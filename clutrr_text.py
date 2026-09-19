"""CLUTRR from TEXT: facts read from the story sentences, relation words as operators with a REVERSE hop per word, so a fact
read from either end composes; inverses emerge from use (no gender table, no inverse table).
reading : per relation word in a sentence, owner/object by local cues ('s before the word -> owner is the nearest name
          before; 'of' after -> owner is the nearest name after 'of', object the nearest name before; otherwise owner =
          nearest name before, object = nearest name after). A cue-based reader in code — the ceiling of a light-program
          reader over the same cues (its port to the machine is mechanical; see read_math.py).
path    : BFS over the story's extracted facts from the query's first name to its second, traversing a fact either way;
          the relation sequence with directions; no path -> abstain.
operators: 20 relation words x 2 directions (ResonatE forward / reverse hops), trained on TRAINING stories' extracted
          paths only: applying the sequence to a random probe must land where the target word's forward hop lands
          (kinship_ops.py objective). Abstention by the validation margin (95th pct of wrong-answer margins).
reported: test exact / abstain / wrong by hops on the 1,146 test stories, from text; path recoverability as the ceiling."""
import argparse, time, json, sys, collections, numpy as np, torch, torch.nn.functional as F
from common import *
pass
ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=8); ap.add_argument("--block", type=int, default=16); ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results/clutrr_text"); ap.add_argument("--probes", type=int, default=64); ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--tied", action="store_true", help="reverse hop = adjoint of the forward block (an inverse while near-unitary)"); ap.add_argument("--train-on", default="text", help="text: paths over facts read from the training stories' text; gold: the gold edge sequences (graph input)")
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time()
from clutrr_read import load, parse, extract_all, story_facts_all, path, sentences, RELS, RELSET
import ast
rid = {r: i for i, r in enumerate(RELS)}; NREL = len(RELS)
def instances(split):
    """(sequence of (relation, direction), target, hops) per story; direction -1 = the fact is traversed from object to subject"""
    out = []
    for r in load(split):
        names, gold, g, q, tgt = parse(r); L = len(ast.literal_eval(r["edge_types"]))
        if a.train_on == "gold" and split == "train": seq = [(t, +1) for t in ast.literal_eval(r["edge_types"])]
        else: seq = path(story_facts_all(r), q[0], q[1])
        out.append({"seq": seq, "target": tgt, "L": L})
    return out
data = {s: instances(s) for s in ("train", "validation", "test")}
for s, items in data.items(): print(f"{s}: {len(items)} stories; path over extracted facts found for {np.mean([it['seq'] is not None for it in items]):.3f}" + ("" if s != "test" else "; by hops: " + " ".join(f"{L}:{np.mean([it['seq'] is not None for it in items if it['L'] == L]):.2f}" for L in range(2, 11))), flush=True)
train = [it for it in data["train"] if it["seq"]]; val = [it for it in data["validation"] if it["seq"]]; test = data["test"]
TARGETS = sorted({it["target"] for it in data["train"]}); tid = torch.tensor([rid[t] for t in TARGETS], device=dev)
model = ResonatE(4, 2 * NREL, k=a.k, block=True, block_size=a.block, tied_reverse=a.tied).to(dev); M = model.m                                 # relation r forward = r, traversed backwards = r + NREL (the reverse hop)
def opid(rel, d): return rid[rel] + (0 if d > 0 else NREL)
def apply_chains(x, chains):
    L = max(len(c) for c in chains); out = x.clone()
    for j in range(L):
        has = torch.tensor([len(c) > j for c in chains], device=dev); r = torch.tensor([opid(*c[j]) if len(c) > j else 0 for c in chains], device=dev)
        y = model.hop(out, r); out = torch.where(has[:, None], y, out)
    return out
def targets_of(x): return torch.stack([model.hop(x, torch.full((x.shape[0],), int(t), device=dev, dtype=torch.long)) for t in tid], 1)
opt = torch.optim.Adam(model.parameters(), lr=a.lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
print(f"training on {len(train)} {a.train_on} paths (train stories with a recoverable path); validation {len(val)}", flush=True)
for step in range(1, a.steps + 1):
    sel = [train[i] for i in rng.integers(0, len(train), a.batch)]
    x = cnorm(torch.randn(len(sel), M, dtype=torch.complex64, device=dev)); tgt = torch.tensor([TARGETS.index(it["target"]) for it in sel], device=dev)
    y = apply_chains(x, [it["seq"] for it in sel]); T = targets_of(x); S = torch.real((y[:, None, :] * T.conj()).sum(-1)) * model.log_tau.exp(); loss = F.cross_entropy(S, tgt)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 2000 == 0 or step == 1: print(f"step {step}/{a.steps}  loss {loss.item():.4f}  exact {(S.argmax(1) == tgt).float().mean().item():.3f}  lr {sched.get_last_lr()[0]:.2e}  tau {model.log_tau.exp().item():.1f}  elapsed {time.time()-t0:.0f}s", flush=True)
model.eval(); res = {"path_recoverable": {s: float(np.mean([it["seq"] is not None for it in items])) for s, items in data.items()}}
with torch.no_grad():
    def evaluate(items):
        out = []
        for i in range(0, len(items), 64):
            chunk = items[i:i + 64]; B = len(chunk); x = cnorm(torch.randn(B * a.probes, M, dtype=torch.complex64, device=dev)); chains = [it["seq"] for it in chunk for _ in range(a.probes)]
            y = apply_chains(x, chains); T = targets_of(x); c = torch.real((y[:, None, :] * T.conj()).sum(-1)).reshape(B, a.probes, -1).mean(1)
            for b in range(B): best = int(c[b].argmax()); out.append((TARGETS[best], float(c[b, best]), float(c[b].sort(descending=True).values[1])))
        return out
    v = evaluate(val); margins_bad = [m - m2 for (w, m, m2), it in zip(v, val) if w != it["target"]]; theta = float(np.percentile(margins_bad, 95)) if margins_bad else 0.0
    print(f"validation (stories with a path): exact {np.mean([w == it['target'] for (w, m, m2), it in zip(v, val)]):.3f}; abstention margin threshold {theta:.3f}", flush=True)
    for split, items in (("validation", data["validation"]), ("test", test)):
        withp = [it for it in items if it["seq"]]; ev = dict(zip([id(it) for it in withp], evaluate(withp))); by = collections.defaultdict(list)
        for it in items:
            if not it["seq"]: by[it["L"]].append("abstain"); continue
            w, m, m2 = ev[id(it)]; by[it["L"]].append("exact" if (w == it["target"] and m - m2 >= theta) else "abstain" if m - m2 < theta else "wrong")
        res[split] = {L: {"n": len(vs), "exact": vs.count("exact") / len(vs), "abstain": vs.count("abstain") / len(vs), "wrong": vs.count("wrong") / len(vs)} for L, vs in sorted(by.items())}
        print(f"{split} FROM TEXT — exact/abstain/wrong by hops: " + "  ".join(f"{L}: {r['exact']:.2f}/{r['abstain']:.2f}/{r['wrong']:.2f} (n={r['n']})" for L, r in res[split].items()), flush=True)
        allv = [x for vs in by.values() for x in vs]; print(f"   overall: exact {allv.count('exact')/len(allv):.3f} abstain {allv.count('abstain')/len(allv):.3f} wrong {allv.count('wrong')/len(allv):.3f}", flush=True)
    # do the reverse hops act as inverses? word w forward then w backward should return to the probe
    x = cnorm(torch.randn(256, M, dtype=torch.complex64, device=dev)); inv = {}
    for w in RELS: y = model.hop(model.hop(x, torch.full((256,), rid[w], device=dev, dtype=torch.long)), torch.full((256,), rid[w] + NREL, device=dev, dtype=torch.long)); inv[w] = float(torch.real((y * x.conj()).sum(-1)).mean())
    print("forward-then-reverse returns to the probe (cosine), per word: " + " ".join(f"{w}:{c:.2f}" for w, c in inv.items()), flush=True); res["inverse_consistency"] = inv
save_json(res, HERE / a.out / "clutrr_text.json"); torch.save({"model": model.state_dict(), "RELS": RELS, "TARGETS": TARGETS}, HERE / a.out / "ops.pt"); print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)
