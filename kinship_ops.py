"""CLUTRR unsupervised: relation words as OPERATORS, composition as multiplication, the algebra emergent.
No steps, no genders, no reduction rules, no word meanings. Each of the 20 relation words is a learned block-diagonal
operator (a ResonatE relation). A training story (r1, r2 -> r3) [or (r1, r2, r3 -> r4)] says only: applying the chain's
operators to any person must land where the target word's operator lands. Trained on random probe places.
Answering a chain of any length: apply its operators to a probe, then read which word's single operator lands in the
same place (the light over the 18 target words), abstain if none is close (threshold calibrated on validation).
Reported: CLUTRR test chain stories, exact / abstain / wrong by length (train chains 2-3, test 2-10)."""
import argparse, time, json, collections, numpy as np, torch, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=12); ap.add_argument("--block", type=int, default=16); ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results/kinship_ops"); ap.add_argument("--probes", type=int, default=8); ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--batch", type=int, default=256); ap.add_argument("--shuffle-targets", action="store_true", help="control: permute training targets; test must fall to chance")
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time()
D = json.load(open(str(DATA / "clutrr_structured.json"))); RELS = D["rels"]; rid = {r: i for i, r in enumerate(RELS)}; TARGETS = sorted({d["target"] for d in D["data"]["train"]}); tid = torch.tensor([rid[t] for t in TARGETS], device=dev)
train = D["data"]["train"]; val = D["data"]["validation"]; test = D["data"]["test"]
if a.shuffle_targets: perm = rng.permutation(len(train)); train = [dict(it, target=train[j]["target"]) for it, j in zip(train, perm)]; print("CONTROL: training targets permuted", flush=True)
model = ResonatE(4, len(RELS), k=a.k, block=True, block_size=a.block).to(dev); M = model.m                                  # rows unused; only the 20 relation operators matter
def apply_chain(x, chain):
    for r in chain: x = model.hop(x, torch.full((x.shape[0],), rid[r], device=dev, dtype=torch.long))
    return x
def apply_chains(x, chains):
    """batched: chains is a list of relation-name lists (ragged); one hop per position with per-sample relation ids"""
    L = max(len(c) for c in chains); out = x.clone()
    for j in range(L):
        has = torch.tensor([len(c) > j for c in chains], device=dev); r = torch.tensor([rid[c[j]] if len(c) > j else 0 for c in chains], device=dev)
        y = model.hop(out, r); out = torch.where(has[:, None], y, out)
    return out
def targets_of(x):
    """each target word's single operator applied to the same probe: (B, T, M)"""
    return torch.stack([model.hop(x, torch.full((x.shape[0],), int(t), device=dev, dtype=torch.long)) for t in tid], 1)
def batch(items, n):
    sel = [items[i] for i in rng.integers(0, len(items), n)]; return sel
opt = torch.optim.Adam(model.parameters(), lr=a.lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
for step in range(1, a.steps + 1):
    sel = batch(train, a.batch); loss = 0.0
    x = cnorm(torch.randn(len(sel), M, dtype=torch.complex64, device=dev)); tgt = torch.tensor([TARGETS.index(it["target"]) for it in sel], device=dev)
    y = apply_chains(x, [it["edges"] for it in sel]); T = targets_of(x)
    S = torch.real((y[:, None, :] * T.conj()).sum(-1)) * model.log_tau.exp(); loss = F.cross_entropy(S, tgt)                    # the composed place must be the target word's place, not another word's
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 1000 == 0 or step == 1: print(f"step {step}/{a.steps}  loss {loss.item():.4f}  exact {(S.argmax(1) == tgt).float().mean().item():.3f}  lr {sched.get_last_lr()[0]:.2e}  tau {model.log_tau.exp().item():.1f}  elapsed {time.time()-t0:.0f}s", flush=True)
model.eval(); res = {}
with torch.no_grad():
    def evaluate(items, theta=None, n_probe=None):
        """answer each story: the word whose operator lands closest to the composed place (averaged over probes); abstain below theta"""
        n_probe = n_probe or a.probes; out = []
        for i in range(0, len(items), 64):
            chunk = items[i:i + 64]; B = len(chunk); x = cnorm(torch.randn(B * n_probe, M, dtype=torch.complex64, device=dev)); chains = [it["edges"] for it in chunk for _ in range(n_probe)]
            y = apply_chains(x, chains); T = targets_of(x); c = torch.real((y[:, None, :] * T.conj()).sum(-1)).reshape(B, n_probe, -1).mean(1)
            for b in range(B): best = int(c[b].argmax()); out.append((TARGETS[best], float(c[b, best]), float(c[b].sort(descending=True).values[1])))
        return out
    v = evaluate(val); margins_ok = [m - m2 for (w, m, m2), it in zip(v, val) if w == it["target"]]; margins_bad = [m - m2 for (w, m, m2), it in zip(v, val) if w != it["target"]]
    theta = float(np.percentile(margins_bad, 95)) if margins_bad else 0.0; print(f"validation: exact {np.mean([w == it['target'] for (w, m, m2), it in zip(v, val)]):.3f}; abstention margin threshold {theta:.3f} (95th pct of wrong-answer margins)", flush=True)
    for split, items in (("validation", val), ("test", test)):
        ev = evaluate(items); by = collections.defaultdict(list)
        for (w, m, m2), it in zip(ev, items): by[it["L"]].append("exact" if (w == it["target"] and m - m2 >= theta) else "abstain" if m - m2 < theta else "wrong")
        res[split] = {L: {"n": len(vs), "exact": vs.count("exact") / len(vs), "abstain": vs.count("abstain") / len(vs), "wrong": vs.count("wrong") / len(vs)} for L, vs in sorted(by.items())}
        print(f"{split} chain stories — exact/abstain/wrong by length: " + "  ".join(f"{L}: {r['exact']:.2f}/{r['abstain']:.2f}/{r['wrong']:.2f} (n={r['n']})" for L, r in res[split].items()), flush=True)
    # do the operators form an algebra? check the emergent products against the training compositions, and inverse pairs
    pairs = collections.Counter((it["edges"][0], it["edges"][1], it["target"]) for it in train if it["L"] == 2)
    ok = 0
    for (r1, r2, t), _ in pairs.items():
        x = cnorm(torch.randn(16, M, dtype=torch.complex64, device=dev)); y = apply_chain(x, [r1, r2]); T = targets_of(x); ok += int(TARGETS[int(torch.real((y[:, None, :] * T.conj()).sum(-1)).mean(0).argmax())] == t)
    res["train_pairs_consistent"] = [ok, len(pairs)]; print(f"emergent algebra: {ok}/{len(pairs)} training length-2 compositions reproduced by operator products", flush=True)
save_json(res, HERE / a.out / "kinship_ops.json"); torch.save({"model": model.state_dict(), "RELS": RELS, "TARGETS": TARGETS}, HERE / a.out / "ops.pt"); print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)
