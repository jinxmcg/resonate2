"""CLUTRR text -> facts with a LOCAL converter: the cue reader with its labels learned from the gold graph instead of written.
Why local: a BiGRU over the whole sentence memorises the 4,620 training templates (val 0.97) and transfers poorly to the
1,423 unseen test templates (precision 0.55); the cue reader's local rules transfer at 0.96. So: classify each
(relation-word mention, ordered name pair) from a small window around the word.
relation-word candidates: discovered from data — words whose presence in a sentence with a gold pair has high mutual
    information with the gold label (count >= 20, PMI-like ratio >= 3); 'dad', 'mom', 'kids' come in this way.
features (all local): the word; the +-3 window tokens by offset; 's before / of after; for each of the two names:
    before/after the word, nearest name on that side or not, same sentence or not; the order of the two names.
label: none, or the gold relation from x to y (gold convention), taught on every mention in a sentence with the pair.
rung 1: pairs within a sentence. rung 2: mentions in a sentence containing at least one of the two names (cross-sentence).
Facts -> the Part-19b operators (gold-graph trained) -> answer / abstain. Gold facts through the same pipeline = ceiling."""
import argparse, time, json, ast, collections, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common import *
from clutrr_read import load, parse, tok, sentences, RELS

ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--emb", type=int, default=64); ap.add_argument("--hidden", type=int, default=256); ap.add_argument("--window", type=int, default=3)
ap.add_argument("--ops", default="results/kin_full_k8/ops.pt"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results/clutrr_local"); ap.add_argument("--probes", type=int, default=64)
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time()
rid = {r: i for i, r in enumerate(RELS)}; NREL = len(RELS); W = a.window
def story_tokens(r):
    names, gold, g, q, tgt = parse(r); toks, ment, sent_id = [], collections.defaultdict(list), []
    for si, s in enumerate(sentences(r["story"])):
        for w in tok(s):
            if w.startswith("["): ment[w[1:-1]].append(len(toks)); toks.append("NAME")
            else: toks.append(w.lower())
            sent_id.append(si)
    return {"toks": toks, "sent": sent_id, "ment": dict(ment), "names": names, "gold": gold, "q": q, "target": tgt, "L": len(ast.literal_eval(r["edge_types"]))}
data = {s: [story_tokens(r) for r in load(s)] for s in ("train", "validation", "test")}
cnt = collections.Counter(w for st in data["train"] for w in st["toks"]); VOCAB = ["<pad>", "<unk>", "NAME"] + sorted(w for w, c in cnt.items() if c >= 2 and w != "NAME"); wid = {w: i for i, w in enumerate(VOCAB)}
# ------------------------------------------------------------------ relation-word candidates from data: words associated with a SPECIFIC gold label
cw = collections.Counter(); cwt = collections.Counter(); ct = collections.Counter(); N = 0; templates_of = {}
for st in data["train"]:
    by_sent = collections.defaultdict(set)
    for i, s in enumerate(st["sent"]): by_sent[s].add(st["toks"][i])
    for (x, y), t in st["gold"].items():
        if x not in st["ment"] or y not in st["ment"]: continue
        for s in {st["sent"][p] for p in st["ment"][x]} & {st["sent"][p] for p in st["ment"][y]}:
            N += 1; ct[t] += 1; tmpl = " ".join(st["toks"][i] for i in range(len(st["toks"])) if st["sent"][i] == s)
            for w in by_sent[s]: cw[w] += 1; cwt[(w, t)] += 1; templates_of.setdefault(w, set()).add(tmpl)
def assoc(w): return max((cwt[(w, t)] / cw[w]) / (ct[t] / N) for t in ct)                                          # lift of the best label given the word
CANDS = sorted(w for w in cw if cw[w] >= 20 and len(templates_of[w]) >= 15 and w not in ("NAME", ".", ",", "'s") and assoc(w) >= 3.0)   # recurs across many templates AND predicts a label
print(f"vocabulary {len(VOCAB)}; relation-word candidates discovered from data (label lift >= 3, >= 15 templates): {len(CANDS)} -> {CANDS}  ({time.time()-t0:.0f}s)", flush=True)
CSET = set(CANDS)
# ------------------------------------------------------------------ instances: (mention, ordered pair) -> local features
NF_ = 2 * W + 1
def instances(st, cross):
    """for each ordered name pair and each candidate mention in a sentence containing both names (rung 1) or at least one (rung 2)"""
    out = []; toks, sent = st["toks"], st["sent"]; names = [n for n in st["names"] if n in st["ment"]]
    name_pos = {n: st["ment"][n] for n in names}; all_pos = sorted((p, n) for n, ps in name_pos.items() for p in ps)
    for i, w in enumerate(toks):
        if w not in CSET: continue
        s = sent[i]; before = [(p, n) for p, n in all_pos if p < i]; after = [(p, n) for p, n in all_pos if p > i]
        nb = before[-1][1] if before else None; na = after[0][1] if after else None
        for x in names:
            for y in names:
                if x == y: continue
                sx = s in {sent[p] for p in name_pos[x]}; sy = s in {sent[p] for p in name_pos[y]}
                if not (sx and sy) and not (cross and (sx or sy)): continue
                win = [wid.get(toks[i + d], 1) if 0 <= i + d < len(toks) and sent[i + d] == s else 0 for d in range(-W, W + 1)]
                def side(n):
                    ps = [p for p in name_pos[n] if sent[p] == s]
                    if not ps: return 4                                                                             # not in this sentence
                    p = min(ps, key=lambda p: abs(p - i)); return (0 if p < i else 2) + (1 if (n == nb if p < i else n == na) else 0)   # before/after x nearest/not
                f = win + [wid[w], int(i > 0 and toks[i - 1] == "'s"), int(i + 1 < len(toks) and toks[i + 1] == "of"), side(x), side(y), int(sx), int(sy), int(min(name_pos[x]) < min(name_pos[y]))]
                lab = 1 + rid[st["gold"][(x, y)]] if (x, y) in st["gold"] else 0
                out.append((f, lab, x, y))
    return out
class Local(nn.Module):
    def __init__(self):
        super().__init__(); self.emb = nn.Embedding(len(VOCAB), a.emb, padding_idx=0); self.pos = nn.Parameter(torch.randn(NF_, a.emb) * 0.1); self.small = nn.Embedding(6, 16)
        self.mlp = nn.Sequential(nn.Linear(NF_ * a.emb + a.emb + 7 * 16, a.hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(a.hidden, a.hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(a.hidden, 1 + NREL))
    def forward(self, f):
        f = torch.as_tensor(f, device=dev); win = self.emb(f[:, :NF_]) + self.pos[None]; word = self.emb(f[:, NF_]); small = self.small(f[:, NF_ + 1:]).flatten(1)
        return self.mlp(torch.cat([win.flatten(1), word, small], 1))
model = Local().to(dev); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
def facts_of(items, cross, bs=4096):
    """facts per story: for each ordered pair, the most confident non-none label over its mentions (if its probability beats 'none' summed... simply argmax of the mention with the highest non-none probability)"""
    model.eval(); out = []; tp = fp = fn = 0
    with torch.no_grad():
        for st in items:
            inst = instances(st, cross); best = {}
            if inst:
                P = torch.cat([F.softmax(model([f for f, _, _, _ in inst[i:i + bs]]), 1) for i in range(0, len(inst), bs)])
                for (f, lab, x, y), p in zip(inst, P):
                    nn_p, nn_l = p[1:].max(0); nn_p = float(nn_p); l = int(nn_l) + 1
                    if nn_p > float(p[0]) and nn_p > best.get((x, y), (0, None))[0]: best[(x, y)] = (nn_p, l)
            facts = [(x, RELS[l - 1], y) for (x, y), (pp, l) in best.items()]; out.append(facts)
            got = set(facts); gold = {(x, t, y) for (x, y), t in st["gold"].items()}; tp += len(got & gold); fp += len(got - gold); fn += len(gold - got)
    return out, tp / max(tp + fp, 1), tp / max(tp + fn, 1)
# ------------------------------------------------------------------ end to end (operators of Part 19b, forward paths)
pk = torch.load(HERE / a.ops, map_location=dev, weights_only=False); ORELS = pk["RELS"]; OT = pk["TARGETS"]; orid = {r: i for i, r in enumerate(ORELS)}
ops = ResonatE(4, len(ORELS), k=8, block=True, block_size=16).to(dev); ops.load_state_dict(pk["model"]); ops.eval(); M = ops.m; otid = torch.tensor([orid[t] for t in OT], device=dev)
def apply_chains(x, chains):
    L = max(len(c) for c in chains); out = x.clone()
    for j in range(L):
        has = torch.tensor([len(c) > j for c in chains], device=dev); r = torch.tensor([orid[c[j]] if len(c) > j else 0 for c in chains], device=dev); y = ops.hop(out, r); out = torch.where(has[:, None], y, out)
    return out
def targets_of(x): return torch.stack([ops.hop(x, torch.full((x.shape[0],), int(t), device=dev, dtype=torch.long)) for t in otid], 1)
def fpath(facts, s, o, maxlen=10):
    adj = collections.defaultdict(list); [adj[x].append((y, t)) for x, t, y in facts]
    from collections import deque; Q = deque([(s, [])]); seen = {s}
    while Q:
        x, p = Q.popleft()
        if x == o: return p
        if len(p) >= maxlen: continue
        for y, t in adj[x]:
            if y not in seen: seen.add(y); Q.append((y, p + [t]))
    return None
def answer(chains):
    out = []
    with torch.no_grad():
        for i in range(0, len(chains), 64):
            ch = chains[i:i + 64]; B = len(ch); x = cnorm(torch.randn(B * a.probes, M, dtype=torch.complex64, device=dev)); y = apply_chains(x, [c for c in ch for _ in range(a.probes)]); T = targets_of(x)
            c = torch.real((y[:, None, :] * T.conj()).sum(-1)).reshape(B, a.probes, -1).mean(1)
            for b in range(B): best = int(c[b].argmax()); out.append((OT[best], float(c[b, best]), float(c[b].sort(descending=True).values[1])))
    return out
def end_to_end(items, facts, theta=None):
    seqs = [fpath(f, st["q"][0], st["q"][1]) for f, st in zip(facts, items)]; idx = [i for i, s in enumerate(seqs) if s]; ev = dict(zip(idx, answer([seqs[i] for i in idx]))) if idx else {}
    if theta is None: bad = [m - m2 for i, (w, m, m2) in ev.items() if w != items[i]["target"]]; theta = float(np.percentile(bad, 95)) if bad else 0.0
    by = collections.defaultdict(list)
    for i, st in enumerate(items):
        if i not in ev: by[st["L"]].append("abstain"); continue
        w, m, m2 = ev[i]; by[st["L"]].append("exact" if (w == st["target"] and m - m2 >= theta) else "abstain" if m - m2 < theta else "wrong")
    tab = {L: {"n": len(v), "exact": v.count("exact") / len(v), "abstain": v.count("abstain") / len(v), "wrong": v.count("wrong") / len(v)} for L, v in sorted(by.items())}; allv = [x for v in by.values() for x in v]
    return tab, {"exact": allv.count("exact") / len(allv), "abstain": allv.count("abstain") / len(allv), "wrong": allv.count("wrong") / len(allv)}, theta
def report(name_, facts_val, facts_test, store):
    _, _, theta = end_to_end(data["validation"], facts_val); tab, ov, _ = end_to_end(data["test"], facts_test, theta); store[name_] = {"by_hops": tab, "overall": ov, "theta": theta}
    print(f"  end to end on TEST from text, {name_}: exact/abstain/wrong by hops " + "  ".join(f"{L}: {r['exact']:.2f}/{r['abstain']:.2f}/{r['wrong']:.2f}" for L, r in tab.items()) + f"   overall {ov['exact']:.3f}/{ov['abstain']:.3f}/{ov['wrong']:.3f}", flush=True)
res = {}
goldf = lambda items: [[(x, t, y) for (x, y), t in st["gold"].items()] for st in items]
report("gold facts (ceiling)", goldf(data["validation"]), goldf(data["test"]), res)
for rung, cross in (("rung 1: one hop (mentions in a sentence with both names)", False), ("rung 2: cross-sentence (mentions in a sentence with either name)", True)):
    print(f"\n== {rung}", flush=True); train_inst = [inst for st in data["train"] for inst in instances(st, cross)]; print(f"  {len(train_inst)} training instances", flush=True)
    for ep in range(1, a.epochs + 1):
        model.train(); order = rng.permutation(len(train_inst)); tot = 0.0; cor = 0
        for i in range(0, len(order), 2048):
            b = [train_inst[j] for j in order[i:i + 2048]]; logits = model([f for f, _, _, _ in b]); lab = torch.tensor([l for _, l, _, _ in b], device=dev); loss = F.cross_entropy(logits, lab)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b); cor += (logits.argmax(1) == lab).sum().item()
        _, P, R = facts_of(data["test"], cross); print(f"  epoch {ep}/{a.epochs}  train loss {tot/len(order):.3f} acc {cor/len(order):.3f}   TEST facts: precision {P:.3f} recall {R:.3f}  elapsed {time.time()-t0:.0f}s", flush=True)
    fv, Pv, Rv = facts_of(data["validation"], cross); ft, P, R = facts_of(data["test"], cross); res[rung] = {"val_precision": Pv, "val_recall": Rv, "test_precision": P, "test_recall": R}
    print(f"  facts: validation precision {Pv:.3f} recall {Rv:.3f}; TEST (unseen templates) precision {P:.3f} recall {R:.3f}", flush=True)
    report(f"{rung} facts", fv, ft, res[rung])
save_json(res, HERE / a.out / "clutrr_local.json"); torch.save({"model": model.state_dict(), "vocab": VOCAB, "cands": CANDS}, HERE / a.out / "local.pt"); print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)
