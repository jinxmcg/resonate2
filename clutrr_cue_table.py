"""CLUTRR from text: the cue reader (reads English at 0.96 per clean sentence on unseen templates) + a RELABELLING TABLE
counted from the training data (how the gold graph writes what the cue reader read: (word read, form, gender cue) ->
(gold label, direction)), then forward paths through the Part-19b operators. No training; the table is the only learned
part and it is counted, not fitted. The point: is the reading-to-gold gap a vocabulary/convention gap (fixable by a table)
or a structure gap (not)?"""
import argparse, time, json, ast, collections, numpy as np, torch
from common import *
from clutrr_read import load, parse, tok, sentences, RELS, extract_all

ap = argparse.ArgumentParser(); ap.add_argument("--ops", default="results/kin_full_k8/ops.pt"); ap.add_argument("--out", default="results/clutrr_cue_table"); ap.add_argument("--probes", type=int, default=64); ap.add_argument("--min", type=int, default=5)
a = ap.parse_args(); dev = torch.device("cuda"); t0 = time.time(); torch.manual_seed(0)
def gender_cue(sent, name):
    """'his'/'her'/'he'/'she' nearest to the name's mention (a local cue for the owner's gender), else none"""
    t = [w.lower() for w in tok(sent)]; pos = [i for i, w in enumerate(t) if w == f"[{name.lower()}]"]
    best = None
    for i in pos:
        for d in range(1, 5):
            for j in (i - d, i + d):
                if 0 <= j < len(t) and t[j] in ("his", "her", "he", "she") and best is None: best = t[j]
    return best
def read_story(r):
    """cue facts with their local key: (word, owner-is-first-mention?, gender cue at owner)"""
    out = []
    for s in sentences(r["story"]):
        ns = [w[1:-1] for w in tok(s) if w.startswith("[")]
        for (o, w, b) in extract_all(s): out.append(((w, ns.index(o) < ns.index(b) if o in ns and b in ns else True, gender_cue(s, o)), o, b))
    return out
# ------------------------------------------------------------------ the table, counted on the training stories
table = collections.defaultdict(collections.Counter)
for r in load("train"):
    names, gold, g, q, tgt = parse(r)
    for key, o, b in read_story(r):
        if (o, b) in gold: table[key][("keep", gold[(o, b)])] += 1
        elif (b, o) in gold: table[key][("flip", gold[(b, o)])] += 1
MAP = {k: c.most_common(1)[0][0] for k, c in table.items() if sum(c.values()) >= a.min}; cons = np.mean([c.most_common(1)[0][1] / sum(c.values()) for k, c in table.items() if sum(c.values()) >= a.min])
print(f"relabelling table: {len(MAP)} keys (count >= {a.min}); mean consistency of the majority mapping {cons:.3f}  ({time.time()-t0:.0f}s)", flush=True)
for k in list(MAP)[:8]: print("   ", k, "->", MAP[k], dict(table[k]))
def facts_gold_form(r):
    out = []
    for key, o, b in read_story(r):
        m = MAP.get(key) or MAP.get((key[0], key[1], None))
        if m is None: continue
        out.append((o, m[1], b) if m[0] == "keep" else (b, m[1], o))
    return out
# ------------------------------------------------------------------ fact quality and end to end
pk = torch.load(HERE / a.ops, map_location=dev, weights_only=False); ORELS = pk["RELS"]; OT = pk["TARGETS"]; orid = {r_: i for i, r_ in enumerate(ORELS)}
ops = ResonatE(4, len(ORELS), k=8, block=True, block_size=16).to(dev); ops.load_state_dict(pk["model"]); ops.eval(); M = ops.m; otid = torch.tensor([orid[t] for t in OT], device=dev)
def apply_chains(x, chains):
    L = max(len(c) for c in chains); out = x.clone()
    for j in range(L):
        has = torch.tensor([len(c) > j for c in chains], device=dev); rr = torch.tensor([orid[c[j]] if len(c) > j else 0 for c in chains], device=dev); y = ops.hop(out, rr); out = torch.where(has[:, None], y, out)
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
def evaluate(split, theta=None):
    rows = load(split); tp = fp = fn = 0; seqs = []; meta = []
    for r in rows:
        names, gold, g, q, tgt = parse(r); facts = facts_gold_form(r); got = set(facts); gs = {(x, t, y) for (x, y), t in gold.items()}; tp += len(got & gs); fp += len(got - gs); fn += len(gs - got)
        seqs.append(fpath(facts, q[0], q[1])); meta.append((tgt, len(ast.literal_eval(r["edge_types"]))))
    P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1); idx = [i for i, s_ in enumerate(seqs) if s_]; ev = dict(zip(idx, answer([seqs[i] for i in idx]))) if idx else {}
    if theta is None: bad = [m - m2 for i, (w, m, m2) in ev.items() if w != meta[i][0]]; theta = float(np.percentile(bad, 95)) if bad else 0.0
    by = collections.defaultdict(list)
    for i, (tgt, L) in enumerate(meta):
        if i not in ev: by[L].append("abstain"); continue
        w, m, m2 = ev[i]; by[L].append("exact" if (w == tgt and m - m2 >= theta) else "abstain" if m - m2 < theta else "wrong")
    tab = {L: {"n": len(v), "exact": v.count("exact") / len(v), "abstain": v.count("abstain") / len(v), "wrong": v.count("wrong") / len(v)} for L, v in sorted(by.items())}; allv = [x for v in by.values() for x in v]
    print(f"{split}: facts precision {P:.3f} recall {R:.3f}; path recoverable {len(idx)/len(rows):.3f}; exact/abstain/wrong by hops " + "  ".join(f"{L}: {r_['exact']:.2f}/{r_['abstain']:.2f}/{r_['wrong']:.2f}" for L, r_ in tab.items()) + f"   overall {allv.count('exact')/len(allv):.3f}/{allv.count('abstain')/len(allv):.3f}/{allv.count('wrong')/len(allv):.3f}", flush=True)
    return {"precision": P, "recall": R, "path": len(idx) / len(rows), "by_hops": tab, "theta": theta}, theta
res = {}; res["validation"], theta = evaluate("validation"); res["test"], _ = evaluate("test", theta)
save_json(res, HERE / a.out / "clutrr_cue_table.json"); print(f"saved ({time.time()-t0:.0f}s)", flush=True)
