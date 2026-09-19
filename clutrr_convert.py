"""CLUTRR text -> facts, learned from the paired data (text + gold graph) as a curriculum: rung 1 = one hop (pairs of names
inside one sentence), rung 2 = whole stories (pairs anywhere in the story; a name's mentions are pooled, so a fact whose
names sit in different sentences is reachable). Then the facts go to the Part-19b operators (trained on the gold graph),
which answer or abstain. Supervision: the gold edges (a stronger setting than the text-only baselines — stated as such).
converter: word rows (an embedding per vocabulary word; names are one NAME token, their identity kept aside) -> BiGRU ->
           each name = mean of its mention states; each ordered pair (a, b) -> MLP -> 21 classes (none + 20 relation words,
           in the gold convention: 'a's <relation> is b').
reported : fact precision / recall on the test stories (unseen templates); end-to-end exact / abstain / wrong by hops with
           the 19b operators on the converter's facts; the same with gold facts (ceiling); rung 1 alone vs rung 1+2."""
import argparse, time, json, re, ast, collections, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common import *
from clutrr_read import load, parse, tok, sentences, RELS, path

ap = argparse.ArgumentParser(); ap.add_argument("--epochs1", type=int, default=6); ap.add_argument("--epochs2", type=int, default=10); ap.add_argument("--hidden", type=int, default=256); ap.add_argument("--emb", type=int, default=128)
ap.add_argument("--ops", default="results/kin_full_k8/ops.pt"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results/clutrr_convert"); ap.add_argument("--probes", type=int, default=64)
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time()
rid = {r: i for i, r in enumerate(RELS)}; NREL = len(RELS)
# ------------------------------------------------------------------ data: tokens (names -> NAME), name mention positions, gold pairs
def story_tokens(r):
    names, gold, g, q, tgt = parse(r); toks, ment, sent_id = [], collections.defaultdict(list), []
    for si, s in enumerate(sentences(r["story"])):
        for w in tok(s):
            if w.startswith("["): ment[w[1:-1]].append(len(toks)); toks.append("NAME")
            else: toks.append(w.lower())
            sent_id.append(si)
        toks.append("<eos>"); sent_id.append(si)
    return {"toks": toks, "sent": sent_id, "ment": dict(ment), "names": names, "gold": gold, "q": q, "target": tgt, "L": len(ast.literal_eval(r["edge_types"]))}
data = {s: [story_tokens(r) for r in load(s)] for s in ("train", "validation", "test")}
cnt = collections.Counter(w for st in data["train"] for w in st["toks"]); VOCAB = ["<pad>", "<unk>", "NAME", "<eos>"] + sorted(w for w, c in cnt.items() if c >= 2 and w not in ("NAME", "<eos>")); wid = {w: i for i, w in enumerate(VOCAB)}
print(f"stories train/val/test {len(data['train'])}/{len(data['validation'])}/{len(data['test'])}; vocabulary {len(VOCAB)} (count >= 2; rest <unk>)  ({time.time()-t0:.0f}s)", flush=True)
def pairs_of(st, within_sentence):
    """ordered name pairs and their label (0 = none, 1 + relation id); rung 1 keeps pairs that co-occur in a sentence"""
    out = []
    for x in st["names"]:
        for y in st["names"]:
            if x == y or x not in st["ment"] or y not in st["ment"]: continue
            if within_sentence and not ({st["sent"][i] for i in st["ment"][x]} & {st["sent"][i] for i in st["ment"][y]}): continue
            out.append((x, y, 1 + rid[st["gold"][(x, y)]] if (x, y) in st["gold"] else 0))
    return out
# ------------------------------------------------------------------ the converter
class Converter(nn.Module):
    def __init__(self):
        super().__init__(); self.emb = nn.Embedding(len(VOCAB), a.emb, padding_idx=0); self.rnn = nn.GRU(a.emb, a.hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        H = 2 * a.hidden; self.mlp = nn.Sequential(nn.Linear(3 * H, a.hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(a.hidden, 1 + NREL))
    def forward(self, batch, within):
        """batch: list of stories; returns logits over (pair) and the pair list"""
        ids = [torch.tensor([wid.get(w, 1) for w in st["toks"]], device=dev) for st in batch]; X = nn.utils.rnn.pad_sequence(ids, batch_first=True)
        if self.training: X = torch.where((torch.rand_like(X, dtype=torch.float) < 0.15) & (X > 3), torch.ones_like(X), X)          # word dropout: 15 % of content words -> <unk>
        Hs, _ = self.rnn(self.emb(X))
        feats, labels, plist = [], [], []
        for b, st in enumerate(batch):
            pooled = {n: Hs[b, torch.tensor(pos, device=dev)].mean(0) for n, pos in st["ment"].items()}
            for x, y, lab in pairs_of(st, within): feats.append(torch.cat([pooled[x], pooled[y], pooled[x] * pooled[y]])); labels.append(lab); plist.append((b, x, y))
        return self.mlp(torch.stack(feats)), torch.tensor(labels, device=dev), plist
model = Converter().to(dev); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
def run_epoch(items, within, train=True, bs=32):
    model.train(train); tot, n, correct = 0.0, 0, 0
    order = rng.permutation(len(items)) if train else np.arange(len(items))
    for i in range(0, len(items), bs):
        batch = [items[j] for j in order[i:i + bs]]
        with torch.set_grad_enabled(train):
            logits, labels, _ = model(batch, within); loss = F.cross_entropy(logits, labels)
            if train: opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tot += loss.item() * len(labels); n += len(labels); correct += (logits.argmax(1) == labels).sum().item()
    return tot / n, correct / n
def facts_of(items, within=False, bs=64):
    """the converter's facts per story (gold convention), and precision/recall against the gold edges"""
    model.eval(); out = []; tp = fp = fn = 0
    with torch.no_grad():
        for i in range(0, len(items), bs):
            batch = items[i:i + bs]; logits, labels, plist = model(batch, within); pred = logits.argmax(1).tolist(); per = [[] for _ in batch]
            for (b, x, y), p in zip(plist, pred):
                if p: per[b].append((x, RELS[p - 1], y))
            for b, st in enumerate(batch):
                got = set(per[b]); gold = {(x, t, y) for (x, y), t in st["gold"].items()}; tp += len(got & gold); fp += len(got - gold); fn += len(gold - got); out.append(per[b])
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1); return out, P, R
res = {}
for rung, within, epochs in (("rung 1: one hop (pairs within a sentence)", True, a.epochs1), ("rung 2: whole stories (all pairs)", False, a.epochs2)):
    print(f"\n== {rung}", flush=True)
    for ep in range(1, epochs + 1):
        tl, ta = run_epoch(data["train"], within); vl, va = run_epoch(data["validation"], within, train=False)
        _, P, R = facts_of(data["test"]); print(f"  epoch {ep}/{epochs}  train loss {tl:.3f} pair-acc {ta:.3f}  val loss {vl:.3f} pair-acc {va:.3f}  TEST facts (story-level): precision {P:.3f} recall {R:.3f}  elapsed {time.time()-t0:.0f}s", flush=True)
    res[rung] = {"test_precision": P, "test_recall": R}
    # facts within sentences only vs story-level, on the test set
    _, Pw, Rw = facts_of(data["test"], within=True); print(f"  test facts, pairs within a sentence only: precision {Pw:.3f} recall {Rw:.3f}; all pairs: precision {P:.3f} recall {R:.3f}", flush=True)
    res[rung].update({"test_precision_within": Pw, "test_recall_within": Rw})
    # ------------------------------------------------------------------ end to end with the Part-19b operators (trained on the gold graph)
    pk = torch.load(HERE / a.ops, map_location=dev, weights_only=False); ORELS = pk["RELS"]; OT = pk["TARGETS"]; orid = {r: i for i, r in enumerate(ORELS)}
    ops = ResonatE(4, len(ORELS), k=8, block=True, block_size=16).to(dev); ops.load_state_dict(pk["model"]); ops.eval(); M = ops.m; otid = torch.tensor([orid[t] for t in OT], device=dev)
    def apply_chains(x, chains):
        L = max(len(c) for c in chains); out = x.clone()
        for j in range(L):
            has = torch.tensor([len(c) > j for c in chains], device=dev); r = torch.tensor([orid[c[j]] if len(c) > j else 0 for c in chains], device=dev); y = ops.hop(out, r); out = torch.where(has[:, None], y, out)
        return out
    def targets_of(x): return torch.stack([ops.hop(x, torch.full((x.shape[0],), int(t), device=dev, dtype=torch.long)) for t in otid], 1)
    def fpath(facts, s, o, maxlen=10):
        """forward-only BFS over directed facts (the converter's facts are in the gold convention)"""
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
    def end_to_end(split, fact_source, theta=None):
        items = data[split]; facts = fact_source(items); seqs = [fpath(f, st["q"][0], st["q"][1]) for f, st in zip(facts, items)]; idx = [i for i, s in enumerate(seqs) if s]
        ev = dict(zip(idx, answer([seqs[i] for i in idx]))) if idx else {}
        if theta is None: bad = [m - m2 for i, (w, m, m2) in ev.items() if w != items[i]["target"]]; theta = float(np.percentile(bad, 95)) if bad else 0.0
        by = collections.defaultdict(list)
        for i, st in enumerate(items):
            if i not in ev: by[st["L"]].append("abstain"); continue
            w, m, m2 = ev[i]; by[st["L"]].append("exact" if (w == st["target"] and m - m2 >= theta) else "abstain" if m - m2 < theta else "wrong")
        tab = {L: {"n": len(v), "exact": v.count("exact") / len(v), "abstain": v.count("abstain") / len(v), "wrong": v.count("wrong") / len(v)} for L, v in sorted(by.items())}; allv = [x for v in by.values() for x in v]
        return tab, {"exact": allv.count("exact") / len(allv), "abstain": allv.count("abstain") / len(allv), "wrong": allv.count("wrong") / len(allv)}, theta
    conv = lambda items: facts_of(items)[0]; goldf = lambda items: [[(x, t, y) for (x, y), t in st["gold"].items()] for st in items]
    from clutrr_read import extract_all
    def cuef(items):                                                                                          # the cue reader's facts (its own direction); used forward-only here, so only gold-form facts help
        return [[f for s_ in sentences(r["story"]) for f in extract_all(s_)] for r in (load("test") if items is data["test"] else load("validation"))]
    for name_, src in (("converter facts", conv), ("cue-reader facts", cuef), ("gold facts (ceiling)", goldf)):
        _, _, theta = end_to_end("validation", src)                                                            # abstention calibrated per fact source
        tab, ov, _ = end_to_end("test", src, theta); res[rung][name_] = {"by_hops": tab, "overall": ov, "theta": theta}
        print(f"  end to end on TEST from text, {name_}: exact/abstain/wrong by hops " + "  ".join(f"{L}: {r['exact']:.2f}/{r['abstain']:.2f}/{r['wrong']:.2f}" for L, r in tab.items()) + f"   overall {ov['exact']:.3f}/{ov['abstain']:.3f}/{ov['wrong']:.3f}", flush=True)
save_json(res, HERE / a.out / "clutrr_convert.json"); torch.save({"model": model.state_dict(), "vocab": VOCAB}, HERE / a.out / "convert.pt"); print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)
