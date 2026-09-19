"""Closing the loop without order: add one place at a time.
x, y as path chains -> walk back (digits, exact) -> per place k from units: bind the value anchors of the two digits and
the carry, v(da,k) (.) v(db,k) (.) v(c,k), and read the digit sum by EQUALITY lights against the 20 candidate values
s = 0..18 (s >= 10 as v(1,k+1) (.) v(s-10,k)) -> digit = s mod 10, carry = s div 10 -> rebuild the path chain of the
result -> walk it back. Nothing new is trained: path model (Part 4), value code (Part 6), the program iterates.
Exact on never-trained sums by length and number of carries."""
import argparse, time, numpy as np, torch
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--path", default="results/two_relations_k12_walk2/memory.pt"); ap.add_argument("--value", default="results/two_part_learn/value.pt"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results/add_program")
a = ap.parse_args(); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time()
NODES = [str(d) for d in range(10)] + [str(10 ** k) for k in range(1, 7)]; nid = {n: i for i, n in enumerate(NODES)}; PLUS, TIMES = 0, 1; NR = 2
pk = torch.load(HERE / a.path, map_location=dev, weights_only=False); pcfg = pk["config"]; pmodel = ResonatE(len(NODES), 2 * NR, k=pcfg["k"], block=True, block_size=pcfg["block"]).to(dev); pmodel.load_state_dict(pk["model"]); pmodel.eval()
ANCH = torch.tensor([nid["0"]] + [nid[str(10 ** k)] for k in range(1, 7)], device=dev); Er = cnorm(pmodel.E).detach()
def phop(z, r): return pmodel.hop(z, torch.full((z.shape[0],), r, device=dev, dtype=torch.long))
def plight(z): return torch.real(z @ Er.conj().t())
def digs(ns): return torch.tensor([[int(c) for c in f"{int(n):07d}"] for n in ns], device=dev)
def path_chain(D):
    B = D.shape[0]; acc = None
    for j in range(7):
        k = 6 - j; d = D[:, j]; term = Er[d] if k == 0 else phop(cnorm(Er[d] + Er[ANCH[k]][None].expand(B, -1)), TIMES)
        acc = phop(term, PLUS) if acc is None else phop(cnorm(acc + term), PLUS)
    return acc
def walk_digits(z):
    B = z.shape[0]; out = torch.zeros(B, 7, dtype=torch.long, device=dev); rr = z
    for k in range(7):
        rr = phop(rr, PLUS + NR)
        if k == 0: out[:, 0] = plight(rr).argmax(1)
        else: L = plight(phop(rr, TIMES + NR)); L[:, 10:] = -1e9; out[:, k] = L.argmax(1)
    return out                                                                                     # (B, 7), place 0 first
vk = torch.load(HERE / a.value, map_location=dev, weights_only=False); phi = vk["phi"]; mv = phi.shape[-1]; p = phi.clone(); p[0] = 0; V = torch.polar(torch.ones_like(p), p)   # V[d, k, :]
with torch.no_grad():
    # the 20 candidate values of a digit sum at place k: s = 0..19  ->  v(s%10, k) (.) v(s//10, k+1)   (place 6 cannot carry out: sums stay below 10^7)
    cand = torch.stack([torch.stack([V[s % 10, k] * (V[s // 10, k + 1] if k < 6 and s >= 10 else torch.ones(mv, dtype=torch.complex64, device=dev)) for s in range(20)]) for k in range(7)])   # (7, 19, mv)
    def add(xs, ys):
        Dx, Dy = walk_digits(path_chain(digs(xs))), walk_digits(path_chain(digs(ys))); B = len(xs); carry = torch.zeros(B, dtype=torch.long, device=dev); out = torch.zeros(B, 7, dtype=torch.long, device=dev)
        for k in range(7):
            e = V[Dx[:, k], k] * V[Dy[:, k], k] * V[carry, k]                                        # bind the two digits and the carry at this place
            L = torch.real((e[:, None, :] * cand[k][None].conj()).mean(-1)); s = L.argmax(1)         # equality light against the 19 candidates
            out[:, k] = s % 10; carry = s // 10
        res_digits = out.flip(1)                                                                    # biggest place first, for the chain
        return walk_digits(path_chain(res_digits)), out
    def to_int(D): return [int(sum(int(row[k]) * 10 ** k for k in range(7))) for row in D.cpu().numpy()]
    def n_carries(x, y):
        c = n = 0; dx, dy = f"{x:07d}"[::-1], f"{y:07d}"[::-1]
        for i in range(7): s = int(dx[i]) + int(dy[i]) + c; c = 1 if s >= 10 else 0; n += c
        return n
    res = {}
    for L in range(1, 8):
        pairs = []
        while len(pairs) < 1000:
            s = int(rng.integers(10 ** (L - 1), 10 ** L)); x = int(rng.integers(0, s + 1)); pairs.append((x, s - x))
        walked, direct = add([p[0] for p in pairs], [p[1] for p in pairs]); ok = [w == x + y for w, (x, y) in zip(to_int(walked), pairs)]; nc = [n_carries(x, y) for x, y in pairs]
        row = {"n": len(pairs), "exact": float(np.mean(ok))}
        for c in (0, 1, 2, 3):
            sel = [o for o, k in zip(ok, nc) if (k == c if c < 3 else k >= 3)]
            if sel: row[f"carries_{c}{'+' if c == 3 else ''}"] = {"n": len(sel), "exact": float(np.mean(sel))}
        res[f"sum_len{L}"] = row
        print(f"{L}-digit sums (never trained), walk -> per-place value equality -> path -> walk: exact {np.mean(ok):.3f}   " + "  ".join(f"{k.replace('carries_', 'carries=')} {v['exact']:.3f}(n={v['n']})" for k, v in row.items() if k.startswith("carries")), flush=True)
    for x, y in ((23, 45), (47, 35), (123456, 111111), (999999, 1), (4999999, 4999999), (347, 653)):
        w, _ = add([x], [y]); print(f"{x} + {y} = {x + y}:   read {to_int(w)[0]}", flush=True)
    # the other agent's dev_v1 buckets, for the table: compose (add / x10^k / repeated addition) and decompose
    def evaluate(e):
        if isinstance(e, str): return int(e)
        vals = [evaluate(x) for x in e["args"]]
        if e["op"] == "add":
            acc = vals[0]
            for v in vals[1:]: acc = to_int(add([acc], [v])[0])[0]
            return acc
        acc = vals[0]
        for v in vals[1:]:
            if v and 10 ** (len(str(v)) - 1) == v: acc = to_int(walk_digits(path_chain(digs([acc * v]))))[0]       # x 10^k: the digits move up k places
            elif acc and 10 ** (len(str(acc)) - 1) == acc: acc = to_int(walk_digits(path_chain(digs([v * acc]))))[0]
            else:
                small, big_ = (v, acc) if v <= acc else (acc, v); s = 0
                for _ in range(small): s = to_int(add([s], [big_])[0])[0]
                acc = s
        return acc
    import collections
    for split in ("validation", "test"):
        per = collections.defaultdict(list)
        for r in records(split):
            mi = model_input(r)
            if r["task"] == "compose": ok = evaluate(mi.inputs["expression"]) == int("".join(r["target"]["digits"]))
            elif r["task"] == "decompose":
                n = int(mi.number_literals[0]); dg = walk_digits(path_chain(digs([n])))[0].tolist(); pred = sorted((d, k) for k, d in enumerate(dg) if d); tg = sorted((int(t["coefficient"]), int(t["power10"])) for t in r["target"]["terms"] if t["coefficient"] != "0"); ok = pred == tg or (n == 0 and not pred)
            else: continue
            per[f"{r['task']} {'+'.join(r['metadata']['buckets'])} [{r['metadata']['pool']}]"].append(ok)
        res[split] = {bk: {"n": len(v), "exact": float(np.mean(v))} for bk, v in sorted(per.items())}
        print(f"== dev_v1 {split}"); [print(f"  {bk:62s} n={o['n']:4d}  exact {o['exact']:.3f}", flush=True) for bk, o in res[split].items() if "operator_change" not in bk and "grouping" not in bk]
save_json(res, HERE / a.out / "add_program.json"); print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)
