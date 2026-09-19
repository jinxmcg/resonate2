"""Numbers as a knowledge graph with the human place concepts, up to millions.
entities  : numbers. Rows for 0..9999 minus a random 30% (held out: no row, no edge; 0..19 always kept for the counting
            line) plus a sparse band (1000 random numbers each with 5, 6, 7 digits).
relations : units, tens, hundreds, thousands, ten_thousands, hundred_thousands, millions  (n --place_k--> digit row,
            zero digits included so every number has all 7 edges), and successor (n --> n+1 where both have rows).
            Free reverse operators, as in the research trainer (id r + 8).
Trained exactly like a KG: cross-entropy of the full light at the target row, both directions.
compose(n)= cnorm( sum_k hop(E_{d_k}, place_k^-1) )        the AND of seven probes: a spot when n has no row
decode(z) = [ argmax over rows 0..9 of light(hop(z, place_k)) ]_k           read the digits by the memory's own relations
add       = a FIXED program: per place read both digits, walk successor on the 0..19 rows, read units and carry, rebind.
The memory never sees + or x. Exact scores, by digit length, on numbers that have no row.
"""
import argparse, json, time, collections, numpy as np, torch, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=12); ap.add_argument("--block", type=int, default=16); ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--batch", type=int, default=8192); ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--dense-max", type=int, default=9999)
ap.add_argument("--delete", type=float, default=0.3); ap.add_argument("--sparse-per-len", type=int, default=1000); ap.add_argument("--out", default="results/number_kg_k12"); ap.add_argument("--resume", action="store_true"); ap.add_argument("--chain", type=int, default=1, help="counting by d: successor^d from row n must land on row n+d, d<=chain, on the 0..19 rows"); ap.add_argument("--adjoint", action="store_true", help="no learned reverse: the adjoint of the forward operator is the reverse light, exactly; forward loss only")
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time()
PLACES = ["units", "tens", "hundreds", "thousands", "ten_thousands", "hundred_thousands", "millions"]; NP = len(PLACES); SUCC = NP; NR = NP + 1
# ------------------------------------------------------------------ the graph
dense = np.arange(a.dense_max + 1); deletable = dense[dense >= 20]; deleted = set(rng.choice(deletable, int(a.delete * len(deletable)), replace=False).tolist())
sparse = [int(x) for L in (5, 6, 7) for x in rng.integers(10 ** (L - 1), 10 ** L, a.sparse_per_len)]
stored = sorted((set(dense.tolist()) - deleted) | set(sparse)); rid = {n: i for i, n in enumerate(stored)}; N = len(stored)
def digs(n): return [int(c) for c in f"{int(n):07d}"[::-1]]                                # place 0 first, 7 places
edges = [(rid[n], k, rid[d]) for n in stored for k, d in enumerate(digs(n))] + [(rid[n], SUCC, rid[n + 1]) for n in stored if n + 1 in rid]
E_ = torch.tensor(edges, device=dev); print(f"rows {N} (dense kept {len(set(dense.tolist()) - deleted)}, deleted {len(deleted)}, sparse {len(sparse)})  edges {len(edges)}  relations {NR} (+reverses)", flush=True)
model = ResonatE(N, 2 * NR, k=a.k, block=True, block_size=a.block, tied_reverse=a.adjoint).to(dev); M = model.m
def hop(z, r): return model.hop(z, torch.full((z.shape[0],), r, device=dev, dtype=torch.long))
def light(z): return torch.real(z @ cnorm(model.E).conj().t()) * model.log_tau.exp()
opt = torch.optim.Adam(model.parameters(), lr=a.lr); start = 1; ckpt = HERE / a.out / "ckpt.pt"; (HERE / a.out).mkdir(parents=True, exist_ok=True)
if a.resume and ckpt.is_file():
    ck = torch.load(ckpt, map_location=dev, weights_only=False); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start = ck["step"] + 1; print(f"resumed at {start}", flush=True)
for step in range(start, a.steps + 1):
    b = E_[torch.randint(0, len(E_), (a.batch,), device=dev)]; h, r, t = b[:, 0], b[:, 1], b[:, 2]
    zf = model.hop(model.embed(h), r); zr = model.hop(model.embed(t), r + NR)
    loss = F.cross_entropy(light(zf), t) + (0 if a.adjoint else F.cross_entropy(light(zr), h))
    if a.chain > 1:
        idx = torch.tensor([rid[n] for n in range(20)], device=dev); z = model.embed(idx)
        for d in range(2, a.chain + 1):
            z = hop(z, SUCC); loss = loss + F.cross_entropy(light(z[:20 - d]), idx[d:]) / (a.chain - 1)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0 or step == 1:
        with torch.no_grad(): acc = (light(zf).argmax(1) == t).float().mean().item(); mrr = (1.0 / (1 + (light(zf) > light(zf).gather(1, t[:, None])).sum(1)).float()).mean().item()
        eta = (time.time() - t0) / max(step - start + 1, 1) * (a.steps - step)
        print(f"step {step}/{a.steps}  loss {loss.item():.4f}  forward exact {acc:.3f}  mrr {mrr:.3f}  lr {a.lr}  tau {model.log_tau.exp().item():.1f}  elapsed {time.time()-t0:.0f}s  ETA {eta:.0f}s", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "config": vars(a), "stored": stored, "deleted": sorted(deleted)}, ckpt)
model.eval(); res = {"config": vars(a), "rows": N, "edges": len(edges), "deleted": len(deleted)}
with torch.no_grad():
    Eu = cnorm(model.E); D = Eu[[rid[d] for d in range(10)]]                              # the ten digit rows
    def compose(ns):
        """AND of the seven digit probes, batched. A spot when n has no row."""
        Dg = torch.tensor([digs(n) for n in ns], device=dev); z = 0
        for k in range(NP): z = z + hop(D[Dg[:, k]], k + NR)
        return cnorm(z)
    def state(ns): return torch.stack([Eu[rid[n]] if n in rid else compose([n])[0] for n in ns])   # row if stored, else its spot
    def decode(z):
        ds = torch.stack([torch.real(hop(z, k) @ D.conj().t()).argmax(1) for k in range(NP)], 1).cpu().numpy()
        return [int(sum(int(d) * 10 ** k for k, d in enumerate(row))) for row in ds]
    L19 = Eu[[rid[n] for n in range(20)]]
    def add_digits(da, db, carry):
        z = D[da][None]
        for _ in range(db + carry): z = hop(z, SUCC)
        v = int(torch.real(z @ L19.conj().t()).argmax()); return v % 10, v // 10
    def add(za, zb):
        A_, B_ = decode(za)[0], decode(zb)[0]; da, db = digs(A_), digs(B_); carry, out = 0, []           # digits read from the areas by the place relations
        for k in range(NP): u, carry = add_digits(da[k], db[k], carry); out.append(u)
        n = sum(u * 10 ** k for k, u in enumerate(out)); return compose([n]), n
    # ---------------- (a) unstored numbers: does the AND-spot decode exactly? by length; plus controls
    held = {L: [n for n in sorted(deleted) if len(str(n)) == L] for L in (2, 3, 4)}
    for L in (5, 6, 7): held[L] = [int(x) for x in rng.integers(10 ** (L - 1), 10 ** L, 500) if int(x) not in rid]
    for L, ns in held.items():
        if not ns: continue
        ex = float(np.mean([d == n for d, n in zip(decode(compose(ns)), ns)])); res[f"spot_decode_len{L}"] = {"n": len(ns), "exact": ex}
        print(f"unstored numbers, {L} digits: n={len(ns):4d}  AND-spot decodes exactly {ex:.3f}", flush=True)
    st = [n for n in stored if n >= 20][:3000]; ex_row = float(np.mean([d == n for d, n in zip(decode(Eu[[rid[n] for n in st]]), st)])); ex_spot = float(np.mean([d == n for d, n in zip(decode(compose(st)), st)]))
    res["stored_decode"] = {"from_row": ex_row, "from_spot": ex_spot}; print(f"stored numbers: decode from row {ex_row:.3f}, from their AND-spot {ex_spot:.3f}", flush=True)
    rnd = cnorm(torch.randn(500, M, dtype=torch.complex64, device=dev)); res["control_random_state_decode"] = float(np.mean([d == n for d, n in zip(decode(rnd), held[4][:500])])); print(f"control: random states decode as the intended number {res['control_random_state_decode']:.3f}", flush=True)
    # spot of a deleted number vs the row it would have had? none exists; instead: spot consistency = distinct spots for distinct numbers
    zs = compose(held[4][:1000]); S = torch.real(zs @ zs.conj().t()); S.fill_diagonal_(-1); res["spot_nearest_other_cos"] = S.max(1).values.mean().item(); print(f"spots of 1000 unstored 4-digit numbers: nearest-other cosine {res['spot_nearest_other_cos']:.3f}", flush=True)
    # ---------------- (b) addition program, by result length, operands mostly unstored
    for L in range(1, 8):
        pairs = []
        while len(pairs) < 300:
            s = int(rng.integers(10 ** (L - 1), 10 ** L)); x = int(rng.integers(0, s + 1)); pairs.append((x, s - x))
        ok = [add(state([x])[0][None], state([y])[0][None])[1] == x + y for x, y in pairs]; both_unstored = [x not in rid and y not in rid for x, y in pairs]
        res[f"add_len{L}"] = {"n": len(pairs), "exact": float(np.mean(ok)), "share_both_operands_unstored": float(np.mean(both_unstored))}
        print(f"a+b with {L}-digit result: n={len(pairs)}  exact {np.mean(ok):.3f}  (operands both unstored: {np.mean(both_unstored):.2f})", flush=True)
    # ---------------- (c) the other agent's dev_v1 buckets, same program (x10^k = shift of digits; general x = repeated addition)
    def evaluate(e):
        if isinstance(e, str): return state([int(e)])[0][None]
        vals = [evaluate(x) for x in e["args"]]
        if e["op"] == "add":
            acc = vals[0]
            for v in vals[1:]: acc = add(acc, v)[0]
            return acc
        acc = vals[0]
        for v in vals[1:]:
            dv, da_ = decode(v)[0], decode(acc)[0]
            if dv and 10 ** (len(str(dv)) - 1) == dv: acc = compose([da_ * dv])                          # x 10^k: the digits move up k places
            elif da_ and 10 ** (len(str(da_)) - 1) == da_: acc = compose([dv * da_])
            else:
                small, big_ = (dv, acc) if dv <= da_ else (da_, v); s = D[0][None]
                for _ in range(small): s = add(s, big_)[0]
                acc = s
        return acc
    for split in ("validation", "test"):
        per = collections.defaultdict(list)
        for r in records(split):
            mi = model_input(r)
            if r["task"] == "compose": ok = decode(evaluate(mi.inputs["expression"]))[0] == int("".join(r["target"]["digits"]))
            elif r["task"] == "decompose":
                n = int(mi.number_literals[0]); dg = digs(decode(state([n])[0][None])[0]); pred = sorted((d, k) for k, d in enumerate(dg) if d); tg = sorted((int(t["coefficient"]), int(t["power10"])) for t in r["target"]["terms"] if t["coefficient"] != "0"); ok = pred == tg or (n == 0 and not pred)
            else: continue
            per[f"{r['task']} {'+'.join(r['metadata']['buckets'])} [{r['metadata']['pool']}]"].append(ok)
        res[split] = {bk: {"n": len(v), "exact": float(np.mean(v))} for bk, v in sorted(per.items())}
        print(f"== dev_v1 {split}"); [print(f"  {bk:62s} n={o['n']:4d}  exact {o['exact']:.3f}", flush=True) for bk, o in res[split].items()]
save_json(res, HERE / a.out / "number_kg.json"); torch.save({"model": model.state_dict(), "config": vars(a), "stored": stored, "deleted": sorted(deleted)}, HERE / a.out / "memory.pt"); print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)
