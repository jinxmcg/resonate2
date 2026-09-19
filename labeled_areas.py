"""Numbers as labeled areas, constructed the way their labels are.
  rows      : the ten digit symbols 0-9 — the only stored entities.
  operators : S (successor / counting), and the scales T (tens), H (hundreds), K (thousands), X (ten-thousands);
              tied_reverse gives the exact adjoint of every operator, so unbinding is the memory's own inverse hop.
  area(n)   : cnorm( sum over non-zero places k of hop(E_{d_k}, scale_k) )   (units: no operator; 0 itself is E_0)
              "thirty" = 3 bound to tens.  "347" = 3 hundreds + 4 tens + 7.  No number above 9 has a row.
  decode    : place k of an area z = argmax over the digit rows of light(hop(z, scale_k^-1)), or 0 if that light is
              below a threshold calibrated once on the grounding numerals (absent place).
The memory learns only label-level facts: counting 0..19 (S maps area(n) to area(n+1)) and that the grammar's binding
is invertible (unbinding a constructed area recovers its digits). It never sees + or x.
Step 2: constructed areas of never-seen numbers are distinct and decode exactly.
Step 3: a FIXED program over these primitives (no learned parameters) performs a+b with carry: per place, unbind both
digits, walk the successor line, read units directly and the carry by unbinding tens, rebind; multiply(d, 10^k) is a
binding; general multiply is repeated addition. Scored exactly by decoding the result area — no row, no label needed.
"""
import argparse, json, time, collections, numpy as np, torch, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=8); ap.add_argument("--block", type=int, default=16); ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--count-to", type=int, default=19); ap.add_argument("--out", default="results/numbers_k8")
ap.add_argument("--no-train", action="store_true", help="random unitary operators and random rows: is the algebra alone enough?"); ap.add_argument("--resume", action="store_true"); ap.add_argument("--unitary", action="store_true", help="operators parametrised as exp(A - A^H): every hop is a rotation, unbinding is exact"); ap.add_argument("--chain", type=int, default=1, help="counting objective also on S^d chains up to d (counting by d; still counting)")
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time()
SC = ["T", "H", "K", "X"]; NOPS = 1 + len(SC)                                             # op ids: 0 = S, 1..4 = scales; +NOPS = adjoint
model = ResonatE(10, 2 * NOPS, k=a.k, block=True, block_size=a.block, tied_reverse=True).to(dev); M = model.m
E = lambda: cnorm(model.E)
if a.unitary:                                                                                  # blocks U = exp(A - A^H); adjoint for ids >= NOPS
    nblk = M // a.block; A = torch.nn.Parameter(0.7 * torch.randn(NOPS, nblk, a.block, a.block, dtype=torch.complex64, device=dev)); model.H.requires_grad_(False)
    def U(): return torch.linalg.matrix_exp(A - A.conj().transpose(-1, -2))
    def hop(z, r):
        h = U()[r % NOPS]; h = h.conj().transpose(-1, -2) if r >= NOPS else h
        return cnorm(torch.einsum("kij,bkj->bki", h, z.reshape(z.shape[0], nblk, a.block)).reshape(z.shape[0], -1))
else:
    def hop(z, r): return model.hop(z, torch.full((z.shape[0],), r, device=dev, dtype=torch.long))
def digits(n): return [int(c) for c in reversed(str(int(n)))]                              # place 0 first


def area(ns):
    """constructed areas for a list of ints, batched: one hop per place for all numerals at once. (B, M)"""
    Er = E(); D = torch.tensor([[int(c) for c in f"{int(n):05d}"[::-1]] for n in ns], device=dev)      # (B, 5) place 0 first
    keep = (D > 0).float(); keep[:, 0] = ((D[:, 0] > 0) | (D.sum(1) == 0)).float()                     # zero digits omitted; 0 itself is E_0
    z = Er[D[:, 0]] * keep[:, 0:1]
    for k in range(1, 5): z = z + hop(Er[D[:, k]], k) * keep[:, k:k + 1]
    return cnorm(z)


def place_lights(z, k):
    """cosine light over the ten digit rows of place k of z (k=0: direct)."""
    return torch.real((z if k == 0 else hop(z, k + NOPS)) @ E().conj().t())


# ------------------------------------------------------------------ step 1: the memory learns counting and the grammar
ground = sorted({int(x) for x in grounded_labels()} | set(range(a.count_to + 1)))          # numerals whose labels the memory sees
count_pairs = [(n, n + 1) for n in range(a.count_to)]
opt = torch.optim.Adam(list(model.parameters()) + ([A] if a.unitary else []), lr=a.lr); tau = model.log_tau
steps = 0 if a.no_train else a.steps; start = 1; ckpt = HERE / a.out / "ckpt.pt"
if a.resume and ckpt.is_file():
    ck = torch.load(ckpt, map_location=dev, weights_only=False); model.load_state_dict(ck["model"]); (A.data.copy_(ck["A"]) if a.unitary else None); opt.load_state_dict(ck["opt"]); start = ck["step"] + 1; torch.set_rng_state(ck["rng"]); print(f"resumed from step {ck['step']}", flush=True)
(HERE / a.out).mkdir(parents=True, exist_ok=True)
for step in range(start, steps + 1):
    line = area(list(range(a.count_to + 1)))                                               # the counting candidates (areas, not rows)
    nxt = hop(line[:-1], 0); loss_c = F.cross_entropy(torch.real(nxt @ line.conj().t()) * tau.exp(), torch.arange(1, a.count_to + 1, device=dev))
    z = line
    for d in range(2, a.chain + 1):                                                            # S^d from area(n) must land on area(n+d)
        z = hop(z, 0); loss_c = loss_c + F.cross_entropy(torch.real(z[:-d] @ line.conj().t()) * tau.exp(), torch.arange(d, a.count_to + 1, device=dev)) / a.chain
    zs = area(ground); loss_u = 0.0; n_u = 0
    for k in range(5):
        L = place_lights(zs, k); d = torch.tensor([digits(n)[k] if k < len(digits(n)) else 0 for n in ground], device=dev)
        present = d > 0 if k > 0 else torch.ones_like(d, dtype=torch.bool)
        if present.any(): loss_u = loss_u + F.cross_entropy(L[present] * tau.exp(), d[present]); n_u += 1
        if (~present).any(): loss_u = loss_u + F.relu(L[~present].max(1).values - 0.25).mean()   # absent place: nothing bright
    loss = loss_c + loss_u / max(n_u, 1); opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0 or step == 1:
        with torch.no_grad(): acc_c = (torch.real(nxt @ line.conj().t()).argmax(1) == torch.arange(1, a.count_to + 1, device=dev)).float().mean().item()
        eta = (time.time() - t0) / max(step - start + 1, 1) * (steps - step)
        print(f"step {step}/{steps}  loss {loss.item():.4f} (count {loss_c.item():.3f} unbind {float(loss_u)/max(n_u,1):.3f})  counting exact {acc_c:.3f}  lr {a.lr}  tau {tau.exp().item():.1f}  elapsed {time.time()-t0:.0f}s  ETA {eta:.0f}s", flush=True)
        torch.save({"model": model.state_dict(), "A": (A.detach() if a.unitary else None), "opt": opt.state_dict(), "step": step, "rng": torch.get_rng_state(), "config": vars(a)}, ckpt)
model.eval(); res = {"config": vars(a), "rows": 10, "operators": ["S"] + SC, "grounding_numerals": len(ground)}
with torch.no_grad():
    # threshold for "absent place": midpoint between the weakest present digit light and the strongest absent light, on the grounding numerals
    zs = area(ground); pres, absn = [], []
    for k in range(1, 5):
        L = place_lights(zs, k)
        for i, n in enumerate(ground):
            d = digits(n)[k] if k < len(digits(n)) else 0
            (pres if d else absn).append(L[i, d].item() if d else L[i].max().item())
    theta = (min(pres) + max(absn)) / 2 if pres and absn else 0.3; res["threshold"] = {"theta": theta, "min_present": min(pres), "max_absent": max(absn)}
    print(f"absent-place threshold {theta:.3f} (weakest present {min(pres):.3f}, strongest absent {max(absn):.3f})")

    def decode(z):
        out = []
        for i in range(z.shape[0]):
            n = 0
            for k in range(5):
                L = place_lights(z[i:i + 1], k)[0]; d = int(L.argmax()); n += d * 10 ** k if (k == 0 or L[d] >= theta) else 0
            out.append(n)
        return out

    # line check: does the successor act as a line on 0..count_to? and is it a rotation-like (near-unitary) operator?
    line = area(list(range(a.count_to + 1))); nxt = hop(line[:-1], 0); res["counting_exact"] = float((torch.real(nxt @ line.conj().t()).argmax(1).cpu() == torch.arange(1, a.count_to + 1)).float().mean())
    # ---------------------------------------------------------------- step 2: unseen numbers, no rows: distinct and exactly decodable
    seen = set(ground); pools = manifest()["pools"]; C = [n for n in pools["C_test"] if n not in seen]; big = [int(x) for x in rng.integers(10000, 100000, 2000)]
    for name, ns in (("grounding", ground), ("C_test (never seen)", rng.choice(C, 2000, replace=False).tolist()), ("10000-99999 (extrapolation)", big)):
        z = area(ns); dec = decode(z); ex = float(np.mean([d == n for d, n in zip(dec, ns)]))
        S = torch.real(z @ z.conj().t()); S.fill_diagonal_(-1); nn_cos = S.max(1).values
        res[f"construct_{name}"] = {"n": len(ns), "decode_exact": ex, "nearest_other_cos_mean": nn_cos.mean().item(), "nearest_other_cos_max": nn_cos.max().item()}
        print(f"construct {name:28s} n={len(ns):5d}  decode exact {ex:.3f}  nearest-other cosine mean {nn_cos.mean():.3f} max {nn_cos.max():.3f}")

    # ---------------------------------------------------------------- step 3: the fixed program over the primitives
    def add_digit_areas(za, db, carry):
        """walk the successor line: area of (digit a) + db + carry, all in-space; returns (units digit, carry digit) read by lights"""
        z = za
        for _ in range(db + carry): z = hop(z, 0)
        Lu = place_lights(z, 0)[0]; Lt = place_lights(z, 1)[0]; u = int(Lu.argmax()); t = int(Lt.argmax()); return u, (t if Lt[t] >= theta else 0)

    def add(za, zb):
        """a + b as a program: per place unbind both digits, add on the line, carry by unbinding tens, rebind."""
        carry, parts, Er = 0, [], E()
        for k in range(5):
            La = place_lights(za, k)[0]; Lb = place_lights(zb, k)[0]
            da = int(La.argmax()) if (k == 0 or La[La.argmax()] >= theta) else 0
            db = int(Lb.argmax()) if (k == 0 or Lb[Lb.argmax()] >= theta) else 0
            u, carry = add_digit_areas(Er[da][None], db, carry)
            if u: parts.append(Er[u] if k == 0 else hop(Er[u][None], k)[0])
        # a carry out of the fifth place has no scale operator here (numbers stay below 100000); it is dropped and counted as an error
        if not parts: return Er[0][None]
        return cnorm(torch.stack(parts).sum(0)[None])

    def shift(z, k):
        """multiply by 10^k: rebind every place k up (unbind, rebind) — the label grammar moved, nothing computed"""
        Er = E(); parts = []
        for j in range(5 - k):
            L = place_lights(z, j)[0]; d = int(L.argmax())
            if d and (j == 0 or L[d] >= theta): parts.append(hop(Er[d][None], j + k)[0])
        return cnorm(torch.stack(parts).sum(0)[None]) if parts else Er[0][None]

    def evaluate(e):
        if isinstance(e, str): return area([int(e)])
        vals = [evaluate(x) for x in e["args"]]
        if e["op"] == "add":
            acc = vals[0]
            for v in vals[1:]: acc = add(acc, v)
            return acc
        acc = vals[0]
        for v in vals[1:]:
            dv = decode(v)[0]; da = decode(acc)[0]
            if dv and 10 ** (len(str(dv)) - 1) == dv: acc = shift(acc, len(str(dv)) - 1)               # x 10^k: a binding
            elif da and 10 ** (len(str(da)) - 1) == da: acc = shift(v, len(str(da)) - 1)
            else:                                                                                       # general product: repeated addition of the smaller count
                small, big_ = (dv, acc) if dv <= da else (da, v); s = E()[0][None]
                for _ in range(small): s = add(s, big_)
                acc = s
        return acc

    for split in ("validation", "test", "train"):
        per = collections.defaultdict(list); steps_ = collections.defaultdict(list)
        for r in records(split):
            if r["task"] != "compose": continue
            mi = model_input(r); z = evaluate(mi.inputs["expression"]); ok = decode(z)[0] == int("".join(r["target"]["digits"]))
            per["+".join(r["metadata"]["buckets"]) + f" [{r['metadata']['pool']}]"].append(ok)
        for r in records(split):
            if r["task"] != "decompose": continue
            mi = model_input(r); z = area([int(mi.number_literals[0])]); pred = [(int(place_lights(z, k)[0].argmax()), k) for k in range(5)]
            pred = [(d, k) for d, k in pred if d and (k == 0 or place_lights(z, k)[0][d] >= theta)]; tg = sorted((int(t["coefficient"]), int(t["power10"])) for t in r["target"]["terms"] if t["coefficient"] != "0")
            per[f"decompose [{r['metadata']['pool']}]"].append(sorted(pred) == tg or (int(mi.number_literals[0]) == 0 and not pred))
        res[split] = {bk: {"n": len(v), "exact": float(np.mean(v))} for bk, v in sorted(per.items())}
        print(f"== {split}"); [print(f"  {bk:55s} n={o['n']:4d}  exact {o['exact']:.3f}") for bk, o in res[split].items()]
save_json(res, HERE / a.out / "numbers.json"); torch.save({"state": model.state_dict(), "A": (A.detach() if a.unitary else None), "config": vars(a), "theta": theta}, HERE / a.out / "memory.pt"); print(f"saved {a.out} ({time.time()-t0:.0f}s)")
