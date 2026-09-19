"""A number as a region with two parts: how it is BUILT (path) and how much it IS (value).
path part  : the directed chain of two_relations.py (loaded frozen from results/two_relations_k12_walk2): composition by
             superposition + direction, decomposition by walking back one light per step — exact.
value part : 63 term anchors d x 10^k (d = 1..9, k = 0..6) as unit-modulus phase vectors v(d,k) = exp(i phi[d,k,:]);
             a number's value = the BINDING (elementwise product) of its term anchors: v_347 = v(3,2) (.) v(4,1) (.) v(7,0).
             Binding is multiplicative, so phases add: if the anchors form a line, v_a (.) v_b = v_{a+b} exactly.
             Learned ONLY from single-place digit equations (a*10^k + b*10^k = (a+b)*10^k, carries included, 20% held out)
             plus 'different numbers are different' (InfoNCE over random numbers). Nothing multi-place is trained.
'=' is the light : the value region of the expression a+b must shine on the value region of the numeral a+b and on
             nothing nearby (rank among a+b +- 1..9, 10..90, ..., 10^6).
addition   : v_a (.) v_b, then digits read from the value by lights (per place from the top: which of the ten candidate
             terms, unbound, leaves a remainder that lights the 'numbers below 10^k' region), then the path chain is
             rebuilt from those digits and walked back. Exact = all seven digits right on never-trained sums.
Controls   : random (untrained) phases; the oracle line phi = theta_j * d * 10^k."""
import argparse, time, numpy as np, torch, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--mv", type=int, default=64); ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--lr", type=float, default=1e-2)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--mode", default="learn", choices=["learn", "random", "oracle"]); ap.add_argument("--path", default="results/two_relations_k12_walk2/memory.pt"); ap.add_argument("--out", default="results/two_part")
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time()
# ------------------------------------------------------------------ path part (frozen, exact)
NODES = [str(d) for d in range(10)] + [str(10 ** k) for k in range(1, 7)]; nid = {n: i for i, n in enumerate(NODES)}; PLUS, TIMES = 0, 1; NR = 2
pk = torch.load(HERE / a.path, map_location=dev, weights_only=False); pcfg = pk["config"]; pmodel = ResonatE(len(NODES), 2 * NR, k=pcfg["k"], block=True, block_size=pcfg["block"]).to(dev); pmodel.load_state_dict(pk["model"]); pmodel.eval(); pmodel.requires_grad_(False)
ANCH = torch.tensor([nid["0"]] + [nid[str(10 ** k)] for k in range(1, 7)], device=dev)
def phop(z, r): return pmodel.hop(z, torch.full((z.shape[0],), r, device=dev, dtype=torch.long))
def plight(z): return torch.real(z @ cnorm(pmodel.E).conj().t())
def digs(ns): return torch.tensor([[int(c) for c in f"{int(n):07d}"] for n in ns], device=dev)
def path_chain(D):
    Er = cnorm(pmodel.E); B = D.shape[0]; acc = None
    for j in range(7):
        k = 6 - j; d = D[:, j]; term = Er[d] if k == 0 else phop(cnorm(Er[d] + Er[ANCH[k]][None].expand(B, -1)), TIMES)
        acc = phop(term, PLUS) if acc is None else phop(cnorm(acc + term), PLUS)
    return acc
def path_walk(z):
    B = z.shape[0]; out = torch.zeros(B, 7, dtype=torch.long, device=dev); rr = z
    for k in range(7):
        rr = phop(rr, PLUS + NR)
        if k == 0: out[:, 0] = plight(rr).argmax(1)
        else: L = plight(phop(rr, TIMES + NR)); L[:, 10:] = -1e9; out[:, k] = L.argmax(1)
    return [int(sum(int(row[k]) * 10 ** k for k in range(7))) for row in out.cpu().numpy()]
# ------------------------------------------------------------------ value part
phi = torch.nn.Parameter((torch.rand(10, 7, a.mv, device=dev) * 2 - 1) * np.pi)                 # phi[d, k, :]; d = 0 is fixed to phase 0 (no term)
if a.mode == "oracle":
    theta = torch.tensor(np.exp(rng.uniform(np.log(2 * np.pi / 3e7), np.log(2 * np.pi / 3), a.mv)), dtype=torch.float32, device=dev)
    with torch.no_grad(): phi.copy_(torch.tensor([[d * 10 ** k for k in range(7)] for d in range(10)], dtype=torch.float32, device=dev)[:, :, None] * theta)
log_t = torch.nn.Parameter(torch.tensor(np.log(10.0), device=dev))
def anchors():
    p = phi.clone(); p[0] = 0; return torch.polar(torch.ones_like(p), p)                            # (10, 7, mv) unit modulus
def value(ns):
    """binding of the term anchors of each number. (B, mv)"""
    V = anchors(); D = digs(ns); v = torch.ones(len(ns), a.mv, dtype=torch.complex64, device=dev)
    for j in range(7): v = v * V[D[:, j], 6 - j]
    return v
def vcos(x, y): return torch.real((x * y.conj()).mean(-1))                                          # cosine of unit-modulus vectors
facts = [(x, y, k) for x in range(10) for y in range(10) for k in range(7) if (x + y) * 10 ** k < 10 ** 7]; rng.shuffle(facts); n_hold = len(facts) // 5; facts_test, facts_train = facts[:n_hold], facts[n_hold:]
opt = torch.optim.Adam([phi, log_t], lr=a.lr); (HERE / a.out).mkdir(parents=True, exist_ok=True); steps = a.steps if a.mode == "learn" else 0
for step in range(1, steps + 1):
    fb = [facts_train[i] for i in rng.integers(0, len(facts_train), 512)]; xa = [x * 10 ** k for x, y, k in fb]; xb = [y * 10 ** k for x, y, k in fb]; xc = [(x + y) * 10 ** k for x, y, k in fb]
    loss_eq = (1 - vcos(value(xa) * value(xb), value(xc))).mean()                                   # the equation: binding lands on the numeral's value
    ns = [int(x) for x in rng.integers(0, 10 ** 7, 1024)]; v = value(ns); S = torch.real(v @ v.conj().t()) / a.mv * log_t.exp(); loss_d = F.cross_entropy(S, torch.arange(len(ns), device=dev))
    loss = loss_eq + loss_d; opt.zero_grad(); loss.backward(); opt.step()
    if step % 250 == 0 or step == 1:
        eta = (time.time() - t0) / step * (steps - step); print(f"step {step}/{steps}  equation loss {loss_eq.item():.4f}  distinctness loss {loss_d.item():.4f}  lr {a.lr}  t {log_t.exp().item():.1f}  elapsed {time.time()-t0:.0f}s  ETA {eta:.0f}s", flush=True)
        torch.save({"phi": phi.detach(), "log_t": log_t.detach(), "step": step, "config": vars(a)}, HERE / a.out / "ckpt.pt")
res = {"config": vars(a), "facts_train": len(facts_train), "facts_test": len(facts_test)}
with torch.no_grad():
    # emergence: is each frequency's phase a line in the value?  corr(phi[d,k,j], d*10^k) after unwrapping is not well defined; use the
    # held-out additivity instead (below) and the local test: v(d,k) (.) v(1,k) vs v(d+1,k)
    loc = [vcos(anchors()[d, k] * anchors()[1, k], anchors()[d + 1, k]).item() for d in range(1, 9) for k in range(7)]; res["local_line_cos"] = float(np.mean(loc))
    print(f"anchor line check  v(d,k)(.)v(1,k) vs v(d+1,k): mean cosine {np.mean(loc):.3f}", flush=True)
    # '=' as the light: the expression's value must shine on the numeral's value and beat its neighbours
    def neighbours(c): return [c + s * m for m in (1, 10, 100, 1000, 10000, 100000, 1000000) for s in (-9, -5, -2, -1, 1, 2, 5, 9) if 0 <= c + s * m < 10 ** 7 and s * m != 0]
    def eq_light(pairs):
        top, cs = [], []
        for x, y in pairs:
            c = x + y; cand = [c] + neighbours(c); e = value([x]) * value([y]); L = torch.real((e * value(cand).conj()).mean(-1)); top.append(int(L.argmax()) == 0); cs.append(L[0].item())
        return float(np.mean(top)), float(np.mean(cs))
    for name, fs in (("digit equations (trained)", facts_train), ("digit equations (held out)", facts_test)):
        t1, c = eq_light([(x * 10 ** k, y * 10 ** k) for x, y, k in fs]); res[name] = {"n": len(fs), "light_points_to_sum": t1, "cos": c}
        print(f"{name:30s} n={len(fs):4d}  '=' light points to the sum (vs neighbours) {t1:.3f}   cos(expression, numeral) {c:.3f}", flush=True)
    # reading digits off a value by lights: per place from the top, unbind each candidate term, keep the one whose remainder lights 'below 10^k'
    below = {}
    for k in range(1, 7):
        r = [int(x) for x in rng.integers(0, 10 ** k, 8192)]; below[k] = value(r).mean(0)                # the region 'numbers below 10^k' (superposition of their values)
    below[0] = value([0])[0]
    def read_value(v):
        B = v.shape[0]; out = torch.zeros(B, 7, dtype=torch.long, device=dev); V = anchors(); rem = v
        for k in range(6, -1, -1):
            cand = V[:, k][None].expand(B, -1, -1); r = rem[:, None, :] * cand.conj(); sc = torch.real((r * below[k][None, None, :].conj()).mean(-1)); d = sc.argmax(1)
            out[:, k] = d; rem = r[torch.arange(B), d]
        return [int(sum(int(row[k]) * 10 ** k for k in range(7))) for row in out.cpu().numpy()]
    ns = [int(x) for x in rng.integers(0, 10 ** 7, 2000)]; res["value_read_exact"] = float(np.mean([d == n for d, n in zip(read_value(value(ns)), ns)])); print(f"digits read off a value by lights (no addition): exact {res['value_read_exact']:.3f}", flush=True)
    # addition: bind the values, read the digits, rebuild the path chain, walk it back — exact on never-trained sums, by length and carries
    def n_carries(x, y):
        c = n = 0; dx, dy = f"{x:07d}"[::-1], f"{y:07d}"[::-1]
        for i in range(7): s = int(dx[i]) + int(dy[i]) + c; c = 1 if s >= 10 else 0; n += c
        return n
    for L in range(1, 8):
        pairs = []
        while len(pairs) < 600:
            s = int(rng.integers(10 ** (L - 1), 10 ** L)); x = int(rng.integers(0, s + 1)); pairs.append((x, s - x))
        got = read_value(value([p[0] for p in pairs]) * value([p[1] for p in pairs])); ok = [g == x + y for g, (x, y) in zip(got, pairs)]
        walked = path_walk(path_chain(digs(got))); loop_ok = [w == x + y for w, (x, y) in zip(walked, pairs)]; nc = [n_carries(x, y) for x, y in pairs]
        row = {"n": len(pairs), "exact": float(np.mean(ok)), "exact_after_path_walk": float(np.mean(loop_ok))}
        for c in (0, 1, 2, 3):
            sel = [o for o, k in zip(ok, nc) if (k == c if c < 3 else k >= 3)]
            if sel: row[f"carries_{c}{'+' if c == 3 else ''}"] = {"n": len(sel), "exact": float(np.mean(sel))}
        res[f"sum_len{L}"] = row
        print(f"{L}-digit sums (never trained): exact {np.mean(ok):.3f}  (after rebuilding the path and walking back {np.mean(loop_ok):.3f})   " + "  ".join(f"{k.replace('carries_', 'carries=')} {v['exact']:.3f}(n={v['n']})" for k, v in row.items() if k.startswith("carries")), flush=True)
    for x, y in ((23, 45), (47, 35), (123456, 111111), (999999, 1)):
        g = read_value(value([x]) * value([y]))[0]; print(f"{x} + {y} = {x + y}:   value read {g}   path walk {path_walk(path_chain(digs([g])))[0]}", flush=True)
save_json(res, HERE / a.out / "two_part.json"); torch.save({"phi": phi.detach(), "log_t": log_t.detach(), "config": vars(a)}, HERE / a.out / "value.pt"); print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)
