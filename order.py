"""Stage 2 on the value code: add ORDER as new facts to the already-trained model (no restart), measure gain and retention.
Start from results/two_part_learn/value.pt (phases that already add exactly; no order in them).
New facts: x < y on random pairs at every scale. The comparison is a light: by additivity the difference is
v_y (.) conj(v_x) = v_{y-x}; 'x < y' means that difference shines on a learnable POSITIVE region P (one vector).
Replay: the single-place digit equations (explicit). No distinctness term (it is what killed the order before).
After: retention of '=' (held-out digit equations, unseen multi-digit sums); comparison on unseen pairs by distance;
cos(n, n+1); and the closed loop — digits read off a value by COMPARISON lights (largest d with d*10^k <= remainder),
the path chain rebuilt and walked back, exact on never-trained sums."""
import argparse, time, numpy as np, torch, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--init", default="results/two_part_learn/value.pt"); ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--path", default="results/two_relations_k12_walk2/memory.pt"); ap.add_argument("--out", default="results/order"); ap.add_argument("--slopes", action="store_true", help="reparametrise the emerged line: phi[d,k,j] = d*10^k*theta_j with theta_j free (init from the learned units slope)")
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time()
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
# ------------------------------------------------------------------ the trained value code, continued
ck = torch.load(HERE / a.init, map_location=dev, weights_only=False); phi = torch.nn.Parameter(ck["phi"].clone()); mv = phi.shape[-1]
P_re = torch.nn.Parameter(torch.randn(mv, 2, device=dev) / np.sqrt(mv)); P = None; log_t = torch.nn.Parameter(torch.tensor(np.log(10.0), device=dev))
if a.slopes:
    theta = torch.nn.Parameter(ck["phi"][1, 0].clone()); scale = torch.tensor([[d * 10 ** k for k in range(7)] for d in range(10)], dtype=torch.float32, device=dev)
    def anchors(): return torch.polar(torch.ones(10, 7, mv, device=dev), scale[:, :, None] * theta[None, None, :])
else:
    def anchors(): p = phi.clone(); p[0] = 0; return torch.polar(torch.ones_like(p), p)
def value(ns):
    V = anchors(); D = digs(ns); v = torch.ones(len(ns), mv, dtype=torch.complex64, device=dev)
    for j in range(7): v = v * V[D[:, j], 6 - j]
    return v
def vcos(x, y): return torch.real((x * y.conj()).mean(-1))
def less(vx, vy): return torch.real(((vy * vx.conj()) * torch.view_as_complex(P_re).conj()).mean(-1)) * log_t.exp()       # 'x < y' as a light: the difference shines on P
facts = [(x, y, k) for x in range(10) for y in range(10) for k in range(7) if (x + y) * 10 ** k < 10 ** 7]; rng.shuffle(facts); n_hold = len(facts) // 5; facts_test, facts_train = facts[:n_hold], facts[n_hold:]
def cmp_pairs(n):
    x = rng.integers(0, 10 ** 7, n); u = rng.integers(0, 7, n); d = rng.integers(1, 10 ** u + 1) * rng.choice([-1, 1], n); y = np.clip(x + d, 0, 10 ** 7 - 1)
    keep = y != x; return [int(v) for v in x[keep]], [int(v) for v in y[keep]]
def eq_light(pairs):
    def neighbours(c): return [c + s * m for m in (1, 10, 100, 1000, 10000, 100000, 1000000) for s in (-9, -5, -2, -1, 1, 2, 5, 9) if 0 <= c + s * m < 10 ** 7 and s * m != 0]
    top = []
    for x, y in pairs:
        c = x + y; cand = [c] + neighbours(c); e = value([x]) * value([y]); top.append(int(torch.real((e * value(cand).conj()).mean(-1)).argmax()) == 0)
    return float(np.mean(top))
def cmp_eval(n=3000):
    xs, ys = cmp_pairs(n); s = less(value(xs), value(ys)); ok = ((s > 0) == torch.tensor([y > x for x, y in zip(xs, ys)], device=dev)).cpu().numpy(); dist = np.abs(np.array(ys) - np.array(xs))
    by = {}
    for o, d in zip(ok, dist): by.setdefault(int(np.floor(np.log10(d))), []).append(o)
    return float(ok.mean()), {f"10^{k}": float(np.mean(v)) for k, v in sorted(by.items())}
with torch.no_grad():
    before = {"eq_heldout": eq_light([(x * 10 ** k, y * 10 ** k) for x, y, k in facts_test]), "cmp": cmp_eval()[0], "cos_n_n1": vcos(value(list(range(1000, 1500))), value(list(range(1001, 1501)))).mean().item()}
print(f"BEFORE stage 2: '=' light held-out {before['eq_heldout']:.3f}   x<y (untrained P, chance) {before['cmp']:.3f}   cos(n,n+1) {before['cos_n_n1']:.3f}", flush=True)
opt = torch.optim.Adam([theta if a.slopes else phi, P_re, log_t], lr=a.lr); (HERE / a.out).mkdir(parents=True, exist_ok=True)
for step in range(1, a.steps + 1):
    xs, ys = cmp_pairs(1024); vx, vy = value(xs), value(ys); lab = torch.tensor([float(y > x) for x, y in zip(xs, ys)], device=dev)
    loss_c = F.binary_cross_entropy_with_logits(less(vx, vy), lab)
    fb = [facts_train[i] for i in rng.integers(0, len(facts_train), 256)]; loss_eq = (1 - vcos(value([x * 10 ** k for x, y, k in fb]) * value([y * 10 ** k for x, y, k in fb]), value([(x + y) * 10 ** k for x, y, k in fb]))).mean()
    loss = loss_c + loss_eq; opt.zero_grad(); loss.backward(); opt.step()
    if step % 250 == 0 or step == 1:
        with torch.no_grad(): acc = ((less(vx, vy) > 0).float() == lab).float().mean().item()
        eta = (time.time() - t0) / step * (a.steps - step); print(f"step {step}/{a.steps}  x<y loss {loss_c.item():.4f} exact {acc:.3f}   equation replay loss {loss_eq.item():.5f}   lr {a.lr}  t {log_t.exp().item():.1f}  elapsed {time.time()-t0:.0f}s  ETA {eta:.0f}s", flush=True)
        torch.save({"phi": phi.detach(), "P": torch.view_as_complex(P_re).detach(), "log_t": log_t.detach(), "step": step, "config": vars(a)}, HERE / a.out / "ckpt.pt")
res = {"config": vars(a), "before": before}
with torch.no_grad():
    # retention and gain
    res["after"] = {"eq_heldout": eq_light([(x * 10 ** k, y * 10 ** k) for x, y, k in facts_test]), "cos_n_n1": vcos(value(list(range(1000, 1500))), value(list(range(1001, 1501)))).mean().item()}
    c_all, c_by = cmp_eval(6000); res["after"]["cmp"] = c_all; res["after"]["cmp_by_distance"] = c_by
    print(f"AFTER stage 2:  '=' light held-out {res['after']['eq_heldout']:.3f} (retention)   x<y on unseen pairs {c_all:.3f}  by distance " + "  ".join(f"{k} {v:.3f}" for k, v in c_by.items()) + f"   cos(n,n+1) {res['after']['cos_n_n1']:.3f}", flush=True)
    sums = [(int(rng.integers(0, 5 * 10 ** 6)), int(rng.integers(0, 5 * 10 ** 6))) for _ in range(500)]; res["after"]["eq_unseen_sums"] = eq_light(sums); print(f"'=' light on unseen multi-digit sums (retention) {res['after']['eq_unseen_sums']:.3f}", flush=True)
    # the loop: digits read off a value by comparison lights — largest d with d*10^k <= remainder — then path rebuilt and walked back
    def read_value(v):
        B = v.shape[0]; out = torch.zeros(B, 7, dtype=torch.long, device=dev); V = anchors(); rem = v
        for k in range(6, -1, -1):
            cand = V[:, k][None].expand(B, -1, -1); r = rem[:, None, :] * cand.conj()                   # remainder after removing d*10^k, for d = 0..9
            notneg = less(r.reshape(-1, mv), value([0])[0][None].expand(B * 10, -1)).reshape(B, 10) < 0   # remainder >= 0  <=>  not (remainder < 0)
            notneg[:, 0] = True; d = (notneg.float() * torch.arange(10, device=dev)[None]).argmax(1)                  # largest d whose remainder is still non-negative
            out[:, k] = d; rem = r[torch.arange(B), d]
        return [int(sum(int(row[k]) * 10 ** k for k in range(7))) for row in out.cpu().numpy()]
    ns = [int(x) for x in rng.integers(0, 10 ** 7, 2000)]; res["value_read_exact"] = float(np.mean([d == n for d, n in zip(read_value(value(ns)), ns)])); print(f"digits read off a value by comparison lights (decomposition of a value): exact {res['value_read_exact']:.3f}", flush=True)
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
        print(f"{L}-digit sums (never trained): value bound, digits read by comparison, path walked: exact {np.mean(loop_ok):.3f}   " + "  ".join(f"{k.replace('carries_', 'carries=')} {v['exact']:.3f}(n={v['n']})" for k, v in row.items() if k.startswith("carries")), flush=True)
    for x, y in ((23, 45), (47, 35), (123456, 111111), (999999, 1), (4999999, 4999999)):
        g = read_value(value([x]) * value([y]))[0]; print(f"{x} + {y} = {x + y}:   read {g}   walk {path_walk(path_chain(digs([g])))[0]}   and  {x} < {y}? light says {'yes' if less(value([x]), value([y])).item() > 0 else 'no'}", flush=True)
save_json(res, HERE / a.out / "order.json"); torch.save({"phi": phi.detach(), "P": torch.view_as_complex(P_re).detach(), "log_t": log_t.detach(), "config": vars(a)}, HERE / a.out / "value.pt"); print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)
