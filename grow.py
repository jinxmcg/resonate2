"""Growing the number system from millions to billions by adding knowledge: ONLY the new nodes learn.
path  : load the 16-node chain model (Part 4), append two rows (10^7, 10^8), keep + and x, continue the walk objective on
        9-term chains (numbers < 10^9). Report walk-back exactness on 9-term chains before and after, by digit length.
value : load the 63-anchor phase code (Part 6), append 18 anchors (places 7, 8), continue on single-place digit equations
        at all nine places (carries tie the new places to the old) + distinctness. Report held-out equations by place.
program: unchanged (per-place equality against 20 candidates; comparison by walking from the top), now over 9 places.
Exact on never-trained 8- and 9-digit sums and comparisons."""
import argparse, time, numpy as np, torch, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--np", type=int, default=9); ap.add_argument("--path", default="results/two_relations_k12_walk2/memory.pt"); ap.add_argument("--value", default="results/two_part_learn/value.pt")
ap.add_argument("--path-steps", type=int, default=2000); ap.add_argument("--value-steps", type=int, default=2000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--moves-learn", action="store_true", help="operators keep learning too; old numbers replayed so nothing is forgotten"); ap.add_argument("--out", default="results/grow_1e9")
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time(); NP = a.np; MAXN = 10 ** NP
NODES = [str(d) for d in range(10)] + [str(10 ** k) for k in range(1, NP)]; nid = {n: i for i, n in enumerate(NODES)}; PLUS, TIMES = 0, 1; NR = 2
pk = torch.load(HERE / a.path, map_location=dev, weights_only=False); pcfg = pk["config"]; old_n = pk["model"]["E"].shape[0]
pmodel = ResonatE(len(NODES), 2 * NR, k=pcfg["k"], block=True, block_size=pcfg["block"]).to(dev)
with torch.no_grad():
    sd = pk["model"]; E_new = pmodel.E.data.clone(); E_new[:old_n] = sd["E"]; sd = dict(sd); sd["E"] = E_new; pmodel.load_state_dict(sd)      # old rows kept, new rows random
ANCH = torch.tensor([nid["0"]] + [nid[str(10 ** k)] for k in range(1, NP)], device=dev)
def phop(z, r): return pmodel.hop(z, torch.full((z.shape[0],), r, device=dev, dtype=torch.long))
def plight(z): return torch.real(z @ cnorm(pmodel.E).conj().t())
def digs(ns): return torch.tensor([[int(c) for c in f"{int(n):0{NP}d}"] for n in ns], device=dev)
def path_chain(D):
    Er = cnorm(pmodel.E); B = D.shape[0]; acc = None
    for j in range(NP):
        k = NP - 1 - j; d = D[:, j]; term = Er[d] if k == 0 else phop(cnorm(Er[d] + Er[ANCH[k]][None].expand(B, -1)), TIMES)
        acc = phop(term, PLUS) if acc is None else phop(cnorm(acc + term), PLUS)
    return acc
def walk_loss(z, D):
    B = D.shape[0]; loss = 0.0; rr = z
    for j in range(NP):
        k = j; d = D[:, NP - 1 - j]; rr = phop(rr, PLUS + NR); L = plight(rr) * pmodel.log_tau.exp() if k == 0 else plight(phop(rr, TIMES + NR)) * pmodel.log_tau.exp()
        tgt = torch.zeros_like(L); tgt[torch.arange(B), d] = 1
        if k > 0: tgt[:, ANCH[k]] = 1
        loss = loss + F.binary_cross_entropy_with_logits(L, tgt)
    return loss / NP
def walk_digits(z):
    B = z.shape[0]; out = torch.zeros(B, NP, dtype=torch.long, device=dev); rr = z
    for k in range(NP):
        rr = phop(rr, PLUS + NR)
        if k == 0: out[:, 0] = plight(rr).argmax(1)
        else: L = plight(phop(rr, TIMES + NR)); L[:, 10:] = -1e9; out[:, k] = L.argmax(1)
    return out
def to_int(D): return [int(sum(int(row[k]) * 10 ** k for k in range(NP))) for row in D.cpu().numpy()]
def walk_exact_by_len(n=400):
    out = {}
    for L in range(1, NP + 1):
        ns = [int(x) for x in rng.integers(10 ** (L - 1), 10 ** L, n)]; out[L] = float(np.mean([d == m for d, m in zip(to_int(walk_digits(path_chain(digs(ns)))), ns)]))
    return out
with torch.no_grad(): before = walk_exact_by_len(); print("path BEFORE continuation (2 new random rows, 9-term chains): walk-back exact by digits " + " ".join(f"{L}d {v:.2f}" for L, v in before.items()), flush=True)
if a.moves_learn: opt = torch.optim.Adam(pmodel.parameters(), lr=3e-3)
else:
    pmodel.requires_grad_(False); pmodel.E.requires_grad_(True); rowmask = torch.zeros_like(pmodel.E); rowmask[old_n:] = 1; pmodel.E.register_hook(lambda g: g * rowmask)   # only the two new rows learn
    opt = torch.optim.Adam([pmodel.E], lr=3e-3)
(HERE / a.out).mkdir(parents=True, exist_ok=True)
for step in range(1, a.path_steps + 1):
    ns = [int(x) for x in rng.integers(10 ** (old_n - 9), MAXN, 1536)] + ([int(x) for x in rng.integers(0, 10 ** (old_n - 9), 512)] if a.moves_learn else []); D = digs(ns)   # new-place examples (+ replay of old numbers)
    loss = walk_loss(path_chain(D), D); opt.zero_grad(); loss.backward(); opt.step()
    if step % 250 == 0 or step == 1: print(f"path step {step}/{a.path_steps}  walk loss {loss.item():.4f}  lr 0.003  tau {pmodel.log_tau.exp().item():.1f}  elapsed {time.time()-t0:.0f}s  ETA {(time.time()-t0)/step*(a.path_steps-step):.0f}s", flush=True)
pmodel.eval(); res = {"config": vars(a)}
with torch.no_grad(): after = walk_exact_by_len(); res["path_walk_before"] = before; res["path_walk_after"] = after; print("path AFTER continuation: walk-back exact by digits " + " ".join(f"{L}d {v:.3f}" for L, v in after.items()), flush=True)
torch.save({"model": pmodel.state_dict(), "config": {**pcfg, "np": NP}}, HERE / a.out / "path.pt")
# ------------------------------------------------------------------ value code, continued with two new places
vk = torch.load(HERE / a.value, map_location=dev, weights_only=False); old = vk["phi"]; mv = old.shape[-1]
phi = torch.nn.Parameter(torch.cat([old, (torch.rand(10, NP - old.shape[1], mv, device=dev) * 2 - 1) * np.pi], 1)); OLDP = old.shape[1]
phimask = torch.zeros_like(phi); phimask[:, OLDP:] = 1
if not a.moves_learn: phi.register_hook(lambda g: g * phimask)
log_t = torch.nn.Parameter(torch.tensor(np.log(10.0), device=dev))
def anchors(): p = phi.clone(); p[0] = 0; return torch.polar(torch.ones_like(p), p)
def value(ns):
    V = anchors(); D = digs(ns); v = torch.ones(len(ns), mv, dtype=torch.complex64, device=dev)
    for j in range(NP): v = v * V[D[:, j], NP - 1 - j]
    return v
def vcos(x, y): return torch.real((x * y.conj()).mean(-1))
facts = [(x, y, k) for x in range(10) for y in range(10) for k in range(NP) if (x + y) * 10 ** k < MAXN]; rng.shuffle(facts); n_hold = len(facts) // 5; facts_test, facts_train = facts[:n_hold], facts[n_hold:]
def eq_by_place(fs):
    by = {}
    for x, y, k in fs: by.setdefault(k, []).append(vcos(value([x * 10 ** k]) * value([y * 10 ** k]), value([(x + y) * 10 ** k])).item() > 0.99)
    return {k: float(np.mean(v)) for k, v in sorted(by.items())}
with torch.no_grad(): vb = eq_by_place(facts_test); print("value BEFORE continuation: held-out equations satisfied by place " + " ".join(f"p{k} {v:.2f}" for k, v in vb.items()), flush=True)
opt = torch.optim.Adam([phi, log_t], lr=1e-2)
for step in range(1, a.value_steps + 1):
    newf = [f for f in facts_train if f[2] >= OLDP - 1]                                                                       # facts at the new places, plus the carry facts that link place 6 to 7
    fb = [newf[i] for i in rng.integers(0, len(newf), 384)] + ([facts_train[i] for i in rng.integers(0, len(facts_train), 128)] if a.moves_learn else []); loss_eq = (1 - vcos(value([x * 10 ** k for x, y, k in fb]) * value([y * 10 ** k for x, y, k in fb]), value([(x + y) * 10 ** k for x, y, k in fb]))).mean()
    ns = [int(x) for x in rng.integers(10 ** OLDP, MAXN, 1024)]; v = value(ns); S = torch.real(v @ v.conj().t()) / mv * log_t.exp(); loss_d = F.cross_entropy(S, torch.arange(len(ns), device=dev))
    loss = loss_eq + (0 if a.moves_learn else loss_d); opt.zero_grad(); loss.backward(); opt.step()
    if step % 250 == 0 or step == 1: print(f"value step {step}/{a.value_steps}  equation loss {loss_eq.item():.5f}  distinctness {loss_d.item():.4f}  lr 0.01  elapsed {time.time()-t0:.0f}s  ETA {(time.time()-t0)/step*(a.value_steps-step):.0f}s", flush=True)
with torch.no_grad():
    va = eq_by_place(facts_test); res["value_eq_before"] = vb; res["value_eq_after"] = va; print("value AFTER continuation: held-out equations satisfied by place " + " ".join(f"p{k} {v:.3f}" for k, v in va.items()), flush=True)
    torch.save({"phi": phi.detach(), "config": {**vk["config"], "np": NP}}, HERE / a.out / "value.pt")
    # ------------------------------------------------------------------ the programs over NP places
    V = anchors(); one = torch.ones(mv, dtype=torch.complex64, device=dev)
    cand = torch.stack([torch.stack([V[s % 10, k] * (V[s // 10, k + 1] if k < NP - 1 and s >= 10 else one) for s in range(20)]) for k in range(NP)])
    def add(xs, ys):
        Dx, Dy = walk_digits(path_chain(digs(xs))), walk_digits(path_chain(digs(ys))); B = len(xs); carry = torch.zeros(B, dtype=torch.long, device=dev); out = torch.zeros(B, NP, dtype=torch.long, device=dev)
        for k in range(NP):
            e = V[Dx[:, k], k] * V[Dy[:, k], k] * V[carry, k]; s = torch.real((e[:, None, :] * cand[k][None].conj()).mean(-1)).argmax(1); out[:, k] = s % 10; carry = s // 10
        return walk_digits(path_chain(out.flip(1)))
    for L in range(1, NP + 1):
        pairs = []
        while len(pairs) < 500:
            s = int(rng.integers(10 ** (L - 1), 10 ** L)); x = int(rng.integers(0, s + 1)); pairs.append((x, s - x))
        ok = [w == x + y for w, (x, y) in zip(to_int(add([p[0] for p in pairs], [p[1] for p in pairs])), pairs)]; res[f"add_len{L}"] = float(np.mean(ok))
        print(f"{L}-digit sums (never trained): exact {np.mean(ok):.3f}", flush=True)
    Q = torch.load(HERE / "results/compare/digit_order.pt", map_location=dev, weights_only=False)["Q"]; s_d = torch.real(cnorm(pmodel.E)[:10] @ Q.conj())
    def compare(xs, ys):
        Dx, Dy = walk_digits(path_chain(digs(xs))), walk_digits(path_chain(digs(ys))); sx, sy = s_d[Dx], s_d[Dy]; out = torch.zeros(len(xs), dtype=torch.long, device=dev)
        for k in range(NP - 1, -1, -1):
            und = out == 0; out[und & (sx[:, k] < sy[:, k])] = 1; out[und & (sx[:, k] > sy[:, k])] = 2
        return out.cpu().numpy()
    x = rng.integers(0, MAXN, 4000); u = rng.integers(0, NP, 4000); d = rng.integers(1, 10 ** u + 1) * rng.choice([-1, 1], 4000); y = np.clip(x + d, 0, MAXN - 1); xs, ys = [int(v) for v in x], [int(v) for v in y]
    truth = np.array([0 if p == q else (1 if p < q else 2) for p, q in zip(xs, ys)]); ok = compare(xs, ys) == truth; dist = np.abs(np.array(ys) - np.array(xs)); by = {}
    for o, dd in zip(ok, dist): by.setdefault("0" if dd == 0 else f"10^{int(np.floor(np.log10(dd)))}", []).append(o)
    res["compare"] = {"exact": float(ok.mean()), "by_distance": {k: float(np.mean(v)) for k, v in sorted(by.items())}}; print(f"compare on unseen pairs below 10^{NP}: exact {ok.mean():.3f}   by distance " + "  ".join(f"{k} {np.mean(v):.3f}" for k, v in sorted(by.items())), flush=True)
    for x_, y_ in ((123456789, 876543211), (999999999, 0), (500000000, 499999999), (347, 437)): print(f"{x_} + {y_} = {x_ + y_}: read {to_int(add([x_], [y_]))[0]}   {x_} vs {y_}: {['equal', 'x < y', 'x > y'][compare([x_], [y_])[0]]}", flush=True)
save_json(res, HERE / a.out / "grow.json"); print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)
