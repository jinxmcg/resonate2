"""Comparison as a program on the path code: we already decompose (walk back), so x < y is decided by walking both
chains back (one hop + one light per step), then from the most significant place the first digit that differs decides,
using an ORDER ON THE TEN DIGIT SYMBOLS read as a light: score(d) = Re<E_d, Q>, Q one template vector learned from the
45 facts d < d'. Nothing else is trained; the path model is frozen. Exact on unseen pairs at every distance, incl. equal."""
import argparse, time, numpy as np, torch, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--path", default="results/two_relations_k12_walk2/memory.pt"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results/compare")
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time()
NODES = [str(d) for d in range(10)] + [str(10 ** k) for k in range(1, 7)]; nid = {n: i for i, n in enumerate(NODES)}; PLUS, TIMES = 0, 1; NR = 2
pk = torch.load(HERE / a.path, map_location=dev, weights_only=False); pcfg = pk["config"]; pmodel = ResonatE(len(NODES), 2 * NR, k=pcfg["k"], block=True, block_size=pcfg["block"]).to(dev); pmodel.load_state_dict(pk["model"]); pmodel.eval(); pmodel.requires_grad_(False)
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
    """decomposition: one hop + one light per step; returns (B, 7) digits, place 0 first"""
    B = z.shape[0]; out = torch.zeros(B, 7, dtype=torch.long, device=dev); rr = z
    for k in range(7):
        rr = phop(rr, PLUS + NR)
        if k == 0: out[:, 0] = plight(rr).argmax(1)
        else: L = plight(phop(rr, TIMES + NR)); L[:, 10:] = -1e9; out[:, k] = L.argmax(1)
    return out
# ------------------------------------------------------------------ the order on the ten digit symbols: one template, 45 facts
Q = torch.nn.Parameter(torch.randn(Er.shape[1], 2, device=dev) / np.sqrt(Er.shape[1])); opt = torch.optim.Adam([Q], lr=3e-2)
pairs = [(d, e) for d in range(10) for e in range(10) if d < e]
def dscore(): return torch.real(Er[:10] @ torch.view_as_complex(Q).conj())                       # light of each digit row on Q
for step in range(1, 1001):
    s = dscore(); loss = F.softplus(-(s[[e for d, e in pairs]] - s[[d for d, e in pairs]]) * 10).mean(); opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    s = dscore(); order_ok = float(np.mean([(s[e] > s[d]).item() for d, e in pairs])); print(f"digit order template: 45 facts, all satisfied {order_ok:.3f}; scores " + " ".join(f"{d}:{s[d].item():+.2f}" for d in range(10)), flush=True)
    # ------------------------------------------------------------------ the program: walk both back, first differing place from the top decides by the digit-order light
    def compare(xs, ys):
        Dx, Dy = walk_digits(path_chain(digs(xs))), walk_digits(path_chain(digs(ys))); sx, sy = s[Dx], s[Dy]                  # (B, 7) digit lights
        out = torch.zeros(len(xs), dtype=torch.long, device=dev)                                                            # 0: equal, 1: x<y, 2: x>y
        for k in range(6, -1, -1):
            und = out == 0; lt = und & (sx[:, k] < sy[:, k]); gt = und & (sx[:, k] > sy[:, k]); out[lt] = 1; out[gt] = 2
        return out.cpu().numpy()
    res = {"digit_order_ok": order_ok}
    x = rng.integers(0, 10 ** 7, 6000); u = rng.integers(0, 7, 6000); d = rng.integers(1, 10 ** u + 1) * rng.choice([-1, 1], 6000); y = np.clip(x + d, 0, 10 ** 7 - 1)
    xs, ys = [int(v) for v in x], [int(v) for v in y]; truth = np.array([0 if a_ == b_ else (1 if a_ < b_ else 2) for a_, b_ in zip(xs, ys)]); pred = compare(xs, ys); ok = pred == truth
    dist = np.abs(np.array(ys) - np.array(xs)); by = {}
    for o, dd in zip(ok, dist): by.setdefault("0" if dd == 0 else f"10^{int(np.floor(np.log10(dd)))}", []).append(o)
    res["unseen_pairs"] = {"n": len(xs), "exact": float(ok.mean()), "by_distance": {k: float(np.mean(v)) for k, v in sorted(by.items())}}
    print(f"x<y / x>y / x=y on {len(xs)} unseen pairs, decided by walking both back: exact {ok.mean():.3f}   by distance " + "  ".join(f"{k} {np.mean(v):.3f}(n={len(v)})" for k, v in sorted(by.items())), flush=True)
    eq = [int(v) for v in rng.integers(0, 10 ** 7, 1000)]; res["equal_pairs_exact"] = float((compare(eq, eq) == 0).mean()); print(f"x = x recognised as equal: {res['equal_pairs_exact']:.3f}", flush=True)
    for x_, y_ in ((347, 437), (437, 347), (999999, 1000000), (120123, 120123), (5060, 6050), (7, 12)):
        print(f"{x_} vs {y_}: {['equal', 'x < y', 'x > y'][compare([x_], [y_])[0]]}", flush=True)
save_json(res, HERE / a.out / "compare.json"); torch.save({"Q": torch.view_as_complex(Q).detach()}, HERE / a.out / "digit_order.pt"); print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)
