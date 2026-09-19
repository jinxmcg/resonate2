"""'=' as one more relation. Nodes 0-9 and 10^1..10^6 (16 rows); relations x, +, = (learned reverses).
numeral region : the directed 7-term chain of two_relations.py (first term through its own +).
expression a+b : the directed 14-term chain — a's seven terms bigger to smaller, then b's seven — same + and x.
'=' facts      : region(a + b) --=--> region(numeral a+b), trained ONLY on single-place digit equations:
                 a*10^k + b*10^k = (a+b)*10^k for a, b in 0..9 at every place k (carry facts included, e.g. 7 tens + 5 tens
                 = 1 hundred 2 tens); a random 20% of (a, b, k) held out. Nothing multi-place is ever trained.
Reading        : hop(region(a+b), =) walked back one hop + one light per step (the numeral decode), exact = all 7 digits.
Prediction     : multi-place sums WITHOUT carry follow by linearity; sums WITH carry do not (a carried 1 stays a second term
                 at its place instead of merging into the digit there) — the carry is where a decision, not a move, begins."""
import argparse, time, numpy as np, torch, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=12); ap.add_argument("--block", type=int, default=16); ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--batch", type=int, default=1024); ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results/equals_k12")
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time()
NODES = [str(d) for d in range(10)] + [str(10 ** k) for k in range(1, 7)]; nid = {n: i for i, n in enumerate(NODES)}; PLUS, TIMES, EQ = 0, 1, 2; NR = 3
ANCH = torch.tensor([nid["0"]] + [nid[str(10 ** k)] for k in range(1, 7)], device=dev)        # anchor row per place (place 0: none)
model = ResonatE(len(NODES), 2 * NR, k=a.k, block=True, block_size=a.block).to(dev); M = model.m
def hop(z, r): return model.hop(z, torch.full((z.shape[0],), r, device=dev, dtype=torch.long))
def E(): return cnorm(model.E)
def light(z): return torch.real(z @ E().conj().t()) * model.log_tau.exp()
def digs(ns): return torch.tensor([[int(c) for c in f"{int(n):07d}"] for n in ns], device=dev)     # (B, 7) biggest place first


def chain(D):
    """directed chain over terms given as (B, T) digits with places (6..0 repeated); every term enters through its own +."""
    Er = E(); B, T = D.shape; acc = None
    for j in range(T):
        k = 6 - (j % 7); d = D[:, j]
        term = Er[d] if k == 0 else hop(cnorm(Er[d] + Er[ANCH[k]][None].expand(B, -1)), TIMES)
        acc = hop(term, PLUS) if acc is None else hop(cnorm(acc + term), PLUS)
    return acc


def walk_loss(z, D):
    """walk back: after j reverse + hops (and x^-1 above place 0) the light shows exactly the j-th term from the end."""
    B, T = D.shape; loss = 0.0; rr = z
    for j in range(T):
        k = 6 - ((T - 1 - j) % 7); d = D[:, T - 1 - j]; rr = hop(rr, PLUS + NR)
        L = light(rr) if k == 0 else light(hop(rr, TIMES + NR)); tgt = torch.zeros_like(L); tgt[torch.arange(B), d] = 1
        if k > 0: tgt[:, ANCH[k]] = 1
        loss = loss + F.binary_cross_entropy_with_logits(L, tgt)
    return loss / T


def walk_read(z, T=7):
    """the numeral decode: one hop + one light per step; returns ints (7-term chains only)."""
    B = z.shape[0]; out = torch.zeros(B, 7, dtype=torch.long, device=dev); rr = z
    for j in range(7):
        k = j; rr = hop(rr, PLUS + NR)
        if k == 0: out[:, 0] = light(rr).argmax(1)
        else:
            L = light(hop(rr, TIMES + NR)); L[:, 10:] = -1e9; out[:, k] = L.argmax(1)          # the digit among the digit rows (anchor known by the step)
    return [int(sum(int(row[k]) * 10 ** k for k in range(7))) for row in out.cpu().numpy()]


# ------------------------------------------------------------------ data
numerals = [int(x) for x in rng.integers(0, 10 ** 7, 20000)]
facts = [(x, y, k) for x in range(10) for y in range(10) for k in range(7) if (x + y) * 10 ** k < 10 ** 7]
rng.shuffle(facts); n_hold = len(facts) // 5; facts_test, facts_train = facts[:n_hold], facts[n_hold:]
print(f"digit equations: {len(facts_train)} train, {len(facts_test)} held out (single place, carries included); numerals for the walk {len(numerals)}", flush=True)
def expr(pairs): return torch.cat([digs([p[0] for p in pairs]), digs([p[1] for p in pairs])], 1)   # (B, 14)
opt = torch.optim.Adam(model.parameters(), lr=a.lr); (HERE / a.out).mkdir(parents=True, exist_ok=True)
for step in range(1, a.steps + 1):
    ns = [numerals[i] for i in rng.integers(0, len(numerals), a.batch)]; Dn = digs(ns); loss = walk_loss(chain(Dn), Dn)
    fb = [facts_train[i] for i in rng.integers(0, len(facts_train), a.batch)]; pairs = [(x * 10 ** k, y * 10 ** k) for x, y, k in fb]; De = expr(pairs)
    ze = chain(De); loss = loss + walk_loss(ze, De) + walk_loss(hop(ze, EQ), digs([p[0] + p[1] for p in pairs]))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 250 == 0 or step == 1:
        eta = (time.time() - t0) / step * (a.steps - step); print(f"step {step}/{a.steps}  loss {loss.item():.4f}  lr {a.lr}  tau {model.log_tau.exp().item():.1f}  elapsed {time.time()-t0:.0f}s  ETA {eta:.0f}s", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "config": vars(a)}, HERE / a.out / "ckpt.pt")
model.eval(); res = {"config": vars(a), "facts_train": len(facts_train), "facts_test": len(facts_test)}
with torch.no_grad():
    def eq_exact(pairs):
        dec = walk_read(hop(chain(expr(pairs)), EQ)); return [d == x + y for d, (x, y) in zip(dec, pairs)]
    # sanity: numerals and expressions read back their own terms
    ns = [int(x) for x in rng.integers(0, 10 ** 7, 2000)]; res["numeral_walk_exact"] = float(np.mean([d == n for d, n in zip(walk_read(chain(digs(ns))), ns)]))
    print(f"numeral chains walk back exactly: {res['numeral_walk_exact']:.3f}", flush=True)
    # digit equations: trained and held out
    for name, fs in (("digit equations (trained)", facts_train), ("digit equations (held out)", facts_test)):
        ok = eq_exact([(x * 10 ** k, y * 10 ** k) for x, y, k in fs]); carry = [x + y >= 10 for x, y, k in fs]
        res[name] = {"n": len(fs), "exact": float(np.mean(ok)), "no_carry": float(np.mean([o for o, c in zip(ok, carry) if not c])), "carry": float(np.mean([o for o, c in zip(ok, carry) if c]))}
        print(f"{name:30s} n={len(fs):4d}  exact {np.mean(ok):.3f}   no-carry {res[name]['no_carry']:.3f}   carry {res[name]['carry']:.3f}", flush=True)
    # multi-place sums, never trained: by digit length and by number of carries
    def n_carries(x, y):
        c = 0; n = 0; dx, dy = f"{x:07d}"[::-1], f"{y:07d}"[::-1]
        for i in range(7):
            s = int(dx[i]) + int(dy[i]) + c; c = 1 if s >= 10 else 0; n += c
        return n
    for L in range(2, 8):
        pairs = []
        while len(pairs) < 600:
            s = int(rng.integers(10 ** (L - 1), 10 ** L)); x = int(rng.integers(0, s + 1)); pairs.append((x, s - x))
        ok = eq_exact(pairs); nc = [n_carries(x, y) for x, y in pairs]
        row = {"n": len(pairs), "exact": float(np.mean(ok))}
        for c in (0, 1, 2, 3):
            sel = [o for o, k in zip(ok, nc) if (k == c if c < 3 else k >= 3)]
            if sel: row[f"carries_{c}{'+' if c == 3 else ''}"] = {"n": len(sel), "exact": float(np.mean(sel))}
        res[f"sum_len{L}"] = row
        print(f"{L}-digit sums (never trained): exact {np.mean(ok):.3f}   " + "  ".join(f"{k.replace('carries_', 'carries=')} {v['exact']:.3f}(n={v['n']})" for k, v in row.items() if k.startswith("carries")), flush=True)
    # a few examples
    for x, y in ((23, 45), (47, 35), (123456, 111111), (999999, 1)):
        print(f"{x} + {y} = {x + y}:   read {walk_read(hop(chain(expr([(x, y)])), EQ))[0]}", flush=True)
save_json(res, HERE / a.out / "equals.json"); torch.save({"model": model.state_dict(), "config": vars(a)}, HERE / a.out / "memory.pt"); print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)
