"""A number as a chain of its digits under ONE repeated move — no place rows, no fixed length. Rows: the ten digits and an END
symbol (the alphabet). Move: NEXT. A number is built most-significant digit first onto END, so the walk back (NEXT^-1, one
light per step over the eleven rows) reads the units digit first and meets END when nothing is left. Trained on random
digit strings of length 1..Lmax with the walk-back objective (self-supervised: no labels, nothing arithmetic).
Reported: read-back exact by length, beyond the training lengths (the range ceiling of this representation is the depth
at which the walk stays exact, for a given width)."""
import argparse, time, json, numpy as np, torch, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=20); ap.add_argument("--block", type=int, default=16); ap.add_argument("--steps", type=int, default=12000); ap.add_argument("--lmax", type=int, default=24); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="")
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time(); out = a.out or f"results/digit_chain_k{a.k}"
END = 10; model = ResonatE(11, 2, k=a.k, block=True, block_size=a.block).to(dev); M = model.m; NEXT = 0
def E(): return cnorm(model.E)
def hop(z, r): return model.hop(z, torch.full((z.shape[0],), r, device=dev, dtype=torch.long))
def random_numbers(B, Lmax, Lmin=1):
    """digit strings (most significant first, no leading zero unless the number is 0) padded on the left; mask marks real digits"""
    L = rng.integers(Lmin, Lmax + 1, B); D = np.zeros((B, Lmax), dtype=np.int64); mask = np.zeros((B, Lmax), dtype=bool)
    for b in range(B):
        d = rng.integers(0, 10, L[b]); d[0] = rng.integers(1, 10) if L[b] > 1 else d[0]
        if rng.random() < 0.1: d[:] = rng.integers(1, 10)                                                          # a tenth of the strings are runs of one digit (random strings never show them; a k = 12 memory misread 999999999)
        D[b, Lmax - L[b]:] = d; mask[b, Lmax - L[b]:] = True
    return torch.tensor(D, device=dev), torch.tensor(mask, device=dev)
def chain(D, mask):
    """acc = NEXT(END); then for each real digit, most significant first: acc = NEXT(acc + E[d])"""
    Er = E(); B, L = D.shape; acc = hop(Er[END][None].expand(B, -1), NEXT)
    for j in range(L):
        step = hop(cnorm(acc + Er[D[:, j]]), NEXT); acc = torch.where(mask[:, j][:, None], step, acc)
    return acc
def walk(z, n):
    """n reverse hops; the light over the eleven rows at each; after END is met the state is frozen"""
    Er = E(); B = z.shape[0]; out = torch.full((B, n), END, dtype=torch.long, device=dev); rr = z; ended = torch.zeros(B, dtype=torch.bool, device=dev)
    for k in range(n):
        nxt = hop(rr, NEXT + 1); rr = torch.where(ended[:, None], rr, nxt); d = torch.real(rr @ Er.conj().t()).argmax(1); d = torch.where(ended, torch.full_like(d, END), d); out[:, k] = d; ended = ended | (d == END)
    return out
opt = torch.optim.Adam(model.parameters(), lr=3e-3); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
for step in range(1, a.steps + 1):
    B = 256; D, mask = random_numbers(B, a.lmax); z = chain(D, mask); Er = E(); tau = model.log_tau.exp(); rr = z; loss = 0.0; n = 0
    L = mask.sum(1)
    for k in range(a.lmax + 1):                                                                                       # k-th reverse hop must light the k-th digit from the units, or END once past the last
        rr = hop(rr, NEXT + 1); tgt = torch.where(k < L, D[:, max(a.lmax - 1 - k, 0)] if k < a.lmax else torch.full_like(L, END), torch.full_like(L, END)); has = k <= L
        if has.any(): loss = loss + F.cross_entropy((torch.real(rr @ Er.conj().t()) * tau)[has], tgt[has]); n += 1
        rr = torch.where((k < L)[:, None], rr, rr.detach())                                                           # past END the state is not trained further
    loss = loss / n; opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 1000 == 0 or step == 1:
        with torch.no_grad(): got = walk(z.detach(), a.lmax + 1); truth = torch.cat([torch.where(mask.flip(1), D.flip(1), torch.full_like(D, END)), torch.full((B, 1), END, device=dev)], 1); ex = float((got == truth).all(1).float().mean())
        print(f"step {step}/{a.steps}  loss {loss.item():.4f}  exact read-back (batch, 1–{a.lmax} digits) {ex:.3f}  lr {sched.get_last_lr()[0]:.1e}  tau {tau.item():.1f}  elapsed {time.time()-t0:.0f}s  ETA {(time.time()-t0)/step*(a.steps-step):.0f}s", flush=True)
model.eval(); res = {"k": a.k, "M": M, "lmax_train": a.lmax, "readback_by_length": {}}
with torch.no_grad():
    for L in list(range(1, a.lmax + 1)) + [a.lmax + 2, a.lmax + 4, a.lmax + 8, a.lmax + 16]:
        D, mask = random_numbers(400, L, L); got = walk(chain(D, mask), L + 1); truth = torch.cat([D.flip(1), torch.full((400, 1), END, device=dev)], 1)
        res["readback_by_length"][L] = float((got == truth).all(1).float().mean())
    print(f"digit chain k={a.k} (M={M}): read-back exact by length (train 1–{a.lmax}): " + " ".join(f"{L}:{v:.3f}" for L, v in res["readback_by_length"].items()), flush=True)
    runs = torch.tensor([[d] * a.lmax for d in range(1, 10)], device=dev); got = walk(chain(runs, torch.ones_like(runs, dtype=torch.bool)), a.lmax + 1); ok = (got == torch.cat([runs, torch.full((9, 1), END, device=dev)], 1)).all(1)
    res["runs_exact"] = {str(d): bool(o) for d, o in zip(range(1, 10), ok.tolist())}; print(f"  runs of one digit at length {a.lmax} (1..9): " + " ".join(f"{d}:{'ok' if o else 'MISREAD'}" for d, o in res["runs_exact"].items()), flush=True)
(HERE / out).mkdir(parents=True, exist_ok=True); torch.save({"model": model.state_dict(), "config": {"k": a.k, "block": a.block, "lmax": a.lmax}}, HERE / out / "model.pt"); save_json(res, HERE / out / "digit_chain.json"); print(f"saved {out} ({time.time()-t0:.0f}s)", flush=True)
