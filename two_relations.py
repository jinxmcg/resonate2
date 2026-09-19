"""One step on ResonatE, output fed back. Nodes: 0-9 and 10, 100, ..., 1000000 (16 rows). Relations: + and x only.
347 is built bigger to smaller, one hop per step, operands entering a hop as a SUPERPOSITION:
    term_2 = hop(E_3 (+) E_100, x)          acc = term_2
    term_1 = hop(E_4 (+) E_10,  x)          acc = hop(acc (+) term_1, +)
    term_0 = E_7                            acc = hop(acc (+) term_0, +)
Every place is present (zero digits too), so every number is a 7-term chain and batches by shape. No composite number
has a row; the memory only ever sees the 16 nodes and the two relations.
Training = the reverse must reconstruct: hop(term_k, x^-1) lights d_k and 10^k; hop(acc_k, +^-1) lights the stored
operand (units digit at k=0) and matches the superposition of its two operands (cosine).
Evaluation: NO reasoner, NO iteration — one light. (a) the composed state's own light over the 16 nodes; (b) one reverse
hop (+^-1, x^-1) and its light. Scored as sets: the nodes that should shine are the number's non-zero digits and the
anchors of its non-zero places; exact = the top-|set| nodes of the light are exactly that set. 347 vs 437 reported."""
import argparse, time, numpy as np, torch, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=12); ap.add_argument("--block", type=int, default=16); ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--batch", type=int, default=2048); ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results/two_relations_k12"); ap.add_argument("--walk", action="store_true", help="train the directed reverse walk (one light per step) instead of bag reconstruction")
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time()
NODES = [str(d) for d in range(10)] + [str(10 ** k) for k in range(1, 7)]; nid = {n: i for i, n in enumerate(NODES)}; PLUS, TIMES = 0, 1; NR = 2
model = ResonatE(len(NODES), 2 * NR, k=a.k, block=True, block_size=a.block).to(dev); M = model.m
def hop(z, r): return model.hop(z, torch.full((z.shape[0],), r, device=dev, dtype=torch.long))
def E(): return cnorm(model.E)
def light(z): return torch.real(z @ E().conj().t()) * model.log_tau.exp()
def digs(ns): return torch.tensor([[int(c) for c in f"{int(n):07d}"] for n in ns], device=dev)     # (B, 7) biggest place first


def compose(ns, trace=False):
    """bigger to smaller, one hop per step, output fed back. Returns acc (B, M) [and the per-step states]."""
    Er = E(); D = digs(ns); B = len(ns); acc = None; terms, accs = [], []
    for j in range(7):
        k = 6 - j; d = D[:, j]
        term = Er[d] if k == 0 else hop(cnorm(Er[d] + Er[nid[str(10 ** k)]][None].expand(B, -1)), TIMES)
        acc = hop(term, PLUS) if acc is None else hop(cnorm(acc + term), PLUS); terms.append(term); accs.append(acc)   # the first term also enters through its own +
    return (acc, terms, accs) if trace else acc


def walk_loss(ns):
    """the directed chain read backwards: after j reverse + hops (and one x^-1), the light shows the j-th term's operands
    and NOTHING else (BCE over all 16 nodes). Units first (j=0: the light of +^-1 shows d_0 on the digit rows)."""
    Er = E(); D = digs(ns); B = len(ns); acc = compose(ns); loss = 0.0; n = 0; rr = acc
    for j in range(7):                                                                          # j-th term back: place k = j
        k = j; d = D[:, 6 - j]
        rr = hop(rr, PLUS + NR)                                                                 # step back over one + (every term has one)
        if k == 0: L = light(rr); tgt = torch.zeros_like(L); tgt[torch.arange(B), d] = 1
        else:
            L = light(hop(rr, TIMES + NR)); tgt = torch.zeros_like(L); tgt[torch.arange(B), d] = 1; tgt[:, nid[str(10 ** k)]] = 1
        loss = loss + F.binary_cross_entropy_with_logits(L, tgt); n += 1
    return loss / n


def recon_loss(ns):
    """reverse hops must reconstruct: x^-1 of a term lights its digit and anchor; +^-1 of an accumulator lights the
    stored operand (units digit at the last step) and matches the superposition of its two operands."""
    Er = E(); D = digs(ns); B = len(ns); acc, terms, accs = compose(ns, trace=True); loss = 0.0; n = 0
    for j in range(7):
        k = 6 - j; d = D[:, j]
        if k > 0:
            L = light(hop(terms[j], TIMES + NR)); tgt = torch.zeros_like(L); tgt[torch.arange(B), d] = 1; tgt[:, nid[str(10 ** k)]] = 1
            loss = loss + F.binary_cross_entropy_with_logits(L, tgt); n += 1
        if j > 0:
            r = hop(accs[j], PLUS + NR); want = cnorm(accs[j - 1] + terms[j]); loss = loss + (1 - torch.real((r * want.conj()).sum(-1))).mean(); n += 1
            if k == 0:
                L = light(r); tgt = torch.zeros_like(L); tgt[torch.arange(B), d] = 1; loss = loss + F.binary_cross_entropy_with_logits(L[:, :10], tgt[:, :10]); n += 1
    return loss / n


train = [int(x) for x in rng.integers(0, 10 ** 7, 20000)]; test = [int(x) for x in rng.integers(0, 10 ** 7, 3000) if int(x) not in set(train)]
opt = torch.optim.Adam(model.parameters(), lr=a.lr); (HERE / a.out).mkdir(parents=True, exist_ok=True)
for step in range(1, a.steps + 1):
    b = [train[i] for i in rng.integers(0, len(train), a.batch)]; loss = (walk_loss if a.walk else recon_loss)(b); opt.zero_grad(); loss.backward(); opt.step()
    if step % 250 == 0 or step == 1:
        eta = (time.time() - t0) / step * (a.steps - step); print(f"step {step}/{a.steps}  recon loss {loss.item():.4f}  lr {a.lr}  tau {model.log_tau.exp().item():.1f}  elapsed {time.time()-t0:.0f}s  ETA {eta:.0f}s", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "config": vars(a)}, HERE / a.out / "ckpt.pt")
model.eval(); res = {"config": vars(a)}
with torch.no_grad():
    Er = E()
    def parts(n):
        s = f"{int(n):07d}"; out = set()
        for j, c in enumerate(s):
            k = 6 - j
            if c != "0": out.add(nid[c]); out.add(nid[str(10 ** k)]) if k > 0 else None
        return out or {nid["0"]}
    def set_scores(L, ns):
        """for each number: are the top-|parts| nodes of the light exactly its parts? plus mean precision of that top set."""
        ex, prec = [], []
        for i, n in enumerate(ns):
            p = parts(n); top = set(L[i].topk(len(p)).indices.tolist()); ex.append(top == p); prec.append(len(top & p) / len(p))
        return float(np.mean(ex)), float(np.mean(prec))
    for name, ns in (("train", train[:3000]), ("test (never seen)", test)):
        z = compose(ns); reads = {"state light": light(z), "one hop +^-1 then light": light(hop(z, PLUS + NR)), "one hop x^-1 then light": light(hop(z, TIMES + NR)),
                                  "+^-1 then x^-1 then light": light(hop(hop(z, PLUS + NR), TIMES + NR))}
        res[name] = {}
        for rd, L in reads.items():
            ex, pr = set_scores(L, ns); res[name][rd] = {"exact_set": ex, "precision": pr}; print(f"{name:18s} {rd:28s}: the shining set is exactly the parts {ex:.3f}   precision of the top set {pr:.3f}", flush=True)
        by = {}
        for i, n in enumerate(ns): by.setdefault(len(str(n)), []).append(set(reads["one hop +^-1 then light"][i].topk(len(parts(n))).indices.tolist()) == parts(n))
        print(f"{name:18s} +^-1 light exact set by digit length: " + "  ".join(f"{L}d {np.mean(v):.2f}" for L, v in sorted(by.items())), flush=True)
    # the directed walk back: one hop and one light per step; the j-th step must show the j-th term's operands
    def walk_read(ns):
        D = digs(ns); B = len(ns); rr = compose(ns); ok = torch.ones(B, dtype=torch.bool, device=dev); step_ok = []
        for j in range(7):
            k = j; d = D[:, 6 - j]
            rr = hop(rr, PLUS + NR)
            if k == 0: hit = light(rr).argmax(1) == d
            else:
                top = light(hop(rr, TIMES + NR)).topk(2).indices; want = torch.stack([d, torch.full_like(d, nid[str(10 ** k)])], 1)
                hit = (top.sort(1).values == want.sort(1).values).all(1)
            step_ok.append(hit.float().mean().item()); ok &= hit
        return ok.float().mean().item(), step_ok
    for name, ns in (("train", train[:3000]), ("test (never seen)", test)):
        ex, per = walk_read(ns); res[name]["reverse_walk"] = {"exact_all_7_steps": ex, "per_step": per}
        print(f"{name:18s} reverse walk, one light per step: all 7 terms right {ex:.3f}   per step (units first) " + " ".join(f"{p:.2f}" for p in per), flush=True)
    # what does one light show for 347?  and can it tell 347 from 437?
    for n in (347, 437, 7, 1000000, 120123):
        z = compose([n]); L = light(hop(z, PLUS + NR))[0]; top = L.topk(6); print(f"{n}: one +^-1 light, brightest nodes: " + ", ".join(f"{NODES[i]}({v:.1f})" for v, i in zip(top.values.tolist(), top.indices.tolist())), flush=True)
    for x, y in ((347, 437), (347, 743), (1234, 4321), (5060, 6050)):
        c = float(torch.real(compose([x]) @ compose([y]).conj().t())); res[f"cos_{x}_{y}"] = c; print(f"{x} vs {y}: cosine of the composed states {c:.3f}", flush=True)
    ctrl = list(zip(rng.integers(0, 10 ** 7, 500).tolist(), rng.integers(0, 10 ** 7, 500).tolist())); zc = torch.real((compose([p[0] for p in ctrl]) * compose([p[1] for p in ctrl]).conj()).sum(-1)).mean().item()
    perm = []
    for n in test[:500]:
        sl = list(str(n)); rng.shuffle(sl); m = int("".join(sl))
        if m != n: perm.append((n, m))
    zp = torch.real((compose([p[0] for p in perm]) * compose([p[1] for p in perm]).conj()).sum(-1)).mean().item()
    res["cos_random_pairs"] = zc; res["cos_digit_permutations"] = zp; print(f"mean cosine: random number pairs {zc:.3f}   same digits permuted {zp:.3f}", flush=True)
save_json(res, HERE / a.out / "two_relations.json"); torch.save({"model": model.state_dict(), "config": vars(a)}, HERE / a.out / "memory.pt"); print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)
