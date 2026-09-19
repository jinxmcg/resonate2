"""Step 1 — ground the memory as a number graph and check that number structure EMERGES (nothing is enforced).
Train: rows for the 221 grounded numerals + one operator per (op, b) seen as a depth-1 training fact, both directions.
Loss: cross-entropy of the full light of hop(E_a, (op,b)) at row c, and of hop(E_c, (op,b)^-1) at row a.
Emergence checks (all held out from the operator training):
  named   : validation/test B depth-1 facts, hop with the (op,b) operator -> exact top-1 / MRR over all rows
  compose : (+b then +c) vs the row a+b+c, on grounded triples whose pair-sums were never trained together
  unnamed : C depth-1 facts, hop lands on a spot with no row -> same-number vs different-number cosine (purity)
"""
import argparse, time, numpy as np, torch, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=8); ap.add_argument("--block", type=int, default=16); ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--lr", type=float, default=5e-3); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results/ground_k8")
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time()
labels = grounded_labels(); train = facts(records("train")); rels = sorted({(op, b) for _, op, b, _ in train})
mem = Memory(labels, rels, k=a.k, block_size=a.block).to(dev)
print(f"rows {len(labels)}  relations {len(rels)} (x2 with reverses)  facts {len(train)}  M={mem.M}  params {mem.model.n_params() if hasattr(mem.model,'n_params') else sum(p.numel() for p in mem.parameters())}", flush=True)
A = torch.tensor([mem.id[x] for x, _, _, _ in train], device=dev); C = torch.tensor([mem.id[c] for _, _, _, c in train], device=dev)
R = torch.tensor([mem.rid[(op, b)] for _, op, b, _ in train], device=dev); Rinv = R + mem.nb
opt = torch.optim.Adam(mem.parameters(), lr=a.lr)
for step in range(1, a.steps + 1):
    z = mem.hop(mem.model.embed(A), R); loss = F.cross_entropy(mem.light(z), C)
    zr = mem.hop(mem.model.embed(C), Rinv); loss = loss + F.cross_entropy(mem.light(zr), A)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0 or step == 1:
        with torch.no_grad(): acc = (mem.light(z).argmax(1) == C).float().mean().item()
        print(f"step {step}/{a.steps}  loss {loss.item():.4f}  train exact {acc:.3f}  tau {mem.model.log_tau.exp().item():.2f}  lr {a.lr}  {time.time()-t0:.0f}s", flush=True)
mem.eval(); res = {"config": vars(a), "rows": len(labels), "relations": len(rels), "train_facts": len(train)}
with torch.no_grad():
    # named held-out facts: B composition (withheld from training) through the trained operator, when the operator exists
    for split in ("validation", "test"):
        fs = [f for f in facts(records(split)) if f[3] in mem.id and f[0] in mem.id]; cov = [f for f in fs if (f[1], f[2]) in mem.rid]
        if cov:
            z = mem.hop(mem.embed([f[0] for f in cov]), torch.tensor([mem.rid[(f[1], f[2])] for f in cov], device=dev)); ex, mrr = rank_metrics(mem.light(z), torch.tensor([mem.id[f[3]] for f in cov], device=dev))
            by = {}
            for f, e in zip(cov, ex.tolist()): by.setdefault(f[1], []).append(e)
            res[f"named_{split}"] = {"n": len(fs), "with_operator": len(cov), "exact": ex.mean().item(), "mrr": mrr.mean().item(), "exact_by_op": {k: float(np.mean(v)) for k, v in by.items()}}
            print(f"named {split}: {len(cov)}/{len(fs)} facts have a trained operator; exact {ex.mean():.3f} mrr {mrr.mean():.3f} by op {res[f'named_{split}']['exact_by_op']}", flush=True)
    # composition: (+b)(+c) vs row a+b+c where neither (a, +(b+c)) nor (a+b, +c) is a training fact
    trained = {(x, op, b) for x, op, b, _ in train}; adds = sorted({int(b) for op, b in rels if op == "add"}); trip = []
    for x in labels:
        for b in adds:
            for c in adds:
                s1, s2 = str(int(x) + b), str(int(x) + b + c)
                if s2 in mem.id and ("add", str(b + c)) not in mem.rid and (x, "add", str(b)) not in trained and (s1, "add", str(c)) not in trained: trip.append((x, str(b), str(c), s2))
    if trip:
        trip = [trip[i] for i in rng.choice(len(trip), min(4000, len(trip)), replace=False)]
        z = mem.hop(mem.hop(mem.embed([t[0] for t in trip]), torch.tensor([mem.rid[("add", t[1])] for t in trip], device=dev)), torch.tensor([mem.rid[("add", t[2])] for t in trip], device=dev))
        ex, mrr = rank_metrics(mem.light(z), torch.tensor([mem.id[t[3]] for t in trip], device=dev)); res["compose_2hop"] = {"n": len(trip), "exact": ex.mean().item(), "mrr": mrr.mean().item()}
        print(f"composition (+b)(+c) on {len(trip)} untrained triples: exact {ex.mean():.3f} mrr {mrr.mean():.3f}", flush=True)
    # unnamed: C facts -> spot; is the spot the same for different expressions of the same unnamed number?
    for split in ("validation", "test"):
        fs = [f for f in facts(records(split)) if f[3] not in mem.id and f[0] in mem.id and (f[1], f[2]) in mem.rid]
        if fs:
            z = mem.hop(mem.embed([f[0] for f in fs]), torch.tensor([mem.rid[(f[1], f[2])] for f in fs], device=dev)); sc = spot_consistency(z, [f[3] for f in fs], rng)
            ctrl = spot_consistency(mem.embed([f[0] for f in fs]), [f[3] for f in fs], rng)                       # control: the operand rows themselves (no hop)
            res[f"unnamed_{split}"] = {"n": len(fs), **sc, "control_operand_only": ctrl}
            print(f"unnamed {split}: {len(fs)} facts, same-number cos {sc.get('cos_same', float('nan')):.3f} vs different {sc.get('cos_diff', float('nan')):.3f} purity {sc.get('purity', float('nan')):.3f} | control (operand rows only) purity {ctrl.get('purity', float('nan')):.3f}", flush=True)
    # does the number line show? correlation of row similarity with |a-b| among the dense band
    dense = [l for l in labels if int(l) <= 20]; Z = mem.embed(dense); S = torch.real(Z @ Z.conj().t()).cpu().numpy(); d = np.abs(np.subtract.outer(np.arange(21), np.arange(21)))
    iu = np.triu_indices(21, 1); res["dense_band_cos_vs_distance_corr"] = float(np.corrcoef(S[iu], d[iu])[0, 1]); print(f"dense band: corr(cos, |a-b|) = {res['dense_band_cos_vs_distance_corr']:.3f}")
save_json(res, HERE / a.out / "emergence.json"); torch.save({"labels": labels, "relations": rels, "k": a.k, "block": a.block, "state": mem.state_dict()}, HERE / a.out / "memory.pt")
print(f"saved {a.out}  ({time.time()-t0:.0f}s)")
