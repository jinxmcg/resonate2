"""Step 1 (strict) — ground the memory WITHOUT arithmetic: it only sees which numerals occur together.
Each training record contributes a bag of numerals (its operands; with --bag equation also its result / its decomposition
anchors). No operator, no order, no tree. Objective: for every pair (i, j) in a bag, the light of row i must be bright at
row j (softmax over all rows) — co-occurrence in the same area, nothing else. The table is then frozen for the reasoner.
"""
import argparse, time, itertools, collections, numpy as np, torch, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=8); ap.add_argument("--block", type=int, default=16); ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--bag", default="operands", choices=["operands", "equation"]); ap.add_argument("--buckets", default="all"); ap.add_argument("--lr", type=float, default=5e-3); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results/bag_k8")
ap.add_argument("--oracle-line", action="store_true", help="CONTROL: rows are a fixed systematic number line z_n[j]=exp(i*theta_j*n) (encodes magnitude; not a learned memory)")
a = ap.parse_args(); torch.manual_seed(a.seed); dev = torch.device("cuda"); t0 = time.time()
labels = grounded_labels(); mem = Memory(labels, [("none", "0")], k=a.k, block_size=a.block).to(dev); want = None if a.buckets == "all" else set(a.buckets.split(","))
bags = []
for r in records("train"):
    if want is not None and not (set(r["metadata"]["buckets"]) & want): continue
    mi = model_input(r); bag = set(mi.number_literals)
    if a.bag == "equation":
        if r["task"] == "compose": bag.add("".join(r["target"]["digits"]))
        elif r["task"] == "decompose": bag |= {str(int(t["coefficient"]) * 10 ** int(t["power10"])) for t in r["target"]["terms"]}
    bag = [x for x in bag if x in mem.id]
    if len(bag) >= 2: bags.append(bag)
pairs = torch.tensor([(mem.id[i], mem.id[j]) for b in bags for i, j in itertools.permutations(b, 2)], device=dev)
print(f"bags {len(bags)}  pairs {len(pairs)}  rows {len(labels)}  M={mem.M}  bag={a.bag}", flush=True)
if a.oracle_line:
    with torch.no_grad():
        theta = torch.tensor(np.exp(np.random.default_rng(a.seed).uniform(np.log(2 * np.pi / 20000), np.log(2 * np.pi / 2), mem.M)), dtype=torch.float32, device=dev)
        n = torch.tensor([float(l) for l in labels], device=dev); mem.model.E.copy_(cnorm(torch.polar(torch.ones(len(n), mem.M, device=dev), n[:, None] * theta[None, :])))
    a.steps = 0
opt = torch.optim.Adam(mem.parameters(), lr=a.lr)
for step in range(1, a.steps + 1):
    light = mem.light(mem.model.embed(pairs[:, 0])); loss = F.cross_entropy(light, pairs[:, 1])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0 or step == 1: print(f"step {step}/{a.steps}  loss {loss.item():.4f}  tau {mem.model.log_tau.exp().item():.1f}  {time.time()-t0:.0f}s", flush=True)
with torch.no_grad():
    Z = mem.E; S = torch.real(Z @ Z.conj().t()).cpu().numpy(); n = np.array([int(l) for l in labels]); iu = np.triu_indices(len(n), 1)
    corr = float(np.corrcoef(S[iu], np.log1p(np.abs(np.subtract.outer(n, n)))[iu])[0, 1])
    cov = collections.Counter(x for b in bags for x in b); seen = sum(1 for l in labels if cov[l] > 0)
print(f"rows with any co-occurrence signal {seen}/{len(labels)};  corr(row cosine, log|a-b|) = {corr:.3f}  (memory knows nothing of arithmetic by construction; this is a diagnostic)")
save_json({"config": vars(a), "bags": len(bags), "pairs": len(pairs), "rows_with_signal": seen, "cos_vs_logdist_corr": corr}, HERE / a.out / "grounding.json")
torch.save({"labels": labels, "relations": [("none", "0")], "k": a.k, "block": a.block, "state": mem.state_dict()}, HERE / a.out / "memory.pt"); print(f"saved {a.out}")
