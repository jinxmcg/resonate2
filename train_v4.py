"""Step 2 — V4: memory rows (states) + primitive reasoner (relationships), trained through the loop.
Every compose training record is folded bottom-up (one node per iteration, intermediates unsupervised); the ONLY loss is
the cross-entropy of the root response's full light at the target row.  --frozen <memory.pt> freezes the rows (V4-v1,
grounded first); otherwise rows and reasoner train together from random rows (V4-v2, joint).
--buckets selects which expression families train (dense_basic, place_value, arithmetic_variation, grouping_change, ...).
Evaluation (validation and test, every bucket separately):
  named   (B pool)   : exact top-1 / MRR of the target row in the root light (all 221 rows are candidates)
  unnamed (C pools)  : spot consistency of the root state across different expressions of the same unseen number,
                       against different numbers; control = the same statistic on the first operand's row
  intermediates      : for place-value trees, does the unsupervised (d×10^k) node light its anchor row? (diagnostic)
"""
import argparse, json, time, collections, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=8); ap.add_argument("--block", type=int, default=16); ap.add_argument("--n-gen", type=int, default=16)
ap.add_argument("--buckets", default="all"); ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--frozen", default=""); ap.add_argument("--ops", default="labeled", choices=["labeled", "none"]); ap.add_argument("--flat", action="store_true", help="no grouping either: fold the leaf sequence left in the given order"); ap.add_argument("--out", default="results/v4_joint_k8"); ap.add_argument("--report-every", type=int, default=250)
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time()
labels = grounded_labels(); want = None if a.buckets == "all" else set(a.buckets.split(","))


def template(e, leaves_):
    if isinstance(e, str): leaves_.append(e); return len(leaves_) - 1
    tpl = {"op": e["op"], "args": [template(x, leaves_) for x in e["args"]]}
    if a.flat and not leaves_ is None and e is _root[0]: return {"op": "combine", "args": list(range(len(leaves_)))}       # grouping discarded
    return tpl
_root = [None]


def groups(recs, bucket_filter=None):
    """Group compose records by tree shape so a whole group folds as one batch. Returns {shape_json: (template, leaves(B,L) labels, targets, ids, buckets)}"""
    g = collections.defaultdict(list)
    for r in recs:
        if r["task"] != "compose": continue
        if bucket_filter is not None and not (set(r["metadata"]["buckets"]) & bucket_filter): continue
        mi = model_input(r); lv = []; _root[0] = mi.inputs["expression"]; tpl = template(mi.inputs["expression"], lv)
        g[json.dumps(tpl)].append((tpl, lv, "".join(r["target"]["digits"]), r["metadata"]["identity"], tuple(r["metadata"]["buckets"]), r["metadata"]["pool"]))
    return g


if a.frozen:
    ck = torch.load(HERE / a.frozen, map_location="cpu"); mem = Memory(ck["labels"], ck["relations"], k=ck["k"], block_size=ck["block"]); mem.load_state_dict(ck["state"]); mem.to(dev); mem.requires_grad_(False); a.k, a.block = ck["k"], ck["block"]
else:
    mem = Memory(labels, [("none", "0")], k=a.k, block_size=a.block).to(dev)
reasoner = Reasoner(mem.M, a.block, n_gen=a.n_gen, prims=("add", "multiply") if a.ops == "labeled" else ("combine",), both=(a.ops == "none")).to(dev)
if a.frozen:                                                                                    # frozen table: the reasoner owns the loss temperature
    reasoner.log_tau = nn.Parameter(torch.tensor(float(mem.model.log_tau)))
    light_fn = lambda z: torch.real(z @ mem.E.conj().t()) * reasoner.log_tau.exp()
else: light_fn = mem.light
train = groups(records("train"), want); n_train = sum(len(v) for v in train.values())
# training operands must be grounded rows (they are: all train compose leaves are grounded numerals)
print(f"M={mem.M} block={a.block} gens={a.n_gen}  train records {n_train} in {len(train)} shapes  buckets={a.buckets}  frozen={bool(a.frozen)}  reasoner params {sum(p.numel() for p in reasoner.parameters())}  rows {len(mem.labels)}x{mem.M}", flush=True)


def fold_group(tpl, lv_batch, states_of):
    """lv_batch: list of leaf-label lists (B, L). states_of(labels list)->(B,M). Folds the template; returns root state and node trace."""
    L = len(lv_batch[0]); cols = [states_of([lv[i] for lv in lv_batch]) for i in range(L)]; trace = []
    def go(e):
        if isinstance(e, int): return cols[e]
        p = reasoner.pid[e["op"]]; acc = go(e["args"][0])
        for x in e["args"][1:]: acc = reasoner.apply(p, acc, go(x)); trace.append((e["op"], acc))
        return acc
    return go(tpl), trace


def decomp(recs, pools=None):
    """decompose records whose input numeral has a row: (label, [target row label per place 0..4]); absent place -> row '0'."""
    out = []
    for r in recs:
        if r["task"] != "decompose": continue
        mi = model_input(r); lab = mi.number_literals[0]
        if lab not in mem.id: continue
        terms = {int(t["power10"]): t["coefficient"] for t in r["target"]["terms"]}
        tg = [str(int(terms[k]) * 10 ** k) if k in terms and terms[k] != "0" else "0" for k in range(reasoner.n_place)]
        if all(x in mem.id for x in tg): out.append((lab, tg, r["metadata"]["pool"]))
    return out


dtrain = decomp(records("train")) if (want is None or "decompose" in want) else []; print(f"decomposition train records {len(dtrain)}", flush=True)
def decomp_loss(items):
    z = mem.embed([it[0] for it in items]); tot, hits = 0.0, torch.ones(len(items), dtype=torch.bool, device=dev)
    for k in range(reasoner.n_place):
        light = light_fn(reasoner.place(k, z)); tgt = torch.tensor([mem.id[it[1][k]] for it in items], device=dev)
        tot = tot + F.cross_entropy(light, tgt, reduction="sum"); hits &= light.argmax(1) == tgt
    return tot, hits


params = list(reasoner.parameters()) + ([] if a.frozen else list(mem.parameters())); opt = torch.optim.Adam(params, lr=a.lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
for step in range(1, a.steps + 1):
    tot, correct = 0.0, 0
    for key, items in train.items():
        tpl = items[0][0]; z, _ = fold_group(tpl, [it[1] for it in items], mem.embed); tgt = torch.tensor([mem.id[it[2]] for it in items], device=dev)
        light = light_fn(z); loss = F.cross_entropy(light, tgt, reduction="sum"); tot = tot + loss; correct += (light.argmax(1) == tgt).sum().item()
    dacc = float("nan")
    if dtrain: dl, hits = decomp_loss(dtrain); tot = tot + dl / reasoner.n_place; dacc = hits.float().mean().item()
    opt.zero_grad(); (tot / (n_train + len(dtrain))).backward(); opt.step(); sched.step()
    if step % a.report_every == 0 or step == 1:
        print(f"step {step}/{a.steps}  loss {tot.item()/(n_train+len(dtrain)):.4f}  train compose exact {correct/n_train:.3f}  decompose exact {dacc:.3f}  lr {sched.get_last_lr()[0]:.2e}  tau {(reasoner.log_tau if a.frozen else mem.model.log_tau).exp().item():.1f}  {time.time()-t0:.0f}s", flush=True)

# ------------------------------------------------------------------ evaluation
reasoner.eval(); res = {"config": vars(a), "train_records": n_train, "reasoner_params": sum(p.numel() for p in reasoner.parameters())}
anchor = lambda d, p: str(int(d) * int(p)) if int(p) in (1, 10, 100, 1000, 10000) and 1 <= int(d) <= 9 else None
with torch.no_grad():
    for split in ("validation", "test", "train"):
        allg = groups(records(split)); named, unnamed = collections.defaultdict(list), collections.defaultdict(list); inter = collections.defaultdict(list)
        for key, items in allg.items():
            tpl = items[0][0]; ok = [it for it in items if all(l in mem.id for l in it[1])]                  # operands must have rows (C operands do not)
            if not ok: continue
            z, trace = fold_group(tpl, [it[1] for it in ok], mem.embed); light = mem.light(z)
            for i, it in enumerate(ok):
                bk = "+".join(it[4]) + f" [{it[5]}]"
                if it[2] in mem.id: ex, mrr = rank_metrics(light[i:i + 1], torch.tensor([mem.id[it[2]]], device=dev)); named[bk].append((ex.item(), mrr.item()))
                else: unnamed[bk].append((z[i], it[3]))
            # intermediate diagnostic: multiply nodes whose operands are (digit, power of ten) -> anchor row lit?
            def walk(e, tr):
                if isinstance(e, int): return None
                for x in e["args"]: walk(x, tr)
                node = tr.pop(0)
                if e["op"] == "multiply" and len(e["args"]) == 2 and all(isinstance(x, int) for x in e["args"]):
                    for i, it in enumerate(ok):
                        d, p = it[1][e["args"][0]], it[1][e["args"][1]]; an = anchor(d, p) or anchor(p, d)
                        if an and an in mem.id: inter[split].append(float(mem.light(node[1][i:i + 1]).argmax(1).item() == mem.id[an]))
            tr = list(trace); walk(tpl, tr)
        out = {}
        for bk, v in sorted(named.items()): out[bk] = {"n": len(v), "exact": float(np.mean([x[0] for x in v])), "mrr": float(np.mean([x[1] for x in v]))}
        for bk, v in sorted(unnamed.items()):
            st = torch.stack([x[0] for x in v]); sc = spot_consistency(st, [x[1] for x in v], rng); out[bk] = {"n": len(v), **sc}
        dv = decomp(records(split))
        for pool in sorted({it[2] for it in dv}):
            items = [it for it in dv if it[2] == pool]; _, hits = decomp_loss(items); out[f"decompose [{pool}]"] = {"n": len(items), "exact_canonical": hits.float().mean().item()}
        if inter[split]: out["intermediate_anchor_hit"] = {"n": len(inter[split]), "rate": float(np.mean(inter[split]))}
        res[split] = out
        print(f"== {split}"); [print(f"  {bk:55s} n={o['n']:4d}  " + (f"exact {o['exact']:.3f}  mrr {o['mrr']:.3f}" if "exact" in o else f"spot: same {o.get('cos_same', float('nan')):.3f} diff {o.get('cos_diff', float('nan')):.3f} purity {o.get('purity', float('nan')):.3f}" if "n_same" in o else f"anchor-hit {o['rate']:.3f}" if "rate" in o else f"exact canonical {o['exact_canonical']:.3f}")) for bk, o in out.items()]
save_json(res, HERE / a.out / "results.json"); torch.save({"memory": {"labels": mem.labels, "k": mem.k, "block": mem.block_size, "state": mem.state_dict()}, "reasoner": reasoner.state_dict(), "config": vars(a)}, HERE / a.out / "v4.pt")
print(f"saved {a.out} ({time.time()-t0:.0f}s)")
