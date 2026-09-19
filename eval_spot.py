"""Re-score saved V4 checkpoints with the label-free exact test:
  compose the expression into a SPOT (a native state; the number may have no row), then decompose the spot with the
  place primitives and read each place at the anchor rows (d*10^k, or row 0).  Exact = every place right.
This needs no row for the result, so it applies to strict-unseen C numbers exactly as to B/A/G.  Scoring uses the
record's identity value only on the evaluator side (canonical terms), never in the model input.
Also reported: rank of the target row when it exists (named), and the lights' top-1 label per place for inspection."""
import argparse, json, collections, numpy as np, torch
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); a = ap.parse_args(); dev = torch.device("cuda")


def canonical(n):
    return [str(int(d) * 10 ** k) if d != "0" else "0" for k, d in enumerate(reversed(str(n)))] + ["0"] * (5 - len(str(n)))


def template(e, leaves_, flat, root):
    if isinstance(e, str): leaves_.append(e); return len(leaves_) - 1
    tpl = {"op": e["op"], "args": [template(x, leaves_, flat, root) for x in e["args"]]}
    if flat and e is root: return {"op": "combine", "args": list(range(len(leaves_)))}
    return tpl


for run in a.runs:
    ck = torch.load(HERE / run / "v4.pt", map_location="cpu", weights_only=False); cfg = ck["config"]; mm = ck["memory"]
    mem = Memory(mm["labels"], [("none", "0")], k=mm["k"], block_size=mm["block"]); mem.load_state_dict(mm["state"]); mem.to(dev).eval()
    ops = cfg.get("ops", "labeled"); reasoner = Reasoner(mem.M, mm["block"], n_gen=cfg["n_gen"], prims=("add", "multiply") if ops == "labeled" else ("combine",), both=(ops == "none")).to(dev)
    if "log_tau" in ck["reasoner"]: reasoner.log_tau = nn.Parameter(torch.zeros(()))
    reasoner.load_state_dict(ck["reasoner"]); reasoner.eval(); out = {}
    with torch.no_grad():
        for split in ("validation", "test", "train"):
            g = collections.defaultdict(list)
            for r in records(split):
                if r["task"] != "compose": continue
                mi = model_input(r); lv = []; tpl = template(mi.inputs["expression"], lv, cfg.get("flat", False), mi.inputs["expression"])
                if all(l in mem.id for l in lv): g[json.dumps(tpl)].append((tpl, lv, r["metadata"]["identity"], "+".join(r["metadata"]["buckets"]) + f" [{r['metadata']['pool']}]"))
            per = collections.defaultdict(list)
            for key, items in g.items():
                tpl = items[0][0]; cols = [mem.embed([it[1][i] for it in items]) for i in range(len(items[0][1]))]
                def go(e):
                    if isinstance(e, int): return cols[e]
                    p = reasoner.pid[e["op"]]; acc = go(e["args"][0])
                    for x in e["args"][1:]: acc = reasoner.apply(p, acc, go(x))
                    return acc
                z = go(tpl); pred = [[mem.labels[i] for i in mem.light(reasoner.place(k, z)).argmax(1).tolist()] for k in range(reasoner.n_place)]
                for i, it in enumerate(items):
                    tg = canonical(it[2]); ok = all(pred[k][i] == tg[k] for k in range(5)); per[it[3]].append(ok)
            out[split] = {bk: {"n": len(v), "spot_decompose_exact": float(np.mean(v))} for bk, v in sorted(per.items())}
    print(f"== {run}  (ops={ops}, flat={cfg.get('flat', False)}, frozen={bool(cfg.get('frozen'))}, buckets={cfg['buckets']}, M={mem.M})")
    for split in ("validation", "test", "train"):
        for bk, o in out[split].items():
            if split == "train" and o["n"] < 50: continue
            print(f"  {split:10s} {bk:52s} n={o['n']:4d}  spot->decompose exact {o['spot_decompose_exact']:.3f}")
    save_json(out, HERE / run / "spot_decompose.json")
