"""Aggregate the ten-seed CLUTRR runs (results/kin_seeds/s*/kinship_ops.json): mean ± std of exact / abstain / wrong by hops,
the per-seed ten-hop exact, and validation. Writes results/kin_seeds/summary.json and prints a markdown table."""
import json, glob, numpy as np
from common import HERE
runs = sorted(glob.glob(str(HERE / "results/kin_seeds/s*/kinship_ops.json"))); R = [json.load(open(p)) for p in runs]; print(f"{len(R)} seeds")
hops = sorted(int(h) for h in R[0]["test"]); out = {"n_seeds": len(R), "by_hops": {}}
for h in hops:
    for k in ("exact", "abstain", "wrong"):
        v = np.array([r["test"][str(h)][k] for r in R]); out["by_hops"].setdefault(str(h), {})[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "min": float(v.min()), "max": float(v.max())}
ten = [r["test"]["10"]["exact"] for r in R]; out["ten_hop_exact_per_seed"] = ten
allv = {k: [np.mean([r["test"][str(h)][k] for h in hops]) for r in R] for k in ("exact", "abstain", "wrong")}
print("| hops | " + " | ".join(str(h) for h in hops) + " |"); print("|---|" + "---|" * len(hops))
for k in ("exact", "abstain", "wrong"): print(f"| {k} | " + " | ".join(f"{out['by_hops'][str(h)][k]['mean']:.3f} ± {out['by_hops'][str(h)][k]['std']:.3f}" for h in hops) + " |")
print(f"ten-hop exact per seed: {' '.join(f'{x:.3f}' for x in ten)}  → {np.mean(ten):.3f} ± {np.std(ten, ddof=1):.3f} (min {min(ten):.3f}, max {max(ten):.3f})")
json.dump(out, open(HERE / "results/kin_seeds/summary.json", "w"), indent=1)
