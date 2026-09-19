"""Train the learned proposer by imitation on the programs already found on the digit-chain machine, with one operation held
out (the one the search will then have to find). For each training program: random corruptions (1-2 instructions
replaced), each evaluated on examples for its failing-place light; then every position of the corrupted program is an
example: (slot kinds before the position, failing places, previous opcode, part, position) -> the ORIGINAL instruction
there. The policy learns 'given what the slots hold and where the answer fails, what goes here' from add, sub, cmp,
muld, mul, double, triple — never from the held-out program."""
import sys, json, time, numpy as np, torch
ARGS = sys.argv[1:]; HOLD = ARGS[ARGS.index("--holdout") + 1] if "--holdout" in ARGS else "sumd"; NCOR = int(ARGS[ARGS.index("--corruptions") + 1]) if "--corruptions" in ARGS else 40
sys.argv = [sys.argv[0]] + [x for i, x in enumerate(ARGS) if x not in ("--holdout", "--corruptions") and (i == 0 or ARGS[i - 1] not in ("--holdout", "--corruptions"))]
src = open(__file__.replace("proposer_train.py", "digits.py")).read(); src = src[:src.index("# ------------------------------------------------------------------ the five found programs")]
g = {"__name__": "digits"}; exec(compile(src, "digits.py", "exec"), g); globals().update({k: v for k, v in g.items() if not k.startswith("__")})
from proposer import OPS, PARTS, features, train_policy, parse_ins
found = json.load(open(HERE / "results/places/found_programs.json")); PROGRAMS.update(found)
LIB = dict(found)
for f_, key in (("results/discover_io/double.json", "double"), ("results/discover_io/triple.json", "triple"), ("results/digits_find_k20/digits_find.json", "sumd")):
    try:
        d = json.load(open(HERE / f_)); LIB[key] = d[key]["program"]
    except (FileNotFoundError, KeyError): pass
if "double" not in LIB: LIB["double"] = json.load(open(HERE / "results/digits_find_k20/digits_find.json"))["double"]["program"]
for k_ in LIB: LIB[k_].setdefault("kind", "number")
PROGRAMS.update({k_: v for k_, v in LIB.items() if k_ != HOLD})
def digitsum(x): return sum(int(c) for c in str(x))
FN = {"add": lambda x, y: x + y, "sub": lambda x, y: x - y, "cmp": lambda x, y: EQ if x == y else (LT if x < y else GT), "muld": lambda x, y: x * y, "mul": lambda x, y: x * y, "double": lambda x, y: 2 * x, "triple": lambda x, y: 3 * x, "sumd": lambda x, y: digitsum(x)}
def sample_for(name, n, max_digits=3):
    if name in ("add", "sub", "cmp", "muld", "mul"): return sample(n, max_digits, name)
    return [rnum(int(rng.integers(1, max_digits + 1))) for _ in range(n)], [rnum(int(rng.integers(1, max_digits + 1))) for _ in range(n)]
def fail_places(progs, name, xs, ys):
    """per program: fraction of examples wrong at output place 0..3, and beyond (verdicts: the wrong fraction in every entry)"""
    pl = execute(progs, xs, ys); P, E = len(progs), len(xs); f = FN[name]; targets = [f(x, y) for x, y in zip(xs, ys)]
    if LIB[name]["kind"] == "verdict":
        w = (pl.answer.reshape(P, E) != torch.tensor(targets, device=dev)[None]).float().mean(1).cpu().numpy(); return np.stack([w] * 5, 1)
    T = digs([min(t, MAXN - 1) for t in targets]).flip(1).repeat(P, 1); npl = torch.tensor([max(len(str(t)), 1) for t in targets], device=dev).repeat(P); place = torch.arange(NP, device=dev)[None]
    got = digs(pl.result()).flip(1); wrong = ((got != T) & (place < npl[:, None])).float().reshape(P, E, NP).mean(1)
    return torch.cat([wrong[:, :4], wrong[:, 4:].max(1, keepdim=True).values], 1).cpu().numpy()
def flat(p): return [(x, "init") for x in p["init"]] + [(x, "body") for x in p["body"]] + [(x, "outro") for x in p["outro"]]
def unflat(seq, p):
    q = {"kind": p.get("kind", "number"), "init": [x for x, pt in seq if pt == "init"], "body": [x for x, pt in seq if pt == "body"], "outro": [x for x, pt in seq if pt == "outro"], "stop": p["stop"]}
    if not q["body"]: q["body"] = [seq[0][0]]
    return q
def corrupt(p, INS):
    seq = flat(p); q = list(seq)
    for _ in range(int(rng.integers(1, 3))): i = int(rng.integers(len(q))); q[i] = (INS[rng.integers(len(INS))], q[i][1])
    return unflat(q, p)
examples = []; t0 = time.time()
with torch.no_grad():
    INS = instruction_set()
    for name, p in LIB.items():
        if name == HOLD: continue
        variants = [p] + [corrupt(p, INS) for _ in range(NCOR)]; xs, ys = sample_for(name, 8); fails = fail_places(variants, name, xs, ys)
        for v, fl in zip(variants, fails):
            probe = {}; execute([v], [xs[0]], [ys[0]], probe=probe); seq = flat(v); orig = flat(p)
            if len(seq) != len(orig): continue
            for i, (ins, part) in enumerate(seq):
                kinds = probe.get(i) or (probe[max(k for k in probe if k <= i)] if any(k <= i for k in probe) else None)
                if kinds is None: continue
                prev_op = parse_ins(seq[i - 1][0])[0] if i > 0 else "none"; part_len = sum(1 for _, pt in seq if pt == part); pos_in = sum(1 for j in range(i) if seq[j][1] == part)
                examples.append((features(kinds, fl, prev_op, part, pos_in / max(part_len, 1), K), orig[i][0]))
        print(f"  {name}: {len(variants)} variants, mean failing places {fails[1:].mean():.2f}, examples so far {len(examples)}  ({time.time()-t0:.0f}s)", flush=True)
print(f"training the proposer on {len(examples)} examples from {[n for n in LIB if n != HOLD]} (held out: {HOLD})", flush=True)
m = train_policy(examples, K, dev, steps=4000)
(HERE / "results/proposer").mkdir(parents=True, exist_ok=True); out = HERE / f"results/proposer/proposer_holdout_{HOLD}.pt"; torch.save({"model": m.state_dict(), "K": K, "holdout": HOLD, "n_examples": len(examples)}, out); print(f"saved {out} ({time.time()-t0:.0f}s)", flush=True)
