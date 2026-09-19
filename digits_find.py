"""Discovering NEW operations on the digit chain from three worked examples, with the found add/sub/cmp/muld/mul callable
(Part 14's search on the machine of Part 23). Candidates that work units-up, as every program here does:
  sumd   — the sum of a number's digits (two accumulators: units in the value code, tens by carries; a structure no
           earlier program has), result of up to three digits;
  double — x + x through a call with the same operand twice.
Operations that read a number from the top (halving, division, square root) are not attempted: the chain walks units
first, and 'reading from the top' is the missing move (Part 14) — reversing a number (rev = rev*10 + d) would give it.
Ground-truth programs are used only to check the machine and to produce the three worked traces; acceptance requires
exactness on numbers longer than any shown plus the carry regression patterns."""
import argparse, sys, json, time, numpy as np
sys.argv = [sys.argv[0]] + [x for x in sys.argv[1:]]
FIND = [x for i, x in enumerate(sys.argv[1:]) if sys.argv[i] == "--find"]; sys.argv = [x for i, x in enumerate(sys.argv) if x != "--find" and (i == 0 or sys.argv[i - 1] != "--find")]
src = open(__file__.replace("digits_find.py", "digits.py")).read(); src = src[:src.index("# ------------------------------------------------------------------ the five found programs")]
g = {"__name__": "digits"}; exec(compile(src, "digits.py", "exec"), g); globals().update({k: v for k, v in g.items() if not k.startswith("__")})
found = json.load(open(HERE / "results/places/found_programs.json")); PROGRAMS.update(found)
def digitsum(x): return sum(int(c) for c in str(x))
SUMD = {"kind": "number", "init": ["SET 1<0", "SET 2<0", "SET 4<0", "SET 5<0"], "body": ["LOAD 0>3", "BIND 2,3", "READ 2", "SETD 2", "SETC 3", "BIND 4,3", "READ 4", "SETD 4", "SETC 3", "BIND 5,3", "READ 5", "SETD 5"], "outro": ["WRITE 2", "WRITE 4", "WRITE 5"], "stop": 1}
DOUBLE = {"kind": "number", "init": ["COPY 0>1", "SET 3<0"], "body": ["CALL add 0,1>2"], "outro": ["ANSWER 2"], "stop": 3}
TRUTH.update({"sumd": (SUMD, lambda x, y: digitsum(x)), "double": (DOUBLE, lambda x, y: 2 * x)})
# sumd regression includes sums of 108, 216 and 160: three digits (a first found program wrote a constant 0 for the hundreds)
REGRESSION.update({"sumd": [(0, 0), (9, 0), (99, 0), (999999999, 0), (1000000000, 0), (123456789, 0), (909090909090, 0), (999999999999, 0), (int("9" * 24), 0), (int("8" * 20), 0)], "double": [(0, 0), (5, 0), (50, 0), (99, 0), (999999, 0), (123456789012, 0)]})
_sample = sample
def sample(n, max_digits, task):
    if task not in ("sumd", "double"): return _sample(n, max_digits, task)
    xs, ys = [], []
    for _ in range(n): L = int(rng.integers(1, max_digits + 1)); xs.append(rnum(L)); ys.append(0)
    return xs, ys
g["sample"] = sample; globals()["sample"] = sample
def demos(name, n=3):
    prog = TRUTH[name][0]; out = []
    for _ in range(n):
        xs, ys = sample(1, a.max_digits, name); cycles = max(len(str(xs[0])), 1)
        out.append((xs[0], ys[0], list(prog["init"]) + [i for _ in range(cycles) for i in prog["body"]] + list(prog["outro"])))
    return out
def from_demos(dm, kind):
    x, y, tr = dm[rng.integers(len(dm))]; blen = int(rng.integers(2, 13)); start = int(rng.integers(0, max(1, len(tr) - blen + 1))); body = tr[start:start + blen]
    init = tr[:start][-int(rng.integers(0, 4)):] if start > 0 and rng.random() < 0.7 else []; outro = tr[-int(rng.integers(1, 7)):] if rng.random() < 0.7 else []
    return {"kind": kind, "init": list(init), "body": list(body), "outro": list(outro), "stop": int(rng.integers(0, K))}
def mutate(p):
    INS = instruction_set(); q = {k: (list(v) if isinstance(v, list) else v) for k, v in p.items()}; part = rng.choice(["body", "body", "body", "init", "outro"]); seq = q[part]; r = rng.random()
    if r < 0.4 and seq: seq[rng.integers(len(seq))] = INS[rng.integers(len(INS))]
    elif r < 0.6 and len(seq) < 13: seq.insert(int(rng.integers(len(seq) + 1)), INS[rng.integers(len(INS))])
    elif r < 0.75 and len(seq) > (1 if part == "body" else 0): seq.pop(int(rng.integers(len(seq))))
    elif r < 0.9 and len(seq) > 1: i, j = rng.integers(len(seq), size=2); seq[i], seq[j] = seq[j], seq[i]
    else: q["stop"] = int(rng.integers(K))
    if not q["body"]: q["body"] = [INS[rng.integers(len(INS))]]
    return q
def show(p): return f"init [{', '.join(p['init'])}]  body [{', '.join(p['body'])}]  outro [{', '.join(p['outro'])}]  stop {p['stop']}"
def score_regression(prog, name):
    rx, ry = zip(*REGRESSION[name]); f, e = score([prog], name, list(rx), list(ry)); return float(e[0])
results = {}
with torch.no_grad():
    for name in (FIND or ["sumd", "double"]):
        v = verify(TRUTH[name][0], name, lengths=(1, 3, 6, 12), n=60); print(f"ground truth {name} on the digit chain: " + " ".join(f"{L}d {e:.3f}" if L != "reg" else f"regression {e:.3f}" for L, e in v.items()), flush=True)
        kind = TRUTH[name][0]["kind"]; dm = demos(name); print(f"\n== {name}: worked examples " + "; ".join(f"{x} ({len(tr)} steps)" for x, y, tr in dm) + f"   callable: {sorted(PROGRAMS)}", flush=True)
        pop = [from_demos(dm, kind) for _ in range(a.pop)]; best, best_f = None, -1; found_p = None
        for gen in range(1, a.gens + 1):
            xs, ys = sample(a.examples, a.max_digits, name); f, e = score(pop, name, xs, ys); order = np.argsort(-f)
            if f[order[0]] > best_f: best, best_f = pop[order[0]], float(f[order[0]])
            if gen % 5 == 0 or gen == 1: print(f"  gen {gen}  best {f[order[0]]:.3f} (exact {e[order[0]]:.3f})  mean {f.mean():.3f}  elapsed {time.time()-t0:.0f}s", flush=True)
            if e[order[0]] >= 1.0:
                cands = [pop[i] for i in order if e[i] >= 1.0]; cands.sort(key=lambda p: len(p["init"]) + len(p["body"]) + len(p["outro"]))
                for cand in cands[:5]:
                    xs2, ys2 = sample(48, a.max_digits + 3, name); fc, ec = score([cand], name, xs2, ys2)
                    if ec[0] >= 1.0 and score_regression(cand, name) >= 1.0: found_p = cand; break
                if found_p is not None: print(f"  found at generation {gen}: {show(found_p)}", flush=True); break
            elite = [pop[i] for i in order[:max(2, a.pop // 20)]]; new = list(elite)
            while len(new) < a.pop:
                i, j = order[rng.integers(0, max(4, a.pop // 4), size=2)]; child = mutate(pop[i] if f[i] >= f[j] else pop[j]); new.append(mutate(child) if rng.random() < 0.3 else child)
            pop = new
        prog = found_p or best; v = verify(prog, name, lengths=(1, 2, 3, 4, 6, 8, 12, 16, 24), n=60)
        print(f"  {name} winner by digits (>{a.max_digits} never seen): " + " ".join(f"{L}d {e:.3f}" if L != "reg" else f"regression {e:.3f}" for L, e in v.items()), flush=True)
        results[name] = {"found": found_p is not None, "generation": gen, "program": prog, "verify": v}
        if found_p is not None: PROGRAMS[name] = prog
save_json(results, HERE / a.out / "digits_find.json"); print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)
