"""Discovery from input-output pairs ALONE — no worked traces anywhere — on the digit-chain machine (Part 23) with the found
programs callable. The population starts from random programs (blank slate) or from a neighbouring program (the ladder);
the fitness sees only (x, y) -> answer pairs. Slope without traces: per-digit credit (as before) plus credit for the
right value sitting in ANY slot at the end (getting the number somewhere counts before learning to ANSWER it);
acceptance still requires the official answer to be exact on numbers longer than any shown and on the regression set.
Rungs: double from (x, 2x) blank-slate; triple from (x, 3x) seeded with double; digit sum from (x, digit sum) seeded
with add. Part 13 showed blank-slate search of addition has no slope; this asks whether a library and a ladder give it."""
import argparse, sys, json, time, numpy as np
ARGS = sys.argv[1:]; VALUED = ("--target", "--seed-program", "--proposer", "--ptemp", "--tier1"); FLAGS = ("--any-slot", "--typed", "--worked")
sys.argv = [sys.argv[0]] + [x for i, x in enumerate(ARGS) if x not in VALUED + FLAGS and (i == 0 or ARGS[i - 1] not in VALUED)]
def opt(name, default=None):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default
TARGET = opt("--target", "double"); SEEDP = opt("--seed-program", ""); ANY = "--any-slot" in ARGS; PROP = opt("--proposer", ""); PTEMP = float(opt("--ptemp", "1.0")); WORKED = "--worked" in ARGS; TIER1 = int(opt("--tier1", "0")); GEN = [0]
PLUMBING = ("CALL", "LOAD", "NUM", "COPY", "SET", "SHIFT", "ANSWER")
def named_ops():
    """the operations the worked examples name (a teacher says 'add'): the callables offered in tier 1"""
    try: return {c[0] for x, y in zip(*sample(6, a.max_digits, TARGET)) for c in expected_calls(x, y)} or None
    except Exception: return None
def allowed_ins():
    """reuse before invention: for the first --tier1 generations only calls to the NAMED library programs and plumbing; then everything"""
    INS = instruction_set()
    if GEN[0] > TIER1: return INS
    named = named_ops(); return [n for n in INS if (n.split()[0] in PLUMBING and n.split()[0] != "CALL") or (n.split()[0] == "CALL" and (named is None or n.split()[1] in named))]
src = open(__file__.replace("discover_io.py", "digits.py")).read(); src = src[:src.index("# ------------------------------------------------------------------ the five found programs")]
g = {"__name__": "digits"}; exec(compile(src, "digits.py", "exec"), g); globals().update({k: v for k, v in g.items() if not k.startswith("__")})
found = json.load(open(HERE / "results/places/found_programs.json")); PROGRAMS.update(found)
for extra in ("results/discover_io/double.json", "results/discover_io/triple.json"):                                  # earlier rungs, if found, are callable
    try: PROGRAMS.update({k: v["program"] for k, v in json.load(open(HERE / extra)).items() if v.get("found")})
    except FileNotFoundError: pass
def digitsum(x): return sum(int(c) for c in str(x))
TASKS = {"double": lambda x, y: 2 * x, "triple": lambda x, y: 3 * x, "sumd": lambda x, y: digitsum(x), "addy2": lambda x, y: x + 2 * y}
REG = {"double": [(0, 3), (5, 0), (50, 7), (99, 99), (999999, 1), (123456789, 42)], "triple": [(0, 9), (4, 0), (34, 34), (99, 1), (333333, 5), (123456789, 8)],
       "sumd": [(0, 7), (9, 0), (99, 3), (999999999, 0), (1000000000, 5), (123456789, 0), (909090909090, 1), (999999999999, 0), (9999999999999, 2), (8888888888888, 0)], "addy2": [(0, 0), (1, 1), (99, 1), (5, 99), (123456, 654321)]}
TRUTH[TARGET] = ({"kind": "number"}, TASKS[TARGET]); REGRESSION[TARGET] = REG[TARGET]
_sample = sample
def sample(n, max_digits, task):
    if task not in TASKS: return _sample(n, max_digits, task)
    xs, ys = [], []
    for _ in range(n): L = int(rng.integers(1, max_digits + 1)); xs.append(rnum(L)); ys.append(rnum(int(rng.integers(1, max_digits + 1))))      # unary tasks get a RANDOM second operand: a found program cannot lean on it
    return xs, ys
g["sample"] = sample                                                                                                  # the machine's verify uses the same sampler
FAILS = []
def expected_calls(x, y):
    """the worked example's named operations: '123: add(1,2)=3, add(3,3)=6' — as library calls with their operands (order-free)"""
    ds = [int(c) for c in str(x)][::-1]
    if TARGET == "sumd":
        calls, tot = [], ds[0]
        for d in ds[1:]: calls.append(("add", tot, d)); tot += d
        return calls
    if TARGET == "double": return [("add", x, x)]
    if TARGET == "triple": return [("add", x, x), ("add", 2 * x, x)]
    return []
def intermediates(x, y):
    """the worked example in the machine's terms: the values a teacher would write on the way ('123 = 3 + 2 + 1 = 6' read units first)"""
    ds = [int(c) for c in str(x)][::-1]
    if TARGET == "sumd": run = list(np.cumsum(ds)); return sorted({*ds, *run})
    if TARGET == "double": return [x, 2 * x]
    if TARGET == "triple": return [x, 2 * x, 3 * x]
    return [f(x, y) for f in (TASKS[TARGET],)]
def score_io(progs, xs, ys):
    """fitness: per-digit credit of the official result + exact bonus, plus (if --any-slot) the best per-digit credit over all walkable slots; exact from the official result only"""
    pl = execute(progs, xs, ys, track=WORKED); P, E = len(progs), len(xs); f = TASKS[TARGET]; targets = [f(x, y) for x, y in zip(xs, ys)]
    T = digs([min(t, MAXN - 1) for t in targets]).flip(1).repeat(P, 1); npl = torch.tensor([max(len(str(t)), 1) for t in targets], device=dev).repeat(P); place = torch.arange(NP, device=dev)[None]
    got = digs(pl.result()).flip(1); credit = ((got == T) & (place < npl[:, None])).float().sum(1) / npl.float(); ex = (got == T).all(1).float()
    wrong = ((got != T) & (place < npl[:, None])).float().reshape(P, E, NP).mean(1); FAILS[:] = torch.cat([wrong[:, :4], wrong[:, 4:].max(1, keepdim=True).values], 1).cpu().numpy().tolist()
    if ANY:
        best = credit.clone(); ar = torch.arange(P * E, device=dev)
        for s in range(K):
            wk = pl.walkable[:, s]
            if wk.any():
                idx = ar[wk]; gs = digs(pl.number_of(idx, s)).flip(1); c = ((gs == T[idx]) & (place < npl[idx][:, None])).float().sum(1) / npl[idx].float(); best[idx] = torch.maximum(best[idx], c)
        credit = 0.5 * credit + 0.5 * best
    if WORKED:                                                                                                       # credit for every named intermediate that lit somewhere during execution
        inter = [[v for v in intermediates(x, y) if v < 256] for x, y in zip(xs, ys)]; seen = pl.seen.cpu().numpy()
        wc = torch.tensor([float(np.mean([seen[r, v] for v in inter[r % E]])) if inter[r % E] else 0.0 for r in range(P * E)], device=dev); credit = credit + wc
        exp = [expected_calls(x, y) for x, y in zip(xs, ys)]
        def hit(r):
            made = pl.calls_seen[r]; want = exp[r % E]
            return float(np.mean([((n, u, v) in made) or ((n, v, u) in made) for n, u, v in want])) if want else 0.0
        credit = credit + torch.tensor([hit(r) for r in range(P * E)], device=dev)                                  # credit for the named library calls actually made
    return (credit + ex).reshape(P, E).mean(1).cpu().numpy(), ex.reshape(P, E).mean(1).cpu().numpy()
def random_program():
    INS = allowed_ins(); r = lambda n: [INS[rng.integers(len(INS))] for _ in range(n)]
    return {"kind": "number", "init": r(int(rng.integers(0, 3))), "body": r(int(rng.integers(1, 6))), "outro": r(int(rng.integers(0, 3))), "stop": int(rng.integers(0, K))}
TYPED = "--typed" in ARGS
import re as _re
def applicable(name, kinds):
    """can this instruction act on what the slots hold (walkable numbers / digits / verdicts)? the lights type the proposals"""
    walk, sym = kinds; op = name.split()[0]
    rest = name.split(" ", 2)[2] if op == "CALL" else (name.split(" ", 1)[1] if " " in name else "")
    nums = [int(x) for x in _re.findall(r"\d+", rest.replace("<EQ", ""))]
    isdig = lambda i: 0 <= sym[i] <= 9 and not walk[i]; isnum = lambda i: bool(walk[i]); isval = lambda i: (not walk[i]) and sym[i] != -9
    if op in ("LOAD", "SHIFT"): return isnum(nums[0])
    if op == "NUM": return isdig(nums[0])
    if op in ("BIND", "UNBIND", "READMUL"): return isval(nums[0]) and isdig(nums[1])
    if op in ("READ", "READ2", "READN"): return isval(nums[0])
    if op == "ORDER": return isdig(nums[0]) and isdig(nums[1])
    if op == "CALL": return isnum(nums[0]) and isnum(nums[1])
    if op == "WRITE": return isdig(nums[0])
    if op == "ANSWER": return isnum(nums[0]) or sym[nums[0]] >= 0
    return True                                                                                                     # SET, SETD, SETC, COPY
_KINDS = {}
def pkey(p): return (tuple(p["init"]), tuple(p["body"]), tuple(p["outro"]), p["stop"])
def kinds_batch(progs):
    """slot kinds before each instruction, for many programs in ONE execution (one example each)"""
    todo = [p for p in progs if pkey(p) not in _KINDS]
    if not todo: return
    if len(_KINDS) > 5000: _KINDS.clear()
    d = {}
    with torch.no_grad(): execute(todo, [rnum(int(rng.integers(1, a.max_digits + 1)))], [rnum(1)], probe=d)
    for i, p in enumerate(todo): _KINDS[pkey(p)] = d.get(i, {})
def kinds_of(p):
    """cached kinds of p; probes it alone if it was not batched"""
    if pkey(p) not in _KINDS: kinds_batch([p])
    return _KINDS[pkey(p)]
def propose(INS, kinds):
    if kinds is None: return INS[rng.integers(len(INS))]
    ok = [n for n in INS if applicable(n, kinds)]; return ok[rng.integers(len(ok))] if ok else INS[rng.integers(len(INS))]
PROPOSER = None
if PROP:
    from proposer import Proposer, parse_ins as _parse; PROPOSER = Proposer(HERE / PROP, dev); print(f"learned proposer loaded from {PROP} (held out: {torch.load(HERE / PROP, map_location='cpu')['holdout']}), temperature {PTEMP}", flush=True)
def mutate(p, fail=None, kd=None):
    INS = allowed_ins(); q = {k: (list(v) if isinstance(v, list) else v) for k, v in p.items()}; part = rng.choice(["body", "body", "init", "outro"]); seq = q[part]; r = rng.random()
    kinds = None
    if PROPOSER is not None and r < 0.6 and rng.random() < (0.5 if (TIER1 and GEN[0] <= TIER1) else 0.8):                 # in tier 1 half the proposals are typed-random: the new primitive is not in the proposer's vocabulary                                                       # the learned proposer picks the instruction from what the machine shows
        try:
            kd = kd if kd is not None else kinds_of(p); base = {"init": 0, "body": len(p["init"]), "outro": len(p["init"]) + len(p["body"])}[part]; j = int(rng.integers(len(seq) + (0 if r < 0.4 else 1))) if (seq or r >= 0.4) else 0; pos = base + j
            kinds = kd.get(pos) or (kd[max(k for k in kd if k <= pos)] if any(k <= pos for k in kd) else None)
            flat_ = list(p["init"]) + list(p["body"]) + list(p["outro"]); prev_op = _parse(flat_[pos - 1])[0] if pos > 0 and pos - 1 < len(flat_) else "none"
            if kinds is not None:
                name = PROPOSER.propose(set(INS), kinds, fail if fail is not None else [0.5] * 5, prev_op, part, j / max(len(seq), 1), T=PTEMP, applicable=applicable if TYPED else None)
                if name is not None:
                    if r < 0.4 and seq: seq[j] = name
                    elif len(seq) < 13: seq.insert(j, name)
                    if not q["body"]: q["body"] = [INS[rng.integers(len(INS))]]
                    return q
        except Exception: pass
    if TYPED and r < 0.6:
        try:
            kd = kd if kd is not None else kinds_of(p); base = {"init": 0, "body": len(p["init"]), "outro": len(p["init"]) + len(p["body"])}[part]; pos = base + (int(rng.integers(len(seq))) if seq else 0)
            kinds = kd.get(pos) or (kd[max(k for k in kd if k <= pos)] if any(k <= pos for k in kd) else None)
        except Exception: kinds = None
    if r < 0.4 and seq: seq[rng.integers(len(seq))] = propose(INS, kinds)
    elif r < 0.6 and len(seq) < 13: seq.insert(int(rng.integers(len(seq) + 1)), propose(INS, kinds))
    elif r < 0.75 and len(seq) > (1 if part == "body" else 0): seq.pop(int(rng.integers(len(seq))))
    elif r < 0.9 and len(seq) > 1: i, j = rng.integers(len(seq), size=2); seq[i], seq[j] = seq[j], seq[i]
    else: q["stop"] = int(rng.integers(K))
    if not q["body"]: q["body"] = [INS[rng.integers(len(INS))]]
    return q
def show(p): return f"init [{', '.join(p['init'])}]  body [{', '.join(p['body'])}]  outro [{', '.join(p['outro'])}]  stop {p['stop']}"
seed_prog = None
if SEEDP:
    src_ = json.load(open(HERE / SEEDP)) if SEEDP.endswith(".json") else None; seed_prog = (src_[list(src_)[0]]["program"] if src_ and "program" in src_[list(src_)[0]] else src_[list(src_)[0]]) if src_ else PROGRAMS[SEEDP]
    seed_prog = dict(seed_prog); seed_prog.setdefault("kind", "number"); print(f"seeded with {SEEDP}: {show(seed_prog)}", flush=True)
print(f"target {TARGET}: input-output pairs only, {'seeded' if seed_prog else 'blank slate'}, {K} slots, {len(instruction_set())} instructions, callable {sorted(PROGRAMS)}, any-slot credit {ANY}, typed proposals {TYPED}, learned proposer {bool(PROP)}, worked-example intermediates {WORKED}, tier-1 (calls+plumbing only) for {TIER1} generations", flush=True)
pop = [(mutate(seed_prog) if rng.random() < 0.8 else dict(seed_prog)) if seed_prog else random_program() for _ in range(a.pop)]; best, best_f = pop[0], -1; found_p = None
with torch.no_grad():
    POOL = sample(24, a.max_digits, TARGET)                                                                            # a fixed pool of training examples: repeated operands make library calls lookups (acceptance draws fresh numbers)
    for gen in range(1, a.gens + 1):
        GEN[0] = gen
        if TIER1 and gen == TIER1 + 1: print(f"  tier 1 (calls + plumbing) exhausted at generation {gen}: the full instruction set is now allowed", flush=True)
        pick = rng.choice(24, a.examples, replace=False); xs, ys = [POOL[0][i] for i in pick], [POOL[1][i] for i in pick]; f, e = score_io(pop, xs, ys); order = np.argsort(-f); fails = list(FAILS)
        if f[order[0]] > best_f: best, best_f = pop[order[0]], float(f[order[0]])
        if True: print(f"  gen {gen}/{a.gens}  best {f[order[0]]:.3f} (exact {e[order[0]]:.3f})  mean {f.mean():.3f}  candidates {gen * a.pop}  elapsed {time.time()-t0:.0f}s  ETA {(time.time()-t0)/gen*(a.gens-gen):.0f}s  | best so far: {show(best)[:110]}", flush=True)
        if e[order[0]] >= 1.0:
            cands = [pop[i] for i in order if e[i] >= 1.0]; cands.sort(key=lambda p: len(p["init"]) + len(p["body"]) + len(p["outro"]))
            for cand in cands[:5]:
                xs2, ys2 = sample(48, a.max_digits + 3, TARGET); _, ec = score_io([cand], xs2, ys2); rx, ry = zip(*REG[TARGET]); _, er = score_io([cand], list(rx), list(ry))
                if ec[0] >= 1.0 and er[0] >= 1.0: found_p = cand; break
            if found_p is not None: print(f"  FOUND at generation {gen}: {show(found_p)}", flush=True); break
        elite = [pop[i] for i in order[:max(2, a.pop // 20)]]; new = list(elite); kinds_batch([pop[i] for i in order[:max(4, a.pop // 4)]])            # one probe for all parents of this generation
        while len(new) < a.pop:
            i, j = order[rng.integers(0, max(4, a.pop // 4), size=2)]; par = i if f[i] >= f[j] else j; kdp = _KINDS.get(pkey(pop[par])); child = mutate(pop[par], fails[par], kdp); new.append(mutate(child, fails[par], kdp) if rng.random() < 0.3 else child)
            if rng.random() < 0.05: new.append(random_program())                                                       # a trickle of fresh random programs
        pop = new[:a.pop]
    prog = found_p or best; TRUTH[TARGET] = ({"kind": "number"}, TASKS[TARGET]); v = verify(prog, TARGET, lengths=tuple(L for L in (1, 2, 3, 4, 6, 8, 12, 16, 24) if L <= NP - 1), n=60)
    print(f"  {TARGET} winner by digits (>{a.max_digits} never seen): " + " ".join(f"{L}d {e:.3f}" if L != "reg" else f"regression {e:.3f}" for L, e in v.items()), flush=True)
(HERE / "results/discover_io").mkdir(parents=True, exist_ok=True); save_json({TARGET: {"found": found_p is not None, "generation": gen, "program": prog, "verify": v, "seeded": SEEDP, "any_slot": ANY, "typed": TYPED, "proposer": PROP, "slots": K, "candidates": gen * a.pop}}, HERE / ("results/discover_io/" + TARGET + ("_typed" if TYPED else "") + ("_proposer" if PROP else "") + ("_worked" if WORKED else "") + ".json")); print(f"saved ({time.time()-t0:.0f}s)", flush=True)
