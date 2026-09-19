"""Program by observation: a worked example is PLAYED on the machine and recorded as a trace; the loop is then FOUND from
the trace by the Part-14 search (slices of the trace as candidate init/body/outro, selected by exactness on numbers
longer than the example) — no rule says where the loop is.
The teacher's example '123 = 0 + 3 + 2 + 1 = 6' (units first, a zero to start) is a sequence of events: a value 0
appears; the digit 3 appears; add(0, 3) happens; the digit 2 appears; add(3, 2); the digit 1; add(5, 1); the answer 6.
An observer stands at the machine: for each event it tries the moves that could make it happen — one instruction, or
a plumbing move followed by one — over the alphabet the example names (calls to 'add') plus plumbing (LOAD, NUM, COPY,
SET, SHIFT, ANSWER), all candidates as ONE batched execution, and keeps the shortest that realises the event (ties: the
move used for the same kind of event before, so repetitions look alike). The recorded traces are the worked examples
of the search of Part 14: a population of slices of them, fitness = the machine's answers on fresh short numbers,
acceptance = exact on numbers longer than any shown and on the regression cases. Cost: minutes."""
import sys, json, time, numpy as np, torch
ARGS = sys.argv[1:]; VALUED = ("--target", "--examples-shown", "--name"); FLAGS = ()
sys.argv = [sys.argv[0]] + [x for i, x in enumerate(ARGS) if x not in VALUED and (i == 0 or ARGS[i - 1] not in VALUED)]
def opt(name, default=None): return ARGS[ARGS.index(name) + 1] if name in ARGS else default
TARGET = opt("--target", "sumd"); SHOWN = [int(v) for v in opt("--examples-shown", "123,90,5").split(",")]; NAME = opt("--name", {"sumd": "digitsum", "double": "double", "triple": "triple", "iseven": "iseven"}.get(TARGET, TARGET))   # the word the worked example uses: the program's label
src = open(__file__.replace("observe.py", "digits.py")).read(); src = src[:src.index("# ------------------------------------------------------------------ the five found programs")]
g = {"__name__": "digits"}; exec(compile(src, "digits.py", "exec"), g); globals().update({k: v for k, v in g.items() if not k.startswith("__")})
found = json.load(open(HERE / "results/places/found_programs.json")); PROGRAMS.update(found)
def digitsum(x): return sum(int(c) for c in str(x))
TASKS = {"sumd": lambda x, y: digitsum(x), "double": lambda x, y: 2 * x, "triple": lambda x, y: 3 * x, "iseven": lambda x, y: EVEN if x % 2 == 0 else ODD}
KIND = "verdict" if TARGET == "iseven" else "number"
def events(x):
    """the worked example as events, in the machine's reading order (units first)"""
    ds = [int(c) for c in str(x)][::-1]
    if TARGET == "sumd":
        ev = [("value", 0)]; tot = 0
        for d in ds: ev += [("digit", d), ("call", "add", tot, d, tot + d)]; tot += d
        return ev + [("answer", tot)]
    if TARGET == "iseven": code = EVEN if x % 2 == 0 else ODD; return [("digit", ds[0]), ("parity", code), ("answer", code)]   # 'the last digit is 4; 4 is even; even'
    if TARGET == "double": return [("call", "add", x, x, 2 * x), ("answer", 2 * x)]
    if TARGET == "triple": return [("call", "add", x, x, 2 * x), ("call", "add", 2 * x, x, 3 * x), ("answer", 3 * x)]
NAMED = sorted({e[1] for x in SHOWN for e in events(x) if e[0] == "call"}); PLUMB = ("LOAD", "NUM", "COPY", "SET", "SHIFT", "ANSWER", "PARITY")
ALPHA = [n for n in instruction_set() if n.split()[0] in PLUMB or (n.split()[0] == "CALL" and n.split()[1] in NAMED)]
PREP = [n for n in ALPHA if n.split()[0] in ("NUM", "COPY", "SET", "LOAD", "SHIFT")]
print(f"observer: alphabet {len(ALPHA)} (named callables {NAMED} + plumbing), examples shown {SHOWN}", flush=True)
def realised(pl, ev, rows):
    """for each row: did the event happen in the machine's state?"""
    kind = ev[0]; B = len(rows); out = torch.zeros(B, dtype=torch.bool, device=dev)
    if kind == "digit": out = ((pl.sym[rows] == ev[1]) & ~pl.walkable[rows]).any(1)
    elif kind == "value":
        out = ((pl.sym[rows] == ev[1]) & ~pl.walkable[rows]).any(1)                                                # a digit tag with that value counts: it is an operand
        for s_ in range(K):
            wk = pl.walkable[rows, s_]
            if wk.any(): nums = torch.tensor(pl.number_of(rows[wk], s_), device=dev); out[wk.nonzero().flatten()] |= nums == ev[1]
    elif kind == "call": out = torch.tensor([((ev[1], ev[2], ev[3]) in pl.calls_seen[r]) or ((ev[1], ev[3], ev[2]) in pl.calls_seen[r]) for r in rows.tolist()], device=dev)
    elif kind == "parity": out = (pl.sym[rows] == ev[1]).any(1)
    elif kind == "answer" and KIND == "verdict": out = pl.answer[rows] == ev[1]
    elif kind == "answer": res = pl.result(); out = torch.tensor([res[i] == ev[1] and int(pl.ans_slot[r]) >= 0 for i, r in enumerate(rows.tolist())], device=dev)
    return out
def run_prefixes(prefixes, x):
    """execute each prefix (as an init-only program) on x; returns the Places state"""
    progs = [{"kind": KIND, "init": list(pf), "body": [], "outro": [], "stop": 0} for pf in prefixes]
    return execute(progs, [x], [47], track=True)                                                                   # a non-zero second operand: no free zero in a slot
def slots_of(pl, r=0):
    """what each slot holds, by the lights: ('num', value) for a walkable place, ('dig', d) for a digit tag, None if empty"""
    out = []
    for s_ in range(K):
        if bool(pl.walkable[r, s_]): out.append(("num", pl.number_of(torch.tensor([r], device=dev), s_)[0]))
        elif 0 <= int(pl.sym[r, s_]) <= 9: out.append(("dig", int(pl.sym[r, s_])))
        elif int(pl.sym[r, s_]) >= 10: out.append(("sym", int(pl.sym[r, s_])))
        else: out.append(None)
    return out
def derive(pl, ev, prefer):
    """the moves the event and the lights determine — a few candidates, not a search"""
    st = slots_of(pl); free = [s_ for s_ in range(K) if st[s_] is None]; holds = lambda v: [s_ for s_ in range(K) if st[s_] is not None and st[s_][1] == v]
    def first(kind_key, default): return prefer.get(kind_key, default)
    if ev[0] == "value":                                                                                            # a zero to start: a digit 0 in a free slot (a digit is a fine operand)
        return [[f"SET {s_}<{ev[1]}"] for s_ in free if ev[1] in (0, 1)] or [[]]
    if ev[0] == "digit":                                                                                            # the next digit comes from the slot that still holds the number being read
        nums = [s_ for s_ in range(K) if st[s_] is not None and st[s_][0] == "num" and s_ in (0,)] or [s_ for s_ in range(K) if st[s_] is not None and st[s_][0] == "num"]
        tgt = [t for t in free] + [t for t in range(K) if st[t] is not None and st[t][0] == "dig"]                  # into a free slot, or over a digit no longer needed
        return [[f"LOAD {n}>{t}"] for n in nums for t in tgt if n != t]
    if ev[0] == "call":
        name, a_, b_, r_ = ev[1], ev[2], ev[3], ev[4]; A, B = holds(a_), holds(b_)
        if a_ == b_: pairs = [(u, v) for u in A for v in B if u != v]
        else: pairs = [(u, v) for u in A for v in B if u != v] + [(v, u) for u in A for v in B if u != v]
        outs = lambda u, v: [u, v] + free                                                                            # the result replaces an operand (a running total) or takes a free slot
        return [[f"CALL {name} {u},{v}>{w}"] for u, v in pairs for w in outs(u, v)]
    if ev[0] == "parity": return [[f"PARITY {s_}>{u}"] for s_ in range(K) if st[s_] is not None and st[s_][0] == "dig" for u in free]
    if ev[0] == "answer": return [[f"ANSWER {s_}"] for s_ in holds(ev[1])]
    return []
def observe(x, prefer, beam=8):
    """play the example x: a BEAM of partial traces; at each event the moves are DERIVED from the event and the lights (which
    slots hold the named operands), verified by executing; entries whose choices make the next event impossible die there"""
    beams = [[]]; t0 = time.time()
    for ev in events(x):
        survivors = []
        with torch.no_grad():
            for tr in beams:
                pl0 = run_prefixes([tr], x); cands = derive(pl0, ev, prefer)
                if not cands: continue
                pl = run_prefixes([tr + c for c in cands], x); ok = realised(pl, ev, torch.arange(len(cands), device=dev))
                survivors += [tr + cands[j] for j in ok.nonzero().flatten().tolist()]
        if not survivors: print(f"  event {ev}: no derived move realises it", flush=True); return None
        seen_ = set(); uniq = []
        for tr in sorted(survivors, key=lambda t: (len(t), 0 if (len(t) and tuple(t[-1:]) == prefer.get(ev[0])) else 1)):
            if tuple(tr) not in seen_: seen_.add(tuple(tr)); uniq.append(tr)
        beams = uniq[:beam]; prefer[ev[0]] = tuple(beams[0][-1:]); print(f"  event {ev[:2] if ev[0] != 'call' else ev[1:4]}: {len(survivors)} derived moves realise it, beam keeps {len(beams)}, first {beams[0][-1:]}   [{time.time()-t0:.0f}s]", flush=True)
    return beams[0]
def show(p): return f"init [{', '.join(p['init'])}]  body [{', '.join(p['body'])}]  outro [{', '.join(p['outro'])}]  stop {p['stop']}"
TRUTH[TARGET] = ({"kind": KIND}, TASKS[TARGET]); REGRESSION[TARGET] = {"iseven": [(0, 3), (1, 0), (10, 5), (11, 2), (999999999999, 1), (1000000000000, 4), (2468, 0), (13579, 7)], "sumd": [(0, 7), (9, 0), (99, 3), (999999999, 0), (1000000000, 5), (123456789, 0), (909090909090, 1), (999999999999, 0), (9999999999999, 2), (8888888888888, 0)], "double": [(0, 3), (5, 0), (99, 99), (999999, 1)], "triple": [(0, 9), (4, 0), (99, 1), (333333, 5)]}[TARGET]
_sample = sample
def sample(n, max_digits, task):
    if task != TARGET: return _sample(n, max_digits, task)
    return [rnum(int(rng.integers(1, max_digits + 1))) for _ in range(n)], [rnum(int(rng.integers(1, max_digits + 1))) for _ in range(n)]
g["sample"] = sample
t0 = time.time(); prefer = {}; traces = {}
for x in SHOWN:
    print(f"\n== playing {TARGET}({x}) = {TASKS[TARGET](x, 0)}", flush=True); tr = observe(x, prefer); traces[x] = tr
    if tr: print(f"  trace ({len(tr)} moves): {tr}", flush=True)
res = {"target": TARGET, "shown": SHOWN, "alphabet": len(ALPHA), "traces": {str(k): v for k, v in traces.items()}}
demos = [tr for tr in traces.values() if tr]
# ------------------------------------------------------------------ the loop is found, not folded: Part 14's search over slices of the observed traces
def from_demo():
    tr = demos[rng.integers(len(demos))]; blen = int(rng.integers(1, 9)); start = int(rng.integers(0, max(1, len(tr) - blen + 1))); body = tr[start:start + blen]
    init = tr[:start][-int(rng.integers(0, 6)):] if start > 0 and rng.random() < 0.8 else []; outro = tr[start + blen:][-int(rng.integers(0, 4)):] if rng.random() < 0.8 else []
    return {"kind": KIND, "init": list(init), "body": list(body) or [tr[0]], "outro": list(outro), "stop": int(rng.integers(0, K))}
def mutate(p):
    q = {k: (list(v) if isinstance(v, list) else v) for k, v in p.items()}; part = rng.choice(["body", "body", "init", "outro"]); seq = q[part]; r = rng.random(); pool = [m for tr in demos for m in tr]
    if r < 0.4 and seq: seq[rng.integers(len(seq))] = pool[rng.integers(len(pool))]
    elif r < 0.6 and len(seq) < 12: seq.insert(int(rng.integers(len(seq) + 1)), pool[rng.integers(len(pool))])
    elif r < 0.75 and len(seq) > (1 if part == "body" else 0): seq.pop(int(rng.integers(len(seq))))
    elif r < 0.9 and len(seq) > 1: i, j = rng.integers(len(seq), size=2); seq[i], seq[j] = seq[j], seq[i]
    else: q["stop"] = int(rng.integers(K))
    if not q["body"]: q["body"] = [pool[rng.integers(len(pool))]]
    return q
if demos:
    with torch.no_grad():
        pop = [from_demo() for _ in range(150)]; found_p = None; best, best_f = pop[0], -1
        for gen in range(1, 61):
            xs, ys = sample(12, 3, TARGET); f, e = score(pop, TARGET, xs, ys); order = np.argsort(-f)
            if f[order[0]] > best_f: best, best_f = pop[order[0]], float(f[order[0]])
            print(f"  search gen {gen}/60  best {f[order[0]]:.3f} (exact {e[order[0]]:.3f})  mean {f.mean():.3f}  candidates {gen * 150}  [{time.time()-t0:.0f}s]", flush=True)
            if e[order[0]] >= 1.0:
                cands = [pop[i] for i in order if e[i] >= 1.0]; cands.sort(key=lambda p: len(p["init"]) + len(p["body"]) + len(p["outro"]))
                for cand in cands[:8]:
                    xs2, ys2 = sample(48, 6, TARGET); _, ec = score([cand], TARGET, xs2, ys2); rx, ry = zip(*REGRESSION[TARGET]); _, er = score([cand], TARGET, list(rx), list(ry))
                    if ec[0] >= 1.0 and er[0] >= 1.0: found_p = cand; break
                if found_p: break
            elite = [pop[i] for i in order[:8]]; new = list(elite)
            while len(new) < 150:
                i, j = order[rng.integers(0, 40, size=2)]; child = mutate(pop[i] if f[i] >= f[j] else pop[j]); new.append(mutate(child) if rng.random() < 0.3 else child)
            pop = new
        prog = found_p or best; v = verify(prog, TARGET, lengths=(1, 2, 3, 4, 6, 8, 12, 13), n=60)
        print(("  FOUND at generation %d (%d candidates): " % (gen, gen * 150) if found_p else "  not found; best: ") + show(prog), flush=True)
        print(f"  {TARGET} by digits (>3 never shown): " + " ".join(f"{L}d {e:.3f}" if L != "reg" else f"regression {e:.3f}" for L, e in v.items()), flush=True)
        res.update({"found": found_p is not None, "generation": gen, "candidates": gen * 150, "program": prog, "verify": v, "name": NAME})
        if found_p is not None:                                                                                     # accepted -> labelled in the library, automatically: from here on it is a call, not a search
            lib = json.load(open(HERE / "results/places/found_programs.json")); lib[NAME] = {**prog, "kind": KIND, "arity": 1, "named_by": "worked example", "shown": SHOWN}
            json.dump(lib, open(HERE / "results/places/found_programs.json", "w"), indent=1); print(f"  labelled '{NAME}' in the library ({len(lib)} programs); callable by name from now on", flush=True)
(HERE / "results/observe").mkdir(parents=True, exist_ok=True); save_json(res, HERE / f"results/observe/{TARGET}.json"); print(f"saved ({time.time()-t0:.0f}s)", flush=True)
