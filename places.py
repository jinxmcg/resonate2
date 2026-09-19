"""The places machine: every slot holds a PLACE; every instruction moves, lights or writes places.
A place = a state in the memory: a path state (walkable: a number) and/or a value state (bindable: a digit) and an optional
tag. The query's operands arrive as places in slots 0 and 1; a number result is built on a tape and becomes a place.
moves  : LOAD s>t     walk the place in s one step back (units first); the digit's place goes to t; s keeps the remainder
         BIND / UNBIND s,t   value binding (digit arithmetic, exact);   SHIFT s   move the place in s one digit up (x10)
         CALL p s,t>u  run a found program p on the places in s and t; its result place goes to u  (programs call programs)
lights : READ s (0..19), READN s (-10..9), READ2 s (0..99), READMUL s,t (the times table: 100 facts keyed by the pair)
         ORDER s,t>u  the digit-order lights of s and t; writes LT/GT into u only if they differ (EQ leaves u alone)
decode : SETD t (digit of the last light), SETC t (its carry / borrow)
writes : SET t<0, SET t<1, SET t<EQ, COPY s>t, WRITE s (append the digit place in s to the tape), ANSWER s
Program = init + body (once per place) + outro + stop slot. The loop ends when every operand slot that was walked has
nothing left and the stop slot reads 0 (or after NP cycles). 'Anything left?' is a peek over the remaining places.
Ground truth programs are checked to 1.000 first; then each operation is searched from 3 worked examples in curriculum
order (add, sub, cmp, muld, mul), with the earlier ones available as CALLs; every earlier one is re-verified after."""
import argparse, time, json, numpy as np, torch
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--path", default="results/grow_1e9_moves/path.pt"); ap.add_argument("--value", default="results/grow_1e9_moves/value.pt"); ap.add_argument("--order", default="results/compare/digit_order.pt")
ap.add_argument("--slots", type=int, default=6); ap.add_argument("--pop", type=int, default=200); ap.add_argument("--gens", type=int, default=150); ap.add_argument("--examples", type=int, default=24); ap.add_argument("--max-digits", type=int, default=3)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results/places"); ap.add_argument("--only-check", action="store_true"); ap.add_argument("--resume-programs", default=""); ap.add_argument("--only", default="")
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time(); K = a.slots
pk = torch.load(HERE / a.path, map_location=dev, weights_only=False); pcfg = pk["config"]; NP = int(pcfg.get("np", 9)); MAXN = 10 ** NP                                    # the number of places comes from the table
NODES = [str(d) for d in range(10)] + [str(10 ** k) for k in range(1, NP)]; nid = {n: i for i, n in enumerate(NODES)}; PLUS, TIMES = 0, 1; NR = 2
NR = pk["model"]["H"].shape[0] // 2                      # the table may hold more rows/relations than the number part (one memory)
pmodel = ResonatE(pk["model"]["E"].shape[0], 2 * NR, k=pcfg["k"], block=True, block_size=pcfg["block"]).to(dev); pmodel.load_state_dict(pk["model"]); pmodel.eval(); pmodel.requires_grad_(False)
ANCH = torch.tensor([nid["0"]] + [nid[str(10 ** k)] for k in range(1, NP)], device=dev); Er = cnorm(pmodel.E).detach(); M = Er.shape[1]
def phop(z, r): return pmodel.hop(z, torch.full((z.shape[0],), r, device=dev, dtype=torch.long))
def plight(z): return torch.real(z @ Er.conj().t())
def digs(ns): return torch.tensor([[int(c) for c in f"{int(n):0{NP}d}"] for n in ns], device=dev)
def path_chain(D):
    B = D.shape[0]; acc = None
    for j in range(NP):
        k = NP - 1 - j; d = D[:, j]; term = Er[d] if k == 0 else phop(cnorm(Er[d] + Er[ANCH[k]][None].expand(B, -1)), TIMES)
        acc = phop(term, PLUS) if acc is None else phop(cnorm(acc + term), PLUS)
    return acc
def walk_all(z):
    B = z.shape[0]; out = torch.zeros(B, NP, dtype=torch.long, device=dev); rr = z
    for k in range(NP):
        rr = phop(rr, PLUS + NR)
        if k == 0: out[:, 0] = plight(rr).argmax(1)
        else: L = plight(phop(rr, TIMES + NR)); L[:, 10:] = -1e9; out[:, k] = L.argmax(1)
    return out
def to_int(D): return [int(sum(int(row[k]) * 10 ** k for k in range(NP))) for row in D.cpu().numpy()]
vk = torch.load(HERE / a.value, map_location=dev, weights_only=False); phi = vk["phi"]; mv = phi.shape[-1]; p_ = phi.clone(); p_[0] = 0; V = torch.polar(torch.ones_like(p_), p_); ONE = torch.ones(mv, dtype=torch.complex64, device=dev)
def vnum(n): return V[n % 10, 0] * (V[n // 10, 1] if n >= 10 else ONE)
LINE = torch.stack([vnum(s) for s in range(20)]); LINE2 = torch.stack([vnum(s) for s in range(100)]); NLINE = torch.stack([(V[1, 1].conj() if s == -10 else V[-s, 0].conj() if s < 0 else V[s, 0]) for s in range(-10, 10)])
MULKEY = torch.stack([V[d, 0] * V[e, 1] for d in range(10) for e in range(10)]); MULVAL = torch.tensor([d * e for d in range(10) for e in range(10)], device=dev)
Q = torch.load(HERE / a.order, map_location=dev, weights_only=False)["Q"]; SDIG = torch.real(Er[:10] @ Q.conj()); LT, GT, EQ = 10, 11, 12
PROGRAMS = {}                                                                                      # found programs, callable by later ones
def instruction_set():
    pairs = [(s, t) for s in range(K) for t in range(K) if s != t]; triples = [(s, t, u) for s, t in pairs for u in range(K) if u != s and u != t]
    ins = [f"LOAD {s}>{t}" for s, t in pairs] + [f"BIND {s},{t}" for s, t in pairs] + [f"UNBIND {s},{t}" for s, t in pairs] + [f"SHIFT {s}" for s in range(K)] \
        + [f"READ {s}" for s in range(K)] + [f"READN {s}" for s in range(K)] + [f"READ2 {s}" for s in range(K)] + [f"READMUL {s},{t}" for s, t in pairs] + [f"ORDER {s},{t}>{u}" for s, t, u in triples] \
        + [f"SETD {s}" for s in range(K)] + [f"SETC {s}" for s in range(K)] + [f"SET {s}<0" for s in range(K)] + [f"SET {s}<1" for s in range(K)] + [f"SET {s}<EQ" for s in range(K)] + [f"COPY {s}>{t}" for s, t in pairs] \
        + [f"WRITE {s}" for s in range(K)] + [f"ANSWER {s}" for s in range(K)]
    for name in PROGRAMS: ins += [f"CALL {name} {s},{t}>{u}" for s, t in pairs for u in range(K)]                 # a call may write its result back into an operand slot (accumulators)
    return ins


class Places:
    def __init__(self, B):
        self.B = B; self.path = torch.zeros(B, K, M, dtype=torch.complex64, device=dev); self.ptr = torch.zeros(B, K, dtype=torch.long, device=dev); self.walkable = torch.zeros(B, K, dtype=torch.bool, device=dev)
        self.val = torch.ones(B, K, mv, dtype=torch.complex64, device=dev); self.sym = torch.full((B, K), -1, dtype=torch.long, device=dev)
        self.tape = torch.zeros(B, NP, dtype=torch.long, device=dev); self.wp = torch.zeros(B, dtype=torch.long, device=dev); self.answer = torch.full((B,), -1, dtype=torch.long, device=dev); self.ans_slot = torch.full((B,), -1, dtype=torch.long, device=dev); self.last = torch.zeros(B, dtype=torch.long, device=dev)
    def put_number(self, idx, s, ns):
        self.path[idx, s] = path_chain(digs(ns)); self.ptr[idx, s] = 0; self.walkable[idx, s] = True; self.sym[idx, s] = -1; self.val[idx, s] = ONE
    def put_digit(self, mask, s, d):
        self.val[mask, s] = V[d[mask], 0]; self.sym[mask, s] = d[mask]; self.walkable[mask, s] = False
    def number_of(self, idx, s):
        """the number a place stands for: the remaining digits of a walkable place, or the digit tag"""
        rem = walk_all(self.path[idx, s]); out = []
        for i, row in enumerate(rem.cpu().numpy()):
            j = int(idx[i]); out.append(int(sum(int(row[k]) * 10 ** k for k in range(NP))) if bool(self.walkable[j, s]) else max(int(self.sym[j, s]), 0) if self.sym[j, s] < 10 else 0)
        return out
    def result(self):
        """a number program's result: the tape if it wrote digits, else the place it ANSWERed (a walkable slot)"""
        tape_n = to_int(walk_all(path_chain(self.tape.flip(1)))); out = []
        for i in range(self.B):
            if int(self.wp[i]) > 0 or int(self.ans_slot[i]) < 0: out.append(tape_n[i])
            else: out.append(self.number_of(torch.tensor([i], device=dev), int(self.ans_slot[i]))[0])
        return out
    def left(self, s):
        out = torch.ones(self.B, device=dev); rr = self.path[:, s]; ptr = self.ptr[:, s]
        for k in range(NP):
            rr = phop(rr, PLUS + NR); nz = (plight(phop(rr, TIMES + NR))[:, :10].argmax(1) != 0) & (ptr + k < NP); out = torch.where(nz, torch.zeros_like(out), out)
        return torch.where(self.walkable[:, s], out, torch.ones_like(out))


def execute(progs, xs, ys, depth=0, max_cycles=NP + 1):
    P, E = len(progs), len(xs); B = P * E; pl = Places(B); ar = torch.arange(B, device=dev)
    pl.put_number(ar, 0, [x for _ in progs for x in xs]); pl.put_number(ar, 1, [y for _ in progs for y in ys])
    INS = instruction_set(); I = {n: i for i, n in enumerate(INS)}
    maxlen = max(len(p["init"]) for p in progs) + max_cycles * max(len(p["body"]) for p in progs) + max(len(p["outro"]) for p in progs) + 1
    code = torch.full((B, maxlen), -1, dtype=torch.long, device=dev); cstart = torch.zeros(B, maxlen, dtype=torch.bool, device=dev); outro_at = torch.full((B,), -1, dtype=torch.long, device=dev)
    stop_slot = torch.tensor([p["stop"] for p in progs], device=dev).repeat_interleave(E)
    for pi, p in enumerate(progs):
        seq = [I[x] for x in p["init"]]; starts = []
        for c in range(max_cycles): starts.append(len(seq)); seq += [I[x] for x in p["body"]]
        rows = slice(pi * E, (pi + 1) * E); code[rows, :len(seq)] = torch.tensor(seq, device=dev); cstart[rows, starts] = True
        code[rows, len(seq):len(seq) + len(p["outro"])] = torch.tensor([I[x] for x in p["outro"]] or [], device=dev, dtype=torch.long) if p["outro"] else code[rows, len(seq):len(seq)]
    # the outro of each program is stored after its full unrolled loop; when the loop ends early we jump there
    outro_code = [[I[x] for x in p["outro"]] for p in progs]; loop_done = torch.zeros(B, dtype=torch.bool, device=dev); outro_pc = torch.zeros(B, dtype=torch.long, device=dev); finished = torch.zeros(B, dtype=torch.bool, device=dev)
    ncyc = torch.zeros(B, dtype=torch.long, device=dev); walked = torch.zeros(B, 2, dtype=torch.bool, device=dev)
    def parse(name):
        op = name.split()[0]; rest = name[len(op):].strip(); sub = None
        if op == "CALL": sub, rest = rest.split(" ", 1)
        nums = [int(x) for x in __import__("re").findall(r"\d+", rest.replace("<EQ", ""))]; return op, nums, sub, name
    PARSED = [parse(n) for n in INS]
    def step(ins, active):
        present = torch.unique(ins[active]).tolist(); calls = {}
        for iid in present:
            op, nums, sub, name = PARSED[iid]; mk = active & (ins == iid)
            if op == "LOAD":
                s, tt = nums; rr = phop(pl.path[:, s], PLUS + NR); L = torch.where((pl.ptr[:, s] == 0)[:, None], plight(rr), plight(phop(rr, TIMES + NR)))[:, :10]; d = L.argmax(1)
                pl.path[mk, s] = rr[mk]; pl.ptr[mk, s] = pl.ptr[mk, s] + 1; pl.put_digit(mk, tt, d)
                if s < 2: walked[mk, s] = True
            elif op == "BIND": s, tt = nums; pl.val[mk, s] = pl.val[mk, s] * pl.val[mk, tt]; pl.sym[mk, s] = -1
            elif op == "UNBIND": s, tt = nums; pl.val[mk, s] = pl.val[mk, s] * pl.val[mk, tt].conj(); pl.sym[mk, s] = -1
            elif op == "COPY":
                s, tt = nums
                for attr in ("path", "ptr", "walkable", "val", "sym"): getattr(pl, attr)[mk, tt] = getattr(pl, attr)[mk, s]
            elif op == "READMUL": s, tt = nums; key = pl.val[:, s] * V[pl.sym[:, tt].clamp(min=0, max=9), 1]; idx = torch.real((key[:, None, :] * MULKEY[None].conj()).mean(-1)).argmax(1); pl.last = torch.where(mk, MULVAL[idx], pl.last)
            elif op == "ORDER":
                s, tt, u = nums; ds, dt = SDIG[pl.sym[:, s].clamp(min=0, max=9)], SDIG[pl.sym[:, tt].clamp(min=0, max=9)]; lt = mk & (ds < dt - 1e-4); gt = mk & (ds > dt + 1e-4)
                pl.sym[lt, u] = LT; pl.sym[gt, u] = GT; pl.walkable[lt | gt, u] = False
            elif op == "CALL": calls.setdefault(sub, []).append((mk, *nums))
            elif op == "SHIFT": s, = nums; idx = mk.nonzero().flatten(); pl.put_number(idx, s, [min(v * 10, MAXN - 1) for v in pl.number_of(idx, s)])
            elif op in ("READ", "READ2", "READN"):
                s, = nums; LN, off = {"READ": (LINE, 0), "READ2": (LINE2, 0), "READN": (NLINE, -10)}[op]; pl.last = torch.where(mk, torch.real((pl.val[:, s][:, None, :] * LN[None].conj()).mean(-1)).argmax(1) + off, pl.last)
            elif op == "SETD": s, = nums; pl.put_digit(mk, s, pl.last % 10)
            elif op == "SETC": s, = nums; pl.put_digit(mk, s, torch.where(pl.last < 0, torch.ones_like(pl.last), pl.last // 10))
            elif op == "SET":
                if name.endswith("<EQ"): s, = nums; pl.sym[mk, s] = EQ; pl.walkable[mk, s] = False
                else: s, cval = nums; pl.put_digit(mk, s, torch.full_like(pl.last, cval))
            elif op == "WRITE": s, = nums; d = pl.sym[:, s].clamp(min=0, max=9); pl.tape[mk, pl.wp[mk].clamp(max=NP - 1)] = d[mk]; pl.wp = pl.wp + mk.long()
            elif op == "ANSWER": s, = nums; pl.answer = torch.where(mk, pl.sym[:, s], pl.answer); pl.ans_slot = torch.where(mk, torch.full_like(pl.ans_slot, s), pl.ans_slot)
        # all calls of the same sub-program at this step run as ONE nested batch, whatever slots they use
        for sub, items in calls.items():
            if depth >= 3 or sub not in PROGRAMS: continue
            idx_all, xs_, ys_, us = [], [], [], []
            for mk, s, tt, u in items:
                idx = mk.nonzero().flatten()
                if len(idx) == 0: continue
                idx_all.append(idx); xs_ += pl.number_of(idx, s); ys_ += pl.number_of(idx, tt); us += [u] * len(idx)
            if not idx_all: continue
            idx_all = torch.cat(idx_all); sub_pl = execute([PROGRAMS[sub]], xs_, ys_, depth + 1)
            if PROGRAMS[sub]["kind"] == "number":
                res = [min(v, MAXN - 1) for v in sub_pl.result()]
                for u in set(us): sel = torch.tensor([i for i, uu in enumerate(us) if uu == u], device=dev); pl.put_number(idx_all[sel], u, [res[i] for i in sel.tolist()])
            else:
                for u in set(us): sel = torch.tensor([i for i, uu in enumerate(us) if uu == u], device=dev); pl.sym[idx_all[sel], u] = sub_pl.answer[sel]; pl.walkable[idx_all[sel], u] = False
    for t in range(maxlen):
        ins = code[:, t]; active = ~loop_done & (ins >= 0)
        if cstart[:, t].any():
            cs = cstart[:, t] & active; l0, l1 = pl.left(0), pl.left(1); s0 = pl.sym[ar, stop_slot]
            ok0 = ~walked[:, 0] | (l0 > 0.5); ok1 = ~walked[:, 1] | (l1 > 0.5)
            fin = cs & (ncyc > 0) & ok0 & ok1 & (s0 == 0) | (cs & (ncyc >= NP)); loop_done = loop_done | fin; active = active & ~fin; ncyc = ncyc + (cs & ~fin).long()
        if active.any(): step(ins, active)
        if not (~loop_done & (code[:, t + 1:] >= 0).any(1)).any() and t > 0: pass
        if loop_done.all() or not ((code[:, t + 1:] >= 0).any()): break
    loop_done[:] = True
    # outro: run each program's outro instructions once
    max_o = max(len(o) for o in outro_code)
    for j in range(max_o):
        ins = torch.tensor([o[j] if j < len(o) else -1 for o in outro_code], device=dev).repeat_interleave(E); step(ins, ins >= 0)
    return pl


# ------------------------------------------------------------------ ground truth (checked, then used only for demonstrations)
ADD = {"kind": "number", "init": ["SET 2<0"], "body": ["LOAD 0>3", "LOAD 1>4", "BIND 3,4", "BIND 3,2", "READ 3", "SETD 5", "SETC 2", "WRITE 5"], "outro": [], "stop": 2}
SUB = {"kind": "number", "init": ["SET 2<0"], "body": ["LOAD 0>3", "LOAD 1>4", "UNBIND 3,4", "UNBIND 3,2", "READN 3", "SETD 5", "SETC 2", "WRITE 5"], "outro": [], "stop": 2}
CMP = {"kind": "verdict", "init": ["SET 2<0", "SET 5<EQ"], "body": ["LOAD 0>3", "LOAD 1>4", "ORDER 3,4>5"], "outro": ["ANSWER 5"], "stop": 2}
MULD = {"kind": "number", "init": ["SET 2<0", "LOAD 1>4"], "body": ["LOAD 0>3", "READMUL 3,4", "SETD 5", "SETC 3", "BIND 5,2", "READ 5", "SETD 5", "SETC 2", "BIND 2,3", "READ 2", "SETD 2", "WRITE 5"], "outro": [], "stop": 2}
MUL = {"kind": "number", "init": ["SET 2<0", "COPY 0>4"], "body": ["LOAD 1>3", "CALL muld 4,3>5", "CALL add 2,5>2", "SHIFT 4"], "outro": ["ANSWER 2"], "stop": 3}
TRUTH = {"add": (ADD, lambda x, y: x + y), "sub": (SUB, lambda x, y: x - y), "cmp": (CMP, lambda x, y: EQ if x == y else (LT if x < y else GT)), "muld": (MULD, lambda x, y: x * y), "mul": (MUL, lambda x, y: x * y)}


def sample(n, max_digits, task):
    xs, ys = [], []
    for _ in range(n):
        L = int(rng.integers(1, max_digits + 1)); s = int(rng.integers(10 ** (L - 1), 10 ** L))
        # operands of INDEPENDENT lengths (the earlier sampler drew the sum first and split it: unequal lengths with a carry out of
        # the longer operand's top digit occurred 0.09 % of the time and a found add with exactly that hole passed acceptance)
        if task == "add": L2 = int(rng.integers(1, max_digits + 1)); x = int(rng.integers(0, 10 ** L)); y = int(rng.integers(0, 10 ** L2)); xs.append(min(x, MAXN - 1 - y)); ys.append(y)
        elif task == "sub": L2 = int(rng.integers(1, L + 1)); y = int(rng.integers(0, min(s, 10 ** L2 - 1) + 1)); xs.append(s); ys.append(y)
        elif task == "cmp": y = int(np.clip(s + rng.integers(-10 ** int(rng.integers(0, L)), 10 ** int(rng.integers(0, L)) + 1), 0, 10 ** L - 1)); xs.append(s); ys.append(int(y))
        elif task == "muld": xs.append(int(rng.integers(0, 10 ** min(L, NP - 1)))); ys.append(int(rng.integers(0, 10)))
        else: La = max(1, L // 2); xs.append(int(rng.integers(0, 10 ** La))); ys.append(int(rng.integers(0, 10 ** (L - La + 1))))
    return xs, ys


def score(progs, task, xs, ys):
    """fitness per program: per-place digit credit + exact bonus (numbers) or exact (verdicts); returns (fitness, exact)"""
    pl = execute(progs, xs, ys); P, E = len(progs), len(xs); f = TRUTH[task][1]; targets = [f(x, y) for x, y in zip(xs, ys)]
    if TRUTH[task][0]["kind"] == "verdict":
        ex = (pl.answer.reshape(P, E) == torch.tensor(targets, device=dev)[None]).float(); return (2 * ex.mean(1)).cpu().numpy(), ex.mean(1).cpu().numpy()
    T = digs([min(t, MAXN - 1) for t in targets]).flip(1).repeat(P, 1); npl = torch.tensor([max(len(str(t)), 1) for t in targets], device=dev).repeat(P); place = torch.arange(NP, device=dev)[None]
    got = digs(pl.result()).flip(1)
    credit = ((got == T) & (place < npl[:, None])).float().sum(1) / npl.float(); ex = (got == T).all(1).float()
    return (credit + ex).reshape(P, E).mean(1).cpu().numpy(), ex.reshape(P, E).mean(1).cpu().numpy()


REGRESSION = {"add": [(99, 1), (1, 99), (999, 1), (95, 9), (9, 95), (99999, 1), (95365, 9114), (9999, 9999), (5, 5), (90, 10), (999999, 999999), (123456, 7), (0, 0), (0, 7)],
              "sub": [(100, 1), (1000, 1), (104, 9), (100000, 1), (104479, 9114), (10, 10), (7, 0), (0, 0)],
              "cmp": [(99, 100), (100, 99), (7, 7), (1000, 999), (10, 9)], "muld": [(99, 9), (999, 9), (0, 5), (100, 0), (12345678, 9)], "mul": [(99, 99), (999, 9), (9, 999), (101, 11), (0, 12), (999, 1001), (4321, 999)]}
def verify(prog, task, lengths=range(1, NP + 1), n=200):
    """exact by length on fresh samples, plus a fixed regression set of the carry-out / unequal-length patterns (key 'reg')"""
    out = {}
    for L in lengths:
        xs, ys = sample(n, L, task); f, e = score([prog], task, xs, ys); out[L] = float(e[0])
    xs, ys = zip(*REGRESSION[task]); f, e = score([prog], task, list(xs), list(ys)); out["reg"] = float(e[0])
    return out


# ------------------------------------------------------------------ 1. the machine: ground-truth programs must be exact
with torch.no_grad():
    for name in ("add", "sub", "cmp", "muld", "mul"):
        prog = TRUTH[name][0]; v = verify(prog, name, lengths=(1, 3, 5, 7), n=100); print(f"ground truth {name:4s} on the places machine: exact by digits " + " ".join(f"{L}d {e:.3f}" if L != "reg" else f"regression {e:.3f}" for L, e in v.items()) + f"  ({time.time()-t0:.0f}s)", flush=True)
        PROGRAMS[name] = prog                                                                       # ground truth is registered only so that later ground-truth programs can CALL it during this check
    if a.only_check: raise SystemExit
    if a.only: pass
PROGRAMS.clear()
if a.resume_programs:
    for k_, v_ in json.load(open(HERE / a.resume_programs)).items(): PROGRAMS[k_] = v_
    print(f"resumed found programs: {list(PROGRAMS)}", flush=True)
# ------------------------------------------------------------------ 2. the curriculum: each operation from 3 worked examples, earlier ones callable
def demos(name, n=3):
    prog = TRUTH[name][0]; out = []
    for _ in range(n):
        xs, ys = sample(1, a.max_digits, name); cycles = max(len(str(max(xs[0], ys[0], TRUTH[name][1](xs[0], ys[0]) if name != "cmp" else 1))), 1)
        out.append((xs[0], ys[0], list(prog["init"]) + [i for _ in range(cycles) for i in prog["body"]] + list(prog["outro"])))
    return out
def from_demos(dm, kind):
    x, y, tr = dm[rng.integers(len(dm))]; blen = int(rng.integers(2, 13)); start = int(rng.integers(0, max(1, len(tr) - blen + 1))); body = tr[start:start + blen]
    init = tr[:start][-int(rng.integers(0, 3)):] if start > 0 and rng.random() < 0.7 else []; outro = [tr[-1]] if kind == "verdict" and rng.random() < 0.7 else []
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
results = {"machine_check": {}, "curriculum": {}}
with torch.no_grad():
    for name in ([a.only] if a.only else ("add", "sub", "cmp", "muld", "mul")):
        kind = TRUTH[name][0]["kind"]; dm = demos(name); print(f"\n== {name}: worked examples " + "; ".join(f"{x},{y} ({len(tr)} steps)" for x, y, tr in dm) + f"   callable: {list(PROGRAMS)}", flush=True)
        pop = [from_demos(dm, kind) for _ in range(a.pop)]; best, best_f = None, -1; found = None
        for g in range(1, a.gens + 1):
            xs, ys = sample(a.examples, a.max_digits, name); f, e = score(pop, name, xs, ys); order = np.argsort(-f)
            if f[order[0]] > best_f: best, best_f = pop[order[0]], float(f[order[0]])
            if g % 10 == 0 or g == 1: print(f"  gen {g}  best {f[order[0]]:.3f} (exact {e[order[0]]:.3f})  mean {f.mean():.3f}  elapsed {time.time()-t0:.0f}s", flush=True)
            if e[order[0]] >= 1.0:
                # accept only if exact on a held-out batch LONGER than anything shown (the teacher's test), preferring the shortest such candidate
                cands = [pop[i] for i in order if e[i] >= 1.0]; cands.sort(key=lambda p: len(p["init"]) + len(p["body"]) + len(p["outro"]))
                for cand in cands[:5]:
                    xs, ys = sample(64, a.max_digits + 2, name); fc, ec = score([cand], name, xs, ys); rx, ry = zip(*REGRESSION[name]); fr, er = score([cand], name, list(rx), list(ry))
                    if ec[0] >= 1.0 and er[0] >= 1.0: found = cand; break                                              # held-out longer cases AND the regression patterns
                if found is not None: print(f"  found at generation {g}: {show(found)}", flush=True); break
            elite = [pop[i] for i in order[:max(2, a.pop // 20)]]; new = list(elite)
            while len(new) < a.pop:
                i, j = order[rng.integers(0, max(4, a.pop // 4), size=2)]; child = mutate(pop[i] if f[i] >= f[j] else pop[j]); new.append(mutate(child) if rng.random() < 0.3 else child)
            pop = new
        prog = found or best; v = verify(prog, name, lengths=range(1, (NP if name in ("muld", "mul") else NP) + 1), n=150)
        print(f"  {name} winner on 1-9 digits (>{a.max_digits} never seen): " + " ".join(f"{L}d {e:.3f}" if L != "reg" else f"regression {e:.3f}" for L, e in v.items()), flush=True)
        results["curriculum"][name] = {"found": found is not None, "generation": g, "program": prog, "verify": v}
        PROGRAMS[name] = prog                                                                       # the found program becomes callable by the next operations
        # retention: every earlier operation re-verified with the current machine (programs are frozen, so this is a check of the machine, reported)
        ret = {m: verify(PROGRAMS[m], m, lengths=(2, 5, 8), n=60) for m in PROGRAMS if m != name}
        if ret: print("  retention: " + "  ".join(f"{m} " + "/".join(f"{e:.2f}" for e in v.values()) for m, v in ret.items()), flush=True)
        results["curriculum"][name]["retention"] = ret
save_json(results, HERE / a.out / "places.json"); print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)
