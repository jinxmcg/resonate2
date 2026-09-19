"""The places machine of Part 14 on the DIGIT CHAIN: a number is a chain of its digits under one repeated move (NEXT) — no
place rows, no fixed length; the ten digit rows and an END symbol are the alphabet. Everything else is places.py unchanged:
the same primitives, the same executor, the same found programs (results/places/found_programs.json). What changes is the
representation underneath four primitives: LOAD walks one digit (END -> a 0 and the place stays), SHIFT appends a zero,
"anything left?" peeks along the chain to END, and a number is built/read by the one move. The value code is the digit and
carry phases only (places 0 and 1 of the Part-6 code: the programs never used more). The digit-order light is re-fitted on
the chain's digit rows (45 facts, seconds). Reported: the five found programs, unchanged, by digit length up to --np."""
import argparse, time, json, numpy as np, torch
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--path", default="results/digit_chain_k20/model.pt"); ap.add_argument("--np", type=int, default=24, help="longest number handled: a cycle bound, not a row count"); ap.add_argument("--n", type=int, default=100, help="cases per length in verification"); ap.add_argument("--value", default="results/grow_1e9_moves/value.pt"); ap.add_argument("--order", default="results/compare/digit_order.pt")
ap.add_argument("--slots", type=int, default=6); ap.add_argument("--pop", type=int, default=200); ap.add_argument("--gens", type=int, default=150); ap.add_argument("--examples", type=int, default=24); ap.add_argument("--max-digits", type=int, default=3)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results/places"); ap.add_argument("--only-check", action="store_true"); ap.add_argument("--resume-programs", default=""); ap.add_argument("--only", default="")
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time(); K = a.slots
pk = torch.load(HERE / a.path, map_location=dev, weights_only=False); pcfg = pk["config"]; NP = a.np; MAXN = 10 ** NP
END = 10; NEXT = 0; NR = 1; LMAX_TRAIN = int(pcfg.get("lmax", 0))
pmodel = ResonatE(11, 2, k=pcfg["k"], block=True, block_size=pcfg["block"]).to(dev); pmodel.load_state_dict(pk["model"]); pmodel.eval(); pmodel.requires_grad_(False)
Er = cnorm(pmodel.E).detach(); M = Er.shape[1]
def phop(z, r): return pmodel.hop(z, torch.full((z.shape[0],), r, device=dev, dtype=torch.long))
def plight(z): return torch.real(z @ Er.conj().t())
def digs(ns): return torch.tensor([[int(c) for c in f"{int(n):0{NP}d}"] for n in ns], device=dev)
def path_chain(D):
    """the digit chain: NEXT(END), then NEXT(acc + digit) for each digit, most significant first (leading zeros skipped)"""
    B = D.shape[0]; nz = D != 0; first = torch.where(nz.any(1), nz.float().argmax(1), torch.full((B,), NP - 1, device=dev)); mask = torch.arange(NP, device=dev)[None] >= first[:, None]
    acc = phop(Er[END][None].expand(B, -1), NEXT)
    for j in range(NP): step = phop(cnorm(acc + Er[D[:, j]]), NEXT); acc = torch.where(mask[:, j][:, None], step, acc)
    return acc
def walk_all(z):
    """(B, NP) digits, units first, by one reverse hop and one light per step; END reads as 0 and freezes the walk"""
    B = z.shape[0]; out = torch.zeros(B, NP, dtype=torch.long, device=dev); rr = z; ended = torch.zeros(B, dtype=torch.bool, device=dev)
    for k in range(NP):
        nxt = phop(rr, NEXT + NR); rr = torch.where(ended[:, None], rr, nxt); d = plight(rr).argmax(1); is_end = d == END; out[:, k] = torch.where(ended | is_end, torch.zeros_like(d), d); ended = ended | is_end
    return out
def to_int(D): return [int(sum(int(row[k]) * 10 ** k for k in range(NP))) for row in D.cpu().numpy()]
vk = torch.load(HERE / a.value, map_location=dev, weights_only=False); phi = vk["phi"]; mv = phi.shape[-1]; p_ = phi.clone(); p_[0] = 0; V = torch.polar(torch.ones_like(p_), p_); ONE = torch.ones(mv, dtype=torch.complex64, device=dev)
def vnum(n): return V[n % 10, 0] * (V[n // 10, 1] if n >= 10 else ONE)
LINE = torch.stack([vnum(s) for s in range(20)]); LINE2 = torch.stack([vnum(s) for s in range(100)]); NLINE = torch.stack([(V[1, 1].conj() if s == -10 else V[-s, 0].conj() if s < 0 else V[s, 0]) for s in range(-10, 10)])
MULKEY = torch.stack([V[d, 0] * V[e, 1] for d in range(10) for e in range(10)]); MULVAL = torch.tensor([d * e for d in range(10) for e in range(10)], device=dev)
# the digit-order light (Part 8): one template fitted to the 45 facts d < d' on THIS memory's digit rows
Qp = torch.nn.Parameter(torch.randn(M, 2, device=dev) * 0.1); qopt = torch.optim.Adam([Qp], lr=3e-2); pairs_ = [(d, e) for d in range(10) for e in range(d + 1, 10)]
for _ in range(600):
    s_ = torch.real(Er[:10] @ torch.view_as_complex(Qp).conj()); qloss = torch.nn.functional.softplus(-(s_[[e for d, e in pairs_]] - s_[[d for d, e in pairs_]]) * 10).mean(); qopt.zero_grad(); qloss.backward(); qopt.step()
SDIG = torch.real(Er[:10] @ torch.view_as_complex(Qp).conj()).detach(); print(f"digit order on the chain's rows: {int(sum((SDIG[e] > SDIG[d]).item() for d, e in pairs_))}/45 facts", flush=True); LT, GT, EQ = 10, 11, 12; EVEN, ODD = 13, 14
# the parity light (given, like the digit-order facts): one template fitted to the ten facts 'd is even / odd' on this memory's digit rows
Pp = torch.nn.Parameter(torch.randn(M, 2, device=dev) * 0.1); popt = torch.optim.Adam([Pp], lr=3e-2); PSIGN = torch.tensor([1.0 if d % 2 == 0 else -1.0 for d in range(10)], device=dev)
for _ in range(600): sp_ = torch.real(Er[:10] @ torch.view_as_complex(Pp).conj()); ploss = torch.nn.functional.softplus(-sp_ * PSIGN * 10).mean(); popt.zero_grad(); ploss.backward(); popt.step()
SPAR = torch.real(Er[:10] @ torch.view_as_complex(Pp).conj()).detach(); print(f"parity light on the digit rows: {int(((SPAR > 0) == (PSIGN > 0)).sum())}/10 facts", flush=True)
CALL_MEMO = {}                                                                                     # (program, x, y) -> its result
class Library(dict):
    """the found programs, callable by name; any (re)definition forgets the remembered call results"""
    def __setitem__(self, k, v): CALL_MEMO.clear(); super().__setitem__(k, v)
    def update(self, *a_, **k_): CALL_MEMO.clear(); super().update(*a_, **k_)
    def clear(self): CALL_MEMO.clear(); super().clear()
PROGRAMS = Library()                                                                               # found programs, callable by later ones
_INS_CACHE = {}
def instruction_set():
    pairs = [(s, t) for s in range(K) for t in range(K) if s != t]; triples = [(s, t, u) for s, t in pairs for u in range(K) if u != s and u != t]
    ins = [f"LOAD {s}>{t}" for s, t in pairs] + [f"NUM {s}>{t}" for s, t in pairs] + [f"BIND {s},{t}" for s, t in pairs] + [f"UNBIND {s},{t}" for s, t in pairs] + [f"SHIFT {s}" for s in range(K)] \
        + [f"READ {s}" for s in range(K)] + [f"READN {s}" for s in range(K)] + [f"READ2 {s}" for s in range(K)] + [f"READMUL {s},{t}" for s, t in pairs] + [f"ORDER {s},{t}>{u}" for s, t, u in triples] + [f"PARITY {s}>{t}" for s, t in pairs] \
        + [f"SETD {s}" for s in range(K)] + [f"SETC {s}" for s in range(K)] + [f"SET {s}<{d}" for s in range(K) for d in range(10)] + [f"SET {s}<EQ" for s in range(K)] + [f"COPY {s}>{t}" for s, t in pairs] \
        + [f"WRITE {s}" for s in range(K)] + [f"ANSWER {s}" for s in range(K)]
    for name in PROGRAMS: ins += [f"CALL {name} {s},{t}>{u}" for s, t in pairs for u in range(K)]                 # a call may write its result back into an operand slot (accumulators)
    return ins


class Places:
    def __init__(self, B):
        self.B = B; self.path = torch.zeros(B, K, M, dtype=torch.complex64, device=dev); self.ptr = torch.zeros(B, K, dtype=torch.long, device=dev); self.walkable = torch.zeros(B, K, dtype=torch.bool, device=dev)
        self.val = torch.ones(B, K, mv, dtype=torch.complex64, device=dev); self.sym = torch.full((B, K), -1, dtype=torch.long, device=dev)
        self.seen = torch.zeros(B, 256, dtype=torch.bool, device=dev); self.calls_seen = [set() for _ in range(B)]; self.tape = torch.zeros(B, NP, dtype=torch.long, device=dev); self.wp = torch.zeros(B, dtype=torch.long, device=dev); self.answer = torch.full((B,), -1, dtype=torch.long, device=dev); self.ans_slot = torch.full((B,), -1, dtype=torch.long, device=dev); self.last = torch.zeros(B, dtype=torch.long, device=dev)
    def put_number(self, idx, s, ns):
        self.path[idx, s] = path_chain(digs(ns)); self.ptr[idx, s] = 0; self.walkable[idx, s] = True; self.sym[idx, s] = -1; self.val[idx, s] = ONE
    def put_digit(self, mask, s, d):
        self.val[mask, s] = V[d[mask], 0]; self.sym[mask, s] = d[mask]; self.walkable[mask, s] = False
    def put_digit_rows(self, idx, s, d):
        """rows idx, per-row slot s (tensor), digits d (tensor)"""
        self.val[idx, s] = V[d, 0]; self.sym[idx, s] = d; self.walkable[idx, s] = False
    def number_of(self, idx, s):
        """the number a place stands for: the remaining digits of a walkable place, or the digit tag — computed for all rows at once"""
        rem = walk_all(self.path[idx, s]).cpu().numpy(); wk = self.walkable[idx, s].cpu().numpy(); sym = self.sym[idx, s].cpu().numpy()
        vals = [int(sum(int(row[k]) * 10 ** k for k in range(NP))) for row in rem]
        return [v if w else (max(int(z), 0) if z < 10 else 0) for v, w, z in zip(vals, wk, sym)]
    def result(self):
        """a number program's result: the tape if it wrote digits, else the place it ANSWERed (a walkable slot)"""
        tape_n = to_int(walk_all(path_chain(self.tape.flip(1)))); out = []
        for i in range(self.B):
            if int(self.wp[i]) > 0 or int(self.ans_slot[i]) < 0: out.append(tape_n[i])
            else: out.append(self.number_of(torch.tensor([i], device=dev), int(self.ans_slot[i]))[0])
        return out
    def left(self, s, rows=None):
        """1 if nothing nonzero is left along the chain in slot s (peek to END), for the given rows"""
        rows = torch.arange(self.B, device=dev) if rows is None else rows; n = len(rows); out = torch.ones(n, device=dev)
        if n == 0: return out
        _ = None; rr = self.path[rows, s]; ended = torch.zeros(n, dtype=torch.bool, device=dev)
        for k in range(NP):
            nxt = phop(rr, NEXT + NR); d = plight(nxt).argmax(1); is_end = d == END; nz = ~ended & ~is_end & (d != 0); out = torch.where(nz, torch.zeros_like(out), out); ended = ended | is_end; rr = torch.where(ended[:, None], rr, nxt)
            if bool(ended.all()): break
        return torch.where(self.walkable[rows, s], out, torch.ones_like(out))


def execute(progs, xs, ys, depth=0, max_cycles=NP + 1, probe=None, track=False):
    P, E = len(progs), len(xs); B = P * E; pl = Places(B); ar = torch.arange(B, device=dev)
    pl.put_number(ar, 0, [x for _ in progs for x in xs]); pl.put_number(ar, 1, [y for _ in progs for y in ys])
    global _INS_CACHE
    key_ = tuple(sorted(PROGRAMS))
    if _INS_CACHE.get("key") != key_:
        INS_ = instruction_set(); I_ = {n: i for i, n in enumerate(INS_)}
        def parse(name):
            op = name.split()[0]; rest = name[len(op):].strip(); sub = None
            if op == "CALL": sub, rest = rest.split(" ", 1)
            nums = [int(x) for x in __import__("re").findall(r"\d+", rest.replace("<EQ", ""))]; return op, nums, sub, name
        PARSED_ = [parse(n) for n in INS_]; OPC = sorted({p_[0] for p_ in PARSED_}); OPID_ = {o: i for i, o in enumerate(OPC)}
        _INS_CACHE = {"key": key_, "INS": INS_, "I": I_, "PARSED": PARSED_, "OPCODES": OPC, "T_OP": torch.tensor([OPID_[p_[0]] for p_ in PARSED_], device=dev), "T_ARG": torch.tensor([(p_[1] + [0, 0, 0])[:3] for p_ in PARSED_], device=dev), "T_SETEQ": torch.tensor([p_[3].endswith("<EQ") for p_ in PARSED_], device=dev)}
    INS, I, PARSED, OPCODES, T_OP, T_ARG, T_SETEQ = (_INS_CACHE[k] for k in ("INS", "I", "PARSED", "OPCODES", "T_OP", "T_ARG", "T_SETEQ"))
    maxlen = max(len(p["init"]) for p in progs) + max_cycles * max(len(p["body"]) for p in progs) + max(len(p["outro"]) for p in progs) + 1
    code = torch.full((B, maxlen), -1, dtype=torch.long, device=dev); cstart = torch.zeros(B, maxlen, dtype=torch.bool, device=dev); outro_at = torch.full((B,), -1, dtype=torch.long, device=dev)
    stop_slot = torch.tensor([p["stop"] for p in progs], device=dev).repeat_interleave(E)
    posmaps = []
    for pi, p in enumerate(progs):
        seq = [I[x] for x in p["init"]]; starts = []; pm = list(range(len(p["init"])))
        for c in range(max_cycles): starts.append(len(seq)); seq += [I[x] for x in p["body"]]; pm += [len(p["init"]) + i for i in range(len(p["body"]))]
        pm += [len(p["init"]) + len(p["body"]) + i for i in range(len(p["outro"]))]; posmaps.append(pm)
        rows = slice(pi * E, (pi + 1) * E); code[rows, :len(seq)] = torch.tensor(seq, device=dev); cstart[rows, starts] = True
        code[rows, len(seq):len(seq) + len(p["outro"])] = torch.tensor([I[x] for x in p["outro"]] or [], device=dev, dtype=torch.long) if p["outro"] else code[rows, len(seq):len(seq)]
    # the outro of each program is stored after its full unrolled loop; when the loop ends early we jump there
    outro_code = [[I[x] for x in p["outro"]] for p in progs]; loop_done = torch.zeros(B, dtype=torch.bool, device=dev); outro_pc = torch.zeros(B, dtype=torch.long, device=dev); finished = torch.zeros(B, dtype=torch.bool, device=dev)
    ncyc = torch.zeros(B, dtype=torch.long, device=dev); walked = torch.zeros(B, 2, dtype=torch.bool, device=dev)
    def step(ins, active):
        """one batched operation per OPCODE present, slot arguments per row (a population of distinct programs costs no more than one program)"""
        calls = {}; ins_c = ins.clamp(min=0); opn = torch.where(active, T_OP[ins_c], torch.full_like(ins, -1)); A0, A1, A2 = T_ARG[ins_c, 0], T_ARG[ins_c, 1], T_ARG[ins_c, 2]; ar_ = torch.arange(len(ins), device=dev)
        for oc in torch.unique(opn[active]).tolist():
            op = OPCODES[oc]; mk = opn == oc; idx = mk.nonzero().flatten(); a0, a1, a2 = A0[idx], A1[idx], A2[idx]
            if op == "LOAD":
                wk = pl.walkable[idx, a0]; idx, a0, a1 = idx[wk], a0[wk], a1[wk]                                   # only a slot that holds a number can be walked (a digit tag is not a chain)
                if len(idx) == 0: continue
                rr = phop(pl.path[idx, a0], NEXT + NR); L = plight(rr); is_end = L.argmax(1) == END; d = torch.where(is_end, torch.zeros_like(is_end, dtype=torch.long), L[:, :10].argmax(1))
                adv = idx[~is_end]; pl.path[adv, a0[~is_end]] = rr[~is_end]; pl.ptr[idx, a0] = pl.ptr[idx, a0] + 1; pl.put_digit_rows(idx, a1, d)                       # at END: a 0, and the place stays
                w0 = idx[a0 == 0]; w1 = idx[a0 == 1]; walked[w0, 0] = True; walked[w1, 1] = True
            elif op == "NUM":                                                                                        # a digit becomes a one-digit NUMBER (a chain of one digit): the bridge from value code to place
                ok = (pl.sym[idx, a0] >= 0) & (pl.sym[idx, a0] <= 9) & ~pl.walkable[idx, a0]; rows_ = idx[ok]; tt = a1[ok]
                if len(rows_): pl.path[rows_, tt] = path_chain(digs(pl.sym[rows_, a0[ok]].tolist())); pl.ptr[rows_, tt] = 0; pl.walkable[rows_, tt] = True; pl.sym[rows_, tt] = -1; pl.val[rows_, tt] = ONE
            elif op == "BIND": pl.val[idx, a0] = pl.val[idx, a0] * pl.val[idx, a1]; pl.sym[idx, a0] = -1
            elif op == "UNBIND": pl.val[idx, a0] = pl.val[idx, a0] * pl.val[idx, a1].conj(); pl.sym[idx, a0] = -1
            elif op == "COPY":
                for attr in ("path", "ptr", "walkable", "val", "sym"): t_ = getattr(pl, attr); t_[idx, a1] = t_[idx, a0]
            elif op == "READMUL": key = pl.val[idx, a0] * V[pl.sym[idx, a1].clamp(min=0, max=9), 1]; j = torch.real((key[:, None, :] * MULKEY[None].conj()).mean(-1)).argmax(1); pl.last[idx] = MULVAL[j]
            elif op == "ORDER":
                ds, dt = SDIG[pl.sym[idx, a0].clamp(min=0, max=9)], SDIG[pl.sym[idx, a1].clamp(min=0, max=9)]; lt = ds < dt - 1e-4; gt = ds > dt + 1e-4
                pl.sym[idx[lt], a2[lt]] = LT; pl.sym[idx[gt], a2[gt]] = GT; pl.walkable[idx[lt | gt], a2[lt | gt]] = False
            elif op == "PARITY":                                                                                     # the units digit's region: even or odd, by the parity light
                d = pl.sym[idx, a0]; ok = (d >= 0) & (d <= 9) & ~pl.walkable[idx, a0]; r_ = idx[ok]; ev = SPAR[d[ok]] > 0
                pl.sym[r_, a1[ok]] = torch.where(ev, torch.full_like(d[ok], EVEN), torch.full_like(d[ok], ODD)); pl.walkable[r_, a1[ok]] = False
            elif op == "CALL":
                for iid in torch.unique(ins[idx]).tolist(): _, nums, sub, _ = PARSED[iid]; calls.setdefault(sub, []).append((active & (ins == iid), *nums))
            elif op == "SHIFT": pl.path[idx, a0] = phop(cnorm(pl.path[idx, a0] + Er[0][None]), NEXT)                                                              # x10 = append a zero
            elif op in ("READ", "READ2", "READN"):
                LN, off = {"READ": (LINE, 0), "READ2": (LINE2, 0), "READN": (NLINE, -10)}[op]; pl.last[idx] = torch.real((pl.val[idx, a0][:, None, :] * LN[None].conj()).mean(-1)).argmax(1) + off
            elif op == "SETD": pl.put_digit_rows(idx, a0, pl.last[idx] % 10)
            elif op == "SETC": pl.put_digit_rows(idx, a0, torch.where(pl.last[idx] < 0, torch.ones_like(pl.last[idx]), pl.last[idx] // 10))
            elif op == "SET":
                eq = T_SETEQ[ins_c[idx]]; e_ = idx[eq]; pl.sym[e_, a0[eq]] = EQ; pl.walkable[e_, a0[eq]] = False
                n_ = ~eq; pl.put_digit_rows(idx[n_], a0[n_], a1[n_])
            elif op == "WRITE": d = pl.sym[idx, a0].clamp(min=0, max=9); pl.tape[idx, pl.wp[idx].clamp(max=NP - 1)] = d; pl.wp[idx] = pl.wp[idx] + 1
            elif op == "ANSWER": pl.answer[idx] = pl.sym[idx, a0]; pl.ans_slot[idx] = a0
        # all calls of the same sub-program at this step run as ONE nested batch, whatever slots they use
        for sub, items in calls.items():
            if depth >= 3 or sub not in PROGRAMS: continue
            idx_all, xs_, ys_, us = [], [], [], []
            for mk, s, tt, u in items:
                idx = mk.nonzero().flatten(); filled = lambda q: pl.walkable[idx, q] | ((pl.sym[idx, q] >= 0) & (pl.sym[idx, q] <= 9)); idx = idx[filled(s) & filled(tt)]   # a call needs two filled operands (a number or a digit); an empty slot is not 0
                if len(idx) == 0: continue
                idx_all.append(idx); xs_ += pl.number_of(idx, s); ys_ += pl.number_of(idx, tt); us += [u] * len(idx)
            if not idx_all: continue
            idx_all = torch.cat(idx_all); kind = PROGRAMS[sub]["kind"]
            # a called program is an exact function of its operand values: remember what it returned (CALL_MEMO) and run only the calls not seen before
            todo = sorted({(x, y) for x, y in zip(xs_, ys_) if (sub, x, y) not in CALL_MEMO})
            if todo:
                sub_pl = execute([PROGRAMS[sub]], [x for x, y in todo], [y for x, y in todo], depth + 1); vals = [min(v, MAXN - 1) for v in sub_pl.result()] if kind == "number" else sub_pl.answer.tolist()
                for (x, y), v in zip(todo, vals): CALL_MEMO[(sub, x, y)] = v
            res = [CALL_MEMO[(sub, x, y)] for x, y in zip(xs_, ys_)]
            for r_, x, y in zip(idx_all.tolist(), xs_, ys_): pl.calls_seen[r_].add((sub, x, y))                         # which library calls each row actually made (the worked example may name them)
            if kind == "number":
                for u in set(us): sel = torch.tensor([i for i, uu in enumerate(us) if uu == u], device=dev); pl.put_number(idx_all[sel], u, [res[i] for i in sel.tolist()])
            else:
                for u in set(us): sel = torch.tensor([i for i, uu in enumerate(us) if uu == u], device=dev); pl.sym[idx_all[sel], u] = torch.tensor([res[i] for i in sel.tolist()], device=dev); pl.walkable[idx_all[sel], u] = False
    for t in range(maxlen):
        ins = code[:, t]; active = ~loop_done & (ins >= 0)
        if cstart[:, t].any():
            cs = cstart[:, t] & active; rows_ = cs.nonzero().flatten(); l0 = torch.ones(B, device=dev); l1 = torch.ones(B, device=dev); l0[rows_] = pl.left(0, rows_); l1[rows_] = pl.left(1, rows_); s0 = pl.sym[ar, stop_slot]
            ok0 = ~walked[:, 0] | (l0 > 0.5); ok1 = ~walked[:, 1] | (l1 > 0.5)
            fin = cs & (ncyc > 0) & ok0 & ok1 & (s0 == 0) | (cs & (ncyc >= NP)); loop_done = loop_done | fin; active = active & ~fin; ncyc = ncyc + (cs & ~fin).long()
        if probe is not None:                                                                                        # what each program's first row holds before this instruction (probe: program index -> position -> kinds)
            act = active.nonzero().flatten().tolist(); wk = pl.walkable.tolist(); sy = pl.sym.tolist()
            for r_ in act:
                if r_ % E == 0:
                    pi_ = r_ // E; pm = posmaps[pi_]
                    if t < len(pm) and pm[t] not in probe.setdefault(pi_, {}): probe[pi_][pm[t]] = (wk[r_], sy[r_])
        if active.any(): step(ins, active)
        if track:                                                                                                    # which small values have LIT anywhere so far: the last READ, digit slots, value-code slots against the 0..99 line
            pl.seen[ar, pl.last.clamp(min=0, max=255)] |= pl.last >= 0
            dg = pl.sym.clamp(min=0, max=9); pl.seen.scatter_(1, dg, ((pl.sym >= 0) & (pl.sym <= 9)) | pl.seen.gather(1, dg))
            vc = torch.real((pl.val[:, :, None, :] * LINE2[None, None].conj()).mean(-1)); best = vc.argmax(-1); ok = (vc.max(-1).values > 0.95) & ~pl.walkable
            pl.seen.scatter_(1, best.clamp(max=255), ok | pl.seen.gather(1, best.clamp(max=255)))
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


def rnum(L): return int("".join(map(str, rng.integers(0, 10, L)))) if L > 0 else 0
def sample(n, max_digits, task):
    xs, ys = [], []
    for _ in range(n):
        L = int(rng.integers(1, max_digits + 1)); s = int(str(int(rng.integers(1, 10))) + "".join(map(str, rng.integers(0, 10, L - 1))))
        if task == "add": L2 = int(rng.integers(1, max_digits + 1)); x = rnum(L); y = rnum(L2); xs.append(min(x, MAXN - 1 - y)); ys.append(y)
        elif task == "sub": L2 = int(rng.integers(1, L + 1)); y = min(s, rnum(L2)); xs.append(s); ys.append(y)
        elif task == "cmp": u = int(rng.integers(0, L)); d = rnum(u + 1) * int(rng.choice([-1, 1])); xs.append(s); ys.append(int(min(max(s + d, 0), 10 ** L - 1)))
        elif task == "muld": xs.append(rnum(min(L, NP - 1))); ys.append(int(rng.integers(0, 10)))
        else: La = max(1, L // 2); xs.append(rnum(La)); ys.append(rnum(L - La + 1))
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



# ------------------------------------------------------------------ the five found programs of Part 14, unchanged, on the digit chain
found = json.load(open(HERE / a.resume_programs)) if a.resume_programs else json.load(open(HERE / "results/places/found_programs.json")); PROGRAMS.update(found); res = {"k": pcfg["k"], "M": M, "lmax_train": LMAX_TRAIN}
with torch.no_grad():
    for name in ("add", "sub", "cmp", "muld", "mul"):
        v = verify(found[name], name, lengths=range(1, NP + 1), n=a.n); res[name] = v
        print(f"found {name:4s} on the digit chain (k={pcfg['k']}, train lengths 1–{LMAX_TRAIN}), unchanged: " + " ".join(f"{L}d {e:.2f}" if L != "reg" else f"regression {e:.2f}" for L, e in v.items()), flush=True)
save_json(res, HERE / a.out / "digits.json"); print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)
