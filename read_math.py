"""The language layer on the arithmetic machine: sentences that CALL the found programs, and sentences that DEFINE new ones.
tokens : every token is a row: the ten digit words, the function words (add, and, from, by, to, it, itself, '.', '?', ...)
         are stored rows; a new word ('double') is a fresh row. A numeral is read digit by digit ('3 4 7'); the number is
         the place reached by walking its digits (Part 4's construction) — no number place sits in a sentence, and where
         one operand ends and the next begins is for the reader to find. A sentence is one place: a NEXT chain over its
         token places, read back by walking (one light per token).
machine: K slots holding tokens; LOAD (walk the sentence one token forward), ISW s,w>u (the token is the stored word w),
         ISDIG (it is a digit word), ISPROG (it names a program: a stored program word or a name defined in this
         episode), ISNEW (a fresh word that names nothing yet), CCOPY v,s>t / CWRITE v,s (copy / append the token to the
         current operand — a digit extends it, 'it'/'itself' make it the argument — only if v is YES; the flag is consumed),
         CNEWOP v (close the current operand), CSWAP v (swap the two operands), CALLW c (run the program named by the token
         in c on the operands: each operand's digits walked into a number place for the places machine), CDEF n,c (define the program named by the token in n as
         'run c on these operands', if n holds a new word), SET, COPY, IFYES. Programs = init + loop body + outro + stop.
         Calls run on the places machine of Part 14 (`places.py`, the found programs add/sub/cmp/muld/mul) — imported, not
         re-implemented. Defined programs are entries next to the found ones (their promotion to pages is Part 16).
search : the reader is found from worked examples (traces of a ground-truth reader on four sentences); the fitness is
         the machine's answer only. Accepted only if exact on longer numbers and more filler words than shown.
episodes: two sentences sharing tokens: two commands ('please add the number 347 and 12 .', 'subtract 12 from 347 .',
         'multiply 34 by 7 .', 'compare 12 and 7 ?'), or a definition + a use ('to double something , add it to itself .',
         'double 34324 .'); definitions may carry a constant ('to triple something , multiply it by 3 .'). A definition
         sentence must produce no answer; an unknown command word ('frobnicate 12 and 7 .') must produce NONE (abstain)."""
import argparse, time, json, sys, runpy, numpy as np, torch, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--path", default="results/onemem/path.pt"); ap.add_argument("--programs", default="results/places/found_programs.json"); ap.add_argument("--steps", type=int, default=8000)
ap.add_argument("--pop", type=int, default=200); ap.add_argument("--gens", type=int, default=200); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results/read_math"); ap.add_argument("--slots", type=int, default=10)
ap.add_argument("--check-only", action="store_true"); ap.add_argument("--ladder", action="store_true", help="find the command reader first, then the definition reader seeded from it"); ap.add_argument("--lk", type=int, default=20, help="width of the language memory (its own; k*k)"); ap.add_argument("--lm-resume", default="", help="load a trained language memory instead of training")
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time(); K = a.slots
# ------------------------------------------------------------------ the arithmetic machine (places.py) with the FOUND programs, imported as a module
src = open(HERE / "places.py").read(); src = src[:src.index("# ------------------------------------------------------------------ 1. the machine")]          # the machine's definitions, up to its own ground-truth check
sys.argv = ["places.py", "--path", a.path]; PLACES = {"__name__": "places", "__file__": str(HERE / "places.py")}; exec(compile(src, str(HERE / "places.py"), "exec"), PLACES)
PROGRAMS = PLACES["PROGRAMS"]; PROGRAMS.clear(); found = json.load(open(HERE / a.programs)); PROGRAMS.update(found)
ALIAS = {"add": "add", "subtract": "sub", "multiply": "mul", "compare": "cmp"}                                     # the sentence word for each program (a naming alias, nothing learned)
for w, p in ALIAS.items(): PROGRAMS[w] = found[p]
pexec, path_chain, digs, walk_all, to_int = PLACES["execute"], PLACES["path_chain"], PLACES["digs"], PLACES["walk_all"], PLACES["to_int"]; M = PLACES["M"]; NP = PLACES["NP"]; LT, GT, EQ = PLACES["LT"], PLACES["GT"], PLACES["EQ"]
print(f"places machine loaded: programs {list(PROGRAMS)}  M {M}  ({time.time()-t0:.0f}s)", flush=True)
MEMO = {}                                                                                                             # (program, x, y) -> the machine's answer; the machine is deterministic, so a call is run once
def run_calls(cmds, xs, ys):
    """execute program cmds[i] on (xs[i], ys[i]); distinct calls only, grouped by program, remembered; returns ints, verdict codes (LT/GT/EQ), or None"""
    out = [None] * len(cmds); todo = {}
    for i, c in enumerate(cmds):
        if c is not None and (c, xs[i], ys[i]) not in MEMO: todo.setdefault(c, set()).add((xs[i], ys[i]))
    for c, pairs in todo.items():
        pairs = sorted(pairs); pl = pexec([PROGRAMS[c]], [x for x, y in pairs], [y for x, y in pairs])
        vals = [int(v) for v in pl.answer.tolist()] if PROGRAMS[c]["kind"] == "verdict" else [int(v) for v in pl.result()]
        for (x, y), v in zip(pairs, vals): MEMO[(c, x, y)] = v
    for i, c in enumerate(cmds):
        if c is not None: out[i] = MEMO[(c, xs[i], ys[i])]
    return out
# ------------------------------------------------------------------ the language memory: stored words + NEXT, trained on random token sequences (random places and number places)
VOCAB = [str(d) for d in range(10)] + ["add", "subtract", "multiply", "compare", "and", "from", "by", "to", "than", "the", "number", "please", "it", "itself", "something", ",", ".", "?"]; V = len(VOCAB); vid = {w: i for i, w in enumerate(VOCAB)}; NF = 6
PROGWORDS = {"add", "subtract", "multiply", "compare"}; ARGWORDS = {"it", "itself"}
lm = ResonatE(V, 2, k=a.lk, block=True, block_size=16).to(dev); LM = lm.m; NEXT = 0
def E(): return cnorm(lm.E)
def hop(z, r): return lm.hop(z, torch.full((z.shape[0],), r, device=dev, dtype=torch.long))
def chain(W, mask):
    B, L, _ = W.shape; acc = None
    for j in range(L - 1, -1, -1):
        has = mask[:, j]; step = hop(W[:, j], NEXT) if acc is None else hop(cnorm(acc + W[:, j]), NEXT); acc = step if acc is None else torch.where(has[:, None], step, acc)
    return acc
def light(z, cand, cmask):
    sc = torch.real((z[:, None, :] * cand.conj()).sum(-1)); sc[~cmask] = -1e9; return sc
def random_tokens(B, Lmax=20, Lmin=2):
    fresh = cnorm(torch.randn(B, NF, LM, dtype=torch.complex64, device=dev)); cand = torch.cat([E()[None].expand(B, -1, -1), fresh], 1); cmask = torch.ones(B, V + NF, dtype=torch.bool, device=dev)
    L = torch.tensor(rng.integers(Lmin, Lmax + 1, B), device=dev); w = rng.integers(0, V + NF, (B, Lmax))
    runs = rng.random(B) < 0.15                                                                                      # runs of one token (a numeral like 4444444) never arise from random strings: a memory trained without them misreads them (Part 23b)
    for b in np.nonzero(runs)[0]: r0 = int(rng.integers(0, Lmax)); r1 = int(rng.integers(r0 + 2, Lmax + 1)) if r0 + 2 <= Lmax else Lmax; w[b, r0:r1] = rng.integers(0, 10)
    wid = torch.tensor(w, device=dev); mask = torch.arange(Lmax, device=dev)[None] < L[:, None]
    return cand, cmask, wid, mask
opt = torch.optim.Adam(lm.parameters(), lr=3e-3); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
if a.lm_resume: lm.load_state_dict(torch.load(HERE / a.lm_resume, map_location=dev)["lm"]); a.steps = 0; print(f"language memory loaded from {a.lm_resume}", flush=True)
for step in range(1, a.steps + 1):
    B = 256; cand, cmask, wid, mask = random_tokens(B); z = chain(cand[torch.arange(B, device=dev)[:, None], wid], mask); tau = lm.log_tau.exp(); loss = 0.0; n = 0; rr = z
    for j in range(wid.shape[1]):
        rr = hop(rr, NEXT + 1); has = mask[:, j]
        if has.any(): loss = loss + F.cross_entropy(light(rr, cand, cmask)[has] * tau, wid[has, j]); n += 1
    loss = loss / n; opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 500 == 0 or step == 1: print(f"language memory step {step}/{a.steps}  loss {loss.item():.4f}  lr {sched.get_last_lr()[0]:.1e}  tau {tau.item():.1f}  elapsed {time.time()-t0:.0f}s", flush=True)
lm.eval(); lm.requires_grad_(False); res = {}
if not a.lm_resume: (HERE / a.out).mkdir(parents=True, exist_ok=True); torch.save({"lm": lm.state_dict(), "lk": a.lk}, HERE / a.out / "lm.pt")
with torch.no_grad():
    by = {}
    for L in range(2, 23):
        cand, cmask, wid, mask = random_tokens(400, Lmax=L, Lmin=L); z = chain(cand[torch.arange(400, device=dev)[:, None], wid], mask); rr = z; ok = torch.ones(400, dtype=torch.bool, device=dev)
        for j in range(L): rr = hop(rr, NEXT + 1); ok &= light(rr, cand, cmask).argmax(1) == wid[:, j]
        by[L] = float(ok.float().mean())
    res["sentence_readback_by_length"] = by; print("sentences as one place (stored words incl. digits, fresh words), read back by walking, exact by length (train 2–20): " + " ".join(f"{L}:{v:.3f}" for L, v in by.items()), flush=True)
# ------------------------------------------------------------------ episodes
POOL = {d: sorted({int(rng.integers(10 ** (d - 1), 10 ** d)) if d > 1 else int(rng.integers(0, 10)) for _ in range(60)}) for d in range(1, NP + 1)}; USE_POOL = False    # the search draws numbers from a fixed pool (calls are memoised); acceptance and tests draw fresh numbers
def numeral(n_digits):
    if USE_POOL: return int(rng.choice(POOL[n_digits]))
    return int(rng.integers(10 ** (n_digits - 1), 10 ** n_digits)) if n_digits > 1 else int(rng.integers(0, 10))
def fillers(n): return [["please"], ["the", "number"]][:0] if n == 0 else None
class Episodes:
    """B episodes of two sentences over shared tokens. Token ids: 0..V-1 stored words, V.. sentence-specific (number places and fresh words).
    per episode: kinds[b] = ('cmd', 'cmd') or ('def', 'use'); truth[b] = [answer1, answer2] (None = no answer expected, 'NONE' = must abstain)"""
    def __init__(self, B, n_digits=(1, 3), n_fill=(0, 1), kinds=None, unknown=0.0, mul_digits=3):
        self.B = B; self.fresh = cnorm(torch.randn(B, NF, LM, dtype=torch.complex64, device=dev)); self.sents = [[], []]; self.truth = []; self.kind = []; self.intended = []
        for b in range(B):
            kind = kinds[b] if kinds is not None else rng.choice(["cmd", "def"]); self.kind.append(kind); nd = lambda: int(rng.integers(n_digits[0], n_digits[1] + 1)); nf = lambda: int(rng.integers(n_fill[0], n_fill[1] + 1))
            toks = []                                                                                                # sentence-specific tokens: fresh words
            def num(n): return [vid[c] for c in str(n)]                                                             # a numeral is its digit words
            def new(): toks.append("word"); return V + len(toks) - 1
            def fill(): return [vid[["please", "the"][rng.integers(2)]] for _ in range(nf())]                       # single-word fillers
            def command(unknown_word=False):
                op = rng.choice(["add", "subtract", "multiply", "compare"]); d1, d2 = nd(), nd()
                if op == "multiply": d1, d2 = min(d1, mul_digits), min(d2, mul_digits)
                x, y = numeral(d1), numeral(d2)
                if op == "subtract" and x < y: x, y = y, x
                cw = new() if unknown_word else vid[op]; nx, ny = num(x), num(y)
                if op == "add": s = fill() + [cw] + fill() + nx + [vid["and"]] + ny + [vid["."]]; ans = x + y
                elif op == "subtract": s = fill() + [cw] + fill() + ny + [vid["from"]] + nx + [vid["."]]; ans = x - y
                elif op == "multiply": s = fill() + [cw] + fill() + nx + [vid["by"]] + ny + [vid["."]]; ans = x * y
                else: s = fill() + [cw] + fill() + nx + [vid["and"]] + ny + [vid["?"]]; ans = LT if x < y else GT if x > y else EQ
                return s, ("NONE" if unknown_word else ans), (None if unknown_word else (op, x, y))
            if kind == "cmd":
                u1, u2 = rng.random() < unknown, rng.random() < unknown; s1, t1, i1 = command(u1); s2, t2, i2 = command(u2)
            else:
                name = new(); op = rng.choice(["add", "subtract", "multiply", "add_c"]); k_ = numeral(1) if rng.random() < 0.5 else numeral(2)
                if op == "add": body = [vid["add"], vid["it"], vid["to"], vid["itself"]]; f_ = lambda n: n + n; call = lambda n: ("add", n, n)
                elif op == "add_c": body = [vid["add"]] + num(k_) + [vid["to"], vid["it"]]; f_ = lambda n, k_=k_: n + k_; call = lambda n, k_=k_: ("add", k_, n)
                elif op == "subtract": body = [vid["subtract"]] + num(k_) + [vid["from"], vid["it"]]; f_ = lambda n, k_=k_: n - k_; call = lambda n, k_=k_: ("subtract", n, k_)
                else: k_ = numeral(1) + 1; body = [vid["multiply"], vid["it"], vid["by"]] + num(k_); f_ = lambda n, k_=k_: n * k_; call = lambda n, k_=k_: ("multiply", n, k_)
                s1 = [vid["to"], name, vid["something"], vid[","]] + body + [vid["."]]; t1 = None; i1 = None
                d = nd(); n = numeral(min(d, mul_digits) if op == "multiply" else d)
                if op == "subtract": n = max(n, k_)
                s2 = fill() + [name] + fill() + num(n) + [vid["."]]; t2 = f_(n); i2 = call(n)
            self.sents[0].append(s1); self.sents[1].append(s2); self.truth.append([t1, t2]); self.intended.append([i1, i2])
        # the truth is the MACHINE's answer to the intended call (the reader is what is tested; the found programs have their own holes, logged separately)
        flat = [c for pair in self.intended for c in pair]; ans = run_calls([c[0] if c else None for c in flat], [c[1] if c else 0 for c in flat], [c[2] if c else 0 for c in flat])
        for b in range(B):
            for j in range(2):
                if self.intended[b][j]: self.truth[b][j] = ans[2 * b + j]
        self.cand = torch.cat([E()[None].expand(B, -1, -1), self.fresh], 1); self.cmask = torch.ones(B, V + NF, dtype=torch.bool, device=dev); self.z = []
        for j in range(2):
            Lmax = max(len(s) for s in self.sents[j]); wid = torch.zeros(B, Lmax, dtype=torch.long, device=dev); mask = torch.zeros(B, Lmax, dtype=torch.bool, device=dev)
            for b, s in enumerate(self.sents[j]): wid[b, :len(s)] = torch.tensor(s, device=dev); mask[b, :len(s)] = True
            self.z.append(chain(self.cand[torch.arange(B, device=dev)[:, None], wid], mask))
    def text(self, b, j): return " ".join(VOCAB[t] if t < V else f"W{t - V}" for t in self.sents[j][b])
# ------------------------------------------------------------------ the reading machine
def instruction_set():
    S = K - 3                                                                                                          # slots K-3, K-2 = the operand list; K-1 unused by the search
    pairs = [(s, t) for s in range(S) for t in range(S) if s != t]
    return [f"LOAD {s}>{t}" for s, t in pairs] + [f"ISW {s},{SAFE.get(w, w)}>{u}" for s in range(S) for w in VOCAB for u in range(S) if u != s] + [f"{op} {s}>{u}" for op in ("ISDIG", "ISPROG", "ISNEW") for s in range(S) for u in range(S) if u != s] \
        + [f"CCOPY {v},{s}>{t}" for v in range(S) for s in range(S) for t in range(S) if len({v, s, t}) == 3] + [f"CWRITE {v},{s}" for v in range(S) for s in range(S) if v != s] + [f"CNEWOP {v}" for v in range(S)] + [f"CSWAP {v}" for v in range(S)] \
        + [f"CALLW {c}" for c in range(S)] + [f"CDEF {n},{c}" for n in range(S) for c in range(S) if n != c] + [f"SET {u}<NONE" for u in range(S)] + [f"SET {u}<YES" for u in range(S)] + [f"COPY {s}>{t}" for s, t in pairs]
SAFE = {",": "comma", ".": "dot", "?": "qmark", **{str(d): f"d{d}" for d in range(10)}}; UNSAFE = {v: k for k, v in SAFE.items()}
INS = instruction_set(); I = {n: i for i, n in enumerate(INS)}; PARSED = []
for name in INS:
    op, rest = name.split(" ", 1); parts = rest.replace(">", ",").replace("<", ",").split(","); nums = [int(x) for x in parts if x.strip().isdigit()]; sym = [UNSAFE.get(x, x) for x in parts if not x.strip().isdigit()]; PARSED.append((op, nums, sym[0] if sym else None))
OPCODES = ["LOAD", "ISW", "ISDIG", "ISPROG", "ISNEW", "CCOPY", "CWRITE", "CNEWOP", "CSWAP", "CALLW", "CDEF", "SET", "COPY"]; OPID = {o: i for i, o in enumerate(OPCODES)}
T_OP = torch.tensor([OPID[p[0]] for p in PARSED], device=dev); T_ARG = torch.tensor([(p[1] + [0, 0, 0, 0])[:4] for p in PARSED], device=dev)
T_SYM = torch.tensor([(vid[p[2]] if p[0] == "ISW" else (-1 if p[2] == "NONE" else -2) if p[0] == "SET" else 0) for p in PARSED], device=dev)      # the word an ISW tests / the value a SET writes
print(f"reading machine: {len(INS)} instructions, {K} slots", flush=True)
class State:
    """per (program x episode): defined names (token id) and their meaning; carried from sentence 1 to sentence 2"""
    def __init__(self, N): self.def_name = torch.full((N,), -1, dtype=torch.long, device=dev); self.def_cmd = [None] * N; self.def_ops = [None] * N
def execute(progs, ep, j, st, max_cycles=None):
    max_cycles = max_cycles or (max(len(x) for x in ep.sents[j]) + 1)                                                  # unroll the loop to the batch's longest sentence
    """P programs x B episodes on sentence j. Returns per row: the call made (cmd name or None), operands, and whether anything was answered"""
    P, B = len(progs), ep.B; N = P * B; bidx = torch.arange(N, device=dev) % B; ar = torch.arange(N, device=dev); bnp = bidx.cpu().numpy()
    wid = torch.full((N, K), -9, dtype=torch.long, device=dev); place = torch.zeros(N, K, LM, dtype=torch.complex64, device=dev); place[:, 0] = ep.z[j][bidx]; wid[:, 0] = -5
    cand, cmask = ep.cand[bidx], ep.cmask[bidx]
    ops = torch.full((N, 2, NP), -1, dtype=torch.long, device=dev); olen = torch.zeros(N, 2, dtype=torch.long, device=dev); oarg = torch.zeros(N, 2, dtype=torch.bool, device=dev); cur = torch.zeros(N, dtype=torch.long, device=dev)   # the operand list
    isprog_stored = torch.tensor([w in PROGWORDS for w in VOCAB] + [False] * NF, device=dev); isdig = torch.tensor([w.isdigit() for w in VOCAB] + [False] * NF, device=dev); isarg = torch.tensor([w in ARGWORDS for w in VOCAB] + [False] * NF, device=dev)
    def nonempty(rows): return (olen[rows, cur[rows]] > 0) | oarg[rows, cur[rows]]
    calls = [None] * N; call_ops = [None] * N; defined_now = torch.zeros(N, dtype=torch.bool, device=dev)
    maxlen = max(len(p["init"]) for p in progs) + max_cycles * max(max(len(p["body"]) for p in progs), 1) + max(len(p["outro"]) for p in progs) + 1
    code = torch.full((N, maxlen), -1, dtype=torch.long, device=dev); cstart = torch.zeros(N, maxlen, dtype=torch.bool, device=dev); ostart = torch.zeros(N, dtype=torch.long, device=dev); stop_slot = torch.tensor([p["stop"] for p in progs], device=dev).repeat_interleave(B)
    for pi, p in enumerate(progs):
        seq = [I[x] for x in p["init"]]; starts = []
        for c in range(max_cycles if p["body"] else 0): starts.append(len(seq)); seq += [I[x] for x in p["body"]]
        rows = slice(pi * B, (pi + 1) * B); ostart[rows] = len(seq); seq += [I[x] for x in p["outro"]]; code[rows, :len(seq)] = torch.tensor(seq, device=dev); cstart[rows, starts] = True
    loop_done = torch.zeros(N, dtype=torch.bool, device=dev); ncyc = torch.zeros(N, dtype=torch.long, device=dev); pc = torch.zeros(N, dtype=torch.long, device=dev)
    def operands(i):
        """the two operands of row i: 'ARG', an int (its digits walked into a number: the value of the place they reach), or None"""
        out = []
        for o in range(2):
            if oarg[i, o]: out.append("ARG")
            elif olen[i, o] > 0: out.append(int("".join(VOCAB[int(d)] for d in ops[i, o, :int(olen[i, o])].tolist())))
            else: out.append(None)
        return out
    for t_ in range(maxlen + 2):
        pcc = pc.clamp(max=maxlen - 1); ins = torch.where(pc < maxlen, code[ar, pcc], torch.full_like(pc, -1)); active = ins >= 0
        if not active.any(): break
        cs = cstart[ar, pcc] & active & ~loop_done
        if cs.any():
            sv = wid[ar, stop_slot]; term = sv == -2; fin = cs & ((ncyc > 0) & term | (ncyc >= max_cycles - 1))
            loop_done = loop_done | fin; pc = torch.where(fin, ostart, pc); ncyc = ncyc + (cs & ~fin).long(); pcc = pc.clamp(max=maxlen - 1); ins = torch.where(pc < maxlen, code[ar, pcc], torch.full_like(pc, -1)); active = ins >= 0
        opn = torch.where(active, T_OP[ins.clamp(min=0)], torch.full_like(ins, -1)); A = T_ARG[ins.clamp(min=0)]; SY = T_SYM[ins.clamp(min=0)]
        for oc in torch.unique(opn[active]).tolist():                                                                  # one batched step per OPCODE present, slot arguments per row
            op = OPCODES[oc]; idx = (opn == oc).nonzero().flatten(); a0, a1, a2, a3 = A[idx, 0], A[idx, 1], A[idx, 2], A[idx, 3]
            if op == "LOAD":
                okm = wid[idx, a0] == -5; ok = idx[okm]
                if len(ok): rr = hop(place[ok, a0[okm]], NEXT + 1); w = light(rr, cand[ok], cmask[ok]).argmax(1); place[ok, a0[okm]] = rr; wid[ok, a1[okm]] = w; place[ok, a1[okm]] = cand[ok, w]
                wid[idx[~okm], a1[~okm]] = -1
            elif op == "ISW": hit = wid[idx, a0] == SY[idx]; wid[idx[hit], a1[hit]] = -2
            elif op == "ISDIG": t = wid[idx, a0]; hit = (t >= 0) & isdig[t.clamp(min=0)]; wid[idx[hit], a1[hit]] = -2
            elif op == "ISPROG": t = wid[idx, a0]; hit = (t >= 0) & (isprog_stored[t.clamp(min=0)] | (t == st.def_name[idx])); wid[idx[hit], a1[hit]] = -2
            elif op == "ISNEW": t = wid[idx, a0]; hit = (t >= V) & (t != st.def_name[idx]); wid[idx[hit], a1[hit]] = -2
            elif op == "CCOPY": hm = wid[idx, a0] == -2; hit = idx[hm]; wid[hit, a2[hm]] = wid[hit, a1[hm]]; place[hit, a2[hm]] = place[hit, a1[hm]]; wid[hit, a0[hm]] = -1
            elif op == "CWRITE":
                flag = wid[idx, a0] == -2; t = wid[idx, a1]; c_ = cur[idx]
                dm = flag & (t >= 0) & isdig[t.clamp(min=0)] & (olen[idx, c_] < NP); dg = idx[dm]                        # a digit extends the current operand
                ops[dg, cur[dg], olen[dg, cur[dg]]] = t[dm]; olen[dg, cur[dg]] += 1
                am = flag & (t >= 0) & isarg[t.clamp(min=0)]; ag = idx[am]; oarg[ag, cur[ag]] = True                    # it / itself: the operand is the argument
                wid[idx[flag], a0[flag]] = -1
            elif op == "CNEWOP": hm = wid[idx, a0] == -2; hit = idx[hm]; adv = hit[nonempty(hit)]; cur[adv] = (cur[adv] + 1).clamp(max=1); wid[hit, a0[hm]] = -1
            elif op == "CSWAP": hm = wid[idx, a0] == -2; hit = idx[hm]; ops[hit] = ops[hit].flip(1); olen[hit] = olen[hit].flip(1); oarg[hit] = oarg[hit].flip(1); wid[hit, a0[hm]] = -1
            elif op == "SET": wid[idx, a0] = SY[idx]
            elif op == "COPY": wid[idx, a1] = wid[idx, a0]; place[idx, a1] = place[idx, a0]
            elif op == "CDEF":
                t = wid[idx, a0]; cw = wid[idx, a1]; filled = (olen[idx] > 0) | oarg[idx]; hm = (t >= V) & (cw >= 0) & isprog_stored[cw.clamp(min=0)] & filled.all(1)
                for i, n_, c in zip(idx[hm].tolist(), a0[hm].tolist(), a1[hm].tolist()): st.def_name[i] = int(wid[i, n_]); st.def_cmd[i] = VOCAB[int(wid[i, c])]; st.def_ops[i] = operands(i); defined_now[i] = True
            elif op == "CALLW":
                cw = wid[idx, a0]; hm = (cw >= 0) & ((olen[idx, 0] > 0) | oarg[idx, 0])
                for i, c in zip(idx[hm].tolist(), a0[hm].tolist()):
                    if calls[i] is not None or defined_now[i]: continue
                    tk = int(wid[i, c]); calls[i] = ("stored", VOCAB[tk]) if tk < V and VOCAB[tk] in PROGWORDS else ("defined", tk) if tk == int(st.def_name[i]) else ("unknown", tk); call_ops[i] = operands(i)
        pc = pc + 1
    return calls, call_ops, bnp
def answers_of(calls, call_ops, ep, bnp, st):
    """resolve each row's call to (program, x, y) — a 'defined' call substitutes its argument for ARG — and run the machine"""
    N = len(calls); cmds, xs, ys, out = [None] * N, [0] * N, [0] * N, [None] * N
    for i, c in enumerate(calls):
        if c is None: continue
        if c[0] == "unknown": out[i] = "NONE"; continue
        o = call_ops[i]
        if c[0] == "stored":
            x, y = o
            if not isinstance(x, int) or not isinstance(y, int): out[i] = "NONE"; continue
        else:
            arg = o[0]; dop = st.def_ops[i]
            if not isinstance(arg, int) or dop is None: out[i] = "NONE"; continue
            x, y = [arg if d == "ARG" else d for d in dop]
            if not isinstance(x, int) or not isinstance(y, int): out[i] = "NONE"; continue
        if x >= 10 ** NP or y >= 10 ** NP: out[i] = "NONE"; continue
        cmds[i], xs[i], ys[i] = (c[1] if c[0] == "stored" else st.def_cmd[i]), x, y
    r = run_calls(cmds, xs, ys)
    for i in range(N):
        if cmds[i] is not None: out[i] = r[i]
    return out
def run(progs, ep):
    """both sentences of each episode; returns answers (P*B, 2)"""
    N = len(progs) * ep.B; st = State(N); outs = []
    for j in range(2): calls, call_ops, bnp = execute(progs, ep, j, st); outs.append(answers_of(calls, call_ops, ep, bnp, st))
    return outs
def score(progs, ep):
    outs = run(progs, ep); P, B = len(progs), ep.B; ex = np.zeros((P, B)); part = np.zeros((P, B))
    for i in range(P * B):
        b = i % B; ok = True; credit = 0.0
        for j in range(2):
            t = ep.truth[b][j]; got = outs[j][i]
            if t is None: ok &= got is None; credit += 0.5 * (got is None)
            else: ok &= got == t; credit += 0.5 * (got == t) + 0.1 * (got is not None and got != "NONE" and t != "NONE")
        ex[i // B, b] = ok; part[i // B, b] = credit
    return ex.mean(1), (ex + part).mean(1)
# ground truth (to check the machine and to produce the worked examples only)
TRUTH_READ = {"init": ["SET 4<NONE"],
              "body": ["LOAD 0>1", "ISPROG 1>4", "CCOPY 4,1>2", "ISDIG 1>4", "ISW 1,it>4", "ISW 1,itself>4", "CWRITE 4,1", "ISNEW 1>4", "CCOPY 4,1>3",
                       "ISW 1,and>5", "ISW 1,by>5", "ISW 1,to>5", "ISW 1,from>5", "CNEWOP 5", "ISW 1,from>6", "ISW 1,dot>4", "ISW 1,qmark>4"],
              "outro": ["CSWAP 6", "CDEF 3,2", "CALLW 2"], "stop": 4}
def show(p): return f"init {p['init']}  body {p['body']}  outro {p['outro']}  stop {p['stop']}"
def trace_of(p, n_tokens): return list(p["init"]) + [x for _ in range(n_tokens) for x in p["body"]] + list(p["outro"])
def mutate(p):
    q = {k: (list(v) if isinstance(v, list) else v) for k, v in p.items()}; part = rng.choice(["body", "body", "init", "outro"]); seq = q[part]; r = rng.random()
    if r < 0.4 and seq: seq[rng.integers(len(seq))] = INS[rng.integers(len(INS))]
    elif r < 0.6 and len(seq) < 20: seq.insert(int(rng.integers(len(seq) + 1)), INS[rng.integers(len(INS))])
    elif r < 0.75 and seq: seq.pop(int(rng.integers(len(seq))))
    elif r < 0.9 and len(seq) > 1: i, j = rng.integers(len(seq), size=2); seq[i], seq[j] = seq[j], seq[i]
    else: q["stop"] = int(rng.integers(K - 3))
    return q
with torch.no_grad():
    for kinds, label in ((["cmd"], "commands"), (["def"], "definition + use")):
        ep = Episodes(120, kinds=kinds * 120, n_fill=(0, 2)); ex, _ = score([TRUTH_READ], ep); print(f"ground-truth reader on the machine, {label} (0–2 fillers, 1–3 digits): exact {ex[0]:.3f}", flush=True)
    ep = Episodes(6, kinds=["cmd", "def"] * 3); outs = run([TRUTH_READ], ep)
    for b in range(6): print("   ", ep.text(b, 0), "->", outs[0][b], " | ", ep.text(b, 1), "->", outs[1][b], "  (truth", ep.truth[b], ")", flush=True)
if a.check_only: raise SystemExit
# ------------------------------------------------------------------ the search: worked examples = traces of the ground truth; fitness = the machine's answers
def search(demos, kinds, label, seed_prog=None, accept_kinds=None):
    def from_demo():
        if seed_prog is not None and rng.random() < 0.5: return mutate(seed_prog) if rng.random() < 0.7 else {k: (list(v) if isinstance(v, list) else v) for k, v in seed_prog.items()}
        tr = demos[rng.integers(len(demos))]; blen = int(rng.integers(1, 20)); start = int(rng.integers(0, max(1, len(tr) - blen + 1))); body = tr[start:start + blen]
        init = tr[:start][-int(rng.integers(0, 4)):] if start > 0 and rng.random() < 0.8 else []; outro = tr[-int(rng.integers(1, 5)):] if rng.random() < 0.8 else []
        return {"init": list(init), "body": list(body), "outro": list(outro), "stop": int(rng.integers(0, K - 3))}
    pop = [from_demo() for _ in range(a.pop)]; found_p = None; best, best_f = pop[0], -1; ak = accept_kinds or kinds
    for g in range(1, a.gens + 1):
        global USE_POOL; USE_POOL = True; ep = Episodes(40, kinds=[kinds[i % len(kinds)] for i in range(40)], n_fill=(0, 1)); USE_POOL = False; ex, f = score(pop, ep); order = np.argsort(-f)
        if f[order[0]] > best_f: best, best_f = pop[order[0]], float(f[order[0]])
        if g % 5 == 0 or g == 1: print(f"  {label} gen {g}  best exact {ex[order[0]]:.3f} fitness {f[order[0]]:.3f}  mean {f.mean():.3f}  elapsed {time.time()-t0:.0f}s", flush=True)
        if ex[order[0]] >= 1.0:
            cands = [pop[i] for i in order if ex[i] >= 1.0]; cands.sort(key=lambda p: len(p["init"]) + len(p["body"]) + len(p["outro"]))
            for cand_ in cands[:5]:
                if score([cand_], Episodes(64, kinds=[ak[i % len(ak)] for i in range(64)], n_digits=(4, 5), n_fill=(1, 2)))[0][0] >= 1.0: found_p = cand_; break      # accept only on longer numbers and more fillers than shown
            if found_p: print(f"  {label} found at generation {g}: {show(found_p)}", flush=True); return found_p, g, True
        elite = [pop[i] for i in order[:max(2, a.pop // 20)]]; new = list(elite)
        while len(new) < a.pop:
            i, j = order[rng.integers(0, max(4, a.pop // 4), size=2)]; child = mutate(pop[i] if f[i] >= f[j] else pop[j]); new.append(mutate(child) if rng.random() < 0.3 else child)
        pop = new
    print(f"  {label} not found; best fitness {best_f:.3f}: {show(best)}", flush=True); return best, g, False
with torch.no_grad():
    demos = {"cmd": [], "def": []}
    for kinds, nfill in (("cmd", 1), ("cmd", 0), ("def", 0)):
        ep1 = Episodes(1, kinds=[kinds], n_fill=(nfill, nfill)); demos[kinds] += [trace_of(TRUTH_READ, len(ep1.sents[j][0])) for j in range(2)]; print("worked example:", ep1.text(0, 0), "|", ep1.text(0, 1), "->", ep1.truth[0], flush=True)
    if a.ladder:
        print("\n== rung 1: the command reader (two command episodes as worked examples)", flush=True)
        p1, g1, ok1 = search(demos["cmd"], ["cmd"], "commands"); res["reader_commands"] = {"found": ok1, "generation": g1, "program": p1}
        print("\n== rung 2: the definition reader, seeded with the command reader (one definition episode as worked example)", flush=True)
        prog, g, ok = search(demos["def"], ["cmd", "def"], "definitions", seed_prog=p1); found_p = prog if ok else None
    else:
        prog, g, ok = search(demos["cmd"] + demos["def"], ["cmd", "def"], "reader"); found_p = prog if ok else None
    prog = found_p or best; res["reader"] = {"found": found_p is not None, "generation": g, "program": prog}
    # tests: by number length (mul operands capped at 4 digits), fillers, definitions with unseen names/bodies, unknown command words
    v = {d: float(score([prog], Episodes(100, kinds=["cmd"] * 100, n_digits=(d, d), n_fill=(0, 1), mul_digits=4))[0][0]) for d in range(1, 8)}
    print("  commands by number length 1..7 digits (>3 never shown): " + " ".join(f"{d}:{e:.2f}" for d, e in v.items()), flush=True); res["commands_by_digits"] = v
    v = {nf: float(score([prog], Episodes(100, kinds=["cmd"] * 100, n_fill=(nf, nf)))[0][0]) for nf in range(0, 5)}
    print("  commands by filler words 0..4 (>1 never shown): " + " ".join(f"{nf}:{e:.2f}" for nf, e in v.items()), flush=True); res["commands_by_fillers"] = v
    v = {d: float(score([prog], Episodes(100, kinds=["def"] * 100, n_digits=(d, d), mul_digits=4))[0][0]) for d in range(1, 8)}
    print("  definition + use by number length 1..7 (unseen names, bodies add/add k/subtract k/multiply k): " + " ".join(f"{d}:{e:.2f}" for d, e in v.items()), flush=True); res["definitions_by_digits"] = v
    ep = Episodes(200, kinds=["cmd"] * 200, unknown=1.0); outs = run([prog], ep); ab = np.mean([outs[j][b] in (None, "NONE") for b in range(200) for j in range(2)]); print(f"  unknown command word ('W 12 and 7 .'): no call made or NONE {ab:.2f}", flush=True); res["unknown_word_abstain"] = float(ab)
    ep = Episodes(4, kinds=["cmd", "def"] * 2, n_digits=(4, 6)); outs = run([prog], ep)
    for b in range(4): print("   ", ep.text(b, 0), "->", outs[0][b], " | ", ep.text(b, 1), "->", outs[1][b], "  (truth", ep.truth[b], ")", flush=True)
save_json(res, HERE / a.out / "read_math.json"); torch.save({"lm": lm.state_dict(), "lk": a.lk, "reader": prog}, HERE / a.out / "read_math.pt"); print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)
