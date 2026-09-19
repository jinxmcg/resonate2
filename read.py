"""Reading, the way arithmetic was done: a sentence is a place (a directed chain of word places, one move NEXT); reading is a
program over lights, found from worked examples; word order, the role of each position and which words carry structure
are not given — the program has to find them.
memory : rows for a few stored words (is, than, by, much, very, '.', '?'); every other word (names, relation words) is a
         FRESH random row, never trained. NEXT chains a sentence: acc = NEXT(w_L); acc = NEXT(acc + w_j) for j = L-1..1, so
         walking back (NEXT^-1, one light per step against the sentence's own words + the stored words) reads the words
         first to last. Three role relations make a fact place s⊗SUBJECT + r⊗RELATION + o⊗OBJECT (the given format).
         Both trained once on random sequences / random triples of random places — nothing about sentences or templates.
machine: K slots; each holds a word (index into the sentence's candidate words) or a symbol, and a walkable place.
         LOAD s>t (walk the sentence in s one word forward, the word into t), ISW s,w>u (YES if the word in s is the stored
         word w), FACT s,r,o / CFACT v,s,r,o (write the fact place of three slots into slot K-1, the latter only if v is YES),
         SET, COPY, IFYES. Programs = init + loop body + outro + stop slot, as before.
stage 1: sentences as chains — read-back exact vs length, unseen words.
stage 2: the reading program from 3 worked (sentence -> fact) examples; templates 'X is R than Y', with modifiers ('much',
         'very' stacked, so lengths vary) and the passive 'Y is R by X' (roles reversed); accepted only if exact on
         unseen words and sentences LONGER than shown; tested on more modifiers than shown.
stage 6: no fact supervision — only (story text, question text, answer). The reading program is scored by the answer a
         fixed decision routine (the Part-17 chain/decide program, executed as code) gives over the facts it wrote.
         (a) blank slate; (b) seeded with the stage-2 program found on statements only, asked to extend to the passive
         from answers alone."""
import argparse, time, json, numpy as np, torch, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=12); ap.add_argument("--block", type=int, default=16); ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--pop", type=int, default=150); ap.add_argument("--gens", type=int, default=150); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results/read"); ap.add_argument("--slots", type=int, default=8)
ap.add_argument("--stages", default="1,2,6")
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time(); K = a.slots
VOCAB = ["is", "than", "by", "much", "very", ".", "?"]; V = len(VOCAB); vid = {w: i for i, w in enumerate(VOCAB)}; NF = 6                      # stored words; NF fresh rows per sentence/story
NEXT, SUBJ, REL, OBJ = 0, 1, 2, 3; NR = 4
model = ResonatE(V, 2 * NR, k=a.k, block=True, block_size=a.block).to(dev); M = model.m
def E(): return cnorm(model.E)
def hop(z, r): return model.hop(z, torch.full((z.shape[0],), r, device=dev, dtype=torch.long))
def fact(s, r, o): return cnorm(hop(s, SUBJ) + hop(r, REL) + hop(o, OBJ))
def key(s, r): return cnorm(hop(s, SUBJ) + hop(r, REL))
def chain(W, mask):
    """W (B, L, M) word places, mask (B, L): the sentence place; built last word first so the walk back reads first word first"""
    B, L, _ = W.shape; acc = None
    for j in range(L - 1, -1, -1):
        has = mask[:, j]
        step = hop(W[:, j], NEXT) if acc is None else hop(cnorm(acc + W[:, j]), NEXT)
        acc = step if acc is None else torch.where(has[:, None], step, acc)
    return acc
def light(z, cand, cmask):
    sc = torch.real((z[:, None, :] * cand.conj()).sum(-1)); sc[~cmask] = -1e9; return sc
# ------------------------------------------------------------------ 1. the memory: NEXT learns to chain (walk-back objective), the roles learn what a fact is
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
def random_sentences(B, Lmax=9, Lmin=2):
    """random word sequences over [stored words + NF fresh rows]; returns cand (B, V+NF, M), cmask, wid (B, Lmax), mask"""
    fresh = cnorm(torch.randn(B, NF, M, dtype=torch.complex64, device=dev)); cand = torch.cat([E()[None].expand(B, -1, -1), fresh], 1); cmask = torch.ones(B, V + NF, dtype=torch.bool, device=dev)
    L = torch.tensor(rng.integers(Lmin, Lmax + 1, B), device=dev); wid = torch.tensor(rng.integers(0, V + NF, (B, Lmax)), device=dev); mask = torch.arange(Lmax, device=dev)[None] < L[:, None]
    return cand, cmask, wid, mask, L
for step in range(1, a.steps + 1):
    B = 256; cand, cmask, wid, mask, L = random_sentences(B); W = cand[torch.arange(B, device=dev)[:, None], wid]; z = chain(W, mask); tau = model.log_tau.exp(); loss = 0.0; n = 0
    rr = z
    for j in range(wid.shape[1]):                                                                             # walk back: the j-th reverse hop must light word j
        rr = hop(rr, NEXT + NR); has = mask[:, j]
        if has.any(): loss = loss + F.cross_entropy(light(rr, cand, cmask)[has] * tau, wid[has, j]); n += 1
    loss = loss / n
    s = torch.tensor(rng.integers(0, V + NF, B), device=dev); r = torch.tensor(rng.integers(0, V + NF, B), device=dev); o = torch.tensor(rng.integers(0, V + NF, B), device=dev); ar = torch.arange(B, device=dev)
    f = fact(cand[ar, s], cand[ar, r], cand[ar, o])
    for role, tgt in ((SUBJ, s), (REL, r), (OBJ, o)): loss = loss + F.cross_entropy(light(hop(f, role + NR), cand, cmask) * tau, tgt)   # read the three parts back
    loss = loss + F.cross_entropy(torch.real(key(cand[ar, s], cand[ar, r]) @ f.conj().t()) * tau, ar)                                    # find the fact by its key among others
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0 or step == 1: print(f"memory step {step}/{a.steps}  loss {loss.item():.4f}  lr 3e-3  tau {tau.item():.1f}  elapsed {time.time()-t0:.0f}s", flush=True)
model.eval(); model.requires_grad_(False); res = {}
# ------------------------------------------------------------------ stage 1: sentences as chains, read back by walking
def walk(z, cand, cmask, L):
    out = torch.zeros(z.shape[0], L, dtype=torch.long, device=dev); rr = z
    for j in range(L): rr = hop(rr, NEXT + NR); out[:, j] = light(rr, cand, cmask).argmax(1)
    return out
if "1" in a.stages.split(","):
    with torch.no_grad():
        by = {}
        for L in range(2, 13):
            cand, cmask, wid, mask, _ = random_sentences(500, Lmax=L, Lmin=L); W = cand[torch.arange(500, device=dev)[:, None], wid]; z = chain(W, mask); got = walk(z, cand, cmask, L)
            by[L] = float((got == wid).all(1).float().mean())
        res["stage1_readback_by_length"] = by; print("stage 1 — sentence as one place, read back by walking, exact by length (train 2–9, fresh words): " + " ".join(f"{L}:{v:.3f}" for L, v in by.items()), flush=True)
# ------------------------------------------------------------------ sentences from templates: names and relation words are fresh rows (indices V.. in the sentence's candidates)
# fresh slots: 0 X, 1 Y, 2 R, 3.. spare names (stories use several names; one relation word per story)
def render(template, X, Y, R, mods):
    """template 'active': X is (mods) R than Y .   'passive': Y is (mods) R by X .   returns word ids (into cand) and the (s, r, o) truth"""
    m = [vid[w] for w in mods]
    if template == "active": return [X, vid["is"]] + m + [R, vid["than"], Y, vid["."]], (X, R, Y)
    else: return [Y, vid["is"]] + m + [R, vid["by"], X, vid["."]], (X, R, Y)
class Sentences:
    """B sentences over per-row candidate words: cand (B, V+NF, M); wid lists; the chain place z; truth (s, r, o) or None"""
    def __init__(self, B, templates, max_mods, n_mods=None, fresh=None, X=None, Y=None, R=None):
        self.B = B; self.fresh = cnorm(torch.randn(B, NF, M, dtype=torch.complex64, device=dev)) if fresh is None else fresh; self.cand = torch.cat([E()[None].expand(B, -1, -1), self.fresh], 1); self.cmask = torch.ones(B, V + NF, dtype=torch.bool, device=dev)
        self.wids, self.truth, self.template = [], [], []
        for b in range(B):
            t = templates[rng.integers(len(templates))]; nm = int(n_mods if n_mods is not None else rng.integers(0, max_mods + 1)); mods = [["much", "very"][rng.integers(2)] for _ in range(nm)]
            x = V + 0 if X is None else X[b]; y = V + 1 if Y is None else Y[b]; r = V + 2 if R is None else R[b]
            w, tr = render(t, x, y, r, mods); self.wids.append(w); self.truth.append(tr); self.template.append(t)
        Lmax = max(len(w) for w in self.wids); wid = torch.zeros(B, Lmax, dtype=torch.long, device=dev); mask = torch.zeros(B, Lmax, dtype=torch.bool, device=dev)
        for b, w in enumerate(self.wids): wid[b, :len(w)] = torch.tensor(w, device=dev); mask[b, :len(w)] = True
        self.wid, self.mask = wid, mask; self.z = chain(self.cand[torch.arange(B, device=dev)[:, None], wid], mask)
# ------------------------------------------------------------------ the reading machine
SYMC = {"NONE": -1, "YES": -2, "NO": -3, "UNKNOWN": -4}
def instruction_set():
    pairs = [(s, t) for s in range(K - 1) for t in range(K - 1) if s != t]
    return [f"LOAD {s}>{t}" for s, t in pairs] + [f"ISW {s},{w}>{u}" for s in range(K - 1) for w in VOCAB for u in range(K - 1) if u != s] \
        + [f"FACT {s},{r},{o}" for s in range(K - 1) for r in range(K - 1) for o in range(K - 1) if len({s, r, o}) == 3] \
        + [f"CFACT {v},{s},{r},{o}" for v in range(K - 1) for s in range(K - 1) for r in range(K - 1) for o in range(K - 1) if len({v, s, r, o}) == 4] \
        + [f"SET {u}<{y}" for u in range(K - 1) for y in ("NONE", "YES")] + [f"COPY {s}>{t}" for s, t in pairs] + [f"IFYES {v},{y}>{u}" for v in range(K - 1) for y in ("YES", "NONE") for u in range(K - 1) if u != v]
INS = instruction_set(); I = {n: i for i, n in enumerate(INS)}; PARSED = []
for name in INS:
    op, rest = name.split(" ", 1); nums = [int(x) for x in rest.replace(">", ",").replace("<", ",").split(",") if x.strip().lstrip("-").isdigit()]; sym = [x for x in rest.replace(">", ",").replace("<", ",").split(",") if not x.strip().lstrip("-").isdigit()]
    PARSED.append((op, nums, sym[0] if sym else None))
def execute(progs, sn, max_cycles=10):
    """P programs x B sentences. Returns the fact place written to slot K-1 (N, M) and a written flag (N,)"""
    P, B = len(progs), sn.B; N = P * B; bidx = torch.arange(N, device=dev) % B; ar = torch.arange(N, device=dev)
    wid = torch.full((N, K), -9, dtype=torch.long, device=dev); place = torch.zeros(N, K, M, dtype=torch.complex64, device=dev); place[:, 0] = sn.z[bidx]; wid[:, 0] = -5      # slot 0: the sentence (walkable)
    cand, cmask = sn.cand[bidx], sn.cmask[bidx]; written = torch.zeros(N, dtype=torch.bool, device=dev)
    maxlen = max(len(p["init"]) for p in progs) + max_cycles * max(max(len(p["body"]) for p in progs), 1) + max(len(p["outro"]) for p in progs) + 1
    code = torch.full((N, maxlen), -1, dtype=torch.long, device=dev); cstart = torch.zeros(N, maxlen, dtype=torch.bool, device=dev); ostart = torch.zeros(N, dtype=torch.long, device=dev); stop_slot = torch.tensor([p["stop"] for p in progs], device=dev).repeat_interleave(B)
    for pi, p in enumerate(progs):
        seq = [I[x] for x in p["init"]]; starts = []
        for c in range(max_cycles if p["body"] else 0): starts.append(len(seq)); seq += [I[x] for x in p["body"]]
        rows = slice(pi * B, (pi + 1) * B); ostart[rows] = len(seq); seq += [I[x] for x in p["outro"]]; code[rows, :len(seq)] = torch.tensor(seq, device=dev); cstart[rows, starts] = True
    loop_done = torch.zeros(N, dtype=torch.bool, device=dev); ncyc = torch.zeros(N, dtype=torch.long, device=dev); pc = torch.zeros(N, dtype=torch.long, device=dev)
    for t in range(maxlen + 2):
        pcc = pc.clamp(max=maxlen - 1); ins = torch.where(pc < maxlen, code[ar, pcc], torch.full_like(pc, -1)); active = ins >= 0
        if not active.any(): break
        cs = cstart[ar, pcc] & active & ~loop_done
        if cs.any():
            sv = wid[ar, stop_slot]; term = (sv == -1) | (sv == -2); fin = cs & ((ncyc > 0) & term | (ncyc >= max_cycles - 1))
            loop_done = loop_done | fin; pc = torch.where(fin, ostart, pc); ncyc = ncyc + (cs & ~fin).long(); pcc = pc.clamp(max=maxlen - 1); ins = torch.where(pc < maxlen, code[ar, pcc], torch.full_like(pc, -1)); active = ins >= 0
        for iid in torch.unique(ins[active]).tolist():
            op, nums, sym = PARSED[iid]; mk = active & (ins == iid); idx = mk.nonzero().flatten()
            if op == "LOAD":
                s, tt = nums; ok = idx[wid[idx, s] == -5]                                                       # only a walkable (sentence) place can be walked
                if len(ok): rr = hop(place[ok, s], NEXT + NR); w = light(rr, cand[ok], cmask[ok]).argmax(1); place[ok, s] = rr; wid[ok, tt] = w; place[ok, tt] = cand[ok, w]
                bad = idx[wid[idx, s] != -5]; wid[bad, tt] = -1
            elif op == "ISW": s, u = nums; hit = wid[idx, s] == vid[sym]; wid[idx[hit], u] = -2
            elif op in ("FACT", "CFACT"):
                if op == "CFACT": v, s, r, o = nums; ok = idx[(wid[idx, v] == -2) & (wid[idx, s] >= 0) & (wid[idx, r] >= 0) & (wid[idx, o] >= 0)]
                else: s, r, o = nums; ok = idx[(wid[idx, s] >= 0) & (wid[idx, r] >= 0) & (wid[idx, o] >= 0)]
                if len(ok): place[ok, K - 1] = fact(place[ok, s], place[ok, r], place[ok, o]); written[ok] = True
            elif op == "SET": u = nums[0]; wid[idx, u] = SYMC[sym]
            elif op == "COPY": s, tt = nums; wid[idx, tt] = wid[idx, s]; place[idx, tt] = place[idx, s]
            elif op == "IFYES": v, u = nums; hit = wid[idx, v] == -2; wid[idx[hit], u] = SYMC[sym]
        pc = pc + 1
    return place[:, K - 1], written
def decode(f, cand, cmask):
    """the three roles of a fact place read back as word indices"""
    return torch.stack([light(hop(f, role + NR), cand, cmask).argmax(1) for role in (SUBJ, REL, OBJ)], 1)
def score_facts(progs, sn):
    f, wr = execute(progs, sn); P = len(progs); bidx = torch.arange(P * sn.B, device=dev) % sn.B; got = decode(f, sn.cand[bidx], sn.cmask[bidx]); truth = torch.tensor(sn.truth, device=dev)[bidx]
    ex = (wr & (got == truth).all(1)).float().reshape(P, sn.B); part = (wr[:, None] & (got == truth)).float().mean(1).reshape(P, sn.B)        # fitness: exact + per-role credit
    return ex.mean(1).cpu().numpy(), (ex + 0.5 * part).mean(1).cpu().numpy()
# ground truth (to check the machine and to produce worked examples only): walk to 'is', skip modifiers to the relation word, read the marker, then the last name; roles by the marker
TRUTH_READ = {"init": ["LOAD 0>1", "LOAD 0>2"], "body": ["LOAD 0>2", "ISW 2,much>3", "ISW 2,very>3", "IFYES 3,NONE>3", "SET 3<NONE", "COPY 2>4"], "outro": [], "stop": 3}      # placeholder, replaced below
# simpler, exact: init reads X and 'is'; the loop loads words until one is not a modifier (that word is R); then marker, last name; two conditional facts
TRUTH_READ = {"init": ["LOAD 0>1", "LOAD 0>2", "SET 3<NONE"],
              "body": ["LOAD 0>4", "SET 3<YES", "ISW 4,much>5", "ISW 4,very>5", "IFYES 5,NONE>3", "SET 5<NONE"],                                     # stop slot 3 is YES once a non-modifier word (R) was loaded
              "outro": ["LOAD 0>5", "LOAD 0>6", "ISW 5,than>2", "CFACT 2,1,4,6", "SET 2<NONE", "ISW 5,by>2", "CFACT 2,6,4,1"], "stop": 3}
def show(p): return f"init {p['init']}  body {p['body']}  outro {p['outro']}  stop {p['stop']}"
def mutate(p):
    q = {k: (list(v) if isinstance(v, list) else v) for k, v in p.items()}; part = rng.choice(["body", "init", "outro", "outro"]); seq = q[part]; r = rng.random()
    if r < 0.4 and seq: seq[rng.integers(len(seq))] = INS[rng.integers(len(INS))]
    elif r < 0.6 and len(seq) < 10: seq.insert(int(rng.integers(len(seq) + 1)), INS[rng.integers(len(INS))])
    elif r < 0.75 and seq: seq.pop(int(rng.integers(len(seq))))
    elif r < 0.9 and len(seq) > 1: i, j = rng.integers(len(seq), size=2); seq[i], seq[j] = seq[j], seq[i]
    else: q["stop"] = int(rng.integers(K - 1))
    return q
def trace_of(p, sn_one):
    L = len(sn_one.wids[0]); n_mods = L - 6; return list(p["init"]) + [x for _ in range(n_mods + 1) for x in p["body"]] + list(p["outro"])
def search(fitness_fn, accept_fn, demos, seed_prog=None, label=""):
    """demos: list of instruction traces (flat step lists) or None (blank slate); returns (program, generation, found)"""
    def from_demo():
        if demos is None and seed_prog is None: return {"init": [INS[rng.integers(len(INS))] for _ in range(int(rng.integers(1, 5)))], "body": [INS[rng.integers(len(INS))] for _ in range(int(rng.integers(0, 5)))], "outro": [INS[rng.integers(len(INS))] for _ in range(int(rng.integers(0, 6)))], "stop": int(rng.integers(K - 1))}
        if seed_prog is not None and (demos is None or rng.random() < 0.5): return mutate(seed_prog) if rng.random() < 0.7 else {k: (list(v) if isinstance(v, list) else v) for k, v in seed_prog.items()}
        tr = demos[rng.integers(len(demos))]; blen = int(rng.integers(0, 7)); start = int(rng.integers(0, max(1, len(tr) - blen + 1))); body = tr[start:start + blen]
        init = tr[:start][-int(rng.integers(0, 6)):] if start > 0 and rng.random() < 0.8 else []; outro = tr[start + blen:][:int(rng.integers(0, 9))] if rng.random() < 0.8 else []
        return {"init": list(init), "body": list(body), "outro": list(outro), "stop": int(rng.integers(0, K - 1))}
    pop = [from_demo() for _ in range(a.pop)]; found = None; best, best_f = pop[0], -1
    for g in range(1, a.gens + 1):
        ex, f = fitness_fn(pop); order = np.argsort(-f)
        if f[order[0]] > best_f: best, best_f = pop[order[0]], float(f[order[0]])
        if g % 10 == 0 or g == 1: print(f"  {label} gen {g}  best exact {ex[order[0]]:.3f} fitness {f[order[0]]:.3f}  mean {f.mean():.3f}  elapsed {time.time()-t0:.0f}s", flush=True)
        if ex[order[0]] >= 1.0:
            cands = [pop[i] for i in order if ex[i] >= 1.0]; cands.sort(key=lambda p: len(p["init"]) + len(p["body"]) + len(p["outro"]))
            for cand_ in cands[:5]:
                if accept_fn(cand_): found = cand_; break
            if found: print(f"  {label} found at generation {g}: {show(found)}", flush=True); return found, g, True
        elite = [pop[i] for i in order[:max(2, a.pop // 20)]]; new = list(elite)
        while len(new) < a.pop:
            i, j = order[rng.integers(0, max(4, a.pop // 4), size=2)]; child = mutate(pop[i] if f[i] >= f[j] else pop[j]); new.append(mutate(child) if rng.random() < 0.3 else child)
        pop = new
    print(f"  {label} not found in {a.gens} generations; best fitness {best_f:.3f}: {show(best)}", flush=True); return best, g, False
# ------------------------------------------------------------------ stage 2: the reading program from 3 worked (sentence -> fact) examples
READ = None
if "2" in a.stages.split(","):
    with torch.no_grad():
        chk = {t: float(score_facts([TRUTH_READ], Sentences(200, [t], 2))[0][0]) for t in ("active", "passive")}; print(f"ground-truth reading program on the machine (fresh words, 0–2 modifiers): {chk}", flush=True)
        for setting, templates in (("active", ["active"]), ("both", ["active", "passive"])):
            demos = []
            for i in range(3):
                sn1 = Sentences(1, [templates[i % len(templates)]], 1, n_mods=i % 2); demos.append(trace_of(TRUTH_READ, sn1))
            print(f"\n== stage 2 ({setting}): worked examples " + "; ".join(f"{' '.join(VOCAB[w] if w < V else 'XYR'[w - V] for w in sn.wids[0]) if False else len(d)} steps" for d, sn in zip(demos, [None]*3)) + f"  (templates {templates}, 0–1 modifiers)", flush=True)
            fit = lambda pop: score_facts(pop, Sentences(48, templates, 1))
            acc = lambda p: score_facts([p], Sentences(64, templates, 3, n_mods=3))[0][0] >= 1.0                                        # accept only on LONGER sentences than shown
            prog, g, ok = search(fit, acc, demos, label=f"read/{setting}")
            v = {t: {nm: float(score_facts([prog], Sentences(100, [t], 0, n_mods=nm))[0][0]) for nm in range(0, 5)} for t in ("active", "passive")}
            print(f"  read/{setting} by template and modifiers 0..4 (>1 never shown), fresh words: " + "  ".join(f"{t}: " + " ".join(f"{nm}:{e:.2f}" for nm, e in d.items()) for t, d in v.items()), flush=True)
            res[f"stage2_{setting}"] = {"found": ok, "generation": g, "program": prog, "by_template_mods": v}
            if setting == "active": READ_ACTIVE = prog
            else: READ = prog
# ------------------------------------------------------------------ stage 6: answers only. stories of 2–3 statements + a question; a fixed decision routine over the facts the program wrote
def story_batch(B, templates, max_links=3, lengths=None):
    """each story: names 0..L (chain of the story's relation R in order, statements shuffled), one question 'a is R than b ?'; answer YES/NO/UNKNOWN by structure"""
    fresh = cnorm(torch.randn(B, NF, M, dtype=torch.complex64, device=dev)); R = V + NF - 1; S = max_links; sents, qs, ans = [[] for _ in range(S)], [], []
    Xs = [[V] * B for _ in range(S)]; Ys = [[V] * B for _ in range(S)]; valid = torch.zeros(B, S, dtype=torch.bool, device=dev)
    for b in range(B):
        L = int(lengths[b] if lengths is not None else rng.integers(1, max_links + 1)); names = list(range(V, V + L + 1)); extra = V + L + 1
        links = [(names[i], names[i + 1]) for i in range(L)]; rng.shuffle(links)
        for j, (x, y) in enumerate(links): Xs[j][b], Ys[j][b] = x, y; valid[b, j] = True
        kind = rng.choice(["yes", "no", "unknown"])
        if kind == "yes": i = int(rng.integers(0, L)); j = int(rng.integers(i + 1, L + 1)); qs.append((names[i], names[j])); ans.append("YES")
        elif kind == "no": i = int(rng.integers(0, L)); j = int(rng.integers(i + 1, L + 1)); qs.append((names[j], names[i])); ans.append("NO")
        else: qs.append((names[int(rng.integers(0, L + 1))], extra)); ans.append("UNKNOWN")
    sn = [Sentences(B, templates, 1, fresh=fresh, X=Xs[j], Y=Ys[j], R=[R] * B) for j in range(S)]
    qn = Sentences(B, ["active"], 1, fresh=fresh, X=[q[0] for q in qs], Y=[q[1] for q in qs], R=[R] * B)
    for b in range(B): qn.wids[b][-1] = vid["?"]
    Lmax = max(len(w) for w in qn.wids); wid = torch.zeros(B, Lmax, dtype=torch.long, device=dev); mask = torch.zeros(B, Lmax, dtype=torch.bool, device=dev)
    for b, w in enumerate(qn.wids): wid[b, :len(w)] = torch.tensor(w, device=dev); mask[b, :len(w)] = True
    qn.wid, qn.mask = wid, mask; qn.z = chain(qn.cand[torch.arange(B, device=dev)[:, None], wid], mask)
    return sn, qn, valid, ans
def decide(facts, fmask, cand, cmask, q):
    """the Part-17 decide program as code: YES if chain(a -> b), NO if chain(b -> a), else UNKNOWN; each read verified by read-back"""
    N = facts.shape[0]; ar = torch.arange(N, device=dev)
    def chain_to(a_, b_):
        cur = a_.clone(); hit = torch.zeros(N, dtype=torch.bool, device=dev); alive = torch.ones(N, dtype=torch.bool, device=dev)
        for _ in range(10):
            k = key(cand[ar, cur], cand[ar, q[:, 1]]); sc = torch.real((k[:, None, :] * facts.conj()).sum(-1)); sc[~fmask] = -1e9; best = sc.argmax(1); f = facts[ar, best]
            s_ok = light(hop(f, SUBJ + NR), cand, cmask).argmax(1) == cur; r_ok = light(hop(f, REL + NR), cand, cmask).argmax(1) == q[:, 1]; o = light(hop(f, OBJ + NR), cand, cmask).argmax(1)
            alive = alive & s_ok & r_ok & fmask[ar, best] & (o != cur); hit = hit | (alive & (o == b_)); cur = torch.where(alive, o, cur); alive = alive & ~hit
        return hit
    yes = chain_to(q[:, 0], q[:, 2]); no = chain_to(q[:, 2], q[:, 0]); out = torch.full((N,), -4, dtype=torch.long, device=dev); out[no] = -3; out[yes] = -2; return out
def score_answers(progs, sn, qn, valid, ans):
    P, B = len(progs), qn.B; N = P * B; bidx = torch.arange(N, device=dev) % B; S = len(sn)
    facts = torch.zeros(N, S, M, dtype=torch.complex64, device=dev); fmask = torch.zeros(N, S, dtype=torch.bool, device=dev)
    for j in range(S): f, wr = execute(progs, sn[j]); facts[:, j] = f; fmask[:, j] = wr & valid[bidx, j]
    fq, wq = execute(progs, qn); q = decode(fq, qn.cand[bidx], qn.cmask[bidx])                                          # the question read by the same program: (a, R, b)
    out = decide(facts, fmask, qn.cand[bidx], qn.cmask[bidx], q); out[~wq] = -9
    truth = torch.tensor([SYMC[x] for x in ans], device=dev)[bidx]; ex = (out == truth).float().reshape(P, B)
    return ex.mean(1).cpu().numpy(), ex.mean(1).cpu().numpy()
if "6" in a.stages.split(","):
    with torch.no_grad():
        for templates in (["active"], ["active", "passive"]):
            sn, qn, valid, ans = story_batch(300, templates); ex = score_answers([TRUTH_READ], sn, qn, valid, ans)[0][0]
            print(f"ground-truth reading + fixed decide on text stories {templates}: answer exact {ex:.3f} (chance 0.33)", flush=True)
        fitb = lambda pop: score_answers(pop, *story_batch(48, ["active", "passive"]))
        accb = lambda p: score_answers([p], *story_batch(64, ["active", "passive"], max_links=6))[0][0] >= 1.0
        print("\n== stage 6a: blank slate, answers only (stories with active and passive statements)", flush=True)
        prog, g, ok = search(fitb, accb, None, label="answers/blank"); res["stage6_blank"] = {"found": ok, "generation": g, "program": prog}
        seed_prog = res.get("stage2_active", {}).get("program") if "2" in a.stages.split(",") else TRUTH_READ
        print("\n== stage 6b: seeded with the statements-only reading program, asked to extend to the passive from answers alone", flush=True)
        base = score_answers([seed_prog], *story_batch(300, ["active", "passive"]))[0][0]; print(f"  seed program on active+passive stories before search: {base:.3f}", flush=True)
        prog, g, ok = search(fitb, accb, None, seed_prog=seed_prog, label="answers/seeded"); res["stage6_seeded"] = {"found": ok, "generation": g, "program": prog, "seed_exact": float(base)}
        for name_, p in (("blank", res["stage6_blank"]["program"]), ("seeded", prog)):
            v = {L: float(score_answers([p], *story_batch(100, ["active", "passive"], lengths=[L] * 100))[0][0]) for L in range(1, 9)}
            vf = {t: float(score_facts([p], Sentences(200, [t], 2))[0][0]) for t in ("active", "passive")}
            print(f"  {name_}: answers by chain length 1..8 (>3 never shown): " + " ".join(f"{L}:{e:.2f}" for L, e in v.items()) + f";  its facts decoded against the never-used labels: {vf}", flush=True)
            res[f"stage6_{name_}"]["by_length"] = v; res[f"stage6_{name_}"]["facts_vs_labels"] = vf
save_json(res, HERE / a.out / "read.json"); torch.save({"model": model.state_dict(), "programs": {k: v.get("program") for k, v in res.items() if isinstance(v, dict) and "program" in v}}, HERE / a.out / "read.pt"); print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)
