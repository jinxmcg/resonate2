"""Transitive reasoning from three worked examples, on a story memory.
memory : a fact = subject(x)SUBJECT + relation(x)RELATION + object(x)OBJECT — three role relations, one place. A story is a set
         of fact places in working memory (cleared per question). Names are FRESH random rows never seen in training.
         The roles are trained once on random triples (read the three parts back; find the fact keyed by (subject,
         relation) among others) — nothing about stories, chains or transitivity.
machine: slots hold name places or symbols. Primitives: READF s,r>t (the story's fact keyed by the name in s and the
         relation r; its object into t; NONE if none lights), SAME s,t>u (YES into u if the names match), IFYES v,sym>u
         (write sym into u if v reads YES — the branch), SET, COPY, ANSWER, CALL.  Loop stops when the stop slot reads a
         terminal symbol (NONE / YES).
programs to FIND from 3 worked stories each (teachers never seen): chain_bigger (follow 'bigger' until the target or
         nothing), decide_bigger (YES if chain a->b, NO if chain b->a, else UNKNOWN; calls chain), and the non-transitive
         trap: decide_knows (one hop only).
test   : chain lengths 1..10 (train stories <= 3 links), fresh names, distractors, reversed and unrelated queries."""
import argparse, time, json, numpy as np, torch, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=12); ap.add_argument("--block", type=int, default=16); ap.add_argument("--steps", type=int, default=2500)
ap.add_argument("--pop", type=int, default=150); ap.add_argument("--gens", type=int, default=120); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results/story")
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time()
NTRAIN = 120; RELS = ["bigger", "knows"]; SYMS = ["NONE", "YES", "NO", "UNKNOWN"]
NODES = [f"n{i}" for i in range(NTRAIN)] + RELS + SYMS; nid = {n: i for i, n in enumerate(NODES)}; ROLES = ["SUBJECT", "RELATION", "OBJECT"]; NR = 3; SUBJ, REL, OBJ = 0, 1, 2
model = ResonatE(len(NODES), 2 * NR, k=a.k, block=True, block_size=a.block).to(dev); M = model.m
def E(): return cnorm(model.E)
def hop(z, r): return model.hop(z, torch.full((z.shape[0],), r, device=dev, dtype=torch.long))
def fact(s, r, o): return cnorm(hop(s, SUBJ) + hop(r, REL) + hop(o, OBJ))
def key(s, r): return cnorm(hop(s, SUBJ) + hop(r, REL))
# ------------------------------------------------------------------ 1. the memory learns what a fact is (random triples over TRAINING names; nothing about stories)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
for step in range(1, a.steps + 1):
    Er = E(); B = 256; s = torch.tensor(rng.integers(0, NTRAIN, B), device=dev); o = torch.tensor(rng.integers(0, NTRAIN, B), device=dev); r = torch.tensor(rng.integers(NTRAIN, NTRAIN + len(RELS), B), device=dev)
    f = fact(Er[s], Er[r], Er[o]); tau = model.log_tau.exp(); loss = 0.0
    for role, tgt in ((SUBJ, s), (REL, r), (OBJ, o)): loss = loss + F.cross_entropy(torch.real(hop(f, role + NR) @ Er.conj().t()) * tau, tgt)          # read the parts back
    S = torch.real(key(Er[s], Er[r]) @ f.conj().t()) * tau; loss = loss + F.cross_entropy(S, torch.arange(B, device=dev))                              # find the fact by its key among others
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0 or step == 1: print(f"memory step {step}/{a.steps}  loss {loss.item():.4f}  lr 0.003  tau {tau.item():.1f}  elapsed {time.time()-t0:.0f}s", flush=True)
model.eval()
# ------------------------------------------------------------------ stories with FRESH names (random places, never trained), as batched working memory
class Stories:
    """B stories: names (B, NMAX, M), facts (B, FMAX, M) with masks; queries: (a, b, relation) and the answer."""
    def __init__(self, B, max_links, n_extra=2, n_knows=2, lengths=None):
        Er = E(); self.B = B; self.names = cnorm(torch.randn(B, 14, M, dtype=torch.complex64, device=dev)); self.nmask = torch.zeros(B, 14, dtype=torch.bool, device=dev)
        facts, self.fmask, self.q, self.fs = [], torch.zeros(B, 16, dtype=torch.bool, device=dev), [], []; self.facts = torch.zeros(B, 16, M, dtype=torch.complex64, device=dev)
        for b in range(B):
            L = int(lengths[b] if lengths is not None else rng.integers(1, max_links + 1)); chain = list(range(L + 1)); extra = list(range(L + 1, L + 1 + n_extra)); n = L + 1 + n_extra; self.nmask[b, :n] = True
            fs = [(chain[i], "bigger", chain[i + 1]) for i in range(L)]
            for _ in range(n_knows): x, y = rng.choice(n, 2, replace=False); fs.append((int(x), "knows", int(y)))
            rng.shuffle(fs); self.fs.append(fs)
            for j, (s, r, o) in enumerate(fs): self.facts[b, j] = fact(self.names[b, s][None], Er[nid[r]][None], self.names[b, o][None])[0]; self.fmask[b, j] = True
            kind = rng.choice(["yes", "no", "unknown", "knows_direct", "knows_chain"]);
            if kind == "yes": i = int(rng.integers(0, L)); j = int(rng.integers(i + 1, L + 1)); self.q.append((i, j, "bigger", "YES", j - i))
            elif kind == "no": i = int(rng.integers(0, L)); j = int(rng.integers(i + 1, L + 1)); self.q.append((j, i, "bigger", "NO", j - i))
            elif kind == "unknown": self.q.append((int(rng.integers(0, L + 1)), extra[0], "bigger", "UNKNOWN", 0))
            else:
                kf = [(s, o) for s, r, o in fs if r == "knows"]
                if kind == "knows_direct": s, o = kf[0]; self.q.append((s, o, "knows", "YES", 1))
                else:                                                                                         # a 2-step knows path must NOT be followed: UNKNOWN unless a direct fact exists
                    s, o = kf[0]; two = [(o, o2) for s2, o2 in kf if s2 == o]; tgt = two[0][1] if two else extra[-1]
                    self.q.append((s, tgt, "knows", "YES" if (s, tgt) in kf else "UNKNOWN", 2))
    def readf(self, cur, rel):
        """the fact keyed by (cur, rel) in each story: object name index (or -1 if none lights), by lights"""
        Er = E(); k = key(cur, Er[rel][None].expand(self.B, -1)); sc = torch.real((k[:, None, :] * self.facts.conj()).sum(-1)); sc[~self.fmask] = -1e9; best = sc.argmax(1)
        f = self.facts[torch.arange(self.B, device=dev), best]
        # verify the fact by reading it back: its SUBJECT light must be the query name, its RELATION light the query relation (no threshold)
        sb = torch.real((hop(f, SUBJ + NR)[:, None, :] * self.names.conj()).sum(-1)); sb[~self.nmask] = -1e9; subj_ok = torch.real((hop(f, SUBJ + NR) * cur.conj()).sum(-1)) >= sb.max(1).values - 1e-6
        R0 = len(NODES) - len(RELS) - len(SYMS); rl = torch.real(hop(f, REL + NR) @ Er[R0:R0 + len(RELS)].conj().t()); rel_ok = rl.argmax(1) == (rel - R0)
        ob = torch.real((hop(f, OBJ + NR)[:, None, :] * self.names.conj()).sum(-1)); ob[~self.nmask] = -1e9; o = ob.argmax(1)
        return torch.where(subj_ok & rel_ok, o, torch.full_like(o, -1))
theta = None
with torch.no_grad():                                                                                     # calibrate 'no fact here': the key light of (name, relation) pairs that have a fact vs those that do not
    st = Stories(256, 3); Er = E(); pres, absn = [], []
    for b in range(st.B):
        have = {(s_, r_) for s_, r_, o_ in st.fs[b]}
        for nm in range(int(st.nmask[b].sum())):
            for r in RELS:
                sc = torch.real((key(st.names[b, nm][None], Er[nid[r]][None]) * st.facts[b][st.fmask[b]].conj()).sum(-1)).max().item(); (pres if (nm, r) in have else absn).append(sc)
    theta = (np.percentile(pres, 1) + np.percentile(absn, 99)) / 2; Stories.theta = theta
print(f"fact-present threshold {theta:.3f} (present: min {np.min(pres):.2f} mean {np.mean(pres):.2f}; absent: max {np.max(absn):.2f} mean {np.mean(absn):.2f}); {time.time()-t0:.0f}s", flush=True)
# ------------------------------------------------------------------ the machine over stories
K = 6; INS = [f"READF {s},{r}>{t}" for s in range(K) for r in RELS for t in range(K) if s != t] + [f"HASF {s},{r},{t}>{u}" for s in range(K) for r in RELS for t in range(K) for u in range(K) if s != t and u != s and u != t] + [f"SAME {s},{t}>{u}" for s in range(K) for t in range(K) for u in range(K) if s != t and u != s and u != t] \
    + [f"IFYES {v},{y}>{u}" for v in range(K) for y in ("YES", "NO", "UNKNOWN") for u in range(K) if u != v] + [f"SET {u}<{y}" for u in range(K) for y in SYMS] + [f"COPY {s}>{t}" for s in range(K) for t in range(K) if s != t] + [f"ANSWER {u}" for u in range(K)]
PROGRAMS = {}
def instruction_set(): return INS + [f"CALL {n} {s},{t}>{u}" for n in PROGRAMS for s in range(K) for t in range(K) for u in range(K) if s != t and u != s and u != t]
def execute(progs, st, depth=0, max_cycles=12):
    """P programs x B stories: slots hold a name index (>=0) or a symbol (-1 NONE, -2 YES, -3 NO, -4 UNKNOWN, -9 empty)"""
    P, B = len(progs), st.B; N = P * B; slot = torch.full((N, K), -9, dtype=torch.long, device=dev); INS_ = instruction_set(); I = {n: i for i, n in enumerate(INS_)}
    qa = torch.tensor([q[0] for q in st.q] * P, device=dev); qb = torch.tensor([q[1] for q in st.q] * P, device=dev); slot[:, 0] = qa; slot[:, 1] = qb
    SYMC = {"NONE": -1, "YES": -2, "NO": -3, "UNKNOWN": -4}; ans = torch.full((N,), -9, dtype=torch.long, device=dev); bidx = torch.arange(N, device=dev) % B
    maxlen = max(len(p["init"]) for p in progs) + max_cycles * max(max(len(p["body"]) for p in progs), 1) + max(len(p["outro"]) for p in progs) + 1
    code = torch.full((N, maxlen), -1, dtype=torch.long, device=dev); cstart = torch.zeros(N, maxlen, dtype=torch.bool, device=dev); ostart = torch.zeros(N, dtype=torch.long, device=dev); stop_slot = torch.tensor([p["stop"] for p in progs], device=dev).repeat_interleave(B)
    for pi, p in enumerate(progs):
        seq = [I[x] for x in p["init"]]; starts = []
        for c in range(max_cycles if p["body"] else 0): starts.append(len(seq)); seq += [I[x] for x in p["body"]]
        rows = slice(pi * B, (pi + 1) * B); ostart[rows] = len(seq); seq += [I[x] for x in p["outro"]]; code[rows, :len(seq)] = torch.tensor(seq, device=dev); cstart[rows, starts] = True
    loop_done = torch.zeros(N, dtype=torch.bool, device=dev); ncyc = torch.zeros(N, dtype=torch.long, device=dev); pc = torch.zeros(N, dtype=torch.long, device=dev)
    ar = torch.arange(N, device=dev)
    for t in range(maxlen + 2):
        pcc = pc.clamp(max=maxlen - 1); ins = torch.where(pc < maxlen, code[ar, pcc], torch.full_like(pc, -1)); active = ins >= 0
        if not active.any(): break
        cs = cstart[ar, pcc] & active & ~loop_done
        if cs.any():
            term = (slot[torch.arange(N, device=dev), stop_slot] == -1) | (slot[torch.arange(N, device=dev), stop_slot] == -2); fin = cs & ((ncyc > 0) & term | (ncyc >= max_cycles - 1))
            loop_done = loop_done | fin; pc = torch.where(fin, ostart, pc); ncyc = ncyc + (cs & ~fin).long(); pcc = pc.clamp(max=maxlen - 1); ins = torch.where(pc < maxlen, code[ar, pcc], torch.full_like(pc, -1)); active = ins >= 0
        for iid in torch.unique(ins[active]).tolist():
            name = INS_[iid]; mk = active & (ins == iid); op = name.split()[0]; idx = mk.nonzero().flatten()
            if op == "READF":
                s_, rest = name[6:].split(","); r_, t_ = rest.split(">"); s_, t_ = int(s_), int(t_); ok = idx[slot[idx, s_] >= 0]
                if len(ok):
                    sub = Stories.__new__(Stories); sub.B = len(ok); sub.names = st.names[bidx[ok]]; sub.nmask = st.nmask[bidx[ok]]; sub.facts = st.facts[bidx[ok]]; sub.fmask = st.fmask[bidx[ok]]
                    o = sub.readf(st.names[bidx[ok], slot[ok, s_]], nid[r_]); slot[ok, t_] = o
                slot[idx[slot[idx, s_] < 0], t_] = -1
            elif op == "HASF":                                                                          # is the fact (s r t) in the story? the whole fact lit against the story's facts
                s_, r_, rest = name[5:].split(","); t_, u_ = rest.split(">"); s_, t_, u_ = int(s_), int(t_), int(u_); ok = idx[(slot[idx, s_] >= 0) & (slot[idx, t_] >= 0)]
                if len(ok):
                    Er = E(); f = fact(st.names[bidx[ok], slot[ok, s_]], Er[nid[r_]][None].expand(len(ok), -1), st.names[bidx[ok], slot[ok, t_]]); sc = torch.real((f[:, None, :] * st.facts[bidx[ok]].conj()).sum(-1)); sc[~st.fmask[bidx[ok]]] = -1e9
                    hit = sc.max(1).values > 0.9; slot[ok[hit], u_] = -2
            elif op == "SAME":
                s_, rest = name[5:].split(","); t_, u_ = rest.split(">"); s_, t_, u_ = int(s_), int(t_), int(u_); same = (slot[idx, s_] >= 0) & (slot[idx, s_] == slot[idx, t_]); slot[idx[same], u_] = -2
            elif op == "IFYES":
                v_, rest = name[6:].split(","); y_, u_ = rest.split(">"); v_, u_ = int(v_), int(u_); hit = slot[idx, v_] == -2; slot[idx[hit], u_] = SYMC[y_]
            elif op == "SET": u_, y_ = name[4:].split("<"); slot[idx, int(u_)] = SYMC[y_]
            elif op == "COPY": s_, t_ = name[5:].split(">"); slot[idx, int(t_)] = slot[idx, int(s_)]
            elif op == "ANSWER": ans[idx] = slot[idx, int(name[7:])]
            elif op == "CALL" and depth < 2:
                nm, rest = name[5:].split(" ", 1); s_, rest = rest.split(","); t_, u_ = rest.split(">"); s_, t_, u_ = int(s_), int(t_), int(u_); ok = idx[(slot[idx, s_] >= 0) & (slot[idx, t_] >= 0)]
                if len(ok):
                    sub = Stories.__new__(Stories); sub.B = len(ok); sub.names = st.names[bidx[ok]]; sub.nmask = st.nmask[bidx[ok]]; sub.facts = st.facts[bidx[ok]]; sub.fmask = st.fmask[bidx[ok]]; sub.q = [(int(slot[i, s_]), int(slot[i, t_]), None, None, 0) for i in ok.tolist()]
                    r = execute([PROGRAMS[nm]], sub, depth + 1); slot[ok, u_] = r
        pc = pc + 1
    return ans
def score(progs, st):
    ans = execute(progs, st).reshape(len(progs), st.B); truth = torch.tensor([{"YES": -2, "NO": -3, "UNKNOWN": -4}[q[3]] for q in st.q], device=dev)
    return (ans == truth[None]).float().mean(1).cpu().numpy()
# ------------------------------------------------------------------ ground truth (only to check the machine and to produce worked examples)
CHAIN_B = {"init": ["SET 3<NONE", "SET 4<NONE"], "body": ["READF 0,bigger>3", "SAME 3,1>4", "COPY 3>0"], "outro": ["ANSWER 4"], "stop": 4}      # stop when slot 4 is YES; slot 3 NONE also ends (READF gives NONE -> COPY makes slot 0 NONE -> READF NONE)
CHAIN_B = {"init": ["SET 4<NONE"], "body": ["READF 0,bigger>3", "SAME 3,1>4", "COPY 3>0"], "outro": ["ANSWER 4"], "stop": 3}
DECIDE_B = {"init": ["SET 5<UNKNOWN", "CALL chain_bigger 0,1>3", "IFYES 3,YES>5", "CALL chain_bigger 1,0>4", "IFYES 4,NO>5"], "body": [], "outro": ["ANSWER 5"], "stop": 2}
DECIDE_K = {"init": ["SET 5<UNKNOWN", "HASF 0,knows,1>4", "IFYES 4,YES>5"], "body": [], "outro": ["ANSWER 5"], "stop": 2}
TRUTH = {"chain_bigger": CHAIN_B, "decide_bigger": DECIDE_B, "decide_knows": DECIDE_K}
def stories_for(name, B, max_links, lengths=None):
    """labels from the story's STRUCTURE: chain nodes are 0..L in order (a > b iff a < b as indices); knows only direct"""
    st = Stories(B, max_links, lengths=lengths); q2 = []
    for b, q in enumerate(st.q):
        L = sum(1 for f in st.fs[b] if f[1] == "bigger"); a_, c_ = q[0], q[1]; inchain = a_ <= L and c_ <= L
        if name == "chain_bigger": lab = "YES" if inchain and a_ < c_ else "NONE"; rel = "bigger"
        elif name == "decide_bigger": lab = ("YES" if a_ < c_ else "NO") if inchain and a_ != c_ else "UNKNOWN"; rel = "bigger"
        else: lab = "YES" if (a_, "knows", c_) in st.fs[b] else "UNKNOWN"; rel = "knows"
        q2.append((a_, c_, rel, lab, abs(c_ - a_) if inchain else 0))
    st.q = q2; return st
def score_named(progs, name, st):
    ans = execute(progs, st).reshape(len(progs), st.B); tmap = {"YES": -2, "NO": -3, "UNKNOWN": -4, "NONE": -1}; truth = torch.tensor([tmap[q[3]] for q in st.q], device=dev)
    return (ans == truth[None]).float().mean(1).cpu().numpy()
res = {"theta": theta}
with torch.no_grad():
    for name, p in TRUTH.items():
        PROGRAMS.update({k: v for k, v in TRUTH.items() if k != name and k in ("chain_bigger",)})
        st = stories_for(name, 200, 3); ex = score_named([p], name, st)[0]; print(f"ground truth {name} on the story machine (train-length stories, fresh names): exact {ex:.3f}", flush=True)
    PROGRAMS.clear()
# ------------------------------------------------------------------ 2. find each program from 3 worked stories; earlier ones callable
def trace_of(p, max_links=3):
    want = {"chain_bigger": ["YES"], "decide_bigger": ["YES", "NO", "UNKNOWN"], "decide_knows": ["YES", "UNKNOWN"]}[name_cur]
    while True:
        st = stories_for(name_cur, 1, max_links)
        if st.q[0][3] in want and (name_cur != "chain_bigger" or st.q[0][4] >= 2): break
    n_links = st.q[0][4] if st.q[0][4] else 2; return list(p["init"]) + [x for _ in range(max(n_links, 1)) for x in p["body"]] + list(p["outro"]), st.q[0]
def mutate(p):
    INS_ = instruction_set(); q = {k: (list(v) if isinstance(v, list) else v) for k, v in p.items()}; part = rng.choice(["body", "init", "init", "outro"]); seq = q[part]; r = rng.random()
    if r < 0.4 and seq: seq[rng.integers(len(seq))] = INS_[rng.integers(len(INS_))]
    elif r < 0.6 and len(seq) < 8: seq.insert(int(rng.integers(len(seq) + 1)), INS_[rng.integers(len(INS_))])
    elif r < 0.75 and seq: seq.pop(int(rng.integers(len(seq))))
    elif r < 0.9 and len(seq) > 1: i, j = rng.integers(len(seq), size=2); seq[i], seq[j] = seq[j], seq[i]
    else: q["stop"] = int(rng.integers(K))
    return q
def show(p): return f"init {p['init']}  body {p['body']}  outro {p['outro']}  stop {p['stop']}"
with torch.no_grad():
    for name_cur, p in TRUTH.items():
        dm = [trace_of(p) for _ in range(3)]; print(f"\n== {name_cur}: worked examples " + "; ".join(f"{q[3]} in {len(tr)} steps" for tr, q in dm) + f"   callable: {list(PROGRAMS)}", flush=True)
        def from_demo():
            tr, q = dm[rng.integers(3)]; blen = int(rng.integers(0, 5)); start = int(rng.integers(0, max(1, len(tr) - blen + 1))); body = tr[start:start + blen]
            init = tr[:start][-int(rng.integers(0, 6)):] if start > 0 and rng.random() < 0.8 else []; outro = [tr[-1]] if rng.random() < 0.8 else []
            return {"init": list(init), "body": list(body), "outro": list(outro), "stop": int(rng.integers(0, K))}
        pop = [from_demo() for _ in range(a.pop)]; found = None; best, best_f = None, -1
        for g in range(1, a.gens + 1):
            st = stories_for(name_cur, 32, 3); f = score_named(pop, name_cur, st); order = np.argsort(-f)
            if f[order[0]] > best_f: best, best_f = pop[order[0]], float(f[order[0]])
            if g % 10 == 0 or g == 1: print(f"  gen {g}  best {f[order[0]]:.3f}  mean {f.mean():.3f}  elapsed {time.time()-t0:.0f}s", flush=True)
            if f[order[0]] >= 1.0:
                cands = [pop[i] for i in order if f[i] >= 1.0]; cands.sort(key=lambda p: len(p["init"]) + len(p["body"]) + len(p["outro"]))
                for cand in cands[:5]:
                    st2 = stories_for(name_cur, 64, 6)                                                    # accept only if exact on LONGER chains than shown, fresh names
                    if score_named([cand], name_cur, st2)[0] >= 1.0: found = cand; break
                if found: print(f"  found at generation {g}: {show(found)}", flush=True); break
            elite = [pop[i] for i in order[:max(2, a.pop // 20)]]; new = list(elite)
            while len(new) < a.pop:
                i, j = order[rng.integers(0, max(4, a.pop // 4), size=2)]; child = mutate(pop[i] if f[i] >= f[j] else pop[j]); new.append(mutate(child) if rng.random() < 0.3 else child)
            pop = new
        prog = found or best; PROGRAMS[name_cur] = prog; v = {}
        for L in range(1, 11):
            st = stories_for(name_cur, 100, L, lengths=[L] * 100); v[L] = float(score_named([prog], name_cur, st)[0])
        print(f"  {name_cur} winner by chain length 1..10 (>3 never shown): " + " ".join(f"{L}:{e:.2f}" for L, e in v.items()), flush=True)
        res[name_cur] = {"found": found is not None, "generation": g, "program": prog, "by_length": v}
    # case-type breakdown for decide_bigger at length 6
    st = stories_for("decide_bigger", 300, 6, lengths=[6] * 300); ans = execute([PROGRAMS["decide_bigger"]], st); tmap = {"YES": -2, "NO": -3, "UNKNOWN": -4}; by = {}
    for q, a_ in zip(st.q, ans.tolist()): by.setdefault(q[3], []).append(a_ == tmap[q[3]])
    res["decide_bigger_by_case_len6"] = {k: float(np.mean(v)) for k, v in by.items()}; print("decide_bigger at 6 links, by case: " + "  ".join(f"{k} {np.mean(v):.2f} (n={len(v)})" for k, v in by.items()), flush=True)
save_json(res, HERE / a.out / "story.json"); torch.save({"model": model.state_dict(), "theta": theta, "programs": PROGRAMS}, HERE / a.out / "story.pt"); print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)
