"""CLUTRR (structured, chain-shaped stories): kinship inference over a story memory, the program found from three worked
examples, the composition table learned from training stories as facts in the same memory.
memory : as story.py — a fact = subject(x)SUBJECT + relation(x)RELATION + object(x)OBJECT; roles trained once on random triples.
         The composition table (r1, r2 -> r3), learned from the length-2/3 TRAINING stories (117 entries, any bracketing
         closure), is stored as facts too: (r1, r2, r3) with the same three roles, read by the same keyed read.
machine: NEXT s>t,u (the story fact whose subject is in s: its object into t, its relation into u — verified by read-back),
         COMPOSE u,v>w (the table fact keyed by (u, v): its third part into w; NONE if the pair was never learned),
         SAME s,t>x (YES if names match), ANSWER, CALL.  Loop stops when the stop slot reads YES / NONE.
program to FIND (3 worked stories of 2–3 links): walk the chain from the query's first name to its last, composing the
         relations; answer the composed relation (or NONE = abstain when the table does not cover a step).
test   : the 713 chain-shaped test stories (chains 2–10; training chains were 2–3), exact / abstain / wrong by length."""
import argparse, time, json, numpy as np, torch, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=12); ap.add_argument("--block", type=int, default=16); ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--pop", type=int, default=150); ap.add_argument("--gens", type=int, default=100); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results/clutrr")
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time()
D = json.load(open(str(DATA / "clutrr_structured.json"))); RELS = D["rels"]; TABLE = D["table"]; SYMS = ["NONE", "YES"]
NTRAIN = 120; NODES = [f"n{i}" for i in range(NTRAIN)] + RELS + SYMS; nid = {n: i for i, n in enumerate(NODES)}; R0 = NTRAIN; NR = 3; SUBJ, REL, OBJ = 0, 1, 2
model = ResonatE(len(NODES), 2 * NR, k=a.k, block=True, block_size=a.block).to(dev); M = model.m
def E(): return cnorm(model.E)
def hop(z, r): return model.hop(z, torch.full((z.shape[0],), r, device=dev, dtype=torch.long))
def fact(s, r, o): return cnorm(hop(s, SUBJ) + hop(r, REL) + hop(o, OBJ))
# ------------------------------------------------------------------ 1. the memory learns what a fact is (random triples; subjects/objects fresh random places or relation rows)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
for step in range(1, a.steps + 1):
    Er = E(); B = 256; tau = model.log_tau.exp(); r = torch.tensor(rng.integers(R0, R0 + len(RELS), B), device=dev)
    if step % 2 == 0: S = cnorm(torch.randn(B, M, dtype=torch.complex64, device=dev)); O = cnorm(torch.randn(B, M, dtype=torch.complex64, device=dev)); names = torch.cat([S, O]); tS, tO = torch.arange(B, device=dev), torch.arange(B, 2 * B, device=dev)
    else: s_ = torch.tensor(rng.integers(R0, R0 + len(RELS), B), device=dev); o_ = torch.tensor(rng.integers(R0, R0 + len(RELS), B), device=dev); S, O = Er[s_], Er[o_]; names = Er; tS, tO = s_, o_    # table facts: relations in all three roles
    f = fact(S, Er[r], O); loss = F.cross_entropy(torch.real(hop(f, SUBJ + NR) @ names.conj().t()) * tau, tS) + F.cross_entropy(torch.real(hop(f, OBJ + NR) @ names.conj().t()) * tau, tO) + F.cross_entropy(torch.real(hop(f, REL + NR) @ Er.conj().t()) * tau, r)
    loss = loss + F.cross_entropy(torch.real(cnorm(hop(S, SUBJ)) @ f.conj().t()) * tau, torch.arange(B, device=dev)) + F.cross_entropy(torch.real(cnorm(hop(S, SUBJ) + hop(Er[r], REL)) @ f.conj().t()) * tau, torch.arange(B, device=dev))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0 or step == 1: print(f"memory step {step}/{a.steps}  loss {loss.item():.4f}  lr 0.003  tau {tau.item():.1f}  elapsed {time.time()-t0:.0f}s", flush=True)
model.eval()
with torch.no_grad():
    Er = E(); TAB = torch.stack([fact(Er[nid[x]][None], Er[nid[y]][None], Er[nid[z]][None])[0] for x, y, z in TABLE])            # the composition table as facts in memory
class Stories:
    def __init__(self, items):
        Er = E(); self.B = len(items); NMAX = 12; FMAX = 12; self.names = cnorm(torch.randn(self.B, NMAX, M, dtype=torch.complex64, device=dev)); self.nmask = torch.zeros(self.B, NMAX, dtype=torch.bool, device=dev)
        self.facts = torch.zeros(self.B, FMAX, M, dtype=torch.complex64, device=dev); self.fmask = torch.zeros(self.B, FMAX, dtype=torch.bool, device=dev); self.q = []
        for b, it in enumerate(items):
            L = it["L"]; self.nmask[b, :L + 1] = True
            for i, r in enumerate(it["edges"]): self.facts[b, i] = fact(self.names[b, i][None], Er[nid[r]][None], self.names[b, i + 1][None])[0]; self.fmask[b, i] = True
            self.q.append((0, L, it["target"], L))
    def nxt(self, idx, cur):
        """the story fact whose subject is cur (verified): object name index, relation row index (or -1, -1)"""
        Er = E(); k = cnorm(hop(cur, SUBJ)); F_ = self.facts[idx]; sc = torch.real((k[:, None, :] * F_.conj()).sum(-1)); sc[~self.fmask[idx]] = -1e9; best = sc.argmax(1); f = F_[torch.arange(len(idx), device=dev), best]
        sb = torch.real((hop(f, SUBJ + NR)[:, None, :] * self.names[idx].conj()).sum(-1)); sb[~self.nmask[idx]] = -1e9; subj_ok = torch.real((hop(f, SUBJ + NR) * cur.conj()).sum(-1)) >= sb.max(1).values - 1e-6
        ob = torch.real((hop(f, OBJ + NR)[:, None, :] * self.names[idx].conj()).sum(-1)); ob[~self.nmask[idx]] = -1e9; o = ob.argmax(1); rl = torch.real(hop(f, REL + NR) @ Er[R0:R0 + len(RELS)].conj().t()).argmax(1) + R0
        return torch.where(subj_ok, o, torch.full_like(o, -1)), torch.where(subj_ok, rl, torch.full_like(rl, -1))
def compose(r1, r2):
    """table fact keyed by (r1, r2): its third part (relation row index) or -1"""
    Er = E(); k = cnorm(hop(Er[r1], SUBJ) + hop(Er[r2], REL)); sc = torch.real(k @ TAB.conj().t()); best = sc.argmax(1); f = TAB[best]
    s_ok = torch.real(hop(f, SUBJ + NR) @ Er[R0:R0 + len(RELS)].conj().t()).argmax(1) + R0 == r1; r_ok = torch.real(hop(f, REL + NR) @ Er[R0:R0 + len(RELS)].conj().t()).argmax(1) + R0 == r2
    out = torch.real(hop(f, OBJ + NR) @ Er[R0:R0 + len(RELS)].conj().t()).argmax(1) + R0; return torch.where(s_ok & r_ok, out, torch.full_like(out, -1))
# ------------------------------------------------------------------ the machine: slots hold a name index (>=0, tagged by +1000 offset? no: names 0..11, relations >= R0, symbols -1 NONE, -2 YES, -9 empty)
K = 6; INS = [f"NEXT {s}>{t},{u}" for s in range(K) for t in range(K) for u in range(K) if u != s and u != t] + [f"COMPOSE {u},{v}>{w}" for u in range(K) for v in range(K) for w in range(K) if u != v] \
    + [f"SAME {s},{t}>{x}" for s in range(K) for t in range(K) for x in range(K) if len({s, t, x}) == 3] + [f"COPY {s}>{t}" for s in range(K) for t in range(K) if s != t] + [f"ANSWER {u}" for u in range(K)]
I = {n: i for i, n in enumerate(INS)}
def execute(progs, st, max_cycles=12):
    P, B = len(progs), st.B; N = P * B; slot = torch.full((N, K), -9, dtype=torch.long, device=dev); ar = torch.arange(N, device=dev); bidx = ar % B
    slot[:, 0] = torch.tensor([q[0] for q in st.q] * P, device=dev); slot[:, 1] = torch.tensor([q[1] for q in st.q] * P, device=dev); ans = torch.full((N,), -9, dtype=torch.long, device=dev)
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
            v = slot[ar, stop_slot]; fin = cs & ((ncyc > 0) & ((v == -1) | (v == -2)) | (ncyc >= max_cycles - 1)); loop_done = loop_done | fin; pc = torch.where(fin, ostart, pc); ncyc = ncyc + (cs & ~fin).long()
            pcc = pc.clamp(max=maxlen - 1); ins = torch.where(pc < maxlen, code[ar, pcc], torch.full_like(pc, -1)); active = ins >= 0
        for iid in torch.unique(ins[active]).tolist():
            name = INS[iid]; mk = active & (ins == iid); idx = mk.nonzero().flatten(); op = name.split()[0]
            if op == "NEXT":
                s_, rest = name[5:].split(">"); t_, u_ = rest.split(","); s_, t_, u_ = int(s_), int(t_), int(u_); ok = idx[(slot[idx, s_] >= 0) & (slot[idx, s_] < 100)]
                if len(ok): o, r = st.nxt(bidx[ok], st.names[bidx[ok], slot[ok, s_]]); slot[ok, t_] = o; slot[ok, u_] = r
                bad = idx[~((slot[idx, s_] >= 0) & (slot[idx, s_] < 100))]; slot[bad, t_] = -1; slot[bad, u_] = -1
            elif op == "COMPOSE":
                u_, rest = name[8:].split(","); v_, w_ = rest.split(">"); u_, v_, w_ = int(u_), int(v_), int(w_); ok = idx[(slot[idx, u_] >= R0) & (slot[idx, v_] >= R0)]
                if len(ok): slot[ok, w_] = compose(slot[ok, u_], slot[ok, v_])
                first = idx[(slot[idx, u_] == -9) & (slot[idx, v_] >= R0)]; slot[first, w_] = slot[first, v_]                 # composing with 'nothing yet' (empty) = the relation itself
                bad = idx[(slot[idx, v_] < R0) | (slot[idx, u_] == -1)]; slot[bad, w_] = -1                                   # NONE is sticky: an uncovered step means abstain
            elif op == "SAME":
                s_, rest = name[5:].split(","); t_, x_ = rest.split(">"); s_, t_, x_ = int(s_), int(t_), int(x_); same = (slot[idx, s_] >= 0) & (slot[idx, s_] == slot[idx, t_]); slot[idx[same], x_] = -2
            elif op == "COPY": s_, t_ = name[5:].split(">"); slot[idx, int(t_)] = slot[idx, int(s_)]
            elif op == "ANSWER": ans[idx] = slot[idx, int(name[7:])]
        pc = pc + 1
    return ans
def score(progs, st):
    ans = execute(progs, st).reshape(len(progs), st.B); truth = torch.tensor([nid[q[2]] for q in st.q], device=dev); return (ans == truth[None]).float().mean(1).cpu().numpy(), ans
# ground truth (only to check the machine and to produce worked examples)
WALK = {"init": ["COPY 0>2", "NEXT 2>2,4", "COMPOSE 3,4>3", "SAME 2,1>5"], "body": ["NEXT 2>2,4", "COMPOSE 3,4>3", "SAME 2,1>5"], "outro": ["ANSWER 3"], "stop": 5}
def sample(split, n, lengths=None):
    pool = [d for d in D["data"][split] if lengths is None or d["L"] in lengths]; return [pool[i] for i in rng.integers(0, len(pool), n)]
with torch.no_grad():
    st = Stories(sample("train", 300)); f, _ = score([WALK], st); print(f"ground truth walk+compose on training chains (fresh names): exact {f[0]:.3f}", flush=True)
    st = Stories(D["data"]["test"]); f, ans = score([WALK], st); by = {}
    for q, a_ in zip(st.q, ans.flatten().tolist()): by.setdefault(q[3], []).append(("exact" if a_ == nid[q[2]] else "abstain" if a_ == -1 else "wrong"))
    print("ground truth on TEST by length (exact/abstain/wrong): " + "  ".join(f"{L}: {sum(v=='exact' for v in vs)/len(vs):.2f}/{sum(v=='abstain' for v in vs)/len(vs):.2f}/{sum(v=='wrong' for v in vs)/len(vs):.2f} (n={len(vs)})" for L, vs in sorted(by.items())), flush=True)
# ------------------------------------------------------------------ 2. find the program from three worked training stories
def trace(it): return list(WALK["init"]) + [x for _ in range(it["L"] - 1) for x in WALK["body"]] + list(WALK["outro"])
dm = [trace(it) for it in sample("train", 3, lengths=(3,))]; print("worked examples: " + "; ".join(f"{len(tr)} steps" for tr in dm), flush=True)
def from_demo():
    tr = dm[rng.integers(3)]; blen = int(rng.integers(0, 6)); start = int(rng.integers(0, max(1, len(tr) - blen + 1))); body = tr[start:start + blen]; init = tr[:start][-int(rng.integers(0, 6)):] if start > 0 and rng.random() < 0.8 else []
    return {"init": list(init), "body": list(body), "outro": [tr[-1]] if rng.random() < 0.8 else [], "stop": int(rng.integers(0, K))}
def mutate(p):
    q = {k_: (list(v) if isinstance(v, list) else v) for k_, v in p.items()}; part = rng.choice(["body", "body", "init", "init", "outro"]); seq = q[part]; r = rng.random()
    if r < 0.4 and seq: seq[rng.integers(len(seq))] = INS[rng.integers(len(INS))]
    elif r < 0.6 and len(seq) < 8: seq.insert(int(rng.integers(len(seq) + 1)), INS[rng.integers(len(INS))])
    elif r < 0.75 and seq: seq.pop(int(rng.integers(len(seq))))
    elif r < 0.9 and len(seq) > 1: i, j = rng.integers(len(seq), size=2); seq[i], seq[j] = seq[j], seq[i]
    else: q["stop"] = int(rng.integers(K))
    return q
res = {}
with torch.no_grad():
    pop = [from_demo() for _ in range(a.pop)]; found = None; best, best_f = None, -1
    for g in range(1, a.gens + 1):
        st = Stories(sample("train", 32)); f, _ = score(pop, st); order = np.argsort(-f)
        if f[order[0]] > best_f: best, best_f = pop[order[0]], float(f[order[0]])
        if g % 10 == 0 or g == 1: print(f"  gen {g}  best {f[order[0]]:.3f}  mean {f.mean():.3f}  elapsed {time.time()-t0:.0f}s", flush=True)
        if f[order[0]] >= 1.0:
            cands = [pop[i] for i in order if f[i] >= 1.0]; cands.sort(key=lambda p: len(p["init"]) + len(p["body"]) + len(p["outro"]))
            for cand in cands[:5]:
                st2 = Stories(sample("validation", 64, lengths=(3,))); fc, _ = score([cand], st2)                        # accept only if exact on held-out stories (validation) too
                if fc[0] >= 1.0: found = cand; break
            if found: print(f"  found at generation {g}: init {found['init']}  body {found['body']}  outro {found['outro']}  stop {found['stop']}", flush=True); break
        elite = [pop[i] for i in order[:max(2, a.pop // 20)]]; new = list(elite)
        while len(new) < a.pop:
            i, j = order[rng.integers(0, max(4, a.pop // 4), size=2)]; child = mutate(pop[i] if f[i] >= f[j] else pop[j]); new.append(mutate(child) if rng.random() < 0.3 else child)
        pop = new
    prog = found or best; st = Stories(D["data"]["test"]); f, ans = score([prog], st); by = {}
    for q, a_ in zip(st.q, ans.flatten().tolist()): by.setdefault(q[3], []).append("exact" if a_ == nid[q[2]] else "abstain" if a_ == -1 else "wrong")
    res = {"found": found is not None, "generation": g, "program": prog, "test_by_length": {L: {"n": len(vs), "exact": sum(v == "exact" for v in vs) / len(vs), "abstain": sum(v == "abstain" for v in vs) / len(vs), "wrong": sum(v == "wrong" for v in vs) / len(vs)} for L, vs in sorted(by.items())}, "test_exact_all": float(f[0])}
    print("FOUND program on CLUTRR test (chain stories; train chains 2-3) by length — exact/abstain/wrong: " + "  ".join(f"{L}: {v['exact']:.2f}/{v['abstain']:.2f}/{v['wrong']:.2f} (n={v['n']})" for L, v in res["test_by_length"].items()) + f"   overall exact {f[0]:.3f}", flush=True)
save_json(res, HERE / a.out / "clutrr.json"); print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)
