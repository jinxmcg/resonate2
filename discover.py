"""Discovering programs on the register machine from examples only (no teacher).
Machine (K slots): moves LOAD_X/Y>s, BIND s,t, UNBIND s,t (bind with the conjugate — the memory's inverse);
lights READ s (the 0..19 line), READN s (the -10..9 line, conjugate anchors); decoders SETD t (digit of the last light),
SETC t (its carry / borrow); writes SET t<0, SET t<1, COPY s>t, WRITE s.  A program = init (run once) + body (repeated
once per place) + the slot whose zero, together with 'nothing left', ends the loop.
Fitness on a batch of examples (x, y, target): per-place digit credit (units first) + a bonus for exact; no teacher.
Search: evolutionary — mutate (replace / insert / delete / swap / change stop slot), tournament, elitism; every candidate
of a generation runs as one batched rollout.  --task sub --seed-program add : seeded (subtraction from addition);
--task add unseeded : addition from random programs.  The winner is verified on never-seen 6–9-digit numbers."""
import argparse, time, json, numpy as np, torch, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--path", default="results/grow_1e9_moves/path.pt"); ap.add_argument("--value", default="results/grow_1e9_moves/value.pt")
ap.add_argument("--task", default="add", choices=["add", "sub"]); ap.add_argument("--seed-program", default="", help="add: start the population from mutations of the known addition program")
ap.add_argument("--slots", type=int, default=6); ap.add_argument("--pop", type=int, default=200); ap.add_argument("--gens", type=int, default=300); ap.add_argument("--examples", type=int, default=24); ap.add_argument("--max-digits", type=int, default=3)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results/discover_add"); ap.add_argument("--demos", type=int, default=0, help="seed from a few WORKED EXAMPLES: straight-line traces of the teacher on N small sums (no loop structure given); candidate programs are windows of them")
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time(); NP = 9; K = a.slots
NODES = [str(d) for d in range(10)] + [str(10 ** k) for k in range(1, NP)]; nid = {n: i for i, n in enumerate(NODES)}; PLUS, TIMES = 0, 1; NR = 2
pk = torch.load(HERE / a.path, map_location=dev, weights_only=False); pcfg = pk["config"]; pmodel = ResonatE(len(NODES), 2 * NR, k=pcfg["k"], block=True, block_size=pcfg["block"]).to(dev); pmodel.load_state_dict(pk["model"]); pmodel.eval(); pmodel.requires_grad_(False)
ANCH = torch.tensor([nid["0"]] + [nid[str(10 ** k)] for k in range(1, NP)], device=dev); Er = cnorm(pmodel.E).detach()
def phop(z, r): return pmodel.hop(z, torch.full((z.shape[0],), r, device=dev, dtype=torch.long))
def plight(z): return torch.real(z @ Er.conj().t())
def digs(ns): return torch.tensor([[int(c) for c in f"{int(n):0{NP}d}"] for n in ns], device=dev)
def path_chain(D):
    B = D.shape[0]; acc = None
    for j in range(NP):
        k = NP - 1 - j; d = D[:, j]; term = Er[d] if k == 0 else phop(cnorm(Er[d] + Er[ANCH[k]][None].expand(B, -1)), TIMES)
        acc = phop(term, PLUS) if acc is None else phop(cnorm(acc + term), PLUS)
    return acc
vk = torch.load(HERE / a.value, map_location=dev, weights_only=False); phi = vk["phi"]; mv = phi.shape[-1]; p_ = phi.clone(); p_[0] = 0; V = torch.polar(torch.ones_like(p_), p_); ONE = torch.ones(mv, dtype=torch.complex64, device=dev)
LINE = torch.stack([V[s % 10, 0] * (V[s // 10, 1] if s >= 10 else ONE) for s in range(20)])                                  # 0..19
NLINE = torch.stack([(V[1, 1].conj() if s == -10 else V[-s, 0].conj() if s < 0 else V[s, 0]) for s in range(-10, 10)])           # -10..9 (negatives = conjugates; -10 = one ten, negated)
# ------------------------------------------------------------------ instructions
INS = [f"LOAD_X>{s}" for s in range(K)] + [f"LOAD_Y>{s}" for s in range(K)] + [f"BIND {s},{t}" for s in range(K) for t in range(K) if s != t] + [f"UNBIND {s},{t}" for s in range(K) for t in range(K) if s != t] \
    + [f"READ {s}" for s in range(K)] + [f"READN {s}" for s in range(K)] + [f"SETD {s}" for s in range(K)] + [f"SETC {s}" for s in range(K)] + [f"SET {s}<0" for s in range(K)] + [f"SET {s}<1" for s in range(K)] \
    + [f"COPY {s}>{t}" for s in range(K) for t in range(K) if s != t] + [f"WRITE {s}" for s in range(K)]
I = {n: i for i, n in enumerate(INS)}; NI = len(INS); print(f"machine: {K} slots, {NI} instructions", flush=True)
ADD_PROGRAM = {"init": ["SET 2<0"], "body": ["LOAD_X>0", "LOAD_Y>1", "BIND 0,1", "BIND 0,2", "READ 0", "SETD 3", "SETC 2", "WRITE 3"], "stop": 2}
SUB_PROGRAM = {"init": ["SET 2<0"], "body": ["LOAD_X>0", "LOAD_Y>1", "UNBIND 0,1", "UNBIND 0,2", "READN 0", "SETD 3", "SETC 2", "WRITE 3"], "stop": 2}


def run_programs(progs, xs, ys, max_cycles=NP + 1):
    """all programs on all examples as one batch. Returns tapes (P*E, NP) place-0-first and the step count."""
    P, E = len(progs), len(xs); B = P * E; X = [x for _ in progs for x in xs]; Y = [y for _ in progs for y in ys]
    zx, zy = path_chain(digs(X)), path_chain(digs(Y)); px = torch.zeros(B, dtype=torch.long, device=dev); py = torch.zeros(B, dtype=torch.long, device=dev)
    vec = torch.zeros(B, K, mv, dtype=torch.complex64, device=dev); sym = torch.full((B, K), -1, dtype=torch.long, device=dev); has = torch.zeros(B, K, dtype=torch.bool, device=dev)
    tape = torch.zeros(B, NP, dtype=torch.long, device=dev); wp = torch.zeros(B, dtype=torch.long, device=dev); done = torch.zeros(B, dtype=torch.bool, device=dev); last = torch.zeros(B, dtype=torch.long, device=dev)
    # per-episode instruction streams: init then body repeated; stop check at each body start
    L_init = [len(p["init"]) for p in progs]; L_body = [len(p["body"]) for p in progs]; maxlen = max(L_init) + max_cycles * max(L_body) + 1
    code = torch.full((B, maxlen), -1, dtype=torch.long, device=dev); cycle_start = torch.zeros(B, maxlen, dtype=torch.bool, device=dev); stop_slot = torch.tensor([p["stop"] for p in progs], device=dev).repeat_interleave(E)
    for pi, p in enumerate(progs):
        seq = [I[x] for x in p["init"]]; starts = []
        for c in range(max_cycles): starts.append(len(seq)); seq += [I[x] for x in p["body"]]
        rows = slice(pi * E, (pi + 1) * E); code[rows, :len(seq)] = torch.tensor(seq, device=dev); cycle_start[rows, starts] = True
    def walk(z, ptr, mask):
        rr = phop(z, PLUS + NR); L = torch.where((ptr == 0)[:, None], plight(rr), plight(phop(rr, TIMES + NR)))[:, :10]; return torch.where(mask[:, None], rr, z), L.argmax(1)
    def left(z, ptr):
        out = torch.ones(B, device=dev); rr = z
        for k in range(NP):
            rr = phop(rr, PLUS + NR); nz = (plight(phop(rr, TIMES + NR))[:, :10].argmax(1) != 0) & (ptr + k < NP); out = torch.where(nz, torch.zeros_like(out), out)
        return out
    ar = torch.arange(B, device=dev)
    for t in range(maxlen):
        ins = code[:, t]; active = ~done & (ins >= 0)
        if not active.any(): break
        if cycle_start[:, t].any():                                                                  # stop rule: nothing left on both sides and the stop slot reads 0 (or too many cycles)
            cs = cycle_start[:, t] & active; lx, ly = left(zx, px), left(zy, py); s0 = sym[ar, stop_slot]
            fin = cs & (px > 0) & (lx > 0.5) & (ly > 0.5) & (s0 == 0) | (cs & (px >= NP)); done = done | fin; active = active & ~fin
        for s in range(K):
            mk = active & (ins == I[f"LOAD_X>{s}"])
            if mk.any(): zx, d = walk(zx, px, mk); vec[mk, s] = V[d[mk], 0]; sym[mk, s] = d[mk]; has[mk, s] = True; px = px + mk.long()
            mk = active & (ins == I[f"LOAD_Y>{s}"])
            if mk.any(): zy, d = walk(zy, py, mk); vec[mk, s] = V[d[mk], 0]; sym[mk, s] = d[mk]; has[mk, s] = True; py = py + mk.long()
            for tt in range(K):
                if s == tt: continue
                mk = active & (ins == I[f"BIND {s},{tt}"])
                if mk.any(): vec[mk, s] = vec[mk, s] * vec[mk, tt]; sym[mk, s] = -1
                mk = active & (ins == I[f"UNBIND {s},{tt}"])
                if mk.any(): vec[mk, s] = vec[mk, s] * vec[mk, tt].conj(); sym[mk, s] = -1
                mk = active & (ins == I[f"COPY {s}>{tt}"])
                if mk.any(): vec[mk, tt] = vec[mk, s]; sym[mk, tt] = sym[mk, s]; has[mk, tt] = has[mk, s]
            mk = active & (ins == I[f"READ {s}"])
            if mk.any(): last = torch.where(mk, torch.real((vec[:, s][:, None, :] * LINE[None].conj()).mean(-1)).argmax(1), last)                 # value in 0..19
            mk = active & (ins == I[f"READN {s}"])
            if mk.any(): last = torch.where(mk, torch.real((vec[:, s][:, None, :] * NLINE[None].conj()).mean(-1)).argmax(1) - 10, last)          # value in -10..9
            mk = active & (ins == I[f"SETD {s}"])
            if mk.any(): d = last % 10; vec[mk, s] = V[d[mk], 0]; sym[mk, s] = d[mk]; has[mk, s] = True
            mk = active & (ins == I[f"SETC {s}"])
            if mk.any(): c = torch.where(last < 0, torch.ones_like(last), last // 10); vec[mk, s] = V[c[mk], 0]; sym[mk, s] = c[mk]; has[mk, s] = True
            for cval in (0, 1):
                mk = active & (ins == I[f"SET {s}<{cval}"])
                if mk.any(): vec[mk, s] = V[cval, 0]; sym[mk, s] = cval; has[mk, s] = True
            mk = active & (ins == I[f"WRITE {s}"])
            if mk.any(): d = sym[:, s].clamp(min=0, max=9); tape[mk, wp[mk].clamp(max=NP - 1)] = d[mk]; wp = wp + mk.long()
    return tape


def fitness(progs, xs, ys, targets):
    tape = run_programs(progs, xs, ys); P, E = len(progs), len(xs); T = digs(targets).flip(1).repeat(P, 1)                                     # place 0 first
    nplaces = torch.tensor([max(len(str(t)), 1) for t in targets], device=dev).repeat(P); place = torch.arange(NP, device=dev)[None]
    credit = ((tape == T) & (place < nplaces[:, None])).float().sum(1) / nplaces.float(); exact = (tape == T).all(1).float()
    f = (credit + exact).reshape(P, E).mean(1); return f.cpu().numpy(), exact.reshape(P, E).mean(1).cpu().numpy()


def sample(n, max_digits, task):
    xs, ys, ts = [], [], []
    for _ in range(n):
        L = int(rng.integers(1, max_digits + 1)); s = int(rng.integers(10 ** (L - 1), 10 ** L)); x = int(rng.integers(0, s + 1))
        if task == "add": xs.append(x); ys.append(s - x); ts.append(s)
        else: xs.append(s); ys.append(x); ts.append(s - x)
    return xs, ys, ts


def random_program(): return {"init": [INS[i] for i in rng.integers(0, NI, rng.integers(0, 3))], "body": [INS[i] for i in rng.integers(0, NI, rng.integers(3, 11))], "stop": int(rng.integers(0, K))}
def mutate(p):
    q = {"init": list(p["init"]), "body": list(p["body"]), "stop": p["stop"]}; part = "body" if rng.random() < 0.85 else "init"; seq = q[part]; r = rng.random()
    if r < 0.4 and seq: seq[rng.integers(len(seq))] = INS[rng.integers(NI)]
    elif r < 0.6 and len(seq) < 12: seq.insert(int(rng.integers(len(seq) + 1)), INS[rng.integers(NI)])
    elif r < 0.75 and len(seq) > (1 if part == "body" else 0): seq.pop(int(rng.integers(len(seq))))
    elif r < 0.9 and len(seq) > 1: i, j = rng.integers(len(seq), size=2); seq[i], seq[j] = seq[j], seq[i]
    else: q["stop"] = int(rng.integers(K))
    if not q["body"]: q["body"] = [INS[rng.integers(NI)]]
    return q
def show(p): return f"init [{', '.join(p['init'])}]  body [{', '.join(p['body'])}]  stop-slot {p['stop']}"


# ------------------------------------------------------------------ ground truth check of the hand-written programs (never used by the search)
xs, ys, ts = sample(64, a.max_digits, a.task); f_ref, e_ref = fitness([ADD_PROGRAM if a.task == "add" else SUB_PROGRAM], xs, ys, ts)
print(f"hand-written {a.task} program on the machine (ground truth only): fitness {f_ref[0]:.3f} exact {e_ref[0]:.3f}", flush=True)
# ------------------------------------------------------------------ search
def demo_traces(n):
    """what a child is shown: the steps of n worked examples, as flat instruction lists (init + as many cycles as the sum needed)"""
    prog = ADD_PROGRAM if a.task == "add" else SUB_PROGRAM; out = []
    for _ in range(n):
        L = int(rng.integers(2, a.max_digits + 1)); s = int(rng.integers(10 ** (L - 1), 10 ** L)); x = int(rng.integers(0, s + 1)); ys_ = s - x if a.task == "add" else x; xs_ = x if a.task == "add" else s
        cycles = len(str(s)); out.append((xs_, ys_, list(prog["init"]) + [ins for _ in range(cycles) for ins in prog["body"]]))
    return out
def from_demos(demos):
    """a candidate = a window of a demo trace as the body, an earlier window as init, a random stop slot"""
    x, y, tr = demos[rng.integers(len(demos))]; blen = int(rng.integers(3, 13)); start = int(rng.integers(0, max(1, len(tr) - blen + 1)))
    body = tr[start:start + blen]; init = tr[:start][-int(rng.integers(0, 3)):] if start > 0 and rng.random() < 0.7 else []
    return {"init": list(init), "body": list(body), "stop": int(rng.integers(0, K))}
if a.demos:
    demos = demo_traces(a.demos); print(f"worked examples shown: " + "; ".join(f"{x}{'+' if a.task=='add' else '-'}{y} ({len(tr)} steps)" for x, y, tr in demos), flush=True)
    pop = [from_demos(demos) for _ in range(a.pop)]
else: pop = [mutate(mutate(ADD_PROGRAM)) if a.seed_program == "add" else random_program() for _ in range(a.pop)]
if a.seed_program == "add": pop[0] = {"init": list(ADD_PROGRAM["init"]), "body": list(ADD_PROGRAM["body"]), "stop": ADD_PROGRAM["stop"]}
best, best_f, best_e, history = None, -1, 0, []; (HERE / a.out).mkdir(parents=True, exist_ok=True)
for g in range(1, a.gens + 1):
    xs, ys, ts = sample(a.examples, a.max_digits, a.task); f, e = fitness(pop, xs, ys, ts); order = np.argsort(-f)
    if f[order[0]] > best_f or (f[order[0]] == best_f and len(pop[order[0]]["body"]) < len(best["body"])): best, best_f, best_e = pop[order[0]], float(f[order[0]]), float(e[order[0]])
    history.append({"gen": g, "best_f": float(f[order[0]]), "best_exact": float(e[order[0]]), "mean_f": float(f.mean())})
    if g % 10 == 0 or g == 1:
        eta = (time.time() - t0) / g * (a.gens - g); print(f"gen {g}/{a.gens}  best fitness {f[order[0]]:.3f} (exact {e[order[0]]:.3f})  mean {f.mean():.3f}  best-so-far {best_f:.3f}  elapsed {time.time()-t0:.0f}s  ETA {eta:.0f}s\n   best: {show(pop[order[0]])}", flush=True)
        json.dump({"best": best, "best_f": best_f, "history": history}, open(HERE / a.out / "search.json", "w"), indent=1)
    if best_e >= 1.0 and f[order[0]] >= 2.0 and g > 5:
        # confirm on a fresh, larger batch before declaring
        xs, ys, ts = sample(128, a.max_digits, a.task); fc, ec = fitness([best], xs, ys, ts)
        if ec[0] >= 1.0: print(f"found at generation {g}: {show(best)}", flush=True); break
    elite = [pop[i] for i in order[:max(2, a.pop // 20)]]; new = list(elite)
    while len(new) < a.pop:
        i, j = order[rng.integers(0, max(4, a.pop // 4), size=2)]; parent = pop[i] if f[i] >= f[j] else pop[j]; child = mutate(parent)
        if rng.random() < 0.3: child = mutate(child)
        new.append(child)
    pop = new
# ------------------------------------------------------------------ verification of the winner on never-seen lengths
res = {"task": a.task, "seeded": bool(a.seed_program), "best": best, "best_fitness_train": best_f, "generations": g, "history": history}
with torch.no_grad():
    for L in range(1, NP + 1):
        xs, ys, ts = sample(300, L, a.task); keep = [i for i in range(len(ts)) if len(str(ts[i])) == L or L == 1]
        if not keep: continue
        xs, ys, ts = [xs[i] for i in keep], [ys[i] for i in keep], [ts[i] for i in keep]; f_, e_ = fitness([best], xs, ys, ts); res[f"verify_len{L}"] = {"n": len(xs), "exact": float(e_[0])}
        print(f"winner on {L}-digit {a.task}{' (never seen in the search)' if L > a.max_digits else ''}: n={len(xs):3d}  exact {e_[0]:.3f}", flush=True)
print(f"winner: {show(best)}", flush=True); save_json(res, HERE / a.out / "discover.json"); print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)
