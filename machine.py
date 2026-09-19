"""A general working memory: K anonymous slots and three kinds of primitive — move, light, write.
slots      : each holds a state in the value code (place-0 anchors) with an optional symbol tag; nothing says 'carry' or
             'digit' — the controller decides what goes where. Cleared after every query (a cache).
moves      : LOAD_X->s / LOAD_Y->s  walk that operand one place back, put the digit's state in slot s
             BIND s,t->s            bind two slots in the value code (exact addition of what they hold)
light      : READ s                 the slot's light over the 0..19 line (20 candidates) — the only way anything is 'seen'
             (every slot's symbol also shines on the digit-order template Q, always visible: 'how big')
writes     : SET s<-symbol (digits 0-9, LT, GT, EQ), COPY s->t, WRITE_TAPE s (result digit), ANSWER s, STOP
Operations are programs over these — addition and comparison are the two teachers; nothing task-specific is hardware.
Controller: GRU over lights + slot tags; imitation of the per-state teacher on numbers < 10^4, DAgger, then executed
alone up to 9 digits. Promotion demo: a validated sum (path result == value-code equality) is written to long-term
memory as a labelled row and is then found by one light."""
import argparse, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--path", default="results/grow_1e9_moves/path.pt"); ap.add_argument("--value", default="results/grow_1e9_moves/value.pt"); ap.add_argument("--order", default="results/compare/digit_order.pt")
ap.add_argument("--slots", type=int, default=4); ap.add_argument("--train-max-digits", type=int, default=4); ap.add_argument("--traces", type=int, default=30000); ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--dagger", type=int, default=3); ap.add_argument("--dagger-episodes", type=int, default=6000)
ap.add_argument("--hidden", type=int, default=256); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results/machine")
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
def walk_digits(z):
    B = z.shape[0]; out = torch.zeros(B, NP, dtype=torch.long, device=dev); rr = z
    for k in range(NP):
        rr = phop(rr, PLUS + NR)
        if k == 0: out[:, 0] = plight(rr).argmax(1)
        else: L = plight(phop(rr, TIMES + NR)); L[:, 10:] = -1e9; out[:, k] = L.argmax(1)
    return out
vk = torch.load(HERE / a.value, map_location=dev, weights_only=False); phi = vk["phi"]; mv = phi.shape[-1]; p_ = phi.clone(); p_[0] = 0; V = torch.polar(torch.ones_like(p_), p_)
CAND = torch.stack([V[s % 10, 0] * (V[s // 10, 1] if s >= 10 else torch.ones(mv, dtype=torch.complex64, device=dev)) for s in range(20)])   # the 0..19 line at place 0
Q = torch.load(HERE / a.order, map_location=dev, weights_only=False)["Q"]; SDIG = torch.real(Er[:10] @ Q.conj()); SDIG = (SDIG - SDIG.mean()) / SDIG.std()
ZERO_REM = []
with torch.no_grad():
    rr = path_chain(digs([0]))
    for k in range(NP): rr = phop(rr, PLUS + NR); ZERO_REM.append(rr[0].clone())
# ------------------------------------------------------------------ symbols, actions
SYM = [str(d) for d in range(10)] + ["LT", "GT", "EQ"]; NS = len(SYM); EMPTY = -1; ORDER_OF = torch.cat([SDIG, torch.zeros(3, device=dev)])
ACTS = [f"LOAD_X>{s}" for s in range(K)] + [f"LOAD_Y>{s}" for s in range(K)] + [f"BIND {s},{t}" for s in range(K) for t in range(K) if s != t] + [f"READ {s}" for s in range(K)] \
     + [f"SET {s}<{y}" for s in range(K) for y in SYM] + [f"COPY {s}>{t}" for s in range(K) for t in range(K) if s != t] + [f"WRITE {s}" for s in range(K)] + [f"ANSWER {s}" for s in range(K)] + ["STOP"]
A = {n: i for i, n in enumerate(ACTS)}; NA = len(ACTS); OBS = K * (NS + 2) + 20 + K + 2 + NA + 2                       # slot tags + order lights + has-vec; last READ light + which slot; left x/y; prev action; task
print(f"machine: {K} slots, {NA} actions, {NS} symbols", flush=True)


class Env:
    def __init__(self, xs, ys, task):
        self.B = len(xs); self.task = task; self.zx, self.zy = path_chain(digs(xs)), path_chain(digs(ys)); self.px = torch.zeros(self.B, dtype=torch.long, device=dev); self.py = torch.zeros(self.B, dtype=torch.long, device=dev)
        self.vec = torch.zeros(self.B, K, mv, dtype=torch.complex64, device=dev); self.has = torch.zeros(self.B, K, dtype=torch.bool, device=dev); self.sym = torch.full((self.B, K), EMPTY, dtype=torch.long, device=dev)
        self.tape = torch.zeros(self.B, NP, dtype=torch.long, device=dev); self.wp = torch.zeros(self.B, dtype=torch.long, device=dev); self.answer = torch.full((self.B,), -1, dtype=torch.long, device=dev); self.done = torch.zeros(self.B, dtype=torch.bool, device=dev)
        self.lread = torch.zeros(self.B, 20, device=dev); self.rslot = torch.zeros(self.B, K, device=dev); self.prev = torch.zeros(self.B, NA, device=dev)
    def _walk(self, z, ptr, mask):
        rr = phop(z, PLUS + NR); L = torch.where((ptr == 0)[:, None], plight(rr), plight(phop(rr, TIMES + NR)))[:, :10]; return torch.where(mask[:, None], rr, z), L.argmax(1)
    def _left(self):
        """'anything left?' as a light: peek the remaining places of each operand (walk them, read the digit lights); 1 = all zero"""
        def peek(z, ptr):
            out = torch.ones(self.B, device=dev); rr = z
            for k in range(NP):
                rr = phop(rr, PLUS + NR); L = plight(phop(rr, TIMES + NR))[:, :10]; nz = (L.argmax(1) != 0) & (ptr + k < NP) & (ptr > 0)     # places still ahead of the pointer
                out = torch.where(nz, torch.zeros_like(out), out)
            return torch.where(ptr > 0, out, torch.zeros_like(out))
        return peek(self.zx, self.px), peek(self.zy, self.py)
    def obs(self):
        tags = F.one_hot(self.sym.clamp(min=0), NS).float() * (self.sym >= 0)[:, :, None].float(); order = ORDER_OF[self.sym.clamp(min=0)] * (self.sym >= 0).float()
        lx, ly = self._left()
        return torch.cat([tags.reshape(self.B, -1), order, self.has.float(), self.lread, self.rslot, lx[:, None], ly[:, None], self.prev, F.one_hot(torch.full((self.B,), self.task, device=dev), 2).float()], 1)
    def put(self, mask, s, d):
        """put digit d (tensor) into slot s for masked episodes: value state at place 0 and the symbol tag"""
        self.vec[mask, s] = V[d[mask], 0]; self.has[mask, s] = True; self.sym[mask, s] = d[mask]
    def act(self, act):
        act = torch.where(self.done, torch.full_like(act, A["STOP"]), act); m = lambda name: (act == A[name]) & ~self.done; ar = torch.arange(self.B, device=dev)
        for s in range(K):
            mk = m(f"LOAD_X>{s}")
            if mk.any(): self.zx, d = self._walk(self.zx, self.px, mk); self.put(mk, s, d); self.px = self.px + mk.long()
            mk = m(f"LOAD_Y>{s}")
            if mk.any(): self.zy, d = self._walk(self.zy, self.py, mk); self.put(mk, s, d); self.py = self.py + mk.long()
            for t in range(K):
                if s == t: continue
                mk = m(f"BIND {s},{t}")
                if mk.any(): self.vec[mk, s] = self.vec[mk, s] * self.vec[mk, t]; self.sym[mk, s] = EMPTY                 # a bound state has no symbol until READ
                mk = m(f"COPY {s}>{t}")
                if mk.any(): self.vec[mk, t] = self.vec[mk, s]; self.has[mk, t] = self.has[mk, s]; self.sym[mk, t] = self.sym[mk, s]
            mk = m(f"READ {s}")
            if mk.any(): L = torch.real((self.vec[:, s][:, None, :] * CAND[None].conj()).mean(-1)) * 10; self.lread = torch.where(mk[:, None], L, self.lread); self.rslot = torch.where(mk[:, None], F.one_hot(torch.full((self.B,), s, device=dev), K).float(), self.rslot)
            for y, name in enumerate(SYM):
                mk = m(f"SET {s}<{name}")
                if mk.any():
                    if y < 10: self.vec[mk, s] = V[y, 0]
                    self.has[mk, s] = True; self.sym[mk, s] = y
            mk = m(f"WRITE {s}")
            if mk.any(): d = self.sym[:, s].clamp(min=0, max=9); self.tape[mk, self.wp[mk].clamp(max=NP - 1)] = d[mk]; self.wp = self.wp + mk.long()
            mk = m(f"ANSWER {s}")
            if mk.any(): self.answer = torch.where(mk, self.sym[:, s], self.answer); self.done = self.done | mk
        self.done = self.done | m("STOP"); self.prev = F.one_hot(act, NA).float()
    def result_numbers(self): return [int(sum(int(row[k]) * 10 ** k for k in range(NP))) for row in walk_digits(path_chain(self.tape.flip(1))).cpu().numpy()]


# ------------------------------------------------------------------ the teachers: programs over the machine, as per-state policies (next instruction from what is visible)
def teacher(env, truth_cmp=None):
    B = env.B; prev = torch.where(env.prev.sum(1) == 0, torch.full((B,), -1, dtype=torch.long, device=dev), env.prev.argmax(1)); lx, ly = env._left()
    act = torch.full((B,), A["STOP"], dtype=torch.long, device=dev); s = env.lread.argmax(1)
    if env.task == 0:
        # slots: 0 = x digit / sum, 1 = y digit, 2 = carry, 3 = result digit.  cycle: LOAD_X>0, LOAD_Y>1, BIND 0,1, BIND 0,2, READ 0, SET 3<digit, SET 2<carry, WRITE 3
        fin = (env.px > 0) & (lx > 0.95) & (ly > 0.95) & (env.sym[:, 2] == 0) | (env.px >= NP)
        start = (prev == -1) | (prev == A["WRITE 3"])
        act = torch.where(start & ~env.has[:, 2], torch.full_like(act, A["SET 2<0"]), act)                                  # carry slot must exist first
        act = torch.where(start & env.has[:, 2] & ~fin, torch.full_like(act, A["LOAD_X>0"]), act)
        act = torch.where((prev == A["SET 2<0"]) & ~env.has[:, 3], torch.full_like(act, A["LOAD_X>0"]), act)              # the initial carry: no result digit exists yet
        act = torch.where(prev == A["LOAD_X>0"], torch.full_like(act, A["LOAD_Y>1"]), act); act = torch.where(prev == A["LOAD_Y>1"], torch.full_like(act, A["BIND 0,1"]), act)
        act = torch.where(prev == A["BIND 0,1"], torch.full_like(act, A["BIND 0,2"]), act); act = torch.where(prev == A["BIND 0,2"], torch.full_like(act, A["READ 0"]), act)
        act = torch.where(prev == A["READ 0"], A["SET 3<0"] + s % 10, act); act = torch.where((prev >= A["SET 3<0"]) & (prev <= A["SET 3<9"]), A["SET 2<0"] + s // 10, act)
        act = torch.where(((prev == A["SET 2<0"]) | (prev == A["SET 2<1"])) & env.has[:, 3], torch.full_like(act, A["WRITE 3"]), act)
        act = torch.where(prev == A["WRITE 3"], torch.where(fin, torch.full_like(act, A["STOP"]), torch.full_like(act, A["LOAD_X>0"])), act)
        act = torch.where(start & env.has[:, 2] & fin, torch.full_like(act, A["STOP"]), act)
    else:
        # slots: 0 = x digit, 1 = y digit, 3 = verdict.  cycle: LOAD_X>0, LOAD_Y>1, [SET 3<LT/GT if they differ]; at the end ANSWER 3 (EQ if never set)
        fin = (env.px > 0) & (lx > 0.95) & (ly > 0.95) | (env.px >= NP); d0, d1 = env.sym[:, 0], env.sym[:, 1]
        differ = (prev == A["LOAD_Y>1"]) & (d0 != d1)
        act = torch.where((prev == -1) | (prev == A["SET 3<LT"]) | (prev == A["SET 3<GT"]) | ((prev == A["LOAD_Y>1"]) & ~differ), torch.full_like(act, A["LOAD_X>0"]), act)
        act = torch.where(prev == A["LOAD_X>0"], torch.full_like(act, A["LOAD_Y>1"]), act)
        act = torch.where(differ, torch.where(d0 < d1, torch.full_like(act, A["SET 3<LT"]), torch.full_like(act, A["SET 3<GT"])), act)
        end = fin & ((prev == A["SET 3<LT"]) | (prev == A["SET 3<GT"]) | ((prev == A["LOAD_Y>1"]) & ~differ))
        act = torch.where(end & ~env.has[:, 3], torch.full_like(act, A["SET 3<EQ"]), act); act = torch.where(end & env.has[:, 3], torch.full_like(act, A["ANSWER 3"]), act)
        act = torch.where(prev == A["SET 3<EQ"], torch.full_like(act, A["ANSWER 3"]), act)
    return act


def sample_pairs(n, max_digits, task):
    xs, ys = [], []
    for _ in range(n):
        L = int(rng.integers(1, max_digits + 1)); s = int(rng.integers(10 ** (L - 1), 10 ** L))
        if task == 0: x = int(rng.integers(0, s + 1)); xs.append(x); ys.append(s - x)
        else: y = int(np.clip(s + rng.integers(-10 ** int(rng.integers(0, L)), 10 ** int(rng.integers(0, L)) + 1), 0, 10 ** L - 1)); xs.append(s); ys.append(int(y))
    return xs, ys


class Controller(nn.Module):
    def __init__(self, h): super().__init__(); self.gru = nn.GRUCell(OBS, h); self.out = nn.Linear(h, NA)
    def forward(self, o, hid): hid = self.gru(o, hid); return self.out(hid), hid


ctrl = Controller(a.hidden).to(dev); opt = torch.optim.Adam(ctrl.parameters(), lr=1e-3); (HERE / a.out).mkdir(parents=True, exist_ok=True); ANS = {"EQ": 0, "LT": 1, "GT": 2}
def rollout(n_ep, task, mix):
    xs, ys = sample_pairs(n_ep, a.train_max_digits, task); env = Env(xs, ys, task); hid = torch.zeros(len(xs), a.hidden, device=dev)
    truth = torch.tensor([0 if x == y else (1 if x < y else 2) for x, y in zip(xs, ys)], device=dev); O, T = [], []
    for _ in range(120):
        o = env.obs(); lab = teacher(env, truth); O.append(o); T.append(lab)
        if mix > 0: logits, hid = ctrl(o, hid); act = torch.where(torch.rand(len(xs), device=dev) < mix, logits.argmax(1), lab)
        else: act = lab
        env.act(act)
        if env.done.all(): break
    ans = torch.tensor([ANS.get(SYM[i], -1) if i >= 0 else -1 for i in env.answer.tolist()], device=dev)
    ok = (np.array(env.result_numbers()) == np.array(xs) + np.array(ys)) if task == 0 else (ans == truth).cpu().numpy()
    return torch.stack(O, 1), torch.stack(T, 1), float(ok.mean())
traces = []
with torch.no_grad():
    for task in (0, 1):
        for _ in range(a.traces // 2 // 256): traces.append(rollout(256, task, 0.0))
    print(f"teacher programs on the machine: {len(traces) * 256} episodes, exact on their own runs: add {np.mean([t[2] for t in traces[:len(traces)//2]]):.3f}  cmp {np.mean([t[2] for t in traces[len(traces)//2:]]):.3f}  mean length add {np.mean([t[0].shape[1] for t in traces[:len(traces)//2]]):.0f} cmp {np.mean([t[0].shape[1] for t in traces[len(traces)//2:]]):.0f}  ({time.time()-t0:.0f}s)", flush=True)
def train(epochs, tag):
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(traces)); tot = 0.0; n = 0; acc = 0.0
        for i in order:
            O, T, _ = traces[i]; hid = torch.zeros(O.shape[0], a.hidden, device=dev); loss = 0.0; hits = 0.0
            for t in range(O.shape[1]):
                logits, hid = ctrl(O[:, t], hid); loss = loss + F.cross_entropy(logits, T[:, t]); hits += (logits.argmax(1) == T[:, t]).float().mean().item()
            loss = loss / O.shape[1]; opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); acc += hits / O.shape[1]; n += 1
        if ep % 10 == 0 or ep == epochs: print(f"{tag} epoch {ep}/{epochs}  imitation loss {tot/n:.4f}  step accuracy {acc/n:.4f}  lr 0.001  traces {len(traces)}  elapsed {time.time()-t0:.0f}s", flush=True); torch.save({"ctrl": ctrl.state_dict(), "config": vars(a)}, HERE / a.out / "ckpt.pt")
train(a.epochs, "imitation")
for r in range(1, a.dagger + 1):
    with torch.no_grad():
        for task in (0, 1):
            for _ in range(a.dagger_episodes // 256): traces.append(rollout(256, task, 0.5))
    train(20, f"dagger {r}")
# ------------------------------------------------------------------ execution alone
res = {"config": vars(a), "actions": NA}
with torch.no_grad():
    def run(xs, ys, task, max_steps=120):
        env = Env(xs, ys, task); hid = torch.zeros(len(xs), a.hidden, device=dev)
        for _ in range(max_steps):
            logits, hid = ctrl(env.obs(), hid); env.act(logits.argmax(1))
            if env.done.all(): break
        return env
    for task, name in ((0, "add"), (1, "compare")):
        for L in range(1, NP + 1):
            xs, ys = sample_pairs(3000, L, task); keep = [i for i, (x, y) in enumerate(zip(xs, ys)) if len(str(x + y if task == 0 else max(x, y))) == L][:500]; xs, ys = [xs[i] for i in keep], [ys[i] for i in keep]
            if not xs: continue
            env = run(xs, ys, task)
            if task == 0: ok = np.array(env.result_numbers()) == np.array(xs) + np.array(ys)
            else: ok = np.array([ANS.get(SYM[i], -1) if i >= 0 else -1 for i in env.answer.tolist()]) == np.array([0 if x == y else (1 if x < y else 2) for x, y in zip(xs, ys)])
            res[f"{name}_len{L}"] = {"n": len(xs), "exact": float(ok.mean())}; print(f"{name:8s} {L}-digit{' (length never seen)' if L > a.train_max_digits else ''}: n={len(xs):3d}  exact {ok.mean():.3f}", flush=True)
    # promotion: a validated sum becomes long-term memory.  validation = two routes agree: the path result and the value-code equality light
    xs, ys = sample_pairs(200, 6, 0); env = run(xs, ys, 0); got = env.result_numbers()
    Vfull = lambda ns: torch.stack([torch.prod(torch.stack([V[d, NP - 1 - j] for j, d in enumerate(row)]), 0) for row in digs(ns).tolist()])
    agree = torch.real((Vfull(xs) * Vfull(ys) * Vfull(got).conj()).mean(-1)) > 0.99; correct = np.array(got) == np.array(xs) + np.array(ys)
    res["promotion"] = {"n": len(xs), "validated": float(agree.float().mean()), "validated_and_correct": float((agree.cpu().numpy() & correct).sum() / max(agree.sum().item(), 1)), "correct_but_rejected": float(((~agree.cpu().numpy()) & correct).mean())}
    print(f"promotion check on 200 sums (≤6 digits): validated by two routes agreeing {res['promotion']['validated']:.3f}; of the validated, correct {res['promotion']['validated_and_correct']:.3f}; correct but rejected {res['promotion']['correct_but_rejected']:.3f}", flush=True)
    keep = [i for i in range(len(xs)) if agree[i]][:20]; rows = path_chain(digs([got[i] for i in keep])); labels = [f"{xs[i]}+{ys[i]}" for i in keep]        # written to long-term memory as rows with labels
    probe = path_chain(digs([xs[keep[0]] + ys[keep[0]]])); hit = int(torch.real(probe @ rows.conj().t()).argmax()); print(f"long-term memory now holds {len(keep)} validated sums as labelled rows; a single light from the place of {xs[keep[0]]}+{ys[keep[0]]} finds row '{labels[hit]}'", flush=True)
save_json(res, HERE / a.out / "machine.json"); torch.save({"ctrl": ctrl.state_dict(), "config": vars(a)}, HERE / a.out / "controller.pt"); print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)
