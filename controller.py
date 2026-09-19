"""The reasoning layer, learned: a controller that reads lights and chooses moves.
Environment = today's exact primitives over the 9-place memory (path model + value code of Part 10, digit order of Part 8):
  STEP_X / STEP_Y : walk that operand one place back (units first); returns the 10-digit light of that place
  SUM             : bind the last-read digits of x and y and the current carry in the value code; returns the 20-candidate
                    equality light (digit sums 0..19)
  WRITE d (0-9)   : put d at the current result place and advance;   CARRY c (0/1): set the carry
  STOP            : done (the result path is rebuilt from the written digits and walked back to verify)
  STEP_BOTH / ANSWER_LT / ANSWER_GT / ANSWER_EQ : the comparison task (task token given)
Observation each step: the lights the last primitive returned (10 + 10 + 20), the digit-order lights of the two last
digits (2), 'anything left?' lights for x and y (cosine of the remaining chain to the all-zero remainder), the carry,
the previous action (one-hot), the task token. The controller never sees a number.
Controller: a GRU. Trained by IMITATION of the hand-written programs' traces on numbers < 10^4 only, then executed
on its own; scored by exact outcome on 1-4 digits (seen lengths) and 5-9 digits (never seen). Chance ≈ 0."""
import argparse, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--path", default="results/grow_1e9_moves/path.pt"); ap.add_argument("--value", default="results/grow_1e9_moves/value.pt"); ap.add_argument("--order", default="results/compare/digit_order.pt")
ap.add_argument("--train-max-digits", type=int, default=4); ap.add_argument("--traces", type=int, default=30000); ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--dagger", type=int, default=3); ap.add_argument("--dagger-episodes", type=int, default=6000); ap.add_argument("--hidden", type=int, default=128); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results/controller")
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time(); NP = 9; MAXN = 10 ** NP
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
vk = torch.load(HERE / a.value, map_location=dev, weights_only=False); phi = vk["phi"]; mv = phi.shape[-1]; p_ = phi.clone(); p_[0] = 0; V = torch.polar(torch.ones_like(p_), p_); one = torch.ones(mv, dtype=torch.complex64, device=dev)
CAND = torch.stack([torch.stack([V[s % 10, k] * (V[s // 10, k + 1] if k < NP - 1 and s >= 10 else one) for s in range(20)]) for k in range(NP)])
Q = torch.load(HERE / a.order, map_location=dev, weights_only=False)["Q"]; SD = torch.real(Er[:10] @ Q.conj()); SD = (SD - SD.mean()) / SD.std()
ZERO_REM = []                                                                                    # the remaining chain of the all-zero number after k reverse + hops, for the 'anything left?' light
with torch.no_grad():
    rr = path_chain(digs([0]))
    for k in range(NP): rr = phop(rr, PLUS + NR); ZERO_REM.append(rr[0].clone())
# ------------------------------------------------------------------ actions and the environment (batched over episodes)
ACTS = ["STEP_X", "STEP_Y", "SUM"] + [f"WRITE_{d}" for d in range(10)] + ["CARRY_0", "CARRY_1", "STOP", "STEP_BOTH", "ANSWER_LT", "ANSWER_GT", "ANSWER_EQ"]; A = {n: i for i, n in enumerate(ACTS)}; NA = len(ACTS)
OBS = 10 + 10 + 20 + 2 + 2 + 1 + NA + 2                                                       # lights x, y, sum; order lights; left x, y; carry; prev action; task


class Env:
    def __init__(self, xs, ys, task):
        self.B = len(xs); self.task = task; self.zx, self.zy = path_chain(digs(xs)), path_chain(digs(ys)); self.px = torch.zeros(self.B, dtype=torch.long, device=dev); self.py = torch.zeros(self.B, dtype=torch.long, device=dev)
        self.dx = torch.zeros(self.B, dtype=torch.long, device=dev); self.dy = torch.zeros(self.B, dtype=torch.long, device=dev); self.carry = torch.zeros(self.B, dtype=torch.long, device=dev)
        self.res = torch.zeros(self.B, NP, dtype=torch.long, device=dev); self.wp = torch.zeros(self.B, dtype=torch.long, device=dev); self.answer = torch.full((self.B,), -1, dtype=torch.long, device=dev); self.done = torch.zeros(self.B, dtype=torch.bool, device=dev)
        self.lx = torch.zeros(self.B, 10, device=dev); self.ly = torch.zeros(self.B, 10, device=dev); self.ls = torch.zeros(self.B, 20, device=dev); self.prev = torch.zeros(self.B, NA, device=dev)
    def _step(self, z, ptr, mask):
        """walk one place back for the masked episodes; returns new state, digit light"""
        rr = phop(z, PLUS + NR); L = torch.where((ptr == 0)[:, None], plight(rr), plight(phop(rr, TIMES + NR)))[:, :10]
        z_new = torch.where(mask[:, None], rr, z); return z_new, L, rr
    def obs(self):
        left = lambda z, ptr: torch.stack([torch.real((z[i] * ZERO_REM[min(int(ptr[i]), NP) - 1].conj()).sum()) if ptr[i] > 0 else torch.tensor(0.0, device=dev) for i in range(self.B)])
        lx_, ly_ = left(self.zx, self.px), left(self.zy, self.py)
        return torch.cat([self.lx, self.ly, self.ls, SD[self.dx][:, None], SD[self.dy][:, None], lx_[:, None], ly_[:, None], self.carry[:, None].float(), self.prev, F.one_hot(torch.full((self.B,), self.task, device=dev), 2).float()], 1)
    def act(self, act):
        act = torch.where(self.done, torch.full_like(act, A["STOP"]), act); m = lambda name: (act == A[name]) & ~self.done
        for name, which in (("STEP_X", "x"), ("STEP_Y", "y"), ("STEP_BOTH", "xy")):
            mk = m(name)
            if mk.any():
                if "x" in which: z, L, rr = self._step(self.zx, self.px, mk); self.zx = z; self.lx = torch.where(mk[:, None], L * 10, self.lx); self.dx = torch.where(mk, L.argmax(1), self.dx); self.px = self.px + mk.long()
                if "y" in which: z, L, rr = self._step(self.zy, self.py, mk); self.zy = z; self.ly = torch.where(mk[:, None], L * 10, self.ly); self.dy = torch.where(mk, L.argmax(1), self.dy); self.py = self.py + mk.long()
        mk = m("SUM")
        if mk.any():
            k = (self.wp.clamp(max=NP - 1)); e = V[self.dx, k] * V[self.dy, k] * V[self.carry, k]; L = torch.real((e[:, None, :] * CAND[k].conj()).mean(-1)); self.ls = torch.where(mk[:, None], L * 10, self.ls)
        for d in range(10):
            mk = m(f"WRITE_{d}")
            if mk.any(): self.res[mk, self.wp[mk].clamp(max=NP - 1)] = d; self.wp = self.wp + mk.long()
        for c in (0, 1):
            mk = m(f"CARRY_{c}"); self.carry = torch.where(mk, torch.full_like(self.carry, c), self.carry)
        for name, v in (("ANSWER_LT", 1), ("ANSWER_GT", 2), ("ANSWER_EQ", 0)):
            mk = m(name); self.answer = torch.where(mk, torch.full_like(self.answer, v), self.answer); self.done = self.done | mk
        self.done = self.done | m("STOP"); self.prev = F.one_hot(act, NA).float()
    def result_numbers(self):
        return [int(sum(int(row[k]) * 10 ** k for k in range(NP))) for row in walk_digits(path_chain(self.res.flip(1))).cpu().numpy()]


# ------------------------------------------------------------------ the teacher: the hand-written programs, emitting actions from what they read
def teacher_add(env):
    """per place: STEP_X, STEP_Y, SUM, WRITE (light argmax mod 10), CARRY (div 10); STOP when both 'left' lights say nothing remains and carry is 0."""
    acts = []
    while True:
        o = env.obs(); leftx, lefty = o[:, 42], o[:, 43]; cur = env.px
        stop = (cur > 0) & (leftx > 0.98) & (lefty > 0.98) & (env.carry == 0) | (cur >= NP)
        if stop.all(): break
        for name in ("STEP_X", "STEP_Y", "SUM"):
            act = torch.where(stop, torch.full((env.B,), A["STOP"], device=dev), torch.full((env.B,), A[name], device=dev)); acts.append((env.obs(), act)); env.act(act)
        s = env.ls.argmax(1); act = torch.where(stop, torch.full_like(s, A["STOP"]), A["WRITE_0"] + s % 10); acts.append((env.obs(), act)); env.act(act)
        act = torch.where(stop, torch.full_like(s, A["STOP"]), A["CARRY_0"] + s // 10); acts.append((env.obs(), act)); env.act(act)
    return acts
def teacher_cmp(env):
    """STEP_BOTH until nothing is left on either side, tracking the last differing place; then ANSWER."""
    acts = []; verdict = torch.zeros(env.B, dtype=torch.long, device=dev)
    for k in range(NP):
        act = torch.full((env.B,), A["STEP_BOTH"], device=dev); acts.append((env.obs(), act)); env.act(act)
        o = env.obs(); sx, sy = o[:, 40], o[:, 41]; verdict = torch.where(sx < sy - 1e-4, torch.ones_like(verdict), torch.where(sx > sy + 1e-4, torch.full_like(verdict, 2), verdict))
        if k >= 0 and (o[:, 42] > 0.98).all() and (o[:, 43] > 0.98).all(): break
    act = torch.tensor([A["ANSWER_EQ"], A["ANSWER_LT"], A["ANSWER_GT"]], device=dev)[verdict]; acts.append((env.obs(), act)); env.act(act)
    return acts


def teacher_policy(env, truth_cmp=None):
    """the hand-written program as a per-state policy: next action from the env's current state (used to label states the controller visits)"""
    o = env.obs(); prev = torch.where(env.prev.sum(1) == 0, torch.full((env.B,), -1, dtype=torch.long, device=dev), env.prev.argmax(1)); act = torch.full((env.B,), A["STOP"], dtype=torch.long, device=dev)   # -1 = no previous action
    if env.task == 0:
        stop = (env.px > 0) & (o[:, 42] > 0.98) & (o[:, 43] > 0.98) & (env.carry == 0) | (env.px >= NP)
        phase_x = (prev == -1) | (prev == A["CARRY_0"]) | (prev == A["CARRY_1"]) | (prev == A["STOP"]) | (prev >= A["STEP_BOTH"])
        s = env.ls.argmax(1)
        act = torch.where(phase_x, torch.full_like(act, A["STEP_X"]), act); act = torch.where(prev == A["STEP_X"], torch.full_like(act, A["STEP_Y"]), act); act = torch.where(prev == A["STEP_Y"], torch.full_like(act, A["SUM"]), act)
        act = torch.where(prev == A["SUM"], A["WRITE_0"] + s % 10, act); act = torch.where((prev >= A["WRITE_0"]) & (prev <= A["WRITE_9"]), A["CARRY_0"] + s // 10, act)
        act = torch.where(stop & phase_x, torch.full_like(act, A["STOP"]), act)
    else:
        fin = (env.px > 0) & (o[:, 42] > 0.98) & (o[:, 43] > 0.98) | (env.px >= NP)
        act = torch.where(fin, torch.tensor([A["ANSWER_EQ"], A["ANSWER_LT"], A["ANSWER_GT"]], device=dev)[truth_cmp], torch.full_like(act, A["STEP_BOTH"]))
    return act


def sample_pairs(n, max_digits, task):
    xs, ys = [], []
    for _ in range(n):
        L = int(rng.integers(1, max_digits + 1)); s = int(rng.integers(10 ** (L - 1), 10 ** L))
        if task == 0: x = int(rng.integers(0, s + 1)); xs.append(x); ys.append(s - x)
        else: y = int(np.clip(s + rng.integers(-10 ** int(rng.integers(0, L)), 10 ** int(rng.integers(0, L)) + 1), 0, 10 ** L - 1)); xs.append(s); ys.append(int(y))
    return xs, ys


class Controller(nn.Module):
    def __init__(self, h):
        super().__init__(); self.gru = nn.GRUCell(OBS, h); self.out = nn.Linear(h, NA); self.h = h
    def forward(self, o, hid): hid = self.gru(o, hid); return self.out(hid), hid


ctrl = Controller(a.hidden).to(dev); opt = torch.optim.Adam(ctrl.parameters(), lr=1e-3); (HERE / a.out).mkdir(parents=True, exist_ok=True)
def rollout_label(n_ep, task, mix=0.5):
    """episodes driven by a mixture of the controller and the teacher (mix = share of controller actions; 0 = pure teacher);
    every visited state is labelled by the per-state teacher policy — one consistent rule everywhere"""
    xs, ys = sample_pairs(n_ep, a.train_max_digits, task); env = Env(xs, ys, task); hid = torch.zeros(len(xs), a.hidden, device=dev)
    truth = torch.tensor([0 if x == y else (1 if x < y else 2) for x, y in zip(xs, ys)], device=dev); O, T = [], []
    for _ in range(80):
        o = env.obs(); lab = teacher_policy(env, truth); O.append(o); T.append(lab)
        if mix > 0: logits, hid = ctrl(o, hid); act = torch.where(torch.rand(len(xs), device=dev) < mix, logits.argmax(1), lab)
        else: act = lab
        env.act(act)
        if env.done.all(): break
    ok = (np.array(env.result_numbers()) == np.array(xs) + np.array(ys)) if task == 0 else (env.answer.cpu().numpy() == truth.cpu().numpy())
    return torch.stack(O, 1), torch.stack(T, 1), float(ok.mean())
traces = []
with torch.no_grad():                                                                             # teacher traces: the per-state policy rolled out (consistent with every later label)
    for task in (0, 1):
        for _ in range(a.traces // 2 // 256): traces.append(rollout_label(256, task, mix=0.0))
    print(f"teacher traces: {len(traces) * 256} episodes, teacher exact on its own traces: add {np.mean([t[2] for t in traces[:len(traces)//2]]):.3f}  cmp {np.mean([t[2] for t in traces[len(traces)//2:]]):.3f}  ({time.time()-t0:.0f}s)", flush=True)
def train(epochs, tag):
  for ep in range(1, epochs + 1):
    order = rng.permutation(len(traces)); tot = 0.0; n = 0
    for i in order:
        O, T, _ = traces[i]; hid = torch.zeros(O.shape[0], a.hidden, device=dev); loss = 0.0
        for t in range(O.shape[1]):
            logits, hid = ctrl(O[:, t], hid); loss = loss + F.cross_entropy(logits, T[:, t])
        loss = loss / O.shape[1]; opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); n += 1
    if ep % 10 == 0 or ep == epochs: print(f"{tag} epoch {ep}/{epochs}  imitation loss {tot/n:.4f}  lr 0.001  traces {len(traces)}  elapsed {time.time()-t0:.0f}s", flush=True); torch.save({"ctrl": ctrl.state_dict(), "epoch": ep, "config": vars(a)}, HERE / a.out / "ckpt.pt")
train(a.epochs, "imitation")
for r in range(1, a.dagger + 1):
    with torch.no_grad():
        for task in (0, 1):
            for _ in range(a.dagger_episodes // 256):
                traces.append(rollout_label(256, task, mix=0.5))
    train(20, f"dagger {r}")
# ------------------------------------------------------------------ execution: the controller acts alone, scored by outcome
res = {"config": vars(a)}
with torch.no_grad():
    def run(xs, ys, task, max_steps=80):
        env = Env(xs, ys, task); hid = torch.zeros(len(xs), a.hidden, device=dev)
        for _ in range(max_steps):
            logits, hid = ctrl(env.obs(), hid); env.act(logits.argmax(1))
            if env.done.all(): break
        return env
    for task, name in ((0, "add"), (1, "compare")):
        for L in range(1, NP + 1):
            xs, ys = sample_pairs(3000, L, task); xs, ys = [x for x, y in zip(xs, ys) if len(str(x + y if task == 0 else max(x, y))) == L], [y for x, y in zip(xs, ys) if len(str(x + y if task == 0 else max(x, y))) == L]
            if not xs: continue
            xs, ys = xs[:500], ys[:500]; env = run(xs, ys, task)
            if task == 0: ok = np.array(env.result_numbers()) == np.array(xs) + np.array(ys)
            else: ok = env.answer.cpu().numpy() == np.array([0 if x == y else (1 if x < y else 2) for x, y in zip(xs, ys)])
            res[f"{name}_len{L}"] = {"n": len(xs), "exact": float(ok.mean()), "seen_length": L <= a.train_max_digits}
            print(f"{name:8s} {L}-digit{' (length never seen in training)' if L > a.train_max_digits else ''}: n={len(xs):3d}  exact {ok.mean():.3f}", flush=True)
save_json(res, HERE / a.out / "controller.json"); torch.save({"ctrl": ctrl.state_dict(), "config": vars(a)}, HERE / a.out / "controller.pt"); print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)
