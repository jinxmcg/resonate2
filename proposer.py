"""A learned proposer for program search: the Part-12 controller turned around. A stateless policy reads what the machine
shows at a position — the slot kinds (empty / number / digit / verdict) before the instruction, which output places the
candidate gets wrong, the previous opcode, the program part — and emits an instruction (opcode, callee, slot arguments).
Trained by imitation on programs already found (each instruction of each program, under random corruptions of the
program so the failing-place light is informative), with the target operation HELD OUT. Used in discover_io.py as the
mutation operator in place of a random draw; measured as candidates-to-acceptance with and without it.
Shared module: `Proposer` (features, model, sampling) and `imitation_data` (corruptions -> examples), parameterised by the
machine's globals so it runs on whichever machine execs it."""
import re, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
OPS = ["LOAD", "BIND", "UNBIND", "SHIFT", "READ", "READN", "READ2", "READMUL", "ORDER", "SETD", "SETC", "SET", "COPY", "WRITE", "ANSWER", "CALL"]
CALLEES = ["none", "add", "sub", "cmp", "muld", "mul", "double", "triple", "sumd", "other"]
CONSTS = ["0", "1", "EQ"]; PARTS = ["init", "body", "outro"]; NPL = 4                                                   # output places tracked (units..thousands) + beyond
def parse_ins(name):
    op = name.split()[0]; callee = "none"; rest = name[len(op):].strip()
    if op == "CALL": callee, rest = rest.split(" ", 1); callee = callee if callee in CALLEES else "other"
    const = None
    if op == "SET": s, c = rest.split("<"); return op, callee, [int(s)], c
    nums = [int(x) for x in re.findall(r"\d+", rest)]; return op, callee, nums, const
def build_ins(op, callee, args, const, K):
    a = [int(x) % K for x in args] + [0, 0, 0]; s, t, u = a[0], a[1], a[2]
    if op in ("SHIFT", "READ", "READN", "READ2", "SETD", "SETC", "WRITE", "ANSWER"): return f"{op} {s}"
    if op in ("BIND", "UNBIND", "READMUL"): return f"{op} {s},{t}" if s != t else None
    if op == "LOAD" or op == "COPY": return f"{op} {s}>{t}" if s != t else None
    if op == "ORDER": return f"ORDER {s},{t}>{u}" if len({s, t, u}) == 3 else None
    if op == "SET": return f"SET {s}<{const}"
    if op == "CALL": return f"CALL {callee} {s},{t}>{u}" if s != t else None
    return None
def kind_vec(kinds, K):
    """slot kinds -> one-hot (K x 4): empty, number (walkable), digit, other symbol (verdict/EQ)"""
    walk, sym = kinds; v = np.zeros((K, 4), dtype=np.float32)
    for i in range(K):
        v[i, 1 if walk[i] else (0 if sym[i] == -9 else (2 if 0 <= sym[i] <= 9 else 3))] = 1
    return v.reshape(-1)
def features(kinds, fail, prev_op, part, pos_frac, K):
    f = [kind_vec(kinds, K), np.asarray(fail, dtype=np.float32)[:NPL + 1], np.eye(len(OPS) + 1, dtype=np.float32)[OPS.index(prev_op) if prev_op in OPS else len(OPS)], np.eye(3, dtype=np.float32)[PARTS.index(part)], np.array([pos_frac], dtype=np.float32)]
    return np.concatenate(f)
class Policy(nn.Module):
    def __init__(self, K, hidden=192):
        super().__init__(); d = K * 4 + NPL + 1 + len(OPS) + 1 + 3 + 1; self.K = K
        self.body = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.h_op = nn.Linear(hidden, len(OPS)); self.h_callee = nn.Linear(hidden, len(CALLEES)); self.h_a = nn.ModuleList([nn.Linear(hidden + len(OPS), K) for _ in range(3)]); self.h_const = nn.Linear(hidden, len(CONSTS))
    def forward(self, x, op_onehot=None):
        h = self.body(x); lo = self.h_op(h); op1 = F.one_hot(lo.argmax(1), len(OPS)).float() if op_onehot is None else op_onehot
        hh = torch.cat([h, op1], 1); return lo, self.h_callee(h), [hd(hh) for hd in self.h_a], self.h_const(h)
class Proposer:
    """sampling wrapper: given the machine's instruction set at search time, propose an instruction for a position"""
    def __init__(self, path, dev):
        pk = torch.load(path, map_location=dev); self.K = pk["K"]; self.model = Policy(self.K).to(dev); self.model.load_state_dict(pk["model"]); self.model.eval(); self.dev = dev; self.rng = np.random.default_rng(0)
    def propose(self, INS_set, kinds, fail, prev_op, part, pos_frac, T=1.0, tries=12, applicable=None):
        x = torch.tensor(features(kinds, fail, prev_op, part, pos_frac, self.K), device=self.dev)[None]
        with torch.no_grad():
            lo, lc, la, lk = self.model(x); p_op = F.softmax(lo[0] / T, 0).cpu().numpy()
            for _ in range(tries):
                oi = int(self.rng.choice(len(OPS), p=p_op)); op = OPS[oi]; op1 = F.one_hot(torch.tensor([oi], device=self.dev), len(OPS)).float()
                _, lc2, la2, lk2 = self.model(x, op1); callee = CALLEES[int(self.rng.choice(len(CALLEES), p=F.softmax(lc2[0] / T, 0).cpu().numpy()))]
                args = [int(self.rng.choice(self.K, p=F.softmax(l[0] / T, 0).cpu().numpy())) for l in la2]; const = CONSTS[int(self.rng.choice(len(CONSTS), p=F.softmax(lk2[0] / T, 0).cpu().numpy()))]
                name = build_ins(op, callee, args, const, self.K)
                if name is not None and name in INS_set and (applicable is None or applicable(name, kinds)): return name
        return None
def targets_of(name):
    op, callee, args, const = parse_ins(name); a = (args + [0, 0, 0])[:3]
    return OPS.index(op), CALLEES.index(callee), a, (CONSTS.index(const) if const in CONSTS else 0)
def train_policy(examples, K, dev, steps=4000, lr=2e-3, seed=0):
    """examples: list of (feature vector, target instruction name)"""
    torch.manual_seed(seed); X = torch.tensor(np.stack([f for f, _ in examples]), device=dev); T = [targets_of(n) for _, n in examples]
    y_op = torch.tensor([t[0] for t in T], device=dev); y_c = torch.tensor([t[1] for t in T], device=dev); y_a = torch.tensor([t[2] for t in T], device=dev); y_k = torch.tensor([t[3] for t in T], device=dev)
    op1 = F.one_hot(y_op, len(OPS)).float(); m = Policy(K).to(dev); opt = torch.optim.Adam(m.parameters(), lr=lr); n = len(examples)
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (min(512, n),), device=dev); lo, lc, la, lk = m(X[idx], op1[idx])
        loss = F.cross_entropy(lo, y_op[idx]) + F.cross_entropy(lc, y_c[idx]) + sum(F.cross_entropy(l, y_a[idx, j]) for j, l in enumerate(la)) + F.cross_entropy(lk, y_k[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0 or step == 1:
            with torch.no_grad(): lo, lc, la, lk = m(X, op1); acc = (lo.argmax(1) == y_op).float().mean().item(); acc_a = float(np.mean([(l.argmax(1) == y_a[:, j]).float().mean().item() for j, l in enumerate(la)]))
            print(f"  proposer step {step}/{steps}  loss {loss.item():.3f}  opcode acc {acc:.3f}  arg acc {acc_a:.3f}", flush=True)
    return m
