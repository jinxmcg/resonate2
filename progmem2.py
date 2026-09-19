"""Programs in the memory. An instruction is four terms — opcode, arg1, arg2, arg3 — each a symbol row bound to a ROLE
anchor (like a digit bound to a place); a program is one directed chain of those terms (like a number is a chain of its
terms), composed with the same + and x, walked back one hop + one light per step to execute. A found program is promoted
to long-term memory as a labelled row; CALL loads the callee's place and walks it.
1. train the chain model on random symbol chains up to 64 terms (walk-back objective; programs are longer than numbers)
2. encode the five found programs, promote them (rows + labels), walk them back: exact instruction lists?
3. execute the five operations FROM MEMORY (instructions read by walking): still 1.000?"""
import argparse, time, json, sys, numpy as np, torch, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=12); ap.add_argument("--block", type=int, default=16); ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--max-terms", type=int, default=64)
ap.add_argument("--programs", default="results/places/found_programs.json"); ap.add_argument("--eval-only", action="store_true"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results/progmem")
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time()
found = json.load(open(HERE / a.programs))
# ------------------------------------------------------------------ the alphabet: symbol rows + role anchors
OPS = ["LOAD", "BIND", "UNBIND", "SHIFT", "READ", "READN", "READ2", "READMUL", "ORDER", "SETD", "SETC", "SET", "COPY", "WRITE", "ANSWER", "CALL", "SECTION", "STOP"]
ARGS = [f"s{i}" for i in range(6)] + ["c0", "c1", "cEQ"] + list(found) + ["init", "body", "outro", "none"]
SYM = OPS + ARGS; sid = {s: i for i, s in enumerate(SYM)}; ROLES = ["OPCODE", "ARG1", "ARG2", "ARG3"]; NS, NRl = len(SYM), len(ROLES)
NODES = SYM; nid = {n: i for i, n in enumerate(NODES)}; PLUS, TIMES = 0, 1; NR = 2 + NRl; ROLE_R = {r: 2 + i for i, r in enumerate(ROLES)}   # relations: +, x, OP, R1, R2, R3 (with learned reverses)
model = ResonatE(len(NODES), 2 * NR, k=a.k, block=True, block_size=a.block).to(dev); M = model.m
def E(): return cnorm(model.E)
def hop(z, r): return model.hop(z, torch.full((z.shape[0],), r, device=dev, dtype=torch.long))
def light(z): return torch.real(z @ E().conj().t())


def chain(T):
    """T: (B, L) symbol ids, L a multiple of 4: each INSTRUCTION = one term = sum over its four roles of hop(E_symbol, role);
    the program = the instructions chained by + (first through its own +). Depth = number of instructions, not symbols."""
    Er = E(); B, L = T.shape; acc = None
    for j in range(0, L, 4):
        term = cnorm(sum(hop(Er[T[:, j + r]], ROLE_R[ROLES[r]]) for r in range(4)))
        acc = hop(term, PLUS) if acc is None else hop(cnorm(acc + term), PLUS)
    return acc


def walk_loss(z, T):
    """walk back: after j reverse + hops, the four role lights (role^-1 then the symbol rows) show exactly the j-th instruction from the end"""
    B, L = T.shape; n = L // 4; loss = 0.0; rr = z
    for j in range(n):
        rr = hop(rr, PLUS + NR); i = n - 1 - j
        for r in range(4):
            Lg = light(hop(rr, ROLE_R[ROLES[r]] + NR)) * model.log_tau.exp(); tgt = torch.zeros_like(Lg); tgt[torch.arange(B), T[:, 4 * i + r]] = 1; loss = loss + F.binary_cross_entropy_with_logits(Lg, tgt)
    return loss / (4 * n)


def walk_back(z, L):
    """read L symbols (L/4 instructions) back, last instruction first: one + step, then four role lights"""
    B = z.shape[0]; n = L // 4; out = torch.zeros(B, L, dtype=torch.long, device=dev); rr = z
    for j in range(n):
        rr = hop(rr, PLUS + NR); i = n - 1 - j
        for r in range(4): out[:, 4 * i + r] = light(hop(rr, ROLE_R[ROLES[r]] + NR)).argmax(1)
    return out


# ------------------------------------------------------------------ instruction <-> four symbols
def encode_instruction(ins):
    """'BIND 3,4' -> [BIND, s3, s4, none]; 'CALL muld 4,3>5' -> [CALL, muld, s4, s3] + ... (CALL needs 4 args: name, s, t, u -> two terms)"""
    op = ins.split()[0]; rest = ins[len(op):].strip()
    if op == "CALL": name, args = rest.split(" ", 1); nums = [int(x) for x in __import__("re").findall(r"\d+", args)]; return [["CALL", name, f"s{nums[0]}", f"s{nums[1]}"], ["CALL", f"s{nums[2]}", "none", "none"]]
    if op == "SET": s = int(rest.split("<")[0]); c = rest.split("<")[1]; return [["SET", f"s{s}", "c" + c, "none"]]
    nums = [int(x) for x in __import__("re").findall(r"\d+", rest)]; return [[op] + [f"s{n}" for n in nums] + ["none"] * (3 - len(nums))]
def decode_terms(syms):
    """four-symbol groups back to instruction strings (CALL spans two groups)"""
    out = []; i = 0
    while i < len(syms):
        g = syms[i:i + 4]; op = g[0]
        if op == "SECTION": out.append(("SECTION", g[1])); i += 4; continue
        if op == "STOP": out.append(("STOP", int(g[1][1]))); i += 4; continue
        if op == "CALL": g2 = syms[i + 4:i + 8]; out.append(("INS", f"CALL {g[1]} {g[2][1]},{g[3][1]}>{g2[1][1]}")); i += 8; continue
        if op == "SET": out.append(("INS", f"SET {g[1][1]}<{g[2][1:]}")); i += 4; continue
        nums = [x[1] for x in g[1:] if x != "none"]
        if op in ("LOAD", "COPY"): out.append(("INS", f"{op} {nums[0]}>{nums[1]}"))
        elif op in ("BIND", "UNBIND", "READMUL"): out.append(("INS", f"{op} {nums[0]},{nums[1]}"))
        elif op == "ORDER": out.append(("INS", f"ORDER {nums[0]},{nums[1]}>{nums[2]}"))
        else: out.append(("INS", f"{op} {nums[0]}"))
        i += 4
    return out
def program_symbols(p):
    """a program as a flat symbol list, in EXECUTION order: SECTION init, ..., SECTION body, ..., SECTION outro, ..., STOP s"""
    syms = []
    for sec in ("init", "body", "outro"):
        syms += ["SECTION", sec, "none", "none"]
        for ins in p[sec]: syms += [x for grp in encode_instruction(ins) for x in grp]
    syms += ["STOP", f"s{p['stop']}", "none", "none"]; return syms
def symbols_to_program(syms, kind):
    p = {"kind": kind, "init": [], "body": [], "outro": [], "stop": 0}; cur = None
    try: items = decode_terms(syms)
    except Exception: return {"kind": kind, "init": ["?"], "body": ["?"], "outro": [], "stop": -1}                       # unreadable: a mis-read symbol
    for tag, v in items:
        if tag == "SECTION": cur = v
        elif tag == "STOP": p["stop"] = v
        elif cur is not None: p[cur].append(v)
    return p


# ------------------------------------------------------------------ 1. train the chain model on random symbol chains (programs are long)
opt = torch.optim.Adam(model.parameters(), lr=1e-3); (HERE / a.out).mkdir(parents=True, exist_ok=True); best_state, best_ex = None, -1
if a.eval_only: best_state = torch.load(HERE / a.out / "ckpt.pt", map_location=dev, weights_only=False)["model"]; a.steps = 0
def val_exact(L, n=200):
    L = max(4, L - L % 4)
    with torch.no_grad(): T = torch.tensor(rng.integers(0, NS, (n, L)), device=dev); return (walk_back(chain(T), L) == T).all(1).float().mean().item()
for step in range(1, a.steps + 1):
    cur_max = int(8 + (a.max_terms - 8) * min(1.0, step / (0.6 * a.steps)))                        # length curriculum: short chains first
    L = int(rng.integers(4, cur_max + 1)); L -= L % 4; L = max(L, 4); T = torch.tensor(rng.integers(0, NS, (256, L)), device=dev)
    loss = walk_loss(chain(T), T); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if step % 250 == 0 or step == 1:
        ex = val_exact(min(cur_max, a.max_terms))
        if ex >= best_ex: best_ex = ex; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"step {step}/{a.steps}  walk loss {loss.item():.4f}  max len {cur_max}  exact@{min(cur_max, a.max_terms)} {ex:.3f}  lr 0.001  tau {model.log_tau.exp().item():.1f}  elapsed {time.time()-t0:.0f}s  ETA {(time.time()-t0)/step*(a.steps-step):.0f}s", flush=True)
        torch.save({"model": best_state, "step": step}, HERE / a.out / "ckpt.pt")
model.load_state_dict(best_state); model.eval(); res = {"config": vars(a), "symbols": NS}
with torch.no_grad():
    for L in (8, 32, 64, 96, 128):
        T = torch.tensor(rng.integers(0, NS, (300, L)), device=dev); ex = (walk_back(chain(T), L) == T).all(1).float().mean().item(); res[f"walk_len{L}"] = ex
        print(f"random programs of {L // 4} instructions ({L} symbols){' (longer than trained)' if L > a.max_terms else ''}: walk back exactly {ex:.3f}", flush=True)
    # ------------------------------------------------------------------ 2. promote the five found programs as PAGES of <= PAGE instructions (a place holds ~6 exactly)
    PAGE = 6; CONT = ["SECTION", "none", "none", "none"]                                                # 'continue on the next page' marker (an unused instruction shape; no new row)
    def pages_of(syms):
        ins = [syms[i:i + 4] for i in range(0, len(syms), 4)]; out = []
        while ins:
            take = ins[:PAGE - 1] if len(ins) > PAGE else ins; ins = ins[len(take):]; page = [x for g in take for x in g] + ([x for x in CONT] if ins else []); out.append(page)
        return out
    LTM, labels, page_len = [], [], []
    for name, p in found.items():
        syms = program_symbols(p); pgs = pages_of(syms); back_all = []
        for i, pg in enumerate(pgs):
            T = torch.tensor([[sid[x] for x in pg]], device=dev); z = chain(T); LTM.append(z[0]); labels.append(f"{name}#{i}"); page_len.append(len(pg))
            back = [SYM[j] for j in walk_back(z, len(pg))[0].tolist()]; back_all += back[:-4] if back[-4:] == CONT else back
        p2 = symbols_to_program(back_all, p["kind"]); ok = (p2["init"], p2["body"], p2["outro"], p2["stop"]) == (p["init"], p["body"], p["outro"], p["stop"])
        res[f"promote_{name}"] = {"instructions": len(syms) // 4, "pages": len(pgs), "walks_back_exactly": bool(ok)}
        print(f"program '{name}': {len(syms) // 4} instructions in {len(pgs)} pages; promoted as {len(pgs)} labelled rows; walked back from memory page by page -> {'identical' if ok else 'DIFFERENT'}", flush=True)
    LTM = torch.stack(LTM)
    # ------------------------------------------------------------------ 3. execute from memory: instructions read by walking the pages
    sys.argv = ["places", "--only-check"]; src = open(HERE / "places.py").read().split("# ------------------------------------------------------------------ 1. the machine")[0]; g = {}; exec(src, g)
    PROGRAMS_mem = {}
    for name, p in found.items():
        syms = []
        for i, lab in enumerate(labels):
            if lab.split("#")[0] != name: continue
            back = [SYM[j] for j in walk_back(LTM[i][None], page_len[i])[0].tolist()]; syms += back[:-4] if back[-4:] == CONT else back
        PROGRAMS_mem[name] = symbols_to_program(syms, p["kind"])
    g["PROGRAMS"].update(PROGRAMS_mem)
    for name in found:
        v = g["verify"](PROGRAMS_mem[name], name, lengths=(1, 3, 5, 7), n=100); res[f"exec_from_memory_{name}"] = v
        print(f"executing '{name}' read from memory (its CALLs read from memory too): exact " + " ".join(f"{L}d {e:.3f}" for L, e in v.items()), flush=True)
save_json(res, HERE / a.out / "progmem.json"); torch.save({"model": model.state_dict(), "SYM": SYM, "ROLES": ROLES, "LTM": LTM.cpu(), "labels": labels, "page_len": page_len}, HERE / a.out / "progmem.pt"); print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)
