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
ap.add_argument("--programs", default="results/places/found_programs.json"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results/progmem")
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time()
found = json.load(open(HERE / a.programs))
# ------------------------------------------------------------------ the alphabet: symbol rows + role anchors
OPS = ["LOAD", "BIND", "UNBIND", "SHIFT", "READ", "READN", "READ2", "READMUL", "ORDER", "SETD", "SETC", "SET", "COPY", "WRITE", "ANSWER", "CALL", "SECTION", "STOP"]
ARGS = [f"s{i}" for i in range(6)] + ["c0", "c1", "cEQ"] + list(found) + ["init", "body", "outro", "none"]
SYM = OPS + ARGS; sid = {s: i for i, s in enumerate(SYM)}; ROLES = ["OPCODE", "ARG1", "ARG2", "ARG3"]; NS, NRl = len(SYM), len(ROLES)
NODES = SYM + ROLES; nid = {n: i for i, n in enumerate(NODES)}; PLUS, TIMES = 0, 1; NR = 2
model = ResonatE(len(NODES), 2 * NR, k=a.k, block=True, block_size=a.block).to(dev); M = model.m
def E(): return cnorm(model.E)
def hop(z, r): return model.hop(z, torch.full((z.shape[0],), r, device=dev, dtype=torch.long))
def light(z): return torch.real(z @ E().conj().t())


def chain(T):
    """T: (B, L) symbol ids; term j has role j mod 4; each term = hop(E_sym (+) E_role, x); chained by + (first through its own +)."""
    Er = E(); B, L = T.shape; acc = None
    for j in range(L):
        role = nid[ROLES[j % 4]]; term = hop(cnorm(Er[T[:, j]] + Er[role][None].expand(B, -1)), TIMES)
        acc = hop(term, PLUS) if acc is None else hop(cnorm(acc + term), PLUS)
    return acc


def walk_loss(z, T):
    """walk back: after j reverse + hops and x^-1, the light shows exactly the j-th term from the end (its symbol and its role)"""
    B, L = T.shape; loss = 0.0; rr = z
    for j in range(L):
        rr = hop(rr, PLUS + NR); Lg = light(hop(rr, TIMES + NR)) * model.log_tau.exp(); tgt = torch.zeros_like(Lg)
        tgt[torch.arange(B), T[:, L - 1 - j]] = 1; tgt[:, nid[ROLES[(L - 1 - j) % 4]]] = 1; loss = loss + F.binary_cross_entropy_with_logits(Lg, tgt)
    return loss / L


def walk_back(z, L):
    """read L terms back (last first): symbol ids among the symbol rows"""
    B = z.shape[0]; out = torch.zeros(B, L, dtype=torch.long, device=dev); rr = z
    for j in range(L):
        rr = hop(rr, PLUS + NR); Lg = light(hop(rr, TIMES + NR)); Lg[:, NS:] = -1e9; out[:, L - 1 - j] = Lg.argmax(1)
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
    for tag, v in decode_terms(syms):
        if tag == "SECTION": cur = v
        elif tag == "STOP": p["stop"] = v
        elif cur is not None: p[cur].append(v)
    return p


# ------------------------------------------------------------------ 1. train the chain model on random symbol chains (programs are long)
opt = torch.optim.Adam(model.parameters(), lr=1e-3); (HERE / a.out).mkdir(parents=True, exist_ok=True); best_state, best_ex = None, -1
def val_exact(L, n=200):
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
        print(f"random symbol chains of {L} terms{' (longer than trained)' if L > a.max_terms else ''}: walk back exactly {ex:.3f}", flush=True)
    # ------------------------------------------------------------------ 2. promote the five found programs to memory as labelled places
    rows, labels, lengths, syms_of = [], [], [], {}
    for name, p in found.items():
        syms = program_symbols(p); syms_of[name] = syms; T = torch.tensor([[sid[s] for s in syms]], device=dev); z = chain(T); rows.append(z[0]); labels.append(name); lengths.append(len(syms))
        back = [SYM[i] for i in walk_back(z, len(syms))[0].tolist()]; p2 = symbols_to_program(back, p["kind"]); ok = (p2["init"], p2["body"], p2["outro"], p2["stop"]) == (p["init"], p["body"], p["outro"], p["stop"])
        res[f"promote_{name}"] = {"terms": len(syms), "walks_back_exactly": bool(ok)}; print(f"program '{name}': {len(syms)} terms as one place; promoted; walked back from memory -> {'identical' if ok else 'DIFFERENT'}", flush=True)
    LTM = torch.stack(rows)                                                                            # long-term memory: the program rows
    # programs that share structure light alike: cosine of the promoted places
    S = torch.real(LTM @ LTM.conj().t()).cpu().numpy(); print("program places, cosine matrix (add sub cmp muld mul):\n" + "\n".join("  " + " ".join(f"{v:5.2f}" for v in row) for row in S), flush=True)
    # ------------------------------------------------------------------ 3. execute from memory: instructions are read by walking the program's place
    sys.argv = ["places", "--only-check"]; src = open(HERE / "places.py").read().split("# ------------------------------------------------------------------ 1. the machine")[0]; g = {}; exec(src, g)
    PROGRAMS_mem = {}
    for i, name in enumerate(labels):                                                                  # read each program back from its row, by walking
        back = [SYM[j] for j in walk_back(LTM[i][None], lengths[i])[0].tolist()]; PROGRAMS_mem[name] = symbols_to_program(back, found[name]["kind"])
    g["PROGRAMS"].update(PROGRAMS_mem)
    for name in labels:
        v = g["verify"](PROGRAMS_mem[name], name, lengths=(1, 3, 5, 7), n=100); res[f"exec_from_memory_{name}"] = v
        print(f"executing '{name}' read from memory (and its CALLs read from memory): exact " + " ".join(f"{L}d {e:.3f}" for L, e in v.items()), flush=True)
save_json(res, HERE / a.out / "progmem.json"); torch.save({"model": model.state_dict(), "SYM": SYM, "ROLES": ROLES, "LTM": LTM.cpu(), "labels": labels, "lengths": lengths}, HERE / a.out / "progmem.pt"); print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)
