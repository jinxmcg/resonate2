"""One memory: numbers and programs in the same table, with the same + and x.
Start from the number chain model of Part 10 (rows: digits 0-9, anchors 10^1..10^8; relations +, x). Add by continuation:
39 instruction-symbol rows and 4 role relations (OP, R1, R2, R3); operators and old rows keep learning, numbers replayed.
Verify: (1) numbers 1-9 digits walk back exactly (retention); (2) random programs of <= 6 instructions walk back exactly;
(3) the five found programs promoted as pages into THIS table execute at 1.000 from memory (the places machine now runs
on the merged table); (4) a NEW program whose name is a freshly appended random row: is a CALL to it read back through
the existing role operators without any training? If so, adding a program is a write, nothing else."""
import argparse, time, json, sys, numpy as np, torch, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--path", default="results/grow_1e9_moves/path.pt"); ap.add_argument("--programs", default="results/places/found_programs.json"); ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results/onemem"); ap.add_argument("--eval-only", action="store_true")
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device("cuda"); t0 = time.time(); NP = 9
found = json.load(open(HERE / a.programs))
NUMROWS = [str(d) for d in range(10)] + [str(10 ** k) for k in range(1, NP)]
OPS = ["LOAD", "BIND", "UNBIND", "SHIFT", "READ", "READN", "READ2", "READMUL", "ORDER", "SETD", "SETC", "SET", "COPY", "WRITE", "ANSWER", "CALL", "SECTION", "STOP"]
ARGS = [f"s{i}" for i in range(6)] + ["c0", "c1", "cEQ"] + list(found) + ["init", "body", "outro", "none"]
SYM = OPS + ARGS; NODES = NUMROWS + SYM; nid = {n: i for i, n in enumerate(NODES)}; sid = {s: nid[s] for s in SYM}; ROLES = ["OPCODE", "ARG1", "ARG2", "ARG3"]
PLUS, TIMES = 0, 1; NOPS = 2 + len(ROLES); NR = NOPS; ROLE_R = {r: 2 + i for i, r in enumerate(ROLES)}
pk = torch.load(HERE / a.path, map_location=dev, weights_only=False); pcfg = pk["config"]; old = pk["model"]; n_old = old["E"].shape[0]
model = ResonatE(len(NODES), 2 * NR, k=pcfg["k"], block=True, block_size=pcfg["block"]).to(dev); M = model.m
with torch.no_grad():                                                                                   # old rows and the old +, x (and their reverses) carried over; new rows/relations random
    model.E.data[:n_old] = old["E"]; H = model.H.data; Hold = old["H"]; H[0], H[1] = Hold[0], Hold[1]; H[NR + 0], H[NR + 1] = Hold[2], Hold[3]; model.log_tau.data = old["log_tau"]
def E(): return cnorm(model.E)
def hop(z, r): return model.hop(z, torch.full((z.shape[0],), r, device=dev, dtype=torch.long))
def light(z): return torch.real(z @ E().conj().t())
ANCH = torch.tensor([nid["0"]] + [nid[str(10 ** k)] for k in range(1, NP)], device=dev)
def digs(ns): return torch.tensor([[int(c) for c in f"{int(n):0{NP}d}"] for n in ns], device=dev)
def num_chain(D):
    Er = E(); B = D.shape[0]; acc = None
    for j in range(NP):
        k = NP - 1 - j; d = D[:, j]; term = Er[d] if k == 0 else hop(cnorm(Er[d] + Er[ANCH[k]][None].expand(B, -1)), TIMES)
        acc = hop(term, PLUS) if acc is None else hop(cnorm(acc + term), PLUS)
    return acc
def num_walk_loss(z, D):
    B = D.shape[0]; loss = 0.0; rr = z
    for j in range(NP):
        k = j; d = D[:, NP - 1 - j]; rr = hop(rr, PLUS + NR); Lg = (light(rr) if k == 0 else light(hop(rr, TIMES + NR))) * model.log_tau.exp(); tgt = torch.zeros_like(Lg); tgt[torch.arange(B), d] = 1
        if k > 0: tgt[:, ANCH[k]] = 1
        loss = loss + F.binary_cross_entropy_with_logits(Lg, tgt)
    return loss / NP
def num_walk(z):
    B = z.shape[0]; out = torch.zeros(B, NP, dtype=torch.long, device=dev); rr = z
    for k in range(NP):
        rr = hop(rr, PLUS + NR)
        if k == 0: out[:, 0] = light(rr)[:, :10].argmax(1)
        else: L = light(hop(rr, TIMES + NR)); L[:, 10:] = -1e9; out[:, k] = L.argmax(1)
    return out
def prog_chain(T):
    Er = E(); B, L = T.shape; acc = None
    for j in range(0, L, 4):
        term = cnorm(sum(hop(Er[T[:, j + r]], ROLE_R[ROLES[r]]) for r in range(4))); acc = hop(term, PLUS) if acc is None else hop(cnorm(acc + term), PLUS)
    return acc
def prog_walk_loss(z, T):
    B, L = T.shape; n = L // 4; loss = 0.0; rr = z
    for j in range(n):
        rr = hop(rr, PLUS + NR); i = n - 1 - j
        for r in range(4): Lg = light(hop(rr, ROLE_R[ROLES[r]] + NR)) * model.log_tau.exp(); tgt = torch.zeros_like(Lg); tgt[torch.arange(B), T[:, 4 * i + r]] = 1; loss = loss + F.binary_cross_entropy_with_logits(Lg, tgt)
    return loss / (4 * n)
def prog_walk(z, L):
    B = z.shape[0]; n = L // 4; out = torch.zeros(B, L, dtype=torch.long, device=dev); rr = z
    for j in range(n):
        rr = hop(rr, PLUS + NR); i = n - 1 - j
        for r in range(4): out[:, 4 * i + r] = light(hop(rr, ROLE_R[ROLES[r]] + NR)).argmax(1)
    return out
SYMIDS = torch.tensor([sid[s] for s in SYM], device=dev)
def rand_progs(B, n_ins): return SYMIDS[torch.tensor(rng.integers(0, len(SYM), (B, 4 * n_ins)), device=dev)]
with torch.no_grad():
    ns = [int(x) for x in rng.integers(0, 10 ** NP, 300)]; before = (num_walk(num_chain(digs(ns))) == digs(ns).flip(1)).all(1).float().mean().item()
print(f"BEFORE continuation: numbers walk back exactly {before:.3f} (old rows and operators carried over; {len(SYM)} symbol rows and 4 role relations added, random)", flush=True)
(HERE / a.out).mkdir(parents=True, exist_ok=True); best_state, best_score = None, -1
if a.eval_only: best_state = torch.load(HERE / a.out / "ckpt.pt", map_location=dev, weights_only=False)["model"]; a.steps = 0
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(1, a.steps + 1):
    ns = [int(x) for x in rng.integers(0, 10 ** NP, 256)]; D = digs(ns); loss_n = num_walk_loss(num_chain(D), D)                # numbers replayed
    n_ins = int(rng.integers(1, 7)); T = rand_progs(256, n_ins); loss_p = prog_walk_loss(prog_chain(T), T)                          # programs of <= 6 instructions
    loss = loss_n + loss_p; opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if step % 250 == 0 or step == 1:
        with torch.no_grad():
            ns = [int(x) for x in rng.integers(0, 10 ** NP, 200)]; exn = (num_walk(num_chain(digs(ns))) == digs(ns).flip(1)).all(1).float().mean().item(); T = rand_progs(200, 6); exp_ = (prog_walk(prog_chain(T), 24) == T).all(1).float().mean().item()
        if exn + exp_ >= best_score: best_score = exn + exp_; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; torch.save({"model": best_state, "step": step}, HERE / a.out / "ckpt.pt")
        print(f"step {step}/{a.steps}  loss numbers {loss_n.item():.4f} programs {loss_p.item():.4f}  exact: numbers(9d) {exn:.3f}  programs(6 instr) {exp_:.3f}  lr 0.001  tau {model.log_tau.exp().item():.1f}  elapsed {time.time()-t0:.0f}s  ETA {(time.time()-t0)/step*(a.steps-step):.0f}s", flush=True)
model.load_state_dict(best_state); model.eval(); res = {"config": vars(a), "rows": len(NODES)}
with torch.no_grad():
    for L in range(1, NP + 1):
        ns = [int(x) for x in rng.integers(10 ** (L - 1), 10 ** L, 200)]; ex = (num_walk(num_chain(digs(ns))) == digs(ns).flip(1)).all(1).float().mean().item(); res[f"numbers_len{L}"] = ex
    print("numbers walk back exactly by digits: " + " ".join(f"{L}d {res[f'numbers_len{L}']:.3f}" for L in range(1, NP + 1)), flush=True)
    for n_ins in (2, 4, 6, 8):
        T = rand_progs(300, n_ins); ex = (prog_walk(prog_chain(T), 4 * n_ins) == T).all(1).float().mean().item(); res[f"programs_{n_ins}"] = ex; print(f"random programs of {n_ins} instructions: walk back exactly {ex:.3f}", flush=True)
    # promote the five programs as pages into THIS table, execute from memory with the places machine running on THIS table
    sys.argv = ["progmem2", "--eval-only", "--out", "results/progmem2"]; src2 = open(HERE / "progmem2.py").read().split("# ------------------------------------------------------------------ 1. train")[0]; g2 = {}; exec(src2, g2)
    program_symbols, symbols_to_program = g2["program_symbols"], g2["symbols_to_program"]; PAGE = 4; CONT = ["SECTION", "none", "none", "none"]                                                  # measured capacity of a place on the merged table
    def pages_of(syms):
        ins = [syms[i:i + 4] for i in range(0, len(syms), 4)]; out = []
        while ins:
            take = ins[:PAGE - 1] if len(ins) > PAGE else ins; ins = ins[len(take):]; out.append([x for gg in take for x in gg] + (list(CONT) if ins else []))
        return out
    LTM, labels, plen, PROGRAMS_mem = [], [], [], {}
    for name, p in found.items():
        syms = program_symbols(p); back_all = []
        for i, pg in enumerate(pages_of(syms)):
            T = torch.tensor([[sid[x] for x in pg]], device=dev); z = prog_chain(T); LTM.append(z[0]); labels.append(f"{name}#{i}"); plen.append(len(pg))
            back = [NODES[j] for j in prog_walk(z, len(pg))[0].tolist()]; back_all += back[:-4] if back[-4:] == CONT else back
        p2 = symbols_to_program(back_all, p["kind"]); ok = (p2["init"], p2["body"], p2["outro"], p2["stop"]) == (p["init"], p["body"], p["outro"], p["stop"]); PROGRAMS_mem[name] = p2 if ok else p
        res[f"promote_{name}"] = bool(ok); print(f"program '{name}' promoted into the one table as {len(pages_of(syms))} pages; walked back -> {'identical' if ok else 'DIFFERENT'}", flush=True)
    torch.save({"model": model.state_dict(), "config": {**pcfg, "np": NP}}, HERE / a.out / "path.pt")                    # the merged table IS the places machine's path model
    sys.argv = ["places", "--only-check", "--path", "results/onemem/path.pt"]; src = open(HERE / "places.py").read().split("# ------------------------------------------------------------------ 1. the machine")[0]; g = {}; exec(src, g)
    g["PROGRAMS"].update(PROGRAMS_mem)
    for name in found:
        v = g["verify"](PROGRAMS_mem[name], name, lengths=(1, 3, 5, 7), n=100); res[f"exec_{name}"] = v; print(f"executing '{name}' from the one memory: exact " + " ".join(f"{L}d {e:.3f}" for L, e in v.items()), flush=True)
    # a NEW program with a fresh random row as its name: can a CALL to it be read back with no training at all?
    new_row = cnorm(torch.randn(1, M, dtype=torch.complex64, device=dev)); model.E.data = torch.cat([model.E.data, new_row]); model.n_entities += 1; NODES.append("plus"); nid["plus"] = len(NODES) - 1; sid["plus"] = nid["plus"]
    T = torch.tensor([[sid[x] for x in ["CALL", "plus", "s0", "s1"]] + [sid[x] for x in ["CALL", "s2", "none", "none"]] + [sid[x] for x in ["ANSWER", "s2", "none", "none"]]], device=dev)
    back = [NODES[j] for j in prog_walk(prog_chain(T), 12)[0].tolist()]; ok = back[1] == "plus"; res["new_name_row_readable_without_training"] = bool(ok)
    print(f"a new program name as a freshly appended random row, referenced by a CALL, read back through the existing role operators without training: {'yes' if ok else 'no'} (read: {back[:4]})", flush=True)
save_json(res, HERE / a.out / "onemem.json"); print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)
