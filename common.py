"""V4 (separated state/relation learning) over the existing ResonatE space — my own implementation.

memory   : a ResonatE table (unit-norm complex rows, block-diagonal complex operators) over the GROUNDED numerals only,
           trained as a number graph from the depth-1 training facts: relation (op, b) maps row a to row c  (a op b = c).
reasoner : primitives add / multiply (and, in stage 2, > < =). Op_p(b) = sum_g w_pg(E_b) G_pg  — the operator is
           GENERATED from the operand's activation (16 generators per primitive, our biokg finding), then applied to
           the other operand:  response = cnorm(Op_p(E_b) E_a).  The response is a native state: it can be read
           (full light over the table, a spot if the number has no row) or fed back as an operand.
loop     : an expression tree is folded bottom-up through query-local working memory, one node per iteration; only
           the root gets a loss (intermediates unsupervised).  The schedule comes from the expression's binding.
Data and generator of ../geocore/reasoning are used read-only through their own model_input whitelist.
"""
import importlib.util, json, math, sys, types
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

import os
HERE = Path(__file__).resolve().parent
THEIRS = Path(os.environ.get("REASONING_DATA", "/mnt/nvme990/geocore/reasoning"))          # the dev_v1 generator data used in Part 1 (optional)
DATA = HERE / "data" / "clutrr"                                                              # CLUTRR csvs + clutrr_structured.json (see clutrr_prepare.py)
RESONATE = Path(os.environ.get("RESONATE_PY", "")) if os.environ.get("RESONATE_PY") else next((p for p in (HERE.parent / "resonate" / "resonate.py", Path("/mnt/nvme990/geocore/resonate/resonate.py")) if p.exists()), HERE.parent / "resonate" / "resonate.py")   # jinxmcg/resonate cloned next to this repo, or $RESONATE_PY
sys.dont_write_bytecode = True                             # never leave __pycache__ in their folders


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


model_input = _load("their_interface", THEIRS / "reasoning_data/interface.py").model_input
_res = _load("resonate_core", RESONATE); ResonatE, cnorm = _res.ResonatE, _res.cnorm


# ------------------------------------------------------------------ data (read-only)
def records(split, stage=1, data="dev_v1"):
    return [json.loads(l) for l in open(THEIRS / "data" / data / f"stage{stage}" / f"{split}.jsonl")]


def manifest(data="dev_v1"):
    return json.load(open(THEIRS / "data" / data / "manifest.json"))


def grounded_labels(data="dev_v1"):
    p = manifest(data)["pools"]; return [str(n) for n in sorted(set(p["G"]) | set(p["A"]) | set(p["B"]))]


def leaves(e):
    return [e] if isinstance(e, str) else [x for a in e["args"] for x in leaves(a)]


def depth(e):
    return 0 if isinstance(e, str) else 1 + max(depth(a) for a in e["args"])


def facts(recs):
    """depth-1 binary facts (a, op, b, c) from compose records, via the whitelist; c from the target (training use only)."""
    out = []
    for r in recs:
        mi = model_input(r)
        if mi.task != "compose": continue
        e = mi.inputs["expression"]
        if depth(e) == 1 and len(e["args"]) == 2: out.append((e["args"][0], e["op"], e["args"][1], "".join(r["target"]["digits"])))
    return out


# ------------------------------------------------------------------ memory
class Memory(nn.Module):
    """The ResonatE table for the grounded numerals + the (op, b) relation vocabulary of the training facts (with reverses)."""
    def __init__(self, labels, relations, k=8, block_size=16):
        super().__init__()
        self.labels = list(labels); self.id = {l: i for i, l in enumerate(self.labels)}
        self.relations = list(relations); self.rid = {r: i for i, r in enumerate(self.relations)}; nb = len(self.relations)
        self.model = ResonatE(len(self.labels), 2 * nb, k=k, block=True, block_size=block_size)
        self.nb = nb; self.k = k; self.block_size = block_size

    @property
    def M(self): return self.model.m
    @property
    def E(self): return cnorm(self.model.E)
    def embed(self, labels): return self.model.embed(torch.tensor([self.id[l] for l in labels], device=self.model.E.device))
    def hop(self, z, r): return self.model.hop(z, r)
    def light(self, z): return torch.real(z @ self.E.conj().t()) * self.model.log_tau.exp()      # (B, N) the full light

    def add_rows(self, new_labels):
        """Stage-2 grounding: append rows (new numerals, or the truth symbols) without touching existing ones."""
        with torch.no_grad():
            e = torch.randn(len(new_labels), self.M, dtype=torch.complex64, device=self.model.E.device) / math.sqrt(self.M)
            self.model.E = nn.Parameter(torch.cat([self.model.E.data, cnorm(e)])); self.model.n_entities = len(self.model.E)
        for l in new_labels: self.id[l] = len(self.labels); self.labels.append(l)


# ------------------------------------------------------------------ reasoner
PRIMS = ("add", "multiply", ">", "<", "=")


class Reasoner(nn.Module):
    """One generator bank per primitive; the mixing weights are a function of the operand's activation."""
    def __init__(self, M, block_size, n_gen=16, prims=PRIMS, hidden=128, n_place=5, both=False):
        super().__init__()
        self.M, self.bs, self.nblk, self.n_gen, self.both = M, block_size, M // block_size, n_gen, both; self.prims = list(prims); self.pid = {p: i for i, p in enumerate(self.prims)}
        if prims == ("combine",): self.pid = {"add": 0, "multiply": 0, "combine": 0}                   # unlabeled: one bank, the operation is inferred from the areas
        q = torch.linalg.qr(torch.randn(len(prims), n_gen, self.nblk, block_size, block_size, dtype=torch.complex64))[0]
        self.G = nn.Parameter(q)                                                                # (P, n_gen, nblk, bs, bs), unitary init
        self.w = nn.ModuleList([nn.Sequential(nn.Linear((4 if both else 2) * M, hidden), nn.GELU(), nn.Linear(hidden, n_gen)) for _ in prims])
        self.n_place = n_place; self.P = nn.Parameter(torch.linalg.qr(torch.randn(n_place, self.nblk, block_size, block_size, dtype=torch.complex64))[0])   # unary place primitives

    def place(self, k, z):
        """decomposition primitive: the activation of the 10^k term of z (read at the anchor rows d*10^k, or row 0)"""
        return cnorm(torch.einsum("kij,bkj->bki", self.P[k], z.reshape(z.shape[0], self.nblk, self.bs)).reshape(z.shape[0], -1))

    def op(self, p, zb, za=None):
        """Operator generated from the operand activation(s). (B, nblk, bs, bs)"""
        feat = torch.view_as_real(zb).flatten(1) if not self.both else torch.cat([torch.view_as_real(za).flatten(1), torch.view_as_real(zb).flatten(1)], 1)
        w = torch.softmax(self.w[p](feat), -1)                                                # (B, n_gen)
        return torch.einsum("bg,gkij->bkij", w.to(torch.complex64), self.G[p])

    def apply(self, p, za, zb):
        """response = cnorm(Op_p(zb) za)"""
        h = self.op(p, zb, za); zab = za.reshape(za.shape[0], self.nblk, self.bs)
        return cnorm(torch.einsum("bkij,bkj->bki", h, zab).reshape(za.shape[0], -1))

    def fold(self, expr, operand, n_ary="left"):
        """Bottom-up evaluation of a binding tree. operand(label) -> (1, M) activation. Returns the root state and the
        working-memory trace [(op, deps, state)]. n-ary add/multiply are folded left (commutative, order is randomised
        upstream by the data)."""
        trace = []
        def go(e):
            if isinstance(e, str): return operand(e), f"input:{e}"
            p = self.pid[e["op"]]; acc, dep = go(e["args"][0])
            for a in e["args"][1:]:
                zb, dep_b = go(a); acc = self.apply(p, acc, zb); trace.append((e["op"], (dep, dep_b), acc)); dep = f"write:{len(trace) - 1}"
            return acc, dep
        z, _ = go(expr); return z, trace


# ------------------------------------------------------------------ metrics
def rank_metrics(light, target):
    """exact top-1 and MRR of the target row within the full light (all rows are candidates)."""
    s = light.gather(1, target[:, None]); rank = 1 + (light > s).sum(1)
    return (rank == 1).float(), 1.0 / rank.float()


def spot_consistency(states, ids, rng):
    """Unnamed results: cosine between states that should be the same number vs. states of different numbers, and a
    dimension-aware baseline (random pairs of different identities). Returns dict."""
    z = cnorm(states); ids = np.asarray(ids); same, diff = [], []
    for i in range(len(ids)):
        j = np.where(ids == ids[i])[0]; j = j[j != i]
        if len(j): same.append(float(torch.real(z[i] @ z[j[rng.integers(len(j))]].conj())))
        k = np.where(ids != ids[i])[0]
        if len(k): diff.append(float(torch.real(z[i] @ z[k[rng.integers(len(k))]].conj())))
    if not same: return {"n_same": 0}
    same, diff = np.array(same), np.array(diff); thr = (same.mean() + diff.mean()) / 2
    return {"n_same": len(same), "cos_same": float(same.mean()), "cos_diff": float(diff.mean()),
            "purity": float(((same > thr).mean() + (diff <= thr).mean()) / 2)}


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True); json.dump(obj, open(path, "w"), indent=1)
