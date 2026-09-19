"""The set of logically valid answers for a CLUTRR chain, by simulation — scoring only (world knowledge: what the twenty words
mean as moves in a family). Random families (3-4 generations, couples with 1-4 children, everyone married); a chain is
realised by following each word's move from a start person; every realisation's end person is then named by every word
whose move from the start reaches it. The valid set is the union over many families. Kinship is one-to-many ('mother >
daughter' is you or your sister), so a chain can have several valid answers; the benchmark's label is one of them."""
import json, collections, numpy as np
from common import HERE, DATA
rng = np.random.default_rng(0)
class Family:
    def __init__(self, gens=4, kids=(1, 4)):
        self.par = {}; self.kids = collections.defaultdict(list); self.sp = {}; self.g = {}; n = [0]
        def person(gender): i = n[0]; n[0] += 1; self.g[i] = gender; return i
        def marry(i): j = person("f" if self.g[i] == "m" else "m"); self.sp[i] = j; self.sp[j] = i; return j
        def grow(i, depth):
            if depth == 0: return
            j = marry(i)
            for _ in range(int(rng.integers(kids[0], kids[1] + 1))):
                c = person(rng.choice(["m", "f"])); self.par[c] = (i, j); self.kids[i].append(c); self.kids[j].append(c); grow(c, depth - 1)
        grow(person(rng.choice(["m", "f"])), gens)
    def parents(self, x): return list(self.par.get(x, ()))
    def sibs(self, x): return [s for p in self.parents(x)[:1] for s in self.kids[p] if s != x]
    def move(self, x, w):
        g = self.g; P = self.parents(x); K = self.kids[x]; S = self.sibs(x); SP = [self.sp[x]] if x in self.sp else []
        table = {"mother": [p for p in P if g[p] == "f"], "father": [p for p in P if g[p] == "m"], "daughter": [k for k in K if g[k] == "f"], "son": [k for k in K if g[k] == "m"],
                 "sister": [s for s in S if g[s] == "f"], "brother": [s for s in S if g[s] == "m"], "wife": [s for s in SP if g[s] == "f"], "husband": [s for s in SP if g[s] == "m"],
                 "grandmother": [q for p in P for q in self.parents(p) if g[q] == "f"], "grandfather": [q for p in P for q in self.parents(p) if g[q] == "m"],
                 "granddaughter": [c for k in K for c in self.kids[k] if g[c] == "f"], "grandson": [c for k in K for c in self.kids[k] if g[c] == "m"],
                 "aunt": [s for p in P for s in self.sibs(p) if g[s] == "f"], "uncle": [s for p in P for s in self.sibs(p) if g[s] == "m"],
                 "niece": [c for s in S for c in self.kids[s] if g[c] == "f"], "nephew": [c for s in S for c in self.kids[s] if g[c] == "m"],
                 "mother-in-law": [p for s in SP for p in self.parents(s) if g[p] == "f"], "father-in-law": [p for s in SP for p in self.parents(s) if g[p] == "m"],
                 "daughter-in-law": [self.sp[k] for k in K if k in self.sp and g[self.sp[k]] == "f"], "son-in-law": [self.sp[k] for k in K if k in self.sp and g[self.sp[k]] == "m"]}
        return table[w]
    def names(self, x, y, words): return {w for w in words if y in self.move(x, w)}
def valid_sets(chains, words, n_families=60):
    """chain (tuple of words) -> set of valid answer words, union over families and start persons"""
    out = {c: set() for c in chains}; fams = [Family() for _ in range(n_families)]
    for fam in fams:
        people = list(fam.g)
        for c in chains:
            for x in people:
                ends = {x}
                for w in c:
                    ends = {e2 for e in ends for e2 in fam.move(e, w)}
                    if not ends: break
                for e in ends:
                    if e != x: out[c] |= fam.names(x, e, words)
    return out
if __name__ == "__main__":
    D = json.load(open(DATA / "clutrr_structured.json")); words = D["rels"]; test = D["data"]["test"]
    chains = sorted({tuple(it["edges"]) for it in test}); VS = valid_sets(chains, words)
    gold_in = np.mean([it["target"] in VS[tuple(it["edges"])] for it in test]); multi = np.mean([len(VS[tuple(it["edges"])]) > 1 for it in test])
    print(f"{len(chains)} distinct test chains; gold answer inside the simulated valid set for {gold_in:.3f} of test stories; stories whose chain has more than one valid answer: {multi:.3f}")
    print("examples:", {" > ".join(c): sorted(VS[c]) for c in chains[:3]}, {"wife > son > grandmother": sorted(VS[("wife", "son", "grandmother")])} if ("wife", "son", "grandmother") in VS else "")
    json.dump({" > ".join(c): sorted(v) for c, v in VS.items()}, open(HERE / "results/kin_seeds/valid_sets.json", "w"), indent=0)
