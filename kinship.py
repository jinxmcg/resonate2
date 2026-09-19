"""Learning what the kinship words MEAN, so that composition is derivable and the learned table is no longer needed.
World knowledge given (the family algebra): a relation is a short path in a family — steps up (parent), down (child),
side (sibling), sp (spouse) — plus the gender of the person reached; two paths compose by concatenation and reduce by
the family's rules (parent's child = sibling, child's parent = spouse, sibling's sibling = sibling, sibling's parent =
parent, child's sibling = child, parent's spouse = parent, spouse's child = child, spouse's spouse = self).
Learned from the training stories: the descriptor (path, gender) of each of the 20 relation words, by coordinate-descent
search for the assignment that makes the most training compositions (r1, r2 -> r3) consistent under the algebra.
Then CLUTRR is answered by walking the chain and composing DESCRIPTORS, naming the result (or abstaining if no word
has that descriptor). Reported: descriptors found, training consistency, CLUTRR test exact / abstain / wrong by length."""
import json, itertools, collections, numpy as np, ast, csv, time
from common import HERE, DATA
rng = np.random.default_rng(0); t0 = time.time()
D = json.load(open(str(DATA / "clutrr_structured.json"))); RELS = D["rels"]
rows = {s: list(csv.DictReader(open(str(DATA / f"{s}.csv")))) for s in ("train", "validation", "test")}
comp = collections.Counter()
for r in rows["train"]:
    et = ast.literal_eval(r["edge_types"])
    if len(et) == 2: comp[(et[0], et[1], r["target_text"])] += 1
TRIPLES = sorted({k for k in comp}); print(f"training compositions (length-2 stories): {len(TRIPLES)} distinct (r1, r2 -> r3)")
comp3 = collections.Counter()
for r in rows["train"]:
    et = ast.literal_eval(r["edge_types"])
    if len(et) == 3: comp3[(et[0], et[1], et[2], r["target_text"])] += 1
CHAINS3 = sorted({k for k in comp3}); print(f"training compositions (length-3 stories): {len(CHAINS3)} distinct")
# ------------------------------------------------------------------ the family algebra
STEPS = ["up", "down", "side", "sp"]; PAIRS = [(x, y) for x in STEPS for y in STEPS]; OPTIONS = ["keep", "self", "up", "down", "side", "sp"]
RULES = {}                                                                                            # learned below: pair -> reduction (absent = keep)
def set_rules(assign):
    RULES.clear()
    for p, o in zip(PAIRS, assign):
        if o == "self": RULES[p] = ()
        elif o != "keep": RULES[p] = (o,)
KINDS = [("up",), ("down",), ("side",), ("sp",), ("up", "up"), ("down", "down"), ("up", "side"), ("side", "down"), ("sp", "up"), ("down", "sp")]
def reduce_path(p):
    p = list(p); changed = True
    while changed:
        changed = False
        for i in range(len(p) - 1):
            if (p[i], p[i + 1]) in RULES: p[i:i + 2] = list(RULES[(p[i], p[i + 1])]); changed = True; break
    return tuple(p)
def compose(d1, d2):
    """descriptors (path, gender): the composed path reduced; the gender is that of the person reached (d2's)"""
    return (reduce_path(d1[0] + d2[0]), d2[1])
# ------------------------------------------------------------------ gender of each word is observable: the gender of the person an edge of that word reaches
gen_of = collections.defaultdict(collections.Counter)
for r in rows["train"]:
    et = ast.literal_eval(r["edge_types"]); ed = ast.literal_eval(r["story_edges"]); g = dict(x.split(":") for x in r["genders"].split(",")); names = {}
    story = r["story"]; import re; ments = re.findall(r"\[(.*?)\]", story)
    # node ids -> names: CLUTRR numbers nodes by first mention order in the story
    order = []; [order.append(m) for m in ments if m not in order]
    for (i, j), w in zip(ed, et):
        if j < len(order) and order[j] in g: gen_of[w][g[order[j]][0]] += 1
# more direct: the query's second name is the person the TARGET word describes; its gender is given
tg = collections.defaultdict(collections.Counter)
for r in rows["train"]:
    q = ast.literal_eval(r["query"]); g = dict(x.split(":") for x in r["genders"].split(","))
    if q[1] in g: tg[r["target_text"]][g[q[1]][0]] += 1
GENDER = {w: (tg[w].most_common(1)[0][0] if tg[w] else gen_of[w].most_common(1)[0][0] if gen_of[w] else "m") for w in RELS}; print("gender of each word from the data:", {w: GENDER[w] for w in RELS})
# ------------------------------------------------------------------ learn the path of each word: consistency with training compositions, distinct words distinct meanings
def score_desc(desc):
    ok = sum(1 for r1, r2, r3 in TRIPLES if compose(desc[r1], desc[r2]) == desc[r3]) + sum(1 for a_, b_, c_, t in CHAINS3 if compose(compose(desc[a_], desc[b_]), desc[c_]) == desc[t]) * 0.5
    coll = len(RELS) - len(set(desc.values())); return ok - 1000 * coll
# alternating search with a prior: start from a generic family algebra, fit the words, let the inconsistencies revise the rules, refit, repeat
PRIOR = {("up", "down"): "side", ("down", "up"): "sp", ("side", "side"): "side", ("side", "up"): "up", ("down", "side"): "down", ("up", "sp"): "up", ("sp", "down"): "down", ("sp", "sp"): "self"}
assign = [PRIOR.get(p, "keep") for p in PAIRS]; set_rules(assign); desc = None; best = -1e9
def fit_words(desc0):
    bo, bd = -1e9, None
    for restart in range(60):
        d = dict(desc0) if desc0 and restart == 0 else {w: (KINDS[rng.integers(len(KINDS))], GENDER[w]) for w in RELS}; b = score_desc(d)
        for sweep in range(30):
            improved = False
            for w in rng.permutation(RELS):
                cur = d[w]; scores = []
                for k in KINDS: d[w] = (k, GENDER[w]); scores.append((score_desc(d), (k, GENDER[w])))
                top = max(x[0] for x in scores); cands = [c for sc, c in scores if sc == top]; d[w] = cands[int(rng.integers(len(cands)))] if top >= b else cur
                if top > b: b = top; improved = True
            if not improved: break
        if b > bo: bo, bd = b, dict(d)
    return bd, bo
def fit_rules(d, assign):
    b = score_desc(d)
    for sweep in range(10):
        improved = False
        for pi in rng.permutation(len(PAIRS)):
            cur_o = assign[pi]; scores = []
            for o in OPTIONS: assign[pi] = o; set_rules(assign); scores.append((score_desc(d), o))
            top = max(x[0] for x in scores); cands = [o for sc, o in scores if sc == top]; assign[pi] = cands[int(rng.integers(len(cands)))] if top > b else cur_o; set_rules(assign)   # rules change only on STRICT improvement: minimal edits to the prior
            if top > b: b = top; improved = True
        if not improved: break
    return assign, b
for it_ in range(4):
    desc, best = fit_words(desc); assign, best = fit_rules(desc, assign); print(f"  alternation {it_ + 1}: score {best:.1f}  rules changed from prior: {[(f'{x}∘{y}', o) for (x, y), o in zip(PAIRS, assign) if PRIOR.get((x, y), 'keep') != o]}", flush=True)
best_desc, best_rules = desc, list(assign)
desc = best_desc; set_rules(best_rules); print("learned rules (pair -> reduction, 'keep' omitted):", {f"{x}∘{y}": o for (x, y), o in zip(PAIRS, best_rules) if o != "keep"})
ok = sum(1 for r1, r2, r3 in TRIPLES if compose(desc[r1], desc[r2]) == desc[r3]); coll = len(RELS) - len(set(desc.values()))
ok3 = sum(1 for a_, b_, c_, t in CHAINS3 if compose(compose(desc[a_], desc[b_]), desc[c_]) == desc[t])
print(f"learned words + rules (alternating, prior-corrected): {ok}/{len(TRIPLES)} length-2 and {ok3}/{len(CHAINS3)} length-3 training compositions consistent, {coll} meaning collisions ({time.time()-t0:.0f}s)")
for w in RELS: print(f"  {w:16s} = {'-'.join(desc[w][0]):10s} {desc[w][1]}")
NAME = {}
for w in RELS: NAME.setdefault(desc[w], w)
# ------------------------------------------------------------------ CLUTRR by composing meanings (chain stories and ALL stories via the edge list along the query path)
def answer(edge_types):
    d = desc[edge_types[0]]
    for e in edge_types[1:]: d = compose(d, desc[e])
    return NAME.get(d)
res = {"descriptors": {w: ["-".join(desc[w][0]), desc[w][1]] for w in RELS}, "rules": {f"{x}∘{y}": o for (x, y), o in zip(PAIRS, best_rules)}, "train_consistency": [ok, len(TRIPLES), ok3, len(CHAINS3)]}
for split in ("validation", "test"):
    by = collections.defaultdict(list)
    for it in D["data"][split]:
        a_ = answer(it["edges"]); by[it["L"]].append("exact" if a_ == it["target"] else "abstain" if a_ is None else "wrong")
    res[f"{split}_chain_by_length"] = {L: {"n": len(v), "exact": v.count("exact") / len(v), "abstain": v.count("abstain") / len(v), "wrong": v.count("wrong") / len(v)} for L, v in sorted(by.items())}
    print(f"{split} chain stories, composing learned meanings — exact/abstain/wrong: " + "  ".join(f"{L}: {v.count('exact')/len(v):.2f}/{v.count('abstain')/len(v):.2f}/{v.count('wrong')/len(v):.2f} (n={len(v)})" for L, v in sorted(by.items())))
json.dump(res, open(str(HERE / "results/kinship.json"), "w"), indent=1); print(f"saved ({time.time()-t0:.0f}s)")
