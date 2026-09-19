"""Type sentences to the reader. Numerals are handed over as digit characters (the machine reads a numeral digit by digit);
new words become fresh rows. Definitions persist for the session, so 'to double something , add it to itself .' followed
by 'double 34324 .' works across lines. Usage:  python say.py "add 3421 and 57 ." "to double something , add it to itself ." "double 34324 ."
                                      python say.py            (interactive)"""
import sys, json, re, torch
ARGS = [x for x in sys.argv[1:]]; sys.argv = ["read_math.py", "--lm-resume", "results/read_math_k32/lm.pt", "--lk", "32", "--check-only"]
src = open("read_math.py").read(); src = src[:src.index("with torch.no_grad():\n    for kinds, label in")]
g = {"__name__": "__main__"}; exec(compile(src, "read_math.py", "exec"), g)
V, VOCAB, vid, NF, LM, M, E, chain, cnorm, dev, execute, answers_of, State, LT, GT, EQ = [g[k] for k in ("V", "VOCAB", "vid", "NF", "LM", "M", "E", "chain", "cnorm", "dev", "execute", "answers_of", "State", "LT", "GT", "EQ")]
READER = json.load(open("results/read_math_ladder/read_math.json"))["reader"]["program"]
LABELS = {}                                                                                                          # every found program is callable by its label; unary ones (found from worked examples) take the sentence's number
for n_, p_ in json.load(open("results/places/found_programs.json")).items():
    if n_ not in g["PROGRAMS"]: g["PROGRAMS"][n_] = p_
    if p_.get("arity") == 1: LABELS[n_] = (n_, ["ARG", 0])
def tokenize(text):
    """words on spaces; a numeral is its digit characters; punctuation split off"""
    out = []
    for w in re.findall(r"[A-Za-z]+|\d+|[.,?]", text):
        out += list(w) if w.isdigit() else [w]
    return out
class Session:
    """one shared token table across sentences: stored words + fresh rows for new words (kept, so a defined name stays the same row)"""
    def __init__(self): self.fresh_words = []; self.fresh = cnorm(torch.randn(NF, LM, dtype=torch.complex64, device=dev)); self.st = State(1); self.defs = {}   # defs: name token -> (cmd, operands); the machine's episode state holds one definition, so the session loads the one a sentence mentions (owed: definitions as pages in the one memory)
    def ids(self, toks):
        ids = []
        for t in toks:
            if t in vid: ids.append(vid[t])
            else:
                if t not in self.fresh_words: assert len(self.fresh_words) < NF, "too many new words for one session"; self.fresh_words.append(t)
                ids.append(V + self.fresh_words.index(t))
        return ids
    def say(self, text):
        toks = tokenize(text); ids = self.ids(toks); self.st = State(1)
        for t, w in zip(ids, toks):
            if w in LABELS and t >= V and t not in self.defs: self.defs[t] = LABELS[w]
            if t in self.defs: self.st.def_name[0] = t; self.st.def_cmd[0], self.st.def_ops[0] = self.defs[t]
        class Ep: pass
        ep = Ep(); ep.B = 1; ep.cand = torch.cat([E()[None], self.fresh[None]], 1); ep.cmask = torch.ones(1, V + NF, dtype=torch.bool, device=dev); ep.sents = [[ids]]
        wid = torch.tensor([ids], device=dev); mask = torch.ones(1, len(ids), dtype=torch.bool, device=dev); ep.z = [chain(ep.cand[0][wid], mask)]
        with torch.no_grad(): calls, ops, bnp = execute([READER], ep, 0, self.st); out = answers_of(calls, ops, ep, bnp, self.st)[0]
        name = {LT: "smaller", GT: "bigger", EQ: "equal"}.get(out, out) if isinstance(out, int) and out in (LT, GT, EQ) else out
        dn = int(self.st.def_name[0]); new_def = dn >= V and dn not in self.defs
        if new_def: self.defs[dn] = (self.st.def_cmd[0], self.st.def_ops[0])
        return " ".join(toks), ("(no answer — " + (f"defined {self.fresh_words[dn - V]} = {self.st.def_cmd[0]} {self.st.def_ops[0]}" if new_def else "nothing called") + ")") if out is None else name
s = Session()
if ARGS:
    for text in ARGS: t, r = s.say(text); print(f"{t}\n  -> {r}")
else:
    for line in sys.stdin:
        if line.strip(): t, r = s.say(line.strip()); print(f"  -> {r}", flush=True)
