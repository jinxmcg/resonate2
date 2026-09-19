"""CLUTRR text reading helpers: tokenisation, gold parsing, the cue-based fact reader (extract_all), and the path search over facts traversed either way. The inverse() table is DIAGNOSTIC ONLY (scoring against the gold convention); the system never uses it."""
import csv, re, ast, collections, json
from common import HERE, DATA
RELS=json.load(open(str(DATA / "clutrr_structured.json")))["rels"]; RELSET=set(RELS)
def load(split): return list(csv.DictReader(open(str(DATA / f"{split}.csv"))))
def tok(s): s=s.replace("’","'"); return re.findall(r"\[[^\]]+\]|[A-Za-z]+(?:-in-law)?|'s|[.,;!?]", s)
def sentences(story): return [x for x in re.split(r"(?<=[.!?])\s+", story.strip()) if x.strip()]
def parse(r):
    g=dict(x.split(":") for x in r["genders"].split(",")); names=list(g); ed=ast.literal_eval(r["story_edges"]); et=ast.literal_eval(r["edge_types"])
    gold={(names[i],names[j]):t for (i,j),t in zip(ed,et)}; q=ast.literal_eval(r["query"]); return names, gold, g, q, r["target_text"]
PAIR={"father":"son","mother":"son","son":"father","daughter":"father","brother":"brother","sister":"brother","husband":"husband","wife":"husband","grandfather":"grandson","grandmother":"grandson","grandson":"grandfather","granddaughter":"grandfather","uncle":"nephew","aunt":"nephew","nephew":"uncle","niece":"uncle","father-in-law":"son-in-law","mother-in-law":"son-in-law","son-in-law":"father-in-law","daughter-in-law":"father-in-law"}
FEM={"son":"daughter","father":"mother","brother":"sister","husband":"wife","grandson":"granddaughter","grandfather":"grandmother","nephew":"niece","uncle":"aunt","son-in-law":"daughter-in-law","father-in-law":"mother-in-law"}
def inverse(r, gender_of_subject):
    """(a, r, b) means a's r is b; the inverse (b, r', a): r' depends on a's gender. DIAGNOSTIC ONLY (world knowledge)"""
    m=PAIR[r]; return FEM[m] if gender_of_subject=="female" else m
def extract(sent):
    t=tok(sent); names=[]; R=None; flip=False
    for i,w in enumerate(t):
        if w.startswith("["): names.append(w[1:-1])
        elif w.lower() in RELSET and R is None:
            R=w.lower(); seen=len(names)
            if i>0 and t[i-1]=="'s" and seen>=2: flip=True
            if i+1<len(t) and t[i+1].lower()=="of" and seen==1: flip=True
    if R is None or len(names)<2: return None
    a,b=names[0],names[1]; return (b,R,a) if flip else (a,R,b)
def story_facts(r): return [f for s in sentences(r["story"]) for f in [extract(s)] if f]
def path(facts, a, b, maxlen=10):
    """BFS over extracted facts traversed either way; returns [(rel, dir)] or None"""
    adj=collections.defaultdict(list)
    for s,rl,o in facts: adj[s].append((o,rl,+1)); adj[o].append((s,rl,-1))
    from collections import deque; Q=deque([(a,[])]); seen={a}
    while Q:
        x,p=Q.popleft()
        if x==b: return p
        if len(p)>=maxlen: continue
        for y,rl,d in adj[x]:
            if y not in seen: seen.add(y); Q.append((y,p+[(rl,d)]))
    return None
if __name__=="__main__":
    for split in ("train","test"):
        rows=load(split); c=collections.Counter(); pc=collections.Counter(); byL=collections.defaultdict(list)
        for r in rows:
            names,gold,g,q,tgt=parse(r)
            for s in sentences(r["story"]):
                ns=[w[1:-1] for w in tok(s) if w.startswith("[")]; pair={(x,y) for x in ns for y in ns if x!=y}; gf=[(x,y,gold[(x,y)]) for (x,y) in pair if (x,y) in gold]; f=extract(s)
                if not gf: c["no gold; none" if f is None else "no gold; spurious"]+=1; continue
                gx,gy,gr=gf[0]
                if f is None: c["missed"]+=1
                elif f==(gx,gr,gy): c["exact"]+=1
                elif f==(gy,inverse(gr,g[gx]),gx): c["exact (inverse form)"]+=1
                else: c["wrong"]+=1
            p=path(story_facts(r), q[0], q[1]); L=len(ast.literal_eval(r["edge_types"])); byL[L].append(p is not None); pc["path found" if p else "no path"]+=1
        tot=sum(c.values()); print(f"{split} sentences: " + "  ".join(f"{k}: {v/tot:.3f}" for k,v in c.most_common()))
        print(f"{split} stories: query path recoverable from extracted facts (either direction): {pc['path found']/len(rows):.3f}; by hops: " + " ".join(f"{L}:{sum(v)/len(v):.2f}" for L,v in sorted(byL.items())))

POSS={"his","her","their","my","our"}
def extract_all(sent):
    """one fact per relation word: owner/object by local cues. 's before R -> owner = nearest name before; 'of' after R -> owner = nearest name after 'of',
    object = nearest name before; otherwise owner = nearest name before R (the subject), object = nearest name after R"""
    t=tok(sent); low=[w.lower() for w in t]; isn=[w.startswith("[") for w in t]; facts=[]
    def name_before(i):
        for j in range(i-1,-1,-1):
            if isn[j]: return t[j][1:-1]
    def name_after(i):
        for j in range(i+1,len(t)):
            if isn[j]: return t[j][1:-1]
    for i,w in enumerate(low):
        if w not in RELSET: continue
        if i>0 and t[i-1]=="'s": owner=name_before(i); obj=name_after(i)
        elif i+1<len(t) and low[i+1]=="of": owner=name_after(i+1); obj=name_before(i)
        else: owner=name_before(i); obj=name_after(i)
        if owner and obj and owner!=obj: facts.append((owner,w,obj))
    return facts
def story_facts_all(r): return [f for s in sentences(r["story"]) for f in extract_all(s)]
