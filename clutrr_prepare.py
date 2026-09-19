"""Download CLUTRR (gen_train23_test2to10: train on 2-3 hops, test on 2-10) and build data/clutrr/clutrr_structured.json,
the file every CLUTRR script here reads: for each story the relation-word sequence along the query path ('edges'), the
target word and the hop count; the 20 relation words; and the composition table used by the structured machine of
Part 18 (length-2 training pairs, plus pairs derivable through length-3 training stories). Nothing else is derived."""
import csv, ast, json, urllib.request, collections
from common import DATA
BASE = "https://raw.githubusercontent.com/kliang5/CLUTRR_huggingface_dataset/main/gen_train23_test2to10/"
DATA.mkdir(parents=True, exist_ok=True)
for split in ("train", "validation", "test"):
    p = DATA / f"{split}.csv"
    if not p.exists(): print("downloading", split, flush=True); urllib.request.urlretrieve(BASE + f"{split}.csv", p)
data = {}
for split in ("train", "validation", "test"):
    rows = list(csv.DictReader(open(DATA / f"{split}.csv"))); items = []
    for r in rows:
        ed = ast.literal_eval(r["story_edges"]); et = ast.literal_eval(r["edge_types"]); qe = ast.literal_eval(r["query_edge"])
        assert all(ed[i][1] == ed[i + 1][0] for i in range(len(ed) - 1)) and ed[0][0] == qe[0] and ed[-1][1] == qe[1], "every CLUTRR story is a walk from the query's first node to its second"
        items.append({"edges": et, "target": r["target_text"], "L": len(et)})
    data[split] = items; print(split, len(items), "stories; by hops:", dict(sorted(collections.Counter(i["L"] for i in items).items())), flush=True)
rels = sorted({w for s in data for i in data[s] for w in i["edges"]} | {i["target"] for s in data for i in data[s]})
table = {}
for i in data["train"]:
    if i["L"] == 2: table.setdefault("|".join(i["edges"]), i["target"])
for i in data["train"]:
    if i["L"] == 3:
        a, b, c = i["edges"]
        if f"{a}|{b}" in table: table.setdefault(f"{table[f'{a}|{b}']}|{c}", i["target"])
        if f"{b}|{c}" in table: table.setdefault(f"{a}|{table[f'{b}|{c}']}", i["target"])
json.dump({"rels": rels, "table": table, "data": data}, open(DATA / "clutrr_structured.json", "w"))
print(f"{len(rels)} relation words; composition table {len(table)} entries; wrote {DATA / 'clutrr_structured.json'}", flush=True)
