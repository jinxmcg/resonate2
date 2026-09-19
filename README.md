# ResonatE 2 — a memory that computes

Code, results and logs behind [resonate.page/resonate2](https://resonate.page/resonate2/): the ResonatE knowledge-graph
memory of [jinxmcg/resonate](https://github.com/jinxmcg/resonate) — unit-norm rows, block-diagonal relation operators,
**unchanged** — asked to hold numbers it never stored, to learn arithmetic from three worked examples, to take
instructions in words, and to reason over kinship chains it has never seen. One GTX 1080 Ti, negatives kept; ten seeds on the CLUTRR headline, one seed elsewhere.

The full research record, part by part with every table and every negative, is in [`RESEARCH.md`](RESEARCH.md).
The argument for the write-up is in [`paper_log.md`](paper_log.md). What was optimised and what was not is in
[`optimizations.md`](optimizations.md).

## The two results people ask about first

**CLUTRR from the graph, 0.99 at ten hops** (wrong: ≤ 1.2 % at 4–10 hops and 2.2 % ± 2.3 at three hops over ten seeds — see the table; every "1.000" in this repository is sampled exactness on n cases per length plus a regression set, not a proof over all numbers) (`kinship_ops.py`, `results/kin_full_k8/`, `logs/kin_full_k8.log`).
Twenty kinship words become twenty ResonatE operators on 64-dimensional places (k = 8, four 16×16 blocks per word,
~20k complex parameters). A training story of 2–3 hops says only: the product of the chain's operators must land where
the target word's operator lands. Test stories of 2–10 hops, all 1,146 of them:

| hops | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| exact, seed 0 | 1.00 | 0.86 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.94 | **0.99** |
| wrong, seed 0 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 |
| exact, ten seeds | 1.000 ± .000 | 0.893 ± .039 | 0.995 ± .000 | 0.989 ± .000 | 0.982 ± .005 | 0.985 ± .009 | 0.992 ± .008 | 0.976 ± .013 | **0.988 ± .008** |
| wrong, ten seeds | 0.000 | 0.022 ± .023 | 0.004 ± .002 | 0.009 ± .003 | 0.012 ± .006 | 0.006 ± .006 | 0.001 ± .004 | 0.004 ± .004 | 0.003 ± .006 |

Ten seeds (`kin_seeds.sh`, `results/kin_seeds/`, `logs/kin_seed_*.log`; ten-hop exact per seed 0.992 0.983 0.983 0.983
1.000 0.992 1.000 0.992 0.983 0.975). Note the wrong rate at three hops, 0.022 ± 0.023 over seeds: seed 0 abstained on the
ambiguous chain, other seeds answer it one way or the other. Every remaining miss is a chain with two logically valid answers (*wife → son →
grandmother* is a mother or a mother-in-law; *… → daughter → sister* a daughter or a niece) and the model's two top
words are those two, 0.00–0.05 apart — kinship is one-to-many and the benchmark labels one member. We tried a
second, set-valued scorer by simulating families (`kin_valid.py`) and dropped it: CLUTRR's own rules differ from
real-world kinship in places (a wife's sister is labelled *sister*; there are no in-law siblings), so a second ground
truth would be ours, not the benchmark's.

The rest abstains, and the abstentions are the benchmark's own ambiguities (all fifteen at length 3 are the chain
*wife → son → grandmother*, labelled *mother* ten times and *mother-in-law* five). Best published on this split at ten
hops: CTP_A 0.90 (Minervini et al., ICML 2020). Training: 20,000 steps, **339 s on a 1080 Ti** while another run shared
the GPU (the first run of this code, 4,000 steps in about a minute, gave 0.78; the whole gap was convergence and read
noise, not capacity). Nothing in `resonate.py` was changed for any of this.

```
python clutrr_prepare.py                                              # downloads CLUTRR, builds data/clutrr/clutrr_structured.json
python kinship_ops.py --k 8 --steps 20000 --probes 64 --seed 0 --out results/kin_full_k8
python kinship_ops.py --k 8 --steps 4000 --probes 64 --shuffle-targets --out results/kin_k8_shuffled   # control: 0.00 exact, 1.00 abstain
```

**Arithmetic from three worked examples, then in words** (`places.py`, `onemem.py`, `read_math.py`, `say.py`).
Eighteen stored rows (digits, place anchors) and two moves hold every number below a billion as a place you walk to
(grown to a trillion by continuation in 513 s: `grow.py --np 12`, `results/grow_1e12/`; and, without place rows, the digit chain `digit_chain.py` — note that the published k = 12 chain checkpoint misreads runs of nines from 9 digits, the k = 20 checkpoint used for the program results does not, and the trainer now samples runs); add, sub, cmp, muld, mul are
programs over lights found from three worked examples each, exact at 1–9 digits, stored as pages in the same table —
and, unchanged, exact at 1–12 digits on the trillion table (`results/places_1e12_found.json`); and a reader found the same way takes commands and definitions:

```
python say.py "add 3421 and 57 ." "to double something , add it to itself ." "double 34324 ."
  -> 3478   -> (defined double = add ['ARG','ARG'])   -> 68648
```

## Layout

- `common.py` — loads `resonate.py` from `$RESONATE_PY` or from `../resonate/resonate.py` (clone jinxmcg/resonate next to this repo).
- Numbers: `labeled_areas.py`, `number_kg.py`, `two_relations.py`, `two_part.py`, `order.py`, `compare.py`, `add_program.py`, `grow.py` (Parts 2–10).
- Programs: `machine.py`, `controller.py`, `discover.py`, `places.py` (the places machine and the arithmetic curriculum; `results/places/found_programs.json`), `progmem.py`, `progmem2.py`, `onemem.py` (Parts 11–16).
- Language on the machine: `read.py`, `read_math.py`, `say.py` (Part 20).
- Reasoning: `story.py` (Part 17), `clutrr.py` (Part 18, structured), `kinship.py` (the hand-written algebra, a negative), `kinship_ops.py` (Part 19, the 0.99), `clutrr_read.py`, `clutrr_text.py`, `clutrr_convert.py`, `clutrr_local.py`, `clutrr_cue_table.py` (Part 21, from text — a negative with a located cause).
- `results/` — every run's JSON, and the small checkpoints (found programs, operator tables, the one memory). `logs/` — the run logs the tables were read from.
- `AGENTS.md` — the working agreement the runs were made under (status every two minutes, unbuffered logs, resumable checkpoints).

## Setup

```
git clone https://github.com/jinxmcg/resonate        # resonate.py is imported by path; nothing in it is modified
git clone https://github.com/jinxmcg/resonate2 && cd resonate2
python -m venv .venv && . .venv/bin/activate && pip install torch numpy
python clutrr_prepare.py
```

PyTorch 2.6 with CUDA was used; the runs assume a GPU. The Part-1 experiments (`train_v4.py`, `ground*.py`, `equals.py`,
`eval_spot.py`) used a private generator's data (`$REASONING_DATA`) and are kept for the record only.

## What is and is not established

Ten seeds on the CLUTRR headline, one seed on the arithmetic and program-search tables. The machine's primitives and the word→program aliases are ours; the programs are found, the
instruction set is not; "found at generation 1" means the search selected the program from worked traces. Owed:
seeds on the arithmetic tables, NPI-style and VSA baselines on the same tasks, a real-graph result. From raw text CLUTRR
is a negative (0.08 exact, 0.90 abstain) whose cause is located in the benchmark's fact form (`RESEARCH.md`, Part 21b).

MIT license.
