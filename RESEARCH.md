# reasoning_ant — my own V4 over the ResonatE space, on the other agent's data

Date: 2026-09-14/15. Data and generator: `/mnt/nvme990/geocore/reasoning/data/dev_v1` (read-only, through their own
`model_input` whitelist; no bytecode written there). Memory: the real `ResonatE` class from `geocore/resonate/resonate.py`,
imported by path. Everything else here is mine. GPU: 1080 Ti, no rental. Total compute ≈ 25 runs × 4–7 min.

## The paradigm under test (the user's V4)

- **Memory** = the ResonatE table: one unit-norm complex row per grounded numeral (221 rows: pools G, A, B). Rows are
  *areas*; the numeral is an optional label. The memory must **not** see `+` or `×`: it only sees which numerals occur
  together (`3, 100, 4, 10, 7` as a bag — no operator, no order, no tree), then it is frozen.
- **Reasoner** = the only thing that learns operations. Input: the activations of the operand rows and the grouping
  `((3,100),(4,10),7)`. Output: a *response activation* in the same space. It is read against the frozen table
  (bright at `347`'s row if one exists, a spot otherwise) or fed back as an operand for the next node.
- **Loop**: the expression tree is folded bottom-up through query-local working memory, one node per iteration; the only
  loss is at the root (intermediates unsupervised).

Reasoner design: `response = cnorm(Op(z_a, z_b) · z_a)` with `Op = Σ_g w_g(z_a, z_b) G_g` — a bank of 16 block-diagonal
complex generators (our biokg finding), mixed by a small controller that reads the operand activations. `--ops labeled`
gives one bank per named primitive (add / multiply); `--ops none` gives one unlabeled bank and the controller must infer
the operation from the areas alone; `--flat` also removes the grouping. Decomposition uses five unary `place_k`
primitives read at the anchor rows (`236 → 200, 30, 6`, empty places → row `0`).

## Runs

| run | memory | ops given | buckets | k | gens | train compose¹ | test B `a+b` | test B place-value | test A decompose | C spot purity |
|---|---|---|---|---|---|---|---|---|---|---|
| ground_k{2,3,4,6,8} (step 1) | free operator per `(op,b)`, trained as a number graph | — | depth-1 | 2–8 | — | 1.00 | 0.00–0.09 | — | — | 0.55–0.74 |
| strict_k8 operands all | co-occurrence (operands), frozen | labeled | all | 8 | 16 | 0.98 | 0.066 | 0.012 | 0.000 | 0.58 |
| strict_k8 operands pv | co-occurrence (operands), frozen | labeled | pv | 8 | 16 | 1.00² | 0.012 | 0.012 | 0.000 | 0.55 |
| strict_k8 equation all | co-occurrence (+result/anchors), frozen | labeled | all | 8 | 16 | 1.00 | 0.136 | 0.012 | 0.000 | 0.59 |
| strict_k8 equation pv | co-occurrence (+result/anchors), frozen | labeled | pv | 8 | 16 | 1.00² | 0.070 | 0.012 | 0.000 | 0.61 |
| unlab_k8 operands all | co-occurrence, frozen | **none** | all | 8 | 16 | 0.85 | 0.047 | 0.000 | 0.000 | 0.55 |
| unlab_k8 operands pv | co-occurrence, frozen | **none** | pv | 8 | 16 | 0.77² | 0.012 | 0.000 | 0.000 | 0.62 |
| flat_k8 operands all | co-occurrence, frozen | none, no grouping | all | 8 | 16 | 0.82 | 0.035 | 0.000 | 0.000 | 0.76 |
| flat_k8 operands pv | co-occurrence, frozen | none, no grouping | pv | 8 | 16 | 0.78² | 0.004 | 0.000 | 0.000 | 0.69 |
| joint_k2 g4 all | rows shaped by the loss (control) | labeled | all | 2 | 4 | 0.63 | 0.043 | 0.000 | 0.000 | 0.74 |
| joint_k2 g16 all | rows shaped by the loss (control) | labeled | all | 2 | 16 | 0.94 | **0.140** | 0.000 | 0.000 | 0.66 |
| joint_k2 g16 pv | rows shaped by the loss (control) | labeled | pv | 2 | 16 | 1.00² | 0.016 | 0.000 | 0.000 | 0.52 |
| joint_k4 g16 all | rows shaped by the loss (control) | labeled | all | 4 | 16 | 1.00 | 0.093 | 0.000 | 0.000 | 0.61 |
| joint_k4 g16 pv | rows shaped by the loss (control) | labeled | pv | 4 | 16 | 1.00² | 0.043 | 0.000 | 0.000 | 0.70 |
| **oracle_lab all** | fixed line `e^{iθn}` (control) | labeled | all | 8 | 16 | 1.00 | **0.397** (MRR 0.52) | **0.098** (val 0.159) | 0.000 (val 0.054) | 0.58 |
| **oracle_unlab all** | fixed line `e^{iθn}` (control) | **none** | all | 8 | 16 | 0.90 | **0.175** (MRR 0.29) | 0.000 (val 0.023) | 0.000 (val 0.054) | 0.58 |

¹ exact top-1 over all 221 rows. ² trained on composition-only buckets (dense basics + place value + decomposition); the
train number is on those buckets. Chance for exact top-1 is 1/221 = 0.0045. "pv" = `dense_basic,place_value,decompose`.
B `a+b` = `arithmetic_variation [B]` (e.g. `223+13=236`, carries included); B place-value = `(2×100)+(3×10)+6=236` where
`236`'s row was positioned only by co-occurrence / decomposition; A decompose = canonical place-value terms of a row whose
decomposition was never trained; C spot purity = same-number vs different-number cosine separation of the root state for
strictly unseen numbers (no row); 0.5 = none. Label-free exact test (`eval_spot.py`: compose into a spot, decompose the
spot with `place_k`, all places right) = **0.000 on every bucket of every run**, including training records.

## What happened

1. **Grounding the memory as a number graph memorises at every width** (step 1, `ground.py`): one free operator per
   `(+b)` / `(×b)` sees ~5 facts on average and can map them arbitrarily; held-out facts and 2-hop composition sit at
   chance for k = 2…8. So V4-v1 "ground, freeze, then learn relations" has nothing usable to freeze if grounding means a
   relation table — the operator has to be a function of the operand's activation. (`ground.py`, `results/ground_*`)

2. **Strict V4 fits the training trees on a frozen co-occurrence table and generalises nothing.** Train compose reaches
   0.98–1.00 (labeled) or 0.77–0.85 (unlabeled) — the frozen table is expressive enough to compose over — but every
   held-out bucket is at chance: B place-value 0.00–0.01, A decomposition 0.00, spot→decompose 0.00, C spot purity
   0.55–0.62. Bag type (operands vs. equation), bucket selection, op labels, grouping: none of it moves the held-out
   numbers. The training fit itself is knife-edge: the composed spot has cosine 0.37–0.79 with its target row and is top-1
   only because τ runs to ≈200.

3. **Why (diagnosed, not guessed).** An operation can generalise only over areas that are *systematic*: `236`'s area
   must be predictable from `200, 30, 6`'s (or `2, 3, 6`'s). Rows placed by co-occurrence are, for numbers, essentially
   arbitrary (corr(row cosine, log|a−b|) = −0.27 at k=8), so every primitive — `add`, `multiply`, `place_k` — degenerates
   into a lookup table over the rows it was trained on. `place_k` on the row of `151` (an A number, decomposition never
   trained) returns `1, 70, 0`. Nothing in the strict setup can supply that structure, because the only thing that
   *creates* it is the operation acting on the rows during training.

4. **The one non-chance number is in the control.** With the rows shaped by the loss at k=2 (M=4), 16 generators, all
   buckets: B `a+b` = 0.140 exact / 0.171 MRR — addition facts start to generalise, place-value composition still does
   not. Same regime where addition became a rotation on our number line two days ago: structure appears when the
   operation is allowed to shape the areas, and only under a bottleneck.

5. **Oracle control** (`--oracle-line`: rows fixed to `z_n[j] = e^{iθ_j n}`, never learned; encodes magnitude, so a
   control, not a candidate): the *same* reasoner generalises. Held-out sums with carries: 0.397 exact / 0.52 MRR
   (labeled) and 0.175 (unlabeled) vs 0.01–0.14 on every learned memory; held-out place-value composition 0.10–0.16
   (labeled) vs 0.00–0.01; τ settles at 35–45 instead of ≈200. So the reasoner does learn a transferable `+` when the
   areas are systematic, and inferring the operation from the areas alone (no op names) costs about half of it.
   Two things stay at the floor even here: (a) decomposition of untrained rows and the label-free spot→decompose test
   (0.000) — a single linear block-diagonal `place_k` map cannot extract a digit from a line (that is a modular,
   non-linear readout), so this is a limit of my decomposition primitive, not of the memory; (b) unseen-number spot
   consistency (purity 0.58) — the oracle line is smooth, so nearby numbers have nearby spots and the same/different
   separation is small by construction. Two runs on the oracle (composition-only buckets) were stopped unfinished when
   the decision was taken to stop spending compute.

## What this means for the paradigm

Separating "areas" (memory) from "operations" (reasoner) is not free. The areas must carry structure the operations can be
rules over; bags of co-occurring numerals do not supply it, and on this data no width, bag, bucket, or labelling choice
changes that. The negative is clean and reproducible (`*.sh`, `logs/`, `results/*/results.json`). The two things that
did work are the ones that break the strict separation: rows shaped by the loss under a bottleneck (k=2), and a memory
with a built-in line (oracle). The protocol's "do not fill up memory and assume relationships will flow" is exactly
what this measures.

## Suggestions on their data / generator (nothing changed there)

- The canonical place-value form `(3×100)+(4×10)+(7)` exists for every selected identity (both leaf orders), no
  `3×10×10+…` forms — but it is a minority: train has 234 place-value rows vs 765 arithmetic variations (`55+96`) and
  144 grouping changes. A per-bucket weight/count in `data_v1.json` would let composition be the primary family.
- `347` has no records (eval identities are sampled 24 per band). An `eval_identity_include` list would guarantee the
  canonical example is in the unseen test.
- Named held-out composition (B pool) is a strong test only if the target row is *systematically* placed. With a memory
  that never sees operations it cannot be, so that bucket measures the memory, not the reasoner. The label-free test
  (compose → spot → decompose with the place primitives → exact canonical terms) is the one that matches "the computed
  point just is"; it is implemented in `eval_spot.py` and should be their primary exact metric for C numbers.
- B rows get training signal only as operands (64/85) and through decomposition (85/85); 21 B numbers never co-occur
  with anything, so the B composition bucket has a floor by construction.

## Files

`common.py` (data access via their whitelist, `Memory` over the real `ResonatE`, `Reasoner`, metrics) ·
`ground.py` (step 1, number-graph grounding, emergence checks) · `ground_bag.py` (strict grounding: co-occurrence bags;
`--oracle-line` control) · `train_v4.py` (V4 trainer: `--frozen`, `--ops labeled|none`, `--flat`, `--buckets`, loop
with endpoint loss, evaluation per bucket) · `eval_spot.py` (label-free compose→decompose exact test) ·
`sweep.sh strict.sh unlabeled.sh oracle.sh` (what was run) · `logs/` · `results/<run>/{results.json, v4.pt, spot_decompose.json}`.

---

# Part 2 — numbers as labeled areas (2026-09-15)

The user's fix for the diagnosed failure: big numbers get *systematic* areas because their areas are **constructed the
way their labels are**. "Thirty" = 3 bound to *tens*; "347" = 3 hundreds + 4 tens + 7. `labeled_areas.py`.

- rows: the ten digit symbols only. Operators: S (successor), T/H/K/X (tens … ten-thousands), with the exact adjoint
  (`tied_reverse`) so unbinding is the memory's own inverse hop.
- `area(n) = cnorm(Σ_{nonzero places k} hop(E_{d_k}, scale_k))` — binding + superposition, both already in the space.
  No number above 9 has a row. Decoding = unbind each place, read the digit rows; absent place = light below a threshold
  calibrated once on the grounding numerals.
- The memory learns only **label-level facts**: counting 0..19 (S maps area(n) to area(n+1)) and that the grammar's
  binding is invertible (unbinding a constructed area recovers its digits). It never sees + or ×.
- Step 3 is a **fixed program** (zero learned parameters) over the primitives: per place unbind both digits, walk the
  successor line, read units directly and the carry by unbinding tens, rebind; ×10^k = re-binding; general × = repeated
  addition. Scored **exactly by decoding the result area** — no row, no label. Chance ≈ 0.

| exact | learned V4 (best of 14) | algebra, untrained | areas k=8 (M=64) | **areas k=12 (M=144)** | areas k=16 | k=12 + chain-9 counting |
|---|---|---|---|---|---|---|
| construct-decode, never-seen numbers | — | 0.79 | 0.88 | **0.93** | 0.85 | 0.88 |
| decompose never-seen C (test) | 0.000 | 0.83 | 0.96 | **0.98** | 0.95 | 0.95 |
| decompose held-out A rows | 0.000 | 0.90 | 1.00 | **1.00** | 1.00 | — |
| `(d×100)+(d×10)+d`, strictly unseen (val / test) | 0.000 | 0.000 | 0.46 / 0.39 | **1.00 / 0.87** | 1.00 / 0.75 | 0.01 / 0.03 |
| B place-value (val / test) | 0.01 | 0.000 | 0.47 / 0.45 | **1.00 / 1.00** | 1.00 / 1.00 | 0.01 / 0.04 |
| `a+b`, strictly unseen, carries (val / test) | 0.000 | 0.03 / 0.005 | 0.39 / 0.37 | **0.69 / 0.59** | 0.62 / 0.51 | 0.03 / 0.03 |
| `a+b`, B pool (val / test) | 0.14 (row hit) | 0.12 | 0.46 / 0.44 | **0.74 / 0.71** | 0.71 / 0.65 | 0.15 / 0.16 |
| single-digit sums ≤ 20 (train, sanity) | — | 0.27 | 0.75 | 0.84 | 0.82 | 0.35 |

Diagnosis on k=8 (`results/areas_k8`): (a) **walk drift** — S is exact for one step (top-1 of 20) but lands at cosine
only 0.65 to the true next area; nine hops drift to 0.18; single-digit sums are exact for walks ≤ 3 and fall to 0 at
8–9 (carry cases 0.36 vs no-carry 0.85). (b) **cross-talk** — decode is exact for ≤ 3 non-zero places, 0.99 for 4,
0.82 for 5 at M=64; nearest-other cosine stays ≈ 0.75 at every M, so this is the free scale operators not scattering
cleanly rather than width alone. Training counting on S^d chains (`--chain 9`) did **not** fix drift — it wrecked the
line (loss plateaus at 0.36; a single free block operator cannot be exact for all nine chain lengths) — recorded as a
negative. Untried, cheap: keep S and the scales unitary (a parametrisation, not a data constraint), so a hop is a
rotation and cannot drift.

What this is and is not: exact arithmetic on numbers that have no row, verified by decoding with no label, on a memory
that never saw an operation; the *control* (which primitive, in which order) is a hand-written program, declared as
such. It answers "do the areas support the operations?" (yes, once they are constructed systematically) and not yet
"can the program be learned" — that is the next, separate question, and it now has a clean target to imitate.

**Unitary operators (`--unitary`, `exp(A−Aᴴ)`), k=8 and k=12 — negative.** Construct-decode of never-seen numbers
0.59 / 0.65 (vs 0.93 free), unseen place-value 0.42 / 0.37 (vs 0.87), unseen `a+b` 0.28 / 0.22 (vs 0.59), single-digit
sums 0.69 / 0.64 (vs 0.84). Reading: the free scale operators use non-unitarity to *attenuate* cross-talk between bound
places (a rotation cannot), and a unitary S does not stop drift because the 0–19 line is not a rotation orbit — areas
10–19 are bundles (`1 ten + d`), so no rotation maps `E_9` exactly onto `norm(T·E_1 + E_0)`. Drift comes from how the
line is constructed, not from S's freedom. Untried next: 0–19 as atoms (rows, as the language has words for them), S a
line over those rows, grammar facts on those rows as unbinding targets; construction starts at 20. `results/areas_*`.

---

# Part 3 — the number KG (2026-09-15, `number_kg.py`)

The user's next step: **KG the composition** with the human place concepts up to millions. Entities = numbers; every
number has exactly seven edges to the digit rows `0–9` (`units … millions`, zeros included) plus `successor`. Rows for
0–9,999 minus a random 30 % (held out entirely) and 1,000 sparse numbers each of 5, 6, 7 digits. Trained like any KG
(k=12, 8 relations with learned reverses, 74,876 edges). **A number with no row = the AND of its seven digit probes**
(adding the reverse-hop probes); decoding = the seven forward hops read on the digit rows.

| numbers with no row, exact decode of their AND-spot | 2 digits | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|
| learned reverse (run 1) | **1.000** | **0.996** | **1.000** | **0.992** | **0.994** | **0.930** |
| adjoint instead of learned reverse | 0.000 | 0.047 | 0.331 | 0.000 | 0.164 | 0.514 |
| learned reverse + successor trained on chains ≤ 9 | 1.000 | 0.945 | 0.996 | 0.949 | 0.260 | 0.454 |

n = 28 / 256 / 2,710 / 491 / 500 / 500. Controls: random states decode as the intended number 0.000; stored numbers
decode 1.000 from their row and 0.999 from their spot. dev_v1: unseen decomposition 1.000, held-out A decomposition
1.000 (run 1). **The reverse operators must be learned**: the adjoint gives the right *scores over stored rows* (the
light identity) but not a usable *state* — `H_k H_kᵀ ≠ I` for a many-to-one operator, so forward hops off an adjoint-
built spot fail; the learned reverse puts the probe in the numbers' area.

Addition through the successor walk fails in every variant (1-digit sums 0.65 / 0.61 / 0.25; anything longer ≈ 0),
and drags the dev_v1 composition buckets down with it (0.06–0.25) because `300+40+7` walks `0+4`. With the
`labeled_areas` chain and unitary runs, that is three parametrisations in which a free block operator hopped d times is
not an adder: the counting line is exact for one step and is not a walk. Open decision (the user's): the digit addition
table as memory facts (`3 —plus_4→ 7`, 100 facts, the protocol's own dense 0–20 band) or as symbols inside the program.
Runs: `results/number_kg_k12{,_adjoint,_chain9}`, logs in `logs/`.

---

# Part 4 — the step back: two relations, anchors as nodes, one light (2026-09-15, `two_relations.py`)

Nodes `0–9, 10, 100, …, 1000000` (16 rows); relations `+`, `×` only; `347` built bigger to smaller, one hop per step,
output fed back, operands entering a hop as a superposition (`hop(E_3 ⊕ E_100, ×)` … `hop(acc ⊕ E_7, +)`). Trained so
the reverse hops reconstruct the operands (loss → 0.0001). Evaluation with no reasoner and no iteration: one light.

| | exact set (top-|parts| = parts) | precision |
|---|---|---|
| state light | 0.000 | 0.56 |
| one hop `+⁻¹`, light | 0.008 | 0.78 |
| one hop `×⁻¹`, light | 0.000 | 0.65 |

`347` → `+⁻¹` light: `7` (+15.4), everything else ≈ −6. `120123` → `3`. `1000000` → `0`. One light shows the last
operand added; the earlier steps are carried as an opaque state. `cos(347, 437) = 0.993`, `cos(5060, 6050) = 0.996`,
`cos(347, 743) = 0.43` (last digit differs); same-digits-permuted 0.55 vs random pairs 0.49.

Why, structurally: a hop is linear, so a hop on a superposition is a superposition of hops — the number becomes the bag
`{3, 100, 4, 10, 7}` with no record of the pairing. Superposition carries *what* is present, never *what is paired with
what*. **A scale has to be an operator (or a relation), not a node**: binding `3` to hundreds needs `×100` applied *to*
`E_3` — which is what `labeled_areas` (scale operators) and `number_kg` (place relations) did, and why they compose
unseen numbers exactly. With anchors as entities and one shared `×`, the pairing is unrepresentable in this algebra.

**Correction (same day): the graph is directed, and that carries the pairing.** The bag result above came from an
objective that never needed the direction, so `+` collapsed to near-identity. Training the *directed reverse walk*
instead (`--walk`: after j reverse `+` hops and one `×⁻¹`, the light must show the j-th term's operands and nothing
else) gives per-step reads 1.00 1.00 1.00 1.00 1.00 0.17 0.18 — the last two fail because the chain started with
`acc = term_6`, so the first `+` combined two terms symmetrically. Giving the first term its own `+` step (every term at
a distinct depth):

| reverse walk, one hop + one light per step | all 7 terms | per step (units → millions) |
|---|---|---|
| train | 1.000 | 1.00 1.00 1.00 1.00 1.00 1.00 1.00 |
| **test, never seen** | **1.000** | 1.00 1.00 1.00 1.00 1.00 1.00 1.00 |

`347`: step 1 lights `7` (+9.5, rest ≈ −10.5); step 2 `4, 10`; step 3 `3, 100`. 16 nodes, two relations, no row for any
number, no reasoner, nothing arithmetic learned. `cos(347, 437)` = 0.982 — the states are near-parallel as vectors and
perfectly distinct under the walk: the position marker is a small rotation; cosine is the wrong similarity for a
directed chain, the walk is the right one. `results/two_relations_k12_walk2`. So the rule from the bag run is amended:
superposition alone loses the pairing; **superposition + direction keeps it**.

---

# Part 5 — is an equation an edge? (2026-09-15, `equals.py`)

Same 16-node chain memory plus one relation `=`: `region(a+b) →=→ region(numeral a+b)`, trained on single-place digit
equations only (524, carries included at every place; 131 held out), read by walking `hop(region, =)` back.

| | exact |
|---|---|
| digit equations, trained | 0.252 (no-carry 0.265, carry 0.234) |
| digit equations, held out | 0.023 |
| 2–7-digit sums, never trained, any number of carries | 0.000–0.007 |
| numeral chains walk back (sanity, same run) | 1.000 |

`=` cannot even memorise the single-place facts. A linear `=` on a superposition yields `f(a)+g(b)`; matching the
region of `a+b` for all pairs would need the digit rows collinear with their value, which the walk-back (digits as
distinct symbols) forbids. So: an equation is not a shared destination (a region encodes its path), not an edge
between regions (linearity), and the table-as-edges (`plus_5`) was rejected by design. **Evaluation is not in the map.**
Composition, decomposition, facts and dependency are directed-graph moves (all exact this week); the thing that reads
two lit digits and produces a third is the program — the boundary starts at the first digit sum, before the carry.

---

# Part 6 — a region with two parts: path and value (2026-09-15, `two_part.py`)

Path part = the exact chain of Part 4 (frozen). Value part = 63 term anchors `d×10^k` as learnable phase vectors,
a number's value = the **binding** (elementwise product) of its term anchors. Trained only on single-place digit
equations (`a·10^k + b·10^k = (a+b)·10^k`, carries included, 20 % held out) plus InfoNCE distinctness over random
numbers. Both losses → 0 in one minute.

| | learned | random | oracle line |
|---|---|---|---|
| anchors form a line: `v(d,k)⊙v(1,k)` vs `v(d+1,k)` | 1.000 | −0.002 | 1.000 |
| `=` as the light, held-out digit equations (vs neighbours) | **1.000** | 0.26 | 1.000 |
| `=` as the light, never-trained 2–7-digit sums, any carries | **1.000** (cos 1.0000) | — | 1.000 |
| digits read off a value by lights | 0.000 | 0.000 | 0.998 |
| never-trained sums, exact via value→digits→path→walk | 0.000 | 0.000 | 0.993–1.000 |

`=` is the light: the value of `a` bound with the value of `b` **is** the value of `a+b`, exactly, for any sum below ten
million, carries free, emerged from digit equations alone. What the learned code lacks is **order**: `cos(n, n+1)` =
−0.07 (oracle ≈ 0.9); the distinctness objective pushed neighbours apart like any other pair, so "is the remainder
below 10^k" has no light and digits cannot be read back. Additivity and order are separate structures; equations give
the first, comparison facts (the protocol's stage 2, `347 < 1000`) are the candidate source of the second. Next run.

---

# Part 7 — stage 2 on the value code: adding order as facts (2026-09-15, `order.py`)

Continue the trained value code (no restart) with `x < y` facts at every scale, the comparison as a light (the
difference `v_y ⊙ conj(v_x)` shining on a learnable positive region `P`), explicit replay of the digit equations.

| | 3k steps, lr 3e-3 | 8k steps, lr 1e-2 | slopes free (line reparametrised) |
|---|---|---|---|
| `=` retention (held-out equations / unseen sums) | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| `x<y` 1 apart / 10 apart / ≥100 apart | 1.000 / 0.87 / chance | 1.000 / 0.96 / chance | 0.80 / 0.48 / chance |
| `cos(n, n+1)` | −0.065 | −0.065 | −0.015 |
| digits read off a value, closed loop | 0.000 | 0.000 | 0.000 |

Dynamic update itself works: new facts went in, nothing known was lost. But order does not come from facts by
gradient on this code: global order needs a slow frequency (slope ≈ π/10⁷ per unit) and the comparison loss is
oscillatory in the slope, so descent from fast slopes never finds it; only `P` learns, memorising small differences.
Additivity emerged because every slope satisfies the equations; order did not because almost none satisfies the
comparisons. Ways forward: comparison as a program on the path code (walk back from the top, first differing digit,
10-symbol order) — no training; or slow frequencies as an initialisation prior in the value code (a built-in).

---

# Part 8 — comparison as a program on the path code (2026-09-15, `compare.py`)

We already decompose (the walk back), so `x < y` = walk both chains back, look from the most significant place, the
first digit that differs decides — using an order on the ten digit symbols read as a light: one template `Q`, fitted to
the 45 facts `d < d'` (all satisfied; digit lights `0: −3.3 … 9: +2.7`). Path model frozen, nothing else trained.

| unseen pairs (6,000), decided by walking both back | exact |
|---|---|
| all | **1.000** |
| by distance 1 / 10 / 10² / 10³ / 10⁴ / 10⁵ | 1.000 each |
| `x = x` (1,000) | 1.000 |

`999999 < 1000000`, `347 < 437`, `120123 = 120123`. Order that would not come from facts on the value code (Part 7)
takes ten symbols and forty-five facts on the path code, because the places are explicit there. Standing: the path
code composes, decomposes and compares exactly; the value code adds exactly (`=` is the light) but cannot be read back;
the bridge value → digits needs order in the value code.

---

# Part 9 — the loop closes: addition one place at a time (2026-09-15, `add_program.py`)

The missing bridge (value → digits) needed order the value code does not have. Do not ask for it: never form the whole
sum as one value. Walk both operands back (path code, exact) → per place, bind the two digit anchors and the carry in the
value code and read the digit sum by the **equality light against the 20 possible digit sums** (`s ≥ 10` as
`v(1,k+1) ⊙ v(s−10,k)`) → digit and carry → rebuild the result's path → walk it back. Nothing new is trained; the
program reads lights and chooses moves.

| never-trained sums, exact | 1 digit | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| all (1,000 each) | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| 3+ carries | — | — | — | 1.000 (130) | 1.000 (295) | 1.000 (480) | 1.000 (604) |

dev_v1 (the other agent's data): unseen addition 1.000 / 1.000, unseen place-value 1.000 / 1.000, B composition
0.993 / 1.000, A and unseen decomposition 1.000 — every bucket the learned variants of Part 1 scored 0.000 on.
(First run had 19 candidates; `9+9+carry = 19` needs 20 — that was the only source of error, 0.975–0.997 → 1.000.)

**What was built, in one phrase:** a number system with no numbers stored in it — 16 stored symbols and two moves in
which any number below ten million exists as a place you can walk to, walk back from, compare and add, exactly, with the
control a nine-line program over lights. The principles carried out of it: superposition keeps *what*, direction keeps
*pairing*, multiplication keeps *value*; moves live in the memory, decisions live in the program.

---

# Part 10 — growing to a billion by continuation (2026-09-15, `grow.py`)

Two new nodes (`10^7`, `10^8`), 18 new value anchors, chains of 9 terms. Two ways to continue the trained system:

| | only the new rows/anchors learn, moves frozen, new-place examples only | moves learn too, old numbers replayed |
|---|---|---|
| walk back, 1–7 / 8–9 digits | 0.000 / 0.10 | **1.000 / 1.000** |
| held-out equations by place 0…8 | old 1.00, new 0.45 / 0.87 | **1.000 everywhere** |
| never-trained sums, 1–9 digits | 0.000 | **1.000** |
| comparison, distances 1…10⁷ | 0.85 (0.94–0.98 below 10⁶) | **1.000** |

Rows alone cannot be added to a frozen memory when the new knowledge is **depth**: `+` was trained to keep terms
separable for 7 steps and no choice of two new rows changes that property of the move. With the operators allowed to
keep learning and old numbers replayed, everything old is kept and everything new fits in 7 minutes. Also: padding every
number to 9 places changed every old chain (leading zero terms push each term two steps deeper); variable-length chains
would avoid that at the cost of a "anything left?" decision — one for the learned controller.

---

# Part 11 — the reasoning layer, learned (2026-09-15, `controller.py`)

A GRU controller that reads lights and chooses moves. Environment = the exact primitives (`STEP_X/Y` walk one place
back → digit light; `SUM` → 20-candidate equality light; `WRITE d`; `CARRY c`; `STOP`; `STEP_BOTH`; `ANSWER <,>,=`),
plus an "anything left?" light (cosine of the remaining chain to the all-zero remainder) so stopping is a decision from
a light. The controller never sees a number. Trained by imitation of the hand-written programs as a per-state policy
(traces on numbers **< 10⁴ only**), then 3 DAgger rounds (controller-driven episodes, every visited state labelled by
the same policy), then executed alone and scored by exact outcome.

| trained on < 10⁴ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 digits |
|---|---|---|---|---|---|---|---|---|---|
| add (n = 346–500 per length) | 1.000 | 1.000 | 0.994 | 0.998 | **0.992** | **0.996** | **0.993** | 0.792 | 0.728 |
| compare | 1.000 | 1.000 | 0.998 | 0.996 | 0.400 | 0.247 | 0.212 | 0.236 | 0.237 |

Addition generalises to lengths never seen (5–7 digits at 99 %): its loop is local — each cycle needs only the current
lights and the carry. Comparison does not: "first difference from the top", read units-first, is "last difference so
far", which the controller held in its hidden state, and a hidden state trained on ≤ 4 steps does not survive 9. (An
earlier run with a buggy teacher that always stepped 9 times generalised comparison *better*, 0.92 at 5 digits — same
diagnosis from the other side.) Two bugs on the way, both recorded: a batch-level teacher whose stop rule depended on
the batch (labels conflicted with the per-state policy, addition collapsed under DAgger), and a first-step `argmax` of
an all-zero previous action reading as `STEP_X` (teacher 0.058 on its own rollouts). Fix for comparison, untried: an
explicit working-memory verdict slot the controller writes (`SET_LT/GT`) so every decision is local to the lights.

---

# Part 12 — the general working memory: a register machine of lights and moves (2026-09-15, `machine.py`)

K = 4 anonymous slots (a state in the value code + optional symbol tag; cleared per query), three kinds of primitive:
**move** (`LOAD_X/Y→s` walk one place back; `BIND s,t` value binding), **light** (`READ s` over the 0..19 line; every
slot's symbol also shines on the digit-order template `Q`; "anything left?" = peek the remaining places and read their
digit lights), **write** (`SET s←symbol`, `COPY`, `WRITE s` to the result tape, `ANSWER s`, `STOP`). 97 actions,
13 symbols. Nothing in the hardware knows what a carry or a verdict is; addition and comparison are programs.
Controller = GRU (256), imitation of the per-state teacher programs on numbers **< 10⁴** (step accuracy 1.0000),
3 DAgger rounds, then executed alone.

| trained on < 10⁴ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 digits |
|---|---|---|---|---|---|---|---|---|---|
| add (n = 346–500) | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| compare | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

Compared with Part 11 (task-specific slots, verdict in the hidden state: compare 0.24 at 9 digits; add 0.73): with
every piece of context in a slot, every decision is local to the current lights and both programs generalise without
limit in the lengths tested. **Promotion:** 200 sums validated by two routes agreeing (path result vs value-code
equality light): 1.000 validated, 1.000 of those correct, 0.000 correct-but-rejected; 20 written to long-term memory as
labelled rows and found afterwards by a single light. Two teacher bugs fixed on the way (an initialisation rule that
leaked a leading zero; the cosine "anything left" light being blunt for a lone high digit — replaced by a peek).
Principle: **the controller is stateless; the context lives in memory.**

Decision (user): K = 6 slots from here on; the number is arbitrary — a bound on what is held "in hand", not on what a
computation can use (a program can promote a slot to long-term memory and load it back). Status of "hard-coded":
the machine is not (move / light / write only); the *programs* for + and > were hand-written teachers that the
controller learned by imitation — learned to execute, not yet found. Next: subtraction — two memory-side additions
(`UNBIND` = bind with the conjugate; a READ set extended to −10…9), continual learning on the same controller with
replay (retention of + and >), then program *search* from the addition program as a seed with exact outcome as the
score — the first discovery test.

---

# Part 13 — discovering programs from examples (2026-09-15, `discover.py`)

Machine grown to K = 6 (arbitrary) and 144 instructions: moves `LOAD_X/Y`, `BIND`, `UNBIND` (bind with the conjugate —
the memory's inverse); lights `READ` (0..19 line), `READN` (−10..9 line, conjugate anchors); decoders `SETD` (digit of
the last light), `SETC` (its carry/borrow); writes `SET <0/1`, `COPY`, `WRITE`. A program = init + loop body + the slot
whose zero (with "nothing left") ends the loop. Fitness = per-place digit credit + exact bonus on examples only; no
teacher in the loop. Evolutionary search, all candidates of a generation as one batched rollout. Hand-written programs
are used only as ground truth (fitness 2.000, exact 1.000) and to produce demonstrations.

| how the operation entered | given | found | exact, 1–9 digits (4–9 never seen by the search) |
|---|---|---|---|
| **addition from 3 worked examples** (`12+11`, `383+47`, `19+65` as flat step lists, no loop marked) | the steps of 3 sums | gen 6 (~1 min) | **1.000** at every length |
| **subtraction from a neighbour** (addition's program + `(x,y,x−y)` examples) | a program for a related operation | gen 21 (~3 min) | **1.000** at every length |
| addition from nothing (examples only, random programs) | nothing | plateau, 300 generations | — |

The found subtraction is not the hand-written one: it adds the borrow to y's digit and subtracts once
(`BIND 3,2 … UNBIND 0,3`), keeps a permanent zero in a slot for its stop rule, and carries dead instructions — an
evolved idiom. The found addition recovers the exact 8-instruction cycle and its initialisation from the traces and
picks its own stop slot. Blank-slate search has no slope (the six-instruction core gives no credit until complete);
this is the user's point made precise — nobody discovers addition, one is shown a few worked examples or counts —
and the two seeded rows are what learning from examples looks like here. Random search is scaffolding to measure
the landscape; with slope established, the proposer can be learned (the controller with outcome reward) or a
language prior. Files: `results/discover_{add_demos3,sub_seeded,add_unseeded}/`.

---

# Part 14 — the places machine and the arithmetic curriculum (2026-09-15/16, `places.py`)

Every slot holds a **place** (a walkable path state for a number, a bindable value state for a digit, an optional tag);
the operands are places in slots 0 and 1; a program can return a place. Primitives: moves `LOAD s>t` (walk any slot's
place one step back), `BIND/UNBIND`, `SHIFT` (one digit up), `CALL p s,t>u` (run a found program on two places — programs
call programs); lights `READ/READN/READ2` (the 0..19 / −10..9 / 0..99 lines), `READMUL` (the times table, 100 facts),
`ORDER` (writes LT/GT only if the digits differ); decoders `SETD/SETC`; writes `SET`, `COPY`, `WRITE`, `ANSWER`. K = 6
(arbitrary). 576 instructions. Ground-truth programs are used only to check the machine and to produce three worked
examples per operation; the search never sees them. Acceptance requires exactness on held-out cases **two digits longer
than any shown** (a first `mul` accepted without that rule carried an extra `SHIFT`, perfect to 4 digits, 0.5 at 9).

| operation | callable when found | found at | exact 1–8 digits (>3 never shown) | 9 digits |
|---|---|---|---|---|
| add | — | gen 1 | 1.000 | 1.000 |
| sub | add | gen 1 | 1.000 | 1.000 |
| cmp | add, sub | gen 1 | 1.000 | 1.000 |
| muld (× digit) | add, sub, cmp | gen 1 (rotated loop, first product in the init) | 1.000 | 1.000 |
| mul | add, sub, cmp, muld | gen 5 (`SHIFT, CALL add, LOAD, CALL muld` over the found programs) | 1.000 | 0.893 (the memory's 10⁹ range, not the program) |

Retention of every earlier operation after each rung: 1.00. Executor dispatch made general and 10× faster (only the
instructions present in a step are dispatched; all calls to the same sub-program at a step run as one nested batch).
Found programs: `results/places/found_programs.json`. Next: division (`sub`, `cmp`, `muld` callable), programs stored
as chains of instruction-places in the memory itself, and a learned proposer in place of random mutation.

**Open (noted 2026-09-16, not run):** (a) range to 10¹² by continuation — three new nodes (`10⁹, 10¹⁰, 10¹¹`), 27 value
anchors, operators learning with replay (the Part 10 recipe, ~7 min); the found programs must then verify unchanged,
since they reference slots and moves, never the memory's width. (b) range with no ceiling — the recursive grammar
(`n = rest(n)·10 + units(n)`, Part 4 discussion) with a variable-depth walk model and the "anything left?" peek as the
top-of-number light; a larger change, kept for later.

---

# Part 15 — programs in the memory (2026-09-16, `progmem2.py`)

An instruction is **one place**: its opcode and up to three arguments bound to four role *relations* (`OP, R1, R2, R3`) and
superposed — as a number is digits bound to places. A program is a chain of instruction-places (same `+`/`×`, first term
through its own `+`), read back one `+⁻¹` step and four role lights per instruction. The chain model is trained once on
random instruction chains (curriculum to 24 instructions, gradient clipping, best checkpoint by validation); after that,
promoting a program is composition, not training.

Capacity, measured (k = 12, 39 symbol rows): a place holds **6 instructions exactly** (1.000), 8 at 0.12, 16+ at 0.000 —
the earliest-composed instructions wash out first. (A first encoding with one term per *symbol* — 4× deeper — read back
nothing beyond 8 terms; recorded.) So long programs are stored as **pages**: a linked list of places of ≤ 6 instructions,
the last instruction of a page a "continue" marker (an unused instruction shape, no new row).

| program | instructions | pages (rows) | walks back from memory | executed from memory (CALLs read from memory too), exact 1/3/5/7 digits |
|---|---|---|---|---|
| add | 13 | 3 | identical | 1.000 |
| sub | 13 | 3 | identical | 1.000 |
| cmp | 10 | 2 | identical | 1.000 |
| muld | 23 | 5 | identical | 1.000 |
| mul | 22 | 5 | identical | 1.000 |

Long-term memory now holds numbers' parts, facts, and procedures as the same kind of object; `CALL mul` loads the rows
labelled `mul#0…mul#4` and walks them. Open: one memory for digits and instructions (merge by continuation); names as
composed places so a new program needs no new row; a wider program memory (k = 16–20) to raise the page size.

---

# Part 16 — one memory (2026-09-16, `onemem.py`)

The number table of Part 10 (digits, place anchors, `+`, `×`) grown by continuation with 36 instruction-symbol rows
and 4 role relations; old rows and operators keep learning, numbers replayed; 3,000 steps, ~4 min. One table, one `+`,
one `×`: numbers, facts and procedures are the same kind of object in the same space.

| | |
|---|---|
| numbers 1–9 digits walk back (retention) | 1.000 at every length |
| random programs of 2 / 4 / 6 / 8 instructions per place | 1.000 / 1.000 / 0.32 / 0.00 → page size 4 on this table |
| five programs promoted as pages (add 4, sub 4, cmp 3, muld 8, mul 7 rows) | all walk back identical |
| executed from the one memory (the places machine now runs on the merged table; calls read from memory) | 1.000 for all five at 1/3/5/7 digits |
| **new program name = a freshly appended random row; a `CALL` to it read back with no training** | **yes** |

So adding a program is a write: append its pages, and calls to it are readable through the existing role operators.
Per-place capacity is lower on the shared table (4 instructions vs 6 dedicated) — width or a longer continuation would
raise it (`optimizations.md`). Speed: unchanged per hop; what the merge removes is duplication (one model, one loader,
uniform `CALL`/promotion).

---

# Part 17 — transitive reasoning from three worked stories (2026-09-16, `story.py`)

Story memory: a fact = `subject⊗SUBJECT + relation⊗RELATION + object⊗OBJECT` (three role relations trained once on random
triples, 25 s; nothing about stories); a story = a set of fact places in working memory; names = fresh random places
never trained. Reads: `READF s,r>t` (the fact keyed by (name, relation), **verified by reading its subject and relation
back** — no threshold), `HASF s,r,t>u` (the whole fact lit against the story), `SAME`, `IFYES` (the branch), `CALL`.

| program (3 worked stories each, ≤ 3 links) | found | chain length 1–10, fresh names (> 3 never shown) |
|---|---|---|
| `chain_bigger` | gen 1 | 1.000 ×10 |
| `decide_bigger` (calls `chain`; YES / NO / UNKNOWN) | gen 1 | 1.000 ×10 |
| `decide_knows` (non-transitive trap: direct fact only) | gen 1 | 1.000 ×10 |

At 6 links by case: YES 1.00 (69), NO 1.00 (86), UNKNOWN 1.00 (145). Ground-truth programs also 1.000 on the machine.
Fixes on the way: keyed reads verified by read-back instead of a 0.72-vs-0.71 cosine cutoff; a whole-fact light for
direct checks (a subject may have several facts of one relation); and the first "3–7 % errors" were generator labels,
not the machine — labels now come from the story's structure. Next: CLUTRR structured (kinship chains; compositions
as neighbour edits), bAbI 16–18, templated sentences.

---

# Part 18 — CLUTRR, structured (2026-09-16, `clutrr.py`)

CLUTRR `gen_train23_test2to10` (train chains 2–3, test 2–10), structured form, the 713 chain-shaped test stories (the
other 433 contain a detour and need path search — next). Story memory as Part 17; the **composition table** `(r₁, r₂) → r₃`
is learned from the training stories (117 entries after both-bracketing closure over train+val) and stored as facts in
the same memory, read by the same keyed read. Program found from three worked training stories: walk the chain from
the query's first name, `NEXT` (verified read), `COMPOSE` (table read; NONE is sticky = abstain), `SAME` to stop, `ANSWER`.

| chain length | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| exact | 1.00 | 0.52 | 0.72 | 0.62 | 0.49 | 0.39 | 0.26 | 0.31 | 0.28 |
| abstain (composition never learned) | 0.00 | 0.43 | 0.25 | 0.37 | 0.51 | 0.61 | 0.74 | 0.69 | 0.72 |
| wrong | 0.00 | 0.05 | 0.03 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

n = 38 / 105 / 177 / 131 / 63 / 61 / 61 / 45 / 32. From six links up the machine is never wrong; every miss is a declared
"I don't know this composition". The reasoning generalises in length; the *knowledge* (the table) does not cover the
compositions long chains need. Fix on the knowledge side: learn the relation words as compositions of a few kinship
primitives (parent / child / sibling / spouse × gender) so the table is complete by construction — the "words express
primitives" thread. Bugs on the way: a 2-D `.tolist()` summary; NONE not sticky (turned abstentions into wrong answers).

---

# Part 19 — CLUTRR unsupervised: relation words as operators (2026-09-16, `kinship_ops.py`)

After Part 18's abstentions I tried to *learn the meaning* of the kinship words as paths in a hand-written family algebra
(`kinship.py`): the words came out right (father = up♂, niece = side-down♀ …, all 20) but every hand-written or
hand-repaired rule made generalisation worse (54/62 → wrong overgeneralisation → 25/62). Recorded as a negative.

The unsupervised version: each of the 20 relation words is a learned ResonatE operator; a training story
`(r₁, r₂ → r₃)` says only that applying the chain must land where the target word lands (cross-entropy over the 18
target words' landings on a random probe). Nothing else: no steps, genders, rules or meanings. A test chain is the
product of its operators on a probe, named by the closest word's single hop, abstaining when the margin is below the
95th percentile of wrong-answer margins on validation.

| chain length | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| exact | 1.00 | 0.90 | 0.98 | 0.97 | 0.89 | 0.87 | 0.79 | 0.73 | 0.78 |
| abstain | 0.00 | 0.06 | 0.02 | 0.03 | 0.11 | 0.13 | 0.20 | 0.24 | 0.16 |
| wrong | 0.00 | 0.04 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.02 | 0.06 |

Validation 0.996; all 62 training compositions reproduced by operator products. Train chains 2–3, test to 10, n as in
Part 18. The algebra is in the matrices — the biokg "relation families emerge unlabelled" result, now doing inference.
Owed: the 433 non-chain stories (path search), seeds, and the natural-language surface.

## Part 19b — the gap to 0.99 was effort, not capacity; the full test set (2026-09-16)

Same code, no new primitive: 20 000 steps instead of 4 000 (the operators had not converged: loss 0.05 → 0.004) and 64
probes instead of 8 at answer time. Width does not matter — k = 8 (64-wide places, 4 blocks of 16×16 per word,
~20k complex parameters in total), k = 12 and k = 16 give the same numbers within one or two stories per length.

The 433 test stories excluded in Part 18 as "non-chain" are walks that revisit a node (0→1→2→1→3); the operator product
along the edge sequence answers them unchanged, so the whole CLUTRR test set (1146 stories) is answered here.

**k = 8, 20 000 steps, 64 probes, seed 0, all 1146 test stories** (train chains 2–3):

| chain length | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| n | 38 | 105 | 190 | 174 | 107 | 144 | 150 | 119 | 119 |
| exact | 1.00 | 0.86 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.94 | **0.99** |
| abstain | 0.00 | 0.14 | 0.01 | 0.00 | 0.00 | 0.00 | 0.01 | 0.06 | 0.01 |
| wrong | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 |

Validation 0.996; 62/62 training compositions reproduced. Published on this split at 10 hops (Minervini et al. 2020,
full test set, no abstention): CTP_A 0.90 ± 0.07, LSTM 0.75, GAT 0.45, GCN 0.39, GNTP 0.31.

**The abstentions are the dataset's ambiguities, not the model's misses.** All 15 abstentions at length 3 (15/105 =
0.14) are the 15 test occurrences of one chain, *wife → son → grandmother*, which is either *mother* or *mother-in-law*
— and the test set labels it both ways (10 vs 5). Integer counts by length (exact/abstain/wrong): 2: 38/0/0, 3: 90/15/0,
4: 189/1/0, 5: 172/0/2, 6: 106/0/1, 7: 143/0/1, 8: 148/2/0, 9: 112/7/0, 10: 118/1/0. The length-9 abstentions are walks that can return to self (*father → wife → son* = me or my brother, then
*daughter → sister* = daughter or niece); the two candidate words light within 0.01–0.04 of each other. CLUTRR's labels
assume the walk never returns to self; the operators see that it can.

Controls: with training targets randomly permuted the same run gives exact 0.00 / abstain 1.00 at every length (the
operators learn nothing consistent and the margin test refuses everything); the model sees only the relation-word
sequence — no names, genders or text; every length-2 test sequence occurs in training (38/38, the table itself); no test
sequence of ≥ 3 words occurs in training (0/105 at 3, 0/1003 at 4–10). Training: 9,074 stories (4,076 two-hop, 4,998
three-hop; 383 distinct sequences), batch 256 with a fresh probe each, cross-entropy over the 18 target landings.
40k steps (same seed): 10 hops unchanged (118/1/0) but length 3 starts guessing on the ambiguous chain (3 wrong) —
longer training erodes abstention; 20k is the reported run. Caveats: graph input (the CTP/GAT setting, not the text setting); one seed
(±1–2 stories per length between k = 8/12/16 runs).

# Part 20 — the language layer on the arithmetic machine (2026-09-16, `read_math.py`)

Sentences that call the found programs, and sentences that define new ones. Every token is a row: the ten digit words
and the function words (`add`, `and`, `from`, `by`, `to`, `it`, `itself`, `.`, `?`, …) are stored; a new word (`double`)
is a fresh row. A numeral is read digit by digit (`3 4 7`) and the number is the place its digits walk to (Part 4) —
no number place sits in a sentence (as tokens they are near-parallel, cos 0.98, and a light cannot tell two apart;
walking the recovered place was worse still: two negatives on the way). A sentence is one place, a `NEXT` chain over
its token places, read back by walking: exact to 10 tokens, ≥ 0.99 to 14, 0.93 at 18 (k = 32 language memory, its own
width; 15k steps). The reader is a program over lights on the machine of Part 14 with 13 primitives (`LOAD`, `ISW`,
`ISDIG`, `ISPROG`, `ISNEW`, gated `CCOPY`/`CWRITE`/`CNEWOP`/`CSWAP`, `CALLW`, `CDEF`, `SET`, `COPY`); its output is a
call into the found programs (`add`/`sub`/`mul`/`cmp`, reached through their sentence words — a naming alias) or a
definition. Fitness = the machine's answer only. Two-sentence episodes: two commands, or a definition and a use.

| never shown | exact |
|---|---|
| commands (`please add the 3 4 7 and 1 2 .`, `subtract … from …`, `multiply … by …`, `compare … and … ?`), 1–7-digit numbers (train 1–3) | 1.00 at every length |
| commands with 0–4 filler words at two positions (train 0–1) | 1.00 at every count |
| definition + use (`to W something , add it to itself .` / `add 5 to it` / `subtract 1 5 from it` / `multiply it by 3`; then `W 3 4 3 2 4 .`), unseen names, 1–7 digits | 1.00 at every length |
| unknown command word (`W 1 2 and 7 .`) | no call made, 400/400 |

The two-sentence sample: *to W0 something , multiply it by 7 .* → nothing; *the W0 4 4 6 .* → 3122.
Found in the ladder (commands first, then definitions seeded from the command reader), each at generation 1 — which
must be read correctly: the worked traces come from one ground-truth reader that handles both kinds, so the initial
population already contained the full 17-instruction cycle and the search *selected* it against the machine's
answers (as for add/cmp/muld in Part 14). The joint search without the ladder plateaued at 0.28 exact over 40
generations. What the result establishes is the reading: order, the role of positions, which words carry structure
and where operands begin and end are used correctly on sentences and numbers never shown, and a program defined by a
sentence is callable by the next one. What it does not establish: discovery of the reader from answers alone (the
traces were given), and the 13 primitives and the word→program aliases are ours.
**Side finding:** the found `add` of Part 14 has a hole — `95365 + 9114 → 4479` (second operand shorter, carry out of
the top digit; the acceptance sample never produced the case). Owed: acceptance samples with unequal lengths and
carry-out, `add` re-found, dependent tables re-run. Reading is scored against the machine's own answer to the intended
call. Optimisations that made the search feasible (`optimizations.md`): memoised, deduplicated calls (92 % of a
generation's calls were duplicates), a fixed number pool during search, opcode-level dispatch, tight unrolling.

## Part 14b — the addition hole, fixed permanently (2026-09-16)

Characterised: the Part-14 `add` failed whenever the **first operand was longer and the sum carried out of its top
digit** (`99+1 = 0`, `999+1 = 0`, `95+9 = 4`, `95365+9114 = 4479`); the reversed order was fine (`1+99 = 100`). The
sampler drew the sum first and split it, which yields that pattern 0.09 % of the time — 150 samples per length never
saw it; the acceptance test was blind by construction. Fix in `places.py`, for every future search: (1) operand
lengths sampled independently (add, sub); (2) a fixed **regression set** of carry-out / unequal-length patterns per
operation that acceptance and verification must pass in addition to the fresh held-out cases. Re-found `add` from the
same three worked examples (generation 1): the same 8-instruction cycle with stop slot 0 instead of 4 — exact at 1–9
digits and on the regression set; `sub`, `cmp`, `muld`, `mul` (which calls `add`) re-verified 1.00 with it. The hand-
written `add` had no hole (the machine was never the problem; the found program's stop rule was). `found_programs.json`
updated; the one memory re-promoted with the fixed program (Part 16 re-run). Still to audit with the same regression
set: the Part-13 `discover.py` addition (older machine). Lesson for the paper: **an acceptance test must sample the
failure patterns explicitly; "exact on n fresh cases per length" is not exactness.**

# Part 21 — CLUTRR from text: a negative (2026-09-16, `clutrr_read.py`, `clutrr_text.py`)

The graph result (Part 19b, 0.99 at ten hops) with the sentences read instead of given. Reader: per relation word in a
sentence, owner/object by local cues (`'s` before the word, `of` after it, otherwise nearest name before/after) — in
code, as the ceiling of a light-program reader over the same cues. Path search over the story's read facts from either
end; the relation sequence with directions; operators with a reverse hop per word; abstain if no path or low margin.

What the text is: 4,620 sentence templates in training (names masked), 1,423 of the 1,438 test templates unseen,
2,432 word types. On clean sentences (one relation word, two names; 14,876 occurrences) the cue reader is **0.963**
correct and its errors are template-consistent — 73 templates whose direction contradicts their own gold graph
("X adores her daughter X" with the graph saying the reverse): dataset noise no reading of English recovers. The losses
are elsewhere: facts spread over sentences by pronouns (28 % of unrecovered gold facts), sentences without any of the
20 relation words — `dad`, `mom`, `daughters`, `anniversary` (22 %), two-fact sentences with ellipsis/appositives/
pronoun owners (40 %). Net: 0.87 of read facts true, 0.77 of gold facts recovered, and the whole query path intact for
0.51 of test stories (0.66 at two hops → 0.39 at ten).

| test, from text (1,146 stories) | exact | abstain | wrong |
|---|---|---|---|
| operators trained on text-read paths, free reverse hops | 0.084 | 0.896 | 0.020 |
| … tied reverse (adjoint) | 0.003 | 0.996 | 0.002 |
| operators trained on the gold graph (the 0.99 algebra), tested from text, tied reverse | 0.017 | 0.980 | 0.003 |

Two causes, separable: (1) operators trained on read paths never form a consistent algebra (train exact 0.78; 13 %
of facts wrong → contradictory instances) and the abstention threshold, calibrated on validation, then refuses nearly
everything; (2) traversing a fact backwards is not an inverse: free reverse hops learn nothing (forward-then-reverse
cosine ≈ 0.05), the adjoint is an inverse only while blocks are unitary and they drift (≈ 0.5). Compounding does the
rest: a 0.96-per-fact reader gives at most 0.96¹⁰ ≈ 0.66 at ten hops.
Why arithmetic-from-language worked and this does not: a controlled grammar (one fact per sentence, local cues, closed
vocabulary, exact supervision) versus crowd-written text (coreference, synonyms, ellipsis, contradictory templates).
The memory and the chain are not what failed; reading messy text is a different problem.
Ways forward, each measurable against this table: keep blocks unitary (Cayley/exponential parametrisation) so the
reverse hop is the exact inverse; relation words discovered from the "X's W Y" pattern with their own operators
(`dad` → `father`'s by use); pronoun/appositive/ellipsis cues; or delegate reading to an LLM/parser and keep this
memory as the exact, abstaining reasoner.

## Part 21b — text → gold facts: the wall is the benchmark's fact form (2026-09-16, `clutrr_convert.py`, `clutrr_local.py`, `clutrr_cue_table.py`)

Three learned converters from the paired data (text + gold graph), as a curriculum (one hop, then whole stories):
a BiGRU over the story with pooled name mentions and a pair classifier; a local classifier over (relation-word mention,
name pair) with ±3-token windows and cue features (relation words discovered by label lift across templates: `dad`,
`daughters`, `kids`, `married`, `anniversary` come in); and the cue reader with a counted relabelling table.

| converter | validation (seen templates) P / R | test (unseen templates) P / R | end-to-end test exact |
|---|---|---|---|
| BiGRU pair classifier, whole stories | 0.99 pair-acc | 0.46 / 0.41 | 0.027 (wrong 0.033) |
| local window classifier, one hop | 0.84 / 0.74 | 0.56 / 0.44 | 0.002 |
| cue reader + counted table | 0.64 / 0.54 | 0.59 / 0.50 | 0.000 |
| gold facts through the same pipeline | — | — | **0.974** (0.97 at 10 hops) |

Why they all stop near 0.55: the gold graph writes the *same reading* in different forms depending on the template —
"A's son B" (no gender cue) is (A, son, B) in 900 training sentences and (B, father/mother, A) in 378; within a
template the form is fixed (99 %), across templates it is not, and the test templates are unseen. So the form of a
gold fact is not predictable from the sentence, and no sentence-level converter can exceed that ceiling. (The path
recoverability with either-direction traversal of true facts, 0.51, is the reading ceiling; with forward-only gold-form
facts it is 0.12.) The consequence is architectural, not lexical: the reasoner must accept a fact in whichever form
the text gives it, which needs exact inverses in the operators — blocks kept unitary by parametrisation so the reverse
hop is the inverse (free reverse hops learned nothing; the adjoint drifted to cosine ≈ 0.5). That is the next
experiment on this line, on the graph task first.

# Part 22 — to a trillion by continuation; the found programs unchanged (2026-09-19, `grow.py --np 12`)

The Part-10 recipe once more: from the 10⁹ table (18 rows), three new place anchors (10⁹, 10¹⁰, 10¹¹), 30 new value
anchors, operators and old rows learning with old numbers replayed; 3,000 + 3,000 steps, **513 s**. Before the
continuation every chain reads 0.00 (three new leading-zero terms push every old term deeper — the fixed-length chain's
known cost); after it:

| 10¹² table | 1–12 digits |
|---|---|
| walk-back on unseen numbers | 1.000 at every length |
| value code, held-out equations by place | 1.000 at every place |
| never-trained sums | 1.000 at every length |
| comparison, 6,000 unseen pairs, distances 1 … 10¹¹ | 1.000 (digit-order template re-fitted: 4 s, 45 facts) |
| **the five found programs of Part 14, unchanged** | add / sub / cmp / muld 1.000 at 1–12 digits; mul 1.000 to 11 digits, 0.933 at 12 (products beyond the table's range) |

The programs were found on the 10⁹ table from three-digit examples; they reference slots and moves, never the table's
width, and run exactly on numbers three places longer than the table they were found on could hold. The only thing that
had to be re-fitted was the ten-symbol digit-order light (`Q`), because the digit rows moved during the continuation;
`places.py` and `compare.py` now take the number of places from the checkpoint. 10¹⁵ is the same step again. The
unbounded version — a number as a chain of its digits under one repeated move, no place rows — is Part 23's candidate;
its ceiling would be the depth at which the walk stays exact rather than a row count.

# Part 23 — the digit chain: ten symbols, one move, no place rows (2026-09-19, `digit_chain.py`, `digits.py`)

The last hand-placed structure in the numbers track was the place rows (`10 … 10¹¹`) and the fixed chain length they
imply. Removed: a number is now a chain of its digits under **one repeated move**, `NEXT`, built onto an END symbol —
eleven rows in all (the alphabet), no row per magnitude, no length. Reading it back is the same move in reverse, one
digit per step, until END lights. The move is trained self-supervised on random digit strings of 1–24 digits (walk-back
objective: no labels, nothing arithmetic; 12k steps, ~20 min at k = 20 on the 1080 Ti sharing the GPU). The places
machine of Part 14 runs on it with four primitives re-based on the chain (`LOAD` walks one digit; `SHIFT` appends a
zero; "anything left?" peeks to END; a number is built by the move) — the executor, the other primitives and the
**found programs are byte-for-byte those of Part 14**, found on the anchored 10⁹ table from three-digit examples. The
value code shrinks to the digit and carry phases (the programs never used more); the digit-order light is re-fitted on
the chain's rows (45 facts, seconds).

| digit chain, train lengths 1–24 | k = 12 (M = 144) | k = 20 (M = 400) | k = 32 (M = 1024) |
|---|---|---|---|
| read-back exact, lengths 1–24 | ≥ 0.993 (1.000 at 21 of 24) | 1.000 at every length | 1.000 at every length |
| read-back at 26 / 28 / 32 digits (never trained) | 0.15 / 0.00 / 0.00 | 0.29 / 0.07 / 0.00 | 0.75 / 0.22 / 0.01 |
| found `add`, `sub`, `cmp`, `muld`, unchanged, 1–24 digits | 1.00 at every length, regression 1.00 | 1.00 at every length, regression 1.00 | — |
| found `mul`, unchanged | 1.00 to 23 digits, 0.98 at 24 | 1.00 to 23 digits, 0.99 at 24 | — |
| `mul` at exactly 12 × 13 digits (products of 24 or 25 digits) | — | 24-digit products 11/11 exact; 25-digit products 0/47 — the answer is the product with its top digit dropped: the chain holds 24 |

**A note on the tables' rows:** in every `verify` table (Parts 14, 22, 23) the row "L digits" samples operand lengths
uniformly from 1 to L, so it reads "numbers of up to L digits"; 1.000 there covers every sampled length, and the
read-back tests use exact lengths. So the numbers track reduces to: **ten symbols, one move, three worked examples per
operation** — exact to 10²⁴ on a 144-dimensional memory. The honest ceiling: the range is the depth the move was *trained* at — all three widths are exact through 24 and fall off
just beyond, width buying only a little extrapolation (26 digits: 0.15 / 0.29 / 0.75). Width does set a *capacity*: at
k = 4 (16 dimensions, a single block) the 24-deep chain cannot be held (Part 23b), while CLUTRR's twenty operators fit
in the same 16 dimensions — chains need width proportional to depth, operator algebras barely need any. "Unbounded"
is therefore not a claim; "as deep as the move is trained, at a cost that is one row of training data per digit" is.
Owed: the width-versus-depth curve (train to 40 digits at k = 12 and 32) to see where a small memory stops following.

## Part 23b — a corner the random strings never showed (2026-09-19)

The k = 12 digit chain reads random strings exactly to 24 digits and **misreads runs of nines from length 9**
(`999999999 → 999999799`); runs of fives and alternating patterns are fine to 13, and k = 20 / 32 read every run
correctly. Repeated addition of the same row along the move is a pattern with probability 10⁻⁹ in random training
strings, and for one symbol the small memory loses its depth on it. Found while trying to discover a digit-sum program
(its sums of all-nines came out wrong; the program was fine). Same lesson as Part 14b, on the memory side: **test the
patterns that random sampling never produces.** The trainer now draws a tenth of its strings as runs of one digit and
the read-back test includes runs of every digit at full length; the seed runs use the patched trainer.
Small-width results (Part 23c, below) locate where width becomes the limit.

## Part 23c — where width becomes the limit (2026-09-19)

| width | dimensions | digit chain, read-back exact to | CLUTRR, ten hops (k = 8 setting otherwise) |
|---|---|---|---|
| k = 4 | 16 | 6 digits (0.53 at 7, 0.00 from 15) | **0.87** exact, 0.04 wrong (0.99 at 4–8 hops) |
| k = 8 | 64 | 13 digits; ≥ 0.9 to 23; 0.70 at 24 | 0.99 (Part 19b) |
| k = 12 | 144 | 24 (every trained length) | 0.99 |
| k = 20 / 32 | 400 / 1024 | 24; beyond training 0.29 / 0.75 at 26 | 0.99 |

Two regimes: an operator algebra (twenty relation words) needs almost no width — a single 16×16 block composes ten hops
at 0.87 — while a chain needs width in proportion to its depth, roughly **depth ≈ dimension / 3** at this alphabet,
until the trained length caps it. So the numbers track's ceiling is two-part: width sets the capacity, training sets
the depth actually reached within it; the reasoning track's ceiling is neither — it is the objective.

# Part 24 — discovery from input–output pairs: what worked, and why the search is wrong (2026-09-19, `discover_io.py`)

No worked traces anywhere: the fitness sees only `(x, y) → answer` pairs; the found programs are callable; the population
starts from random programs (blank slate) or from a neighbouring program (the ladder). Proposals can be *typed* by the
lights — an instruction is proposed for a position only if the slots hold what it acts on (a walkable number, a digit, a
verdict), read from the candidate's own execution on one example.

| rung | start | proposals | result |
|---|---|---|---|
| `triple` from `(x, 3x)` | seeded with `double` | untyped | **found at generation 2**: double's body + one more `CALL add` — exact 1–8 digits and the regression set |
| `double` from `(x, 2x)` | blank slate, library callable | typed | exact program at generation 9 (~10 min) — but it leaned on a constant second operand (the sampler's y = 0); with y random the honest run was stopped at gen 1 |
| `digit sum` from `(x, Σ digits)` | seeded with `add`'s program | typed, y random | 0.75 exact by generation 10 and climbing (population mean 0.67 → 1.16 by gen 20), not found in the ~30 generations run |

Two things stand: a **library effect** — the next program is cheap when a neighbour exists (triple in 2 generations) — and
a **sampler lesson** (again): a unary task must get a random second operand or the search finds a program that uses it.
And one thing does not stand, which is why the line was stopped: the search is brute force. Typing the proposals only
shrinks the alphabet per position; the *choice* is still a random draw, and nothing the search observes — which output
place is wrong, what the slots held when it went wrong — is used to pick the next candidate. That is the opposite of
how every other part of this machine works. The right proposer is the Part-12 controller turned around: a stateless
policy that reads (slot kinds, the failing output place) and emits an instruction, trained by imitation on the programs
already found, then measured as candidates-to-acceptance with and without it. Not done; recorded as the next design.

# Part 25 — program by observation: a worked example played on the machine (2026-09-19, `observe.py`)

The search of Part 24 was brute force with a typed alphabet; this replaces the search with an **observer**. A worked
example is a sequence of events in the machine's own terms — for `digitsum(123) = 6`, read units first with a zero to
start: *a value 0 appears; the digit 3 appears; add(0, 3); the digit 2; add(3, 2); the digit 1; add(5, 1); the answer 6*.
The numbers in it are variables, the named operation is a function. At each event the observer **derives** the moves
from the event and the lights — which slot holds the named operands, which slot still holds the number being read,
which slot lights the answer — a handful of candidates (`CALL add 2,3>2` with the result in one of a few slots; `LOAD`
from the slot that holds the input), each verified by executing; a beam of 8 partial traces carries the genuine
choices forward and an entry whose choice makes the next event impossible dies there, with no rule about which slots
to spare. The recorded traces are then the worked examples of Part 14's search: slices of them as candidates, fitness
= the machine's answers on fresh short numbers, acceptance = exact on numbers longer than any shown and on the
all-nines regression cases. No rule folds the loop.

| shown | trace recorded | found | exact on |
|---|---|---|---|
| `digitsum(123) = 6`, `(90) = 9`, `(5) = 5` | `SET 2<0` · (`LOAD 0>3`, `CALL add 2,3>2`) per digit · `ANSWER 2` — in 13 / 8 / 4 s | generation 1, **150 candidates**, ~1 min in all | 1–13 digits, all-nines to 13 digits (117): 1.000 |

The found program: init `SET 2<0`; body `LOAD 0>3, CALL add 2,3>2`; outro `ANSWER 2`. It was then **labelled**
`digitsum` in the library — a write — and answers by name through the language layer: `digitsum 12346 .` → 16,
`digitsum 999999999 .` → 72 where 81 is right — the program is exact on both machines (81 on the anchored table directly); the *reader* drops a digit at 9-digit numerals, one past the length it was verified to (7). The label step is automatic: the observer writes an accepted program into the library under the worked example's word, and the language layer labels every library program on load. The brute-force arm of Part 24 on the same task (named
callables + plumbing, proposer, worked-example credit) was at 7,800 candidates without a find when stopped.

What was learned on the way, each a closed leak in the machine: an empty slot read as the number 0 in a call (now a
call needs two filled operands); a `LOAD` could walk a chain that was still underneath a digit tag written over it (now
only a walkable slot can be walked); the second operand of a unary task must be random, or a program leans on it.
Lessons for the write-up: (1) the worked example's *structure* — values as variables, named operations as functions —
is the program's structure; reading it off the machine costs seconds where searching for it costs hours; (2) the
observer is the controller's stance (read lights, choose moves) applied to *learning*, not just execution; (3) what is
still given: the event grammar for each task (what counts as "the digit appears", "add happens") — the teacher's
vocabulary — and the acceptance test. What is not: the program, the loop, the slots, the stop rule.
