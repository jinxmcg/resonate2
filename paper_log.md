# Paper log

Running record for the write-up: what was done, what it showed, what a reviewer will ask, what is still owed.
One entry per session block. Numbers live in README.md (Parts 1–16) and results/*.json; this file is the argument.

## Claims we can currently defend (each with a pointer)

1. **A computed place exists without a row.** Numbers below a billion from 18 rows; composed, decomposed, compared,
   added exactly (Parts 4, 8, 9, 10, 12). Live-Wikidata time-split spots from the earlier work (README top, wikinew).
2. **Superposition keeps *what*, direction keeps *pairing*, multiplication keeps *value*.** Bag encoding: 347 ≡ 437
   (Part 4); directed chain: 1.000 walk-back; phase code: `=` is the light, carries free (Part 6).
3. **Evaluation is not a move.** `=` as an edge cannot even memorise single-place facts (Part 5); learned operator
   generators memorise and do not generalise (Part 1, fourteen configurations at 0.000); the successor walk is not an
   adder in three parametrisations (Parts 2–3).
4. **Reasoning is a program over lights; the controller must be stateless.** Verdict in the hidden state: compare 0.24
   at 9 digits; verdict in a slot: 1.000 (Parts 11–12).
5. **Operations are induced from three worked examples and generalise in length.** add, sub, cmp, muld, mul, each
   1.000 on lengths never shown; sub found from add with its own idiom; blank-slate search has no slope (Parts 13–14).
6. **Programs are places in the same memory.** Stored as pages, executed by walking, calls resolved by walking; a new
   program's name row needs no training (Parts 15–16). One table for numbers and procedures.
7. **Knowledge grows by continuation without forgetting; depth lives in the operator, not in rows** (Part 10).

## What a reviewer will say, and our answer

- *Program induction from traces is known (NPI 2016, recursive NPI 2017, DreamCoder).* Yes; our primitives are the
  memory's own moves and lights on a learned relational geometry, execution is exact and auditable, and the procedures
  are stored in the same space as the data. We need an NPI-style baseline on the same tasks.
- *Binding/superposition is VSA/HRR.* Yes; the difference is a learned geometry (ResonatE) rather than random codes,
  exact walk-back through learned operators, and the measured capacity/paging. We need a VSA baseline on the chains.
- *It's a calculator.* The graph port (planned) is the answer; until then the claim is about the memory, not arithmetic.
- *One seed.* Owed: 5 seeds on the headline tables (Parts 12, 13, 14, 16).

## Still owed before writing

- [ ] seeds on the headline tables
- [ ] NPI-style and VSA baselines on the same tasks
- [ ] one real-graph result (compositional queries on wikikg2/biokg from three worked examples)
- [ ] related-work pass (NPI, DreamCoder, HRR/VSA, KG embeddings, CLUTRR/bAbI reasoning systems)
- [ ] ablation table consolidated from the negatives

---

## Log

### 2026-09-16 — logic: transitive reasoning from three worked examples (start)

Decision: test the "program over a story memory" claim on transitive inference before the graph port, because it is
the same length-generalisation claim as arithmetic and there are benchmarks with ground truth (CLUTRR structured;
bAbI 16–18 as sanity; ProofWriter later). Order: synthetic transitive world → CLUTRR → bAbI → templated sentences.

Design (`story.py`):
- Story memory: a fact = `subject⊗SUBJECT + relation⊗RELATION + object⊗OBJECT` (three role relations), one place;
  a story = a set of fact places in working memory (the cache); names are rows never seen in training (fresh).
- Roles are trained once on random triples with a read-back objective (unsupervised w.r.t. stories); nothing per story.
- Primitives added: `READF s,r>t` (the fact keyed by the subject in `s` and the relation `r`, read against the story's
  facts; its object into `t`; NONE if nothing lights), `SAME s,t>u` (writes YES into `u` if the names match).
  Both are `READ against a set` + a gated write — no new kind.
- Programs to find from three worked stories each: `chain` (follow a relation until the target or nothing),
  `decide` (YES if chain(a→b), NO if chain(b→a), else UNKNOWN) — the second calls the first.
- Traps: a non-transitive relation (`knows`) in the same stories; distractor facts; reversed queries; unrelated pairs.
- Test: chain length 1–10 with train stories ≤ 3 links and unseen names; exact accuracy per case type.

Build notes (same day): the story memory's fact roles train in 25 s; a keyed read separates present from absent facts
only narrowly by cosine (0.72 vs 0.71), so 'is there a fact?' is decided by **reading the best fact back** (its SUBJECT and
RELATION lights must equal the query) — a light-verified read, no threshold, consistent with the rest of the machine.
Ground-truth programs execute at 0.97 (chain) / 0.93 (decide) / 0.915 (knows) on fresh-name stories — not exact;
residual to diagnose (suspect: object read-back cross-talk among fresh random names at M = 144). Worked examples for
`chain` must be walks that reach the target (first draft drew three 'nothing found' stories — fixed).

**Result (same day):** all three logic programs found at generation 1 from three worked stories each and exact
(1.000) on chains 1–10 with fresh names; YES/NO/UNKNOWN each 1.00 at six links; the non-transitive relation is never
propagated. This is the arithmetic length-generalisation claim reproduced on inference over a story memory — the
paper's logic table. Two lessons for the write-up: (1) every read the machine makes should be *verified by a light*,
never by a threshold; (2) label your generator from structure, and check the machine against ground truth before
blaming it (the first 'errors' were ours).
Owed next: CLUTRR structured (the external benchmark), bAbI 16–18, then the templated-sentence parser as a found program.

### 2026-09-16 — CLUTRR structured (chain stories)
Found program = ground-truth walk; exact 1.00 at 2 links falling to ~0.3 at 10, but **wrong = 0.00 from 6 links up** —
all misses are abstentions from an incomplete learned composition table (117 pairs from training). This is the honest
headline: exact where it knows, silent where it does not, versus baselines that guess. The knowledge gap is the next
experiment: relation words as compositions of few primitives → complete table. Also owed: the 433 non-chain stories
(path search with a visited set — the machine needs a small stack or a `NEXT` that excludes a visited name).

### 2026-09-16 — CLUTRR: hand-written algebra vs emergent operators
Three hours of hand-repairing a kinship algebra (`kinship.py`) produced correct word meanings and worse generalisation
each time a rule was touched. The version with **no** rules — relation words as learned operators, trained only on
"this chain lands where that word lands" — reaches 0.78 exact / 0.16 abstain / 0.06 wrong at 10 links (train 2–3),
validation 0.996. For the paper: (1) this is the biokg emergence result turned into inference; (2) abstention is a
first-class outcome and should be reported next to exact; (3) the lesson "do not write the algebra" is a finding.

**Same day, later — the gap closed.** 0.78 → **0.99 exact at 10 hops on the full 1146-story test set, wrong ≤ 0.01 at
every length** (README Part 19b). Nothing added: the 4k-step run was unconverged (20k steps) and 8 probes were noisy
(64). Width irrelevant: k = 8 (64-wide, ~20k complex parameters) = k = 12 = k = 16. The "non-chain" stories were walks
with revisits — the product along the edge sequence answers them, no path search needed; that removed a caveat rather
than adding one. Reference: CTP_A 0.90 ± 0.07 at 10 hops on this split (Minervini et al. 2020, arXiv 2007.06477);
GAT 0.45, GNTP 0.31 — the text baselines I quoted earlier (0.2–0.4) are the wrong comparison, graph input is the right one.
Finding worth a paragraph: the residual abstentions are *ambiguities in the benchmark* (wife→son→grandmother is
mother or mother-in-law and the test set labels it both ways; long walks that can return to self), detected because two
words light equally. A model that must answer would score those by luck; ours says "two answers".
Controls done: shuffled targets → 0.00 exact / 1.00 abstain; input is the relation-word sequence only. Owed: seeds
(now a formality, ±1–2 stories/length), a 40k-step run (queued), the text surface.

### 2026-09-16 — ARC-AGI-2 as the external few-shot benchmark (`ARC2026/`)
Decision: enter ARC Prize 2026 for the **Paper Prize** (own Kaggle competition, deadline Nov 9) with a linked ARC-AGI-2 code
submission (Nov 2), not as a leaderboard contender; ARC-AGI-3 not pursued. First measured fact, `ARC2026/README.md`: a
place holds ≈ M/9 cells exactly (16 / 32 / ~50 at k = 12 / 16 / 20) and the learned operators give no capacity advantage
over untrained phase codes (worse on real grids) — so grids are *sets* of places in working memory, and the claim for
the paper is about the operations, not the code's capacity. The blank-slate gap (Part 13: no slope from I/O pairs) is
the whole ARC problem and must be measured, not hidden. Docs for the six required paper sections and the six-criterion
rubric are in `ARC2026/docs/`.

### 2026-09-16 — the language layer on the arithmetic machine (`read_math.py`, in progress)
Design: every token is a row (digit words included); a numeral is read digit by digit and the number is the place its
digits walk to; a sentence is a NEXT chain over token places, read back by walking. The reader is a program over
lights found from worked traces; its output is a CALL into the found programs or a DEFINITION of a new one ("to double
something, add it to itself."); fitness = the machine's answer only. Two lessons already, both general:
1. **Number places must not sit in a sentence.** As tokens they are near-parallel (cos 0.98) and a cosine light cannot
   tell two numerals apart; identifying them by walking the recovered place was worse still (0.93 at two tokens).
   Digits as words removed the problem entirely (read-back exact to 10 tokens, ≥ 0.99 to 14 at k = 32). Part 4's
   lesson again: the walk is the similarity for a chain, the light is the similarity for a row.
2. **The language layer found a hole in a found arithmetic program.** `95365 + 9114 → 4479`: the found `add` (Part 14)
   stops one cycle early when the second operand is shorter and the top digit carries out; equal-length operands
   with carry-out work (9999 + 9999), so the acceptance sample never hit the case. Owed: the acceptance sampler for
   arithmetic must include unequal lengths with carry-out, and `add` re-found; every table that used `add` re-run.
   For the reading experiment the target is now the machine's own answer to the intended call.
Also: the joint search (commands + definitions in one search) plateaued at 0.28 exact; the ladder (commands first,
definitions seeded from the command reader) reaches 1.0 on training episodes by generation 5 — the subtraction-from-
addition lesson again. And an acceptance set the ground truth itself fails is a broken yardstick: check it first.

### 2026-09-16 — CLUTRR from text: negative (README Part 21)
0.08 / 0.02 / 0.90 (exact / wrong / abstain) overall from text with a cue reader that is 0.96 per clean sentence; the
losses are coreference, lexicon, two-fact sentences, 4 % contradictory templates, and — a model finding — reverse
traversal is not an inverse (free reverse hops ≈ 0.05, adjoint ≈ 0.5 once blocks drift from unitary). For the paper:
the graph result stands; the text result is honest and low, with wrong ≤ 2 % because the machine abstains; the claim
"reasoning stays exact and abstains, reading is the open part" is the accurate one. Owed: unitary-kept operators (the
inverse fix is a one-line parametrisation change worth testing on the graph task first), lexical expansion from the
possessive pattern, and the LLM-as-reader variant, which would test the division of labour directly.
Addendum (same day): three converters text→gold facts all stop at ~0.55 precision on unseen templates because the
gold *form* of a fact is template-specific and unpredictable from the sentence ("A's son B" → (A, son, B) or
(B, parent, A) by template). The fix is exact inverses in the operators (unitary-kept blocks), not better reading;
with gold facts the same pipeline gives 0.974. README Part 21b.

**Same day, later — the machine on ARC-AGI-2 (`ARC2026/README.md`).** Built and measured, all local, nothing submitted:
storage exact (phase codes; learned operators only add cross-talk — three variants); a hardware-level machine (47 tokens)
that re-finds crop / extend / shift / symmetry from a flat worked trace in 1–7 generations *and* from I/O pairs alone
(blank slate has slope on drawing tasks — per-cell credit — unlike arithmetic; but blank-slate programs overfit and need a
held-out demo); a library of 3 programs from 300 training tasks; **0.0000 on the 120 public eval tasks, 0 wrong**, traction
on 40. Controls all at 0.0000 too: global transforms, dense learned operator (memorises, LOO catches it), feature lookup
(no eval task is a function of ≤ 2 local features), a 70-instruction hand-written DSL (no eval task is ≤ 3 of its moves).
For the paper: the machine claims hold (found programs, exactness, abstention, length/size generalisation by construction);
the accuracy row is conceded honestly; the diagnosis is quantitative (31 shape-unreachable / 49 no-slope / 40 traction).
Lesson recorded: a hand-written DSL primitive is hand-written knowledge (`CROP` *is* the solution to `1cf80156`) — the
CLUTRR lesson, caught by the user; kept as a labelled baseline. Live Kaggle leaderboard 76.94 (2026-09-14).

### 2026-09-16 — ARC: the zoom-out — strokes and object-level physics (`ARC2026/strokes.py`, `objpass.py`)
The user's diagnosis of the search: it works per pixel where the picture is about whole objects (a full ray, gravity) and
about strokes (lines that turn colour on impact). Two hardware layers added, each token one write / one move / one light:
14 object-level physics operations (`FALL`, `CAST`, `SHEAR_*`, `ROT_OBJ*`, `REFLECT_OBJ_*`, `SCALE2`, `DEL_OBJ`, `RECOLOUR`),
`DIR_TO2`, and flag forms of the selectors. The object pass executes `LOOP_OBJ [select] [dir] [colour] OP END` on the machine
over every demo (8–28k candidates, 5–15 s), accepts exact single-stage programs, feeds diff reducers to the descent. First
results: `d037b0a7` (cast), `1e0a9b12` (gravity with stacking via `REPEAT`), `05f2a901` (`SLIDE_TO2` re-expressed as
`DIR_TO2 FALL`) each solved by ONE object operation; a clean denoise stage for `0607ce86` (43 cells, 0 wrong).
Strokes: minimal stroke cover of the changed cells as the zoom-out; it separates line pictures (`0f63c0b9`: 92 cells → 11
strokes) from floods (`1190bc91`: 14 strokes of length 4) and names the missing reads (L-connect between objects; nearest-
object / row-Voronoi). `docs/lessons.md`: the ledger, one line per task with the tokens used for the first time; 49 % of
token uses across the 19 solved programs were reuse. For the paper: the "physics / patterns / logic" taxonomy the user gave
maps onto three token families and the per-task ledger is the honest metric while the solve count is small.

### 2026-09-17 — ARC: causality and orientation as lights (`ARC2026/pairlights.py`)
The token lights name the operator family; they do not say WHO decides an object's move and in WHICH frame. `pairlights.py`
reads each participating object's effect from the diff (move / cast / recolour) and tallies which cause explains all of them —
a uniquely coloured object, the largest / smallest, a related object, the object's own orientation, or a constant (gravity) —
in a cell frame and a tile frame. Verdicts read like the human's: `18286ef8` "the 6 decides the direction (tile frame); the 9
decides the colour", `13713586` "the largest decides the cast", `d037b0a7` / `17b80ad2` "constant S / N", `1e0a9b12` nothing
lit (one object already resting — correct). One read added (`DIR_TILE2`); the lit cause instantiates the operator families
first. `18286ef8` solved in 69 s with exactly the light's two verdicts as its two stages. Three general fixes fell out of the
two-checks rule: a background under 0-separators; the callable-stage cap was being consumed by the seeded library (the
leave-one-out folds could not chain the found stages); and the answer rule — a select that can be removed while staying exact
is unevidenced and does not vote; a prediction whose shortest program is ≥ 2 tokens longer than the shortest is dropped when it
would force an abstain (MDL with a margin; the margin because a 1-token-shorter program had been the wrong one on the same
task). 32 solved, 0 wrong, the object solves re-found with the pair light in front. For the paper: causality here is a
supervised light over (effect, candidate cause, frame) triples — no rule written, the machine verifies the named tokens.

### 2026-09-17 — ARC: programs out of the search, blocks in; cloning; effects by reconstruction (`ARC2026/`)
The user's diagnosis, confirmed by the numbers (one library replay in five fresh slices; library chance-fits cost a solve):
a solved program is one task. It stays for retention and for RECOGNITION (evaluated on a new task's demos: a lookup) and leaves
the search entirely. Reuse is carried by BLOCKS — skeletons with holes (`<OBJ2> <DIR> <NUM> <OP>` = "toward the second object,
by an amount, do the operation", shared by three tasks) filled from the library's fillers and the task's lights; two solved
tasks re-found from other tasks' blocks in ~2 s. Attractor / repeller as a polarity parameter (`DIR_FLIP`); the "repellers"
the light found were CLONING (`045e512c`, discovered by the machine once `CLONE` existed). Effects are now read by
reconstruction on the machine and attributed to the actor (an object completed by another's clone did not act) — which also
makes the segmentation a light (coherent causes under SEG8). An inventory of simple operators added (overlay of panels,
interior crop, symmetry completion about the object's own axis, fill holes, connect, scale-N, invert, multicolour objects).
Composition-level never-wrong: all exact finishers vote (a single returned finisher was wrong on the test twice). 35 solved,
0 wrong, retention 35/35.

### 2026-09-18 — ARC: primitives v3, the concept vocabulary as a generator (`ARC2026/v3.py`, `docs/primitives_v3.md`)
The 35 solved programs are two shapes — an object operation with parameters, and PAINT(domain, map, condition, colour) — so
the 294-token vocabulary was mostly hole-fillers. v3: ~24 operations × 7 slot kinds, each slot's values ordered by a light
(pair light for cause / frame / polarity / segmentation, effect kind for the operation, token lights for colour and constant
direction, a per-operation select light evaluated with the second-object read in place, number reads), one best-first cost
over operations and slots, compiled to the unchanged machine. The seven object solves are found at ranks 0–19 of ~13k
candidates (built in ~1 s); through the pipeline all re-found, 0 wrong, retention 35/35. A domain map names what is still
missing (legend indexing, statistics / majority, axis detection, order / rank, distance fields, z-order, clone window).
For the paper: the search is now light-first by construction — enumeration is the tail of one ordered list, not a stage.

### 2026-09-18 — ARC: fresh slice 6 under v3: 5/10 unattended (slices 1–5: 1, 1, 0, 0, 1)
Two solves by recognition (stored programs exact on new tasks — a lookup), one by an inventory primitive written for another
task's ask (interior crop), two by PAINT in seconds; none by enumeration; 0 wrong. The statistics domain (approximate period,
lattice phase, majority) closed `0607ce86` with a one-token program. 41 solved, reuse 59 %. The claim for the paper is now
measurable: with programs removed from the search and ~60 lit concept primitives in their place, the unattended transfer rate
on unseen tasks went from ~1/10 to 5/10 on the first slice tried; the next slices decide whether it holds.

### 2026-09-18 — ARC: slice 6's misses as concepts; 44 solved
Each skipped task of the first v3 slice named one basic concept, four of them slot VALUES (alignment as a frame the pair light
learns, orientation as a predicate, the radial direction per cell, membership in the second object as a paint condition) and one
operator (periodic shear). Three of the five were then discovered by the machine. The ranking layer is where the work went: a
segmentation light, a shear effect, a per-colour select light, interleaving of unnamed operations, and the two attempts as the two
most supported predictions when they carry >= 75 % of the exact programs. The user's correction on 1d61978c stands as a rule: the
colour is tied to the example, never to the concept — the machine pairs classes with colours; we only supply the class read.

### 2026-09-18 — ARC: colour by class, the class discovered by table consistency (`1d61978c`); slice 6 at 10/10
The user's rule: a task's colours are arbitrary but consistent across its demos; the machine must find the class read under
which the demos' colour table is consistent, and use the demos' colours. Implemented as class reads as numbers (side of the
partner, orientation code, the majority of any class read) feeding the existing learned table; the machine chose a class the
author had not proposed (the line's size under 8-connectivity) and was exact on the test. Slice 6 now 10/10 (5 unattended, 5
after one general concept each: alignment, periodic shear, radial direction, markers / annotation, class by table). 46 solved,
0 wrong.

### 2026-09-18 — ARC: compaction as a step; slice 7 at 2/10; the markers family grows
The loop is now explicit: expand (one general concept per miss) -> compact (sibling operations folded into a family with a
member slot; tokens untouched) -> measure (re-find gate 14/14, then a fresh slice). Slice 7: 2/10 unattended, 0 wrong — the
series 1, 1, 0, 0, 1 | 5, 2 says v3 lifted the rate and slice 6 was friendly. The gate caught a third route by which a single
finisher escapes the vote (the residual search de-duplicating by effect); the residual is now searched directly. Markers gained a
second operation (project onto the facing edge), discovered in 9.5 s once it existed. Two domains named for the next expansion:
temporal (one step, N steps, until an event) and order / rank. 49 solved, 0 wrong.

### 2026-09-18 — ARC: the legend / preview domain (`1e81d6f9`, `1da012fc`); 53 solved
The user read two open tasks as a PANEL WITH INSTRUCTIONS where the author had read geometry (rooms): the largest object frames a
legend, the objects inside are the instructions — a colour to remove, a miniature map of what to recolour. One panel read, two
operations; both tasks discovered by the machine in under 30 s. The domain unifies the earlier legend indexing and the tile pointer.
53 solved, 0 wrong.

### 2026-09-18 — ARC: slices 7–8 closed at 7/10 and 4/10 after one concept each; 59 solved
Unattended rates 2/10 and 2/10; after one general concept per miss (read with the user from the demo pages): the legend / preview
panel as three primitives, markers projected onto an edge, a marker inside an object as a direction, counting (a counter by the
found add, a count rendered as a length by the found cmp, the select chosen by count agreement), the tiles light and the reader's
observed stride / colour / directions as slot fillers, the quadrant of a symmetric block, the nearest LARGER object. Two path
faults found by the misses: crop programs hidden by an accidental constant size rule; the object pass's learned table facts
dropped before the answer. The pattern holds: operators exist; what a miss names is a LIGHT for a slot or a read one level up.
59 solved, 0 wrong.

### 2026-09-18 — ARC: slice 8 at 7/10; object algebra, zoom + texture, the lattice frame; 62 solved
Three more slice-8 misses read with the user, one basic concept each, all then discovered by the machine: object algebra
(`OBJ_DIFF / UNION / INTER` — 'the missing pieces to make the first object look like the second'); optics `ZOOM` (the output
cell reads the input cell it magnifies — we had repeat and shrink, no magnify) with a texture table keyed by the cell's phase
inside its block ('just fill with pattern'); and the tiles light generalised from separator lines to a point lattice, with
`CENTRE_TILE` ('centre the blue in the red grid'). None is task-shaped: each is one read or one move in a domain the map already
had (optics, canvas / frame, object algebra), and each solve came out shorter than its hand description. The unattended rate on
fresh slices is still 2/10; the quantity that improved is the cost per miss (one concept, then discovery) and reuse (~60% of solves
use blocks from earlier tasks). 62 solved (1 assisted), 0 wrong, retention 62/62.

### 2026-09-18 — ARC: slice 8 at 9/10; order / z-order, the legend as a dictionary, corner casts; 64 solved
`20818e16` opened the order / z-order domain (loop order = draw order; an occluded rectangle drawn whole). `20fb2937` extended the
legend domain: a panel legend, a dictionary of labelled samples, a stamp, the legend consumed by the crop. `212895b5` gave the
corner cast and a physical rule (a diagonal ray cannot pass between two lit cells) but its zigzag pinwheel is task-shaped: left
open, the machine abstains. Three path faults found by the misses, all lights or paths rather than operators: a constant size rule
hiding the crop composition (second occurrence), background detection demanding every demo coloured, the select light scoring
without the legend tagging. 64 solved (1 assisted), 0 wrong.

### 2026-09-18 — ARC: a cast with learned facts (`212895b5`); slice 8 closes 10/10; 65 solved
The zigzag rays looked like task knowledge (side, step, turn sense). The user's rule reframed it: the ray's shape and colour are
properties READ from the examples, like a texture table's entries. `CAST_RAYS` learns, per port of the body, a stroke in the port's
frame and a colour, pools observations per port class, and replays until blocked — the same learn/verify/LOO discipline as the
colour tables. The lesson for the map: an operator's parameters may be facts, not only slot values; what is written is the frame
(ports, strokes, blocking), what is read is the content. Slice 8: 2/10 unattended → 10/10 after one concept each (five of them
lights or paths, not operators). 65 solved (1 assisted), 0 wrong.

### 2026-09-18 — ARC: slice 9 at 1/10; the AREA BANK (latent places) and the z-order domain; 68 solved
Slice 9 unattended 1/10, 0 wrong. Reading the nine misses with the user: seven were a missing PLACE, not a missing operator — a
centre, a free corner, the crossing of a marked row and column, a rank in an order, a layer. The user's reframing: numbers were
places in the latent space and the arithmetic found its own rules by casting on them; bake the spatial concepts the same way, as
AREAS, and let the machine cast on them. Built as `M.fields()` — a bank of per-cell fields (own colour, colours on my lines and in
each direction, object centre, bbox corner / free corner, enclosed, inside a bbox, rank mod k, quadrant, distance, edge) — plus a
colour table keyed by the cell's place in one or two areas and the areas as conditions; the fast pass ranks the areas by how well
they predict which cells change. Two misses fell with no task token (`22a4bbc2` in 1.2 s, `2281f1f4` in 10.5 s); a third has its
place read and waits on the finish. The z-order domain (hull, covers, a topological layer order, the list as an output form)
found `22425bda` on the demos and correctly ABSTAINS on its test: two lines never meet, their order is unknowable. Costs paid:
a loosened background rule and a crowded candidate cap each broke a task until the gates caught them (retention 66/66, re-find
13/14 → fixed). 68 solved (1 assisted), 0 wrong.

### 2026-09-18 — ARC: the object bank, the MATRIX pass, and an A/B of the latent places; 71 solved
The latent move completed at both layers: object places (`ofields`) with learned tables for which objects act, their direction and
their colour — `22208ba4` fell to a learned direction per rank — and the search itself moved into the latent form: for a (domain,
move, order) the conditions are mask vectors and the colour sources value vectors, so the whole condition × colour space is two
matrix products per demo (`A @ Wᵀ == 0` and coverage), no candidate cap. Fast-pass solves land in 0.5–3 s; the pass's wall time
halved. A/B on a fresh slice (same ten tasks, same code, 300 s wall): banks off 1 solved / 0 wrong, banks on 2 / 0 — the gain is
the task whose answer is a place. Costs found and paid: a chance-fit table on 1×1 outputs gave the second wrong answer ever (rule:
a learned table must compress, ≥ 16 bits of evidence); the relational read on a 500-object grid took 25 silent minutes (an index
answers identically beyond 80 objects); leave-one-out re-derivations ran unbounded (now half the budget); learned slot values
placed before plain ones crowded a known program out (MDL: learned = later). Two re-find gates: 13/14 and 16/17, each miss a
crowding effect. 71 solved (1 assisted), 2 wrong ever, both recorded with their rule.

### 2026-09-18 — ARC: the reader census; the first expert is a read; 84 solved, 3 wrong ever
The effect reader alone over the 1000 training tasks (a minute): it names a domain on 78% of same-size tasks and the unsolved
mass sits inside the named domains — the gap is the slots, not the concepts; ~280 size-changing tasks form an untouched
'value / list' domain. Running the machine on the colour tasks the census said were one program away: 13 of 34 solved, four of
them by the learned latent tables on unseen tasks; one wrong (a chance condition on four 3x3 demos — rule: tiny evidence needs two
structurally different programs). The colour EXPERT built as a pure read — every select a mask over the demo objects, every
colour source a value, the exact pairs verified — reads 8 of 115 test-exact with 0 wrong at 5 s a task, including the task the
search answered wrong. A cross-task classifier from object place to 'acts' read nothing (0/111): the decision is per task; what
transfers is the kind of rule — the prior to train. The architecture the user named — router (the reader), experts per domain
that read and suggest, the matrix as verifier, the chain to mix them — is half built. 84 solved (1 assisted), 3 wrong ever.

### 2026-09-18 — ARC: the move expert; five new solves by reading; 89 solved
The second expert built as a read: per object the observed displacement, every direction and amount source compared per object,
gravity verified as a landing. Over the 153 move-domain tasks: 9 test-exact, 0 wrong, 5 s a task — five of them tasks the machine
had never solved, all confirmed through the pipeline's gates (rise by own height; fall right until stable; fly outward along the
diagonal; the attractor; align to a row). Two experts, both pure reads, zero wrong between them. The residue is the mixture:
128 of 153 read a move that is one stage of a mixed task — each expert must return its clean partial stage for the chain to
hand the residual on. 89 solved (1 assisted), 3 wrong ever.

### 2026-09-18 — ARC: the holographic expert and its decoder; the space answers without a written rule; 91 solved
The experts moved onto the holographic substrate: an object's place is the bundle of its (field, value) codes, a rule outcome a
code, the demos' (place ⊛ outcome) pairs superposed into one task bundle, a test object read by lighting its place against it —
the select is the bundle, nothing enumerated. On the colour domain the bundle reads every held-out demo object right on 20/105
tasks (the enumeration expert: 4), once where no select phrase exists in the vocabulary at all. A decoder (`HOLO:<fields>`) turns
the reading into a machine program; through the gates: 7 test-exact, two new solves entered the ledger, two would-be wrongs caught
in the test — both bundles keyed on absolute rank, which memorises a demo's places. That failure names the training: the field
PRIOR (which places transfer within a domain) is what to learn across tasks; within a task the demos are the training and the
bundle is the model. 91 solved (1 assisted), 3 wrong ever, retention 91/91.

### 2026-09-18 — ARC: one holographic mechanism, a registry of readers, and a prior that trains itself from the gates
The experts folded into one mechanism (encode / bundle / decode) with per-domain outcome readers, and the first cross-task
training: a field prior voted by the gates — test-exact bundles vote their fields up, wrong ones down, no human label. On the
colour domain one epoch took the bundle programs from 20 exact / 17 wrong to 17 exact / 2 wrong at the same budget, demoting
absolute rank and promoting colour, holes, height, rank mod 2. The wrong answers of the evening became the training signal of the
night. Transfer to the move domain read nothing — the move reader's outcomes must be sources, not constants; a reader fix, not a
mechanism fault. State: 91 solved (1 assisted), 3 wrong ever, retention 91/91; ResonatE's substrate now carries places, outcomes,
counts and relations alike, the machine executes what the space reads, the gates keep it never wrong.

### 2026-09-18 — ARC: the prior transfers across domains; 92 solved
The move reader's outcomes became sources (which direction and amount source explain the displacement), and the field prior the
colour domain trained was carried to the move domain unchanged: 46 test-exact / 2 wrong at epoch 0, 51 / 0 wrong at epoch 1; a
task never solved (`4364c1c4`) read whole and gated into the ledger; 68 partial move stages read for the mixture. Self-labelled
training on one domain improved the read on another — the property the reader was built for. 92 solved (1 assisted), 3 wrong ever.

### 2026-09-19 — ARC: the third reader, unsupervised training over the set, and an honest zero
Rays reader built (cast forms + learned strokes, actor arbitration by coverage), holographic stages wired into the chain with cached
reads, the prior trained unsupervised over all 670 same-size training tasks in 66 s an epoch (the gates as teacher, the held-out slice
never voting): stable transferring fields — shape, holes, colour, height, line — and a bundle that reads ~1350 partial stages across
the set but only ~10 whole tasks. Fresh slice 11: 0/10, 0 wrong. Half the slice is size-changing (no reader covers the value / crop /
list domain), the rest are mixtures the chain cannot compose within the wall. The reading grew where it could read; the fresh tasks
need a domain no reader has and a chain that composes faster. The user's two rules of the night stand: never train on the validation
set; and never-wrong is only a quality together with coverage — 92 answered, 3 wrong ever, coverage the number to grow.

### 2026-09-19 — ARC: knowledge transfer as a rule; the syllabus; 113 solved, 4 wrong ever
The user reframed the discipline: a child is taught physics, not the answer to exercise 7 — the memory may be TAUGHT concepts from
any public source, never a task's solution. The public arc-dsl (160 primitives, 400 solvers) mined for concepts only: the absent ones
by use are the bbox REGION family (taught as cell fields — the area form), concat, counter-mirror, downscale, template occurrences,
and a per-cell ray already present under another name. The value reader (a select isolates the answer's source object, a transform
makes the output) read 19 size-changing tasks, 6 new solves and the 4th wrong ever — a modal-shape rule absent from the vocabulary,
the answer key no gate can supply, now taught as a statistic. The census's cheapest gain: nine tasks whose whole program existed and
had never reached the answer rule. Acceptance of the lesson on 40 tasks: 6 solved, 0 wrong — by the old tools, the syllabus's value
being the pointing. ResonatE's own assessment holds: composition of learned operations is the real advantage; reading the right
composition from the demos alone is the unproven step, and the next build. 113 solved (1 assisted), 4 wrong ever.

### 2026-09-19 — ARC: recognition to execution — procedures retrieved by their effects
ResonatE's own gap named and closed at first strength: the memory's stored procedures gain a second access path — by recognising
their effects. Twenty-one skill families dreamed on ~1,500 random scenes (the program as its own simulator); each concrete program a
stored procedure with a name code; each procedure's prototype signature (what its transformation looks like: the effect reader's kind,
movers stopped against obstacles, dominant direction, new cells in lines or boxes, size / count / colour change) bound to that name in
one holographic memory. Held-out scenes: recognise → retrieve → execute exact 30 % within top-3, 43 % with the retrieved procedures'
variants explored — chance 2 %. Real solved tasks: an exact program in 0.1 s on 10 of 113 by retrieval alone; the dream router lifts
recognition of the solving family to 49 % top-3 given only the demos; gate 8/8. The kid was shown gravity on many scenes; now, seeing
things settle, it reaches for gravity first — and keeps alignment and shift in hand, never declaring. 113 solved, 4 wrong ever.

### 2026-09-19 — ARC: hierarchical retrieval with learned signatures; recognition finds 16 of 113 solved tasks by itself
A flat effect-memory broke when the procedure space grew from 135 to 492 (top-3 30 % → 5 %): no hand signature discriminates that
many. The fix was the same discipline that fixed the field prior — learn the reader: a bundle over (signature ⊛ label) with the field
subset chosen by leave-one-out on the dreams, and a hierarchy: family first, then the procedure within the family, then the variants.
Over the full space: family 72 %, procedure 57 %, with exploration 67 % on held-out dreams; on the real solved tasks recognition alone
finds an exact program on 16 of 113 in under a second each. The dreams label themselves; the readers pick what to look at; the
procedures are the same executable objects the arithmetic calls by name. 113 solved, 4 wrong ever.

### 2026-09-19 — seeds on the CLUTRR headline
Ten seeds, k = 8, 20k steps: exact at ten hops 0.988 ± 0.008 (0.975–1.000), wrong 0.003 ± 0.006; all lengths in
`results/kin_seeds/summary.json`. Every miss in seed 0 (30 stories) is a two-valid-answer chain with the two words
0.00–0.05 apart. A set-valued scorer by family simulation was tried and dropped: CLUTRR's kinship differs from
real-world kinship (wife's sister = "sister"), so it would be a second ground truth of our own making. Code and results
public at github.com/jinxmcg/resonate2; the page carries the ten-seed rows.

### 2026-09-19 — ARC: the memory grows from its own solves; the solve-predicting reader does not yet work
The loop the user asked for runs unattended over the training set: recognise by effect, explore, and every gated solve leaves its stage
skeletons — constants as slots, the select kept — as named procedures whose effects are bound in the memory, so the next puzzle that
shows the effect retrieves the block and fills it from its own demos. Two negatives recorded plainly: recognition alone finds 17 of
1,000 (fast confirmation of what search finds, not new coverage), and a reader trained to predict WHETHER a retrieved family will solve
is uninformative on the real distribution — 78 positives among 9,700 pairs, a one-field bundle. What it needs is what only more solves
give. 113 solved, 4 wrong ever.

### 2026-09-19 — 10¹² by continuation
513 s, three new anchors, everything 1.000 at 1–12 digits, and the five found programs execute unchanged on the new
table (mul limited by range at 12 digits). For the paper: the growth claim (Part 10) and the program-generalisation
claim (Part 14) compose — knowledge added by continuation is immediately usable by programs found before it existed.

### 2026-09-19 — ARC: the evaluation set through recognition alone — zero, and the reason
Validation, nothing learned: 120 evaluation tasks, recognition only (dreamed procedures, experts' whole reads, complete families):
0 exact, 0 wrong, 120 abstain. The readers name a domain on 74 of 81 same-size tasks — the same domains as training — yet not one
task admits a single-stage exact program: every evaluation task is a composition. The pieces are perceived; their assembly is not
read. This is the unproven step in ResonatE's own words, now measured as the whole of the gap on the set that matters.

### 2026-09-19 — the digit chain (README Part 23)
Place rows removed: eleven rows, one move, self-supervised; the five found programs unchanged at 1.00 for 1–24 digits
on k = 12 and k = 20. For the paper this replaces Parts 4/10/22's anchored table as the numbers representation, with
those parts kept as the path that led here (and as the "depth lives in the operator" evidence). Ceiling = trained
depth, not width; state it as such.

### 2026-09-19 — ARC: the composition sweep — what the machine can and cannot yet assemble; 117 solved
Over 661 same-size training tasks a bounded beam applied the readers' clean stages, re-read each residual, and kept every path
that closed exactly: 29 test-exact (the test as judge only), one wrong held by the pipeline's tiny-evidence gate, four new solves
gated into the ledger. But only six compositions were multi-stage, and nearly all the same expert on successive object classes;
genuinely cross-domain sequences were not found. The lesson is precise: the assembly is not the bottleneck now — the STAGES are.
The proposers offer whole-object recolours and moves; the tasks need stage shapes the readers do not yet produce (cell-level,
region-level, causal — a colour that comes from a covering or a crossing). The order reader waits for its training set; the next
lesson is the causal stage vocabulary, taught as dreamed families, judged by the sweep. 117 solved (1 assisted), 4 wrong ever.

### 2026-09-19 — ARC: relations as places, the ladder, correspondence; 123 solved
The review named the missing part of the substrate: every place was a single object's property; the tasks the readers name and cannot
close are relations. The relation bank (partner fields in the object place bank, partner sources and amounts, partner dream families,
relational signature fields) passed its gates (retention 117/117, prior voted every field up) and forms partner bundles on 11/55 census
tasks; the gated run over 70 relational-census tasks gave 6 solves, 0 wrong — none by a partner primitive, 64 at the wall. Dream
hierarchy refit with effect-family labels (77 % top-3 held-out; product place beats sum place; fine labels split same-effect variants).
Two pipeline faults found in the logs and fixed (GPU pickle on CPU workers; unbounded matrix pass). The ladder (bases reader: 27 per-base
holographic yes/no readers labelled by every dream's and every solve's own bases; read bases searched first, rung 2 without cursor moves,
votes as training signal) is built behind a flag; its A/B on the 64 wall-hits is running. Correspondence (Hungarian input<->output object
match) labels every object where the move reader left 3,464 unlabelled; census says 382 of 584 unsolved same-size tasks are fully
keep/move/recol/delete. For the paper: the substrate now carries places, relations, outcomes, readings and votes; the claim to test is
that reading-directed search (ladder) plus correspondence-labelled outcomes converts recognition into solves — the A/B is that test.
123 solved (1 assisted), 4 wrong ever.

### 2026-09-19 — ARC: the seven stages as a pass; three faults from the logs; wrong #5 and its rule; 123 solved
The user's procedure — read what is at play, what changed, the mechanism repeated over objects, try it per object and read each
object's PECULIARITY, compare, hand the residual on with the same destination — is now a pass (`mechanism.py`) and a debugging table
(`stage_report.py`). Its per-object deviation read (no-act / early-stop with the blocker / colour-diff / what it did instead) turns
mechanism + exceptions into one program; class-exact scoring handles mixtures; learned mechanisms are simulated leave-one-out. Three
tasks fall in seconds where the full pipeline never closed. Reading the run logs exposed three faults: device-bound holographic codes
(GPU measurements vs CPU pipeline), reading-based proposals cut by a cap before the search, and a memorised colour table that answered
wrong on the test (wrong #5). The fifth wrong gave the never-wrong rule its next clause: a fact taught by one demo of four is not
evidence. Retention 123/123. The census by stage over the unsolved names what is missing — mechanisms on the drawn side (repeat, tile,
move-onto-marker), and exception forms for the select — rather than domains. 123 solved (1 assisted), 5 wrong ever.

### 2026-09-19 — discovery from I/O pairs (README Part 24): stopped as brute force
triple found from (x, 3x) in 2 generations with double callable and no trace — the library effect is real and cheap to
show. Digit sum from add's program climbed to 0.75 exact but was stopped: random mutation with typed alphabets is still
a lottery. For the paper: keep the ladder results as "reuse compounds"; do NOT claim discovery. Owed if we return: the
learned proposer (controller reads slot kinds + failing place → instruction), scored as candidates-to-acceptance.

### 2026-09-19 — program by observation (README Part 25)
digitsum found from three worked examples in ~1 min / 150 candidates: the example is played as events (value, digit,
add(a,b), answer), moves derived from the lights (which slot holds what), a beam over the few genuine choices, then
Part-14's search over the recorded traces picks the loop. Brute force with everything else equal: not found at 7,800
candidates. Labelled 'digitsum' in the library; callable by name from a sentence. For the paper this is the learning
chapter's centrepiece: worked example → trace → program, values as variables and named operations as functions; the
given parts are the event vocabulary and the acceptance test. Owed: a second task (e.g. product of digits: multiply
named), and one where the named callable is itself a found program (triple via double) to show the library compounds.
