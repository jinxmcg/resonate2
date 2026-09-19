# Optimizations

Things we noticed could be faster or more general, and whether they are done. Add one when found; tick it when fixed.

- [x] **Executor dispatch** — at every step the executor tested a mask for every possible instruction (~1,000 GPU
  syncs) and spawned one nested run per `CALL` variant. Now only the instructions present in the step are dispatched
  and all calls to the same sub-program at a step run as one nested batch. General (no operation knowledge). 10×:
  ground-truth `mul` check 80 s → 8 s. (`places.py`, 2026-09-16)
- [x] **Batched training of `area()`** — per-numeral Python loop replaced by one hop per place over the whole batch;
  same result, 8× faster (`labeled_areas.py`, 2026-09-15)
- [ ] **Multithreading across questions** — the executor already runs a batch of independent episodes, each with its
  own K slots, different programs allowed in the same batch (the search: 200 programs × 24 questions in flight). Not
  yet: nested calls within one question run one level at a time (a call tree could be flattened into one batch per
  depth); promotion to long-term memory is a single writer and should be a queue with validation before write.
- [ ] **Nested-call overhead** — a `CALL` converts places to digits and back (`number_of` → `put_number`) instead of
  passing the place itself; pass the place state directly and keep the walked digits cached.
- [ ] **`anything left?` peek** — walks all remaining places (9 hops) at every cycle start for both operand slots;
  could be maintained incrementally (one hop per LOAD) or read from a single light once the walk model exposes it.
- [ ] **Per-generation fitness noise** — the search scores each generation on a fresh random batch; a fixed batch (or
  a fixed core plus a fresh part) gives stabler selection.
- [x] **Instruction rows for programs-in-memory** — compose instructions as opcode ⊗ argument roles (≈ 26 rows) rather
  than one row per instruction (576), when programs are stored as chains. Done: 39 symbol rows + 4 role relations (`progmem2.py`).
- [ ] **Page size** — a place holds 6 instructions at k = 12; a wider program memory (k = 16–20) or a separate
  chain model per page length would raise it and cut page count.
- [x] **One memory** — merged by continuation (`onemem.py`, Part 16): one table, one `+`/`×`; new program names need no training. Per-place capacity 4 instructions on the shared table (6 dedicated) — see page size.

## Reader / language machine (2026-09-16)
- [x] **memoised, deduplicated calls into the places machine** — within a generation 92 % of calls were duplicates (3,461 → 276 distinct); the machine is deterministic, so `(program, x, y)` is run once and remembered. General.
- [x] **fixed number pool during search** (acceptance and tests draw fresh numbers) — with the memo, the arithmetic cost of a generation goes to ~0 after the first few. General to any search whose fitness calls the machine.
- [x] **opcode-level dispatch with per-row slot arguments** — one batched step per opcode present (~13) instead of one per distinct instruction string (~100 across a population). General; small gain here (2.6 → 2.3 s per sentence) because the remaining cost is the per-step Python loop.
- [x] **tight unrolling** — the loop unrolls to the batch's longest sentence, not a fixed cycle count.
- [ ] the per-step Python loop itself (≈ 450 steps × 13 opcodes per sentence-batch): CUDA graphs are out (data-dependent masks); the remaining lever is fewer steps — shorter programs (a ladder of rungs) or a stepping scheme where finished rows leave the batch.
- [ ] GPU: a 5090 would give ~1.5–2× on these runs (launch-latency bound), not 10×; the levers above matter more.
