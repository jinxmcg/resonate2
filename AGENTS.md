# Working agreement — reasoning_ant

- This folder is mine. `/mnt/nvme990/geocore/reasoning` (the other agent's) and `/mnt/nvme990/geocore/resonate` are
  read-only: import by path with `sys.dont_write_bytecode = True`, never write there.
- **Status every 2 minutes** while anything runs: phase, step/total, elapsed, **ETA**, and the metrics that process
  actually has — training: loss, LR, exact accuracy, MRR where a ranked candidate set exists; grounding/eval: what
  finished and what is next. Never invent MRR/LR for a job that has none.
- **No buffered output.** Launch every long script with `python -u` (and print with `flush=True`),  writing straight to a log file
  (`> logs/<run>.log 2>&1`), never through a `grep | tail` pipe (that hides progress until the end). Monitor the log.
- **Checkpoints so runs can be restarted.** Training scripts save `ckpt.pt` (model, optimizer, step, RNG, config) every
  500 steps and at the end, and accept `--resume`; evaluation-only phases save their JSON as soon as each part is done.
  Never kill a run to fix logging unless it can be resumed.
- Vectorise the inner loop before launching: a per-item Python loop over 200 numerals × 3 hops per step is CPU-bound
  (this is what made `labeled_areas.py` take 20+ min instead of 3). Batch by shape.
- Held-out numbers are the result; training accuracy is a sanity check. Report chance level next to every exact score.
- Keep failed runs and their logs; the README table lists every run, including the stopped ones.
