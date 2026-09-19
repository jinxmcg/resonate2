#!/bin/zsh
# ten seeds of the headline CLUTRR run (k=8, 20k steps, 64 probes), two at a time
P="/mnt/nvme990/latent/.venv/bin/python -u"
for pair in "0 1" "2 3" "4 5" "6 7" "8 9"; do
  for s in ${=pair}; do ${=P} kinship_ops.py --k 8 --steps 20000 --probes 64 --seed $s --out results/kin_seeds/s$s > logs/kin_seed_$s.log 2>&1 & done
  wait
done
echo DONE > logs/kin_seeds.done
