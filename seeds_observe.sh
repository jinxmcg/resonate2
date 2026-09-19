#!/bin/zsh
# seeds for the observation results (run after seeds_all.sh): each seed re-plays the worked examples and re-finds the program
P="/mnt/nvme990/latent/.venv/bin/python -u"; M="--path results/digit_chain_k20/model.pt --slots 6 --max-digits 3 --np 14"
until [ -f logs/seeds/all.done ]; do sleep 60; done
for s in 0 1 2 3 4 5 6 7 8 9; do
  mkdir -p results/seeds/observe_s$s
  ${=P} observe.py --target sumd --examples-shown 123,90,5 ${=M} --seed $s --no-label --out results/seeds/observe_s$s > logs/seeds/observe_sumd_s$s.log 2>&1
  ${=P} observe.py --target iseven --examples-shown 124,37 ${=M} --seed $s --no-label --out results/seeds/observe_s$s > logs/seeds/observe_iseven_s$s.log 2>&1
  ${=P} observe.py --target square --examples-shown 7,12 ${=M} --seed $s --no-label --out results/seeds/observe_s$s > logs/seeds/observe_square_s$s.log 2>&1
done
echo DONE > logs/seeds/observe.done
