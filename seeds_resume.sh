#!/bin/zsh
# resume the seed queue where it left off (skips seeds whose result file exists). Run overnight: nohup ./seeds_resume.sh > logs/seeds_resume.log 2>&1 &
P="/mnt/nvme990/latent/.venv/bin/python -u"; mkdir -p results/seeds logs/seeds
for s in 0 1 2 3 4 5 6 7 8 9; do
  [ -f results/seeds/digits_s$s/digits.json ] && continue
  [ -f results/seeds/digit_chain_s$s/model.pt ] || ${=P} digit_chain.py --k 20 --steps 6000 --seed $s --out results/seeds/digit_chain_s$s > logs/seeds/digit_chain_s$s.log 2>&1
  mkdir -p results/seeds/digits_s$s; ${=P} digits.py --path results/seeds/digit_chain_s$s/model.pt --np 24 --n 40 --seed $s --out results/seeds/digits_s$s > logs/seeds/digits_s$s.log 2>&1
done
for s in 0 1 2 3 4; do [ -f results/seeds/places_s$s/places.json ] || { mkdir -p results/seeds/places_s$s; ${=P} places.py --seed $s --out results/seeds/places_s$s > logs/seeds/places_s$s.log 2>&1; }; done
for s in 0 1 2 3 4 5 6 7 8 9; do [ -f results/seeds/story_s$s/story.json ] || { mkdir -p results/seeds/story_s$s; ${=P} story.py --seed $s --out results/seeds/story_s$s > logs/seeds/story_s$s.log 2>&1; }; done
for s in 0 1 2; do [ -f results/seeds/read_math_s$s/read_math.json ] || { mkdir -p results/seeds/read_math_s$s; ${=P} read_math.py --lk 32 --steps 15000 --ladder --seed $s --out results/seeds/read_math_s$s > logs/seeds/read_math_s$s.log 2>&1; }; done
echo DONE > logs/seeds/all.done
M="--path results/digit_chain_k20/model.pt --slots 6 --max-digits 3 --np 14"
for s in 0 1 2 3 4 5 6 7 8 9; do
  mkdir -p results/seeds/observe_s$s
  for t in "sumd 123,90,5" "iseven 124,37" "square 7,12"; do set -- ${=t}; [ -f results/seeds/observe_s$s/$1.json ] || ${=P} observe.py --target $1 --examples-shown $2 ${=M} --seed $s --no-label --out results/seeds/observe_s$s > logs/seeds/observe_$1_s$s.log 2>&1; done
done
echo DONE > logs/seeds/observe.done
