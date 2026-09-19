#!/bin/zsh
P="/mnt/nvme990/latent/.venv/bin/python -u"
${=P} kinship_ops.py --k 8 --steps 20000 --probes 64 --seed 0 --out results/kin_full_k8 > logs/kin_full_k8.log 2>&1
${=P} kinship_ops.py --k 8 --steps 40000 --probes 64 --seed 0 --out results/kin_full_k8_40k > logs/kin_full_k8_40k.log 2>&1
echo DONE > logs/kin_sweep.done
