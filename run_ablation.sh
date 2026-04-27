#!/usr/bin/env bash
set -euo pipefail

# Edit this list to the proportions you want to test
proportions=(0.1 0.3 0.5 0.6 0.8)

for p in "${proportions[@]}"; do
  echo "Running proportion=${p}"
  python run_ablation_experiments.py --mode ablation --num-workers 4 \
    --override "experiment.proportion=${p}"
done
