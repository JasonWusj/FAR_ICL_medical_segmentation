#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
# Every ablation is also independently callable through the per-method scripts.
for seed in ${SEEDS:-42 43 44}; do
  SEED="$seed"
  for size in 1 2 4 8 16; do
    K="$size"
    for method in random knn; do
      run_cli run --method "$method" --split "$SPLIT" --tag "${method}_k${K}_${SPLIT}_s${SEED}" "$@"
    done
  done
  K=4
  for mode in image shape boundary failure full; do
    run_cli run --method utility --resume --split "$SPLIT" --set "feature_mode=$mode" \
      --tag "utility_${mode}_${SPLIT}_s${SEED}" "$@"
  done
  for mode in none tta context; do
    run_cli run --method mmr --resume --split "$SPLIT" --set "uncertainty=$mode" \
      --tag "mmr_${mode}_${SPLIT}_s${SEED}" "$@"
  done
  for method in shape failure utility mmr set adaptive adaptive_learned roles; do
    run_cli run --method "$method" --resume --split "$SPLIT" --tag "${method}_${SPLIT}_s${SEED}" "$@"
  done
  # One-pass utility uses image-only features; full utility performs two passes.
  for loss in regression pairwise listwise; do
    run_cli run --method utility --split "$SPLIT" --set "loss=$loss" \
      --tag "utility_${loss}_${SPLIT}_s${SEED}" "$@"
  done
done
run_cli report "$@"
