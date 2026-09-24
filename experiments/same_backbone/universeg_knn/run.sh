#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../../common.sh"
OUT="$OUT_BASE/same_backbone/universeg_knn"
run_eval knn configs/isic.yaml "$OUT" val "knn_k${K}_val_seed${SEED}"
run_eval knn configs/isic.yaml "$OUT" test "knn_k${K}_test_seed${SEED}"
