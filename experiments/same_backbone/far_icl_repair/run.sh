#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../../common.sh"
OUT="$OUT_BASE/same_backbone/far_icl_repair"
mkdir -p "$OUT/logs"
"$PYTHON" -m far_icl.cli run --method repair --split val --resume \
  --tag "repair_k${K}_val_seed${SEED}" --config configs/isic.yaml \
  "${common_args[@]}" --set "output=$OUT" \
  --set "epochs=${EPOCHS:-100}" --set "patience=${PATIENCE:-20}" \
  --set "lr=${LR:-0.00003}" 2>&1 | tee "$OUT/logs/train_val.log"
"$PYTHON" -m far_icl.cli evaluate --method repair --split test \
  --tag "repair_k${K}_test_seed${SEED}" --config configs/isic.yaml \
  "${common_args[@]}" --set "output=$OUT" \
  --set "epochs=${EPOCHS:-100}" --set "patience=${PATIENCE:-20}" \
  --set "lr=${LR:-0.00003}" 2>&1 | tee "$OUT/logs/test.log"
