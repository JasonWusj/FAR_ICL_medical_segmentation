#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../../common.sh"
OUT="$OUT_BASE/supervised/unet"
mkdir -p "$OUT/logs"
"$PYTHON" -m experiments.unet_train --config configs/isic.yaml \
  "${common_args[@]}" --set "output=$OUT" \
  --epochs "${UNET_EPOCHS:-100}" --patience "${UNET_PATIENCE:-20}" \
  --batch-size "${UNET_BATCH_SIZE:-8}" 2>&1 | tee "$OUT/logs/train_eval.log"
