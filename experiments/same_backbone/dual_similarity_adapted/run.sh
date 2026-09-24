#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../../common.sh"
OUT="$OUT_BASE/same_backbone/dual_similarity_adapted"
mkdir -p "$OUT/logs"
for SPLIT in val test; do
  "$PYTHON" -m experiments.frozen_adaptations --method dual_similarity_adapted \
    --split "$SPLIT" --config configs/isic.yaml \
    "${common_args[@]}" --set "output=$OUT" \
    2>&1 | tee "$OUT/logs/${SPLIT}.log"
done
