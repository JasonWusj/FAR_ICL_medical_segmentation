#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../../common.sh"
if [[ -z "${IRIS_PRED_DIR:-}" ]]; then
  echo 'Iris official ISIC inference is unavailable here. Set IRIS_PRED_DIR to predictions from an independently obtained Iris implementation (case_id.png).' >&2
  exit 2
fi
OUT="$OUT_BASE/published_icl/iris_external"
mkdir -p "$OUT"
"$PYTHON" -m experiments.score_masks --config configs/isic.yaml \
  "${common_args[@]}" --set "output=$OUT" --pred-dir "$IRIS_PRED_DIR" \
  --method iris_external --split test --support-k "${IRIS_SUPPORT_K:--1}"
