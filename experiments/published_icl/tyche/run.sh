#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../../common.sh"
if [[ ! -d "$REPO/external/Tyche" ]]; then
  echo 'Tyche code missing. Run INSTALL_TYCHE=1 bash scripts/setup_linux.sh first.' >&2
  exit 2
fi
export PYTHONPATH="$REPO/external/Tyche:$PYTHONPATH"
OUT="$OUT_BASE/published_icl/tyche"
run_eval knn configs/tyche.yaml "$OUT" val "tyche_knn_k${K}_val_seed${SEED}" \
  --set "tyche_samples=${TYCHE_SAMPLES:-8}"
run_eval knn configs/tyche.yaml "$OUT" test "tyche_knn_k${K}_test_seed${SEED}" \
  --set "tyche_samples=${TYCHE_SAMPLES:-8}"
