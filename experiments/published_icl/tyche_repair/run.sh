#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../../common.sh"
if [[ ! -d "$REPO/external/Tyche" ]]; then
  echo 'Tyche code missing. Run INSTALL_TYCHE=1 bash scripts/setup_linux.sh first.' >&2
  exit 2
fi
export PYTHONPATH="$REPO/external/Tyche:$PYTHONPATH"

# Keep this output and checkpoint separate from the UniverSeg repair run.
# The frozen Tyche weights and KNN candidate bank are the same as tyche/run.sh.
OUT="$OUT_BASE/published_icl/tyche_repair"
KNN_RESULT="$OUT_BASE/published_icl/tyche/results/tyche_knn_k${K}_test_seed${SEED}/per_case.json"
if [[ ! -f "$KNN_RESULT" ]]; then
  echo 'Matching Tyche+KNN result missing; running its validation and test launcher first.'
  TYCHE_SAMPLES="${TYCHE_SAMPLES:-8}" bash experiments/published_icl/tyche/run.sh
fi
# Reject a stale baseline before the expensive supervision pass starts.
"$PYTHON" - "$KNN_RESULT" "$MANIFEST" "$IDENTITY_SCOPE" "$K" "$SEED" \
  "$CANDIDATE_N" "$INITIAL_K" "$UNCERTAINTY_SAMPLES" "${TYCHE_SAMPLES:-8}" <<'PY'
import json
import sys
from pathlib import Path

result, manifest, scope, k, seed, candidate_n, initial_k, uncertainty_samples, samples = sys.argv[1:]
cfg = json.loads(Path(result).with_name("config.json").read_text())
expected = {
    "identity_scope": scope,
    "k": int(k),
    "seed": int(seed),
    "candidate_n": int(candidate_n),
    "initial_k": int(initial_k),
    "max_k": int(k),
    "uncertainty_samples": int(uncertainty_samples),
    "tyche_samples": int(samples),
    "segmenter": "tyche",
    "encoder": "resnet50",
    "encoder_repo": None,
    "encoder_weights": None,
    "segmenter_weights": None,
}
bad = {key: (cfg.get(key), value) for key, value in expected.items() if cfg.get(key) != value}
if Path(cfg["manifest"]).resolve() != Path(manifest).resolve():
    bad["manifest"] = (cfg["manifest"], manifest)
if bad:
    raise SystemExit(f"Existing Tyche+KNN result uses different settings: {bad}. Use a fresh OUT_BASE.")
PY
mkdir -p "$OUT/logs"
train_args=(
  --set "epochs=${EPOCHS:-100}" --set "patience=${PATIENCE:-20}"
  --set "lr=${LR:-0.00003}" --set "tyche_samples=${TYCHE_SAMPLES:-8}"
)

"$PYTHON" -m far_icl.cli run --method repair --split val --resume \
  --tag "tyche_repair_k${K}_val_seed${SEED}" --config configs/tyche.yaml \
  "${common_args[@]}" "${train_args[@]}" --set "output=$OUT" \
  2>&1 | tee "$OUT/logs/train_val.log"

"$PYTHON" -m far_icl.cli evaluate --method repair --split test \
  --tag "tyche_repair_k${K}_test_seed${SEED}" --config configs/tyche.yaml \
  "${common_args[@]}" "${train_args[@]}" --set "output=$OUT" \
  2>&1 | tee "$OUT/logs/test.log"

"$PYTHON" scripts/compare_results.py "$KNN_RESULT" \
  "$OUT/results/tyche_repair_k${K}_test_seed${SEED}/per_case.json" \
  --seed "$SEED" | tee "$OUT/paired_test_comparison.json"
