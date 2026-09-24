#!/usr/bin/env bash
# One-command ISIC 2018 exploratory repair experiment and paired test report.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${K:=2}"
: "${SEED:=42}"
export K SEED
source "$ROOT/scripts/common.sh"

MANIFEST="${MANIFEST:-/home/featurize/work/isic2018_full_with_test.csv}"
OUT="${OUT:-/home/featurize/work/far_icl_runs/isic2018_full_repair}"
TORCH_HOME="${TORCH_HOME:-/home/featurize/work/.cache/torch}"
CANDIDATE_N="${CANDIDATE_N:-12}"
INITIAL_K="${INITIAL_K:-2}"
UNCERTAINTY_SAMPLES="${UNCERTAINTY_SAMPLES:-2}"
LR="${LR:-0.00003}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-20}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-false}"
export TORCH_HOME

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 2
fi

mkdir -p "$OUT/logs" "$OUT/results"

# Validate the split protocol without decoding or hashing every image. The full
# check-data command is intentionally omitted because it rereads all image data.
"$PYTHON" - "$MANIFEST" <<'PY'
import csv
import sys
from collections import Counter
from pathlib import Path

manifest = Path(sys.argv[1]).resolve()
with manifest.open(newline="") as stream:
    rows = list(csv.DictReader(stream))
counts = Counter(row.get("split", "") for row in rows)
expected = {"train": 2594, "val": 100, "test": 1000}
if dict(counts) != expected:
    raise SystemExit(f"Expected ISIC Task 1 splits {expected}, found {dict(counts)}")
case_ids = [row.get("case_id", "") for row in rows]
if not all(case_ids) or len(case_ids) != len(set(case_ids)):
    raise SystemExit("Manifest case_id values must be nonempty and unique")
bad_identity = [
    row["case_id"] for row in rows
    if row.get("patient_id") != f"image:{row['case_id']}"
]
if bad_identity:
    raise SystemExit(
        "This exploratory script requires image-level IDs (patient_id=image:<case_id>); "
        f"first mismatch: {bad_identity[0]}"
    )
print(
    "Manifest protocol OK: 2594 train, 100 val, 1000 test; "
    "image-level identity (not patient-level)."
)
PY

COMMON=(
  --set identity_scope=image
  --set "manifest=$MANIFEST"
  --set "output=$OUT"
  --set "candidate_n=$CANDIDATE_N"
  --set "initial_k=$INITIAL_K"
  --set max_k=2
  --set "uncertainty_samples=$UNCERTAINTY_SAMPLES"
  --set "lr=$LR"
  --set "epochs=$EPOCHS"
  --set "patience=$PATIENCE"
  --set "save_predictions=$SAVE_PREDICTIONS"
)

echo "[1/4] Train repair ranker; validation selects best checkpoint."
SPLIT=val bash "$ROOT/scripts/run_repair.sh" "${COMMON[@]}" \
  2>&1 | tee "$OUT/logs/repair_train_val.log"

KNN_TAG="knn_test_k${K}_n${CANDIDATE_N}_seed${SEED}"
REPAIR_TAG="repair_test_lr${LR}_k${K}_seed${SEED}"
KNN_JSON="${KNN_RESULT:-}"

if [[ -n "$KNN_JSON" && ! -s "$KNN_JSON" ]]; then
  echo "KNN_RESULT does not exist or is empty: $KNN_JSON" >&2
  exit 2
fi

if [[ -z "$KNN_JSON" ]]; then
  KNN_JSON="$("$PYTHON" - "$OUT" "$MANIFEST" "$CONFIG" "$K" "$SEED" "$CANDIDATE_N" <<'PY'
import json
import sys
from pathlib import Path

from far_icl.config import load_config

root, manifest, config_path = Path(sys.argv[1]), str(Path(sys.argv[2]).resolve()), sys.argv[3]
k, seed, candidate_n = map(int, sys.argv[4:7])
expected = load_config(
    config_path,
    [
        f"manifest={manifest}", "identity_scope=image", f"k={k}",
        f"seed={seed}", f"candidate_n={candidate_n}",
    ],
)
keys = (
    "manifest", "identity_scope", "seed", "k", "candidate_n", "encoder",
    "encoder_repo", "encoder_weights", "segmenter", "segmenter_weights",
    "image_size", "mask_values", "bank_domains", "query_domains",
)
candidates = []
for result in (root / "results").glob("*/per_case.json"):
    config_file = result.with_name("config.json")
    if not config_file.is_file():
        continue
    try:
        cfg = json.loads(config_file.read_text())
        data = json.loads(result.read_text())
    except (OSError, json.JSONDecodeError):
        continue
    if any(cfg.get(key) != expected.get(key) for key in keys):
        continue
    rows = data.get("rows", [])
    if (
        data.get("identity_scope") == "image"
        and len(rows) == 1000
        and all(row.get("method") == "knn" and row.get("split") == "test" for row in rows)
    ):
        candidates.append((result.stat().st_mtime, result))
if candidates:
    print(max(candidates, key=lambda item: item[0])[1])
PY
  )"
fi

if [[ -z "$KNN_JSON" ]]; then
  echo "[2/4] Run KNN baseline on the held-out test set."
  SPLIT=test bash "$ROOT/scripts/eval_knn.sh" "${COMMON[@]}" --tag "$KNN_TAG" \
    2>&1 | tee "$OUT/logs/knn_test.log"
  KNN_JSON="$OUT/results/$KNN_TAG/per_case.json"
else
  echo "[2/4] Reuse compatible KNN test result: $KNN_JSON"
fi

echo "[3/4] Evaluate repair on the same held-out test images."
SPLIT=test bash "$ROOT/scripts/eval_repair.sh" "${COMMON[@]}" --tag "$REPAIR_TAG" \
  2>&1 | tee "$OUT/logs/repair_test.log"
REPAIR_JSON="$OUT/results/$REPAIR_TAG/per_case.json"

echo "[4/4] Paired image-level bootstrap comparison and summary report."
"$PYTHON" "$ROOT/scripts/compare_results.py" \
  "$KNN_JSON" "$REPAIR_JSON" --seed "$SEED" --samples "$BOOTSTRAP_SAMPLES" \
  | tee "$OUT/paired_test_comparison.json"
bash "$ROOT/scripts/report.sh" --set "output=$OUT"

cat > "$OUT/experiment_protocol.md" <<EOF
# FAR-ICL Repair exploratory experiment

- Data: ISIC 2018 Task 1, 2,594 train / 100 validation / 1,000 official test images.
- Training: gradients use the train split; validation targets measure val loss/regret and select the best checkpoint.
- Test comparison: KNN and repair use the same test manifest and case IDs; primary endpoint is paired image-macro Dice.
- Uncertainty: paired bootstrap with $BOOTSTRAP_SAMPLES resamples; see paired_test_comparison.json for the 95% interval.
- Identity scope: image-level because a patient mapping is unavailable. This is exploratory evidence, not patient-independent validation.
- Reproducibility: seed=$SEED, K=$K, candidate_n=$CANDIDATE_N, learning_rate=$LR.

Interpret the Dice difference together with its interval and the repair summary fields (second_pass_gain, negative_second_pass, repair_fraction, harm_fraction). A positive image-level interval does not establish benefit for every case or independent patients.
EOF

echo "Experiment outputs: $OUT"
echo "Paired comparison: $OUT/paired_test_comparison.json"
echo "Protocol note: $OUT/experiment_protocol.md"
