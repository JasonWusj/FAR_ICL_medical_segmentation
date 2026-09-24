#!/usr/bin/env bash
set -Eeuo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TORCH_HOME="${TORCH_HOME:-/home/featurize/work/.cache/torch}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
MANIFEST="${MANIFEST:-/home/featurize/work/isic2018_full_with_test.csv}"
OUT_BASE="${OUT_BASE:-/home/featurize/work/far_icl_paper_runs}"
IDENTITY_SCOPE="${IDENTITY_SCOPE:-image}"
K="${K:-2}"
SEED="${SEED:-42}"
CANDIDATE_N="${CANDIDATE_N:-12}"
INITIAL_K="${INITIAL_K:-2}"
UNCERTAINTY_SAMPLES="${UNCERTAINTY_SAMPLES:-2}"
if [[ ! -x "$PYTHON" ]]; then echo "Python environment missing: $PYTHON (run bash scripts/setup_linux.sh)" >&2; exit 2; fi
if [[ ! -f "$MANIFEST" ]]; then echo "Manifest missing: $MANIFEST" >&2; exit 2; fi
if (( CANDIDATE_N < K || CANDIDATE_N < INITIAL_K )); then echo 'CANDIDATE_N must cover K and INITIAL_K' >&2; exit 2; fi
common_args=(
  --set "manifest=$MANIFEST" --set "identity_scope=$IDENTITY_SCOPE"
  --set "seed=$SEED" --set "k=$K" --set "candidate_n=$CANDIDATE_N"
  --set "initial_k=$INITIAL_K" --set "max_k=$K"
  --set "uncertainty_samples=$UNCERTAINTY_SAMPLES" --set save_predictions=false
)
run_eval() {
  local method="$1" config="$2" output="$3" split="$4" tag="$5"
  mkdir -p "$output/logs"
  "$PYTHON" -m far_icl.cli evaluate --method "$method" --split "$split" \
    --tag "$tag" --config "$config" "${common_args[@]}" --set "output=$output" \
    2>&1 | tee "$output/logs/${tag}.log"
}
