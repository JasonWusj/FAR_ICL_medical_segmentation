#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
method="$1"
shift
export TORCH_HOME="${TORCH_HOME:-/home/featurize/work/.cache/torch}"
export PYTHONUNBUFFERED=1
args=(
  --method "$method"
  --manifest "${MANIFEST:-/home/featurize/work/isic2018_full_with_test.csv}"
  --out-base "${OUT_BASE:-/home/featurize/work/far_icl_comparison_runs}"
  --split "${SPLIT:-val}" --identity-scope "${IDENTITY_SCOPE:-image}"
  --k "${K:-2}" --seed "${SEED:-42}" --initial-k "${INITIAL_K:-2}"
  --mmr-lambda "${MMR_LAMBDA:-0.5}" --random-repeats "${RANDOM_REPEATS:-1}"
)
if [[ -n "${CANDIDATE_N:-}" ]]; then args+=(--candidate-n "$CANDIDATE_N"); fi
if [[ -n "${UNIVERSEG_WEIGHTS:-}" ]]; then args+=(--segmenter-weights "$UNIVERSEG_WEIGHTS"); fi
if [[ -n "${RESNET50_WEIGHTS:-}" ]]; then args+=(--encoder-weights "$RESNET50_WEIGHTS"); fi
exec "${PYTHON:-$PWD/.venv/bin/python}" -m experiments.baselines.run "${args[@]}" "$@"
