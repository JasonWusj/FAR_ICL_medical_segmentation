#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../../common.sh"

OUT="$OUT_BASE/same_backbone/far_icl_swap"
SPLIT="${SPLIT:-val}"
VARIANT="${VARIANT:-full}"
UNCERTAINTY="${UNCERTAINTY:-none}"
setting_tag="k${K}_n${CANDIDATE_N}_${UNCERTAINTY}"
if [[ -n "${SWAP_THRESHOLD:-}" ]]; then setting_tag+="_t${SWAP_THRESHOLD}"; fi
mkdir -p "$OUT/logs"
args=(--config configs/isic.yaml "${common_args[@]}" --set "initial_k=$K"
      --set "output=$OUT" --set "uncertainty=$UNCERTAINTY" --set "epochs=${EPOCHS:-40}"
      --set "patience=${PATIENCE:-8}" --set "lr=${LR:-0.0001}")
run_variant() {
  local command="$1" variant="$2" split="$3"
  local method="$variant" weight=1
  case "$variant" in
    full) method=swap ;;
    dice) method=swap; weight=0 ;;
    knn|repair_cover) ;;
    *) echo "Unknown VARIANT: $variant" >&2; return 2 ;;
  esac
  local tag="${variant}_${setting_tag}_${split}_seed${SEED}"
  "$PYTHON" -m far_icl.cli "$command" --method "$method" --split "$split" --tag "$tag" --resume \
    "${args[@]}" --set "swap_transition_weight=$weight" \
    --set "swap_threshold=${SWAP_THRESHOLD:-null}" 2>&1 | tee "$OUT/logs/$tag.log"
}
case "${1:-train}" in
  oracle)
    "$PYTHON" -m far_icl.cli swap-oracle --split val "${args[@]}" 2>&1 | tee "$OUT/logs/oracle.log"
    ;;
  train) run_variant run "$VARIANT" val ;;
  eval) run_variant evaluate "$VARIANT" "$SPLIT" ;;
  suite)
    for variant in knn repair_cover dice full; do run_variant run "$variant" val; done
    suffix="_${setting_tag}_val_seed${SEED}"
    for variant in knn repair_cover dice; do
      "$PYTHON" scripts/compare_results.py "$OUT/results/$variant$suffix/per_case.json" \
        "$OUT/results/full$suffix/per_case.json" --seed "$SEED" | tee "$OUT/logs/full_vs_$variant$suffix.json"
    done
    ;;
  *) echo "Usage: bash run.sh [oracle|train|eval|suite]" >&2; exit 2 ;;
esac
