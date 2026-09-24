#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../../common.sh"
OUT="$OUT_BASE/supervised/nnunet"
NNUNET_ENV="${NNUNET_ENV:-$REPO/.venv}"
if [[ ! -x "$NNUNET_ENV/bin/nnUNetv2_train" ]]; then
  "$NNUNET_ENV/bin/python" -m pip install 'nnunetv2==2.8.1'
fi
export nnUNet_raw="$OUT/nnUNet_raw"
export nnUNet_preprocessed="$OUT/nnUNet_preprocessed"
export nnUNet_results="$OUT/nnUNet_results"
mkdir -p "$OUT/logs"
"$PYTHON" -m experiments.prepare_nnunet --config configs/isic.yaml \
  "${common_args[@]}" --set "output=$OUT" --dataset-id "${NNUNET_DATASET_ID:-901}" \
  2>&1 | tee "$OUT/logs/prepare.log"
DATASET_ID="${NNUNET_DATASET_ID:-901}"
"$NNUNET_ENV/bin/nnUNetv2_plan_and_preprocess" -d "$DATASET_ID" --verify_dataset_integrity \
  2>&1 | tee "$OUT/logs/preprocess.log"
"$PYTHON" -m experiments.prepare_nnunet --config configs/isic.yaml \
  "${common_args[@]}" --set "output=$OUT" --dataset-id "$DATASET_ID" --write-split-only
"$NNUNET_ENV/bin/nnUNetv2_train" "$DATASET_ID" 2d 0 -tr "${NNUNET_TRAINER:-nnUNetTrainer}" \
  2>&1 | tee "$OUT/logs/train.log"
"$NNUNET_ENV/bin/nnUNetv2_predict" -i "$OUT/nnUNet_raw/Dataset${DATASET_ID}_ISIC2018/imagesTs" \
  -o "$OUT/predictions" -d "$DATASET_ID" -c 2d -f 0 -tr "${NNUNET_TRAINER:-nnUNetTrainer}" \
  -chk checkpoint_best.pth \
  2>&1 | tee "$OUT/logs/predict.log"
"$PYTHON" -m experiments.score_masks --config configs/isic.yaml \
  "${common_args[@]}" --set "output=$OUT" --pred-dir "$OUT/predictions" \
  --method nnunet_2d --split test
