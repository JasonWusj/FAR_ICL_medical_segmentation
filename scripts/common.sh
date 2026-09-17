#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -d "$ROOT/external/Tyche" ]]; then
  export PYTHONPATH="$ROOT/external/Tyche:$PYTHONPATH"
fi
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then PYTHON=python3; fi
CONFIG="${CONFIG:-configs/isic.yaml}"
SPLIT="${SPLIT:-val}"
SEED="${SEED:-42}"
K="${K:-4}"
run_cli() {
  "$PYTHON" -m far_icl.cli "$@" --config "$CONFIG" --set "seed=$SEED" --set "k=$K"
}
