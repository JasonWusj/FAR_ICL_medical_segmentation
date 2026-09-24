#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
"$PYTHON" -m experiments.compare_all --out-base "$OUT_BASE" --seed "$SEED"
