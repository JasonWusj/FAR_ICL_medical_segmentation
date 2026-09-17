#!/usr/bin/env bash
set -Eeuo pipefail
export CONFIG="${CONFIG:-configs/tyche.yaml}"
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
run_cli run --method "${METHOD:-mmr}" --resume --split "$SPLIT" "$@"
