#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
run_cli run --method adaptive_learned --split "$SPLIT" --resume "$@"
