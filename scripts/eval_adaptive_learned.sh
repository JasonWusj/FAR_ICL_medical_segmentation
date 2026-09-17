#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
run_cli evaluate --method adaptive_learned --split "$SPLIT" "$@"
