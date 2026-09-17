#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
run_cli generate --kind marginal --split train "$@"
run_cli generate --kind marginal --split val "$@"
run_cli train --kind marginal --resume "$@"
