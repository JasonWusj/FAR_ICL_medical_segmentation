#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
run_cli generate --kind correction --split train "$@"
run_cli generate --kind correction --split val "$@"
run_cli train --kind correction --resume "$@"
