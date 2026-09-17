#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
"${BOOTSTRAP_PYTHON:-python3}" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# Choose a PyTorch wheel index compatible with your Linux driver, e.g. cu124 or cpu.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
.venv/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url "$TORCH_INDEX_URL"
.venv/bin/python -m pip install -e '.[dev]'
mkdir -p external
if [[ ! -d external/UniverSeg/.git ]]; then git clone https://github.com/JJGO/UniverSeg.git external/UniverSeg; fi
git -C external/UniverSeg checkout 833a0c34c65e38d675e21bd48ddec6797cc03259
# Upstream dependencies use Pydantic v1. Do not install Tyche's old torch pin.
.venv/bin/python -m pip install 'pydantic>=1.10,<2' 'einops>=0.7,<0.9' 'validation==0.8.3' 'nibabel>=5'
.venv/bin/python -m pip install --no-deps -e external/UniverSeg
if [[ "${INSTALL_TYCHE:-0}" == 1 ]]; then
  if [[ ! -d external/Tyche/.git ]]; then git clone https://github.com/mariannerakic/Tyche.git external/Tyche; fi
  git -C external/Tyche checkout e8031e18f15583c2de8077d37b6d0ac5132a3aba
fi
.venv/bin/python -m pip freeze > environment.lock.txt
git -C external/UniverSeg rev-parse HEAD > universeg_revision.txt
printf '%s\n' 'Installed. Configure your data manifest before running experiments.'
