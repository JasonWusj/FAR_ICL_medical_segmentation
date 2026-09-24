#!/usr/bin/env bash
# Install FAR-ICL and official model code into an already working CUDA PyTorch env.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$(command -v python3)}"
if [[ ! -x "$PYTHON" ]]; then echo "Python not executable: $PYTHON" >&2; exit 2; fi
"$PYTHON" - <<'PY'
import sys
import torch
import torchvision

print("torch:", torch.__version__, "torchvision:", torchvision.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if sys.version_info < (3, 10):
    raise SystemExit("Python >=3.10 is required")
version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
if not ((2, 2) <= version < (3, 0)):
    raise SystemExit("This project requires PyTorch >=2.2,<3")
if not torch.cuda.is_available():
    raise SystemExit("Current PyTorch cannot see the GPU. Activate a CUDA-enabled environment first.")
print("GPU:", torch.cuda.get_device_name(0))
PY
"$PYTHON" -m pip install \
  'numpy>=1.24,<3' 'scipy>=1.10,<2' 'Pillow>=10' 'PyYAML>=6' \
  'surface-distance>=0.1' 'matplotlib>=3.7' 'tqdm>=4.66' \
  'pydantic>=1.10,<2' 'einops>=0.7,<0.9' 'validation==0.8.3' 'nibabel>=5'
"$PYTHON" -m pip install --no-deps -e .
mkdir -p external
if [[ ! -d external/UniverSeg/.git ]]; then
  git clone https://github.com/JJGO/UniverSeg.git external/UniverSeg
fi
git -C external/UniverSeg checkout 833a0c34c65e38d675e21bd48ddec6797cc03259
"$PYTHON" -m pip install --no-deps -e external/UniverSeg
if [[ "${INSTALL_TYCHE:-1}" == 1 ]]; then
  if [[ ! -d external/Tyche/.git ]]; then
    git clone https://github.com/mariannerakic/Tyche.git external/Tyche
  fi
  git -C external/Tyche checkout e8031e18f15583c2de8077d37b6d0ac5132a3aba
fi
"$PYTHON" - <<'PY'
import torch
import torchvision
import universeg

assert torch.cuda.is_available()
print("Ready:", torch.cuda.get_device_name(0), "torch", torch.__version__)
PY
printf '%s\n' 'Use: export PYTHON="'"$PYTHON"'" before running experiments.'
printf '%s\n' 'nnU-Net v2 installs on first supervised/nnunet run; model weights download on first use.'
