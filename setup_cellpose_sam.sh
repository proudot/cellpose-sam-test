#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="cellpose-sam"
PYTHON_VERSION="3.10"

echo "[1/4] Creating conda environment: ${ENV_NAME}"
conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"

echo "[2/4] Activating environment"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "[3/4] Installing PyTorch"
# CPU-only default: works everywhere.
python -m pip install --upgrade pip
python -m pip install torch torchvision

# If you have an NVIDIA GPU and want CUDA wheels instead, comment the line above
# and use one of the official PyTorch install commands matching your CUDA version.

echo "[4/4] Installing Cellpose-SAM dependencies"
python -m pip install cellpose
python -m pip install numpy scipy matplotlib scikit-image tifffile imageio tqdm

echo
echo "Environment ready."
echo "Activate with:"
echo "  conda activate ${ENV_NAME}"
echo
echo "Test imports with:"
echo "  python - <<'PY'"
echo "from cellpose import models"
echo "m = models.CellposeModel(pretrained_model='cpsam', gpu=False)"
echo "print('Cellpose-SAM ready')"
echo "PY"
