#!/usr/bin/env bash
# install_nerfstudio_conda_exact.sh
# Conda-first minimal installer that adheres to:
#  - Torch 2.1.2 with CUDA 11.8 (recommended by Nerfstudio docs)
#  - numpy == 1.26.4
#
# References:
#  - Nerfstudio installation recommends Torch 2.1.2 + CUDA 11.8.
#  - NumPy 1.26.4 release (we pin this exact version).
#
set -eo pipefail

ENV_NAME="FiGS"
PYTHON_VERSION="3.10"
NUMPY_VERSION="1.26.4"
TORCH_VER="2.1.2"
TORCHVISION_VER="0.16.2"
TORCHAUDIO_VER="2.1.2"   # torchaudio pair; acceptable to install alongside
CUDA_MAJOR_MINOR="11.8"
TINY_CUDA_NN_REPO="git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

# Help
if [[ "${1-}" == "-h" ]] || [[ "${1-}" == "--help" ]]; then
  cat <<EOF
Usage: $0

This script creates a conda env named '${ENV_NAME}' with:
  - python ${PYTHON_VERSION}
  - numpy ${NUMPY_VERSION}
  - pytorch ${TORCH_VER} + CUDA ${CUDA_MAJOR_MINOR} (via conda channels)
  - tiny-cuda-nn (pip from NVlabs repo)
  - nerfstudio (pip)
EOF
  exit 0
fi

echo "=========================================="
echo "STEP 1: Nerfstudio Base Installation"
echo "=========================================="
echo ""

command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found in PATH. Install Miniconda/Anaconda first."; exit 1; }

# initialize conda for this shell (works for bash/zsh)
eval "$(conda shell.bash hook)"

echo "=== Creating conda env: ${ENV_NAME} (python ${PYTHON_VERSION}) ==="
conda create -n "${ENV_NAME}" -y python=="${PYTHON_VERSION}" numpy=="${NUMPY_VERSION}"

echo "=== Activating ${ENV_NAME} ==="
conda activate "${ENV_NAME}"

echo "=== Upgrading pip inside env ==="
python -m pip install --upgrade pip

conda env config vars set PYTHONNOUSERSITE=1
conda deactivate
conda activate "${ENV_NAME}"

echo "=== Installing PyTorch ${TORCH_VER} + CUDA ${CUDA_MAJOR_MINOR} ==="
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

echo "=== Creating constraints file for critical dependencies ==="
cat > /tmp/constraints.txt <<EOF
numpy==${NUMPY_VERSION}
torch==2.1.2+cu118
torchvision==0.16.2+cu118
EOF

echo "=== Installing CUDA Toolkit ==="
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y

echo "=== Installing tiny-cuda-nn (torch bindings) ==="
# Use local copy instead of cloning from GitHub
TCNN_LOCAL_PATH="/home/cyrus/workspaces/StanfordMSL/tiny-cuda-nn/bindings/torch"
if [ -d "$TCNN_LOCAL_PATH" ]; then
  echo "Using local tiny-cuda-nn at: $TCNN_LOCAL_PATH"
  python -m pip install ninja "$TCNN_LOCAL_PATH" --no-build-isolation
else
  echo "Local copy not found, falling back to GitHub (may fail with network issues)"
    python -m pip install ninja "${TINY_CUDA_NN_REPO}" --no-build-isolation
fi

echo "=== Installing COLMAP ==="
conda install -y -c conda-forge colmap

echo "=== Installing ffmpeg ==="
conda install -y -c conda-forge ffmpeg

if [ -d "./Hierarchical-Localization" ]; then
    echo "=== Installing Hierarchical-Localization ==="
    pip install --constraint /tmp/constraints.txt -e ./Hierarchical-Localization
fi

echo "=== Installing nerfstudio via pip ==="
python -m pip install nerfstudio

ns-install-cli

echo
echo "=== Quick verification ==="
python - <<PY
import sys, importlib
import numpy as np
import torch
print("python:", sys.version.split()[0])
print("numpy:", np.__version__)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
try:
    print("cuda device count:", torch.cuda.device_count())
    print("cuda current device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
except Exception as e:
    print("cuda info error:", e)
# sanity checks for versions (exit non-zero if mismatch)
if np.__version__ != "${NUMPY_VERSION}":
    print("ERROR: numpy version mismatch (expected ${NUMPY_VERSION}, got", np.__version__, ")")
    sys.exit(2)
if not torch.__version__.startswith("${TORCH_VER}"):
    print("WARNING: torch version does not start with ${TORCH_VER} -- installed:", torch.__version__)
PY

echo
echo "=== Nerfstudio base installation complete ==="

echo "=== Uninstalling JIT gsplat & reinstalling functioning version ==="
pip uninstall gsplat -y
pip install gsplat==1.4.0 --index-url https://docs.gsplat.studio/whl/pt21cu118

echo ""
echo "=========================================="
echo "Installing FiGS-specific dependencies"
echo "=========================================="
echo ""

echo "=== Installing misc dependencies ==="
# conda install -c conda-forge albumentations --freeze-installed
pip install albumentations --no-deps
conda install -y -c conda-forge qpsolvers
conda install -y -c conda-forge tabulate
conda install -y -c conda-forge cython 
pip install ipykernel --no-deps
pip install ipympl --no-deps
pip install rich imageio[ffmpeg]

# Install editable packages if they exist
if [ -d "../acados/interfaces/acados_template" ]; then
    echo "=== Installing acados_template ==="
    pip install -e ../acados/interfaces/acados_template
fi

# echo "=== Installing Remaining conda dependencies ==="
# conda install -y -c conda-forge albumentations qpsolvers gdown ipykernel ipympl "matplotlib<3.9" tqdm tabulate cython "numpy==${NUMPY_VERSION}"

# # Install pip packages
# echo "=== Installing FiGS pip dependencies ==="
# pip install rich imageio[ffmpeg]

echo "=== Installing FiGS (current package) ==="
pip install -e .

echo ""
echo "=========================================="
echo "Patching COLMAP parameter names"
echo "=========================================="
echo ""

# Fix deprecated SIFT parameter names in nerfstudio's colmap_utils.py
COLMAP_UTILS_PATH="${CONDA_PREFIX}/lib/python${PYTHON_VERSION}/site-packages/nerfstudio/process_data/colmap_utils.py"

if [ -f "$COLMAP_UTILS_PATH" ]; then
    echo "=== Patching COLMAP parameters in colmap_utils.py ==="
    # Replace SiftExtraction with FeatureExtraction
    sed -i 's/--SiftExtraction\.use_gpu/--FeatureExtraction.use_gpu/g' "$COLMAP_UTILS_PATH"
    # Replace SiftMatching with FeatureMatching
    sed -i 's/--SiftMatching\.use_gpu/--FeatureMatching.use_gpu/g' "$COLMAP_UTILS_PATH"
    echo "Successfully patched COLMAP parameters"
else
    echo "WARNING: Could not find colmap_utils.py at expected location: $COLMAP_UTILS_PATH"
fi