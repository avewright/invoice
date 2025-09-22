#!/usr/bin/env bash
set -euo pipefail

# Portable setup script for the playground notebook.
# - Creates a Python venv in .venv
# - Installs JupyterLab and notebook deps
# - Installs PaddleOCR with CPU by default, or GPU build when requested
#
# Usage examples:
#   ./setup.sh                # CPU install (default)
#   ./setup.sh --gpu cuda118  # GPU install for CUDA 11.8 wheel
#   ./setup.sh --gpu cuda121  # GPU install for CUDA 12.1 wheel
#   ./setup.sh --python 3.11  # Prefer Python 3.11 when creating venv

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN=""
GPU_MODE="cpu"
CUDA_TAG=""

print_usage() {
  echo "Usage: $0 [--python <3.x>] [--gpu <cuda118|cuda121>]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="python$2"
      shift 2
      ;;
    --gpu)
      GPU_MODE="gpu"
      CUDA_TAG="$2"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      print_usage
      exit 1
      ;;
  esac
done

# Resolve a python interpreter
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    echo "No python interpreter found" >&2
    exit 1
  fi
fi

echo "Using Python: $(${PYTHON_BIN} -V 2>&1)"

# Create venv
if [[ ! -d "${VENV_DIR}" ]]; then
  ${PYTHON_BIN} -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel setuptools

# Base requirements
BASE_REQS=(
  "numpy"
  "pillow"
  "matplotlib"
  "jupyterlab"
  "notebook"
)

pip install "${BASE_REQS[@]}"

# Install PaddleOCR and PaddlePaddle
PADDLEOCR_SPEC="paddleocr>=2.7.0"

if [[ "${GPU_MODE}" == "gpu" ]]; then
  case "${CUDA_TAG}" in
    cuda118)
      # PaddlePaddle GPU wheel for CUDA 11.8
      PADDLE_SPEC="paddlepaddle-gpu==2.6.1.post118"
      ;;
    cuda121)
      # PaddlePaddle GPU wheel for CUDA 12.1
      PADDLE_SPEC="paddlepaddle-gpu==2.6.1.post121"
      ;;
    *)
      echo "Unknown or missing CUDA tag for --gpu. Use cuda118 or cuda121." >&2
      exit 1
      ;;
  esac
else
  PADDLE_SPEC="paddlepaddle==2.6.1"
fi

echo "Installing: ${PADDLE_SPEC} and ${PADDLEOCR_SPEC}"
pip install "${PADDLE_SPEC}" "${PADDLEOCR_SPEC}"

echo "Setup complete. Activate with: source .venv/bin/activate"
echo "Launch Jupyter: ./run.sh"


