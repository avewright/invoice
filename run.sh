#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

# Defaults for serving
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
WORKERS="${WORKERS:-1}"
RELOAD="${RELOAD:-0}"

# Defaults for setup
PYTHON_BIN=""
GPU_MODE="cpu"
CUDA_TAG=""

print_usage() {
  echo "Usage: $0 <command> [options]"
  echo
  echo "Commands:"
  echo "  setup   [--python <3.x>] [--gpu <cuda118|cuda121>]    Create venv and install deps (project + Jupyter + PaddleOCR)"
  echo "  serve                                               Start FastAPI service (uses env: PORT HOST WORKERS RELOAD)"
  echo "  jupyter                                             Launch Jupyter Lab (env: JUPYTER_PORT)"
  echo
  echo "Examples:"
  echo "  $0 setup --python 3.11"
  echo "  GPU=1 $0 setup --gpu cuda121"
  echo "  PORT=8000 RELOAD=1 $0 serve"
  echo "  JUPITER_PORT=8888 $0 jupyter"
}

ensure_python_and_venv() {
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    PYTHON_BIN="${VENV_DIR}/bin/python"
  elif [[ -n "${PYTHON_BIN}" ]]; then
    : # respect user-provided PYTHON_BIN like python3.11
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "No python interpreter found" >&2
    exit 1
  fi

  if [[ ! -d "${VENV_DIR}" ]]; then
    ${PYTHON_BIN} -m venv "${VENV_DIR}"
  fi

  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"

  python -m pip install --upgrade pip wheel setuptools
}

install_project_requirements() {
  if [[ -f "${PROJECT_ROOT}/requirements.txt" ]]; then
    pip install -r "${PROJECT_ROOT}/requirements.txt"
    echo "Installing project requirements"
  else
    pip install fastapi "uvicorn[standard]" transformers torch accelerate safetensors
  fi
}

install_notebook_and_paddle() {
  local base_reqs=(
    "numpy"
    "pillow"
    "matplotlib"
    "jupyterlab"
    "notebook"
  )
  pip install "${base_reqs[@]}"

  local paddle_spec="paddlepaddle==2.6.1"
  if [[ "${GPU_MODE}" == "gpu" ]]; then
    case "${CUDA_TAG}" in
      cuda118)
        paddle_spec="paddlepaddle-gpu==2.6.1.post118"
        ;;
      cuda121)
        paddle_spec="paddlepaddle-gpu==2.6.1.post121"
        ;;
      *)
        echo "Unknown or missing CUDA tag for --gpu. Use cuda118 or cuda121." >&2
        exit 1
        ;;
    esac
  fi
  echo "Installing: ${paddle_spec} and paddleocr>=2.7.0"
  pip install "${paddle_spec}" "paddleocr>=2.7.0"
}

cmd_setup() {
  # Parse flags for setup
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
        echo "Unknown arg for setup: $1" >&2
        print_usage
        exit 1
        ;;
    esac
  done

  ensure_python_and_venv
  install_project_requirements
  install_notebook_and_paddle

  echo "Setup complete. Activate with: source .venv/bin/activate"
}

cmd_serve() {
  ensure_python_and_venv

  if ! command -v uvicorn >/dev/null 2>&1; then
    install_project_requirements
  fi

  local app_module="llm:app"
  local uvicorn_opts=("--host" "${HOST}" "--port" "${PORT}")
  if [[ "${RELOAD}" == "1" ]]; then
    uvicorn_opts+=("--reload")
  fi
  if [[ "${WORKERS}" != "1" ]]; then
    uvicorn_opts+=("--workers" "${WORKERS}")
  fi

  exec uvicorn "${app_module}" "${uvicorn_opts[@]}"
}

cmd_jupyter() {
  ensure_python_and_venv
  if ! python -c 'import jupyterlab' >/dev/null 2>&1; then
    pip install jupyterlab notebook
  fi
  local jp_port="${JUPYTER_PORT:-8888}"
  exec jupyter lab --ip 0.0.0.0 --port "${jp_port}" --no-browser --NotebookApp.token=''
}

main() {
  local cmd="${1:-serve}"
  shift || true
  case "${cmd}" in
    setup)
      cmd_setup "$@"
      ;;
    serve)
      cmd_serve "$@"
      ;;
    jupyter)
      cmd_jupyter "$@"
      ;;
    -h|--help|help)
      print_usage
      ;;
    *)
      echo "Unknown command: ${cmd}" >&2
      print_usage
      exit 1
      ;;
  esac
}

main "$@"

