#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo ".venv not found. Run ./setup.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

export PYTHONUNBUFFERED=1

jupyter lab --ip=0.0.0.0 --port="${PORT:-8888}" --no-browser


