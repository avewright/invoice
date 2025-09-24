#!/usr/bin/env bash
set -euo pipefail

#
# vLLM startup script for Qwen2.5-VL-32B on multi-GPU (A40)
# Environment variables can override the defaults below.
#

# CUDA / NCCL defaults for single-node multi-GPU
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# PyTorch / vLLM tuning
export PYTORCH_CUDA_USE_FLASH_SDP="${PYTORCH_CUDA_USE_FLASH_SDP:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

# User-tunable knobs
MODEL="${MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
PORT="${PORT:-8080}"
TP="${TP:-4}"
DTYPE="${DTYPE:-float16}"
MAX_LEN="${MAX_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.60}"
SWAP_GB="${SWAP_GB:-16}"
BACKEND="${BACKEND:-mp}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Optional cleanup of stray processes (disabled by default)
# pkill -f vllm || true; pkill -f python || true

if [[ "${TP}" == "auto" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    # Count GPUs from CUDA_VISIBLE_DEVICES (comma-separated list)
    IFS=',' read -r -a _gpus <<< "${CUDA_VISIBLE_DEVICES}"
    TP="${#_gpus[@]}"
  else
    # Fallback: use nvidia-smi to count GPUs
    TP="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    [[ -z "${TP}" || "${TP}" == "0" ]] && TP=1
  fi
fi

echo "Starting vLLM with model=${MODEL}, TP=${TP}, dtype=${DTYPE}, max_len=${MAX_LEN}, util=${GPU_UTIL}, port=${PORT}"

exec vllm serve "${MODEL}" \
  --tensor-parallel-size "${TP}" \
  --distributed-executor-backend "${BACKEND}" \
  --dtype "${DTYPE}" \
  --enforce-eager \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --swap-space "${SWAP_GB}" \
  --port "${PORT}" \
  ${EXTRA_ARGS}


