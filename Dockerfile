# Base image with CUDA + vLLM OpenAI server
FROM ghcr.io/vllm-project/vllm-openai:latest

WORKDIR /app

# Copy startup script
COPY start_vllm.sh /app/start_vllm.sh
RUN chmod +x /app/start_vllm.sh

# Defaults (override via Runpod env)
ENV MODEL=Qwen/Qwen2.5-VL-32B-Instruct \
    TP=auto \
    DTYPE=float16 \
    MAX_LEN=8192 \
    GPU_UTIL=0.60 \
    SWAP_GB=16 \
    BACKEND=mp \
    PORT=8080 \
    NCCL_IB_DISABLE=1 \
    NCCL_SOCKET_IFNAME=lo \
    PYTORCH_CUDA_USE_FLASH_SDP=0 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn

EXPOSE 8080

ENTRYPOINT ["/bin/bash","-lc","/app/start_vllm.sh"]


