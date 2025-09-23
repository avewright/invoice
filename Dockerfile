# Linux + NVIDIA GPU ready image using Python 3.12
FROM python:3.12-slim

# Enable NVIDIA runtime (host must provide NVIDIA Container Toolkit)
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# System deps frequently needed by your locked packages (opencv, cairo/pdf, ocrmypdf, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    tesseract-ocr \
    poppler-utils \
    ghostscript \
    qpdf \
    fonts-dejavu \
  && rm -rf /var/lib/apt/lists/*

# Faster, deterministic pip
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# Copy only the lock first to leverage build cache
COPY requirements-lock.txt ./

# Install Python dependencies.
# - Use PyTorch cu121 extra index so torch/vision/audio +cu121 resolve
# - Then swap CPU Paddle for GPU Paddle built for CUDA 12.1
RUN python -m pip install --upgrade pip setuptools wheel \
  && pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements-lock.txt \
  && pip uninstall -y paddlepaddle \
  && pip install paddlepaddle-gpu==2.6.1.post121

# Project files
COPY playground.ipynb ./
COPY construction-invoice-template-1x.png ./

# JupyterLab port
EXPOSE 8888

# Start JupyterLab
CMD ["python", "-m", "jupyterlab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.token="]
