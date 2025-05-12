FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    build-essential \
    ninja-build \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch with CUDA
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install numpy first (dependency issue workaround)
RUN pip3 install --no-cache-dir numpy

# Install vLLM using a specific version known to work
RUN pip3 install --no-cache-dir "vllm==0.2.1"

# Install additional dependencies
RUN pip3 install --no-cache-dir \
    accelerate \
    transformers \
    huggingface_hub \
    flask \
    requests \
    pydantic

# Copy script
COPY src/vllm/server.py /app/server.py

# Default model path
ENV MODEL_PATH="/models/tinyllama"
ENV MODEL_ID="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Expose port
EXPOSE 8000

# Command to run vLLM server
CMD ["python3", "/app/server.py"] 