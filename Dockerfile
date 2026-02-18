FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (avoids pulling the huge CUDA wheels)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the package in editable mode so code changes are picked up via bind mount
COPY pyproject.toml .
COPY medvqa_probe/ medvqa_probe/
RUN pip install --no-cache-dir -e .

# Outputs and configs are provided at runtime via bind mount; nothing to COPY here.
ENV PYTHONUNBUFFERED=1
