# Multi-stage Dockerfile for ML Retinal Disease Classification

# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY config.yaml config_ci.yaml run.py ./
COPY scripts/ ./scripts/
COPY src/ ./src/

# Create necessary directories
RUN mkdir -p src/data/images/train \
    src/data/images/val \
    src/data/images/offsite_test \
    src/data/pretrained_backbone \
    src/models \
    src/results

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "run.py", "--help"]
