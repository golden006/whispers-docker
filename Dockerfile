# ---- Build stage ----
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv \
    ffmpeg build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency specs first (cache layer optimization)
COPY pyproject.toml uv.lock .python-version ./

# Create venv and install deps
ENV UV_COMPILE_BYTECODE=1 \
    UV_INDEX_URL=https://pypi.org/simple \
    UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

RUN uv venv && \
    uv sync --frozen && \
    uv pip install "uvicorn[standard]>=0.30.0" "fastapi>=0.110.0" "python-multipart>=0.0.9"

# Copy application code
COPY . .

# ---- Runtime stage ----
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv \
    ffmpeg tini curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy venv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application
COPY . .

ENV PATH="/app/.venv/bin:$PATH"

# Expose transcription service port
EXPOSE 2948

# Pre-download NLTK data
RUN python -c "import nltk; nltk.download('punkt_tab', download_dir='/tmp/nltk'); import shutil; shutil.move('/tmp/nltk', '/root/.nltk')" 2>/dev/null || true

ENTRYPOINT ["tini", "--"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "2948"]
