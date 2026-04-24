FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv \
    ffmpeg build-essential curl git tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Create venv early so all subsequent pip calls use it
RUN python3.10 -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

RUN pip install --upgrade pip setuptools

# Install CUDA PyTorch — cu129 index has aarch64 wheels (cu128 only has x86_64)
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu129 \
        --extra-index-url https://pypi.org/simple \
        "torch==2.8.0" \
        "torchaudio==2.8.0" \
        "torchvision==0.23.0"

# Install ctranslate2 with CUDA support for aarch64 from Jetson AI Lab index
# The standard PyPI aarch64 wheel is CPU-only; this one includes CUDA
RUN pip install --no-cache-dir \
        --extra-index-url https://pypi.jetson-ai-lab.dev/jp6/cu126 \
        "ctranslate2>=4.5.0"

# Install remaining dependencies from PyPI
RUN pip install --no-cache-dir \
        "faster-whisper>=1.2.0" \
        "nltk>=3.9.1" \
        "numpy>=2.1.0" \
        "omegaconf>=2.3.0" \
        "pandas>=2.2.3" \
        "pyannote-audio>=4.0.0" \
        "huggingface-hub<1.0.0" \
        "transformers>=4.48.0" \
        "uvicorn[standard]>=0.30.0" \
        "fastapi>=0.110.0" \
        "python-multipart>=0.0.9"

# Copy source and install whisperx — no-deps prevents pip re-resolving torch from PyPI
COPY . .
RUN pip install --no-cache-dir --no-deps -e .

EXPOSE 2948

RUN python -c "import nltk; nltk.download('punkt_tab', download_dir='/tmp/nltk'); import shutil; shutil.move('/tmp/nltk', '/root/.nltk')" 2>/dev/null || true

ENTRYPOINT ["tini", "--"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "2948"]
