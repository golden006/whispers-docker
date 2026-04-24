FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

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

COPY pyproject.toml ./

# Install dependencies (CUDA PyTorch) — source not needed yet for deps
RUN pip install --upgrade pip setuptools && \
    pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        "ctranslate2>=4.5.0" \
        "faster-whisper>=1.2.0" \
        "nltk>=3.9.1" \
        "numpy>=2.1.0" \
        "omegaconf>=2.3.0" \
        "pandas>=2.2.3" \
        "pyannote-audio>=4.0.0" \
        "huggingface-hub<1.0.0" \
        "torch~=2.8.0" \
        "torchaudio~=2.8.0" \
        "torchvision~=0.23.0" \
        "transformers>=4.48.0" \
        "triton>=3.3.0" \
        "uvicorn[standard]>=0.30.0" \
        "fastapi>=0.110.0" \
        "python-multipart>=0.0.9"

# Copy source and install the whisperx package in editable mode
COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 2948

RUN python -c "import nltk; nltk.download('punkt_tab', download_dir='/tmp/nltk'); import shutil; shutil.move('/tmp/nltk', '/root/.nltk')" 2>/dev/null || true

ENTRYPOINT ["tini", "--"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "2948"]
