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

# Install CUDA PyTorch wheels directly by URL — avoids any index resolution ambiguity
RUN pip install --upgrade pip setuptools && \
    pip install --no-cache-dir \
        "https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp310-cp310-linux_x86_64.whl" \
        "https://download.pytorch.org/whl/cu128/torchaudio-2.8.0%2Bcu128-cp310-cp310-linux_x86_64.whl" \
        "https://download.pytorch.org/whl/cu128/torchvision-0.23.0%2Bcu128-cp310-cp310-linux_x86_64.whl"

COPY pyproject.toml ./

# Install remaining dependencies — torch is already installed so pip won't touch it
RUN pip install --no-cache-dir \
        "ctranslate2>=4.5.0" \
        "faster-whisper>=1.2.0" \
        "nltk>=3.9.1" \
        "numpy>=2.1.0" \
        "omegaconf>=2.3.0" \
        "pandas>=2.2.3" \
        "pyannote-audio>=4.0.0" \
        "huggingface-hub<1.0.0" \
        "transformers>=4.48.0" \
        "triton>=3.3.0" \
        "uvicorn[standard]>=0.30.0" \
        "fastapi>=0.110.0" \
        "python-multipart>=0.0.9"

# Copy source and install whisperx — no-deps prevents pip re-resolving torch
COPY . .
RUN pip install --no-cache-dir --no-deps -e .

EXPOSE 2948

RUN python -c "import nltk; nltk.download('punkt_tab', download_dir='/tmp/nltk'); import shutil; shutil.move('/tmp/nltk', '/root/.nltk')" 2>/dev/null || true

ENTRYPOINT ["tini", "--"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "2948"]
