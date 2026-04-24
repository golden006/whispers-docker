# whispers-docker

WhisperX transcription service via FastAPI. Supports GPU (NVIDIA) and CPU.

## Prerequisites

### NVIDIA GPU machine (Spark)

Install the NVIDIA Container Toolkit (one time per machine):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Setup

```bash
git clone https://github.com/golden006/whispers-docker
cd whispers-docker
cp .env.example .env
# Edit .env and add your HuggingFace token (required for diarization)
# Get a free read token at https://huggingface.co/settings/tokens
# Accept the model license at https://huggingface.co/pyannote/speaker-diarization-community-1
```

## Start

**GPU (Spark)** — port 2948, uses large-v2 model:
```bash
sudo docker compose --profile gpu up --build -d
```

**CPU only** — port 2949, uses tiny model:
```bash
docker compose up --build -d
```

> First GPU build takes ~20 minutes (compiles ctranslate2 from source for aarch64).
> Subsequent builds are cached.

## Verify

```bash
curl http://localhost:2948/health
# GPU should show: "cuda": true, "cuda_device": "NVIDIA GB10"
```

## Transcribe

```bash
# Basic
curl -s -N -X POST http://localhost:2948/transcribe \
  -F "file=@audio/your_file.mp3" \
  -F "language=en" \
  -F "stream=true" \
  | grep "^data:" | tail -1 | sed 's/^data: //' | jq '.result' > output.json

# With speaker diarization (requires HF_TOKEN in .env)
curl -s -N -X POST http://localhost:2948/transcribe \
  -F "file=@audio/your_file.mp3" \
  -F "language=en" \
  -F "diarize=true" \
  -F "stream=true" \
  | grep "^data:" | tail -1 | sed 's/^data: //' | jq '.result' > output.json
```

## Output format

```json
{
  "language": "English",
  "language_code": "en",
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "transcription": [
    { "timestamp": "0:00", "speaker": "SPEAKER_00", "text": "Hello." },
    { "timestamp": "0:03", "speaker": "SPEAKER_01", "text": "Hi there." }
  ]
}
```

## Key options

| Field | Default | Description |
|---|---|---|
| `model` | `large-v2` (GPU) / `tiny` (CPU) | Whisper model size |
| `language` | auto-detect | ISO code e.g. `en`, `es`, `fr` |
| `diarize` | `false` | Identify speakers (requires `HF_TOKEN`) |
| `no_align` | `false` | Skip word-level alignment (faster) |
| `stream` | `false` | SSE progress stream |
