"""FastAPI web service wrapping WhisperX transcription."""

import asyncio
import json
import logging
import tempfile
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from whisperx.transcribe import transcribe_task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WhisperX Transcription Service", version="3.8.5")

# Model cache to avoid reloading between requests
_model_cache: dict = {}


def _get_model(name: str, device: str, device_index: int, model_dir: str | None,
               compute_type: str, hf_token: str | None) -> object:
    """Load or return cached WhisperX model."""
    key = (name, device, device_index, model_dir, compute_type, hf_token)
    if key not in _model_cache:
        logger.info(f"Loading model {name} on {device}...")
        from whisperx.asr import load_model

        _model_cache[key] = load_model(
            name,
            device=device,
            device_index=device_index,
            download_root=model_dir,
            compute_type=compute_type,
            asr_options={
                "beam_size": 5,
                "patience": 1.0,
                "length_penalty": 1.0,
                "temperatures": [0],
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": False,
                "initial_prompt": None,
                "hotwords": None,
                "suppress_tokens": [-1],
                "suppress_numerals": False,
            },
            vad_method="pyannote",
            vad_options={"chunk_size": 30, "vad_onset": 0.5, "vad_offset": 0.363},
            task="transcribe",
            local_files_only=False,
            threads=4,
            use_auth_token=hf_token,
        )
        logger.info("Model loaded.")
    return _model_cache[key]


def _build_args(audio_path: str, model: str, device: str, device_index: int,
                model_dir: str | None, compute_type: str, batch_size: int,
                language: str | None, output_dir: str, output_format: str,
                align: bool, interpolate_method: str, no_align: bool,
                return_char_alignments: bool, vad_method: str,
                vad_onset: float, vad_offset: float, chunk_size: int,
                diarize: bool, min_speakers: int | None, max_speakers: int | None,
                diarize_model: str, hf_token: str | None, temperature: float,
                beam_size: int, patience: float, length_penalty: float,
                suppress_tokens: str, suppress_numerals: bool,
                initial_prompt: str | None, hotwords: str | None,
                logprob_threshold: float, no_speech_threshold: float,
                threads: int) -> dict:
    """Build the args dict that transcribe_task expects."""
    return {
        "audio": [audio_path],
        "model": model,
        "model_cache_only": False,
        "model_dir": model_dir,
        "device": device,
        "device_index": device_index,
        "batch_size": batch_size,
        "compute_type": compute_type,
        "verbose": False,
        "output_dir": output_dir,
        "output_format": output_format,
        "task": "transcribe",
        "language": language,
        "align_model": None,
        "interpolate_method": interpolate_method,
        "no_align": no_align,
        "return_char_alignments": return_char_alignments,
        "hf_token": hf_token,
        "vad_method": vad_method,
        "vad_onset": vad_onset,
        "vad_offset": vad_offset,
        "chunk_size": chunk_size,
        "diarize": diarize,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "diarize_model": diarize_model,
        "speaker_embeddings": False,
        "temperature": temperature,
        "beam_size": beam_size,
        "patience": patience,
        "length_penalty": length_penalty,
        "suppress_tokens": suppress_tokens,
        "suppress_numerals": suppress_numerals,
        "initial_prompt": initial_prompt,
        "hotwords": hotwords,
        "condition_on_previous_text": False,
        "fp16": True,
        "temperature_increment_on_fallback": 0.2,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": logprob_threshold,
        "no_speech_threshold": no_speech_threshold,
        "max_line_width": None,
        "max_line_count": None,
        "highlight_words": False,
        "segment_resolution": "sentence",
        "threads": threads,
        "print_progress": False,
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("small"),
    device: str = Form(None),
    device_index: int = Form(0),
    model_dir: str | None = Form(None),
    compute_type: str = Form("default"),
    batch_size: int = Form(8),
    language: str | None = Form(None),
    output_format: str = Form("json"),
    align: bool = Form(True),
    interpolate_method: str = Form("nearest"),
    no_align: bool = Form(False),
    return_char_alignments: bool = Form(False),
    vad_method: str = Form("pyannote"),
    vad_onset: float = Form(0.5),
    vad_offset: float = Form(0.363),
    chunk_size: int = Form(30),
    diarize: bool = Form(False),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
    diarize_model: str = Form("pyannote/speaker-diarization-community-1"),
    hf_token: str | None = Form(None),
    temperature: float = Form(0.0),
    beam_size: int = Form(5),
    patience: float = Form(1.0),
    length_penalty: float = Form(1.0),
    suppress_tokens: str = Form("-1"),
    suppress_numerals: bool = Form(False),
    initial_prompt: str | None = Form(None),
    hotwords: str | None = Form(None),
    logprob_threshold: float = Form(-1.0),
    no_speech_threshold: float = Form(0.6),
    threads: int = Form(0),
):
    """Transcribe an uploaded audio file."""
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate output_format
    valid_formats = {"txt", "srt", "vtt", "tsv", "json", "aud"}
    if output_format not in valid_formats:
        raise HTTPException(status_code=400, detail=f"Invalid output_format: {output_format}")

    # Validate interpolate_method
    if interpolate_method not in ("nearest", "linear", "ignore"):
        raise HTTPException(status_code=400, detail=f"Invalid interpolate_method: {interpolate_method}")

    # Validate vad_method
    if vad_method not in ("pyannote", "silero"):
        raise HTTPException(status_code=400, detail=f"Invalid vad_method: {vad_method}")

    tmpdir = Path(tempfile.mkdtemp())
    audio_path = tmpdir / file.filename
    output_dir = tmpdir / "output"
    output_dir.mkdir()

    try:
        content = await file.read()
        audio_path.write_bytes(content)

        args = _build_args(
            audio_path=str(audio_path),
            model=model, device=device, device_index=device_index,
            model_dir=model_dir, compute_type=compute_type,
            batch_size=batch_size, language=language,
            output_dir=str(output_dir), output_format=output_format,
            align=align, interpolate_method=interpolate_method,
            no_align=no_align, return_char_alignments=return_char_alignments,
            vad_method=vad_method, vad_onset=vad_onset, vad_offset=vad_offset,
            chunk_size=chunk_size, diarize=diarize,
            min_speakers=min_speakers, max_speakers=max_speakers,
            diarize_model=diarize_model, hf_token=hf_token,
            temperature=temperature, beam_size=beam_size,
            patience=patience, length_penalty=length_penalty,
            suppress_tokens=suppress_tokens, suppress_numerals=suppress_numerals,
            initial_prompt=initial_prompt, hotwords=hotwords,
            logprob_threshold=logprob_threshold,
            no_speech_threshold=no_speech_threshold, threads=threads,
        )

        model_obj = _get_model(
            name=model, device=device, device_index=device_index,
            model_dir=model_dir, compute_type=compute_type, hf_token=hf_token,
        )

        # Run transcription in a threadpool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: transcribe_task(args, None),
        )

        # Read the output file
        basename = Path(file.filename).stem
        output_file = output_dir / f"{basename}.{output_format}"

        if not output_file.exists():
            available = list(output_dir.rglob("*"))
            raise HTTPException(
                status_code=500,
                detail=f"Output file not created. Available files: {available}",
            )

        result_text = output_file.read_text()

        if output_format == "json":
            return JSONResponse(content=json.loads(result_text))

        return JSONResponse(content={"text": result_text, "format": output_format})

    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_count": torch.cuda.device_count(),
        "model_cache_size": len(_model_cache),
    }


@app.post("/shutdown")
async def shutdown():
    """Shutdown the server (only if SHUTDOWN_ENDPOINT=true)."""
    import os
    if os.environ.get("SHUTDOWN_ENDPOINT") != "true":
        raise HTTPException(status_code=404, detail="Shutdown endpoint disabled")
    import sys
    asyncio.get_event_loop().stop()
    return {"message": "Shutting down"}
