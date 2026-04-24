"""FastAPI web service wrapping WhisperX transcription."""

import asyncio
import gc
import json
import logging
import os
import queue
import tempfile
import threading
import warnings
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WhisperX Transcription Service", version="3.8.5")

# ISO 639-1 code → human-readable language name
_LANGUAGE_NAMES: dict[str, str] = {
    "af": "Afrikaans", "ar": "Arabic", "hy": "Armenian", "az": "Azerbaijani",
    "be": "Belarusian", "bs": "Bosnian", "bg": "Bulgarian", "ca": "Catalan",
    "zh": "Chinese", "hr": "Croatian", "cs": "Czech", "da": "Danish",
    "nl": "Dutch", "en": "English", "et": "Estonian", "fi": "Finnish",
    "fr": "French", "gl": "Galician", "de": "German", "el": "Greek",
    "he": "Hebrew", "hi": "Hindi", "hu": "Hungarian", "is": "Icelandic",
    "id": "Indonesian", "it": "Italian", "ja": "Japanese", "kn": "Kannada",
    "kk": "Kazakh", "ko": "Korean", "lv": "Latvian", "lt": "Lithuanian",
    "mk": "Macedonian", "ms": "Malay", "mr": "Marathi", "mi": "Maori",
    "ne": "Nepali", "no": "Norwegian", "fa": "Persian", "pl": "Polish",
    "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sr": "Serbian",
    "sk": "Slovak", "sl": "Slovenian", "es": "Spanish", "sw": "Swahili",
    "sv": "Swedish", "tl": "Tagalog", "ta": "Tamil", "th": "Thai",
    "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese",
    "cy": "Welsh", "yi": "Yiddish",
}


def _seconds_to_timestamp(seconds: float) -> str:
    """Convert fractional seconds to M:SS or H:MM:SS string."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _format_result(raw: dict) -> dict:
    """Shape the raw whisperx result into the clean response format."""
    lang_code = raw.get("language", "en")
    lang_name = _LANGUAGE_NAMES.get(lang_code, lang_code)

    lines = []
    for seg in raw.get("segments", []):
        ts = _seconds_to_timestamp(seg.get("start", 0))
        entry: dict = {"timestamp": ts, "text": seg["text"].strip()}
        if "speaker" in seg:
            entry["speaker"] = seg["speaker"]
        lines.append(entry)

    result: dict = {
        "language": lang_name,
        "language_code": lang_code,
        "transcription": lines,
    }

    # Include unique speaker list when diarization was run
    speakers = sorted({seg["speaker"] for seg in raw.get("segments", []) if "speaker" in seg})
    if speakers:
        result["speakers"] = speakers

    return result


def _is_cpu() -> bool:
    return not torch.cuda.is_available()


def _default_model() -> str:
    return "tiny" if _is_cpu() else "large-v2"


def _default_batch_size() -> int:
    return 2 if _is_cpu() else 16


def _default_compute_type() -> str:
    return "int8" if _is_cpu() else "float16"


def _run_transcription(
    audio_path: str,
    model_name: str,
    device: str,
    device_index: int,
    model_dir: str | None,
    compute_type: str,
    batch_size: int,
    language: str | None,
    output_dir: str,
    output_format: str,
    no_align: bool,
    interpolate_method: str,
    return_char_alignments: bool,
    vad_method: str,
    vad_onset: float,
    vad_offset: float,
    chunk_size: int,
    diarize: bool,
    min_speakers: int | None,
    max_speakers: int | None,
    diarize_model: str,
    hf_token: str | None,
    temperature: float,
    beam_size: int,
    patience: float,
    length_penalty: float,
    suppress_tokens: str,
    suppress_numerals: bool,
    initial_prompt: str | None,
    hotwords: str | None,
    logprob_threshold: float,
    no_speech_threshold: float,
    threads: int,
    progress_queue: queue.Queue | None = None,
):
    """Run full transcription pipeline synchronously (called in threadpool)."""
    def emit(msg: str):
        logger.info(msg)
        if progress_queue is not None:
            progress_queue.put(msg)

    from whisperx.alignment import align as wx_align, load_align_model
    from whisperx.audio import load_audio
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers
    from whisperx.utils import get_writer

    os.makedirs(output_dir, exist_ok=True)

    if language is not None:
        language = language.lower()

    align_language = language if language is not None else "en"

    temperature_val = temperature
    temperatures = tuple(np.arange(temperature_val, 1.0 + 1e-6, 0.2))

    faster_whisper_threads = 4
    if threads > 0:
        torch.set_num_threads(threads)
        faster_whisper_threads = threads

    asr_options = {
        "beam_size": beam_size,
        "patience": patience,
        "length_penalty": length_penalty,
        "temperatures": temperatures,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": logprob_threshold,
        "no_speech_threshold": no_speech_threshold,
        "condition_on_previous_text": False,
        "initial_prompt": initial_prompt,
        "hotwords": hotwords,
        "suppress_tokens": [int(x) for x in suppress_tokens.split(",")],
        "suppress_numerals": suppress_numerals,
    }

    writer = get_writer(output_format, output_dir)
    writer_args = {"highlight_words": False, "max_line_count": None, "max_line_width": None}

    # Part 1: VAD & ASR
    from whisperx.asr import load_model

    emit(f"Loading model '{model_name}' on {device}...")
    model = load_model(
        model_name,
        device=device,
        device_index=device_index,
        download_root=model_dir,
        compute_type=compute_type,
        language=language,
        asr_options=asr_options,
        vad_method=vad_method,
        vad_options={"chunk_size": chunk_size, "vad_onset": vad_onset, "vad_offset": vad_offset},
        task="transcribe",
        local_files_only=False,
        threads=faster_whisper_threads,
        use_auth_token=hf_token,
    )

    emit("Loading audio...")
    audio = load_audio(audio_path)

    emit("Performing voice activity detection and transcription...")
    result = model.transcribe(
        audio,
        batch_size=batch_size,
        chunk_size=chunk_size,
        print_progress=True,
        verbose=False,
    )

    segment_count = len(result.get("segments", []))
    emit(f"Transcription complete — {segment_count} segments found.")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Part 2: Align
    if not no_align:
        emit("Loading alignment model...")
        align_model_obj, align_metadata = load_align_model(
            align_language, device, model_name=None, model_dir=model_dir, model_cache_only=False
        )
        if align_model_obj is not None and len(result["segments"]) > 0:
            emit("Performing word-level alignment...")
            result = wx_align(
                result["segments"],
                align_model_obj,
                align_metadata,
                audio,
                device,
                interpolate_method=interpolate_method,
                return_char_alignments=return_char_alignments,
                print_progress=False,
            )
            emit("Alignment complete.")
        del align_model_obj
        gc.collect()
        torch.cuda.empty_cache()

    # Part 3: Diarize
    if diarize:
        if hf_token is None:
            warnings.warn("No hf_token provided; diarization model may fail to load.")
        emit("Performing speaker diarization...")
        diarize_pipeline = DiarizationPipeline(
            model_name=diarize_model, token=hf_token, device=device, cache_dir=model_dir
        )
        diarize_segments = diarize_pipeline(
            audio_path, min_speakers=min_speakers, max_speakers=max_speakers, return_embeddings=False
        )
        result = assign_word_speakers(diarize_segments, result, None)
        emit("Diarization complete.")

    result["language"] = align_language
    emit("Writing output...")
    writer(result, audio_path, writer_args)
    emit("__DONE__")


def _transcription_kwargs(
    audio_path, model, device, device_index, model_dir, compute_type,
    batch_size, language, output_dir, output_format, no_align,
    interpolate_method, return_char_alignments, vad_method, vad_onset,
    vad_offset, chunk_size, diarize, min_speakers, max_speakers,
    diarize_model, hf_token, temperature, beam_size, patience,
    length_penalty, suppress_tokens, suppress_numerals, initial_prompt,
    hotwords, logprob_threshold, no_speech_threshold, threads,
    progress_queue=None,
):
    return dict(
        audio_path=audio_path,
        model_name=model,
        device=device,
        device_index=device_index,
        model_dir=model_dir,
        compute_type=compute_type,
        batch_size=batch_size,
        language=language,
        output_dir=output_dir,
        output_format=output_format,
        no_align=no_align,
        interpolate_method=interpolate_method,
        return_char_alignments=return_char_alignments,
        vad_method=vad_method,
        vad_onset=vad_onset,
        vad_offset=vad_offset,
        chunk_size=chunk_size,
        diarize=diarize,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        diarize_model=diarize_model,
        hf_token=hf_token,
        temperature=temperature,
        beam_size=beam_size,
        patience=patience,
        length_penalty=length_penalty,
        suppress_tokens=suppress_tokens,
        suppress_numerals=suppress_numerals,
        initial_prompt=initial_prompt,
        hotwords=hotwords,
        logprob_threshold=logprob_threshold,
        no_speech_threshold=no_speech_threshold,
        threads=threads,
        progress_queue=progress_queue,
    )


async def _sse_generator(
    audio_path: str,
    output_dir: str,
    output_format: str,
    filename: str,
    kwargs: dict,
) -> AsyncGenerator[str, None]:
    """Run transcription in a thread and stream SSE progress events."""
    q: queue.Queue = queue.Queue()
    error_holder: list = []
    kwargs["progress_queue"] = q

    def worker():
        try:
            _run_transcription(**kwargs)
        except Exception as exc:
            logger.exception("Transcription failed")
            q.put(f"__ERROR__:{exc}")

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    loop = asyncio.get_event_loop()

    while True:
        try:
            msg = await loop.run_in_executor(None, lambda: q.get(timeout=1))
        except queue.Empty:
            # Send keepalive comment so the connection doesn't time out
            yield ": keepalive\n\n"
            if not t.is_alive():
                break
            continue

        if msg == "__DONE__":
            # Read the output file and send as final event
            basename = Path(filename).stem
            output_file = Path(output_dir) / f"{basename}.{output_format}"
            if output_file.exists():
                result_text = output_file.read_text()
                if output_format == "json":
                    raw = json.loads(result_text)
                    payload = json.dumps({"status": "done", "result": _format_result(raw)})
                else:
                    payload = json.dumps({"status": "done", "result": result_text, "format": output_format})
                yield f"data: {payload}\n\n"
            else:
                available = list(Path(output_dir).rglob("*"))
                payload = json.dumps({"status": "error", "detail": f"Output file not created. Available: {available}"})
                yield f"data: {payload}\n\n"
            break
        elif msg.startswith("__ERROR__:"):
            payload = json.dumps({"status": "error", "detail": msg[len("__ERROR__:"):]})
            yield f"data: {payload}\n\n"
            break
        else:
            yield f"data: {json.dumps({'status': 'progress', 'message': msg})}\n\n"

    t.join(timeout=5)

    import shutil
    shutil.rmtree(Path(output_dir).parent, ignore_errors=True)


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(None),
    device: str = Form(None),
    device_index: int = Form(0),
    model_dir: str | None = Form(None),
    compute_type: str = Form(None),
    batch_size: int = Form(None),
    language: str | None = Form(None),
    output_format: str = Form("json"),
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
    stream: bool = Form(False),
):
    """Transcribe an uploaded audio file.

    Set stream=true to receive Server-Sent Events with progress updates.
    Each SSE event is a JSON object:
      {"status": "progress", "message": "..."}  — progress update
      {"status": "done",     "result": ...}     — final result
      {"status": "error",    "detail": "..."}   — error
    """
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Fall back to environment variable if no token provided in request
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN") or None

    # Apply CPU-friendly defaults
    if model is None:
        model = _default_model()
    if batch_size is None:
        batch_size = _default_batch_size()
    if compute_type is None:
        compute_type = _default_compute_type()

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    valid_formats = {"txt", "srt", "vtt", "tsv", "json", "aud"}
    if output_format not in valid_formats:
        raise HTTPException(status_code=400, detail=f"Invalid output_format: {output_format}")

    if interpolate_method not in ("nearest", "linear", "ignore"):
        raise HTTPException(status_code=400, detail=f"Invalid interpolate_method: {interpolate_method}")

    if vad_method not in ("pyannote", "silero"):
        raise HTTPException(status_code=400, detail=f"Invalid vad_method: {vad_method}")

    tmpdir = Path(tempfile.mkdtemp())
    audio_path = tmpdir / file.filename
    output_dir = tmpdir / "output"
    output_dir.mkdir()

    content = await file.read()
    audio_path.write_bytes(content)

    kwargs = _transcription_kwargs(
        audio_path=str(audio_path),
        model=model, device=device, device_index=device_index,
        model_dir=model_dir, compute_type=compute_type,
        batch_size=batch_size, language=language,
        output_dir=str(output_dir), output_format=output_format,
        no_align=no_align, interpolate_method=interpolate_method,
        return_char_alignments=return_char_alignments,
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

    if stream:
        return StreamingResponse(
            _sse_generator(
                audio_path=str(audio_path),
                output_dir=str(output_dir),
                output_format=output_format,
                filename=file.filename,
                kwargs=kwargs,
            ),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    # Non-streaming: run synchronously and return JSON
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: _run_transcription(**kwargs))

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
            return JSONResponse(content=_format_result(json.loads(result_text)))

        return JSONResponse(content={"text": result_text, "format": output_format})

    except HTTPException:
        raise
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
        "default_model": _default_model(),
        "default_compute_type": _default_compute_type(),
    }


@app.post("/shutdown")
async def shutdown():
    """Shutdown the server (only if SHUTDOWN_ENDPOINT=true)."""
    if os.environ.get("SHUTDOWN_ENDPOINT") != "true":
        raise HTTPException(status_code=404, detail="Shutdown endpoint disabled")
    asyncio.get_event_loop().stop()
    return {"message": "Shutting down"}
