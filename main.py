"""
High-Fidelity 432Hz Adaptive Audio Tuning Engine
Backend: FastAPI + Librosa + SciPy  (no FFmpeg required)

Pipeline:
  1. Audio streamed to disk in 1 MB chunks  (memory-safe for large files)
  2. Librosa  -> detect master tuning frequency via estimate_tuning()
  3. SciPy    -> varispeed resample (resample_poly) -- zero phase artifacts
  4. Soundfile -> write lossless WAV output
  5. Temp files deleted in background after response is sent
"""

import os
import uuid
import asyncio
import logging
from fractions import Fraction

import aiofiles
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import resample_poly

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Config
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR   = os.environ.get("TEMP_DIR", os.path.join(BASE_DIR, "temp"))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(TEMP_DIR, exist_ok=True)

TARGET_HZ         = 432.0
STANDARD_HZ       = 440.0
ANALYSIS_DURATION = 60
CHUNK_SIZE        = 1024 * 1024
MAX_FILE_SIZE_MB  = 500
MAX_RATIO_DENOM   = 1000

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("432hz-tuner")

app = FastAPI(title="432Hz Adaptive Tuning Engine", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


def detect_master_frequency(input_path: str) -> float:
    logger.info(f"Analysing pitch: {input_path}")
    y, sr = librosa.load(input_path, sr=22050, mono=True,
                         duration=ANALYSIS_DURATION, res_type="kaiser_fast")
    y_trim, _ = librosa.effects.trim(y, top_db=20)
    if len(y_trim) < sr:
        y_trim = y
    y_trim = y_trim[: sr * ANALYSIS_DURATION]
    tuning_offset = librosa.estimate_tuning(y=y_trim, sr=sr)
    f_current = STANDARD_HZ * (2 ** (tuning_offset / 12))
    logger.info(f"Tuning offset : {tuning_offset:+.4f} semitones")
    logger.info(f"Master freq   : {f_current:.4f} Hz")
    return f_current


def _ratio_to_fraction(ratio: float, max_denom: int = MAX_RATIO_DENOM):
    frac = Fraction(ratio).limit_denominator(max_denom)
    return frac.numerator, frac.denominator


def convert_to_432_scipy(input_path: str, output_path: str, ratio: float) -> None:
    ext = os.path.splitext(input_path)[1].lower()

    if ext in (".wav", ".flac", ".aiff", ".aif"):
        data, original_sr = sf.read(input_path, always_2d=True, dtype="float32")
    else:
        # MP3 / OGG / M4A handled by librosa via audioread
        y, original_sr = librosa.load(input_path, sr=None, mono=False,
                                       res_type="kaiser_best")
        data = y[:, np.newaxis] if y.ndim == 1 else y.T

    logger.info(f"Read  : {data.shape[0]} samples x {data.shape[1]} ch @ {original_sr} Hz")

    up, down = _ratio_to_fraction(ratio)
    logger.info(f"Resample {ratio:.8f}  -> {up}/{down}")

    resampled_channels = []
    for ch in range(data.shape[1]):
        resampled = resample_poly(data[:, ch], up, down).astype(np.float32)
        resampled_channels.append(resampled)

    output_data = np.stack(resampled_channels, axis=1)
    logger.info(f"Output: {output_data.shape[0]} samples x {output_data.shape[1]} ch")

    sf.write(output_path, output_data, original_sr, subtype="FLOAT")
    logger.info(f"Saved : {output_path}")


def process_audio_to_432(input_path: str, output_path: str):
    f_current = detect_master_frequency(input_path)
    ratio     = TARGET_HZ / f_current
    logger.info(f"Ratio : {ratio:.8f}  ({f_current:.4f} Hz -> {TARGET_HZ} Hz)")
    convert_to_432_scipy(input_path, output_path, ratio)
    return f_current, ratio


def cleanup_files(*paths: str) -> None:
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
                logger.info(f"Deleted: {p}")
        except Exception as e:
            logger.warning(f"Could not delete {p}: {e}")


ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".aiff", ".aif", ".ogg", ".m4a"}


@app.post("/api/tune/")
async def tune_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    original_name = file.filename or "audio.wav"
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415,
            detail=f"Unsupported type '{ext}'. Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    file_id     = str(uuid.uuid4())
    input_path  = os.path.join(TEMP_DIR, f"{file_id}_input{ext}")
    output_path = os.path.join(TEMP_DIR, f"{file_id}_output.wav")
    base_name   = os.path.splitext(original_name)[0]
    output_name = f"432Hz_{base_name}.wav"

    total_bytes = 0
    async with aiofiles.open(input_path, "wb") as f_out:
        while chunk := await file.read(CHUNK_SIZE):
            total_bytes += len(chunk)
            if total_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
                cleanup_files(input_path)
                raise HTTPException(status_code=413,
                    detail=f"File exceeds {MAX_FILE_SIZE_MB} MB limit.")
            await f_out.write(chunk)

    logger.info(f"Received {total_bytes / 1e6:.2f} MB  ->  {input_path}")

    try:
        loop = asyncio.get_event_loop()
        original_hz, ratio = await loop.run_in_executor(
            None, process_audio_to_432, input_path, output_path)
    except Exception as e:
        cleanup_files(input_path, output_path)
        logger.error(f"Processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    background_tasks.add_task(cleanup_files, input_path, output_path)

    return FileResponse(
        path=output_path,
        filename=output_name,
        media_type="audio/wav",
        headers={
            "X-Original-Hz": f"{original_hz:.4f}",
            "X-Ratio":       f"{ratio:.8f}",
            "X-Target-Hz":   f"{TARGET_HZ}",
            "Access-Control-Expose-Headers": "X-Original-Hz, X-Ratio, X-Target-Hz",
        },
    )


@app.get("/api/health")
async def health():
    return JSONResponse({
        "status":  "ok",
        "engine":  "scipy.signal.resample_poly (no FFmpeg required)",
        "target_hz": TARGET_HZ,
    })


@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
