"""
High-Fidelity 432Hz Adaptive Audio Tuning Engine
Backend: FastAPI + Librosa + FFmpeg (with SciPy fallback)

Pipeline:
  1. Audio streamed to disk in 1 MB chunks
  2. Librosa   -> detect master tuning frequency via estimate_tuning()
  3. FFmpeg    -> lossless varispeed resampling (asetrate + libsoxr)
     Fallback  -> scipy.signal.resample_poly if FFmpeg not installed
  4. Processed file returned as downloadable stream
  5. Temp files deleted in background after response ships
"""

import os
import uuid
import asyncio
import logging
import shutil
import subprocess
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

# ─── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# On Railway/Render use /tmp; locally use ./temp
TEMP_DIR   = os.environ.get("TEMP_DIR", os.path.join(BASE_DIR, "temp"))
os.makedirs(TEMP_DIR, exist_ok=True)

TARGET_HZ         = 432.0
STANDARD_HZ       = 440.0
ANALYSIS_DURATION = 60          # seconds analysed for pitch detection
CHUNK_SIZE        = 1024 * 1024 # 1 MB upload chunks
MAX_FILE_SIZE_MB  = 500
MAX_RATIO_DENOM   = 1000        # precision for scipy rational resampling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("432hz-tuner")

# ─── Detect FFmpeg once at startup ────────────────────────────────────────────
FFMPEG_BIN  = shutil.which("ffmpeg")
FFPROBE_BIN = shutil.which("ffprobe")
FFMPEG_AVAILABLE = FFMPEG_BIN is not None and FFPROBE_BIN is not None

if FFMPEG_AVAILABLE:
    logger.info(f"FFmpeg found : {FFMPEG_BIN}  (high-quality engine active)")
else:
    logger.warning("FFmpeg NOT found — using scipy fallback engine. "
                   "Output will be WAV regardless of input format.")

# ─── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="432Hz Adaptive Tuning Engine", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ─── DSP: Pitch Detection (shared by both engines) ────────────────────────────

def detect_master_frequency(input_path: str) -> float:
    """
    Analyse up to ANALYSIS_DURATION seconds of audio and return the
    detected master A frequency in Hz.

    librosa.estimate_tuning() returns semitone offset from A=440.
    Formula: f_current = 440 * 2^(offset / 12)
    """
    logger.info(f"Analysing: {input_path}")
    y, sr = librosa.load(
        input_path, sr=22050, mono=True,
        duration=ANALYSIS_DURATION, res_type="kaiser_fast",
    )
    y_trim, _ = librosa.effects.trim(y, top_db=20)
    if len(y_trim) < sr:
        y_trim = y
    y_trim = y_trim[: sr * ANALYSIS_DURATION]

    tuning_offset = librosa.estimate_tuning(y=y_trim, sr=sr)
    f_current = STANDARD_HZ * (2 ** (tuning_offset / 12))

    logger.info(f"Tuning offset : {tuning_offset:+.4f} semitones")
    logger.info(f"Master freq   : {f_current:.4f} Hz")
    return f_current


# ─── Engine A: FFmpeg (preferred) ─────────────────────────────────────────────

def _get_sample_rate(input_path: str) -> int:
    """Read native sample rate with ffprobe."""
    cmd = [
        FFPROBE_BIN, "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return int(result.stdout.strip())


def _output_ext_and_codec(input_ext: str):
    """Return (output_extension, ffmpeg_codec_args) for the given input."""
    if input_ext in (".flac",):
        return ".flac", ["-c:a", "flac", "-compression_level", "8"]
    elif input_ext in (".wav", ".aiff", ".aif"):
        return ".wav",  ["-c:a", "pcm_s24le"]
    elif input_ext in (".mp3",):
        return ".mp3",  ["-c:a", "libmp3lame", "-q:a", "0"]
    elif input_ext in (".m4a", ".aac"):
        return ".m4a",  ["-c:a", "aac", "-b:a", "320k"]
    else:
        return ".mp3",  ["-c:a", "libmp3lame", "-q:a", "0"]


def convert_ffmpeg(input_path: str, output_path: str, ratio: float) -> None:
    """
    Varispeed conversion using FFmpeg + libsoxr:
      asetrate = original_sr * ratio   (reinterpret sample rate — shifts pitch+tempo)
      aresample = original_sr          (restore sample rate via sinc interpolation)

    Result: pitch shifted by ratio, tempo changed by the same tiny amount (~1.8%),
    phase coherence and transients perfectly preserved.
    """
    original_sr = _get_sample_rate(input_path)
    target_sr   = int(round(original_sr * ratio))

    ext = os.path.splitext(input_path)[1].lower()
    out_ext, codec_args = _output_ext_and_codec(ext)

    # Ensure output path uses the right extension
    out_base    = os.path.splitext(output_path)[0]
    output_path = out_base + out_ext

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", input_path,
        "-af", (
            f"asetrate={target_sr},"
            f"aresample={original_sr}:resampler=soxr:precision=33"
        ),
        *codec_args,
        output_path,
    ]
    logger.info(f"FFmpeg: asetrate={target_sr} → aresample={original_sr}  cmd: {' '.join(cmd)}")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error(f"FFmpeg stderr: {proc.stderr}")
        raise RuntimeError(f"FFmpeg failed: {proc.stderr[-600:]}")

    logger.info(f"FFmpeg done: {output_path}")
    return output_path


# ─── Engine B: SciPy (fallback — no FFmpeg needed) ────────────────────────────

def _ratio_to_fraction(ratio: float, max_denom: int = MAX_RATIO_DENOM):
    frac = Fraction(ratio).limit_denominator(max_denom)
    return frac.numerator, frac.denominator


def convert_scipy(input_path: str, output_path: str, ratio: float) -> str:
    """
    Varispeed conversion using scipy.signal.resample_poly.
    Always writes 32-bit float WAV.
    """
    ext = os.path.splitext(input_path)[1].lower()

    if ext in (".wav", ".flac", ".aiff", ".aif"):
        data, sr = sf.read(input_path, always_2d=True, dtype="float32")
    else:
        y, sr = librosa.load(input_path, sr=None, mono=False, res_type="kaiser_best")
        data = y[:, np.newaxis] if y.ndim == 1 else y.T

    logger.info(f"SciPy read: {data.shape[0]} samples × {data.shape[1]} ch @ {sr} Hz")

    up, down = _ratio_to_fraction(ratio)
    logger.info(f"SciPy resample_poly {ratio:.8f} → {up}/{down}")

    channels = [
        resample_poly(data[:, ch], up, down).astype(np.float32)
        for ch in range(data.shape[1])
    ]
    output_data = np.stack(channels, axis=1)

    # scipy engine always outputs WAV
    out_base    = os.path.splitext(output_path)[0]
    output_path = out_base + ".wav"
    sf.write(output_path, output_data, sr, subtype="FLOAT")
    logger.info(f"SciPy done: {output_path}")
    return output_path


# ─── Full Pipeline ─────────────────────────────────────────────────────────────

def process_audio_to_432(input_path: str, output_path: str):
    """Detect pitch → calculate ratio → convert (FFmpeg if available, else SciPy)."""
    f_current = detect_master_frequency(input_path)
    ratio     = TARGET_HZ / f_current
    logger.info(f"Ratio: {ratio:.8f}  ({f_current:.4f} Hz → {TARGET_HZ} Hz)")

    if FFMPEG_AVAILABLE:
        final_output = convert_ffmpeg(input_path, output_path, ratio)
    else:
        final_output = convert_scipy(input_path, output_path, ratio)

    return f_current, ratio, final_output


# ─── Cleanup ───────────────────────────────────────────────────────────────────

def cleanup_files(*paths: str) -> None:
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
                logger.info(f"Deleted: {p}")
        except Exception as e:
            logger.warning(f"Could not delete {p}: {e}")


# ─── API ───────────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".aiff", ".aif", ".ogg", ".m4a"}


@app.post("/api/tune/")
async def tune_audio(background_tasks: BackgroundTasks,
                     file: UploadFile = File(...)):

    # Validate
    original_name = file.filename or "audio.wav"
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415,
            detail=f"Unsupported type '{ext}'. "
                   f"Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    # Stream upload to disk
    file_id     = str(uuid.uuid4())
    input_path  = os.path.join(TEMP_DIR, f"{file_id}_input{ext}")
    output_path = os.path.join(TEMP_DIR, f"{file_id}_output")  # ext added by engine
    base_name   = os.path.splitext(original_name)[0]

    total_bytes = 0
    async with aiofiles.open(input_path, "wb") as f_out:
        while chunk := await file.read(CHUNK_SIZE):
            total_bytes += len(chunk)
            if total_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
                cleanup_files(input_path)
                raise HTTPException(status_code=413,
                    detail=f"File exceeds {MAX_FILE_SIZE_MB} MB limit.")
            await f_out.write(chunk)

    logger.info(f"Received {total_bytes / 1e6:.2f} MB → {input_path}")

    # Process (run in thread so async loop isn't blocked)
    try:
        loop = asyncio.get_event_loop()
        original_hz, ratio, final_output = await loop.run_in_executor(
            None, process_audio_to_432, input_path, output_path)
    except Exception as e:
        cleanup_files(input_path, output_path)
        logger.error(f"Processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    # Derive output filename and MIME type from actual output file
    out_ext   = os.path.splitext(final_output)[1].lower()
    out_name  = f"432Hz_{base_name}{out_ext}"
    mime_map  = {".wav": "audio/wav", ".flac": "audio/flac",
                 ".mp3": "audio/mpeg", ".m4a": "audio/mp4"}
    mime_type = mime_map.get(out_ext, "audio/wav")

    background_tasks.add_task(cleanup_files, input_path, final_output)

    return FileResponse(
        path=final_output,
        filename=out_name,
        media_type=mime_type,
        headers={
            "X-Original-Hz": f"{original_hz:.4f}",
            "X-Ratio":       f"{ratio:.8f}",
            "X-Target-Hz":   f"{TARGET_HZ}",
            "X-Engine":      "ffmpeg" if FFMPEG_AVAILABLE else "scipy",
            "Access-Control-Expose-Headers":
                "X-Original-Hz, X-Ratio, X-Target-Hz, X-Engine",
        },
    )


@app.get("/api/health")
async def health():
    return JSONResponse({
        "status":    "ok",
        "ffmpeg":    FFMPEG_AVAILABLE,
        "engine":    "ffmpeg+libsoxr" if FFMPEG_AVAILABLE else "scipy.resample_poly",
        "target_hz": TARGET_HZ,
    })


# ─── Frontend ──────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")