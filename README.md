# 432Hz Adaptive Tuning Engine

A high-fidelity, locally-hosted web application that uploads an audio file,
auto-detects its master tuning frequency, and converts it to **A = 432 Hz**
using lossless varispeed resampling — zero phase-vocoder artifacts.

---

## Prerequisites

| Requirement | Version | Install |
|-------------|---------|---------|
| Python      | 3.10+   | [python.org](https://python.org) |
| FFmpeg      | 5.0+    | See below |

### Install FFmpeg

**macOS**
```bash
brew install ffmpeg
```

**Ubuntu / Debian**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Windows**
Download from [ffmpeg.org/download.html](https://ffmpeg.org/ffmpeg-builds.html)
and add the `bin/` folder to your system PATH.

---

## Setup & Run

```bash
# 1. Clone / unzip the project
cd 432hz-tuner

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn main:app --reload

# 5. Open your browser
#    http://127.0.0.1:8000
```

---

## How It Works

### 1. Pitch Detection (Librosa)
`librosa.estimate_tuning()` analyses the first 60 seconds of the uploaded
audio using parabolic interpolation on a thresholded STFT (Short-Time
Fourier Transform).  It returns the master tuning offset in semitones
relative to A = 440 Hz, which is converted to an exact Hz value:

```
f_current = 440 × 2^(offset_semitones / 12)
```

### 2. Ratio Calculation
```
ratio = 432 / f_current
```
E.g. if your track is at 441.5 Hz → ratio ≈ 0.97845

### 3. Varispeed Resampling (FFmpeg + libsoxr)
Instead of a destructive phase-vocoder pitch-shift, the engine uses
**sample-rate conversion** — the same principle as a tape deck:

```
ffmpeg -i input.flac \
  -af "asetrate=<original_sr × ratio>,aresample=<original_sr>:resampler=soxr:precision=33" \
  -c:a flac output.flac
```

- `asetrate` reinterprets the sample rate, changing both pitch and tempo
- `aresample` restores the original sample rate using libsoxr sinc
  interpolation (33-bit precision)
- The tempo change is ≈ −1.82% — inaudible to human perception

### Why Varispeed, Not Phase-Vocoder?
| Method | Phase Coherence | Transient Integrity | Artifacts |
|--------|----------------|---------------------|-----------|
| Phase Vocoder (Rubberband, etc.) | ✗ Degraded | ✗ Smeared | "Phasiness", metallic shimmer |
| Varispeed (this app) | ✓ Perfect | ✓ Razor-sharp | None |

---

## API Reference

### `POST /api/tune/`
Upload an audio file and receive the 432 Hz-converted version.

**Request:** `multipart/form-data`
- `file` — audio file (MP3, WAV, FLAC, AIFF, OGG, M4A, AAC)

**Response:** Binary audio stream (same format as input)

**Response Headers:**
- `X-Original-Hz` — detected master frequency
- `X-Ratio` — applied resampling ratio
- `X-Target-Hz` — target frequency (432.0)

### `GET /api/health`
Returns JSON: `{ "status": "ok", "ffmpeg": true, "target_hz": 432.0 }`

---

## Project Structure

```
432hz-tuner/
├── main.py            ← FastAPI backend (pitch detect + FFmpeg convert)
├── requirements.txt   ← Python dependencies
├── README.md
├── temp/              ← Auto-created; stores files during processing
└── static/
    └── index.html     ← Single-page frontend
```

---

## Configuration (main.py)

| Constant | Default | Description |
|----------|---------|-------------|
| `TARGET_HZ` | 432.0 | Target tuning frequency |
| `ANALYSIS_DURATION` | 60 | Seconds of audio used for pitch detection |
| `MAX_FILE_SIZE_MB` | 500 | Upload size limit |
| `CHUNK_SIZE` | 1 MB | Streaming chunk size for large file uploads |

---

## Notes on Audio Quality

- **FLAC / WAV input** → FLAC / WAV output (lossless to lossless)
- **MP3 / OGG / M4A input** → MP3 output at VBR q=0 (~245 kbps)
- Temp files are automatically deleted after each request
- Files are streamed to disk in 1 MB chunks — no memory overload on large files
