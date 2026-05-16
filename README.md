# whisper-transcribe

Transcribe MP4 video audio and generate SRT subtitles + text transcript using [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

Produces:
- `<video>.en.srt` — subtitles in original language
- `<video>.en.txt` — transcript with timestamps
- `<video>.it.srt` — subtitles translated to Italian (via Google Translate)
- `<video>.it.txt` — translated transcript

## Requirements

- Python 3.11+
- [ffmpeg](https://ffmpeg.org/) installed on your system (`brew install ffmpeg` on macOS)

## Installation

```bash
git clone https://github.com/mvarani/whisper-transcribe.git
cd whisper-transcribe

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

```bash
source venv/bin/activate

# Basic — auto-detects language, translates to Italian
python transcribe.py video.mp4

# Force source language (faster, more accurate)
python transcribe.py video.mp4 --language en

# Different model (tiny/base/small/medium/large-v2/large-v3)
python transcribe.py video.mp4 --language en --model large-v3

# Translate to a different language
python transcribe.py video.mp4 --language en --translate-to fr

# Disable translation (only original language files)
python transcribe.py video.mp4 --language en --translate-to none

# Custom output directory
python transcribe.py video.mp4 --language en -o /path/to/output/
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `-m` / `--model` | `medium` | Whisper model size |
| `-l` / `--language` | auto | Source language code (`en`, `it`, `fr`, …) |
| `--translate-to` | `it` | Target language for translation (`none` to skip) |
| `--device` | `cpu` | `cpu` or `cuda` (NVIDIA GPU) |
| `-o` / `--output-dir` | video folder | Output directory |

## Models

| Model | Speed | Accuracy | VRAM |
|-------|-------|----------|------|
| `tiny` | fastest | lowest | ~1 GB |
| `base` | fast | low | ~1 GB |
| `small` | fast | medium | ~2 GB |
| `medium` | moderate | good | ~5 GB |
| `large-v3` | slow | best | ~10 GB |

On CPU, `medium` with `int8` quantization is a good balance of speed and accuracy.
