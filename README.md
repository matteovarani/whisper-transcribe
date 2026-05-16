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
git clone https://github.com/matteovarani/whisper-transcribe.git
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

# Enable noise reduction (useful for low-quality or noisy audio)
python transcribe.py video.mp4 --language en --denoise

# Use a larger model for better accuracy on difficult audio
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
| `--denoise` | off | Enable FFT noise reduction before transcription |
| `--device` | `cpu` | `cpu` or `cuda` (NVIDIA GPU) |
| `-o` / `--output-dir` | video folder | Output directory |

## Audio preprocessing

The script applies an audio filter chain via ffmpeg before transcription to improve recognition quality:

1. **`highpass=f=80`** — removes low-frequency rumble (fans, AC units)
2. **`loudnorm`** — normalizes volume to EBU R128 broadcast standard (fixes quiet audio)
3. **`dynaudnorm`** — smooths dynamic volume changes throughout the video

With `--denoise`, an additional FFT-based filter (`afftdn`) is applied to reduce background noise like hiss or ambient sound.

## Models

| Model | Speed (CPU) | Accuracy |
|-------|-------------|----------|
| `tiny` | very fast | low |
| `base` | fast | low |
| `small` | fast | medium |
| `medium` *(default)* | moderate | good |
| `large-v2` | slow | very good |
| `large-v3` | slow | best |

On CPU, `medium` with `int8` quantization offers a good balance of speed and accuracy. Switch to `large-v3` only when `medium` produces clearly wrong results — it is significantly slower without a proportional quality gain in most cases.
