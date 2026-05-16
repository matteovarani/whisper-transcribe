#!/usr/bin/env python3
"""Transcribe MP4 audio and generate SRT subtitles + text transcript using faster-whisper."""

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def extract_audio(video_path: Path, audio_path: Path, denoise: bool = False) -> None:
    print(f"Estrazione audio da {video_path.name}...")

    # loudnorm: normalizza il volume secondo lo standard EBU R128
    # highpass: rimuove i bassi sotto 80Hz (rumori di fondo, ventole)
    # dynaudnorm: normalizza dinamicamente picchi e cali di volume
    filters = "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11,dynaudnorm"
    if denoise:
        # afftdn: riduzione rumore FFT — utile per audio con fruscio/ambiente
        filters = f"afftdn=nf=-25,{filters}"
        print("  Riduzione rumore abilitata")

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-af", filters,
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Errore ffmpeg:", result.stderr, file=sys.stderr)
        sys.exit(1)


def transcribe(audio_path: Path, model_size: str, language: str | None, device: str) -> tuple[list, object]:
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Caricamento modello '{model_size}' su {device} ({compute_type})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print("Trascrizione in corso...")
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=False,
        word_timestamps=False,
    )

    detected = info.language
    print(f"Lingua rilevata: {detected} (confidenza: {info.language_probability:.0%})")
    return list(segments), info


def translate_segments(segments: list, source_lang: str, target_lang: str) -> list[str]:
    """Translate segment texts using Google Translate. Returns list of translated strings."""
    print(f"Traduzione {source_lang} → {target_lang} in corso...")
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    translated = []
    for i, seg in enumerate(segments):
        text = seg.text.strip()
        if not text:
            translated.append("")
            continue
        try:
            result = translator.translate(text)
            translated.append(result or text)
        except Exception as e:
            print(f"  Avviso: traduzione segmento {i+1} fallita ({e}), uso testo originale")
            translated.append(text)
        # avoid hammering the API
        if i > 0 and i % 20 == 0:
            time.sleep(0.5)
    return translated


def write_srt(segments: list, texts: list[str], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (seg, text) in enumerate(zip(segments, texts), start=1):
            start = format_timestamp(seg.start)
            end = format_timestamp(seg.end)
            f.write(f"{i}\n{start} --> {end}\n{text.strip()}\n\n")
    print(f"  Salvato: {output_path}")


def write_txt(segments: list, texts: list[str], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for seg, text in zip(segments, texts):
            start = format_timestamp(seg.start)
            f.write(f"[{start}] {text.strip()}\n")
    print(f"  Salvato: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe MP4 and generate SRT subtitles + text transcript"
    )
    parser.add_argument("video", type=Path, help="File MP4 da trascrivere")
    parser.add_argument(
        "-m", "--model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Modello Whisper (default: medium)",
    )
    parser.add_argument(
        "-l", "--language",
        default=None,
        help="Lingua originale del video, es. 'en', 'it' (default: auto)",
    )
    parser.add_argument(
        "--translate-to",
        default="it",
        help="Lingua per la traduzione dei sottotitoli (default: it). Usa 'none' per disabilitare.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device di inferenza (default: cpu)",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Abilita riduzione rumore FFT (utile per audio con fruscio o rumore ambiente)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Cartella di output (default: stessa del video)",
    )
    args = parser.parse_args()

    video_path: Path = args.video.resolve()
    if not video_path.exists():
        print(f"Errore: file non trovato: {video_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.output_dir.resolve() if args.output_dir else video_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        audio_path = Path(tmp.name)
        extract_audio(video_path, audio_path, denoise=args.denoise)
        segments, info = transcribe(audio_path, args.model, args.language, args.device)

    if not segments:
        print("Nessun segmento trovato nel file audio.")
        sys.exit(0)

    source_lang = info.language
    orig_texts = [seg.text.strip() for seg in segments]

    # --- file in lingua originale ---
    print(f"\nFile in lingua originale ({source_lang}):")
    write_srt(segments, orig_texts, out_dir / f"{stem}.{source_lang}.srt")
    write_txt(segments, orig_texts, out_dir / f"{stem}.{source_lang}.txt")

    # --- traduzione ---
    if args.translate_to.lower() != "none" and args.translate_to != source_lang:
        target_lang = args.translate_to
        translated_texts = translate_segments(segments, source_lang, target_lang)
        print(f"\nFile tradotti ({target_lang}):")
        write_srt(segments, translated_texts, out_dir / f"{stem}.{target_lang}.srt")
        write_txt(segments, translated_texts, out_dir / f"{stem}.{target_lang}.txt")

    print(f"\nFatto! {len(segments)} segmenti.")


if __name__ == "__main__":
    main()
