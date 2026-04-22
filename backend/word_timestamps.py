"""
Word-level timestamp extraction using faster-whisper.
Used by chord_sheet.py to accurately place chords above lyrics.
"""

import logging
import os
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

_model = None

# Use medium model on CPU (large-v3 OOMs on 8GB VPS with float32)
# medium still gives excellent word timestamps at ~1.5GB RAM vs ~6GB for large
import torch as _torch
_model_size = "large-v3" if _torch.cuda.is_available() or (hasattr(_torch.backends, 'mps') and _torch.backends.mps.is_available()) else "medium"
_compute_type = "float16" if _torch.cuda.is_available() else "int8" if _model_size == "medium" else "float32"
del _torch


def _get_model():
    """Lazy-load the Whisper model."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        logger.info(f"Loading faster-whisper model ({_model_size}, compute_type={_compute_type})...")
        _model = WhisperModel(_model_size, device="cpu", compute_type=_compute_type)
        logger.info("faster-whisper model loaded")
    return _model


def get_word_timestamps(audio_path: str) -> List[Dict]:
    """
    Extract word-level timestamps from an audio file (ideally a vocal stem).

    Returns list of {"word": str, "start": float, "end": float}
    """
    model = _get_model()

    segments, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        language="en",
        vad_filter=True,
    )

    words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                words.append({
                    "word": w.word.strip(),
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                })

    logger.info(f"Extracted {len(words)} word timestamps from {os.path.basename(audio_path)}")
    return words


def align_lyrics_with_words(synced_lyrics: List[Dict], word_timestamps: List[Dict]) -> List[Dict]:
    """
    Enhance synced lyrics with per-word timing from Whisper.

    Takes lrclib synced lyrics (line-level) and Whisper word timestamps,
    matches them up, and returns lyrics with word-level timing.

    Returns list of {
        "time": float,          # line start time
        "text": str,            # full line text
        "words": [              # per-word timing
            {"word": str, "start": float, "char_pos": int},
            ...
        ]
    }
    """
    if not word_timestamps:
        return synced_lyrics

    # Build a flat list of all whisper words with their times
    whisper_words = [(w["word"].lower().strip(".,!?;:'\"()"), w["start"], w["end"]) for w in word_timestamps]
    wi = 0  # whisper word index

    enhanced = []
    for li, lyric in enumerate(synced_lyrics):
        text = lyric.get("text", "").strip()
        line_time = lyric["time"]
        next_time = synced_lyrics[li + 1]["time"] if li + 1 < len(synced_lyrics) else line_time + 15

        if not text:
            enhanced.append({"time": line_time, "text": text, "words": []})
            continue

        lyric_words = text.split()
        word_info = []
        char_pos = 0

        for lw in lyric_words:
            lw_clean = lw.lower().strip(".,!?;:'\"()")

            # Find best matching whisper word near this line's time window
            best_match = None
            best_dist = float("inf")

            # Search forward from current position
            for j in range(wi, min(wi + 30, len(whisper_words))):
                ww_text, ww_start, ww_end = whisper_words[j]

                # Must be in reasonable time range for this line
                if ww_start < line_time - 2.0:
                    continue
                if ww_start > next_time + 2.0:
                    break

                # Check text similarity
                if ww_text == lw_clean or lw_clean.startswith(ww_text) or ww_text.startswith(lw_clean):
                    dist = abs(ww_start - line_time)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = j

            if best_match is not None:
                _, start, _ = whisper_words[best_match]
                word_info.append({
                    "word": lw,
                    "start": start,
                    "char_pos": char_pos,
                })
                wi = best_match + 1
            else:
                # No match — estimate from line time
                word_info.append({
                    "word": lw,
                    "start": None,
                    "char_pos": char_pos,
                })

            char_pos += len(lw) + 1

        enhanced.append({
            "time": line_time,
            "text": text,
            "words": word_info,
        })

    return enhanced
