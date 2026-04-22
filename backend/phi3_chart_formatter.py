"""
Phi-3 chart formatter client — calls the deployed Modal function for LLM-based
chord chart formatting.

Takes raw chord events + word timestamps and produces clean, structured chord charts
using a QLoRA fine-tuned Phi-3-mini-4k-instruct model running on Modal GPU.

Three output formats:
  - "json"     → StemScriber library JSON (default, used by frontend)
  - "slash"    → Slash notation (Jeff's handwritten style)
  - "chordpro" → ChordPro interchange format

The Modal function must be deployed first:
    cd ~/stemscribe/backend && ../venv311/bin/python -m modal deploy modal_chart_formatter.py

Falls back to rule-based chart_formatter.py on failure.

Usage:
    from phi3_chart_formatter import format_chart_phi3
    chart = format_chart_phi3(chord_events, word_timestamps, title="...", artist="...")
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ADAPTER_DIR = Path(__file__).parent / "models" / "pretrained" / "chart_formatter_lora"
# Adapter lives on Modal volume, not required locally on VPS
ADAPTER_AVAILABLE = True

# Try to import Modal for remote inference
try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Prompt construction (must match training format exactly)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a professional chord chart formatter. Given raw chord detection data with timestamps, format a clean, readable chord chart."

FORMAT_LABELS = {
    "json": "StemScriber JSON",
    "slash": "slash notation",
    "chordpro": "ChordPro",
}


def _build_user_prompt(
    chord_events: List[Dict],
    word_timestamps: List[Dict],
    title: str,
    artist: str,
    key: str,
    tempo: int = 120,
    time_sig: str = "4/4",
    output_format: str = "json",
) -> str:
    """Build the user prompt matching the training data format."""
    format_label = FORMAT_LABELS.get(output_format, "StemScriber JSON")

    # Compact chord events (keep only time, duration, chord, confidence)
    compact_chords = []
    for c in chord_events:
        compact_chords.append({
            "time": round(c.get("time", 0), 3),
            "duration": round(c.get("duration", 0.5), 3),
            "chord": c.get("chord", "N"),
            "confidence": round(c.get("confidence", 0.8), 2),
        })

    # Compact word timestamps
    compact_words = []
    for w in word_timestamps:
        compact_words.append({
            "word": w.get("word", ""),
            "start": round(w.get("start", 0), 3),
            "end": round(w.get("end", 0), 3),
        })

    chords_json = json.dumps(compact_chords, separators=(",", ":"))
    words_json = json.dumps(compact_words, separators=(",", ":"))

    return (
        f"Format this as a {format_label} chord chart:\n\n"
        f"Title: {title}\n"
        f"Artist: {artist}\n"
        f"Key: {key}\n"
        f"Tempo: {tempo} BPM\n"
        f"Time Signature: {time_sig}\n\n"
        f"Chord events:\n{chords_json}\n\n"
        f"Lyrics with timing:\n{words_json}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def format_chart_phi3(
    chord_events: List[Dict],
    word_timestamps: List[Dict],
    title: str = "Unknown",
    artist: str = "Unknown",
    key: str = "Unknown",
    tempo: int = 120,
    time_sig: str = "4/4",
    output_format: str = "json",
) -> Optional[Dict]:
    """
    Format a chord chart using the fine-tuned Phi-3 model on Modal.

    Calls the deployed "stemscribe-chart-formatter" Modal app via
    modal.Function.from_name(). Falls back gracefully if Modal is
    unavailable or the function is not deployed.

    Args:
        chord_events: List of {"time": float, "duration": float, "chord": str, ...}
        word_timestamps: List of {"word": str, "start": float, "end": float}
        title, artist, key: Song metadata
        tempo: BPM (default 120)
        time_sig: Time signature string (default "4/4")
        output_format: "json" (default), "slash", or "chordpro"

    Returns:
        Dict (chord library JSON) if format is "json", or None on failure.
        For "slash"/"chordpro", returns {"text": str, "format": str}.
    """
    if not chord_events:
        logger.warning("Phi-3 formatter: no chord events provided")
        return None

    if not MODAL_AVAILABLE:
        logger.warning("Phi-3 formatter: Modal not available, cannot run inference")
        return None

    user_prompt = _build_user_prompt(
        chord_events=chord_events,
        word_timestamps=word_timestamps,
        title=title,
        artist=artist,
        key=key,
        tempo=tempo,
        time_sig=time_sig,
        output_format=output_format,
    )

    try:
        # Look up the deployed Modal function by app/function name
        format_fn = modal.Function.from_name(
            "stemscribe-chart-formatter", "format_chart_gpu"
        )
        raw_output = format_fn.remote(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_new_tokens=2048,
        )

        if not raw_output:
            logger.warning("Phi-3 formatter returned empty output")
            return None

        # Log first 500 chars for debugging
        logger.info(f"Phi-3 raw output ({len(raw_output)} chars): {raw_output[:500]}")

        if output_format == "json":
            return _parse_json_output(raw_output, title, artist, key)
        else:
            return {"text": raw_output, "format": output_format}

    except Exception as e:
        logger.error(f"Phi-3 chart formatter failed: {e}")
        return None


def _parse_json_output(raw: str, title: str, artist: str, key: str) -> Optional[Dict]:
    """Parse and validate JSON output from the model."""
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        chart = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the output
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                chart = json.loads(text[start:end])
            except json.JSONDecodeError:
                logger.error("Phi-3 formatter: could not parse JSON output")
                return None
        else:
            logger.error("Phi-3 formatter: no JSON object found in output")
            return None

    # Validate required fields
    if not isinstance(chart, dict) or "sections" not in chart:
        logger.error("Phi-3 formatter: output missing 'sections' field")
        return None

    # Ensure metadata is set
    chart.setdefault("title", title)
    chart.setdefault("artist", artist)
    chart.setdefault("key", key)
    chart.setdefault("source", "StemScriber AI (Phi-3)")
    chart.setdefault("capo", 0)

    # Validate sections structure
    for section in chart.get("sections", []):
        if not isinstance(section, dict):
            continue
        section.setdefault("name", "Section")
        section.setdefault("lines", [])

    # Collect chords_used if not present
    if "chords_used" not in chart:
        chords_seen = []
        seen_set = set()
        for section in chart.get("sections", []):
            for line in section.get("lines", []):
                chord_str = line.get("chords", "")
                for chord in chord_str.split():
                    chord = chord.strip()
                    if chord and chord not in seen_set and chord not in ("-", "|", "/"):
                        seen_set.add(chord)
                        chords_seen.append(chord)
        chart["chords_used"] = chords_seen

    # Strip internal fields the model may have copied from training data
    chart.pop("_source_file", None)

    logger.info(
        f"Phi-3 formatter produced chart: {len(chart.get('sections', []))} sections, "
        f"{sum(len(s.get('lines', [])) for s in chart.get('sections', []))} lines"
    )
    return chart


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def is_phi3_formatter_available() -> bool:
    """Check if the Phi-3 formatter can be used (adapter exists + Modal available)."""
    return ADAPTER_AVAILABLE and MODAL_AVAILABLE
