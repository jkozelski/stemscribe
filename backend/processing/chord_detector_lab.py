"""Shared helpers for parsing Harte-notation `.lab` chord files.

Both the consonance-ACE and Chord-CNN-LSTM detectors produce `.lab` files in
standard Harte format (one row: `start_time<TAB>end_time<TAB>label`). The label
uses `ROOT:QUALITY[/BASS]` form (e.g. `E:min7`, `C:maj/5`, `A:min6/6`).

This module converts those `.lab` rows into the StemScriber pipeline's expected
shape: `List[Dict]` with `{time, duration, chord, root, quality, confidence,
bass, candidates}` keys. The rest of the pipeline (`chart_formatter`,
`bar_grid` build, scoring) consumes that shape unchanged.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Harte degree → semitone offset from root.
_DEGREE_TO_SEMITONE = {
    "1": 0, "2": 2, "3": 4, "4": 5, "5": 7, "6": 9, "7": 11,
    "b2": 1, "#2": 3, "b3": 3, "#4": 6, "b5": 6, "#5": 8,
    "b6": 8, "b7": 10, "#7": 11,
}

# Pitch class lookup for root + bass note conversion.
_NOTE_PC = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11,
}

_PC_TO_NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Harte quality → StemScriber chord-string suffix.
_QUALITY_MAP = {
    "maj":    "",
    "min":    "m",
    "dim":    "dim",
    "aug":    "aug",
    "7":      "7",
    "maj7":   "maj7",
    "min7":   "m7",
    "minmaj7": "mmaj7",
    "min6":   "m6",
    "maj6":   "6",
    "6":      "6",
    "dim7":   "dim7",
    "hdim7":  "m7b5",
    "sus2":   "sus2",
    "sus4":   "sus4",
    "9":      "9",
    "maj9":   "maj9",
    "min9":   "m9",
    "min11":  "m11",
    "11":     "11",
    "13":     "13",
}

# Family classification — used for "min/maj/dim/etc." root family compat
_QUALITY_FAMILY = {
    "":     "maj",
    "m":    "min",
    "dim":  "dim",
    "aug":  "aug",
    "7":    "maj",  # dominant 7 is in major family
    "maj7": "maj",
    "m7":   "min",
    "mmaj7": "min",
    "m6":   "min",
    "6":    "maj",
    "dim7": "dim",
    "m7b5": "dim",
    "sus2": "sus",
    "sus4": "sus",
    "9":    "maj",
    "maj9": "maj",
    "m9":   "min",
    "m11":  "min",
    "11":   "maj",
    "13":   "maj",
}


# ---------------------------------------------------------------------------
# Same dataclasses as chord_detector_librosa so callers can swap detectors
# behind one interface. If we keep multiple detectors long-term, factor these
# into a shared models module.
# ---------------------------------------------------------------------------

@dataclass
class ChordEvent:
    time: float
    duration: float
    chord: str
    root: str
    quality: str
    confidence: float
    bass: Optional[str] = None
    candidates: List[Dict] = field(default_factory=list)


@dataclass
class ChordProgression:
    chords: List[ChordEvent]
    key: str
    tuning_info: Optional[dict] = None


# ---------------------------------------------------------------------------
# Harte → StemScriber conversion
# ---------------------------------------------------------------------------

_CHORD_RE = re.compile(r"^([A-G][#b]?)(?::(.+?))?(?:/(.+))?$")


def harte_to_components(label: str) -> Optional[Dict[str, str]]:
    """Parse a Harte chord label.

    Returns dict {chord, root, quality, bass} where:
      - chord: StemScriber-format string ("Em7", "C/G", "F#m7b5")
      - root: pitch name (sharps), e.g. "F#"
      - quality: short-form quality, e.g. "m7", "" for major
      - bass: pitch name or None
    Returns None for "N" (no chord) or unparseable.
    """
    if not label or label in ("N", "X"):
        return None
    s = label.strip()

    # Split slash bass.
    main, _, bass_raw = s.partition("/")
    m = _CHORD_RE.match(main + ("/" + bass_raw if bass_raw else ""))
    if not m:
        m = _CHORD_RE.match(main)
        if not m:
            return None
    root, qual, bass = m.groups()

    if root not in _NOTE_PC:
        return None
    root_pc = _NOTE_PC[root]

    # Quality — strip parenthetical add-ons (e.g. "sus4(b7)" → "sus4")
    if qual:
        qual = re.sub(r"\(.*?\)", "", qual).strip()
    qual = qual or "maj"
    short_qual = _QUALITY_MAP.get(qual, qual)

    # Bass — could be a degree ("5", "b7") or a note name ("G", "F#")
    bass_note: Optional[str] = None
    if bass:
        bass = bass.strip()
        if bass in _DEGREE_TO_SEMITONE:
            bass_pc = (root_pc + _DEGREE_TO_SEMITONE[bass]) % 12
            bass_note = _PC_TO_NOTE[bass_pc]
        elif bass in _NOTE_PC:
            bass_note = _PC_TO_NOTE[_NOTE_PC[bass]]
        # Don't drop the slash if we can't parse — keep raw for forensic logs
        else:
            bass_note = bass

    chord_str = f"{root}{short_qual}"
    if bass_note and bass_note != root:
        chord_str += f"/{bass_note}"

    family = _QUALITY_FAMILY.get(short_qual, "maj")

    return {
        "chord": chord_str,
        "root": root,
        "quality": short_qual,
        "family": family,
        "bass": bass_note,
    }


# ---------------------------------------------------------------------------
# .lab file → ChordProgression
# ---------------------------------------------------------------------------

def parse_lab_file(lab_path: str | Path) -> List[ChordEvent]:
    """Read a Harte `.lab` file and return a list of ChordEvent.

    File format: `start_time<TAB>end_time<TAB>label` per line.
    Returns empty list if the file is missing or no parseable rows.
    """
    path = Path(lab_path)
    if not path.exists():
        logger.warning(f"[chord_detector_lab] .lab file missing: {path}")
        return []

    events: List[ChordEvent] = []
    skipped = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            parts = line.split("\t")
        if len(parts) < 3:
            skipped += 1
            continue
        try:
            start = float(parts[0])
            end = float(parts[1])
        except ValueError:
            skipped += 1
            continue
        label = " ".join(parts[2:])
        comps = harte_to_components(label)
        if comps is None:
            continue  # 'N' (no chord) or unparseable — skip cleanly
        duration = max(0.05, end - start)
        events.append(ChordEvent(
            time=start,
            duration=duration,
            chord=comps["chord"],
            root=comps["root"],
            quality=comps["quality"],
            confidence=1.0,  # CNN/Conformer models don't emit per-event scores via .lab
            bass=comps["bass"] or comps["root"],
            candidates=[],   # top-K not exposed by ACE/Jiang CLIs
        ))
    if skipped:
        logger.debug(f"[chord_detector_lab] skipped {skipped} unparseable rows in {path}")
    return events


# ---------------------------------------------------------------------------
# Pipeline-shaped wrapper
# ---------------------------------------------------------------------------

def chord_events_to_job_format(events: List[ChordEvent]) -> List[Dict]:
    """Convert ChordEvent dataclasses to the dict shape `job.chord_progression` expects."""
    return [
        {
            "time": ce.time,
            "duration": ce.duration,
            "chord": ce.chord,
            "root": ce.root,
            "quality": ce.quality,
            "confidence": ce.confidence,
            "bass": ce.bass,
            "candidates": ce.candidates,
        }
        for ce in events
    ]


def detect_key_from_events(events: List[ChordEvent]) -> str:
    """Re-use stem_chord_detector's K-K key detector. Lazy import to keep this
    module load-light when key detection isn't needed.
    """
    if not events:
        return "Unknown"
    try:
        from stem_chord_detector import detect_key_from_chords  # type: ignore
    except ImportError:
        try:
            from backend.stem_chord_detector import detect_key_from_chords  # type: ignore
        except ImportError:
            logger.warning("[chord_detector_lab] stem_chord_detector not importable — key=Unknown")
            return "Unknown"
    return detect_key_from_chords(events)
