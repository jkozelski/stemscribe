"""Jiang Chord-CNN-LSTM detector wrapper (ISMIR 2019, large-vocab 301 classes).

Wraps the third-party detector at `backend/external/chord_cnn_lstm/`
(`music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition`). Same subprocess-
isolated shape as the consonance-ACE wrapper.

Empirical baseline (2026-05-13, 6-song A/B vs current prod):
  - Average root F1: 0.786 (+0.07 over current prod librosa+V1).
  - Wins on 5/6 songs. Mary Jane's regression (-0.21) is fixed by router
    branching to ACE for that song's failure mode.
  - Strong on Hotel California (0.98), House of Rising Sun (0.99), Crazy On You
    (0.92). Used as the "clean rock" path in the per-bar router.

Used via chord_router.py — not called directly by pipeline. The chord_router
picks ACE or Jiang per-song based on bar-agreement + event-density heuristic.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from processing.chord_detector_lab import (
    ChordProgression,
    chord_events_to_job_format,
    detect_key_from_events,
    parse_lab_file,
)

logger = logging.getLogger(__name__)


# Repo layout: backend/processing/chord_detector_jiang.py
# Jiang lives at: backend/external/chord_cnn_lstm/
_REPO_ROOT = Path(__file__).resolve().parent.parent
_JIANG_DIR = _REPO_ROOT / "external" / "chord_cnn_lstm"
# Jiang ships 5 ensemble checkpoints in cache_data/ (joint_chord_net_*.best.sdict).
# Verify at least one is present at module-import time.
_JIANG_CACHE = _JIANG_DIR / "cache_data"
_JIANG_CLI = _JIANG_DIR / "chord_recognition.py"

# Same 5-min timeout shape as ACE wrapper.
_TIMEOUT_S = float(os.environ.get("JIANG_INFERENCE_TIMEOUT_S") or "300.0")


def _jiang_available() -> bool:
    if not _JIANG_CLI.exists():
        logger.warning(f"[jiang] inference CLI missing: {_JIANG_CLI}")
        return False
    if not _JIANG_CACHE.exists():
        logger.warning(f"[jiang] cache_data dir missing: {_JIANG_CACHE}")
        return False
    sdicts = list(_JIANG_CACHE.glob("joint_chord_net_*.best.sdict"))
    if not sdicts:
        logger.warning(f"[jiang] no .sdict checkpoints found in {_JIANG_CACHE}")
        return False
    return True


def detect_chords_jiang(audio_path: str) -> ChordProgression:
    """Run Jiang Chord-CNN-LSTM on a single audio file."""
    audio = Path(audio_path)
    if not audio.exists():
        logger.error(f"[jiang] audio file missing: {audio}")
        return ChordProgression(chords=[], key="Unknown")
    if not _jiang_available():
        return ChordProgression(chords=[], key="Unknown")

    with tempfile.NamedTemporaryFile(suffix=".lab", delete=False) as tmp:
        lab_path = Path(tmp.name)

    try:
        cmd = [sys.executable, "chord_recognition.py", str(audio), str(lab_path)]
        logger.info(f"[jiang] running inference on {audio.name} (timeout {_TIMEOUT_S}s)")
        try:
            result = subprocess.run(
                cmd,
                cwd=str(_JIANG_DIR),
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            logger.error(f"[jiang] inference timed out after {_TIMEOUT_S}s on {audio.name}")
            return ChordProgression(chords=[], key="Unknown")

        if result.returncode != 0:
            logger.error(
                f"[jiang] inference failed rc={result.returncode} on {audio.name}: "
                f"{(result.stderr or '')[:400]}"
            )
            return ChordProgression(chords=[], key="Unknown")

        events = parse_lab_file(lab_path)
        if not events:
            logger.warning(f"[jiang] empty chord output for {audio.name}")
            return ChordProgression(chords=[], key="Unknown")

        key = detect_key_from_events(events)
        logger.info(f"[jiang] {audio.name}: {len(events)} chord events, key={key}")
        return ChordProgression(chords=events, key=key)
    finally:
        try:
            lab_path.unlink()
        except OSError:
            pass


def detect_chords_for_job_jiang(job, audio_path) -> None:
    progression = detect_chords_jiang(str(audio_path))
    job.chord_progression = chord_events_to_job_format(progression.chords)
    job.detected_key = progression.key
    logger.info(
        f"[jiang] wrote {len(job.chord_progression)} events, key={job.detected_key}"
    )
