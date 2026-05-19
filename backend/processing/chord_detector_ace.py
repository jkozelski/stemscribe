"""consonance-ACE chord detector wrapper.

Wraps the third-party consonance-ACE inference CLI at
`backend/external/consonance-ACE/ACE/inference.py`. Subprocess isolation keeps
PyTorch state (and any model crashes) from polluting the parent Python process.

Empirical baseline (2026-05-13, 13-song UG benchmark):
  - Average root F1: 0.852 (+0.139 over current prod librosa+V1).
  - Wins on 12/13 songs vs V1.
  - Catches slash chords + extensions natively (Em7, Em6, Em/C#, Am6/F#).
  - Known weakness: flattens jazz extensions on Steely Dan / Jamiroquai
    (Aadd9 → A, Bm11 → Bm7). Routed away via detector_router → 'jazz' path.

Gated behind USE_ACE_DETECTOR=true env var (set in detector chain logic, not
imported as a default).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from processing.chord_detector_lab import (
    ChordProgression,
    chord_events_to_job_format,
    detect_key_from_events,
    parse_lab_file,
)

logger = logging.getLogger(__name__)


# Repo layout: backend/processing/chord_detector_ace.py
# ACE lives at:  backend/external/consonance-ACE/
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ACE_DIR = _REPO_ROOT / "external" / "consonance-ACE"
_ACE_CKPT = _ACE_DIR / "ACE" / "checkpoints" / "conformer_decomposed_smooth.ckpt"

# Inference timeout: typical 4-min song = 7-15s on M3 Max, est 15-40s on CPX41
# CPU. Padded for very long uploads. Bound prevents post-sep semaphore deadlock.
_TIMEOUT_S = float(os.environ.get("ACE_INFERENCE_TIMEOUT_S") or "300.0")

# Default tuning — bake-off found defaults (threshold 0.5, min_dur 0.5) win on
# 12/13 songs. Tuning them HIGHER backfires (avg 0.852 → 0.667). Knobs exposed
# via env so we can sweep without code change if needed.
_THRESHOLD = float(os.environ.get("ACE_THRESHOLD") or "0.5")
_MIN_DURATION = float(os.environ.get("ACE_MIN_DURATION") or "0.5")


def _ace_available() -> bool:
    """Verify the ACE checkpoint and entrypoint are present."""
    if not _ACE_CKPT.exists():
        logger.warning(f"[ace] checkpoint missing: {_ACE_CKPT}")
        return False
    if not (_ACE_DIR / "ACE" / "inference.py").exists():
        logger.warning(f"[ace] inference module missing in {_ACE_DIR}")
        return False
    return True


def detect_chords_ace(audio_path: str) -> ChordProgression:
    """Run consonance-ACE on a single audio file.

    Args:
        audio_path: path to MP3/WAV/etc audio file (full mix preferred).

    Returns:
        ChordProgression with chord_events and a K-K derived key. Empty
        progression on failure (caller can fall back).
    """
    audio = Path(audio_path)
    if not audio.exists():
        logger.error(f"[ace] audio file missing: {audio}")
        return ChordProgression(chords=[], key="Unknown")
    if not _ace_available():
        return ChordProgression(chords=[], key="Unknown")

    # Use a tempfile for the .lab output so concurrent jobs don't collide.
    with tempfile.NamedTemporaryFile(suffix=".lab", delete=False) as tmp:
        lab_path = Path(tmp.name)

    try:
        cmd = [
            sys.executable, "-m", "ACE.inference",
            "--audio", str(audio),
            "--out", str(lab_path),
            "--threshold", str(_THRESHOLD),
            "--chord-min-duration", str(_MIN_DURATION),
        ]
        logger.info(f"[ace] running inference on {audio.name} (timeout {_TIMEOUT_S}s)")
        try:
            result = subprocess.run(
                cmd,
                cwd=str(_ACE_DIR),
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            logger.error(f"[ace] inference timed out after {_TIMEOUT_S}s on {audio.name}")
            return ChordProgression(chords=[], key="Unknown")

        if result.returncode != 0:
            logger.error(
                f"[ace] inference failed rc={result.returncode} on {audio.name}: "
                f"{(result.stderr or '')[:400]}"
            )
            return ChordProgression(chords=[], key="Unknown")

        events = parse_lab_file(lab_path)
        if not events:
            logger.warning(f"[ace] empty chord output for {audio.name}")
            return ChordProgression(chords=[], key="Unknown")

        key = detect_key_from_events(events)
        logger.info(f"[ace] {audio.name}: {len(events)} chord events, key={key}")
        return ChordProgression(chords=events, key=key)
    finally:
        try:
            lab_path.unlink()
        except OSError:
            pass


def detect_chords_for_job_ace(job, audio_path) -> None:
    """Pipeline-shaped wrapper. Populates job.chord_progression + job.detected_key.

    On detector failure (timeout, crash, empty output), leaves both fields
    in their unset state so the caller can detect failure via empty
    chord_progression and fall through to a fallback.
    """
    progression = detect_chords_ace(str(audio_path))
    job.chord_progression = chord_events_to_job_format(progression.chords)
    job.detected_key = progression.key
    logger.info(
        f"[ace] wrote {len(job.chord_progression)} events, key={job.detected_key}"
    )
