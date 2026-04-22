"""
Tempo and beat-grid extraction.

Produces the musical grid that everything downstream should quantize to:
  - tempo_bpm:     float, beats per minute
  - time_signature: "4/4" (assumed — we don't detect meter yet)
  - beat_times:    list[float], every beat in seconds
  - downbeat_times: list[float], every bar-start in seconds
  - bar_count:     int, number of full bars in the song

Design choices:
  * Prefer the drums stem if available — cleaner beat than the full mix,
    especially on heavily produced pop/rock where melodic content can fool
    onset detection.
  * Fall back to the full mix if no drums stem provided or drums are too
    quiet (e.g. ballad with brushes, acoustic track).
  * Time signature is assumed 4/4. Real meter detection (3/4, 6/8, 7/8)
    is a future problem — 4/4 is right for ~95% of pop/rock/funk/jazz.
  * Downbeat inference uses librosa's beat tracker and then anchors bar
    starts to the first beat whose preceding onset-strength peak is highest
    (typical downbeat heuristic). This is not perfect but catches the common
    case where the tracker locks onto the off-beat.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Plausible pop/rock/jazz tempo range. Outside this we refuse to trust the
# estimate and fall back to 120 BPM.
MIN_BPM = 40.0
MAX_BPM = 220.0
DEFAULT_BPM = 120.0

# How many beats per bar we assume.
BEATS_PER_BAR = 4


def extract_grid(
    drums_path: Optional[str] = None,
    mix_path: Optional[str] = None,
    sample_rate: int = 22050,
) -> Dict:
    """
    Extract tempo and beat grid from an audio file.

    Preference order:
      1. drums_path if provided and has content
      2. mix_path (required fallback)

    Returns a dict safe to store in job_metadata.json["grid"]. Never raises —
    returns a DEFAULT grid on failure so downstream steps can still run.
    """
    source = _pick_source(drums_path, mix_path)
    if source is None:
        logger.warning("tempo_beats: no audio source available, returning default grid")
        return _default_grid()

    try:
        import librosa
        import numpy as np
    except ImportError as e:
        logger.error(f"tempo_beats: librosa not installed ({e}), returning default grid")
        return _default_grid()

    try:
        audio, sr = librosa.load(source, sr=sample_rate, mono=True)
    except Exception as e:
        logger.error(f"tempo_beats: failed to load {source}: {e}")
        return _default_grid()

    if audio.size == 0:
        logger.warning(f"tempo_beats: {source} is empty")
        return _default_grid()

    try:
        tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr, units="frames")
    except Exception as e:
        logger.error(f"tempo_beats: beat_track failed on {source}: {e}")
        return _default_grid()

    tempo_bpm = _coerce_tempo(tempo)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    if len(beat_times) < BEATS_PER_BAR:
        logger.warning(f"tempo_beats: only {len(beat_times)} beats detected, insufficient")
        return _default_grid(tempo=tempo_bpm)

    downbeat_times, downbeat_offset = _infer_downbeats(audio, sr, beat_times)

    song_duration = float(len(audio)) / sr
    bar_count = len(downbeat_times)

    grid = {
        "tempo_bpm": round(tempo_bpm, 2),
        "time_signature": f"{BEATS_PER_BAR}/4",
        "beat_times": [round(t, 4) for t in beat_times],
        "downbeat_times": [round(t, 4) for t in downbeat_times],
        "bar_count": bar_count,
        "downbeat_offset": downbeat_offset,
        "source": Path(source).name,
        "song_duration_sec": round(song_duration, 2),
    }
    logger.info(
        f"tempo_beats: {Path(source).name} -> {tempo_bpm:.1f} BPM, "
        f"{len(beat_times)} beats, {bar_count} bars (downbeat offset {downbeat_offset})"
    )
    return grid


def time_to_bar(t: float, grid: Dict) -> int:
    """
    Map a time in seconds to a bar number (1-indexed). Returns 0 for times
    before the first downbeat (pickup measure).
    """
    downbeats = grid.get("downbeat_times", [])
    if not downbeats or t < downbeats[0]:
        return 0
    # Binary search would be tighter but bar counts are small
    for i in range(len(downbeats) - 1):
        if downbeats[i] <= t < downbeats[i + 1]:
            return i + 1
    return len(downbeats)


def bar_to_time(bar: int, grid: Dict) -> float:
    """Inverse of time_to_bar: bar number -> downbeat time in seconds."""
    downbeats = grid.get("downbeat_times", [])
    if bar < 1 or bar > len(downbeats):
        return 0.0
    return downbeats[bar - 1]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _pick_source(drums_path: Optional[str], mix_path: Optional[str]) -> Optional[str]:
    """Prefer drums, fall back to mix."""
    if drums_path and Path(drums_path).is_file() and _has_content(drums_path):
        return drums_path
    if mix_path and Path(mix_path).is_file():
        return mix_path
    return None


def _has_content(path: str, min_size_bytes: int = 50_000) -> bool:
    """Quick size-based check to skip near-silent stems."""
    try:
        return Path(path).stat().st_size > min_size_bytes
    except OSError:
        return False


def _coerce_tempo(raw) -> float:
    """
    librosa.beat_track returns either a scalar float or a numpy array
    depending on version. Also clamp to a sane range.
    """
    try:
        if hasattr(raw, "__len__") and len(raw) > 0:
            val = float(raw[0])
        else:
            val = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_BPM
    if not (MIN_BPM <= val <= MAX_BPM):
        logger.warning(f"tempo_beats: tempo {val:.1f} out of range [{MIN_BPM}, {MAX_BPM}], using default")
        return DEFAULT_BPM
    return val


def _infer_downbeats(audio, sr: int, beat_times: List[float]):
    """
    Pick which beat is beat-1 of each bar.

    librosa.beat_track returns evenly-spaced beats but no bar alignment — we
    could be locked onto beat 2 or 4 of the measure. The heuristic:
      1. Compute onset strength (percussive energy) at every beat.
      2. For each of the BEATS_PER_BAR candidate phases, sum the onset
         strengths at positions [phase, phase+4, phase+8, ...].
      3. The phase with maximum total is the downbeat phase.

    Downbeats are typically louder (kick drum, bass hit), so this works well
    on pop/rock. Fails on syncopated music (jazz, prog, funk with pushing-
    the-one grooves) — we accept this and will add metadata-based overrides
    later (e.g. "user says bar 1 starts at t=2.41").

    Returns (downbeat_times, offset) where offset is the beat index (0-3)
    that was chosen as the downbeat phase.
    """
    try:
        import librosa
        import numpy as np
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        onset_at_beats = []
        for t in beat_times:
            frame = librosa.time_to_frames(t, sr=sr)
            idx = min(max(0, int(frame)), len(onset_env) - 1)
            onset_at_beats.append(float(onset_env[idx]))
        onset_at_beats = np.array(onset_at_beats)

        best_phase = 0
        best_score = -1.0
        for phase in range(BEATS_PER_BAR):
            score = float(onset_at_beats[phase::BEATS_PER_BAR].sum())
            if score > best_score:
                best_score = score
                best_phase = phase

        downbeats = beat_times[best_phase::BEATS_PER_BAR]
        return downbeats, best_phase
    except Exception as e:
        logger.warning(f"tempo_beats: downbeat inference fell back to phase 0: {e}")
        return beat_times[::BEATS_PER_BAR], 0


def _default_grid(tempo: float = DEFAULT_BPM) -> Dict:
    """Safe fallback when detection fails — downstream code can still run."""
    return {
        "tempo_bpm": tempo,
        "time_signature": f"{BEATS_PER_BAR}/4",
        "beat_times": [],
        "downbeat_times": [],
        "bar_count": 0,
        "downbeat_offset": 0,
        "source": None,
        "song_duration_sec": 0.0,
    }


# ---------------------------------------------------------------------------
# CLI — run on a file so Jeff can sanity-check the output before wiring up
# ---------------------------------------------------------------------------

def _cli():
    import argparse, json, sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Extract tempo + beat grid from audio")
    p.add_argument("audio", help="Path to audio file (drums stem preferred, mix OK)")
    p.add_argument("--mix", help="Optional fallback mix path")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    p.add_argument("--head", type=int, default=0, help="Only print first N beats/downbeats")
    args = p.parse_args()

    grid = extract_grid(drums_path=args.audio, mix_path=args.mix)

    if args.head > 0:
        grid = dict(grid)
        grid["beat_times"] = grid["beat_times"][: args.head]
        grid["downbeat_times"] = grid["downbeat_times"][: args.head]

    json.dump(grid, sys.stdout, indent=2 if args.pretty else None)
    sys.stdout.write("\n")


if __name__ == "__main__":
    _cli()
