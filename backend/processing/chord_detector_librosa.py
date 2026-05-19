"""
librosa-based chord detector — chroma + 24-template (12 maj + 12 min) matcher.

Replacement for the stem_chord_detector bass-root-first approach. Operates on
the FULL MIX audio (not stems) which empirically beat the stem-aware path by
+38 points on rock and tied/won on jazz in the 2026-05-06 audit.

Output shape matches stem_chord_detector.ChordProgression so the rest of the
pipeline (chart_formatter, bar_grid build) works unchanged.

Gated behind USE_LIBROSA_DETECTOR=true env var. When unset, this module is
not imported and the legacy stem-aware detector runs as before.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Same dataclasses as stem_chord_detector — duplicated here so callers can use
# either detector behind the same interface. If we standardize, move these to
# a shared models module.
@dataclass
class ChordEvent:
    time: float
    duration: float
    chord: str
    root: str
    quality: str
    confidence: float
    bass: Optional[str] = None
    # Top-K candidates from the template scorer — populated by detect_chords_librosa.
    # Each entry: {"chord": "Am", "score": 0.91, "root": "A", "quality": "min"}.
    # Used by the re-ranker corrector (chord_corrector_anthropic.apply_correction_reranker)
    # to let Claude pick from multiple chord guesses per beat. Empty list when
    # the detector is the legacy stem_chord_detector path.
    candidates: List[Dict] = field(default_factory=list)


@dataclass
class ChordProgression:
    chords: List[ChordEvent]
    key: str
    tuning_info: Optional[dict] = None


# 12-tone pitch class names (using sharps; the rest of the pipeline accepts both)
PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Major triad template: root + maj3 + 5
MAJOR_T = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
# Minor triad template: root + min3 + 5
MINOR_T = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)


def _build_templates():
    """Return (24, 12) array of templates and parallel list of (root_idx, quality) labels."""
    templates = []
    labels = []  # list of (root_idx, 'maj' | 'min')
    for root in range(12):
        templates.append(np.roll(MAJOR_T, root))
        labels.append((root, 'maj'))
        templates.append(np.roll(MINOR_T, root))
        labels.append((root, 'min'))
    return np.array(templates), labels


_TEMPLATES, _LABELS = _build_templates()


def _label_to_chord_str(root_idx: int, quality: str) -> str:
    """Render (root_idx, quality) → chord string compatible with chart_formatter."""
    name = PITCH_NAMES[root_idx]
    if quality == 'maj':
        return name           # bare letter for major
    elif quality == 'min':
        return name + 'm'     # 'Am', 'F#m'
    return name


def detect_chords_librosa(audio_path: str,
                          sample_rate: int = 22050,
                          hop_length: int = 512) -> ChordProgression:
    """
    Run chroma + template chord detection on the full-mix audio file.

    Args:
        audio_path: path to MP3/WAV/etc audio file (full mix preferred)
        sample_rate: librosa load rate
        hop_length: chromagram frame hop

    Returns:
        ChordProgression with chord_events and a derived key.
    """
    import librosa  # imported here so module loads even when librosa is absent

    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

    # CQT-based chromagram (better octave equivariance than STFT chroma)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

    # Beat-synchronous segmentation. One chord per beat.
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    if len(beat_frames) < 4:
        # Fallback: 1-second window aggregation when beat-tracking fails (drum-light tracks)
        beat_frames = np.arange(0, chroma.shape[1], max(1, sr // hop_length))

    # Average chroma per beat (median is more robust to noise than mean)
    beat_chroma = librosa.util.sync(chroma, beat_frames, aggregate=np.median)

    # Convert beat frame indices to seconds for chord_event timestamps
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length).tolist()
    # Add a synthetic end-time at the audio duration so the last chord has a duration
    audio_duration = len(y) / sr
    beat_times.append(audio_duration)

    # Top-K extraction config (see docs/reranker-design-2026-05-11.md §2).
    # K=4 by default — the 24-template bank caps useful re-ranks well below
    # this. Min-score floor drops template misfires on silence/percussion-only
    # beats so Claude doesn't waste budget arguing over noise.
    topk_k = max(1, int(os.environ.get("ANTHROPIC_CORRECTION_RERANKER_K") or "4"))
    topk_min_score = float(
        os.environ.get("ANTHROPIC_CORRECTION_RERANKER_MIN_SCORE") or "0.55"
    )

    # Score each beat against all 24 templates; argmax → chord
    chord_events: List[ChordEvent] = []
    for i in range(beat_chroma.shape[1]):
        v = beat_chroma[:, i]
        if v.sum() < 1e-3:
            continue  # silence / no harmonic content
        v_norm = v / (np.linalg.norm(v) + 1e-9)
        scores = _TEMPLATES @ v_norm  # (24,)
        best = int(np.argmax(scores))
        confidence = float(scores[best])  # 0..1 cosine-ish
        root_idx, quality = _LABELS[best]
        chord_str = _label_to_chord_str(root_idx, quality)

        # Compute top-K candidates above min-score floor for the re-ranker
        # corrector. The argmax above is always candidates[0] when it clears
        # the floor. Keep candidates a plain list of dicts so this serializes
        # cleanly into chord_chart.json.
        top_idx = np.argsort(-scores)[:topk_k]
        candidates: List[Dict] = []
        for idx in top_idx:
            idx_i = int(idx)
            s = float(scores[idx_i])
            if s < topk_min_score:
                continue
            cand_root_idx, cand_quality = _LABELS[idx_i]
            candidates.append({
                "chord": _label_to_chord_str(cand_root_idx, cand_quality),
                "score": round(s, 4),
                "root": PITCH_NAMES[cand_root_idx],
                "quality": cand_quality,
            })

        start = beat_times[i] if i < len(beat_times) else audio_duration
        end = beat_times[i + 1] if (i + 1) < len(beat_times) else audio_duration
        duration = max(0.05, end - start)

        chord_events.append(ChordEvent(
            time=float(start),
            duration=float(duration),
            chord=chord_str,
            root=PITCH_NAMES[root_idx],
            quality=quality,
            confidence=confidence,
            bass=PITCH_NAMES[root_idx],  # template detector has no inversion info
            candidates=candidates,
        ))

    # Derive song-level key via PCP-weighted Krumhansl-Kessler. Naive
    # most-time-occupied-root picks the V on V-dominant pop (e.g. "Your
    # Wildest Dreams" lands on C when the song is in G; "Bad Company" lands
    # on D#m when it's in G). detect_key_from_chords sums chord-implied
    # pitch classes (root + 3rd + 5th, weighted by duration*confidence)
    # before correlating against K-K major/minor profiles, which is robust
    # to V-heavy progressions and slash-chord moments. chart_formatter's
    # _redetect_key_from_bargrid is a no-op when bar_grid lacks root/quality
    # fields, so this upstream key has to be right on its own.
    if chord_events:
        from stem_chord_detector import detect_key_from_chords
        key_str = detect_key_from_chords(chord_events)
    else:
        key_str = 'Unknown'

    logger.info(
        f"librosa detector: {len(chord_events)} chord events, "
        f"key={key_str}, tempo={float(tempo):.1f} BPM"
    )

    return ChordProgression(
        chords=chord_events,
        key=key_str,
        tuning_info=None,
    )


def detect_chords_for_job_librosa(job, audio_path) -> None:
    """
    Pipeline-shaped wrapper: takes a job + audio path, populates job.chord_progression
    and job.detected_key. Drop-in replacement for `detect_chords_for_job` from the
    legacy detector path.
    """
    audio_path_str = str(audio_path)
    logger.info(f"librosa detector: processing {audio_path_str}")

    progression = detect_chords_librosa(audio_path_str)

    # job.chord_progression expects a list of dicts (NOT ChordEvent dataclasses)
    # per chart_formatter's signature: List of {"time": float, "duration": float, "chord": str, ...}
    # The optional `candidates` field carries top-K data for the re-ranker
    # corrector — downstream consumers may ignore it.
    job.chord_progression = [
        {
            'time': ce.time,
            'duration': ce.duration,
            'chord': ce.chord,
            'root': ce.root,
            'quality': ce.quality,
            'confidence': ce.confidence,
            'bass': ce.bass,
            'candidates': ce.candidates,
        }
        for ce in progression.chords
    ]
    job.detected_key = progression.key
    logger.info(
        f"librosa detector wrote {len(job.chord_progression)} chord_events "
        f"to job, key={job.detected_key}"
    )
