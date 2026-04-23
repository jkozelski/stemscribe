"""
MIDI-intermediate chord detector.

Replaces the CQT→BTC→pitch-class-pruning matching layer with a note-based
matcher fed by Spotify Basic Pitch MIDI. The 2026-04-23 Phase-0 prototype
(see docs/midi-prototype-results-2026-04-23.md) confirmed that Basic Pitch's
raw note stream contains the 7ths on 100% of ground-truth m7 bars in
Jamiroquai's "Alright" — the existing pipeline loses them only at the
`_onset_weighted_pitch_classes_in_segment` aggregation layer.

This detector:
  1. Runs Basic Pitch on each provided harmonic stem (guitar, piano).
     Bass is NOT routed through Basic Pitch — the per-bar bass root already
     comes from pyin via processing/bass_root_extraction.py.
  2. Aggregates all notes per bar window (from the grid's downbeat_times).
  3. Computes amplitude×duration weighted pitch-class activation per bar.
  4. Scores each candidate chord quality's interval template against the
     bar's pc weights and selects the best match, with the bass root as the
     anchor.
  5. Emits ChordEvents compatible with the existing pipeline (same dataclass
     as stem_chord_detector).

Config (env, read lazily at call time so .env reloads don't require restart):
  ENABLE_MIDI_DETECTOR       -- "true"/"false", default "false". Gating is
                                done by the caller (transcription.py); this
                                module doesn't read it — callers check.
  MIDI_DETECTOR_ENERGY_FLOOR -- float, default 0.05. A pitch class is
                                considered "strong" in a bar if its weight
                                is at least this fraction of the bar's
                                loudest pc. Below the floor, a pc can still
                                earn a partial-credit bonus when it matches
                                a template interval but is not counted as a
                                spurious extra when it doesn't.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Output types match the existing detectors so the formatter doesn't care
# which detector produced the ChordEvents.
from stem_chord_detector import ChordEvent, ChordProgression

logger = logging.getLogger(__name__)

NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

ROOT_PC: Dict[str, int] = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
}

# (internal_key, intervals_from_root_mod12, display_suffix).
# Ordering is informational; _select_chord picks the best score with a
# tie-break that prefers the richer (longer-interval-list) template.
QUALITY_TEMPLATES: List[Tuple[str, Tuple[int, ...], str]] = [
    ('5',       (0, 7),              '5'),       # power chord
    ('maj',     (0, 4, 7),           ''),
    ('min',     (0, 3, 7),           'm'),
    ('sus2',    (0, 2, 7),           'sus2'),
    ('sus4',    (0, 5, 7),           'sus4'),
    ('aug',     (0, 4, 8),           'aug'),
    ('6',       (0, 4, 7, 9),        '6'),
    ('m6',      (0, 3, 7, 9),        'm6'),
    ('7',       (0, 4, 7, 10),       '7'),
    ('m7',      (0, 3, 7, 10),       'm7'),
    ('maj7',    (0, 4, 7, 11),       'maj7'),
    ('m7b5',    (0, 3, 6, 10),       'm7b5'),
    ('dim7',    (0, 3, 6, 9),        'dim7'),
    ('add9',    (0, 4, 7, 2),        'add9'),    # 9th is +14 ≡ +2 mod 12
    ('9',       (0, 4, 7, 10, 2),    '9'),
    ('maj9',    (0, 4, 7, 11, 2),    'maj9'),
    ('m9',      (0, 3, 7, 10, 2),    'm9'),
]

DEFAULT_ENERGY_FLOOR = 0.05


@dataclass
class MidiNote:
    """A single detected note event from Basic Pitch."""
    start: float       # seconds
    end: float         # seconds
    pitch: int         # MIDI pitch (0-127)
    amplitude: float   # Basic Pitch emits in [0, 1]
    stem: str = ""     # which stem produced it ("guitar", "piano")


@dataclass
class MidiChordDetectorResult:
    """Full detector output, including diagnostics for per-bar inspection."""
    chord_progression: ChordProgression
    per_bar: List[Dict] = field(default_factory=list)
    basic_pitch_ok: bool = True
    notes_by_stem: Dict[str, int] = field(default_factory=dict)


def _resolve_energy_floor(explicit: Optional[float]) -> float:
    """Pick the effective floor: explicit arg > env var > default."""
    if explicit is not None:
        return float(explicit)
    raw = os.environ.get("MIDI_DETECTOR_ENERGY_FLOOR")
    if raw:
        try:
            return float(raw)
        except ValueError:
            logger.warning(
                "MIDI_DETECTOR_ENERGY_FLOOR=%r is not a float, falling back to %s",
                raw, DEFAULT_ENERGY_FLOOR,
            )
    return DEFAULT_ENERGY_FLOOR


def _pc_to_name(pc: int) -> str:
    return NOTE_NAMES[pc % 12]


def _bar_pc_weights(notes: List[MidiNote], t0: float, t1: float) -> List[float]:
    """Amplitude×duration-weighted activation for each pitch class in [t0, t1].

    A note's contribution is `amplitude * overlap_seconds`. Notes that don't
    touch the window contribute nothing. This is deliberately simple — the
    point is to give a stable ranking, not an acoustically-calibrated
    loudness model.
    """
    weights = [0.0] * 12
    if t1 <= t0:
        return weights
    for n in notes:
        overlap = min(n.end, t1) - max(n.start, t0)
        if overlap <= 0:
            continue
        weights[n.pitch % 12] += n.amplitude * overlap
    return weights


def _score_template(
    pc_weights: List[float],
    root_pc: int,
    intervals: Tuple[int, ...],
    energy_floor: float,
) -> Tuple[float, List[int], List[int], float]:
    """Score one template.

    Returns (normalized_score, matched_strong_pcs, extras_strong_pcs,
    matched_strength_sum).

    Scoring:
      +2   for each template interval pc that is "strong" (>= floor of peak)
      +1   for each template interval present-but-quiet (0 < weight < floor)
      -0.5 for each non-template pc that is strong (spurious extra)

    Normalized by max possible score (2 * len(intervals)) so short and long
    templates compete on the same scale.

    `matched_strength_sum` is the sum of the matched pcs' absolute weights —
    used by the caller as a secondary tie-break when two templates have
    indistinguishable normalized scores (e.g. Cm6 vs Cm7 on a bar that
    contains every tone of both).
    """
    wmax = max(pc_weights) if pc_weights else 0.0
    if wmax <= 0:
        return 0.0, [], [], 0.0

    template_pcs = set((root_pc + i) % 12 for i in intervals)
    matched_strong: List[int] = []
    matched_strength_sum = 0.0
    score = 0.0

    for i in intervals:
        pc = (root_pc + i) % 12
        w = pc_weights[pc]
        rel = w / wmax
        if rel >= energy_floor:
            score += 2.0
            matched_strong.append(pc)
            matched_strength_sum += w
        elif w > 0:
            score += 1.0

    extras_strong: List[int] = []
    for pc in range(12):
        if pc in template_pcs:
            continue
        rel = pc_weights[pc] / wmax
        if rel >= energy_floor:
            score -= 0.5
            extras_strong.append(pc)

    max_score = 2.0 * len(intervals)
    normalized = score / max_score if max_score > 0 else 0.0
    return normalized, matched_strong, extras_strong, matched_strength_sum


def _select_chord(
    pc_weights: List[float],
    root_pc: int,
    energy_floor: float,
) -> Tuple[str, str, float, Dict]:
    """Select the highest-scoring quality template for this bar.

    Returns (quality_key, display_suffix, score, diagnostics_dict).

    Tie-break order when normalized scores are within 0.01:
      1. Prefer the template with more intervals (m7 > min when both fit).
      2. Among equal-length templates, prefer the one whose matched tones
         are LOUDER (so Cm7 beats Cm6 when the bar has Bb=strong and A=weak).
    """
    # Collect all candidates, then pick the best via bucketed sort.
    candidates: List[Tuple[float, int, float, str, Tuple[int, ...], str,
                            List[int], List[int]]] = []
    for q_key, intervals, suffix in QUALITY_TEMPLATES:
        score, matched, extras, strength = _score_template(
            pc_weights, root_pc, intervals, energy_floor,
        )
        candidates.append((score, len(intervals), strength, q_key, intervals,
                            suffix, matched, extras))

    if not candidates:  # defensive; QUALITY_TEMPLATES is non-empty
        return 'maj', '', 0.0, {'matched_intervals': [], 'extras_strong_pcs': [],
                                 'template_size': 3}

    # Primary sort by normalized score (desc)
    best_score = max(c[0] for c in candidates)
    # Consider anything within 0.01 of best as tied on primary score.
    top_bucket = [c for c in candidates if c[0] >= best_score - 0.01]
    # Within the tied bucket: longer template wins, then louder matched tones.
    top_bucket.sort(key=lambda c: (-c[1], -c[2]))
    winner = top_bucket[0]

    score, n_intervals, strength, q_key, intervals, suffix, matched, extras = winner
    diag = {
        'matched_intervals': sorted([(pc - root_pc) % 12 for pc in matched]),
        'extras_strong_pcs': [_pc_to_name(pc) for pc in extras],
        'template_size': n_intervals,
        'matched_strength': round(strength, 4),
    }
    return q_key, suffix, score, diag


def _run_basic_pitch_on_stem(
    stem_path: str,
    stem_name: str,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_length_ms: int = 58,
) -> List[MidiNote]:
    """Invoke Basic Pitch on one stem file. Lazy import so the module itself
    is importable under pytest where basic_pitch is mocked.
    """
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    _, _, note_events = predict(
        stem_path,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_length_ms,
    )
    notes: List[MidiNote] = []
    for ev in note_events:
        # Basic Pitch event tuple: (start, end, pitch, amplitude, pitch_bends)
        notes.append(MidiNote(
            start=float(ev[0]),
            end=float(ev[1]),
            pitch=int(ev[2]),
            amplitude=float(ev[3]),
            stem=stem_name,
        ))
    return notes


def _bar_windows_from_grid(grid: Dict) -> List[Tuple[int, float, float]]:
    """Turn grid['downbeat_times'] into [(bar_number, start, end), ...]."""
    downbeats = list((grid or {}).get("downbeat_times") or [])
    if len(downbeats) < 2:
        return []
    song_end = float(
        (grid or {}).get("song_duration_sec")
        or (downbeats[-1] + (downbeats[-1] - downbeats[-2]))
    )
    out: List[Tuple[int, float, float]] = []
    for i, t0 in enumerate(downbeats):
        t1 = downbeats[i + 1] if i + 1 < len(downbeats) else song_end
        out.append((i + 1, float(t0), float(t1)))
    return out


def detect_chords_from_midi(
    guitar_path: Optional[str],
    piano_path: Optional[str],
    bass_path: Optional[str],
    grid: Dict,
    bass_roots: List[Dict],
    min_note_length_ms: int = 58,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    bar_energy_floor: Optional[float] = None,
    _injected_notes: Optional[List[MidiNote]] = None,
) -> MidiChordDetectorResult:
    """Main entry.

    `_injected_notes` is a test seam — callers in production never pass it.
    When provided, Basic Pitch is skipped entirely and the supplied notes
    are used as-if they came from BP. This keeps integration tests
    audio-free.
    """
    floor = _resolve_energy_floor(bar_energy_floor)
    result = MidiChordDetectorResult(
        chord_progression=ChordProgression(chords=[], key='Unknown'),
        basic_pitch_ok=True,
    )

    bar_windows = _bar_windows_from_grid(grid)
    if len(bar_windows) < 4:
        logger.warning("midi_detector: grid has fewer than 4 bars (%d) — aborting",
                       len(bar_windows))
        result.basic_pitch_ok = False
        return result

    # Collect notes from harmonic stems (bass stays out — pyin handles roots).
    all_notes: List[MidiNote] = []
    if _injected_notes is not None:
        all_notes = list(_injected_notes)
        for n in all_notes:
            result.notes_by_stem[n.stem] = result.notes_by_stem.get(n.stem, 0) + 1
    else:
        for stem_name, stem_path in (('guitar', guitar_path), ('piano', piano_path)):
            if not stem_path or not os.path.isfile(stem_path):
                continue
            try:
                notes = _run_basic_pitch_on_stem(
                    stem_path, stem_name,
                    onset_threshold=onset_threshold,
                    frame_threshold=frame_threshold,
                    min_note_length_ms=min_note_length_ms,
                )
            except Exception as e:
                logger.warning("midi_detector: Basic Pitch failed on %s: %s",
                               stem_name, e)
                result.basic_pitch_ok = False
                return result
            result.notes_by_stem[stem_name] = len(notes)
            all_notes.extend(notes)
            logger.info("midi_detector: %s → %d notes", stem_name, len(notes))

    if not all_notes:
        logger.warning("midi_detector: no notes from any harmonic stem")
        result.basic_pitch_ok = False
        return result

    all_notes.sort(key=lambda n: n.start)

    # Per-bar bass root lookup
    bass_root_by_bar: Dict[int, Dict] = {
        b['bar']: b for b in (bass_roots or []) if isinstance(b, dict) and 'bar' in b
    }

    chord_events: List[ChordEvent] = []
    per_bar_diag: List[Dict] = []

    for bar_num, t0, t1 in bar_windows:
        pc_weights = _bar_pc_weights(all_notes, t0, t1)
        wmax = max(pc_weights) if pc_weights else 0.0

        # Silent bar — emit N.C. regardless of bass root.
        if wmax <= 0:
            chord_events.append(ChordEvent(
                time=t0, duration=max(0.0, t1 - t0), chord='N.C.',
                root='N', quality='none', confidence=0.0,
            ))
            per_bar_diag.append(dict(bar=bar_num, start=t0, end=t1,
                                      chord='N.C.', reason='silent'))
            continue

        # Root — prefer bass_roots; fall back to strongest pc.
        br = bass_root_by_bar.get(bar_num)
        if (br and br.get('root') and
                br.get('source') in ('pyin', 'inherit') and
                ROOT_PC.get(br['root']) is not None):
            root_name = br['root']
            root_pc = ROOT_PC[root_name]
            root_source = br.get('source', 'pyin')
        else:
            root_pc = max(range(12), key=lambda pc: pc_weights[pc])
            root_name = _pc_to_name(root_pc)
            root_source = 'dominant_pc_fallback'

        q_key, suffix, score, diag = _select_chord(pc_weights, root_pc, floor)
        chord_name = root_name + suffix
        # Existing detectors use 'maj' when there's no suffix in the name.
        quality = suffix if suffix else 'maj'
        chord_events.append(ChordEvent(
            time=t0,
            duration=max(0.0, t1 - t0),
            chord=chord_name,
            root=root_name,
            quality=quality,
            confidence=max(0.0, min(1.0, score)),
        ))
        per_bar_diag.append(dict(
            bar=bar_num, start=t0, end=t1,
            chord=chord_name, root=root_name, quality=q_key,
            score=round(float(score), 3), root_source=root_source,
            pc_weights={
                _pc_to_name(i): round(pc_weights[i] / (wmax or 1.0), 3)
                for i in range(12)
            },
            **diag,
        ))

    result.chord_progression = ChordProgression(chords=chord_events, key='Unknown')
    result.per_bar = per_bar_diag
    return result
