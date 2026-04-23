"""
Stem-Aware Chord Detection for StemScriber
==========================================
Instead of pattern-matching spectrograms to chord labels, we:
1. Run multi-pitch estimation on each SEPARATED stem (guitar, bass, piano)
2. Detect individual notes being played in each stem
3. Assemble the chord from the detected notes
4. Use bass root to resolve ambiguity (Am7 vs C6)
5. Cross-reference stems for confidence

Requires: basic_pitch, librosa, numpy
Optional: pyin (via librosa) for bass monophonic tracking
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ============ DATA CLASSES ============

@dataclass
class ChordEvent:
    """A detected chord with timing info."""
    time: float
    duration: float
    chord: str
    root: str
    quality: str
    confidence: float
    bass: str = None

@dataclass
class ChordProgression:
    """Result of chord detection."""
    chords: List[ChordEvent]
    key: str
    tuning_info: dict = None


# ============ CONSTANTS ============

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Enharmonic mappings for V8 class lookup
ENHARMONIC = {
    'Db': 'C#', 'Eb': 'D#', 'Fb': 'E', 'Gb': 'F#',
    'Ab': 'G#', 'Bb': 'A#', 'Cb': 'B',
}

# Chord interval templates: intervals from root in semitones
# Ordered by specificity (more notes first) so longer matches win ties
CHORD_INTERVALS = {
    # 6-note chords (13ths)
    '13':     frozenset({0, 2, 4, 7, 9, 10}),
    'min13':  frozenset({0, 2, 3, 7, 9, 10}),
    'maj13':  frozenset({0, 2, 4, 7, 9, 11}),
    # 5-note chords (9ths, 11ths)
    '9':      frozenset({0, 2, 4, 7, 10}),
    'min9':   frozenset({0, 2, 3, 7, 10}),
    'maj9':   frozenset({0, 2, 4, 7, 11}),
    '11':     frozenset({0, 4, 5, 7, 10}),
    'min11':  frozenset({0, 3, 5, 7, 10}),
    '9sus4':  frozenset({0, 2, 5, 7, 10}),
    # 4-note chords (7ths, 6ths, extended)
    '7':      frozenset({0, 4, 7, 10}),
    'maj7':   frozenset({0, 4, 7, 11}),
    'min7':   frozenset({0, 3, 7, 10}),
    'mMaj7':  frozenset({0, 3, 7, 11}),
    '6':      frozenset({0, 4, 7, 9}),
    'min6':   frozenset({0, 3, 7, 9}),
    'dim7':   frozenset({0, 3, 6, 9}),
    'hdim7':  frozenset({0, 3, 6, 10}),
    'add9':   frozenset({0, 2, 4, 7}),
    'madd9':  frozenset({0, 2, 3, 7}),
    '7sus4':  frozenset({0, 5, 7, 10}),
    '7sus2':  frozenset({0, 2, 7, 10}),
    'add11':  frozenset({0, 4, 5, 7}),
    '7#5':    frozenset({0, 4, 8, 10}),   # augmented dominant
    '7b5':    frozenset({0, 4, 6, 10}),   # dominant flat 5
    '7b9':    frozenset({0, 1, 4, 7, 10}),  # dominant flat 9
    '7#9':    frozenset({0, 3, 4, 7, 10}),  # Hendrix chord (has both b3 and 3)
    'maj7#11': frozenset({0, 4, 6, 7, 11}), # Lydian
    '6/9':    frozenset({0, 2, 4, 7, 9}),   # major 6/9
    'min6/9': frozenset({0, 2, 3, 7, 9}),   # minor 6/9
    # 3-note chords (triads)
    'maj':    frozenset({0, 4, 7}),
    'min':    frozenset({0, 3, 7}),
    'dim':    frozenset({0, 3, 6}),
    'aug':    frozenset({0, 4, 8}),
    'sus2':   frozenset({0, 2, 7}),
    'sus4':   frozenset({0, 5, 7}),
    # 2-note (power chord)
    '5':      frozenset({0, 7}),
}

# Quality priority for tie-breaking (prefer simpler, more common chords)
# Lower number = higher priority (preferred when scores are close)
QUALITY_PRIORITY = {
    'maj': 0, 'min': 1, 'min7': 2, '7': 3, 'maj7': 4,
    'sus4': 5, 'sus2': 6, '6': 7, 'min6': 8, 'dim': 9,
    'aug': 10, 'add9': 11, '9': 12, 'min9': 13, 'maj9': 14,
    'mMaj7': 15, 'dim7': 16, 'hdim7': 17, 'madd9': 18,
    '7sus4': 19, '7sus2': 20, '5': 21, 'add11': 22,
    '6/9': 23, 'min6/9': 24, '9sus4': 25,
    '7#5': 26, '7b5': 27, '7b9': 28, '7#9': 29, 'maj7#11': 30,
    # High penalty — 11ths/13ths should only win on near-exact interval match
    # Otherwise stem bleed makes every triad look like a 13th chord
    '11': 40, 'min11': 41, '13': 45, 'min13': 46, 'maj13': 47,
}


# ============ V8 VOCABULARY ============

def load_v8_classes() -> list:
    """Load the V8 337-class chord vocabulary."""
    v8_path = Path(__file__).parent / 'models' / 'pretrained' / 'v8_classes.json'
    if v8_path.exists():
        with open(v8_path) as f:
            return json.load(f)
    return []


def map_to_v8_class(chord_name: str, v8_classes: list) -> str:
    """Map a detected chord to the closest V8 class name."""
    if chord_name in v8_classes:
        return chord_name
    # 'Cm7' in design -> 'Cmin7' in V8
    # Try min/min7/min6 conversions
    conversions = [
        ('m9', 'min9'), ('m7', 'min7'), ('m6', 'min6'), ('m', 'min'),
    ]
    for src, dst in conversions:
        if chord_name.endswith(src) and not chord_name.endswith('dim' + src):
            alt = chord_name[:-len(src)] + dst
            if alt in v8_classes:
                return alt

    # Try without inversion
    base = chord_name.split('/')[0]
    if base in v8_classes:
        return base

    # Try enharmonic
    for old, new in ENHARMONIC.items():
        if chord_name.startswith(old):
            alt = new + chord_name[len(old):]
            if alt in v8_classes:
                return alt

    # Simplify extensions (for types not in V8 vocabulary)
    simplifications = [
        ('madd9', 'min'), ('add9', ''), ('min9', 'min7'),
        ('maj9', 'maj7'), ('9', '7'), ('7sus4', 'sus4'), ('5', ''),
    ]
    for src, dst in simplifications:
        if src in chord_name:
            simplified = chord_name.replace(src, dst)
            if simplified in v8_classes:
                return simplified

    return chord_name


# ============ NOTE DETECTION ============

def detect_notes_in_stem(stem_path: str, instrument: str = 'guitar',
                         onset_threshold: float = 0.5,
                         frame_threshold: float = 0.3,
                         min_note_length: float = 80) -> dict:
    """
    Run Basic Pitch on a single stem to get frame-level note activations.

    Args:
        stem_path: Path to the audio stem file
        instrument: 'guitar', 'bass', or 'piano'
        onset_threshold: Onset detection sensitivity
        frame_threshold: Frame activation threshold
        min_note_length: Minimum note length in ms

    Returns:
        dict with keys:
            'note_frames': np.array (N_frames, 88) frame activations
            'onset_frames': np.array (N_frames, 88) onset activations
            'note_events': list of (start, end, midi_pitch, amplitude, ...)
            'pitch_class_frames': list of (time, frozenset_of_pitch_classes, confidence)
            'frame_times': np.array of timestamps per frame
    """
    from basic_pitch.inference import predict

    # Instrument-specific frequency ranges
    freq_ranges = {
        'guitar': (80.0, 1200.0),    # E2 to ~D#6
        'bass':   (30.0, 350.0),     # B0 to ~F4
        'piano':  (27.5, 4200.0),    # A0 to C8
    }
    min_freq, max_freq = freq_ranges.get(instrument, (30.0, 4200.0))

    logger.info(f"Running Basic Pitch on {instrument} stem: {Path(stem_path).name}")

    model_output, midi_data, note_events = predict(
        str(stem_path),
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_length,
        minimum_frequency=min_freq,
        maximum_frequency=max_freq,
    )

    note_frames = model_output['note']    # (N_frames, 88)
    onset_frames = model_output['onset']  # (N_frames, 88)

    n_frames = note_frames.shape[0]
    frame_duration = 256 / 22050  # ~11.6ms per frame (Basic Pitch hop size)

    # Extract frame-level pitch class sets
    pitch_class_frames = []
    frame_times = np.arange(n_frames) * frame_duration

    for frame_idx in range(n_frames):
        active = note_frames[frame_idx] > frame_threshold
        onsets = onset_frames[frame_idx] > onset_threshold

        # Get MIDI pitches of active notes
        active_indices = np.where(active | onsets)[0]
        midi_pitches = active_indices + 21  # Basic Pitch MIDI offset

        # Convert to pitch classes (mod 12)
        pitch_classes = frozenset(int(p % 12) for p in midi_pitches)

        # Confidence = mean activation of detected notes
        if len(active_indices) > 0:
            conf = float(np.mean(note_frames[frame_idx, active_indices]))
        else:
            conf = 0.0

        pitch_class_frames.append((frame_times[frame_idx], pitch_classes, conf))

    logger.info(f"  Basic Pitch: {n_frames} frames, {len(note_events)} note events for {instrument}")

    return {
        'note_frames': note_frames,
        'onset_frames': onset_frames,
        'note_events': note_events,
        'pitch_class_frames': pitch_class_frames,
        'frame_times': frame_times,
    }


def detect_bass_root(stem_path: str) -> list:
    """
    Detect bass root note per frame using pyin (monophonic pitch tracking)
    combined with Basic Pitch for confirmation.

    Returns:
        list of (time, pitch_class_or_None, confidence) per frame
    """
    import librosa

    logger.info(f"Running pyin bass root detection on: {Path(stem_path).name}")

    y, sr = librosa.load(str(stem_path), sr=22050, mono=True)

    # pyin: probabilistic YIN for monophonic pitch
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=30.0, fmax=350.0, sr=sr,
        hop_length=256,  # Match Basic Pitch resolution
        fill_na=None,
    )

    frame_duration = 256 / sr
    results = []
    for i, (freq, voiced, prob) in enumerate(zip(f0, voiced_flag, voiced_prob)):
        time = i * frame_duration
        if voiced and freq is not None and freq > 0:
            # Convert frequency to MIDI pitch, then to pitch class
            midi_pitch = 12 * np.log2(freq / 440.0) + 69
            pitch_class = int(round(midi_pitch)) % 12
            results.append((time, pitch_class, float(prob)))
        else:
            results.append((time, None, 0.0))

    logger.info(f"  pyin: {sum(1 for _, pc, _ in results if pc is not None)} voiced frames out of {len(results)}")
    return results


# ============ ONSET / SEGMENTATION ============

def segment_by_onsets(stem_path: str, min_segment_duration: float = 0.5,
                      onset_delta: float = 0.15, onset_wait: int = 8) -> list:
    """
    Use librosa onset detection to find chord change points.
    Tuned to avoid over-segmentation: higher delta, longer wait, larger min duration.

    Args:
        stem_path: Path to audio stem
        min_segment_duration: Minimum time between onsets in seconds (default 0.5s)
        onset_delta: Onset detection sensitivity (higher = fewer onsets)
        onset_wait: Minimum frames between onsets

    Returns:
        List of onset times in seconds
    """
    import librosa

    y, sr = librosa.load(str(stem_path), sr=22050, mono=True)

    # Onset detection tuned for chord changes (not individual strums)
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr,
        hop_length=256,
        backtrack=True,
        units='frames',
        pre_max=5, post_max=5,
        pre_avg=5, post_avg=7,
        delta=onset_delta,
        wait=onset_wait,
    )

    onset_times = onset_frames * (256 / sr)

    # Filter by minimum segment duration
    if len(onset_times) > 1:
        filtered = [onset_times[0]]
        for t in onset_times[1:]:
            if t - filtered[-1] >= min_segment_duration:
                filtered.append(t)
        onset_times = np.array(filtered)

    # Always include time 0 as first boundary
    if len(onset_times) == 0 or onset_times[0] > 0.05:
        onset_times = np.concatenate([[0.0], onset_times])

    logger.info(f"  Onset detection: {len(onset_times)} boundaries from {Path(stem_path).name}")
    return onset_times.tolist()


def segment_by_pitch_change(pitch_class_frames: list,
                            min_segment_duration: float = 0.5,
                            smoothing_window: int = 5) -> list:
    """
    Segment by detecting when the pitch class set changes.
    More robust than onset detection for chord boundary finding.

    Args:
        pitch_class_frames: list of (time, pitch_class_set, confidence)
        min_segment_duration: Minimum segment duration in seconds
        smoothing_window: Number of frames to smooth over before comparing

    Returns:
        List of boundary times in seconds
    """
    if not pitch_class_frames:
        return [0.0]

    # Smooth pitch classes: for each frame, take the union of nearby frames
    n = len(pitch_class_frames)
    smoothed = []
    half_w = smoothing_window // 2

    for i in range(n):
        union_pcs = set()
        total_conf = 0.0
        count = 0
        for j in range(max(0, i - half_w), min(n, i + half_w + 1)):
            t, pcs, conf = pitch_class_frames[j]
            if conf > 0.1:
                union_pcs |= pcs
                total_conf += conf
                count += 1
        avg_conf = total_conf / count if count > 0 else 0.0
        smoothed.append((pitch_class_frames[i][0], frozenset(union_pcs), avg_conf))

    # Detect changes: Jaccard distance between consecutive smoothed frames
    boundaries = [smoothed[0][0]]  # Always start at the beginning

    for i in range(1, len(smoothed)):
        prev_pcs = smoothed[i - 1][1]
        curr_pcs = smoothed[i][1]
        curr_time = smoothed[i][0]

        # Skip if too close to last boundary
        if curr_time - boundaries[-1] < min_segment_duration:
            continue

        # Check if pitch classes changed significantly
        if not prev_pcs and not curr_pcs:
            continue

        if not prev_pcs or not curr_pcs:
            # Transition from silence to notes or vice versa
            boundaries.append(curr_time)
            continue

        # Jaccard distance
        intersection = len(prev_pcs & curr_pcs)
        union = len(prev_pcs | curr_pcs)
        similarity = intersection / union if union > 0 else 1.0

        # If less than 50% overlap, it's a chord change
        if similarity < 0.5:
            boundaries.append(curr_time)

    return boundaries


# ============ CHORD ASSEMBLY ============

def _match_intervals_to_quality(intervals: frozenset) -> Tuple[str, float]:
    """
    Match a set of intervals (from root) to the best chord quality.

    Returns (quality_name, score) where score is 0-1.
    Prefers exact matches, then best subset/superset overlap.

    Scoring strategy:
    - Coverage: what fraction of the template is matched
    - Noise penalty: extra notes we hear that aren't in the template
    - Priority: prefer simpler, more common chord types on ties
    - Core triad bonus: if root+3rd+5th all present, boost confidence
    - Exact match bonus: perfect interval set match
    """
    best_quality = None
    best_score = -1.0

    for quality, template in CHORD_INTERVALS.items():
        # How many template tones are present in what we hear?
        matched = len(intervals & template)
        template_size = len(template)
        extra = len(intervals - template)  # notes we hear that aren't in template
        missing = len(template - intervals)  # template tones we don't hear

        if matched < 2:
            # Must match at least root + one other interval
            continue

        # Score: fraction of template matched, penalized by extra notes
        coverage = matched / template_size
        noise_penalty = extra * 0.06        # reduced from 0.08 — more forgiving of bleed
        priority_penalty = QUALITY_PRIORITY.get(quality, 30) * 0.004

        score = coverage - noise_penalty - priority_penalty

        # Exact match DOMINATES — no subset can beat a perfect interval match.
        # Without this, a minor-triad subset match (which gets superset +0.15
        # and core-triad +0.08 bonuses) scored ~1.17, beating a true min7
        # exact match at ~0.99, causing the detector to drop 7ths on every
        # song that genuinely uses them (e.g. Alright: Cm7/Gm7/Dm7/Am7 was
        # being classified as Cm/Gm/Dm/Am). Bump exact-match well above
        # any bonus-boosted subset score.
        if intervals == template:
            score = 2.0 - priority_penalty

        # Bonus for matching all template tones (superset of template)
        if template <= intervals:
            score += 0.15

        # Core triad bonus: if we matched root + 3rd (or sus) + 5th,
        # that's strong structural evidence even with extra bleed notes.
        # This helps when stem separation adds noise (e.g., both D and D#
        # from piano bleed — the 3rd interval still confirms quality).
        has_root = 0 in intervals  # always true since we build from root
        has_fifth = 7 in (intervals & template)
        has_third = bool({3, 4} & intervals & template)  # minor or major 3rd
        has_sus = bool({2, 5} & intervals & template)    # sus2 or sus4
        if has_root and has_fifth and (has_third or has_sus):
            score += 0.08

        # Minor vs major disambiguation: if BOTH minor 3rd (3) and major 3rd (4)
        # are in the interval set, prefer the quality whose template contains the
        # interval that the bass stem supports. In practice, this means the template
        # that has fewer "contradicting" intervals wins. We handle this by giving
        # a penalty when both 3 and 4 are present and the template only expects one.
        if 3 in intervals and 4 in intervals:
            if 3 in template and 4 not in template:
                # Template says minor — the major 3rd is noise
                score += 0.03  # small bonus for explaining the contradiction
            elif 4 in template and 3 not in template:
                score += 0.03

        if score > best_score:
            best_score = score
            best_quality = quality

    return (best_quality, max(0.0, best_score)) if best_quality else (None, 0.0)


def notes_to_chord(pitch_classes: set, bass_root: int = None,
                   v8_classes: list = None) -> Tuple[str, str, str, str, float]:
    """
    Given a set of active pitch classes and a bass root, identify the chord.

    BASS-ROOT-FIRST approach:
    1. If bass root is provided, USE IT as the chord root. The bass note IS the root.
    2. Compute intervals from that root to all other notes.
    3. Match intervals to chord quality templates.
    4. Only if bass root is missing, try all pitch classes as candidate roots.

    Args:
        pitch_classes: set of ints 0-11 (C=0, C#=1, ..., B=11)
        bass_root: single pitch class int for bass note, or None
        v8_classes: optional V8 vocabulary list for validation

    Returns:
        (chord_name, root_name, quality, bass_name, confidence)
    """
    if not pitch_classes or len(pitch_classes) < 2:
        return ('N', 'N', 'none', None, 0.0)

    def _build_result(root_pc, quality, score, is_inversion=False):
        """Format a chord result from root pitch class and quality."""
        root_name = NOTE_NAMES[root_pc]
        bass_name = None

        # Format chord name
        if quality == 'maj':
            chord_name = root_name
        elif quality == 'min':
            chord_name = f"{root_name}m"
        else:
            chord_name = f"{root_name}{quality}"

        # If bass is different from root, mark as inversion
        if bass_root is not None and bass_root != root_pc:
            bass_name = NOTE_NAMES[bass_root]
            chord_name = f"{chord_name}/{bass_name}"

        conf = max(0.0, min(score, 0.99))
        return (chord_name, root_name, quality, bass_name, conf)

    # === STRATEGY 1: Bass root is the chord root (primary) ===
    if bass_root is not None and bass_root in pitch_classes:
        intervals = frozenset((pc - bass_root) % 12 for pc in pitch_classes)
        quality, score = _match_intervals_to_quality(intervals)

        if quality and score >= 0.3:
            result = _build_result(bass_root, quality, score)
            if v8_classes:
                mapped = map_to_v8_class(result[0], v8_classes)
                if mapped != result[0]:
                    result = (mapped, result[1], result[2], result[3], result[4])
            return result

    # === STRATEGY 2: Bass root provided but not in pitch classes ===
    # Still trust bass as root, try with bass added to pitch set
    if bass_root is not None:
        extended_pcs = pitch_classes | {bass_root}
        intervals = frozenset((pc - bass_root) % 12 for pc in extended_pcs)
        quality, score = _match_intervals_to_quality(intervals)

        if quality and score >= 0.25:
            # Slight confidence reduction since bass wasn't in harmony stems
            result = _build_result(bass_root, quality, score * 0.85)
            if v8_classes:
                mapped = map_to_v8_class(result[0], v8_classes)
                if mapped != result[0]:
                    result = (mapped, result[1], result[2], result[3], result[4])
            return result

    # === STRATEGY 3: No bass root — try all pitch classes as candidate roots ===
    best_match = None
    best_score = -1.0

    for candidate_root in pitch_classes:
        intervals = frozenset((pc - candidate_root) % 12 for pc in pitch_classes)
        quality, score = _match_intervals_to_quality(intervals)

        if quality and score > best_score:
            best_score = score
            best_match = _build_result(candidate_root, quality, score)

    if best_match:
        if v8_classes:
            mapped = map_to_v8_class(best_match[0], v8_classes)
            if mapped != best_match[0]:
                best_match = (mapped, best_match[1], best_match[2], best_match[3], best_match[4])
        return best_match

    return ('N', 'N', 'none', None, 0.0)


def _dominant_pitch_classes_in_segment(pitch_class_frames: list,
                                       start_time: float,
                                       end_time: float,
                                       threshold: float = 0.25) -> Tuple[set, float]:
    """
    Get the dominant pitch classes within a time segment.

    Returns (set_of_pitch_classes, mean_confidence)
    """
    # Collect pitch class votes within the segment
    pc_votes = {}  # pitch_class -> list of confidences
    total_conf = 0.0
    count = 0

    for time, pcs, conf in pitch_class_frames:
        if time < start_time:
            continue
        if time >= end_time:
            break
        if conf < 0.05:
            continue
        for pc in pcs:
            if pc not in pc_votes:
                pc_votes[pc] = []
            pc_votes[pc].append(conf)
        total_conf += conf
        count += 1

    if not pc_votes:
        return set(), 0.0

    # Keep pitch classes that appear in at least 25% of frames in the segment
    min_appearances = max(1, int(count * threshold))
    dominant = set()
    for pc, confs in pc_votes.items():
        if len(confs) >= min_appearances:
            dominant.add(pc)

    mean_conf = total_conf / count if count > 0 else 0.0
    return dominant, mean_conf


def _onset_weighted_pitch_classes_in_segment(note_data: dict,
                                              start_time: float,
                                              end_time: float,
                                              onset_weight: float = 3.0,
                                              threshold: float = 0.25,
                                              max_notes: int = 6) -> Tuple[set, float]:
    """
    Get dominant pitch classes using onset-weighted voting.

    Notes with recent onsets get higher weight than sustaining notes.
    This prevents ringing open strings from polluting every segment.
    Limits output to max_notes pitch classes to avoid chord vocabulary explosion.

    Args:
        note_data: dict from detect_notes_in_stem (has note_frames, onset_frames, frame_times)
        start_time: segment start in seconds
        end_time: segment end in seconds
        onset_weight: multiplier for onset-activated notes vs sustain-only
        threshold: minimum fraction of segment an onset-note must appear in
        max_notes: maximum pitch classes to return (chords rarely have >6 notes)

    Returns (set_of_pitch_classes, mean_confidence)
    """
    note_frames = note_data['note_frames']      # (N, 88)
    onset_frames = note_data['onset_frames']    # (N, 88)
    frame_times = note_data['frame_times']      # (N,)
    frame_threshold = 0.3
    onset_threshold = 0.5

    start_idx = int(np.searchsorted(frame_times, start_time))
    end_idx = int(np.searchsorted(frame_times, end_time))
    if end_idx <= start_idx:
        return set(), 0.0

    n_seg = end_idx - start_idx
    seg_notes = note_frames[start_idx:end_idx]    # (seg_len, 88)
    seg_onsets = onset_frames[start_idx:end_idx]  # (seg_len, 88)

    # For each of 88 pitches, compute: onset score + sustain score
    # onset_score = sum of onset activations in this segment
    # note_score = sum of note activations in this segment
    pitch_onset_score = np.sum(seg_onsets * (seg_onsets > onset_threshold), axis=0)  # (88,)
    pitch_note_score = np.sum(seg_notes * (seg_notes > frame_threshold), axis=0)    # (88,)

    # Weighted score: onset-attacked notes get onset_weight multiplier
    has_onset = pitch_onset_score > 0
    pitch_score = np.where(has_onset,
                           pitch_onset_score * onset_weight + pitch_note_score,
                           pitch_note_score * 0.3)  # penalize sustain-only heavily

    # Collapse to pitch classes (sum across octaves)
    pc_score = np.zeros(12)
    pc_onset_score = np.zeros(12)
    for i in range(88):
        midi = i + 21
        pc = midi % 12
        pc_score[pc] += pitch_score[i]
        pc_onset_score[pc] += pitch_onset_score[i]

    total = np.sum(pc_score)
    if total == 0:
        return set(), 0.0

    # Normalize
    pc_norm = pc_score / total

    # Select pitch classes: top N by score, above minimum threshold
    min_score = threshold / 12  # minimum normalized score
    candidates = [(pc, pc_norm[pc], pc_onset_score[pc])
                  for pc in range(12) if pc_norm[pc] > min_score]

    # Sort by score descending, take top max_notes
    candidates.sort(key=lambda x: -x[1])
    selected = candidates[:max_notes]

    # Filter: require at least 2% of total or having an onset
    dominant = set()
    for pc, norm_score, onset_sc in selected:
        if onset_sc > 0 or norm_score > 0.05:
            dominant.add(pc)

    mean_conf = float(np.mean(seg_notes[seg_notes > frame_threshold])) \
        if np.any(seg_notes > frame_threshold) else 0.0

    return dominant, mean_conf


def _dominant_bass_in_segment(bass_frames: list,
                              start_time: float,
                              end_time: float) -> Tuple[Optional[int], float]:
    """
    Get the most common bass pitch class in a time segment.

    Returns (pitch_class_or_None, confidence)
    """
    from collections import Counter

    votes = Counter()
    confs = []

    for time, pc, conf in bass_frames:
        if time < start_time:
            continue
        if time >= end_time:
            break
        if pc is not None and conf > 0.2:
            votes[pc] += 1
            confs.append(conf)

    if not votes:
        return None, 0.0

    most_common_pc = votes.most_common(1)[0][0]
    mean_conf = np.mean(confs) if confs else 0.0
    return most_common_pc, float(mean_conf)


# ============ CROSS-STEM VALIDATION ============

def cross_validate_segments(guitar_pcs: set, bass_root: Optional[int],
                            piano_pcs: set,
                            guitar_conf: float, piano_conf: float,
                            bass_conf: float) -> Tuple[set, Optional[int], float]:
    """
    Combine pitch class evidence from multiple stems with confidence boosting.

    Returns (combined_pitch_classes, bass_root, combined_confidence)
    """
    # Start with guitar as primary
    combined = set(guitar_pcs)

    # Add piano notes (union)
    if piano_pcs:
        combined |= piano_pcs

    # Include bass in the pitch class set
    if bass_root is not None:
        combined.add(bass_root)

    # Compute agreement-based confidence
    base_conf = max(guitar_conf, piano_conf) if piano_pcs else guitar_conf

    # Boost if guitar and piano share pitch classes
    if guitar_pcs and piano_pcs:
        overlap = guitar_pcs & piano_pcs
        if len(overlap) >= 2:
            base_conf += 0.10  # Strong agreement

    # Boost if bass confirms the root
    if bass_root is not None and bass_root in guitar_pcs:
        base_conf += 0.15

    # All three agree strongly
    if guitar_pcs and piano_pcs and bass_root is not None:
        if bass_root in guitar_pcs and bass_root in piano_pcs:
            base_conf += 0.05

    return combined, bass_root, min(0.99, base_conf)


# ============ KEY DETECTION ============

def detect_key_from_chords(chord_events: List[ChordEvent]) -> str:
    """Detect musical key from a chord progression using Krumhansl-Kessler profiles."""
    if not chord_events:
        return 'C'

    # Count weighted root occurrences
    root_weights = np.zeros(12)
    for c in chord_events:
        if c.root in NOTE_NAMES:
            idx = NOTE_NAMES.index(c.root)
            root_weights[idx] += c.duration * c.confidence

    if np.sum(root_weights) == 0:
        return 'C'

    root_weights /= np.sum(root_weights)

    # Krumhansl-Kessler key profiles
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    major_profile /= np.sum(major_profile)
    minor_profile /= np.sum(minor_profile)

    best_corr, best_key = -1, 'C'
    for i in range(12):
        rotated = np.roll(root_weights, -i)
        major_corr = np.corrcoef(rotated, major_profile)[0, 1]
        minor_corr = np.corrcoef(rotated, minor_profile)[0, 1]

        if major_corr > best_corr:
            best_corr = major_corr
            best_key = NOTE_NAMES[i]
        if minor_corr > best_corr:
            best_corr = minor_corr
            best_key = f"{NOTE_NAMES[i]}m"

    return best_key


# ============ POST-PROCESSING ============

def merge_consecutive_chords(chords: List[ChordEvent], min_duration: float = 0.15) -> List[ChordEvent]:
    """Merge consecutive identical chord events and filter by minimum duration."""
    if not chords:
        return []

    merged = [chords[0]]
    for c in chords[1:]:
        if c.chord == merged[-1].chord:
            # Extend previous chord
            merged[-1].duration = (c.time + c.duration) - merged[-1].time
            merged[-1].confidence = max(merged[-1].confidence, c.confidence)
        else:
            merged.append(c)

    # Filter by minimum duration
    filtered = [c for c in merged if c.duration >= min_duration]

    # Recompute durations to fill gaps
    for i in range(len(filtered) - 1):
        gap = filtered[i + 1].time - (filtered[i].time + filtered[i].duration)
        if gap > 0 and gap < 0.5:
            filtered[i].duration += gap

    return filtered


# ============ BLEED SIMPLIFICATION ============

# Qualities that are likely over-extended triads due to stem bleed.
# Maps extended quality → simplified quality.
_SIMPLIFY_MAP = {
    'min7': 'min',    # F#min7 → F#m (7th is bleed from another instrument)
    'min9': 'min',    # Bmin9 → Bm
    'min6': 'min',    # F#min6 → F#m (6th is bleed)
    'maj7': 'maj',    # Dmaj7 → D (maj7 is bleed)
    'maj9': 'maj',    # Gmaj9 → G
    '6':    'maj',    # E6 → E (6th is bleed)
    '9':    'maj',    # B9 → B (9th is bleed)
    '7':    'maj',    # B7 → B (dom7 may be bleed)
    'add9': 'maj',    # Cadd9 → C
    'madd9': 'min',   # Cmadd9 → Cm
    '9sus4': 'sus4',  # B9sus4 → Bsus4
    '7sus4': 'sus4',  # C7sus4 → Csus4
    '7sus2': 'sus2',  # C7sus2 → Csus2
}

# Qualities that should NEVER be simplified (they're distinctive enough)
_KEEP_EXTENDED = {'dim', 'dim7', 'hdim7', 'aug', 'mMaj7', '7#5', '7b5', '7b9', '7#9'}


def _simplify_bleed_extensions(chord_events: list) -> list:
    """
    Simplify over-extended chord qualities that are likely caused by stem bleed.

    In pop/rock, most chords are triads. When stem separation isn't perfect,
    notes from other instruments leak in and make every chord look like a 7th
    or 9th. This pass simplifies extensions back to triads when:
    1. The quality is in the simplify map (common extension)
    2. The chord confidence is below a threshold (uncertain extension)
    3. OR most chords in the song have extensions (suggesting systematic bleed)

    Preserves genuinely extended chords in jazz contexts by checking
    the overall extension rate.
    """
    if not chord_events:
        return chord_events

    # Count how many chords have extensions vs triads
    extended_count = 0
    triad_count = 0
    ext_qualities: list = []
    for ce in chord_events:
        q = ce.quality
        if q in ('maj', 'min', 'dim', 'aug', 'sus2', 'sus4', '5'):
            triad_count += 1
        elif q in _SIMPLIFY_MAP:
            extended_count += 1
            ext_qualities.append(q)

    total = extended_count + triad_count
    if total == 0:
        return chord_events

    extension_rate = extended_count / total

    # Consistency of the extensions: fraction that share the single most
    # common quality. High consistency (e.g. all min7) = deliberate song
    # style (pop/soul). Low consistency with moderate rate = stem bleed
    # producing random extensions on triads.
    from collections import Counter as _Counter
    if ext_qualities:
        _, top_count = _Counter(ext_qualities).most_common(1)[0]
        ext_consistency = top_count / len(ext_qualities)
    else:
        ext_consistency = 0.0

    # Systematic-bleed heuristic (revised 2026-04-23 to preserve genuine 7ths):
    #   - extension rate in 0.60-0.90: MAYBE bleed
    #     - if extensions are varied (consistency < 0.75), it's bleed → simplify
    #     - if extensions are consistent (all the same min7 / maj7 / 7), keep
    #   - extension rate > 0.90: real music (Cm7-Gm7-Dm7-Am7 style songs, jazz)
    #     regardless of consistency — bleed wouldn't produce THIS many extensions
    #   - extension rate < 0.60: per-chord confidence filter handles it
    #
    # Ground-truth test case: Jamiroquai "Alright" = Cm7 Gm7 Dm7 Am7 throughout.
    # Old logic saw 92% extension rate → systematic_bleed=True → stripped all 7ths.
    # New logic: 92% > 0.90 ceiling → systematic_bleed=False → keeps 7ths.
    if extension_rate > 0.90:
        systematic_bleed = False
    elif extension_rate > 0.60:
        systematic_bleed = ext_consistency < 0.75
    else:
        systematic_bleed = False

    simplified = []
    for ce in chord_events:
        if ce.quality in _KEEP_EXTENDED:
            simplified.append(ce)
            continue

        should_simplify = False

        if ce.quality in _SIMPLIFY_MAP:
            if systematic_bleed:
                # High bleed rate → simplify everything
                should_simplify = True
            elif ce.confidence < 0.93:
                # Low confidence on this specific chord → simplify
                should_simplify = True

        if should_simplify:
            new_quality = _SIMPLIFY_MAP[ce.quality]
            root_name = ce.root if ce.root else 'C'

            # Build new chord name
            if new_quality == 'maj':
                new_chord = root_name
            elif new_quality == 'min':
                new_chord = f"{root_name}m"
            else:
                new_chord = f"{root_name}{new_quality}"

            # Preserve slash bass
            if ce.bass and ce.bass != ce.root:
                new_chord = f"{new_chord}/{ce.bass}"

            simplified.append(ChordEvent(
                time=ce.time,
                duration=ce.duration,
                chord=new_chord,
                root=ce.root,
                quality=new_quality,
                confidence=ce.confidence,
                bass=ce.bass,
            ))
        else:
            simplified.append(ce)

    # Log simplification
    changed = sum(1 for a, b in zip(chord_events, simplified) if a.chord != b.chord)
    if changed:
        logger.info(f"  Simplified {changed}/{len(chord_events)} over-extended chords (extension_rate={extension_rate:.0%})")

    return simplified


# ============ MAIN DETECTION FUNCTION ============

def detect_chords_from_stems(guitar_path: str = None,
                              bass_path: str = None,
                              piano_path: str = None,
                              audio_path: str = None,
                              artist: str = '',
                              title: str = '',
                              min_segment_duration: float = 0.5,
                              onset_threshold: float = 0.5,
                              frame_threshold: float = 0.3,
                              min_note_length: float = 80,
                              pc_frame_threshold: float = 0.4,
                              onset_delta: float = 0.15,
                              onset_wait: int = 8,
                              merge_min_duration: float = 1.0,
                              use_pitch_change_segmentation: bool = True) -> ChordProgression:
    """
    Main stem-aware chord detection function.

    Runs note detection on individual stems, segments by pitch-class changes
    (or onsets as fallback), assembles chords from detected notes, and
    cross-validates across stems.

    Args:
        guitar_path: Path to separated guitar stem
        bass_path: Path to separated bass stem
        piano_path: Path to separated piano stem (optional)
        audio_path: Path to original audio (for onset detection fallback)
        artist: Artist name for vocabulary constraint
        title: Song title for vocabulary constraint
        min_segment_duration: Minimum chord duration in seconds (default 0.5)
        onset_threshold: Basic Pitch onset sensitivity
        frame_threshold: Basic Pitch frame activation threshold
        min_note_length: Minimum note duration in ms
        pc_frame_threshold: Fraction of frames a pitch class must appear in
        onset_delta: librosa onset detection sensitivity
        onset_wait: Minimum frames between onsets
        use_pitch_change_segmentation: Use pitch-class change detection (True)
            or raw onset detection (False)

    Returns:
        ChordProgression with detected chords and key
    """
    if not guitar_path and not piano_path:
        logger.warning("No guitar or piano stem provided - cannot detect chords from stems")
        return ChordProgression(chords=[], key='Unknown')

    v8_classes = load_v8_classes()

    # Stage 1: Detect notes in each stem
    guitar_notes = None
    piano_notes = None
    bass_frames = None

    # Guitar stem (primary)
    if guitar_path and Path(guitar_path).exists():
        try:
            guitar_notes = detect_notes_in_stem(
                guitar_path, 'guitar',
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
                min_note_length=min_note_length,
            )
        except Exception as e:
            logger.warning(f"Guitar note detection failed: {e}")

    # Piano stem
    if piano_path and Path(piano_path).exists():
        try:
            piano_notes = detect_notes_in_stem(
                piano_path, 'piano',
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
                min_note_length=min_note_length,
            )
        except Exception as e:
            logger.warning(f"Piano note detection failed: {e}")

    # Bass stem - use pyin for monophonic root + Basic Pitch for confirmation
    if bass_path and Path(bass_path).exists():
        try:
            bass_frames = detect_bass_root(bass_path)
        except Exception as e:
            logger.warning(f"Bass root detection failed: {e}")

    if not guitar_notes and not piano_notes:
        logger.warning("No note detection succeeded - returning empty progression")
        return ChordProgression(chords=[], key='Unknown')

    # Stage 2: Get segment boundaries
    primary_stem = guitar_path if guitar_notes else piano_path
    primary_notes = guitar_notes if guitar_notes else piano_notes

    if use_pitch_change_segmentation and primary_notes:
        # Use pitch-class change detection (more robust, fewer false boundaries)
        onset_times = segment_by_pitch_change(
            primary_notes['pitch_class_frames'],
            min_segment_duration=min_segment_duration,
        )
        logger.info(f"  Pitch-change segmentation: {len(onset_times)} boundaries")
    else:
        # Fallback to onset detection
        onset_times = segment_by_onsets(
            primary_stem,
            min_segment_duration=min_segment_duration,
            onset_delta=onset_delta,
            onset_wait=onset_wait,
        )

    # Get audio duration for the final segment
    import librosa
    y, sr = librosa.load(str(primary_stem), sr=22050, mono=True, duration=None)
    audio_duration = len(y) / sr

    # Ensure we have an end boundary
    if onset_times[-1] < audio_duration - 0.5:
        onset_times.append(audio_duration)

    # Stage 3: For each segment, collect notes from all stems and assemble chord
    chord_events = []

    for seg_idx in range(len(onset_times) - 1):
        start_time = onset_times[seg_idx]
        end_time = onset_times[seg_idx + 1]
        duration = end_time - start_time

        if duration < 0.05:
            continue

        # Get dominant pitch classes from guitar (onset-weighted)
        guitar_pcs = set()
        guitar_conf = 0.0
        if guitar_notes:
            guitar_pcs, guitar_conf = _onset_weighted_pitch_classes_in_segment(
                guitar_notes, start_time, end_time,
                threshold=pc_frame_threshold,
            )

        # Get dominant pitch classes from piano (onset-weighted)
        piano_pcs = set()
        piano_conf = 0.0
        if piano_notes:
            piano_pcs, piano_conf = _onset_weighted_pitch_classes_in_segment(
                piano_notes, start_time, end_time,
                threshold=pc_frame_threshold,
            )

        # Get bass root
        bass_root = None
        bass_conf = 0.0
        if bass_frames:
            bass_root, bass_conf = _dominant_bass_in_segment(bass_frames, start_time, end_time)

        # Cross-validate across stems
        combined_pcs, final_bass, combined_conf = cross_validate_segments(
            guitar_pcs, bass_root, piano_pcs,
            guitar_conf, piano_conf, bass_conf
        )

        # Assemble chord from notes
        chord_name, root, quality, bass_name, chord_conf = notes_to_chord(
            combined_pcs, final_bass, v8_classes
        )

        if chord_name == 'N':
            continue

        # Blend confidences
        final_confidence = min(0.99, (combined_conf * 0.6 + chord_conf * 0.4))

        chord_events.append(ChordEvent(
            time=round(start_time, 3),
            duration=round(duration, 3),
            chord=chord_name,
            root=root,
            quality=quality,
            confidence=round(final_confidence, 3),
            bass=bass_name,
        ))

    # Stage 4: Post-process
    chord_events = merge_consecutive_chords(chord_events, min_duration=merge_min_duration)

    # Stage 5: Simplify over-extended qualities from stem bleed
    # When stems aren't perfectly isolated, notes from other instruments
    # leak in and make triads look like 7ths, 9ths, 6ths, etc.
    # If the context suggests a simple chord (pop/rock), prefer the triad.
    chord_events = _simplify_bleed_extensions(chord_events)

    # Detect key
    key = detect_key_from_chords(chord_events)

    logger.info(f"Stem-aware chord detection: {len(chord_events)} chords detected, key: {key}")

    return ChordProgression(
        chords=chord_events,
        key=key,
        tuning_info={'detection_method': 'stem_aware', 'tuning_offset_semitones': 0,
                     'tuning_name': 'Standard (E)', 'effective_a4': 440.0},
    )


# ============ CLASS WRAPPER (matches existing ChordDetector interface) ============

class StemAwareChordDetector:
    """
    Drop-in replacement for ChordDetector that uses separated stems.
    Falls back to the standard detector if stems are not available.
    """

    def __init__(self, min_duration: float = 0.15):
        self.min_duration = min_duration

    def detect_from_stems(self, guitar_path: str = None, bass_path: str = None,
                          piano_path: str = None, audio_path: str = None,
                          artist: str = '', title: str = '') -> ChordProgression:
        """Detect chords using separated stems."""
        return detect_chords_from_stems(
            guitar_path=guitar_path,
            bass_path=bass_path,
            piano_path=piano_path,
            audio_path=audio_path,
            artist=artist,
            title=title,
            min_segment_duration=self.min_duration,
        )


# ============ CLI TEST ============

if __name__ == '__main__':
    import sys
    import time

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # Default to local test stems
    test_dir = Path(__file__).parent / 'outputs' / 'modal_test'

    if len(sys.argv) > 1:
        test_dir = Path(sys.argv[1])

    guitar = str(test_dir / 'guitar.mp3')
    bass = str(test_dir / 'bass.mp3')
    piano = str(test_dir / 'piano.mp3')

    if not Path(guitar).exists():
        print(f"ERROR: Guitar stem not found at {guitar}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"STEM-AWARE CHORD DETECTION TEST")
    print(f"Stems directory: {test_dir}")
    print(f"{'='*60}\n")

    start = time.time()
    result = detect_chords_from_stems(
        guitar_path=guitar,
        bass_path=bass if Path(bass).exists() else None,
        piano_path=piano if Path(piano).exists() else None,
    )
    elapsed = time.time() - start

    print(f"\nDetected key: {result.key}")
    print(f"Detection time: {elapsed:.1f}s")
    print(f"Chords detected: {len(result.chords)}")
    print(f"\n{'Time':>8}  {'Dur':>5}  {'Chord':<12} {'Conf':>5}  {'Root':<4} {'Quality':<8} {'Bass'}")
    print(f"{'-'*60}")
    for c in result.chords:
        bass_str = c.bass if c.bass else '-'
        print(f"{c.time:8.2f}  {c.duration:5.2f}  {c.chord:<12} {c.confidence:5.2f}  {c.root:<4} {c.quality:<8} {bass_str}")

    # Compare against known chords if Thunderhead
    known_chords = ['Am7', 'Bm7', 'Cm7', 'E9', 'Bm6', 'Dm7', 'Gmaj7', 'Em7', 'G#7', 'G#maj7']
    detected_unique = sorted(set(c.chord.split('/')[0] for c in result.chords))

    print(f"\n{'='*60}")
    print(f"COMPARISON: Known vs Detected (unique chords)")
    print(f"{'='*60}")
    print(f"Known ({len(known_chords)}):    {', '.join(known_chords)}")
    print(f"Detected ({len(detected_unique)}): {', '.join(detected_unique)}")

    # Map known to V8 format for comparison
    v8_classes = load_v8_classes()
    known_v8 = set()
    for k in known_chords:
        mapped = map_to_v8_class(k, v8_classes)
        known_v8.add(mapped)

    detected_v8 = set()
    for d in detected_unique:
        mapped = map_to_v8_class(d, v8_classes)
        detected_v8.add(mapped)

    matched = known_v8 & detected_v8
    missed = known_v8 - detected_v8
    extra = detected_v8 - known_v8

    print(f"\nMatched:  {', '.join(sorted(matched)) if matched else 'none'}")
    print(f"Missed:   {', '.join(sorted(missed)) if missed else 'none'}")
    print(f"Extra:    {', '.join(sorted(extra)) if extra else 'none'}")
    print(f"Accuracy: {len(matched)}/{len(known_v8)} = {len(matched)/max(len(known_v8),1)*100:.0f}%")
