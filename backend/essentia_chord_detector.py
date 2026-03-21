"""
Essentia-based chord detection for StemScribe.

Provides standalone Essentia ChordsDetection as a complementary detector
to BTC v10. Can be used as:
  - Fallback when BTC confidence is low
  - Ensemble member for majority-vote chord reconciliation
  - Primary detector when BTC model files are missing

Essentia ChordsDetection handles major/minor triads only (no 7ths/extensions),
but is fast (~0.3s), reliable, and runs natively on ARM64.

Output format matches BTC: {chord, time, duration, confidence, root, quality}
"""

import logging
import numpy as np
from typing import List, Optional
from collections import Counter

logger = logging.getLogger(__name__)

# Reuse the shared ChordEvent/ChordProgression from chord_detector_v10
try:
    from chord_detector_v10 import ChordEvent, ChordProgression, _parse_chord
except ImportError:
    # Standalone fallback definitions
    from dataclasses import dataclass

    @dataclass
    class ChordEvent:
        time: float
        duration: float
        chord: str
        root: str
        quality: str
        confidence: float

    @dataclass
    class ChordProgression:
        chords: List[ChordEvent]
        key: str

    def _parse_chord(name: str) -> tuple:
        if not name or name == 'N':
            return ('N', 'none')
        root = name[0]
        rest = name[1:]
        if rest and rest[0] in ('#', 'b'):
            root += rest[0]
            rest = rest[1:]
        return (root, rest if rest else 'maj')


# Check Essentia availability at import time
ESSENTIA_AVAILABLE = False
try:
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    logger.warning("Essentia not available — essentia_chord_detector disabled")


def detect_chords_essentia(audio_path: str, min_duration: float = 0.3,
                           smoothing_window: int = 10) -> ChordProgression:
    """
    Detect chords using Essentia's ChordsDetection algorithm.

    Args:
        audio_path: Path to audio file (WAV, MP3, etc.)
        min_duration: Minimum chord duration in seconds
        smoothing_window: Half-width of majority-vote smoothing window (frames)

    Returns:
        ChordProgression with detected chords and key
    """
    if not ESSENTIA_AVAILABLE:
        logger.error("Essentia not available")
        return ChordProgression(chords=[], key="Unknown")

    try:
        import essentia.standard as es

        # Load audio
        audio = es.MonoLoader(filename=str(audio_path), sampleRate=44100)()
        sr = 44100
        duration_sec = len(audio) / sr
        logger.info(f"Essentia chord detection: {duration_sec:.1f}s audio")

        # Key detection
        key_ext = es.KeyExtractor()
        key_name, scale, key_strength = key_ext(audio)
        detected_key = f"{key_name}m" if scale == "minor" else key_name
        logger.info(f"Essentia key: {detected_key} (strength={key_strength:.2f})")

        # HPCP computation
        frame_size = 8192
        hop = 4096
        w = es.Windowing(type='blackmanharris62')
        spectrum = es.Spectrum()
        peaks = es.SpectralPeaks(
            orderBy='magnitude', magnitudeThreshold=0.00001,
            minFrequency=40, maxFrequency=5000, maxPeaks=60
        )
        hpcp = es.HPCP(
            size=36, referenceFrequency=440, bandPreset=False,
            minFrequency=40, maxFrequency=5000, weightType='cosine'
        )
        chords_algo = es.ChordsDetection(hopSize=hop, sampleRate=sr)

        hpcp_frames = []
        for fstart in range(0, len(audio) - frame_size, hop):
            frame = audio[fstart:fstart + frame_size]
            h = hpcp(*peaks(spectrum(w(frame))))
            hpcp_frames.append(h)

        if not hpcp_frames:
            return ChordProgression(chords=[], key=detected_key)

        hpcp_array = np.array(hpcp_frames)
        chord_list, strength_list = chords_algo(hpcp_array)

        # Majority vote smoothing
        half = smoothing_window
        smoothed = []
        for i in range(len(chord_list)):
            s, e = max(0, i - half), min(len(chord_list), i + half + 1)
            window = [c for c in chord_list[s:e] if c != 'N']
            smoothed.append(
                Counter(window).most_common(1)[0][0] if window else chord_list[i]
            )

        # Map Essentia strength values to confidence
        # Essentia strengths are 0-1, but often cluster around 0.4-0.8
        strength_arr = np.array(strength_list) if len(strength_list) > 0 else np.array([0.5])

        # Consolidate consecutive same-chord frames into events
        events = []
        current = smoothed[0]
        start_idx = 0
        conf_sum = float(strength_arr[0]) if len(strength_arr) > 0 else 0.5
        conf_count = 1

        for i in range(1, len(smoothed)):
            if smoothed[i] != current:
                t0 = start_idx * hop / sr
                dur = (i - start_idx) * hop / sr
                if dur >= min_duration and current != 'N':
                    root, quality = _parse_chord(current)
                    avg_conf = conf_sum / max(conf_count, 1)
                    # Scale Essentia confidence to be comparable with BTC
                    # Essentia strengths are typically 0.3-0.8; map to 0.5-0.9
                    scaled_conf = min(0.95, 0.4 + avg_conf * 0.6)
                    events.append(ChordEvent(
                        time=float(t0),
                        duration=float(dur),
                        chord=current,
                        root=root,
                        quality=quality,
                        confidence=float(scaled_conf),
                    ))
                current = smoothed[i]
                start_idx = i
                conf_sum = float(strength_arr[i]) if i < len(strength_arr) else 0.5
                conf_count = 1
            else:
                conf_sum += float(strength_arr[i]) if i < len(strength_arr) else 0.5
                conf_count += 1

        # Handle last segment
        if current != 'N':
            t0 = start_idx * hop / sr
            dur = duration_sec - t0
            if dur >= min_duration:
                root, quality = _parse_chord(current)
                avg_conf = conf_sum / max(conf_count, 1)
                scaled_conf = min(0.95, 0.4 + avg_conf * 0.6)
                events.append(ChordEvent(
                    time=float(t0),
                    duration=float(dur),
                    chord=current,
                    root=root,
                    quality=quality,
                    confidence=float(scaled_conf),
                ))

        logger.info(f"Essentia detected {len(events)} chord events, key={detected_key}")
        return ChordProgression(chords=events, key=detected_key)

    except Exception as e:
        logger.error(f"Essentia chord detection failed: {e}", exc_info=True)
        return ChordProgression(chords=[], key="Unknown")


def ensemble_chords(btc_chords: List[ChordEvent], essentia_chords: List[ChordEvent],
                    btc_confidence_threshold: float = 0.5,
                    time_tolerance: float = 0.5) -> List[ChordEvent]:
    """
    Combine BTC and Essentia chord detections using confidence-weighted ensemble.

    Strategy:
    - For each BTC chord event, find overlapping Essentia chord(s)
    - If BTC confidence is high (>threshold), keep BTC result
    - If BTC confidence is low, check if Essentia agrees on root:
      - If same root: boost confidence, keep BTC's quality (more detailed)
      - If different root: use Essentia's chord (it's more reliable for roots)
    - Essentia only does major/minor, so BTC's quality labels (7ths, etc.) are preferred
      when BTC confidence is reasonable

    Args:
        btc_chords: Chords from BTC v10
        essentia_chords: Chords from Essentia
        btc_confidence_threshold: Below this, consider Essentia's opinion
        time_tolerance: Max time difference for chord overlap matching

    Returns:
        Merged chord list with improved accuracy
    """
    if not btc_chords:
        return essentia_chords
    if not essentia_chords:
        return btc_chords

    def _find_overlapping(event: ChordEvent, candidates: List[ChordEvent]) -> Optional[ChordEvent]:
        """Find the candidate chord that overlaps most with the given event."""
        best = None
        best_overlap = 0
        ev_start = event.time
        ev_end = event.time + event.duration
        for c in candidates:
            c_start = c.time
            c_end = c.time + c.duration
            overlap = max(0, min(ev_end, c_end) - max(ev_start, c_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best = c
        return best if best_overlap > 0 else None

    merged = []
    for btc_ev in btc_chords:
        ess_ev = _find_overlapping(btc_ev, essentia_chords)

        if btc_ev.confidence >= btc_confidence_threshold:
            # BTC is confident — keep it, but note if Essentia agrees
            if ess_ev and ess_ev.root == btc_ev.root:
                # Both agree on root — slight confidence boost
                boosted_conf = min(0.99, btc_ev.confidence + 0.05)
                merged.append(ChordEvent(
                    time=btc_ev.time,
                    duration=btc_ev.duration,
                    chord=btc_ev.chord,
                    root=btc_ev.root,
                    quality=btc_ev.quality,
                    confidence=boosted_conf,
                ))
            else:
                merged.append(btc_ev)
        else:
            # BTC confidence is low — consider Essentia
            if ess_ev:
                if ess_ev.root == btc_ev.root:
                    # Same root, different confidence — keep BTC chord name (more detail)
                    # but boost confidence since both detectors agree
                    boosted_conf = min(0.95, max(btc_ev.confidence, ess_ev.confidence) + 0.1)
                    merged.append(ChordEvent(
                        time=btc_ev.time,
                        duration=btc_ev.duration,
                        chord=btc_ev.chord,
                        root=btc_ev.root,
                        quality=btc_ev.quality,
                        confidence=boosted_conf,
                    ))
                else:
                    # Disagreement — Essentia is more reliable for root detection
                    # Use Essentia's chord but with moderate confidence
                    merged.append(ChordEvent(
                        time=btc_ev.time,
                        duration=btc_ev.duration,
                        chord=ess_ev.chord,
                        root=ess_ev.root,
                        quality=ess_ev.quality,
                        confidence=float(max(ess_ev.confidence, 0.55)),
                    ))
            else:
                # No Essentia match — keep BTC with reduced confidence
                merged.append(ChordEvent(
                    time=btc_ev.time,
                    duration=btc_ev.duration,
                    chord=btc_ev.chord,
                    root=btc_ev.root,
                    quality=btc_ev.quality,
                    confidence=btc_ev.confidence * 0.8,
                ))

    logger.info(f"Ensemble: {len(btc_chords)} BTC + {len(essentia_chords)} Essentia -> {len(merged)} merged")
    return merged


def get_chord_at_time(chords: list, time: float) -> Optional[dict]:
    """
    Look up what chord is playing at a given time.

    Args:
        chords: List of chord dicts with 'time', 'duration', 'chord', 'root'
        time: Time in seconds to query

    Returns:
        Chord dict if found, None otherwise
    """
    for c in chords:
        c_time = c.get('time', c.time if hasattr(c, 'time') else 0)
        c_dur = c.get('duration', c.duration if hasattr(c, 'duration') else 0)
        if c_time <= time < c_time + c_dur:
            return c
    return None


def chord_to_pitch_classes(chord_name: str) -> set:
    """
    Convert a chord name to a set of pitch classes (0-11, C=0).

    Handles: major, minor, 7th, maj7, min7, sus2, sus4, dim, aug, power chords.

    Returns:
        Set of pitch class integers, or empty set if unknown
    """
    NOTE_MAP = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8,
        'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
    }

    if not chord_name or chord_name == 'N':
        return set()

    # Parse root
    root_str = chord_name[0]
    rest = chord_name[1:]
    if rest and rest[0] in ('#', 'b'):
        root_str += rest[0]
        rest = rest[1:]

    root_pc = NOTE_MAP.get(root_str)
    if root_pc is None:
        return set()

    # Determine intervals from quality
    quality = rest.lower()
    if 'dim7' in quality:
        intervals = [0, 3, 6, 9]
    elif 'dim' in quality:
        intervals = [0, 3, 6]
    elif 'aug' in quality:
        intervals = [0, 4, 8]
    elif 'sus2' in quality:
        intervals = [0, 2, 7]
    elif 'sus4' in quality:
        intervals = [0, 5, 7]
    elif 'mmaj7' in quality or 'mmaj7' in quality:
        intervals = [0, 3, 7, 11]
    elif 'm7' in quality or 'min7' in quality:
        intervals = [0, 3, 7, 10]
    elif 'maj7' in quality:
        intervals = [0, 4, 7, 11]
    elif 'hdim7' in quality:
        intervals = [0, 3, 6, 10]
    elif '7' in quality:
        intervals = [0, 4, 7, 10]
    elif 'm' in quality or 'min' in quality:
        intervals = [0, 3, 7]
    elif '5' in quality:
        intervals = [0, 7]  # Power chord
    elif quality in ('', 'maj', '6', 'add9'):
        intervals = [0, 4, 7]
    else:
        intervals = [0, 4, 7]  # Default to major triad

    return {(root_pc + i) % 12 for i in intervals}
