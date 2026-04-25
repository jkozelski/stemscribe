"""
Bass-root extraction from a bass stem, anchored to the bar grid.

For each bar in the provided tempo grid, extract the dominant pitch from the
bass stem and convert it to a pitch class (C, C#, D, ..., B). This gives a
high-confidence chord ROOT per bar.

Why this exists: the polyphonic chord detector infers the root from a dense
mix and frequently confuses roots in keyboard-heavy songs (Alright's verse
detected as Cm-Gm-Dm-Cm instead of Cm-Gm-Dm-Am). The bass stem is typically
monophonic, so pitch detection via librosa.pyin is very accurate. Combine
bass-anchored ROOTS with detector-inferred QUALITY (maj/min/7) for the best
chord-per-bar estimate.

Output aligned to `tempo_beats.extract_grid` so everything downstream shares
one notion of "bar 1, bar 2, bar 3."
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Electric/upright bass range. Fundamental of low E1 is ~41 Hz.
# Upper bound is generous to catch higher basslines (e.g. slap above G3).
BASS_FMIN = 40.0
BASS_FMAX = 400.0

# Minimum ratio of voiced frames within a bar to trust the pitch estimate.
# Below this, the bar is mostly silence or unvoiced — inherit the previous root.
MIN_VOICED_RATIO = 0.3

# Pyin frame parameters
FRAME_LENGTH = 2048
HOP_LENGTH = 512


def extract_bass_roots(
    bass_path: str,
    grid: Dict,
    sample_rate: int = 22050,
) -> List[Dict]:
    """
    Extract one bass root per bar.

    Returns a list, one entry per bar where we could estimate (or inherit) a root:
        [{"bar": 1, "root": "C", "confidence": 0.87, "source": "pyin"},
         {"bar": 2, "root": "G", "confidence": 0.92, "source": "pyin"},
         {"bar": 3, "root": "G", "confidence": 0.0,  "source": "inherit"}, ...]

    `source`:
      - "pyin"    : pitch was detected for at least MIN_VOICED_RATIO of the bar
      - "inherit" : bar was mostly silent; root carried over from previous bar
      - (no entry): bar is before any voiced content (can't inherit yet)
    """
    downbeats = (grid or {}).get("downbeat_times") or []
    song_end = (grid or {}).get("song_duration_sec") or 0.0
    if not downbeats or not Path(bass_path).is_file():
        logger.warning("bass_root: missing grid or bass file, returning empty")
        return []

    try:
        import librosa
        import numpy as np
    except ImportError as e:
        logger.error(f"bass_root: librosa/numpy not available ({e})")
        return []

    try:
        audio, sr = librosa.load(bass_path, sr=sample_rate, mono=True)
    except Exception as e:
        logger.error(f"bass_root: failed to load {bass_path}: {e}")
        return []

    if audio.size == 0:
        logger.warning(f"bass_root: {bass_path} is empty")
        return []

    try:
        f0, voiced, voiced_prob = librosa.pyin(
            audio,
            fmin=BASS_FMIN,
            fmax=BASS_FMAX,
            sr=sr,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
        )
    except Exception as e:
        logger.error(f"bass_root: pyin failed: {e}")
        return []

    frame_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP_LENGTH)

    results: List[Dict] = []
    # Only look at the first DOWNBEAT_WINDOW_RATIO of each bar — that's when
    # the bass player hits the root on beat 1 before walking off into passing
    # notes. Averaging across a full bar of walking bass collapses the root
    # into whatever pitch is held longest, which is often the 5th.
    DOWNBEAT_WINDOW_RATIO = 0.35
    for i, start in enumerate(downbeats):
        end = downbeats[i + 1] if i + 1 < len(downbeats) else song_end
        # Shrink the analysis window to the front of the bar
        bar_length = end - start
        window_end = start + bar_length * DOWNBEAT_WINDOW_RATIO
        mask = (frame_times >= start) & (frame_times < window_end)
        if not np.any(mask):
            continue
        bar_f0 = f0[mask]
        bar_voiced = voiced[mask]

        voiced_count = int(np.nansum(bar_voiced.astype(bool)))
        voiced_ratio = voiced_count / max(1, len(bar_voiced))

        valid_f0 = bar_f0[np.logical_and(bar_voiced.astype(bool), ~np.isnan(bar_f0))]

        if voiced_ratio < MIN_VOICED_RATIO or len(valid_f0) == 0:
            if results:
                prev = results[-1]
                results.append({
                    "bar": i + 1,
                    "root": prev["root"],
                    "confidence": 0.0,
                    "source": "inherit",
                    "voiced_ratio": round(voiced_ratio, 3),
                })
            # else: skip — nothing yet to inherit from
            continue

        pitch_classes = _hz_to_pitch_class(valid_f0)
        from collections import Counter
        pc_counter = Counter(pitch_classes)
        top_pc, top_count = pc_counter.most_common(1)[0]
        confidence = top_count / len(pitch_classes)
        results.append({
            "bar": i + 1,
            "root": NOTE_NAMES[top_pc],
            "confidence": round(float(confidence), 3),
            "source": "pyin",
            "voiced_ratio": round(float(voiced_ratio), 3),
        })

    logger.info(
        f"bass_root: extracted {len(results)} bar-roots from {Path(bass_path).name} "
        f"({sum(1 for r in results if r['source']=='pyin')} pyin, "
        f"{sum(1 for r in results if r['source']=='inherit')} inherited)"
    )
    return results


def _hz_to_pitch_class(f0_array):
    """Convert frequencies (Hz) to pitch class integers (0=C, 1=C#, ..., 11=B)."""
    import numpy as np
    midi = 69.0 + 12.0 * np.log2(f0_array / 440.0)
    pc = np.round(midi).astype(int) % 12
    return pc.tolist()


# ---------------------------------------------------------------------------
# Combine bass roots + detector-inferred qualities into final chords per bar
# ---------------------------------------------------------------------------

import re as _re
_ROOT_RE = _re.compile(r'^([A-G][#b]?)(.*)$')


def combine_with_detector_quality(
    bass_roots: List[Dict],
    detector_chord_events: List[Dict],
    grid: Dict,
    min_bass_confidence: float = 0.4,
) -> List[Dict]:
    """
    Merge bass-anchored roots (reliable) with detector-inferred chord qualities
    (less reliable on root, usually right on quality) into a final chord per bar.

    For each bar:
      1. Find the detector chord event with the largest time-overlap inside
         the bar.
      2. Strip its root letter, keep the quality suffix (m, m7, maj7, 7, etc.).
      3. If bass confidence is high enough, use bass root + detector quality.
         Otherwise fall back to the detector's chord entirely.

    Output: same shape as _quantize_chords_to_bars output so callers can swap
    them interchangeably:
        [{"bar": 1, "chord": "Cm", "start_time": 39.87, "end_time": 42.12,
          "bass_root": "C", "detector_root": "C", "source": "bass+detector"}, ...]
    """
    downbeats = (grid or {}).get("downbeat_times") or []
    song_end = (grid or {}).get("song_duration_sec") or 0.0
    if not downbeats:
        return []

    # Index bass roots by bar for O(1) lookup
    by_bar = {r["bar"]: r for r in bass_roots}

    combined: List[Dict] = []
    for i, start in enumerate(downbeats):
        end = downbeats[i + 1] if i + 1 < len(downbeats) else song_end
        bar = i + 1

        # Dominant detector chord in this bar
        overlaps: Dict[str, float] = {}
        for c in detector_chord_events:
            c_start = c["time"]
            c_end = c_start + c.get("duration", 1.0)
            overlap = max(0.0, min(end, c_end) - max(start, c_start))
            if overlap > 0:
                overlaps[c["chord"]] = overlaps.get(c["chord"], 0.0) + overlap
        detector_chord = max(overlaps, key=overlaps.get) if overlaps else None

        # Split detector chord into root + quality
        detector_root, detector_quality = "", ""
        if detector_chord:
            m = _ROOT_RE.match(detector_chord)
            if m:
                detector_root, detector_quality = m.group(1), m.group(2)

        bass_entry = by_bar.get(bar)
        bass_root = bass_entry["root"] if bass_entry else None
        bass_conf = bass_entry["confidence"] if bass_entry else 0.0

        # Decide which root wins
        if bass_root and bass_conf >= min_bass_confidence:
            final_root = bass_root
            source = "bass+detector" if detector_quality else "bass"
        elif detector_root:
            final_root = detector_root
            source = "detector-fallback"
        else:
            # No info — hold previous bar
            if combined:
                final_root = combined[-1]["chord"][:1]
                detector_quality = combined[-1]["chord"][1:]
                source = "hold"
            else:
                continue

        chord = final_root + detector_quality
        combined.append({
            "bar": bar,
            "chord": chord,
            "start_time": round(start, 3),
            "end_time": round(end, 3),
            "bass_root": bass_root,
            "bass_confidence": bass_conf,
            "detector_root": detector_root,
            "detector_quality": detector_quality,
            "source": source,
        })
    return combined


# ---------------------------------------------------------------------------
# Quality smoother — majority-vote snap per bass root
# ---------------------------------------------------------------------------

# Quality suffixes we treat as "informative extensions" — when the detector
# surfaces these on a root, it's positive evidence the chord IS extended.
# Triad-like qualities ('', 'm', 'sus2', 'sus4', etc.) carry no extension
# information either way — a bar showing 'Cm' may genuinely be Cm or may be
# a Cm7 where the detector failed to see the b7.
_EXTENSION_QUALITIES = {
    'm7', 'maj7', '7',
    'm9', 'maj9', '9',
    'm11', 'maj11', '11',
    '13', 'maj13', 'm13',
    'm6', '6',
    'add9', 'madd9',
    '7sus4', '9sus4',
    'dim7', 'hdim7',
}


def smooth_qualities(
    bar_grid: List[Dict],
    min_occurrences: int = 2,
    majority_ratio: float = 0.5,
    extension_min_occurrences: int = 3,
    extension_min_ratio: float = 0.10,
) -> List[Dict]:
    """
    Snap ambiguous quality labels to the most informative quality for each
    bass root.

    Two passes:

    1. **Extension promotion.** Extensions (m7/maj7/7/etc.) carry strictly
       more information than triads. When the detector surfaces an extension
       on a root at least `extension_min_occurrences` times AND in at least
       `extension_min_ratio` of that root's bars, snap ALL bars of that root
       to the extension. Reason: when a song is genuinely Cm7 throughout, the
       detector inconsistently catches the b7 — bars where the b7 wasn't
       loud enough get labeled plain Cm. Without promotion, the legacy
       majority-ratio rule would snap the minority Cm7 evidence to the
       majority Cm — destroying the signal that proves the chord is m7.
       This is the bug Apr 24's _simplify_bleed_extensions fix tried to
       address but couldn't reach (it operates upstream on chord_events,
       not on bar_grid).

    2. **Triad smoothing (legacy).** If no extension promoted for a root,
       fall back to the original majority-ratio rule: snap minority quality
       to the dominant quality when it accounts for >majority_ratio of bars.
       This still handles cases like '' (major triad) being inconsistently
       paired with 'm' (minor triad) on the same root.

    Returns a new list (does not mutate the input). Source field is updated
    to track which bars were smoothed.
    """
    if not bar_grid:
        return bar_grid
    from collections import Counter

    qualities_by_root: Dict[str, Counter] = {}
    for b in bar_grid:
        root = b.get("bass_root")
        if not root:
            continue
        qualities_by_root.setdefault(root, Counter())[b.get("detector_quality", "")] += 1

    dominant_quality: Dict[str, str] = {}
    for root, counter in qualities_by_root.items():
        total = sum(counter.values())

        # Pass 1: prefer the most-common informative extension if it clears
        # both the absolute and relative thresholds.
        ext_counts = Counter({q: n for q, n in counter.items() if q in _EXTENSION_QUALITIES})
        if ext_counts:
            top_ext_q, top_ext_n = ext_counts.most_common(1)[0]
            if (top_ext_n >= extension_min_occurrences
                    and top_ext_n / total >= extension_min_ratio):
                dominant_quality[root] = top_ext_q
                continue

        # Pass 2: legacy triad-majority smoothing.
        top_q, top_n = counter.most_common(1)[0]
        if top_n >= min_occurrences and top_n / total > majority_ratio:
            dominant_quality[root] = top_q

    smoothed: List[Dict] = []
    for b in bar_grid:
        root = b.get("bass_root")
        if root and root in dominant_quality:
            dq = dominant_quality[root]
            if b.get("detector_quality", "") != dq:
                new_b = dict(b)
                new_b["chord"] = root + dq
                new_b["detector_quality"] = dq
                new_b["source"] = (b.get("source", "") or "") + "+smoothed"
                smoothed.append(new_b)
                continue
        smoothed.append(b)
    return smoothed


# ---------------------------------------------------------------------------
# CLI for sanity check
# ---------------------------------------------------------------------------

def _cli():
    import argparse, json, sys
    from processing.tempo_beats import extract_grid
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("bass", help="Bass stem audio file (.mp3 or .wav)")
    p.add_argument("--drums", help="Drums stem for grid extraction")
    p.add_argument("--mix", help="Mix for grid fallback")
    p.add_argument("--head", type=int, default=0, help="Only print first N bars")
    args = p.parse_args()

    grid = extract_grid(
        drums_path=args.drums,
        mix_path=args.mix or args.bass,
    )
    roots = extract_bass_roots(args.bass, grid)
    if args.head > 0:
        roots = roots[: args.head]
    json.dump({
        "tempo_bpm": grid.get("tempo_bpm"),
        "bar_count": grid.get("bar_count"),
        "bass_roots": roots,
    }, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    _cli()
