"""
Chord Pattern Analyzer for StemScribe
======================================
Detects song structure by analyzing repeating chord progressions.

Core idea: Songs repeat chord patterns within sections. When the pattern
changes, it's a new section. When a pattern returns, it's the same section type.

This is the PRIMARY signal for structure detection — more reliable than
audio energy features alone.

Algorithm:
1. Build a sequence of chord names in order of appearance
2. Use sliding windows of N chords (N=4,6,8) to find repeating subsequences
3. Compute a self-similarity matrix of windows using chord-set Jaccard distance
4. Detect boundaries where similarity drops (new section)
5. Label sections by pattern frequency and position

Dependencies: numpy (required), librosa (optional, for beat tracking from audio)
"""

import logging
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# ==============================================================================
# Public API
# ==============================================================================

def analyze_chord_patterns(
    chords: List[Dict],
    audio_path: Optional[str] = None,
    beats_per_bar: int = 4,
) -> List[Dict]:
    """
    Analyze chord progressions to detect song structure via repeating patterns.

    Args:
        chords: List of chord events, each with keys:
            chord (str), time (float), duration (float), confidence (float).
        audio_path: Optional path to audio file for beat tracking.
            If None, tempo is estimated from chord durations.
        beats_per_bar: Beats per bar (default 4 for 4/4 time).

    Returns:
        List of pattern sections:
        [
            {
                "pattern": "A",
                "start_time": 0.0,
                "end_time": 25.0,
                "chords": ["F#m", "A", "Bm", "E"],
                "bar_count": 4,
                "section_hint": "verse",
            },
            ...
        ]
    """
    if not chords or len(chords) < 2:
        return []

    # Parse chord events into a clean sequence
    events = _parse_events(chords)
    if len(events) < 4:
        return []

    # Try chord-sequence approach (primary)
    sections = _chord_sequence_segmentation(events)

    if not sections:
        return []

    # Label patterns by frequency and position
    sections = _label_patterns(sections)

    logger.info(f"Chord pattern analysis: {len(sections)} sections detected")
    for s in sections:
        logger.info(
            f"  Pattern {s['pattern']} ({s['section_hint']}): "
            f"{s['start_time']:.1f}-{s['end_time']:.1f}s  "
            f"chords={s['chords']}"
        )

    return sections


# ==============================================================================
# Event parsing
# ==============================================================================

def _parse_events(chords: List[Dict]) -> List[Dict]:
    """Parse chord list into clean event dicts."""
    events = []
    for c in chords:
        name = c.get("chord", "N")
        if not name or name == "N":
            continue
        t = float(c.get("time", c.get("start", 0)))
        d = float(c.get("duration", 1.5))
        events.append({
            "chord": name,
            "time": t,
            "end": t + d,
            "duration": d,
        })
    # Sort by time
    events.sort(key=lambda e: e["time"])
    return events


# ==============================================================================
# Chord-sequence segmentation
# ==============================================================================

def _chord_sequence_segmentation(events: List[Dict]) -> List[Dict]:
    """
    Segment the chord sequence by finding repeating chord-set patterns.

    Strategy:
    1. Divide the chord timeline into fixed-duration windows (~6s each,
       roughly 2 bars at typical tempos).
    2. For each window, compute its chord-set fingerprint.
    3. Compare windows using Jaccard similarity on chord sets.
    4. Group consecutive windows with the same fingerprint.
    5. Where the fingerprint changes, mark a section boundary.
    """
    if not events:
        return []

    first_t = events[0]["time"]
    last_t = events[-1]["end"]
    total_dur = last_t - first_t

    # Estimate a good window size from chord durations
    # Each chord change is roughly one beat — 4 changes = 1 bar
    # We want ~2 bars per window (8 chord changes worth of time)
    durations = [e["duration"] for e in events]
    median_dur = float(np.median(durations))
    # Window = ~4 chord durations (roughly 1 bar), but we'll try multiples
    base_window = median_dur * 4  # ~1 bar

    # Clamp to reasonable range
    base_window = max(3.0, min(8.0, base_window))

    logger.info(f"Median chord dur: {median_dur:.2f}s, base window: {base_window:.2f}s")

    # Try multiple window sizes and pick the one that produces the best segmentation
    best_sections = None
    best_score = -1

    for multiplier in [1, 2, 3]:
        win_dur = base_window * multiplier
        if win_dur > total_dur / 3:
            continue

        windows = _build_time_windows(events, first_t, last_t, win_dur)
        if len(windows) < 3:
            continue

        # Build fingerprints and cluster
        fingerprints = [_window_fingerprint(w) for w in windows]
        clusters = _cluster_fingerprints(fingerprints)

        # Score: prefer fewer unique clusters with good repetition
        n_unique = len(set(clusters))
        cluster_counts = Counter(clusters)
        n_repeating = sum(1 for v in cluster_counts.values() if v >= 2)
        max_size = max(cluster_counts.values())

        # Penalty for too many or too few clusters
        ideal_clusters = max(3, min(8, len(windows) // 3))
        cluster_penalty = abs(n_unique - ideal_clusters)

        score = n_repeating * 15 + max_size * 5 - cluster_penalty * 3 - n_unique

        if score > best_score:
            best_score = score
            best_sections = _clusters_to_sections(windows, clusters)

    return best_sections or []


def _build_time_windows(
    events: List[Dict],
    start_t: float,
    end_t: float,
    win_dur: float,
) -> List[Dict]:
    """Build non-overlapping time windows with their chord content."""
    windows = []
    t = start_t
    while t < end_t - win_dur * 0.3:  # Allow partial last window
        w_end = min(t + win_dur, end_t)
        # Find chords in this window
        w_chords = []
        w_chord_seq = []  # Ordered sequence
        for e in events:
            if e["end"] > t and e["time"] < w_end:
                w_chord_seq.append(e["chord"])
                if e["chord"] not in w_chords:
                    w_chords.append(e["chord"])

        if w_chords:
            windows.append({
                "start_time": t,
                "end_time": w_end,
                "chord_set": frozenset(w_chords),
                "chord_seq": tuple(w_chord_seq),
                "chords": w_chords,
            })
        t += win_dur

    return windows


def _window_fingerprint(window: Dict) -> Tuple:
    """
    Create a fingerprint for a window that captures its harmonic character.

    Uses the chord set (unordered) — this is more robust than exact sequence
    matching since chord detection timing can be noisy.
    """
    return window["chord_set"]


def _jaccard_similarity(set_a: frozenset, set_b: frozenset) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _sequence_similarity(seq_a: List[str], seq_b: List[str]) -> float:
    """
    Compute similarity between two chord sequences using normalized edit distance.
    Returns 0.0 (completely different) to 1.0 (identical).
    """
    if not seq_a and not seq_b:
        return 1.0
    if not seq_a or not seq_b:
        return 0.0
    if seq_a == seq_b:
        return 1.0

    n, m = len(seq_a), len(seq_b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    max_len = max(n, m)
    return 1.0 - dp[n][m] / max_len


def _cluster_fingerprints(
    fingerprints: List[frozenset],
    threshold: float = 0.55,
) -> List[int]:
    """
    Cluster window fingerprints by Jaccard similarity.

    Uses greedy assignment: each window joins the best matching existing cluster,
    or starts a new one if similarity is below threshold.
    """
    n = len(fingerprints)
    clusters = [-1] * n
    centroids = []  # List of (cluster_id, representative_fingerprint)
    next_id = 0

    for i in range(n):
        fp_i = fingerprints[i]
        best_cluster = -1
        best_sim = threshold

        for cid, rep_fp in centroids:
            sim = _jaccard_similarity(fp_i, rep_fp)
            if sim > best_sim:
                best_sim = sim
                best_cluster = cid

        if best_cluster >= 0:
            clusters[i] = best_cluster
        else:
            clusters[i] = next_id
            centroids.append((next_id, fp_i))
            next_id += 1

    return clusters


def _clusters_to_sections(
    windows: List[Dict],
    clusters: List[int],
) -> List[Dict]:
    """
    Convert clustered windows into section dicts.
    Merge consecutive windows with the same cluster.
    """
    if not windows or not clusters:
        return []

    # Merge consecutive same-cluster windows
    sections = []
    current_cluster = clusters[0]
    current_start = 0

    for i in range(1, len(windows)):
        if clusters[i] != current_cluster:
            sections.append(_make_section(
                windows, current_start, i, f"P{current_cluster}"
            ))
            current_cluster = clusters[i]
            current_start = i
    sections.append(_make_section(
        windows, current_start, len(windows), f"P{current_cluster}"
    ))

    # Merge very short sections (1 window) into their best neighbor
    sections = _merge_short_sections(sections, min_windows=1)

    return sections


def _make_section(
    windows: List[Dict],
    start_idx: int,
    end_idx: int,
    pattern: str,
) -> Dict:
    """Create a section dict from a range of windows."""
    all_chords = []
    for w in windows[start_idx:end_idx]:
        for ch in w["chords"]:
            if ch not in all_chords:
                all_chords.append(ch)

    return {
        "pattern": pattern,
        "start_time": round(windows[start_idx]["start_time"], 2),
        "end_time": round(windows[end_idx - 1]["end_time"], 2),
        "chords": all_chords,
        "bar_count": end_idx - start_idx,  # Window count, roughly bar pairs
        "section_hint": "",
    }


def _merge_short_sections(sections: List[Dict], min_windows: int = 1) -> List[Dict]:
    """Merge sections that are only 1 window into adjacent sections."""
    if len(sections) <= 2:
        return sections

    merged = [sections[0]]
    for i in range(1, len(sections)):
        s = sections[i]
        prev = merged[-1]

        # If this section is very short (1 window) and pattern doesn't repeat elsewhere
        if s["bar_count"] <= min_windows:
            # Check if this pattern appears elsewhere
            pattern_elsewhere = any(
                other["pattern"] == s["pattern"] and j != i
                for j, other in enumerate(sections)
            )
            if not pattern_elsewhere:
                # Merge into previous
                prev["end_time"] = s["end_time"]
                prev["bar_count"] += s["bar_count"]
                for ch in s["chords"]:
                    if ch not in prev["chords"]:
                        prev["chords"].append(ch)
                continue

        merged.append(s)

    return merged


# ==============================================================================
# Pattern labeling
# ==============================================================================

def _label_patterns(sections: List[Dict]) -> List[Dict]:
    """
    Assign meaningful labels to patterns based on frequency and position.

    Rules:
    - Most common pattern (by total bar/window count) = likely verse -> "A"
    - Second most common = likely chorus -> "B"
    - Pattern appearing only once = bridge -> "C", "D", etc.
    - First section (if unique pattern) = intro
    - Last section (if matches intro pattern) = outro
    """
    if not sections:
        return sections

    # Count total windows per pattern
    pattern_bars = Counter()
    pattern_occurrences = Counter()
    for s in sections:
        pattern_bars[s["pattern"]] += s["bar_count"]
        pattern_occurrences[s["pattern"]] += 1

    # Rank patterns: most total windows first
    ranked = [p for p, _ in pattern_bars.most_common()]

    # Assign clean letter labels
    old_to_new = {}
    letter_idx = 0
    for pat in ranked:
        old_to_new[pat] = chr(ord("A") + letter_idx)
        letter_idx = min(letter_idx + 1, 25)  # Cap at Z

    # Update pattern labels
    for s in sections:
        s["pattern"] = old_to_new[s["pattern"]]

    # Re-count with new labels
    pattern_occurrences_new = Counter()
    pattern_bars_new = Counter()
    for s in sections:
        pattern_occurrences_new[s["pattern"]] += 1
        pattern_bars_new[s["pattern"]] += s["bar_count"]

    ranked_new = [p for p, _ in pattern_bars_new.most_common()]

    # Determine verse/chorus candidates
    verse_pattern = ranked_new[0] if ranked_new else None
    chorus_pattern = ranked_new[1] if len(ranked_new) > 1 else None

    for i, s in enumerate(sections):
        pat = s["pattern"]
        occ = pattern_occurrences_new[pat]

        # First section with a unique or uncommon pattern = intro
        if i == 0 and pat != verse_pattern and occ <= 2:
            s["section_hint"] = "intro"
        elif pat == verse_pattern:
            s["section_hint"] = "verse"
        elif pat == chorus_pattern:
            s["section_hint"] = "chorus"
        elif occ == 1:
            s["section_hint"] = "bridge"
        else:
            s["section_hint"] = "section"

    # Last section: if it matches intro or is unique, call it outro
    if len(sections) >= 2:
        last = sections[-1]
        first = sections[0]
        if first.get("section_hint") == "intro":
            if last["pattern"] == first["pattern"]:
                last["section_hint"] = "outro"
            elif last["chords"] == first["chords"]:
                last["section_hint"] = "outro"

    return sections


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python chord_pattern_analyzer.py <chords.json> [audio_path]")
        print("\nAnalyzes chord patterns to detect song structure.")
        sys.exit(1)

    chords_file = sys.argv[1]
    audio = sys.argv[2] if len(sys.argv) >= 3 else None

    with open(chords_file) as f:
        chord_data = json.load(f)

    result = analyze_chord_patterns(chord_data, audio)
    print(json.dumps(result, indent=2))
