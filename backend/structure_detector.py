"""
Song Structure Detection for StemScriber — Ensemble Detector
============================================================
Detects song sections (Intro, Verse, Chorus, Bridge, Outro, Solo/Instrumental)
by combining three signal sources:

1. Chord Pattern Analyzer (primary) — repeating chord progressions
2. Lyrics Repetition Detector — chorus identification via repeated lyrics
3. Audio Boundary Detector — precise boundary timestamps from audio features

Ensemble logic:
- Start with chord patterns as the primary segmentation
- Overlay lyrics repetition to identify choruses
- Refine boundary positions using audio boundary detection
- Apply labeling rules based on position, repetition, and content

Dependencies: numpy, librosa (pre-installed in venv311).
"""

import numpy as np
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

logger = logging.getLogger(__name__)

try:
    import librosa
    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    logger.warning(f"structure_detector: missing dependency — {e}")


# ==============================================================================
# Public API
# ==============================================================================

def detect_structure(
    audio_path: str,
    chords: Optional[List[Dict]] = None,
    lyrics: Optional[List[Dict]] = None,
    output_path: Optional[str] = None,
) -> List[Dict]:
    """
    Detect song sections from audio, chords, and optional lyrics.

    Uses an ensemble of three detectors:
    1. Chord pattern analyzer (primary signal)
    2. Lyrics repetition detector (chorus identification)
    3. Audio boundary detector (boundary refinement)

    Args:
        audio_path: Path to audio file (full mix preferred).
        chords: List of dicts with keys: chord, time, duration, confidence.
        lyrics: Optional list of dicts with keys: word/text, start_time/start, end_time/end.
        output_path: Optional path to save structure.json.

    Returns:
        List of section dicts: [{"name": str, "start_time": float, "end_time": float}, ...]
    """
    if not DEPS_AVAILABLE:
        logger.error("Cannot detect structure — missing dependencies")
        return _fallback_from_chords(chords) if chords else []

    try:
        logger.info(f"Detecting structure (ensemble) for: {audio_path}")

        # Get audio duration
        y, sr = librosa.load(str(audio_path), sr=22050)
        duration = librosa.get_duration(y=y, sr=sr)
        logger.info(f"Audio duration: {duration:.1f}s")

        # --- Signal 1: Chord patterns (primary) ---
        chord_sections = _get_chord_sections(chords)

        # --- Signal 2: Lyrics repetition ---
        lyrics_result = _get_lyrics_sections(lyrics)

        # --- Signal 3: Audio boundaries ---
        audio_boundaries = _get_audio_boundaries(audio_path)

        # --- Ensemble: combine signals ---
        segments = _ensemble_combine(
            chord_sections=chord_sections,
            lyrics_result=lyrics_result,
            audio_boundaries=audio_boundaries,
            chords=chords,
            lyrics=lyrics,
            duration=duration,
            y=y,
            sr=sr,
        )

        # Post-process
        segments = _postprocess(segments, duration)

        logger.info(f"Detected {len(segments)} sections:")
        for s in segments:
            logger.info(f"  {s['name']}: {s['start_time']:.1f}s - {s['end_time']:.1f}s")

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(segments, f, indent=2)
            logger.info(f"Saved structure to {output_path}")

        return segments

    except Exception as e:
        logger.error(f"Structure detection failed: {e}", exc_info=True)
        return _fallback_from_chords(chords) if chords else []


# ==============================================================================
# Signal acquisition — safely import and run each sub-detector
# ==============================================================================

def _get_chord_sections(chords):
    """Run chord pattern analyzer. Returns list of pattern sections or []."""
    if not chords or len(chords) < 4:
        return []
    try:
        from chord_pattern_analyzer import analyze_chord_patterns
        sections = analyze_chord_patterns(chords)
        logger.info(f"Chord pattern analyzer: {len(sections)} sections")
        return sections
    except Exception as e:
        logger.warning(f"Chord pattern analyzer failed: {e}")
        return []


def _get_lyrics_sections(lyrics):
    """Run lyrics repetition detector. Returns result dict or empty."""
    if not lyrics:
        return {"chorus_lines": [], "sections": []}
    try:
        # Normalize lyrics format
        normalized = _normalize_lyrics(lyrics)
        if not normalized:
            return {"chorus_lines": [], "sections": []}
        from lyrics_repetition_detector import detect_lyrics_repetition
        result = detect_lyrics_repetition(normalized)
        logger.info(f"Lyrics repetition: {len(result.get('sections', []))} sections, "
                     f"{len(result.get('chorus_lines', []))} repeating lines")
        return result
    except Exception as e:
        logger.warning(f"Lyrics repetition detector failed: {e}")
        return {"chorus_lines": [], "sections": []}


def _get_audio_boundaries(audio_path):
    """Run audio boundary detector. Returns list of boundaries or []."""
    try:
        from audio_boundary_detector import detect_audio_boundaries
        boundaries = detect_audio_boundaries(audio_path)
        logger.info(f"Audio boundary detector: {len(boundaries)} boundaries")
        return boundaries
    except Exception as e:
        logger.warning(f"Audio boundary detector failed: {e}")
        return []


def _normalize_lyrics(lyrics):
    """Normalize lyrics to the format expected by lyrics_repetition_detector."""
    if not lyrics:
        return []
    normalized = []
    for w in lyrics:
        text = w.get("text", w.get("word", ""))
        start = w.get("start_time", w.get("start", None))
        end = w.get("end_time", w.get("end", None))
        if text and start is not None and end is not None:
            normalized.append({
                "text": text,
                "start_time": float(start),
                "end_time": float(end),
            })
    return normalized


# ==============================================================================
# Ensemble combination logic
# ==============================================================================

def _ensemble_combine(
    chord_sections,
    lyrics_result,
    audio_boundaries,
    chords,
    lyrics,
    duration,
    y,
    sr,
):
    """
    Combine the three signal sources into a final section list.

    Priority:
    1. Chord patterns provide the primary segmentation and boundaries
    2. Lyrics repetition overrides labeling (repeated lyrics = Chorus)
    3. Audio boundaries refine boundary positions (snap within 5s)
    """

    # --- Step 1: Build initial segments from chord patterns ---
    if chord_sections:
        segments = _segments_from_chord_patterns(chord_sections, duration)
    elif chords and len(chords) >= 4:
        # Fallback: basic chord-based segmentation using energy
        segments = _basic_chord_segmentation(chords, duration, y, sr)
    elif lyrics_result and lyrics_result.get("sections"):
        # Lyrics only
        segments = _segments_from_lyrics(lyrics_result, duration)
    else:
        # Audio-only fallback
        segments = _audio_only_segmentation(y, sr, duration, audio_boundaries)

    if not segments:
        return [{"name": "Song", "start_time": 0.0, "end_time": round(duration, 2)}]

    # --- Step 2: Refine boundaries with audio boundaries ---
    if audio_boundaries:
        segments = _refine_with_audio_boundaries(segments, audio_boundaries)

    # --- Step 3: Add audio-only boundaries if high confidence ---
    if audio_boundaries:
        segments = _add_missed_boundaries(segments, audio_boundaries, min_confidence=0.7)

    # --- Step 4: Overlay lyrics for chorus identification ---
    if lyrics_result:
        segments = _overlay_lyrics_labels(segments, lyrics_result, lyrics)

    # --- Step 5: Apply labeling rules ---
    segments = _apply_labeling_rules(segments, lyrics, duration)

    # --- Step 6: Detect chorus from distinctive chords ---
    # If chords are available, look for rare/distinctive chord clusters
    # that indicate chorus sections missed by the pattern analyzer
    if chords and len(chords) >= 4:
        segments = _detect_chorus_from_distinctive_chords(segments, chords, lyrics)

    # --- Step 7: Cap all chorus sections at ~20 seconds ---
    # Bug fix #3: Choruses in popular music are typically 10-20s. Any chorus
    # longer than 20s indicates a boundary error. Split it.
    MAX_CHORUS_DURATION = 20.0
    capped = []
    for seg in segments:
        seg_dur = seg["end_time"] - seg["start_time"]
        if seg["name"] == "Chorus" and seg_dur > MAX_CHORUS_DURATION:
            chorus_part = dict(seg)
            chorus_part["end_time"] = round(seg["start_time"] + MAX_CHORUS_DURATION, 2)
            capped.append(chorus_part)

            remainder = dict(seg)
            remainder["start_time"] = chorus_part["end_time"]
            remainder["name"] = "Verse"
            capped.append(remainder)
            logger.info(f"Capped {seg_dur:.1f}s chorus at {chorus_part['end_time']:.1f}s")
        else:
            capped.append(seg)
    segments = capped

    # --- Step 8: Bridge detection (after all other labeling) ---
    # Bug fix #2: A bridge is a new chord pattern that appears after the last
    # chorus. Any vocal section after the last chorus with no following chorus
    # should be labeled "Bridge" instead of "Verse."
    last_chorus_idx = None
    for i in range(len(segments) - 1, -1, -1):
        if segments[i]["name"] == "Chorus":
            last_chorus_idx = i
            break

    if last_chorus_idx is not None:
        for i in range(last_chorus_idx + 1, len(segments)):
            seg = segments[i]
            if seg["name"] in ("Intro", "Outro", "Instrumental"):
                continue
            # Any vocal section after the last chorus -> Bridge
            if seg["name"] in ("Verse", "Unknown"):
                seg["name"] = "Bridge"
                logger.info(f"Bridge detected at {seg['start_time']:.1f}s-{seg['end_time']:.1f}s "
                            f"(after last chorus at idx {last_chorus_idx})")

    return segments


def _segments_from_chord_patterns(chord_sections, duration):
    """Convert chord pattern analyzer output into segment dicts."""
    segments = []
    for cs in chord_sections:
        segments.append({
            "name": "Unknown",
            "start_time": round(cs["start_time"], 2),
            "end_time": round(cs["end_time"], 2),
            "_pattern": cs.get("pattern", "?"),
            "_chords": cs.get("chords", []),
            "_hint": cs.get("section_hint", ""),
            "_bar_count": cs.get("bar_count", 0),
        })

    # Extend last segment to duration if close
    if segments and duration - segments[-1]["end_time"] < 5.0:
        segments[-1]["end_time"] = round(duration, 2)

    return segments


def _segments_from_lyrics(lyrics_result, duration):
    """Build segments from lyrics-only detection."""
    segments = []
    for ls in lyrics_result.get("sections", []):
        segments.append({
            "name": ls.get("name", "Unknown"),
            "start_time": round(ls["start_time"], 2),
            "end_time": round(ls["end_time"], 2),
            "_pattern": "?",
            "_chords": [],
            "_hint": ls.get("name", "").lower(),
            "_bar_count": 0,
        })
    return segments


def _audio_only_segmentation(y, sr, duration, audio_boundaries):
    """Build segments from audio boundaries alone."""
    if not audio_boundaries:
        return [{"name": "Unknown", "start_time": 0.0, "end_time": round(duration, 2),
                 "_pattern": "?", "_chords": [], "_hint": "", "_bar_count": 0}]

    times = [0.0] + [b["time"] for b in audio_boundaries] + [duration]
    segments = []
    for i in range(len(times) - 1):
        if times[i + 1] - times[i] < 3.0 and segments:
            segments[-1]["end_time"] = round(times[i + 1], 2)
            continue
        segments.append({
            "name": "Unknown",
            "start_time": round(times[i], 2),
            "end_time": round(times[i + 1], 2),
            "_pattern": f"S{i}",
            "_chords": [],
            "_hint": "",
            "_bar_count": 0,
        })
    return segments


def _basic_chord_segmentation(chords, duration, y, sr):
    """Fallback chord segmentation when chord_pattern_analyzer isn't available."""
    # Use the old windowed approach as fallback
    events = []
    for c in chords:
        t = float(c.get("time", c.get("start", 0)))
        d = float(c.get("duration", 1.5))
        name = c.get("chord", "N")
        if name and name != "N":
            events.append((t, t + d, name))
    if not events:
        return []

    first_t = events[0][0]
    last_t = events[-1][1]
    win_size = 6.0
    windows = []
    t = first_t
    while t < last_t:
        w_end = min(t + win_size, duration)
        cs = frozenset(n for s, e, n in events if e > t and s < w_end)
        windows.append((t, w_end, cs))
        t += win_size

    if not windows:
        return []

    # Simple run-length encoding by chord set
    segments = []
    current_cs = windows[0][2]
    seg_start = windows[0][0]
    seg_idx = 0

    for i in range(1, len(windows)):
        if windows[i][2] != current_cs:
            segments.append({
                "name": "Unknown",
                "start_time": round(seg_start, 2),
                "end_time": round(windows[i - 1][1], 2),
                "_pattern": f"F{seg_idx}",
                "_chords": list(current_cs),
                "_hint": "",
                "_bar_count": i - seg_idx,
            })
            current_cs = windows[i][2]
            seg_start = windows[i][0]
            seg_idx = i

    segments.append({
        "name": "Unknown",
        "start_time": round(seg_start, 2),
        "end_time": round(duration, 2),
        "_pattern": f"F{seg_idx}",
        "_chords": list(current_cs),
        "_hint": "",
        "_bar_count": len(windows) - seg_idx,
    })

    return segments


# ==============================================================================
# Boundary refinement
# ==============================================================================

def _refine_with_audio_boundaries(segments, audio_boundaries):
    """
    Snap segment boundaries to nearby audio boundaries for precision.
    If an audio boundary is within the snap threshold of a chord-pattern
    boundary, use it.

    Important: shared boundaries between adjacent segments are snapped
    together so that one segment doesn't absorb time from the next.
    The snap threshold is tighter (3s) to avoid overriding chord-pattern
    boundaries that are already precise.
    """
    snap_threshold = 3.0
    min_duration = 3.0
    boundary_times = sorted(b["time"] for b in audio_boundaries)

    def _nearest_audio_boundary(t, exclude_zero=False):
        """Find the nearest audio boundary to time t within threshold."""
        best = None
        best_dist = snap_threshold
        for bt in boundary_times:
            if exclude_zero and bt == 0.0:
                continue
            dist = abs(bt - t)
            if dist < best_dist:
                best = bt
                best_dist = dist
        return best

    # Process shared boundaries between adjacent segments together.
    # When seg[i].end_time == seg[i+1].start_time, snap both to the
    # same audio boundary so the boundary moves as a unit.
    shared_boundaries = {}  # original_time -> snapped_time
    for i in range(len(segments) - 1):
        shared_t = segments[i]["end_time"]
        if abs(shared_t - segments[i + 1]["start_time"]) < 0.5:
            if shared_t not in shared_boundaries:
                snapped = _nearest_audio_boundary(shared_t)
                shared_boundaries[shared_t] = round(snapped, 2) if snapped is not None else shared_t

    for seg in segments:
        orig_start = seg["start_time"]
        orig_end = seg["end_time"]

        # Use shared boundary snap if this boundary is shared with an adjacent segment
        new_start = orig_start
        new_end = orig_end

        # Check if start_time matches a shared boundary
        for orig_t, snapped_t in shared_boundaries.items():
            if abs(orig_start - orig_t) < 0.5:
                new_start = snapped_t
                break
        else:
            # Not a shared boundary — snap independently
            best = _nearest_audio_boundary(orig_start, exclude_zero=True)
            if best is not None:
                new_start = round(best, 2)

        # Check if end_time matches a shared boundary
        for orig_t, snapped_t in shared_boundaries.items():
            if abs(orig_end - orig_t) < 0.5:
                new_end = snapped_t
                break
        else:
            # Not a shared boundary — snap independently
            best = _nearest_audio_boundary(orig_end)
            if best is not None:
                new_end = round(best, 2)

        # Apply snaps only if the resulting segment stays long enough
        if new_end - new_start >= min_duration:
            seg["start_time"] = new_start
            seg["end_time"] = new_end
        elif new_start != orig_start and orig_end - new_start >= min_duration:
            seg["start_time"] = new_start
        elif new_end != orig_end and new_end - orig_start >= min_duration:
            seg["end_time"] = new_end

    return segments


def _add_missed_boundaries(segments, audio_boundaries, min_confidence=0.7):
    """
    Add boundaries detected by audio that chord/lyrics missed.
    Only add if confidence > threshold and it falls inside an existing segment.
    """
    new_segments = []
    for seg in segments:
        # Find audio boundaries that fall inside this segment
        inner_bounds = [
            b for b in audio_boundaries
            if seg["start_time"] + 5.0 < b["time"] < seg["end_time"] - 5.0
            and b["confidence"] >= min_confidence
        ]

        if not inner_bounds:
            new_segments.append(seg)
            continue

        # Split at the strongest boundary
        best = max(inner_bounds, key=lambda b: b["confidence"])
        split_time = round(best["time"], 2)

        seg1 = dict(seg)
        seg1["end_time"] = split_time

        seg2 = dict(seg)
        seg2["start_time"] = split_time
        seg2["_pattern"] = seg["_pattern"] + "'"

        new_segments.append(seg1)
        new_segments.append(seg2)

    return new_segments


# ==============================================================================
# Lyrics overlay
# ==============================================================================

def _overlay_lyrics_labels(segments, lyrics_result, lyrics):
    """
    Use lyrics repetition data to identify chorus sections.
    If a segment contains repeated lyrics, mark it as chorus.
    """
    chorus_sections = [
        s for s in lyrics_result.get("sections", [])
        if s.get("name") == "Chorus"
    ]

    if not chorus_sections:
        return segments

    for seg in segments:
        for cs in chorus_sections:
            # Check overlap: does this chorus section overlap significantly with segment?
            overlap_start = max(seg["start_time"], cs["start_time"])
            overlap_end = min(seg["end_time"], cs["end_time"])
            overlap = overlap_end - overlap_start
            seg_duration = seg["end_time"] - seg["start_time"]
            chorus_duration = cs["end_time"] - cs["start_time"]

            # Require strong overlap: >50% of chorus OR >50% of segment
            if overlap > 0 and (overlap > chorus_duration * 0.5 or overlap > seg_duration * 0.5):
                seg["_lyrics_chorus"] = True
                break

    return segments


# ==============================================================================
# Section labeling
# ==============================================================================

def _apply_labeling_rules(segments, lyrics, duration):
    """
    Apply labeling rules in priority order:
    1. Before first lyrics -> Intro
    2. After last lyrics (>10s of music remains) -> Outro
    3. Contains repeated lyrics -> Chorus
    4. Most common chord pattern (excl. intro/outro) -> Verse
    5. Pattern appears only once AND not verse/chorus -> Bridge
    6. Instrumental section between vocal sections -> Solo/Instrumental
    """
    if not segments:
        return segments

    # Determine lyrics time range
    first_lyric_time = None
    last_lyric_time = None
    if lyrics:
        times = []
        for w in lyrics:
            t = w.get("start_time", w.get("start", None))
            if t is not None:
                times.append(float(t))
        end_times = []
        for w in lyrics:
            t = w.get("end_time", w.get("end", None))
            if t is not None:
                end_times.append(float(t))
        if times:
            first_lyric_time = min(times)
        if end_times:
            last_lyric_time = max(end_times)

    # --- Step 1: Mark Intro (everything before first lyric timestamp) ---
    # Bug fix #1: Intro boundary = first lyric start time, not audio features.
    # Split any segment that straddles the first lyric time so the intro ends
    # exactly when vocals begin.
    intro_patterns = set()
    if first_lyric_time is not None and first_lyric_time > 2.0:
        new_segments = []
        for seg in segments:
            if seg["name"] != "Unknown":
                new_segments.append(seg)
                continue
            # Segment entirely before first lyric
            if seg["end_time"] <= first_lyric_time + 0.5:
                seg["name"] = "Intro"
                intro_patterns.add(seg.get("_pattern", "?"))
                new_segments.append(seg)
            # Segment straddles first lyric — split it
            elif seg["start_time"] < first_lyric_time and seg["end_time"] > first_lyric_time + 0.5:
                intro_seg = dict(seg)
                intro_seg["name"] = "Intro"
                intro_seg["end_time"] = round(first_lyric_time, 2)
                intro_patterns.add(intro_seg.get("_pattern", "?"))
                new_segments.append(intro_seg)

                rest_seg = dict(seg)
                rest_seg["start_time"] = round(first_lyric_time, 2)
                new_segments.append(rest_seg)
            else:
                new_segments.append(seg)
        segments = new_segments
    else:
        # No lyrics or lyrics start very early — fall back to hint-based intro
        for seg in segments:
            if seg["name"] != "Unknown":
                continue
            if first_lyric_time is None and seg.get("_hint") == "intro":
                seg["name"] = "Intro"
                intro_patterns.add(seg.get("_pattern", "?"))

    # --- Step 2: Mark Outro (after last lyrics, >10s remaining) ---
    for seg in reversed(segments):
        if seg["name"] != "Unknown":
            continue
        if last_lyric_time is not None and seg["start_time"] >= last_lyric_time - 2.0:
            remaining = duration - seg["start_time"]
            if remaining > 10.0:
                seg["name"] = "Outro"
                break

    # --- Step 3: Mark Chorus (lyrics-identified — highest priority) ---
    for seg in segments:
        if seg["name"] != "Unknown":
            continue
        if seg.get("_lyrics_chorus"):
            seg["name"] = "Chorus"

    # --- Step 4: Build pattern statistics excluding intro/outro ---
    # Note: Don't propagate chorus labels to all segments with the same pattern.
    # A chord pattern can appear in both chorus and verse contexts.
    # Only the specific segments with lyrics chorus overlap are choruses.

    # Count patterns in the song body (not intro/outro)
    body_pattern_counts = Counter()
    body_pattern_bars = Counter()
    for seg in segments:
        pat = seg.get("_pattern", "?")
        if seg["name"] in ("Intro", "Outro"):
            continue
        if pat in intro_patterns:
            continue
        body_pattern_counts[pat] += 1
        body_pattern_bars[pat] += seg.get("_bar_count", 1)

    # Collect chord pattern analyzer hints (excluding intro patterns)
    hint_chorus = set()
    hint_bridge = set()
    for seg in segments:
        hint = seg.get("_hint", "")
        pat = seg.get("_pattern", "?")
        if hint == "chorus" and pat not in intro_patterns:
            hint_chorus.add(pat)
        elif hint == "bridge" and pat not in intro_patterns:
            hint_bridge.add(pat)

    # Patterns that EXCLUSIVELY appear in chorus-labeled segments
    # (don't include patterns that also appear in non-chorus segments)
    chorus_only_patterns = set()
    chorus_segment_patterns = set()
    non_chorus_patterns = set()
    for seg in segments:
        pat = seg.get("_pattern", "?")
        if seg["name"] == "Chorus":
            chorus_segment_patterns.add(pat)
        elif seg["name"] not in ("Intro", "Outro"):
            non_chorus_patterns.add(pat)
    chorus_only_patterns = chorus_segment_patterns - non_chorus_patterns

    # If no lyrics-based chorus found, use chord hints
    has_lyrics_chorus = any(seg["name"] == "Chorus" for seg in segments)
    if not has_lyrics_chorus and hint_chorus:
        # Use chord hints for chorus — only mark segments with those patterns
        for seg in segments:
            if seg["name"] == "Unknown" and seg.get("_pattern", "?") in hint_chorus:
                seg["name"] = "Chorus"
        chorus_only_patterns = hint_chorus

    # Determine verse pattern: most common body pattern excluding intro and chorus-only
    verse_pattern = None
    verse_candidates = [
        (p, bars) for p, bars in body_pattern_bars.most_common()
        if p not in chorus_only_patterns and p not in intro_patterns
    ]
    if verse_candidates:
        verse_pattern = verse_candidates[0][0]

    logger.info(f"Labeling: intro_patterns={intro_patterns}, "
                f"chorus_only_patterns={chorus_only_patterns}, "
                f"verse_pattern={verse_pattern}")

    # --- Step 5: Label verse sections ---
    for seg in segments:
        if seg["name"] != "Unknown":
            continue
        pat = seg.get("_pattern", "?")
        if pat == verse_pattern:
            seg["name"] = "Verse"
        elif pat in chorus_only_patterns:
            seg["name"] = "Chorus"

    # --- Step 6: Label remaining by pattern frequency, lyrics, and hints ---
    for seg in segments:
        if seg["name"] != "Unknown":
            continue
        pat = seg.get("_pattern", "?")
        hint = seg.get("_hint", "")

        # Check if segment contains lyrics
        has_lyrics = False
        if lyrics:
            for w in lyrics:
                t = w.get("start_time", w.get("start", 0))
                if seg["start_time"] <= float(t) <= seg["end_time"]:
                    has_lyrics = True
                    break

        if hint == "bridge" or pat in hint_bridge:
            seg["name"] = "Bridge"
        elif body_pattern_counts.get(pat, 0) == 1 and pat not in intro_patterns:
            if has_lyrics:
                seg["name"] = "Bridge"
            else:
                seg["name"] = "Instrumental"
        elif has_lyrics:
            seg["name"] = "Verse"
        else:
            seg["name"] = "Instrumental"

    return segments


# ==============================================================================
# Distinctive chord detection (supplements pattern analyzer)
# ==============================================================================

def _detect_chorus_from_distinctive_chords(segments, chords, lyrics):
    """
    Look for distinctive chord clusters that may indicate chorus sections
    missed by the pattern analyzer's windowed approach.

    Strategy: find chords that ONLY appear in specific time windows (not
    scattered throughout the song). These "section-exclusive" chords mark
    sections with unique harmonic content — often the chorus.
    """
    if not chords or len(chords) < 8:
        return segments

    # Parse chord events
    events = []
    for c in chords:
        name = c.get("chord", "N")
        if not name or name == "N":
            continue
        t = float(c.get("time", c.get("start", 0)))
        d = float(c.get("duration", 1.5))
        events.append({"chord": name, "time": t, "end": t + d})

    if len(events) < 8:
        return segments

    # Find chords that appear ONLY in specific compact time windows
    # (not scattered throughout the song). These mark distinct sections.
    chord_freq = Counter(e["chord"] for e in events)

    # For each chord, find its time clusters
    chord_clusters = {}
    for chord_name in chord_freq:
        chord_events = sorted([e for e in events if e["chord"] == chord_name],
                              key=lambda e: e["time"])
        if len(chord_events) < 2:
            continue

        # Group into clusters (events within 10s of each other)
        cls = []
        cur = [chord_events[0]]
        for i in range(1, len(chord_events)):
            if chord_events[i]["time"] - cur[-1]["end"] < 10.0:
                cur.append(chord_events[i])
            else:
                cls.append(cur)
                cur = [chord_events[i]]
        cls.append(cur)
        chord_clusters[chord_name] = cls

    # Find "section-exclusive" chords: appear in exactly 2+ compact clusters
    # (each cluster < 15s), not spread across the whole song
    section_exclusive = {}
    for chord_name, cls in chord_clusters.items():
        if len(cls) < 2:
            continue
        # Check that each cluster is compact
        all_compact = all(
            (cl[-1]["end"] - cl[0]["time"]) < 15.0
            for cl in cls
        )
        if not all_compact:
            continue
        section_exclusive[chord_name] = cls

    if not section_exclusive:
        return segments

    # Find groups of section-exclusive chords that co-occur in the same time windows
    # (same section). Build time windows from cluster overlaps.
    all_windows = []
    for chord_name, cls in section_exclusive.items():
        for cl in cls:
            start = cl[0]["time"]
            end = cl[-1]["end"]
            all_windows.append((start, end, chord_name))

    # Group overlapping windows into sections
    all_windows.sort()
    merged_windows = []
    if all_windows:
        cur_start, cur_end, cur_chords = all_windows[0][0], all_windows[0][1], {all_windows[0][2]}
        for start, end, chord_name in all_windows[1:]:
            if start <= cur_end + 2.0:  # overlap or very close
                cur_end = max(cur_end, end)
                cur_chords.add(chord_name)
            else:
                merged_windows.append((cur_start, cur_end, frozenset(cur_chords)))
                cur_start, cur_end, cur_chords = start, end, {chord_name}
        merged_windows.append((cur_start, cur_end, frozenset(cur_chords)))

    # Find windows with the same chord set (repeating sections)
    window_sigs = [w[2] for w in merged_windows]
    sig_counts = Counter(window_sigs)
    repeating_sigs = {sig for sig, cnt in sig_counts.items()
                      if cnt >= 2 and len(sig) >= 2}

    if not repeating_sigs:
        return segments

    # These are the chorus time ranges
    chorus_ranges = []
    for start, end, sig in merged_windows:
        if sig in repeating_sigs:
            chorus_ranges.append((start, end))

    if not chorus_ranges:
        return segments

    # Extend each chorus range backward to include the chord that starts
    # the section. The section-exclusive chords (e.g., C#m) may not be the
    # very first chord of the chorus — often there's a lead-in chord (e.g., D)
    # that transitions from the verse pattern. Look for the chord event
    # immediately before each chorus range and include it if it's close
    # (within one chord duration) and co-occurs in ALL chorus ranges.
    extended_ranges = []
    for cr_start, cr_end in chorus_ranges:
        # Find chord events just before this chorus range
        preceding = [e for e in events if e["end"] <= cr_start + 0.5 and e["time"] >= cr_start - 5.0]
        if preceding:
            preceding.sort(key=lambda e: e["time"], reverse=True)
            lead_chord = preceding[0]
            # Only extend if the lead chord is close (within ~2s before chorus)
            if cr_start - lead_chord["time"] < 3.0:
                extended_ranges.append((lead_chord["time"], cr_end, lead_chord["chord"]))
            else:
                extended_ranges.append((cr_start, cr_end, None))
        else:
            extended_ranges.append((cr_start, cr_end, None))

    # Only apply extension if the SAME lead chord appears before ALL chorus
    # instances (consistent pattern = same section structure)
    lead_chords = [lc for _, _, lc in extended_ranges if lc is not None]
    if lead_chords and len(set(lead_chords)) == 1 and len(lead_chords) == len(chorus_ranges):
        # All chorus instances have the same lead-in chord — extend all ranges
        chorus_ranges = [(start, end) for start, end, _ in extended_ranges]
        logger.info(f"Extended chorus ranges with lead-in chord '{lead_chords[0]}': {chorus_ranges}")
    # else: keep original ranges

    logger.info(f"Distinctive chord clusters (likely chorus): {chorus_ranges}")

    # Split segments at chorus boundaries
    new_segments = []
    for seg in segments:
        if not (seg["name"].startswith("Verse") or seg["name"] == "Instrumental"
               or seg["name"] == "Bridge"):
            new_segments.append(seg)
            continue

        # Check if any chorus range falls inside this verse segment
        splits = []
        seg_dur = seg["end_time"] - seg["start_time"]
        for cr_start, cr_end in chorus_ranges:
            # Chorus range must substantially overlap this segment
            overlap_start = max(seg["start_time"], cr_start)
            overlap_end = min(seg["end_time"], cr_end)
            overlap = overlap_end - overlap_start
            chorus_dur = cr_end - cr_start

            # Match if: overlap covers >60% of chorus OR >50% of segment
            if overlap > 0 and (overlap > chorus_dur * 0.6 or overlap > seg_dur * 0.5) and overlap > 2.0:
                splits.append((cr_start, cr_end))

        if not splits:
            new_segments.append(seg)
            continue

        # Sort splits by start time
        splits.sort()

        # Split the segment
        current_start = seg["start_time"]
        for cr_start, cr_end in splits:
            # Pre-chorus part (if >3s)
            if cr_start - current_start > 3.0:
                pre = dict(seg)
                pre["start_time"] = round(current_start, 2)
                pre["end_time"] = round(cr_start, 2)
                # Keep original name (Verse)
                new_segments.append(pre)

            # Chorus part
            chorus_seg = dict(seg)
            chorus_seg["name"] = "Chorus"
            chorus_seg["start_time"] = round(max(cr_start, current_start), 2)
            chorus_seg["end_time"] = round(min(cr_end, seg["end_time"]), 2)
            new_segments.append(chorus_seg)

            current_start = cr_end

        # Post-chorus remainder (if >3s)
        if seg["end_time"] - current_start > 3.0:
            post = dict(seg)
            post["start_time"] = round(current_start, 2)
            post["end_time"] = round(seg["end_time"], 2)
            new_segments.append(post)

    # Re-sort by start time
    new_segments.sort(key=lambda s: s["start_time"])

    return new_segments


# ==============================================================================
# Post-processing (preserved from original for test compatibility)
# ==============================================================================

def _postprocess(segments, duration):
    if not segments:
        return segments

    # Bug fix #4: Remove zero-length, negative-length, or sub-1-second segments
    segments = [s for s in segments
                if s["end_time"] > s["start_time"] and
                (s["end_time"] - s["start_time"]) >= 1.0]

    if not segments:
        return segments

    # Make segments contiguous (no gaps)
    for i in range(1, len(segments)):
        if segments[i]["start_time"] > segments[i - 1]["end_time"] + 0.1:
            segments[i]["start_time"] = segments[i - 1]["end_time"]
        elif segments[i]["start_time"] < segments[i - 1]["end_time"] - 0.1:
            segments[i]["start_time"] = segments[i - 1]["end_time"]

    # Second pass: filter out segments that became zero/sub-1s after contiguity fix
    segments = [s for s in segments
                if s["end_time"] > s["start_time"] and
                (s["end_time"] - s["start_time"]) >= 1.0]

    # Re-apply contiguity after removing collapsed segments
    for i in range(1, len(segments)):
        if segments[i]["start_time"] > segments[i - 1]["end_time"] + 0.1:
            segments[i]["start_time"] = segments[i - 1]["end_time"]

    # Merge adjacent same-label (but don't merge Chorus beyond 20s)
    MAX_CHORUS_DURATION = 20.0
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg["name"] == prev["name"] and seg["name"] not in ("Unknown",):
            # Bug fix #3: Don't merge choruses if it would exceed 20s
            proposed_dur = seg["end_time"] - prev["start_time"]
            if prev["name"] == "Chorus" and proposed_dur > MAX_CHORUS_DURATION:
                # Start a new segment instead of merging
                seg_copy = seg.copy()
                seg_copy["name"] = "Verse"  # likely verse/bridge after chorus
                merged.append(seg_copy)
            else:
                prev["end_time"] = seg["end_time"]
        else:
            merged.append(seg.copy())

    # Ensure full coverage
    if merged[0]["start_time"] > 0.5:
        merged[0]["start_time"] = 0.0
    if merged[-1]["end_time"] < duration - 0.5:
        merged[-1]["end_time"] = round(duration, 2)

    # Number repeated labels
    counts = Counter(s["name"] for s in merged)
    seen = {}
    for seg in merged:
        name = seg["name"]
        if counts[name] > 1 and name in ("Verse", "Chorus", "Bridge", "Solo", "Instrumental", "Section"):
            seen[name] = seen.get(name, 0) + 1
            seg["name"] = f"{name} {seen[name]}"

    # Clean internal keys
    for seg in merged:
        for k in [k for k in seg if k.startswith("_")]:
            del seg[k]

    return merged


# ==============================================================================
# Fallback (preserved from original for test compatibility)
# ==============================================================================

def _fallback_from_chords(chords):
    if not chords:
        return []
    last = chords[-1]
    total = last.get("time", 0) + last.get("duration", 3) + 2
    return [{"name": "Song", "start_time": 0.0, "end_time": round(total, 2)}]


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python structure_detector.py <audio_path> [chords.json] [lyrics.json]")
        sys.exit(1)

    audio = sys.argv[1]
    chords_data = None
    lyrics_data = None

    if len(sys.argv) >= 3:
        with open(sys.argv[2]) as f:
            chords_data = json.load(f)
    if len(sys.argv) >= 4:
        with open(sys.argv[3]) as f:
            lyrics_data = json.load(f)

    result = detect_structure(audio, chords_data, lyrics_data)
    print(json.dumps(result, indent=2))
