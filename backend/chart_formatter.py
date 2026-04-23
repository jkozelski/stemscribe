"""
Chart Formatter — transforms raw chord detection + Whisper word timestamps
into clean, Ultimate-Guitar-style chord charts.

Takes:
  - Chord events: [{"time": float, "duration": float, "chord": str, ...}, ...]
  - Word timestamps: [{"word": str, "start": float, "end": float}, ...]

Produces:
  - Chord library JSON format matching ~/stemscribe/backend/chord_library/ schema:
    {
      "title": str,
      "artist": str,
      "key": str,
      "capo": int,
      "source": "StemScriber AI",
      "chords_used": [str, ...],
      "sections": [
        {"name": str, "lines": [{"chords": str, "lyrics": str|null}, ...]}
      ]
    }

Section detection uses:
  - Lyric gaps (>3s pause = new section)
  - Chord progression fingerprinting (repeated patterns = same section type)
  - Chord density changes (instrumental vs vocal sections)
  - Repeated section labeling (Verse 1, Verse 2, etc.)

Usage:
    from chart_formatter import format_chart
    chart = format_chart(chord_events, word_timestamps, title="My Song", artist="My Band", key="Am")
"""

import json
import logging
import re
from typing import List, Dict, Optional, Tuple
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum gap (seconds) between words to split into separate lyric lines
LINE_GAP_THRESHOLD = 0.4

# Minimum gap (seconds) between lyric lines to consider a section break
SECTION_GAP_THRESHOLD = 4.0

# Maximum words per lyric line (prevents overly long lines)
MAX_WORDS_PER_LINE = 12

# Minimum words per lyric line (prevents tiny fragments — but set low to respect natural phrasing)
MIN_WORDS_PER_LINE = 2

# How close (seconds) a chord must be to a word to be placed above it
CHORD_SNAP_TOLERANCE = 0.8


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class _LyricLine:
    """A single line of lyrics with timing."""
    words: List[Dict]          # [{"word": str, "start": float, "end": float}]
    start_time: float = 0.0
    end_time: float = 0.0
    text: str = ""

    def __post_init__(self):
        if self.words:
            self.start_time = self.words[0]["start"]
            self.end_time = self.words[-1]["end"]
            self.text = " ".join(w["word"] for w in self.words)


@dataclass
class _Section:
    """A song section (Verse, Chorus, etc.)."""
    name: str
    lines: List[dict] = field(default_factory=list)  # [{"chords": str, "lyrics": str|null}]
    chord_pattern: str = ""     # fingerprint for section matching
    start_time: float = 0.0
    end_time: float = 0.0
    has_lyrics: bool = True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def format_chart(
    chord_events: List[Dict],
    word_timestamps: List[Dict],
    title: str = "Unknown",
    artist: str = "Unknown",
    key: str = "Unknown",
    capo: int = 0,
    stem_paths: Optional[Dict[str, str]] = None,
    grid: Optional[Dict] = None,
    bass_roots: Optional[List[Dict]] = None,
) -> Dict:
    """
    Format raw chord detection + Whisper word timestamps into a clean chord chart.

    Args:
        chord_events: List of {"time": float, "duration": float, "chord": str, ...}
                      (output from chord_detector_v10.detect_chords)
        word_timestamps: List of {"word": str, "start": float, "end": float}
                         (output from word_timestamps.get_word_timestamps)
        title: Song title
        artist: Artist name
        key: Detected key (e.g. "Am", "C", "F#m")
        capo: Capo position
        stem_paths: Optional dict of {stem_name: file_path}. When provided, instrumental
                    sections are labeled by the loudest non-vocal stem (e.g. "Piano Solo").

    Returns:
        Dict in chord library JSON format
    """
    if not chord_events:
        logger.warning("No chord events provided")
        return _empty_chart(title, artist, key, capo)

    # Sort inputs by time
    chords = sorted(chord_events, key=lambda c: c["time"])
    words = sorted(word_timestamps, key=lambda w: w["start"]) if word_timestamps else []

    # Filter out 'N' (no chord) events
    chords = [c for c in chords if c.get("chord", "N") != "N"]

    if not chords:
        logger.warning("All chord events were 'N' (no chord)")
        return _empty_chart(title, artist, key, capo)

    # Step 1a: Consolidate chord events — drop transient outliers and merge adjacent duplicates.
    # Without this, real verse loops like Cm-Gm-Dm-Am get polluted with one-off Gsus2/Baug/D#
    # events that are detector noise, and the formatter shows 20 chord labels per section
    # instead of the 4-chord vamp.
    before_count = len(chords)
    chords = _consolidate_chord_events(chords)
    logger.info(f"Consolidated chord events: {before_count} -> {len(chords)}")

    # Step 1: Split words into lyric lines
    lyric_lines = _split_into_lines(words)
    logger.info(f"Split {len(words)} words into {len(lyric_lines)} lyric lines")

    # Step 2: Identify instrumental regions (before lyrics, between sections, after lyrics)
    song_end = max(c["time"] + c.get("duration", 1.0) for c in chords)

    # Step 3: Build raw sections from lyric gaps and instrumental regions
    raw_sections = _build_raw_sections(lyric_lines, chords, song_end)
    logger.info(f"Built {len(raw_sections)} raw sections")

    # Step 3b: Pickup anacrusis correction. Vocalists often enter on the LAST chord
    # of a loop (e.g. Am, the 4th chord of Cm-Gm-Dm-Am) rather than the downbeat.
    # The lyric-gap section builder sees the first vocal note and labels THAT the
    # start of the verse — so the verse displays starting on Am when it should
    # start on Cm one bar later. Shift section starts forward to the loop downbeat
    # and move the pickup lyric(s) back to the preceding section.
    _align_sections_to_loop_downbeat(raw_sections, chords)

    # Bar-aligned chord grid. Preference order:
    #   1. bass_roots + detector quality (most reliable — bass pitch anchors the root)
    #   2. detector-only quantized to bars (when no bass stem / grid)
    #   3. raw detector events (when no grid at all)
    if grid and bass_roots:
        from processing.bass_root_extraction import combine_with_detector_quality, smooth_qualities
        bar_grid = combine_with_detector_quality(bass_roots, chords, grid)
        bar_grid = smooth_qualities(bar_grid)
        smoothed_count = sum(1 for b in bar_grid if "smoothed" in (b.get("source") or ""))
        logger.info(
            f"chart_formatter: bar grid built from bass+detector "
            f"({len(bar_grid)} bars, {smoothed_count} quality-smoothed)"
        )
    elif grid:
        bar_grid = _quantize_chords_to_bars(chords, grid)
        logger.info(f"chart_formatter: bar grid built from detector-only ({len(bar_grid)} bars)")
    else:
        bar_grid = []

    placement_chords = _bar_grid_to_chord_events(bar_grid) if bar_grid else chords

    # Step 4: Assign chords to lines within each section
    for section in raw_sections:
        _assign_chords_to_section(section, placement_chords)

    # Step 4b: Re-anchor rendered chord strings to bar_grid order. Without this
    # step, vocal sections that enter on a pickup bar render their chords in
    # vocal-onset order (e.g. "Alright" verse starting on Am instead of Cm),
    # because _assign_chords_to_section walks lyric lines and the lyric onset
    # trails the bar downbeat. The bar_grid already reflects the true
    # bass-anchored downbeat sequence, so we replay it over each section's
    # existing lyric lines. Runs only when bar_grid is populated.
    if bar_grid:
        try:
            _rebuild_section_chords_from_bar_grid(raw_sections, bar_grid)
        except Exception as e:
            logger.warning(f"bar-grid section rebuild failed, keeping vocal-aligned chords: {e}")

    # Step 5: Detect section types (Verse, Chorus, Bridge, etc.)
    _label_sections(raw_sections)

    # Step 6: Detect and label repeated sections
    _number_repeated_sections(raw_sections)

    # Step 6b: If stem audio is available, detect instrumental subsections hidden
    # inside vocal sections (Whisper often phantom-labels lyrics during a solo).
    # Then rename generic Solo/Interlude sections with the dominant instrument.
    if stem_paths:
        try:
            raw_sections = _split_vocal_tail_into_solo(raw_sections, stem_paths)
            _label_solo_instruments(raw_sections, stem_paths)
        except Exception as e:
            logger.warning(f"Solo-instrument labeling failed, keeping generic names: {e}")

    # Step 6c: If any section still carries lyric text under a Drum Break /
    # Bass Solo / Drum Solo / Guitar Solo / Piano Solo / Keys Solo name (a
    # Whisper phantom-lyric artefact), relabel it so a lyrical section never
    # reads as a "solo" to the musician.
    _rename_solo_sections_with_lyrics(raw_sections)

    # Step 6d: Collapse each section into the 4-bar slash-notation chord/lyric
    # chunk layout. Each section becomes one or more lines of:
    #   - chords: up to 4 chord slots where each slot is "<chord> ////"
    #     (chord name + four beat slashes for a 4/4 bar), producing the
    #     Nashville / lead-sheet idiom
    #     "Cm  ////  Gm  ////  Dm  ////  Am  ////".
    #   - lyrics: concatenation of all Whisper words whose start timestamp
    #     falls inside the chunk's 4-bar time window.
    # Section boundaries are honored: a chunk never spans two sections, so the
    # final chunk of a section may be shorter than 4 bars (partial chunks OK
    # at boundaries — we emit only the bars present, no rest-padding).
    if bar_grid:
        # Step 6c.5: Snap non-first section starts FORWARD to the nearest
        # phrase downbeat so sections line up to the 4-bar vamp rather than
        # to the vocal pickup bar. Without this, a verse that enters on a
        # pickup bar (bar 3 = Dm of a Cm-Gm-Dm-Am vamp) would read as if the
        # verse progression starts on Dm, which is musically wrong. Bars
        # absorbed into the previous section preserve their pickup lyrics
        # under the previous section's final chunk.
        try:
            _snap_sections_to_phrase_boundaries(raw_sections, bar_grid)
        except Exception as e:
            logger.warning(f"Phrase-boundary snap failed, keeping prior boundaries: {e}")

        try:
            _rebuild_sections_as_4bar_chunks(raw_sections, bar_grid, words)
        except Exception as e:
            logger.warning(f"4-bar chunk rebuild failed, keeping prior layout: {e}")
        # Rebuild may have changed has_lyrics (e.g. Whisper phantom words land
        # inside a previously-instrumental window); re-run the relabel so a
        # section that just gained real lyrics drops its Solo/Drum-Break name.
        _rename_solo_sections_with_lyrics(raw_sections)
        # Step 6e: Re-number Verse/Bridge after renaming. A "Bass Solo" that
        # got relabeled to "Verse" would otherwise produce two un-numbered
        # "Verse" entries back-to-back; this pass collapses them into
        # "Verse 1", "Verse 2", etc.
        _number_repeated_sections(raw_sections)
        # Step 6f: Merge adjacent sections that now share the same name.
        # Post-rename, a trailing "[Outro] [Outro]" pair (vocal outro followed
        # by a 1-bar instrumental tail) should render as a single Outro whose
        # final chunk is just the tail chord.
        raw_sections = _merge_adjacent_sections_with_same_name(raw_sections)

    # Step 7: Collect unique chords used
    chords_used = []
    seen = set()
    for c in chords:
        ch = c["chord"]
        if ch not in seen:
            seen.add(ch)
            chords_used.append(ch)

    # Step 7b: Strip internal keys now that all chord-placement passes are done.
    for sec in raw_sections:
        for line in sec.lines:
            line.pop("_words", None)
            line.pop("_start", None)
            line.pop("_end", None)

    # Step 8: Build output
    sections_out = []
    for sec in raw_sections:
        sections_out.append({
            "name": sec.name,
            "lines": sec.lines,
        })

    # bar_grid was computed up top so it could drive chord placement; include
    # it in the output for any renderer that wants the per-bar chord list.
    return {
        "title": title,
        "artist": artist,
        "key": key,
        "capo": capo,
        "source": "StemScriber AI",
        "chords_used": chords_used,
        "sections": sections_out,
        "bar_grid": bar_grid,
        "grid": {
            "tempo_bpm": (grid or {}).get("tempo_bpm"),
            "time_signature": (grid or {}).get("time_signature"),
            "bar_count": (grid or {}).get("bar_count"),
        } if grid else None,
    }


# ---------------------------------------------------------------------------
# Bar quantization — assign one chord per bar based on the tempo grid
# ---------------------------------------------------------------------------

def _bar_grid_to_chord_events(bar_grid: List[Dict]) -> List[Dict]:
    """
    Convert a bar-indexed grid into chord events, collapsing adjacent bars
    that share a chord. These events are then fed to the normal chord-over-word
    placement logic — but because their `time` values sit on bar downbeats,
    chord labels land on the first word of each new bar/phrase, not on whatever
    word the detector's raw timestamp happened to align with.

    Example:
        bar_grid = [(1,Cm,0..2), (2,Cm,2..4), (3,Gm,4..6), (4,Gm,6..8), (5,Dm,8..10)]
        returns  = [{time:0,dur:4,chord:Cm}, {time:4,dur:4,chord:Gm}, {time:8,dur:2,chord:Dm}]
    """
    if not bar_grid:
        return []
    events: List[Dict] = []
    first = bar_grid[0]
    current = {
        "time": first["start_time"],
        "duration": first["end_time"] - first["start_time"],
        "chord": first["chord"],
    }
    for bar in bar_grid[1:]:
        bar_dur = bar["end_time"] - bar["start_time"]
        if bar["chord"] == current["chord"]:
            current["duration"] += bar_dur
        else:
            events.append(current)
            current = {
                "time": bar["start_time"],
                "duration": bar_dur,
                "chord": bar["chord"],
            }
    events.append(current)
    return events


def _quantize_chords_to_bars(
    chords: List[Dict],
    grid: Dict,
) -> List[Dict]:
    """
    Map consolidated chord events onto the bar grid.

    For each bar, pick the chord whose total duration overlapping that bar is
    largest. Bars with no chord overlap inherit the previous bar's chord
    (chord is held). Output:
      [{"bar": 1, "chord": "Cm", "start_time": 38.73, "end_time": 40.99},
       {"bar": 2, "chord": "Gm", "start_time": 40.99, "end_time": 43.25},
       ...]

    When this grid is available downstream, the pickup-anacrusis problem
    disappears: a short pickup chord lives in its pickup bar and the next
    downbeat is the natural section start.
    """
    if not chords or not grid:
        return []
    downbeats = grid.get("downbeat_times") or []
    if len(downbeats) < 2:
        return []

    bars: List[Dict] = []
    song_end = grid.get("song_duration_sec") or (
        downbeats[-1] + (downbeats[-1] - downbeats[-2])
    )

    for i, start in enumerate(downbeats):
        end = downbeats[i + 1] if i + 1 < len(downbeats) else song_end
        # Accumulate overlap per chord within this bar
        overlaps: Dict[str, float] = {}
        for c in chords:
            c_start = c["time"]
            c_end = c_start + c.get("duration", 1.0)
            overlap = max(0.0, min(end, c_end) - max(start, c_start))
            if overlap > 0:
                overlaps[c["chord"]] = overlaps.get(c["chord"], 0.0) + overlap
        if overlaps:
            chord = max(overlaps.items(), key=lambda kv: kv[1])[0]
        elif bars:
            # No chord overlap — hold the previous bar's chord
            chord = bars[-1]["chord"]
        else:
            continue  # nothing yet, skip
        bars.append({
            "bar": i + 1,
            "chord": chord,
            "start_time": round(start, 3),
            "end_time": round(end, 3),
        })
    return bars


# ---------------------------------------------------------------------------
# Render-layer bar-grid anchoring — rebuild section chord lines from bar_grid
# so the chart follows the bass-anchored downbeat order instead of the
# vocal-onset order. Fixes the "Alright by Jamiroquai" verse starting on the
# pickup Am instead of the true Cm downbeat: the detector + bass pipeline
# already produced bar_grid = [Cm, Gm, Dm, Am, ...], we just have to render
# in that order rather than in word-timestamp order.
# ---------------------------------------------------------------------------

def _rebuild_section_chords_from_bar_grid(
    sections: List[_Section],
    bar_grid: List[Dict],
) -> None:
    """
    Rewrite each section's lines[].chords to reflect bar-grid chord order.

    For vocal sections: keep each line's lyrics intact, but rebuild the
    `chords` string in Ultimate-Guitar style — each chord character-aligned
    directly above the lyric word whose timestamp matches the chord's start.
    Consecutive duplicate chords are collapsed. The `segments` array (used by
    the lead-sheet renderer) is also rebuilt so the chord ORDER matches the
    bass-anchored downbeat sequence rather than the vocal-onset sequence.

    For instrumental sections: rebuild `lines` as 4-bar groups of bar-grid
    chord labels (matching the existing `_assign_chords_to_section` format).

    Mutates sections in place. No-op when bar_grid is empty.
    """
    if not bar_grid:
        return

    for sec in sections:
        sec_bars = [
            b for b in bar_grid
            if b["end_time"] > sec.start_time - 0.5
            and b["start_time"] < sec.end_time + 0.5
        ]
        if not sec_bars:
            continue

        if not sec.has_lyrics:
            # Instrumental: collapse consecutive duplicates, then group by 4
            chord_names: List[str] = []
            for b in sec_bars:
                if not chord_names or chord_names[-1] != b["chord"]:
                    chord_names.append(b["chord"])
            lines = []
            for i in range(0, len(chord_names), 4):
                batch = chord_names[i:i + 4]
                lines.append({"chords": "  ".join(batch), "lyrics": None})
            if lines:
                sec.lines = lines
            continue

        # Vocal: pair bars to existing lyric lines by time overlap.
        for line in sec.lines:
            l_start = line.get("_start")
            l_end = line.get("_end")
            if l_start is None or l_end is None:
                # Lyric-only line without timing (shouldn't normally happen
                # post-_assign_chords_to_section which strips _start/_end);
                # leave the line alone.
                continue

            # Bars whose midpoint falls inside this lyric line's window.
            line_bars = []
            for b in sec_bars:
                mid = 0.5 * (b["start_time"] + b["end_time"])
                if l_start - 0.3 <= mid <= l_end + 0.3:
                    line_bars.append(b)

            # If no bar midpoint lands inside the line, fall back to any bar
            # that overlaps the line window at all.
            if not line_bars:
                for b in sec_bars:
                    if b["end_time"] > l_start and b["start_time"] < l_end:
                        line_bars.append(b)

            if not line_bars:
                continue

            # Collapse consecutive duplicates into (chord, start_time) pairs,
            # preserving each chord's downbeat time so we can character-align
            # it over the correct lyric word.
            chord_events_line: List[Dict] = []
            for b in line_bars:
                if not chord_events_line or chord_events_line[-1]["chord"] != b["chord"]:
                    chord_events_line.append({
                        "chord": b["chord"],
                        "time": b["start_time"],
                        "duration": b["end_time"] - b["start_time"],
                    })
                else:
                    # Extend duration of the held chord
                    prev = chord_events_line[-1]
                    prev["duration"] = (b["end_time"] - prev["time"])

            words_data = line.get("_words", [])
            lyrics_text = line.get("lyrics", "") or ""

            # Ultimate-Guitar style alignment: each chord sits directly above
            # the word/syllable whose onset matches the chord's downbeat.
            positions = _place_chords_on_words(chord_events_line, words_data, lyrics_text)
            line["chords"] = _build_chord_line(positions, len(lyrics_text))

            # Rebuild segments so the lead-sheet renderer (practice.html
            # preferred path) walks chords in bar-grid/downbeat order with
            # the words spoken under each chord's time window.
            if words_data:
                durations = [max(0.1, c["duration"]) for c in chord_events_line]
                unit_bar = min(durations) if durations else 1.0
                new_segments = []
                for c in chord_events_line:
                    c_start = c["time"]
                    c_end = c_start + max(0.1, c["duration"])
                    seg_words = [
                        {"w": w["word"], "t": round(w["start"], 3)}
                        for w in words_data
                        if c_start - 0.1 <= w["start"] < c_end
                    ]
                    bars = max(1, min(8, round(c["duration"] / unit_bar)))
                    new_segments.append({
                        "chord": c["chord"],
                        "start": round(c_start, 3),
                        "end":   round(c_end, 3),
                        "duration": round(c["duration"], 3),
                        "bars":    bars,
                        "words":   seg_words,
                    })
                line["segments"] = new_segments


# ---------------------------------------------------------------------------
# 4-bar slash-notation chunk layout
#
# Collapses each section's lines into one or more 4-bar chord-sheet lines.
# Each bar renders as the chord name followed by four slashes (one slash per
# beat of a 4/4 measure). All Whisper words whose start_time falls in that
# 4-bar time window flow underneath as the lyric line. This is the standard
# Nashville / lead-sheet idiom that reading musicians already know.
#
# Example for Alright's verse (Cm Gm Dm Am):
#
#     Cm  ////  Gm  ////  Dm  ////  Am  ////
#     I've been seeing angels turn into devils in my mind
#
# Section boundaries are honored: a chunk never spans two sections, so the
# final chunk of a section may be shorter than 4 bars (partial chunks at
# boundaries emit only the bars that are present — no rest-padding).
# ---------------------------------------------------------------------------

# Bars per rendered chord line.
_CHORDS_PER_LINE = 4

# Chord-name field width inside a slot. Fits "Cm", "F#m", "Am" cleanly;
# longer names like "Cm7b5" overflow but still render legibly (we always
# keep at least one space before the slashes).
_CHORD_NAME_WIDTH = 4

# Separator between bar-slots ("////" of one bar + "  " + next chord name).
_SLOT_SEPARATOR = "  "


def _format_slash_bar(chord: str) -> str:
    """Render one bar as '<chord-name-padded> ////' (chord + 4 beat slashes).

    Chord name is left-justified to `_CHORD_NAME_WIDTH` so typical 2-3 char
    chord names line up in even columns. Longer chord names overflow into
    their padding but always keep at least one space before the slashes so
    the bar stays unambiguous.

    Example:
        "Cm"    -> "Cm  ////"
        "F#m"   -> "F#m ////"
        "Cm7b5" -> "Cm7b5 ////"   (overflow, keeps one space)
    """
    name = (chord or "").strip()
    if len(name) < _CHORD_NAME_WIDTH:
        name = name.ljust(_CHORD_NAME_WIDTH)
    else:
        # Overflow: keep one space before the slashes
        name = name + " "
    return f"{name}////"


def _build_slash_chord_line(chord_names: List[str]) -> str:
    """Render up to `_CHORDS_PER_LINE` chord names as a 4-bar slash line.

    Example:
        ["Cm", "Gm", "Dm", "Am"] -> "Cm  ////  Gm  ////  Dm  ////  Am  ////"

    Empty input -> empty string. Partial chunks (1-3 chords at a section
    boundary) render with only the bars that are present; no padding to 4.
    """
    if not chord_names:
        return ""
    return _SLOT_SEPARATOR.join(_format_slash_bar(c) for c in chord_names)


def _snap_sections_to_phrase_boundaries(
    sections: List[_Section],
    bar_grid: List[Dict],
    phrase_bars: int = _CHORDS_PER_LINE,
) -> None:
    """Snap non-first section starts forward to the nearest phrase downbeat.

    Pop songs with a 4-bar vamp (Cm-Gm-Dm-Am repeating every 4 bars) can have
    their vocal sections enter on a pickup bar mid-phrase. The stem-RMS
    section labeler uses vocal onset as the section boundary, which produces
    "Verse 1 starts on Dm" when musically the verse progression is still
    Cm-Gm-Dm-Am. This pass re-aligns each non-first section's start time
    to the bar_grid phrase grid so the chunker emits full 4-bar phrases
    starting on the vamp's downbeat chord.

    Algorithm:
      1. Treat bar_grid[0] as phrase beat 1. Phrase downbeats are at bar
         indices 0, phrase_bars, 2*phrase_bars, ...
      2. For each section after the first, find the bar index whose midpoint
         is closest to the section's start_time.
      3. If that index is NOT on a phrase downbeat (index % phrase_bars != 0),
         snap forward to the next phrase downbeat.
      4. Update the section's start_time to that bar's start_time; push the
         preceding section's end_time to match. Bars absorbed into the
         preceding section keep their chords AND any pickup lyrics that fell
         in those bars (chunker reads those via word timestamps).

    Runs in-place. No-op when bar_grid is empty or there are fewer than 2
    sections. Safe: if a snapped index would exceed len(bar_grid), leave the
    section boundary alone rather than push it off the end.
    """
    if not bar_grid or len(sections) < 2 or phrase_bars <= 1:
        return

    def _bar_index_for_time(t: float) -> int:
        """Index in bar_grid of the bar whose midpoint is closest to t."""
        best_i = 0
        best_d = float("inf")
        for i, b in enumerate(bar_grid):
            mid = 0.5 * (b["start_time"] + b["end_time"])
            d = abs(mid - t)
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    for i in range(1, len(sections)):
        sec = sections[i]
        first_idx = _bar_index_for_time(sec.start_time)
        offset = first_idx % phrase_bars
        if offset == 0:
            continue  # already on a phrase downbeat
        snap_forward_by = phrase_bars - offset
        new_idx = first_idx + snap_forward_by
        if new_idx >= len(bar_grid):
            # Would push off the end; leave this boundary alone. The
            # remaining short tail stays with the preceding section.
            continue
        new_start = bar_grid[new_idx]["start_time"]
        if new_start >= sec.end_time:
            # Whole section would vanish — skip the snap.
            continue
        # Extend previous section to absorb the pickup bars.
        sections[i - 1].end_time = new_start
        sec.start_time = new_start


def _rebuild_sections_as_4bar_chunks(
    sections: List[_Section],
    bar_grid: List[Dict],
    words: List[Dict],
) -> None:
    """Rewrite `sections[].lines` in-place as 4-bar slash-notation chunks.

    For each section, finds the bars from `bar_grid` whose midpoint falls
    inside the section's time window, chunks them in groups of up to 4, and
    for each chunk emits ONE `{chords, lyrics, segments}` line where:
      - chords: "<Chord1>  ////  <Chord2>  ////  <Chord3>  ////  <Chord4>  ////"
        (partial chunks at section boundaries emit fewer slots). ONE chord
        per bar — no consecutive-duplicate collapsing, so the musician sees
        each bar's label even when a chord is held for multiple bars.
      - lyrics: concatenation of all `words` whose `start` falls inside the
        chunk's time window, joined by single spaces. `None` when the chunk
        is fully instrumental.
      - segments: per-bar word groups for the lead-sheet renderer
        (practice.html's preferred path), one segment per bar in the chunk.

    Section boundaries are honored: a chunk never spans two sections, so the
    final chunk of a section may be shorter than 4 bars (e.g. a 3-bar Intro
    emits one 3-bar chunk). No-op when `bar_grid` is empty. Sections with no
    bars overlapping their time window are left unchanged (rare).

    The prior UG-style per-syllable chord-over-word alignment (via
    `_place_chords_on_words` + `_build_chord_line`) is bypassed here: the
    slash-notation idiom communicates bar and beat position without needing
    character-column alignment with the lyric text below.
    """
    if not bar_grid:
        return

    words_sorted = sorted(words or [], key=lambda w: w.get("start", 0.0))

    for sec in sections:
        sec_start = sec.start_time
        sec_end = sec.end_time if sec.end_time > sec_start else sec_start + 0.001

        # Bars whose midpoint lies inside the section window. The 0.3s slop
        # tolerates rounding noise at section boundaries.
        sec_bars: List[Dict] = []
        for b in bar_grid:
            mid = 0.5 * (b["start_time"] + b["end_time"])
            if sec_start - 0.3 <= mid <= sec_end + 0.3:
                sec_bars.append(b)

        if not sec_bars:
            # Keep whatever the earlier passes produced for this section
            # (usually empty / an instrumental placeholder).
            continue

        new_lines: List[dict] = []
        for i in range(0, len(sec_bars), _CHORDS_PER_LINE):
            chunk = sec_bars[i:i + _CHORDS_PER_LINE]
            chunk_start = chunk[0]["start_time"]
            chunk_end = chunk[-1]["end_time"]

            # Slash notation: one chord per bar, four slashes per bar.
            chord_names = [b["chord"] for b in chunk]
            chords_line = _build_slash_chord_line(chord_names)

            # Words whose onset lies inside the chunk's time window. A small
            # leading tolerance (0.1s) catches words that begin on the bar
            # downbeat but whose Whisper timestamp rounds slightly low.
            chunk_word_objs = [
                w for w in words_sorted
                if chunk_start - 0.1 <= w.get("start", 0.0) < chunk_end
            ]
            lyrics_line = " ".join(
                (w.get("word") or "") for w in chunk_word_objs
                if (w.get("word") or "").strip()
            ).strip()

            # Per-bar word segments, one per bar in the chunk, for the
            # lead-sheet renderer that walks segments in bar-grid order.
            segments: List[dict] = []
            for b in chunk:
                b_start = b["start_time"]
                b_end = b["end_time"]
                seg_words = [
                    {"w": w.get("word", ""), "t": round(w.get("start", 0.0), 3)}
                    for w in chunk_word_objs
                    if b_start - 0.1 <= w.get("start", 0.0) < b_end
                ]
                segments.append({
                    "chord": b["chord"],
                    "start": round(b_start, 3),
                    "end":   round(b_end, 3),
                    "duration": round(b_end - b_start, 3),
                    "bars":    1,
                    "words":   seg_words,
                })

            new_lines.append({
                "chords": chords_line,
                "lyrics": lyrics_line if lyrics_line else None,
                "segments": segments,
            })

        # Shift pickup words across chunk boundaries: a word sung near the
        # end of line N that begins a phrase completed on line N+1 should
        # start line N+1, not end line N. Musically this is the intra-section
        # analogue of the phrase-boundary snap (which handles Section→Section
        # pickups). Without it, verses read "... love. You," | "give me light"
        # when the phrase is actually "You give me light" starting on the Cm
        # downbeat of the next line.
        _shift_pickup_words_across_lines(new_lines)

        sec.lines = new_lines
        # Reflect whether the section actually carries lyric text post-chunk.
        sec.has_lyrics = any(ln.get("lyrics") for ln in new_lines)


def _shift_pickup_words_across_lines(
    lines: List[dict],
    tail_window_sec: float = 0.5,
    gap_window_sec: float = 1.0,
) -> None:
    """Move last word of line N to line N+1 when it's a musical pickup.

    Heuristic: if line N's final segment ends with a word sung near the bar
    end (start within ``tail_window_sec`` of the segment's end) AND line N+1's
    first segment begins with a word sung close in time (gap within
    ``gap_window_sec``), treat the word as a pickup anacrusis for line N+1's
    phrase and shift it forward. Runs in-place, updates both the per-segment
    ``words`` arrays and each line's ``lyrics`` string.

    Only shifts ONE word per boundary per pass — avoids cascading shifts that
    could re-order entire phrases. If the heuristic's thresholds don't fire,
    the line boundary stays where it was (safe default).
    """
    if not lines or len(lines) < 2:
        return

    for i in range(len(lines) - 1):
        cur = lines[i]
        nxt = lines[i + 1]
        cur_segs = cur.get("segments") or []
        nxt_segs = nxt.get("segments") or []
        if not cur_segs or not nxt_segs:
            continue
        cur_last_seg = cur_segs[-1]
        nxt_first_seg = nxt_segs[0]
        cur_last_words = cur_last_seg.get("words") or []
        nxt_first_words = nxt_first_seg.get("words") or []
        if not cur_last_words or not nxt_first_words:
            continue

        last_w = cur_last_words[-1]
        first_w_next = nxt_first_words[0]
        try:
            last_t = float(last_w.get("t", 0.0))
            first_t_next = float(first_w_next.get("t", 0.0))
            seg_end = float(cur_last_seg.get("end", 0.0))
        except (TypeError, ValueError):
            continue

        # Gap to first word of next line must be short (tight phrase cluster).
        tight_cluster = (first_t_next - last_t) <= gap_window_sec

        # Heuristic A (time-based): word sung near the end of its segment AND
        # tight cluster with next line. Catches straight pickup anacrusis.
        near_segment_end = last_t >= seg_end - tail_window_sec

        # Heuristic B (phrase-based): the word BEFORE the last word ends in a
        # sentence-terminating punctuation (. ! ? ;), meaning the last word
        # starts a new lyric phrase. Lyric-phrase starts belong on the next
        # chord line's downbeat regardless of how the Whisper timestamps
        # happened to land. This is the "oh that sounds like the start of
        # the next verse — put it there" rule.
        starts_new_phrase = False
        if len(cur_last_words) >= 2:
            prev_w_text = (cur_last_words[-2].get("w") or "").strip()
            if prev_w_text.endswith((".", "!", "?", ";")):
                starts_new_phrase = True

        # Only shift when at least one trigger fires AND the cluster is tight.
        if not tight_cluster:
            continue
        if not (near_segment_end or starts_new_phrase):
            continue
        # Shift.
        cur_last_words.pop()
        nxt_first_words.insert(0, last_w)

    # Rebuild each line's lyrics string from the updated per-segment words.
    for ln in lines:
        all_words = []
        for seg in ln.get("segments") or []:
            for w in seg.get("words") or []:
                txt = (w.get("w") or "").strip()
                if txt:
                    all_words.append(txt)
        joined = " ".join(all_words).strip()
        ln["lyrics"] = joined if joined else None


def _rename_solo_sections_with_lyrics(sections: List[_Section]) -> None:
    """Relabel sections that still read as a "solo" but contain lyric text.

    Whisper sometimes transcribes lyrics over instrumental passages (bleed
    from previous phrase, ad-libs, or phantom words under a solo), and the
    stem-RMS solo-labeler can then tag a section as "Drum Break" / "Bass
    Solo" while lyric text remains in the line dicts. A chord chart that
    shows "Drum Break" above real lyrics is the wrong affordance — the
    musician needs to see it as a normal vocal section.

    Fix: if any line in such a section has non-empty lyric text, rename it
    to Verse/Chorus/Bridge based on nearby sections' labels. Solo-style
    names (Drum Break, Bass Solo, Guitar Solo, Piano Solo, Keys Solo,
    Interlude, Outro) are reserved for sections with zero lyric lines.
    """
    solo_names = {
        "Drum Break", "Drums Solo", "Drum Solo",
        "Bass Solo", "Guitar Solo", "Piano Solo", "Keys Solo",
        "Solo", "Interlude",
        # Outro is also an instrumental label in this context — a tail
        # section that unexpectedly carries lyrics is a vocal outro, which
        # we'd rather call Verse (or Chorus if the chord pattern matches)
        # than Outro.
        "Outro",
    }

    def _nearest_vocal_label(idx: int) -> str:
        """Borrow a vocal label from an adjacent section if one is available."""
        # Prefer the preceding vocal section — avoids labeling a tail as
        # Chorus when the song just left a Chorus into a (phantom-) solo.
        for j in range(idx - 1, -1, -1):
            nm = sections[j].name or ""
            base = nm.split(" ")[0]
            if base in ("Verse", "Chorus", "Bridge", "Pre-Chorus"):
                return base
        for j in range(idx + 1, len(sections)):
            nm = sections[j].name or ""
            base = nm.split(" ")[0]
            if base in ("Verse", "Chorus", "Bridge", "Pre-Chorus"):
                return base
        return "Verse"

    # Identify the last section that still carries lyric text so a tail solo
    # section with residual Whisper lyrics becomes "Outro" (not "Bridge" or
    # "Verse"). This matches the directive's expectation that the final
    # chunk of a song ends under an Outro label.
    last_lyric_idx = -1
    for i, sec in enumerate(sections):
        if any((ln.get("lyrics") or "").strip()
               for ln in (sec.lines or []) if isinstance(ln, dict)):
            last_lyric_idx = i

    for i, sec in enumerate(sections):
        if not sec.name:
            continue
        base = sec.name.split(" (")[0]
        # Strip trailing numbering like "Bass Solo 2"
        base_stripped = base.rsplit(" ", 1)[0] if base.rsplit(" ", 1)[-1].isdigit() else base
        if base not in solo_names and base_stripped not in solo_names:
            continue
        has_lyric_text = any(
            (ln.get("lyrics") or "").strip()
            for ln in (sec.lines or []) if isinstance(ln, dict)
        )
        if not has_lyric_text:
            continue
        sec.has_lyrics = True
        # Tail-of-song vocal section → Outro. Interior → borrow nearest
        # vocal label from a neighbouring section.
        if i == last_lyric_idx:
            sec.name = "Outro"
        else:
            sec.name = _nearest_vocal_label(i)


def _merge_adjacent_sections_with_same_name(
    sections: List[_Section],
) -> List[_Section]:
    """Merge consecutive sections that share the same display name.

    After solo-rename + re-numbering, it's common to see a trailing vocal
    section and a 1-bar instrumental tail both labeled "Outro". The chart
    reads cleaner when they render as a single Outro section whose last
    chunk is the instrumental tail, so we concatenate line lists whenever two
    adjacent sections carry identical names. Returns a new list; the
    original sections are not mutated.
    """
    if not sections:
        return sections
    merged: List[_Section] = [sections[0]]
    for sec in sections[1:]:
        prev = merged[-1]
        if prev.name == sec.name:
            prev.lines = list(prev.lines) + list(sec.lines)
            prev.end_time = sec.end_time
            prev.has_lyrics = prev.has_lyrics or sec.has_lyrics
        else:
            merged.append(sec)
    return merged


# ---------------------------------------------------------------------------
# Post-rebuild consolidation (unused — 4-bar-chunk rebuild handles
# fragmentation and held-chord rendering directly). Kept here as reference
# if a future layout needs to pre-process sections before chunking.
# ---------------------------------------------------------------------------


def _consolidate_short_lyric_lines(
    sections: List[_Section],
    max_lines_before_consolidation: int = 8,
    max_avg_chars: int = 30,
    target_line_chars: int = 60,
) -> None:
    """Merge adjacent short lyric lines so a verse doesn't render as 14 stubs.

    Whisper's word timestamps plus the short LINE_GAP_THRESHOLD can split a
    single sung phrase ("I need your love, I need your love, I need your love,")
    across multiple lines. Post-chord-placement, collapse those stubs so a
    section reads as 4-8 lines of ~60 chars instead of 14 lines of ~15 chars.

    Rules:
      - Skip instrumental sections (has_lyrics=False or no lyric text).
      - Only act when a section has > `max_lines_before_consolidation` lines
        AND the average lyric length is < `max_avg_chars`.
      - Greedily merge adjacent lines while the merged lyric stays under
        `target_line_chars`; break when adding the next line would overflow.
      - Concatenate chord strings with two spaces as the separator and
        collapse consecutive identical chord names (e.g. "Cm  Cm  Gm" -> "Cm  Gm").
      - Segments (if present) concatenate as-is. `notes` is preserved from the
        first line of each merged group.
    """
    for sec in sections:
        lines = sec.lines or []
        if not lines or len(lines) <= max_lines_before_consolidation:
            continue
        # Only consolidate lyric-bearing sections
        lyric_lines = [ln for ln in lines if isinstance(ln, dict) and (ln.get("lyrics") or "").strip()]
        if not lyric_lines:
            continue
        avg_chars = sum(len(ln.get("lyrics") or "") for ln in lyric_lines) / max(1, len(lyric_lines))
        if avg_chars >= max_avg_chars:
            continue

        merged: List[dict] = []
        for ln in lines:
            if not isinstance(ln, dict):
                merged.append(ln)
                continue
            ly = (ln.get("lyrics") or "").strip()
            if not merged or not ly:
                merged.append(dict(ln))
                continue
            prev = merged[-1]
            prev_ly = (prev.get("lyrics") or "").strip()
            # Only merge when the previous line also has lyrics and together
            # we stay under the target width (add +1 for the joining space).
            if not prev_ly or len(prev_ly) + 1 + len(ly) > target_line_chars:
                merged.append(dict(ln))
                continue
            # Merge ly -> prev
            new_lyrics = prev_ly + " " + ly
            prev_chords = (prev.get("chords") or "").strip()
            this_chords = (ln.get("chords") or "").strip()
            if prev_chords and this_chords:
                combined = prev_chords + "  " + this_chords
            else:
                combined = prev_chords or this_chords
            # Collapse consecutive identical chord tokens ("Cm  Cm  Gm" -> "Cm  Gm")
            tokens = [t for t in combined.split() if t]
            deduped: List[str] = []
            for t in tokens:
                if not deduped or deduped[-1] != t:
                    deduped.append(t)
            prev["chords"] = "  ".join(deduped)
            prev["lyrics"] = new_lyrics
            # Merge segments if both sides have them
            prev_segs = prev.get("segments") or []
            this_segs = ln.get("segments") or []
            if prev_segs or this_segs:
                prev["segments"] = list(prev_segs) + list(this_segs)

        sec.lines = merged


def _compact_held_chord_lines(sections: List[_Section]) -> None:
    """Collapse "Gm  Gm  Gm  Gm" style repetition into "Gm (x4)".

    Legacy helper from earlier layouts that emitted bare chord names per
    bar. Under the current 4-bar slash-notation layout each bar renders as
    "<chord> ////" (tokens are [chord, "////", chord, "////", ...]), so this
    pass is effectively a no-op because `////` always interrupts the run of
    identical chord tokens. Retained in case a future layout reverts to a
    bare-chord chord line that would benefit from held-chord compaction.

    For each chord line, collapse runs of 3+ identical consecutive chord
    tokens into "{name} (x{count})". Shorter runs (1-2 repeats) are left
    alone so a straight Cm-Gm-Dm-Am verse vamp still reads as the full
    progression.
    """
    for sec in sections:
        for line in sec.lines or []:
            if not isinstance(line, dict):
                continue
            chords = (line.get("chords") or "").strip()
            if not chords:
                continue
            tokens = [t for t in chords.split() if t]
            if len(tokens) < 3:
                continue
            compact: List[str] = []
            i = 0
            n = len(tokens)
            while i < n:
                run_end = i + 1
                while run_end < n and tokens[run_end] == tokens[i]:
                    run_end += 1
                run_len = run_end - i
                if run_len >= 3:
                    compact.append(f"{tokens[i]} (x{run_len})")
                else:
                    compact.extend(tokens[i:run_end])
                i = run_end
            line["chords"] = "  ".join(compact)


# ---------------------------------------------------------------------------
# Step 1a: Consolidate chord events (drop outliers + merge adjacent duplicates)
# ---------------------------------------------------------------------------

_SUS_AUG_STRIP_RE = re.compile(r'(sus[24]?|aug\d*|\+|add\d+|#5|b5)', re.IGNORECASE)


def _simplify_uncommon_quality(chord: str) -> str:
    """
    Strip chord-quality decorations that are almost always detector noise:
    sus2/sus4, aug/+, add9/add11, altered fifths. Leaves major/minor/7/maj7/m7
    and dim alone (those are real jazz chord qualities).

    Examples:
      "Csus2"  -> "C"
      "Gaug"   -> "G"
      "Dsus4"  -> "D"
      "Baug"   -> "B"
      "Cm7"    -> "Cm7"  (unchanged)
      "Fdim"   -> "Fdim" (unchanged)
    """
    if not chord or chord == "N":
        return chord
    simplified = _SUS_AUG_STRIP_RE.sub('', chord)
    # If stripping left just a root (e.g. "Cm" -> "Cm", "C" -> "C"), keep it.
    return simplified or chord


def _consolidate_chord_events(
    chords: List[Dict],
    min_count: int = 4,
    min_duration: float = 1.0,
    min_vocab_fraction: float = 0.03,
) -> List[Dict]:
    """
    Pre-pass cleanup of raw chord-detector output before section assignment.

    Four passes:
      0. Simplify uncommon qualities — strip sus/aug/add/altered-5 decorations
         that are almost always detector noise on real pop/rock audio.
      1. Drop transient outliers — chords whose label appears fewer than `min_count`
         times AND whose duration is shorter than `min_duration` seconds. Dropped
         duration folds into the preceding chord so the timeline stays continuous.
      2. Merge adjacent identical chords into a single longer event.
      3. Vocabulary pareto — drop any chord whose TOTAL duration is less than
         `min_vocab_fraction` of the song. A 4-minute song (240s) with a 3% floor
         means any chord summing to less than 7.2s gets dropped.
    """
    if not chords:
        return chords
    from collections import Counter

    # Pass 0: simplify qualities
    normalized: List[Dict] = []
    for c in chords:
        nc = dict(c)
        nc["chord"] = _simplify_uncommon_quality(c["chord"])
        normalized.append(nc)
    chords = normalized

    chord_counts = Counter(c["chord"] for c in chords)

    # Pass 1: drop rare+short outliers
    kept: List[Dict] = []
    for c in chords:
        dur = c.get("duration", 1.0)
        count = chord_counts[c["chord"]]
        if count < min_count and dur < min_duration and kept:
            prev = dict(kept[-1])
            prev["duration"] = prev.get("duration", 1.0) + dur
            kept[-1] = prev
        else:
            kept.append(c)

    # Pass 2: merge adjacent duplicates
    if not kept:
        return kept
    merged: List[Dict] = [dict(kept[0])]
    for c in kept[1:]:
        if c["chord"] == merged[-1]["chord"]:
            merged[-1]["duration"] = merged[-1].get("duration", 1.0) + c.get("duration", 1.0)
        else:
            merged.append(dict(c))

    # Pass 3: vocabulary pareto — drop chords with tiny total duration share
    if merged:
        total_dur = sum(c.get("duration", 1.0) for c in merged)
        if total_dur > 0:
            by_chord_dur: Dict[str, float] = {}
            for c in merged:
                by_chord_dur[c["chord"]] = by_chord_dur.get(c["chord"], 0.0) + c.get("duration", 1.0)
            keepable = {ch for ch, d in by_chord_dur.items() if d / total_dur >= min_vocab_fraction}
            pared: List[Dict] = []
            for c in merged:
                if c["chord"] in keepable:
                    pared.append(c)
                elif pared:
                    pared[-1] = dict(pared[-1])
                    pared[-1]["duration"] = pared[-1].get("duration", 1.0) + c.get("duration", 1.0)
                # else: drop (can't fold into anything)
            # Re-merge adjacents after drops
            if pared:
                final: List[Dict] = [dict(pared[0])]
                for c in pared[1:]:
                    if c["chord"] == final[-1]["chord"]:
                        final[-1]["duration"] = final[-1].get("duration", 1.0) + c.get("duration", 1.0)
                    else:
                        final.append(dict(c))
                return final
    return merged


# ---------------------------------------------------------------------------
# Step 1: Split words into lyric lines
# ---------------------------------------------------------------------------

def _split_into_lines(words: List[Dict]) -> List[_LyricLine]:
    """Split word timestamps into natural lyric lines based on timing gaps."""
    if not words:
        return []

    lines = []
    current_words = [words[0]]

    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        word_count = len(current_words)

        # Split on timing gap or max word count
        should_split = (
            gap >= LINE_GAP_THRESHOLD
            or word_count >= MAX_WORDS_PER_LINE
        )

        # Hard break at max words regardless of MIN_WORDS_PER_LINE
        force_split = word_count >= MAX_WORDS_PER_LINE

        if force_split or (should_split and word_count >= MIN_WORDS_PER_LINE):
            lines.append(_LyricLine(words=current_words))
            current_words = [words[i]]
        else:
            current_words.append(words[i])

    # Don't lose the last line
    if current_words:
        lines.append(_LyricLine(words=current_words))

    return lines


# ---------------------------------------------------------------------------
# Step 3: Build raw sections from lyric lines and chord events
# ---------------------------------------------------------------------------

def _build_raw_sections(
    lyric_lines: List[_LyricLine],
    chords: List[Dict],
    song_end: float,
) -> List[_Section]:
    """
    Group lyric lines into sections based on timing gaps.
    Also creates instrumental sections for chord-only regions.
    """
    if not lyric_lines:
        # No lyrics at all — one big instrumental section
        sec = _Section(name="Instrumental", has_lyrics=False, start_time=0.0, end_time=song_end)
        return [sec]

    sections = []

    # Check for intro (chords before first lyrics)
    first_lyric_time = lyric_lines[0].start_time
    intro_chords = [c for c in chords if c["time"] + c.get("duration", 0) < first_lyric_time - 0.5]
    if intro_chords and first_lyric_time > 3.0:
        sec = _Section(
            name="Intro",
            has_lyrics=False,
            start_time=intro_chords[0]["time"],
            end_time=first_lyric_time,
        )
        sections.append(sec)

    # Group lyric lines into sections based on gaps
    current_lines = [lyric_lines[0]]
    current_start = lyric_lines[0].start_time

    for i in range(1, len(lyric_lines)):
        gap = lyric_lines[i].start_time - lyric_lines[i - 1].end_time

        if gap >= SECTION_GAP_THRESHOLD:
            # Close current section
            sec = _Section(
                name="Section",
                has_lyrics=True,
                start_time=current_start,
                end_time=lyric_lines[i - 1].end_time,
            )
            sec.lines = _lines_to_output(current_lines)
            sections.append(sec)

            # Check for instrumental bridge in the gap
            gap_chords = [
                c for c in chords
                if c["time"] >= lyric_lines[i - 1].end_time + 0.5
                and c["time"] + c.get("duration", 0) < lyric_lines[i].start_time - 0.5
            ]
            if gap_chords and gap > 6.0:
                inst_sec = _Section(
                    name="Instrumental",
                    has_lyrics=False,
                    start_time=lyric_lines[i - 1].end_time,
                    end_time=lyric_lines[i].start_time,
                )
                sections.append(inst_sec)

            current_lines = [lyric_lines[i]]
            current_start = lyric_lines[i].start_time
        else:
            current_lines.append(lyric_lines[i])

    # Close last section
    if current_lines:
        sec = _Section(
            name="Section",
            has_lyrics=True,
            start_time=current_start,
            end_time=current_lines[-1].end_time,
        )
        sec.lines = _lines_to_output(current_lines)
        sections.append(sec)

    # Check for outro (chords after last lyrics)
    last_lyric_time = lyric_lines[-1].end_time
    outro_chords = [c for c in chords if c["time"] > last_lyric_time + 1.0]
    if outro_chords and (song_end - last_lyric_time) > 3.0:
        sec = _Section(
            name="Outro",
            has_lyrics=False,
            start_time=last_lyric_time,
            end_time=song_end,
        )
        sections.append(sec)

    return sections


def _lines_to_output(lyric_lines: List[_LyricLine]) -> List[dict]:
    """Convert _LyricLine objects to output dicts (chords assigned later)."""
    return [
        {"chords": "", "lyrics": ll.text, "_words": ll.words, "_start": ll.start_time, "_end": ll.end_time}
        for ll in lyric_lines
    ]


# ---------------------------------------------------------------------------
# Step 3b: Pickup anacrusis correction
# ---------------------------------------------------------------------------

def _align_sections_to_loop_downbeat(
    sections: List[_Section],
    chords: List[Dict],
    max_shift_seconds: float = 3.0,
    min_section_chords: int = 6,
    pickup_ratio: float = 0.75,
) -> None:
    """
    Shift vocal sections that start on a pickup chord forward to the loop downbeat.

    Pickup signal: the first chord of the section has significantly shorter
    duration than later occurrences of the SAME chord within the section. In a
    Cm-Gm-Dm-Am vamp where the vocal enters on the last Am of the intro loop,
    that pickup Am is typically a fraction of a bar while later Am's in the verse
    are a full bar — so the pickup duration is visibly shorter.

    If pickup detected:
      1. Shift section.start_time to the second chord (the loop downbeat).
      2. Move any lyric lines whose _start is before the new boundary back to
         the preceding section (flip prev to has_lyrics=True if instrumental).

    Mutates sections in place. No-ops when the first chord looks like a normal
    downbeat, when the section is too short, when the shift would strip the
    section, or when no preceding section exists.
    """
    for i, sec in enumerate(sections):
        if not sec.has_lyrics or not sec.lines:
            continue
        sec_chords = [c for c in chords if sec.start_time - 0.3 <= c["time"] < sec.end_time]
        if len(sec_chords) < min_section_chords:
            continue

        first = sec_chords[0]
        first_name = first["chord"]
        first_dur = first.get("duration", 1.0)

        # If next chord is the same, that's not a pickup situation
        if sec_chords[1]["chord"] == first_name:
            continue

        # Compare duration to later instances of the same chord
        later_same = [c.get("duration", 1.0) for c in sec_chords[1:] if c["chord"] == first_name]
        if not later_same:
            continue  # first chord never reappears — can't judge
        later_same.sort()
        median_later = later_same[len(later_same) // 2]
        if first_dur >= pickup_ratio * median_later:
            continue  # first chord is long enough to be the downbeat

        shift_time = sec_chords[1]["time"]
        if shift_time - sec.start_time > max_shift_seconds:
            continue

        kept_lines = []
        moved_lines = []
        for line in sec.lines:
            if line.get("_start", 0) < shift_time:
                moved_lines.append(line)
            else:
                kept_lines.append(line)

        if not moved_lines or not kept_lines:
            continue  # would strip the section entirely — leave alone

        prev_sec = sections[i - 1] if i > 0 else None
        sec.lines = kept_lines
        sec.start_time = shift_time
        if prev_sec is not None:
            prev_sec.lines = list(prev_sec.lines) + moved_lines
            prev_sec.end_time = shift_time
            if moved_lines:
                prev_sec.has_lyrics = True
        logger.info(
            f"Pickup alignment: section '{sec.name}' shifted "
            f"({first_name} pickup {first_dur:.2f}s -> downbeat {sec_chords[1]['chord']} at {shift_time:.2f}s); "
            f"{len(moved_lines)} pickup line(s) moved to prev section"
        )


# ---------------------------------------------------------------------------
# Step 4: Assign chords to lines
# ---------------------------------------------------------------------------

def _assign_chords_to_section(section: _Section, all_chords: List[Dict]):
    """Place chords above the correct words in each line of a section."""

    if not section.has_lyrics:
        # Instrumental section: just list chord names
        sec_chords = [
            c for c in all_chords
            if section.start_time - 0.5 <= c["time"] <= section.end_time + 0.5
        ]
        if not sec_chords:
            return

        # Group into lines of ~4 chords each (like UG instrumental lines)
        chord_names = [c["chord"] for c in sec_chords]
        lines = []
        for i in range(0, len(chord_names), 4):
            batch = chord_names[i:i + 4]
            lines.append({"chords": "  ".join(batch), "lyrics": None})
        section.lines = lines

        # Build fingerprint for section matching
        section.chord_pattern = _chord_fingerprint(sec_chords)
        return

    # Vocal section: place chords above words
    for line in section.lines:
        words_data = line.get("_words", [])
        lyrics_text = line.get("lyrics", "")
        line_start = line.get("_start", 0)
        line_end = line.get("_end", 0)

        if not lyrics_text:
            continue

        # Find chords that fall within this line's time range
        # Include chord active at the start of the line
        line_chords = []
        active_at_start = None

        for c in all_chords:
            chord_end = c["time"] + c.get("duration", 1.0)
            # Chord starts within this line
            if line_start - 0.3 <= c["time"] <= line_end + 0.3:
                line_chords.append(c)
            # Chord was playing when this line started (carry-over)
            elif c["time"] < line_start and chord_end > line_start:
                active_at_start = c

        if not line_chords and active_at_start:
            # Only show carry-over chord if it's different from the previous line's last chord
            line_chords = [active_at_start]

        if not line_chords:
            line["chords"] = ""
            continue

        # Deduplicate consecutive same chords
        deduped = [line_chords[0]]
        for lc in line_chords[1:]:
            if lc["chord"] != deduped[-1]["chord"]:
                deduped.append(lc)
        line_chords = deduped

        # Place chords above words using timing alignment
        chord_positions = _place_chords_on_words(line_chords, words_data, lyrics_text)
        line["chords"] = _build_chord_line(chord_positions, len(lyrics_text))

        # Build timed segments: one per chord in the line, each with the words
        # spoken during that chord's time window + bar count computed from
        # relative chord durations (shortest chord = 1 bar reference).
        durations = [max(0.1, c.get("duration", 1.0)) for c in line_chords]
        unit_bar = min(durations) if durations else 1.0
        segments = []
        for idx, c in enumerate(line_chords):
            c_start = c["time"]
            c_dur = max(0.1, c.get("duration", 1.0))
            c_end = c_start + c_dur
            seg_words = [
                {"w": w["word"], "t": round(w["start"], 3)}
                for w in words_data
                if c_start - 0.1 <= w["start"] < c_end
            ]
            # Bars = round(this_duration / shortest_in_line), min 1, max 8
            bars = max(1, min(8, round(c_dur / unit_bar)))
            segments.append({
                "chord": c["chord"],
                "start": round(c_start, 3),
                "end":   round(c_end, 3),
                "duration": round(c_dur, 3),
                "bars":    bars,
                "words":   seg_words,
            })
        line["segments"] = segments

    # Build fingerprint
    sec_chords = [
        c for c in all_chords
        if section.start_time - 0.5 <= c["time"] <= section.end_time + 0.5
    ]
    section.chord_pattern = _chord_fingerprint(sec_chords)

    # NOTE: Internal keys (_words, _start, _end) are intentionally NOT stripped
    # here — downstream steps (bar-grid section rebuild) still need _start/_end
    # to pair bars to lyric lines. Stripping happens once at the end of
    # format_chart, just before the JSON output is built.


def _place_chords_on_words(
    line_chords: List[Dict],
    words_data: List[Dict],
    lyrics_text: str,
) -> List[Tuple[int, str]]:
    """
    Return list of (char_position, chord_name) for placing chords above lyrics.
    Uses word-level timestamps for accurate placement.
    """
    if not words_data or not line_chords:
        # Fallback: evenly space chords
        positions = []
        spacing = max(1, len(lyrics_text) // max(len(line_chords), 1))
        for i, c in enumerate(line_chords):
            positions.append((i * spacing, c["chord"]))
        return positions

    # Build character position map for each word
    char_positions = []
    pos = 0
    for w in words_data:
        char_positions.append({"word": w["word"], "start": w["start"], "char_pos": pos})
        pos += len(w["word"]) + 1  # +1 for space

    positions = []
    for chord in line_chords:
        ct = chord["time"]

        # Find the word closest in time to this chord
        best_idx = 0
        best_dist = float("inf")
        for j, wp in enumerate(char_positions):
            dist = abs(wp["start"] - ct)
            if dist < best_dist:
                best_dist = dist
                best_idx = j

        char_pos = char_positions[best_idx]["char_pos"]

        # Avoid placing two chords at the exact same position
        while any(p == char_pos for p, _ in positions):
            char_pos += 1

        positions.append((char_pos, chord["chord"]))

    # Sort by position
    positions.sort(key=lambda x: x[0])
    return positions


def _build_chord_line(positions: List[Tuple[int, str]], lyrics_len: int) -> str:
    """
    Build the chord string that sits above a lyrics line.
    Chords are spaced to align with the character positions in the lyrics below.

    Returns a string like: "Am                C"
    which sits above:      "Hello darkness my old friend"
    """
    if not positions:
        return ""

    # Build the chord line character by character. Enforce a minimum two-space
    # gap between adjacent chord labels so "Cm" + "Gm" never render fused as
    # "CmGm" (which happens when two chord onsets crowd into a single short
    # word). Readable gig-chart convention: chord labels must be visually
    # separated even when the underlying time grid is dense.
    MIN_GAP = 2
    result = []
    cursor = 0

    for char_pos, chord_name in positions:
        required_pos = char_pos
        if cursor > 0 and required_pos < cursor + MIN_GAP:
            required_pos = cursor + MIN_GAP
        # Pad to reach this position
        if required_pos > cursor:
            result.append(" " * (required_pos - cursor))
            cursor = required_pos

        # Place the chord
        result.append(chord_name)
        cursor += len(chord_name)

    return "".join(result).rstrip()


# ---------------------------------------------------------------------------
# Step 5: Section type detection
# ---------------------------------------------------------------------------

def _chord_fingerprint(chords: List[Dict]) -> str:
    """
    Create a fingerprint of a chord progression for pattern matching.
    Uses relative intervals (semitones between consecutive roots) rather than
    absolute chord names, so transposed songs still match.
    """
    if not chords:
        return ""

    NOTE_MAP = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
        'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
    }

    roots = []
    for c in chords:
        name = c.get("chord", "N")
        if name == "N":
            continue
        root = name[0]
        if len(name) > 1 and name[1] in ('#', 'b'):
            root = name[:2]
        pc = NOTE_MAP.get(root, -1)
        if pc >= 0:
            roots.append(pc)

    if len(roots) < 2:
        return ",".join(str(r) for r in roots)

    # Convert to intervals
    intervals = []
    for i in range(1, len(roots)):
        interval = (roots[i] - roots[i - 1]) % 12
        intervals.append(str(interval))

    return ",".join(intervals)


def _label_sections(sections: List[_Section]):
    """
    Label sections as Verse, Chorus, Bridge, etc. based on heuristics:

    1. Instrumental sections without lyrics keep their name (Intro/Outro/Instrumental)
    2. The most-repeated chord pattern with lyrics = Verse
    3. Second most-repeated = Chorus (or if higher energy / different pattern)
    4. Unique patterns that appear once = Bridge
    5. Short sections with few chords near choruses = Pre-Chorus
    """
    vocal_sections = [s for s in sections if s.has_lyrics and s.chord_pattern]

    if not vocal_sections:
        return

    # Count pattern occurrences
    pattern_counts = Counter(s.chord_pattern for s in vocal_sections)

    # Rank patterns by frequency
    ranked = pattern_counts.most_common()

    if not ranked:
        for s in vocal_sections:
            s.name = "Verse"
        return

    # Most common pattern = Verse
    verse_pattern = ranked[0][0]

    # Second most common = Chorus (if exists and appears 2+ times)
    chorus_pattern = None
    if len(ranked) > 1 and ranked[1][1] >= 2:
        chorus_pattern = ranked[1][0]
    elif len(ranked) > 1:
        # Even if it appears once, if it's distinct enough, call it a chorus
        chorus_pattern = ranked[1][0]

    # Patterns that appear exactly once and aren't verse/chorus = Bridge
    bridge_patterns = set()
    for pattern, count in ranked[2:]:
        if count == 1:
            bridge_patterns.add(pattern)

    # Apply labels
    verse_num = 0
    chorus_num = 0
    bridge_num = 0

    for s in sections:
        if not s.has_lyrics:
            continue  # Already labeled (Intro/Outro/Instrumental)

        if s.chord_pattern == verse_pattern:
            s.name = "Verse"
        elif chorus_pattern and s.chord_pattern == chorus_pattern:
            s.name = "Chorus"
        elif s.chord_pattern in bridge_patterns:
            s.name = "Bridge"
        else:
            # Could be pre-chorus (short section between verse and chorus)
            line_count = len(s.lines)
            if line_count <= 2:
                s.name = "Pre-Chorus"
            else:
                s.name = "Verse"

    # Heuristic: if a short section always appears between Verse and Chorus, it's Pre-Chorus
    for i, s in enumerate(sections):
        if not s.has_lyrics:
            continue
        if s.name in ("Verse", "Chorus", "Bridge"):
            continue
        # Check if it's sandwiched between Verse and Chorus
        prev_vocal = _prev_vocal_section(sections, i)
        next_vocal = _next_vocal_section(sections, i)
        if prev_vocal and next_vocal:
            if prev_vocal.name == "Verse" and next_vocal.name == "Chorus":
                s.name = "Pre-Chorus"


def _prev_vocal_section(sections, idx):
    for i in range(idx - 1, -1, -1):
        if sections[i].has_lyrics:
            return sections[i]
    return None


def _next_vocal_section(sections, idx):
    for i in range(idx + 1, len(sections)):
        if sections[i].has_lyrics:
            return sections[i]
    return None


# ---------------------------------------------------------------------------
# Step 6: Number repeated sections
# ---------------------------------------------------------------------------

def _number_repeated_sections(sections: List[_Section]):
    """
    Add numbers to repeated section types: Verse 1, Verse 2, Chorus, Chorus, etc.
    Only numbers verses (Verse 1, Verse 2...). Choruses stay as "Chorus" unless
    their lyrics differ.
    """
    # Count how many of each type
    type_counts = Counter(s.name for s in sections)

    # Number verses
    verse_idx = 0
    for s in sections:
        if s.name == "Verse":
            if type_counts["Verse"] > 1:
                verse_idx += 1
                s.name = f"Verse {verse_idx}"

    # Number bridges if multiple
    bridge_idx = 0
    for s in sections:
        if s.name == "Bridge":
            if type_counts["Bridge"] > 1:
                bridge_idx += 1
                s.name = f"Bridge {bridge_idx}"

    # Rename instrumental sections in the middle to "Interlude" or "Solo"
    for s in sections:
        if s.name == "Instrumental":
            # If it's not the first or last section, call it Interlude
            idx = sections.index(s)
            if idx > 0 and idx < len(sections) - 1:
                s.name = "Interlude"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_vocal_tail_into_solo(sections: List[_Section], stem_paths: Dict[str, str]) -> List[_Section]:
    """
    If the TAIL of a vocal section (last 2+ lines) has:
      - vocals stem RMS < 0.7 × dominant instrument stem RMS, AND
      - a chord pattern that differs from the section's opening
    …split the tail off into its own section labeled "{Instrument} Solo".
    Returns the new list of sections (may be same length or longer).

    Silent on any failure; returns input unchanged if stems can't be loaded.
    """
    try:
        import librosa
        import numpy as np
    except ImportError:
        return sections

    INSTRUMENT_STEMS = {"guitar", "bass", "piano", "drums", "other"}
    INSTRUMENT_LABEL = {
        "guitar": "Guitar Solo",
        "bass":   "Bass Solo",
        "piano":  "Keys Solo",
        "drums":  "Drum Break",
        "other":  "Instrumental",
    }

    # Pre-load stems
    stem_audio: Dict[str, tuple] = {}
    vocals_audio = None
    for name, path in (stem_paths or {}).items():
        base = name.lower().split("_")[0]
        try:
            y, sr = librosa.load(path, sr=22050, mono=True)
        except Exception:
            continue
        if base == "vocals":
            vocals_audio = (y, sr)
        elif base in INSTRUMENT_STEMS:
            stem_audio[base] = (y, sr)

    if not stem_audio or vocals_audio is None:
        return sections

    def _rms_in(y, sr, t0, t1):
        i0 = max(0, int(t0 * sr)); i1 = min(len(y), int(t1 * sr))
        if i1 <= i0:
            return 0.0
        seg = y[i0:i1]
        return float(np.sqrt(np.mean(seg ** 2) + 1e-12)) if len(seg) else 0.0

    result: List[_Section] = []
    for sec in sections:
        # Only consider sections with enough lyric lines to have a tail
        lines = sec.lines or []
        if not sec.has_lyrics or len(lines) < 4:
            result.append(sec); continue

        # Find the midpoint line, try a clean split
        split_idx = len(lines) // 2
        tail_lines = lines[split_idx:]
        head_lines = lines[:split_idx]
        if not tail_lines or not head_lines:
            result.append(sec); continue

        # Tail time window
        def _line_bounds(ll):
            # Lines carry _start/_end when built by _build_raw_sections
            s = ll.get('_start') if isinstance(ll, dict) else None
            e = ll.get('_end')   if isinstance(ll, dict) else None
            return (s, e)
        tail_s, _ = _line_bounds(tail_lines[0])
        _, tail_e = _line_bounds(tail_lines[-1])
        if tail_s is None or tail_e is None or tail_e <= tail_s:
            result.append(sec); continue

        # Compare chord signatures head vs. tail
        def _sig(ls):
            chs = []
            for ln in ls:
                segs = ln.get('segments') if isinstance(ln, dict) else None
                if segs:
                    chs.extend([s.get('chord', '') for s in segs])
            return '|'.join(chs)
        head_sig = _sig(head_lines)
        tail_sig = _sig(tail_lines)
        if not tail_sig or head_sig == tail_sig:
            result.append(sec); continue

        # Stem evidence: vocals must be clearly quieter than a dominant instrument
        v_rms = _rms_in(vocals_audio[0], vocals_audio[1], tail_s, tail_e)
        inst_rms = {k: _rms_in(y, sr, tail_s, tail_e) for k, (y, sr) in stem_audio.items()}
        if not inst_rms:
            result.append(sec); continue
        top_inst = max(inst_rms, key=inst_rms.get)
        top_rms = inst_rms[top_inst]
        if top_rms <= 0 or v_rms / (top_rms + 1e-9) >= 0.7:
            result.append(sec); continue  # vocals still present → it's the chorus

        # Split: keep head as-is, emit a new solo section for the tail.
        head_sec = _Section(
            name=sec.name, has_lyrics=True,
            start_time=sec.start_time, end_time=tail_s,
        )
        head_sec.lines = head_lines
        head_sec.chord_pattern = sec.chord_pattern

        # Tail still contains Whisper-labeled lyric text, even if the vocal
        # stem RMS dropped. has_lyrics tracks whether the output should be
        # rendered as vocal-with-chords or instrumental-only, so mirror the
        # actual line content: a tail whose lines carry non-empty lyric
        # strings stays a vocal section, and keeps a generic name
        # (Section / Outro) rather than being falsely renamed "Bass Solo".
        tail_has_lyrics = any(
            (ln.get('lyrics') or '').strip()
            for ln in tail_lines if isinstance(ln, dict)
        )
        solo_name = INSTRUMENT_LABEL.get(top_inst, "Solo") if not tail_has_lyrics else sec.name

        solo_sec = _Section(
            name=solo_name,
            has_lyrics=tail_has_lyrics,
            start_time=tail_s, end_time=tail_e,
        )
        solo_sec.lines = tail_lines
        solo_sec.chord_pattern = tail_sig

        result.extend([head_sec, solo_sec])

    return result


def _label_solo_instruments(sections: List[_Section], stem_paths: Dict[str, str]) -> None:
    """
    For any section whose current name is generic (Solo/Interlude/Instrumental/Intro/Outro),
    compute RMS of each non-vocal stem over the section's time window and relabel with the
    loudest instrument: "Piano Solo", "Guitar Solo", "Bass Solo", "Drums Solo".

    If the loudest stem is vocals (unusual during labeled Solo/Interlude), leave name unchanged.
    If the section has lyrics (has_lyrics=True), skip — those are vocal sections.

    Silent on any stem load/decode failure — individual stems are best-effort.
    """
    try:
        import librosa
        import numpy as np
    except ImportError:
        logger.info("librosa/numpy unavailable, skipping solo-instrument labeling")
        return

    INSTRUMENT_STEMS = {"guitar", "bass", "piano", "drums", "other"}
    INSTRUMENT_LABEL = {
        "guitar": "Guitar Solo",
        "bass":   "Bass Solo",
        "piano":  "Piano Solo",
        "drums":  "Drum Break",
        "other":  "Instrumental",
    }
    RENAMEABLE = {"Solo", "Interlude", "Instrumental", "Intro", "Outro"}

    # Pre-load each stem once (at 22050 mono for speed)
    stem_audio: Dict[str, tuple] = {}
    for name, path in (stem_paths or {}).items():
        base = name.lower().split("_")[0]  # handle "guitar_lead" → "guitar"
        if base not in INSTRUMENT_STEMS:
            continue
        try:
            y, sr = librosa.load(path, sr=22050, mono=True)
            stem_audio[base] = (y, sr)
        except Exception as e:
            logger.debug(f"Could not load stem {name}: {e}")

    if not stem_audio:
        logger.info("No instrument stems loaded; skipping solo labeling")
        return

    for sec in sections:
        if sec.has_lyrics:
            continue
        # Safety guard: if the section's line dicts carry any non-empty lyric
        # text, treat it as a vocal section regardless of the has_lyrics flag.
        # Prevents "Bass Solo" / "Drum Break" labels landing on a section with
        # actual lyrics (a Whisper phantom-lyric inside a quiet solo).
        if any((ln.get('lyrics') or '').strip() for ln in (sec.lines or []) if isinstance(ln, dict)):
            continue
        # Strip possible numbering like "Solo 2" before comparing
        base_name = sec.name.split()[0] if sec.name else ""
        if base_name not in RENAMEABLE and sec.name not in RENAMEABLE:
            continue

        # Compute RMS of each stem in the section window
        rms_by_instrument: Dict[str, float] = {}
        for base, (y, sr) in stem_audio.items():
            i0 = max(0, int(sec.start_time * sr))
            i1 = min(len(y), int(sec.end_time * sr))
            if i1 <= i0:
                continue
            segment = y[i0:i1]
            if len(segment) == 0:
                continue
            rms_by_instrument[base] = float(np.sqrt(np.mean(segment ** 2) + 1e-12))

        if not rms_by_instrument:
            continue
        top = max(rms_by_instrument, key=rms_by_instrument.get)
        top_rms = rms_by_instrument[top]
        # Only relabel if the top stem is clearly dominant (≥ 1.3× the second-loudest);
        # otherwise keep the generic name.
        others = [v for k, v in rms_by_instrument.items() if k != top]
        second = max(others) if others else 0.0
        if second > 0 and top_rms / (second + 1e-9) < 1.3:
            continue
        # For Intro/Outro we usually DON'T rename (they're named for their position in the song).
        if sec.name in ("Intro", "Outro"):
            continue
        sec.name = INSTRUMENT_LABEL.get(top, sec.name)


def _empty_chart(title, artist, key, capo):
    """Return a minimal empty chart."""
    return {
        "title": title,
        "artist": artist,
        "key": key,
        "capo": capo,
        "source": "StemScriber AI",
        "chords_used": [],
        "sections": [],
    }


# ---------------------------------------------------------------------------
# Convenience: format from raw files
# ---------------------------------------------------------------------------

def format_chart_from_files(
    chords_json_path: str,
    words_json_path: str,
    title: str = "Unknown",
    artist: str = "Unknown",
    key: str = "Unknown",
) -> Dict:
    """Load chord events and word timestamps from JSON files and format a chart."""
    with open(chords_json_path) as f:
        chord_events = json.load(f)
    with open(words_json_path) as f:
        word_timestamps = json.load(f)
    return format_chart(chord_events, word_timestamps, title=title, artist=artist, key=key)


# ---------------------------------------------------------------------------
# Plain text rendering (for preview / debugging)
# ---------------------------------------------------------------------------

def render_plain_text(chart: Dict) -> str:
    """
    Render a chord chart dict as plain text in Ultimate Guitar style.

    Example output:
        [Verse 1]
        Am                C
        Hello darkness my old friend
        G                 Am
        I've come to talk with you again
    """
    lines = []

    if chart.get("title") or chart.get("artist"):
        lines.append(f"{chart.get('title', '')} - {chart.get('artist', '')}")

    if chart.get("key"):
        meta = f"Key: {chart['key']}"
        if chart.get("capo"):
            meta += f"  Capo: {chart['capo']}"
        lines.append(meta)

    lines.append("")

    for section in chart.get("sections", []):
        lines.append(f"[{section['name']}]")
        for line in section.get("lines", []):
            chords = line.get("chords", "")
            lyrics = line.get("lyrics")

            if chords:
                lines.append(chords)
            if lyrics:
                lines.append(lyrics)

        lines.append("")  # blank line between sections

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test / demo
# ---------------------------------------------------------------------------

def _run_test():
    """
    Test the formatter with sample data resembling "Sound of Silence" by Simon & Garfunkel.
    Demonstrates chord-over-word alignment and section detection.
    """
    # Simulated chord detection output (like chord_detector_v10.detect_chords)
    chord_events = [
        # Intro
        {"time": 0.0, "duration": 2.5, "chord": "Am", "root": "A", "quality": "m", "confidence": 0.9},
        # Verse 1
        {"time": 2.5, "duration": 3.0, "chord": "Am", "root": "A", "quality": "m", "confidence": 0.92},
        {"time": 5.5, "duration": 2.5, "chord": "G", "root": "G", "quality": "maj", "confidence": 0.88},
        {"time": 8.0, "duration": 3.0, "chord": "Am", "root": "A", "quality": "m", "confidence": 0.91},
        {"time": 11.0, "duration": 2.5, "chord": "G", "root": "G", "quality": "maj", "confidence": 0.87},
        {"time": 13.5, "duration": 3.0, "chord": "Am", "root": "A", "quality": "m", "confidence": 0.90},
        {"time": 16.5, "duration": 2.5, "chord": "G", "root": "G", "quality": "maj", "confidence": 0.86},
        {"time": 19.0, "duration": 2.0, "chord": "F", "root": "F", "quality": "maj", "confidence": 0.84},
        {"time": 21.0, "duration": 2.0, "chord": "C", "root": "C", "quality": "maj", "confidence": 0.89},
        # Chorus
        {"time": 27.0, "duration": 2.5, "chord": "F", "root": "F", "quality": "maj", "confidence": 0.88},
        {"time": 29.5, "duration": 2.5, "chord": "C", "root": "C", "quality": "maj", "confidence": 0.90},
        {"time": 32.0, "duration": 2.5, "chord": "F", "root": "F", "quality": "maj", "confidence": 0.87},
        {"time": 34.5, "duration": 2.5, "chord": "C", "root": "C", "quality": "maj", "confidence": 0.89},
        # Instrumental break
        {"time": 40.0, "duration": 2.0, "chord": "Am", "root": "A", "quality": "m", "confidence": 0.85},
        {"time": 42.0, "duration": 2.0, "chord": "G", "root": "G", "quality": "maj", "confidence": 0.84},
        {"time": 44.0, "duration": 2.0, "chord": "F", "root": "F", "quality": "maj", "confidence": 0.83},
        {"time": 46.0, "duration": 2.0, "chord": "C", "root": "C", "quality": "maj", "confidence": 0.82},
        # Verse 2
        {"time": 52.0, "duration": 3.0, "chord": "Am", "root": "A", "quality": "m", "confidence": 0.91},
        {"time": 55.0, "duration": 2.5, "chord": "G", "root": "G", "quality": "maj", "confidence": 0.87},
        {"time": 57.5, "duration": 3.0, "chord": "Am", "root": "A", "quality": "m", "confidence": 0.90},
        {"time": 60.5, "duration": 2.5, "chord": "G", "root": "G", "quality": "maj", "confidence": 0.86},
        {"time": 63.0, "duration": 3.0, "chord": "Am", "root": "A", "quality": "m", "confidence": 0.89},
        {"time": 66.0, "duration": 2.5, "chord": "G", "root": "G", "quality": "maj", "confidence": 0.85},
        {"time": 68.5, "duration": 2.0, "chord": "F", "root": "F", "quality": "maj", "confidence": 0.83},
        {"time": 70.5, "duration": 2.0, "chord": "C", "root": "C", "quality": "maj", "confidence": 0.88},
        # Chorus 2
        {"time": 76.0, "duration": 2.5, "chord": "F", "root": "F", "quality": "maj", "confidence": 0.87},
        {"time": 78.5, "duration": 2.5, "chord": "C", "root": "C", "quality": "maj", "confidence": 0.89},
        {"time": 81.0, "duration": 2.5, "chord": "F", "root": "F", "quality": "maj", "confidence": 0.86},
        {"time": 83.5, "duration": 2.5, "chord": "C", "root": "C", "quality": "maj", "confidence": 0.88},
        # Outro
        {"time": 90.0, "duration": 3.0, "chord": "Am", "root": "A", "quality": "m", "confidence": 0.80},
        {"time": 93.0, "duration": 4.0, "chord": "Am", "root": "A", "quality": "m", "confidence": 0.75},
    ]

    # Simulated Whisper word timestamps (like word_timestamps.get_word_timestamps)
    word_timestamps = [
        # Verse 1 line 1
        {"word": "Hello", "start": 2.8, "end": 3.3},
        {"word": "darkness", "start": 3.4, "end": 4.0},
        {"word": "my", "start": 4.1, "end": 4.3},
        {"word": "old", "start": 4.4, "end": 4.6},
        {"word": "friend", "start": 4.7, "end": 5.2},
        # Verse 1 line 2
        {"word": "I've", "start": 5.8, "end": 6.1},
        {"word": "come", "start": 6.2, "end": 6.5},
        {"word": "to", "start": 6.6, "end": 6.7},
        {"word": "talk", "start": 6.8, "end": 7.1},
        {"word": "with", "start": 7.2, "end": 7.4},
        {"word": "you", "start": 7.5, "end": 7.7},
        {"word": "again", "start": 7.8, "end": 8.3},
        # Verse 1 line 3
        {"word": "Because", "start": 8.8, "end": 9.3},
        {"word": "a", "start": 9.4, "end": 9.5},
        {"word": "vision", "start": 9.6, "end": 10.1},
        {"word": "softly", "start": 10.2, "end": 10.7},
        {"word": "creeping", "start": 10.8, "end": 11.3},
        # Verse 1 line 4
        {"word": "Left", "start": 11.8, "end": 12.1},
        {"word": "its", "start": 12.2, "end": 12.4},
        {"word": "seeds", "start": 12.5, "end": 12.9},
        {"word": "while", "start": 13.0, "end": 13.2},
        {"word": "I", "start": 13.3, "end": 13.4},
        {"word": "was", "start": 13.5, "end": 13.7},
        {"word": "sleeping", "start": 13.8, "end": 14.3},
        # Verse 1 line 5
        {"word": "And", "start": 14.8, "end": 15.0},
        {"word": "the", "start": 15.1, "end": 15.2},
        {"word": "vision", "start": 15.3, "end": 15.8},
        {"word": "that", "start": 15.9, "end": 16.1},
        {"word": "was", "start": 16.2, "end": 16.3},
        {"word": "planted", "start": 16.4, "end": 16.9},
        {"word": "in", "start": 17.0, "end": 17.1},
        {"word": "my", "start": 17.2, "end": 17.3},
        {"word": "brain", "start": 17.4, "end": 17.9},
        # Verse 1 line 6
        {"word": "Still", "start": 18.8, "end": 19.1},
        {"word": "remains", "start": 19.2, "end": 19.8},
        {"word": "within", "start": 20.0, "end": 20.4},
        {"word": "the", "start": 20.5, "end": 20.6},
        {"word": "sound", "start": 20.7, "end": 21.2},
        {"word": "of", "start": 21.3, "end": 21.4},
        {"word": "silence", "start": 21.5, "end": 22.2},

        # Chorus line 1 (after gap)
        {"word": "In", "start": 27.2, "end": 27.4},
        {"word": "restless", "start": 27.5, "end": 28.0},
        {"word": "dreams", "start": 28.1, "end": 28.5},
        {"word": "I", "start": 28.6, "end": 28.7},
        {"word": "walked", "start": 28.8, "end": 29.2},
        {"word": "alone", "start": 29.3, "end": 29.8},
        # Chorus line 2
        {"word": "Narrow", "start": 30.2, "end": 30.7},
        {"word": "streets", "start": 30.8, "end": 31.2},
        {"word": "of", "start": 31.3, "end": 31.4},
        {"word": "cobblestone", "start": 31.5, "end": 32.2},
        # Chorus line 3
        {"word": "Beneath", "start": 32.5, "end": 33.0},
        {"word": "the", "start": 33.1, "end": 33.2},
        {"word": "halo", "start": 33.3, "end": 33.8},
        {"word": "of", "start": 33.9, "end": 34.0},
        {"word": "a", "start": 34.1, "end": 34.2},
        {"word": "street", "start": 34.3, "end": 34.7},
        {"word": "lamp", "start": 34.8, "end": 35.3},

        # Verse 2 line 1 (after instrumental gap)
        {"word": "And", "start": 52.3, "end": 52.5},
        {"word": "in", "start": 52.6, "end": 52.7},
        {"word": "the", "start": 52.8, "end": 52.9},
        {"word": "naked", "start": 53.0, "end": 53.5},
        {"word": "light", "start": 53.6, "end": 54.0},
        {"word": "I", "start": 54.1, "end": 54.2},
        {"word": "saw", "start": 54.3, "end": 54.7},
        # Verse 2 line 2
        {"word": "Ten", "start": 55.3, "end": 55.6},
        {"word": "thousand", "start": 55.7, "end": 56.2},
        {"word": "people", "start": 56.3, "end": 56.8},
        {"word": "maybe", "start": 56.9, "end": 57.4},
        {"word": "more", "start": 57.5, "end": 57.9},
        # Verse 2 line 3
        {"word": "People", "start": 58.3, "end": 58.8},
        {"word": "talking", "start": 58.9, "end": 59.4},
        {"word": "without", "start": 59.5, "end": 60.0},
        {"word": "speaking", "start": 60.1, "end": 60.7},
        # Verse 2 line 4
        {"word": "People", "start": 61.2, "end": 61.7},
        {"word": "hearing", "start": 61.8, "end": 62.3},
        {"word": "without", "start": 62.4, "end": 62.9},
        {"word": "listening", "start": 63.0, "end": 63.7},
        # Verse 2 line 5
        {"word": "People", "start": 64.2, "end": 64.7},
        {"word": "writing", "start": 64.8, "end": 65.3},
        {"word": "songs", "start": 65.4, "end": 65.8},
        {"word": "that", "start": 65.9, "end": 66.1},
        {"word": "voices", "start": 66.2, "end": 66.7},
        {"word": "never", "start": 66.8, "end": 67.2},
        {"word": "shared", "start": 67.3, "end": 67.8},
        # Verse 2 line 6
        {"word": "And", "start": 68.3, "end": 68.5},
        {"word": "no", "start": 68.6, "end": 68.8},
        {"word": "one", "start": 68.9, "end": 69.1},
        {"word": "dared", "start": 69.2, "end": 69.7},
        {"word": "disturb", "start": 70.0, "end": 70.5},
        {"word": "the", "start": 70.6, "end": 70.7},
        {"word": "sound", "start": 70.8, "end": 71.3},
        {"word": "of", "start": 71.4, "end": 71.5},
        {"word": "silence", "start": 71.6, "end": 72.3},

        # Chorus 2 line 1 (after gap)
        {"word": "Fools", "start": 76.3, "end": 76.8},
        {"word": "said", "start": 76.9, "end": 77.2},
        {"word": "I", "start": 77.3, "end": 77.4},
        {"word": "you", "start": 77.5, "end": 77.7},
        {"word": "do", "start": 77.8, "end": 78.0},
        {"word": "not", "start": 78.1, "end": 78.3},
        {"word": "know", "start": 78.4, "end": 78.9},
        # Chorus 2 line 2
        {"word": "Silence", "start": 79.3, "end": 79.8},
        {"word": "like", "start": 79.9, "end": 80.2},
        {"word": "a", "start": 80.3, "end": 80.4},
        {"word": "cancer", "start": 80.5, "end": 81.0},
        {"word": "grows", "start": 81.1, "end": 81.6},
        # Chorus 2 line 3
        {"word": "Hear", "start": 82.0, "end": 82.3},
        {"word": "my", "start": 82.4, "end": 82.6},
        {"word": "words", "start": 82.7, "end": 83.1},
        {"word": "that", "start": 83.2, "end": 83.4},
        {"word": "I", "start": 83.5, "end": 83.6},
        {"word": "might", "start": 83.7, "end": 84.0},
        {"word": "teach", "start": 84.1, "end": 84.5},
        {"word": "you", "start": 84.6, "end": 84.9},
    ]

    print("=" * 60)
    print("CHART FORMATTER TEST")
    print("=" * 60)
    print()

    chart = format_chart(
        chord_events=chord_events,
        word_timestamps=word_timestamps,
        title="The Sound of Silence",
        artist="Simon & Garfunkel",
        key="Am",
    )

    # Print as plain text
    print(render_plain_text(chart))

    # Print as JSON
    print("=" * 60)
    print("JSON OUTPUT:")
    print("=" * 60)
    print(json.dumps(chart, indent=2))

    return chart


if __name__ == "__main__":
    _run_test()
