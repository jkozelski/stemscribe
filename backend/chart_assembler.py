"""
Chart Assembler — Combines chords + lyrics + song structure into a formatted
chord chart JSON that the frontend's renderManualChordChart() can render.

Output format matches chord_chart.json / chord_chart_manual.json:
{
  "title": "...",
  "artist": "...",
  "source": "auto",
  "sections": [
    {
      "name": "Verse 1",
      "lines": [
        {"chords": "F#m          A", "lyrics": "When the time comes around"},
        ...
      ]
    }
  ]
}
"""

import json
import math
from pathlib import Path
from typing import List, Dict, Optional


def _find_chords_in_range(chords: List[dict], start: float, end: float) -> List[dict]:
    """Return chord events for a lyric line.

    Includes the most recent chord before the line starts (the chord that's
    ringing when the vocal enters) plus any chord changes during the line.
    """
    # Find chords that fall within the line
    in_range = [c for c in chords if start <= c['time'] < end]

    # Find the most recent chord BEFORE this line starts (the active chord)
    preceding = [c for c in chords if c['time'] < start]
    if preceding:
        active_chord = preceding[-1]
        # Only include if it's not already the first chord in range
        # and it's within 2 seconds before the line (still ringing)
        if (start - active_chord['time'] < 2.0 and
                (not in_range or in_range[0]['chord'] != active_chord['chord'])):
            in_range.insert(0, {**active_chord, 'time': start})

    return in_range


def _format_chord_line(chords_in_line: List[dict], lyric_text: str,
                       line_start: float, line_end: float) -> str:
    """
    Build a chord string positioned above the lyric text.

    Each chord is placed at a character position proportional to where it falls
    in the time range of the lyric line.
    """
    if not chords_in_line:
        return ""

    line_len = max(len(lyric_text), 1)
    duration = line_end - line_start if line_end > line_start else 1.0

    # Build list of (char_position, chord_name)
    placements = []
    for c in chords_in_line:
        frac = (c['time'] - line_start) / duration
        frac = max(0.0, min(frac, 1.0))
        char_pos = int(frac * line_len)
        placements.append((char_pos, c['chord']))

    # Sort by position
    placements.sort(key=lambda x: x[0])

    # Resolve overlaps — ensure each chord has room
    resolved = []
    cursor = 0
    for pos, name in placements:
        pos = max(pos, cursor)
        resolved.append((pos, name))
        cursor = pos + len(name) + 1  # at least 1 space after chord

    # Build the padded chord string
    result = []
    cur = 0
    for pos, name in resolved:
        if pos > cur:
            result.append(' ' * (pos - cur))
        result.append(name)
        cur = pos + len(name)

    chord_str = ''.join(result)

    # Ensure chord line is at least as wide as the lyric line
    if len(chord_str) < line_len:
        chord_str = chord_str.ljust(line_len)

    return chord_str


def _format_instrumental_chords(chords_in_section: List[dict]) -> List[dict]:
    """
    For instrumental sections (no lyrics), format chord patterns.
    Detects repeating patterns and uses repeat notation.
    """
    if not chords_in_section:
        return []

    chord_names = [c['chord'] for c in chords_in_section]

    # Try to detect a repeating pattern (2-8 chords)
    for pattern_len in range(2, min(9, len(chord_names) + 1)):
        pattern = chord_names[:pattern_len]
        repeats = 0
        i = 0
        while i + pattern_len <= len(chord_names):
            if chord_names[i:i + pattern_len] == pattern:
                repeats += 1
                i += pattern_len
            else:
                break

        if repeats >= 2 and i >= len(chord_names) - 1:
            # Found a repeating pattern
            spacing = max(12 // pattern_len, 4)
            chord_str = (''.join(c.ljust(spacing) for c in pattern)).rstrip()
            lines = [{"chords": chord_str, "lyrics": f"(x{repeats})"}]
            # Handle leftover chords
            leftover = chord_names[repeats * pattern_len:]
            if leftover:
                leftover_str = (''.join(c.ljust(spacing) for c in leftover)).rstrip()
                lines.append({"chords": leftover_str, "lyrics": ""})
            return lines

    # No repeating pattern — output chords in groups of 4
    lines = []
    for i in range(0, len(chord_names), 4):
        group = chord_names[i:i + 4]
        chord_str = (''.join(c.ljust(12) for c in group)).rstrip()
        lines.append({"chords": chord_str, "lyrics": ""})

    return lines


def _auto_section(lyric_lines: List[dict], section_size: int = 8) -> List[dict]:
    """
    Create generic sections when no structure detection is available.
    Groups every `section_size` lines into a section.
    """
    sections = []
    total = len(lyric_lines)
    verse_num = 0
    for i in range(0, total, section_size):
        verse_num += 1
        chunk = lyric_lines[i:i + section_size]
        start = chunk[0].get('start_time', 0)
        end = chunk[-1].get('end_time', start + 30)
        sections.append({
            'name': f'Verse {verse_num}',
            'start_time': start,
            'end_time': end,
        })
    return sections


def _assign_lines_to_sections(lyrics: List[dict],
                              sections: List[dict]) -> Dict[int, List[dict]]:
    """
    Map each lyric line to its section index.
    Returns {section_idx: [lyric_line, ...]}.
    """
    assigned: Dict[int, List[dict]] = {i: [] for i in range(len(sections))}

    for line in lyrics:
        line_mid = (line.get('start_time', 0) + line.get('end_time', 0)) / 2
        best_idx = 0
        best_dist = float('inf')
        for si, sec in enumerate(sections):
            s = sec.get('start_time', 0)
            e = sec.get('end_time', s + 30)
            if s <= line_mid <= e:
                best_idx = si
                best_dist = 0
                break
            dist = min(abs(line_mid - s), abs(line_mid - e))
            if dist < best_dist:
                best_dist = dist
                best_idx = si
        assigned[best_idx].append(line)

    return assigned


def _estimate_tempo(chords: List[dict]) -> float:
    """
    Estimate tempo (BPM) from chord durations.
    Looks for the most common duration quantum that could be a beat.
    Returns estimated BPM, default 120 if can't determine.
    """
    if not chords or len(chords) < 2:
        return 120.0

    durations = [c.get('duration', 0) for c in chords if c.get('duration', 0) > 0.1]
    if not durations:
        return 120.0

    # Find the GCD-like quantum: the smallest common duration unit
    # Most chord durations should be multiples of the beat duration
    min_dur = min(durations)
    median_dur = sorted(durations)[len(durations) // 2]

    # Try common beat durations (BPM 60-180 -> beat = 1.0s to 0.33s)
    best_bpm = 120.0
    best_score = 0

    for bpm_candidate in range(60, 181, 2):
        beat_dur = 60.0 / bpm_candidate
        score = 0
        for d in durations:
            beats = d / beat_dur
            # How close is this to a whole number of beats?
            rounded = round(beats)
            if rounded >= 1:
                error = abs(beats - rounded) / rounded
                if error < 0.2:  # within 20%
                    score += 1
        if score > best_score:
            best_score = score
            best_bpm = bpm_candidate

    return best_bpm


def _compute_chord_beats_for_line(chords_in_line: List[dict], tempo: float,
                                   line_start: float, line_end: float,
                                   beats_per_bar: int = 4) -> List[dict]:
    """
    Compute beat counts for each chord in a line.

    Returns list of {name: str, beats: int} where beats is quantized to
    whole beats and the total is rounded to a multiple of beats_per_bar.
    """
    if not chords_in_line:
        return []

    beat_duration = 60.0 / tempo if tempo > 0 else 0.5  # seconds per beat

    raw_beats = []
    for i, c in enumerate(chords_in_line):
        # Duration: time until next chord, or until line end
        if i + 1 < len(chords_in_line):
            dur = chords_in_line[i + 1]['time'] - c['time']
        else:
            dur = line_end - c['time']
        dur = max(dur, beat_duration * 0.5)  # at least half a beat

        beats_float = dur / beat_duration
        raw_beats.append({
            'name': c['chord'],
            'beats_float': beats_float,
        })

    # Round each to nearest whole beat (minimum 1)
    result = []
    for rb in raw_beats:
        b = max(1, round(rb['beats_float']))
        result.append({'name': rb['name'], 'beats': b})

    # Quantize total to multiple of beats_per_bar
    total = sum(r['beats'] for r in result)
    target = max(beats_per_bar, round(total / beats_per_bar) * beats_per_bar)

    # Adjust if needed - scale proportionally
    if target != total and total > 0:
        scale = target / total
        new_total = 0
        for r in result:
            r['beats'] = max(1, round(r['beats'] * scale))
            new_total += r['beats']
        # Fix rounding on last chord
        if new_total != target and result:
            result[-1]['beats'] += (target - new_total)
            if result[-1]['beats'] < 1:
                result[-1]['beats'] = 1

    return result


def _compute_instrumental_chord_beats(chords_in_section: List[dict],
                                       tempo: float,
                                       beats_per_bar: int = 4) -> List[List[dict]]:
    """
    Compute chord_beats for instrumental sections (no lyrics).
    Groups chords into lines of ~4 bars each.
    Returns list of chord_beats arrays (one per line).
    """
    if not chords_in_section:
        return []

    beat_duration = 60.0 / tempo if tempo > 0 else 0.5

    # Compute beats for each chord
    all_beats = []
    for i, c in enumerate(chords_in_section):
        if i + 1 < len(chords_in_section):
            dur = chords_in_section[i + 1]['time'] - c['time']
        else:
            dur = c.get('duration', beat_duration * 2)
        dur = max(dur, beat_duration * 0.5)
        b = max(1, round(dur / beat_duration))
        all_beats.append({'name': c['chord'], 'beats': b})

    # Group into lines of ~4 bars (16 beats in 4/4)
    beats_per_line = beats_per_bar * 4
    lines = []
    current_line = []
    current_count = 0

    for cb in all_beats:
        current_line.append(cb)
        current_count += cb['beats']
        if current_count >= beats_per_line:
            lines.append(current_line)
            current_line = []
            current_count = 0

    if current_line:
        lines.append(current_line)

    return lines


def assemble_chart(chords: List[dict],
                   lyrics: Optional[List[dict]] = None,
                   sections: Optional[List[dict]] = None,
                   title: str = "",
                   artist: str = "",
                   tempo: float = 0) -> dict:
    """
    Assemble a chord chart from detected chords, lyrics, and sections.

    Args:
        chords: list of {chord, time, duration, confidence}
        lyrics: list of {text, start_time, end_time} (optional)
        sections: list of {name, start_time, end_time} (optional)
        title: song title
        artist: artist name
        tempo: BPM (0 = auto-estimate)

    Returns:
        dict matching the chord_chart.json format for renderManualChordChart()
    """
    # Sort inputs by time
    chords = sorted(chords, key=lambda c: c.get('time', 0))

    # Estimate tempo if not provided
    if not tempo or tempo <= 0:
        tempo = _estimate_tempo(chords)
    beats_per_bar = 4  # 4/4 time assumed

    # Handle no-chords edge case (lyrics only)
    if not chords:
        if not lyrics:
            return {
                "title": title,
                "artist": artist,
                "source": "auto",
                "sections": []
            }
        # Lyrics only
        out_sections = []
        effective_sections = sections or _auto_section(lyrics)
        assigned = _assign_lines_to_sections(lyrics, effective_sections)
        for si, sec in enumerate(effective_sections):
            lines = []
            for l in assigned.get(si, []):
                lines.append({"chords": "", "lyrics": l.get('text', '')})
            if lines:
                out_sections.append({"name": sec.get('name', f'Section {si+1}'), "lines": lines})
        return {
            "title": title,
            "artist": artist,
            "source": "auto",
            "sections": out_sections
        }

    # --- Chords available ---

    has_lyrics = lyrics and len(lyrics) > 0

    if not has_lyrics:
        # Chord-only chart
        effective_sections = sections or [{
            'name': 'Song',
            'start_time': chords[0]['time'],
            'end_time': chords[-1]['time'] + chords[-1].get('duration', 2.0)
        }]
        out_sections = []
        for sec in effective_sections:
            s = sec.get('start_time', 0)
            e = sec.get('end_time', s + 30)
            sec_chords = _find_chords_in_range(chords, s, e)
            if sec_chords:
                lines = _format_instrumental_chords(sec_chords)
                # Add chord_beats data to instrumental lines
                beat_lines = _compute_instrumental_chord_beats(sec_chords, tempo, beats_per_bar)
                for li, line in enumerate(lines):
                    if li < len(beat_lines):
                        line['chord_beats'] = beat_lines[li]
            else:
                lines = []
            if lines:
                out_sections.append({
                    "name": sec.get('name', 'Section'),
                    "lines": lines
                })
        return {
            "title": title,
            "artist": artist,
            "tempo": round(tempo),
            "beats_per_bar": beats_per_bar,
            "source": "auto",
            "sections": out_sections
        }

    # --- Chords + Lyrics ---
    lyrics = sorted(lyrics, key=lambda l: l.get('start_time', 0))

    # Determine sections
    if sections:
        effective_sections = sorted(sections, key=lambda s: s.get('start_time', 0))
    else:
        effective_sections = _auto_section(lyrics)

    # Refine section boundaries using chord pattern changes.
    # For each section boundary, find the nearest chord change that signals
    # a new pattern (e.g., verse chords → chorus chords) and snap to it.
    if len(effective_sections) > 1 and chords:
        for si in range(1, len(effective_sections)):
            boundary = effective_sections[si].get('start_time', 0)
            # Look for chord changes within 5s before/after the boundary
            nearby_chords = [c for c in chords if abs(c['time'] - boundary) < 5.0]
            if nearby_chords:
                # Find the chord closest to but before the boundary that differs
                # from the previous section's dominant chord pattern
                prev_start = effective_sections[si - 1].get('start_time', 0)
                prev_chords = [c['chord'] for c in chords
                               if prev_start <= c['time'] < boundary - 2]
                if prev_chords:
                    # Most common chord in previous section
                    from collections import Counter
                    prev_common = Counter(prev_chords).most_common(3)
                    prev_roots = {c for c, _ in prev_common}
                    # Find the first chord near the boundary that's NOT in prev pattern
                    for c in sorted(nearby_chords, key=lambda x: x['time']):
                        if c['chord'] not in prev_roots and c['time'] < boundary + 3:
                            # Snap boundary to this chord's time
                            effective_sections[si]['start_time'] = c['time']
                            effective_sections[si - 1]['end_time'] = c['time']
                            break

    assigned = _assign_lines_to_sections(lyrics, effective_sections)

    out_sections = []
    last_chord = None  # track last chord for carry-forward

    for si, sec in enumerate(effective_sections):
        sec_start = sec.get('start_time', 0)
        sec_end = sec.get('end_time', sec_start + 30)
        sec_lyrics = assigned.get(si, [])
        sec_name = sec.get('name', f'Section {si + 1}')

        lines = []

        if not sec_lyrics:
            # Instrumental section
            sec_chords = _find_chords_in_range(chords, sec_start, sec_end)
            if sec_chords:
                last_chord = sec_chords[-1]['chord']
                lines = _format_instrumental_chords(sec_chords)
        else:
            for li, lyric_line in enumerate(sec_lyrics):
                text = lyric_line.get('text', '')
                l_start = lyric_line.get('start_time', sec_start)
                l_end = lyric_line.get('end_time', l_start + 5)

                line_chords = _find_chords_in_range(chords, l_start, l_end)

                if line_chords:
                    last_chord = line_chords[-1]['chord']
                    chord_str = _format_chord_line(line_chords, text, l_start, l_end)
                    chord_beats = _compute_chord_beats_for_line(
                        line_chords, tempo, l_start, l_end, beats_per_bar)
                elif last_chord:
                    # No chord change on this line — show carried chord at start
                    chord_str = last_chord.ljust(len(text)) if text else last_chord
                    chord_beats = [{'name': last_chord, 'beats': beats_per_bar}]
                else:
                    chord_str = ""
                    chord_beats = []

                line_dict = {"chords": chord_str, "lyrics": text}
                if chord_beats:
                    line_dict["chord_beats"] = chord_beats
                lines.append(line_dict)

        # Also check for instrumental gap before first lyric in this section
        if sec_lyrics:
            first_lyric_start = sec_lyrics[0].get('start_time', sec_start)
            if first_lyric_start - sec_start > 2.0:
                # There's an instrumental intro to this section
                intro_chords = _find_chords_in_range(chords, sec_start, first_lyric_start)
                if intro_chords:
                    intro_lines = _format_instrumental_chords(intro_chords)
                    lines = intro_lines + lines

        if lines:
            out_sections.append({"name": sec_name, "lines": lines})

    return {
        "title": title,
        "artist": artist,
        "tempo": round(tempo),
        "beats_per_bar": beats_per_bar,
        "source": "auto",
        "sections": out_sections
    }


def save_chart(chart: dict, output_path: str) -> None:
    """Save chart dict to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(chart, f, indent=2)


def generate_chord_chart_for_job(
    job_id: str,
    chords: List[dict],
    vocals_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    title: str = "Untitled",
    artist: str = "",
    output_dir: Optional[str] = None,
) -> Optional[dict]:
    """
    Full chord chart generation pipeline for a job.

    Orchestrates:
    1. Lyrics extraction (if vocals_path provided)
    2. Structure detection
    3. Chart assembly + save

    Fault-tolerant: if lyrics or structure detection fails,
    falls back to a chord-only chart.

    Returns the assembled chord chart dict, or None on total failure.
    """
    import logging
    import os

    logger = logging.getLogger(__name__)

    from models.job import OUTPUT_DIR

    if output_dir is None:
        output_dir = str(OUTPUT_DIR / job_id)

    os.makedirs(output_dir, exist_ok=True)

    lyrics = None
    sections = None

    # Step 1: Extract lyrics from vocals stem
    if vocals_path and os.path.exists(vocals_path):
        try:
            from lyrics_extractor import extract_lyrics
            lyrics = extract_lyrics(
                vocals_path,
                output_dir=output_dir,
                audio_path=audio_path,
                chords=chords,
            )
            logger.info(f"Extracted {len(lyrics)} lyric lines")
        except Exception as e:
            logger.warning(f"Lyrics extraction failed (will generate chord-only chart): {e}")
            lyrics = None

    # Step 2: Detect song structure
    try:
        from structure_detector import detect_structure
        structure_path = os.path.join(output_dir, 'structure.json')

        # structure_detector requires audio_path as first arg
        source_audio = audio_path or vocals_path
        if source_audio and os.path.exists(source_audio):
            sections = detect_structure(
                audio_path=source_audio,
                chords=chords,
                lyrics=lyrics,
                output_path=structure_path,
            )
            logger.info(f"Detected {len(sections)} sections")
        else:
            logger.warning("No audio path for structure detection — skipping")
            sections = None
    except Exception as e:
        logger.warning(f"Structure detection failed (will use flat chart): {e}")
        sections = None

    # Step 3: Assemble and save chart
    try:
        chart_path = os.path.join(output_dir, 'chord_chart.json')

        # Don't overwrite a manual chart — save as auto variant
        manual_path = os.path.join(output_dir, 'chord_chart_manual.json')
        if os.path.exists(manual_path):
            logger.info("Manual chord chart exists — saving auto chart as chord_chart_auto.json")
            chart_path = os.path.join(output_dir, 'chord_chart_auto.json')

        chart = assemble_chart(
            chords=chords,
            lyrics=lyrics,
            sections=sections,
            title=title,
            artist=artist,
        )
        save_chart(chart, chart_path)
        logger.info(f"Chord chart saved to {chart_path}")
        return chart
    except Exception as e:
        logger.error(f"Chart assembly failed: {e}")
        return None
