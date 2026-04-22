"""
Chord sheet generator — combines chord events + synced lyrics
into UG-style formatted chord sheets with chords positioned above lyrics.

Supports two sources:
  1. Songsterr song_id → fetches chords from Songsterr
  2. StemScriber job_id → uses our own BTC v10 chord detection

Uses faster-whisper word-level timestamps when audio is available
for accurate chord-over-word placement.
"""

import logging
import os
import re
import glob
from flask import Blueprint, request, jsonify, Response
from models.job import get_job
from middleware.validation import validate_job_id

logger = logging.getLogger(__name__)

chord_sheet_bp = Blueprint("chord_sheet", __name__)

# Directory where StemScriber stores job outputs
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def _find_vocal_stem(title, artist):
    """Try to find a vocal stem file for this song."""
    if not os.path.isdir(OUTPUTS_DIR):
        return None
    # Look for vocal stems matching the song
    patterns = [
        f"*{title}*vocals*",
        f"*{title}*vocal*",
        f"*{artist}*{title}*vocals*",
    ]
    for pattern in patterns:
        matches = glob.glob(os.path.join(OUTPUTS_DIR, "**", pattern), recursive=True)
        for m in matches:
            if m.endswith((".wav", ".mp3", ".flac")):
                return m
    return None


def _get_word_timestamps_for_lyrics(synced_lyrics, title="", artist=""):
    """
    Try to get word-level timestamps using faster-whisper on the vocal stem.
    Returns enhanced lyrics with per-word timing, or None if unavailable.
    """
    vocal_path = _find_vocal_stem(title, artist)
    if not vocal_path:
        return None

    try:
        from word_timestamps import get_word_timestamps, align_lyrics_with_words
        word_ts = get_word_timestamps(vocal_path)
        if word_ts:
            return align_lyrics_with_words(synced_lyrics, word_ts)
    except Exception as e:
        logger.warning(f"Word timestamp extraction failed: {e}")

    return None


def _place_chord_on_word(chord_time, lyric_entry):
    """
    Find the best word position for a chord using word-level timestamps.
    Returns char_pos or None if no good match.
    """
    words = lyric_entry.get("words", [])
    if not words:
        return None

    best_idx = 0
    best_dist = float("inf")
    for i, w in enumerate(words):
        if w.get("start") is not None:
            dist = abs(w["start"] - chord_time)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

    return words[best_idx].get("char_pos", 0)


def _estimate_word_position(chord_time, line_time, next_time, words, char_positions):
    """
    Fallback: estimate chord position using compressed time model.
    Words are assumed to be sung in ~70% of the line duration (rest is pause).
    Skip common articles/prepositions — snap forward to content words.
    """
    SKIP_WORDS = {"a", "an", "the", "to", "in", "on", "at", "by", "of", "is",
                  "it", "and", "or", "but", "with", "for", "so", "no", "not",
                  "i", "my", "me", "we", "he", "she", "up", "if", "do", "be"}

    line_duration = next_time - line_time
    if line_duration <= 0:
        line_duration = 5.0

    # Compress word times to 70% of line duration (accounts for inter-line pause)
    effective_duration = line_duration * 0.7
    n = len(words)

    word_times = [line_time + (wi / max(n, 1)) * effective_duration for wi in range(n)]

    # Find closest word by time
    best_idx = 0
    best_dist = abs(word_times[0] - chord_time)
    for wi in range(1, n):
        dist = abs(word_times[wi] - chord_time)
        if dist < best_dist:
            best_dist = dist
            best_idx = wi

    # Skip articles/prepositions — snap forward to next content word
    while best_idx < n - 1 and words[best_idx].lower().strip(".,!?;:'\"()") in SKIP_WORDS:
        best_idx += 1

    return char_positions[best_idx]


def _align_chords_to_lyrics(chord_events, synced_lyrics, sections, title="", artist="", tempo=0):
    """
    Align chord events (with timestamps) to synced lyrics (with timestamps).
    Returns a formatted chord sheet string with [ch]...[/ch] markup for chords.
    """
    if not chord_events or not synced_lyrics:
        return None

    # Build section time map
    section_times = {}
    if sections:
        beats_per_measure = 4
        if tempo and tempo > 0:
            sec_per_measure = (60.0 / tempo) * beats_per_measure
            for s in sections:
                measure = s.get("measure", 1)
                section_times[round((measure - 1) * sec_per_measure, 1)] = s.get("name", "")

    # Sort events by time
    chord_events = sorted(chord_events, key=lambda e: e["time"])
    synced_lyrics = sorted(synced_lyrics, key=lambda l: l["time"])

    # Try to get word-level timestamps from Whisper
    enhanced_lyrics = _get_word_timestamps_for_lyrics(synced_lyrics, title, artist)
    has_word_timestamps = enhanced_lyrics is not None
    if has_word_timestamps:
        logger.info("Using Whisper word-level timestamps for chord placement")
    else:
        logger.info("Using estimated word timing (no vocal stem available)")

    # --- Assign each chord to the lyric line it falls within ---
    line_chords = [[] for _ in synced_lyrics]
    first_lyric_time = synced_lyrics[0]["time"]
    pre_lyric_chords = []

    for ce in chord_events:
        ct = ce["time"]
        if ct < first_lyric_time - 0.5:
            pre_lyric_chords.append(ce)
            continue

        assigned = False
        for li in range(len(synced_lyrics) - 1, -1, -1):
            if ct >= synced_lyrics[li]["time"] - 0.5:
                line_chords[li].append(ce)
                assigned = True
                break
        if not assigned:
            pre_lyric_chords.append(ce)

    # Find which chord is active at any given time
    def chord_at_time(t):
        active = None
        for e in chord_events:
            if e["time"] <= t + 0.5:
                active = e["chord"]
            else:
                break
        return active

    lines = []
    used_sections = set()
    last_chord_name = ""

    # Add intro chords (before lyrics start)
    if pre_lyric_chords:
        for st, sname in sorted(section_times.items()):
            if st < first_lyric_time and st not in used_sections:
                lines.append("")
                lines.append("[" + sname + "]")
                used_sections.add(st)
        chord_line = "  ".join("[ch]" + c["chord"] + "[/ch]" for c in pre_lyric_chords)
        lines.append(chord_line)
        last_chord_name = pre_lyric_chords[-1]["chord"]

    for i, lyric in enumerate(synced_lyrics):
        text = lyric.get("text", "").strip()
        line_time = lyric["time"]
        next_time = synced_lyrics[i + 1]["time"] if i + 1 < len(synced_lyrics) else line_time + 10

        # Section markers
        for st, sname in sorted(section_times.items()):
            if st not in used_sections and st <= line_time + 1.0:
                lines.append("")
                lines.append("[" + sname + "]")
                used_sections.add(st)

        my_chords = line_chords[i]

        # Deduplicate: skip if first chord same as last chord from previous line
        if my_chords and my_chords[0]["chord"] == last_chord_name:
            my_chords = my_chords[1:]

        if my_chords:
            last_chord_name = my_chords[-1]["chord"]

        if not text:
            if my_chords:
                chord_line = "  ".join("[ch]" + c["chord"] + "[/ch]" for c in my_chords)
                lines.append(chord_line)
            continue

        if not my_chords:
            active = chord_at_time(line_time)
            if active and active != last_chord_name:
                lines.append("[ch]" + active + "[/ch]")
                last_chord_name = active
            lines.append(text)
            continue

        # Position chords above correct words
        words = text.split()
        if not words:
            chord_line = "  ".join("[ch]" + c["chord"] + "[/ch]" for c in my_chords)
            lines.append(chord_line)
            lines.append(text)
            continue

        # Character positions for each word
        char_positions = []
        pos = 0
        for w in words:
            char_positions.append(pos)
            pos += len(w) + 1

        # Build chord placement
        placement_data = []
        for ce in my_chords:
            ct = ce["time"]
            chord_name = ce["chord"]

            # Use Whisper word timestamps if available
            if has_word_timestamps and i < len(enhanced_lyrics):
                char_pos = _place_chord_on_word(ct, enhanced_lyrics[i])
                if char_pos is None:
                    char_pos = _estimate_word_position(ct, line_time, next_time, words, char_positions)
            else:
                char_pos = _estimate_word_position(ct, line_time, next_time, words, char_positions)

            # Avoid overlapping with previously placed chords
            for prev_pos, prev_name in placement_data:
                if char_pos >= prev_pos and char_pos < prev_pos + len(prev_name) + 1:
                    char_pos = prev_pos + len(prev_name) + 1

            placement_data.append((char_pos, chord_name))

        if placement_data:
            lines.append(_build_tagged_line(placement_data, len(text) + 20))
        lines.append(text)

    # Remaining sections
    for st, sname in sorted(section_times.items()):
        if st not in used_sections:
            lines.append("")
            lines.append("[" + sname + "]")

    return "\n".join(lines)


def _build_tagged_line(placements, line_width):
    """Build a chord line with [ch]...[/ch] tags at correct character positions."""
    placements = sorted(placements, key=lambda p: p[0])

    result = []
    pos = 0
    for char_pos, chord_name in placements:
        while pos < char_pos:
            result.append(" ")
            pos += 1
        result.append("[ch]" + chord_name + "[/ch]")
        pos += len(chord_name)

    return "".join(result).rstrip()


def _build_chord_sheet(chord_events, synced_lyrics, sections, title="", artist="", tempo=0):
    """Build a complete chord sheet with header and content."""
    content = _align_chords_to_lyrics(chord_events, synced_lyrics, sections, title, artist, tempo)
    if not content:
        return None

    seen = set()
    chords_used = []
    for e in chord_events:
        ch = e["chord"]
        if ch not in seen:
            seen.add(ch)
            chords_used.append(ch)

    return {
        "title": title,
        "artist": artist,
        "tempo": tempo,
        "chords_used": chords_used,
        "content": content,
        "source": "StemScriber AI",
    }


@chord_sheet_bp.route("/api/chord-sheet/<int:song_id>", methods=["GET"])
def generate_chord_sheet(song_id):
    """Generate a formatted chord sheet from Songsterr data + synced lyrics."""
    try:
        from routes.songsterr import fetch_chord_data

        data = fetch_chord_data(song_id)

        chord_events = data.get("chordEvents", [])
        synced_lyrics_raw = data.get("syncedLyrics", [])
        sections = data.get("sections", [])
        title = data.get("title", "")
        artist = data.get("artist", "")
        tempo = data.get("tempo", 0)
        plain_lyrics = data.get("lyrics", "")

        # Parse synced lyrics
        synced_lyrics = []
        if isinstance(synced_lyrics_raw, list):
            synced_lyrics = synced_lyrics_raw
        elif isinstance(synced_lyrics_raw, str):
            import re
            for line in synced_lyrics_raw.split("\n"):
                m = re.match(r"\[(\d+):(\d+\.\d+)\](.*)", line)
                if m:
                    mins, secs, text = m.groups()
                    t = float(mins) * 60 + float(secs)
                    synced_lyrics.append({"time": t, "text": text.strip()})

        if not synced_lyrics and plain_lyrics:
            return jsonify({
                "title": title,
                "artist": artist,
                "tempo": tempo,
                "chords_used": list({e["chord"] for e in chord_events}),
                "content": plain_lyrics,
                "source": "StemScriber (no synced lyrics)",
            })

        if not chord_events:
            return jsonify({"error": "No chord data available for this song"}), 404

        result = _build_chord_sheet(chord_events, synced_lyrics, sections, title, artist, tempo)
        if not result:
            return jsonify({"error": "Could not generate chord sheet"}), 500

        return jsonify(result)

    except Exception as e:
        logger.error(f"Chord sheet generation failed: {e}", exc_info=True)
        return jsonify({"error": "Chord sheet generation failed", "details": str(e)}), 500


@chord_sheet_bp.route("/api/chord-sheet/job/<job_id>", methods=["GET"])
def generate_chord_sheet_from_job(job_id):
    """Generate a chord sheet from a StemScriber job's own chord detection (BTC v10)."""
    if not validate_job_id(job_id):
        return jsonify({"error": "Invalid job ID"}), 400
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    # Prefer the pre-built chord_chart.json which has sections with lyrics
    from models.job import OUTPUT_DIR
    chart_path = OUTPUT_DIR / job_id / "chord_chart.json"
    if chart_path.exists():
        try:
            import json as _json
            with open(chart_path) as f:
                chart_data = _json.load(f)
            if chart_data.get("sections"):
                return jsonify(chart_data)
        except Exception:
            pass  # Fall through to generated version

    chord_progression = job.chord_progression
    if not chord_progression:
        return jsonify({"error": "No chord data detected for this job"}), 404

    # Build chord events in the format _build_chord_sheet expects
    chord_events = [{"time": c["time"], "chord": c["chord"]} for c in chord_progression]

    title = ""
    artist = ""
    if job.metadata:
        title = job.metadata.get("title", "")
        artist = job.metadata.get("artist", "")
    if not title:
        title = getattr(job, "original_filename", "") or "Unknown"

    detected_key = getattr(job, "detected_key", None) or ""

    # Build unique chord list preserving order
    seen = set()
    chords_used = []
    for e in chord_events:
        ch = e["chord"]
        if ch not in seen:
            seen.add(ch)
            chords_used.append(ch)

    # Build a simple chord-only sheet (no lyrics — we don't have synced lyrics for uploaded audio)
    lines = []
    if detected_key:
        lines.append(f"Key: {detected_key}")
        lines.append("")

    # Group chords by time blocks (~4 seconds per line for readability)
    LINE_DURATION = 4.0
    current_line_chords = []
    line_start = chord_events[0]["time"] if chord_events else 0

    for ce in chord_events:
        if ce["time"] - line_start > LINE_DURATION and current_line_chords:
            chord_line = "  ".join("[ch]" + c + "[/ch]" for c in current_line_chords)
            lines.append(chord_line)
            current_line_chords = []
            line_start = ce["time"]
        current_line_chords.append(ce["chord"])

    if current_line_chords:
        chord_line = "  ".join("[ch]" + c + "[/ch]" for c in current_line_chords)
        lines.append(chord_line)

    detector_version = chord_progression[0].get("detector_version", "v10") if chord_progression else "v10"

    return jsonify({
        "title": title,
        "artist": artist,
        "key": detected_key,
        "chords_used": chords_used,
        "content": "\n".join(lines),
        "source": f"StemScriber AI ({detector_version})",
        "chord_events": chord_events,
    })


# ============ EXPORT HELPERS ============

def _get_job_info(job_id):
    """Get job, chord events, title, artist, key, and tempo for export endpoints."""
    if not validate_job_id(job_id):
        return None, "Invalid job ID"
    job = get_job(job_id)
    if not job:
        return None, "Job not found"

    chord_progression = job.chord_progression
    if not chord_progression:
        return None, "No chord data detected for this job"

    chord_events = [{"time": c["time"], "chord": c["chord"]} for c in chord_progression]

    title = ""
    artist = ""
    if job.metadata:
        title = job.metadata.get("title", "")
        artist = job.metadata.get("artist", "")
    if not title:
        title = getattr(job, "original_filename", "") or "Unknown"

    detected_key = getattr(job, "detected_key", None) or ""

    # Clean YouTube junk from title (e.g. "Kozelski - The Time Comes  -  Lyric Video")
    clean_title = title
    clean_title = re.sub(
        r'\s*[-–—|]\s*(Official\s*(Video|Audio|Music\s*Video|Lyric\s*Video)?|'
        r'Lyric\s*Video|Music\s*Video|'
        r'HQ|HD|4K|Lyrics?|Audio|Full\s*Album|Remaster(ed)?|Live|'
        r'ft\.?\s*.+|feat\.?\s*.+)\s*$',
        '', clean_title, flags=re.IGNORECASE
    )
    clean_title = re.sub(
        r'\s*[\[\(](Official|Video|Audio|Lyrics?|HQ|HD|4K|Live|Remastered|Lyric Video)[\]\)]\s*',
        '', clean_title, flags=re.IGNORECASE
    )
    # If title looks like "Artist - Song Title", extract just the song title
    for sep in [' - ', ' – ', ' — ', ' | ']:
        if sep in clean_title:
            parts = clean_title.split(sep, 1)
            if not artist:
                artist = parts[0].strip()
            clean_title = parts[1].strip()
            break
    # Strip any remaining trailing junk after split
    clean_title = re.sub(
        r'\s*[-–—|]\s*(Lyric\s*Video|Music\s*Video|Official\s*Video|Audio|Live)\s*$',
        '', clean_title, flags=re.IGNORECASE
    )
    clean_title = clean_title.strip(' -–—')

    # Estimate tempo from chord events if we have enough data
    tempo = job.metadata.get("tempo", 0) if job.metadata else 0

    return {
        "job": job,
        "chord_events": chord_events,
        "title": clean_title or title,
        "artist": artist,
        "key": detected_key,
        "tempo": tempo,
    }, None


def _fetch_synced_lyrics_for_job(job, title, artist):
    """
    Try to get synced lyrics for a job:
    1. LRCLIB (online lyrics database)
    2. Whisper transcription of the vocal stem (fallback)

    Returns list of {"time": float, "text": str} or empty list.
    """
    import requests

    LRCLIB_API = "https://lrclib.net/api"

    # --- Try LRCLIB first ---
    try:
        if artist and title:
            params = {"artist_name": artist, "track_name": title}
        else:
            params = {"q": title}

        logger.info(f"ChordPro: Trying LRCLIB for '{title}' by '{artist}'")
        resp = requests.get(f"{LRCLIB_API}/search", params=params, timeout=10)
        resp.raise_for_status()
        results = resp.json()

        if not results and artist:
            resp2 = requests.get(f"{LRCLIB_API}/search", params={"q": f"{artist} {title}"}, timeout=10)
            resp2.raise_for_status()
            results = resp2.json()

        if results:
            # Prefer result with synced lyrics
            best = None
            for r in results:
                if r.get("syncedLyrics"):
                    best = r
                    break
            if not best:
                best = results[0]

            synced_lrc = best.get("syncedLyrics")
            if synced_lrc:
                from routes.lyrics import _parse_lrc
                parsed = _parse_lrc(synced_lrc)
                if parsed:
                    logger.info(f"ChordPro: Got {len(parsed)} synced lyrics lines from LRCLIB")
                    return parsed
    except Exception as e:
        logger.warning(f"ChordPro: LRCLIB lookup failed: {e}")

    # --- Fallback: Whisper on vocal stem ---
    vocal_path = None
    if job.stems and "vocals" in job.stems:
        vp = job.stems["vocals"]
        if os.path.exists(vp):
            vocal_path = vp

    if not vocal_path:
        vocal_path = _find_vocal_stem(title, artist)

    if not vocal_path:
        logger.info("ChordPro: No vocal stem found for Whisper fallback")
        return []

    try:
        logger.info(f"ChordPro: Running Whisper on vocal stem: {os.path.basename(vocal_path)}")
        from word_timestamps import get_word_timestamps

        word_ts = get_word_timestamps(vocal_path)
        if not word_ts:
            return []

        # Group words into lines (~3-5 seconds per line, or break on long pauses)
        synced_lines = []
        current_words = []
        line_start = None
        MAX_LINE_DURATION = 5.0
        PAUSE_THRESHOLD = 1.5  # seconds of silence triggers new line

        for i, w in enumerate(word_ts):
            if line_start is None:
                line_start = w["start"]
                current_words.append(w["word"])
                continue

            # Check for pause gap
            prev_end = word_ts[i - 1]["end"]
            gap = w["start"] - prev_end

            # Check if line is getting too long
            line_duration = w["end"] - line_start

            if gap > PAUSE_THRESHOLD or line_duration > MAX_LINE_DURATION:
                # Flush current line
                if current_words:
                    synced_lines.append({
                        "time": line_start,
                        "text": " ".join(current_words),
                    })
                current_words = [w["word"]]
                line_start = w["start"]
            else:
                current_words.append(w["word"])

        # Flush last line
        if current_words:
            synced_lines.append({
                "time": line_start,
                "text": " ".join(current_words),
            })

        logger.info(f"ChordPro: Whisper produced {len(synced_lines)} lyric lines")
        return synced_lines

    except Exception as e:
        logger.warning(f"ChordPro: Whisper transcription failed: {e}")
        return []


def _build_chordpro(chord_events, synced_lyrics, title, artist, key, tempo):
    """
    Build a ChordPro (.cho) formatted string.

    Chords go inline with lyrics: [Am]word [G]word
    Lines without lyrics get chords only: [F#m] [A] [Bm] [E]
    Section markers: {comment: Section Name}
    """
    lines = []

    # Header directives
    lines.append(f"{{title: {title}}}")
    if artist:
        lines.append(f"{{artist: {artist}}}")
    if key:
        lines.append(f"{{key: {key}}}")
    if tempo:
        lines.append(f"{{tempo: {tempo}}}")
    lines.append("")

    if not synced_lyrics:
        # No lyrics — output chords grouped by time
        lines.append("{comment: Chords}")
        LINE_DURATION = 4.0
        current_line = []
        line_start = chord_events[0]["time"] if chord_events else 0

        for ce in chord_events:
            if ce["time"] - line_start > LINE_DURATION and current_line:
                lines.append(" ".join(f"[{c}]" for c in current_line))
                current_line = []
                line_start = ce["time"]
            current_line.append(ce["chord"])

        if current_line:
            lines.append(" ".join(f"[{c}]" for c in current_line))

        return "\n".join(lines)

    # Sort both by time
    chord_events = sorted(chord_events, key=lambda e: e["time"])
    synced_lyrics = sorted(synced_lyrics, key=lambda l: l["time"])

    # Assign chords to lyric lines
    first_lyric_time = synced_lyrics[0]["time"]
    line_chords = [[] for _ in synced_lyrics]
    pre_lyric_chords = []

    for ce in chord_events:
        ct = ce["time"]
        if ct < first_lyric_time - 0.5:
            pre_lyric_chords.append(ce)
            continue

        assigned = False
        for li in range(len(synced_lyrics) - 1, -1, -1):
            if ct >= synced_lyrics[li]["time"] - 0.5:
                line_chords[li].append(ce)
                assigned = True
                break
        if not assigned:
            pre_lyric_chords.append(ce)

    last_chord = ""

    # Pre-lyrics (intro)
    if pre_lyric_chords:
        lines.append("{comment: Intro}")
        chord_line = " ".join(f"[{c['chord']}]" for c in pre_lyric_chords)
        lines.append(chord_line)
        lines.append("")
        last_chord = pre_lyric_chords[-1]["chord"]

    # Detect sections from gaps in lyrics (>5 seconds gap = likely new section)
    section_count = 0
    section_names = ["Verse", "Chorus", "Verse 2", "Chorus 2", "Bridge", "Verse 3",
                     "Chorus 3", "Outro"]

    for i, lyric in enumerate(synced_lyrics):
        text = lyric.get("text", "").strip()
        line_time = lyric["time"]
        next_time = synced_lyrics[i + 1]["time"] if i + 1 < len(synced_lyrics) else line_time + 10

        # Detect section breaks (large gap before this line)
        if i > 0:
            prev_time = synced_lyrics[i - 1]["time"]
            gap = line_time - prev_time
            if gap > 6.0:
                lines.append("")
                section_label = section_names[section_count] if section_count < len(section_names) else f"Section {section_count + 1}"
                lines.append(f"{{comment: {section_label}}}")
                section_count += 1

                # Any instrumental chords in the gap
                gap_chords = [ce for ce in chord_events
                              if prev_time + 1 < ce["time"] < line_time - 0.5
                              and ce["chord"] != last_chord]
                if gap_chords:
                    lines.append(" ".join(f"[{c['chord']}]" for c in gap_chords))
                    last_chord = gap_chords[-1]["chord"]
                    lines.append("")

        my_chords = line_chords[i]

        # Deduplicate leading chord if same as last
        if my_chords and my_chords[0]["chord"] == last_chord:
            my_chords = my_chords[1:]
        if my_chords:
            last_chord = my_chords[-1]["chord"]

        if not text:
            # Instrumental line — chords only
            if my_chords:
                lines.append(" ".join(f"[{c['chord']}]" for c in my_chords))
            continue

        if not my_chords:
            # Lyric line with no new chords
            lines.append(text)
            continue

        # Insert chords inline with lyrics
        words = text.split()
        if not words:
            lines.append(text)
            continue

        # Estimate which word each chord falls on
        n_words = len(words)
        line_duration = next_time - line_time
        if line_duration <= 0:
            line_duration = 5.0
        effective_duration = line_duration * 0.7

        # Build word timing estimates
        word_times = [line_time + (wi / max(n_words, 1)) * effective_duration for wi in range(n_words)]

        # Assign each chord to the closest word
        chord_word_map = {}  # word_index -> chord_name
        for ce in my_chords:
            ct = ce["time"]
            best_idx = 0
            best_dist = abs(word_times[0] - ct)
            for wi in range(1, n_words):
                dist = abs(word_times[wi] - ct)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = wi
            # Don't overwrite if already assigned — shift to next word
            while best_idx in chord_word_map and best_idx < n_words - 1:
                best_idx += 1
            chord_word_map[best_idx] = ce["chord"]

        # Build the line with inline chords
        result_parts = []
        for wi, word in enumerate(words):
            if wi in chord_word_map:
                result_parts.append(f"[{chord_word_map[wi]}]{word}")
            else:
                result_parts.append(word)

        lines.append(" ".join(result_parts))

    return "\n".join(lines)


def _build_plain_text_sheet(chord_events, synced_lyrics, title, artist, key, tempo):
    """
    Build a plain text chord sheet (UG-style: chords above lyrics).
    """
    lines = []

    # Header
    if title:
        lines.append(title)
    if artist:
        lines.append(f"by {artist}")
    if key:
        lines.append(f"Key: {key}")
    if tempo:
        lines.append(f"Tempo: {tempo} BPM")
    if lines:
        lines.append("")
        lines.append("=" * 50)
        lines.append("")

    if not synced_lyrics:
        # No lyrics — chords only
        LINE_DURATION = 4.0
        current_line = []
        line_start = chord_events[0]["time"] if chord_events else 0

        for ce in chord_events:
            if ce["time"] - line_start > LINE_DURATION and current_line:
                lines.append("  ".join(current_line))
                current_line = []
                line_start = ce["time"]
            current_line.append(ce["chord"])

        if current_line:
            lines.append("  ".join(current_line))

        return "\n".join(lines)

    # Sort
    chord_events = sorted(chord_events, key=lambda e: e["time"])
    synced_lyrics = sorted(synced_lyrics, key=lambda l: l["time"])

    # Assign chords to lines
    first_lyric_time = synced_lyrics[0]["time"]
    line_chords = [[] for _ in synced_lyrics]
    pre_lyric_chords = []

    for ce in chord_events:
        ct = ce["time"]
        if ct < first_lyric_time - 0.5:
            pre_lyric_chords.append(ce)
            continue
        assigned = False
        for li in range(len(synced_lyrics) - 1, -1, -1):
            if ct >= synced_lyrics[li]["time"] - 0.5:
                line_chords[li].append(ce)
                assigned = True
                break
        if not assigned:
            pre_lyric_chords.append(ce)

    last_chord = ""
    section_count = 0
    section_names = ["Verse", "Chorus", "Verse 2", "Chorus 2", "Bridge", "Verse 3",
                     "Chorus 3", "Outro"]

    # Intro chords
    if pre_lyric_chords:
        lines.append("[Intro]")
        lines.append("  ".join(c["chord"] for c in pre_lyric_chords))
        lines.append("")
        last_chord = pre_lyric_chords[-1]["chord"]

    for i, lyric in enumerate(synced_lyrics):
        text = lyric.get("text", "").strip()
        line_time = lyric["time"]
        next_time = synced_lyrics[i + 1]["time"] if i + 1 < len(synced_lyrics) else line_time + 10

        # Section breaks on large gaps
        if i > 0:
            prev_time = synced_lyrics[i - 1]["time"]
            gap = line_time - prev_time
            if gap > 6.0:
                lines.append("")
                section_label = section_names[section_count] if section_count < len(section_names) else f"Section {section_count + 1}"
                lines.append(f"[{section_label}]")
                section_count += 1

        my_chords = line_chords[i]
        if my_chords and my_chords[0]["chord"] == last_chord:
            my_chords = my_chords[1:]
        if my_chords:
            last_chord = my_chords[-1]["chord"]

        if not text:
            if my_chords:
                lines.append("  ".join(c["chord"] for c in my_chords))
            continue

        if not my_chords:
            lines.append(text)
            continue

        # Position chords above lyrics
        words = text.split()
        n_words = len(words)
        line_duration = next_time - line_time
        if line_duration <= 0:
            line_duration = 5.0
        effective_duration = line_duration * 0.7
        word_times = [line_time + (wi / max(n_words, 1)) * effective_duration for wi in range(n_words)]

        # Character positions for each word
        char_positions = []
        pos = 0
        for w in words:
            char_positions.append(pos)
            pos += len(w) + 1

        # Build chord line
        placements = []
        for ce in my_chords:
            ct = ce["time"]
            best_idx = 0
            best_dist = abs(word_times[0] - ct)
            for wi in range(1, n_words):
                dist = abs(word_times[wi] - ct)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = wi
            char_pos = char_positions[best_idx]
            # Avoid overlap
            for prev_pos, prev_name in placements:
                if char_pos >= prev_pos and char_pos < prev_pos + len(prev_name) + 1:
                    char_pos = prev_pos + len(prev_name) + 1
            placements.append((char_pos, ce["chord"]))

        # Render chord line
        chord_line_chars = [" "] * (len(text) + 20)
        for cpos, cname in sorted(placements):
            for ci, ch in enumerate(cname):
                if cpos + ci < len(chord_line_chars):
                    chord_line_chars[cpos + ci] = ch
        lines.append("".join(chord_line_chars).rstrip())
        lines.append(text)

    return "\n".join(lines)


# ============ EXPORT ENDPOINTS ============

@chord_sheet_bp.route("/api/chord-sheet/job/<job_id>/chordpro", methods=["GET"])
def export_chordpro(job_id):
    """Export chord sheet as ChordPro (.cho) file for OnSong compatibility."""
    info, err = _get_job_info(job_id)
    if not info:
        return jsonify({"error": err}), 404

    job = info["job"]
    chord_events = info["chord_events"]
    title = info["title"]
    artist = info["artist"]
    key = info["key"]
    tempo = info["tempo"]

    # Get synced lyrics
    synced_lyrics = _fetch_synced_lyrics_for_job(job, title, artist)

    # Build ChordPro
    chordpro_content = _build_chordpro(chord_events, synced_lyrics, title, artist, key, tempo)

    # Generate filename
    safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
    if not safe_title:
        safe_title = "chord_sheet"
    filename = f"{safe_title}.cho"

    return Response(
        chordpro_content,
        mimetype="text/plain",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Type": "text/plain; charset=utf-8",
        },
    )


@chord_sheet_bp.route("/api/chord-sheet/job/<job_id>/text", methods=["GET"])
def export_text(job_id):
    """Export chord sheet as plain text (UG-style chords above lyrics)."""
    info, err = _get_job_info(job_id)
    if not info:
        return jsonify({"error": err}), 404

    job = info["job"]
    chord_events = info["chord_events"]
    title = info["title"]
    artist = info["artist"]
    key = info["key"]
    tempo = info["tempo"]

    # Get synced lyrics
    synced_lyrics = _fetch_synced_lyrics_for_job(job, title, artist)

    # Build plain text sheet
    text_content = _build_plain_text_sheet(chord_events, synced_lyrics, title, artist, key, tempo)

    # Generate filename
    safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
    if not safe_title:
        safe_title = "chord_sheet"
    filename = f"{safe_title}.txt"

    return Response(
        text_content,
        mimetype="text/plain",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Type": "text/plain; charset=utf-8",
        },
    )
