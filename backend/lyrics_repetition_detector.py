"""
Lyrics Repetition Detector — identifies choruses and section boundaries
by finding repeating lyric patterns.

Choruses repeat with the same or similar lyrics. This module uses fuzzy
string matching (difflib) to find those repetitions and infer song structure.

Usage:
    from lyrics_repetition_detector import detect_lyrics_repetition
    result = detect_lyrics_repetition(lyrics)
    # result["chorus_lines"] — lines that repeat across the song
    # result["sections"] — inferred section boundaries with labels
"""

import logging
from difflib import SequenceMatcher
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Minimum similarity ratio to consider two lines as matching
SIMILARITY_THRESHOLD = 0.7

# Minimum gap (seconds) between lyric lines to consider an instrumental break
INSTRUMENTAL_GAP = 5.0

# Minimum number of instances for a line to be considered "repeating"
MIN_REPEAT_COUNT = 2

# Minimum group size (consecutive repeating lines) to form a chorus block
MIN_CHORUS_GROUP = 2


def _normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation."""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def _similarity(a: str, b: str) -> float:
    """Fuzzy similarity ratio between two normalized strings."""
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _find_repeating_lines(lyrics: List[Dict]) -> List[Dict]:
    """
    Find individual lyric lines that appear multiple times in the song.
    Returns list of {text, instances: [{start, end}]}.
    """
    n = len(lyrics)
    used = [False] * n  # track which lines have been grouped
    repeating = []

    for i in range(n):
        if used[i]:
            continue

        instances = [{"start": lyrics[i]["start_time"], "end": lyrics[i]["end_time"]}]

        for j in range(i + 1, n):
            if used[j]:
                continue
            sim = _similarity(lyrics[i]["text"], lyrics[j]["text"])
            if sim >= SIMILARITY_THRESHOLD:
                instances.append({"start": lyrics[j]["start_time"], "end": lyrics[j]["end_time"]})
                used[j] = True

        if len(instances) >= MIN_REPEAT_COUNT:
            used[i] = True
            repeating.append({
                "text": lyrics[i]["text"],
                "instances": instances,
            })

    return repeating


def _find_chorus_blocks(lyrics: List[Dict]) -> List[List[int]]:
    """
    Find groups of consecutive lines that repeat together as a unit (chorus blocks).

    Returns list of groups, where each group is a list of lyric indices
    representing one chorus instance. Groups that appear 2+ times are choruses.
    """
    n = len(lyrics)
    if n < MIN_CHORUS_GROUP:
        return []

    # Build a similarity matrix for all line pairs
    sim_matrix = {}
    for i in range(n):
        for j in range(i + 1, n):
            s = _similarity(lyrics[i]["text"], lyrics[j]["text"])
            if s >= SIMILARITY_THRESHOLD:
                sim_matrix[(i, j)] = s

    # Find matching consecutive sequences
    # For each pair of starting positions (i, j), extend as long as lines match
    best_blocks = []

    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in sim_matrix:
                continue

            # Extend the match
            length = 1
            while (i + length < j and  # don't overlap
                   j + length < n and
                   (i + length, j + length) in sim_matrix):
                length += 1

            if length >= MIN_CHORUS_GROUP:
                block_i = list(range(i, i + length))
                block_j = list(range(j, j + length))
                best_blocks.append((block_i, block_j, length))

    # Deduplicate: keep longest non-overlapping blocks
    best_blocks.sort(key=lambda x: x[2], reverse=True)
    used_indices = set()
    chorus_instances = []

    for block_i, block_j, length in best_blocks:
        # Check if any index already used
        all_indices = set(block_i + block_j)
        if all_indices & used_indices:
            continue

        chorus_instances.append((block_i, block_j))
        used_indices.update(all_indices)

    return chorus_instances


def _is_chorus_line(idx: int, chorus_blocks: List[Tuple]) -> bool:
    """Check if a lyric line index belongs to any chorus block."""
    for block_pair in chorus_blocks:
        for block in block_pair:
            if idx in block:
                return True
    return False


def _get_chorus_ranges(lyrics: List[Dict], chorus_blocks: List[Tuple]) -> List[Dict]:
    """
    Convert chorus block indices into time ranges.
    Returns list of {start_time, end_time} for each chorus instance.
    """
    ranges = []
    for block_pair in chorus_blocks:
        for block in block_pair:
            start = lyrics[block[0]]["start_time"]
            end = lyrics[block[-1]]["end_time"]
            ranges.append({"start_time": start, "end_time": end})

    ranges.sort(key=lambda x: x["start_time"])
    return ranges


def _build_sections(lyrics: List[Dict], chorus_ranges: List[Dict],
                    chorus_blocks: List[Tuple]) -> List[Dict]:
    """
    Build section labels from lyrics and identified chorus positions.

    Logic:
    - Lines before first chorus = Verse (or Intro if instrumental gap precedes them)
    - Repeated blocks = Chorus
    - Lines between choruses = Verse
    - Lines after last chorus that don't repeat = Bridge/Outro
    - Instrumental gaps (>5s) at start/end = Intro/Outro
    """
    if not lyrics:
        return []

    sections = []
    song_start = lyrics[0]["start_time"]
    song_end = lyrics[-1]["end_time"]

    # Detect instrumental intro (gap before first lyric)
    if song_start > INSTRUMENTAL_GAP:
        sections.append({
            "name": "Intro",
            "start_time": 0.0,
            "end_time": song_start,
            "confidence": 0.7,
        })

    if not chorus_ranges:
        # No chorus detected — label everything as verses with gaps as instrumental
        _add_verse_sections(lyrics, sections)
        return sections

    # Build chorus index set for quick lookup
    chorus_line_indices = set()
    for block_pair in chorus_blocks:
        for block in block_pair:
            chorus_line_indices.update(block)

    # Walk through lyrics and assign sections
    current_section_lines = []
    current_section_start = None
    verse_count = 0
    chorus_count = 0
    in_chorus = False

    for i, line in enumerate(lyrics):
        is_chorus = i in chorus_line_indices

        if is_chorus and not in_chorus:
            # Flush previous non-chorus section as verse
            if current_section_lines:
                verse_count += 1
                label = "Verse" if verse_count <= 10 else "Bridge"
                sections.append({
                    "name": label,
                    "start_time": current_section_start,
                    "end_time": current_section_lines[-1]["end_time"],
                    "confidence": 0.75,
                })
                current_section_lines = []
                current_section_start = None
            in_chorus = True
            current_section_start = line["start_time"]
            current_section_lines = [line]

        elif not is_chorus and in_chorus:
            # Flush chorus section
            chorus_count += 1
            sections.append({
                "name": "Chorus",
                "start_time": current_section_start,
                "end_time": current_section_lines[-1]["end_time"],
                "confidence": 0.9,
            })
            current_section_lines = []
            current_section_start = None
            in_chorus = False
            # Start new non-chorus section
            current_section_start = line["start_time"]
            current_section_lines = [line]

        else:
            # Continuing same section type
            if current_section_start is None:
                current_section_start = line["start_time"]
            current_section_lines.append(line)

        # Check for instrumental gap between this line and next
        if i + 1 < len(lyrics):
            gap = lyrics[i + 1]["start_time"] - line["end_time"]
            if gap > INSTRUMENTAL_GAP and current_section_lines:
                # Flush current section
                if in_chorus:
                    chorus_count += 1
                    sections.append({
                        "name": "Chorus",
                        "start_time": current_section_start,
                        "end_time": line["end_time"],
                        "confidence": 0.9,
                    })
                else:
                    verse_count += 1
                    sections.append({
                        "name": "Verse",
                        "start_time": current_section_start,
                        "end_time": line["end_time"],
                        "confidence": 0.75,
                    })

                # Insert instrumental/solo section for the gap
                sections.append({
                    "name": "Solo",
                    "start_time": line["end_time"],
                    "end_time": lyrics[i + 1]["start_time"],
                    "confidence": 0.5,
                })

                current_section_lines = []
                current_section_start = None
                in_chorus = False

    # Flush remaining section
    if current_section_lines:
        if in_chorus:
            chorus_count += 1
            sections.append({
                "name": "Chorus",
                "start_time": current_section_start,
                "end_time": current_section_lines[-1]["end_time"],
                "confidence": 0.9,
            })
        else:
            # After last chorus — could be bridge or outro
            is_after_chorus = any(
                cr["end_time"] <= current_section_start
                for cr in chorus_ranges
            )
            label = "Bridge" if is_after_chorus and verse_count >= 2 else "Verse"
            verse_count += 1
            sections.append({
                "name": label,
                "start_time": current_section_start,
                "end_time": current_section_lines[-1]["end_time"],
                "confidence": 0.65,
            })

    # Sort by time
    sections.sort(key=lambda s: s["start_time"])

    return sections


def _add_verse_sections(lyrics: List[Dict], sections: List[Dict]):
    """When no chorus is detected, split lyrics into verse sections based on gaps."""
    if not lyrics:
        return

    current_start = lyrics[0]["start_time"]
    current_end = lyrics[0]["end_time"]
    verse_num = 0

    for i in range(1, len(lyrics)):
        gap = lyrics[i]["start_time"] - lyrics[i - 1]["end_time"]
        if gap > INSTRUMENTAL_GAP:
            verse_num += 1
            sections.append({
                "name": "Verse",
                "start_time": current_start,
                "end_time": current_end,
                "confidence": 0.5,
            })
            # Insert Solo for the instrumental gap
            sections.append({
                "name": "Solo",
                "start_time": current_end,
                "end_time": lyrics[i]["start_time"],
                "confidence": 0.5,
            })
            current_start = lyrics[i]["start_time"]

        current_end = lyrics[i]["end_time"]

    verse_num += 1
    sections.append({
        "name": "Verse",
        "start_time": current_start,
        "end_time": current_end,
        "confidence": 0.5,
    })


def detect_lyrics_repetition(lyrics: List[Dict]) -> Dict:
    """
    Analyze lyric lines for repetition to identify choruses and section boundaries.

    Args:
        lyrics: List of {"text": str, "start_time": float, "end_time": float}

    Returns:
        {
            "chorus_lines": [{"text": str, "instances": [{"start": float, "end": float}]}],
            "sections": [{"name": str, "start_time": float, "end_time": float, "confidence": float}]
        }
    """
    if not lyrics:
        return {"chorus_lines": [], "sections": []}

    logger.info(f"Analyzing {len(lyrics)} lyric lines for repetition...")

    # Step 1: Find individual lines that repeat
    repeating_lines = _find_repeating_lines(lyrics)
    logger.info(f"Found {len(repeating_lines)} repeating lines")

    # Step 2: Find chorus blocks (consecutive repeating lines)
    chorus_blocks = _find_chorus_blocks(lyrics)
    logger.info(f"Found {len(chorus_blocks)} chorus block pairs")

    # Step 3: Get chorus time ranges
    chorus_ranges = _get_chorus_ranges(lyrics, chorus_blocks)

    # Step 4: Build section labels
    sections = _build_sections(lyrics, chorus_ranges, chorus_blocks)

    result = {
        "chorus_lines": repeating_lines,
        "sections": sections,
    }

    logger.info(f"Detected {len(sections)} sections, "
                f"{sum(1 for s in sections if s['name'] == 'Chorus')} choruses")

    return result


if __name__ == "__main__":
    import json
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python lyrics_repetition_detector.py <lyrics.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        lyrics_data = json.load(f)

    result = detect_lyrics_repetition(lyrics_data)

    print(f"\n{'='*60}")
    print(f"  Chorus Lines ({len(result['chorus_lines'])})")
    print(f"{'='*60}")
    for cl in result["chorus_lines"]:
        times = ", ".join(f"{inst['start']:.1f}-{inst['end']:.1f}s" for inst in cl["instances"])
        print(f"  \"{cl['text']}\" -> [{times}]")

    print(f"\n{'='*60}")
    print(f"  Sections ({len(result['sections'])})")
    print(f"{'='*60}")
    for s in result["sections"]:
        print(f"  [{s['start_time']:6.1f} - {s['end_time']:6.1f}] {s['name']:12s} (conf={s['confidence']:.2f})")
