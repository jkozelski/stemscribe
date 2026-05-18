#!/usr/bin/env python3
"""
Chord Library Cleanup Script for StemScriber
Audits and cleans all JSON chord charts, removing tab notation,
tab instructions, strumming patterns, and other non-lyric junk from lyrics fields.
"""

import json
import os
import re
import sys
from pathlib import Path

LIBRARY_DIR = Path(__file__).parent / "chord_library"

# --- Patterns that indicate a lyrics line is tab/junk ---

# Tab notation: e|---, B|---, G|--2--, etc. (standard 6-string tuning labels)
TAB_LINE_RE = re.compile(
    r'^[eEBGDAb]\|[\d\-\/\\hpbx~()\s|]+$'  # e|---2---3--- etc
)

# More generous tab detection: starts with a string name + pipe
TAB_PREFIX_RE = re.compile(
    r'^[eEBGDA]\|'
)

# Fret diagrams like "022xxx", "x244xx" embedded in lyrics
FRET_DIAGRAM_ONLY_RE = re.compile(
    r'^[\dxX]{4,8}$'
)

# Tab instruction patterns
TAB_INSTRUCTIONS = [
    re.compile(r'\bb\s*=\s*bend\b', re.IGNORECASE),
    re.compile(r'\bh\s*=\s*hammer[\s\-]*on\b', re.IGNORECASE),
    re.compile(r'\bp\s*=\s*pull[\s\-]*off\b', re.IGNORECASE),
    re.compile(r'[/\\]\s*=\s*slide[\s\-]*(up|down|note\s*(up|down))\b', re.IGNORECASE),
    re.compile(r'\b~\s*=\s*vibrato\b', re.IGNORECASE),
    re.compile(r'\bx\s*=\s*(mute|dead\s*note)\b', re.IGNORECASE),
    re.compile(r'\bt\s*=\s*tap\b', re.IGNORECASE),
    re.compile(r'\br\s*=\s*release\b', re.IGNORECASE),
    re.compile(r'\bpm\s*=\s*palm\s*mute\b', re.IGNORECASE),
    re.compile(r'^\s*\*+\s*(b|h|p|/|\\|~|x|t|r|pm)\s*=', re.IGNORECASE),
]

# Strumming pattern descriptions
STRUMMING_RE = [
    re.compile(r'strumming\s*pattern', re.IGNORECASE),
    re.compile(r'suggested\s+strumming', re.IGNORECASE),
    re.compile(r'strum\s*:\s*[DU\s]+', re.IGNORECASE),
    re.compile(r'^[DUdu\s\-x]+$'),  # Pure D U D U pattern
    re.compile(r'^\s*(down|up)[\s,]+(down|up)', re.IGNORECASE),
    re.compile(r'^\s*\(?\s*strumming\s*\)?\s*$', re.IGNORECASE),  # "(Strumming)" alone
    re.compile(r'strumming\s*pattern\s*varies', re.IGNORECASE),
]

# Timing/rhythm markers that aren't lyrics
TIMING_RE = [
    re.compile(r'^\s*\d+/\d+\s*$'),  # "4/4", "3/4" alone
    re.compile(r'^\s*\\\s*\d+\s*/\s*$'),  # "\ 3 /"
]

# Editorial/instruction text
EDITORIAL_RE = [
    re.compile(r'play\s+(this|these|the\s+following)\s+(part|riff|lick)', re.IGNORECASE),
    re.compile(r'play\s+in\s+between\s+verse\s+vocal\s+lines', re.IGNORECASE),
    re.compile(r'^\s*\(?(tab|tablature|guitar\s*tab)\)?\s*$', re.IGNORECASE),
    re.compile(r'^\s*\(?(standard\s+tuning|drop\s+[a-z](\s+tuning)?)\)?\s*$', re.IGNORECASE),
    re.compile(r'^\s*tuning\s*:', re.IGNORECASE),
    re.compile(r'^\s*capo\s+\d+', re.IGNORECASE),
    re.compile(r'^\s*(let\s+ring|palm\s+mute|mute\s+strings)', re.IGNORECASE),
    re.compile(r'^\s*\(?\s*repeat\s*(x\s*\d+|\d+\s*times?)?\s*\)?\s*$', re.IGNORECASE),
    re.compile(r'^\s*x\s*\d+\s*$', re.IGNORECASE),  # "x2", "x4"
    re.compile(r'^\s*\(?\s*x\s*\d+\s*\)?\s*$', re.IGNORECASE),  # "(x2)"
    re.compile(r'^\s*\[?fill\s*\d*\]?\s*\(?\s*strumming\s*\)?\s*$', re.IGNORECASE),  # "[Fill 3] (Strumming)"
    re.compile(r'^\s*\w+\s*:\s*\(?\s*strumming\s*(in|out)?\s*\)?\s*$', re.IGNORECASE),  # "Chorus: (Strumming)"
    re.compile(r'one\s+guitar\.\s*the\s+strumming\s+pattern', re.IGNORECASE),
    re.compile(r'^\s*strumming\s*(for|is|:)', re.IGNORECASE),  # "Strumming for the first part..."
    re.compile(r'^\s*strumming\s*:\s*$', re.IGNORECASE),  # "Strumming:" alone
    re.compile(r'^\s*\[?\s*(hard\s+)?strumming\s*(interlude|once|rapidly)?\s*[\],]?\s*$', re.IGNORECASE),
    re.compile(r'^\s*\*+\s*(fast\s+)?strumming\s+to\s+end', re.IGNORECASE),
    re.compile(r'chord\s+and\s+sing.*without?\s+strumming', re.IGNORECASE),
    re.compile(r'finger[\s-]*pick.*or\s+listen\s+to\s+the\s+song', re.IGNORECASE),
    re.compile(r'strumming\s+is\s+a\s+little\s+difficult', re.IGNORECASE),
    re.compile(r'strumming,?\s+all\s+open\s+chords', re.IGNORECASE),
    re.compile(r'strumming\s+pattern\s+(for|with|is)\b', re.IGNORECASE),
    re.compile(r'^\s*\[?\s*strumming\s+once', re.IGNORECASE),
    re.compile(r'^\s*--+\s*strumming\s*--+\s*$', re.IGNORECASE),  # "--Strumming--"
    re.compile(r'listen\s+for\s+strumming', re.IGNORECASE),
    re.compile(r'single\s+strum.*intermit', re.IGNORECASE),
    re.compile(r'the\s+other\s+tab\s+was\s+okay', re.IGNORECASE),
    re.compile(r'chord\s+and\s+sing\s+the\s+lyrics', re.IGNORECASE),
    re.compile(r'strumming\s+pattern\s*[\(:]', re.IGNORECASE),  # "Strumming pattern:" or "Strumming Pattern (..."
    re.compile(r'strumming\s*,?\s*(down|up)\s+(down|up)', re.IGNORECASE),  # "Strumming, Down Down Up"
    re.compile(r'chord\s+strumming\s*$', re.IGNORECASE),
    re.compile(r'^\s*\(?\s*(quick\s+)?strumming\s*(now)?\s*\)?\s*$', re.IGNORECASE),  # "(Strumming now)"
    re.compile(r'just\s+one\s+strumming', re.IGNORECASE),
    re.compile(r'carry\s+on\s+strumming\s+the\s+chord', re.IGNORECASE),
    re.compile(r'picking\s*/\s*strumming\s+style', re.IGNORECASE),
    re.compile(r'transition.*strumming.*pattern', re.IGNORECASE),
    re.compile(r'note.*song.*arranged.*bars', re.IGNORECASE),  # timing explanations
    re.compile(r'need\s+to\s+listen\s+to\b', re.IGNORECASE),
    re.compile(r'riff\s+or\s+chord\s+strumming', re.IGNORECASE),
    re.compile(r'swing\s+feel\s+part', re.IGNORECASE),
    re.compile(r'strumming\s+pattern\s+\^', re.IGNORECASE),  # "Strumming Pattern  ^"
    re.compile(r'^\s*\([A-G]#?\)\s*\(.*strumming\s*\)\s*$', re.IGNORECASE),  # "(A) (quick Strumming)"
    re.compile(r'strings,?\s+ie\s+\d', re.IGNORECASE),  # "strings, ie 3200xx"
    re.compile(r'^\s*\(?\s*(no|now|faster|start|end|continue|back\s+to|change\s+up|syncopated|continuous|staccato|reggae|down)\s*strumming', re.IGNORECASE),
    re.compile(r'^\s*\(?\s*strumming\s*(again|speed|same|16th|direction)\b', re.IGNORECASE),
    re.compile(r'listen\s+to\s+(the\s+)?(song|record|original)', re.IGNORECASE),
    re.compile(r'whatever\s+your\s+strumming', re.IGNORECASE),
    re.compile(r'(here\'?s?\s+the|basic|main|chorus|intro|verse)\s+(chords?\s+for\s+)?strumming', re.IGNORECASE),
    re.compile(r'for\s+(the\s+)?(correct|strumming)\s+strumming', re.IGNORECASE),
    re.compile(r'^\s*>?\s*-?\s*continue\s+strumming', re.IGNORECASE),
    re.compile(r'you\s+(just\s+)?have\s+to\s+get\s+the\s+strumming', re.IGNORECASE),
    re.compile(r'figure\s+out\s+(the\s+)?strumming', re.IGNORECASE),
    re.compile(r'same\s+(basic\s+)?strumming\s+(technique|pattern)', re.IGNORECASE),
    re.compile(r'guitar\s+tab\s+for\s+the\s+intro', re.IGNORECASE),
    re.compile(r'these\s+are\s+the\s+(voicings|chords)\s+used', re.IGNORECASE),
    re.compile(r'^\s*\*+\s*(for|when|note|listen|please|you)', re.IGNORECASE),
    re.compile(r'keep\s+strumming\b', re.IGNORECASE),
    re.compile(r'instead\s+of\s+(just\s+)?strumming', re.IGNORECASE),
    re.compile(r'(pick|fingerpick|finger[\s-]*pick)\s+(the\s+)?(root|bass)\s+note', re.IGNORECASE),
    re.compile(r'^\s*\(?\s*strumming\s*(same|chords?\s+as)', re.IGNORECASE),
    re.compile(r'played?\s+(by|with|without|in)\s+.*strumming', re.IGNORECASE),
    re.compile(r'intersperse\s+these\s+notes', re.IGNORECASE),
    re.compile(r'suggested\s+strumming', re.IGNORECASE),
    re.compile(r'you\s+can\s+(also|either|jangle|make)', re.IGNORECASE),
    re.compile(r'^\s*\(?\s*solo\s+with\s+\w+\s+strumming\s*\)?\s*$', re.IGNORECASE),
    re.compile(r'^\s*\[?\s*verse\s+\d+\s*\]?\s*\(.*strumming\b', re.IGNORECASE),
    re.compile(r'^\s*\[?\s*chorus\s*(again)?\s*\]?\s*\(.*strumming\b', re.IGNORECASE),
    re.compile(r'^\s*\[?\s*chorus\s+again\s+but\s+just\s+play', re.IGNORECASE),
    re.compile(r'muted\s+strum', re.IGNORECASE),
    re.compile(r'^\s*this\s+(tab|song|are)\s+is\s+(purely|pretty|fairly|stripped)', re.IGNORECASE),
    re.compile(r'^\s*(im|i\'?m)\s+no\s+good\s+at\s+writing', re.IGNORECASE),
    re.compile(r'^\s*i\s+(put|can\'?t)\s+(some|figure)', re.IGNORECASE),
    re.compile(r'^\s*well\s+anyways\s+the\s+strumming', re.IGNORECASE),
    re.compile(r'^\s*really\s+fast\s+song', re.IGNORECASE),
    re.compile(r'^\s*easy\s+strumming\s+pattern', re.IGNORECASE),
    re.compile(r'^\s*very\s+very\s+simple\s+song', re.IGNORECASE),
    re.compile(r'the\s+trick\s+to\s+this\s+song', re.IGNORECASE),
    re.compile(r'hammer\s+on\s+starts\s+on\s+beat', re.IGNORECASE),
    re.compile(r'^\s*↓↑\s*=\s*strumming', re.IGNORECASE),
    re.compile(r'^\s*\/\/\/', re.IGNORECASE),  # "///instead of..."
]

# Lines that are ONLY dashes, pipes, numbers (tab-like without the string prefix)
TAB_NUMBERS_RE = re.compile(
    r'^[\d\-|/\\hpbx~()\s]+$'
)

# Chord diagram ASCII art (e.g., lines of just x's, o's, dashes for fret diagrams)
CHORD_DIAGRAM_RE = re.compile(
    r'^[xoXO\d\s\-|]+$'
)


def is_tab_line(text: str) -> bool:
    """Check if a lyrics string contains tab notation."""
    if not text:
        return False
    text = text.strip()
    if not text:
        return False

    # Direct tab notation match
    if TAB_PREFIX_RE.match(text):
        return True

    # Tab notation anywhere in the text (e.g., "Amaj7 x 0 6 6 5 x e|---|")
    if re.search(r'[eEBGDA]\|[\d\-/\\hpbx~()\s|]{3,}', text):
        return True

    # Check for tab lines that might have content after them
    # e.g., "e|---2---| some text"
    for line in text.split('\n'):
        line = line.strip()
        if TAB_PREFIX_RE.match(line):
            return True

    return False


def is_tab_instruction(text: str) -> bool:
    """Check if text is a tab instruction (legend/key)."""
    if not text:
        return False
    text = text.strip()
    for pattern in TAB_INSTRUCTIONS:
        if pattern.search(text):
            return True
    return False


def is_strumming_pattern(text: str) -> bool:
    """Check if text is a strumming pattern description."""
    if not text:
        return False
    text = text.strip()
    for pattern in STRUMMING_RE:
        if pattern.search(text):
            # But if the line also has substantial words, it might be lyrics
            # Only flag if it's mostly pattern notation
            words = re.findall(r'[a-zA-Z]{3,}', text)
            pattern_words = {'down', 'up', 'strum', 'strumming', 'pattern', 'note', 'mute'}
            real_words = [w for w in words if w.lower() not in pattern_words]
            if len(real_words) < 3:
                return True
    return False


def is_timing_marker(text: str) -> bool:
    """Check if text is just a timing/rhythm marker."""
    if not text:
        return False
    text = text.strip()
    for pattern in TIMING_RE:
        if pattern.match(text):
            return True
    return False


def is_editorial_note(text: str) -> bool:
    """Check if text is an editorial/instructional note, not lyrics."""
    if not text:
        return False
    text = text.strip()
    for pattern in EDITORIAL_RE:
        if pattern.search(text):
            return True

    # Broad heuristic: if line mentions "strumming" plus instructional keywords,
    # it's almost certainly an editorial note, not lyrics.
    # Real lyrics with "strumming" are like "strumming my pain", "strumming on my heart",
    # "strumming the lute", "strumming on an old guitar" -- poetic phrases.
    if re.search(r'strumming', text, re.IGNORECASE):
        # Words that strongly indicate editorial/instructional content
        instructional_keywords = [
            r'\bpattern\b', r'\bchord\b', r'\bverse\b', r'\bchorus\b', r'\bintro\b',
            r'\bsong\b', r'\bplay\b', r'\blisten\b', r'\bfigure\s+out\b', r'\bbeat\b',
            r'\bbar\b', r'\bnote\b', r'\bpick\b', r'\bmute\b', r'\bpalm\b',
            r'\btempo\b', r'\btab\b', r'\btuning\b', r'\bcapo\b',
            r'\beasy\b', r'\bsimple\b', r'\bbasic\b', r'\bversion\b',
            r'\bdownstroke\b', r'\bupstroke\b', r'\breggae\b', r'\bska\b',
            r'\bthroughout\b', r'\brepeat\b', r'\briff\b', r'\bsolo\b',
            r'\bhammer\b', r'\bslide\b', r'\bbend\b', r'\bacoustic\b', r'\belectric\b',
            r'\bfingerpick\b', r'\bfinger[\s-]?pick\b', r'\bsound\b', r'\bright\b',
            r'\btiming\b', r'\bguide\b', r'\bfret\b', r'\bstring\b',
        ]
        text_lower = text.lower()
        keyword_hits = sum(1 for kw in instructional_keywords if re.search(kw, text_lower))
        if keyword_hits >= 2:
            return True
        # Even with just 1 keyword: if it's long (>60 chars), it's editorial
        if keyword_hits >= 1 and len(text) > 60:
            return True
        # Lines that are just strumming annotations in brackets/parens
        if re.match(r'^\s*[\[\(].*strumming.*[\]\)]\s*$', text, re.IGNORECASE):
            return True
        # Short editorial phrases with strumming
        if re.match(r'^\s*\(?\s*(and\s+)?(continue|end|start|begin|keep|big|down|more|faster|continuous|light|energetic|no|super|staccato)\s*[\-]?strumming', text, re.IGNORECASE):
            return True
        if re.match(r'^\s*chords?\s+(for|of)\s+strumming', text, re.IGNORECASE):
            return True
        if re.match(r'^\s*strumming[,.]?\s*(\(|$)', text, re.IGNORECASE):
            return True
        if re.search(r'end\s+(the\s+song\s+)?by\s+strumming', text, re.IGNORECASE):
            return True
        if re.match(r'^\s*(besides\s+that,?\s+)?it\'?s?\s+all\s+just\s+.*strumming', text, re.IGNORECASE):
            return True
        if re.search(r'just\s+(go\s+)?crazy\s+with\s+(the\s+)?strumming', text, re.IGNORECASE):
            return True
        if re.match(r'^\s*\(?\s*strumming\s*\)?\s*\(?\s*strumming\s*\)?\s*$', text, re.IGNORECASE):
            return True
        if re.search(r'instrumental\s+break.*strumming', text, re.IGNORECASE):
            return True
        if re.search(r'back\s+to\s+\w.*strumming', text, re.IGNORECASE):
            return True
        if re.search(r'youtube\s+video.*strumming', text, re.IGNORECASE):
            return True
        if re.search(r'method\s*:', text, re.IGNORECASE):
            return True
        if re.search(r'use\s+this\s+strumming', text, re.IGNORECASE):
            return True
        if re.search(r'vary\s+your\s+strumming', text, re.IGNORECASE):
            return True

    return False


def is_only_whitespace(text: str) -> bool:
    """Check if a lyrics field is only whitespace."""
    if text is None:
        return False
    return len(text.strip()) == 0 and len(text) > 0


def is_junk_lyrics(text: str) -> bool:
    """Master check: is this lyrics field junk that should be removed?"""
    if text is None:
        return False  # null lyrics are fine (instrumental lines)

    text_stripped = text.strip()
    if not text_stripped:
        return True  # empty/whitespace-only lyrics = junk

    # Tab notation
    if is_tab_line(text):
        return True

    # Tab instructions
    if is_tab_instruction(text):
        return True

    # Strumming patterns
    if is_strumming_pattern(text):
        return True

    # Timing markers
    if is_timing_marker(text):
        return True

    # Editorial notes
    if is_editorial_note(text):
        return True

    # Multi-line: if ALL lines in the text are tab lines
    lines = text_stripped.split('\n')
    if len(lines) > 1:
        if all(is_tab_line(l) or not l.strip() for l in lines):
            return True

    return False


def has_real_content(line: dict) -> bool:
    """Check if a line has real content (lyrics or chords)."""
    chords = line.get("chords")
    lyrics = line.get("lyrics")

    # If line has chords (even without lyrics), it's valid
    if chords and chords.strip():
        # But check if chords field itself is tab notation junk
        if TAB_PREFIX_RE.match(chords.strip()):
            return False
        return True

    # If line has real lyrics (not junk), it's valid
    if lyrics and lyrics.strip() and not is_junk_lyrics(lyrics):
        return True

    return False


def clean_line(line: dict) -> dict | None:
    """Clean a single line. Returns cleaned line or None if line should be removed."""
    chords = line.get("chords")
    lyrics = line.get("lyrics")

    # If chords field has tab notation, remove entire line
    if chords and TAB_PREFIX_RE.match(chords.strip()):
        return None

    # If lyrics is junk, decide what to do
    if is_junk_lyrics(lyrics):
        # If there are valid chords, keep line but null out the lyrics
        if chords and chords.strip():
            # But first, clean the chords of any fret diagram notation
            # e.g., "E5  022xxx  B5  x244xx" -> "E5  B5"
            cleaned_chords = clean_chords_field(chords)
            if cleaned_chords:
                return {"chords": cleaned_chords, "lyrics": None}
            else:
                return None
        else:
            # No chords and junk lyrics -> remove line
            return None

    # Line is fine as-is. But still clean fret notation from chords field.
    if chords:
        cleaned_chords = clean_chords_field(chords)
        if cleaned_chords != chords:
            return {"chords": cleaned_chords, "lyrics": lyrics}

    return line


def clean_chords_field(chords: str) -> str | None:
    """Remove fret diagram notation and tab notation from chords field.
    E.g., 'E5  022xxx  B5  x244xx' -> 'E5  B5'
    E.g., 'C*  G|---0h20---|' -> 'C*'
    """
    if not chords:
        return chords

    # First, strip any tab notation from the chords field
    # Remove anything that looks like string|tab notation
    chords = re.sub(r'[eEBGDA]\|[\d\-/\\hpbx~()\s|]+', '', chords)

    # Remove strumming instructions from chords field
    chords = re.sub(r'\(?\s*strumming\s*(in|out)?\s*\)?\s*', '', chords, flags=re.IGNORECASE)

    # Split by whitespace and filter out fret diagram tokens
    tokens = chords.split()
    cleaned = []
    for token in tokens:
        # Skip pure fret diagram notation (e.g., 022xxx, x244xx)
        if re.match(r'^[0-9xX]{4,8}$', token):
            continue
        # Skip tokens that are just numbers
        if re.match(r'^\d+$', token):
            continue
        # Skip tokens that are just dashes/pipes (leftover tab fragments)
        if re.match(r'^[\-|]+$', token):
            continue
        cleaned.append(token)

    if not cleaned:
        return None

    result = "  ".join(cleaned)
    # Try to preserve spacing if possible, otherwise use double-space join
    return result if result.strip() else None


def clean_chart(data: dict) -> dict:
    """Clean an entire chart. Returns cleaned chart."""
    sections = data.get("sections", [])
    cleaned_sections = []

    for section in sections:
        lines = section.get("lines", [])
        cleaned_lines = []

        for line in lines:
            cleaned = clean_line(line)
            if cleaned is not None:
                cleaned_lines.append(cleaned)

        if cleaned_lines:
            cleaned_sections.append({
                "name": section.get("name"),
                "lines": cleaned_lines,
            })

    data["sections"] = cleaned_sections
    return data


def chart_is_empty(data: dict) -> bool:
    """Check if chart has no meaningful content after cleaning."""
    sections = data.get("sections", [])
    if not sections:
        return True
    for section in sections:
        if section.get("lines"):
            return False
    return True


def process_file(filepath: Path) -> str:
    """Process a single JSON file. Returns 'cleaned', 'deleted', or 'untouched'."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"  ERROR reading {filepath}: {e}", file=sys.stderr)
        return 'error'

    original_json = json.dumps(data, indent=2, ensure_ascii=False)

    cleaned_data = clean_chart(data)

    if chart_is_empty(cleaned_data):
        # All content was junk — delete the file
        os.remove(filepath)
        return 'deleted'

    cleaned_json = json.dumps(cleaned_data, indent=2, ensure_ascii=False)

    if cleaned_json != original_json:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned_json)
            f.write('\n')
        return 'cleaned'

    return 'untouched'


def main():
    if not LIBRARY_DIR.exists():
        print(f"ERROR: Library directory not found: {LIBRARY_DIR}")
        sys.exit(1)

    json_files = sorted(LIBRARY_DIR.rglob("*.json"))
    total = len(json_files)
    print(f"Found {total} JSON files to process")

    stats = {'cleaned': 0, 'deleted': 0, 'untouched': 0, 'error': 0}
    cleaned_files = []
    deleted_files = []

    for i, filepath in enumerate(json_files):
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i+1}/{total}")

        result = process_file(filepath)
        stats[result] += 1

        if result == 'cleaned':
            cleaned_files.append(str(filepath.relative_to(LIBRARY_DIR)))
        elif result == 'deleted':
            deleted_files.append(str(filepath.relative_to(LIBRARY_DIR)))

    print(f"\n=== CLEANUP COMPLETE ===")
    print(f"Total files:    {total}")
    print(f"Cleaned:        {stats['cleaned']}")
    print(f"Deleted:        {stats['deleted']}")
    print(f"Untouched:      {stats['untouched']}")
    print(f"Errors:         {stats['error']}")

    # Write report data for the summary doc
    report = {
        'total': total,
        'cleaned': stats['cleaned'],
        'deleted': stats['deleted'],
        'untouched': stats['untouched'],
        'errors': stats['error'],
        'cleaned_files_sample': cleaned_files[:50],
        'deleted_files_sample': deleted_files[:50],
        'deleted_files_all': deleted_files,
    }
    report_path = LIBRARY_DIR.parent / "chord_library_cleanup_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
