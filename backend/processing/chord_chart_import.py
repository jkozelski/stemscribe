"""Import chord charts from user-uploaded files (PDF, TXT, ChordPro).

Output: dict matching the shape of outputs/<job_id>/chord_chart.json so the
practice-page renderer treats it identically to auto-detected charts.

v1 scope: text-based PDFs (UG export, print-to-PDF), .txt, .chordpro, .cho.
Scanned/image PDFs (OCR) and notation PDFs (OMR) are intentionally NOT handled.
"""

import re


# Mirror the JS chord regex from practice.html (the _parseChordsFromString helper).
# Same chord vocabulary as the renderer expects.
CHORD_RE = re.compile(
    r"([A-G][#b]?"
    # Quality: order matters — longer alternatives first so they match before "m" alone.
    # mMaj / mmaj = minor-major (e.g. CmMaj7); add must come before m for greedy match.
    r"(?:mMaj|mmaj|maj|m|dim|aug|sus|add)?"
    r"[0-9]*"
    r"(?:[#b][0-9]+)?"
    r"(?:sus[24])?"
    r"(?:add[0-9]+)?"
    r"(?:/[A-G][#b]?)?"
    r")"
)

# Junk-line patterns commonly seen in OCR'd UG PDF preambles. These get dropped
# before parsing so the rendered chart doesn't show "Tuning: EADGBE / CHORDS /
# 342 123 231" type noise as the first section's content.
JUNK_LINE_RE = re.compile(
    r"^\s*("
    r"tuning\s*[:.]"                # Tuning: EADGBE
    r"|chords?\s*$"                  # Just the word "CHORDS" / "CHORD"
    r"|strumming\s+pattern"         # STRUMMING PATTERN
    r"|whole\s+song\s+\d+\s*bpm"    # WHOLE SONG 74 bpm
    r"|page\s+\d+\s*/\s*\d+"        # Page 1/3
    r"|\d+\s*/\s*\d+\s*$"           # 1/3 alone
    r"|[\d\s]{6,}$"                  # Long digit-and-space runs (chord diagram fingerings: 342 342 123 231)
    r"|[1-9]\s*&\s*[1-9]\s*&"       # 1 & 2 & strumming counts
    r"|\(c\)\s*\d{4}"                # (c) 2024 copyright lines
    r"|all\s+rights\s+reserved"     # All Rights Reserved
    r")",
    re.IGNORECASE,
)

CHORD_TOKEN_RE = re.compile(r"^" + CHORD_RE.pattern + r"$")
SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")
INLINE_CHORD_RE = re.compile(r"\[([^\]]+)\]")
META_RE = re.compile(r"^\s*\{([^:}]+):\s*([^}]*)\}\s*$")  # {title: ...}, {capo: 3}
CAPO_LINE_RE = re.compile(r"^\s*capo\s*[:=]?\s*(\d+)", re.IGNORECASE)
KEY_LINE_RE = re.compile(r"^\s*key\s*[:=]\s*([A-G][#b]?m?)\s*$", re.IGNORECASE)
TITLE_LINE_RE = re.compile(r"^\s*title\s*[:=]\s*(.+?)\s*$", re.IGNORECASE)
ARTIST_LINE_RE = re.compile(r"^\s*(?:artist|by|subtitle)\s*[:=]\s*(.+?)\s*$", re.IGNORECASE)
TEMPO_LINE_RE = re.compile(r"^\s*(?:bpm|tempo)\s*[:=]\s*(\d+)", re.IGNORECASE)


def extract_text_from_pdf(path: str) -> str:
    """Extract text from a PDF. Tries pdfplumber first (fast, free for text PDFs);
    falls back to Tesseract OCR for scanned/image PDFs (e.g. Ultimate Guitar's
    "Save as PDF" exports, which are scanned images with zero extractable text).

    Returns empty string if neither pdfplumber nor OCR produced anything useful.
    """
    import pdfplumber

    out = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            out.append(t)
    combined = "\n".join(out)

    # If pdfplumber returned no usable text, fall back to OCR.
    # <40 chars across the whole doc = effectively empty (page numbers, etc.).
    if len(combined.strip()) < 40:
        try:
            from pdf2image import convert_from_path
            import pytesseract

            # 200 dpi is a good speed/accuracy tradeoff for tab/chord text.
            images = convert_from_path(path, dpi=200)
            ocr_parts = []
            for img in images:
                page_text = pytesseract.image_to_string(img, lang="eng")
                ocr_parts.append(page_text)
            combined = "\n".join(ocr_parts)
        except Exception as e:
            # Don't crash the request — return whatever pdfplumber got (likely
            # empty). The endpoint then surfaces the "couldn't read" message.
            import logging
            logging.getLogger(__name__).warning(
                f"OCR fallback failed for {path}: {e}"
            )

    return combined


def is_chord_line(line: str) -> bool:
    """Heuristic: a chord line is mostly chord tokens with whitespace.

    Rejects very long lines (likely lyrics), empty lines, and lines where
    less than 60% of whitespace-separated tokens look like chords.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    tokens = stripped.split()
    if not tokens or len(tokens) > 16:
        return False
    matches = sum(1 for t in tokens if CHORD_TOKEN_RE.match(t))
    return matches / len(tokens) >= 0.6


def _strip_chordpro_inline(line: str):
    """Convert a ChordPro inline-chord line like 'When the [Am]sun comes [F]up'
    into (chord_line, lyric_line) with chords positioned above their lyric chars.
    Returns (None, None) if the line has no inline chords."""
    if "[" not in line or "]" not in line:
        return None, None
    chord_line_chars = []
    lyric_line_chars = []
    i = 0
    while i < len(line):
        if line[i] == "[":
            j = line.find("]", i)
            if j == -1:
                lyric_line_chars.append(line[i])
                i += 1
                continue
            chord = line[i + 1 : j]
            # Pad chord_line out to current lyric position
            while len(chord_line_chars) < len(lyric_line_chars):
                chord_line_chars.append(" ")
            for ch in chord:
                chord_line_chars.append(ch)
            # Pad lyric_line out to absorb the chord width minus one space
            # so the next lyric character lines up under the first chord char.
            i = j + 1
        else:
            # Keep lyric and chord_line in sync by adding a space to chord_line
            # whenever lyric advances past the chord-line cursor.
            if len(chord_line_chars) <= len(lyric_line_chars):
                chord_line_chars.append(" ")
            lyric_line_chars.append(line[i])
            i += 1
    chord_line = "".join(chord_line_chars).rstrip()
    lyric_line = "".join(lyric_line_chars).rstrip()
    if not chord_line:
        return None, None
    return chord_line, lyric_line


def _is_metadata_line(line: str) -> bool:
    """True if a line is a recognised metadata directive (title/artist/key/capo/tempo).
    Used by the preamble cleaner so we keep these even when they appear above the
    first [Section] marker."""
    return bool(
        META_RE.match(line)
        or CAPO_LINE_RE.match(line)
        or KEY_LINE_RE.match(line)
        or TITLE_LINE_RE.match(line)
        or ARTIST_LINE_RE.match(line)
        or TEMPO_LINE_RE.match(line)
    )


_TAB_HEAD_RE = re.compile(
    # First char or two: a string label (E, B, G, D, A) followed by '|' or '['.
    # OCR commonly mangles E|--- into "Eero", "E[O---", "Eo|---" and B|--- into
    # "a |---", "8|---" or pure gibberish like "RSSeesReeeee". To catch the
    # mangled variants we also accept lines where >=40% of the chars are dashes
    # or pipes (tab rows are mostly those two), checked in the heuristic below.
    r"^\s*[EBGDAebgda][\s|\[\(o0O@]"
)


def _is_tab_line(line: str) -> bool:
    """Heuristic: line is part of a guitar tab grid (6 strings of dashes/numbers).

    Triggered by, in priority order:
      1) Starts with a string label (E|, B|, G|, D|, A|, E|) — even OCR-mangled
      2) Contains a '|' character at all — lyrics + chord-name lines never use
         a pipe, but every row of a tab grid uses several. This single rule
         catches the most-mangled OCR output (e.g. "Eero [pea 6-|" or
         "a | RSSeesReeeee 6-|") where the dash-density heuristic falls below
         threshold because OCR replaced too many dashes with letters.
      3) Dash-and-pipe dense (>=40% of non-space chars are '-' or '|') as a
         belt-and-suspenders catch for tab rows where '|' got OCR'd into 'l'
         or '1'.
    Excludes section markers ([Verse] etc.) and very short lines.
    Excludes ChordPro inline chord lines ("[Am]when [C]the sun") which contain
    bracket pairs but no '|' — those are handled by the inline-chord path.
    """
    s = line.strip()
    if not s or len(s) < 6:
        return False
    if SECTION_RE.match(line):
        return False
    # ChordPro inline chord lines have [Chord] tokens but no '|' — let them pass.
    if "|" not in s and INLINE_CHORD_RE.search(s):
        return False
    if _TAB_HEAD_RE.match(line):
        return True
    if "|" in s:
        return True
    non_space = [c for c in s if not c.isspace()]
    if not non_space:
        return False
    dash_pipe = sum(1 for c in non_space if c in "-|")
    if dash_pipe / len(non_space) >= 0.40 and dash_pipe >= 4:
        return True
    return False


# Recognises the start of an alternate-tuning arrangement that often appears
# after the main chord chart in UG-style PDFs ("Open D version", "Drop D
# version", "shapes for DADF#AD", a row of >=20 dashes used as a separator).
# OCR is tolerant: "open d" or "Open\xa0D" both match.
_VERSION_BREAK_RE = re.compile(
    r"^\s*(?:"
    r"-{20,}"                                       # row of dashes used as separator
    r"|\s*open\s*[d]\b.*\bversion"                 # "Open D version"
    r"|\s*drop\s*[d]\b.*\bversion"                 # "Drop D version"
    r"|\s*shapes?\s+for\b"                          # "Shapes for DADF#AD"
    r"|\s*alt(?:ernate)?\s+(?:tuning|version)"     # "Alternate tuning"
    r")",
    re.IGNORECASE,
)


def _is_version_separator(line: str) -> bool:
    return bool(_VERSION_BREAK_RE.match(line.strip()))


def _clean_ocr_preamble(lines: list) -> list:
    """Drop UG-PDF preamble noise + tab grids + alternate-version sections.

    Four passes:
      1) Drop lines matching JUNK_LINE_RE (tuning info, CHORDS header,
         strumming pattern, page numbers, fingering digit rows, copyright).
      2) Drop guitar-tab grid lines (E|---, B|---, etc.) — OCR mangles the
         top two strings into nonsense like "Eero [pea" / "RSSeesReeeee"
         and we don't render ASCII tab anyway. Chord-over-lyric is the
         only thing the practice page draws.
      3) Truncate at the first "Open D version" / "Shapes for..." /
         long-dash separator — stops the alternate-arrangement section
         (different chord vocabulary) from being concatenated onto the
         main chart.
      4) If a [Section] marker exists anywhere, drop lines BEFORE the first
         one unless they're metadata directives (title/artist/key/capo).
         Everything else above the first section is OCR-extracted
         chord-diagram summary rows, watermark gibberish, etc.

    Returns a cleaned line list ready for the main parser.
    """
    # Pass 1: drop known-junk lines outright
    cleaned = [ln for ln in lines if not JUNK_LINE_RE.match(ln)]

    # Pass 2: drop tab-grid lines
    cleaned = [ln for ln in cleaned if not _is_tab_line(ln)]

    # Pass 3: truncate at first alternate-version marker
    for i, ln in enumerate(cleaned):
        if _is_version_separator(ln):
            cleaned = cleaned[:i]
            break

    # Pass 4: if any section marker exists, strip non-metadata noise before it
    first_section_idx = None
    for i, ln in enumerate(cleaned):
        if SECTION_RE.match(ln):
            first_section_idx = i
            break
    if first_section_idx is not None and first_section_idx > 0:
        head = [ln for ln in cleaned[:first_section_idx] if _is_metadata_line(ln)]
        cleaned = head + cleaned[first_section_idx:]

    return cleaned


def parse_chord_chart_text(text: str, source: str = "paste") -> dict:
    """Parse UG-style chord-over-lyric text into the internal chart JSON shape.

    source: 'paste' | 'pdf' | 'txt' | 'chordpro' | 'cho' | 'pro' — metadata only.
    """
    raw_lines = [ln.rstrip() for ln in text.splitlines()]
    lines = _clean_ocr_preamble(raw_lines)
    sections = []
    current_section = {"name": "Intro", "lines": []}
    meta_title = ""
    meta_artist = ""
    meta_key = ""
    meta_capo = None
    meta_bpm = None

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        # ChordPro directives: {title: ...}, {artist: ...}, {key: G}, {capo: 3}
        m = META_RE.match(line)
        if m:
            tag = m.group(1).strip().lower()
            val = m.group(2).strip()
            if tag in ("title", "t"):
                meta_title = val
            elif tag in ("artist", "a", "subtitle", "st"):
                meta_artist = meta_artist or val
            elif tag == "key":
                meta_key = val
            elif tag == "capo":
                try:
                    meta_capo = int(val)
                except ValueError:
                    pass
            i += 1
            continue

        # Bare "Capo: 3rd fret" or "Key: Am" lines
        capo_m = CAPO_LINE_RE.match(line)
        if capo_m:
            try:
                meta_capo = int(capo_m.group(1))
            except ValueError:
                pass
            i += 1
            continue
        key_m = KEY_LINE_RE.match(line)
        if key_m:
            meta_key = meta_key or key_m.group(1)
            i += 1
            continue
        title_m = TITLE_LINE_RE.match(line)
        if title_m and not is_chord_line(line):
            meta_title = meta_title or title_m.group(1)
            i += 1
            continue
        artist_m = ARTIST_LINE_RE.match(line)
        if artist_m and not is_chord_line(line):
            meta_artist = meta_artist or artist_m.group(1)
            i += 1
            continue
        tempo_m = TEMPO_LINE_RE.match(line)
        if tempo_m:
            try:
                meta_bpm = int(tempo_m.group(1))
            except ValueError:
                pass
            i += 1
            continue

        # Section header: [Verse], [Chorus 2], [Bridge]
        sec = SECTION_RE.match(line)
        if sec:
            # Close current section if it has content
            if current_section["lines"]:
                sections.append(current_section)
            current_section = {"name": sec.group(1).strip(), "lines": []}
            i += 1
            continue

        # ChordPro inline chord line: "[Am]when the [C]sun"
        chord_line_from_inline, lyric_line_from_inline = _strip_chordpro_inline(line)
        if chord_line_from_inline is not None:
            current_section["lines"].append(
                {
                    "chords": chord_line_from_inline,
                    "lyrics": lyric_line_from_inline,
                    "segments": [],
                }
            )
            i += 1
            continue

        # Chord-over-lyric pattern: a chord line followed (optionally after a
        # blank line) by a lyric line.
        if is_chord_line(line):
            chord_line = line
            lyric_line = ""
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if (
                j < len(lines)
                and not is_chord_line(lines[j])
                and not SECTION_RE.match(lines[j])
                and not META_RE.match(lines[j])
            ):
                lyric_line = lines[j]
                i = j + 1
            else:
                i += 1
            current_section["lines"].append(
                {
                    "chords": chord_line,
                    "lyrics": lyric_line,
                    "segments": [],
                }
            )
            continue

        # Standalone lyric line with no chord line above it.
        current_section["lines"].append(
            {
                "chords": "",
                "lyrics": line,
                "segments": [],
            }
        )
        i += 1

    if current_section["lines"]:
        sections.append(current_section)

    # Collect unique chord names for the renderer's chords_used field.
    chords_used = []
    seen = set()
    for sec_obj in sections:
        for ln in sec_obj["lines"]:
            for m in CHORD_RE.finditer(ln.get("chords", "")):
                name = m.group(1)
                if name and name not in seen:
                    seen.add(name)
                    chords_used.append(name)

    return {
        "title": meta_title,
        "artist": meta_artist,
        "key": meta_key,
        "capo": meta_capo,
        "bpm": meta_bpm,
        "source": f"imported:{source}",
        "chords_used": chords_used,
        "sections": sections,
    }
