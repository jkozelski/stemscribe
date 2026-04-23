"""Regression tests for 2026-04-22 chart_formatter fixes.

Covers the four render-layer fixes applied to chart_formatter.py:
  1. Section naming post-filter — a section with lyric text never carries
     a solo-style label ("Drum Break", "Bass Solo", etc.).
  2. Short-fragment lyric line consolidation — a section with many tiny
     lyric stubs merges to ~60-char readable lines.
  3. Compact held-chord rendering — a run of 3+ identical consecutive
     chord labels collapses to "{name} (x{count})".
  4. Minimum chord-label gap — two adjacent chord labels never render
     fused (e.g. "CmGm"), they always have at least two spaces between.
"""

from backend.chart_formatter import (
    _Section,
    _rename_solo_sections_with_lyrics,
    _consolidate_short_lyric_lines,
    _compact_held_chord_lines,
    _build_chord_line,
)


# -- Fix 1 ------------------------------------------------------------------

def _sec(name, lines):
    s = _Section(name=name, has_lyrics=False)
    s.lines = lines
    return s


def test_solo_section_with_lyrics_gets_renamed():
    # Simulates the "Bass Solo" / "Drum Break" labels landing on sections
    # that carry Whisper-transcribed lyrics.
    sections = [
        _sec("Verse 1", [{"chords": "Cm", "lyrics": "line one"}]),
        _sec("Bass Solo", [{"chords": "Gm", "lyrics": "has lyrics"}]),
        _sec("Drum Break", [{"chords": "Am", "lyrics": "more lyrics"}]),
    ]
    _rename_solo_sections_with_lyrics(sections)
    assert sections[1].name != "Bass Solo"
    assert sections[2].name != "Drum Break"
    # Should pick up the Verse label from the preceding section
    assert sections[1].name in ("Verse", "Chorus", "Bridge", "Pre-Chorus")


def test_solo_without_lyrics_keeps_name():
    sections = [
        _sec("Verse 1", [{"chords": "Cm", "lyrics": "line one"}]),
        _sec("Bass Solo", [{"chords": "Gm", "lyrics": None}, {"chords": "Am", "lyrics": ""}]),
    ]
    _rename_solo_sections_with_lyrics(sections)
    assert sections[1].name == "Bass Solo"


def test_outro_with_lyrics_relabeled():
    sections = [
        _sec("Verse 1", [{"chords": "Cm", "lyrics": "a"}]),
        _sec("Chorus", [{"chords": "Gm", "lyrics": "b"}]),
        _sec("Outro", [{"chords": "F", "lyrics": "tail lyrics"}]),
    ]
    _rename_solo_sections_with_lyrics(sections)
    # Outro with lyrics should NOT stay "Outro" — it's a vocal section
    assert sections[2].name != "Outro"


# -- Fix 2 ------------------------------------------------------------------

def test_short_lines_consolidate():
    # 10 fragments of ~15 chars each → should merge to fewer longer lines.
    fragments = [
        {"chords": "Cm", "lyrics": "I need your love,"},
        {"chords": "Gm", "lyrics": "I need your love,"},
        {"chords": "Dm", "lyrics": "I need your love,"},
        {"chords": "Am", "lyrics": "I need your love,"},
        {"chords": "Cm", "lyrics": "I need your love,"},
        {"chords": "Gm", "lyrics": "I need your love,"},
        {"chords": "Dm", "lyrics": "I need your love,"},
        {"chords": "Am", "lyrics": "I need your love,"},
        {"chords": "Cm", "lyrics": "I need your love,"},
        {"chords": "Gm", "lyrics": "I need your love,"},
    ]
    sec = _Section(name="Verse", has_lyrics=True)
    sec.lines = fragments
    _consolidate_short_lyric_lines([sec])
    assert len(sec.lines) < 10  # something got merged
    # Every merged line's lyric is under the 60-char cap + some slack
    for ln in sec.lines:
        if ln.get("lyrics"):
            assert len(ln["lyrics"]) <= 70


def test_long_lines_not_consolidated():
    # Already-long lyric lines shouldn't be touched
    sec = _Section(name="Verse", has_lyrics=True)
    sec.lines = [
        {"chords": "Cm", "lyrics": "A" * 70},
        {"chords": "Gm", "lyrics": "B" * 70},
    ]
    before = len(sec.lines)
    _consolidate_short_lyric_lines([sec])
    assert len(sec.lines) == before


def test_consolidation_dedupes_consecutive_chords():
    # Adjacent lines each with "Cm" should merge to a single "Cm" (not "Cm  Cm")
    lines = [
        {"chords": "Cm", "lyrics": "a b"},
        {"chords": "Cm", "lyrics": "c d"},
        {"chords": "Cm", "lyrics": "e f"},
        {"chords": "Cm", "lyrics": "g h"},
        {"chords": "Cm", "lyrics": "i j"},
        {"chords": "Cm", "lyrics": "k l"},
        {"chords": "Cm", "lyrics": "m n"},
        {"chords": "Cm", "lyrics": "o p"},
        {"chords": "Cm", "lyrics": "q r"},
    ]
    sec = _Section(name="Verse", has_lyrics=True)
    sec.lines = lines
    _consolidate_short_lyric_lines([sec])
    for ln in sec.lines:
        chords = ln.get("chords", "")
        tokens = chords.split()
        # No two consecutive tokens should be equal (consecutive dedup)
        for i in range(1, len(tokens)):
            assert tokens[i] != tokens[i - 1], f"consecutive dup in {chords!r}"


# -- Fix 3 ------------------------------------------------------------------

def test_held_chord_collapses_to_xN():
    sec = _Section(name="Outro", has_lyrics=False)
    sec.lines = [{"chords": "Gm  Gm  Gm  Gm  Gm  Gm  Gm  Gm", "lyrics": None}]
    _compact_held_chord_lines([sec])
    assert "(x8)" in sec.lines[0]["chords"] or "(x " in sec.lines[0]["chords"]
    assert sec.lines[0]["chords"] == "Gm (x8)"


def test_short_runs_not_collapsed():
    # 2 in a row stays as is — don't collapse the natural Cm-Gm-Dm-Am 2-bar pattern
    sec = _Section(name="Verse", has_lyrics=True)
    sec.lines = [{"chords": "Cm  Cm  Gm  Gm  Dm  Dm  Am  Am", "lyrics": "hi"}]
    _compact_held_chord_lines([sec])
    assert "(x" not in sec.lines[0]["chords"]


def test_mixed_runs_partial_collapse():
    sec = _Section(name="Bridge", has_lyrics=False)
    sec.lines = [{"chords": "Cm  Gm  Gm  Gm  Dm  Am", "lyrics": None}]
    _compact_held_chord_lines([sec])
    # Gm×3 collapses; Cm, Dm, Am stay verbatim
    out = sec.lines[0]["chords"]
    assert "Gm (x3)" in out
    assert "Cm" in out and "Dm" in out and "Am" in out


# -- Fix 4 ------------------------------------------------------------------

def test_chord_line_enforces_min_gap():
    # Two chord names placed at char positions 0 and 2 should end up with
    # at least a 2-space gap (so "Cm" + "Gm" doesn't render as "CmGm").
    out = _build_chord_line([(0, "Cm"), (2, "Gm")], lyrics_len=10)
    # Cm occupies chars 0-1, so Gm can't start before char 4 (2 + MIN_GAP)
    # Output should contain "Cm" then >= 2 spaces then "Gm"
    assert out.startswith("Cm")
    assert "  Gm" in out
    # Explicitly no fused "CmGm"
    assert "CmGm" not in out


def test_chord_line_respects_larger_gap():
    # When the natural gap is already >= MIN_GAP, use it.
    out = _build_chord_line([(0, "Cm"), (10, "Gm")], lyrics_len=20)
    # Cm then 8 spaces then Gm
    assert out == "Cm        Gm"
