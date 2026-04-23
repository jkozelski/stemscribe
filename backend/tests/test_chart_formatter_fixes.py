"""Regression tests for 2026-04-22 chart_formatter fixes.

Covers the render-layer fixes applied to chart_formatter.py for the
4-bar slash-notation chord-chunk layout:
  1. Section naming — a section with lyric text never carries a solo-style
     label ("Drum Break", "Bass Solo", etc.). The trailing lyric-bearing
     section in the song becomes "Outro"; interior ones borrow from the
     nearest adjacent Verse/Chorus/Bridge label.
  2. Slash-notation chord line — each bar renders as "<chord> ////"
     (chord name + four slashes for the 4 beats of a 4/4 measure), with
     slots joined by a two-space separator.
  3. 4-bar chunking rebuild — each section's lines are rewritten as one
     or more 4-bar chunks with the slash-notation chord line across the
     top and the Whisper words whose start falls in the chunk window
     flowing underneath as the lyric line.
"""

from backend.chart_formatter import (
    _Section,
    _rename_solo_sections_with_lyrics,
    _merge_adjacent_sections_with_same_name,
    _format_slash_bar,
    _build_slash_chord_line,
    _rebuild_sections_as_4bar_chunks,
)


# -- Fix 1: solo-named sections with lyrics get relabeled -------------------

def _sec(name, lines, start_time=0.0, end_time=0.0, has_lyrics=None):
    s = _Section(name=name, has_lyrics=False)
    s.lines = lines
    s.start_time = start_time
    s.end_time = end_time
    if has_lyrics is not None:
        s.has_lyrics = has_lyrics
    return s


def test_solo_section_with_lyrics_gets_renamed():
    # "Bass Solo" / "Drum Break" labels landing on sections with Whisper-
    # transcribed lyrics must be relabeled to a vocal section name.
    sections = [
        _sec("Verse 1", [{"chords": "Cm  ////", "lyrics": "line one"}]),
        _sec("Bass Solo", [{"chords": "Gm  ////", "lyrics": "has lyrics"}]),
        _sec("Drum Break", [{"chords": "Am  ////", "lyrics": "more lyrics"}]),
        _sec("Chorus", [{"chords": "F   ////", "lyrics": "chorus text"}]),
    ]
    _rename_solo_sections_with_lyrics(sections)
    assert sections[1].name != "Bass Solo"
    assert sections[2].name != "Drum Break"
    # Interior solo-renamed sections borrow a vocal label from a neighbour
    assert sections[1].name in ("Verse", "Chorus", "Bridge", "Pre-Chorus")
    assert sections[2].name in ("Verse", "Chorus", "Bridge", "Pre-Chorus")


def test_solo_without_lyrics_keeps_name():
    sections = [
        _sec("Verse 1", [{"chords": "Cm  ////", "lyrics": "line one"}]),
        _sec("Bass Solo", [
            {"chords": "Gm  ////", "lyrics": None},
            {"chords": "Am  ////", "lyrics": ""},
        ]),
    ]
    _rename_solo_sections_with_lyrics(sections)
    assert sections[1].name == "Bass Solo"


def test_tail_solo_with_lyrics_becomes_outro():
    # A trailing solo-labeled section that actually carries lyric text is
    # the song's final vocal phrase, which should render as "Outro".
    sections = [
        _sec("Verse 1", [{"chords": "Cm  ////", "lyrics": "a"}]),
        _sec("Chorus", [{"chords": "Gm  ////", "lyrics": "b"}]),
        _sec("Bass Solo", [{"chords": "F   ////", "lyrics": "final phrase"}]),
    ]
    _rename_solo_sections_with_lyrics(sections)
    assert sections[2].name == "Outro"
    assert sections[2].has_lyrics is True


# -- Fix 2: slash-notation chord line layout --------------------------------

def test_format_slash_bar_short_chord():
    # "Cm" pads to 4 chars (two trailing spaces) then appends "////".
    assert _format_slash_bar("Cm") == "Cm  ////"


def test_format_slash_bar_long_chord_overflow():
    # Chord names longer than the slot width still keep one space before the
    # slashes so the bar stays unambiguous.
    out = _format_slash_bar("Cm7b5")
    assert out.endswith("////")
    assert " ////" in out  # at least one space before the slashes
    assert "Cm7b5" in out


def test_build_slash_chord_line_full_4_bars():
    out = _build_slash_chord_line(["Cm", "Gm", "Dm", "Am"])
    assert out == "Cm  ////  Gm  ////  Dm  ////  Am  ////"
    # Four bars, four slash groups.
    assert out.count("////") == 4


def test_build_slash_chord_line_partial_chunk():
    # Partial chunks (e.g. 3-bar Intro at a section boundary) still render
    # cleanly with only the bars present — no rest-padding.
    out = _build_slash_chord_line(["Cm", "Gm", "Dm"])
    assert out == "Cm  ////  Gm  ////  Dm  ////"
    assert out.count("////") == 3


def test_build_slash_chord_line_empty():
    assert _build_slash_chord_line([]) == ""


# -- Fix 3: 4-bar chunking rebuild over the bar_grid ------------------------

def _make_bar_grid(chords_per_bar, bar_dur=2.0, start=0.0):
    """Build a bar_grid array where bar `i` has chord `chords_per_bar[i]`."""
    return [
        {
            "bar": i + 1,
            "chord": ch,
            "start_time": round(start + i * bar_dur, 3),
            "end_time":   round(start + (i + 1) * bar_dur, 3),
        }
        for i, ch in enumerate(chords_per_bar)
    ]


def test_4bar_chunks_produce_slash_notation_chord_line():
    # 16-bar Cm-Gm-Dm-Am vamp over a single Verse section.
    bar_grid = _make_bar_grid(["Cm", "Gm", "Dm", "Am"] * 4)
    sections = [_sec("Verse", [], start_time=0.0, end_time=32.0, has_lyrics=True)]
    # Synthesize words across the section
    words = [
        {"word": "I've", "start": 0.5, "end": 1.0},
        {"word": "been", "start": 1.5, "end": 2.0},
        {"word": "seeing", "start": 3.0, "end": 3.5},
        {"word": "angels", "start": 5.0, "end": 5.5},
        {"word": "turn", "start": 7.0, "end": 7.3},
        {"word": "into", "start": 9.0, "end": 9.3},
        {"word": "devils", "start": 11.0, "end": 11.5},
        {"word": "in", "start": 13.0, "end": 13.2},
        {"word": "my", "start": 15.0, "end": 15.2},
        {"word": "mind", "start": 16.5, "end": 17.0},
        {"word": "I", "start": 18.0, "end": 18.1},
        {"word": "need", "start": 20.0, "end": 20.3},
        {"word": "your", "start": 22.0, "end": 22.3},
        {"word": "love", "start": 24.0, "end": 24.5},
    ]
    _rebuild_sections_as_4bar_chunks(sections, bar_grid, words)

    sec = sections[0]
    # 16 bars of Cm-Gm-Dm-Am should yield exactly 4 chunks of 4 bars each.
    assert len(sec.lines) == 4

    expected_chord_line = "Cm  ////  Gm  ////  Dm  ////  Am  ////"
    for line in sec.lines:
        assert line["chords"] == expected_chord_line

    # Lyrics split across the 4 chunks by time window.
    assert "I've" in sec.lines[0]["lyrics"]
    assert "love" in sec.lines[3]["lyrics"]
    # Words flow with single-space separators (no chord-above-word alignment).
    assert "  " not in sec.lines[0]["lyrics"]


def test_4bar_chunks_honor_section_boundaries():
    # Two sections each 4 bars long — should produce 1 chunk per section,
    # each chunk a full 4-bar line, with no chunk spanning two sections.
    bar_grid = _make_bar_grid(["Cm", "Gm", "Dm", "Am", "F", "G", "Am", "Bb"])
    sections = [
        _sec("Intro", [], start_time=0.0,  end_time=8.0,  has_lyrics=False),
        _sec("Verse", [], start_time=8.0,  end_time=16.0, has_lyrics=True),
    ]
    _rebuild_sections_as_4bar_chunks(sections, bar_grid, [])
    # Intro has exactly the first 4 chords (Cm, Gm, Dm, Am).
    intro_line = sections[0].lines[0]["chords"]
    assert "Cm" in intro_line and "Gm" in intro_line
    assert "F " not in intro_line  # F belongs to the next section
    # Verse has exactly the next 4 chords starting from F.
    verse_line = sections[1].lines[0]["chords"]
    assert "F" in verse_line and "Bb" in verse_line
    assert "Cm" not in verse_line


def test_4bar_chunks_partial_boundary_chunk():
    # A 3-bar section emits one partial chunk (3 slash slots, no padding).
    bar_grid = _make_bar_grid(["Cm", "Gm", "Dm"])
    sections = [_sec("Intro", [], start_time=0.0, end_time=6.0, has_lyrics=False)]
    _rebuild_sections_as_4bar_chunks(sections, bar_grid, [])
    line = sections[0].lines[0]["chords"]
    assert line.count("////") == 3  # exactly 3 bars, no padding
    assert line.startswith("Cm")


def test_4bar_chunks_instrumental_section_has_none_lyrics():
    bar_grid = _make_bar_grid(["Cm", "Gm", "Dm"])
    sections = [_sec("Intro", [], start_time=0.0, end_time=6.0, has_lyrics=False)]
    _rebuild_sections_as_4bar_chunks(sections, bar_grid, [])
    assert sections[0].lines[0]["lyrics"] is None


def test_4bar_chunks_consecutive_instrumental_chunks_not_collapsed():
    # 12 bars of solid instrumental Cm — three 4-bar chunks, each emitted as
    # its own line. Don't collapse consecutive same-pattern instrumental
    # chunks: preserving them preserves the song-structure length.
    bar_grid = _make_bar_grid(["Cm"] * 12)
    sections = [_sec("Intro", [], start_time=0.0, end_time=24.0, has_lyrics=False)]
    _rebuild_sections_as_4bar_chunks(sections, bar_grid, [])
    assert len(sections[0].lines) == 3


# -- Fix 4: merging adjacent same-name sections -----------------------------

def test_merge_adjacent_same_name_sections():
    sections = [
        _sec("Outro", [{"chords": "Dm  ////", "lyrics": "final verse"}]),
        _sec("Outro", [{"chords": "D#  ////", "lyrics": None}]),
    ]
    merged = _merge_adjacent_sections_with_same_name(sections)
    assert len(merged) == 1
    assert merged[0].name == "Outro"
    assert len(merged[0].lines) == 2  # lines concatenated


def test_merge_leaves_different_names_alone():
    sections = [
        _sec("Verse 1", [{"chords": "Cm  ////", "lyrics": "a"}]),
        _sec("Chorus",  [{"chords": "Gm  ////", "lyrics": "b"}]),
        _sec("Verse 2", [{"chords": "Dm  ////", "lyrics": "c"}]),
    ]
    merged = _merge_adjacent_sections_with_same_name(sections)
    assert len(merged) == 3
    assert [s.name for s in merged] == ["Verse 1", "Chorus", "Verse 2"]
