"""Tests for chart_assembler module."""

import json
import tempfile
import os
import pytest

from chart_assembler import assemble_chart, save_chart


# ─── Sample data modeled after "The Time Comes" by Kozelski ───

SAMPLE_CHORDS = [
    # Intro: F#m A Bm E pattern x2
    {"chord": "F#m", "time": 0.0, "duration": 2.0, "confidence": 0.9},
    {"chord": "A",   "time": 2.0, "duration": 2.0, "confidence": 0.85},
    {"chord": "Bm",  "time": 4.0, "duration": 2.0, "confidence": 0.88},
    {"chord": "E",   "time": 6.0, "duration": 2.0, "confidence": 0.87},
    {"chord": "F#m", "time": 8.0, "duration": 2.0, "confidence": 0.9},
    {"chord": "A",   "time": 10.0, "duration": 2.0, "confidence": 0.85},
    {"chord": "Bm",  "time": 12.0, "duration": 2.0, "confidence": 0.88},
    {"chord": "E",   "time": 14.0, "duration": 2.0, "confidence": 0.87},
    # Verse 1
    {"chord": "F#m", "time": 16.0, "duration": 3.0, "confidence": 0.9},
    {"chord": "A",   "time": 19.0, "duration": 3.0, "confidence": 0.85},
    {"chord": "F#m", "time": 22.0, "duration": 3.0, "confidence": 0.9},
    {"chord": "B",   "time": 25.0, "duration": 3.0, "confidence": 0.82},
    {"chord": "F#m", "time": 28.0, "duration": 3.0, "confidence": 0.9},
    {"chord": "A",   "time": 31.0, "duration": 3.0, "confidence": 0.85},
    {"chord": "F#m", "time": 34.0, "duration": 3.0, "confidence": 0.9},
    {"chord": "B",   "time": 37.0, "duration": 3.0, "confidence": 0.82},
    # Chorus
    {"chord": "D",   "time": 40.0, "duration": 2.0, "confidence": 0.88},
    {"chord": "C#m", "time": 42.0, "duration": 2.0, "confidence": 0.85},
    {"chord": "Bm",  "time": 44.0, "duration": 2.0, "confidence": 0.87},
    {"chord": "D",   "time": 46.0, "duration": 2.0, "confidence": 0.88},
    {"chord": "C#m", "time": 48.0, "duration": 2.0, "confidence": 0.85},
    {"chord": "G",   "time": 50.0, "duration": 2.0, "confidence": 0.80},
]

SAMPLE_LYRICS = [
    # Verse 1
    {"text": "You say you wanted", "start_time": 16.0, "end_time": 19.5},
    {"text": "Someone to take you away", "start_time": 22.0, "end_time": 25.5},
    {"text": "From out of the deep end", "start_time": 28.0, "end_time": 31.5},
    {"text": "Where the devil's all play", "start_time": 34.0, "end_time": 37.5},
    # Chorus
    {"text": "The time comes to turn it around", "start_time": 40.0, "end_time": 45.0},
    {"text": "All the lines must be put down", "start_time": 46.0, "end_time": 51.0},
]

SAMPLE_SECTIONS = [
    {"name": "Intro", "start_time": 0.0, "end_time": 15.9},
    {"name": "Verse 1", "start_time": 16.0, "end_time": 39.9},
    {"name": "Chorus", "start_time": 40.0, "end_time": 52.0},
]


class TestChartFormat:
    """Verify output matches the format renderManualChordChart expects."""

    def test_basic_structure(self):
        chart = assemble_chart(SAMPLE_CHORDS, SAMPLE_LYRICS, SAMPLE_SECTIONS,
                               title="The Time Comes", artist="Kozelski")
        assert "title" in chart
        assert "artist" in chart
        assert "source" in chart
        assert "sections" in chart
        assert chart["title"] == "The Time Comes"
        assert chart["artist"] == "Kozelski"
        assert chart["source"] == "auto"

    def test_sections_have_name_and_lines(self):
        chart = assemble_chart(SAMPLE_CHORDS, SAMPLE_LYRICS, SAMPLE_SECTIONS,
                               title="The Time Comes", artist="Kozelski")
        for section in chart["sections"]:
            assert "name" in section
            assert "lines" in section
            assert isinstance(section["lines"], list)
            for line in section["lines"]:
                assert "chords" in line
                assert "lyrics" in line

    def test_section_names_match(self):
        chart = assemble_chart(SAMPLE_CHORDS, SAMPLE_LYRICS, SAMPLE_SECTIONS,
                               title="Test", artist="Test")
        names = [s["name"] for s in chart["sections"]]
        assert "Intro" in names
        assert "Verse 1" in names
        assert "Chorus" in names

    def test_intro_is_instrumental(self):
        chart = assemble_chart(SAMPLE_CHORDS, SAMPLE_LYRICS, SAMPLE_SECTIONS,
                               title="Test", artist="Test")
        intro = [s for s in chart["sections"] if s["name"] == "Intro"][0]
        # Intro has no lyrics (or only repeat notation)
        for line in intro["lines"]:
            assert line["lyrics"] == "" or line["lyrics"].startswith("(x")

    def test_verse_has_chords_above_lyrics(self):
        chart = assemble_chart(SAMPLE_CHORDS, SAMPLE_LYRICS, SAMPLE_SECTIONS,
                               title="Test", artist="Test")
        verse = [s for s in chart["sections"] if s["name"] == "Verse 1"][0]
        assert len(verse["lines"]) >= 4
        for line in verse["lines"]:
            if line["lyrics"]:
                assert len(line["chords"]) > 0, "Lyric line should have chords"

    def test_chorus_has_multiple_chords_per_line(self):
        chart = assemble_chart(SAMPLE_CHORDS, SAMPLE_LYRICS, SAMPLE_SECTIONS,
                               title="Test", artist="Test")
        chorus = [s for s in chart["sections"] if s["name"] == "Chorus"][0]
        # First chorus line should have D, C#m, Bm
        first_line = chorus["lines"][0]
        assert "D" in first_line["chords"]
        assert "C#m" in first_line["chords"]

    def test_chord_alignment_order(self):
        """Chords should appear left-to-right in time order."""
        chart = assemble_chart(SAMPLE_CHORDS, SAMPLE_LYRICS, SAMPLE_SECTIONS,
                               title="Test", artist="Test")
        chorus = [s for s in chart["sections"] if s["name"] == "Chorus"][0]
        chords_str = chorus["lines"][0]["chords"]
        d_pos = chords_str.index("D")
        cm_pos = chords_str.index("C#m")
        assert d_pos < cm_pos, "D should appear before C#m"


class TestGracefulDegradation:
    """Test edge cases and degraded inputs."""

    def test_no_lyrics(self):
        chart = assemble_chart(SAMPLE_CHORDS, lyrics=None, sections=SAMPLE_SECTIONS,
                               title="Test", artist="Test")
        assert len(chart["sections"]) > 0
        # Should have chord-only lines
        for section in chart["sections"]:
            for line in section["lines"]:
                assert "chords" in line
                assert len(line["chords"]) > 0

    def test_no_sections(self):
        chart = assemble_chart(SAMPLE_CHORDS, SAMPLE_LYRICS, sections=None,
                               title="Test", artist="Test")
        assert len(chart["sections"]) > 0
        # Should auto-generate section names
        for section in chart["sections"]:
            assert section["name"].startswith("Verse")

    def test_no_chords(self):
        chart = assemble_chart([], SAMPLE_LYRICS, SAMPLE_SECTIONS,
                               title="Test", artist="Test")
        assert len(chart["sections"]) > 0
        for section in chart["sections"]:
            for line in section["lines"]:
                assert "lyrics" in line

    def test_empty_everything(self):
        chart = assemble_chart([], None, None, title="Empty", artist="Nobody")
        assert chart["title"] == "Empty"
        assert chart["sections"] == []

    def test_no_lyrics_no_sections(self):
        chart = assemble_chart(SAMPLE_CHORDS, lyrics=None, sections=None,
                               title="Test", artist="Test")
        assert len(chart["sections"]) > 0

    def test_single_chord(self):
        chart = assemble_chart(
            [{"chord": "Am", "time": 0.0, "duration": 10.0, "confidence": 0.9}],
            [{"text": "Hello world", "start_time": 0.0, "end_time": 5.0}],
            title="Single", artist="Test"
        )
        assert len(chart["sections"]) > 0


class TestRepeatDetection:
    """Test instrumental repeat notation."""

    def test_repeating_pattern_detected(self):
        # 4-chord pattern repeated 4 times
        chords = []
        pattern = ["Am", "C", "G", "F"]
        for rep in range(4):
            for i, name in enumerate(pattern):
                chords.append({
                    "chord": name,
                    "time": rep * 8.0 + i * 2.0,
                    "duration": 2.0,
                    "confidence": 0.9
                })

        chart = assemble_chart(chords, lyrics=None,
                               sections=[{"name": "Intro", "start_time": 0.0, "end_time": 32.0}],
                               title="Test", artist="Test")
        intro = chart["sections"][0]
        # Should detect repeat
        has_repeat = any("(x" in line["lyrics"] for line in intro["lines"])
        assert has_repeat, "Should detect repeating chord pattern"


class TestSaveChart:
    """Test JSON serialization."""

    def test_save_and_load(self):
        chart = assemble_chart(SAMPLE_CHORDS, SAMPLE_LYRICS, SAMPLE_SECTIONS,
                               title="The Time Comes", artist="Kozelski")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "chord_chart.json")
            save_chart(chart, path)
            assert os.path.exists(path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["title"] == chart["title"]
            assert len(loaded["sections"]) == len(chart["sections"])

    def test_save_creates_directories(self):
        chart = assemble_chart(SAMPLE_CHORDS, title="Test", artist="Test")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "chord_chart.json")
            save_chart(chart, path)
            assert os.path.exists(path)


class TestMatchesManualFormat:
    """
    Verify our output structure is compatible with the manual chord_chart_manual.json
    that the frontend already renders successfully.
    """

    def test_output_has_all_required_keys(self):
        chart = assemble_chart(SAMPLE_CHORDS, SAMPLE_LYRICS, SAMPLE_SECTIONS,
                               title="The Time Comes", artist="Kozelski")
        # Top-level keys that renderManualChordChart checks
        assert isinstance(chart.get("sections"), list)
        assert isinstance(chart.get("title"), str)
        assert isinstance(chart.get("artist"), str)

    def test_line_chords_are_strings(self):
        chart = assemble_chart(SAMPLE_CHORDS, SAMPLE_LYRICS, SAMPLE_SECTIONS,
                               title="Test", artist="Test")
        for section in chart["sections"]:
            for line in section["lines"]:
                assert isinstance(line["chords"], str)
                assert isinstance(line["lyrics"], str)

    def test_json_serializable(self):
        chart = assemble_chart(SAMPLE_CHORDS, SAMPLE_LYRICS, SAMPLE_SECTIONS,
                               title="Test", artist="Test")
        # Should not raise
        serialized = json.dumps(chart)
        reloaded = json.loads(serialized)
        assert reloaded == chart


class TestGenerateChordChartForJob:
    """Test the pipeline integration function generate_chord_chart_for_job."""

    def test_generates_chart_file(self):
        from chart_assembler import generate_chord_chart_for_job
        with tempfile.TemporaryDirectory() as tmpdir:
            chart = generate_chord_chart_for_job(
                job_id="test-123",
                chords=SAMPLE_CHORDS,
                vocals_path=None,
                audio_path=None,
                title="Test Song",
                artist="Test Artist",
                output_dir=tmpdir,
            )
            assert chart is not None
            assert chart["title"] == "Test Song"
            assert chart["artist"] == "Test Artist"
            assert chart["source"] == "auto"
            assert len(chart["sections"]) > 0
            # Verify file was saved
            chart_file = os.path.join(tmpdir, "chord_chart.json")
            assert os.path.exists(chart_file)

    def test_does_not_overwrite_manual_chart(self):
        from chart_assembler import generate_chord_chart_for_job
        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create a manual chart
            manual_path = os.path.join(tmpdir, "chord_chart_manual.json")
            with open(manual_path, 'w') as f:
                json.dump({"title": "Manual", "sections": []}, f)

            chart = generate_chord_chart_for_job(
                job_id="test-456",
                chords=SAMPLE_CHORDS,
                vocals_path=None,
                audio_path=None,
                title="Auto Song",
                artist="Auto Artist",
                output_dir=tmpdir,
            )
            assert chart is not None
            # Manual chart should be untouched
            with open(manual_path) as f:
                manual = json.load(f)
            assert manual["title"] == "Manual"
            # Auto chart saved as chord_chart_auto.json
            auto_path = os.path.join(tmpdir, "chord_chart_auto.json")
            assert os.path.exists(auto_path)

    def test_returns_none_on_empty_chords(self):
        from chart_assembler import generate_chord_chart_for_job
        with tempfile.TemporaryDirectory() as tmpdir:
            chart = generate_chord_chart_for_job(
                job_id="test-789",
                chords=[],
                vocals_path=None,
                audio_path=None,
                title="Empty",
                artist="Nobody",
                output_dir=tmpdir,
            )
            # With empty chords, chart is returned but sections will be empty
            assert chart is not None
            assert chart["sections"] == []
