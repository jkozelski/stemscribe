"""Tests for structure_detector.py — Ensemble Detector."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


# ---- Mock data ----

SAMPLE_CHORDS = [
    # Intro: F#m-A-Bm-E pattern (0-36s)
    {"chord": "F#m", "time": 0, "duration": 3, "confidence": 0.95},
    {"chord": "A", "time": 3, "duration": 3, "confidence": 0.95},
    {"chord": "Bm", "time": 6, "duration": 3, "confidence": 0.90},
    {"chord": "E", "time": 9, "duration": 3, "confidence": 0.90},
    {"chord": "F#m", "time": 12, "duration": 3, "confidence": 0.95},
    {"chord": "A", "time": 15, "duration": 3, "confidence": 0.95},
    {"chord": "Bm", "time": 18, "duration": 3, "confidence": 0.90},
    {"chord": "E", "time": 21, "duration": 3, "confidence": 0.90},
    {"chord": "F#m", "time": 24, "duration": 3, "confidence": 0.95},
    {"chord": "A", "time": 27, "duration": 3, "confidence": 0.95},
    {"chord": "Bm", "time": 30, "duration": 3, "confidence": 0.90},
    {"chord": "E", "time": 33, "duration": 3, "confidence": 0.90},
    # Verse: F#m-A / F#m-B pattern (36-54s)
    {"chord": "F#m", "time": 36, "duration": 3, "confidence": 0.95},
    {"chord": "A", "time": 39, "duration": 3, "confidence": 0.95},
    {"chord": "F#m", "time": 42, "duration": 3, "confidence": 0.95},
    {"chord": "B", "time": 45, "duration": 3, "confidence": 0.90},
    {"chord": "F#m", "time": 48, "duration": 3, "confidence": 0.95},
    {"chord": "A", "time": 51, "duration": 3, "confidence": 0.95},
    # Chorus: D-C#m-Bm / D-C#m-G (54-66s)
    {"chord": "D", "time": 54, "duration": 2, "confidence": 0.90},
    {"chord": "C#m", "time": 56, "duration": 2, "confidence": 0.90},
    {"chord": "Bm", "time": 58, "duration": 2, "confidence": 0.85},
    {"chord": "D", "time": 60, "duration": 2, "confidence": 0.90},
    {"chord": "C#m", "time": 62, "duration": 2, "confidence": 0.90},
    {"chord": "G", "time": 64, "duration": 2, "confidence": 0.85},
    # Verse 2: F#m-A / F#m-B (66-84s)
    {"chord": "F#m", "time": 66, "duration": 3, "confidence": 0.95},
    {"chord": "A", "time": 69, "duration": 3, "confidence": 0.95},
    {"chord": "F#m", "time": 72, "duration": 3, "confidence": 0.95},
    {"chord": "B", "time": 75, "duration": 3, "confidence": 0.90},
    {"chord": "F#m", "time": 78, "duration": 3, "confidence": 0.95},
    {"chord": "A", "time": 81, "duration": 3, "confidence": 0.95},
    # Chorus 2: D-C#m-Bm / D-C#m-G (84-96s)
    {"chord": "D", "time": 84, "duration": 2, "confidence": 0.90},
    {"chord": "C#m", "time": 86, "duration": 2, "confidence": 0.90},
    {"chord": "Bm", "time": 88, "duration": 2, "confidence": 0.85},
    {"chord": "D", "time": 90, "duration": 2, "confidence": 0.90},
    {"chord": "C#m", "time": 92, "duration": 2, "confidence": 0.90},
    {"chord": "G", "time": 94, "duration": 2, "confidence": 0.85},
    # Bridge: F#m-A-E-B (96-120s)
    {"chord": "F#m", "time": 96, "duration": 3, "confidence": 0.95},
    {"chord": "A", "time": 99, "duration": 3, "confidence": 0.95},
    {"chord": "E", "time": 102, "duration": 3, "confidence": 0.90},
    {"chord": "B", "time": 105, "duration": 3, "confidence": 0.90},
    {"chord": "F#m", "time": 108, "duration": 3, "confidence": 0.95},
    {"chord": "A", "time": 111, "duration": 3, "confidence": 0.95},
    {"chord": "E", "time": 114, "duration": 3, "confidence": 0.90},
    {"chord": "B", "time": 117, "duration": 3, "confidence": 0.90},
]

SAMPLE_LYRICS = [
    {"text": "You say you wanted", "start_time": 36.0, "end_time": 39.0},
    {"text": "Someone to take you away", "start_time": 42.0, "end_time": 45.0},
    {"text": "The time comes", "start_time": 54.0, "end_time": 57.0},
    {"text": "to turn it around", "start_time": 57.0, "end_time": 60.0},
    {"text": "Free your head now", "start_time": 66.0, "end_time": 69.0},
    {"text": "The time comes", "start_time": 84.0, "end_time": 87.0},
    {"text": "Come to life", "start_time": 96.0, "end_time": 99.0},
]

# Mock chord pattern analyzer output matching The Time Comes structure
MOCK_CHORD_PATTERNS = [
    {
        "pattern": "A",
        "start_time": 0.0,
        "end_time": 36.0,
        "chords": ["F#m", "A", "Bm", "E"],
        "bar_count": 6,
        "section_hint": "intro",
    },
    {
        "pattern": "B",
        "start_time": 36.0,
        "end_time": 54.0,
        "chords": ["F#m", "A", "B"],
        "bar_count": 6,
        "section_hint": "verse",
    },
    {
        "pattern": "C",
        "start_time": 54.0,
        "end_time": 66.0,
        "chords": ["D", "C#m", "Bm", "G"],
        "bar_count": 3,
        "section_hint": "chorus",
    },
    {
        "pattern": "B",
        "start_time": 66.0,
        "end_time": 84.0,
        "chords": ["F#m", "A", "B"],
        "bar_count": 6,
        "section_hint": "verse",
    },
    {
        "pattern": "C",
        "start_time": 84.0,
        "end_time": 96.0,
        "chords": ["D", "C#m", "Bm", "G"],
        "bar_count": 3,
        "section_hint": "chorus",
    },
    {
        "pattern": "D",
        "start_time": 96.0,
        "end_time": 120.0,
        "chords": ["F#m", "A", "E", "B"],
        "bar_count": 4,
        "section_hint": "bridge",
    },
]

# Mock lyrics repetition result
MOCK_LYRICS_RESULT = {
    "chorus_lines": [
        {
            "text": "The time comes",
            "instances": [
                {"start": 54.0, "end": 57.0},
                {"start": 84.0, "end": 87.0},
            ],
        }
    ],
    "sections": [
        {"name": "Intro", "start_time": 0.0, "end_time": 36.0, "confidence": 0.7},
        {"name": "Verse", "start_time": 36.0, "end_time": 54.0, "confidence": 0.75},
        {"name": "Chorus", "start_time": 54.0, "end_time": 60.0, "confidence": 0.9},
        {"name": "Verse", "start_time": 66.0, "end_time": 84.0, "confidence": 0.75},
        {"name": "Chorus", "start_time": 84.0, "end_time": 87.0, "confidence": 0.9},
        {"name": "Bridge", "start_time": 96.0, "end_time": 99.0, "confidence": 0.65},
    ],
}

# Mock audio boundaries
MOCK_AUDIO_BOUNDARIES = [
    {"time": 35.5, "energy_change": 0.3, "spectral_change": 0.4, "onset_density_change": 0.2, "chroma_novelty": 0.6, "confidence": 0.55},
    {"time": 53.8, "energy_change": 0.7, "spectral_change": 0.5, "onset_density_change": 0.4, "chroma_novelty": 0.7, "confidence": 0.65},
    {"time": 65.5, "energy_change": 0.5, "spectral_change": 0.4, "onset_density_change": 0.3, "chroma_novelty": 0.5, "confidence": 0.50},
    {"time": 83.5, "energy_change": 0.6, "spectral_change": 0.5, "onset_density_change": 0.4, "chroma_novelty": 0.6, "confidence": 0.60},
    {"time": 95.8, "energy_change": 0.4, "spectral_change": 0.3, "onset_density_change": 0.2, "chroma_novelty": 0.5, "confidence": 0.45},
]


def _mock_librosa_load(path, sr=22050):
    """Return fake audio of ~120s."""
    n_samples = 120 * sr
    y = np.random.randn(n_samples).astype(np.float32) * 0.01
    for start_s, end_s in [(54, 66), (84, 96)]:
        y[start_s * sr:end_s * sr] *= 5.0
    return y, sr


@pytest.fixture
def mock_librosa():
    """Mock librosa and sub-detectors to avoid needing actual audio files."""
    with patch("structure_detector.librosa") as mock_lib, \
         patch("structure_detector._get_chord_sections") as mock_chords, \
         patch("structure_detector._get_lyrics_sections") as mock_lyrics, \
         patch("structure_detector._get_audio_boundaries") as mock_audio:

        mock_lib.load = _mock_librosa_load
        mock_lib.get_duration = lambda y, sr: len(y) / sr

        # Sub-detectors return mock data
        mock_chords.side_effect = lambda chords: MOCK_CHORD_PATTERNS if chords and len(chords) >= 4 else []
        mock_lyrics.side_effect = lambda lyrics: MOCK_LYRICS_RESULT if lyrics else {"chorus_lines": [], "sections": []}
        mock_audio.return_value = MOCK_AUDIO_BOUNDARIES

        yield mock_lib


@pytest.fixture
def mock_librosa_no_subdetectors():
    """Mock only librosa, let sub-detectors run (for integration-style tests)."""
    with patch("structure_detector.librosa") as mock_lib:
        mock_lib.load = _mock_librosa_load
        mock_lib.get_duration = lambda y, sr: len(y) / sr
        yield mock_lib


class TestDetectStructure:
    """Tests for the main detect_structure function."""

    def test_returns_list(self, mock_librosa):
        from structure_detector import detect_structure
        result = detect_structure("fake_audio.mp3", chords=SAMPLE_CHORDS)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_each_section_has_required_keys(self, mock_librosa):
        from structure_detector import detect_structure
        result = detect_structure("fake_audio.mp3", chords=SAMPLE_CHORDS)
        for section in result:
            assert "name" in section
            assert "start_time" in section
            assert "end_time" in section
            assert section["end_time"] > section["start_time"]

    def test_sections_are_contiguous(self, mock_librosa):
        from structure_detector import detect_structure
        result = detect_structure("fake_audio.mp3", chords=SAMPLE_CHORDS)
        for i in range(1, len(result)):
            gap = abs(result[i]["start_time"] - result[i - 1]["end_time"])
            assert gap < 1.0, f"Gap of {gap}s between sections {i-1} and {i}"

    def test_sections_cover_full_duration(self, mock_librosa):
        from structure_detector import detect_structure
        result = detect_structure("fake_audio.mp3", chords=SAMPLE_CHORDS)
        assert result[0]["start_time"] <= 1.0
        assert result[-1]["end_time"] >= 118.0  # ~120s audio

    def test_detects_chorus_with_distinctive_chords(self, mock_librosa):
        from structure_detector import detect_structure
        result = detect_structure("fake_audio.mp3", chords=SAMPLE_CHORDS)
        chorus_sections = [s for s in result if "Chorus" in s["name"]]
        assert len(chorus_sections) >= 1, "Should detect at least one chorus"

    def test_detects_verse(self, mock_librosa):
        from structure_detector import detect_structure
        result = detect_structure("fake_audio.mp3", chords=SAMPLE_CHORDS)
        verse_sections = [s for s in result if "Verse" in s["name"]]
        assert len(verse_sections) >= 1, "Should detect at least one verse"

    def test_no_internal_keys_in_output(self, mock_librosa):
        from structure_detector import detect_structure
        result = detect_structure("fake_audio.mp3", chords=SAMPLE_CHORDS)
        for section in result:
            for key in section:
                assert not key.startswith("_"), f"Internal key {key} leaked to output"

    def test_lyrics_refine_intro(self, mock_librosa):
        from structure_detector import detect_structure
        result = detect_structure(
            "fake_audio.mp3",
            chords=SAMPLE_CHORDS,
            lyrics=SAMPLE_LYRICS,
        )
        # First section before lyrics (start at 36s) should be Intro
        first = result[0]
        assert first["name"] == "Intro" or first["end_time"] <= 38.0

    def test_empty_chords_returns_fallback(self, mock_librosa):
        from structure_detector import detect_structure
        result = detect_structure("fake_audio.mp3", chords=[])
        # Should still work (audio-only fallback)
        assert isinstance(result, list)

    def test_no_chords_returns_list(self, mock_librosa):
        """Without chords, should return a list (possibly empty if audio fallback fails)."""
        from structure_detector import detect_structure
        result = detect_structure("fake_audio.mp3", chords=None)
        assert isinstance(result, list)


class TestFallback:
    """Test fallback behavior."""

    def test_fallback_from_chords(self):
        from structure_detector import _fallback_from_chords
        result = _fallback_from_chords(SAMPLE_CHORDS)
        assert len(result) == 1
        assert result[0]["name"] == "Song"
        assert result[0]["start_time"] == 0.0
        assert result[0]["end_time"] > 100

    def test_fallback_empty(self):
        from structure_detector import _fallback_from_chords
        assert _fallback_from_chords([]) == []
        assert _fallback_from_chords(None) == []


class TestPostprocess:
    """Test post-processing logic."""

    def test_merge_adjacent_same_label(self):
        from structure_detector import _postprocess
        segments = [
            {"name": "Verse", "start_time": 0, "end_time": 30},
            {"name": "Verse", "start_time": 30, "end_time": 60},
            {"name": "Chorus", "start_time": 60, "end_time": 90},
        ]
        result = _postprocess(segments, 90)
        assert len(result) == 2
        assert result[0]["name"] == "Verse"
        assert result[0]["end_time"] == 60

    def test_numbers_repeated_labels(self):
        from structure_detector import _postprocess
        segments = [
            {"name": "Verse", "start_time": 0, "end_time": 30},
            {"name": "Chorus", "start_time": 30, "end_time": 60},
            {"name": "Verse", "start_time": 60, "end_time": 90},
            {"name": "Chorus", "start_time": 90, "end_time": 120},
        ]
        result = _postprocess(segments, 120)
        names = [s["name"] for s in result]
        assert "Verse 1" in names
        assert "Verse 2" in names
        assert "Chorus 1" in names
        assert "Chorus 2" in names
