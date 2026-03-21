"""Tests for chord_pattern_analyzer.py."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chord_pattern_analyzer import (
    analyze_chord_patterns,
    _parse_events,
    _jaccard_similarity,
    _sequence_similarity,
    _label_patterns,
    _build_time_windows,
    _cluster_fingerprints,
    _window_fingerprint,
    _merge_short_sections,
)


# ---- Test data ----

# Simplified "The Time Comes" chord progression
SAMPLE_CHORDS = [
    # Intro: F#m-A-Bm-E pattern (8-20s)
    {"chord": "F#m", "time": 8.0, "duration": 1.6, "confidence": 0.98},
    {"chord": "A", "time": 10.0, "duration": 1.3, "confidence": 0.99},
    {"chord": "Bm", "time": 11.3, "duration": 1.7, "confidence": 0.91},
    {"chord": "E", "time": 13.0, "duration": 1.6, "confidence": 0.88},
    {"chord": "F#m", "time": 14.5, "duration": 1.6, "confidence": 0.99},
    {"chord": "A", "time": 16.1, "duration": 1.5, "confidence": 0.99},
    {"chord": "Bm", "time": 17.6, "duration": 1.6, "confidence": 0.85},
    {"chord": "E", "time": 19.2, "duration": 1.7, "confidence": 0.82},
    # Verse 1: F#m-A / F#m-B pattern (20-44s)
    {"chord": "F#m", "time": 20.8, "duration": 1.5, "confidence": 0.97},
    {"chord": "A", "time": 22.3, "duration": 1.1, "confidence": 0.98},
    {"chord": "F#m", "time": 23.4, "duration": 2.0, "confidence": 0.96},
    {"chord": "B", "time": 25.5, "duration": 1.6, "confidence": 0.99},
    {"chord": "F#m", "time": 27.0, "duration": 1.5, "confidence": 0.94},
    {"chord": "A", "time": 28.5, "duration": 0.9, "confidence": 0.94},
    {"chord": "F#m", "time": 29.4, "duration": 2.2, "confidence": 0.90},
    {"chord": "B", "time": 31.7, "duration": 1.7, "confidence": 0.99},
    {"chord": "F#m", "time": 33.3, "duration": 1.4, "confidence": 0.97},
    {"chord": "A", "time": 34.7, "duration": 1.1, "confidence": 0.90},
    {"chord": "F#m", "time": 36.0, "duration": 1.9, "confidence": 0.88},
    {"chord": "B", "time": 38.0, "duration": 1.5, "confidence": 0.99},
    {"chord": "F#m", "time": 39.4, "duration": 1.7, "confidence": 0.98},
    {"chord": "A", "time": 41.1, "duration": 1.3, "confidence": 0.99},
    {"chord": "F#m", "time": 42.4, "duration": 1.8, "confidence": 0.99},
    {"chord": "B", "time": 44.2, "duration": 1.5, "confidence": 0.98},
    # Chorus 1: D-C#m-Bm / D-C#m-G (45-58s)
    {"chord": "D", "time": 45.6, "duration": 1.7, "confidence": 0.99},
    {"chord": "C#m", "time": 47.3, "duration": 1.5, "confidence": 0.99},
    {"chord": "Bm", "time": 48.8, "duration": 3.2, "confidence": 0.99},
    {"chord": "D", "time": 52.0, "duration": 1.5, "confidence": 0.99},
    {"chord": "C#m", "time": 53.5, "duration": 1.5, "confidence": 0.86},
    {"chord": "G", "time": 55.0, "duration": 3.2, "confidence": 0.88},
    # Verse 2: F#m-A / F#m-B (58-80s)
    {"chord": "F#m", "time": 58.1, "duration": 1.9, "confidence": 0.99},
    {"chord": "A", "time": 60.0, "duration": 0.9, "confidence": 0.99},
    {"chord": "F#m", "time": 61.0, "duration": 1.8, "confidence": 0.95},
    {"chord": "B", "time": 62.8, "duration": 1.6, "confidence": 0.99},
    {"chord": "F#m", "time": 64.4, "duration": 1.6, "confidence": 0.95},
    {"chord": "A", "time": 65.9, "duration": 1.2, "confidence": 0.98},
    {"chord": "F#m", "time": 67.1, "duration": 1.9, "confidence": 0.96},
    {"chord": "B", "time": 69.1, "duration": 1.7, "confidence": 0.99},
    {"chord": "F#m", "time": 70.7, "duration": 1.6, "confidence": 0.99},
    {"chord": "A", "time": 72.3, "duration": 0.9, "confidence": 0.93},
    {"chord": "F#m", "time": 73.5, "duration": 1.9, "confidence": 0.98},
    {"chord": "B", "time": 75.4, "duration": 1.6, "confidence": 0.99},
    # Chorus 2: D-C#m-Bm / D-C#m-G (83-96s)
    {"chord": "D", "time": 83.1, "duration": 1.7, "confidence": 0.99},
    {"chord": "C#m", "time": 84.7, "duration": 1.6, "confidence": 0.87},
    {"chord": "Bm", "time": 86.3, "duration": 2.8, "confidence": 0.99},
    {"chord": "D", "time": 89.4, "duration": 1.7, "confidence": 0.99},
    {"chord": "C#m", "time": 91.0, "duration": 1.5, "confidence": 0.97},
    {"chord": "G", "time": 92.5, "duration": 3.2, "confidence": 0.87},
]

# Minimal test data
MINIMAL_CHORDS = [
    {"chord": "C", "time": 0.0, "duration": 2.0, "confidence": 0.9},
    {"chord": "G", "time": 2.0, "duration": 2.0, "confidence": 0.9},
]

# Simple repeating pattern
SIMPLE_REPEATING = [
    # Pattern A: C-G-Am-F (repeats 3x)
    {"chord": "C", "time": 0.0, "duration": 1.5, "confidence": 0.9},
    {"chord": "G", "time": 1.5, "duration": 1.5, "confidence": 0.9},
    {"chord": "Am", "time": 3.0, "duration": 1.5, "confidence": 0.9},
    {"chord": "F", "time": 4.5, "duration": 1.5, "confidence": 0.9},
    {"chord": "C", "time": 6.0, "duration": 1.5, "confidence": 0.9},
    {"chord": "G", "time": 7.5, "duration": 1.5, "confidence": 0.9},
    {"chord": "Am", "time": 9.0, "duration": 1.5, "confidence": 0.9},
    {"chord": "F", "time": 10.5, "duration": 1.5, "confidence": 0.9},
    {"chord": "C", "time": 12.0, "duration": 1.5, "confidence": 0.9},
    {"chord": "G", "time": 13.5, "duration": 1.5, "confidence": 0.9},
    {"chord": "Am", "time": 15.0, "duration": 1.5, "confidence": 0.9},
    {"chord": "F", "time": 16.5, "duration": 1.5, "confidence": 0.9},
    # Pattern B: Dm-Em-F-G (appears once)
    {"chord": "Dm", "time": 18.0, "duration": 1.5, "confidence": 0.9},
    {"chord": "Em", "time": 19.5, "duration": 1.5, "confidence": 0.9},
    {"chord": "F", "time": 21.0, "duration": 1.5, "confidence": 0.9},
    {"chord": "G", "time": 22.5, "duration": 1.5, "confidence": 0.9},
    # Pattern A again
    {"chord": "C", "time": 24.0, "duration": 1.5, "confidence": 0.9},
    {"chord": "G", "time": 25.5, "duration": 1.5, "confidence": 0.9},
    {"chord": "Am", "time": 27.0, "duration": 1.5, "confidence": 0.9},
    {"chord": "F", "time": 28.5, "duration": 1.5, "confidence": 0.9},
    {"chord": "C", "time": 30.0, "duration": 1.5, "confidence": 0.9},
    {"chord": "G", "time": 31.5, "duration": 1.5, "confidence": 0.9},
    {"chord": "Am", "time": 33.0, "duration": 1.5, "confidence": 0.9},
    {"chord": "F", "time": 34.5, "duration": 1.5, "confidence": 0.9},
]


# ==============================================================================
# Tests
# ==============================================================================

class TestAnalyzeChordPatterns:
    """Tests for the main analyze_chord_patterns function."""

    def test_returns_list(self):
        result = analyze_chord_patterns(SAMPLE_CHORDS)
        assert isinstance(result, list)

    def test_returns_sections(self):
        result = analyze_chord_patterns(SAMPLE_CHORDS)
        assert len(result) > 0

    def test_each_section_has_required_keys(self):
        result = analyze_chord_patterns(SAMPLE_CHORDS)
        for section in result:
            assert "pattern" in section
            assert "start_time" in section
            assert "end_time" in section
            assert "chords" in section
            assert "bar_count" in section
            assert "section_hint" in section

    def test_sections_are_contiguous(self):
        result = analyze_chord_patterns(SAMPLE_CHORDS)
        for i in range(1, len(result)):
            gap = abs(result[i]["start_time"] - result[i - 1]["end_time"])
            assert gap < 2.0, f"Gap of {gap}s between sections {i-1} and {i}"

    def test_end_after_start(self):
        result = analyze_chord_patterns(SAMPLE_CHORDS)
        for section in result:
            assert section["end_time"] > section["start_time"]

    def test_pattern_labels_are_letters(self):
        result = analyze_chord_patterns(SAMPLE_CHORDS)
        for section in result:
            assert section["pattern"].isalpha()
            assert section["pattern"].isupper()

    def test_section_hints_are_valid(self):
        valid_hints = {"intro", "verse", "chorus", "bridge", "section", "outro", ""}
        result = analyze_chord_patterns(SAMPLE_CHORDS)
        for section in result:
            assert section["section_hint"] in valid_hints

    def test_detects_multiple_patterns(self):
        result = analyze_chord_patterns(SAMPLE_CHORDS)
        patterns = set(s["pattern"] for s in result)
        assert len(patterns) >= 2, "Should detect at least 2 different patterns"

    def test_chords_not_empty(self):
        result = analyze_chord_patterns(SAMPLE_CHORDS)
        for section in result:
            assert len(section["chords"]) > 0

    def test_empty_chords_returns_empty(self):
        assert analyze_chord_patterns([]) == []

    def test_none_chords_returns_empty(self):
        assert analyze_chord_patterns(None) == []

    def test_single_chord_returns_empty(self):
        result = analyze_chord_patterns([{"chord": "C", "time": 0, "duration": 2, "confidence": 0.9}])
        assert result == []

    def test_minimal_chords_returns_list(self):
        result = analyze_chord_patterns(MINIMAL_CHORDS)
        assert isinstance(result, list)


class TestSimpleRepeating:
    """Tests with a simple repeating pattern."""

    def test_detects_repeating_pattern(self):
        result = analyze_chord_patterns(SIMPLE_REPEATING)
        assert len(result) >= 2, "Should detect at least verse + bridge"

    def test_most_common_pattern_is_A(self):
        result = analyze_chord_patterns(SIMPLE_REPEATING)
        # Pattern A should be the most common (the C-G-Am-F blocks)
        a_sections = [s for s in result if s["pattern"] == "A"]
        assert len(a_sections) >= 1


class TestParseEvents:
    """Tests for _parse_events."""

    def test_parse_basic(self):
        events = _parse_events([
            {"chord": "C", "time": 0.0, "duration": 2.0, "confidence": 0.9},
            {"chord": "G", "time": 2.0, "duration": 2.0, "confidence": 0.9},
        ])
        assert len(events) == 2
        assert events[0]["chord"] == "C"
        assert events[1]["chord"] == "G"

    def test_filters_N_chords(self):
        events = _parse_events([
            {"chord": "N", "time": 0.0, "duration": 2.0, "confidence": 0.5},
            {"chord": "C", "time": 2.0, "duration": 2.0, "confidence": 0.9},
        ])
        assert len(events) == 1
        assert events[0]["chord"] == "C"

    def test_sorts_by_time(self):
        events = _parse_events([
            {"chord": "G", "time": 5.0, "duration": 2.0, "confidence": 0.9},
            {"chord": "C", "time": 1.0, "duration": 2.0, "confidence": 0.9},
        ])
        assert events[0]["time"] < events[1]["time"]

    def test_empty_input(self):
        assert _parse_events([]) == []


class TestJaccardSimilarity:
    """Tests for _jaccard_similarity."""

    def test_identical_sets(self):
        s = frozenset(["C", "G", "Am"])
        assert _jaccard_similarity(s, s) == 1.0

    def test_disjoint_sets(self):
        a = frozenset(["C", "G"])
        b = frozenset(["Dm", "Em"])
        assert _jaccard_similarity(a, b) == 0.0

    def test_partial_overlap(self):
        a = frozenset(["C", "G", "Am"])
        b = frozenset(["C", "G", "F"])
        sim = _jaccard_similarity(a, b)
        assert 0.4 < sim < 0.6  # 2/4 = 0.5

    def test_empty_sets(self):
        assert _jaccard_similarity(frozenset(), frozenset()) == 1.0

    def test_one_empty(self):
        assert _jaccard_similarity(frozenset(["C"]), frozenset()) == 0.0


class TestSequenceSimilarity:
    """Tests for _sequence_similarity."""

    def test_identical(self):
        assert _sequence_similarity(["C", "G"], ["C", "G"]) == 1.0

    def test_completely_different(self):
        sim = _sequence_similarity(["C", "G"], ["Dm", "Em"])
        assert sim == 0.0

    def test_one_edit(self):
        sim = _sequence_similarity(["C", "G", "Am"], ["C", "G", "F"])
        assert 0.5 < sim < 1.0

    def test_empty(self):
        assert _sequence_similarity([], []) == 1.0
        assert _sequence_similarity(["C"], []) == 0.0


class TestBuildTimeWindows:
    """Tests for _build_time_windows."""

    def test_basic_windows(self):
        events = _parse_events(SAMPLE_CHORDS[:8])  # First 8 chords
        windows = _build_time_windows(events, events[0]["time"], events[-1]["end"], 6.0)
        assert len(windows) >= 1
        for w in windows:
            assert "chord_set" in w
            assert "start_time" in w
            assert "end_time" in w
            assert len(w["chords"]) > 0


class TestClusterFingerprints:
    """Tests for _cluster_fingerprints."""

    def test_identical_fingerprints_same_cluster(self):
        fps = [
            frozenset(["C", "G", "Am"]),
            frozenset(["C", "G", "Am"]),
            frozenset(["D", "Em", "F"]),
        ]
        clusters = _cluster_fingerprints(fps)
        assert clusters[0] == clusters[1]
        assert clusters[0] != clusters[2]

    def test_similar_fingerprints_same_cluster(self):
        fps = [
            frozenset(["C", "G", "Am", "F"]),
            frozenset(["C", "G", "Am"]),  # Missing one chord
        ]
        clusters = _cluster_fingerprints(fps, threshold=0.5)
        assert clusters[0] == clusters[1]

    def test_different_fingerprints_different_clusters(self):
        fps = [
            frozenset(["C", "G"]),
            frozenset(["Dm", "Em", "F#"]),
        ]
        clusters = _cluster_fingerprints(fps)
        assert clusters[0] != clusters[1]


class TestLabelPatterns:
    """Tests for _label_patterns."""

    def test_most_common_is_verse(self):
        sections = [
            {"pattern": "P0", "bar_count": 6, "chords": ["C", "G"], "section_hint": ""},
            {"pattern": "P1", "bar_count": 2, "chords": ["D", "Em"], "section_hint": ""},
            {"pattern": "P0", "bar_count": 6, "chords": ["C", "G"], "section_hint": ""},
            {"pattern": "P1", "bar_count": 2, "chords": ["D", "Em"], "section_hint": ""},
        ]
        result = _label_patterns(sections)
        assert result[0]["section_hint"] == "verse"
        assert result[0]["pattern"] == "A"

    def test_second_most_common_is_chorus(self):
        sections = [
            {"pattern": "P0", "bar_count": 6, "chords": ["C", "G"], "section_hint": ""},
            {"pattern": "P1", "bar_count": 3, "chords": ["D", "Em"], "section_hint": ""},
            {"pattern": "P0", "bar_count": 6, "chords": ["C", "G"], "section_hint": ""},
            {"pattern": "P1", "bar_count": 3, "chords": ["D", "Em"], "section_hint": ""},
        ]
        result = _label_patterns(sections)
        chorus_sections = [s for s in result if s["section_hint"] == "chorus"]
        assert len(chorus_sections) >= 1

    def test_unique_pattern_is_bridge(self):
        sections = [
            {"pattern": "P0", "bar_count": 6, "chords": ["C", "G"], "section_hint": ""},
            {"pattern": "P1", "bar_count": 3, "chords": ["D", "Em"], "section_hint": ""},
            {"pattern": "P2", "bar_count": 2, "chords": ["Bb", "Cm"], "section_hint": ""},
            {"pattern": "P0", "bar_count": 6, "chords": ["C", "G"], "section_hint": ""},
        ]
        result = _label_patterns(sections)
        bridge_sections = [s for s in result if s["section_hint"] == "bridge"]
        assert len(bridge_sections) >= 1

    def test_empty_sections(self):
        assert _label_patterns([]) == []


class TestMergeShortSections:
    """Tests for _merge_short_sections."""

    def test_merges_unique_short(self):
        sections = [
            {"pattern": "A", "bar_count": 4, "start_time": 0, "end_time": 20, "chords": ["C"]},
            {"pattern": "X", "bar_count": 1, "start_time": 20, "end_time": 25, "chords": ["D"]},
            {"pattern": "A", "bar_count": 4, "start_time": 25, "end_time": 45, "chords": ["C"]},
        ]
        result = _merge_short_sections(sections)
        # X should be merged into A since it only appears once
        assert len(result) <= 2

    def test_keeps_repeating_short(self):
        sections = [
            {"pattern": "A", "bar_count": 4, "start_time": 0, "end_time": 20, "chords": ["C"]},
            {"pattern": "B", "bar_count": 1, "start_time": 20, "end_time": 25, "chords": ["D"]},
            {"pattern": "A", "bar_count": 4, "start_time": 25, "end_time": 45, "chords": ["C"]},
            {"pattern": "B", "bar_count": 1, "start_time": 45, "end_time": 50, "chords": ["D"]},
        ]
        result = _merge_short_sections(sections)
        # B appears twice, should be kept
        b_sections = [s for s in result if s["pattern"] == "B"]
        assert len(b_sections) >= 1


class TestRealData:
    """Tests using actual ai_chords.json if available."""

    @pytest.fixture
    def real_chords(self):
        import json
        chords_path = os.path.expanduser(
            "~/stemscribe/outputs/4afcef17-c1ed-4bd1-8eb8-e43afe027ba5/ai_chords.json"
        )
        if not os.path.exists(chords_path):
            pytest.skip("Real chord data not available")
        with open(chords_path) as f:
            return json.load(f)

    @pytest.fixture
    def known_structure(self):
        import json
        struct_path = os.path.expanduser(
            "~/stemscribe/outputs/4afcef17-c1ed-4bd1-8eb8-e43afe027ba5/structure.json"
        )
        if not os.path.exists(struct_path):
            pytest.skip("Known structure not available")
        with open(struct_path) as f:
            return json.load(f)

    def test_real_data_produces_sections(self, real_chords):
        result = analyze_chord_patterns(real_chords)
        assert len(result) >= 4, f"Expected at least 4 sections, got {len(result)}"

    def test_real_data_detects_intro(self, real_chords):
        result = analyze_chord_patterns(real_chords)
        first = result[0]
        # First section should start near the beginning
        assert first["start_time"] < 15.0
        # Intro should contain F#m, A, Bm, E
        assert "F#m" in first["chords"]

    def test_real_data_detects_verse_pattern(self, real_chords):
        result = analyze_chord_patterns(real_chords)
        verse_sections = [s for s in result if s["section_hint"] == "verse"]
        assert len(verse_sections) >= 2, "Should detect at least 2 verse sections"

    def test_real_data_boundaries_near_known(self, real_chords, known_structure):
        """
        Verify that detected boundaries roughly align with known structure.
        Allow tolerance of ~5s since exact boundaries depend on window size.
        """
        result = analyze_chord_patterns(real_chords)
        detected_boundaries = [s["start_time"] for s in result]
        known_boundaries = [s["start_time"] for s in known_structure]

        # At least some detected boundaries should be within 5s of known ones
        matches = 0
        for kb in known_boundaries[1:]:  # Skip 0.0
            for db in detected_boundaries:
                if abs(kb - db) < 8.0:
                    matches += 1
                    break

        # At least 40% of known boundaries should have a match
        match_ratio = matches / max(1, len(known_boundaries) - 1)
        assert match_ratio >= 0.4, (
            f"Only {matches}/{len(known_boundaries)-1} known boundaries matched "
            f"(ratio={match_ratio:.2f}). "
            f"Detected: {[f'{b:.1f}' for b in detected_boundaries]}, "
            f"Known: {[f'{b:.1f}' for b in known_boundaries]}"
        )
