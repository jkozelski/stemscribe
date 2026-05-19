"""Tests for the CHART_FORMATTER_PLACEMENT_CONSOLIDATE flag and the
beat-grid placement-consolidation pass (_placement_consolidate_to_beatgrid).

Lever: 3 forensic agents + the 5-song honest v2 bench identified PLACEMENT as
the dominant systematic defect, root-caused to detector-side chord-event
over-segmentation (librosa emits ~per-beat events with quality flicker). This
pass snaps event boundaries to the beat grid, re-merges duplicates, folds
same-chord-sandwiched flicker, and absorbs strictly sub-beat fragments into an
established neighbour.

Contract under test:
  1. Flag parsing is conservative (only explicit truthy values opt in).
  2. Flag UNSET  -> the pass is never invoked; format_chart output is
     byte-identical to the historical prod path.
  3. No grid / degenerate grid -> pass is a structural no-op (returns input
     list object unchanged) so the downstream path is unaffected.
  4. Boundary snap collapses per-beat duplicate events to a held chord.
  5. Same-chord-sandwiched short flicker (A X A) is absorbed into A.
  6. Strictly sub-beat fragment between two established chords is absorbed
     into the longer neighbour; a genuine fast (>= one beat) change survives.
  7. Inputs are never mutated.
"""

import copy
import os

import pytest

from backend.chart_formatter import (
    _placement_consolidate_enabled,
    _placement_consolidate_to_beatgrid,
    format_chart,
)


# -- 1. flag parsing --------------------------------------------------------

@pytest.mark.parametrize("val,expected", [
    ("1", True), ("true", True), ("TRUE", True), ("Yes", True),
    ("on", True), (" on ", True),
    ("0", False), ("false", False), ("no", False), ("off", False),
    ("", False), ("2", False), ("maybe", False),
])
def test_flag_parsing_is_conservative(val, expected, monkeypatch):
    monkeypatch.setenv("CHART_FORMATTER_PLACEMENT_CONSOLIDATE", val)
    assert _placement_consolidate_enabled() is expected


def test_flag_unset_is_disabled(monkeypatch):
    monkeypatch.delenv("CHART_FORMATTER_PLACEMENT_CONSOLIDATE", raising=False)
    assert _placement_consolidate_enabled() is False


# -- 3. no-grid / degenerate-grid no-op -------------------------------------

def test_no_grid_returns_same_object():
    ch = [{"time": 0.0, "duration": 1.0, "chord": "C"}]
    assert _placement_consolidate_to_beatgrid(ch, None) is ch
    assert _placement_consolidate_to_beatgrid(ch, {}) is ch
    assert _placement_consolidate_to_beatgrid(ch, {"beat_times": [1.0]}) is ch


def test_empty_input_returns_same_object():
    ch = []
    assert _placement_consolidate_to_beatgrid(ch, {"beat_times": [0, 1, 2]}) is ch


# -- 7. input is never mutated ----------------------------------------------

def test_does_not_mutate_input():
    beats = [round(0.5 * i, 3) for i in range(20)]
    ch = [
        {"time": 0.02, "duration": 0.48, "chord": "C"},
        {"time": 0.50, "duration": 0.50, "chord": "C"},
        {"time": 1.00, "duration": 0.50, "chord": "G"},
    ]
    snapshot = copy.deepcopy(ch)
    _placement_consolidate_to_beatgrid(ch, {"beat_times": beats})
    assert ch == snapshot


# -- 4. per-beat duplicate collapse -----------------------------------------

def test_boundary_snap_collapses_per_beat_duplicates():
    # 8 per-beat events, all C with tiny jitter on the boundaries.
    beats = [round(0.5 * i, 3) for i in range(12)]
    ch = []
    for i in range(8):
        ch.append({
            "time": round(0.5 * i + (0.03 if i % 2 else -0.02), 3),
            "duration": 0.5,
            "chord": "C",
        })
    out = _placement_consolidate_to_beatgrid(ch, {"beat_times": beats})
    assert [c["chord"] for c in out] == ["C"]


# -- 5. same-chord-sandwiched flicker absorption ----------------------------

def test_same_chord_sandwich_flicker_absorbed():
    beats = [round(0.5 * i, 3) for i in range(20)]
    ch = [
        {"time": 0.0, "duration": 2.0, "chord": "C"},   # held C
        {"time": 2.0, "duration": 0.5, "chord": "G"},   # 1-beat flicker
        {"time": 2.5, "duration": 2.0, "chord": "C"},   # back to C
        {"time": 4.5, "duration": 2.0, "chord": "F"},   # genuine change
    ]
    out = _placement_consolidate_to_beatgrid(ch, {"beat_times": beats})
    assert [c["chord"] for c in out] == ["C", "F"]


# -- 6. sub-beat fragment absorption vs genuine fast change -----------------

def test_subbeat_fragment_absorbed_into_longer_neighbour():
    beats = [round(0.5 * i, 3) for i in range(24)]
    ch = [
        {"time": 0.0, "duration": 4.0, "chord": "C"},   # established (bar+)
        {"time": 4.0, "duration": 0.2, "chord": "D"},   # sub-beat blip
        {"time": 4.2, "duration": 4.0, "chord": "G"},   # established
    ]
    out = _placement_consolidate_to_beatgrid(ch, {"beat_times": beats})
    assert "D" not in [c["chord"] for c in out]
    assert [c["chord"] for c in out] == ["C", "G"]


def test_genuine_fast_change_survives():
    # Two real one-bar chords with a real one-beat passing chord between
    # them; the passing chord is a full beat (not sub-beat) and its
    # neighbours differ, so neither flicker nor fragment rule fires.
    beats = [round(0.5 * i, 3) for i in range(24)]
    ch = [
        {"time": 0.0, "duration": 2.0, "chord": "C"},
        {"time": 2.0, "duration": 0.5, "chord": "D"},   # 1 full beat
        {"time": 2.5, "duration": 2.0, "chord": "G"},
    ]
    out = _placement_consolidate_to_beatgrid(ch, {"beat_times": beats})
    assert [c["chord"] for c in out] == ["C", "D", "G"]


# -- 2. byte-identical when flag unset --------------------------------------

def _minimal_inputs():
    """A tiny synthetic song that exercises the bass-less detector-only
    quantize branch (grid present, no bass_roots) — the only branch the
    flag touches."""
    grid = {
        "beat_times": [round(0.5 * i, 3) for i in range(40)],
        "downbeat_times": [round(2.0 * i, 3) for i in range(10)],
        "song_duration_sec": 20.0,
        "time_signature": "4/4",
        "tempo_bpm": 120.0,
    }
    chords = []
    prog = ["C", "C", "G", "G", "Am", "Am", "F", "F", "C", "C"]
    for i, ch in enumerate(prog):
        chords.append({"time": round(2.0 * i, 3), "duration": 2.0, "chord": ch})
    words = [
        {"word": "la", "start": 0.6, "end": 1.0},
        {"word": "la", "start": 4.6, "end": 5.0},
        {"word": "la", "start": 8.6, "end": 9.0},
    ]
    return chords, words, grid


def test_format_chart_byte_identical_when_flag_unset(monkeypatch):
    chords, words, grid = _minimal_inputs()

    def _run(env):
        if env is None:
            monkeypatch.delenv("CHART_FORMATTER_PLACEMENT_CONSOLIDATE",
                               raising=False)
        else:
            monkeypatch.setenv("CHART_FORMATTER_PLACEMENT_CONSOLIDATE", env)
        import json
        return json.dumps(
            format_chart(
                chord_events=copy.deepcopy(chords),
                word_timestamps=copy.deepcopy(words),
                title="t", artist="a", key="C",
                grid=copy.deepcopy(grid), bass_roots=None,
            ),
            sort_keys=True,
        )

    out_unset = _run(None)
    out_false = _run("false")
    out_empty = _run("")
    assert out_unset == out_false == out_empty


def test_consolidation_pass_actually_reduces_oversegmentation():
    """Sanity: the consolidation function is not a no-op — fed per-beat
    over-segmented librosa-style events it materially collapses the event
    count toward the true progression.

    NOTE: this asserts the *pass* changes the event list, NOT that the final
    chart bytes change. On the bass-less detector-only path the downstream
    bar-grid max-overlap vote (_quantize_chords_to_bars) is itself robust to
    flicker, so a clean synthetic case can quantize identically with or
    without the pass — an honest property worth documenting in a test."""
    beats = [round(0.5 * i, 3) for i in range(40)]
    grid = {"beat_times": beats}
    chords = []
    for i in range(36):
        c = "G" if i in (5, 13, 21) else "C"   # 3 transient G flickers
        chords.append({
            "time": round(0.5 * i + (0.04 if i % 2 else -0.03), 3),
            "duration": 0.5, "chord": c,
        })
    out = _placement_consolidate_to_beatgrid(chords, grid)
    assert len(out) < len(chords)            # over-segmentation reduced
    # The held C should dominate; transient G flicker folded away.
    assert [c["chord"] for c in out] == ["C"]
