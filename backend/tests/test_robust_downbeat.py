"""Tests for the feature-flagged robust (madmom) downbeat grid.

Covers:
  * the ROBUST_DOWNBEAT flag parsing (default OFF for prod safety)
  * _madmom_grid output schema matches the legacy grid (downstream contract)
  * conservative octave guard halves only implausibly-fast tempos
  * graceful fallback to the legacy path when madmom is unavailable
  * the default/legacy grids carry the new schema keys (no KeyError downstream)

These run WITHOUT real audio / without madmom installed by mocking the
processors, so they are deterministic and CI-safe.
"""

import os
import sys
import types
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from processing import tempo_beats as tb  # noqa: E402


# ---------------------------------------------------------------------------
# flag parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("val,expected", [
    (None, False), ("", False), ("0", False), ("false", False),
    ("no", False), ("off", False),
    ("1", True), ("true", True), ("TRUE", True), ("yes", True),
    ("on", True), (" On ", True),
])
def test_flag_parsing(val, expected, monkeypatch):
    if val is None:
        monkeypatch.delenv("ROBUST_DOWNBEAT", raising=False)
    else:
        monkeypatch.setenv("ROBUST_DOWNBEAT", val)
    assert tb._robust_downbeat_enabled() is expected


def test_flag_default_off(monkeypatch):
    """Prod safety: with no env var set, the robust path is OFF."""
    monkeypatch.delenv("ROBUST_DOWNBEAT", raising=False)
    assert tb._robust_downbeat_enabled() is False


# ---------------------------------------------------------------------------
# schema compatibility — every downstream consumer indexes these keys
# ---------------------------------------------------------------------------

_GRID_KEYS = {
    "tempo_bpm", "time_signature", "beat_times", "downbeat_times",
    "bar_count", "downbeat_offset", "source", "song_duration_sec",
    "grid_method", "low_confidence",
}


def test_default_grid_schema():
    g = tb._default_grid()
    assert _GRID_KEYS.issubset(g.keys())
    assert g["grid_method"] == "default"
    assert g["low_confidence"] is True


def _fake_madmom(beats_array):
    """Return a fake madmom module whose processors yield `beats_array`."""
    mod = types.ModuleType("madmom")
    feats = types.ModuleType("madmom.features")
    db = types.ModuleType("madmom.features.downbeats")

    class _RNN:
        def __call__(self, _src):
            return np.zeros((100, 2), dtype=float)

    class _DBN:
        def __init__(self, **kw):
            pass

        def __call__(self, _act):
            return np.asarray(beats_array, dtype=float)

    db.RNNDownBeatProcessor = _RNN
    db.DBNDownBeatTrackingProcessor = _DBN
    feats.downbeats = db
    mod.features = feats
    return {"madmom": mod, "madmom.features": feats,
            "madmom.features.downbeats": db}


def test_madmom_grid_schema_matches_legacy():
    # 8 bars at 120 BPM (0.5s/beat, 2.0s/bar), beat_pos 1..4 repeating.
    beats = []
    t = 0.0
    for bar in range(8):
        for pos in range(1, 5):
            beats.append([round(t, 4), pos])
            t += 0.5
    with patch.dict(sys.modules, _fake_madmom(beats)):
        g = tb._madmom_grid("/fake/drums.mp3", 22050)
    assert g is not None
    assert _GRID_KEYS.issubset(g.keys())
    assert g["grid_method"] == "madmom_dbn"
    assert g["low_confidence"] is False
    assert g["bar_count"] == 8
    # downbeats are the beat_pos==1 entries: 0.0, 2.0, 4.0, ...
    assert g["downbeat_times"][:3] == [0.0, 2.0, 4.0]
    assert abs(g["tempo_bpm"] - 120.0) < 1.0


def test_madmom_unavailable_returns_none():
    """No madmom -> _madmom_grid returns None so caller uses legacy path."""
    with patch.dict(sys.modules, {"madmom": None,
                                  "madmom.features": None,
                                  "madmom.features.downbeats": None}):
        assert tb._madmom_grid("/fake/drums.mp3", 22050) is None


def test_madmom_too_few_beats_returns_none():
    with patch.dict(sys.modules, _fake_madmom([[0.0, 1], [0.5, 2]])):
        assert tb._madmom_grid("/fake/drums.mp3", 22050) is None


def test_madmom_no_bar_phase_low_confidence():
    """Beats present but no beat_pos==1 -> best-effort grid, flagged."""
    beats = [[i * 0.5, 2] for i in range(20)]  # never a downbeat
    with patch.dict(sys.modules, _fake_madmom(beats)):
        g = tb._madmom_grid("/fake/drums.mp3", 22050)
    assert g is not None
    assert g["low_confidence"] is True
    assert g["bar_count"] >= 1  # still usable


def test_octave_guard_halves_only_implausibly_fast():
    """A ~250 BPM read (0.96s/bar) is halved; a normal ~108 BPM read is not."""
    # implausibly fast: bar period 0.96s -> 250 BPM, half=125 (in band) -> halve
    fast = []
    t = 0.0
    for bar in range(8):
        for pos in range(1, 5):
            fast.append([round(t, 4), pos])
            t += 0.24  # 0.96s/bar
    with patch.dict(sys.modules, _fake_madmom(fast)):
        g = tb._madmom_grid("/fake/d.mp3", 22050)
    assert g is not None
    assert 120.0 <= g["tempo_bpm"] <= 130.0  # halved from ~250

    # normal 108 BPM (0.555s/beat, 2.22s/bar) must NOT be halved
    normal = []
    t = 0.0
    for bar in range(8):
        for pos in range(1, 5):
            normal.append([round(t, 4), pos])
            t += 0.5556
    with patch.dict(sys.modules, _fake_madmom(normal)):
        g = tb._madmom_grid("/fake/d.mp3", 22050)
    assert g is not None
    assert 105.0 <= g["tempo_bpm"] <= 112.0  # untouched


def test_extract_grid_flag_off_skips_madmom(monkeypatch):
    """With the flag OFF, _madmom_grid is never called even if a source exists."""
    monkeypatch.delenv("ROBUST_DOWNBEAT", raising=False)
    with patch.object(tb, "_pick_source", return_value=None), \
         patch.object(tb, "_madmom_grid") as mm:
        tb.extract_grid(drums_path=None, mix_path=None)
    mm.assert_not_called()
