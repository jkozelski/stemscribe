"""Tests for half-time tempo correction in midi_to_notation._detect_tempo.

Regression coverage for the 2026-04-25 audit finding: librosa occasionally
locks onto half the real tempo on tracks with sparse onsets (isolated bass
stems, quiet intros). Symptom on Aja's bass MusicXML: detected 46 BPM,
actual ~92 BPM, which squeezed all notes into half the measures.

Fix re-runs librosa with start_bpm=140 prior when the original estimate is
suspiciously low (<70 BPM), and prefers the doubled value if it agrees.
"""

import sys
import types
from pathlib import Path
from unittest.mock import patch, MagicMock


def _import_detect_tempo():
    """Import _detect_tempo lazily so tests don't need music21 if unavailable."""
    from backend.midi_to_notation import _detect_tempo
    return _detect_tempo


def _mock_score_no_tempo():
    """Score with no embedded MetronomeMark — forces librosa fallback."""
    score = MagicMock()
    score.recurse.return_value = []
    return score


def test_half_time_corrected_when_alt_agrees(tmp_path):
    """46 BPM (half-time) + 92 BPM (alt run) → returns 92."""
    audio = tmp_path / "fake.mp3"
    audio.write_bytes(b"\x00")  # exists

    detect = _import_detect_tempo()

    # Mock librosa: first call returns 46, alt call (start_bpm=140) returns 92.
    fake_librosa = types.SimpleNamespace()
    call_log = []

    def beat_track(y=None, sr=None, start_bpm=None):
        call_log.append(start_bpm)
        if start_bpm == 140:
            return (92.0, None)
        return (46.0, None)

    fake_librosa.load = lambda path, duration=60: (b"y", 22050)
    fake_librosa.beat = types.SimpleNamespace(beat_track=beat_track)

    with patch.dict(sys.modules, {'librosa': fake_librosa}):
        result = detect(_mock_score_no_tempo(), str(audio))

    assert result == 92.0, f"expected half-time correction → 92, got {result}"
    assert 140 in call_log, "alt-tempo run with start_bpm=140 should have been called"


def test_no_correction_when_alt_disagrees(tmp_path):
    """60 BPM (genuinely slow) + alt 65 BPM (doesn't agree on doubling) → keep 60."""
    audio = tmp_path / "fake.mp3"
    audio.write_bytes(b"\x00")
    detect = _import_detect_tempo()

    fake_librosa = types.SimpleNamespace()

    def beat_track(y=None, sr=None, start_bpm=None):
        if start_bpm == 140:
            return (65.0, None)  # 65 is nowhere near 60×2=120 → no correction
        return (60.0, None)

    fake_librosa.load = lambda path, duration=60: (b"y", 22050)
    fake_librosa.beat = types.SimpleNamespace(beat_track=beat_track)

    with patch.dict(sys.modules, {'librosa': fake_librosa}):
        result = detect(_mock_score_no_tempo(), str(audio))

    assert result == 60.0, f"slow ballad should not be corrected; got {result}"


def test_no_correction_above_threshold(tmp_path):
    """120 BPM is healthy — no alt-run should fire."""
    audio = tmp_path / "fake.mp3"
    audio.write_bytes(b"\x00")
    detect = _import_detect_tempo()

    fake_librosa = types.SimpleNamespace()
    call_count = {'n': 0}

    def beat_track(y=None, sr=None, start_bpm=None):
        call_count['n'] += 1
        return (120.0, None)

    fake_librosa.load = lambda path, duration=60: (b"y", 22050)
    fake_librosa.beat = types.SimpleNamespace(beat_track=beat_track)

    with patch.dict(sys.modules, {'librosa': fake_librosa}):
        result = detect(_mock_score_no_tempo(), str(audio))

    assert result == 120.0
    assert call_count['n'] == 1, "no alt-run should fire when initial tempo is healthy"


def test_alt_above_200_rejected(tmp_path):
    """If alt estimate is unrealistically high (>200), reject the correction."""
    audio = tmp_path / "fake.mp3"
    audio.write_bytes(b"\x00")
    detect = _import_detect_tempo()

    fake_librosa = types.SimpleNamespace()

    def beat_track(y=None, sr=None, start_bpm=None):
        if start_bpm == 140:
            return (210.0, None)  # too fast — reject
        return (50.0, None)

    fake_librosa.load = lambda path, duration=60: (b"y", 22050)
    fake_librosa.beat = types.SimpleNamespace(beat_track=beat_track)

    with patch.dict(sys.modules, {'librosa': fake_librosa}):
        result = detect(_mock_score_no_tempo(), str(audio))

    assert result == 50.0, f"unrealistic alt should be rejected; got {result}"


def test_no_audio_path_returns_default(tmp_path):
    """No audio file → default 120 BPM."""
    detect = _import_detect_tempo()
    result = detect(_mock_score_no_tempo(), None)
    assert result == 120.0
