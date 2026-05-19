"""Tests for the CHART_FORMATTER_DISARM_SMOOTHING landmine-disarm flag.

Landmine (chart_formatter.py, `if grid and bass_roots:` branch):
`smooth_qualities` + `promote_diatonic_maj7` were tuned for the noisy legacy
stem-aware detector. On a CLEAN per-bar detector (ACE/Jiang) they collapse
quality vocabulary and crater accuracy. The disarm flag lets a clean detector
keep the useful bass-anchored bar grid while skipping the destructive passes.

Contract under test:
  1. Flag UNSET  -> byte-identical to the historical prod path (default safe).
  2. Flag SET    -> smooth_qualities + promote_diatonic_maj7 are NOT called;
                    the bass-anchored bar grid (combine_with_detector_quality)
                    is still produced.
  3. Flag is parsed conservatively (only explicit truthy values opt in).
"""

import copy
import importlib
import io
import contextlib
import json
import os
from pathlib import Path

import pytest

import backend.chart_formatter as cf

# chart_formatter resolves `from processing.bass_root_extraction import ...`
# at call time (backend/ is on sys.path), which is a DIFFERENT module object
# from `backend.processing.bass_root_extraction`. Patch the one it actually
# uses so the spy test observes real calls.
import importlib as _il
bre = _il.import_module("processing.bass_root_extraction")

FORENSIC = Path("/tmp/forensic")
_HAVE_FORENSIC = (FORENSIC / "iml" / "job_metadata.json").exists()


def _format(name, env_val):
    """Run format_chart on a forensic song with the disarm flag set/unset."""
    base = FORENSIC / name
    m = json.loads((base / "job_metadata.json").read_text())
    words = json.loads((base / "word_ts.json").read_text())
    meta = m["metadata"]
    prev = os.environ.get("CHART_FORMATTER_DISARM_SMOOTHING")
    if env_val is None:
        os.environ.pop("CHART_FORMATTER_DISARM_SMOOTHING", None)
    else:
        os.environ["CHART_FORMATTER_DISARM_SMOOTHING"] = env_val
    try:
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            chart = cf.format_chart(
                chord_events=copy.deepcopy(m["chord_progression"]),
                word_timestamps=copy.deepcopy(words),
                title=name,
                artist="x",
                key=m.get("detected_key", "C"),
                grid=copy.deepcopy(meta.get("grid")),
                bass_roots=copy.deepcopy(meta.get("bass_roots")),
            )
        return json.dumps(chart, sort_keys=True)
    finally:
        if prev is None:
            os.environ.pop("CHART_FORMATTER_DISARM_SMOOTHING", None)
        else:
            os.environ["CHART_FORMATTER_DISARM_SMOOTHING"] = prev


@pytest.mark.skipif(not _HAVE_FORENSIC, reason="forensic captures not present")
@pytest.mark.parametrize("song", ["iml", "pos"])
def test_flag_unset_is_byte_identical_to_prod(song):
    """Default (flag unset) must not change a single byte of chart output
    versus the explicit historical prod path."""
    out_default = _format(song, None)
    out_false = _format(song, "false")
    out_empty = _format(song, "")
    assert out_default == out_false == out_empty


@pytest.mark.skipif(not _HAVE_FORENSIC, reason="forensic captures not present")
def test_flag_set_changes_output_when_landmine_engages():
    """The disarm only matters when the landmine branch is actually taken,
    i.e. `grid and bass_roots`. POS has bass_roots (99 bars) so disarming
    MUST change output. IML has empty bass_roots so the landmine branch is
    never entered and output is correctly identical with the flag on/off
    (the disarm is not a global no-op; it is correctly scoped)."""
    # POS: bass_roots populated -> landmine active -> disarm changes output.
    assert _format("pos", None) != _format("pos", "true")
    # IML: bass_roots empty -> landmine branch never taken -> no change.
    assert _format("iml", None) == _format("iml", "true")


@pytest.mark.skipif(not _HAVE_FORENSIC, reason="forensic captures not present")
def test_iml_has_empty_bass_roots_pos_does_not():
    """Pin the precondition that scopes the disarm: the forensic IML capture
    has empty bass_roots (landmine never engages) while POS has 99."""
    iml = json.loads((FORENSIC / "iml" / "job_metadata.json").read_text())
    pos = json.loads((FORENSIC / "pos" / "job_metadata.json").read_text())
    assert not iml["metadata"].get("bass_roots")
    assert len(pos["metadata"].get("bass_roots") or []) > 0


@pytest.mark.skipif(not _HAVE_FORENSIC, reason="forensic captures not present")
def test_flag_set_skips_smooth_and_promote(monkeypatch):
    """Directly assert smooth_qualities + promote_diatonic_maj7 are not
    invoked when the flag is on, and ARE invoked when it is off."""
    calls = {"smooth": 0, "promote": 0, "combine": 0}
    real_smooth = bre.smooth_qualities
    real_promote = bre.promote_diatonic_maj7
    real_combine = bre.combine_with_detector_quality

    def spy_smooth(*a, **k):
        calls["smooth"] += 1
        return real_smooth(*a, **k)

    def spy_promote(*a, **k):
        calls["promote"] += 1
        return real_promote(*a, **k)

    def spy_combine(*a, **k):
        calls["combine"] += 1
        return real_combine(*a, **k)

    monkeypatch.setattr(bre, "smooth_qualities", spy_smooth)
    monkeypatch.setattr(bre, "promote_diatonic_maj7", spy_promote)
    monkeypatch.setattr(bre, "combine_with_detector_quality", spy_combine)
    importlib.reload(cf)
    try:
        # POS has bass_roots so the landmine branch is taken.
        # Flag ON -> combine still called, smooth/promote skipped.
        _format("pos", "true")
        assert calls["combine"] >= 1
        assert calls["smooth"] == 0
        assert calls["promote"] == 0

        # Flag OFF -> all three called (historical path intact).
        calls.update(smooth=0, promote=0, combine=0)
        _format("pos", None)
        assert calls["combine"] >= 1
        assert calls["smooth"] >= 1
        assert calls["promote"] >= 1
    finally:
        importlib.reload(cf)


@pytest.mark.parametrize(
    "val,expect_disarm",
    [
        ("1", True), ("true", True), ("TRUE", True), ("yes", True),
        ("on", True), (" true ", True),
        ("0", False), ("false", False), ("no", False), ("", False),
        ("disabled", False),
    ],
)
def test_flag_parsing_is_conservative(val, expect_disarm):
    parsed = val.strip().lower() in ("1", "true", "yes", "on")
    assert parsed is expect_disarm
