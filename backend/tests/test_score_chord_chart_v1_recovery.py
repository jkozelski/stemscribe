"""Differential test: reconstructed audit/score_chord_chart.py (v1) must be
byte-identical in behavior to the verified compiled artifact it was recovered
from.

CONTEXT: the original untracked v1 scorer source was destroyed by a stray
`git clean` during a worktree mishap (2026-05-19). It was never committed, so
it is not git-recoverable. The compiled bytecode survived
(audit/__pycache__/score_chord_chart.cpython-311.pyc) and loads/executes
correctly under the project venv (Python 3.11). The source was reconstructed
from that bytecode. This test pins the reconstruction to the surviving
artifact as the behavioral oracle so any drift is caught.

The v2 honest scorer imports six pure primitives from v1
(NOTE_PC, PC_TO_NOTE, parse_chord, transpose, quality_family,
chord_to_pitch_classes) — those are the differential-tested critical path.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

AUDIT = Path(__file__).resolve().parents[2] / "audit"
PYC = AUDIT / "__pycache__" / "score_chord_chart.cpython-311.pyc"

pytestmark = pytest.mark.skipif(
    not PYC.exists(),
    reason="oracle .pyc not present (recovery artifact); diff-test skipped",
)


def _load_oracle():
    spec = importlib.util.spec_from_file_location("v1_oracle", str(PYC))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_reconstruction():
    sys.path.insert(0, str(AUDIT))
    import score_chord_chart as recon  # noqa: E402
    return recon


ROOTS = ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G",
         "G#", "Ab", "A", "A#", "Bb", "B", "Cb", "Fb", "E#", "B#"]
QUALS = ["", "m", "maj7", "M7", "maj9", "maj11", "maj13", "m7", "m9",
         "m11", "m6", "7", "9", "11", "13", "6", "sus2", "sus4", "sus",
         "add9", "dim", "dim7", "aug", "aug7", "m7b5", "5", "+", "°",
         "o", "7b9", "7#9", "9b13", "13b9", "min", "mmaj7", "6/9",
         "b5", "#5", "madd9"]
BASSES = ["", None, "/G", "/E", "/F#", "/Bb", "/C", "/Db", "/Cb"]


def _vocab():
    v = []
    for r in ROOTS:
        for q in QUALS:
            for b in BASSES:
                v.append(f"{r}{q}{b}" if b else f"{r}{q}")
    v += ["N", "", "x", "  C  ", "foo", "/G", "Cmaj7/G", "Am/F#",
          "Em6", "Dm7/G", "Bbm7b5/Db", "c", "cm", "CMAJ7", "Asus",
          "G#dim7/B"]
    return v


def _cmp(of, rf, *a):
    try:
        ov = ("OK", of(*a))
    except Exception as e:  # noqa: BLE001
        ov = ("ERR", type(e).__name__, str(e))
    try:
        rv = ("OK", rf(*a))
    except Exception as e:  # noqa: BLE001
        rv = ("ERR", type(e).__name__, str(e))
    return ov == rv, ov, rv


def test_primitives_byte_identical_to_oracle():
    O = _load_oracle()
    R = _load_reconstruction()
    assert O.NOTE_PC == R.NOTE_PC
    assert O.PC_TO_NOTE == R.PC_TO_NOTE

    mismatches = []
    for c in _vocab():
        for label, of, rf, args in [
            ("parse_chord", O.parse_chord, R.parse_chord, (c,)),
            ("chord_to_pitch_classes", O.chord_to_pitch_classes,
             R.chord_to_pitch_classes, (c,)),
            ("quality_family",
             lambda x: O.quality_family(O.parse_chord(x)[1]),
             lambda x: R.quality_family(R.parse_chord(x)[1]), (c,)),
        ]:
            ok, ov, rv = _cmp(of, rf, *args)
            if not ok:
                mismatches.append((label, c, ov, rv))
        for lv in ("root", "root_family", "root_quality", "full"):
            ok, ov, rv = _cmp(O.chord_to_key, R.chord_to_key, c, lv)
            if not ok:
                mismatches.append(("chord_to_key", (c, lv), ov, rv))
        for st in range(-15, 16):
            ok, ov, rv = _cmp(O.transpose, R.transpose, c, st)
            if not ok:
                mismatches.append(("transpose", (c, st), ov, rv))
    assert not mismatches, f"{len(mismatches)} primitive mismatches: {mismatches[:5]}"


def test_scoring_byte_identical_on_real_fixtures():
    O = _load_oracle()
    R = _load_reconstruction()
    import glob
    import json
    import random

    charts = []
    for p in glob.glob(str(AUDIT / "fixtures" / "prod_charts" / "*.json")):
        charts.append(json.load(open(p)))
    for p in ("/tmp/forensic/iml/chord_chart.json",
              "/tmp/forensic/pos/chord_chart.json",
              "/tmp/forensic/iml/job_metadata.json",
              "/tmp/forensic/pos/job_metadata.json"):
        if Path(p).exists():
            charts.append(json.load(open(p)))
    if not charts:
        pytest.skip("no chart fixtures available")

    random.seed(1)
    rts = ["C", "D", "E", "F", "G", "A", "B", "F#", "Bb", "Eb", "G#"]
    qs = ["", "m", "7", "m7", "maj7", "sus4", "sus2", "dim", "dim7",
          "aug", "6", "add9", "9", "11", "13", "m7b5", "5", "/G",
          "/B", "m6"]
    gts = []
    for _ in range(40):
        gts.append({
            "sections": [{"lines": [
                [f"{random.choice(rts)}{random.choice(qs)}"
                 for _ in range(random.randint(2, 7))]
                for _ in range(random.randint(1, 5))]}],
            "capo": random.choice([0, 0, 0, 2, -1, 5, 7, -3]),
        })
    for f in ("the-beatles__in-my-life.json",
              "bob-dylan__positively-4th-street.json"):
        p = AUDIT / "fixtures" / "ground_truth" / f
        if p.exists():
            gts.append(json.load(open(p)))

    for gt in gts:
        assert O.flatten_ground_truth(gt) == R.flatten_ground_truth(gt)
        for ch in charts:
            assert O.flatten_detector(ch) == R.flatten_detector(ch)
            assert O.score(gt, ch) == R.score(gt, ch)
