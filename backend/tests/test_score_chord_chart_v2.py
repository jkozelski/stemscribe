"""Tests for the honest chord-chart scorer (audit/score_chord_chart_v2.py).

These lock in the BEHAVIOUR that makes v2 honest, so a future change that
re-introduces the v1 bag-of-chords pathology fails CI:

  * a perfect chart scores 1.0 on every axis (mechanics floor)
  * "In My Life" served prod chart scores LOW (musician verdict bar), in
    contrast to v1's 0.919
  * a detector that strips all extensions is caught by AXIS 3, not hidden
  * detector ranking is sane (oracle > librosa) and librosa never tops
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
AUDIT = REPO / "audit"
sys.path.insert(0, str(AUDIT))

from score_chord_chart_v2 import (  # noqa: E402
    score_v2,
    gt_bar_sequence,
)
from score_chord_chart import score as score_v1, parse_chord  # noqa: E402

GT_DIR = AUDIT / "fixtures" / "ground_truth"
PROD_DIR = AUDIT / "fixtures" / "prod_charts"

IML_GT = GT_DIR / "the-beatles__in-my-life.json"
IML_PROD = PROD_DIR / "the-beatles__in-my-life__fb8a175e.json"
POS_GT = GT_DIR / "bob-dylan__positively-4th-street.json"
POS_PROD = PROD_DIR / "bob-dylan__positively-4th-street__1433ba27.json"


def _chart_from_seq(seq):
    return {"chord_progression": [{"chord": c} for c in seq]}


@pytest.mark.parametrize("gt_path", [IML_GT, POS_GT])
def test_perfect_chart_scores_one_on_every_axis(gt_path):
    """Mechanics floor: GT scored against itself must be 1.0 everywhere.
    A scorer that cannot give a perfect chart 1.0 is broken."""
    gt = json.loads(gt_path.read_text())
    seq = gt_bar_sequence(gt)
    r = score_v2(gt, _chart_from_seq(seq))
    assert r["axes"]["1_root"]["f1"] == 1.0
    assert r["axes"]["2_placement"]["strict_bar"]["f1"] == 1.0
    assert r["axes"]["2_placement"]["hold_invariant"]["f1"] == 1.0
    fl = r["axes"]["3_flavor"]["weighted_flavor"]
    assert fl is None or fl == 1.0
    assert r["composite"]["composite"] == 1.0


def test_in_my_life_rated_low_unlike_v1():
    """The keystone test. v1 root = 0.919 (misleading). v2 must rate this
    chart LOW because it drops every dom7 and mis-places most bars — the
    documented musician verdict that it is 'basically root notes'."""
    gt = json.loads(IML_GT.read_text())
    chart = json.loads(IML_PROD.read_text())

    v1 = score_v1(gt, chart)["scores"]
    assert v1["root"]["f1"] > 0.9, "v1 baseline should still be the misleading ~0.919"

    r = score_v2(gt, chart)
    # composite must be LOW — a musician calls this chart bad
    assert r["composite"]["composite"] < 0.55, r["composite"]
    # the specific failure: every dominant-7 dropped
    dom7 = r["axes"]["3_flavor"]["per_class"]["triad_dom7"]
    assert dom7["graded"] >= 4
    assert dom7["accuracy"] == 0.0, "GT A7/B7 should all be dropped to plain triads"
    # placement is genuinely bad, not just mis-anchored
    assert r["axes"]["2_placement"]["strict_bar"]["f1"] < 0.45
    # and v2 composite must be far below v1's cited number
    assert r["composite"]["composite"] < v1["root"]["f1"] - 0.4


def test_extension_stripping_is_caught_by_flavor_axis():
    """A detector with perfect roots & placement but no extensions must score
    ~1.0 root, ~1.0 placement, but LOW flavor — the exact failure the v1 bag
    `root` level hid. Uses Don't Know Why (the diagnostic extension axis)."""
    gt = json.loads((GT_DIR / "norah-jones__dont-know-why.json").read_text())
    seq = gt_bar_sequence(gt)

    def strip(c):
        r, q, _ = parse_chord(c)
        ql = (q or "").lower()
        keep_min = ql.startswith("m") and not ql.startswith("maj")
        return (r or c) + ("m" if keep_min else "")

    degraded = [strip(c) for c in seq]
    r = score_v2(gt, _chart_from_seq(degraded))

    assert r["axes"]["1_root"]["f1"] == 1.0
    assert r["axes"]["2_placement"]["strict_bar"]["f1"] == 1.0
    # flavor must collapse — this is the headline metric the old set couldn't move
    assert r["axes"]["3_flavor"]["weighted_flavor"] < 0.30
    assert r["composite"]["composite"] < 0.65  # not a "good chart"


def test_detector_ranking_oracle_beats_librosa():
    """Sanity ranking on the served prod charts: a perfect (oracle) chart
    must outrank the served librosa-derived prod chart on composite. A scorer
    that ranks the served chart >= oracle is broken."""
    for gt_path, prod_path in [(IML_GT, IML_PROD), (POS_GT, POS_PROD)]:
        gt = json.loads(gt_path.read_text())
        seq = gt_bar_sequence(gt)
        oracle = score_v2(gt, _chart_from_seq(seq))["composite"]["composite"]
        served = score_v2(gt, json.loads(prod_path.read_text()))["composite"]["composite"]
        assert oracle == 1.0
        assert served < oracle, (gt_path.name, served, oracle)


def test_composite_cannot_be_carried_by_one_axis():
    """Anti-bag-of-chords safeguard: a chart strong on root but broken on
    placement+flavor must NOT get a high composite (the v1 0.919 pathology)."""
    gt = json.loads(IML_GT.read_text())
    r = score_v2(gt, json.loads(IML_PROD.read_text()))
    parts = r["composite"]["parts"]
    # root is decent...
    assert parts["root"] > 0.75
    # ...but the composite is dragged down by the weak axes
    assert r["composite"]["composite"] < parts["root"] - 0.3
    assert r["composite"]["weakest_axis"] <= min(parts.values()) + 1e-9
