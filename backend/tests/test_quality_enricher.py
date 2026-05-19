"""Contract tests for the quality-only enricher.

The enricher must NEVER change a bar's root. It only promotes a librosa plain
triad to ACE's richer quality when ACE agrees with librosa on the root AND
ACE's chord dominates the bar.
"""
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from processing.quality_enricher import enrich_qualities_from_ace  # noqa: E402


@dataclass
class FakeEvent:
    time: float
    duration: float
    chord: str
    root: str
    quality: str


def _bar(n, chord, s, e):
    return {"bar": n, "chord": chord, "start_time": s, "end_time": e}


def test_promotes_when_roots_agree_and_ace_dominates():
    grid = [_bar(1, "A", 0.0, 4.0)]
    ace = [FakeEvent(0.0, 4.0, "A7", "A", "7")]
    out, tel = enrich_qualities_from_ace(grid, "x.mp3", ace_events=ace)
    assert out[0]["chord"] == "A7"
    assert out[0]["enriched_from"] == "A"
    assert tel["promoted"] == 1


def test_never_changes_root_when_ace_root_differs():
    grid = [_bar(1, "A", 0.0, 4.0)]
    ace = [FakeEvent(0.0, 4.0, "Dm", "D", "m")]
    out, tel = enrich_qualities_from_ace(grid, "x.mp3", ace_events=ace)
    assert out[0]["chord"] == "A"  # unchanged — root disagreement
    assert tel["promoted"] == 0
    assert tel["root_disagree_skipped"] == 1


def test_does_not_downgrade_or_flip_family():
    # ACE says A minor-ish but librosa says A major triad: family flip, not a
    # color recovery. Keep librosa.
    grid = [_bar(1, "A", 0.0, 4.0)]
    ace = [FakeEvent(0.0, 4.0, "Am7", "A", "m7")]
    out, tel = enrich_qualities_from_ace(grid, "x.mp3", ace_events=ace)
    assert out[0]["chord"] == "A"
    assert tel["promoted"] == 0


def test_short_passing_ace_chord_does_not_flip_bar():
    # ACE flicks A7 for a fraction of the bar over a sustained A.
    grid = [_bar(1, "A", 0.0, 4.0)]
    ace = [
        FakeEvent(0.0, 3.4, "A", "A", ""),
        FakeEvent(3.4, 0.6, "A7", "A", "7"),
    ]
    out, tel = enrich_qualities_from_ace(grid, "x.mp3", ace_events=ace)
    assert out[0]["chord"] == "A"  # coverage gate keeps librosa
    assert tel["promoted"] == 0


def test_preserves_slash_bass_on_promotion():
    grid = [_bar(1, "A/C#", 0.0, 4.0)]
    ace = [FakeEvent(0.0, 4.0, "A7", "A", "7")]
    out, _ = enrich_qualities_from_ace(grid, "x.mp3", ace_events=ace)
    assert out[0]["chord"] == "A7/C#"


def test_empty_inputs_are_noops():
    out, tel = enrich_qualities_from_ace([], "x.mp3", ace_events=[])
    assert out == []
    out2, tel2 = enrich_qualities_from_ace(
        [_bar(1, "A", 0.0, 4.0)], "x.mp3", ace_events=[]
    )
    assert out2[0]["chord"] == "A"
    assert tel2["promoted"] == 0
