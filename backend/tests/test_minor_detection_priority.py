"""Tests for the m3-detection priority pass in smooth_qualities.

Regression coverage for the "Alright G9 instead of Gm9" bug surfaced by
the 2026-04-25 audit:

  Detector inconsistently catches the b3 (m3 interval). On Alright's G-root
  bars: 8 Gm9 + 25 G9 in detector output. Old logic (extension promotion)
  picked G9 because it had more events. But m3 is asymmetrically reliable —
  when the detector outputs a minor-family quality, the m3 was actually
  heard. When it outputs a major/dominant quality, the m3 may simply not
  have made it into pitch_classes. So even minority m3 evidence wins.
"""

from backend.processing.bass_root_extraction import smooth_qualities


def _bar(bar_num, root, quality):
    return {
        "bar": bar_num,
        "chord": root + quality,
        "bass_root": root,
        "detector_quality": quality,
        "source": "bass+detector",
    }


def test_minority_minor_overrides_majority_dominant():
    """The Alright bug: 8 Gm9 + 25 G9 should produce all Gm9.
    m3 detected 8 times = asymmetric positive evidence for minor.
    """
    bars = [_bar(i, "G", "m9") for i in range(8)] + [_bar(i + 8, "G", "9") for i in range(25)]
    out = smooth_qualities(bars)
    qualities = {b["detector_quality"] for b in out}
    assert qualities == {"m9"}, f"minority m3 should win; got {qualities}"


def test_minority_minor_overrides_majority_major7():
    """4 Gm7 + 30 Gmaj7: m3 detected 4 times, snap all to Gm7."""
    bars = [_bar(i, "G", "m7") for i in range(4)] + [_bar(i + 4, "G", "maj7") for i in range(30)]
    out = smooth_qualities(bars)
    assert all(b["detector_quality"] == "m7" for b in out)


def test_three_min_events_meets_threshold():
    """Exactly 3 minor events trigger the priority pass."""
    bars = [_bar(i, "C", "m7") for i in range(3)] + [_bar(i + 3, "C", "") for i in range(50)]
    out = smooth_qualities(bars)
    assert all(b["detector_quality"] == "m7" for b in out)


def test_two_min_events_below_threshold_falls_through():
    """Only 2 minor events — too sparse to trust as the song's quality.
    Falls through to legacy logic.
    """
    bars = [_bar(i, "C", "m7") for i in range(2)] + [_bar(i + 2, "C", "") for i in range(50)]
    out = smooth_qualities(bars)
    # Pass 0 doesn't fire (< 3 minor events). Pass 1 (extension promotion)
    # also doesn't fire (m7=2 below the 3-event extension threshold).
    # Pass 2 (legacy majority): "" has 50/52 = 96% → snap all to "".
    assert all(b["detector_quality"] == "" for b in out)


def test_minor_priority_picks_dominant_minor_extension():
    """When multiple minor extensions present, pick the most common one."""
    bars = (
        [_bar(i, "G", "m9") for i in range(10)]
        + [_bar(i + 10, "G", "m7") for i in range(4)]
        + [_bar(i + 14, "G", "9") for i in range(20)]
    )
    out = smooth_qualities(bars)
    # m9 wins (10 > 4 within minor family)
    assert all(b["detector_quality"] == "m9" for b in out)


def test_minor_priority_falls_back_to_plain_min_if_no_extension():
    """If only plain Cm minor triad is detected (no m7/m9 extension),
    snap all to Cm."""
    bars = [_bar(i, "C", "m") for i in range(5)] + [_bar(i + 5, "C", "") for i in range(20)]
    out = smooth_qualities(bars)
    assert all(b["detector_quality"] == "m" for b in out)


def test_independent_roots_get_independent_treatment():
    """G can promote to minor while D stays major if D has no m3 evidence."""
    bars = (
        [_bar(i, "G", "m7") for i in range(5)]   # G → minor evidence
        + [_bar(i + 5, "G", "9") for i in range(10)]
        + [_bar(i + 15, "D", "") for i in range(20)]  # D → no minor evidence
    )
    out = smooth_qualities(bars)
    g_qualities = {b["detector_quality"] for b in out if b["bass_root"] == "G"}
    d_qualities = {b["detector_quality"] for b in out if b["bass_root"] == "D"}
    assert g_qualities == {"m7"}
    assert d_qualities == {""}


def test_alright_full_pattern():
    """Full Alright simulation: each of 4 roots has min7/min9 minorities
    plus G9/A9/etc. dominants. All 4 should snap to minor extensions."""
    bars = (
        # C: 6 Cm9 + 6 Cm7 + 2 C
        [_bar(i, "C", "m9") for i in range(6)]
        + [_bar(i + 6, "C", "m7") for i in range(6)]
        + [_bar(i + 12, "C", "") for i in range(2)]
        # G: 4 Gm9 + 28 G9
        + [_bar(i + 14, "G", "m9") for i in range(4)]
        + [_bar(i + 18, "G", "9") for i in range(28)]
        # D: 16 Dmin7 (already correct)
        + [_bar(i + 46, "D", "min7") for i in range(16)]
        # A: 5 Am7 + 26 A9
        + [_bar(i + 62, "A", "m7") for i in range(5)]
        + [_bar(i + 67, "A", "9") for i in range(26)]
    )
    out = smooth_qualities(bars)
    by_root = {}
    for b in out:
        by_root.setdefault(b["bass_root"], set()).add(b["detector_quality"])
    # All 4 roots should now be minor-family
    assert all(b["bass_root"] not in ("",) for b in out)
    for root, qs in by_root.items():
        for q in qs:
            assert q.startswith("m"), (
                f"root {root} should be minor-family after pass 0; got {q}"
            )
