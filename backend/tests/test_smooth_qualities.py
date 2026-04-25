"""Tests for smooth_qualities — extension-promotion regression cover.

Regression coverage for the "smooth_qualities destroys 7ths" bug surfaced
by the 2026-04-25 audit:

  Detector inconsistently catches the b7 on a 7ths-heavy song. bar_grid
  ends up with e.g. 30 Cm + 5 Cm7. Pre-fix smooth_qualities snapped the
  minority Cm7 → Cm because Cm was the count-majority. Result: every
  bar wrong, 7 of 8 audited songs had ZERO 7ths in output.

  Post-fix logic: extensions are strictly more informative than triads,
  so a small but real extension presence promotes the whole root.
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


# ============ extension promotion ============

def test_minority_min7_promotes_dominant_min():
    """The Alright bug: 30 Cm + 5 Cm7 should produce all Cm7."""
    bars = [_bar(i, "C", "m") for i in range(30)] + [_bar(i + 30, "C", "m7") for i in range(5)]
    out = smooth_qualities(bars)
    qualities = {b["detector_quality"] for b in out}
    assert qualities == {"m7"}, f"expected all m7, got {qualities}"


def test_minority_maj7_promotes_dominant_major():
    """36 D + 4 Dmaj7 (10% extension rate) should produce all Dmaj7.

    Note: 4/54 (7.4%) does NOT promote — that's the noise-resistance
    threshold doing its job. This test uses 4/40 to land at the 10% floor.
    """
    bars = [_bar(i, "D", "") for i in range(36)] + [_bar(i + 36, "D", "maj7") for i in range(4)]
    out = smooth_qualities(bars)
    assert all(b["detector_quality"] == "maj7" for b in out)


def test_minority_dom7_promotes_dominant_major():
    """Blues: lots of A, a few A7. Should promote to A7."""
    bars = [_bar(i, "A", "") for i in range(20)] + [_bar(i + 20, "A", "7") for i in range(3)]
    out = smooth_qualities(bars)
    assert all(b["detector_quality"] == "7" for b in out)


def test_extension_below_count_threshold_does_not_promote():
    """One stray maj7 in a sea of major triads is noise, not signal."""
    bars = [_bar(i, "C", "") for i in range(30)] + [_bar(30, "C", "maj7")]
    out = smooth_qualities(bars)
    # Should NOT promote — only 1 occurrence, below extension_min_occurrences=3
    assert all(b["detector_quality"] == "" for b in out)


def test_extension_below_ratio_threshold_does_not_promote():
    """3 maj7 occurrences in 100 bars is too sparse to trust."""
    bars = [_bar(i, "C", "") for i in range(97)] + [_bar(i + 97, "C", "maj7") for i in range(3)]
    # 3/100 = 3% < 10% threshold → don't promote
    out = smooth_qualities(bars)
    assert all(b["detector_quality"] == "" for b in out)


def test_extension_at_threshold_promotes():
    """At exactly 10% with ≥3 occurrences, promotion fires."""
    bars = [_bar(i, "C", "") for i in range(27)] + [_bar(i + 27, "C", "maj7") for i in range(3)]
    # 3/30 = 10.0% → promote
    out = smooth_qualities(bars)
    assert all(b["detector_quality"] == "maj7" for b in out)


def test_competing_extensions_pick_most_common():
    """If a root has both m7 and m9 detected, prefer the more-common one."""
    bars = (
        [_bar(i, "C", "m") for i in range(15)]
        + [_bar(i + 15, "C", "m7") for i in range(8)]
        + [_bar(i + 23, "C", "m9") for i in range(3)]
    )
    out = smooth_qualities(bars)
    # m7 has 8 occurrences (more than m9's 3), so promote to m7
    assert all(b["detector_quality"] == "m7" for b in out)


# ============ legacy triad smoothing still works ============

def test_legacy_minor_triad_smoothing_still_works():
    """Pre-existing behavior: 25 Cm + 5 C smooths to all Cm (m wins majority)."""
    bars = [_bar(i, "C", "m") for i in range(25)] + [_bar(i + 25, "C", "") for i in range(5)]
    out = smooth_qualities(bars)
    assert all(b["detector_quality"] == "m" for b in out)


def test_no_dominant_quality_leaves_bars_unchanged():
    """50/50 split — neither quality clears majority. Bars unchanged."""
    bars = [_bar(i, "C", "m") for i in range(5)] + [_bar(i + 5, "C", "") for i in range(5)]
    out = smooth_qualities(bars)
    qualities = [b["detector_quality"] for b in out]
    # Original distribution preserved
    assert qualities.count("m") == 5
    assert qualities.count("") == 5


def test_different_roots_smoothed_independently():
    """Smoothing per-root, not globally."""
    bars = (
        [_bar(i, "C", "m") for i in range(10)]
        + [_bar(i + 10, "C", "m7") for i in range(4)]  # promote C → Cm7
        + [_bar(i + 14, "G", "") for i in range(20)]   # G stays as is
    )
    out = smooth_qualities(bars)
    c_qualities = {b["detector_quality"] for b in out if b["bass_root"] == "C"}
    g_qualities = {b["detector_quality"] for b in out if b["bass_root"] == "G"}
    assert c_qualities == {"m7"}, f"C should promote to m7, got {c_qualities}"
    assert g_qualities == {""}, f"G should stay major triad, got {g_qualities}"


def test_smoothed_bars_get_source_label():
    """Bars whose quality changed should be labeled '+smoothed' in source."""
    bars = [_bar(i, "C", "m") for i in range(10)] + [_bar(i + 10, "C", "m7") for i in range(4)]
    out = smooth_qualities(bars)
    smoothed = [b for b in out if "smoothed" in (b.get("source") or "")]
    # 10 bars with original quality "m" got snapped to "m7" — those should be labeled
    assert len(smoothed) == 10


def test_chord_string_rebuilt_after_promotion():
    """The 'chord' field should reflect the new quality after promotion."""
    bars = [_bar(i, "F", "") for i in range(15)] + [_bar(i + 15, "F", "maj7") for i in range(4)]
    out = smooth_qualities(bars)
    chords = {b["chord"] for b in out}
    assert chords == {"Fmaj7"}, f"expected all Fmaj7, got {chords}"


# ============ edge cases ============

def test_empty_bar_grid_returns_empty():
    assert smooth_qualities([]) == []


def test_bars_without_bass_root_left_alone():
    bars = [
        _bar(1, "C", "m"),
        {"bar": 2, "chord": "?", "bass_root": None, "detector_quality": ""},
        _bar(3, "C", "m"),
    ]
    out = smooth_qualities(bars)
    # Bar 2 has no bass root — passed through unchanged
    assert out[1]["bass_root"] is None
