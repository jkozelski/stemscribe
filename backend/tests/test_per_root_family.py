"""Tests for per-root family-consistency override in _simplify_bleed_extensions.

Regression coverage for the 2026-04-25 PM fix:

  Songs with diverse chord palettes (Peg: Cmaj9 / Bm7b5 / E7 / Am7) have
  LOW global family consistency (multiple families per song) but HIGH
  per-root family consistency (each root is internally coherent).

  Pre-fix global-only logic stripped all extensions on Peg because the
  global family consistency was 46%. The per-root override trusts
  extensions on any root with ≥3 events that are ≥50% in one family,
  even when the song's global picture is varied.
"""

from backend.stem_chord_detector import _simplify_bleed_extensions, ChordEvent


def _ev(chord, quality, conf=0.85, t=0.0):
    root = chord.split('m')[0].split('7')[0].split('9')[0]
    if not root or root[0] not in 'ABCDEFG':
        root = chord[0]
    return ChordEvent(
        time=t, duration=2.0, chord=chord, root=root,
        quality=quality, confidence=conf,
    )


def test_peg_pattern_each_root_keeps_its_family():
    """Peg-like: 4 different roots, each with its own coherent family.
    Global family consistency is ~30% but each root is internally consistent.
    All extensions should survive.
    """
    events = (
        # C-root: all major family (3 maj9 + 3 6 = 6 events, 100% maj-family)
        [_ev("Cmaj9", "maj9") for _ in range(3)]
        + [_ev("C6", "6") for _ in range(3)]
        # B-root: all minor family
        + [_ev("Bm7", "min7") for _ in range(4)]
        + [_ev("Bm9", "min9") for _ in range(2)]
        # E-root: dominant family (real E7 in jazz)
        + [_ev("E7", "7") for _ in range(3)]
        + [_ev("E9", "9") for _ in range(2)]
        # A-root: minor family
        + [_ev("Am7", "min7") for _ in range(4)]
    )
    out = _simplify_bleed_extensions(events)
    # All events should survive untouched — each root has ≥3 events all
    # in the same family.
    out_qualities = [(e.root, e.quality) for e in out]
    in_qualities = [(e.root, e.quality) for e in events]
    assert out_qualities == in_qualities, (
        f"per-root family override should preserve all extensions; got changes:\n"
        f"  in:  {in_qualities[:5]}\n  out: {out_qualities[:5]}"
    )


def test_root_with_mixed_families_still_simplified():
    """A root that has events split 50/50 across families is bleed —
    fall back to global logic which (with low global consistency)
    should simplify."""
    events = (
        # C-root: 3 major + 3 minor = mixed families, fails per-root threshold
        [_ev("Cmaj7", "maj7") for _ in range(3)]
        + [_ev("Cm7", "min7") for _ in range(3)]
        # Add some triads to push global into bleed-suspect territory
        + [_ev("D", "maj") for _ in range(8)]
    )
    out = _simplify_bleed_extensions(events)
    # The C-root extensions (mixed families) should get simplified
    c_qualities = [e.quality for e in out if e.root == "C"]
    has_extension = any(q in {"maj7", "min7"} for q in c_qualities)
    assert not has_extension, (
        f"mixed-family C-root should simplify; got {c_qualities}"
    )


def test_root_with_too_few_events_falls_back_to_global():
    """A root with only 2 extension events doesn't establish a per-root
    pattern (threshold is ≥3). Falls back to global logic."""
    events = (
        # X-root: only 2 events, not enough to trust per-root
        [_ev("Xm7", "min7") for _ in range(2)]
        # Plus enough triads to look like a triad-dominant song
        + [_ev("D", "maj") for _ in range(10)]
        + [_ev("G", "maj") for _ in range(10)]
    )
    # In this scenario, ext_rate = 2/22 = 9% (below 60%) → systematic_bleed=False
    # So per-chord conf gate kicks in. With conf=0.85 < 0.93, simplifies.
    out = _simplify_bleed_extensions(events)
    x_qualities = [e.quality for e in out if e.root == "X"]
    # Should have been simplified by the conf gate
    assert all(q == "min" for q in x_qualities), (
        f"insufficient per-root data should fall through to confidence gate; got {x_qualities}"
    )


def test_alright_pattern_still_works_via_per_root():
    """The Apr 25 morning fix case still works: Alright's C-root has
    mostly minor-family extensions (Cmin7/Cmin9), per-root protects.
    """
    events = (
        [_ev("Cm7", "min7") for _ in range(8)]
        + [_ev("Cm9", "min9") for _ in range(15)]
        + [_ev("Gm7", "min7") for _ in range(10)]
        + [_ev("Gm9", "min9") for _ in range(15)]
        + [_ev("Dm7", "min7") for _ in range(10)]
        + [_ev("Am7", "min7") for _ in range(10)]
    )
    out = _simplify_bleed_extensions(events)
    out_qualities = {e.quality for e in out}
    # All extensions preserved — per-root families are clean
    assert "min" not in out_qualities, (
        f"Alright pattern should preserve all extensions; got {out_qualities}"
    )
    assert {"min7", "min9"} <= out_qualities


def test_single_root_song_works():
    """A song with only one root (like a vamp). All extensions same family."""
    events = [_ev("Cm7", "min7", conf=0.80) for _ in range(20)]
    out = _simplify_bleed_extensions(events)
    assert all(e.quality == "min7" for e in out)


def test_per_root_just_above_50pct_threshold_trusts():
    """4/7 in one family (~57%) clears the strict-majority threshold."""
    events = (
        [_ev("Cm7", "min7") for _ in range(4)]
        + [_ev("Cmaj7", "maj7") for _ in range(3)]
    )
    out = _simplify_bleed_extensions(events)
    # Some extensions should survive — minor-family is the majority
    out_minors = [e for e in out if e.quality == "min7"]
    assert len(out_minors) >= 1, (
        f"Majority family should be trusted; got {[(e.root, e.quality) for e in out]}"
    )
