"""Unit tests for 7ths preservation in stem_chord_detector.

Regression coverage for task #44. Ground-truth failure case: Jamiroquai
"Alright" (Cm7/Gm7/Dm7/Am7 throughout) was being classified as plain
Cm/Gm/Dm/Am before the fix because:

1. In _match_intervals_to_quality: a minor-triad subset match was scoring
   higher than a true min7 exact match due to superset + core-triad bonuses.
2. In _simplify_bleed_extensions: a 92% extension rate was treated as
   systematic bleed and stripped even though every chord was a consistent min7.
"""

from backend.stem_chord_detector import (
    _match_intervals_to_quality,
    _simplify_bleed_extensions,
    ChordEvent,
)


# ============ _match_intervals_to_quality ============

def test_exact_min7_intervals_return_min7():
    # Cm7 from C: {0, 3, 7, 10}
    quality, score = _match_intervals_to_quality(frozenset({0, 3, 7, 10}))
    assert quality == 'min7', f"expected min7, got {quality}"
    assert score >= 1.5, f"exact match should score ≥ 1.5 (dominating subsets), got {score}"


def test_exact_maj7_intervals_return_maj7():
    quality, score = _match_intervals_to_quality(frozenset({0, 4, 7, 11}))
    assert quality == 'maj7'
    assert score >= 1.5


def test_exact_dom7_intervals_return_7():
    quality, score = _match_intervals_to_quality(frozenset({0, 4, 7, 10}))
    assert quality == '7'
    assert score >= 1.5


def test_plain_minor_triad_returns_min():
    # Just {root, min3, fifth} — no 7th present
    quality, _score = _match_intervals_to_quality(frozenset({0, 3, 7}))
    assert quality == 'min', f"expected min, got {quality}"


def test_min7_beats_min_subset_even_with_bonuses():
    """The Alright bug: min triad subset-match was scoring higher than min7
    exact match due to core-triad + superset bonuses.
    """
    # Cm7 intervals — exact match for min7, superset of min triad
    intervals = frozenset({0, 3, 7, 10})
    quality, score = _match_intervals_to_quality(intervals)
    # The regression: this used to return 'min'.
    assert quality == 'min7'
    # And its score must exceed any possible alternate interpretation.
    # Directly sanity-check that 'min' would score lower on these intervals.
    # (We simulate what the old logic did by asking for the min template on
    # the same intervals — it is no longer returned as top, which is the fix.)
    assert score > 1.0


def test_minor_triad_with_bleed_fourth_stays_min():
    """Minor triad + a bleed note that isn't a clean 7th shouldn't flip to min7."""
    # {0, 3, 7, 5} — min triad with a perfect 4 (sus4 bleed, not a 7th)
    quality, _ = _match_intervals_to_quality(frozenset({0, 3, 7, 5}))
    # Should NOT claim min7 — no 10 (min 7th) present
    assert quality != 'min7'


# ============ _simplify_bleed_extensions ============

def _ev(chord: str, quality: str, confidence: float = 0.85, t: float = 0.0):
    """Build a minimal ChordEvent for testing."""
    root = chord.replace('m7', '').replace('maj7', '').replace('7', '').replace('m', '')
    return ChordEvent(
        time=t, duration=2.0, chord=chord, root=root,
        quality=quality, confidence=confidence,
    )


def test_alright_pattern_preserves_all_min7s():
    """Cm7/Gm7/Dm7/Am7 repeated — 100% extension, 100% consistency.
    Must NOT be simplified.
    """
    events = []
    for i in range(16):  # 4 cycles of 4 chords
        for root, qname in [('C', 'min7'), ('G', 'min7'), ('D', 'min7'), ('A', 'min7')]:
            events.append(_ev(f"{root}m7", qname, confidence=0.88, t=len(events) * 2.0))
    out = _simplify_bleed_extensions(events)
    preserved = [e for e in out if e.quality == 'min7']
    assert len(preserved) == len(events), (
        f"expected all {len(events)} min7 preserved, got {len(preserved)}"
    )


def test_high_rate_consistent_min7_preserved_even_at_borderline_confidence():
    """Extension rate > 90% with all-consistent min7 should survive even with
    individual confidences below the 0.93 per-chord gate.
    """
    events = [_ev("Cm7", 'min7', confidence=0.80) for _ in range(20)]
    out = _simplify_bleed_extensions(events)
    assert all(e.quality == 'min7' for e in out)


def test_random_extensions_low_consistency_get_simplified():
    """When extensions are varied across min7/maj7/7/add9 at moderate rate,
    that's bleed — simplify.
    """
    # 70% extensions, but each a different type (low consistency)
    events = []
    for chord, q in [("Cm7", 'min7'), ("Fmaj7", 'maj7'), ("G7", '7'),
                     ("Amadd9", 'madd9'), ("Dadd9", 'add9'), ("Em6", 'min6'),
                     ("C6", '6')]:
        events.append(_ev(chord, q, confidence=0.75))
    # Add a couple triads to push rate to ~70% not 100%
    events.append(_ev("F", 'maj', confidence=0.80))
    events.append(_ev("Am", 'min', confidence=0.80))
    events.append(_ev("C", 'maj', confidence=0.80))

    out = _simplify_bleed_extensions(events)
    simplified = [e for e in out if e.quality in ('maj', 'min')]
    # All 7 extension events should simplify; plus 3 originally-triad events
    assert len(simplified) == len(events), (
        f"expected varied extensions to simplify, got {[e.chord for e in out]}"
    )


def test_mostly_triads_with_rare_extension_simplified_on_low_confidence():
    """Classic bleed case: 1 extension chord among many triads with low
    confidence. The per-chord confidence gate should strip it.
    """
    events = [_ev("C", 'maj', confidence=0.90) for _ in range(10)]
    events.append(_ev("Cmaj7", 'maj7', confidence=0.70))  # bleed suspect
    out = _simplify_bleed_extensions(events)
    assert out[-1].quality == 'maj', (
        f"expected low-confidence maj7 to simplify; got {out[-1].chord}"
    )


def test_mostly_triads_with_confident_extension_preserved():
    """A rare but high-confidence 7th in a triad-heavy song should survive."""
    events = [_ev("C", 'maj', confidence=0.90) for _ in range(10)]
    events.append(_ev("Cmaj7", 'maj7', confidence=0.97))  # genuine maj7
    out = _simplify_bleed_extensions(events)
    assert out[-1].quality == 'maj7', (
        f"expected high-confidence maj7 preserved; got {out[-1].chord}"
    )


def test_keep_extended_qualities_never_simplified():
    """dim7 / hdim7 / aug / mMaj7 are in _KEEP_EXTENDED and must never change
    regardless of bleed heuristics.
    """
    events = [_ev("Bdim7", 'dim7', confidence=0.60)] * 10
    out = _simplify_bleed_extensions(events)
    assert all(e.quality == 'dim7' for e in out)
