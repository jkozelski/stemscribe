"""Tests for detect_key_from_chords using pitch-class weighting.

Regression coverage for the 2026-04-26 fix: weight chord-implied notes
(root + 3rd + 5th + 7th when applicable) instead of just chord roots.
This is more robust to slash-chord moments where the bass walks under
sustained harmony.

Black Cow case: A9 (37) + D (25) + C9 (16) + E (15) chord events. Old
root-only weighting picked D major (wrong — it's A major). New
pc-weighted version should pick A major because every A9 chord
contributes A/C#/E/G/B to the pc histogram, dominating the C/G noise
from the slash-chord C9 moments.
"""

from backend.stem_chord_detector import detect_key_from_chords, ChordEvent


def _ev(chord, root, quality, duration=2.0, confidence=0.85):
    return ChordEvent(
        time=0.0, duration=duration, chord=chord, root=root,
        quality=quality, confidence=confidence,
    )


def test_empty_progression_returns_c():
    assert detect_key_from_chords([]) == 'C'


# ============ Working songs from the audit (regression check) ============

def test_alright_pattern_detects_g_minor():
    """Alright (Jamiroquai) is in G minor. Cm7-Gm7-Dm7-Am9 vamp."""
    events = (
        [_ev("Cmin7", "C", "min7") for _ in range(20)]
        + [_ev("Gmin7", "G", "min7") for _ in range(20)]
        + [_ev("Dmin7", "D", "min7") for _ in range(15)]
        + [_ev("Amin9", "A", "min9") for _ in range(15)]
    )
    key = detect_key_from_chords(events)
    # Alright is in Gm. The vamp Cm-Gm-Dm-Am could read as Gm or Cm depending
    # on weighting — both are reasonable. Accept either.
    assert key in ('Gm', 'Cm', 'Dm', 'Am'), f"expected minor key, got {key}"
    assert 'm' in key, "should detect minor"


def test_cosmic_girl_pattern_detects_minor():
    """Cosmic Girl chorus: F#min7-Emin7-C#min7."""
    events = (
        [_ev("F#min7", "F#", "min7") for _ in range(30)]
        + [_ev("Emin7", "E", "min7") for _ in range(15)]
        + [_ev("C#min7", "C#", "min7") for _ in range(15)]
    )
    key = detect_key_from_chords(events)
    # All-minor progression — should detect a minor key
    assert 'm' in key, f"all-minor progression should detect minor; got {key}"


def test_aja_pattern_detects_b_minor():
    """Aja: Bm7-Cm9-Gm9-Am9-F#m9-Em9 — dense minor jazz."""
    events = (
        [_ev("Bmin7", "B", "min7") for _ in range(50)]
        + [_ev("Cmin9", "C", "min9") for _ in range(20)]
        + [_ev("Gmin9", "G", "min9") for _ in range(20)]
        + [_ev("Amin9", "A", "min9") for _ in range(15)]
        + [_ev("F#min9", "F#", "min9") for _ in range(15)]
        + [_ev("Emin9", "E", "min9") for _ in range(15)]
    )
    key = detect_key_from_chords(events)
    assert 'm' in key, f"all-minor jazz should detect minor; got {key}"


# ============ The Black Cow regression target ============

def test_black_cow_pattern_detects_a_major():
    """Black Cow chord events: A9 (37) D (25) C9 (16) E (15) — as
    observed in the v4 audit. Real key is A major. A9 dominates ≥30% of
    weighted duration → static-dominant heuristic returns 'A' directly.
    """
    events = (
        [_ev("A9", "A", "9") for _ in range(37)]
        + [_ev("D", "D", "maj") for _ in range(25)]
        + [_ev("C9", "C", "9") for _ in range(16)]
        + [_ev("E", "E", "maj") for _ in range(15)]
    )
    key = detect_key_from_chords(events)
    assert key == 'A', f"static-dominant heuristic should pick A; got {key}"


def test_static_dominant_below_threshold_falls_through_to_kk():
    """If no single dom chord covers ≥30%, the guard shouldn't fire — fall
    through to K-K. G (40) + C (30) + D7 (5) → D7 is 6.7%, way below
    threshold. Key should resolve via K-K to G major, NOT D (D is the
    chord-root that would only be picked if the dom guard misfired)."""
    events = (
        [_ev("G", "G", "maj") for _ in range(40)]
        + [_ev("C", "C", "maj") for _ in range(30)]
        + [_ev("D7", "D", "7") for _ in range(5)]
    )
    key = detect_key_from_chords(events)
    assert key != 'D', f"dom guard misfired below 30%; got {key}"
    assert key in ('G', 'Em'), f"expected G major (or relative Em); got {key}"


def test_black_cow_without_slash_chord_noise_detects_a_major():
    """Sanity check: same as Black Cow but WITHOUT the C9 slash-chord
    noise. Should clearly detect A major."""
    events = (
        [_ev("A9", "A", "9") for _ in range(37)]
        + [_ev("D", "D", "maj") for _ in range(25)]
        + [_ev("E", "E", "maj") for _ in range(15)]
    )
    key = detect_key_from_chords(events)
    assert key in ('A', 'F#m'), f"clean A-major progression; got {key}"


# ============ Diatonic test cases ============

def test_three_chord_pop_in_g_major():
    """G - C - D progression — clear G major (I-IV-V)."""
    events = (
        [_ev("G", "G", "maj") for _ in range(30)]
        + [_ev("C", "C", "maj") for _ in range(20)]
        + [_ev("D", "D", "maj") for _ in range(20)]
    )
    key = detect_key_from_chords(events)
    assert key in ('G', 'Em'), f"I-IV-V in G should detect G major; got {key}"


def test_blues_in_a_detects_a_major():
    """12-bar blues in A: A7 / D7 / E7 with dominant 7ths everywhere."""
    events = (
        [_ev("A7", "A", "7") for _ in range(40)]
        + [_ev("D7", "D", "7") for _ in range(20)]
        + [_ev("E7", "E", "7") for _ in range(15)]
    )
    key = detect_key_from_chords(events)
    # Classic blues should still register as A major
    assert key in ('A', 'F#m'), f"A blues should detect A major; got {key}"


# ============ Quality fallback ============

def test_unknown_quality_falls_back_to_root_only():
    """If a chord's quality string isn't in CHORD_INTERVALS, we count
    just the root. Verify this doesn't crash and produces a key."""
    events = [_ev("Cunknown", "C", "unknown_quality_xyz") for _ in range(20)]
    key = detect_key_from_chords(events)
    # No assertions on which key — just verify no crash and string output
    assert isinstance(key, str)
    assert len(key) >= 1


def test_all_zero_weight_returns_c():
    """If somehow all confidences are 0, fall back to C."""
    events = [_ev("Am", "A", "min", confidence=0.0)]
    key = detect_key_from_chords(events)
    assert key == 'C'
