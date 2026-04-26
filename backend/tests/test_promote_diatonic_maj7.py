"""Tests for promote_diatonic_maj7 — key-aware maj7 promotion.

The Black Cow case: detector outputs A9 on the I chord throughout when
the song is actually Amaj7. This pass promotes I and IV degree dom-7
chords to maj-7 in major keys, while protecting:
  - minor-key songs (Cosmic Girl in F#, Aja in Bm, Alright in Gm)
  - blues progressions (V also dom-7)
  - V chords in major keys (could legitimately be V7)
  - bars without dom-quality
"""

from backend.processing.bass_root_extraction import promote_diatonic_maj7


def _bar(bar_num, root, quality, source="bass+detector"):
    return {
        "bar": bar_num,
        "chord": (root + quality) if quality else root,
        "bass_root": root,
        "detector_quality": quality,
        "source": source,
    }


# ============ Black Cow case ============

def test_a9_on_I_in_a_major_promotes_to_amaj9():
    """The Black Cow bug: A9 on the I chord of A major → Amaj9.
    Note: 9 promotes to maj9 (not maj7) — preserves the 9th degree.
    """
    bars = [
        _bar(1, "A", "9"),
        _bar(2, "A", "9"),
        _bar(3, "A", "9"),
    ]
    out = promote_diatonic_maj7(bars, "A")
    assert all(b["detector_quality"] == "maj9" for b in out)
    assert all(b["chord"] == "Amaj9" for b in out)


def test_d9_on_IV_in_a_major_promotes_to_dmaj9():
    """IV chord (D in A major) also promotes; 9 → maj9."""
    bars = [_bar(1, "D", "9")]
    out = promote_diatonic_maj7(bars, "A")
    assert out[0]["detector_quality"] == "maj9"
    assert out[0]["chord"] == "Dmaj9"


def test_dom7_promotes_to_maj7():
    bars = [_bar(1, "A", "7"), _bar(2, "D", "7")]
    out = promote_diatonic_maj7(bars, "A")
    assert out[0]["chord"] == "Amaj7"
    assert out[1]["chord"] == "Dmaj7"


def test_dom13_promotes_to_maj13():
    bars = [_bar(1, "A", "13")]
    out = promote_diatonic_maj7(bars, "A")
    assert out[0]["chord"] == "Amaj13"


# ============ Safety: don't break working songs ============

def test_minor_key_no_promotion():
    """Songs in minor keys (most of our working set: Aja, Alright,
    Cosmic Girl, etc.) should be untouched."""
    bars = [_bar(1, "A", "9"), _bar(2, "D", "7")]
    for minor_key in ["Am", "Amin", "A minor", "A#m", "Bm", "F#m", "Cm"]:
        out = promote_diatonic_maj7(bars, minor_key)
        assert out == bars, f"minor key {minor_key} should not promote"


def test_blues_pattern_preserves_dom7():
    """If V is also dom-7, treat as blues, don't promote anything.
    A blues in A: I7=A7, IV7=D7, V7=E7. All preserved.
    """
    bars = [
        _bar(1, "A", "7"),  # I7
        _bar(2, "D", "7"),  # IV7
        _bar(3, "E", "7"),  # V7 — triggers blues protection
        _bar(4, "A", "7"),
    ]
    out = promote_diatonic_maj7(bars, "A")
    qualities = [b["detector_quality"] for b in out]
    assert qualities == ["7", "7", "7", "7"], (
        f"blues pattern should be untouched; got {qualities}"
    )


def test_v_chord_dom7_preserved_in_non_blues():
    """V can be dom7 even in non-blues — but if I and IV are NOT also
    dom7, blues check passes and only dom7s on I/IV get promoted (which
    are zero here).
    """
    bars = [
        _bar(1, "A", "maj7"),  # I as maj7 (already correct)
        _bar(2, "E", "7"),     # V as dom7 — common in major-key pop
    ]
    out = promote_diatonic_maj7(bars, "A")
    # E should stay as E7, not be promoted (it's V, not I/IV)
    assert out[1]["detector_quality"] == "7"


def test_plain_triad_not_promoted():
    """Bars without dom-quality stay untouched."""
    bars = [_bar(1, "A", ""), _bar(2, "D", "m"), _bar(3, "A", "min7")]
    out = promote_diatonic_maj7(bars, "A")
    assert out == bars


def test_chord_not_on_I_or_IV_not_promoted():
    """In A major: B, C, F#, etc. are not I or IV. dom7s on those
    degrees stay unchanged."""
    bars = [_bar(1, "B", "9"), _bar(2, "C", "7"), _bar(3, "F#", "9")]
    out = promote_diatonic_maj7(bars, "A")
    qualities = [b["detector_quality"] for b in out]
    assert qualities == ["9", "7", "9"], (
        f"non-I/IV chords should be untouched; got {qualities}"
    )


# ============ Cohort regression: working songs unaffected ============

def test_cosmic_girl_pattern_unchanged():
    """Cosmic Girl in F# major (key='F#') has F#min7 / Em7 / C#m7 —
    all minor extensions. Wait — Cosmic Girl was detected as F# major
    in our v4 results. The minor 7 chords (m7) aren't dom-quality, so
    won't promote. Same root family family-aware preservation untouched.
    """
    bars = [
        _bar(1, "F#", "min7"),
        _bar(2, "E", "min7"),
        _bar(3, "C#", "min7"),
    ]
    out = promote_diatonic_maj7(bars, "F#")
    assert out == bars


def test_alright_pattern_unchanged():
    """Alright detected key was 'G' (probably G or Gm). With minor
    chords (Cmin7 etc.), no promotion fires regardless of key polarity."""
    bars = [
        _bar(1, "C", "min7"),
        _bar(2, "G", "min7"),
        _bar(3, "D", "min7"),
        _bar(4, "A", "min9"),
    ]
    # Even if key is interpreted as G major, none of these are dom-7
    out = promote_diatonic_maj7(bars, "G")
    assert out == bars


# ============ Edge cases ============

def test_empty_grid():
    assert promote_diatonic_maj7([], "A") == []


def test_unknown_key_no_op():
    bars = [_bar(1, "A", "9")]
    out = promote_diatonic_maj7(bars, "Unknown")
    assert out == bars


def test_none_key_no_op():
    bars = [_bar(1, "A", "9")]
    out = promote_diatonic_maj7(bars, None)
    assert out == bars


def test_enharmonic_keys_handled():
    """Bb is the same key as A#. F# is the same as Gb."""
    bars = [_bar(1, "Bb", "9")]
    out = promote_diatonic_maj7(bars, "Bb")
    assert out[0]["detector_quality"] == "maj9"


def test_slash_chord_preserved_after_promotion():
    """A9/G should become Amaj9/G after promotion (9 → maj9)."""
    bars = [{
        "bar": 1, "chord": "A9/G", "bass_root": "A",
        "detector_quality": "9", "source": "bass+detector",
    }]
    out = promote_diatonic_maj7(bars, "A")
    assert out[0]["chord"] == "Amaj9/G"
    assert out[0]["detector_quality"] == "maj9"


def test_source_field_marked_when_promoted():
    """Bars whose quality flipped get '+maj7promoted' tag in source."""
    bars = [_bar(1, "A", "9", source="bass+detector")]
    out = promote_diatonic_maj7(bars, "A")
    assert "maj7promoted" in out[0]["source"]
