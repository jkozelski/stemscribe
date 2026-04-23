"""Unit tests for midi_chord_detector.

Tests exercise the pure-Python scoring functions with synthetic pitch-class
weight vectors and MidiNote lists. No audio, no Basic Pitch call; this
suite runs the same way on the VPS as on a dev laptop.
"""
import pytest

from midi_chord_detector import (
    ROOT_PC,
    MidiNote,
    _bar_pc_weights,
    _select_chord,
    _bar_windows_from_grid,
    detect_chords_from_midi,
)


def weights(pc_dict):
    """Build a 12-length weight vector from a {'C': 0.6, ...} shorthand."""
    w = [0.0] * 12
    for name, val in pc_dict.items():
        w[ROOT_PC[name]] = float(val)
    return w


# --- _select_chord: the core template-match ---

def test_cm7_clean():
    q, suffix, score, _ = _select_chord(
        weights({'C': 0.6, 'Eb': 0.6, 'G': 0.6, 'Bb': 0.6}),
        ROOT_PC['C'], 0.05,
    )
    assert q == 'm7'
    assert suffix == 'm7'
    assert score >= 0.99


def test_fmaj7_clean():
    q, suffix, _, _ = _select_chord(
        weights({'F': 0.6, 'A': 0.6, 'C': 0.6, 'E': 0.6}),
        ROOT_PC['F'], 0.05,
    )
    assert q == 'maj7'
    assert suffix == 'maj7'


def test_cm_no_seventh():
    """Plain Cm triad — no Bb — must resolve to 'min', not 'm7'."""
    q, suffix, _, _ = _select_chord(
        weights({'C': 0.6, 'Eb': 0.6, 'G': 0.6}),
        ROOT_PC['C'], 0.05,
    )
    assert q == 'min'
    assert suffix == 'm'


def test_cmaj_with_passing_b7_stays_plain_major():
    """Bb at 2% of peak is below the 5% floor — should not promote to C7."""
    q, suffix, _, _ = _select_chord(
        weights({'C': 0.6, 'E': 0.6, 'G': 0.6, 'Bb': 0.02}),
        ROOT_PC['C'], 0.05,
    )
    assert q == 'maj'
    assert suffix == ''


def test_cmaj_with_real_b7_promotes_to_c7():
    """Bb strongly present — the whole point of the rebuild."""
    q, suffix, _, _ = _select_chord(
        weights({'C': 0.6, 'E': 0.6, 'G': 0.6, 'Bb': 0.5}),
        ROOT_PC['C'], 0.05,
    )
    assert q == '7'
    assert suffix == '7'


def test_messy_extras_still_picks_cm7():
    """Spurious F# and A strong enough to register as extras — penalties
    should not knock m7 off the top."""
    q, suffix, _, _ = _select_chord(
        weights({
            'C': 0.7, 'Eb': 0.7, 'G': 0.7, 'Bb': 0.7,
            'F#': 0.2, 'A': 0.15,
        }),
        ROOT_PC['C'], 0.05,
    )
    assert q == 'm7'


def test_am7_jeffs_case():
    """A, C, E, G — the canonical Am7 voicing."""
    q, suffix, _, _ = _select_chord(
        weights({'A': 0.6, 'C': 0.6, 'E': 0.6, 'G': 0.6}),
        ROOT_PC['A'], 0.05,
    )
    assert q == 'm7'
    assert suffix == 'm7'


def test_power_chord_pure():
    """Just root + 5 — the power-chord case Jeff explicitly asked for."""
    q, suffix, _, _ = _select_chord(
        weights({'C': 0.8, 'G': 0.8}),
        ROOT_PC['C'], 0.05,
    )
    assert q == '5'
    assert suffix == '5'


def test_power_chord_with_suppressed_third():
    """Third at 3% of peak is below floor — still a power chord."""
    q, suffix, _, _ = _select_chord(
        weights({'C': 0.8, 'G': 0.8, 'E': 0.03}),
        ROOT_PC['C'], 0.05,
    )
    assert q == '5'


def test_power_chord_defeated_by_real_third():
    """If the third is actually played (≥ floor), chord becomes a triad."""
    q, suffix, _, _ = _select_chord(
        weights({'C': 0.8, 'G': 0.8, 'E': 0.15}),
        ROOT_PC['C'], 0.05,
    )
    assert q == 'maj'
    assert suffix == ''


def test_dmaj_triad():
    q, suffix, _, _ = _select_chord(
        weights({'D': 0.6, 'F#': 0.6, 'A': 0.6}),
        ROOT_PC['D'], 0.05,
    )
    assert q == 'maj'


def test_bm7b5_half_dim():
    q, suffix, _, _ = _select_chord(
        weights({'B': 0.6, 'D': 0.6, 'F': 0.6, 'A': 0.6}),
        ROOT_PC['B'], 0.05,
    )
    assert q == 'm7b5'


def test_all_zero_returns_zero_score():
    """Empty bar — defensive check that scoring doesn't crash on zeros."""
    q, _, score, _ = _select_chord([0.0] * 12, ROOT_PC['C'], 0.05)
    assert score == 0.0
    # Any template is "tied" at 0; the tie-breaker picks the longest, but
    # the detector-level code treats this as N.C. (see detect_chords tests).


# --- _bar_pc_weights: the windowing primitive ---

def test_bar_pc_weights_note_inside():
    notes = [MidiNote(start=0.0, end=1.0, pitch=60, amplitude=0.5, stem='g')]
    w = _bar_pc_weights(notes, 0.0, 1.0)
    # C=0; amplitude 0.5 × 1.0s overlap = 0.5
    assert w[0] == pytest.approx(0.5)
    assert all(x == 0 for i, x in enumerate(w) if i != 0)


def test_bar_pc_weights_note_outside():
    notes = [MidiNote(start=2.0, end=3.0, pitch=64, amplitude=0.5, stem='g')]
    w = _bar_pc_weights(notes, 0.0, 1.0)
    assert all(x == 0 for x in w)


def test_bar_pc_weights_partial_overlap():
    notes = [MidiNote(start=0.5, end=1.5, pitch=60, amplitude=1.0, stem='g')]
    w = _bar_pc_weights(notes, 0.0, 1.0)
    # 0.5s overlap × amplitude 1.0
    assert w[0] == pytest.approx(0.5)


def test_bar_pc_weights_multiple_same_pc():
    notes = [
        MidiNote(start=0.0, end=1.0, pitch=60, amplitude=0.5, stem='g'),   # C4
        MidiNote(start=0.5, end=1.5, pitch=72, amplitude=0.4, stem='p'),   # C5
    ]
    w = _bar_pc_weights(notes, 0.0, 1.0)
    # C4 contributes 0.5 × 1.0 = 0.5; C5 contributes 0.4 × 0.5 = 0.2. Both are C.
    assert w[0] == pytest.approx(0.7)


# --- detect_chords_from_midi integration (audio-free via _injected_notes) ---

def _single_bar_grid(t0=0.0, t1=2.0):
    """4 bars of equal length. Detector requires >=4 bars to proceed."""
    bar_len = t1 - t0
    return {
        'downbeat_times': [t0, t0 + bar_len, t0 + 2 * bar_len, t0 + 3 * bar_len],
        'song_duration_sec': t0 + 4 * bar_len,
    }


def _bar_root(bar, root, source='pyin'):
    return {'bar': bar, 'root': root, 'source': source, 'confidence': 0.9}


def _chord_across_bar(pitches_with_amp, start, end, stem='guitar'):
    return [MidiNote(start=start, end=end, pitch=p, amplitude=a, stem=stem)
            for p, a in pitches_with_amp]


def test_integration_am7_uses_bass_root():
    grid = _single_bar_grid(0.0, 2.0)
    bass_roots = [_bar_root(1, 'A')]
    # Bar 1 [0, 2]: Am7 tones — A3 (57), C4 (60), E4 (64), G4 (67)
    notes = _chord_across_bar([(57, 0.6), (60, 0.6), (64, 0.6), (67, 0.6)], 0.0, 2.0)
    result = detect_chords_from_midi(
        guitar_path=None, piano_path=None, bass_path=None,
        grid=grid, bass_roots=bass_roots, _injected_notes=notes,
    )
    assert result.basic_pitch_ok
    assert len(result.chord_progression.chords) == 4
    # Bar 1 gets the injected Am7; bars 2-4 are silent → N.C.
    first = result.chord_progression.chords[0]
    assert first.chord == 'Am7'
    assert first.root == 'A'
    assert first.quality == 'm7'


def test_integration_silent_bar_becomes_nc():
    grid = _single_bar_grid(0.0, 2.0)
    bass_roots = [_bar_root(1, 'C')]
    # No notes anywhere.
    result = detect_chords_from_midi(
        guitar_path=None, piano_path=None, bass_path=None,
        grid=grid, bass_roots=bass_roots, _injected_notes=[],
    )
    # Empty note list → basic_pitch_ok = False, empty progression.
    assert not result.basic_pitch_ok
    assert result.chord_progression.chords == []


def test_integration_missing_bass_root_falls_back_to_dominant_pc():
    """Bar has clean Cm7 notes but no bass_roots entry — detector should
    still produce Cm7 (dominant-pc root fallback picks C since the C
    pitch-class has the most weight in this synthetic case)."""
    grid = _single_bar_grid(0.0, 2.0)
    notes = _chord_across_bar(
        # Emphasize C with a longer/louder note so it wins dominant-pc
        [(48, 0.9), (48, 0.9), (51, 0.6), (55, 0.6), (58, 0.6)],
        0.0, 2.0,
    )
    result = detect_chords_from_midi(
        guitar_path=None, piano_path=None, bass_path=None,
        grid=grid, bass_roots=[], _injected_notes=notes,
    )
    assert result.basic_pitch_ok
    first = result.chord_progression.chords[0]
    # Root recovered as C; chord is Cm7.
    assert first.root == 'C'
    assert first.chord == 'Cm7'


def test_integration_grid_too_small_aborts():
    grid = {'downbeat_times': [0.0, 1.0], 'song_duration_sec': 2.0}
    result = detect_chords_from_midi(
        guitar_path=None, piano_path=None, bass_path=None,
        grid=grid, bass_roots=[], _injected_notes=[],
    )
    assert not result.basic_pitch_ok
    assert result.chord_progression.chords == []


# --- _bar_windows_from_grid primitive ---

def test_bar_windows_basic():
    grid = {'downbeat_times': [0.0, 1.0, 2.5, 4.0], 'song_duration_sec': 5.5}
    windows = _bar_windows_from_grid(grid)
    assert [w[0] for w in windows] == [1, 2, 3, 4]
    assert windows[0] == (1, 0.0, 1.0)
    assert windows[3] == (4, 4.0, 5.5)


def test_bar_windows_missing_song_end_extrapolates():
    grid = {'downbeat_times': [0.0, 2.0, 4.0]}
    windows = _bar_windows_from_grid(grid)
    # Last bar extrapolates from the average bar length
    assert windows[-1] == (3, 4.0, 6.0)
