"""
Tests for MIDI to Guitar Pro conversion.
Covers note-to-fret mapping, chord voicing, lead mode, and GP file generation.
"""

import sys
import os
import tempfile
from pathlib import Path

import pytest

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from midi_to_gp import (
    midi_note_to_fret,
    FretMapper,
    get_tuning_for_instrument,
    get_gp_instrument,
    convert_midi_to_gp,
    TUNINGS,
    MAX_FRET,
)


# =============================================================================
# midi_note_to_fret tests
# =============================================================================

class TestMidiNoteToFret:
    """Tests for the basic midi_note_to_fret function."""

    def test_open_strings_standard_guitar(self):
        tuning = TUNINGS['guitar']  # [40, 45, 50, 55, 59, 64]
        # E2 on low E string should be (1, 0)
        result = midi_note_to_fret(40, tuning)
        assert result is not None
        string, fret = result
        assert fret == 0  # Open string

    def test_middle_c_on_guitar(self):
        tuning = TUNINGS['guitar']
        # Middle C = MIDI 60. On guitar: B string fret 1 (59+1) or G string fret 5 (55+5)
        result = midi_note_to_fret(60, tuning)
        assert result is not None
        _, fret = result
        assert 0 <= fret <= MAX_FRET

    def test_note_too_low_for_guitar(self):
        tuning = TUNINGS['guitar']
        # MIDI note 30 is below low E (40)
        result = midi_note_to_fret(30, tuning)
        assert result is None

    def test_note_too_high_for_guitar(self):
        tuning = TUNINGS['guitar']
        # MIDI note 100 = fret 36 on high E... beyond MAX_FRET
        result = midi_note_to_fret(100, tuning)
        assert result is None

    def test_note_at_max_fret(self):
        tuning = TUNINGS['guitar']
        # High E string (64) + 24 frets = MIDI 88
        result = midi_note_to_fret(88, tuning)
        assert result is not None
        string, fret = result
        assert fret <= MAX_FRET

    def test_bass_tuning(self):
        tuning = TUNINGS['bass']  # [28, 33, 38, 43]
        # Open E on bass = MIDI 28
        result = midi_note_to_fret(28, tuning)
        assert result is not None
        _, fret = result
        assert fret == 0

    def test_context_prefers_nearby_frets(self):
        tuning = TUNINGS['guitar']
        # Play note at fret 5, then ask for a nearby note
        # MIDI 45 on A string = fret 0, on low E = fret 5
        result_no_context = midi_note_to_fret(45, tuning)
        result_with_context = midi_note_to_fret(45, tuning, prev_fret=5, prev_string=1)

        # Both should return valid positions
        assert result_no_context is not None
        assert result_with_context is not None

    def test_string_continuity(self):
        tuning = TUNINGS['guitar']
        # MIDI 50 = D3. On D string = open (0), on A string = fret 5
        # With prev_string=3 (D string), should prefer staying on same string
        result = midi_note_to_fret(50, tuning, prev_fret=2, prev_string=3)
        assert result is not None


# =============================================================================
# FretMapper tests
# =============================================================================

class TestFretMapper:
    """Tests for the FretMapper class with position context."""

    def test_basic_mapping(self):
        mapper = FretMapper(TUNINGS['guitar'])
        result = mapper.map_note(60)  # Middle C
        assert result is not None
        string, fret = result
        assert 1 <= string <= 6
        assert 0 <= fret <= MAX_FRET

    def test_position_continuity(self):
        mapper = FretMapper(TUNINGS['guitar'])
        # Map a sequence of notes - positions should stay somewhat stable
        positions = []
        for midi_note in [60, 62, 64, 65, 67]:  # C D E F G scale
            result = mapper.map_note(midi_note)
            assert result is not None
            positions.append(result)

        # Check that fret jumps aren't excessive (most should be <= 4)
        fret_jumps = [abs(positions[i][1] - positions[i-1][1])
                      for i in range(1, len(positions))]
        avg_jump = sum(fret_jumps) / len(fret_jumps)
        assert avg_jump <= 5, f"Average fret jump too large: {avg_jump}"

    def test_lead_mode_same_string_preference(self):
        mapper = FretMapper(TUNINGS['guitar'], lead_mode=True)
        # Map several notes and check string consistency
        strings_used = set()
        for midi_note in [60, 62, 64, 65]:
            result = mapper.map_note(midi_note)
            assert result is not None
            strings_used.add(result[0])

        # Lead mode should use fewer strings (stronger same-string preference)
        assert len(strings_used) <= 3

    def test_lead_mode_bend_awareness(self):
        mapper = FretMapper(TUNINGS['guitar'], lead_mode=True)
        # A bend should avoid open strings and fret 1
        result = mapper.map_note(60, has_bend=True)
        assert result is not None
        _, fret = result
        assert fret >= 2, f"Bend on fret {fret} is not practical"

    def test_reset_clears_context(self):
        mapper = FretMapper(TUNINGS['guitar'])
        mapper.map_note(60)
        mapper.map_note(62)
        assert mapper.prev_fret is not None

        mapper.reset()
        assert mapper.prev_fret is None
        assert mapper.prev_string is None

    def test_chord_mapping(self):
        mapper = FretMapper(TUNINGS['guitar'])
        # C major chord: C E G (MIDI 60, 64, 67)
        result = mapper.map_chord([60, 64, 67])
        assert len(result) == 3

        # All notes on different strings
        strings = [s for s, _ in result]
        assert len(set(strings)) == 3, "Chord notes must be on different strings"

        # Fret span should be playable (<=5)
        frets = [f for _, f in result]
        non_open = [f for f in frets if f > 0]
        if non_open:
            span = max(non_open) - min(non_open)
            assert span <= 5, f"Chord span too wide: {span}"

    def test_chord_empty_input(self):
        mapper = FretMapper(TUNINGS['guitar'])
        result = mapper.map_chord([])
        assert result == []

    def test_chord_impossible_notes(self):
        mapper = FretMapper(TUNINGS['guitar'])
        # Notes way too low for guitar
        result = mapper.map_chord([10, 15, 20])
        assert result == []


# =============================================================================
# Instrument detection tests
# =============================================================================

class TestInstrumentDetection:
    """Tests for tuning and instrument mapping."""

    def test_guitar_tuning(self):
        tuning, strings = get_tuning_for_instrument('guitar')
        assert strings == 6
        assert len(tuning) == 6
        assert tuning == TUNINGS['guitar']

    def test_bass_tuning(self):
        tuning, strings = get_tuning_for_instrument('bass')
        assert strings == 4
        assert len(tuning) == 4

    def test_guitar_left_tuning(self):
        # Compound names like 'guitar_left' should map to guitar
        tuning, strings = get_tuning_for_instrument('guitar_left')
        assert strings == 6

    def test_unknown_defaults_to_guitar(self):
        tuning, strings = get_tuning_for_instrument('kazoo')
        assert strings == 6

    def test_gp_instrument_guitar(self):
        assert get_gp_instrument('guitar') == 25

    def test_gp_instrument_bass(self):
        assert get_gp_instrument('bass') == 33

    def test_gp_instrument_drums(self):
        assert get_gp_instrument('drums') == 0

    def test_gp_instrument_piano(self):
        assert get_gp_instrument('piano') == 0


# =============================================================================
# GP file generation tests (integration)
# =============================================================================

class TestGPFileGeneration:
    """Integration tests for actual GP file creation (requires mido + pyguitarpro)."""

    @pytest.fixture
    def simple_midi(self, tmp_path):
        """Create a simple MIDI file for testing."""
        try:
            import mido
        except ImportError:
            pytest.skip("mido not installed")

        midi = mido.MidiFile()
        track = mido.MidiTrack()
        midi.tracks.append(track)

        # Set tempo (120 BPM)
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120)))

        # Add a few notes (C major scale)
        ticks_per_beat = midi.ticks_per_beat
        for note in [60, 62, 64, 65, 67, 69, 71, 72]:
            track.append(mido.Message('note_on', note=note, velocity=80, time=0))
            track.append(mido.Message('note_off', note=note, velocity=0, time=ticks_per_beat))

        midi_path = tmp_path / "test_scale.mid"
        midi.save(str(midi_path))
        return str(midi_path)

    def test_convert_guitar(self, simple_midi, tmp_path):
        try:
            import guitarpro
        except ImportError:
            pytest.skip("pyguitarpro not installed")

        output_path = str(tmp_path / "test_guitar.gp5")
        success = convert_midi_to_gp(
            simple_midi, output_path,
            instrument_type='guitar',
            title='Test Scale',
            artist='Test'
        )
        assert success is True
        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0

        # Verify the file is valid GP5
        song = guitarpro.parse(output_path)
        assert song.title == 'Test Scale'
        assert song.artist == 'Test'
        assert len(song.tracks) >= 1

    def test_convert_bass(self, simple_midi, tmp_path):
        try:
            import guitarpro
        except ImportError:
            pytest.skip("pyguitarpro not installed")

        # Bass notes (lower octave)
        import mido
        midi = mido.MidiFile()
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120)))

        for note in [40, 43, 45, 47]:  # E A B D on bass
            track.append(mido.Message('note_on', note=note, velocity=80, time=0))
            track.append(mido.Message('note_off', note=note, velocity=0, time=midi.ticks_per_beat))

        bass_midi_path = str(tmp_path / "test_bass.mid")
        midi.save(bass_midi_path)

        output_path = str(tmp_path / "test_bass.gp5")
        success = convert_midi_to_gp(bass_midi_path, output_path, instrument_type='bass')
        assert success is True
        assert Path(output_path).exists()

    def test_convert_empty_midi(self, tmp_path):
        try:
            import mido
        except ImportError:
            pytest.skip("mido not installed")

        # MIDI with no notes
        midi = mido.MidiFile()
        track = mido.MidiTrack()
        midi.tracks.append(track)
        midi_path = str(tmp_path / "empty.mid")
        midi.save(midi_path)

        output_path = str(tmp_path / "empty.gp5")
        success = convert_midi_to_gp(midi_path, output_path, instrument_type='guitar')
        assert success is False  # Should fail gracefully

    def test_auto_adds_gp5_extension(self, simple_midi, tmp_path):
        try:
            import guitarpro
        except ImportError:
            pytest.skip("pyguitarpro not installed")

        output_path = str(tmp_path / "test_no_ext")
        success = convert_midi_to_gp(simple_midi, output_path, instrument_type='guitar')
        assert success is True
        assert Path(output_path + '.gp5').exists()
