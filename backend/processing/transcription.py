"""
Transcription functions — MIDI transcription, quantization, chord detection, MusicXML.

Contains the main transcribe_to_midi routing function which dispatches to the
appropriate transcriber (neural, Basic Pitch, melody, etc.) per stem type.
"""

import os
import json
import logging
from pathlib import Path

from models.job import ProcessingJob, OUTPUT_DIR, save_job_to_disk, save_job_checkpoint
from processing.utils import check_stem_has_content

logger = logging.getLogger(__name__)

# ============ CONDITIONAL IMPORTS ============

try:
    from drum_transcriber import transcribe_drums_to_midi
    DRUM_TRANSCRIBER_AVAILABLE = True
except ImportError:
    DRUM_TRANSCRIBER_AVAILABLE = False

try:
    from transcriber_enhanced import EnhancedTranscriber, transcribe_with_enhanced
    ENHANCED_TRANSCRIBER_AVAILABLE = True
except ImportError:
    ENHANCED_TRANSCRIBER_AVAILABLE = False

try:
    from drum_transcriber_v2 import EnhancedDrumTranscriber, transcribe_drums_to_midi as transcribe_drums_v2
    DRUM_TRANSCRIBER_V2_AVAILABLE = True
except ImportError:
    DRUM_TRANSCRIBER_V2_AVAILABLE = False

try:
    from oaf_drum_transcriber import OaFDrumTranscriber, transcribe_drums, OAF_AVAILABLE
    OAF_DRUM_TRANSCRIBER_AVAILABLE = True
except ImportError:
    OAF_DRUM_TRANSCRIBER_AVAILABLE = False
    OAF_AVAILABLE = False

try:
    from drum_nn_transcriber import (
        NeuralDrumTranscriber, transcribe_drums_nn,
        DRUM_NN_MODEL_AVAILABLE, is_available as drum_nn_is_available
    )
except ImportError:
    DRUM_NN_MODEL_AVAILABLE = False

try:
    from melody_transcriber import MelodyExtractor, transcribe_melody
    MELODY_TRANSCRIBER_AVAILABLE = True
except ImportError:
    MELODY_TRANSCRIBER_AVAILABLE = False

try:
    from guitar_nn_transcriber import (
        GuitarNNTranscriber, transcribe_guitar_nn,
        GUITAR_NN_MODEL_AVAILABLE, is_available as guitar_nn_is_available
    )
except ImportError:
    GUITAR_NN_MODEL_AVAILABLE = False

try:
    from bass_nn_transcriber import (
        BassNNTranscriber, transcribe_bass_nn,
        BASS_NN_MODEL_AVAILABLE, is_available as bass_nn_v3_is_available
    )
except ImportError:
    BASS_NN_MODEL_AVAILABLE = False

try:
    from guitar_tab_transcriber import (
        GuitarTabTranscriber, transcribe_guitar_tab,
        GUITAR_TAB_MODEL_AVAILABLE, is_available as guitar_tab_is_available
    )
except ImportError:
    GUITAR_TAB_MODEL_AVAILABLE = False

try:
    from bass_transcriber import (
        BassTranscriber, transcribe_bass,
        BASS_MODEL_AVAILABLE, is_available as bass_nn_is_available
    )
except ImportError:
    BASS_MODEL_AVAILABLE = False

try:
    from piano_transcriber import (
        PianoTranscriber, transcribe_piano,
        PIANO_MODEL_AVAILABLE, is_available as piano_nn_is_available
    )
except ImportError:
    PIANO_MODEL_AVAILABLE = False

CHORD_DETECTOR_VERSION = None
try:
    from chord_detector_v8 import ChordDetector, detect_chords
    CHORD_DETECTOR_AVAILABLE = True
    CHORD_DETECTOR_VERSION = "v8"
except ImportError:
    try:
        from chord_detector_v7 import ChordDetector, detect_chords
        CHORD_DETECTOR_AVAILABLE = True
        CHORD_DETECTOR_VERSION = "v7"
    except ImportError:
        try:
            from chord_detector import ChordDetector, detect_chords
            CHORD_DETECTOR_AVAILABLE = True
            CHORD_DETECTOR_VERSION = "basic"
        except ImportError:
            CHORD_DETECTOR_AVAILABLE = False

try:
    from midi_to_gp import convert_midi_to_gp, convert_job_midis_to_gp
    GP_CONVERTER_AVAILABLE = True
except ImportError:
    GP_CONVERTER_AVAILABLE = False

try:
    from midi_to_notation import midi_to_musicxml as _notation_convert, MUSIC21_AVAILABLE
    NOTATION_CONVERTER_AVAILABLE = True
except ImportError:
    NOTATION_CONVERTER_AVAILABLE = False
    MUSIC21_AVAILABLE = False


# ============ TRANSCRIPTION FUNCTIONS ============

def quantize_midi(midi_path: str, grid_size: float = 0.125, min_note_length: float = 0.05):
    """
    Quantize MIDI notes to a rhythmic grid for tighter transcription.

    Args:
        midi_path: Path to the MIDI file
        grid_size: Quantization grid in beats (0.125 = 16th notes, 0.25 = 8th notes)
        min_note_length: Minimum note length in seconds (removes tiny noise notes)

    Returns:
        True if quantization succeeded, False otherwise
    """
    try:
        import pretty_midi

        # Load the MIDI file
        midi = pretty_midi.PrettyMIDI(midi_path)

        if not midi.instruments:
            logger.warning(f"No instruments in MIDI file: {midi_path}")
            return False

        # Get tempo for beat calculations (default 120 BPM if not found)
        tempo = 120.0
        if midi.get_tempo_changes()[1].size > 0:
            tempo = midi.get_tempo_changes()[1][0]

        # Convert grid_size from beats to seconds
        beat_duration = 60.0 / tempo
        grid_seconds = grid_size * beat_duration

        total_notes_original = 0
        total_notes_quantized = 0
        notes_removed = 0

        for instrument in midi.instruments:
            original_notes = instrument.notes.copy()
            total_notes_original += len(original_notes)
            quantized_notes = []

            for note in original_notes:
                # Skip very short notes (likely noise)
                note_length = note.end - note.start
                if note_length < min_note_length:
                    notes_removed += 1
                    continue

                # Quantize start time to grid
                quantized_start = round(note.start / grid_seconds) * grid_seconds

                # Quantize end time to grid (minimum 1 grid unit)
                quantized_end = round(note.end / grid_seconds) * grid_seconds
                if quantized_end <= quantized_start:
                    quantized_end = quantized_start + grid_seconds

                # Create quantized note
                note.start = quantized_start
                note.end = quantized_end
                quantized_notes.append(note)

            # Replace notes with quantized versions
            instrument.notes = quantized_notes
            total_notes_quantized += len(quantized_notes)

        # Save the quantized MIDI
        midi.write(midi_path)

        logger.info(f"🎹 Quantized MIDI: {total_notes_original} → {total_notes_quantized} notes "
                   f"(removed {notes_removed} tiny notes, grid={grid_size} beats)")
        return True

    except ImportError:
        logger.warning("pretty_midi not installed - skipping quantization")
        return False
    except Exception as e:
        logger.error(f"MIDI quantization failed: {e}")
        return False


def transcribe_to_midi(job: ProcessingJob, quantize: bool = True, grid_size: float = 0.125,
                       detect_articulations: bool = True):
    """
    Enhanced MIDI transcription with articulation detection and improved drum handling.

    Uses enhanced transcriber for melodic instruments (guitar, bass, piano, vocals)
    and drum transcriber v2 for drums. Falls back to Basic Pitch when modules unavailable.

    Args:
        job: The processing job
        quantize: Whether to quantize MIDI to grid (default True)
        grid_size: Quantization grid in beats (0.125 = 16th notes, 0.25 = 8th notes)
        detect_articulations: Whether to detect bends, slides, hammer-ons (default True)
    """
    import traceback

    job.stage = 'Transcribing to MIDI (enhanced)'
    job.progress = 60
    logger.info("Starting enhanced MIDI transcription")

    midi_output_dir = OUTPUT_DIR / job.job_id / 'midi'
    midi_output_dir.mkdir(parents=True, exist_ok=True)

    total_stems = len(job.stems)
    successful = 0

    # These stems don't work well with transcription (mixed content)
    SKIP_STEMS = {'other', 'instrumental'}

    # Initialize enhanced transcriber if available
    enhanced_transcriber = None
    if ENHANCED_TRANSCRIBER_AVAILABLE:
        try:
            enhanced_transcriber = EnhancedTranscriber()
            logger.info("🎸 Using enhanced transcriber with articulation detection")
        except Exception as e:
            logger.warning(f"Could not initialize enhanced transcriber: {e}")

    for idx, (stem_name, stem_path) in enumerate(job.stems.items()):
        try:
            # Skip mixed stems
            if stem_name.lower() in SKIP_STEMS:
                logger.info(f"⏭️ Skipping {stem_name} (mixed content)")
                continue

            # Verify the stem file exists
            if not Path(stem_path).exists():
                logger.error(f"Stem file not found: {stem_path}")
                continue

            # ==== DRUMS: Neural CRNN -> OaF -> v2 spectral -> v1 ====
            if stem_name.lower() == 'drums' or '_drums' in stem_name.lower():
                drum_midi_path = midi_output_dir / f"{stem_name}_transcribed.mid"

                # HIGHEST PRIORITY: Our trained CRNN drum model (E-GMD, 8 classes)
                if DRUM_NN_MODEL_AVAILABLE:
                    logger.info(f"🥁 Transcribing {stem_name} with neural drum model (CRNN)...")
                    job.stage = f'Transcribing {stem_name} (neural drum model)'

                    try:
                        drum_transcriber = NeuralDrumTranscriber()
                        drum_result = drum_transcriber.transcribe(
                            audio_path=stem_path,
                            output_dir=str(midi_output_dir),
                            sensitivity=0.5,
                            tempo_hint=job.metadata.get('tempo'),
                        )

                        if (drum_result.midi_path
                                and Path(drum_result.midi_path).exists()
                                and drum_result.quality_score > 0.3):
                            job.midi_files[stem_name] = drum_result.midi_path
                            job.transcription_quality[stem_name] = drum_result.quality_score
                            job.transcription_mode[stem_name] = 'drum_nn'

                            logger.info(
                                f"✓ Drum MIDI (neural): "
                                f"{drum_result.total_hits} hits, "
                                f"types: {list(drum_result.hits_by_type.keys())}, "
                                f"tempo: {drum_result.tempo:.0f}, "
                                f"quality: {drum_result.quality_score:.2f}"
                            )
                            successful += 1
                            job.progress = 60 + int((idx + 1) / total_stems * 35)
                            continue
                        else:
                            quality = drum_result.quality_score if drum_result else 0
                            logger.info(
                                f"  Neural drum quality too low ({quality:.2f}), "
                                f"falling back to OaF/spectral"
                            )
                    except Exception as e:
                        logger.warning(f"Neural drum model failed: {e}")
                        # Fall through to OaF / spectral fallbacks

                # Try OaF neural network transcriber (fallback)
                if OAF_DRUM_TRANSCRIBER_AVAILABLE:
                    logger.info(f"🥁 Transcribing {stem_name} with OaF drum transcriber...")
                    job.stage = f'Transcribing {stem_name} (neural network)'

                    try:
                        result = transcribe_drums(
                            audio_path=stem_path,
                            output_path=str(drum_midi_path),
                            sensitivity=0.5,
                            prefer_neural=True  # Use OaF when available
                        )

                        if result.get('success') and result.get('total_hits', 0) > 0:
                            job.midi_files[stem_name] = str(drum_midi_path)
                            # Estimate quality from hit count distribution
                            quality = min(1.0, 0.5 + len(result.get('hits_by_type', {})) * 0.1)
                            job.transcription_quality[stem_name] = quality
                            logger.info(f"✓ Drum MIDI ({result.get('backend', 'unknown')}): "
                                       f"{result['total_hits']} hits, "
                                       f"types: {list(result.get('hits_by_type', {}).keys())}")
                            successful += 1
                            continue
                        else:
                            logger.warning(f"OaF returned no hits, trying v2")
                    except Exception as e:
                        logger.warning(f"OaF drum transcriber failed: {e}")

                # Try v2 spectral transcriber (ghost notes, cymbal detection)
                if DRUM_TRANSCRIBER_V2_AVAILABLE:
                    logger.info(f"🥁 Transcribing {stem_name} with spectral drum transcriber v2...")
                    job.stage = f'Transcribing {stem_name} (ghost notes, cymbals)'

                    try:
                        drum_transcriber = EnhancedDrumTranscriber()
                        stats = drum_transcriber.transcribe(
                            audio_path=stem_path,
                            output_path=str(drum_midi_path),
                            detect_ghost_notes=True,
                            preserve_dynamics=True
                        )

                        if stats.total_hits > 0:
                            job.midi_files[stem_name] = str(drum_midi_path)
                            job.transcription_quality[stem_name] = stats.quality_score
                            logger.info(f"✓ Drum MIDI: {stats.total_hits} hits, "
                                       f"ghost: {stats.ghost_notes}, "
                                       f"quality: {stats.quality_score:.2f}")
                            successful += 1
                        else:
                            logger.warning(f"No drum hits detected in {stem_name}")
                    except Exception as e:
                        logger.warning(f"Drum v2 failed, trying v1: {e}")
                        # Fall back to v1
                        if DRUM_TRANSCRIBER_AVAILABLE:
                            if transcribe_drums_to_midi(stem_path, str(drum_midi_path)):
                                job.midi_files[stem_name] = str(drum_midi_path)
                                successful += 1
                    continue

                elif DRUM_TRANSCRIBER_AVAILABLE:
                    # Fall back to v1 drum transcriber
                    logger.info(f"🥁 Transcribing {stem_name} with drum transcriber v1...")
                    job.stage = f'Transcribing {stem_name}'
                    drum_midi_path = midi_output_dir / f"{stem_name}_transcribed.mid"
                    if transcribe_drums_to_midi(stem_path, str(drum_midi_path)):
                        job.midi_files[stem_name] = str(drum_midi_path)
                        logger.info(f"✓ Created drum MIDI: {drum_midi_path}")
                        successful += 1
                    continue
                else:
                    logger.info(f"⏭️ Skipping {stem_name} (no drum transcriber available)")
                    continue

            # ==== MELODIC INSTRUMENTS ====
            stem_type = stem_name.lower().split('_')[0]  # Handle guitar_left, bass_right, etc.

            # ---- HIGHEST PRIORITY: Guitar v3 NN model ----
            if stem_type == 'guitar' and GUITAR_NN_MODEL_AVAILABLE:
                logger.info(f"Transcribing {stem_name} with guitar v3 NN model...")
                job.stage = f'Transcribing {stem_name} (guitar v3 NN)'

                try:
                    guitar_nn = GuitarNNTranscriber()
                    guitar_result = guitar_nn.transcribe(
                        audio_path=stem_path,
                        output_dir=str(midi_output_dir),
                        tempo_hint=job.metadata.get('tempo'),
                    )

                    if (guitar_result.midi_path
                            and Path(guitar_result.midi_path).exists()
                            and guitar_result.quality_score > 0.3):
                        job.midi_files[stem_name] = guitar_result.midi_path
                        job.transcription_quality[stem_name] = guitar_result.quality_score
                        job.transcription_mode[stem_name] = 'guitar_v3_nn'

                        logger.info(
                            f"Guitar v3 MIDI for {stem_name}: "
                            f"{guitar_result.num_notes} notes, "
                            f"range={guitar_result.pitch_range}, "
                            f"polyphony={guitar_result.polyphony_avg:.1f}, "
                            f"quality: {guitar_result.quality_score:.2f}"
                        )
                        successful += 1
                        job.progress = 60 + int((idx + 1) / total_stems * 35)
                        continue
                    else:
                        quality = guitar_result.quality_score if guitar_result else 0
                        logger.info(
                            f"  Guitar v3 NN quality too low ({quality:.2f}), "
                            f"falling back to Basic Pitch guitar tab"
                        )
                except Exception as e:
                    logger.warning(f"Guitar v3 NN failed for {stem_name}: {e}")

            # ---- FALLBACK: Basic Pitch guitar tab model ----
            if stem_type == 'guitar' and GUITAR_TAB_MODEL_AVAILABLE:
                logger.info(f"Transcribing {stem_name} with guitar tab model (Basic Pitch)...")
                job.stage = f'Transcribing {stem_name} (Basic Pitch guitar)'

                try:
                    tab_transcriber = GuitarTabTranscriber()
                    tab_result = tab_transcriber.transcribe(
                        audio_path=stem_path,
                        output_dir=str(midi_output_dir),
                        tempo_hint=job.metadata.get('tempo'),
                    )

                    if (tab_result.midi_path
                            and Path(tab_result.midi_path).exists()
                            and tab_result.quality_score > 0.3):
                        job.midi_files[stem_name] = tab_result.midi_path
                        job.transcription_quality[stem_name] = tab_result.quality_score
                        job.transcription_mode[stem_name] = 'guitar_tab'

                        logger.info(
                            f"Tab MIDI for {stem_name}: "
                            f"{tab_result.num_notes} notes, "
                            f"{tab_result.num_strings_used} strings, "
                            f"frets {tab_result.fret_range[0]}-{tab_result.fret_range[1]}, "
                            f"quality: {tab_result.quality_score:.2f}"
                        )
                        successful += 1
                        job.progress = 60 + int((idx + 1) / total_stems * 35)
                        continue
                    else:
                        quality = tab_result.quality_score if tab_result else 0
                        logger.info(
                            f"  Tab model quality too low ({quality:.2f}), "
                            f"falling back to melody/enhanced transcriber"
                        )
                except Exception as e:
                    logger.warning(f"Guitar tab model failed for {stem_name}: {e}")
                    # Fall through to melody extractor / enhanced transcriber

            # ---- HIGHEST PRIORITY: Bass v3 NN model ----
            if stem_type == 'bass' and BASS_NN_MODEL_AVAILABLE:
                logger.info(f"Transcribing {stem_name} with bass v3 NN model...")
                job.stage = f'Transcribing {stem_name} (bass v3 NN)'

                try:
                    bass_nn = BassNNTranscriber()
                    bass_nn_result = bass_nn.transcribe(
                        audio_path=stem_path,
                        output_dir=str(midi_output_dir),
                        tempo_hint=job.metadata.get('tempo'),
                    )

                    if (bass_nn_result.midi_path
                            and Path(bass_nn_result.midi_path).exists()
                            and bass_nn_result.quality_score > 0.3):
                        job.midi_files[stem_name] = bass_nn_result.midi_path
                        job.transcription_quality[stem_name] = bass_nn_result.quality_score
                        job.transcription_mode[stem_name] = 'bass_v3_nn'

                        logger.info(
                            f"Bass v3 MIDI for {stem_name}: "
                            f"{bass_nn_result.num_notes} notes, "
                            f"range={bass_nn_result.pitch_range}, "
                            f"polyphony={bass_nn_result.polyphony_avg:.1f}, "
                            f"quality: {bass_nn_result.quality_score:.2f}"
                        )
                        successful += 1
                        job.progress = 60 + int((idx + 1) / total_stems * 35)
                        continue
                    else:
                        quality = bass_nn_result.quality_score if bass_nn_result else 0
                        logger.info(
                            f"  Bass v3 NN quality too low ({quality:.2f}), "
                            f"falling back to Basic Pitch bass"
                        )
                except Exception as e:
                    logger.warning(f"Bass v3 NN failed for {stem_name}: {e}")

            # ---- FALLBACK: Basic Pitch bass model ----
            if stem_type == 'bass' and BASS_MODEL_AVAILABLE:
                logger.info(f"Transcribing {stem_name} with bass model (Basic Pitch)...")
                job.stage = f'Transcribing {stem_name} (Basic Pitch bass)'

                try:
                    bass_transcriber = BassTranscriber()
                    bass_result = bass_transcriber.transcribe(
                        audio_path=stem_path,
                        output_dir=str(midi_output_dir),
                        tempo_hint=job.metadata.get('tempo'),
                    )

                    if (bass_result.midi_path
                            and Path(bass_result.midi_path).exists()
                            and bass_result.quality_score > 0.3):
                        job.midi_files[stem_name] = bass_result.midi_path
                        job.transcription_quality[stem_name] = bass_result.quality_score
                        job.transcription_mode[stem_name] = 'bass_nn'

                        logger.info(
                            f"Bass MIDI for {stem_name}: "
                            f"{bass_result.num_notes} notes, "
                            f"{bass_result.num_strings_used} strings, "
                            f"frets {bass_result.fret_range[0]}-{bass_result.fret_range[1]}, "
                            f"quality: {bass_result.quality_score:.2f}"
                        )
                        successful += 1
                        job.progress = 60 + int((idx + 1) / total_stems * 35)
                        continue
                    else:
                        quality = bass_result.quality_score if bass_result else 0
                        logger.info(
                            f"  Bass model quality too low ({quality:.2f}), "
                            f"falling back to melody/enhanced transcriber"
                        )
                except Exception as e:
                    logger.warning(f"Neural bass model failed for {stem_name}: {e}")
                    # Fall through to melody extractor / enhanced transcriber

            # ---- HIGHEST PRIORITY: Neural piano model for piano/keys stems ----
            if stem_type in ('piano', 'keys', 'keyboard') and PIANO_MODEL_AVAILABLE:
                logger.info(f"🎹 Transcribing {stem_name} with neural piano model (CRNN)...")
                job.stage = f'Transcribing {stem_name} (neural piano model)'

                try:
                    piano_transcriber_inst = PianoTranscriber()
                    piano_result = piano_transcriber_inst.transcribe(
                        audio_path=stem_path,
                        output_dir=str(midi_output_dir),
                        tempo_hint=job.metadata.get('tempo'),
                    )

                    if (piano_result.midi_path
                            and Path(piano_result.midi_path).exists()
                            and piano_result.quality_score > 0.3):
                        job.midi_files[stem_name] = piano_result.midi_path
                        job.transcription_quality[stem_name] = piano_result.quality_score
                        job.transcription_mode[stem_name] = 'piano_nn'

                        logger.info(
                            f"✓ Piano MIDI for {stem_name}: "
                            f"{piano_result.num_notes} notes, "
                            f"range {piano_result.pitch_range[0]}-{piano_result.pitch_range[1]}, "
                            f"polyphony {piano_result.polyphony_avg:.1f}, "
                            f"quality: {piano_result.quality_score:.2f}"
                        )
                        successful += 1
                        job.progress = 60 + int((idx + 1) / total_stems * 35)
                        continue
                    else:
                        quality = piano_result.quality_score if piano_result else 0
                        logger.info(
                            f"  Piano model quality too low ({quality:.2f}), "
                            f"falling back to melody/enhanced transcriber"
                        )
                except Exception as e:
                    logger.warning(f"Neural piano model failed for {stem_name}: {e}")
                    # Fall through to melody extractor / enhanced transcriber

            # ---- Try melody transcriber first for lead/monophonic stems ----
            is_lead_stem = stem_name.lower() in ['guitar_lead', 'vocals', 'vocals_lead', 'bass']
            # Also treat plain 'guitar' as lead candidate if no guitar_lead sub-stem
            if stem_name.lower() == 'guitar' and 'guitar_lead' not in job.stems:
                is_lead_stem = True

            if is_lead_stem and MELODY_TRANSCRIBER_AVAILABLE:
                logger.info(f"🎵 Transcribing {stem_name} with melody extractor (monophonic)...")
                job.stage = f'Transcribing {stem_name} (melody extraction)'

                try:
                    melody_ext = MelodyExtractor(instrument=stem_type)
                    mel_result = melody_ext.transcribe(
                        audio_path=stem_path,
                        output_dir=str(midi_output_dir),
                        instrument=stem_type,
                        tempo_hint=job.metadata.get('tempo'),
                        ensemble=True,
                    )

                    if mel_result.midi_path and Path(mel_result.midi_path).exists() and mel_result.quality_score > 0.4:
                        job.midi_files[stem_name] = mel_result.midi_path
                        job.transcription_quality[stem_name] = mel_result.quality_score
                        job.articulations[stem_name] = mel_result.articulation_count
                        job.transcription_mode[stem_name] = 'melody'

                        if mel_result.detected_key and not job.detected_key:
                            job.detected_key = mel_result.detected_key

                        logger.info(f"✓ Melody MIDI for {stem_name}: "
                                   f"{len(mel_result.notes)} notes, "
                                   f"{mel_result.articulation_count} articulations, "
                                   f"quality: {mel_result.quality_score:.2f}")
                        successful += 1
                        job.progress = 60 + int((idx + 1) / total_stems * 35)
                        continue
                    else:
                        quality = mel_result.quality_score if mel_result else 0
                        logger.info(f"  Melody quality too low ({quality:.2f}), "
                                   f"falling back to enhanced transcriber")
                except Exception as e:
                    logger.warning(f"Melody transcriber failed for {stem_name}: {e}")
                    # Fall through to enhanced transcriber

            # ---- Enhanced transcriber (polyphonic, articulations) ----
            if enhanced_transcriber and stem_type in ['guitar', 'bass', 'vocals', 'piano']:
                logger.info(f"🎼 Transcribing {stem_name} with enhanced transcriber...")
                job.stage = f'Transcribing {stem_name} (articulations)'

                try:
                    result = enhanced_transcriber.transcribe(
                        audio_path=stem_path,
                        output_dir=str(midi_output_dir),
                        stem_type=stem_type,
                        detect_articulations=detect_articulations,
                        quantize=quantize,
                        quantize_grid=grid_size
                    )

                    if result.midi_path and Path(result.midi_path).exists():
                        job.midi_files[stem_name] = result.midi_path
                        job.transcription_quality[stem_name] = result.quality_score
                        job.articulations[stem_name] = len(result.articulations)
                        job.transcription_mode[stem_name] = 'enhanced'

                        # Store detected key if not already set
                        if result.detected_key and not job.detected_key:
                            job.detected_key = result.detected_key

                        logger.info(f"✓ Enhanced MIDI for {stem_name}: "
                                   f"{result.notes_count} notes, "
                                   f"{len(result.articulations)} articulations, "
                                   f"quality: {result.quality_score:.2f}")
                        successful += 1
                        job.progress = 60 + int((idx + 1) / total_stems * 35)
                        continue
                except Exception as e:
                    logger.warning(f"Enhanced transcriber failed for {stem_name}: {e}")
                    # Fall through to Basic Pitch

            # ==== FALLBACK: Use Basic Pitch ====
            try:
                from basic_pitch.inference import predict_and_save
                from basic_pitch import ICASSP_2022_MODEL_PATH
            except ImportError as e:
                logger.warning(f"Basic Pitch not available for {stem_name}: {e}")
                continue

            logger.info(f"Transcribing {stem_name} with Basic Pitch...")
            job.stage = f'Transcribing {stem_name}'

            predict_and_save(
                audio_path_list=[stem_path],
                output_directory=str(midi_output_dir),
                save_midi=True,
                sonify_midi=False,
                save_model_outputs=False,
                save_notes=False,
                model_or_model_path=ICASSP_2022_MODEL_PATH,
                onset_threshold=0.6,
                frame_threshold=0.4,
                minimum_note_length=80,
            )

            # Find the generated MIDI file
            stem_basename = Path(stem_path).stem
            midi_file = midi_output_dir / f"{stem_basename}_basic_pitch.mid"
            if not midi_file.exists():
                midi_file = midi_output_dir / f"{stem_basename}.mid"

            if midi_file.exists():
                if quantize:
                    job.stage = f'Quantizing {stem_name} MIDI'
                    quantize_midi(str(midi_file), grid_size=grid_size)

                job.midi_files[stem_name] = str(midi_file)
                logger.info(f"✓ Created MIDI for {stem_name}: {midi_file}")
                successful += 1
            else:
                logger.warning(f"MIDI file not found at expected path: {midi_file}")

            job.progress = 60 + int((idx + 1) / total_stems * 35)

        except Exception as e:
            logger.error(f"Transcription failed for {stem_name}: {e}")
            logger.error(traceback.format_exc())

    logger.info(f"MIDI transcription complete: {successful}/{total_stems} stems transcribed")
    return successful > 0


def detect_chords_for_job(job: ProcessingJob, audio_path: Path):
    """
    Detect chord progression from original audio or mix of stems.

    Stores chords in job.chord_progression with timestamps.
    """
    if not CHORD_DETECTOR_AVAILABLE:
        logger.info("Chord detector not available - skipping")
        return

    job.stage = 'Detecting chord progression'
    logger.info(f"🎸 Detecting chords (model: {CHORD_DETECTOR_VERSION})...")

    try:
        detector = ChordDetector()

        # Prefer to analyze the original audio if we have it
        analyze_path = str(audio_path) if audio_path.exists() else None

        # If original not available, try guitar or piano stem
        if not analyze_path or not Path(analyze_path).exists():
            if 'guitar' in job.stems:
                analyze_path = job.stems['guitar']
                logger.info("Using guitar stem for chord detection")
            elif 'guitar_left' in job.stems:
                analyze_path = job.stems['guitar_left']
            elif 'piano' in job.stems:
                analyze_path = job.stems['piano']
                logger.info("Using piano stem for chord detection")
            else:
                logger.info("No suitable source for chord detection")
                return

        progression = detector.detect(analyze_path)

        # Store results - V8 includes bass note for inversions, older versions don't
        job.chord_progression = []
        for c in progression.chords:
            chord_data = {
                'time': c.time,
                'duration': c.duration,
                'chord': c.chord,
                'root': c.root,
                'quality': c.quality,
                'confidence': c.confidence
            }
            # V8 adds bass note for chord inversions (e.g., C/E -> bass='E')
            if hasattr(c, 'bass'):
                chord_data['bass'] = c.bass
            chord_data['detector_version'] = CHORD_DETECTOR_VERSION
            job.chord_progression.append(chord_data)

        job.detected_key = progression.key

        logger.info(f"✅ Detected {len(job.chord_progression)} chord changes, key: {progression.key} (model: {CHORD_DETECTOR_VERSION})")

    except Exception as e:
        logger.warning(f"Chord detection failed: {e}")


def convert_midi_to_musicxml(job: ProcessingJob):
    """Convert MIDI files to MusicXML for notation display.

    Uses the midi_to_notation module for superior quality including:
    - Triplet-aware quantization
    - Melody mode with articulation markings (bends, slides, hammer-ons)
    - Dynamic markings from velocity
    - Artifact removal
    - Librosa beat-tracking fallback for tempo detection
    """
    if not NOTATION_CONVERTER_AVAILABLE:
        logger.warning("Notation converter not available - skipping MusicXML conversion")
        return False

    job.stage = 'Converting to notation'
    job.progress = 95
    logger.info("Converting MIDI to MusicXML via notation module")

    xml_output_dir = OUTPUT_DIR / job.job_id / 'musicxml'
    xml_output_dir.mkdir(parents=True, exist_ok=True)

    job.musicxml_files = {}

    # Extract song title from metadata or filename
    song_title = job.metadata.get('title', '') if job.metadata else ''
    if not song_title:
        song_title = job.filename.rsplit('.', 1)[0] if job.filename else "Untitled"
        for suffix in ['_Live', ' Live', ' (Live)', '_Remastered', ' Remastered']:
            if song_title.endswith(suffix):
                song_title = song_title[:-len(suffix)]
        song_title = song_title.replace('_', ' ')

    artist_name = job.metadata.get('artist', '') if job.metadata else ''

    for stem_name, midi_path in job.midi_files.items():
        try:
            stem_type = stem_name.lower().split('_')[0]
            xml_path = xml_output_dir / f"{stem_name}.musicxml"

            # Determine melody_mode from transcription_mode tracking
            is_melody = (hasattr(job, 'transcription_mode') and
                        job.transcription_mode.get(stem_name) == 'melody')

            # Get audio path for librosa tempo fallback
            audio_path = job.stems.get(stem_name) or job.stems.get(stem_type)

            result = _notation_convert(
                midi_path=midi_path,
                output_path=str(xml_path),
                quantize=True,
                stem_type=stem_type,
                title=song_title,
                artist=artist_name,
                melody_mode=is_melody,
                audio_path=audio_path,
            )

            if result and Path(result).exists():
                job.musicxml_files[stem_name] = result
                mode_str = " (melody mode)" if is_melody else ""
                logger.info(f"  ✓ MusicXML for {stem_name}{mode_str}")

        except Exception as e:
            logger.error(f"MusicXML conversion failed for {stem_name}: {e}")

    return len(job.musicxml_files) > 0


