"""
Centralized conditional imports and feature flags.

All try/except import blocks live here so other modules can do:
    from dependencies import CHORD_DETECTOR_AVAILABLE, detect_chords
"""

import logging

logger = logging.getLogger(__name__)

# ============ FEATURE FLAGS (set by try/except blocks below) ============

DRIVE_AVAILABLE = False
SKILLS_AVAILABLE = False
ENHANCER_AVAILABLE = False
DRUM_TRANSCRIBER_AVAILABLE = False
STEREO_SPLITTER_AVAILABLE = False
ENHANCED_SEPARATOR_AVAILABLE = False
GUITAR_SEPARATOR_AVAILABLE = False
TRACK_INFO_AVAILABLE = False
ENHANCED_TRANSCRIBER_AVAILABLE = False
DRUM_TRANSCRIBER_V2_AVAILABLE = False
OAF_DRUM_TRANSCRIBER_AVAILABLE = False
OAF_AVAILABLE = False
DRUM_NN_MODEL_AVAILABLE = False
MODEL_MANAGER_AVAILABLE = False
CHORD_DETECTOR_AVAILABLE = False
CHORD_DETECTOR_VERSION = None
CHORD_THEORY_AVAILABLE = False
MELODY_TRANSCRIBER_AVAILABLE = False
GUITAR_TAB_MODEL_AVAILABLE = False
BASS_MODEL_AVAILABLE = False
PIANO_MODEL_AVAILABLE = False
ARCHIVE_PIPELINE_AVAILABLE = False
GP_CONVERTER_AVAILABLE = False
NOTATION_CONVERTER_AVAILABLE = False
MUSIC21_AVAILABLE = False
ENSEMBLE_SEPARATOR_AVAILABLE = False
EXTENDED_BAND_CONFIG_AVAILABLE = False

# Singletons
_gpu_manager = None
_chord_theory_engine = None

# ============ CONDITIONAL IMPORTS ============

# Google Drive integration
try:
    from drive_service import upload_job_to_drive, cleanup_old_stems, get_drive_stats, get_drive_service  # noqa: F401
    DRIVE_AVAILABLE = True
    logger.info("Google Drive integration available")
except ImportError as e:
    DRIVE_AVAILABLE = False
    logger.warning(f"Google Drive integration not available: {e}")

# Skills system for enhanced stem processing
try:
    from skills import get_skill, get_all_skills, apply_skill, SKILL_REGISTRY, analyze_stems_for_skills  # noqa: F401
    SKILLS_AVAILABLE = True
    logger.info(f"Skills system available with {len(SKILL_REGISTRY)} skills")
except ImportError as e:
    SKILLS_AVAILABLE = False
    logger.warning(f"Skills system not available: {e}")

# Stem enhancement for natural sound
try:
    from enhancer import enhance_stem, enhance_all_stems, PEDALBOARD_AVAILABLE, NOISEREDUCE_AVAILABLE  # noqa: F401
    ENHANCER_AVAILABLE = PEDALBOARD_AVAILABLE
    logger.info(f"Stem enhancer available (pedalboard={PEDALBOARD_AVAILABLE}, noisereduce={NOISEREDUCE_AVAILABLE})")
except ImportError as e:
    ENHANCER_AVAILABLE = False
    logger.warning(f"Stem enhancer not available: {e}")

# Drum transcription (separate from Basic Pitch)
try:
    from drum_transcriber import transcribe_drums_to_midi  # noqa: F401
    DRUM_TRANSCRIBER_AVAILABLE = True
    logger.info("Drum transcriber available")
except ImportError as e:
    DRUM_TRANSCRIBER_AVAILABLE = False
    logger.warning(f"Drum transcriber not available: {e}")

# Stereo splitting for panned instruments
try:
    from stereo_splitter import split_stereo, split_all_stems_by_panning, check_if_splittable  # noqa: F401
    STEREO_SPLITTER_AVAILABLE = True
    logger.info("Stereo splitter available")
except ImportError as e:
    STEREO_SPLITTER_AVAILABLE = False
    logger.warning(f"Stereo splitter not available: {e}")

# Enhanced separator with audio-separator (BS-Roformer, UVR models)
try:
    from enhanced_separator import (
        EnhancedSeparator, AUDIO_SEPARATOR_AVAILABLE,  # noqa: F401
        separate_audio, separate_full_song, split_vocals, MODELS as SEPARATOR_MODELS  # noqa: F401
    )
    ENHANCED_SEPARATOR_AVAILABLE = AUDIO_SEPARATOR_AVAILABLE
    if ENHANCED_SEPARATOR_AVAILABLE:
        logger.info(f"Enhanced separator available ({len(SEPARATOR_MODELS)} models)")
except ImportError as e:
    ENHANCED_SEPARATOR_AVAILABLE = False
    logger.warning(f"Enhanced separator not available: {e}")

# Guitar lead/rhythm separator (trained MelBand-RoFormer)
try:
    from guitar_separator import (
        GuitarSeparator, separate_guitar,  # noqa: F401
        GUITAR_SEPARATOR_AVAILABLE as _gsa, is_available as guitar_separator_is_available  # noqa: F401
    )
    GUITAR_SEPARATOR_AVAILABLE = _gsa
    if GUITAR_SEPARATOR_AVAILABLE:
        logger.info("Guitar lead/rhythm separator available (MelBand-RoFormer)")
except ImportError as e:
    GUITAR_SEPARATOR_AVAILABLE = False
    logger.warning(f"Guitar lead/rhythm separator not available: {e}")

# Track info fetcher for context and learning tips
try:
    from track_info import fetch_track_info, extract_artist_from_title, get_instrument_tips, LOCAL_KNOWLEDGE, should_stereo_split  # noqa: F401
    TRACK_INFO_AVAILABLE = True
    logger.info(f"Track info available (local knowledge: {len(LOCAL_KNOWLEDGE)} artists)")
except ImportError as e:
    TRACK_INFO_AVAILABLE = False
    logger.warning(f"Track info not available: {e}")

# Enhanced transcriber with articulation detection (bends, slides, hammer-ons)
try:
    from transcriber_enhanced import EnhancedTranscriber, transcribe_with_enhanced  # noqa: F401
    ENHANCED_TRANSCRIBER_AVAILABLE = True
    logger.info("Enhanced transcriber available (articulations, polyphony)")
except ImportError as e:
    ENHANCED_TRANSCRIBER_AVAILABLE = False
    logger.warning(f"Enhanced transcriber not available: {e}")

# Improved drum transcription v2 (ghost notes, cymbal differentiation)
try:
    from drum_transcriber_v2 import EnhancedDrumTranscriber, transcribe_drums_to_midi as transcribe_drums_v2  # noqa: F401
    DRUM_TRANSCRIBER_V2_AVAILABLE = True
    logger.info("Drum transcriber v2 available (ghost notes, hi-hat states)")
except ImportError as e:
    DRUM_TRANSCRIBER_V2_AVAILABLE = False
    logger.warning(f"Drum transcriber v2 not available: {e}")

# OaF-based drum transcription (neural network)
try:
    from oaf_drum_transcriber import OaFDrumTranscriber, transcribe_drums, OAF_AVAILABLE as _oaf  # noqa: F401
    OAF_DRUM_TRANSCRIBER_AVAILABLE = True
    OAF_AVAILABLE = _oaf
    logger.info(f"OaF Drum transcriber available (neural: {OAF_AVAILABLE})")
except ImportError as e:
    OAF_DRUM_TRANSCRIBER_AVAILABLE = False
    OAF_AVAILABLE = False
    logger.warning(f"OaF Drum transcriber not available: {e}")

# Neural drum transcription model (CRNN trained on E-GMD, 8 classes)
try:
    from drum_nn_transcriber import (
        NeuralDrumTranscriber, transcribe_drums_nn,  # noqa: F401
        DRUM_NN_MODEL_AVAILABLE as _dnn, is_available as drum_nn_is_available  # noqa: F401
    )
    DRUM_NN_MODEL_AVAILABLE = _dnn
    if DRUM_NN_MODEL_AVAILABLE:
        logger.info("Neural drum model available (CRNN, 8-class, val_loss 0.0335)")
except ImportError as e:
    DRUM_NN_MODEL_AVAILABLE = False
    logger.warning(f"Neural drum model not available: {e}")

# Model manager for pretrained models
try:
    from models.model_manager import ModelManager, ensure_model, list_available_models  # noqa: F401
    MODEL_MANAGER_AVAILABLE = True
    logger.info("Model manager available")
except ImportError as e:
    MODEL_MANAGER_AVAILABLE = False
    logger.warning(f"Model manager not available: {e}")

# Chord detection (V8 Transformer model preferred, falls back to V7 then basic template matching)
try:
    from chord_detector_v8 import ChordDetector, detect_chords
    CHORD_DETECTOR_AVAILABLE = True
    CHORD_DETECTOR_VERSION = 'v8'
    logger.info("Chord detector V8 available (337 classes, inversions, mMaj7)")
except ImportError:
    try:
        from chord_detector_v7 import ChordDetector, detect_chords
        CHORD_DETECTOR_AVAILABLE = True
        CHORD_DETECTOR_VERSION = 'v7'
        logger.info("Chord detector V7 available (25 classes)")
    except ImportError:
        try:
            from chord_detector import ChordDetector, detect_chords  # noqa: F401
            CHORD_DETECTOR_AVAILABLE = True
            CHORD_DETECTOR_VERSION = 'basic'
            logger.info("Basic chord detector available (template matching)")
        except ImportError as e:
            CHORD_DETECTOR_AVAILABLE = False
            CHORD_DETECTOR_VERSION = None
            logger.warning(f"Chord detector not available: {e}")

# Chord theory engine (scale suggestions, Beato-style chord-over-bass analysis)
try:
    from chord_theory import ChordTheoryEngine, get_scale_suggestion, get_progression_analysis  # noqa: F401
    CHORD_THEORY_AVAILABLE = True
    _chord_theory_engine = ChordTheoryEngine()
    logger.info("Chord theory engine available (scale suggestions, polychord analysis)")
except ImportError as e:
    CHORD_THEORY_AVAILABLE = False
    _chord_theory_engine = None
    logger.warning(f"Chord theory engine not available: {e}")

# Monophonic melody/lead transcriber (clean lead lines, articulations)
try:
    from melody_transcriber import MelodyExtractor, transcribe_melody  # noqa: F401
    MELODY_TRANSCRIBER_AVAILABLE = True
    logger.info("Melody transcriber available (monophonic lead extraction, vibrato, bends)")
except ImportError as e:
    MELODY_TRANSCRIBER_AVAILABLE = False
    logger.warning(f"Melody transcriber not available: {e}")

# Guitar tab model (CRNN neural network for string/fret prediction)
try:
    from guitar_tab_transcriber import (
        GuitarTabTranscriber, transcribe_guitar_tab,  # noqa: F401
        GUITAR_TAB_MODEL_AVAILABLE as _gtm, is_available as guitar_tab_is_available  # noqa: F401
    )
    GUITAR_TAB_MODEL_AVAILABLE = _gtm
    if GUITAR_TAB_MODEL_AVAILABLE:
        logger.info("Guitar tab model available (CRNN, 6-string x 20-fret)")
except ImportError as e:
    GUITAR_TAB_MODEL_AVAILABLE = False
    logger.warning(f"Guitar tab model not available: {e}")

# Neural bass transcription model (CRNN trained on Slakh2100, 4-string x 24-fret)
try:
    from bass_transcriber import (
        BassTranscriber, transcribe_bass,  # noqa: F401
        BASS_MODEL_AVAILABLE as _bma, is_available as bass_nn_is_available  # noqa: F401
    )
    BASS_MODEL_AVAILABLE = _bma
    if BASS_MODEL_AVAILABLE:
        logger.info("Neural bass model available (CRNN, 4-string x 24-fret, Slakh2100)")
except ImportError as e:
    BASS_MODEL_AVAILABLE = False
    logger.warning(f"Neural bass model not available: {e}")

# Neural piano transcription model (CRNN trained on MAESTRO, 88 keys)
try:
    from piano_transcriber import (
        PianoTranscriber, transcribe_piano,  # noqa: F401
        PIANO_MODEL_AVAILABLE as _pma, is_available as piano_nn_is_available  # noqa: F401
    )
    PIANO_MODEL_AVAILABLE = _pma
    if PIANO_MODEL_AVAILABLE:
        logger.info("Neural piano model available (CRNN, 88-key, MAESTRO v3)")
except ImportError as e:
    PIANO_MODEL_AVAILABLE = False
    logger.warning(f"Neural piano model not available: {e}")

# Internet Archive Live Music pipeline (search/browse/batch)
try:
    from archive_pipeline import ArchivePipeline, search_archive, get_show_info, get_pipeline as get_archive_pipeline  # noqa: F401
    ARCHIVE_PIPELINE_AVAILABLE = True
    logger.info("Archive.org Live Music pipeline available (250k+ free concert recordings)")
except ImportError as e:
    ARCHIVE_PIPELINE_AVAILABLE = False
    logger.warning(f"Archive.org pipeline not available: {e}")

# MIDI to Guitar Pro conversion
try:
    from midi_to_gp import convert_midi_to_gp, convert_job_midis_to_gp  # noqa: F401
    GP_CONVERTER_AVAILABLE = True
    logger.info("Guitar Pro converter available")
except ImportError as e:
    GP_CONVERTER_AVAILABLE = False
    logger.warning(f"Guitar Pro converter not available: {e}")

# MIDI to MusicXML notation conversion (articulations, melody mode, dynamics)
try:
    from midi_to_notation import midi_to_musicxml as _notation_convert, MUSIC21_AVAILABLE as _m21  # noqa: F401
    NOTATION_CONVERTER_AVAILABLE = True
    MUSIC21_AVAILABLE = _m21
    logger.info("Notation converter available (articulations, dynamics, triplet quantization)")
except ImportError as e:
    NOTATION_CONVERTER_AVAILABLE = False
    MUSIC21_AVAILABLE = False
    logger.warning(f"Notation converter not available: {e}")

# Ensemble separation system (Moises.ai-quality multi-model approach)
try:
    from separation import GPUManager
    _gpu_manager = GPUManager()
    ENSEMBLE_SEPARATOR_AVAILABLE = True
    logger.info(f"Ensemble separator available ({_gpu_manager.device_info.device_name}, {_gpu_manager.device_info.total_memory_gb:.1f}GB)")
except Exception as e:
    ENSEMBLE_SEPARATOR_AVAILABLE = False
    _gpu_manager = None
    logger.warning(f"Ensemble separator not available: {e}")

# Extended band database (Phish, Allman Brothers, 70+ bands)
try:
    from band_config import (
        should_stereo_split as extended_should_stereo_split,  # noqa: F401
        get_player_positions as get_extended_positions,  # noqa: F401
        has_dual_drummers,  # noqa: F401
        get_learning_tips as get_extended_tips,  # noqa: F401
        STEREO_SPLIT_BANDS
    )
    EXTENDED_BAND_CONFIG_AVAILABLE = True
    logger.info(f"Extended band config available ({len(STEREO_SPLIT_BANDS)} bands)")
except ImportError as e:
    EXTENDED_BAND_CONFIG_AVAILABLE = False
    logger.warning(f"Extended band config not available: {e}")
