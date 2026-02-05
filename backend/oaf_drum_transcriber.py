"""
Onsets and Frames (OaF) Drum Transcription for StemScribe
=========================================================
Uses the Magenta OaF Drums model trained on the E-GMD dataset
for high-quality drum transcription.

Falls back to spectral analysis if the OaF model is not available.

The OaF Drums model achieves state-of-the-art drum transcription by:
- Learning from 444 hours of E-GMD drum recordings
- Predicting onsets, frames, and velocities simultaneously
- Supporting 9 drum classes (kick, snare, closed hihat, open hihat,
  low tom, mid tom, high tom, crash, ride)
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)

# Check for TensorFlow/Magenta availability
OAF_AVAILABLE = False
MAGENTA_AVAILABLE = False

try:
    import tensorflow as tf
    # Suppress TF warnings
    tf.get_logger().setLevel('ERROR')

    try:
        from magenta.models.onsets_frames_transcription import train_util
        from magenta.models.onsets_frames_transcription import configs
        from magenta.models.onsets_frames_transcription import infer_util
        MAGENTA_AVAILABLE = True
        OAF_AVAILABLE = True
        logger.info("✅ Magenta OaF available")
    except ImportError:
        logger.info("Magenta not installed - using basic-pitch or spectral fallback")

except ImportError:
    logger.info("TensorFlow not installed - using spectral fallback")

# Try basic-pitch as an alternative
BASIC_PITCH_AVAILABLE = False
try:
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    BASIC_PITCH_AVAILABLE = True
    logger.info("✅ basic-pitch available")
except ImportError:
    logger.debug("basic-pitch not installed")

# Import our spectral fallback
try:
    from drum_transcriber_v2 import (
        EnhancedDrumTranscriber,
        transcribe_drums_to_midi,
        GM_DRUMS
    )
    SPECTRAL_AVAILABLE = True
except ImportError:
    SPECTRAL_AVAILABLE = False
    logger.warning("Spectral drum transcriber not available")


# OaF to General MIDI mapping
# OaF uses 9 classes: kick, snare, closed_hh, open_hh, low_tom, mid_tom, high_tom, crash, ride
OAF_TO_GM = {
    0: 36,   # kick -> Bass Drum 1
    1: 38,   # snare -> Acoustic Snare
    2: 42,   # closed_hh -> Closed Hi-Hat
    3: 46,   # open_hh -> Open Hi-Hat
    4: 45,   # low_tom -> Low Tom
    5: 47,   # mid_tom -> Low-Mid Tom
    6: 50,   # high_tom -> High Tom
    7: 49,   # crash -> Crash Cymbal 1
    8: 51,   # ride -> Ride Cymbal 1
}

OAF_CLASS_NAMES = ['kick', 'snare', 'closed_hh', 'open_hh', 'low_tom', 'mid_tom', 'high_tom', 'crash', 'ride']


class OaFDrumTranscriber:
    """
    Drum transcriber using Onsets and Frames neural network model.
    Falls back to spectral analysis if OaF is not available.
    """

    def __init__(self, model_dir: Optional[str] = None, use_gpu: bool = True):
        """
        Initialize the OaF drum transcriber.

        Args:
            model_dir: Directory containing OaF checkpoint (if not using default)
            use_gpu: Whether to use GPU if available
        """
        self.model_dir = model_dir
        self.use_gpu = use_gpu
        self._model_loaded = False
        self._session = None
        self._estimator = None

        # Determine which backend to use
        if OAF_AVAILABLE:
            self.backend = 'oaf'
            logger.info("🥁 Using OaF Drums neural network backend")
        elif BASIC_PITCH_AVAILABLE:
            self.backend = 'basic_pitch'
            logger.info("🥁 Using basic-pitch backend")
        elif SPECTRAL_AVAILABLE:
            self.backend = 'spectral'
            logger.info("🥁 Using spectral analysis backend")
        else:
            self.backend = None
            logger.error("❌ No drum transcription backend available!")

    def _load_oaf_model(self) -> bool:
        """Load the OaF drums model."""
        if not OAF_AVAILABLE:
            return False

        if self._model_loaded:
            return True

        try:
            # Get drum config
            config = configs.CONFIG_MAP['drums']

            # Find checkpoint
            if self.model_dir:
                checkpoint_dir = Path(self.model_dir)
            else:
                # Check common locations
                from models.model_manager import DEFAULT_MODEL_DIR
                checkpoint_dir = DEFAULT_MODEL_DIR / 'drum_checkpoint'

            if not checkpoint_dir.exists():
                logger.warning(f"OaF checkpoint not found at {checkpoint_dir}")
                return False

            # Create estimator
            self._estimator = train_util.create_estimator(
                config.model_fn,
                checkpoint_dir=str(checkpoint_dir),
                hparams=config.hparams
            )

            self._model_loaded = True
            logger.info("✅ OaF drums model loaded")
            return True

        except Exception as e:
            logger.error(f"Failed to load OaF model: {e}")
            return False

    def transcribe(self, audio_path: str, output_path: str,
                   sensitivity: float = 0.5,
                   min_velocity: int = 20) -> Dict:
        """
        Transcribe drum audio to MIDI.

        Args:
            audio_path: Path to drum stem audio
            output_path: Path for output MIDI file
            sensitivity: Onset detection sensitivity (0-1)
            min_velocity: Minimum velocity to include

        Returns:
            Dict with transcription statistics
        """
        logger.info(f"🥁 Drum transcription: {Path(audio_path).name}")
        logger.info(f"   Backend: {self.backend}")

        if self.backend == 'oaf':
            return self._transcribe_oaf(audio_path, output_path, sensitivity, min_velocity)
        elif self.backend == 'basic_pitch':
            return self._transcribe_basic_pitch(audio_path, output_path, sensitivity, min_velocity)
        elif self.backend == 'spectral':
            return self._transcribe_spectral(audio_path, output_path, sensitivity)
        else:
            logger.error("No transcription backend available")
            return {'success': False, 'error': 'No backend available'}

    def _transcribe_oaf(self, audio_path: str, output_path: str,
                         sensitivity: float, min_velocity: int) -> Dict:
        """Transcribe using OaF neural network."""
        if not self._load_oaf_model():
            logger.warning("OaF model not available, falling back to spectral")
            return self._transcribe_spectral(audio_path, output_path, sensitivity)

        try:
            # Load and preprocess audio
            import librosa
            y, sr = librosa.load(audio_path, sr=16000, mono=True)

            # Get predictions from OaF
            config = configs.CONFIG_MAP['drums']

            predictions = infer_util.predict_sequence(
                estimator=self._estimator,
                examples=[{'audio': y, 'sample_rate': sr}],
                hparams=config.hparams,
                batch_size=1
            )

            # Convert predictions to MIDI
            import pretty_midi
            midi = pretty_midi.PrettyMIDI()
            drum_track = pretty_midi.Instrument(program=0, is_drum=True, name='Drums')

            hits_by_type = {}
            total_hits = 0

            for pred in predictions:
                onset_times = pred['onset_times']
                onset_probs = pred['onset_probs']
                velocities = pred.get('velocities', np.ones_like(onset_probs) * 0.8)

                for i, (time, prob, vel) in enumerate(zip(onset_times, onset_probs, velocities)):
                    if prob < (1 - sensitivity):
                        continue

                    midi_velocity = int(vel * 127)
                    if midi_velocity < min_velocity:
                        continue

                    drum_class = pred.get('drum_class', i % 9)
                    midi_note = OAF_TO_GM.get(drum_class, 38)
                    class_name = OAF_CLASS_NAMES[drum_class] if drum_class < len(OAF_CLASS_NAMES) else 'snare'

                    note = pretty_midi.Note(
                        velocity=midi_velocity,
                        pitch=midi_note,
                        start=time,
                        end=time + 0.05
                    )
                    drum_track.notes.append(note)

                    hits_by_type[class_name] = hits_by_type.get(class_name, 0) + 1
                    total_hits += 1

            midi.instruments.append(drum_track)
            midi.write(output_path)

            logger.info(f"✅ OaF transcription complete: {total_hits} hits")
            return {
                'success': True,
                'total_hits': total_hits,
                'hits_by_type': hits_by_type,
                'backend': 'oaf'
            }

        except Exception as e:
            logger.error(f"OaF transcription failed: {e}")
            logger.info("Falling back to spectral analysis")
            return self._transcribe_spectral(audio_path, output_path, sensitivity)

    def _transcribe_basic_pitch(self, audio_path: str, output_path: str,
                                  sensitivity: float, min_velocity: int) -> Dict:
        """Transcribe using basic-pitch."""
        try:
            # basic-pitch is designed for pitched instruments but can help with toms
            model_output, midi_data, note_events = predict(audio_path)

            # For drums, basic-pitch doesn't work well - fall back to spectral
            logger.info("basic-pitch better suited for pitched instruments, using spectral for drums")
            return self._transcribe_spectral(audio_path, output_path, sensitivity)

        except Exception as e:
            logger.warning(f"basic-pitch failed: {e}")
            return self._transcribe_spectral(audio_path, output_path, sensitivity)

    def _transcribe_spectral(self, audio_path: str, output_path: str,
                               sensitivity: float) -> Dict:
        """Transcribe using spectral analysis (our v2 transcriber)."""
        if not SPECTRAL_AVAILABLE:
            return {'success': False, 'error': 'Spectral transcriber not available'}

        transcriber = EnhancedDrumTranscriber()
        stats = transcriber.transcribe(
            audio_path=audio_path,
            output_path=output_path,
            sensitivity=sensitivity,
            detect_ghost_notes=True,
            preserve_dynamics=True
        )

        return {
            'success': stats.total_hits > 0,
            'total_hits': stats.total_hits,
            'hits_by_type': stats.hits_by_type,
            'ghost_notes': stats.ghost_notes,
            'tempo': stats.tempo,
            'backend': 'spectral'
        }


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def transcribe_drums(audio_path: str, output_path: str,
                     sensitivity: float = 0.5,
                     prefer_neural: bool = True) -> Dict:
    """
    Transcribe drums from audio to MIDI.

    Automatically selects the best available backend:
    1. OaF Drums neural network (if available)
    2. Spectral analysis fallback

    Args:
        audio_path: Path to drum audio
        output_path: Output MIDI path
        sensitivity: Detection sensitivity (0-1)
        prefer_neural: Whether to prefer neural network over spectral

    Returns:
        Dict with transcription results
    """
    transcriber = OaFDrumTranscriber()

    # If neural not preferred, force spectral
    if not prefer_neural and transcriber.backend == 'oaf':
        transcriber.backend = 'spectral'

    return transcriber.transcribe(audio_path, output_path, sensitivity)


if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.INFO)

    print("🥁 OaF Drum Transcriber")
    print(f"   OaF Available: {OAF_AVAILABLE}")
    print(f"   basic-pitch Available: {BASIC_PITCH_AVAILABLE}")
    print(f"   Spectral Available: {SPECTRAL_AVAILABLE}")

    if len(sys.argv) >= 3:
        audio_path = sys.argv[1]
        output_path = sys.argv[2]

        result = transcribe_drums(audio_path, output_path)
        print(f"\n✅ Result: {result}")
