"""
Trimplexx Guitar Tablature Transcriber for StemScriber
======================================================
Uses the trimplexx CRNN model trained on GuitarSet to output string+fret
positions DIRECTLY (not just MIDI pitches like the Kong-style model).

Architecture: GuitarTabCRNN (trimplexx)
  CNN: 5x Conv2d+BN+ReLU+MaxPool -> RNN (BiGRU 768x2) -> onset_fc + fret_fc
  Outputs per frame: onset logits (6 strings) + fret logits (6 strings x 22 classes)
  22 fret classes = frets 0-20 + 1 silence class

Model: backend/models/pretrained/trimplexx_guitar_model.pt (78.9MB)
Config: backend/models/pretrained/trimplexx_run_config.json
Training: train_tab_model/trimplexx/ (GuitarSet, run_72, TDR F1=0.857)

Key advantage over guitar_nn_transcriber.py (Kong-style):
  - Outputs string+fret positions DIRECTLY
  - 85% TDR F1 (string + fret + onset accuracy)
  - Eliminates need for FretMapper/A*-Guitar post-processing
  - ~1.2s per 30s clip on CPU

Input: CQT spectrogram (22050 Hz, hop=512, 168 bins, 24 bins/octave, fmin=E2)
Output: MIDI file + tab_data list with {start_time, end_time, string, fret, pitch_midi}
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, field

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ============================================================================
# PATHS
# ============================================================================

CHECKPOINT_PATH = Path(__file__).parent / 'models' / 'pretrained' / 'trimplexx_guitar_model.pt'
CONFIG_PATH = Path(__file__).parent / 'models' / 'pretrained' / 'trimplexx_run_config.json'
MODEL_AVAILABLE = CHECKPOINT_PATH.exists() and CONFIG_PATH.exists()

# ============================================================================
# CONSTANTS (from trimplexx config - must match training exactly)
# ============================================================================

SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_BINS_CQT = 168
BINS_PER_OCTAVE_CQT = 24
MAX_FRETS = 20
FRET_SILENCE_CLASS_OFFSET = 1
NUM_STRINGS = 6
MIN_NOTE_DURATION_FRAMES = 2

# Standard guitar tuning: string 0=E2(40), 1=A2(45), 2=D3(50), 3=G3(55), 4=B3(59), 5=E4(64)
OPEN_STRING_PITCHES = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}

# CQT fmin = E2 frequency
FMIN_CQT = 82.4068892282175  # librosa.note_to_hz('E2')

# Onset threshold (tuned during training)
DEFAULT_ONSET_THRESHOLD = 0.5

# CNN architecture constants (must match training config exactly)
CNN_INPUT_CHANNELS = 1
CNN_OUTPUT_CHANNELS_LIST = [32, 64, 128, 128, 128]
CNN_KERNEL_SIZES = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
CNN_STRIDES = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
CNN_PADDINGS = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
CNN_POOLING_KERNELS = [(2, 1), (2, 1), (2, 1), (2, 1), (1, 1)]
CNN_POOLING_STRIDES = [(2, 1), (2, 1), (2, 1), (2, 1), (1, 1)]

# Pre-computed: CNN output is (128, 10, N_frames) -> rnn_input_dim = 128*10 = 1280
RNN_INPUT_DIM = 1280

if MODEL_AVAILABLE:
    logger.info(f"Trimplexx guitar model found ({CHECKPOINT_PATH.stat().st_size / 1e6:.1f}MB)")
else:
    logger.debug(f"Trimplexx guitar model not found at {CHECKPOINT_PATH}")


# ============================================================================
# MODEL ARCHITECTURE (must match trimplexx/python/model/architecture.py exactly)
# ============================================================================

class TabCNN(nn.Module):
    """CNN feature extractor from trimplexx."""
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        current_channels = CNN_INPUT_CHANNELS
        for i in range(len(CNN_OUTPUT_CHANNELS_LIST)):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, CNN_OUTPUT_CHANNELS_LIST[i],
                              kernel_size=CNN_KERNEL_SIZES[i],
                              stride=CNN_STRIDES[i],
                              padding=CNN_PADDINGS[i]),
                    nn.BatchNorm2d(CNN_OUTPUT_CHANNELS_LIST[i]),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=CNN_POOLING_KERNELS[i],
                                 stride=CNN_POOLING_STRIDES[i]),
                )
            )
            current_channels = CNN_OUTPUT_CHANNELS_LIST[i]

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x


class GuitarTabCRNN(nn.Module):
    """
    trimplexx CRNN: CNN -> BiGRU -> onset_fc + fret_fc

    Outputs per frame:
      onset_logits: (batch, frames, 6)  -- one onset logit per string
      fret_logits:  (batch, frames, 6, 22) -- 22 fret classes per string
    """
    def __init__(self, rnn_input_dim=RNN_INPUT_DIM, rnn_type="GRU",
                 rnn_hidden_size=768, rnn_layers=2, rnn_dropout=0.5,
                 rnn_bidirectional=True):
        super().__init__()
        self.num_strings = NUM_STRINGS
        self.num_fret_classes = MAX_FRETS + FRET_SILENCE_CLASS_OFFSET + 1  # 22

        self.cnn = TabCNN()
        self.rnn_input_dim = rnn_input_dim

        rnn_cls = nn.GRU if rnn_type.upper() == "GRU" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=rnn_input_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=rnn_bidirectional,
            dropout=rnn_dropout if rnn_layers > 1 else 0,
        )

        rnn_output_size = (2 * rnn_hidden_size) if rnn_bidirectional else rnn_hidden_size
        self.onset_fc = nn.Linear(rnn_output_size, self.num_strings)
        self.fret_fc = nn.Linear(rnn_output_size, self.num_strings * self.num_fret_classes)

    def forward(self, x):
        # x: (batch, n_bins, n_frames) -- CQT spectrogram
        x = x.unsqueeze(1)  # (batch, 1, n_bins, n_frames)
        x_cnn = self.cnn(x)  # (batch, channels, reduced_bins, n_frames)

        batch_size, channels_out, reduced_n_mels, reduced_n_frames = x_cnn.shape

        # Reshape: (batch, n_frames, channels * reduced_bins)
        x_rnn_input = x_cnn.permute(0, 3, 1, 2)
        x_rnn_input = x_rnn_input.reshape(batch_size, reduced_n_frames, channels_out * reduced_n_mels)

        x_rnn_output, _ = self.rnn(x_rnn_input)

        onset_logits = self.onset_fc(x_rnn_output)  # (batch, frames, 6)
        fret_logits_flat = self.fret_fc(x_rnn_output)  # (batch, frames, 6*22)
        fret_logits = fret_logits_flat.reshape(
            batch_size, reduced_n_frames, self.num_strings, self.num_fret_classes
        )

        return onset_logits, fret_logits


# ============================================================================
# RESULT DATACLASS
# ============================================================================

@dataclass
class TrimplexxTranscriptionResult:
    midi_path: Optional[str]
    num_notes: int
    quality_score: float
    method: str
    pitch_range: tuple
    polyphony_avg: float
    tab_data: List[Dict] = field(default_factory=list)


# ============================================================================
# NOTE EXTRACTION (adapted from trimplexx/training/note_conversion_utils.py)
# ============================================================================

def _frames_to_notes(onset_preds_binary, fret_pred_indices, hop_length, sample_rate,
                     max_fret_value=MAX_FRETS, min_note_frames=MIN_NOTE_DURATION_FRAMES):
    """
    Convert frame-level onset + fret predictions to note events.

    Args:
        onset_preds_binary: (num_frames, 6) binary onset predictions
        fret_pred_indices: (num_frames, 6) fret class indices (0-20 = frets, 21 = silence)
        hop_length: audio hop length
        sample_rate: audio sample rate

    Returns:
        List of dicts with start_time, end_time, string, fret, pitch_midi
    """
    num_frames, num_strings = onset_preds_binary.shape
    time_per_frame = hop_length / sample_rate
    silence_class = max_fret_value + FRET_SILENCE_CLASS_OFFSET  # 21
    notes = []

    for string_idx in range(num_strings):
        active_start = None
        active_fret = None

        for frame_idx in range(num_frames):
            is_onset = onset_preds_binary[frame_idx, string_idx] > 0.5
            current_fret = int(fret_pred_indices[frame_idx, string_idx])

            # Check if active note should terminate
            should_terminate = False
            if active_start is not None:
                if is_onset and frame_idx > active_start:
                    should_terminate = True
                elif current_fret == silence_class:
                    should_terminate = True
                elif current_fret != active_fret:
                    should_terminate = True
                elif frame_idx == num_frames - 1:
                    should_terminate = True

                if should_terminate:
                    duration_frames = frame_idx - active_start
                    if duration_frames >= min_note_frames and active_fret != silence_class:
                        if 0 <= active_fret <= max_fret_value:
                            pitch_midi = OPEN_STRING_PITCHES[string_idx] + active_fret
                            notes.append({
                                'start_time': active_start * time_per_frame,
                                'end_time': frame_idx * time_per_frame,
                                'string': string_idx,
                                'fret': int(active_fret),
                                'pitch_midi': int(round(pitch_midi)),
                            })
                    active_start = None
                    active_fret = None

            # Start new note on onset with non-silence fret
            if is_onset and current_fret != silence_class:
                active_start = frame_idx
                active_fret = current_fret

        # Handle note still active at end of audio
        if active_start is not None:
            duration_frames = num_frames - active_start
            if duration_frames >= min_note_frames and active_fret != silence_class:
                if 0 <= active_fret <= max_fret_value:
                    pitch_midi = OPEN_STRING_PITCHES[string_idx] + active_fret
                    notes.append({
                        'start_time': active_start * time_per_frame,
                        'end_time': num_frames * time_per_frame,
                        'string': string_idx,
                        'fret': int(active_fret),
                        'pitch_midi': int(round(pitch_midi)),
                    })

    notes.sort(key=lambda n: n['start_time'])
    return notes


# ============================================================================
# TRANSCRIBER
# ============================================================================

class TrimplexxTranscriber:
    """
    Guitar tablature transcriber using the trimplexx CRNN model.
    Outputs string+fret positions directly -- no FretMapper/A*-Guitar needed.
    Lazy-loads model on first use.
    """

    def __init__(self):
        self._model = None
        self._device = None

    def _ensure_model(self):
        if self._model is not None:
            return

        if not MODEL_AVAILABLE:
            raise RuntimeError(
                f"Trimplexx model not found at {CHECKPOINT_PATH} "
                f"or config not found at {CONFIG_PATH}"
            )

        self._device = torch.device('cpu')  # CPU inference as designed

        # Load config to get hyperparameters
        with open(CONFIG_PATH, 'r') as f:
            run_config = json.load(f)

        hyperparams = run_config['hyperparameters_tuned']

        self._model = GuitarTabCRNN(
            rnn_input_dim=RNN_INPUT_DIM,
            rnn_type=hyperparams.get('RNN_TYPE', 'GRU'),
            rnn_hidden_size=hyperparams['RNN_HIDDEN_SIZE'],
            rnn_layers=hyperparams['RNN_LAYERS'],
            rnn_dropout=hyperparams['RNN_DROPOUT'],
            rnn_bidirectional=hyperparams.get('RNN_BIDIRECTIONAL', True),
        )

        state_dict = torch.load(CHECKPOINT_PATH, map_location=self._device, weights_only=True)

        # Handle DataParallel prefix
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        self._model.load_state_dict(state_dict)
        self._model.eval()

        logger.info(f"Trimplexx guitar model loaded on {self._device} "
                    f"(RNN={hyperparams.get('RNN_TYPE', 'GRU')}, "
                    f"hidden={hyperparams['RNN_HIDDEN_SIZE']}, "
                    f"layers={hyperparams['RNN_LAYERS']})")

    def transcribe(self, audio_path: str, output_dir: str,
                   tempo_hint: float = None,
                   onset_threshold: float = DEFAULT_ONSET_THRESHOLD) -> TrimplexxTranscriptionResult:
        """
        Transcribe guitar audio to MIDI + tab data with string+fret positions.

        Args:
            audio_path: Path to guitar audio file (stem)
            output_dir: Output directory for MIDI file
            tempo_hint: Known tempo (optional)
            onset_threshold: Onset detection threshold (default 0.5)

        Returns:
            TrimplexxTranscriptionResult with MIDI path and tab_data
        """
        import librosa
        import pretty_midi

        self._ensure_model()

        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Transcribing guitar (trimplexx): {audio_path.name}")

        # Load audio at 22050 Hz
        audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        duration = len(audio) / sr

        # Compute CQT spectrogram (MUST match training exactly)
        cqt = librosa.cqt(
            y=audio,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            fmin=FMIN_CQT,
            n_bins=N_BINS_CQT,
            bins_per_octave=BINS_PER_OCTAVE_CQT,
        )
        log_cqt = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)  # (n_bins, n_frames)

        # Run inference
        onset_probs, fret_indices = self._infer(log_cqt)

        # Extract notes with string+fret positions
        onset_binary = (onset_probs > onset_threshold).astype(np.float32)
        notes = _frames_to_notes(onset_binary, fret_indices, HOP_LENGTH, SAMPLE_RATE)

        logger.info(f"Extracted {len(notes)} guitar notes with string+fret positions")

        if not notes:
            return TrimplexxTranscriptionResult(
                midi_path=None, num_notes=0, quality_score=0.0,
                method='trimplexx', pitch_range=(0, 0), polyphony_avg=0.0,
                tab_data=[],
            )

        # Determine tempo
        if tempo_hint and 40 < tempo_hint < 300:
            tempo = tempo_hint
        else:
            try:
                tempo_result = librosa.beat.beat_track(y=audio, sr=SAMPLE_RATE)
                if hasattr(tempo_result[0], '__len__'):
                    tempo = float(tempo_result[0][0])
                else:
                    tempo = float(tempo_result[0])
                tempo = max(40.0, min(300.0, tempo))
            except Exception:
                tempo = 120.0

        # Generate MIDI (for pipeline compatibility)
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        guitar_track = pretty_midi.Instrument(
            program=25, is_drum=False, name='Steel String Guitar'
        )

        pitches = []
        for note in notes:
            guitar_track.notes.append(pretty_midi.Note(
                velocity=100,
                pitch=note['pitch_midi'],
                start=note['start_time'],
                end=note['end_time'],
            ))
            pitches.append(note['pitch_midi'])

        midi.instruments.append(guitar_track)

        midi_filename = f"{audio_path.stem}_guitar.mid"
        midi_path = output_dir / midi_filename
        midi.write(str(midi_path))

        # Compute quality metrics
        polyphony = self._compute_polyphony(notes, duration)
        pitch_range = (min(pitches), max(pitches))
        pitch_span = pitch_range[1] - pitch_range[0]

        quality = 0.0
        quality += 0.2 * min(1.0, len(notes) / (duration * 3))
        quality += 0.2 * min(1.0, pitch_span / 30)
        quality += 0.2 if 1.0 < polyphony < 6.0 else 0.1
        quality += 0.15 if len(notes) > 15 else 0.0
        quality += 0.1 if pitch_range[0] < 55 and pitch_range[1] > 60 else 0.0
        quality += 0.15  # bonus for trimplexx having string+fret directly
        quality = min(1.0, quality)

        logger.info(f"Trimplexx MIDI: {midi_path.name}, {len(notes)} notes, "
                    f"range={pitch_range}, polyphony={polyphony:.1f}, quality={quality:.2f}")

        return TrimplexxTranscriptionResult(
            midi_path=str(midi_path),
            num_notes=len(notes),
            quality_score=quality,
            method='trimplexx',
            pitch_range=pitch_range,
            polyphony_avg=polyphony,
            tab_data=notes,
        )

    def _infer(self, log_cqt: np.ndarray):
        """
        Run model inference on a CQT spectrogram.

        Args:
            log_cqt: (n_bins, n_frames) log-amplitude CQT spectrogram

        Returns:
            onset_probs: (n_frames, 6) onset probabilities per string
            fret_indices: (n_frames, 6) predicted fret index per string
        """
        # Model input: (batch, n_bins, n_frames)
        cqt_tensor = torch.from_numpy(log_cqt).unsqueeze(0).float().to(self._device)

        with torch.no_grad():
            onset_logits, fret_logits = self._model(cqt_tensor)
            # onset_logits: (1, reduced_frames, 6)
            # fret_logits: (1, reduced_frames, 6, 22)

        onset_probs = torch.sigmoid(onset_logits[0]).cpu().numpy()  # (frames, 6)
        fret_indices = torch.argmax(fret_logits[0], dim=-1).cpu().numpy()  # (frames, 6)

        return onset_probs, fret_indices

    def _compute_polyphony(self, notes, duration):
        """Compute average polyphony (simultaneous notes)."""
        if not notes or duration <= 0:
            return 0.0

        # Sample at 100 points through the audio
        sample_times = np.linspace(0, duration, min(100, int(duration * 10)))
        active_counts = []
        for t in sample_times:
            count = sum(1 for n in notes if n['start_time'] <= t < n['end_time'])
            if count > 0:
                active_counts.append(count)

        return float(np.mean(active_counts)) if active_counts else 0.0


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_transcriber: Optional[TrimplexxTranscriber] = None


def transcribe_trimplexx(audio_path: str, output_dir: str,
                         tempo_hint: float = None) -> Optional[str]:
    """Convenience function: returns MIDI path or None."""
    global _transcriber

    if _transcriber is None:
        _transcriber = TrimplexxTranscriber()

    try:
        result = _transcriber.transcribe(audio_path, output_dir, tempo_hint)
        if result.midi_path and result.quality_score > 0.2:
            return result.midi_path
        return None
    except Exception as e:
        logger.error(f"Trimplexx transcription failed: {e}")
        return None


def is_available() -> bool:
    return MODEL_AVAILABLE


TRIMPLEXX_MODEL_AVAILABLE = MODEL_AVAILABLE
