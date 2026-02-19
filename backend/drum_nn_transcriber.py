"""
Neural Drum Transcriber for StemScribe
======================================
Uses a trained OaF-style CRNN (Mel -> Conv2d -> BiLSTM -> onset/frame/velocity)
to transcribe drum audio into MIDI with 8 drum classes and velocity dynamics.

Architecture: DrumTranscriptionModel (OaF-style, 128-bin Mel input, 8-class output)
Checkpoint: backend/models/pretrained/best_drum_model.pt
Training: 50 epochs on E-GMD (444 hours), best val_loss 0.0335 at epoch 50

Drum Classes (General MIDI mapping):
  0: kick       -> GM 36 (Bass Drum 1)
  1: snare      -> GM 38 (Acoustic Snare)
  2: hihat_closed -> GM 42 (Closed Hi-Hat)
  3: hihat_open -> GM 46 (Open Hi-Hat)
  4: tom_low    -> GM 45 (Low Tom)
  5: tom_high   -> GM 50 (High Tom)
  6: crash      -> GM 49 (Crash Cymbal 1)
  7: ride       -> GM 51 (Ride Cymbal 1)
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# PATHS & AVAILABILITY
# ============================================================================

CHECKPOINT_PATH = Path(__file__).parent / 'models' / 'pretrained' / 'best_drum_model.pt'
MODEL_AVAILABLE = CHECKPOINT_PATH.exists()

if MODEL_AVAILABLE:
    logger.info(f"Neural drum model available "
                f"(checkpoint: {CHECKPOINT_PATH.stat().st_size / 1e6:.1f}MB)")
else:
    logger.warning(f"Neural drum model: checkpoint not found at {CHECKPOINT_PATH}")


# ============================================================================
# CONSTANTS (must match training exactly)
# ============================================================================

SAMPLE_RATE = 16000
HOP_LENGTH = 256
N_MELS = 128
N_FFT = 2048
NUM_CLASSES = 8

CHUNK_DURATION = 5.0                                # Training chunk size
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)   # 80000
CHUNK_FRAMES = CHUNK_SAMPLES // HOP_LENGTH           # ~312

# Drum class names and GM MIDI mappings
CLASS_NAMES = ['kick', 'snare', 'hihat_closed', 'hihat_open',
               'tom_low', 'tom_high', 'crash', 'ride']

CLASS_TO_GM = {
    0: 36,   # kick       -> Bass Drum 1
    1: 38,   # snare      -> Acoustic Snare
    2: 42,   # hihat_closed -> Closed Hi-Hat
    3: 46,   # hihat_open -> Open Hi-Hat
    4: 45,   # tom_low    -> Low Tom
    5: 50,   # tom_high   -> High Tom
    6: 49,   # crash      -> Crash Cymbal 1
    7: 51,   # ride       -> Ride Cymbal 1
}


# ============================================================================
# MODEL ARCHITECTURE (must match training notebook cell-8 exactly)
# ============================================================================

class DrumTranscriptionModel(nn.Module):
    """Onsets and Frames style model for drum transcription.

    Input:  Mel spectrogram (batch, 128, time)
    Output: onset_pred  (batch, 8, time) — onset probabilities per class
            frame_pred  (batch, 8, time) — frame activations per class
            velocity_pred (batch, 8, time) — velocity estimates per class
    """

    def __init__(self, n_mels=128, num_classes=8):
        super().__init__()

        # CNN encoder — pools frequency only, preserves time
        self.conv_stack = nn.Sequential(
            # Block 1: n_mels -> n_mels/2
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),       # 128 -> 64 freq, time unchanged
            nn.Dropout(0.25),

            # Block 2: n_mels/2 -> n_mels/4
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),       # 64 -> 32 freq, time unchanged
            nn.Dropout(0.25),

            # Block 3: no further pooling
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        # After conv: (batch, 128, n_mels//4, time) = (batch, 128, 32, time)
        self.flat_size = 128 * (n_mels // 4)    # 4096

        # Onset detection branch
        self.onset_lstm = nn.LSTM(
            self.flat_size, 128, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3
        )
        self.onset_fc = nn.Linear(256, num_classes)

        # Frame detection branch (conditioned on onset predictions)
        self.frame_lstm = nn.LSTM(
            self.flat_size + num_classes, 128, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3
        )
        self.frame_fc = nn.Linear(256, num_classes)

        # Velocity prediction (from onset features)
        self.velocity_fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (batch, n_mels=128, time)
        x = x.unsqueeze(1)                     # (batch, 1, 128, time)

        # CNN
        x = self.conv_stack(x)                 # (batch, 128, 32, time)

        # Reshape for LSTM: (batch, time, features)
        batch, channels, freq, time = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch, time, -1)    # (batch, time, 4096)

        # Onset detection
        onset_features, _ = self.onset_lstm(x)          # (batch, time, 256)
        onset_pred = torch.sigmoid(self.onset_fc(onset_features))   # (batch, time, 8)

        # Frame detection (with onset info)
        frame_input = torch.cat([x, onset_pred], dim=-1)            # (batch, time, 4096+8)
        frame_features, _ = self.frame_lstm(frame_input)            # (batch, time, 256)
        frame_pred = torch.sigmoid(self.frame_fc(frame_features))   # (batch, time, 8)

        # Velocity (from onset features)
        velocity_pred = torch.sigmoid(self.velocity_fc(onset_features))  # (batch, time, 8)

        # Permute to (batch, classes, time)
        onset_pred = onset_pred.permute(0, 2, 1)
        frame_pred = frame_pred.permute(0, 2, 1)
        velocity_pred = velocity_pred.permute(0, 2, 1)

        return onset_pred, frame_pred, velocity_pred


# ============================================================================
# MODEL LOADING
# ============================================================================

def _load_model(device: torch.device = None) -> DrumTranscriptionModel:
    """Load the Drum Transcription model from checkpoint."""
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )

    model = DrumTranscriptionModel(n_mels=N_MELS, num_classes=NUM_CLASSES)

    logger.info(f"Loading drum model from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', '?')
        val_loss = checkpoint.get('val_loss', '?')
        logger.info(f"Checkpoint: epoch={epoch}, val_loss={val_loss}")
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Drum model loaded on {device} ({param_count / 1e6:.1f}M params)")
    return model


# ============================================================================
# RESULT DATACLASS
# ============================================================================

@dataclass
class DrumTranscriptionResult:
    """Result of neural drum transcription."""
    midi_path: Optional[str]
    total_hits: int
    hits_by_type: Dict[str, int]
    quality_score: float            # 0.0 - 1.0
    method: str                     # 'drum_nn_model'
    tempo: Optional[float]


# ============================================================================
# TRANSCRIBER CLASS
# ============================================================================

class NeuralDrumTranscriber:
    """
    Transcribes drum audio to MIDI using the trained OaF-style CRNN model.

    Outputs per-frame onset/frame/velocity predictions for 8 drum classes,
    extracts hits, and writes drum MIDI (channel 10, is_drum=True).

    Usage:
        transcriber = NeuralDrumTranscriber()
        result = transcriber.transcribe('drums.wav', '/output/dir')
    """

    def __init__(self):
        self._model = None
        self._device = None

    def _ensure_model(self):
        """Lazy-load model on first use."""
        if self._model is None:
            self._device = torch.device(
                'cuda' if torch.cuda.is_available() else
                'mps' if torch.backends.mps.is_available() else 'cpu'
            )
            self._model = _load_model(self._device)

    # ----------------------------------------------------------------
    # Mel Spectrogram Feature Extraction
    # ----------------------------------------------------------------

    def _compute_mel(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute log Mel spectrogram matching training preprocessing.

        Training uses torchaudio.transforms.MelSpectrogram then log(x + 1e-8).

        Args:
            audio: Mono audio at SAMPLE_RATE (16000 Hz)

        Returns:
            np.ndarray of shape (128, time_frames), log-scaled Mel
        """
        import torchaudio

        audio_tensor = torch.from_numpy(audio).float()
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )
        mel_spec = mel_transform(audio_tensor)
        mel_spec = torch.log(mel_spec + 1e-8)

        return mel_spec.numpy()     # shape: (128, time_frames)

    # ----------------------------------------------------------------
    # Overlap-Add Inference
    # ----------------------------------------------------------------

    def _infer_full_audio(self, mel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run model inference with overlap-add for full-song Mel spectrogram.

        Args:
            mel: Full Mel spectrogram, shape (128, total_frames)

        Returns:
            onset_pred:   (8, total_frames)
            frame_pred:   (8, total_frames)
            velocity_pred: (8, total_frames)
        """
        total_frames = mel.shape[1]

        # Short audio — process in one pass
        if total_frames <= CHUNK_FRAMES:
            x = torch.from_numpy(mel).float().unsqueeze(0).to(self._device)
            with torch.no_grad():
                onset, frame, velocity = self._model(x)
            return (onset[0].cpu().numpy(),
                    frame[0].cpu().numpy(),
                    velocity[0].cpu().numpy())

        # Overlap-add for long audio
        overlap_ratio = 0.5
        hop_frames = int(CHUNK_FRAMES * (1 - overlap_ratio))

        num_chunks = max(1, (total_frames - CHUNK_FRAMES) // hop_frames + 1)
        pad_needed = (num_chunks - 1) * hop_frames + CHUNK_FRAMES - total_frames
        if pad_needed > 0:
            mel_padded = np.pad(mel, ((0, 0), (0, pad_needed)), mode='constant')
        else:
            mel_padded = mel
            pad_needed = 0

        padded_frames = mel_padded.shape[1]

        # Hann window for crossfade
        window = np.hanning(CHUNK_FRAMES).astype(np.float32)

        # Accumulators
        onset_acc = np.zeros((NUM_CLASSES, padded_frames), dtype=np.float32)
        frame_acc = np.zeros((NUM_CLASSES, padded_frames), dtype=np.float32)
        vel_acc = np.zeros((NUM_CLASSES, padded_frames), dtype=np.float32)
        weight_acc = np.zeros(padded_frames, dtype=np.float32)

        for i in range(num_chunks + 1):
            start = i * hop_frames
            end = start + CHUNK_FRAMES

            if start >= padded_frames:
                break

            # Extract chunk
            if end > padded_frames:
                chunk_mel = mel_padded[:, start:]
                pad_right = CHUNK_FRAMES - chunk_mel.shape[1]
                if pad_right > 0:
                    chunk_mel = np.pad(chunk_mel, ((0, 0), (0, pad_right)))
            else:
                chunk_mel = mel_padded[:, start:end]

            # Model forward pass
            x = torch.from_numpy(chunk_mel).float().unsqueeze(0).to(self._device)
            with torch.no_grad():
                onset_chunk, frame_chunk, vel_chunk = self._model(x)

            onset_np = onset_chunk[0].cpu().numpy()     # (8, time)
            frame_np = frame_chunk[0].cpu().numpy()
            vel_np = vel_chunk[0].cpu().numpy()

            out_t = onset_np.shape[1]
            actual_len = min(out_t, padded_frames - start)
            w = window[:actual_len]

            onset_acc[:, start:start + actual_len] += onset_np[:, :actual_len] * w
            frame_acc[:, start:start + actual_len] += frame_np[:, :actual_len] * w
            vel_acc[:, start:start + actual_len] += vel_np[:, :actual_len] * w
            weight_acc[start:start + actual_len] += w

        # Normalize
        weight_acc = np.maximum(weight_acc, 1e-8)
        onset_acc /= weight_acc
        frame_acc /= weight_acc
        vel_acc /= weight_acc

        return (onset_acc[:, :total_frames],
                frame_acc[:, :total_frames],
                vel_acc[:, :total_frames])

    # ----------------------------------------------------------------
    # Hit Extraction
    # ----------------------------------------------------------------

    def _extract_hits(self, onset_pred: np.ndarray, velocity_pred: np.ndarray,
                      onset_threshold: float = 0.5,
                      min_velocity: int = 20) -> List[dict]:
        """
        Extract drum hits from onset predictions.

        Unlike guitar/piano transcription, drums are percussive events
        (no sustain), so we only need onset detection — no frame tracking.

        Args:
            onset_pred:     (8, time) onset probabilities per class
            velocity_pred:  (8, time) velocity estimates per class
            onset_threshold: Minimum onset probability to detect a hit
            min_velocity:   Minimum MIDI velocity (0-127) to include

        Returns:
            List of hit dicts sorted by time
        """
        hits = []
        num_classes, num_frames = onset_pred.shape

        for class_idx in range(num_classes):
            onset_row = onset_pred[class_idx, :]
            vel_row = velocity_pred[class_idx, :]

            # Find peaks above threshold
            for t in range(num_frames):
                if onset_row[t] > onset_threshold:
                    # Check it's a local peak (not middle of a plateau)
                    if t > 0 and onset_row[t] <= onset_row[t - 1]:
                        continue

                    velocity = int(vel_row[t] * 127)
                    velocity = max(min_velocity, min(127, velocity))

                    if velocity >= min_velocity:
                        hits.append({
                            'class_idx': class_idx,
                            'class_name': CLASS_NAMES[class_idx],
                            'midi_note': CLASS_TO_GM[class_idx],
                            'frame': t,
                            'velocity': velocity,
                            'onset_prob': float(onset_row[t]),
                        })

        # Sort by time
        hits.sort(key=lambda h: h['frame'])
        return hits

    # ----------------------------------------------------------------
    # MIDI Generation
    # ----------------------------------------------------------------

    def _hits_to_midi(self, hits: List[dict],
                      tempo: float = 120.0) -> 'pretty_midi.PrettyMIDI':
        """
        Convert drum hits to MIDI.

        Drum notes use GM channel 10, is_drum=True.
        Each hit is a short note (~50ms) since drums are percussive.
        """
        import pretty_midi

        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        drum_track = pretty_midi.Instrument(program=0, is_drum=True, name='Drums')

        frames_per_sec = SAMPLE_RATE / HOP_LENGTH   # 62.5

        for hit in hits:
            start_time = hit['frame'] / frames_per_sec
            end_time = start_time + 0.05    # 50ms note duration for drums

            note = pretty_midi.Note(
                velocity=hit['velocity'],
                pitch=hit['midi_note'],
                start=start_time,
                end=end_time,
            )
            drum_track.notes.append(note)

        midi.instruments.append(drum_track)
        return midi

    # ----------------------------------------------------------------
    # Main Transcription Pipeline
    # ----------------------------------------------------------------

    def transcribe(self, audio_path: str, output_dir: str,
                   sensitivity: float = 0.5,
                   tempo_hint: float = None) -> DrumTranscriptionResult:
        """
        Full drum transcription pipeline.

        1. Load audio at 16000 Hz
        2. Compute Mel spectrogram (128 bins, matching training)
        3. Run model with overlap-add
        4. Extract hits (onset peaks with velocity)
        5. Write drum MIDI

        Args:
            audio_path: Path to drum audio file
            output_dir: Directory for output MIDI file
            sensitivity: Onset detection threshold (0-1, lower = more sensitive)
            tempo_hint: Known tempo from earlier analysis (optional)

        Returns:
            DrumTranscriptionResult with MIDI path and stats
        """
        import librosa

        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        self._ensure_model()

        logger.info(f"Neural drum transcription: {audio_path.name}")

        # Load audio at training sample rate (mono)
        audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE)

        if len(audio) < SAMPLE_RATE:
            logger.warning(f"Audio too short ({len(audio) / SAMPLE_RATE:.1f}s), skipping")
            return DrumTranscriptionResult(
                midi_path=None, total_hits=0, hits_by_type={},
                quality_score=0.0, method='drum_nn_model', tempo=None,
            )

        # Compute Mel spectrogram
        mel = self._compute_mel(audio)   # (128, frames)
        duration = len(audio) / SAMPLE_RATE
        logger.info(f"Mel: {mel.shape[1]} frames ({duration:.1f}s)")

        # Run model inference with overlap-add
        onset_pred, frame_pred, velocity_pred = self._infer_full_audio(mel)

        # Extract drum hits
        onset_threshold = 1.0 - sensitivity     # sensitivity=0.5 -> threshold=0.5
        hits = self._extract_hits(onset_pred, velocity_pred,
                                  onset_threshold=onset_threshold)
        logger.info(f"Extracted {len(hits)} drum hits")

        if not hits:
            logger.info("No drum hits detected")
            return DrumTranscriptionResult(
                midi_path=None, total_hits=0, hits_by_type={},
                quality_score=0.0, method='drum_nn_model', tempo=None,
            )

        # Count hits by type
        hits_by_type = {}
        for hit in hits:
            name = hit['class_name']
            hits_by_type[name] = hits_by_type.get(name, 0) + 1

        # Tempo detection
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

        # Generate MIDI
        midi_obj = self._hits_to_midi(hits, tempo)
        midi_filename = f"{audio_path.stem}_drums.mid"
        midi_path = output_dir / midi_filename
        midi_obj.write(str(midi_path))

        # Quality heuristic
        hit_density = len(hits) / duration
        num_types = len(hits_by_type)
        quality = 0.0
        quality += 0.25 * min(1.0, len(hits) / (duration * 3))  # Hit density
        quality += 0.25 * min(1.0, num_types / 4)                # Class diversity
        quality += 0.2 if 1.0 < hit_density < 20.0 else 0.0      # Sane density
        quality += 0.15 if 'kick' in hits_by_type else 0.0        # Has kick
        quality += 0.15                                            # Base neural confidence
        quality = min(1.0, quality)

        logger.info(f"Drum MIDI: {midi_path.name}, {len(hits)} hits, "
                    f"types={hits_by_type}, "
                    f"tempo={tempo:.0f}, quality={quality:.2f}")

        return DrumTranscriptionResult(
            midi_path=str(midi_path),
            total_hits=len(hits),
            hits_by_type=hits_by_type,
            quality_score=quality,
            method='drum_nn_model',
            tempo=tempo,
        )


# ============================================================================
# CONVENIENCE FUNCTIONS (matches guitar_tab_transcriber.py pattern)
# ============================================================================

_transcriber: Optional[NeuralDrumTranscriber] = None


def transcribe_drums_nn(audio_path: str, output_dir: str,
                        sensitivity: float = 0.5,
                        tempo_hint: float = None) -> Optional[str]:
    """
    Convenience function: transcribe drum audio to MIDI using neural model.

    Uses a cached singleton transcriber.

    Args:
        audio_path: Path to drum audio file
        output_dir: Output directory for MIDI
        sensitivity: Detection sensitivity (0-1)
        tempo_hint: Known tempo (optional)

    Returns:
        MIDI file path, or None if transcription failed
    """
    global _transcriber

    if _transcriber is None:
        _transcriber = NeuralDrumTranscriber()

    try:
        result = _transcriber.transcribe(
            audio_path=audio_path,
            output_dir=output_dir,
            sensitivity=sensitivity,
            tempo_hint=tempo_hint,
        )
        if result.midi_path and result.quality_score > 0.2:
            return result.midi_path
        return None
    except Exception as e:
        logger.error(f"Neural drum transcription failed: {e}")
        return None


def is_available() -> bool:
    """Check if neural drum model is available."""
    return MODEL_AVAILABLE


# Alias for consistent naming in app.py
DRUM_NN_MODEL_AVAILABLE = MODEL_AVAILABLE


# ============================================================================
# CLI TEST
# ============================================================================

if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.INFO)

    print("Neural Drum Transcriber")
    print(f"  Model available: {MODEL_AVAILABLE}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Sample rate: {SAMPLE_RATE}")
    print(f"  Mel bins: {N_MELS}")
    print(f"  Classes: {CLASS_NAMES}")

    if len(sys.argv) >= 3:
        audio_path = sys.argv[1]
        output_dir = sys.argv[2]

        transcriber = NeuralDrumTranscriber()
        result = transcriber.transcribe(audio_path, output_dir)
        print(f"\nResult: {result}")
