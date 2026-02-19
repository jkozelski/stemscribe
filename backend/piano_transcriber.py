"""
Neural Piano Transcriber for StemScribe
=======================================
Uses a trained CRNN (Mel -> Conv2d -> BiLSTM -> onset/frame/velocity)
to transcribe piano audio into MIDI with 88-key polyphonic output.

Architecture: PianoTranscriptionModel (88 keys, onset-conditioned frames)
Checkpoint: backend/models/pretrained/best_piano_model.pt
Training: 50 epochs on MAESTRO v3.0.0 (198 hours, Disklavier)

Piano Range: A0 (MIDI 21) to C8 (MIDI 108) — 88 keys
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS — MUST match training CONFIG exactly
# ============================================================================

CHECKPOINT_PATH = Path(__file__).parent / 'models' / 'pretrained' / 'best_piano_model.pt'
MODEL_AVAILABLE = CHECKPOINT_PATH.exists()

SAMPLE_RATE = 16000
HOP_LENGTH = 256
N_MELS = 229
N_FFT = 2048
FMIN = 30.0
FMAX = 8000.0
CHUNK_DURATION = 5.0
NUM_KEYS = 88
MIN_MIDI = 21   # A0
MAX_MIDI = 108  # C8

CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)
CHUNK_FRAMES = int(CHUNK_SAMPLES / HOP_LENGTH)

if MODEL_AVAILABLE:
    logger.info(f"Piano transcriber model found ({CHECKPOINT_PATH.stat().st_size / 1e6:.0f}MB)")
else:
    logger.debug(f"Piano model not found at {CHECKPOINT_PATH}")


# ============================================================================
# MODEL (must match training notebook exactly)
# ============================================================================

class PianoTranscriptionModel(nn.Module):
    """
    CRNN for piano transcription: Mel → Conv2D → BiLSTM → onset/frame/velocity

    Input:  (batch, 229, time)
    Output: onset    (batch, 88, time)
            frame    (batch, 88, time)
            velocity (batch, 88, time)
    """

    def __init__(self, n_mels=229, num_keys=88,
                 hidden_size=256, num_layers=2, dropout=0.25):
        super().__init__()

        self.num_keys = num_keys

        # 229 → 114 → 57 → 28 → 14
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(dropout),

            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(dropout),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(dropout),

            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(dropout),
        )

        cnn_output_dim = 128 * 14

        self.onset_lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.frame_lstm = nn.LSTM(
            input_size=cnn_output_dim + num_keys,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        lstm_output_dim = hidden_size * 2

        self.onset_head = nn.Linear(lstm_output_dim, num_keys)
        self.frame_head = nn.Linear(lstm_output_dim, num_keys)
        self.velocity_head = nn.Sequential(
            nn.Linear(lstm_output_dim, num_keys),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch, n_mels, time = x.shape

        cnn_out = self.cnn(x.unsqueeze(1))
        cnn_features = cnn_out.permute(0, 3, 1, 2)
        cnn_features = cnn_features.reshape(batch, time, -1)

        onset_out, _ = self.onset_lstm(cnn_features)
        onset_logits = self.onset_head(onset_out)
        onset_pred = torch.sigmoid(onset_logits)

        frame_input = torch.cat([cnn_features, onset_pred.detach()], dim=-1)
        frame_out, _ = self.frame_lstm(frame_input)
        frame_logits = self.frame_head(frame_out)
        frame_pred = torch.sigmoid(frame_logits)

        velocity_pred = self.velocity_head(onset_out)

        return (
            onset_pred.permute(0, 2, 1),
            frame_pred.permute(0, 2, 1),
            velocity_pred.permute(0, 2, 1),
        )


# ============================================================================
# RESULT DATACLASS
# ============================================================================

@dataclass
class PianoTranscriptionResult:
    midi_path: Optional[str]
    num_notes: int
    quality_score: float
    method: str
    pitch_range: tuple     # (min_midi, max_midi)
    polyphony_avg: float   # average simultaneous notes


# ============================================================================
# TRANSCRIBER
# ============================================================================

class PianoTranscriber:
    """
    Piano transcriber using the trained CRNN model.
    Lazy-loads model on first use.
    """

    def __init__(self):
        self._model = None
        self._device = None

    def _ensure_model(self):
        if self._model is not None:
            return

        if not MODEL_AVAILABLE:
            raise RuntimeError(f"Piano model not found at {CHECKPOINT_PATH}")

        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=self._device, weights_only=False)
        self._model = PianoTranscriptionModel()
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.to(self._device)
        self._model.eval()

        epoch = checkpoint.get('epoch', '?')
        val_loss = checkpoint.get('val_loss', '?')
        logger.info(f"Piano model loaded on {self._device} "
                    f"(epoch={epoch}, val_loss={val_loss})")

    def transcribe(self, audio_path: str, output_dir: str,
                   tempo_hint: float = None) -> PianoTranscriptionResult:
        """
        Transcribe piano audio to MIDI.

        Args:
            audio_path: Path to piano audio file
            output_dir: Output directory for MIDI
            tempo_hint: Known tempo (optional)

        Returns:
            PianoTranscriptionResult
        """
        import librosa
        import pretty_midi

        self._ensure_model()

        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Transcribing piano: {audio_path.name}")

        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        duration = len(audio) / sr

        # Compute mel spectrogram (MUST match training)
        mel = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE,
            n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        )
        mel = np.log(mel + 1e-8)

        total_frames = mel.shape[1]

        # Overlap-add inference
        onset_pred, frame_pred, velocity_pred = self._infer_overlap(mel, total_frames)

        # Extract notes
        notes = self._extract_notes(onset_pred, frame_pred, velocity_pred)
        logger.info(f"Extracted {len(notes)} piano notes")

        if not notes:
            return PianoTranscriptionResult(
                midi_path=None, num_notes=0, quality_score=0.0,
                method='piano_nn_model', pitch_range=(0, 0), polyphony_avg=0.0,
            )

        # Tempo
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
        frames_per_sec = SAMPLE_RATE / HOP_LENGTH
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        piano_track = pretty_midi.Instrument(program=0, is_drum=False, name='Acoustic Grand Piano')

        pitches = []
        for note in notes:
            start_time = note['onset_frame'] / frames_per_sec
            end_time = note['offset_frame'] / frames_per_sec
            midi_pitch = note['pitch']
            velocity = note['velocity']

            piano_track.notes.append(pretty_midi.Note(
                velocity=velocity,
                pitch=midi_pitch,
                start=start_time,
                end=end_time,
            ))
            pitches.append(midi_pitch)

        midi.instruments.append(piano_track)

        midi_filename = f"{audio_path.stem}_piano.mid"
        midi_path = output_dir / midi_filename
        midi.write(str(midi_path))

        # Compute polyphony
        polyphony = self._compute_polyphony(notes, total_frames)

        # Quality heuristic
        pitch_range = (min(pitches), max(pitches))
        pitch_span = pitch_range[1] - pitch_range[0]

        quality = 0.0
        quality += 0.2 * min(1.0, len(notes) / (duration * 3))
        quality += 0.2 * min(1.0, pitch_span / 40)
        quality += 0.2 if 1.0 < polyphony < 8.0 else 0.1
        quality += 0.15 if len(notes) > 20 else 0.0
        quality += 0.1 if pitch_range[0] < 50 and pitch_range[1] > 70 else 0.0
        quality += 0.15
        quality = min(1.0, quality)

        logger.info(f"Piano MIDI: {midi_path.name}, {len(notes)} notes, "
                    f"range={pitch_range}, polyphony={polyphony:.1f}, quality={quality:.2f}")

        return PianoTranscriptionResult(
            midi_path=str(midi_path),
            num_notes=len(notes),
            quality_score=quality,
            method='piano_nn_model',
            pitch_range=pitch_range,
            polyphony_avg=polyphony,
        )

    def _infer_overlap(self, mel, total_frames):
        """Run model with overlap-add for long audio."""
        if total_frames <= CHUNK_FRAMES:
            mel_tensor = torch.from_numpy(mel).unsqueeze(0).float().to(self._device)
            if mel_tensor.shape[2] < CHUNK_FRAMES:
                pad = CHUNK_FRAMES - mel_tensor.shape[2]
                mel_tensor = torch.nn.functional.pad(mel_tensor, (0, pad))

            with torch.no_grad():
                onset, frame, vel = self._model(mel_tensor)

            return (
                onset[0].cpu().numpy()[:, :total_frames],
                frame[0].cpu().numpy()[:, :total_frames],
                vel[0].cpu().numpy()[:, :total_frames],
            )

        # Overlap-add
        overlap = CHUNK_FRAMES // 2
        hop = CHUNK_FRAMES - overlap
        window = np.hanning(CHUNK_FRAMES)

        onset_acc = np.zeros((NUM_KEYS, total_frames), dtype=np.float32)
        frame_acc = np.zeros((NUM_KEYS, total_frames), dtype=np.float32)
        vel_acc = np.zeros((NUM_KEYS, total_frames), dtype=np.float32)
        weight_acc = np.zeros(total_frames, dtype=np.float32)

        start = 0
        while start < total_frames:
            end = min(start + CHUNK_FRAMES, total_frames)
            chunk = mel[:, start:end]

            if chunk.shape[1] < CHUNK_FRAMES:
                pad = CHUNK_FRAMES - chunk.shape[1]
                chunk = np.pad(chunk, ((0, 0), (0, pad)))

            chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).float().to(self._device)
            with torch.no_grad():
                o, f, v = self._model(chunk_tensor)

            actual_len = min(CHUNK_FRAMES, total_frames - start)
            w = window[:actual_len]

            onset_acc[:, start:start + actual_len] += o[0].cpu().numpy()[:, :actual_len] * w
            frame_acc[:, start:start + actual_len] += f[0].cpu().numpy()[:, :actual_len] * w
            vel_acc[:, start:start + actual_len] += v[0].cpu().numpy()[:, :actual_len] * w
            weight_acc[start:start + actual_len] += w

            start += hop

        weight_acc = np.maximum(weight_acc, 1e-8)
        return onset_acc / weight_acc, frame_acc / weight_acc, vel_acc / weight_acc

    def _extract_notes(self, onset_pred, frame_pred, velocity_pred,
                       onset_threshold=0.5, frame_threshold=0.3,
                       min_frames=2):
        """Extract notes using onset-triggered, frame-sustained logic."""
        notes = []
        num_keys, total_frames = onset_pred.shape

        for k in range(num_keys):
            midi_pitch = k + MIN_MIDI
            t = 0
            while t < total_frames:
                if onset_pred[k, t] > onset_threshold:
                    onset_frame = t
                    vel = velocity_pred[k, t]
                    velocity = int(40 + 87 * min(1.0, vel))

                    t += 1
                    while t < total_frames and frame_pred[k, t] > frame_threshold:
                        t += 1

                    offset_frame = t
                    if offset_frame - onset_frame >= min_frames:
                        notes.append({
                            'pitch': midi_pitch,
                            'onset_frame': onset_frame,
                            'offset_frame': offset_frame,
                            'velocity': velocity,
                        })
                else:
                    t += 1

        notes.sort(key=lambda n: n['onset_frame'])
        return notes

    def _compute_polyphony(self, notes, total_frames):
        """Compute average polyphony (simultaneous notes)."""
        if not notes:
            return 0.0

        active = np.zeros(total_frames, dtype=np.int32)
        for note in notes:
            active[note['onset_frame']:note['offset_frame']] += 1

        active_frames = active[active > 0]
        return float(np.mean(active_frames)) if len(active_frames) > 0 else 0.0


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_transcriber: Optional[PianoTranscriber] = None


def transcribe_piano(audio_path: str, output_dir: str,
                     tempo_hint: float = None) -> Optional[str]:
    """Convenience function: returns MIDI path or None."""
    global _transcriber

    if _transcriber is None:
        _transcriber = PianoTranscriber()

    try:
        result = _transcriber.transcribe(audio_path, output_dir, tempo_hint)
        if result.midi_path and result.quality_score > 0.2:
            return result.midi_path
        return None
    except Exception as e:
        logger.error(f"Piano transcription failed: {e}")
        return None


def is_available() -> bool:
    return MODEL_AVAILABLE


PIANO_MODEL_AVAILABLE = MODEL_AVAILABLE
