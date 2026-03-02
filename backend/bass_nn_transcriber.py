"""
Neural Bass Transcriber v3 for StemScribe
==========================================
Uses a trained transfer-learning model (Piano CNN + Adapter + BiLSTM)
to transcribe bass audio into MIDI with 40-key output.

Architecture: BassTranscriptionModel_v3 (simpler than guitar -- no attention)
  Mel(229) -> Piano CNN (frozen) -> Adapter(1792->1024) + Residual
  -> onset_lstm BiLSTM(1024, 128) -> onset_head Linear(256, 40)
  -> frame_lstm BiLSTM(1024+40, 128) -> frame_head Linear(256, 40)
  -> velocity_head Linear(256, 40) + Sigmoid

Checkpoint: backend/models/pretrained/best_bass_v3_model.pt
Training: train_bass_model/train_bass_v3.py (Slakh2100, transfer from piano CNN)

Bass Range: E1 (MIDI 28) to G4 (MIDI 67) -- 40 keys

Falls back to Basic Pitch bass_transcriber when no v3 model exists.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS -- MUST match training CONFIG exactly
# ============================================================================

CHECKPOINT_PATH = Path(__file__).parent / 'models' / 'pretrained' / 'best_bass_v3_model.pt'
PIANO_CHECKPOINT_PATH = Path(__file__).parent / 'models' / 'pretrained' / 'best_piano_model.pt'
MODEL_AVAILABLE = CHECKPOINT_PATH.exists() and PIANO_CHECKPOINT_PATH.exists()

SAMPLE_RATE = 16000
HOP_LENGTH = 256
N_MELS = 229
N_FFT = 2048
FMIN = 30.0
FMAX = 8000.0
CHUNK_DURATION = 5.0
NUM_KEYS = 40
MIN_MIDI = 28   # E1
MAX_MIDI = 67   # G4

CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)
CHUNK_FRAMES = int(CHUNK_SAMPLES / HOP_LENGTH)

if MODEL_AVAILABLE:
    logger.info(f"Bass v3 model found ({CHECKPOINT_PATH.stat().st_size / 1e6:.0f}MB)")
else:
    logger.debug(f"Bass v3 model not found at {CHECKPOINT_PATH}")


# ============================================================================
# PIANO CNN (must match piano_transcriber.py exactly for weight loading)
# ============================================================================

class PianoTranscriptionModel(nn.Module):
    """Exact copy of piano CNN architecture for checkpoint loading."""

    def __init__(self, n_mels=229, num_keys=88,
                 hidden_size=256, num_layers=2, dropout=0.25):
        super().__init__()
        self.num_keys = num_keys

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
            input_size=cnn_output_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.frame_lstm = nn.LSTM(
            input_size=cnn_output_dim + num_keys, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        lstm_output_dim = hidden_size * 2
        self.onset_head = nn.Linear(lstm_output_dim, num_keys)
        self.frame_head = nn.Linear(lstm_output_dim, num_keys)
        self.velocity_head = nn.Sequential(
            nn.Linear(lstm_output_dim, num_keys), nn.Sigmoid(),
        )

    def forward(self, x):
        batch, n_mels, time = x.shape
        cnn_out = self.cnn(x.unsqueeze(1))
        cnn_features = cnn_out.permute(0, 3, 1, 2).reshape(batch, time, -1)
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
# V3 BASS MODEL (must match train_bass_v3.py exactly -- NO attention)
# ============================================================================

class BassTranscriptionModel_v3(nn.Module):
    """
    Transfer-learned bass transcriber. Simpler than guitar v3:
    no multi-head attention, smaller hidden size (128 vs 192).

    Piano CNN (frozen) -> Adapter(1792->1024) -> BiLSTM(128) -> 40-key output

    Input:  Mel spectrogram (B, 229, T)
    Output: onset_logits    (B, 40, T)
            frame_logits    (B, 40, T)
            velocity_pred   (B, 40, T)
    """

    def __init__(self, num_keys=40, hidden_size=128, num_layers=2, dropout=0.25):
        super().__init__()
        self.num_keys = num_keys

        # Frozen piano CNN (set externally)
        self.cnn = None

        # Domain adapter with residual
        self.adapter = nn.Sequential(
            nn.Linear(1792, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.adapter_residual = nn.Linear(1792, 1024)

        # No attention for bass

        # Onset LSTM
        self.onset_lstm = nn.LSTM(
            input_size=1024, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )

        # Frame LSTM (onset-conditioned)
        self.frame_lstm = nn.LSTM(
            input_size=1024 + num_keys, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )

        lstm_dim = hidden_size * 2  # 256

        # Output heads
        self.onset_head = nn.Linear(lstm_dim, num_keys)
        self.frame_head = nn.Linear(lstm_dim, num_keys)
        self.velocity_head = nn.Sequential(
            nn.Linear(lstm_dim, num_keys),
            nn.Sigmoid(),
        )

        nn.init.constant_(self.onset_head.bias, -3.0)
        nn.init.constant_(self.frame_head.bias, -2.0)

    def forward(self, mel):
        batch, n_mels, time = mel.shape

        # Frozen CNN
        with torch.no_grad():
            cnn_out = self.cnn(mel.unsqueeze(1))

        cnn_features = cnn_out.permute(0, 3, 1, 2).reshape(batch, time, -1)

        # Adapter with residual
        adapted = self.adapter(cnn_features) + self.adapter_residual(cnn_features)

        # Onset
        onset_out, _ = self.onset_lstm(adapted)
        onset_logits = self.onset_head(onset_out)
        onset_pred = torch.sigmoid(onset_logits)

        # Frame (onset-conditioned)
        frame_input = torch.cat([adapted, onset_pred.detach()], dim=-1)
        frame_out, _ = self.frame_lstm(frame_input)
        frame_logits = self.frame_head(frame_out)

        # Velocity
        velocity_pred = self.velocity_head(onset_out)

        return (
            onset_logits.permute(0, 2, 1),   # (B, 40, T) -- raw logits
            frame_logits.permute(0, 2, 1),   # (B, 40, T) -- raw logits
            velocity_pred.permute(0, 2, 1),  # (B, 40, T) -- [0, 1]
        )


# ============================================================================
# RESULT DATACLASS
# ============================================================================

@dataclass
class BassTranscriptionResult:
    midi_path: Optional[str]
    num_notes: int
    quality_score: float
    method: str
    pitch_range: tuple
    polyphony_avg: float


# ============================================================================
# TRANSCRIBER
# ============================================================================

class BassNNTranscriber:
    """
    Bass transcriber using the v3 transfer-learning CRNN model.
    Lazy-loads model on first use. Falls back to Basic Pitch when no model exists.
    """

    def __init__(self):
        self._model = None
        self._device = None

    def _ensure_model(self):
        if self._model is not None:
            return

        if not MODEL_AVAILABLE:
            raise RuntimeError(
                f"Bass v3 model not found at {CHECKPOINT_PATH} "
                f"or piano model not found at {PIANO_CHECKPOINT_PATH}"
            )

        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )

        # Load piano CNN from piano checkpoint
        piano_checkpoint = torch.load(
            PIANO_CHECKPOINT_PATH, map_location=self._device, weights_only=False
        )
        piano_model = PianoTranscriptionModel()
        piano_model.load_state_dict(piano_checkpoint['model_state_dict'])
        piano_cnn = piano_model.cnn
        for param in piano_cnn.parameters():
            param.requires_grad = False
        piano_cnn.eval()

        # Load bass v3 model
        bass_checkpoint = torch.load(
            CHECKPOINT_PATH, map_location=self._device, weights_only=False
        )

        self._model = BassTranscriptionModel_v3(num_keys=NUM_KEYS)
        self._model.cnn = piano_cnn

        self._model.load_state_dict(bass_checkpoint['model_state_dict'])
        self._model.to(self._device)
        self._model.eval()

        epoch = bass_checkpoint.get('epoch', '?')
        val_f1 = bass_checkpoint.get('val_f1', '?')
        logger.info(f"Bass v3 model loaded on {self._device} "
                    f"(epoch={epoch}, val_f1={val_f1})")

    def transcribe(self, audio_path: str, output_dir: str,
                   tempo_hint: float = None) -> BassTranscriptionResult:
        """
        Transcribe bass audio to MIDI.

        Args:
            audio_path: Path to bass audio file
            output_dir: Output directory for MIDI
            tempo_hint: Known tempo (optional)

        Returns:
            BassTranscriptionResult
        """
        import librosa
        import pretty_midi

        self._ensure_model()

        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Transcribing bass (v3 NN): {audio_path.name}")

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
        onset_logits, frame_logits, velocity_pred = self._infer_overlap(mel, total_frames)

        # Apply sigmoid to logits for note extraction
        onset_pred = 1.0 / (1.0 + np.exp(-onset_logits))
        frame_pred = 1.0 / (1.0 + np.exp(-frame_logits))

        # Extract notes
        notes = self._extract_notes(onset_pred, frame_pred, velocity_pred)
        logger.info(f"Extracted {len(notes)} bass notes")

        if not notes:
            return BassTranscriptionResult(
                midi_path=None, num_notes=0, quality_score=0.0,
                method='bass_v3_nn', pitch_range=(0, 0), polyphony_avg=0.0,
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
        bass_track = pretty_midi.Instrument(
            program=33, is_drum=False, name='Electric Bass'
        )

        pitches = []
        for note in notes:
            start_time = note['onset_frame'] / frames_per_sec
            end_time = note['offset_frame'] / frames_per_sec
            midi_pitch = note['pitch']
            velocity = note['velocity']

            bass_track.notes.append(pretty_midi.Note(
                velocity=velocity,
                pitch=midi_pitch,
                start=start_time,
                end=end_time,
            ))
            pitches.append(midi_pitch)

        midi.instruments.append(bass_track)

        midi_filename = f"{audio_path.stem}_bass.mid"
        midi_path = output_dir / midi_filename
        midi.write(str(midi_path))

        # Compute polyphony
        polyphony = self._compute_polyphony(notes, total_frames)

        # Quality heuristic
        pitch_range = (min(pitches), max(pitches))
        pitch_span = pitch_range[1] - pitch_range[0]

        quality = 0.0
        quality += 0.25 * min(1.0, len(notes) / (duration * 2))
        quality += 0.2 * min(1.0, pitch_span / 20)
        quality += 0.2 if polyphony < 2.0 else 0.1   # Bass is mostly monophonic
        quality += 0.15 if len(notes) > 10 else 0.0
        quality += 0.05 if pitch_range[0] < 40 else 0.0
        quality += 0.15
        quality = min(1.0, quality)

        logger.info(f"Bass MIDI: {midi_path.name}, {len(notes)} notes, "
                    f"range={pitch_range}, polyphony={polyphony:.1f}, quality={quality:.2f}")

        return BassTranscriptionResult(
            midi_path=str(midi_path),
            num_notes=len(notes),
            quality_score=quality,
            method='bass_v3_nn',
            pitch_range=pitch_range,
            polyphony_avg=polyphony,
        )

    def _infer_overlap(self, mel, total_frames):
        """Run model with overlap-add for long audio. Returns raw logits for onset/frame."""
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
                       min_frames=3):
        """Extract notes using onset-triggered, frame-sustained logic.
        min_frames=3 for bass (longer minimum note than guitar)."""
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

_transcriber: Optional[BassNNTranscriber] = None


def transcribe_bass_nn(audio_path: str, output_dir: str,
                       tempo_hint: float = None) -> Optional[str]:
    """Convenience function: returns MIDI path or None."""
    global _transcriber

    if _transcriber is None:
        _transcriber = BassNNTranscriber()

    try:
        result = _transcriber.transcribe(audio_path, output_dir, tempo_hint)
        if result.midi_path and result.quality_score > 0.2:
            return result.midi_path
        return None
    except Exception as e:
        logger.error(f"Bass v3 transcription failed: {e}")
        return None


def is_available() -> bool:
    return MODEL_AVAILABLE


BASS_NN_MODEL_AVAILABLE = MODEL_AVAILABLE
