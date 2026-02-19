"""
Neural Bass Transcriber for StemScribe
======================================
Uses a trained CRNN (CQT -> Conv2d -> BiLSTM -> onset/frame/velocity)
to transcribe bass audio into MIDI with per-string/fret predictions.

Architecture: BassTranscriptionModel (4 strings × 24 frets)
Checkpoint: backend/models/pretrained/best_bass_model.pt
Training: 50 epochs on Slakh2100 bass stems

Bass Tuning (Standard 4-string):
  String 0: E1 (MIDI 28)
  String 1: A1 (MIDI 33)
  String 2: D2 (MIDI 38)
  String 3: G2 (MIDI 43)
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
# CONSTANTS — MUST match training CONFIG exactly
# ============================================================================

CHECKPOINT_PATH = Path(__file__).parent / 'models' / 'pretrained' / 'best_bass_model.pt'
MODEL_AVAILABLE = CHECKPOINT_PATH.exists()

SAMPLE_RATE = 22050
HOP_LENGTH = 256
N_BINS = 84
BINS_PER_OCTAVE = 12
CHUNK_DURATION = 5.0
NUM_STRINGS = 4
NUM_FRETS = 24
TUNING = [28, 33, 38, 43]  # E1, A1, D2, G2

CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)
CHUNK_FRAMES = int(CHUNK_SAMPLES / HOP_LENGTH)

if MODEL_AVAILABLE:
    logger.info(f"Bass transcriber model found ({CHECKPOINT_PATH.stat().st_size / 1e6:.0f}MB)")
else:
    logger.debug(f"Bass model not found at {CHECKPOINT_PATH}")


# ============================================================================
# MODEL (must match training notebook exactly)
# ============================================================================

class BassTranscriptionModel(nn.Module):
    """
    CRNN for bass transcription: CQT → Conv2D → BiLSTM → onset/frame/velocity

    Input:  (batch, 84, time)
    Output: onset    (batch, 4, 24, time)
            frame    (batch, 4, 24, time)
            velocity (batch, 4, 24, time)
    """

    def __init__(self, n_bins=84, num_strings=4, num_frets=24,
                 hidden_size=256, num_layers=2, dropout=0.25):
        super().__init__()

        self.num_strings = num_strings
        self.num_frets = num_frets
        output_size = num_strings * num_frets

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(dropout),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(dropout),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(dropout),
        )

        cnn_output_dim = 128 * 10

        self.onset_lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.frame_lstm = nn.LSTM(
            input_size=cnn_output_dim + output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        lstm_output_dim = hidden_size * 2

        self.onset_head = nn.Linear(lstm_output_dim, output_size)
        self.frame_head = nn.Linear(lstm_output_dim, output_size)
        self.velocity_head = nn.Sequential(
            nn.Linear(lstm_output_dim, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch, n_bins, time = x.shape

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

        def reshape_output(t):
            return t.permute(0, 2, 1).reshape(
                batch, self.num_strings, self.num_frets, time
            )

        return reshape_output(onset_pred), reshape_output(frame_pred), reshape_output(velocity_pred)


# ============================================================================
# RESULT DATACLASS
# ============================================================================

@dataclass
class BassTranscriptionResult:
    midi_path: Optional[str]
    num_notes: int
    quality_score: float
    method: str
    num_strings_used: int
    fret_range: Tuple[int, int]


# ============================================================================
# TRANSCRIBER
# ============================================================================

class BassTranscriber:
    """
    Bass transcriber using the trained CRNN model.
    Lazy-loads model on first use.
    """

    def __init__(self):
        self._model = None
        self._device = None

    def _ensure_model(self):
        if self._model is not None:
            return

        if not MODEL_AVAILABLE:
            raise RuntimeError(f"Bass model not found at {CHECKPOINT_PATH}")

        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=self._device, weights_only=False)
        self._model = BassTranscriptionModel()
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.to(self._device)
        self._model.eval()

        epoch = checkpoint.get('epoch', '?')
        val_loss = checkpoint.get('val_loss', '?')
        logger.info(f"Bass model loaded on {self._device} "
                    f"(epoch={epoch}, val_loss={val_loss})")

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

        logger.info(f"Transcribing bass: {audio_path.name}")

        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        duration = len(audio) / sr

        # Compute CQT (MUST match training)
        cqt = np.abs(librosa.cqt(
            audio, sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            n_bins=N_BINS,
            bins_per_octave=BINS_PER_OCTAVE,
        ))
        cqt = np.log(cqt + 1e-8)

        total_frames = cqt.shape[1]

        # Overlap-add inference
        onset_pred, frame_pred, velocity_pred = self._infer_overlap(cqt, total_frames)

        # Extract notes
        notes = self._extract_notes(onset_pred, frame_pred, velocity_pred)
        logger.info(f"Extracted {len(notes)} bass notes")

        if not notes:
            return BassTranscriptionResult(
                midi_path=None, num_notes=0, quality_score=0.0,
                method='bass_nn_model', num_strings_used=0, fret_range=(0, 0),
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
        bass_track = pretty_midi.Instrument(program=33, is_drum=False, name='Electric Bass')

        strings_used = set()
        frets_used = []

        for note in notes:
            start_time = note['onset_frame'] / frames_per_sec
            end_time = note['offset_frame'] / frames_per_sec
            midi_pitch = TUNING[note['string']] + note['fret']
            velocity = note['velocity']

            bass_track.notes.append(pretty_midi.Note(
                velocity=velocity,
                pitch=midi_pitch,
                start=start_time,
                end=end_time,
            ))

            strings_used.add(note['string'])
            frets_used.append(note['fret'])

        midi.instruments.append(bass_track)

        midi_filename = f"{audio_path.stem}_bass.mid"
        midi_path = output_dir / midi_filename
        midi.write(str(midi_path))

        # Quality heuristic
        note_density = len(notes) / duration
        num_strings = len(strings_used)
        fret_range = (min(frets_used), max(frets_used)) if frets_used else (0, 0)

        quality = 0.0
        quality += 0.25 * min(1.0, len(notes) / (duration * 2))
        quality += 0.2 * min(1.0, num_strings / 3)
        quality += 0.2 if 0.5 < note_density < 10.0 else 0.0
        quality += 0.2 if fret_range[1] - fret_range[0] > 2 else 0.1
        quality += 0.15
        quality = min(1.0, quality)

        logger.info(f"Bass MIDI: {midi_path.name}, {len(notes)} notes, "
                    f"strings={num_strings}, frets={fret_range}, quality={quality:.2f}")

        return BassTranscriptionResult(
            midi_path=str(midi_path),
            num_notes=len(notes),
            quality_score=quality,
            method='bass_nn_model',
            num_strings_used=num_strings,
            fret_range=fret_range,
        )

    def _infer_overlap(self, cqt, total_frames):
        """Run model with overlap-add for long audio."""
        if total_frames <= CHUNK_FRAMES:
            cqt_tensor = torch.from_numpy(cqt).unsqueeze(0).float().to(self._device)
            if cqt_tensor.shape[2] < CHUNK_FRAMES:
                pad = CHUNK_FRAMES - cqt_tensor.shape[2]
                cqt_tensor = torch.nn.functional.pad(cqt_tensor, (0, pad))

            with torch.no_grad():
                onset, frame, vel = self._model(cqt_tensor)

            return (
                onset[0].cpu().numpy()[:, :, :total_frames],
                frame[0].cpu().numpy()[:, :, :total_frames],
                vel[0].cpu().numpy()[:, :, :total_frames],
            )

        # Overlap-add
        overlap = CHUNK_FRAMES // 2
        hop = CHUNK_FRAMES - overlap
        window = np.hanning(CHUNK_FRAMES)

        onset_acc = np.zeros((NUM_STRINGS, NUM_FRETS, total_frames), dtype=np.float32)
        frame_acc = np.zeros((NUM_STRINGS, NUM_FRETS, total_frames), dtype=np.float32)
        vel_acc = np.zeros((NUM_STRINGS, NUM_FRETS, total_frames), dtype=np.float32)
        weight_acc = np.zeros(total_frames, dtype=np.float32)

        start = 0
        while start < total_frames:
            end = min(start + CHUNK_FRAMES, total_frames)
            chunk = cqt[:, start:end]

            if chunk.shape[1] < CHUNK_FRAMES:
                pad = CHUNK_FRAMES - chunk.shape[1]
                chunk = np.pad(chunk, ((0, 0), (0, pad)))

            chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).float().to(self._device)
            with torch.no_grad():
                o, f, v = self._model(chunk_tensor)

            actual_len = min(CHUNK_FRAMES, total_frames - start)
            w = window[:actual_len]

            onset_acc[:, :, start:start + actual_len] += o[0].cpu().numpy()[:, :, :actual_len] * w
            frame_acc[:, :, start:start + actual_len] += f[0].cpu().numpy()[:, :, :actual_len] * w
            vel_acc[:, :, start:start + actual_len] += v[0].cpu().numpy()[:, :, :actual_len] * w
            weight_acc[start:start + actual_len] += w

            start += hop

        weight_acc = np.maximum(weight_acc, 1e-8)
        return onset_acc / weight_acc, frame_acc / weight_acc, vel_acc / weight_acc

    def _extract_notes(self, onset_pred, frame_pred, velocity_pred,
                       onset_threshold=0.5, frame_threshold=0.3,
                       min_frames=2):
        """Extract notes using onset-triggered, frame-sustained logic."""
        notes = []
        num_strings, num_frets, total_frames = onset_pred.shape

        for s in range(num_strings):
            for f in range(num_frets):
                t = 0
                while t < total_frames:
                    if onset_pred[s, f, t] > onset_threshold:
                        onset_frame = t
                        vel = velocity_pred[s, f, t]
                        velocity = int(40 + 87 * min(1.0, vel))

                        t += 1
                        while t < total_frames and frame_pred[s, f, t] > frame_threshold:
                            t += 1

                        offset_frame = t
                        if offset_frame - onset_frame >= min_frames:
                            notes.append({
                                'string': s,
                                'fret': f,
                                'onset_frame': onset_frame,
                                'offset_frame': offset_frame,
                                'velocity': velocity,
                            })
                    else:
                        t += 1

        notes.sort(key=lambda n: n['onset_frame'])
        return notes


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_transcriber: Optional[BassTranscriber] = None


def transcribe_bass(audio_path: str, output_dir: str,
                    tempo_hint: float = None) -> Optional[str]:
    """Convenience function: returns MIDI path or None."""
    global _transcriber

    if _transcriber is None:
        _transcriber = BassTranscriber()

    try:
        result = _transcriber.transcribe(audio_path, output_dir, tempo_hint)
        if result.midi_path and result.quality_score > 0.2:
            return result.midi_path
        return None
    except Exception as e:
        logger.error(f"Bass transcription failed: {e}")
        return None


def is_available() -> bool:
    return MODEL_AVAILABLE


BASS_MODEL_AVAILABLE = MODEL_AVAILABLE
