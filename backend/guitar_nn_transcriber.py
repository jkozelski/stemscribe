"""
Neural Guitar Transcriber for StemScriber
=========================================
Uses a Kong-style CRNN model fine-tuned on GuitarSet via domain adaptation
from the Kong et al. piano transcription checkpoint.

Architecture: GuitarTranscriptionModel (Kong-style)
  4x AcousticModelCRnn8Dropout (frame, onset, offset, velocity)
  Each: 4 ConvBlocks -> FC(1792,768) -> BiGRU(768,256) -> FC(512, 48)
  Conditioning: onset <- velocity, frame <- onset + offset

Checkpoint: backend/models/pretrained/best_guitar_model.pt
Training: train_guitar_model/train_guitar_runpod.py (GuitarSet, Kong domain adaptation)

Guitar Range: E2 (MIDI 40) to Eb6 (MIDI 87) -- 48 pitches

Falls back to Basic Pitch guitar_tab_transcriber when no checkpoint exists.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS -- MUST match training CONFIG exactly
# ============================================================================

CHECKPOINT_PATH = Path(__file__).parent / 'models' / 'pretrained' / 'best_guitar_model.pt'
MODEL_AVAILABLE = CHECKPOINT_PATH.exists()

SAMPLE_RATE = 16000
HOP_LENGTH = 160
N_MELS = 229
N_FFT = 2048
NUM_KEYS = 48
MIN_MIDI = 40   # E2
MAX_MIDI = 88   # exclusive upper bound (48 pitches: MIDI 40-87)
CHUNK_DURATION = 10.0

CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)
CHUNK_FRAMES = CHUNK_SAMPLES // HOP_LENGTH

if MODEL_AVAILABLE:
    logger.info(f"Guitar NN model found ({CHECKPOINT_PATH.stat().st_size / 1e6:.0f}MB)")
else:
    logger.debug(f"Guitar NN model not found at {CHECKPOINT_PATH}")


# ============================================================================
# KONG-STYLE MODEL ARCHITECTURE (must match train_guitar_runpod.py exactly)
# ============================================================================

def _init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)


def _init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def _init_gru(rnn):
    for name, param in rnn.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            param.data.fill_(0.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        _init_bn(self.bn1)
        _init_bn(self.bn2)
        _init_layer(self.conv1)
        _init_layer(self.conv2)

    def forward(self, x, pool_size=(2, 2), pool_type='avg'):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        return x


class AcousticModelCRnn8Dropout(nn.Module):
    """Kong's CRNN acoustic model: 4 ConvBlocks -> FC -> BiGRU -> Linear"""
    def __init__(self, classes_num, midfeat, momentum):
        super().__init__()
        self.conv_block1 = ConvBlock(1, 48)
        self.conv_block2 = ConvBlock(48, 64)
        self.conv_block3 = ConvBlock(64, 96)
        self.conv_block4 = ConvBlock(96, 128)

        self.fc5 = nn.Linear(midfeat, 768, bias=False)
        self.bn5 = nn.BatchNorm1d(768, momentum=momentum)
        _init_layer(self.fc5)
        _init_bn(self.bn5)

        self.gru = nn.GRU(input_size=768, hidden_size=256, num_layers=2,
                          bias=True, batch_first=True, dropout=0.0, bidirectional=True)
        _init_gru(self.gru)

        self.fc = nn.Linear(512, classes_num, bias=True)
        _init_layer(self.fc)

    def forward(self, input):
        # input: (batch, 1, time, mel_bins)
        x = self.conv_block1(input, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        # (batch, 128, time, mel_bins // 16)
        x = x.transpose(1, 2).flatten(2)  # (batch, time, 128 * mel_bins // 16)
        x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x = F.dropout(x, p=0.5, training=self.training)

        x, _ = self.gru(x)
        x = self.fc(x)
        return x


class GuitarTranscriptionModel(nn.Module):
    """
    Kong-style guitar transcription model with 4 parallel acoustic models
    and onset/frame conditioning. 48 pitches (E2-Eb6).

    Input:  (batch, time, n_mels) -- power_to_db mel spectrogram
    Output: onset_logits    (batch, time, 48)
            frame_logits    (batch, time, 48)
            velocity_pred   (batch, time, 48) -- sigmoid, [0, 1]
    """
    def __init__(self, n_pitches=48):
        super().__init__()
        self.n_pitches = n_pitches

        # midfeat = 128 * (229 // 16) = 128 * 14 = 1792
        midfeat = 128 * (N_MELS // 16)

        # Four parallel acoustic models (matching Kong architecture)
        self.frame_model = AcousticModelCRnn8Dropout(n_pitches, midfeat, momentum=0.01)
        self.onset_model = AcousticModelCRnn8Dropout(n_pitches, midfeat, momentum=0.01)
        self.offset_model = AcousticModelCRnn8Dropout(n_pitches, midfeat, momentum=0.01)
        self.velocity_model = AcousticModelCRnn8Dropout(n_pitches, midfeat, momentum=0.01)

        # Conditioning GRUs (matching Kong)
        self.onset_gru = nn.GRU(input_size=n_pitches * 2, hidden_size=n_pitches,
                                num_layers=1, bias=True, batch_first=True, bidirectional=True)
        _init_gru(self.onset_gru)
        self.onset_fc = nn.Linear(n_pitches * 2, n_pitches, bias=True)
        _init_layer(self.onset_fc)

        self.frame_gru = nn.GRU(input_size=n_pitches * 3, hidden_size=n_pitches,
                                num_layers=1, bias=True, batch_first=True, bidirectional=True)
        _init_gru(self.frame_gru)
        self.frame_fc = nn.Linear(n_pitches * 2, n_pitches, bias=True)
        _init_layer(self.frame_fc)

    def forward(self, mel):
        """
        Args:
            mel: (batch, time, n_mels) log-mel spectrogram

        Returns:
            onset_output:    (batch, time, n_pitches) -- onset logits
            frame_output:    (batch, time, n_pitches) -- frame logits
            velocity_output: (batch, time, n_pitches) -- velocity [0, 1]
        """
        # Reshape for CNN: (batch, 1, time, n_mels)
        x = mel.unsqueeze(1)

        # Run parallel acoustic models
        frame_out = self.frame_model(x)
        onset_out = self.onset_model(x)
        offset_out = self.offset_model(x)
        velocity_out = self.velocity_model(x)

        # Condition onset on velocity (Kong's recipe)
        velocity_sigmoid = torch.sigmoid(velocity_out)
        onset_concat = torch.cat([onset_out, onset_out * velocity_sigmoid], dim=2)
        onset_gru_out, _ = self.onset_gru(onset_concat)
        onset_output = self.onset_fc(onset_gru_out)

        # Condition frame on onset + offset (Kong's recipe)
        onset_sigmoid = torch.sigmoid(onset_output)
        offset_sigmoid = torch.sigmoid(offset_out)
        frame_concat = torch.cat([frame_out, onset_sigmoid, offset_sigmoid], dim=2)
        frame_gru_out, _ = self.frame_gru(frame_concat)
        frame_output = self.frame_fc(frame_gru_out)

        # Velocity uses sigmoid for [0, 1] output
        velocity_output = velocity_sigmoid

        return onset_output, frame_output, velocity_output


# ============================================================================
# RESULT DATACLASS
# ============================================================================

@dataclass
class GuitarTranscriptionResult:
    midi_path: Optional[str]
    num_notes: int
    quality_score: float
    method: str
    pitch_range: tuple
    polyphony_avg: float


# ============================================================================
# TRANSCRIBER
# ============================================================================

class GuitarNNTranscriber:
    """
    Guitar transcriber using the Kong-style CRNN model fine-tuned on GuitarSet.
    Lazy-loads model on first use. Falls back to Basic Pitch when no checkpoint exists.
    """

    def __init__(self):
        self._model = None
        self._device = None

    def _ensure_model(self):
        if self._model is not None:
            return

        if not MODEL_AVAILABLE:
            raise RuntimeError(
                f"Guitar NN model not found at {CHECKPOINT_PATH}"
            )

        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )

        checkpoint = torch.load(
            CHECKPOINT_PATH, map_location=self._device, weights_only=True
        )

        self._model = GuitarTranscriptionModel(n_pitches=NUM_KEYS)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.to(self._device)
        self._model.eval()

        epoch = checkpoint.get('epoch', '?')
        val_f1 = checkpoint.get('val_f1', '?')
        logger.info(f"Guitar NN model loaded on {self._device} "
                    f"(epoch={epoch}, val_f1={val_f1})")

    def transcribe(self, audio_path: str, output_dir: str,
                   tempo_hint: float = None) -> GuitarTranscriptionResult:
        """
        Transcribe guitar audio to MIDI.

        Args:
            audio_path: Path to guitar audio file
            output_dir: Output directory for MIDI
            tempo_hint: Known tempo (optional)

        Returns:
            GuitarTranscriptionResult
        """
        import librosa
        import pretty_midi

        self._ensure_model()

        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Transcribing guitar (NN): {audio_path.name}")

        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        duration = len(audio) / sr

        # Compute mel spectrogram (MUST match training exactly)
        mel = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE,
            n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)  # (n_mels, T)
        mel_db = mel_db.T  # (T, n_mels) -- model expects time-first

        total_frames = mel_db.shape[0]

        # Run inference with overlap-add for long audio
        onset_logits, frame_logits, velocity_pred = self._infer_overlap(mel_db, total_frames)

        # Apply sigmoid to logits for note extraction
        onset_pred = 1.0 / (1.0 + np.exp(-onset_logits))
        frame_pred = 1.0 / (1.0 + np.exp(-frame_logits))

        # Extract notes
        notes = self._extract_notes(onset_pred, frame_pred, velocity_pred)
        logger.info(f"Extracted {len(notes)} guitar notes")

        if not notes:
            return GuitarTranscriptionResult(
                midi_path=None, num_notes=0, quality_score=0.0,
                method='guitar_nn', pitch_range=(0, 0), polyphony_avg=0.0,
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
        guitar_track = pretty_midi.Instrument(
            program=25, is_drum=False, name='Steel String Guitar'
        )

        pitches = []
        for note in notes:
            start_time = note['onset_frame'] / frames_per_sec
            end_time = note['offset_frame'] / frames_per_sec
            midi_pitch = note['pitch']
            velocity = note['velocity']

            guitar_track.notes.append(pretty_midi.Note(
                velocity=velocity,
                pitch=midi_pitch,
                start=start_time,
                end=end_time,
            ))
            pitches.append(midi_pitch)

        midi.instruments.append(guitar_track)

        midi_filename = f"{audio_path.stem}_guitar.mid"
        midi_path = output_dir / midi_filename
        midi.write(str(midi_path))

        # Compute polyphony
        polyphony = self._compute_polyphony(notes, total_frames)

        # Quality heuristic
        pitch_range = (min(pitches), max(pitches))
        pitch_span = pitch_range[1] - pitch_range[0]

        quality = 0.0
        quality += 0.2 * min(1.0, len(notes) / (duration * 3))
        quality += 0.2 * min(1.0, pitch_span / 30)
        quality += 0.2 if 1.0 < polyphony < 6.0 else 0.1
        quality += 0.15 if len(notes) > 15 else 0.0
        quality += 0.1 if pitch_range[0] < 55 and pitch_range[1] > 60 else 0.0
        quality += 0.15
        quality = min(1.0, quality)

        logger.info(f"Guitar MIDI: {midi_path.name}, {len(notes)} notes, "
                    f"range={pitch_range}, polyphony={polyphony:.1f}, quality={quality:.2f}")

        return GuitarTranscriptionResult(
            midi_path=str(midi_path),
            num_notes=len(notes),
            quality_score=quality,
            method='guitar_nn',
            pitch_range=pitch_range,
            polyphony_avg=polyphony,
        )

    def _infer_overlap(self, mel_db, total_frames):
        """
        Run model with overlap-add for long audio.

        Args:
            mel_db: (T, n_mels) power_to_db mel spectrogram, time-first

        Returns:
            onset_logits:  (T, n_pitches) raw logits
            frame_logits:  (T, n_pitches) raw logits
            velocity_pred: (T, n_pitches) sigmoid [0, 1]
        """
        if total_frames <= CHUNK_FRAMES:
            chunk = mel_db
            if chunk.shape[0] < CHUNK_FRAMES:
                pad = CHUNK_FRAMES - chunk.shape[0]
                chunk = np.pad(chunk, ((0, pad), (0, 0)))

            mel_tensor = torch.from_numpy(chunk).unsqueeze(0).float().to(self._device)
            with torch.no_grad():
                onset, frame, vel = self._model(mel_tensor)

            return (
                onset[0, :total_frames].cpu().numpy(),
                frame[0, :total_frames].cpu().numpy(),
                vel[0, :total_frames].cpu().numpy(),
            )

        # Overlap-add for long audio
        overlap = CHUNK_FRAMES // 2
        hop = CHUNK_FRAMES - overlap
        window = np.hanning(CHUNK_FRAMES)

        onset_acc = np.zeros((total_frames, NUM_KEYS), dtype=np.float32)
        frame_acc = np.zeros((total_frames, NUM_KEYS), dtype=np.float32)
        vel_acc = np.zeros((total_frames, NUM_KEYS), dtype=np.float32)
        weight_acc = np.zeros(total_frames, dtype=np.float32)

        start = 0
        while start < total_frames:
            end = min(start + CHUNK_FRAMES, total_frames)
            chunk = mel_db[start:end]

            if chunk.shape[0] < CHUNK_FRAMES:
                pad = CHUNK_FRAMES - chunk.shape[0]
                chunk = np.pad(chunk, ((0, pad), (0, 0)))

            chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).float().to(self._device)
            with torch.no_grad():
                o, f, v = self._model(chunk_tensor)

            actual_len = min(CHUNK_FRAMES, total_frames - start)
            w = window[:actual_len]

            # Model output is (batch, time, pitches)
            onset_acc[start:start + actual_len] += o[0, :actual_len].cpu().numpy() * w[:, None]
            frame_acc[start:start + actual_len] += f[0, :actual_len].cpu().numpy() * w[:, None]
            vel_acc[start:start + actual_len] += v[0, :actual_len].cpu().numpy() * w[:, None]
            weight_acc[start:start + actual_len] += w

            start += hop

        weight_acc = np.maximum(weight_acc, 1e-8)
        return (
            onset_acc / weight_acc[:, None],
            frame_acc / weight_acc[:, None],
            vel_acc / weight_acc[:, None],
        )

    def _extract_notes(self, onset_pred, frame_pred, velocity_pred,
                       onset_threshold=0.5, frame_threshold=0.3,
                       min_frames=2):
        """
        Extract notes using onset-triggered, frame-sustained logic.

        All inputs are (T, n_pitches) -- time-first layout matching model output.
        """
        notes = []
        total_frames, num_keys = onset_pred.shape

        for k in range(num_keys):
            midi_pitch = k + MIN_MIDI
            t = 0
            while t < total_frames:
                if onset_pred[t, k] > onset_threshold:
                    onset_frame = t
                    vel = velocity_pred[t, k]
                    velocity = int(40 + 87 * min(1.0, vel))

                    t += 1
                    while t < total_frames and frame_pred[t, k] > frame_threshold:
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

_transcriber: Optional[GuitarNNTranscriber] = None


def transcribe_guitar_nn(audio_path: str, output_dir: str,
                         tempo_hint: float = None) -> Optional[str]:
    """Convenience function: returns MIDI path or None."""
    global _transcriber

    if _transcriber is None:
        _transcriber = GuitarNNTranscriber()

    try:
        result = _transcriber.transcribe(audio_path, output_dir, tempo_hint)
        if result.midi_path and result.quality_score > 0.2:
            return result.midi_path
        return None
    except Exception as e:
        logger.error(f"Guitar NN transcription failed: {e}")
        return None


def is_available() -> bool:
    return MODEL_AVAILABLE


GUITAR_NN_MODEL_AVAILABLE = MODEL_AVAILABLE
