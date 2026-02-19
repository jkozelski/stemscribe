"""
Guitar Tab Transcriber for StemScribe
=====================================
Uses a trained CRNN (CQT -> Conv2d -> BiLSTM -> onset/frame heads) to
predict per-(string, fret) activations for guitar audio, then converts
the predictions to standard MIDI for downstream Guitar Pro conversion.

Architecture: GuitarTabModel (CRNN, 84-bin CQT input, 6x20 output)
Checkpoint: backend/models/pretrained/best_tab_model.pt
Training: 100 epochs on GuitarSet, best val_loss 0.0498 at epoch 48

The model outputs onset and frame probabilities for each of the 120
(6 strings x 20 frets) positions at each time frame. Notes are extracted
using onset-triggered, frame-sustained logic, then written to standard
MIDI which flows through the existing midi_to_gp.py Guitar Pro pipeline.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# PATHS & AVAILABILITY
# ============================================================================

CHECKPOINT_PATH = Path(__file__).parent / 'models' / 'pretrained' / 'best_tab_model.pt'
MODEL_AVAILABLE = CHECKPOINT_PATH.exists()

if MODEL_AVAILABLE:
    logger.info(f"Guitar tab model available "
                f"(checkpoint: {CHECKPOINT_PATH.stat().st_size / 1e6:.1f}MB)")
else:
    logger.warning(f"Guitar tab model: checkpoint not found at {CHECKPOINT_PATH}")


# ============================================================================
# CONSTANTS (must match training exactly)
# ============================================================================

SAMPLE_RATE = 22050
HOP_LENGTH = 256
N_BINS = 84                     # 7 octaves x 12 semitones
BINS_PER_OCTAVE = 12
NUM_STRINGS = 6
NUM_FRETS = 20                  # Frets 0-19

CHUNK_DURATION = 3.0            # Training chunk size in seconds
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)   # 66150
CHUNK_FRAMES = CHUNK_SAMPLES // HOP_LENGTH           # ~258

# Standard guitar tuning: E2 A2 D3 G3 B3 E4 (MIDI note numbers)
# Matches midi_to_gp.py TUNINGS['guitar'] exactly
TUNING = [40, 45, 50, 55, 59, 64]


# ============================================================================
# MODEL ARCHITECTURE (must match training notebook cell-8 exactly)
# ============================================================================

class GuitarTabModel(nn.Module):
    """CRNN model for guitar tablature transcription.

    Input:  CQT spectrogram (batch, 84, time)
    Output: onset_pred, frame_pred — each (batch, 6, 20, time)
            6 strings x 20 frets, sigmoid activations [0, 1]
    """

    def __init__(self, n_bins=84, num_strings=6, num_frets=20):
        super().__init__()

        self.num_strings = num_strings
        self.num_frets = num_frets

        # CNN encoder — pools frequency only, preserves time dimension
        self.conv_stack = nn.Sequential(
            # Block 1: n_bins -> n_bins/2
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),       # 84 -> 42 freq, time unchanged
            nn.Dropout(0.25),

            # Block 2: n_bins/2 -> n_bins/4
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),       # 42 -> 21 freq, time unchanged
            nn.Dropout(0.25),

            # Block 3: n_bins/4 -> n_bins/8
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),       # 21 -> 10 freq, time unchanged
            nn.Dropout(0.25),
        )

        # After conv: (batch, 128, n_bins//8, time) = (batch, 128, 10, time)
        self.flat_size = 128 * (n_bins // 8)    # 1280

        # Bidirectional LSTM for temporal context
        self.lstm = nn.LSTM(
            self.flat_size, 256, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3
        )
        # Output: (batch, time, 512) [256 x 2 directions]

        # Output heads — predict per-string-fret activation
        self.onset_head = nn.Linear(512, num_strings * num_frets)   # 120
        self.frame_head = nn.Linear(512, num_strings * num_frets)   # 120

    def forward(self, x):
        # x: (batch, n_bins=84, time)
        x = x.unsqueeze(1)                 # (batch, 1, 84, time)

        # CNN
        x = self.conv_stack(x)             # (batch, 128, 10, time)

        # Reshape for LSTM: (batch, time, features)
        batch, channels, freq, time = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch, time, -1)    # (batch, time, 1280)

        # LSTM
        x, _ = self.lstm(x)               # (batch, time, 512)

        # Predictions with sigmoid
        onset_pred = torch.sigmoid(self.onset_head(x))
        frame_pred = torch.sigmoid(self.frame_head(x))

        # Reshape to (batch, strings, frets, time)
        onset_pred = onset_pred.view(batch, time, self.num_strings, self.num_frets)
        onset_pred = onset_pred.permute(0, 2, 3, 1)

        frame_pred = frame_pred.view(batch, time, self.num_strings, self.num_frets)
        frame_pred = frame_pred.permute(0, 2, 3, 1)

        return onset_pred, frame_pred


# ============================================================================
# MODEL LOADING
# ============================================================================

def _load_model(device: torch.device = None) -> GuitarTabModel:
    """Load the Guitar Tab model from checkpoint."""
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )

    model = GuitarTabModel(n_bins=N_BINS, num_strings=NUM_STRINGS, num_frets=NUM_FRETS)

    logger.info(f"Loading guitar tab model from {CHECKPOINT_PATH}")
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
    logger.info(f"Guitar tab model loaded on {device} ({param_count / 1e6:.1f}M params)")
    return model


# ============================================================================
# RESULT DATACLASS
# ============================================================================

@dataclass
class TabTranscriptionResult:
    """Result of guitar tab transcription."""
    midi_path: Optional[str]
    num_notes: int
    quality_score: float            # 0.0 - 1.0
    method: str                     # 'guitar_tab_model'
    num_strings_used: int           # How many strings had notes (0-6)
    fret_range: Tuple[int, int]     # (min_fret, max_fret)


# ============================================================================
# TRANSCRIBER CLASS
# ============================================================================

class GuitarTabTranscriber:
    """
    Transcribes guitar audio to MIDI using the trained CRNN tab model.

    The model outputs per-frame (string, fret) activations. We fuse onset
    and frame predictions, extract notes, and write standard MIDI that flows
    through the existing midi_to_gp.py pipeline.

    Usage:
        transcriber = GuitarTabTranscriber()
        result = transcriber.transcribe('guitar_stem.wav', '/output/dir')
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
    # CQT Feature Extraction
    # ----------------------------------------------------------------

    def _compute_cqt(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute CQT spectrogram matching training preprocessing exactly.

        CRITICAL: The normalization must match training:
            cqt = np.log(np.abs(cqt) + 1e-8)
        NOT np.log1p, NOT librosa.amplitude_to_db.

        Args:
            audio: Mono audio at SAMPLE_RATE (22050 Hz)

        Returns:
            np.ndarray of shape (84, time_frames), log-scaled CQT
        """
        import librosa

        cqt = librosa.cqt(
            audio,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            n_bins=N_BINS,
            bins_per_octave=BINS_PER_OCTAVE,
        )

        # Match training: np.abs then np.log with epsilon
        cqt = np.abs(cqt)
        cqt = np.log(cqt + 1e-8).astype(np.float32)

        return cqt  # shape: (84, time_frames)

    # ----------------------------------------------------------------
    # Overlap-Add Inference
    # ----------------------------------------------------------------

    def _infer_full_audio(self, cqt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run model inference with overlap-add for full-song CQT.

        The model was trained on 3-second chunks (~258 frames). For longer
        audio, we use 50% overlap with Hann window blending.

        Time dimension is preserved through the network because MaxPool2d
        only pools the frequency axis (kernel (2,1)), never time.

        Args:
            cqt: Full CQT spectrogram, shape (84, total_frames)

        Returns:
            onset_pred: (6, 20, total_frames)
            frame_pred: (6, 20, total_frames)
        """
        total_frames = cqt.shape[1]

        # Short audio — process in one pass
        if total_frames <= CHUNK_FRAMES:
            x = torch.from_numpy(cqt).float().unsqueeze(0).to(self._device)
            with torch.no_grad():
                onset, frame = self._model(x)
            return onset[0].cpu().numpy(), frame[0].cpu().numpy()

        # Overlap-add for long audio
        overlap_ratio = 0.5
        hop_frames = int(CHUNK_FRAMES * (1 - overlap_ratio))    # ~129 frames

        num_chunks = max(1, (total_frames - CHUNK_FRAMES) // hop_frames + 1)
        pad_needed = (num_chunks - 1) * hop_frames + CHUNK_FRAMES - total_frames
        if pad_needed > 0:
            cqt_padded = np.pad(cqt, ((0, 0), (0, pad_needed)), mode='constant')
        else:
            cqt_padded = cqt
            pad_needed = 0

        padded_frames = cqt_padded.shape[1]

        # Hann window for crossfade blending
        window = np.hanning(CHUNK_FRAMES).astype(np.float32)

        # Accumulators
        onset_acc = np.zeros((NUM_STRINGS, NUM_FRETS, padded_frames), dtype=np.float32)
        frame_acc = np.zeros((NUM_STRINGS, NUM_FRETS, padded_frames), dtype=np.float32)
        weight_acc = np.zeros(padded_frames, dtype=np.float32)

        for i in range(num_chunks + 1):
            start = i * hop_frames
            end = start + CHUNK_FRAMES

            if start >= padded_frames:
                break

            # Extract chunk, pad if needed
            if end > padded_frames:
                chunk_cqt = cqt_padded[:, start:]
                pad_right = CHUNK_FRAMES - chunk_cqt.shape[1]
                if pad_right > 0:
                    chunk_cqt = np.pad(chunk_cqt, ((0, 0), (0, pad_right)))
            else:
                chunk_cqt = cqt_padded[:, start:end]

            # Model forward pass
            x = torch.from_numpy(chunk_cqt).float().unsqueeze(0).to(self._device)
            with torch.no_grad():
                onset_chunk, frame_chunk = self._model(x)

            onset_np = onset_chunk[0].cpu().numpy()     # (6, 20, time)
            frame_np = frame_chunk[0].cpu().numpy()

            # Verify time dimension preserved (should match input)
            out_t = onset_np.shape[2]
            actual_len = min(out_t, padded_frames - start)
            w = window[:actual_len]

            onset_acc[:, :, start:start + actual_len] += onset_np[:, :, :actual_len] * w
            frame_acc[:, :, start:start + actual_len] += frame_np[:, :, :actual_len] * w
            weight_acc[start:start + actual_len] += w

        # Normalize by accumulated weights
        weight_acc = np.maximum(weight_acc, 1e-8)
        onset_acc /= weight_acc
        frame_acc /= weight_acc

        # Trim to original length
        return onset_acc[:, :, :total_frames], frame_acc[:, :, :total_frames]

    # ----------------------------------------------------------------
    # Note Extraction
    # ----------------------------------------------------------------

    def _extract_notes(self, onset_pred: np.ndarray, frame_pred: np.ndarray,
                       onset_threshold: float = 0.5,
                       frame_threshold: float = 0.3) -> List[dict]:
        """
        Extract discrete notes from onset/frame predictions.

        Uses onset-triggered, frame-sustained logic:
        - A note STARTS when onset > onset_threshold
        - A note CONTINUES while frame > frame_threshold
        - A note ENDS when frame drops below threshold

        Args:
            onset_pred: (6, 20, time) onset probabilities
            frame_pred: (6, 20, time) frame probabilities
            onset_threshold: Threshold for onset detection (default 0.5)
            frame_threshold: Threshold for frame sustain (default 0.3)

        Returns:
            List of note dicts sorted by start time
        """
        notes = []
        num_strings, num_frets, num_frames = onset_pred.shape

        for s in range(num_strings):
            for f in range(num_frets):
                onset_row = onset_pred[s, f, :]
                frame_row = frame_pred[s, f, :]

                in_note = False
                note_start = 0
                peak_onset = 0.0

                for t in range(num_frames):
                    if not in_note and onset_row[t] > onset_threshold:
                        # Note onset detected
                        in_note = True
                        note_start = t
                        peak_onset = onset_row[t]

                    elif in_note:
                        if frame_row[t] < frame_threshold:
                            # Note offset — frame activation dropped
                            duration_frames = t - note_start
                            if duration_frames >= 2:    # Min ~23ms at hop=256/sr=22050
                                midi_note = TUNING[s] + f
                                velocity = int(40 + 87 * min(1.0, peak_onset))
                                notes.append({
                                    'string': s,
                                    'fret': f,
                                    'midi_note': midi_note,
                                    'start_frame': note_start,
                                    'end_frame': t,
                                    'velocity': velocity,
                                })
                            in_note = False
                        else:
                            # Track peak onset for velocity
                            peak_onset = max(peak_onset, onset_row[t])

                # Handle note still active at end of audio
                if in_note:
                    duration_frames = num_frames - note_start
                    if duration_frames >= 2:
                        midi_note = TUNING[s] + f
                        velocity = int(40 + 87 * min(1.0, peak_onset))
                        notes.append({
                            'string': s,
                            'fret': f,
                            'midi_note': midi_note,
                            'start_frame': note_start,
                            'end_frame': num_frames,
                            'velocity': velocity,
                        })

        # Sort by start time, then by string (low to high)
        notes.sort(key=lambda n: (n['start_frame'], n['string']))
        return notes

    # ----------------------------------------------------------------
    # MIDI Generation
    # ----------------------------------------------------------------

    def _notes_to_midi(self, notes: List[dict],
                       tempo: float = 120.0) -> 'pretty_midi.PrettyMIDI':
        """
        Convert extracted notes to a pretty_midi MIDI object.

        MIDI note = TUNING[string] + fret, which matches midi_to_gp.py's
        FretMapper tuning exactly. The GP converter will reconstruct the
        same (string, fret) positions.

        Args:
            notes: List of note dicts from _extract_notes()
            tempo: BPM for the MIDI file

        Returns:
            pretty_midi.PrettyMIDI object
        """
        import pretty_midi

        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=25, name='Guitar')  # Acoustic guitar

        frames_per_sec = SAMPLE_RATE / HOP_LENGTH   # ~86.13

        for note_data in notes:
            start_time = note_data['start_frame'] / frames_per_sec
            end_time = note_data['end_frame'] / frames_per_sec

            # Ensure minimum note length of ~30ms
            if end_time - start_time < 0.03:
                end_time = start_time + 0.03

            midi_note = pretty_midi.Note(
                velocity=note_data['velocity'],
                pitch=note_data['midi_note'],
                start=start_time,
                end=end_time,
            )
            instrument.notes.append(midi_note)

        midi.instruments.append(instrument)
        return midi

    # ----------------------------------------------------------------
    # Main Transcription Pipeline
    # ----------------------------------------------------------------

    def transcribe(self, audio_path: str, output_dir: str,
                   tempo_hint: float = None) -> TabTranscriptionResult:
        """
        Full guitar tab transcription pipeline.

        1. Load audio at 22050 Hz
        2. Compute CQT (84 bins, matching training)
        3. Run model with overlap-add
        4. Extract notes (onset/frame fusion)
        5. Write MIDI

        Args:
            audio_path: Path to guitar audio file (WAV, MP3, etc.)
            output_dir: Directory for output MIDI file
            tempo_hint: Known tempo from earlier analysis (optional)

        Returns:
            TabTranscriptionResult with MIDI path and quality metrics
        """
        import librosa

        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        self._ensure_model()

        logger.info(f"Guitar tab transcription: {audio_path.name}")

        # Load audio at training sample rate (mono)
        audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE)

        if len(audio) < SAMPLE_RATE:
            logger.warning(f"Audio too short ({len(audio) / SAMPLE_RATE:.1f}s), skipping")
            return TabTranscriptionResult(
                midi_path=None, num_notes=0, quality_score=0.0,
                method='guitar_tab_model', num_strings_used=0,
                fret_range=(0, 0),
            )

        # Compute CQT
        cqt = self._compute_cqt(audio)      # (84, frames)
        duration = len(audio) / SAMPLE_RATE
        logger.info(f"CQT: {cqt.shape[1]} frames ({duration:.1f}s)")

        # Run model inference with overlap-add
        onset_pred, frame_pred = self._infer_full_audio(cqt)

        # Extract notes from onset/frame predictions
        notes = self._extract_notes(onset_pred, frame_pred)
        logger.info(f"Extracted {len(notes)} notes from tab model")

        if not notes:
            logger.info("No notes detected by tab model")
            return TabTranscriptionResult(
                midi_path=None, num_notes=0, quality_score=0.0,
                method='guitar_tab_model', num_strings_used=0,
                fret_range=(0, 0),
            )

        # Tempo detection (or use hint from job metadata)
        if tempo_hint and 40 < tempo_hint < 300:
            tempo = tempo_hint
        else:
            try:
                tempo_result = librosa.beat.beat_track(y=audio, sr=SAMPLE_RATE)
                # librosa returns (tempo_array, beat_frames) or (tempo, beat_frames)
                if hasattr(tempo_result[0], '__len__'):
                    tempo = float(tempo_result[0][0])
                else:
                    tempo = float(tempo_result[0])
                tempo = max(40.0, min(300.0, tempo))
            except Exception:
                tempo = 120.0

        # Generate MIDI
        midi_obj = self._notes_to_midi(notes, tempo)
        midi_filename = f"{audio_path.stem}_tab.mid"
        midi_path = output_dir / midi_filename
        midi_obj.write(str(midi_path))

        # Compute quality metrics
        strings_used = set(n['string'] for n in notes)
        frets = [n['fret'] for n in notes]
        note_density = len(notes) / duration

        # Quality heuristic (0.0 - 1.0)
        quality = 0.0
        quality += 0.3 * min(1.0, len(notes) / (duration * 2))     # Note density
        quality += 0.2 * min(1.0, len(strings_used) / 4)            # String diversity
        quality += 0.2 if 1.0 < note_density < 15.0 else 0.0        # Sane density
        quality += 0.15 if max(frets) - min(frets) < 15 else 0.0    # Reasonable range
        quality += 0.15                                               # Base neural confidence
        quality = min(1.0, quality)

        logger.info(f"Tab MIDI: {midi_path.name}, {len(notes)} notes, "
                    f"strings={sorted(strings_used)}, "
                    f"frets={min(frets)}-{max(frets)}, "
                    f"tempo={tempo:.0f}, quality={quality:.2f}")

        return TabTranscriptionResult(
            midi_path=str(midi_path),
            num_notes=len(notes),
            quality_score=quality,
            method='guitar_tab_model',
            num_strings_used=len(strings_used),
            fret_range=(min(frets), max(frets)),
        )


# ============================================================================
# CONVENIENCE FUNCTIONS (matches guitar_separator.py pattern)
# ============================================================================

_transcriber: Optional[GuitarTabTranscriber] = None


def transcribe_guitar_tab(audio_path: str, output_dir: str,
                          tempo_hint: float = None) -> Optional[str]:
    """
    Convenience function: transcribe guitar audio to MIDI using tab model.

    Uses a cached singleton transcriber.

    Args:
        audio_path: Path to guitar audio file
        output_dir: Output directory for MIDI
        tempo_hint: Known tempo (optional)

    Returns:
        MIDI file path, or None if transcription failed/low quality
    """
    global _transcriber

    if _transcriber is None:
        _transcriber = GuitarTabTranscriber()

    try:
        result = _transcriber.transcribe(
            audio_path=audio_path,
            output_dir=output_dir,
            tempo_hint=tempo_hint,
        )
        if result.midi_path and result.quality_score > 0.3:
            return result.midi_path
        return None
    except Exception as e:
        logger.error(f"Guitar tab transcription failed: {e}")
        return None


def is_available() -> bool:
    """Check if guitar tab model is available."""
    return MODEL_AVAILABLE


# Alias for consistent naming in app.py imports
GUITAR_TAB_MODEL_AVAILABLE = MODEL_AVAILABLE


# ============================================================================
# CLI TEST
# ============================================================================

if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.INFO)

    print("Guitar Tab Transcriber")
    print(f"  Model available: {MODEL_AVAILABLE}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Sample rate: {SAMPLE_RATE}")
    print(f"  CQT bins: {N_BINS}")
    print(f"  Tuning: {TUNING}")

    if len(sys.argv) >= 3:
        audio_path = sys.argv[1]
        output_dir = sys.argv[2]

        transcriber = GuitarTabTranscriber()
        result = transcriber.transcribe(audio_path, output_dir)
        print(f"\nResult: {result}")
