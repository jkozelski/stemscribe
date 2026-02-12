"""
Chord Detection V8 for StemScribe
==================================
Advanced chord detection using a Transformer-based neural network.
Trained with 93.61% accuracy on 337 chord classes including:
- Major/minor triads and 7th chords
- mMaj7 (minor-major 7th) for jazz/film scores
- Extended chords: dim, dim7, hdim7, aug, sus2, sus4, 6, min6, 9, add9
- Chord inversions (e.g., C/E, Am/C, Dm7/F)

This is a drop-in replacement for chord_detector.py with the same API.
"""

import json
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - V8 chord detection disabled")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available - chord detection disabled")


@dataclass
class ChordEvent:
    """A detected chord with timing info."""
    time: float
    duration: float
    chord: str      # Full chord name, e.g., "AmMaj7/C"
    root: str       # Root note, e.g., "A"
    quality: str    # Chord quality, e.g., "mMaj7"
    bass: Optional[str]  # Bass note if inversion, e.g., "C"
    confidence: float


@dataclass
class ChordProgression:
    """Result of chord detection."""
    chords: List[ChordEvent]
    key: str


# Note names for chord labeling
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Chord templates for fallback detection
CHORD_TEMPLATES = {
    'maj': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'min': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    '7': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj7': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    'mMaj7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    'dim': [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    'dim7': [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    'hdim7': [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    'aug': [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    'sus2': [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'sus4': [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    '6': [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    'min6': [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    '9': [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'add9': [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
}


class V8TransformerChordModel(nn.Module):
    """
    V8 Transformer architecture for chord recognition.

    Input: 12-dimensional chroma features
    Output: 337 chord classes including inversions

    Architecture:
    - Input projection: 12 -> 128 dimensions
    - Learnable positional encoding
    - 3-layer Transformer encoder (8 attention heads)
    - Classification head with dropout
    """

    def __init__(self, n_classes: int = 337, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 3, seq_len: int = 21):
        super().__init__()
        self.proj = nn.Linear(12, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            batch_first=True,
            dropout=0.1
        )
        self.tf = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

        self.seq_len = seq_len
        self.c = seq_len // 2

    def forward(self, x):
        # Handle both single frames and sequences
        if x.dim() == 2:
            x = x.unsqueeze(1).repeat(1, self.seq_len, 1)

        x = self.proj(x) + self.pos
        x = self.tf(x)

        # Use center frame for classification
        return self.fc(x[:, self.c, :])


class ChordDetector:
    """
    V8 Transformer-based chord detector.

    Drop-in replacement for the original ChordDetector with improved accuracy
    and support for extended chord types and inversions.
    """

    def __init__(self, hop_length: int = 8192, min_duration: float = 0.5,
                 model_path: Optional[str] = None, classes_path: Optional[str] = None,
                 context_frames: int = 21):
        """
        Initialize the V8 chord detector.

        Args:
            hop_length: Samples between chroma frames
            min_duration: Minimum chord duration in seconds
            model_path: Path to trained model weights (.pt file)
            classes_path: Path to class mapping JSON file
            context_frames: Number of frames for temporal context (must be odd)
        """
        self.hop_length = hop_length
        self.min_duration = min_duration
        self.context_frames = context_frames
        self.model = None
        self.device = None
        self.chord_classes = None

        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._load_classes(classes_path)
            self._load_model(model_path)

    def _load_classes(self, classes_path: Optional[str] = None):
        """Load the chord class mapping."""
        if classes_path is None:
            search_paths = [
                Path(__file__).parent / 'models' / 'pretrained' / 'v8_classes.json',
                Path.home() / '.stemscribe' / 'models' / 'v8_classes.json',
            ]
            for path in search_paths:
                if path.exists():
                    classes_path = str(path)
                    break

        if classes_path and Path(classes_path).exists():
            try:
                with open(classes_path, 'r') as f:
                    self.chord_classes = json.load(f)
                logger.info(f"Loaded {len(self.chord_classes)} chord classes from {classes_path}")
            except Exception as e:
                logger.error(f"Failed to load chord classes: {e}")
                self._create_default_classes()
        else:
            self._create_default_classes()

    def _create_default_classes(self):
        """Create default chord classes if JSON not found."""
        self.chord_classes = ['N']
        for q in ['maj', 'min', '7', 'maj7', 'min7', 'mMaj7', 'dim', 'dim7',
                  'hdim7', 'aug', 'sus2', 'sus4', '6', 'min6', '9', 'add9']:
            for n in NOTE_NAMES:
                if q == 'maj':
                    self.chord_classes.append(n)
                elif q == 'min':
                    self.chord_classes.append(f"{n}m")
                else:
                    self.chord_classes.append(f"{n}{q}")
        logger.warning("Using default chord classes (no inversions)")

    def _load_model(self, model_path: Optional[str] = None):
        """Load the V8 Transformer model."""
        if model_path is None:
            search_paths = [
                Path(__file__).parent / 'models' / 'pretrained' / 'v8_chord_model.pt',
                Path.home() / '.stemscribe' / 'models' / 'v8_chord_model.pt',
            ]
            for path in search_paths:
                if path.exists():
                    model_path = str(path)
                    break

        if model_path and Path(model_path).exists():
            try:
                n_classes = len(self.chord_classes) if self.chord_classes else 337
                self.model = V8TransformerChordModel(
                    n_classes=n_classes,
                    d_model=128,
                    nhead=8,
                    num_layers=3,
                    seq_len=self.context_frames
                )

                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()

                logger.info(f"V8 chord model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load V8 model: {e}")
                self.model = None
        else:
            logger.warning("V8 model not found. Falling back to template matching.")
            self.model = None

    def detect(self, audio_path: str) -> ChordProgression:
        """
        Detect chords from an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            ChordProgression with detected chords and key
        """
        if not LIBROSA_AVAILABLE:
            return ChordProgression(chords=[], key="Unknown")

        try:
            # Load audio and extract chroma features
            y, sr = librosa.load(str(audio_path), sr=22050)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
            times = librosa.times_like(chroma, sr=sr, hop_length=self.hop_length)

            # Detect chords using V8 model or fallback
            if self.model is not None:
                frame_results = self._predict_with_model(chroma)
            else:
                frame_results = self._predict_with_templates(chroma)

            # Consolidate into chord regions
            chords = self._consolidate(frame_results, times)

            # Detect key
            detected_key = self._detect_key(chroma)

            logger.info(f"V8 detected {len(chords)} chords, key: {detected_key}")
            return ChordProgression(chords=chords, key=detected_key)

        except Exception as e:
            logger.error(f"V8 chord detection failed: {e}")
            return ChordProgression(chords=[], key="Unknown")

    def _parse_chord_name(self, chord_name: str) -> tuple:
        """Parse a chord name into (root, quality, bass)."""
        if chord_name == 'N':
            return ('N', 'none', None)

        bass = None
        if '/' in chord_name:
            chord_part, bass = chord_name.rsplit('/', 1)
        else:
            chord_part = chord_name

        # Find root and quality
        for note in NOTE_NAMES:
            if chord_part.startswith(note):
                rest = chord_part[len(note):]
                if rest == '':
                    return (note, 'maj', bass)
                elif rest == 'm':
                    return (note, 'min', bass)
                elif rest in CHORD_TEMPLATES or rest in ['min7', 'min6']:
                    return (note, rest, bass)
                break

        return (chord_name, 'unknown', bass)

    def _predict_with_model(self, chroma: np.ndarray) -> List[tuple]:
        """Use V8 Transformer model for chord prediction."""
        n_frames = chroma.shape[1]
        results = []

        # Pad chroma for context window
        pad_size = self.context_frames // 2
        chroma_padded = np.pad(chroma, ((0, 0), (pad_size, pad_size)), mode='edge')

        with torch.no_grad():
            # Process in batches for efficiency
            batch_size = 64
            for batch_start in range(0, n_frames, batch_size):
                batch_end = min(batch_start + batch_size, n_frames)
                batch_frames = []

                for i in range(batch_start, batch_end):
                    # Extract context window
                    context = chroma_padded[:, i:i + self.context_frames].T
                    batch_frames.append(context)

                # Stack into batch tensor
                batch_tensor = torch.tensor(
                    np.array(batch_frames), dtype=torch.float32
                ).to(self.device)

                # Get predictions
                logits = self.model(batch_tensor)
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                confidences = probs.max(dim=-1).values

                # Convert to results
                for pred, conf in zip(predictions.cpu().numpy(), confidences.cpu().numpy()):
                    chord_name = self.chord_classes[pred]
                    root, quality, bass = self._parse_chord_name(chord_name)
                    results.append((chord_name, root, quality, bass, float(conf)))

        return results

    def _predict_with_templates(self, chroma: np.ndarray) -> List[tuple]:
        """Fallback template matching."""
        results = []
        for i in range(chroma.shape[1]):
            chroma_vector = chroma[:, i]

            if np.max(chroma_vector) == 0:
                results.append(("N", "N", "none", None, 0.0))
                continue

            chroma_norm = chroma_vector / np.max(chroma_vector)
            best_score, best_result = -1, ("N", "N", "none", None, 0.0)

            for root_idx in range(12):
                rotated = np.roll(chroma_norm, -root_idx)
                for quality, template in CHORD_TEMPLATES.items():
                    template_arr = np.array(template, dtype=float)
                    score = np.dot(rotated, template_arr) / (
                        np.linalg.norm(rotated) * np.linalg.norm(template_arr) + 1e-10
                    )

                    if score > best_score:
                        best_score = score
                        root = NOTE_NAMES[root_idx]
                        if quality == 'maj':
                            chord = root
                        elif quality == 'min':
                            chord = f"{root}m"
                        else:
                            chord = f"{root}{quality}"
                        best_result = (chord, root, quality, None, float(score))

            results.append(best_result if best_score >= 0.3 else ("N", "N", "none", None, 0.0))

        return results

    def _consolidate(self, frame_results: List[tuple], times: np.ndarray) -> List[ChordEvent]:
        """Consolidate frame-by-frame results into chord regions."""
        if not frame_results:
            return []

        regions = []
        current = frame_results[0]
        start_time = times[0]

        for i, result in enumerate(frame_results[1:], 1):
            if result[0] != current[0]:  # Chord changed
                duration = times[i] - start_time
                if duration >= self.min_duration and current[0] != "N":
                    regions.append(ChordEvent(
                        time=float(start_time),
                        duration=float(duration),
                        chord=current[0],
                        root=current[1],
                        quality=current[2],
                        bass=current[3],
                        confidence=current[4]
                    ))
                current = result
                start_time = times[i]

        # Handle last region
        if len(times) > 0 and current[0] != "N" and times[-1] - start_time >= self.min_duration:
            regions.append(ChordEvent(
                time=float(start_time),
                duration=float(times[-1] - start_time),
                chord=current[0],
                root=current[1],
                quality=current[2],
                bass=current[3],
                confidence=current[4]
            ))

        return regions

    def _detect_key(self, chroma: np.ndarray) -> str:
        """Detect the musical key using Krumhansl-Kessler profiles."""
        chroma_sum = np.sum(chroma, axis=1)
        if np.max(chroma_sum) > 0:
            chroma_sum = chroma_sum / np.max(chroma_sum)

        # Key profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        major_profile /= np.sum(major_profile)
        minor_profile /= np.sum(minor_profile)

        best_corr, best_key = -1, "C"

        for i in range(12):
            rotated = np.roll(chroma_sum, -i)

            major_corr = np.corrcoef(rotated, major_profile)[0, 1]
            minor_corr = np.corrcoef(rotated, minor_profile)[0, 1]

            if major_corr > best_corr:
                best_corr = major_corr
                best_key = NOTE_NAMES[i]
            if minor_corr > best_corr:
                best_corr = minor_corr
                best_key = f"{NOTE_NAMES[i]}m"

        return best_key


def detect_chords(audio_path: str, hop_length: int = 8192, min_duration: float = 0.5,
                  model_path: Optional[str] = None) -> List[dict]:
    """
    Convenience function for V8 chord detection.

    Args:
        audio_path: Path to audio file
        hop_length: Samples between frames
        min_duration: Minimum chord duration
        model_path: Optional path to model weights

    Returns:
        List of chord dictionaries with chord, start, end, confidence, bass
    """
    detector = ChordDetector(hop_length, min_duration, model_path)
    progression = detector.detect(audio_path)
    return [
        {
            'chord': c.chord,
            'start': c.time,
            'end': c.time + c.duration,
            'root': c.root,
            'quality': c.quality,
            'bass': c.bass,
            'confidence': c.confidence
        }
        for c in progression.chords
    ]
