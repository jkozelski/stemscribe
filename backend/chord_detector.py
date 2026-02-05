"""
Chord Detection for StemScribe
==============================
Detects chord progressions from audio using librosa chroma features.
Similar to Logic Pro's chord detection.
"""

import numpy as np
import logging
from pathlib import Path
from typing import List, NamedTuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

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
    chord: str  # Full chord name, e.g., "Am7"
    root: str   # Root note, e.g., "A"
    quality: str  # Chord quality, e.g., "min7"
    confidence: float


@dataclass
class ChordProgression:
    """Result of chord detection."""
    chords: List[ChordEvent]
    key: str


# Chord templates - 12 pitch classes (C through B)
CHORD_TEMPLATES = {
    'maj': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'min': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    '7':   [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj7': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    'dim': [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    'sus4': [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
}

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


class ChordDetector:
    """Main chord detection class."""
    
    def __init__(self, hop_length: int = 8192, min_duration: float = 0.5):
        self.hop_length = hop_length
        self.min_duration = min_duration
    
    def detect(self, audio_path: str) -> ChordProgression:
        """Detect chords from audio file."""
        if not LIBROSA_AVAILABLE:
            return ChordProgression(chords=[], key="Unknown")
        
        try:
            y, sr = librosa.load(str(audio_path), sr=22050)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
            times = librosa.times_like(chroma, sr=sr, hop_length=self.hop_length)
            
            # Detect chord for each frame
            frame_results = [self._match_chord(chroma[:, i]) for i in range(chroma.shape[1])]
            
            # Consolidate into chord regions
            chords = self._consolidate(frame_results, times)
            
            # Detect key
            detected_key = self._detect_key(chroma)
            
            logger.info(f"Detected {len(chords)} chords, key: {detected_key}")
            return ChordProgression(chords=chords, key=detected_key)
            
        except Exception as e:
            logger.error(f"Chord detection failed: {e}")
            return ChordProgression(chords=[], key="Unknown")
    
    def _match_chord(self, chroma_vector: np.ndarray) -> tuple:
        """Match chroma vector to best chord. Returns (chord, root, quality, confidence)."""
        if np.max(chroma_vector) == 0:
            return ("N", "N", "none", 0.0)
        
        chroma_norm = chroma_vector / np.max(chroma_vector)
        best_score, best_result = -1, ("N", "N", "none", 0.0)
        
        for root_idx in range(12):
            rotated = np.roll(chroma_norm, -root_idx)
            for quality, template in CHORD_TEMPLATES.items():
                template_arr = np.array(template, dtype=float)
                score = np.dot(rotated, template_arr) / (np.linalg.norm(rotated) * np.linalg.norm(template_arr) + 1e-10)
                
                if score > best_score:
                    best_score = score
                    root = NOTE_NAMES[root_idx]
                    if quality == 'maj':
                        chord = root
                    elif quality == 'min':
                        chord = f"{root}m"
                    else:
                        chord = f"{root}{quality}"
                    best_result = (chord, root, quality, float(score))
        
        return best_result if best_score >= 0.3 else ("N", "N", "none", 0.0)
    
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
                        confidence=current[3]
                    ))
                current = result
                start_time = times[i]
        
        # Last region
        if current[0] != "N" and times[-1] - start_time >= self.min_duration:
            regions.append(ChordEvent(
                time=float(start_time),
                duration=float(times[-1] - start_time),
                chord=current[0],
                root=current[1],
                quality=current[2],
                confidence=current[3]
            ))
        
        return regions
    
    def _detect_key(self, chroma: np.ndarray) -> str:
        """Detect the musical key from chroma features."""
        # Sum chroma across time
        chroma_sum = np.sum(chroma, axis=1)
        chroma_sum = chroma_sum / np.max(chroma_sum) if np.max(chroma_sum) > 0 else chroma_sum
        
        # Major and minor key profiles (Krumhansl-Kessler)
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


def detect_chords(audio_path: str, hop_length: int = 8192, min_duration: float = 0.5) -> List[dict]:
    """Convenience function for chord detection."""
    detector = ChordDetector(hop_length, min_duration)
    progression = detector.detect(audio_path)
    return [{'chord': c.chord, 'start': c.time, 'end': c.time + c.duration, 'confidence': c.confidence}
            for c in progression.chords]
