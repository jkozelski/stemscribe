"""
StemScribe Skills System
========================
Skills are specialized processing chains that can:
1. Post-process existing stems (EQ, filtering, enhancement)
2. Generate sub-stems from parent stems (e.g., extract horns from "other")
3. Apply genre-specific optimizations

Each skill defines:
- name: Display name
- description: What it does
- target_stems: Which stems it processes
- generates: What new sub-stems it creates
- process(): The actual processing function
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum


class StemType(Enum):
    """Standard stem types from Demucs htdemucs_6s"""
    VOCALS = "vocals"
    BASS = "bass"
    DRUMS = "drums"
    GUITAR = "guitar"
    PIANO = "piano"
    OTHER = "other"


@dataclass
class SubStem:
    """A sub-stem extracted from a parent stem"""
    name: str
    parent: StemType
    audio_path: Optional[str] = None
    description: str = ""


@dataclass
class Skill:
    """Base skill definition"""
    id: str
    name: str
    emoji: str
    description: str
    target_stems: List[StemType]
    generates: List[str]  # Names of sub-stems this skill creates
    genre_tags: List[str] = field(default_factory=list)

    def process(self, stem_paths: Dict[str, str], output_dir: str, sample_rate: int = 44100) -> Dict[str, str]:
        """
        Process stems and return paths to generated sub-stems.
        Override this in specific skill implementations.
        """
        raise NotImplementedError


class FrequencyIsolator:
    """Utility class for frequency-based stem isolation"""

    @staticmethod
    def bandpass_filter(audio: np.ndarray, low_freq: float, high_freq: float,
                        sample_rate: int = 44100, order: int = 5) -> np.ndarray:
        """Apply bandpass filter to isolate frequency range"""
        nyquist = sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist

        # Clamp to valid range
        low = max(0.001, min(low, 0.999))
        high = max(0.001, min(high, 0.999))

        if low >= high:
            return audio

        b, a = signal.butter(order, [low, high], btype='band')

        # Handle stereo
        if len(audio.shape) == 2:
            filtered = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                filtered[:, ch] = signal.filtfilt(b, a, audio[:, ch])
            return filtered
        else:
            return signal.filtfilt(b, a, audio)

    @staticmethod
    def highpass_filter(audio: np.ndarray, cutoff: float,
                        sample_rate: int = 44100, order: int = 5) -> np.ndarray:
        """Apply highpass filter"""
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        normalized_cutoff = max(0.001, min(normalized_cutoff, 0.999))

        b, a = signal.butter(order, normalized_cutoff, btype='high')

        if len(audio.shape) == 2:
            filtered = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                filtered[:, ch] = signal.filtfilt(b, a, audio[:, ch])
            return filtered
        else:
            return signal.filtfilt(b, a, audio)

    @staticmethod
    def lowpass_filter(audio: np.ndarray, cutoff: float,
                       sample_rate: int = 44100, order: int = 5) -> np.ndarray:
        """Apply lowpass filter"""
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        normalized_cutoff = max(0.001, min(normalized_cutoff, 0.999))

        b, a = signal.butter(order, normalized_cutoff, btype='low')

        if len(audio.shape) == 2:
            filtered = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                filtered[:, ch] = signal.filtfilt(b, a, audio[:, ch])
            return filtered
        else:
            return signal.filtfilt(b, a, audio)


class AudioAnalyzer:
    """
    Analyzes audio stems to detect what instruments are likely present.
    Used to decide which skills should run automatically.
    """

    @staticmethod
    def get_rms_energy(audio: np.ndarray) -> float:
        """Calculate RMS energy of audio signal"""
        if len(audio.shape) == 2:
            audio = np.mean(audio, axis=1)  # Convert to mono
        return np.sqrt(np.mean(audio ** 2))

    @staticmethod
    def get_frequency_energy(audio: np.ndarray, low_freq: float, high_freq: float,
                             sample_rate: int = 44100) -> float:
        """Calculate energy in a specific frequency band"""
        if len(audio.shape) == 2:
            audio = np.mean(audio, axis=1)  # Convert to mono

        # Compute FFT
        n = len(audio)
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(n, 1/sample_rate)

        # Find frequency bin indices
        low_idx = np.searchsorted(freqs, low_freq)
        high_idx = np.searchsorted(freqs, high_freq)

        # Calculate energy in band
        band_energy = np.sum(np.abs(fft[low_idx:high_idx]) ** 2)
        total_energy = np.sum(np.abs(fft) ** 2)

        if total_energy == 0:
            return 0

        return band_energy / total_energy

    @staticmethod
    def has_significant_content(audio_path: str, threshold: float = 0.01) -> bool:
        """Check if a stem has significant audio content"""
        try:
            sr, audio = wavfile.read(audio_path)
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0

            rms = AudioAnalyzer.get_rms_energy(audio)
            return rms > threshold
        except Exception:
            return False

    @staticmethod
    def detect_horns(other_stem_path: str, threshold: float = 0.15) -> bool:
        """
        Detect if the 'other' stem likely contains horns/brass.
        Horns have strong energy in 200Hz-3kHz range with characteristic attack.
        """
        try:
            sr, audio = wavfile.read(other_stem_path)
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0

            # Check energy in brass frequency range (200Hz - 3kHz)
            brass_energy = AudioAnalyzer.get_frequency_energy(audio, 200, 3000, sr)

            # Horns typically have significant mid-range presence
            return brass_energy > threshold
        except Exception:
            return False

    @staticmethod
    def detect_strings(other_stem_path: str, threshold: float = 0.12) -> bool:
        """
        Detect if the 'other' stem likely contains orchestral strings.
        Strings have sustained energy across 200Hz-8kHz with smooth envelope.
        """
        try:
            sr, audio = wavfile.read(other_stem_path)
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0

            # Check energy in string frequency range
            string_energy = AudioAnalyzer.get_frequency_energy(audio, 200, 6000, sr)

            return string_energy > threshold
        except Exception:
            return False

    @staticmethod
    def detect_guitar_content(guitar_stem_path: str, threshold: float = 0.02) -> bool:
        """Check if guitar stem has significant content worth splitting"""
        return AudioAnalyzer.has_significant_content(guitar_stem_path, threshold)

    @staticmethod
    def detect_keys_content(piano_stem_path: str, threshold: float = 0.02) -> bool:
        """Check if piano/keys stem has significant content worth splitting"""
        return AudioAnalyzer.has_significant_content(piano_stem_path, threshold)

    @staticmethod
    def detect_vocal_content(vocals_stem_path: str, threshold: float = 0.02) -> bool:
        """Check if vocals stem has significant content worth splitting into lead/harmonies"""
        return AudioAnalyzer.has_significant_content(vocals_stem_path, threshold)


def analyze_stems_for_skills(stem_paths: Dict[str, str]) -> List[str]:
    """
    Analyze separated stems and return list of skill IDs that should run.
    This is the "smart" detection that listens to the audio.
    """
    skills_to_run = []
    analyzer = AudioAnalyzer

    # Check if 'other' stem has horn content
    other_path = stem_paths.get('other')
    if other_path and os.path.exists(other_path):
        if analyzer.detect_horns(other_path):
            skills_to_run.append('horn_hunter')
            print("  🎷 Detected horn/brass content - will extract horns")

        if analyzer.detect_strings(other_path):
            skills_to_run.append('string_section')
            print("  🎻 Detected string content - will extract strings")

    # Check if guitar stem has content worth splitting
    guitar_path = stem_paths.get('guitar')
    if guitar_path and os.path.exists(guitar_path):
        if analyzer.detect_guitar_content(guitar_path):
            skills_to_run.append('guitar_god')
            print("  🎸 Detected guitar content - will split into lead/rhythm/clean/distorted")

    # Check if piano/keys stem has content worth splitting
    piano_path = stem_paths.get('piano')
    if piano_path and os.path.exists(piano_path):
        if analyzer.detect_keys_content(piano_path):
            skills_to_run.append('keys_kingdom')
            print("  🎹 Detected keys content - will split into acoustic/electric/synth")

    # Check if vocals stem has content worth splitting into lead/harmonies
    vocals_path = stem_paths.get('vocals')
    if vocals_path and os.path.exists(vocals_path):
        if analyzer.detect_vocal_content(vocals_path):
            skills_to_run.append('vocal_virtuoso')
            print("  🎤 Detected vocal content - will split into lead/harmony/high/low")

    return skills_to_run


class HornHunterSkill(Skill):
    """
    Horn Hunter Skill
    =================
    Extracts brass/horn instruments from the "other" stem.

    Frequency ranges:
    - Trumpet: 180Hz - 1kHz (fundamental), harmonics up to 10kHz
    - Trombone: 80Hz - 500Hz (fundamental), harmonics up to 5kHz
    - Saxophone: 100Hz - 900Hz (fundamental), harmonics up to 8kHz
    - French Horn: 60Hz - 700Hz

    General brass band: 150Hz - 4kHz with strong presence 800Hz-2kHz
    """

    def __init__(self):
        super().__init__(
            id="horn_hunter",
            name="Horn Hunter",
            emoji="🎷",
            description="Isolate brass and horn instruments (trumpet, sax, trombone)",
            target_stems=[StemType.OTHER],
            generates=["horns", "other_no_horns"],
            genre_tags=["jazz", "funk", "soul", "ska", "big band"]
        )

        # Frequency ranges for different brass instruments
        self.horn_ranges = {
            "low_brass": (80, 400),      # Trombone, tuba fundamentals
            "mid_brass": (300, 1200),    # Trumpet, sax fundamentals
            "high_brass": (800, 4000),   # Brass presence/bite
            "air_harmonics": (3000, 8000) # Upper harmonics, breathiness
        }

    def process(self, stem_paths: Dict[str, str], output_dir: str, sample_rate: int = 44100) -> Dict[str, str]:
        """Extract horns from 'other' stem"""
        other_path = stem_paths.get("other")
        if not other_path or not os.path.exists(other_path):
            return {}

        # Load the other stem
        sr, audio = wavfile.read(other_path)

        # Normalize to float
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0

        isolator = FrequencyIsolator()

        # Extract horn frequencies with multiple bandpass filters
        horn_audio = np.zeros_like(audio)

        # Layer 1: Core brass range (most important)
        core_brass = isolator.bandpass_filter(audio, 200, 2500, sr, order=4)
        horn_audio += core_brass * 0.6

        # Layer 2: Brass presence/attack
        presence = isolator.bandpass_filter(audio, 1500, 4000, sr, order=3)
        horn_audio += presence * 0.25

        # Layer 3: Low brass fundamentals
        low_brass = isolator.bandpass_filter(audio, 80, 300, sr, order=3)
        horn_audio += low_brass * 0.15

        # Normalize
        max_val = np.max(np.abs(horn_audio))
        if max_val > 0:
            horn_audio = horn_audio / max_val * 0.9

        # Create "other without horns" by subtracting
        other_no_horns = audio - (horn_audio * 0.7)  # Partial subtraction to avoid artifacts

        # Normalize
        max_val = np.max(np.abs(other_no_horns))
        if max_val > 0:
            other_no_horns = other_no_horns / max_val * 0.9

        # Save outputs
        horns_path = os.path.join(output_dir, "horns.wav")
        other_remaining_path = os.path.join(output_dir, "other_no_horns.wav")

        # Convert back to int16
        horn_audio_int = (horn_audio * 32767).astype(np.int16)
        other_no_horns_int = (other_no_horns * 32767).astype(np.int16)

        wavfile.write(horns_path, sr, horn_audio_int)
        wavfile.write(other_remaining_path, sr, other_no_horns_int)

        return {
            "horns": horns_path,
            "other_no_horns": other_remaining_path
        }


class GuitarGodSkill(Skill):
    """
    Guitar God Skill
    ================
    Enhances guitar separation and splits into:
    - Clean/acoustic guitar tones
    - Distorted/electric guitar tones
    - Lead guitar (higher register)
    - Rhythm guitar (mid register)
    """

    def __init__(self):
        super().__init__(
            id="guitar_god",
            name="Guitar God",
            emoji="🎸",
            description="Split guitar into clean/distorted and lead/rhythm",
            target_stems=[StemType.GUITAR],
            generates=["guitar_clean", "guitar_distorted", "guitar_lead", "guitar_rhythm"],
            genre_tags=["rock", "metal", "blues", "country", "indie"]
        )

    def process(self, stem_paths: Dict[str, str], output_dir: str, sample_rate: int = 44100) -> Dict[str, str]:
        """Split guitar stem into sub-categories"""
        guitar_path = stem_paths.get("guitar")
        if not guitar_path or not os.path.exists(guitar_path):
            return {}

        sr, audio = wavfile.read(guitar_path)

        # Normalize to float
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0

        isolator = FrequencyIsolator()

        # Lead guitar: Higher frequencies, single note lines
        # Typically 500Hz - 5kHz with emphasis on 1-3kHz
        guitar_lead = isolator.bandpass_filter(audio, 800, 5000, sr, order=3)

        # Rhythm guitar: Lower-mid frequencies, chords
        # Typically 100Hz - 1.5kHz
        guitar_rhythm = isolator.bandpass_filter(audio, 100, 1500, sr, order=3)

        # Clean guitar: Less high-frequency harmonic content
        # Roll off above 4kHz, emphasize 200Hz-2kHz
        guitar_clean = isolator.bandpass_filter(audio, 200, 4000, sr, order=4)
        guitar_clean = isolator.lowpass_filter(guitar_clean, 4500, sr, order=2)

        # Distorted guitar: More harmonic content, presence at 2-5kHz
        # Emphasize the "crunch" frequencies
        guitar_distorted = isolator.bandpass_filter(audio, 150, 6000, sr, order=3)
        distortion_presence = isolator.bandpass_filter(audio, 2000, 5000, sr, order=2)
        guitar_distorted = guitar_distorted + (distortion_presence * 0.3)

        # Normalize all outputs
        outputs = {
            "guitar_lead": guitar_lead,
            "guitar_rhythm": guitar_rhythm,
            "guitar_clean": guitar_clean,
            "guitar_distorted": guitar_distorted
        }

        result_paths = {}
        for name, audio_data in outputs.items():
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * 0.9

            audio_int = (audio_data * 32767).astype(np.int16)
            path = os.path.join(output_dir, f"{name}.wav")
            wavfile.write(path, sr, audio_int)
            result_paths[name] = path

        return result_paths


class StringSectionSkill(Skill):
    """
    String Section Skill
    ====================
    Isolates orchestral strings from the "other" stem.
    """

    def __init__(self):
        super().__init__(
            id="string_section",
            name="String Section",
            emoji="🎻",
            description="Isolate orchestral strings (violin, viola, cello)",
            target_stems=[StemType.OTHER],
            generates=["strings_high", "strings_low"],
            genre_tags=["classical", "orchestral", "cinematic", "pop"]
        )

    def process(self, stem_paths: Dict[str, str], output_dir: str, sample_rate: int = 44100) -> Dict[str, str]:
        """Extract strings from 'other' stem"""
        other_path = stem_paths.get("other")
        if not other_path or not os.path.exists(other_path):
            return {}

        sr, audio = wavfile.read(other_path)

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0

        isolator = FrequencyIsolator()

        # High strings (violin, viola): 200Hz - 8kHz
        strings_high = isolator.bandpass_filter(audio, 400, 8000, sr, order=4)

        # Low strings (cello, bass): 60Hz - 1kHz
        strings_low = isolator.bandpass_filter(audio, 60, 1000, sr, order=4)

        outputs = {
            "strings_high": strings_high,
            "strings_low": strings_low
        }

        result_paths = {}
        for name, audio_data in outputs.items():
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * 0.9

            audio_int = (audio_data * 32767).astype(np.int16)
            path = os.path.join(output_dir, f"{name}.wav")
            wavfile.write(path, sr, audio_int)
            result_paths[name] = path

        return result_paths


class KeysKingdomSkill(Skill):
    """
    Keys Kingdom Skill
    ==================
    Enhances piano/keys separation into sub-categories.
    """

    def __init__(self):
        super().__init__(
            id="keys_kingdom",
            name="Keys Kingdom",
            emoji="🎹",
            description="Split keys into piano, organ, and synth",
            target_stems=[StemType.PIANO, StemType.OTHER],
            generates=["keys_acoustic", "keys_electric", "keys_synth"],
            genre_tags=["jazz", "soul", "electronic", "classical", "rock"]
        )

    def process(self, stem_paths: Dict[str, str], output_dir: str, sample_rate: int = 44100) -> Dict[str, str]:
        """Process piano and extract synth keys from other"""
        piano_path = stem_paths.get("piano")
        other_path = stem_paths.get("other")

        isolator = FrequencyIsolator()
        result_paths = {}

        # Process piano stem if available
        if piano_path and os.path.exists(piano_path):
            sr, audio = wavfile.read(piano_path)

            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0

            # Acoustic piano: Full range with natural rolloff
            keys_acoustic = isolator.bandpass_filter(audio, 27, 4500, sr, order=3)

            # Electric piano (Rhodes, Wurli): Emphasize 200Hz-3kHz
            keys_electric = isolator.bandpass_filter(audio, 200, 3000, sr, order=4)

            for name, audio_data in [("keys_acoustic", keys_acoustic), ("keys_electric", keys_electric)]:
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    audio_data = audio_data / max_val * 0.9
                audio_int = (audio_data * 32767).astype(np.int16)
                path = os.path.join(output_dir, f"{name}.wav")
                wavfile.write(path, sr, audio_int)
                result_paths[name] = path

        # Extract synth from other stem
        if other_path and os.path.exists(other_path):
            sr, audio = wavfile.read(other_path)

            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0

            # Synth: Can be anywhere, but often has strong harmonics
            # Look for sustained tones in mid-high range
            keys_synth = isolator.bandpass_filter(audio, 100, 8000, sr, order=3)

            max_val = np.max(np.abs(keys_synth))
            if max_val > 0:
                keys_synth = keys_synth / max_val * 0.9

            audio_int = (keys_synth * 32767).astype(np.int16)
            path = os.path.join(output_dir, "keys_synth.wav")
            wavfile.write(path, sr, audio_int)
            result_paths["keys_synth"] = path

        return result_paths


class VocalVirtuosoSkill(Skill):
    """
    Vocal Virtuoso Skill
    ====================
    Splits vocals into different components:
    - Lead vocals (center, prominent)
    - Background/harmony vocals (wider stereo, quieter)
    - High vocals (falsetto, high harmonies)
    - Low vocals (bass vocals, low harmonies)

    Uses frequency and stereo analysis to separate vocal parts.
    """

    def __init__(self):
        super().__init__(
            id="vocal_virtuoso",
            name="Vocal Virtuoso",
            emoji="🎤",
            description="Split vocals into lead, harmonies, and ranges",
            target_stems=[StemType.VOCALS],
            generates=["vocals_lead", "vocals_harmony", "vocals_high", "vocals_low"],
            genre_tags=["pop", "r&b", "soul", "gospel", "rock"]
        )

    def process(self, stem_paths: Dict[str, str], output_dir: str, sample_rate: int = 44100) -> Dict[str, str]:
        """Split vocals into components"""
        vocals_path = stem_paths.get("vocals")
        if not vocals_path or not os.path.exists(vocals_path):
            return {}

        sr, audio = wavfile.read(vocals_path)

        # Normalize to float
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0

        isolator = FrequencyIsolator()
        result_paths = {}

        # Check if stereo
        is_stereo = len(audio.shape) == 2 and audio.shape[1] == 2

        if is_stereo:
            left = audio[:, 0]
            right = audio[:, 1]

            # Mid (center) = (L + R) / 2 - typically lead vocals
            mid = (left + right) / 2

            # Side (stereo difference) = (L - R) / 2 - often harmonies, effects
            side = (left - right) / 2

            # Lead vocals: center channel, main vocal range (200Hz - 4kHz)
            vocals_lead = isolator.bandpass_filter(
                np.column_stack([mid, mid]), 200, 4000, sr, order=3
            )

            # Harmony vocals: side channel + wider frequency range
            vocals_harmony = isolator.bandpass_filter(
                np.column_stack([side, side]), 150, 5000, sr, order=3
            )
            # Mix back some center for body
            vocals_harmony = vocals_harmony * 0.7 + isolator.bandpass_filter(
                np.column_stack([mid, mid]), 300, 3000, sr, order=2
            ) * 0.3
        else:
            # Mono: use frequency-based separation
            mono = audio if len(audio.shape) == 1 else np.mean(audio, axis=1)

            # Lead vocals: core vocal range
            vocals_lead = isolator.bandpass_filter(mono, 200, 4000, sr, order=4)
            vocals_lead = np.column_stack([vocals_lead, vocals_lead])

            # Harmony: slightly different EQ curve
            vocals_harmony = isolator.bandpass_filter(mono, 300, 5000, sr, order=3)
            vocals_harmony = np.column_stack([vocals_harmony, vocals_harmony])

        # High vocals: falsetto, high harmonies (800Hz - 8kHz)
        vocals_high = isolator.bandpass_filter(audio, 800, 8000, sr, order=4)

        # Low vocals: bass vocals, low harmonies (80Hz - 400Hz)
        vocals_low = isolator.bandpass_filter(audio, 80, 400, sr, order=4)

        # Save all outputs
        outputs = {
            "vocals_lead": vocals_lead,
            "vocals_harmony": vocals_harmony,
            "vocals_high": vocals_high,
            "vocals_low": vocals_low
        }

        for name, audio_data in outputs.items():
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * 0.9

            audio_int = (audio_data * 32767).astype(np.int16)
            path = os.path.join(output_dir, f"{name}.wav")
            wavfile.write(path, sr, audio_int)
            result_paths[name] = path

        return result_paths


# Registry of all available skills
SKILL_REGISTRY: Dict[str, Skill] = {
    "vocal_virtuoso": VocalVirtuosoSkill(),
    "horn_hunter": HornHunterSkill(),
    "guitar_god": GuitarGodSkill(),
    "string_section": StringSectionSkill(),
    "keys_kingdom": KeysKingdomSkill(),
}


def get_skill(skill_id: str) -> Optional[Skill]:
    """Get a skill by ID"""
    return SKILL_REGISTRY.get(skill_id)


def get_all_skills() -> List[Skill]:
    """Get all available skills"""
    return list(SKILL_REGISTRY.values())


def apply_skill(skill_id: str, stem_paths: Dict[str, str], output_dir: str) -> Dict[str, str]:
    """
    Apply a skill to stems and return paths to generated sub-stems.

    Args:
        skill_id: The skill ID to apply
        stem_paths: Dict mapping stem names to file paths
        output_dir: Directory to save generated sub-stems

    Returns:
        Dict mapping sub-stem names to file paths
    """
    skill = get_skill(skill_id)
    if not skill:
        raise ValueError(f"Unknown skill: {skill_id}")

    # Create output directory if needed
    skill_output_dir = os.path.join(output_dir, f"skill_{skill_id}")
    os.makedirs(skill_output_dir, exist_ok=True)

    return skill.process(stem_paths, skill_output_dir)
