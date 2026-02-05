"""
Stem Enhancement Module for StemScribe

Cleans up separated stems to sound more natural and isolated:
- Spectral gating to remove quiet instrument bleed
- Stem-specific EQ to cut frequencies where bleed lives
- Noise/artifact reduction for cleaner sound
- Light compression to polish the output
"""

import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import audio processing libraries
try:
    import pedalboard
    from pedalboard import (
        Pedalboard,
        Compressor,
        Gain,
        HighpassFilter,
        LowpassFilter,
        HighShelfFilter,
        LowShelfFilter,
        PeakFilter,
        NoiseGate,
        Limiter
    )
    from pedalboard.io import AudioFile
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False
    logger.warning("pedalboard not installed - stem enhancement unavailable")

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    logger.warning("noisereduce not installed - artifact reduction unavailable")


# Stem-specific enhancement profiles
# Each profile defines: EQ, gate threshold, compression settings
ENHANCEMENT_PROFILES = {
    'vocals': {
        'name': 'Vocals',
        'highpass': 80,        # Cut rumble below 80Hz
        'lowpass': 14000,      # Cut harsh highs above 14kHz
        'presence_freq': 3000, # Presence boost frequency
        'presence_gain': 2.0,  # Subtle presence boost (dB)
        'warmth_freq': 250,    # Warmth frequency
        'warmth_gain': 1.5,    # Subtle warmth (dB)
        'gate_threshold': -45, # dB - gate out quiet bleed
        'compression_threshold': -18,
        'compression_ratio': 2.5,
        'noise_reduce': True,
        'noise_reduce_strength': 0.3,  # Gentle - preserve naturalness
    },
    'guitar': {
        'name': 'Guitar',
        'highpass': 80,        # Cut sub-bass (not guitar territory)
        'lowpass': 12000,      # Cut above guitar harmonics
        'presence_freq': 2500, # Guitar presence/bite
        'presence_gain': 2.5,
        'warmth_freq': 400,    # Guitar body
        'warmth_gain': 1.5,
        'cut_freq': 800,       # Cut some piano bleed area
        'cut_gain': -2.0,      # Subtle cut
        'gate_threshold': -40,
        'compression_threshold': -15,
        'compression_ratio': 3.0,
        'noise_reduce': True,
        'noise_reduce_strength': 0.4,
    },
    'piano': {
        'name': 'Piano',
        'highpass': 60,        # Piano goes low
        'lowpass': 15000,      # Keep the sparkle
        'presence_freq': 2000, # Piano clarity
        'presence_gain': 1.5,
        'warmth_freq': 300,    # Piano body/warmth
        'warmth_gain': 2.0,
        'cut_freq': 3500,      # Cut some guitar presence bleed
        'cut_gain': -1.5,
        'gate_threshold': -42,
        'compression_threshold': -16,
        'compression_ratio': 2.5,
        'noise_reduce': True,
        'noise_reduce_strength': 0.35,
    },
    'bass': {
        'name': 'Bass',
        'highpass': 30,        # Keep the sub
        'lowpass': 5000,       # Bass doesn't need highs
        'presence_freq': 700,  # Bass attack/definition
        'presence_gain': 2.0,
        'warmth_freq': 100,    # Sub warmth
        'warmth_gain': 2.5,
        'gate_threshold': -38,
        'compression_threshold': -12,
        'compression_ratio': 4.0,  # Bass likes more compression
        'noise_reduce': True,
        'noise_reduce_strength': 0.5,  # Can be more aggressive on bass
    },
    'drums': {
        'name': 'Drums',
        'highpass': 40,        # Keep kick drum sub
        'lowpass': 16000,      # Keep cymbal air
        'presence_freq': 4000, # Snare crack
        'presence_gain': 2.0,
        'warmth_freq': 100,    # Kick thump
        'warmth_gain': 2.0,
        'gate_threshold': -35, # Drums can gate harder
        'compression_threshold': -14,
        'compression_ratio': 3.5,
        'noise_reduce': True,
        'noise_reduce_strength': 0.3,
    },
    'other': {
        'name': 'Other',
        'highpass': 60,
        'lowpass': 14000,
        'presence_freq': 2500,
        'presence_gain': 1.0,  # Gentle - we don't know what's here
        'warmth_freq': 300,
        'warmth_gain': 1.0,
        'gate_threshold': -45,
        'compression_threshold': -18,
        'compression_ratio': 2.0,
        'noise_reduce': True,
        'noise_reduce_strength': 0.25,
    }
}


def get_profile_for_stem(stem_name: str) -> dict:
    """Get the enhancement profile for a stem type"""
    stem_lower = stem_name.lower()

    # Check for exact match first
    if stem_lower in ENHANCEMENT_PROFILES:
        return ENHANCEMENT_PROFILES[stem_lower]

    # Check for partial matches (e.g., "other_guitar" -> "guitar")
    for key in ENHANCEMENT_PROFILES:
        if key in stem_lower:
            return ENHANCEMENT_PROFILES[key]

    # Default to 'other' profile
    return ENHANCEMENT_PROFILES['other']


def enhance_stem(input_path: str, output_path: str = None, stem_type: str = None) -> str:
    """
    Enhance a stem file to sound more natural and isolated.

    Args:
        input_path: Path to the input stem audio file
        output_path: Path for enhanced output (default: overwrites input)
        stem_type: Type of stem (vocals, guitar, piano, bass, drums, other)
                   If None, will try to detect from filename

    Returns:
        Path to the enhanced file
    """
    if not PEDALBOARD_AVAILABLE:
        logger.warning("pedalboard not available - skipping enhancement")
        return input_path

    input_path = Path(input_path)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return str(input_path)

    # Determine stem type from filename if not provided
    if stem_type is None:
        stem_type = input_path.stem.lower()

    # Get enhancement profile
    profile = get_profile_for_stem(stem_type)
    logger.info(f"🎛️ Enhancing {stem_type} with '{profile['name']}' profile")

    # Set output path
    if output_path is None:
        # Create enhanced version alongside original
        output_path = input_path.parent / f"{input_path.stem}_enhanced{input_path.suffix}"
    output_path = Path(output_path)

    try:
        # Read the audio file
        with AudioFile(str(input_path)) as f:
            audio = f.read(f.frames)
            sample_rate = f.samplerate

        # Step 1: Noise reduction (if available)
        if NOISEREDUCE_AVAILABLE and profile.get('noise_reduce', False):
            strength = profile.get('noise_reduce_strength', 0.3)
            logger.info(f"  → Applying noise reduction (strength={strength})")

            # noisereduce expects (samples, channels) but pedalboard gives (channels, samples)
            audio_for_nr = audio.T if audio.ndim > 1 else audio

            # Apply noise reduction
            audio_reduced = nr.reduce_noise(
                y=audio_for_nr,
                sr=sample_rate,
                prop_decrease=strength,
                stationary=False,  # Non-stationary for music
                n_fft=2048,
                hop_length=512,
            )

            # Convert back to pedalboard format
            audio = audio_reduced.T if audio_reduced.ndim > 1 else audio_reduced
            audio = audio.astype(np.float32)

        # Step 2: Build the pedalboard effects chain
        effects = []

        # Noise gate - remove quiet bleed
        gate_threshold = profile.get('gate_threshold', -45)
        effects.append(NoiseGate(
            threshold_db=gate_threshold,
            attack_ms=5.0,
            release_ms=50.0,
            ratio=2.0
        ))

        # Highpass filter - remove rumble/bleed below instrument range
        if profile.get('highpass'):
            effects.append(HighpassFilter(cutoff_frequency_hz=profile['highpass']))

        # Lowpass filter - remove hiss/bleed above instrument range
        if profile.get('lowpass'):
            effects.append(LowpassFilter(cutoff_frequency_hz=profile['lowpass']))

        # Cut frequency (reduce bleed from other instruments)
        if profile.get('cut_freq') and profile.get('cut_gain'):
            effects.append(PeakFilter(
                cutoff_frequency_hz=profile['cut_freq'],
                gain_db=profile['cut_gain'],
                q=1.0
            ))

        # Warmth boost
        if profile.get('warmth_freq') and profile.get('warmth_gain'):
            effects.append(LowShelfFilter(
                cutoff_frequency_hz=profile['warmth_freq'],
                gain_db=profile['warmth_gain'],
                q=0.7
            ))

        # Presence boost
        if profile.get('presence_freq') and profile.get('presence_gain'):
            effects.append(PeakFilter(
                cutoff_frequency_hz=profile['presence_freq'],
                gain_db=profile['presence_gain'],
                q=1.5
            ))

        # Compression - glue it together
        effects.append(Compressor(
            threshold_db=profile.get('compression_threshold', -18),
            ratio=profile.get('compression_ratio', 2.5),
            attack_ms=10.0,
            release_ms=100.0
        ))

        # Makeup gain
        effects.append(Gain(gain_db=2.0))

        # Limiter - prevent clipping
        effects.append(Limiter(threshold_db=-1.0, release_ms=100.0))

        # Create and apply the pedalboard
        board = Pedalboard(effects)
        logger.info(f"  → Applying {len(effects)} effects")

        enhanced_audio = board(audio, sample_rate)

        # Step 3: Write the enhanced audio
        with AudioFile(str(output_path), 'w', sample_rate, enhanced_audio.shape[0]) as f:
            f.write(enhanced_audio)

        logger.info(f"✅ Enhanced stem saved: {output_path}")
        return str(output_path)

    except Exception as e:
        logger.error(f"Enhancement failed for {input_path}: {e}")
        return str(input_path)  # Return original on failure


def enhance_all_stems(stems_dict: dict, output_dir: str = None) -> dict:
    """
    Enhance all stems in a job.

    Args:
        stems_dict: Dictionary of {stem_name: stem_path}
        output_dir: Directory for enhanced files (default: same as input)

    Returns:
        Dictionary of {stem_name: enhanced_path}
    """
    if not PEDALBOARD_AVAILABLE:
        logger.warning("pedalboard not available - returning original stems")
        return stems_dict

    enhanced_stems = {}

    for stem_name, stem_path in stems_dict.items():
        try:
            if output_dir:
                output_path = Path(output_dir) / f"{stem_name}_enhanced.mp3"
            else:
                output_path = None  # Will create alongside original

            enhanced_path = enhance_stem(
                input_path=stem_path,
                output_path=output_path,
                stem_type=stem_name
            )
            enhanced_stems[stem_name] = enhanced_path

        except Exception as e:
            logger.error(f"Failed to enhance {stem_name}: {e}")
            enhanced_stems[stem_name] = stem_path  # Keep original on failure

    return enhanced_stems


# Quick test
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print(f"Pedalboard available: {PEDALBOARD_AVAILABLE}")
    print(f"Noisereduce available: {NOISEREDUCE_AVAILABLE}")
    print(f"Enhancement profiles: {list(ENHANCEMENT_PROFILES.keys())}")
