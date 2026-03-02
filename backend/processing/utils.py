"""
Audio processing utilities — stem content check, WAV-to-MP3 conversion.
"""

import os
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def check_stem_has_content(stem_path: str, threshold: float = 0.01) -> bool:
    """Check if a stem has significant audio content worth processing further"""
    try:
        from scipy.io import wavfile
        import numpy as np

        # Handle MP3 files by checking file size as proxy
        if stem_path.endswith('.mp3'):
            file_size = os.path.getsize(stem_path)
            # If MP3 is less than 50KB, probably mostly silence
            return file_size > 50000

        sr, audio = wavfile.read(stem_path)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0

        # Calculate RMS energy
        if len(audio.shape) == 2:
            audio = np.mean(audio, axis=1)
        rms = np.sqrt(np.mean(audio ** 2))

        return rms > threshold
    except Exception as e:
        logger.warning(f"Could not check stem content: {e}")
        return True  # Assume it has content if we can't check


def convert_wavs_to_mp3(directory: Path) -> dict:
    """
    Convert all WAV files in a directory to MP3 using ffmpeg.
    This avoids the lameenc dependency which doesn't support Python 3.14.
    Returns dict of {stem_name: mp3_path}
    """
    converted = {}
    for wav_file in directory.glob('*.wav'):
        mp3_file = wav_file.with_suffix('.mp3')
        try:
            _result = subprocess.run([
                'ffmpeg', '-y', '-i', str(wav_file),
                '-codec:a', 'libmp3lame', '-b:a', '320k',
                str(mp3_file)
            ], capture_output=True, text=True)

            if mp3_file.exists():
                wav_file.unlink()  # Remove WAV to save space
                converted[wav_file.stem] = str(mp3_file)
                logger.info(f"Converted {wav_file.stem} to MP3")
            else:
                # Keep WAV if conversion failed
                converted[wav_file.stem] = str(wav_file)
                logger.warning(f"MP3 conversion failed for {wav_file.stem}, keeping WAV")
        except Exception as e:
            converted[wav_file.stem] = str(wav_file)
            logger.warning(f"MP3 conversion error for {wav_file.stem}: {e}")

    return converted
