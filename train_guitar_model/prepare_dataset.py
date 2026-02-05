#!/usr/bin/env python3
"""
Guitar Lead/Rhythm Dataset Preparation

This script prepares training data for a lead/rhythm guitar separation model by:
1. Downloading songs from YouTube (songs known to have panned guitars)
2. Extracting guitar stems using Demucs
3. Splitting stereo guitar into left/right (lead/rhythm) using stereo analysis
4. Organizing into training folder structure

The key insight: Many classic rock songs have rhythm guitar panned to one side
and lead guitar panned to the other. We can use this as "ground truth" for training.

Usage:
    python prepare_guitar_dataset.py --songs songs.txt --output ./dataset
"""

import os
import sys
import json
import shutil
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Songs with known stereo guitar panning (rhythm one side, lead other)
# Format: (YouTube URL or search query, Artist, Title, notes)
RECOMMENDED_SONGS = [
    # Classic Rock - Known for dual guitar panning
    ("Hotel California Eagles", "Eagles", "Hotel California", "Dual guitars panned L/R"),
    ("Stairway to Heaven Led Zeppelin", "Led Zeppelin", "Stairway to Heaven", "12-string rhythm vs lead"),
    ("Free Bird Lynyrd Skynyrd", "Lynyrd Skynyrd", "Free Bird", "Triple guitar harmonies"),
    ("Comfortably Numb Pink Floyd", "Pink Floyd", "Comfortably Numb", "Gilmour lead vs rhythm"),
    ("Sweet Child O Mine Guns N Roses", "Guns N' Roses", "Sweet Child O' Mine", "Slash lead vs rhythm"),
    ("November Rain Guns N Roses", "Guns N' Roses", "November Rain", "Orchestral + dual guitars"),
    ("Wish You Were Here Pink Floyd", "Pink Floyd", "Wish You Were Here", "Acoustic vs lead"),
    ("Black Dog Led Zeppelin", "Led Zeppelin", "Black Dog", "Riff vs fills"),
    ("Whole Lotta Love Led Zeppelin", "Led Zeppelin", "Whole Lotta Love", "Heavy panning"),
    ("Sultans of Swing Dire Straits", "Dire Straits", "Sultans of Swing", "Clean rhythm vs lead"),

    # Grateful Dead - Jerry (lead) vs Bob (rhythm) distinct panning
    ("Scarlet Begonias Grateful Dead", "Grateful Dead", "Scarlet Begonias", "Jerry lead, Bob rhythm"),
    ("Fire on the Mountain Grateful Dead", "Grateful Dead", "Fire on the Mountain", "Clear L/R split"),
    ("Estimated Prophet Grateful Dead", "Grateful Dead", "Estimated Prophet", "Weir rhythm, Garcia lead"),
    ("Franklin's Tower Grateful Dead", "Grateful Dead", "Franklin's Tower", "Dual guitar interplay"),

    # Allman Brothers - Similar dual lead guitar setup
    ("Jessica Allman Brothers", "Allman Brothers Band", "Jessica", "Twin leads panned"),
    ("Whipping Post Allman Brothers", "Allman Brothers Band", "Whipping Post", "Duane vs Dickey"),
    ("Blue Sky Allman Brothers", "Allman Brothers Band", "Blue Sky", "Harmony leads"),

    # More Modern Examples
    ("Under the Bridge Red Hot Chili Peppers", "Red Hot Chili Peppers", "Under the Bridge", "Clean vs distorted"),
    ("Paranoid Android Radiohead", "Radiohead", "Paranoid Android", "Multiple guitar layers"),
    ("Everlong Foo Fighters", "Foo Fighters", "Everlong", "Wall of guitars"),
]


def check_dependencies():
    """Check that required tools are available."""
    deps = {
        'yt-dlp': 'yt-dlp --version',
        'ffmpeg': 'ffmpeg -version',
    }

    missing = []
    for name, cmd in deps.items():
        try:
            subprocess.run(cmd.split(), capture_output=True, check=True)
            logger.info(f"✓ {name} available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing.append(name)
            logger.error(f"✗ {name} not found")

    if missing:
        logger.error(f"Missing dependencies: {missing}")
        return False
    return True


def download_song(query: str, output_dir: Path, artist: str, title: str) -> Optional[Path]:
    """Download a song from YouTube using yt-dlp."""
    safe_name = f"{artist} - {title}".replace("/", "-").replace("\\", "-")
    output_path = output_dir / f"{safe_name}.%(ext)s"

    cmd = [
        'yt-dlp',
        '-x',  # Extract audio
        '--audio-format', 'wav',
        '--audio-quality', '0',
        '-o', str(output_path),
        f'ytsearch1:{query}'  # Search and get first result
    ]

    logger.info(f"Downloading: {artist} - {title}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error(f"Download failed: {result.stderr}")
            return None

        # Find the downloaded file
        for f in output_dir.glob(f"{safe_name}.*"):
            if f.suffix in ['.wav', '.mp3', '.m4a', '.opus']:
                logger.info(f"✓ Downloaded: {f.name}")
                return f

    except subprocess.TimeoutExpired:
        logger.error(f"Download timed out for {query}")
    except Exception as e:
        logger.error(f"Download error: {e}")

    return None


def separate_stems(audio_path: Path, output_dir: Path) -> Dict[str, Path]:
    """Separate audio into stems using Demucs."""
    logger.info(f"Separating stems: {audio_path.name}")

    cmd = [
        sys.executable, '-m', 'demucs',
        '--two-stems', 'other',  # Get "other" which includes guitars
        '-n', 'htdemucs',
        '-o', str(output_dir),
        str(audio_path)
    ]

    # Alternative: Use 6-stem model to get guitar directly
    cmd_6stem = [
        sys.executable, '-m', 'demucs',
        '-n', 'htdemucs_6s',
        '-o', str(output_dir),
        str(audio_path)
    ]

    try:
        # Try 6-stem first (has guitar output)
        result = subprocess.run(cmd_6stem, capture_output=True, text=True, timeout=600)

        if result.returncode == 0:
            stem_dir = output_dir / 'htdemucs_6s' / audio_path.stem
            if stem_dir.exists():
                stems = {}
                for stem_file in stem_dir.glob('*.wav'):
                    stem_name = stem_file.stem
                    stems[stem_name] = stem_file
                    logger.info(f"  ✓ {stem_name}: {stem_file.name}")
                return stems

    except subprocess.TimeoutExpired:
        logger.error("Stem separation timed out")
    except Exception as e:
        logger.error(f"Stem separation error: {e}")

    return {}


def analyze_stereo_field(audio_path: Path) -> Dict:
    """Analyze the stereo field of an audio file."""
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(str(audio_path), sr=None, mono=False, duration=60)

        if y.ndim == 1:
            return {'stereo': False, 'reason': 'mono'}

        left, right = y[0], y[1]

        # Calculate correlation
        correlation = np.corrcoef(left, right)[0, 1]

        # Calculate side ratio
        mid = (left + right) / 2
        side = (left - right) / 2
        mid_energy = np.sum(mid ** 2)
        side_energy = np.sum(side ** 2)
        side_ratio = side_energy / (mid_energy + side_energy + 1e-10)

        # Width metric
        width = 1 - abs(correlation)

        return {
            'stereo': True,
            'correlation': float(correlation),
            'side_ratio': float(side_ratio),
            'width': float(width),
            'splittable': bool(side_ratio > 0.1 and width > 0.15)
        }

    except Exception as e:
        logger.error(f"Stereo analysis failed: {e}")
        return {'stereo': False, 'reason': str(e)}


def split_stereo_guitar(guitar_path: Path, output_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Split guitar stem into lead and rhythm based on stereo panning.

    Many recordings have rhythm guitar panned to one side and lead to the other.
    We use frequency-aware stereo extraction to separate them.
    """
    try:
        import librosa
        import numpy as np
        import soundfile as sf

        y, sr = librosa.load(str(guitar_path), sr=None, mono=False)

        if y.ndim == 1:
            logger.warning("Guitar stem is mono, cannot split")
            return None, None

        left, right = y[0], y[1]

        # Analyze which side has more high-frequency content (usually lead)
        # Using STFT for frequency analysis
        n_fft = 2048
        hop_length = 512

        left_stft = librosa.stft(left, n_fft=n_fft, hop_length=hop_length)
        right_stft = librosa.stft(right, n_fft=n_fft, hop_length=hop_length)

        left_mag = np.abs(left_stft)
        right_mag = np.abs(right_stft)

        # High frequency bins (above ~2kHz where lead guitar presence is stronger)
        high_freq_start = int(2000 / (sr / n_fft))

        left_high = np.mean(left_mag[high_freq_start:, :])
        right_high = np.mean(right_mag[high_freq_start:, :])

        # Determine which side has lead (more high frequency content)
        if left_high > right_high:
            lead_channel = left
            rhythm_channel = right
            logger.info("  Lead guitar detected on LEFT channel")
        else:
            lead_channel = right
            rhythm_channel = left
            logger.info("  Lead guitar detected on RIGHT channel")

        # Apply soft stereo extraction using masks
        total_mag = left_mag + right_mag + 1e-10

        # Create masks emphasizing each channel's unique content
        threshold = 0.55

        if left_high > right_high:
            lead_dominance = left_mag / total_mag
            rhythm_dominance = right_mag / total_mag
        else:
            lead_dominance = right_mag / total_mag
            rhythm_dominance = left_mag / total_mag

        # Apply soft masks
        lead_mask = np.where(lead_dominance > threshold,
                            lead_dominance ** 0.5,
                            lead_dominance * 0.3)

        rhythm_mask = np.where(rhythm_dominance > threshold,
                              rhythm_dominance ** 0.5,
                              rhythm_dominance * 0.3)

        # Reconstruct
        if left_high > right_high:
            lead_stft = left_stft * lead_mask
            rhythm_stft = right_stft * rhythm_mask
        else:
            lead_stft = right_stft * lead_mask
            rhythm_stft = left_stft * rhythm_mask

        lead_audio = librosa.istft(lead_stft, hop_length=hop_length, length=len(left))
        rhythm_audio = librosa.istft(rhythm_stft, hop_length=hop_length, length=len(right))

        # Normalize
        lead_audio = lead_audio / (np.max(np.abs(lead_audio)) + 1e-10) * 0.9
        rhythm_audio = rhythm_audio / (np.max(np.abs(rhythm_audio)) + 1e-10) * 0.9

        # Save
        stem_name = guitar_path.stem
        lead_path = output_dir / f"{stem_name}_lead_guitar.wav"
        rhythm_path = output_dir / f"{stem_name}_rhythm_guitar.wav"

        sf.write(str(lead_path), lead_audio, sr)
        sf.write(str(rhythm_path), rhythm_audio, sr)

        logger.info(f"  ✓ Lead guitar: {lead_path.name}")
        logger.info(f"  ✓ Rhythm guitar: {rhythm_path.name}")

        return lead_path, rhythm_path

    except Exception as e:
        logger.error(f"Guitar splitting failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def prepare_training_structure(dataset_dir: Path, songs_data: List[Dict]):
    """
    Organize processed songs into training folder structure.

    Expected structure:
    dataset/
      song1/
        lead_guitar.wav
        rhythm_guitar.wav
      song2/
        lead_guitar.wav
        rhythm_guitar.wav
    """
    training_dir = dataset_dir / 'training'
    training_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    for song in songs_data:
        if song.get('lead_path') and song.get('rhythm_path'):
            song_dir = training_dir / song['safe_name']
            song_dir.mkdir(exist_ok=True)

            # Copy/move files
            lead_dest = song_dir / 'lead_guitar.wav'
            rhythm_dest = song_dir / 'rhythm_guitar.wav'

            shutil.copy2(song['lead_path'], lead_dest)
            shutil.copy2(song['rhythm_path'], rhythm_dest)

            manifest.append({
                'name': song['safe_name'],
                'artist': song['artist'],
                'title': song['title'],
                'lead_guitar': str(lead_dest),
                'rhythm_guitar': str(rhythm_dest),
                'stereo_analysis': song.get('stereo_analysis', {})
            })

            logger.info(f"✓ Added to training set: {song['safe_name']}")

    # Save manifest
    manifest_path = dataset_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\n📊 Dataset prepared: {len(manifest)} songs")
    logger.info(f"   Training dir: {training_dir}")
    logger.info(f"   Manifest: {manifest_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description='Prepare guitar lead/rhythm separation dataset')
    parser.add_argument('--output', type=str, default='./guitar_dataset',
                       help='Output directory for dataset')
    parser.add_argument('--songs', type=str, default=None,
                       help='Text file with custom song list (one per line: "search query|artist|title")')
    parser.add_argument('--limit', type=int, default=20,
                       help='Maximum number of songs to process')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip download, use existing files in output/downloads')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloads_dir = output_dir / 'downloads'
    downloads_dir.mkdir(exist_ok=True)

    stems_dir = output_dir / 'stems'
    stems_dir.mkdir(exist_ok=True)

    splits_dir = output_dir / 'splits'
    splits_dir.mkdir(exist_ok=True)

    # Check dependencies
    if not args.skip_download:
        if not check_dependencies():
            logger.error("Missing dependencies. Install with: pip install yt-dlp")
            return

    # Get song list
    if args.songs:
        songs = []
        with open(args.songs) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split('|')
                    if len(parts) >= 3:
                        songs.append((parts[0], parts[1], parts[2], ''))
    else:
        songs = RECOMMENDED_SONGS[:args.limit]

    logger.info(f"\n🎸 Guitar Lead/Rhythm Dataset Preparation")
    logger.info(f"   Songs to process: {len(songs)}")
    logger.info(f"   Output: {output_dir}\n")

    processed_songs = []

    for i, (query, artist, title, notes) in enumerate(songs, 1):
        logger.info(f"\n[{i}/{len(songs)}] Processing: {artist} - {title}")
        if notes:
            logger.info(f"   Note: {notes}")

        safe_name = f"{artist} - {title}".replace("/", "-").replace("\\", "-")

        song_data = {
            'query': query,
            'artist': artist,
            'title': title,
            'safe_name': safe_name
        }

        # Step 1: Download (or find existing)
        if args.skip_download:
            audio_files = list(downloads_dir.glob(f"{safe_name}.*"))
            audio_path = audio_files[0] if audio_files else None
        else:
            audio_path = download_song(query, downloads_dir, artist, title)

        if not audio_path:
            logger.warning(f"   ✗ Skipping (no audio)")
            continue

        song_data['audio_path'] = str(audio_path)

        # Step 2: Separate stems
        stems = separate_stems(audio_path, stems_dir)

        guitar_path = stems.get('guitar') or stems.get('other')
        if not guitar_path:
            logger.warning(f"   ✗ Skipping (no guitar stem)")
            continue

        song_data['guitar_stem'] = str(guitar_path)

        # Step 3: Analyze stereo field
        analysis = analyze_stereo_field(guitar_path)
        song_data['stereo_analysis'] = analysis

        if not analysis.get('splittable'):
            logger.warning(f"   ✗ Skipping (not enough stereo separation)")
            logger.warning(f"      Side ratio: {analysis.get('side_ratio', 0):.2f}, Width: {analysis.get('width', 0):.2f}")
            continue

        logger.info(f"   Stereo analysis: width={analysis['width']:.2f}, side_ratio={analysis['side_ratio']:.2f}")

        # Step 4: Split into lead/rhythm
        lead_path, rhythm_path = split_stereo_guitar(guitar_path, splits_dir)

        if lead_path and rhythm_path:
            song_data['lead_path'] = str(lead_path)
            song_data['rhythm_path'] = str(rhythm_path)
            processed_songs.append(song_data)
            logger.info(f"   ✓ Successfully processed")
        else:
            logger.warning(f"   ✗ Guitar split failed")

    # Step 5: Organize into training structure
    if processed_songs:
        manifest = prepare_training_structure(output_dir, processed_songs)

        logger.info(f"\n" + "="*50)
        logger.info(f"🎉 Dataset preparation complete!")
        logger.info(f"   Successfully processed: {len(processed_songs)}/{len(songs)} songs")
        logger.info(f"   Training data: {output_dir}/training")
        logger.info(f"\nNext steps:")
        logger.info(f"   1. Review the separated stems for quality")
        logger.info(f"   2. Run training with:")
        logger.info(f"      python train.py --config_path configs/config_guitar_lead_rhythm.yaml \\")
        logger.info(f"                      --dataset_type custom \\")
        logger.info(f"                      --train_data {output_dir}/training")
    else:
        logger.error("No songs were successfully processed!")


if __name__ == '__main__':
    main()
