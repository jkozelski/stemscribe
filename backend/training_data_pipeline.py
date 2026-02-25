"""
Training Data Pipeline for StemScribe
======================================
Orchestrates the complete process of preparing training data:
1. Parse PDFs to extract notation
2. Download/locate matching audio
3. Run audio through StemScribe for stems
4. Align audio with notation
5. Package for Colab training

Usage:
    python training_data_pipeline.py --pdf-dir /path/to/pdfs --output-dir /path/to/output
    python training_data_pipeline.py --real-book /path/to/REALBK1.PDF --output-dir /path/to/output
"""

import os
import sys
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

# Import our modules
try:
    from pdf_music_parser import PDFMusicParser, ParsedScore, TrainingDataGenerator
    PDF_PARSER_AVAILABLE = True
except ImportError:
    PDF_PARSER_AVAILABLE = False
    logger.warning("PDF parser not available")

# Check for audio processing
LIBROSA_AVAILABLE = False
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    logger.warning("librosa not available: pip install librosa soundfile")


@dataclass
class TrainingExample:
    """A single training example with audio and notation."""
    example_id: str
    song_title: str
    audio_path: str
    stem_paths: Dict[str, str]  # stem_type -> path
    notation_json: str
    midi_path: Optional[str]
    duration_seconds: float
    sample_rate: int = 44100

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TrainingDataset:
    """Complete training dataset."""
    name: str
    created: str
    examples: List[TrainingExample]
    total_duration_hours: float
    instrument_types: List[str]

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'created': self.created,
            'total_examples': len(self.examples),
            'total_duration_hours': self.total_duration_hours,
            'instrument_types': self.instrument_types,
            'examples': [e.to_dict() for e in self.examples]
        }

    def save(self, output_path: str):
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class TrainingDataPipeline:
    """
    Main pipeline for generating training data from PDFs and audio.
    """

    STEM_TYPES = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other']

    def __init__(self, output_dir: str, stemscribe_dir: Optional[str] = None):
        """
        Initialize the pipeline.

        Args:
            output_dir: Directory to save all training data
            stemscribe_dir: Path to StemScribe installation (for stem separation)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.audio_dir = self.output_dir / 'audio'
        self.stems_dir = self.output_dir / 'stems'
        self.notation_dir = self.output_dir / 'notation'
        self.midi_dir = self.output_dir / 'midi'

        for d in [self.audio_dir, self.stems_dir, self.notation_dir, self.midi_dir]:
            d.mkdir(exist_ok=True)

        # StemScribe location
        if stemscribe_dir:
            self.stemscribe_dir = Path(stemscribe_dir)
        else:
            # Try to find it
            self.stemscribe_dir = Path(__file__).parent.parent

        self.pdf_parser = PDFMusicParser() if PDF_PARSER_AVAILABLE else None
        self.examples: List[TrainingExample] = []

    def process_song(self,
                     audio_path: str,
                     pdf_path: str,
                     song_title: Optional[str] = None) -> Optional[TrainingExample]:
        """
        Process a single song: separate stems and parse notation.

        Args:
            audio_path: Path to audio file
            pdf_path: Path to sheet music PDF
            song_title: Optional song title (auto-detected if not provided)

        Returns:
            TrainingExample or None if processing failed
        """
        audio_path = Path(audio_path)
        pdf_path = Path(pdf_path)

        if not audio_path.exists():
            logger.error(f"Audio not found: {audio_path}")
            return None

        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return None

        # Generate unique ID
        example_id = hashlib.md5(
            f"{audio_path.name}{pdf_path.name}".encode()
        ).hexdigest()[:12]

        logger.info(f"\n{'='*60}")
        logger.info(f"🎵 Processing: {audio_path.name}")
        logger.info(f"📄 PDF: {pdf_path.name}")
        logger.info(f"   ID: {example_id}")

        # Step 1: Parse PDF
        logger.info("📖 Step 1: Parsing PDF notation...")
        notation_path = self.notation_dir / f"{example_id}_notation.json"

        if self.pdf_parser:
            try:
                parsed = self.pdf_parser.parse_pdf(str(pdf_path))
                parsed.to_json(str(notation_path))

                if not song_title and parsed.song_titles:
                    song_title = parsed.song_titles[0]

                logger.info(f"   ✅ Extracted: {len(parsed.pages)} pages, "
                           f"{sum(len(p.chords) for p in parsed.pages)} chords")

            except Exception as e:
                logger.error(f"   ❌ PDF parsing failed: {e}")
                notation_path = None
        else:
            logger.warning("   ⚠️ PDF parser not available, skipping notation extraction")
            notation_path = None

        song_title = song_title or audio_path.stem

        # Step 2: Copy/convert audio
        logger.info("🔊 Step 2: Processing audio...")
        processed_audio = self.audio_dir / f"{example_id}.wav"

        try:
            if LIBROSA_AVAILABLE:
                y, sr = librosa.load(str(audio_path), sr=44100, mono=False)
                if y.ndim == 1:
                    y = y.reshape(1, -1)
                sf.write(str(processed_audio), y.T, sr)
                duration = len(y[0]) / sr
            else:
                shutil.copy(audio_path, processed_audio)
                duration = self._get_duration(str(processed_audio))

            logger.info(f"   ✅ Audio: {duration:.1f}s")

        except Exception as e:
            logger.error(f"   ❌ Audio processing failed: {e}")
            return None

        # Step 3: Separate stems using StemScribe
        logger.info("🎛️ Step 3: Separating stems...")
        stem_paths = self._separate_stems(processed_audio, example_id)

        if not stem_paths:
            logger.warning("   ⚠️ Stem separation skipped/failed")
            stem_paths = {'mix': str(processed_audio)}

        # Step 4: Generate aligned MIDI (if we have notation)
        logger.info("🎹 Step 4: Generating MIDI from notation...")
        midi_path = None

        if notation_path and notation_path.exists():
            try:
                midi_path = self.midi_dir / f"{example_id}.mid"
                self._create_midi_from_notation(str(notation_path), str(midi_path))
                logger.info(f"   ✅ MIDI created: {midi_path.name}")
            except Exception as e:
                logger.warning(f"   ⚠️ MIDI creation failed: {e}")

        # Create example object
        example = TrainingExample(
            example_id=example_id,
            song_title=song_title,
            audio_path=str(processed_audio),
            stem_paths=stem_paths,
            notation_json=str(notation_path) if notation_path else "",
            midi_path=str(midi_path) if midi_path else None,
            duration_seconds=duration
        )

        self.examples.append(example)
        logger.info(f"✅ Example {example_id} complete!")

        return example

    def _separate_stems(self, audio_path: Path, example_id: str) -> Dict[str, str]:
        """Separate audio into stems using demucs or audio-separator."""
        stem_paths = {}
        example_stems_dir = self.stems_dir / example_id
        example_stems_dir.mkdir(exist_ok=True)

        try:
            # Try audio-separator first (usually faster)
            result = subprocess.run(
                [
                    sys.executable, '-m', 'audio_separator.separator',
                    str(audio_path),
                    '--output_dir', str(example_stems_dir),
                    '--model_filename', 'htdemucs_6s.yaml'
                ],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode == 0:
                # Find output stems
                for stem_file in example_stems_dir.glob('*.wav'):
                    stem_name = stem_file.stem.split('_')[-1].lower()
                    for stem_type in self.STEM_TYPES:
                        if stem_type in stem_name:
                            stem_paths[stem_type] = str(stem_file)
                            break

                logger.info(f"   ✅ Separated: {list(stem_paths.keys())}")
            else:
                logger.warning(f"   ⚠️ audio-separator failed: {result.stderr[:200]}")

        except FileNotFoundError:
            logger.warning("   ⚠️ audio-separator not found")
        except subprocess.TimeoutExpired:
            logger.warning("   ⚠️ Stem separation timed out")
        except Exception as e:
            logger.warning(f"   ⚠️ Stem separation error: {e}")

        return stem_paths

    def _get_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds."""
        try:
            import wave
            with wave.open(audio_path, 'r') as w:
                return w.getnframes() / w.getframerate()
        except:
            return 0.0

    def _create_midi_from_notation(self, notation_path: str, midi_path: str):
        """Create MIDI file from parsed notation JSON."""
        try:
            import pretty_midi

            with open(notation_path) as f:
                notation = json.load(f)

            midi = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(program=0, name='Melody')

            for page in notation.get('pages', []):
                for note in page.get('notes', []):
                    if note.get('pitch') is not None:
                        midi_note = pretty_midi.Note(
                            velocity=note.get('velocity', 80),
                            pitch=note['pitch'],
                            start=note['start'] * 0.5,
                            end=(note['start'] + note['duration']) * 0.5
                        )
                        inst.notes.append(midi_note)

            midi.instruments.append(inst)
            midi.write(midi_path)

        except ImportError:
            logger.warning("pretty_midi not available")
            raise

    def process_directory(self, pdf_dir: str, audio_dir: str,
                          match_by_name: bool = True) -> int:
        """
        Process all matching PDF/audio pairs in directories.

        Args:
            pdf_dir: Directory containing PDFs
            audio_dir: Directory containing audio files
            match_by_name: If True, match files by similar names

        Returns:
            Number of examples processed
        """
        pdf_dir = Path(pdf_dir)
        audio_dir = Path(audio_dir)

        pdfs = list(pdf_dir.glob('**/*.pdf'))
        audio_files = list(audio_dir.glob('**/*.wav')) + \
                      list(audio_dir.glob('**/*.mp3')) + \
                      list(audio_dir.glob('**/*.flac'))

        logger.info(f"📁 Found {len(pdfs)} PDFs and {len(audio_files)} audio files")

        processed = 0

        if match_by_name:
            # Try to match by filename similarity
            for pdf in pdfs:
                pdf_name = pdf.stem.lower()

                for audio in audio_files:
                    audio_name = audio.stem.lower()

                    # Simple similarity check
                    if pdf_name in audio_name or audio_name in pdf_name:
                        result = self.process_song(str(audio), str(pdf))
                        if result:
                            processed += 1
                        break

        return processed

    def process_real_book(self, real_book_path: str,
                          audio_dir: Optional[str] = None) -> int:
        """
        Process a Real Book PDF.

        Args:
            real_book_path: Path to Real Book PDF
            audio_dir: Optional directory with audio files to match

        Returns:
            Number of songs extracted
        """
        if not self.pdf_parser:
            logger.error("PDF parser not available")
            return 0

        logger.info(f"📚 Processing Real Book: {real_book_path}")

        # Parse Real Book into individual songs
        parsed_songs = self.pdf_parser.parse_real_book(
            real_book_path,
            str(self.notation_dir)
        )

        # If audio directory provided, try to match
        if audio_dir:
            audio_dir = Path(audio_dir)
            audio_files = list(audio_dir.glob('**/*.wav')) + \
                          list(audio_dir.glob('**/*.mp3'))

            for song in parsed_songs:
                if song.song_titles:
                    title = song.song_titles[0].lower()

                    for audio in audio_files:
                        if title in audio.stem.lower():
                            # Found a match!
                            notation_path = self.notation_dir / f"{title.replace(' ', '_')}.json"
                            song.to_json(str(notation_path))

                            self.process_song(
                                str(audio),
                                str(notation_path),
                                song_title=song.song_titles[0]
                            )
                            break

        return len(parsed_songs)

    def process_ultimate_guitar_folder(self, ug_dir: str) -> int:
        """
        Process an Ultimate Guitar style folder with individual instrument PDFs.

        Expected structure:
        Song_Name/
            Song_Name - Guitar.pdf
            Song_Name - Bass.pdf
            Song_Name - Drums.pdf
            Song_Name.mp3 (or .wav)

        Args:
            ug_dir: Path to Ultimate Guitar folder

        Returns:
            Number of songs processed
        """
        ug_dir = Path(ug_dir)
        processed = 0

        for song_dir in ug_dir.iterdir():
            if not song_dir.is_dir():
                continue

            logger.info(f"\n📂 Processing folder: {song_dir.name}")

            # Find audio file
            audio_files = list(song_dir.glob('*.mp3')) + \
                          list(song_dir.glob('*.wav')) + \
                          list(song_dir.glob('*.flac'))

            if not audio_files:
                logger.warning(f"   ⚠️ No audio found in {song_dir.name}")
                continue

            audio_path = audio_files[0]

            # Find PDFs (instrument parts)
            pdfs = list(song_dir.glob('*.pdf'))

            if not pdfs:
                logger.warning(f"   ⚠️ No PDFs found in {song_dir.name}")
                continue

            # Process each instrument PDF
            for pdf in pdfs:
                # Determine instrument from filename
                pdf_name = pdf.stem.lower()
                instrument = None

                for inst in ['guitar', 'bass', 'drums', 'piano', 'vocals', 'keys']:
                    if inst in pdf_name:
                        instrument = inst
                        break

                if instrument:
                    logger.info(f"   🎸 Found {instrument} part: {pdf.name}")

                    # Create example for this instrument
                    example = self.process_song(
                        str(audio_path),
                        str(pdf),
                        song_title=f"{song_dir.name} - {instrument.title()}"
                    )

                    if example:
                        processed += 1

        return processed

    def save_dataset(self, name: str = "stemscribe_training") -> str:
        """
        Save the complete dataset manifest.

        Args:
            name: Dataset name

        Returns:
            Path to saved manifest
        """
        total_duration = sum(e.duration_seconds for e in self.examples) / 3600

        # Collect all instrument types
        instruments = set()
        for e in self.examples:
            instruments.update(e.stem_paths.keys())

        dataset = TrainingDataset(
            name=name,
            created=datetime.now().isoformat(),
            examples=self.examples,
            total_duration_hours=total_duration,
            instrument_types=list(instruments)
        )

        manifest_path = self.output_dir / 'dataset_manifest.json'
        dataset.save(str(manifest_path))

        # Also create a Colab-ready zip
        self._create_colab_package()

        logger.info(f"\n{'='*60}")
        logger.info(f"📦 Dataset saved!")
        logger.info(f"   Examples: {len(self.examples)}")
        logger.info(f"   Duration: {total_duration:.2f} hours")
        logger.info(f"   Instruments: {list(instruments)}")
        logger.info(f"   Manifest: {manifest_path}")

        return str(manifest_path)

    def _create_colab_package(self):
        """Create a zip file ready for upload to Colab."""
        import zipfile

        zip_path = self.output_dir / 'colab_training_data.zip'

        logger.info(f"📦 Creating Colab package: {zip_path.name}")

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add manifest
            manifest = self.output_dir / 'dataset_manifest.json'
            if manifest.exists():
                zf.write(manifest, 'dataset_manifest.json')

            # Add notation files
            for f in self.notation_dir.glob('*.json'):
                zf.write(f, f'notation/{f.name}')

            # Add MIDI files
            for f in self.midi_dir.glob('*.mid'):
                zf.write(f, f'midi/{f.name}')

            # Note: Audio/stems are too large for zip - upload separately

        logger.info(f"   ✅ Created: {zip_path}")
        logger.info(f"   ⚠️ Note: Audio files not included (too large)")
        logger.info(f"   Upload audio separately to Google Drive")


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    parser = argparse.ArgumentParser(
        description='Generate training data from PDFs and audio'
    )
    parser.add_argument('--output', '-o', required=True,
                        help='Output directory for training data')
    parser.add_argument('--pdf-dir', help='Directory containing PDFs')
    parser.add_argument('--audio-dir', help='Directory containing audio files')
    parser.add_argument('--real-book', help='Path to Real Book PDF')
    parser.add_argument('--ug-folder', help='Path to Ultimate Guitar folder')
    parser.add_argument('--name', default='stemscribe_training',
                        help='Dataset name')

    args = parser.parse_args()

    pipeline = TrainingDataPipeline(args.output)

    if args.ug_folder:
        # Process Ultimate Guitar folder structure
        count = pipeline.process_ultimate_guitar_folder(args.ug_folder)
        print(f"\n✅ Processed {count} instrument parts from UG folder")

    if args.real_book:
        # Process Real Book
        count = pipeline.process_real_book(args.real_book, args.audio_dir)
        print(f"\n✅ Extracted {count} songs from Real Book")

    if args.pdf_dir and args.audio_dir:
        # Process matched directories
        count = pipeline.process_directory(args.pdf_dir, args.audio_dir)
        print(f"\n✅ Processed {count} matched pairs")

    # Save dataset
    if pipeline.examples:
        manifest = pipeline.save_dataset(args.name)
        print(f"\n📦 Dataset manifest: {manifest}")
    else:
        print("\n⚠️ No examples processed. Check your input paths.")
