"""
PDF Music Parser for StemScribe Training Data
==============================================
Extracts musical notation from PDF sheet music files.

Supports:
1. Real Book style lead sheets (melody + chords)
2. Individual instrument parts (guitar, bass, drums, etc.)
3. Full band scores

Uses multiple strategies:
- OMR (Optical Music Recognition) via Audiveris
- PDF text extraction for chord symbols
- Image-based analysis for notation
"""

import logging
import subprocess
import tempfile
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Check for dependencies
PDF_LIBS_AVAILABLE = False
try:
    import fitz  # PyMuPDF
    PDF_LIBS_AVAILABLE = True
except ImportError:
    logger.warning("PyMuPDF not installed: pip install pymupdf")

MUSIC21_AVAILABLE = False
try:
    import music21
    MUSIC21_AVAILABLE = True
except ImportError:
    logger.warning("music21 not installed: pip install music21")

PIL_AVAILABLE = False
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    logger.warning("Pillow not installed: pip install Pillow")


class NotationType(Enum):
    """Types of musical notation we can extract."""
    LEAD_SHEET = "lead_sheet"      # Melody + chords (Real Book style)
    SINGLE_PART = "single_part"    # One instrument
    FULL_SCORE = "full_score"      # Multiple instruments
    TAB = "tab"                    # Guitar/bass tablature
    DRUM_NOTATION = "drum"         # Drum notation


@dataclass
class ChordSymbol:
    """Represents a chord symbol from a lead sheet."""
    symbol: str           # e.g., "Cmaj7", "Dm7b5"
    beat_position: float  # Position in beats
    measure: int          # Measure number

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'beat': self.beat_position,
            'measure': self.measure
        }


@dataclass
class NoteEvent:
    """Represents a single note or rest."""
    pitch: Optional[int]    # MIDI pitch (None for rest)
    start_beat: float       # Start position in beats
    duration: float         # Duration in beats
    velocity: int = 80      # Default velocity
    measure: int = 0        # Measure number

    def to_dict(self) -> Dict:
        return {
            'pitch': self.pitch,
            'start': self.start_beat,
            'duration': self.duration,
            'velocity': self.velocity,
            'measure': self.measure
        }


@dataclass
class MusicPage:
    """Parsed content from one page of sheet music."""
    page_num: int
    title: Optional[str] = None
    composer: Optional[str] = None
    key_signature: Optional[str] = None
    time_signature: Optional[str] = None
    tempo: Optional[int] = None
    chords: List[ChordSymbol] = field(default_factory=list)
    notes: List[NoteEvent] = field(default_factory=list)
    measures: int = 0
    notation_type: NotationType = NotationType.LEAD_SHEET
    raw_text: str = ""

    def to_dict(self) -> Dict:
        return {
            'page': self.page_num,
            'title': self.title,
            'composer': self.composer,
            'key': self.key_signature,
            'time': self.time_signature,
            'tempo': self.tempo,
            'chords': [c.to_dict() for c in self.chords],
            'notes': [n.to_dict() for n in self.notes],
            'measures': self.measures,
            'type': self.notation_type.value
        }


@dataclass
class ParsedScore:
    """Complete parsed score from a PDF."""
    source_path: str
    pages: List[MusicPage]
    total_measures: int = 0
    song_titles: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'source': self.source_path,
            'pages': [p.to_dict() for p in self.pages],
            'total_measures': self.total_measures,
            'songs': self.song_titles
        }

    def to_json(self, output_path: str):
        """Save parsed score to JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class PDFMusicParser:
    """
    Main parser class for extracting music from PDFs.
    """

    # Common chord patterns for detection
    CHORD_PATTERN = re.compile(
        r'\b([A-G][#b]?)'                           # Root note
        r'(maj|min|m|dim|aug|\+|\-|°|ø)?'          # Quality
        r'(7|9|11|13|6|69|add9|sus[24]?)?'         # Extensions
        r'([#b][59]|[#b]11|[#b]13)?'               # Alterations
        r'(/[A-G][#b]?)?\b'                         # Bass note
    )

    # Real Book specific patterns
    TITLE_PATTERNS = [
        re.compile(r'^([A-Z][A-Za-z\s\'\-]+)$'),   # All caps or title case
        re.compile(r'^\d+\.\s*(.+)$'),              # Numbered: "1. Song Name"
    ]

    def __init__(self, use_omr: bool = True):
        """
        Initialize the parser.

        Args:
            use_omr: Whether to attempt OMR (requires Audiveris)
        """
        self.use_omr = use_omr
        self._check_omr_available()

    def _check_omr_available(self) -> bool:
        """Check if Audiveris OMR is available."""
        try:
            result = subprocess.run(
                ['audiveris', '-help'],
                capture_output=True,
                timeout=5
            )
            self.omr_available = True
            logger.info("✅ Audiveris OMR available")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.omr_available = False
            logger.info("Audiveris not found - using text/image extraction only")
        return self.omr_available

    def parse_pdf(self, pdf_path: str,
                  page_range: Optional[Tuple[int, int]] = None) -> ParsedScore:
        """
        Parse a PDF file and extract musical content.

        Args:
            pdf_path: Path to PDF file
            page_range: Optional (start, end) page numbers (1-indexed)

        Returns:
            ParsedScore with extracted content
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"📄 Parsing PDF: {pdf_path.name}")

        if not PDF_LIBS_AVAILABLE:
            raise ImportError("PyMuPDF required: pip install pymupdf")

        doc = fitz.open(str(pdf_path))
        pages = []
        song_titles = []

        # Determine page range
        start_page = (page_range[0] - 1) if page_range else 0
        end_page = page_range[1] if page_range else len(doc)

        for page_num in range(start_page, min(end_page, len(doc))):
            logger.info(f"  Processing page {page_num + 1}/{len(doc)}")

            page = doc[page_num]
            parsed = self._parse_page(page, page_num + 1)
            pages.append(parsed)

            if parsed.title and parsed.title not in song_titles:
                song_titles.append(parsed.title)

        doc.close()

        # Calculate total measures
        total_measures = sum(p.measures for p in pages)

        result = ParsedScore(
            source_path=str(pdf_path),
            pages=pages,
            total_measures=total_measures,
            song_titles=song_titles
        )

        logger.info(f"✅ Parsed {len(pages)} pages, {len(song_titles)} songs, {total_measures} measures")
        return result

    def _parse_page(self, page, page_num: int) -> MusicPage:
        """Parse a single page."""
        # Extract text
        text = page.get_text()

        # Extract images for OMR if needed
        images = page.get_images()

        # Create page object
        parsed = MusicPage(
            page_num=page_num,
            raw_text=text
        )

        # Try to detect title (usually at top of page)
        parsed.title = self._extract_title(text)

        # Extract chord symbols
        parsed.chords = self._extract_chords(text)

        # Detect notation type
        parsed.notation_type = self._detect_notation_type(text, images)

        # Extract key/time signatures from text
        parsed.key_signature = self._extract_key_signature(text)
        parsed.time_signature = self._extract_time_signature(text)

        # Estimate measures from chord count and time signature
        if parsed.chords:
            # Rough estimate: 4 chords per line, multiple lines
            parsed.measures = max(len(parsed.chords) // 2, 1)

        # If OMR available and this looks like standard notation, try it
        if self.use_omr and self.omr_available and parsed.notation_type != NotationType.TAB:
            omr_notes = self._run_omr(page)
            if omr_notes:
                parsed.notes = omr_notes

        return parsed

    def _extract_title(self, text: str) -> Optional[str]:
        """Extract song title from page text."""
        lines = text.strip().split('\n')

        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if not line or len(line) < 3:
                continue

            # Skip common non-title text
            if any(skip in line.lower() for skip in ['page', 'copyright', '©', 'real book']):
                continue

            # Check title patterns
            for pattern in self.TITLE_PATTERNS:
                match = pattern.match(line)
                if match:
                    title = match.group(1) if match.groups() else line
                    # Validate it looks like a title
                    if len(title) > 2 and not title.isdigit():
                        return title.strip()

        return None

    def _extract_chords(self, text: str) -> List[ChordSymbol]:
        """Extract chord symbols from text."""
        chords = []
        measure = 0
        beat = 0.0

        # Find all chord matches
        for match in self.CHORD_PATTERN.finditer(text):
            chord_str = match.group(0)

            # Skip if it's likely not a chord (too short, etc.)
            if len(chord_str) < 1:
                continue

            # Create chord symbol
            chord = ChordSymbol(
                symbol=chord_str,
                beat_position=beat,
                measure=measure
            )
            chords.append(chord)

            # Advance position (rough estimate)
            beat += 2.0
            if beat >= 4.0:
                beat = 0.0
                measure += 1

        return chords

    def _extract_key_signature(self, text: str) -> Optional[str]:
        """Extract key signature from text."""
        # Common patterns
        key_patterns = [
            r'Key:\s*([A-G][#b]?\s*(major|minor|maj|min|m)?)',
            r'in\s+([A-G][#b]?)\s+(major|minor)',
            r'([A-G][#b]?)\s*Major',
            r'([A-G][#b]?)\s*Minor',
        ]

        for pattern in key_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _extract_time_signature(self, text: str) -> Optional[str]:
        """Extract time signature from text."""
        # Look for patterns like 4/4, 3/4, 6/8
        match = re.search(r'\b(\d+)/(\d+)\b', text)
        if match:
            return f"{match.group(1)}/{match.group(2)}"
        return "4/4"  # Default assumption

    def _detect_notation_type(self, text: str, images: list) -> NotationType:
        """Detect what type of notation this page contains."""
        text_lower = text.lower()

        # Check for tablature indicators
        if 'tab' in text_lower or re.search(r'[eE]\|[-\d]+', text):
            return NotationType.TAB

        # Check for drum notation
        if any(drum in text_lower for drum in ['drums', 'percussion', 'snare', 'kick', 'hi-hat']):
            return NotationType.DRUM_NOTATION

        # Check for lead sheet (chords present, single staff expected)
        chord_count = len(self.CHORD_PATTERN.findall(text))
        if chord_count > 5:
            return NotationType.LEAD_SHEET

        # Default to single part
        return NotationType.SINGLE_PART

    def _run_omr(self, page) -> List[NoteEvent]:
        """Run OMR on a page image using Audiveris."""
        if not self.omr_available:
            return []

        notes = []

        try:
            # Export page as image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                pix = page.get_pixmap(dpi=300)
                pix.save(tmp.name)
                tmp_path = tmp.name

            # Run Audiveris
            with tempfile.NamedTemporaryFile(suffix='.mxl', delete=False) as out:
                out_path = out.name

            result = subprocess.run(
                ['audiveris', '-batch', '-export', '-output', out_path, tmp_path],
                capture_output=True,
                timeout=60
            )

            if result.returncode == 0 and MUSIC21_AVAILABLE:
                # Parse MusicXML output
                score = music21.converter.parse(out_path)

                for note in score.flat.notes:
                    if hasattr(note, 'pitch'):
                        notes.append(NoteEvent(
                            pitch=note.pitch.midi,
                            start_beat=float(note.offset),
                            duration=float(note.duration.quarterLength),
                            velocity=80
                        ))

            # Cleanup temp files
            Path(tmp_path).unlink(missing_ok=True)
            Path(out_path).unlink(missing_ok=True)

        except Exception as e:
            logger.warning(f"OMR failed: {e}")

        return notes

    def parse_real_book(self, pdf_path: str,
                        output_dir: Optional[str] = None) -> List[ParsedScore]:
        """
        Parse a Real Book PDF, extracting individual songs.

        Real Books have a specific format:
        - Table of contents at the beginning
        - Each song is typically 1-2 pages
        - Title at top, composer below
        - Lead sheet format (melody + chords)

        Args:
            pdf_path: Path to Real Book PDF
            output_dir: Optional directory to save individual song JSONs

        Returns:
            List of ParsedScore objects, one per song
        """
        logger.info(f"📚 Parsing Real Book: {Path(pdf_path).name}")

        if not PDF_LIBS_AVAILABLE:
            raise ImportError("PyMuPDF required: pip install pymupdf")

        doc = fitz.open(pdf_path)
        songs = []
        current_song_pages = []
        current_title = None

        # Skip table of contents (usually first few pages)
        start_page = self._find_first_song_page(doc)

        for page_num in range(start_page, len(doc)):
            page = doc[page_num]
            text = page.get_text()

            # Check if this is a new song
            title = self._extract_title(text)

            if title and title != current_title:
                # Save previous song if exists
                if current_song_pages and current_title:
                    song = ParsedScore(
                        source_path=pdf_path,
                        pages=current_song_pages,
                        total_measures=sum(p.measures for p in current_song_pages),
                        song_titles=[current_title]
                    )
                    songs.append(song)

                    if output_dir:
                        safe_name = re.sub(r'[^\w\s-]', '', current_title).strip()
                        safe_name = re.sub(r'\s+', '_', safe_name)
                        song.to_json(f"{output_dir}/{safe_name}.json")

                # Start new song
                current_title = title
                current_song_pages = []

            # Parse this page
            parsed = self._parse_page(page, page_num + 1)
            if not parsed.title:
                parsed.title = current_title
            current_song_pages.append(parsed)

            if page_num % 20 == 0:
                logger.info(f"  Page {page_num + 1}/{len(doc)}, {len(songs)} songs found")

        # Don't forget the last song
        if current_song_pages and current_title:
            song = ParsedScore(
                source_path=pdf_path,
                pages=current_song_pages,
                total_measures=sum(p.measures for p in current_song_pages),
                song_titles=[current_title]
            )
            songs.append(song)

            if output_dir:
                safe_name = re.sub(r'[^\w\s-]', '', current_title).strip()
                safe_name = re.sub(r'\s+', '_', safe_name)
                song.to_json(f"{output_dir}/{safe_name}.json")

        doc.close()

        logger.info(f"✅ Extracted {len(songs)} songs from Real Book")
        return songs

    def _find_first_song_page(self, doc) -> int:
        """Find the first actual song page (skip TOC)."""
        for i in range(min(10, len(doc))):
            text = doc[i].get_text().lower()
            # Look for musical content indicators
            if self.CHORD_PATTERN.search(doc[i].get_text()):
                return i
        return 0


# ============================================================================
# TRAINING DATA GENERATION
# ============================================================================

class TrainingDataGenerator:
    """
    Generates training data pairs from parsed sheet music.

    For training a transcription model, we need:
    1. Audio files (stems from StemScribe)
    2. Aligned notation (from PDF parser)
    """

    def __init__(self, parser: Optional[PDFMusicParser] = None):
        self.parser = parser or PDFMusicParser()

    def generate_training_pair(self,
                               audio_path: str,
                               pdf_path: str,
                               output_dir: str) -> Dict:
        """
        Generate a training data pair from audio and PDF.

        Args:
            audio_path: Path to audio file (or stem)
            pdf_path: Path to sheet music PDF
            output_dir: Directory to save training data

        Returns:
            Dict with paths to generated files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Parse PDF
        parsed = self.parser.parse_pdf(pdf_path)

        # Generate unique ID
        import hashlib
        audio_hash = hashlib.md5(Path(audio_path).read_bytes()[:1024]).hexdigest()[:8]

        # Save parsed notation
        notation_path = output_dir / f"{audio_hash}_notation.json"
        parsed.to_json(str(notation_path))

        # Create MIDI from notation
        midi_path = output_dir / f"{audio_hash}_score.mid"
        self._create_midi_from_parsed(parsed, str(midi_path))

        # Create metadata file
        metadata = {
            'audio_path': str(audio_path),
            'pdf_path': str(pdf_path),
            'notation_json': str(notation_path),
            'midi_path': str(midi_path),
            'songs': parsed.song_titles,
            'total_measures': parsed.total_measures
        }

        metadata_path = output_dir / f"{audio_hash}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def _create_midi_from_parsed(self, parsed: ParsedScore, output_path: str):
        """Create a MIDI file from parsed notation."""
        try:
            import pretty_midi

            midi = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(program=0, name='Melody')

            for page in parsed.pages:
                for note in page.notes:
                    if note.pitch is not None:
                        midi_note = pretty_midi.Note(
                            velocity=note.velocity,
                            pitch=note.pitch,
                            start=note.start_beat * 0.5,  # Assume 120 BPM
                            end=(note.start_beat + note.duration) * 0.5
                        )
                        inst.notes.append(midi_note)

            midi.instruments.append(inst)
            midi.write(output_path)

        except ImportError:
            logger.warning("pretty_midi not available for MIDI export")


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == '__main__':
    import sys
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    parser = argparse.ArgumentParser(description='Parse music PDFs for training data')
    parser.add_argument('pdf_path', help='Path to PDF file')
    parser.add_argument('--output', '-o', help='Output directory for JSON files')
    parser.add_argument('--real-book', '-r', action='store_true',
                        help='Parse as Real Book (extract individual songs)')
    parser.add_argument('--pages', '-p', help='Page range (e.g., 1-10)')

    args = parser.parse_args()

    pdf_parser = PDFMusicParser()

    if args.real_book:
        # Parse as Real Book
        output_dir = args.output or './parsed_songs'
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        songs = pdf_parser.parse_real_book(args.pdf_path, output_dir)
        print(f"\n✅ Extracted {len(songs)} songs to {output_dir}")

    else:
        # Parse single PDF
        page_range = None
        if args.pages:
            start, end = map(int, args.pages.split('-'))
            page_range = (start, end)

        result = pdf_parser.parse_pdf(args.pdf_path, page_range)

        if args.output:
            result.to_json(args.output)
            print(f"\n✅ Saved to {args.output}")
        else:
            # Print summary
            print(f"\n📄 Parsed: {args.pdf_path}")
            print(f"   Pages: {len(result.pages)}")
            print(f"   Songs: {result.song_titles}")
            print(f"   Total measures: {result.total_measures}")
            print(f"\n   Chords found: {sum(len(p.chords) for p in result.pages)}")
            print(f"   Notes found: {sum(len(p.notes) for p in result.pages)}")
