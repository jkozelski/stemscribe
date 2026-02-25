"""
OMR (Optical Music Recognition) Processor for StemScribe
=========================================================
Handles image-based sheet music using multiple strategies:
1. Audiveris (local, if installed)
2. oemer (Python package for OMR)
3. Image preprocessing for better extraction

Since most sheet music PDFs are scanned images, we need
actual OMR rather than text extraction.
"""

import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

logger = logging.getLogger(__name__)

# Check for dependencies
FITZ_AVAILABLE = False
try:
    import fitz
    FITZ_AVAILABLE = True
except ImportError:
    pass

PIL_AVAILABLE = False
try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    pass

# Check for oemer (deep learning OMR)
OEMER_AVAILABLE = False
try:
    import oemer
    OEMER_AVAILABLE = True
    logger.info("✅ oemer OMR available")
except ImportError:
    logger.debug("oemer not installed")


class OMRProcessor:
    """
    Process image-based sheet music using Optical Music Recognition.
    """

    def __init__(self, use_oemer: bool = True, dpi: int = 300):
        """
        Initialize OMR processor.

        Args:
            use_oemer: Whether to try oemer for deep learning OMR
            dpi: DPI for rendering PDF pages as images
        """
        self.use_oemer = use_oemer
        self.dpi = dpi
        self._check_audiveris()

    def _check_audiveris(self) -> bool:
        """Check if Audiveris is installed."""
        try:
            result = subprocess.run(
                ['audiveris', '-help'],
                capture_output=True,
                timeout=5
            )
            self.audiveris_available = True
            logger.info("✅ Audiveris OMR available")
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.audiveris_available = False
            return False

    def pdf_to_images(self, pdf_path: str, output_dir: str) -> List[str]:
        """
        Convert PDF pages to images for OMR processing.

        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save images

        Returns:
            List of image paths
        """
        if not FITZ_AVAILABLE:
            raise ImportError("PyMuPDF required: pip install pymupdf")

        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(str(pdf_path))
        image_paths = []

        for i, page in enumerate(doc):
            # Render at high DPI for better OMR
            pix = page.get_pixmap(dpi=self.dpi)

            img_path = output_dir / f"page_{i+1:03d}.png"
            pix.save(str(img_path))
            image_paths.append(str(img_path))

        doc.close()
        logger.info(f"Converted {len(image_paths)} pages to images")
        return image_paths

    def process_image_oemer(self, image_path: str) -> Dict:
        """
        Process a music image using oemer.

        Args:
            image_path: Path to image file

        Returns:
            Dict with extracted notation
        """
        if not OEMER_AVAILABLE:
            return {'error': 'oemer not available'}

        try:
            from oemer import inference

            # Run oemer
            result = inference.inference(image_path)

            # Convert to our format
            notation = {
                'notes': [],
                'chords': [],
                'measures': 0
            }

            if hasattr(result, 'notes'):
                for note in result.notes:
                    notation['notes'].append({
                        'pitch': note.pitch,
                        'start': note.start,
                        'duration': note.duration,
                        'staff': note.staff
                    })

            notation['measures'] = len(result.measures) if hasattr(result, 'measures') else 0

            return notation

        except Exception as e:
            logger.error(f"oemer processing failed: {e}")
            return {'error': str(e)}

    def process_image_audiveris(self, image_path: str, output_path: str) -> Optional[str]:
        """
        Process a music image using Audiveris.

        Args:
            image_path: Path to image file
            output_path: Path for output MusicXML

        Returns:
            Path to MusicXML file or None
        """
        if not self.audiveris_available:
            return None

        try:
            result = subprocess.run(
                ['audiveris', '-batch', '-export',
                 '-output', output_path, image_path],
                capture_output=True,
                timeout=120
            )

            if result.returncode == 0:
                # Find the output file
                out_dir = Path(output_path)
                mxl_files = list(out_dir.glob('*.mxl')) + list(out_dir.glob('*.xml'))
                if mxl_files:
                    return str(mxl_files[0])

            return None

        except Exception as e:
            logger.error(f"Audiveris failed: {e}")
            return None

    def process_pdf(self, pdf_path: str, output_dir: str) -> Dict:
        """
        Full OMR pipeline for a PDF file.

        Args:
            pdf_path: Path to PDF
            output_dir: Output directory

        Returns:
            Dict with all extracted notation
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"📄 Processing PDF: {pdf_path.name}")

        # Step 1: Convert to images
        images_dir = output_dir / 'images'
        images = self.pdf_to_images(str(pdf_path), str(images_dir))

        results = {
            'source': str(pdf_path),
            'pages': [],
            'total_notes': 0,
            'total_measures': 0
        }

        # Step 2: Process each image
        for i, img_path in enumerate(images):
            logger.info(f"  Processing page {i+1}/{len(images)}...")

            page_result = {'page': i + 1, 'notes': [], 'measures': 0}

            # Try oemer first
            if self.use_oemer and OEMER_AVAILABLE:
                oemer_result = self.process_image_oemer(img_path)
                if 'error' not in oemer_result:
                    page_result = oemer_result
                    page_result['page'] = i + 1

            # Try Audiveris as fallback
            elif self.audiveris_available:
                mxl_dir = output_dir / 'musicxml'
                mxl_dir.mkdir(exist_ok=True)
                mxl_path = self.process_image_audiveris(img_path, str(mxl_dir))
                if mxl_path:
                    page_result = self._parse_musicxml(mxl_path)
                    page_result['page'] = i + 1

            results['pages'].append(page_result)
            results['total_notes'] += len(page_result.get('notes', []))
            results['total_measures'] += page_result.get('measures', 0)

        # Save results
        json_path = output_dir / f"{pdf_path.stem}_omr.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"✅ OMR complete: {results['total_notes']} notes, {results['total_measures']} measures")
        return results

    def _parse_musicxml(self, mxl_path: str) -> Dict:
        """Parse MusicXML file to our format."""
        try:
            import music21
            score = music21.converter.parse(mxl_path)

            result = {'notes': [], 'measures': 0}

            for note in score.flat.notes:
                if hasattr(note, 'pitch'):
                    result['notes'].append({
                        'pitch': note.pitch.midi,
                        'start': float(note.offset),
                        'duration': float(note.duration.quarterLength)
                    })

            result['measures'] = len(score.parts[0].getElementsByClass('Measure')) if score.parts else 0

            return result

        except Exception as e:
            logger.error(f"MusicXML parsing failed: {e}")
            return {'notes': [], 'measures': 0}


def install_oemer():
    """Instructions for installing oemer."""
    print("""
╔════════════════════════════════════════════════════════════╗
║           INSTALLING OEMER (Deep Learning OMR)             ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  oemer is a neural network-based OMR system that can       ║
║  extract notes, rhythms, and other notation from images.   ║
║                                                            ║
║  Installation (requires ~2GB for model weights):           ║
║                                                            ║
║    pip install oemer                                       ║
║                                                            ║
║  Or for GPU acceleration:                                  ║
║                                                            ║
║    pip install oemer[gpu]                                  ║
║                                                            ║
║  Note: First run will download model weights.              ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
""")


def install_audiveris():
    """Instructions for installing Audiveris."""
    print("""
╔════════════════════════════════════════════════════════════╗
║           INSTALLING AUDIVERIS (Traditional OMR)           ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Audiveris is an open-source OMR application.              ║
║                                                            ║
║  macOS Installation:                                       ║
║                                                            ║
║    brew install --cask audiveris                           ║
║                                                            ║
║  Or download from:                                         ║
║    https://github.com/Audiveris/audiveris/releases         ║
║                                                            ║
║  Audiveris is best for:                                    ║
║    • Clean, printed sheet music                            ║
║    • Standard notation (not handwritten)                   ║
║    • Full scores with multiple staves                      ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
""")


if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("🎼 OMR Processor for StemScribe")
    print(f"   oemer available: {OEMER_AVAILABLE}")
    print(f"   PyMuPDF available: {FITZ_AVAILABLE}")

    processor = OMRProcessor()
    print(f"   Audiveris available: {processor.audiveris_available}")

    if not OEMER_AVAILABLE and not processor.audiveris_available:
        print("\n⚠️ No OMR backend available!")
        print("\nRecommendation: Install oemer for best results:")
        install_oemer()

    if len(sys.argv) >= 2:
        pdf_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) >= 3 else './omr_output'

        results = processor.process_pdf(pdf_path, output_dir)
        print(f"\n✅ Results saved to: {output_dir}")
