"""
Demucs Runner - Robust subprocess handling for Demucs stem separation
Fixes the progress-as-error bug and adds real-time progress streaming

Key improvements:
1. Uses subprocess.Popen for streaming output instead of subprocess.run
2. Parses progress bars from stderr correctly (they're not errors!)
3. Detects actual errors vs progress output
4. Streams progress updates to job status in real-time
5. Handles timeouts gracefully for long songs
"""

import os
import re
import subprocess
import threading
import logging
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Add homebrew to path for macOS
os.environ['PATH'] = os.environ.get('PATH', '') + ':/opt/homebrew/bin:/usr/local/bin'


@dataclass
class DemucsProgress:
    """Progress information from Demucs"""
    percent: float = 0.0
    current_seconds: float = 0.0
    total_seconds: float = 0.0
    rate: float = 0.0  # seconds/s processing rate
    eta_seconds: float = 0.0
    stage: str = "Starting"


@dataclass
class DemucsResult:
    """Result of Demucs separation"""
    success: bool
    output_dir: Optional[Path] = None
    stems: Dict[str, str] = None  # stem_name -> file_path
    error_message: Optional[str] = None
    processing_time_seconds: float = 0.0
    model_used: str = ""


class DemucsRunner:
    """
    Robust Demucs runner with progress streaming and proper error handling.
    
    The key insight: Demucs writes progress bars to stderr, NOT stdout.
    This is normal behavior, not an error. We need to parse stderr for
    progress updates while also detecting actual errors.
    """
    
    # Regex patterns for parsing Demucs output
    PROGRESS_PATTERN = re.compile(
        r'(\d+)%\|[^|]*\|\s*([\d.]+)/([\d.]+)\s*\[.*?,\s*([\d.]+)(\w+)/s'
    )
    # Alternative simpler pattern for percentage only
    PERCENT_PATTERN = re.compile(r'(\d+)%')
    
    # Actual error patterns (these indicate real failures)
    ERROR_PATTERNS = [
        re.compile(r'error:', re.IGNORECASE),
        re.compile(r'exception:', re.IGNORECASE),
        re.compile(r'traceback', re.IGNORECASE),
        re.compile(r'CUDA out of memory', re.IGNORECASE),
        re.compile(r'RuntimeError:', re.IGNORECASE),
        re.compile(r'FileNotFoundError:', re.IGNORECASE),
        re.compile(r'No such file or directory', re.IGNORECASE),
        re.compile(r'Permission denied', re.IGNORECASE),
        re.compile(r'killed', re.IGNORECASE),
    ]
    
    def __init__(self, 
                 model: str = 'htdemucs_6s',
                 device: str = 'auto',
                 progress_callback: Optional[Callable[[DemucsProgress], None]] = None):
        """
        Initialize Demucs runner.
        
        Args:
            model: Demucs model name (htdemucs, htdemucs_ft, htdemucs_6s)
            device: Device to use (auto, cuda, mps, cpu)
            progress_callback: Optional callback for progress updates
        """
        self.model = model
        self.device = device
        self.progress_callback = progress_callback
        self._process: Optional[subprocess.Popen] = None
        self._cancelled = False
        
    def _parse_progress(self, line: str) -> Optional[DemucsProgress]:
        """
        Parse a line of Demucs output for progress information.
        
        Demucs progress looks like:
        5%|██        | 22.35/444.59 [00:05<01:37, 4.33seconds/s]
        """
        # Try full pattern first
        match = self.PROGRESS_PATTERN.search(line)
        if match:
            percent = float(match.group(1))
            current = float(match.group(2))
            total = float(match.group(3))
            rate = float(match.group(4))
            
            # Calculate ETA
            remaining = total - current
            eta = remaining / rate if rate > 0 else 0
            
            return DemucsProgress(
                percent=percent,
                current_seconds=current,
                total_seconds=total,
                rate=rate,
                eta_seconds=eta,
                stage="Separating"
            )
        
        # Try simple percentage pattern
        match = self.PERCENT_PATTERN.search(line)
        if match:
            return DemucsProgress(
                percent=float(match.group(1)),
                stage="Separating"
            )
        
        return None
    
    def _is_actual_error(self, text: str) -> bool:
        """
        Check if text contains an actual error (not just progress output).
        
        Progress bars contain things like "0%|" which look scary but aren't errors.
        """
        for pattern in self.ERROR_PATTERNS:
            if pattern.search(text):
                # Double-check it's not in a progress bar context
                if '|' in text and '%' in text:
                    continue  # Likely a progress bar, not an error
                return True
        return False
    
    def _extract_error_message(self, stderr_text: str) -> Optional[str]:
        """Extract meaningful error message from stderr, ignoring progress output."""
        lines = stderr_text.split('\n')
        error_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip progress bar lines
            if '%|' in line or re.match(r'^\d+%', line):
                continue
            
            # Skip timing lines
            if 'seconds/s' in line or 'it/s' in line:
                continue
                
            # Collect actual error content
            if self._is_actual_error(line) or (error_lines and line):
                error_lines.append(line)
        
        if error_lines:
            return '\n'.join(error_lines[:10])  # Limit to first 10 lines
        return None
    
    def separate(self, 
                 audio_path: Path, 
                 output_dir: Path,
                 timeout_seconds: int = 1800) -> DemucsResult:
        """
        Run Demucs separation with progress streaming.
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save stems
            timeout_seconds: Maximum time to wait (default 30 minutes)
            
        Returns:
            DemucsResult with success status and stem paths
        """
        start_time = time.time()
        self._cancelled = False
        
        # Build command
        cmd = [
            'python3', '-m', 'demucs',
            '--out', str(output_dir),
            '-n', self.model,
        ]
        
        # Add device flag if specified
        if self.device and self.device != 'auto':
            cmd.extend(['-d', self.device])
        
        cmd.append(str(audio_path))
        
        logger.info(f"Starting Demucs: {' '.join(cmd)}")
        
        try:
            # Use Popen for streaming output
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Collect stderr for error detection
            stderr_lines = []
            last_progress = DemucsProgress(stage="Starting")
            
            # Read stderr in a thread to avoid blocking
            def read_stderr():
                for line in self._process.stderr:
                    if self._cancelled:
                        break
                    
                    stderr_lines.append(line)
                    
                    # Parse progress
                    progress = self._parse_progress(line)
                    if progress:
                        nonlocal last_progress
                        last_progress = progress
                        if self.progress_callback:
                            self.progress_callback(progress)
                        logger.debug(f"Demucs progress: {progress.percent:.0f}%")
            
            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stderr_thread.start()
            
            # Wait for process with timeout
            try:
                return_code = self._process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                logger.error(f"Demucs timed out after {timeout_seconds}s")
                self._process.kill()
                return DemucsResult(
                    success=False,
                    error_message=f"Processing timed out after {timeout_seconds // 60} minutes",
                    processing_time_seconds=time.time() - start_time,
                    model_used=self.model
                )
            
            # Wait for stderr thread to finish
            stderr_thread.join(timeout=5)
            
            stderr_text = ''.join(stderr_lines)
            processing_time = time.time() - start_time
            
            # Check for actual errors
            if return_code != 0:
                error_msg = self._extract_error_message(stderr_text)
                if not error_msg:
                    error_msg = f"Demucs exited with code {return_code}"
                
                logger.error(f"Demucs failed: {error_msg}")
                return DemucsResult(
                    success=False,
                    error_message=error_msg,
                    processing_time_seconds=processing_time,
                    model_used=self.model
                )
            
            # Success! Find the output stems
            stem_dir = output_dir / self.model / audio_path.stem
            
            if not stem_dir.exists():
                # Try alternative naming patterns
                for alt_dir in output_dir.glob(f'{self.model}/*'):
                    if alt_dir.is_dir():
                        stem_dir = alt_dir
                        break
            
            if not stem_dir.exists():
                return DemucsResult(
                    success=False,
                    error_message=f"Output directory not found: {stem_dir}",
                    processing_time_seconds=processing_time,
                    model_used=self.model
                )
            
            # Collect stem files
            stems = {}
            for stem_file in stem_dir.glob('*.wav'):
                stems[stem_file.stem] = str(stem_file)
            
            for stem_file in stem_dir.glob('*.mp3'):
                stems[stem_file.stem] = str(stem_file)
            
            if not stems:
                return DemucsResult(
                    success=False,
                    error_message="No stem files were generated",
                    processing_time_seconds=processing_time,
                    model_used=self.model
                )
            
            logger.info(f"Demucs complete: {len(stems)} stems in {processing_time:.1f}s")
            
            return DemucsResult(
                success=True,
                output_dir=stem_dir,
                stems=stems,
                processing_time_seconds=processing_time,
                model_used=self.model
            )
            
        except Exception as e:
            logger.exception(f"Demucs runner error: {e}")
            return DemucsResult(
                success=False,
                error_message=str(e),
                processing_time_seconds=time.time() - start_time,
                model_used=self.model
            )
        finally:
            self._process = None
    
    def cancel(self):
        """Cancel the running separation"""
        self._cancelled = True
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except:
                self._process.kill()


def separate_with_progress(audio_path: Path,
                          output_dir: Path,
                          model: str = 'htdemucs_6s',
                          job=None) -> Tuple[bool, Dict[str, str], Optional[str]]:
    """
    Convenience function to run Demucs with job progress updates.
    
    Args:
        audio_path: Input audio file
        output_dir: Output directory for stems
        model: Demucs model to use
        job: Optional ProcessingJob to update with progress
        
    Returns:
        Tuple of (success, stems_dict, error_message)
    """
    def progress_callback(progress: DemucsProgress):
        if job:
            # Map Demucs progress (0-100%) to job progress (15-40%)
            job_progress = 15 + (progress.percent * 0.25)
            job.progress = int(job_progress)
            
            if progress.eta_seconds > 0:
                eta_min = progress.eta_seconds / 60
                job.stage = f'Separating stems: {progress.percent:.0f}% (ETA: {eta_min:.1f}m)'
            else:
                job.stage = f'Separating stems: {progress.percent:.0f}%'
    
    runner = DemucsRunner(model=model, progress_callback=progress_callback)
    result = runner.separate(audio_path, output_dir)
    
    if result.success:
        return True, result.stems, None
    else:
        return False, {}, result.error_message


# Example usage and testing
if __name__ == '__main__':
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python demucs_runner.py <audio_file>")
        sys.exit(1)
    
    audio_file = Path(sys.argv[1])
    output_dir = Path('./test_output')
    output_dir.mkdir(exist_ok=True)
    
    def print_progress(p: DemucsProgress):
        print(f"\rProgress: {p.percent:.0f}% | {p.current_seconds:.1f}/{p.total_seconds:.1f}s | ETA: {p.eta_seconds:.0f}s", end='', flush=True)
    
    runner = DemucsRunner(progress_callback=print_progress)
    result = runner.separate(audio_file, output_dir)
    
    print()  # New line after progress
    
    if result.success:
        print(f"✅ Success! Stems saved to {result.output_dir}")
        for name, path in result.stems.items():
            print(f"  - {name}: {path}")
    else:
        print(f"❌ Failed: {result.error_message}")
