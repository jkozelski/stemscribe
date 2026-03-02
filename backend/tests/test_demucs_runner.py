"""
Tests for DemucsRunner progress parsing, error detection, and command building.
These tests do NOT require Demucs to be installed -- they test the parsing logic.
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from demucs_runner import DemucsRunner, DemucsProgress


@pytest.fixture
def runner():
    """Create a DemucsRunner without progress callback."""
    return DemucsRunner(model='htdemucs_6s')


@pytest.fixture
def runner_with_extras():
    """Create a DemucsRunner with extra args."""
    return DemucsRunner(
        model='htdemucs',
        extra_args=['--two-stems', 'vocals', '--shifts', '2']
    )


# =============================================================================
# Progress parsing tests
# =============================================================================

class TestProgressParsing:
    """Tests for _parse_progress which parses Demucs tqdm stderr output."""

    def test_full_progress_line(self, runner):
        line = "5%|██        | 22.35/444.59 [00:05<01:37, 4.33seconds/s]"
        progress = runner._parse_progress(line)
        assert progress is not None
        assert progress.percent == 5.0
        assert progress.current_seconds == 22.35
        assert progress.total_seconds == 444.59
        assert progress.rate == 4.33
        assert progress.eta_seconds > 0

    def test_zero_percent(self, runner):
        line = "0%| | 0.0/444.59 [00:00<?, ?seconds/s]"
        progress = runner._parse_progress(line)
        # May or may not parse depending on the ? in rate
        # At minimum should not crash
        if progress is not None:
            assert progress.percent == 0.0

    def test_high_percent(self, runner):
        line = "99%|█████████▉| 440.0/444.59 [02:30<00:01, 2.86seconds/s]"
        progress = runner._parse_progress(line)
        assert progress is not None
        assert progress.percent == 99.0

    def test_simple_percent_pattern(self, runner):
        line = "50% complete"
        progress = runner._parse_progress(line)
        assert progress is not None
        assert progress.percent == 50.0

    def test_non_progress_line(self, runner):
        line = "Loading model htdemucs_6s..."
        progress = runner._parse_progress(line)
        assert progress is None

    def test_empty_line(self, runner):
        progress = runner._parse_progress("")
        assert progress is None

    def test_progress_with_different_rates(self, runner):
        line = "23%|██▎       | 102.5/444.59 [00:35<01:57, 2.93seconds/s]"
        progress = runner._parse_progress(line)
        assert progress is not None
        assert progress.percent == 23.0
        assert abs(progress.rate - 2.93) < 0.01


# =============================================================================
# Error detection tests
# =============================================================================

class TestErrorDetection:
    """Tests for _is_actual_error which distinguishes real errors from progress."""

    def test_actual_error(self, runner):
        assert runner._is_actual_error("RuntimeError: CUDA out of memory") is True

    def test_traceback(self, runner):
        assert runner._is_actual_error("Traceback (most recent call last):") is True

    def test_file_not_found(self, runner):
        assert runner._is_actual_error("FileNotFoundError: No such file: test.wav") is True

    def test_permission_denied(self, runner):
        assert runner._is_actual_error("Permission denied: /tmp/output") is True

    def test_progress_bar_not_error(self, runner):
        # Progress bars contain '%|' which look like errors but aren't
        assert runner._is_actual_error("5%|██ | 22.35/444.59 [00:05<01:37, 4.33seconds/s]") is False

    def test_normal_output_not_error(self, runner):
        assert runner._is_actual_error("Loading model htdemucs_6s...") is False

    def test_empty_not_error(self, runner):
        assert runner._is_actual_error("") is False


# =============================================================================
# Error message extraction tests
# =============================================================================

class TestErrorMessageExtraction:
    """Tests for _extract_error_message which filters meaningful errors from stderr."""

    def test_extracts_runtime_error(self, runner):
        stderr = """5%|██ | 22.35/444.59 [00:05<01:37, 4.33seconds/s]
10%|█  | 44.5/444.59 [00:10<01:27, 2.86seconds/s]
RuntimeError: CUDA out of memory.
Tried to allocate 2.00 GiB"""
        error = runner._extract_error_message(stderr)
        assert error is not None
        assert "CUDA out of memory" in error

    def test_ignores_progress_only(self, runner):
        stderr = """0%| | 0.0/444.59 [00:00<?, ?seconds/s]
5%|██ | 22.35/444.59 [00:05<01:37, 4.33seconds/s]
100%|██████████| 444.59/444.59 [02:33<00:00, 2.93seconds/s]"""
        error = runner._extract_error_message(stderr)
        assert error is None

    def test_extracts_file_not_found(self, runner):
        stderr = "FileNotFoundError: No such file or directory: '/bad/path.wav'"
        error = runner._extract_error_message(stderr)
        assert error is not None
        assert "No such file" in error

    def test_empty_stderr(self, runner):
        error = runner._extract_error_message("")
        assert error is None


# =============================================================================
# Command building tests
# =============================================================================

class TestCommandBuilding:
    """Tests for DemucsRunner initialization and configuration."""

    def test_default_model(self):
        runner = DemucsRunner()
        assert runner.model == 'htdemucs_6s'

    def test_custom_model(self):
        runner = DemucsRunner(model='htdemucs_ft')
        assert runner.model == 'htdemucs_ft'

    def test_extra_args_stored(self, runner_with_extras):
        assert runner_with_extras.extra_args == ['--two-stems', 'vocals', '--shifts', '2']

    def test_no_extra_args_default(self, runner):
        assert runner.extra_args == []

    def test_progress_callback(self):
        received = []

        def callback(progress):
            received.append(progress)

        runner = DemucsRunner(progress_callback=callback)
        assert runner.progress_callback is not None

    def test_cancel_flag(self, runner):
        assert runner._cancelled is False


# =============================================================================
# DemucsProgress dataclass tests
# =============================================================================

class TestDemucsProgress:
    """Tests for the DemucsProgress dataclass."""

    def test_default_values(self):
        p = DemucsProgress()
        assert p.percent == 0.0
        assert p.current_seconds == 0.0
        assert p.total_seconds == 0.0
        assert p.rate == 0.0
        assert p.eta_seconds == 0.0
        assert p.stage == "Starting"

    def test_custom_values(self):
        p = DemucsProgress(
            percent=50.0,
            current_seconds=222.0,
            total_seconds=444.0,
            rate=3.0,
            eta_seconds=74.0,
            stage="Separating"
        )
        assert p.percent == 50.0
        assert p.stage == "Separating"
