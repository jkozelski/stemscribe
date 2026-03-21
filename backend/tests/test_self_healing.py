"""
Tests for the self-healing system: feedback, watchdog, chord accuracy, error tracker.
"""

import json
import time
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# conftest.py already mocked heavy modules
import app as app_mod
from models.job import ProcessingJob, jobs


@pytest.fixture
def client():
    """Create Flask test client."""
    app_mod.app.config['TESTING'] = True
    with app_mod.app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def clean_jobs():
    """Clean up jobs dict before each test."""
    jobs.clear()
    yield
    jobs.clear()


# ============================================================
# Feature 1: User Correction Feedback Loop
# ============================================================

class TestFeedbackChordCorrection:
    """Test POST /api/feedback/chord-correction"""

    def test_chord_correction_saves(self, client, tmp_path):
        with patch('routes.feedback.FEEDBACK_FILE', tmp_path / 'feedback.json'):
            resp = client.post('/api/feedback/chord-correction', json={
                'job_id': 'aabbccdd-1111-2222-3333-444455556666',
                'original_chord': 'Am',
                'corrected_chord': 'Am7',
                'position': 12.5,
                'context': {'before': 'G', 'after': 'F'},
            })
            assert resp.status_code == 201
            data = resp.get_json()
            assert data['status'] == 'saved'
            assert data['correction']['original_chord'] == 'Am'
            assert data['correction']['corrected_chord'] == 'Am7'

    def test_chord_correction_missing_fields(self, client):
        resp = client.post('/api/feedback/chord-correction', json={
            'job_id': 'aabbccdd-1111-2222-3333-444455556666',
            'original_chord': 'Am',
            # missing corrected_chord and position
        })
        assert resp.status_code == 400
        assert 'Missing fields' in resp.get_json()['error']

    def test_chord_correction_no_body(self, client):
        resp = client.post('/api/feedback/chord-correction')
        assert resp.status_code == 400


class TestFeedbackLyricsCorrection:
    """Test POST /api/feedback/lyrics-correction"""

    def test_lyrics_correction_saves(self, client, tmp_path):
        with patch('routes.feedback.FEEDBACK_FILE', tmp_path / 'feedback.json'):
            resp = client.post('/api/feedback/lyrics-correction', json={
                'job_id': 'aabbccdd-1111-2222-3333-444455557777',
                'original_line': 'Is this the real life',
                'corrected_line': 'Is this the real life?',
                'line_index': 0,
            })
            assert resp.status_code == 201
            data = resp.get_json()
            assert data['status'] == 'saved'

    def test_lyrics_correction_missing_fields(self, client):
        resp = client.post('/api/feedback/lyrics-correction', json={
            'job_id': 'aabbccdd-1111-2222-3333-444455557777',
        })
        assert resp.status_code == 400


class TestFeedbackList:
    """Test GET /api/feedback/corrections"""

    def test_list_empty(self, client, tmp_path):
        with patch('routes.feedback.FEEDBACK_FILE', tmp_path / 'feedback.json'):
            resp = client.get('/api/feedback/corrections')
            assert resp.status_code == 200
            data = resp.get_json()
            assert data['total'] == 0

    def test_list_after_saving(self, client, tmp_path):
        fb_file = tmp_path / 'feedback.json'
        with patch('routes.feedback.FEEDBACK_FILE', fb_file):
            client.post('/api/feedback/chord-correction', json={
                'job_id': 'aa000000-0000-0000-0000-000000000001',
                'original_chord': 'C',
                'corrected_chord': 'Cm',
                'position': 5.0,
            })
            client.post('/api/feedback/lyrics-correction', json={
                'job_id': 'aa000000-0000-0000-0000-000000000001',
                'original_line': 'hello',
                'corrected_line': 'Hello',
                'line_index': 0,
            })

            resp = client.get('/api/feedback/corrections')
            data = resp.get_json()
            assert data['total'] == 2
            assert len(data['chord_corrections']) == 1
            assert len(data['lyrics_corrections']) == 1

    def test_filter_by_type(self, client, tmp_path):
        fb_file = tmp_path / 'feedback.json'
        with patch('routes.feedback.FEEDBACK_FILE', fb_file):
            client.post('/api/feedback/chord-correction', json={
                'job_id': 'aa000000-0000-0000-0000-000000000001',
                'original_chord': 'C',
                'corrected_chord': 'Cm',
                'position': 5.0,
            })

            resp = client.get('/api/feedback/corrections?type=lyrics')
            data = resp.get_json()
            assert 'chord_corrections' not in data
            assert len(data['lyrics_corrections']) == 0

    def test_filter_by_job_id(self, client, tmp_path):
        fb_file = tmp_path / 'feedback.json'
        with patch('routes.feedback.FEEDBACK_FILE', fb_file):
            client.post('/api/feedback/chord-correction', json={
                'job_id': 'aa000000-0000-0000-0000-000000000001',
                'original_chord': 'C',
                'corrected_chord': 'Cm',
                'position': 5.0,
            })
            client.post('/api/feedback/chord-correction', json={
                'job_id': 'aa000000-0000-0000-0000-000000000002',
                'original_chord': 'G',
                'corrected_chord': 'G7',
                'position': 10.0,
            })

            resp = client.get('/api/feedback/corrections?job_id=aa000000-0000-0000-0000-000000000001')
            data = resp.get_json()
            assert data['total'] == 1


# ============================================================
# Feature 2: Processing Failure Recovery (Watchdog)
# ============================================================

class TestWatchdog:
    """Test watchdog stall detection and retry logic."""

    def test_stall_detection(self, tmp_path):
        from processing.watchdog import _check_jobs, _job_snapshots, STALL_THRESHOLD_SECONDS

        # Create a stalled job
        job = ProcessingJob('stalled-1', 'test.mp3')
        job.status = 'processing'
        job.progress = 40
        job.stage = 'Separating stems'
        jobs['stalled-1'] = job

        # First check — records baseline
        _job_snapshots.clear()
        _check_jobs()
        assert 'stalled-1' in _job_snapshots

        # Simulate time passing without progress
        _job_snapshots['stalled-1']['last_check'] = time.time() - STALL_THRESHOLD_SECONDS - 10

        with patch('processing.watchdog._retry_job') as mock_retry:
            with patch('processing.watchdog._log_event'):
                _check_jobs()
                mock_retry.assert_called_once()

        _job_snapshots.clear()

    def test_progress_resets_stall_timer(self):
        from processing.watchdog import _check_jobs, _job_snapshots

        job = ProcessingJob('active-1', 'test.mp3')
        job.status = 'processing'
        job.progress = 40
        jobs['active-1'] = job

        _job_snapshots.clear()
        _check_jobs()  # Baseline

        # Job makes progress
        job.progress = 60
        _check_jobs()

        assert _job_snapshots['active-1']['last_progress'] == 60
        _job_snapshots.clear()

    def test_completed_jobs_ignored(self):
        from processing.watchdog import _check_jobs, _job_snapshots

        job = ProcessingJob('done-1', 'test.mp3')
        job.status = 'completed'
        job.progress = 100
        jobs['done-1'] = job

        _job_snapshots.clear()
        _check_jobs()
        assert 'done-1' not in _job_snapshots
        _job_snapshots.clear()

    def test_max_retries_marks_failed(self, tmp_path):
        from processing.watchdog import _retry_job, _job_snapshots, MAX_RETRIES

        job = ProcessingJob('retry-max', 'test.mp3')
        job.status = 'processing'
        job.progress = 30
        jobs['retry-max'] = job

        _job_snapshots['retry-max'] = {
            'last_progress': 30,
            'last_check': time.time(),
            'retries': MAX_RETRIES,  # Already at max
        }

        with patch('processing.watchdog._log_event'):
            with patch('models.job.save_job_checkpoint'):
                _retry_job(job)

        assert job.status == 'failed'
        assert 'retry' in job.error.lower()
        _job_snapshots.clear()


# ============================================================
# Feature 3: Chord Accuracy Scoring
# ============================================================

class TestChordAccuracy:
    """Test chord accuracy scoring."""

    def test_perfect_accuracy(self):
        from chord_accuracy import score_chord_accuracy

        ai = [
            {'chord': 'C', 'time': 0.0},
            {'chord': 'Am', 'time': 2.0},
            {'chord': 'F', 'time': 4.0},
            {'chord': 'G', 'time': 6.0},
        ]
        ref = [
            {'chord': 'C', 'time': 0.1},
            {'chord': 'Am', 'time': 2.1},
            {'chord': 'F', 'time': 4.0},
            {'chord': 'G', 'time': 6.0},
        ]

        result = score_chord_accuracy(ai, ref)
        assert result['accuracy_percent'] == 100.0
        assert result['correct'] == 4
        assert result['wrong'] == 0
        assert result['missed'] == 0

    def test_wrong_chords(self):
        from chord_accuracy import score_chord_accuracy

        ai = [
            {'chord': 'C', 'time': 0.0},
            {'chord': 'Em', 'time': 2.0},  # Wrong (should be Am)
        ]
        ref = [
            {'chord': 'C', 'time': 0.0},
            {'chord': 'Am', 'time': 2.0},
        ]

        result = score_chord_accuracy(ai, ref)
        assert result['correct'] == 1
        assert result['wrong'] == 1
        assert result['accuracy_percent'] == 50.0

    def test_missed_chords(self):
        from chord_accuracy import score_chord_accuracy

        ai = [{'chord': 'C', 'time': 0.0}]
        ref = [
            {'chord': 'C', 'time': 0.0},
            {'chord': 'Am', 'time': 5.0},
        ]

        result = score_chord_accuracy(ai, ref)
        assert result['correct'] == 1
        assert result['missed'] == 1

    def test_extra_chords(self):
        from chord_accuracy import score_chord_accuracy

        ai = [
            {'chord': 'C', 'time': 0.0},
            {'chord': 'Dm', 'time': 20.0},  # No matching reference
        ]
        ref = [{'chord': 'C', 'time': 0.0}]

        result = score_chord_accuracy(ai, ref)
        assert result['correct'] == 1
        assert result['extra'] == 1

    def test_chord_normalization(self):
        from chord_accuracy import score_chord_accuracy

        ai = [{'chord': 'Cmaj', 'time': 0.0}, {'chord': 'Amin', 'time': 2.0}]
        ref = [{'chord': 'C', 'time': 0.0}, {'chord': 'Am', 'time': 2.0}]

        result = score_chord_accuracy(ai, ref)
        assert result['accuracy_percent'] == 100.0

    def test_time_tolerance(self):
        from chord_accuracy import score_chord_accuracy

        ai = [{'chord': 'C', 'time': 0.0}]
        ref = [{'chord': 'C', 'time': 0.8}]  # Within 1s tolerance

        result = score_chord_accuracy(ai, ref, tolerance=1.0)
        assert result['correct'] == 1

        # Beyond tolerance
        ref2 = [{'chord': 'C', 'time': 5.0}]
        result2 = score_chord_accuracy(ai, ref2, tolerance=1.0)
        assert result2['correct'] == 0
        assert result2['missed'] == 1
        assert result2['extra'] == 1

    def test_empty_inputs(self):
        from chord_accuracy import score_chord_accuracy

        result = score_chord_accuracy([], [])
        assert result['accuracy_percent'] == 0.0
        assert result['total_reference_chords'] == 0

    def test_save_and_load(self, tmp_path):
        from chord_accuracy import save_accuracy_score, get_accuracy_score

        with patch('chord_accuracy.ACCURACY_FILE', tmp_path / 'scores.json'):
            report = {'accuracy_percent': 85.0, 'correct': 17, 'wrong': 3}
            save_accuracy_score('job-123', report)

            loaded = get_accuracy_score('job-123')
            assert loaded is not None
            assert loaded['accuracy_percent'] == 85.0

    def test_accuracy_endpoint_no_job(self, client):
        resp = client.get('/api/accuracy/00000000-0000-0000-0000-000000000099')
        assert resp.status_code == 404

    def test_accuracy_list_empty(self, client, tmp_path):
        with patch('chord_accuracy.ACCURACY_FILE', tmp_path / 'scores.json'):
            resp = client.get('/api/accuracy')
            assert resp.status_code == 200
            data = resp.get_json()
            assert data['total'] == 0


# ============================================================
# Feature 4: Error Pattern Detection
# ============================================================

class TestErrorTracker:
    """Test error tracking and pattern detection."""

    def test_log_error(self, tmp_path):
        from error_tracker import log_error, get_recent_errors

        with patch('error_tracker.ERROR_LOG_FILE', tmp_path / 'errors.json'):
            log_error(
                job_id='job-1',
                error_type='separation_failed',
                error_message='Demucs OOM',
                song_duration=600,
                source='youtube',
                processing_stage='separation',
            )

            errors = get_recent_errors(10)
            assert len(errors) == 1
            assert errors[0]['error_type'] == 'separation_failed'
            assert errors[0]['source'] == 'youtube'

    def test_pattern_detection_by_source(self, tmp_path):
        from error_tracker import log_error, get_error_patterns

        with patch('error_tracker.ERROR_LOG_FILE', tmp_path / 'errors.json'):
            for i in range(5):
                log_error(f'job-yt-{i}', 'failure', 'error', source='youtube')
            for i in range(2):
                log_error(f'job-ar-{i}', 'failure', 'error', source='archive')

            patterns = get_error_patterns()
            assert patterns['total_errors'] == 7
            assert patterns['by_source']['youtube'] == 5
            assert patterns['by_source']['archive'] == 2

    def test_pattern_detection_by_duration(self, tmp_path):
        from error_tracker import log_error, get_error_patterns

        with patch('error_tracker.ERROR_LOG_FILE', tmp_path / 'errors.json'):
            # 3 short songs
            for i in range(3):
                log_error(f'job-s-{i}', 'fail', 'err', song_duration=120)
            # 7 long songs
            for i in range(7):
                log_error(f'job-l-{i}', 'fail', 'err', song_duration=600)

            patterns = get_error_patterns()
            assert patterns['duration_buckets']['0-3min'] == 3
            assert patterns['duration_buckets']['8-12min'] == 7
            assert len(patterns['duration_insights']) > 0

    def test_pattern_detection_by_type(self, tmp_path):
        from error_tracker import log_error, get_error_patterns

        with patch('error_tracker.ERROR_LOG_FILE', tmp_path / 'errors.json'):
            log_error('aa000000-0000-0000-0000-000000000001', 'download_failed', 'timeout')
            log_error('j2', 'download_failed', 'timeout')
            log_error('j3', 'separation_failed', 'OOM')

            patterns = get_error_patterns()
            assert patterns['by_type']['download_failed'] == 2
            assert patterns['by_type']['separation_failed'] == 1

    def test_error_patterns_endpoint(self, client, tmp_path):
        with patch('error_tracker.ERROR_LOG_FILE', tmp_path / 'errors.json'):
            resp = client.get('/api/errors/patterns')
            assert resp.status_code == 200
            data = resp.get_json()
            assert 'total_errors' in data

    def test_recent_errors_endpoint(self, client, tmp_path):
        with patch('error_tracker.ERROR_LOG_FILE', tmp_path / 'errors.json'):
            resp = client.get('/api/errors/recent?limit=10')
            assert resp.status_code == 200
            data = resp.get_json()
            assert 'errors' in data

    def test_empty_patterns(self, tmp_path):
        from error_tracker import get_error_patterns

        with patch('error_tracker.ERROR_LOG_FILE', tmp_path / 'errors.json'):
            patterns = get_error_patterns()
            assert patterns['total_errors'] == 0

    def test_top_error_messages(self, tmp_path):
        from error_tracker import log_error, get_error_patterns

        with patch('error_tracker.ERROR_LOG_FILE', tmp_path / 'errors.json'):
            for i in range(5):
                log_error(f'j{i}', 'fail', 'OOM error during separation')
            for i in range(3):
                log_error(f'j{i+5}', 'fail', 'Download timeout')

            patterns = get_error_patterns()
            assert len(patterns['top_error_messages']) >= 2
            assert patterns['top_error_messages'][0]['count'] == 5
