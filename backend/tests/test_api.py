"""
API endpoint smoke tests.

Uses Flask test client to test request/response behavior without
running actual audio processing (mocks the heavy pipeline functions).

Heavy ML dependencies (scipy, torch, demucs, etc.) are pre-mocked
in conftest.py so that `import app` succeeds without real libraries.
"""

from pathlib import Path
from unittest.mock import patch
from io import BytesIO

import pytest

# conftest.py already mocked heavy modules and added backend to sys.path,
# so we can import app directly here.
import app as app_mod

# Import from the new modular locations
from models.job import ProcessingJob, jobs
from routes.api import _validate_job_id, _safe_path
from services.url_resolver import validate_url_no_ssrf as _validate_url_no_ssrf


@pytest.fixture
def client():
    """Create Flask test client."""
    app_mod.app.config['TESTING'] = True
    with app_mod.app.test_client() as c:
        yield c


class TestHealthEndpoint:
    """Test /api/health endpoint."""

    def test_health_returns_ok(self, client):
        resp = client.get('/api/health')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'ok'
        assert 'service' in data

    def test_health_includes_yt_dlp_status(self, client):
        resp = client.get('/api/health')
        data = resp.get_json()
        assert 'yt_dlp_available' in data
        assert isinstance(data['yt_dlp_available'], bool)


class TestUploadEndpoint:
    """Test /api/upload endpoint validation."""

    def test_upload_no_file(self, client):
        resp = client.post('/api/upload')
        assert resp.status_code == 400
        assert 'error' in resp.get_json()

    def test_upload_empty_filename(self, client):
        data = {'file': (BytesIO(b''), '')}
        resp = client.post('/api/upload', data=data, content_type='multipart/form-data')
        assert resp.status_code == 400

    def test_upload_invalid_extension(self, client):
        data = {'file': (BytesIO(b'fake data'), 'song.exe')}
        resp = client.post('/api/upload', data=data, content_type='multipart/form-data')
        assert resp.status_code == 400
        assert 'Invalid file type' in resp.get_json()['error']

    def test_upload_valid_file_starts_processing(self, client):
        with patch('routes.api.process_audio'):
            data = {'file': (BytesIO(b'\x00' * 100), 'test_song.wav')}
            resp = client.post('/api/upload', data=data, content_type='multipart/form-data')
            assert resp.status_code == 200
            result = resp.get_json()
            assert 'job_id' in result
            assert result['message'] == 'Processing started'


class TestURLEndpoint:
    """Test /api/url endpoint validation."""

    def test_url_no_body(self, client):
        resp = client.post('/api/url', content_type='application/json')
        assert resp.status_code == 400

    def test_url_missing_url_field(self, client):
        resp = client.post('/api/url', json={'something': 'else'})
        assert resp.status_code == 400
        assert 'No URL provided' in resp.get_json()['error']

    def test_url_invalid_format(self, client):
        resp = client.post('/api/url', json={'url': 'not-a-url'})
        assert resp.status_code == 400
        assert 'Invalid URL' in resp.get_json()['error']

    def test_url_ssrf_localhost_blocked(self, client):
        resp = client.post('/api/url', json={'url': 'http://127.0.0.1:8080/admin'})
        assert resp.status_code == 400
        assert 'not allowed' in resp.get_json()['error']

    def test_url_ssrf_private_network_blocked(self, client):
        resp = client.post('/api/url', json={'url': 'http://192.168.1.1/internal'})
        assert resp.status_code == 400
        assert 'not allowed' in resp.get_json()['error']

    def test_url_ssrf_metadata_blocked(self, client):
        resp = client.post('/api/url', json={'url': 'http://169.254.169.254/latest/meta-data/'})
        assert resp.status_code == 400

    def test_url_too_long(self, client):
        long_url = 'https://youtube.com/' + 'a' * 2100
        resp = client.post('/api/url', json={'url': long_url})
        assert resp.status_code == 400
        assert 'too long' in resp.get_json()['error']

    def test_url_unsupported_site(self, client):
        resp = client.post('/api/url', json={'url': 'https://example.com/song.mp3'})
        assert resp.status_code == 400
        assert 'Unsupported URL' in resp.get_json()['error']


class TestStatusEndpoint:
    """Test /api/status/<job_id> endpoint."""

    def test_status_nonexistent_job(self, client):
        resp = client.get('/api/status/00000000-0000-0000-0000-000000000000')
        assert resp.status_code == 404
        assert 'error' in resp.get_json()

    def test_status_invalid_job_id(self, client):
        # Path traversal attempts: Flask normalizes ../.. in URLs, so use
        # a non-hex string that still reaches the endpoint
        resp = client.get('/api/status/INVALID_JOB_ID!')
        assert resp.status_code == 400

    def test_status_returns_job_data(self, client):
        test_id = 'aabbccdd-1122-3344-5566-778899aabbcc'
        job = ProcessingJob(test_id, 'test.wav')
        job.status = 'processing'
        job.progress = 50
        job.stage = 'Separating stems'
        jobs[test_id] = job

        resp = client.get(f'/api/status/{test_id}')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['job_id'] == test_id
        assert data['status'] == 'processing'
        assert data['progress'] == 50

        del jobs[test_id]


class TestDownloadEndpoint:
    """Test /api/download/<job_id>/<file_type>/<filename> validation."""

    def test_download_invalid_job_id(self, client):
        resp = client.get('/api/download/../../etc/passwd/stem/test')
        assert resp.status_code in (400, 404)

    def test_download_invalid_file_type(self, client):
        resp = client.get('/api/download/abc12345/script/test')
        assert resp.status_code == 400
        assert 'Invalid file type' in resp.get_json()['error']

    def test_download_path_traversal_in_filename(self, client):
        resp = client.get('/api/download/abc12345/stem/../../../etc/passwd')
        assert resp.status_code in (400, 404)

    def test_download_nonexistent_job(self, client):
        resp = client.get('/api/download/abc12345/stem/vocals')
        assert resp.status_code == 404


class TestLibraryEndpoint:
    """Test /api/library endpoint."""

    def test_library_returns_list(self, client):
        resp = client.get('/api/library')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'library' in data
        assert 'total' in data
        assert isinstance(data['library'], list)


class TestJobsEndpoint:
    """Test /api/jobs endpoint."""

    def test_jobs_returns_list(self, client):
        resp = client.get('/api/jobs')
        assert resp.status_code == 200
        data = resp.get_json()
        # /api/jobs returns either a bare list or {'jobs': [...]}
        if isinstance(data, dict):
            assert 'jobs' in data
            assert isinstance(data['jobs'], list)
        else:
            assert isinstance(data, list)


class TestCleanupEndpoint:
    """Test /api/cleanup input validation."""

    def test_cleanup_bounds_max_age(self, client):
        with patch('dependencies.cleanup_old_stems', return_value={'deleted': 0, 'freed_mb': 0}), \
             patch('dependencies.DRIVE_AVAILABLE', True):
            resp = client.post('/api/cleanup', json={'max_age_days': -5})
            assert resp.status_code == 200

            resp = client.post('/api/cleanup', json={'max_age_days': 99999})
            assert resp.status_code == 200

    def test_cleanup_invalid_type(self, client):
        with patch('dependencies.cleanup_old_stems', return_value={'deleted': 0, 'freed_mb': 0}), \
             patch('dependencies.DRIVE_AVAILABLE', True):
            resp = client.post('/api/cleanup', json={'max_age_days': 'not_a_number'})
            assert resp.status_code == 200  # Falls back to default 7


class TestValidationHelpers:
    """Test the validation helper functions."""

    def test_validate_job_id_valid(self):
        assert _validate_job_id('abc12345')
        assert _validate_job_id('a1b2c3d4-e5f6')

    def test_validate_job_id_invalid(self):
        assert not _validate_job_id('')
        assert not _validate_job_id('../etc')
        assert not _validate_job_id('a' * 50)  # Too long
        assert not _validate_job_id('has spaces')

    def test_validate_url_no_ssrf_allows_public(self):
        assert _validate_url_no_ssrf('https://youtube.com/watch?v=123')
        assert _validate_url_no_ssrf('https://soundcloud.com/track')

    def test_validate_url_no_ssrf_blocks_private(self):
        assert not _validate_url_no_ssrf('http://127.0.0.1/')
        assert not _validate_url_no_ssrf('http://10.0.0.1/')
        assert not _validate_url_no_ssrf('http://192.168.1.1/')
        assert not _validate_url_no_ssrf('http://localhost/')
        assert not _validate_url_no_ssrf('file:///etc/passwd')

    def test_safe_path_valid(self):
        base = Path('/tmp/test_stemscribe')
        base.mkdir(exist_ok=True)
        result = _safe_path(base, 'job123/stems/vocals.mp3')
        assert str(result).startswith(str(base.resolve()))

    def test_safe_path_traversal_blocked(self):
        base = Path('/tmp/test_stemscribe')
        base.mkdir(exist_ok=True)
        with pytest.raises(ValueError, match='Path traversal'):
            _safe_path(base, '../../etc/passwd')
