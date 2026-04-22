"""
Tests for URL-based result caching system.

Tests normalize_url, check_cache, add_to_cache, clone_job, and get_cache_stats
without processing any actual songs.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# conftest.py handles mocking heavy deps before any backend imports
from url_cache import normalize_url, check_cache, add_to_cache, clone_job, get_cache_stats, DB_PATH
import url_cache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Use a temporary database for each test."""
    db_path = tmp_path / 'test_url_cache.db'
    monkeypatch.setattr(url_cache, 'DB_PATH', db_path)
    yield db_path


@pytest.fixture
def fake_source_job(tmp_path, monkeypatch):
    """Create a fake completed job on disk that clone_job can load."""
    from models.job import OUTPUT_DIR

    job_id = 'source-job-1234'
    job_dir = OUTPUT_DIR / job_id
    stems_dir = job_dir / 'stems'
    stems_dir.mkdir(parents=True, exist_ok=True)

    # Create fake stem files
    vocals_path = stems_dir / 'vocals.wav'
    drums_path = stems_dir / 'drums.wav'
    vocals_path.write_bytes(b'\x00' * 100)
    drums_path.write_bytes(b'\x00' * 100)

    # Create fake MIDI file
    midi_dir = job_dir / 'midi'
    midi_dir.mkdir(exist_ok=True)
    midi_path = midi_dir / 'vocals.mid'
    midi_path.write_bytes(b'\x00' * 50)

    # Write job metadata
    metadata = {
        'job_id': job_id,
        'filename': 'Purple Haze - Jimi Hendrix.mp3',
        'source_url': 'https://www.youtube.com/watch?v=abc12345678',
        'status': 'completed',
        'progress': 100,
        'stage': 'Complete',
        'stems': {
            'vocals': str(vocals_path),
            'drums': str(drums_path),
        },
        'enhanced_stems': {},
        'midi_files': {
            'vocals': str(midi_path),
        },
        'musicxml_files': {},
        'gp_files': {},
        'sub_stems': {},
        'selected_skills': [],
        'error': None,
        'metadata': {
            'title': 'Purple Haze',
            'artist': 'Jimi Hendrix',
            'duration': 231.5,
        },
        'chord_progression': [
            {'time': 0.0, 'chord': 'E7#9', 'duration': 2.0},
            {'time': 2.0, 'chord': 'G', 'duration': 2.0},
        ],
        'detected_key': 'E',
        'transcription_quality': {'vocals': 0.85},
        'pro_tabs': {},
        'user_id': None,
        'session_id': None,
    }

    with open(job_dir / 'job_metadata.json', 'w') as f:
        json.dump(metadata, f)

    yield job_id

    # Cleanup
    import shutil
    if job_dir.exists():
        shutil.rmtree(job_dir)


# ---------------------------------------------------------------------------
# normalize_url tests
# ---------------------------------------------------------------------------

class TestNormalizeUrl:
    def test_standard_youtube_url(self):
        assert normalize_url('https://www.youtube.com/watch?v=abc12345678') == 'youtube:abc12345678'

    def test_short_youtube_url(self):
        assert normalize_url('https://youtu.be/abc12345678') == 'youtube:abc12345678'

    def test_music_youtube_url(self):
        assert normalize_url('https://music.youtube.com/watch?v=abc12345678') == 'youtube:abc12345678'

    def test_embed_url(self):
        assert normalize_url('https://www.youtube.com/embed/abc12345678') == 'youtube:abc12345678'

    def test_shorts_url(self):
        assert normalize_url('https://www.youtube.com/shorts/abc12345678') == 'youtube:abc12345678'

    def test_v_url(self):
        assert normalize_url('https://www.youtube.com/v/abc12345678') == 'youtube:abc12345678'

    def test_url_with_extra_params(self):
        url = 'https://www.youtube.com/watch?v=abc12345678&list=PLxyz&t=42'
        assert normalize_url(url) == 'youtube:abc12345678'

    def test_mobile_youtube_url(self):
        assert normalize_url('https://m.youtube.com/watch?v=abc12345678') == 'youtube:abc12345678'

    def test_no_www(self):
        assert normalize_url('https://youtube.com/watch?v=abc12345678') == 'youtube:abc12345678'

    def test_different_urls_same_video(self):
        urls = [
            'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
            'https://youtu.be/dQw4w9WgXcQ',
            'https://music.youtube.com/watch?v=dQw4w9WgXcQ',
            'https://www.youtube.com/embed/dQw4w9WgXcQ',
            'https://m.youtube.com/watch?v=dQw4w9WgXcQ&feature=share',
        ]
        results = [normalize_url(u) for u in urls]
        assert all(r == 'youtube:dQw4w9WgXcQ' for r in results)

    def test_non_youtube_url_returns_none(self):
        assert normalize_url('https://soundcloud.com/artist/song') is None

    def test_spotify_url_returns_none(self):
        assert normalize_url('https://open.spotify.com/track/abc123') is None

    def test_empty_string_returns_none(self):
        assert normalize_url('') is None

    def test_none_returns_none(self):
        assert normalize_url(None) is None

    def test_invalid_video_id_returns_none(self):
        assert normalize_url('https://www.youtube.com/watch?v=short') is None

    def test_whitespace_stripped(self):
        assert normalize_url('  https://youtu.be/abc12345678  ') == 'youtube:abc12345678'


# ---------------------------------------------------------------------------
# add_to_cache / check_cache tests
# ---------------------------------------------------------------------------

class TestCacheOperations:
    def test_add_and_check(self):
        url = 'https://www.youtube.com/watch?v=abc12345678'
        add_to_cache(url, 'job-001', 'Purple Haze', 'Jimi Hendrix', 6, True, True, True)
        result = check_cache(url)
        assert result == 'job-001'

    def test_check_miss(self):
        result = check_cache('https://www.youtube.com/watch?v=nonexistent1')
        assert result is None

    def test_different_url_formats_hit_same_cache(self):
        add_to_cache('https://www.youtube.com/watch?v=abc12345678', 'job-001')
        # Check with short URL
        assert check_cache('https://youtu.be/abc12345678') == 'job-001'
        # Check with music URL
        assert check_cache('https://music.youtube.com/watch?v=abc12345678') == 'job-001'
        # Check with extra params
        assert check_cache('https://www.youtube.com/watch?v=abc12345678&t=30') == 'job-001'

    def test_hit_count_increments(self):
        url = 'https://www.youtube.com/watch?v=abc12345678'
        add_to_cache(url, 'job-001', 'Test Song', 'Test Artist')
        check_cache(url)
        check_cache(url)
        check_cache(url)

        stats = get_cache_stats()
        assert stats['total_hits'] == 3

    def test_non_youtube_url_not_cached(self):
        result = add_to_cache('https://soundcloud.com/artist/song', 'job-001')
        assert result is False

    def test_duplicate_add_ignored(self):
        url = 'https://www.youtube.com/watch?v=abc12345678'
        assert add_to_cache(url, 'job-001', 'Song 1') is True
        # Second add with same URL is silently ignored (INSERT OR IGNORE)
        assert add_to_cache(url, 'job-002', 'Song 1 again') is True
        # Original job_id is preserved
        assert check_cache(url) == 'job-001'


# ---------------------------------------------------------------------------
# clone_job tests
# ---------------------------------------------------------------------------

class TestCloneJob:
    def test_clone_creates_completed_job(self, fake_source_job):
        cloned = clone_job(fake_source_job, 'new-job-5678')
        assert cloned is not None
        assert cloned.job_id == 'new-job-5678'
        assert cloned.status == 'completed'
        assert cloned.progress == 100
        assert 'cached' in cloned.stage.lower()

    def test_clone_has_source_stems(self, fake_source_job):
        cloned = clone_job(fake_source_job, 'new-job-5678')
        assert 'vocals' in cloned.stems
        assert 'drums' in cloned.stems

    def test_clone_has_metadata(self, fake_source_job):
        cloned = clone_job(fake_source_job, 'new-job-5678')
        assert cloned.metadata['title'] == 'Purple Haze'
        assert cloned.metadata['artist'] == 'Jimi Hendrix'
        assert cloned.metadata['cached_from'] == fake_source_job

    def test_clone_has_chords(self, fake_source_job):
        cloned = clone_job(fake_source_job, 'new-job-5678')
        assert len(cloned.chord_progression) == 2
        assert cloned.detected_key == 'E'

    def test_clone_has_midi(self, fake_source_job):
        cloned = clone_job(fake_source_job, 'new-job-5678')
        assert 'vocals' in cloned.midi_files

    def test_clone_missing_source_returns_none(self):
        cloned = clone_job('nonexistent-job', 'new-job-5678')
        assert cloned is None

    def test_clone_points_to_source_files(self, fake_source_job):
        """Verify the clone references the source job's files (no copies)."""
        from models.job import OUTPUT_DIR
        cloned = clone_job(fake_source_job, 'new-job-5678')
        # Stem paths should be under the source job's directory
        for stem_name, stem_path in cloned.stems.items():
            assert fake_source_job in stem_path


# ---------------------------------------------------------------------------
# get_cache_stats tests
# ---------------------------------------------------------------------------

class TestCacheStats:
    def test_empty_stats(self):
        stats = get_cache_stats()
        assert stats['total_cached'] == 0
        assert stats['total_hits'] == 0
        assert stats['top_songs'] == []
        assert stats['estimated_savings'] == '$0.00'

    def test_stats_after_entries(self):
        add_to_cache('https://youtu.be/abc12345678', 'j1', 'Purple Haze', 'Jimi Hendrix', 6, True, True, True)
        add_to_cache('https://youtu.be/def12345678', 'j2', 'Stairway', 'Led Zeppelin', 6, True, False, True)

        # Simulate hits
        check_cache('https://youtu.be/abc12345678')
        check_cache('https://youtu.be/abc12345678')
        check_cache('https://youtu.be/def12345678')

        stats = get_cache_stats()
        assert stats['total_cached'] == 2
        assert stats['total_hits'] == 3
        assert stats['estimated_savings'] == '$0.09'
        assert len(stats['top_songs']) == 2
        # Top song should be Purple Haze with 2 hits
        assert stats['top_songs'][0]['title'] == 'Purple Haze'
        assert stats['top_songs'][0]['hits'] == 2


# ---------------------------------------------------------------------------
# Integration test: full flow
# ---------------------------------------------------------------------------

class TestFullFlow:
    def test_add_check_clone_stats(self, fake_source_job):
        """End-to-end: add to cache, check it, clone it, verify stats."""
        url = 'https://www.youtube.com/watch?v=abc12345678'

        # 1. Add to cache
        add_to_cache(url, fake_source_job, 'Purple Haze', 'Jimi Hendrix', 6, True, True, True)

        # 2. Check cache (simulates second user pasting same URL)
        cached = check_cache(url)
        assert cached == fake_source_job

        # 3. Clone the job
        cloned = clone_job(cached, 'user2-job-9999')
        assert cloned is not None
        assert cloned.status == 'completed'
        assert cloned.metadata['title'] == 'Purple Haze'
        assert len(cloned.stems) >= 2

        # 4. Verify stats
        stats = get_cache_stats()
        assert stats['total_cached'] == 1
        assert stats['total_hits'] == 1
        assert stats['estimated_savings'] == '$0.03'
