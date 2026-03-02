"""
Tests for ProcessingJob serialization and URL validation.
"""

import sys
import os


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import just what we need, avoiding heavy ML imports
# We can't import app.py directly as it tries to import ML libraries,
# so we test the functions that are importable.


class TestURLPatterns:
    """Test URL validation functions from app.py logic."""

    def test_youtube_url(self):
        import re
        patterns = [
            r'(youtube\.com|youtu\.be)',
            r'soundcloud\.com',
            r'bandcamp\.com',
            r'archive\.org/(details|download)',
        ]
        youtube_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://youtube.com/watch?v=abc123",
        ]
        for url in youtube_urls:
            matched = any(re.search(p, url, re.IGNORECASE) for p in patterns)
            assert matched, f"Should match YouTube URL: {url}"

    def test_soundcloud_url(self):
        import re
        patterns = [r'soundcloud\.com']
        url = "https://soundcloud.com/artist/track-name"
        matched = any(re.search(p, url, re.IGNORECASE) for p in patterns)
        assert matched

    def test_archive_org_url(self):
        import re
        patterns = [r'archive\.org/(details|download)']
        urls = [
            "https://archive.org/details/gd1977-05-08.sbd.hicks.4982.sbeok.shnf",
            "https://archive.org/download/gd1977-05-08.sbd/track01.mp3",
        ]
        for url in urls:
            matched = any(re.search(p, url, re.IGNORECASE) for p in patterns)
            assert matched, f"Should match Archive.org URL: {url}"

    def test_invalid_url_no_match(self):
        import re
        patterns = [
            r'(youtube\.com|youtu\.be)',
            r'soundcloud\.com',
            r'bandcamp\.com',
            r'archive\.org/(details|download)',
        ]
        bad_urls = [
            "https://google.com",
            "https://example.com/song.mp3",
            "not-a-url",
        ]
        for url in bad_urls:
            matched = any(re.search(p, url, re.IGNORECASE) for p in patterns)
            assert not matched, f"Should NOT match: {url}"


class TestStreamingURLDetection:
    """Test Spotify/Apple Music URL detection."""

    def test_spotify_track(self):
        import re
        pattern = r'open\.spotify\.com/(track|album|playlist)/([a-zA-Z0-9]+)'
        url = "https://open.spotify.com/track/6rqhFgbbKwnb9MLmUQDhG6"
        match = re.search(pattern, url)
        assert match is not None
        assert match.group(1) == 'track'

    def test_spotify_album(self):
        import re
        pattern = r'open\.spotify\.com/(track|album|playlist)/([a-zA-Z0-9]+)'
        url = "https://open.spotify.com/album/4aawyAB9vmqN3uQ7FjRGTy"
        match = re.search(pattern, url)
        assert match is not None
        assert match.group(1) == 'album'

    def test_apple_music_track(self):
        import re
        pattern = r'music\.apple\.com/.+/album/.+/(\d+)'
        url = "https://music.apple.com/us/album/bohemian-rhapsody/1440806041?i=1440806768"
        match = re.search(pattern, url)
        assert match is not None

    def test_non_streaming_url(self):
        import re
        spotify_pattern = r'open\.spotify\.com/(track|album|playlist)/([a-zA-Z0-9]+)'
        apple_pattern = r'music\.apple\.com/.+/album/.+/(\d+)'
        url = "https://www.youtube.com/watch?v=abc123"
        assert re.search(spotify_pattern, url) is None
        assert re.search(apple_pattern, url) is None


class TestNumpyConversion:
    """Test numpy type conversion for JSON serialization."""

    def test_convert_numpy_int(self):
        import numpy as np
        from models.job import convert_numpy_types

        data = {'count': np.int64(42)}
        result = convert_numpy_types(data)
        assert result['count'] == 42
        assert isinstance(result['count'], int)

    def test_convert_numpy_float(self):
        import numpy as np
        from models.job import convert_numpy_types

        data = {'score': np.float32(0.95)}
        result = convert_numpy_types(data)
        assert abs(result['score'] - 0.95) < 0.01
        assert isinstance(result['score'], float)

    def test_convert_numpy_array(self):
        import numpy as np
        from models.job import convert_numpy_types

        data = {'values': np.array([1, 2, 3])}
        result = convert_numpy_types(data)
        assert result['values'] == [1, 2, 3]
        assert isinstance(result['values'], list)

    def test_convert_nested(self):
        import numpy as np
        from models.job import convert_numpy_types

        data = {
            'outer': {
                'count': np.int64(10),
                'items': [np.float64(1.1), np.float64(2.2)]
            }
        }
        result = convert_numpy_types(data)
        assert result['outer']['count'] == 10
        assert isinstance(result['outer']['items'][0], float)

    def test_passthrough_native_types(self):
        from models.job import convert_numpy_types

        data = {'name': 'test', 'count': 42, 'score': 0.95, 'items': [1, 2]}
        result = convert_numpy_types(data)
        assert result == data
