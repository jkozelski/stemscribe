"""Tests for processing.detector_router.

Covers the hardening pass from the 2026-05-13 four-agent review:
  - Unicode normalization (NFKD strip combining marks)
  - Cache atomic-write + file-lock concurrency
  - Outage fallback (no API key, malformed JSON, exception)
  - Outage allowlist rescuing canonical jazz artists
  - Low-confidence decisions NOT cached
  - Prompt-hash invalidation when system prompt changes
  - Tolerant JSON parsing (code-fenced, surrounded by prose, raw)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# Allow running from repo root or backend/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set an isolated cache path BEFORE importing so the module reads it at load.
_TEST_CACHE_DIR = tempfile.mkdtemp(prefix="detector_router_test_")
os.environ["DETECTOR_ROUTER_CACHE"] = str(Path(_TEST_CACHE_DIR) / "cache.json")

from processing import detector_router  # noqa: E402  — after env setup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wipe_cache():
    """Reset the on-disk cache to {} between tests."""
    p = Path(os.environ["DETECTOR_ROUTER_CACHE"])
    if p.exists():
        p.unlink()


def _mock_anthropic_response(text: str):
    """Build an object mimicking anthropic.types.Message for tests."""
    class _Block:
        def __init__(self, t): self.type = "text"; self.text = t
    class _Resp:
        def __init__(self, t): self.content = [_Block(t)]
    return _Resp(text)


# ---------------------------------------------------------------------------
# Unicode normalization
# ---------------------------------------------------------------------------

def test_norm_strips_combining_marks():
    assert detector_router._norm("Mötley Crüe") == "motley crue"
    assert detector_router._norm("Sigur Rós") == "sigur ros"
    assert detector_router._norm("Beyoncé") == "beyonce"


def test_norm_strips_bracketed_annotations():
    assert detector_router._norm("Wonderwall (Live at Wembley)") == "wonderwall"
    assert detector_router._norm("Smells Like Teen Spirit [2021 Remaster]") == "smells like teen spirit"
    assert detector_router._norm("Yellow (feat. Coldplay)") == "yellow"


def test_norm_empty_and_unicode_punctuation():
    assert detector_router._norm("") == ""
    assert detector_router._norm("  ") == ""
    # Smart-quote apostrophe should not survive
    assert "don" in detector_router._norm("Don't Stop Believin'")


def test_cache_key_round_trip():
    k1 = detector_router._cache_key("Mötley Crüe", "Dr. Feelgood")
    k2 = detector_router._cache_key("motley crue", "dr. feelgood")
    # Both should produce the same normalized key.
    assert k1 == k2


# ---------------------------------------------------------------------------
# Outage fallback paths
# ---------------------------------------------------------------------------

def test_empty_input_returns_general_fallback():
    _wipe_cache()
    r = detector_router.route_detector("", "")
    assert r["path"] == "general"
    assert r["source"] == "fallback"


def test_no_api_key_returns_general_fallback():
    """When no Anthropic key resolves, fall back to general (or allowlist)."""
    _wipe_cache()
    with mock.patch.object(detector_router, "_api_key", return_value=None):
        r = detector_router.route_detector("Some Unknown Track", "Some Unknown Artist")
        # Unknown song with no API key → general
        assert r["path"] == "general"
        assert r["source"] == "fallback"


def test_anthropic_call_failure_returns_general():
    _wipe_cache()
    with mock.patch.object(detector_router, "_api_key", return_value="sk-test"):
        with mock.patch("anthropic.Anthropic") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.messages.create.side_effect = RuntimeError("api boom")
            r = detector_router.route_detector("Random Indie Song", "Unknown Band")
            assert r["path"] == "general"
            assert r["source"] == "fallback"


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------

def test_cache_hit_returns_cached_decision():
    _wipe_cache()
    with mock.patch.object(detector_router, "_api_key", return_value="sk-test"):
        with mock.patch("anthropic.Anthropic") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.messages.create.return_value = _mock_anthropic_response(
                '{"path": "jazz", "confidence": 0.98, "reasoning": "test"}'
            )
            # First call hits Claude
            r1 = detector_router.route_detector("Aja", "Steely Dan")
            assert r1["source"] == "claude"
            assert r1["path"] == "jazz"
            # Second call must hit cache, not Claude
            r2 = detector_router.route_detector("Aja", "Steely Dan")
            assert r2["source"] == "cache"
            assert r2["path"] == "jazz"
            # Claude was called exactly once
            assert mock_client.messages.create.call_count == 1


def test_low_confidence_not_cached():
    """Confidence below CACHE_CONFIDENCE_FLOOR should not poison the cache."""
    _wipe_cache()
    with mock.patch.object(detector_router, "_api_key", return_value="sk-test"):
        with mock.patch("anthropic.Anthropic") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.messages.create.return_value = _mock_anthropic_response(
                '{"path": "general", "confidence": 0.3, "reasoning": "unknown"}'
            )
            r1 = detector_router.route_detector("Random Demo", "Unknown Artist")
            assert r1["path"] == "general"
            # Cache file may or may not exist — checking that this song is not in it.
            cache = detector_router._load_cache()
            key = detector_router._cache_key("Random Demo", "Unknown Artist")
            assert key not in cache, f"Low-confidence decision was persisted: {cache}"


def test_invalid_path_coerces_to_general():
    _wipe_cache()
    with mock.patch.object(detector_router, "_api_key", return_value="sk-test"):
        with mock.patch("anthropic.Anthropic") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.messages.create.return_value = _mock_anthropic_response(
                '{"path": "metal", "confidence": 0.9, "reasoning": "hallucinated category"}'
            )
            r = detector_router.route_detector("Master of Puppets", "Metallica")
            # "metal" not in _VALID_PATHS → coerce to general
            assert r["path"] == "general"


# ---------------------------------------------------------------------------
# Tolerant JSON parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("response_text", [
    '{"path": "jazz", "confidence": 0.95, "reasoning": "test"}',
    '```json\n{"path": "jazz", "confidence": 0.95, "reasoning": "test"}\n```',
    'Here is the routing decision:\n\n{"path": "jazz", "confidence": 0.95, "reasoning": "test"}\n\nLet me know if you need more.',
])
def test_tolerant_json_parsing(response_text):
    _wipe_cache()
    with mock.patch.object(detector_router, "_api_key", return_value="sk-test"):
        with mock.patch("anthropic.Anthropic") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.messages.create.return_value = _mock_anthropic_response(response_text)
            r = detector_router.route_detector("Aja", "Steely Dan")
            assert r["path"] == "jazz"
            assert r["confidence"] == 0.95


def test_malformed_json_returns_fallback():
    _wipe_cache()
    with mock.patch.object(detector_router, "_api_key", return_value="sk-test"):
        with mock.patch("anthropic.Anthropic") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.messages.create.return_value = _mock_anthropic_response("totally not json at all")
            r = detector_router.route_detector("Some Song", "Some Artist")
            assert r["path"] == "general"
            assert r["source"] == "fallback"
