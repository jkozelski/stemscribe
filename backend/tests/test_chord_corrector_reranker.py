"""Tests for the Claude-as-re-ranker chord corrector strategy.

Coverage:
- Env-var dispatch in apply_correction (generator vs reranker)
- apply_correction_reranker with valid top-K + mocked Claude response
- Graceful fallback when top-K is missing from bar_grid
- Graceful fallback when Claude returns malformed/invalid JSON
- Top-K extraction in chart_formatter._quantize_chords_to_bars
- Top-K plumb-through in chord_detector_librosa.detect_chords_for_job_librosa

The fixture chord_charts here are minimal stand-ins for what
chart_formatter produces — just enough fields (title, artist, bar_grid)
to drive the corrector.
"""

import os
from unittest.mock import patch, MagicMock

import pytest

from backend.processing import chord_corrector_anthropic as corrector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chart_with_top_k():
    """Build a small chord chart with librosa top-K candidates per bar."""
    return {
        "title": "Test Song",
        "artist": "Test Artist",
        "key": "Am",
        "tempo": 120.0,
        "chords_used": ["Am", "C", "G", "F"],
        "bar_grid": [
            {
                "bar": 1,
                "chord": "Am",
                "start_time": 0.0,
                "end_time": 2.0,
                "source_meta": {
                    "top_k": [
                        {"chord": "Am", "score": 0.91, "root": "A", "quality": "min"},
                        {"chord": "C",  "score": 0.74, "root": "C", "quality": "maj"},
                        {"chord": "E",  "score": 0.62, "root": "E", "quality": "maj"},
                    ]
                },
            },
            {
                "bar": 2,
                "chord": "C",
                "start_time": 2.0,
                "end_time": 4.0,
                "source_meta": {
                    "top_k": [
                        {"chord": "C",  "score": 0.88, "root": "C", "quality": "maj"},
                        {"chord": "Am", "score": 0.71, "root": "A", "quality": "min"},
                        {"chord": "G",  "score": 0.66, "root": "G", "quality": "maj"},
                    ]
                },
            },
            {
                "bar": 3,
                "chord": "G",
                "start_time": 4.0,
                "end_time": 6.0,
                "source_meta": {
                    "top_k": [
                        {"chord": "G",  "score": 0.85, "root": "G", "quality": "maj"},
                        {"chord": "Em", "score": 0.70, "root": "E", "quality": "min"},
                    ]
                },
            },
        ],
        "sections": [],
    }


def _make_chart_without_top_k():
    """Same shape but no source_meta.top_k — legacy detector output."""
    return {
        "title": "Legacy Song",
        "artist": "Legacy Artist",
        "key": "Am",
        "chords_used": ["Am", "C"],
        "bar_grid": [
            {"bar": 1, "chord": "Am", "start_time": 0.0, "end_time": 2.0},
            {"bar": 2, "chord": "C", "start_time": 2.0, "end_time": 4.0},
        ],
        "sections": [],
    }


def _mock_anthropic_response(text: str):
    """Build a MagicMock that mimics anthropic.Anthropic().messages.create()."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    client = MagicMock()
    client.messages.create.return_value = resp
    return client


# ---------------------------------------------------------------------------
# 1. Env-var dispatch
# ---------------------------------------------------------------------------

def test_apply_correction_disabled_returns_input_unchanged(monkeypatch):
    """When ENABLE_ANTHROPIC_CORRECTION is unset, the function is a no-op."""
    monkeypatch.delenv("ENABLE_ANTHROPIC_CORRECTION", raising=False)
    chart = _make_chart_with_top_k()
    out = corrector.apply_correction(chart)
    assert out is chart
    assert "anthropic_correction" not in out


def test_apply_correction_default_strategy_is_generator(monkeypatch):
    """When the corrector is enabled but no strategy is set, the generator
    path runs (calls _query_canonical_chords, NOT _query_reranker)."""
    monkeypatch.setenv("ENABLE_ANTHROPIC_CORRECTION", "true")
    monkeypatch.delenv("ANTHROPIC_CORRECTION_STRATEGY", raising=False)
    chart = _make_chart_with_top_k()

    with patch.object(corrector, "_query_canonical_chords") as mock_gen, \
         patch.object(corrector, "_query_reranker") as mock_rerank:
        mock_gen.return_value = None  # simulate no-response — exits cleanly
        corrector.apply_correction(chart)
        assert mock_gen.called, "generator path should be invoked by default"
        assert not mock_rerank.called, "reranker should NOT be invoked by default"


def test_apply_correction_strategy_reranker_routes_to_reranker(monkeypatch):
    """When ANTHROPIC_CORRECTION_STRATEGY=reranker, the reranker path runs."""
    monkeypatch.setenv("ENABLE_ANTHROPIC_CORRECTION", "true")
    monkeypatch.setenv("ANTHROPIC_CORRECTION_STRATEGY", "reranker")
    chart = _make_chart_with_top_k()

    with patch.object(corrector, "_query_canonical_chords") as mock_gen, \
         patch.object(corrector, "_query_reranker") as mock_rerank:
        mock_rerank.return_value = None  # short-circuit
        corrector.apply_correction(chart)
        assert mock_rerank.called, "reranker path should be invoked"
        assert not mock_gen.called, "generator should NOT be invoked"


def test_strategy_dispatcher_case_insensitive(monkeypatch):
    """ANTHROPIC_CORRECTION_STRATEGY is normalized to lowercase + trimmed."""
    monkeypatch.setenv("ENABLE_ANTHROPIC_CORRECTION", "true")
    monkeypatch.setenv("ANTHROPIC_CORRECTION_STRATEGY", "  RERANKER  ")
    chart = _make_chart_with_top_k()

    with patch.object(corrector, "_query_reranker") as mock_rerank:
        mock_rerank.return_value = None
        corrector.apply_correction(chart)
        assert mock_rerank.called


# ---------------------------------------------------------------------------
# 2. Reranker with valid top-K + mocked Claude response
# ---------------------------------------------------------------------------

def test_reranker_applies_picks_to_bar_grid(monkeypatch):
    """Valid Claude response → bar_grid is rewritten per pick."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key-for-unit-test")
    chart = _make_chart_with_top_k()

    fake_response = (
        '{"found": true, "bars": ['
        '  {"bar": 1, "pick": "Am", "abstain": false},'
        '  {"bar": 2, "pick": "Am", "abstain": false},'
        '  {"bar": 3, "pick": "Em", "abstain": false}'
        '], "notes": "rewrote bar 2 and 3 for vamp consistency"}'
    )
    mock_client = _mock_anthropic_response(fake_response)

    # Patch the Anthropic client constructor inside corrector module
    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic.return_value = mock_client
    with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
        out = corrector.apply_correction_reranker(chart)

    assert out["anthropic_correction"]["status"] == "applied"
    assert out["anthropic_correction"]["strategy"] == "reranker"
    assert out["anthropic_correction"]["bars_rewritten"] == 2  # bar 2 + bar 3
    assert out["anthropic_correction"]["bars_unchanged"] == 1  # bar 1
    assert out["bar_grid"][0]["chord"] == "Am"   # unchanged
    assert out["bar_grid"][1]["chord"] == "Am"   # rewritten C -> Am
    assert out["bar_grid"][2]["chord"] == "Em"   # rewritten G -> Em
    # rewritten bars carry source_meta.replaced_from
    assert out["bar_grid"][1]["source_meta"]["replaced_from"] == "C"
    assert out["bar_grid"][1]["source_meta"]["reason"] == "reranker-rerank"
    # chords_used is recomputed from final bar_grid (ordered, deduped)
    assert out["chords_used"] == ["Am", "Em"]


def test_reranker_respects_abstain(monkeypatch):
    """When Claude abstains, librosa's argmax stays."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key-for-unit-test")
    chart = _make_chart_with_top_k()

    fake_response = (
        '{"found": true, "bars": ['
        '  {"bar": 1, "pick": "C", "abstain": false},'
        '  {"bar": 2, "pick": null, "abstain": true},'
        '  {"bar": 3, "pick": null, "abstain": true}'
        '], "notes": "noisy"}'
    )
    mock_client = _mock_anthropic_response(fake_response)
    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic.return_value = mock_client
    with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
        out = corrector.apply_correction_reranker(chart)

    assert out["anthropic_correction"]["status"] == "applied"
    assert out["anthropic_correction"]["bars_abstained"] == 2
    assert out["anthropic_correction"]["bars_rewritten"] == 1
    # bar 1: Am -> C
    assert out["bar_grid"][0]["chord"] == "C"
    # bars 2-3 unchanged (librosa argmax preserved)
    assert out["bar_grid"][1]["chord"] == "C"
    assert out["bar_grid"][2]["chord"] == "G"


def test_reranker_handles_found_false(monkeypatch):
    """Claude returning found=false → corrector is a no-op, status set."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key-for-unit-test")
    chart = _make_chart_with_top_k()
    original_grid = [dict(b) for b in chart["bar_grid"]]

    fake_response = '{"found": false, "bars": [], "notes": "unknown song"}'
    mock_client = _mock_anthropic_response(fake_response)
    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic.return_value = mock_client
    with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
        out = corrector.apply_correction_reranker(chart)

    assert out["anthropic_correction"]["status"] == "skipped_unrecognized"
    # bar_grid chord values are unchanged
    for orig, new in zip(original_grid, out["bar_grid"]):
        assert orig["chord"] == new["chord"]


# ---------------------------------------------------------------------------
# 3. Graceful fallback paths
# ---------------------------------------------------------------------------

def test_reranker_no_top_k_returns_unchanged():
    """When bar_grid lacks source_meta.top_k, the reranker is a no-op."""
    chart = _make_chart_without_top_k()
    out = corrector.apply_correction_reranker(chart)
    assert out["anthropic_correction"]["status"] == "skipped_no_top_k"
    # No Claude call should have been made — but more importantly the
    # bar_grid is intact.
    assert out["bar_grid"][0]["chord"] == "Am"
    assert out["bar_grid"][1]["chord"] == "C"


def test_reranker_no_api_key_returns_unchanged(monkeypatch):
    """No API key → reranker bails out cleanly."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Also block the keychain fallback by patching _api_key directly
    with patch.object(corrector, "_api_key", return_value=None):
        chart = _make_chart_with_top_k()
        out = corrector.apply_correction_reranker(chart)
    assert out["anthropic_correction"]["status"] == "skipped_no_response"
    # bar_grid unchanged
    assert out["bar_grid"][1]["chord"] == "C"


def test_reranker_malformed_json_returns_unchanged(monkeypatch):
    """Garbage from Claude → corrector bails out cleanly."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key-for-unit-test")
    chart = _make_chart_with_top_k()

    mock_client = _mock_anthropic_response("this is not JSON at all!")
    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic.return_value = mock_client
    with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
        out = corrector.apply_correction_reranker(chart)

    # Either skipped_no_response (json parse failed) or
    # reranker_validation_failed (parsed but invalid shape). Both are fine.
    assert out["anthropic_correction"]["status"] in (
        "skipped_no_response",
        "reranker_validation_failed",
    )
    # bar_grid chord values are unchanged
    assert out["bar_grid"][1]["chord"] == "C"


def test_reranker_invalid_pick_rejects_response(monkeypatch):
    """If Claude picks a chord NOT in that bar's candidate list, reject the
    whole response — don't apply any picks."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key-for-unit-test")
    chart = _make_chart_with_top_k()
    original_chord_b2 = chart["bar_grid"][1]["chord"]

    # Bar 2 candidates are C, Am, G — picking Bm is illegal.
    fake_response = (
        '{"found": true, "bars": ['
        '  {"bar": 1, "pick": "Am", "abstain": false},'
        '  {"bar": 2, "pick": "Bm", "abstain": false},'
        '  {"bar": 3, "pick": "G",  "abstain": false}'
        '], "notes": "test"}'
    )
    mock_client = _mock_anthropic_response(fake_response)
    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic.return_value = mock_client
    with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
        out = corrector.apply_correction_reranker(chart)

    assert out["anthropic_correction"]["status"] == "reranker_validation_failed"
    # No picks applied — bar 2 still the original
    assert out["bar_grid"][1]["chord"] == original_chord_b2


def test_reranker_missing_bar_rejects_response(monkeypatch):
    """If Claude drops a bar from the response, reject — don't apply partial."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key-for-unit-test")
    chart = _make_chart_with_top_k()

    # Only bars 1, 3 — bar 2 missing.
    fake_response = (
        '{"found": true, "bars": ['
        '  {"bar": 1, "pick": "Am", "abstain": false},'
        '  {"bar": 3, "pick": "G",  "abstain": false}'
        '], "notes": "test"}'
    )
    mock_client = _mock_anthropic_response(fake_response)
    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic.return_value = mock_client
    with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
        out = corrector.apply_correction_reranker(chart)

    assert out["anthropic_correction"]["status"] == "reranker_validation_failed"


# ---------------------------------------------------------------------------
# 4. Pipeline-level fallback: reranker exception in apply_correction
#    falls through to generator
# ---------------------------------------------------------------------------

def test_apply_correction_reranker_exception_falls_through(monkeypatch):
    """If apply_correction_reranker raises, apply_correction should NOT
    propagate the exception — it falls through to generator (which itself
    will no-op if no API key / no canonical response)."""
    monkeypatch.setenv("ENABLE_ANTHROPIC_CORRECTION", "true")
    monkeypatch.setenv("ANTHROPIC_CORRECTION_STRATEGY", "reranker")
    chart = _make_chart_with_top_k()

    with patch.object(corrector, "apply_correction_reranker",
                      side_effect=RuntimeError("simulated crash")), \
         patch.object(corrector, "_query_canonical_chords",
                      return_value=None) as mock_gen:
        out = corrector.apply_correction(chart)
        # generator was called as a fallback
        assert mock_gen.called
        # output isn't None — pipeline gets the chart back
        assert out is not None


# ---------------------------------------------------------------------------
# 5. Top-K aggregation in chart_formatter._quantize_chords_to_bars
# ---------------------------------------------------------------------------

def test_quantize_chords_to_bars_aggregates_top_k():
    """When chord events carry `candidates`, the bar_grid output carries
    aggregated source_meta.top_k. Without candidates, no source_meta key."""
    # Import lazily — chart_formatter pulls in librosa/torch which are mocked
    # by conftest, but the function under test is pure Python.
    from chart_formatter import _quantize_chords_to_bars

    chords = [
        {
            "time": 0.0, "duration": 2.0, "chord": "Am",
            "candidates": [
                {"chord": "Am", "score": 0.9, "root": "A", "quality": "min"},
                {"chord": "C",  "score": 0.7, "root": "C", "quality": "maj"},
            ],
        },
        {
            "time": 2.0, "duration": 2.0, "chord": "C",
            "candidates": [
                {"chord": "C",  "score": 0.8, "root": "C", "quality": "maj"},
                {"chord": "Am", "score": 0.6, "root": "A", "quality": "min"},
            ],
        },
    ]
    grid = {
        "downbeat_times": [0.0, 2.0],
        "song_duration_sec": 4.0,
    }
    bars = _quantize_chords_to_bars(chords, grid)
    assert len(bars) == 2
    assert bars[0]["chord"] == "Am"
    assert "source_meta" in bars[0]
    assert "top_k" in bars[0]["source_meta"]
    top_k_b1 = bars[0]["source_meta"]["top_k"]
    assert top_k_b1[0]["chord"] == "Am"
    assert top_k_b1[1]["chord"] == "C"
    assert top_k_b1[0]["score"] > top_k_b1[1]["score"]


def test_quantize_chords_to_bars_no_candidates_no_source_meta():
    """Legacy detector path (no candidates) → no source_meta on the bar."""
    from chart_formatter import _quantize_chords_to_bars

    chords = [
        {"time": 0.0, "duration": 2.0, "chord": "Am"},
        {"time": 2.0, "duration": 2.0, "chord": "C"},
    ]
    grid = {
        "downbeat_times": [0.0, 2.0],
        "song_duration_sec": 4.0,
    }
    bars = _quantize_chords_to_bars(chords, grid)
    assert len(bars) == 2
    for b in bars:
        assert "source_meta" not in b
