"""Per-bar router: ACE vs Jiang for "general" path songs.

Both detectors run on the same audio. We pick whichever is more appropriate
per the bar-agreement + event-density heuristic from the V3.1 router agent
(docs/v3-agent-3-ensembles-2026-05-13.md):

  if bar_agreement > 0.5 AND ace_events/jiang_events > 1.15:
      → use Jiang (over-eager ACE on a clean-vocab song)
  else:
      → use ACE (default — wins on 12/13 songs)

Empirical lift over ACE-alone: 0.852 → 0.869 root F1 average (zero per-song
regressions vs ACE-alone). Triggers Jiang for Hotel California (+0.157),
Paint It Black (+0.058), Sunshine of Your Love (+0.077).

The Claude-level detector_router (above this) decides general vs jazz first.
Only when general is chosen does THIS router fire. Jazz songs go straight to
stem_chord_detector.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from processing.chord_detector_ace import detect_chords_ace
from processing.chord_detector_jiang import detect_chords_jiang
from processing.chord_detector_lab import (
    ChordProgression,
    chord_events_to_job_format,
    detect_key_from_events,
)

logger = logging.getLogger(__name__)


# Router thresholds — tuned in-sample on the 13-song benchmark (agent 3 report).
# Held-out validation required in Week 2 before flipping prod (#21).
_BAR_AGREEMENT_THRESHOLD = 0.5
_EVENT_DENSITY_RATIO = 1.15


def _bar_agreement_score(ace_events: List, jiang_events: List, n_bars: int = 32) -> float:
    """Fraction of sampled bars where ACE and Jiang agree on the chord root.

    Sample at evenly-spaced timestamps from 0 to min(ace_end, jiang_end). For
    each sample, find the chord active in each detector and compare roots.
    Returns 0-1; higher = more agreement.
    """
    if not ace_events or not jiang_events:
        return 0.0

    ace_end = max((e.time + e.duration) for e in ace_events)
    jiang_end = max((e.time + e.duration) for e in jiang_events)
    end = min(ace_end, jiang_end)
    if end <= 0:
        return 0.0

    matches = 0
    samples = 0
    step = end / n_bars
    for i in range(n_bars):
        t = i * step
        a = next((e for e in ace_events if e.time <= t < e.time + e.duration), None)
        j = next((e for e in jiang_events if e.time <= t < e.time + e.duration), None)
        if a is None or j is None:
            continue
        samples += 1
        if a.root == j.root:
            matches += 1
    return matches / samples if samples else 0.0


def _route_decision(ace_events: List, jiang_events: List) -> Tuple[str, Dict]:
    """Apply the bar-agreement + density rule.

    Returns (winner, telemetry) where winner is "ace" or "jiang".
    """
    if not ace_events:
        return "jiang", {"reason": "ace_empty"}
    if not jiang_events:
        return "ace", {"reason": "jiang_empty"}

    bar_agreement = _bar_agreement_score(ace_events, jiang_events)
    ace_n = len(ace_events)
    jiang_n = len(jiang_events)
    density_ratio = ace_n / max(1, jiang_n)

    telemetry = {
        "bar_agreement": round(bar_agreement, 3),
        "ace_events": ace_n,
        "jiang_events": jiang_n,
        "density_ratio": round(density_ratio, 3),
    }

    if (bar_agreement > _BAR_AGREEMENT_THRESHOLD
            and density_ratio > _EVENT_DENSITY_RATIO):
        telemetry["reason"] = "over-eager-ace"
        return "jiang", telemetry

    telemetry["reason"] = "default-ace"
    return "ace", telemetry


def detect_chords_routed(audio_path: str) -> Tuple[ChordProgression, Dict]:
    """Run both detectors, route per-song, return the winner + telemetry."""
    logger.info(f"[chord_router] running ACE + Jiang on {audio_path}")
    ace = detect_chords_ace(audio_path)
    jiang = detect_chords_jiang(audio_path)

    winner, telemetry = _route_decision(ace.chords, jiang.chords)
    telemetry["winner"] = winner
    logger.info(f"[chord_router] routed to {winner}: {telemetry}")

    progression = ace if winner == "ace" else jiang
    return progression, telemetry


def detect_chords_for_job_routed(job, audio_path) -> None:
    """Pipeline-shaped wrapper. Attaches `job.detector_router_telemetry` for logging."""
    progression, telemetry = detect_chords_routed(str(audio_path))
    job.chord_progression = chord_events_to_job_format(progression.chords)
    job.detected_key = progression.key
    # Attach telemetry for the admin dashboard / nightly benchmark.
    if not hasattr(job, "metadata") or job.metadata is None:
        job.metadata = {}
    job.metadata["detector_router"] = telemetry
    logger.info(
        f"[chord_router] wrote {len(job.chord_progression)} events from "
        f"{telemetry['winner']}, key={job.detected_key}"
    )
