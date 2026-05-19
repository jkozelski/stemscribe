"""Quality-only chord enricher (ACE-secondary, librosa-root-authoritative).

Production librosa is the best ROOT detector we have (measured A/B). It does
not switch detectors. But librosa's 24-template bank can only emit plain
major/minor triads, so it flattens every 7th / maj7 / m7 / sus / 6 color the
song actually contains (e.g. In My Life renders Dm/A7/B7 as D/A/B → plain
triads).

This module runs ACE as a SECONDARY pass on the same audio and, for each bar,
promotes librosa's triad to ACE's richer quality **only when ACE agrees with
librosa on that bar's ROOT**. ACE never changes the root. If ACE disagrees on
the root, or has no richer quality to offer, the librosa chord is kept
verbatim. Conservative by construction: when uncertain, keep librosa.

This is the same pattern `bass_root_extraction.combine_with_detector_quality`
already uses for bass roots — root from the authoritative source, quality from
the secondary detector — generalized to a bar_grid that is already built.

Gated behind `ENABLE_QUALITY_ENRICHER=true`. Default OFF for prod safety; the
test/validation harness sets it ON explicitly.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Enharmonic-normalized pitch class for root comparison.
_NOTE_PC = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "E#": 5,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11, "B#": 0,
}

# chord string → (root, quality, bass)
_CHORD_RE = re.compile(r"^([A-G][#b]?)(.*?)(?:/([A-G][#b]?))?$")


def _parse(chord: str) -> Tuple[Optional[str], str, Optional[str]]:
    if not chord or not isinstance(chord, str):
        return (None, "", None)
    s = chord.strip()
    m = _CHORD_RE.match(s)
    if not m:
        return (None, "", None)
    return (m.group(1), m.group(2) or "", m.group(3))


def _pc(note: Optional[str]) -> Optional[int]:
    if not note:
        return None
    return _NOTE_PC.get(note)


def _same_root(a: Optional[str], b: Optional[str]) -> bool:
    pa, pb = _pc(a), _pc(b)
    return pa is not None and pa == pb


# Qualities that carry strictly MORE harmonic information than a bare triad.
# Promoting librosa's '' (major) or 'm' (minor) to one of these recovers a
# color librosa structurally cannot emit. Anything not in this set (or a plain
# triad) is NOT a promotion target — we never invent an extension librosa and
# ACE don't both support, and we never downgrade.
_INFORMATIVE_EXTENSIONS = {
    "7", "maj7", "m7", "mmaj7",
    "6", "m6",
    "9", "maj9", "m9",
    "11", "m11", "13", "maj13", "m13",
    "add9", "madd9",
    "sus2", "sus4", "7sus4", "9sus4",
    "dim7", "m7b5",
}

# Which librosa base triad each ACE quality is a legal enrichment OF. Key =
# librosa quality (only '' major and 'm' minor exist in the 24-template bank).
# Value = set of ACE qualities we will promote that triad to. This guards
# against e.g. promoting a major-triad bar to m7 (that's a root-family flip,
# not a color recovery — ACE got the chord wrong, keep librosa).
_LEGAL_PROMOTIONS = {
    "": {"7", "maj7", "6", "9", "maj9", "11", "13", "maj13",
         "add9", "sus2", "sus4", "7sus4", "9sus4"},
    "m": {"m7", "mmaj7", "m6", "m9", "m11", "m13", "madd9",
          "dim7", "m7b5"},
}


# A promotion only fires when ACE's winning (root,quality) covers at least
# this fraction of the bar. A short passing chord (e.g. ACE flicks E7 for half
# a beat under a sustained E) must not flip the whole bar to a 7th. Tuned to
# kill the spurious passing-dominant promotions seen in the In My Life A/B.
_MIN_BAR_COVERAGE = 0.55


def _ace_overlap_quality(
    ace_events: List,
    start: float,
    end: float,
) -> Tuple[Optional[str], str]:
    """Return (root, quality) of the ACE chord that DOMINATES [start,end).

    Returns the (root, quality) only if it covers >= _MIN_BAR_COVERAGE of the
    bar duration; otherwise ("", "") so the caller keeps librosa. This guards
    against ACE's frequent short passing-7th flicks on a held dominant chord
    (the dominant cause of the wrong E->E7 / D->Dsus4 promotions in testing).

    `ace_events` are chord_detector_lab.ChordEvent dataclasses with
    .time/.duration/.root/.quality fields.
    """
    bar_dur = max(1e-6, end - start)
    acc: Dict[Tuple[str, str], float] = {}
    for e in ace_events:
        e_start = e.time
        e_end = e.time + e.duration
        ov = max(0.0, min(end, e_end) - max(start, e_start))
        if ov <= 0:
            continue
        key = (e.root, e.quality)
        acc[key] = acc.get(key, 0.0) + ov
    if not acc:
        return (None, "")
    (root, qual), best_overlap = max(acc.items(), key=lambda kv: kv[1])
    if best_overlap / bar_dur < _MIN_BAR_COVERAGE:
        # No single ACE chord confidently owns this bar — too ambiguous to
        # risk promoting. Keep librosa.
        return (None, "")
    return (root, qual)


def enrich_qualities_from_ace(
    bar_grid: List[Dict],
    audio_path: str,
    ace_events: Optional[List] = None,
) -> Tuple[List[Dict], Dict]:
    """Promote librosa triads to ACE's richer quality where ROOTS agree.

    Args:
        bar_grid: list of {"bar", "chord", "start_time", "end_time", ...} as
                  produced by chart_formatter's bar-grid build.
        audio_path: path to the source audio ACE runs on (full mix).
        ace_events: optional pre-computed ACE ChordEvent list (used by the
                    validation harness to avoid re-running ACE per A/B arm).

    Returns:
        (new_bar_grid, telemetry). new_bar_grid is a deep-ish copy: each bar
        dict is copied; promoted bars get an updated "chord" and a
        "quality_enriched" source flag + "enriched_from" original chord.
    """
    telemetry: Dict = {
        "bars_total": len(bar_grid),
        "promoted": 0,
        "root_disagree_skipped": 0,
        "no_richer_quality": 0,
        "ace_events": 0,
        "promotions": [],  # list of {bar, from, to}
    }
    if not bar_grid:
        return bar_grid, telemetry

    if ace_events is None:
        try:
            from processing.chord_detector_ace import detect_chords_ace
            prog = detect_chords_ace(str(audio_path))
            ace_events = prog.chords or []
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"[quality_enricher] ACE pass failed (non-fatal): {e}")
            return bar_grid, telemetry

    telemetry["ace_events"] = len(ace_events)
    if not ace_events:
        logger.info("[quality_enricher] ACE returned no events — keeping librosa")
        return bar_grid, telemetry

    out: List[Dict] = []
    for bar in bar_grid:
        nb = dict(bar)
        lib_chord = nb.get("chord")
        lib_root, lib_qual, lib_bass = _parse(lib_chord)

        # Only triad bars are candidates. If librosa already emitted an
        # extension (it never does today, but be future-proof), leave it.
        if lib_root is None or lib_qual not in ("", "m"):
            out.append(nb)
            continue

        start = float(nb.get("start_time") or 0.0)
        end = float(nb.get("end_time") or start)
        if end <= start:
            out.append(nb)
            continue

        ace_root, ace_qual = _ace_overlap_quality(ace_events, start, end)
        if ace_root is None:
            telemetry["no_richer_quality"] += 1
            out.append(nb)
            continue

        if not _same_root(lib_root, ace_root):
            telemetry["root_disagree_skipped"] += 1
            out.append(nb)
            continue

        # Roots agree. Is ACE offering a richer, legal enrichment?
        legal = _LEGAL_PROMOTIONS.get(lib_qual, set())
        if (
            ace_qual
            and ace_qual != lib_qual
            and ace_qual in _INFORMATIVE_EXTENSIONS
            and ace_qual in legal
        ):
            new_chord = lib_root + ace_qual
            if lib_bass:
                new_chord = f"{new_chord}/{lib_bass}"
            prev_source = nb.get("source") or ""
            nb["chord"] = new_chord
            nb["enriched_from"] = lib_chord
            nb["source"] = (
                f"{prev_source}+quality_enriched" if prev_source
                else "quality_enriched"
            )
            telemetry["promoted"] += 1
            telemetry["promotions"].append({
                "bar": nb.get("bar"),
                "from": lib_chord,
                "to": new_chord,
            })
        else:
            telemetry["no_richer_quality"] += 1
        out.append(nb)

    logger.info(
        "[quality_enricher] %d/%d bars promoted (root-disagree skipped %d, "
        "no-richer %d, ace_events %d)",
        telemetry["promoted"], telemetry["bars_total"],
        telemetry["root_disagree_skipped"], telemetry["no_richer_quality"],
        telemetry["ace_events"],
    )
    return out, telemetry


def is_enricher_enabled() -> bool:
    return os.environ.get("ENABLE_QUALITY_ENRICHER", "").lower() in (
        "1", "true", "yes",
    )
