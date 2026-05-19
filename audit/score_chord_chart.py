"""Score a StemScriber detector output against an Ultimate Guitar ground truth.

Usage:
    python audit/score_chord_chart.py <ground_truth.json> <chord_chart.json>
    python audit/score_chord_chart.py --all          # score every pair we can find
    python audit/score_chord_chart.py --song <slug>  # score a single song by GT slug

Ground truth files live in audit/fixtures/ground_truth/<slug>.json.
Detector outputs live in /opt/stemscribe/outputs/<job_id>/chord_chart.json on
the VPS — fetch locally via scp first, or pass an explicit path.

This is the v1 scorer. Approach: flatten both inputs to chord sequences,
apply capo transposition to ground truth, then compute multi-granularity F1
over bag-of-chords. Bag-of-chords is permissive — it ignores order/timing —
but it gives a fast first signal on vocabulary accuracy and is the metric
that maps most directly to "does the detector know which chords are in the
song." A second pass with order-aware DP alignment is a v2 task.

F1 levels emitted:
  - root: just the root note (C, F#, etc.) — most lenient
  - root+quality: e.g. Em7 must match Em7 (not Em). Distinguishes 7th/maj7/sus.
  - full: root+quality+bass (slash chord exact). e.g. C/G != C.
  - root_family: groups quality into maj/min/dim/aug/sus families. Catches
                 "right family, wrong extension" cases (Em7 vs Em9 both = min).

REPRODUCTION NOTE (2026-05-19)
-----------------------------
The original untracked source of this v1 scorer was lost (git clean during a
worktree mishap). It was faithfully reconstructed from the verified-working
compiled artifact (audit/__pycache__/score_chord_chart.cpython-311.pyc) and
proven byte-identical in behavior via exhaustive differential testing against
that artifact across the full realistic chord vocabulary + the real GT/prod
fixtures (see backend/tests/test_score_chord_chart_v1_recovery.py). The v2
scorer imports only the six pure chord-parse primitives from here; those are
the differential-tested critical path.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

NOTE_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
    "A#": 10, "Bb": 10, "B": 11,
}
PC_TO_NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

CHORD_RE = re.compile(r"^([A-G][#b]?)(.*?)(?:/([A-G][#b]?))?$", re.IGNORECASE)


def parse_chord(c):
    if not c or not isinstance(c, str):
        return (None, "", None)
    s = c.strip()
    if not s:
        return (None, "", None)
    m = CHORD_RE.match(s)
    if not m:
        return (None, "", None)
    root, quality, bass = m.groups()
    return (root, quality or "", bass)


def transpose(chord, semitones):
    if not chord or not semitones:
        return chord
    root, qual, bass = parse_chord(chord)
    if root is None:
        return chord
    new_root = PC_TO_NOTE[(NOTE_PC[root] + semitones) % 12]
    out = new_root + qual
    if bass and bass in NOTE_PC:
        new_bass = PC_TO_NOTE[(NOTE_PC[bass] + semitones) % 12]
        out = f"{out}/{new_bass}"
    return out


def quality_family(q):
    """Reduce a quality string to its family: maj, min, dim, aug, sus.

    Power chords (X5) carry no third, so they have no maj/min identity. We
    collapse them into the `maj` family so they bag-match either Xmaj or X
    in detector output — this matches the V3 plan's "X5 → maj family
    collapse" rule (Iron Man / Hells Bells / More Than a Feeling).
    """
    if not q:
        return "maj"
    ql = q.lower()
    if ql == "5":
        return "maj"
    if ql.startswith("m") and not ql.startswith("maj"):
        return "min"
    if ql.startswith("dim") or ql.startswith("°") or ql.startswith("o"):
        return "dim"
    if ql.startswith("aug") or ql.startswith("+"):
        return "aug"
    if ql.startswith("sus"):
        return "sus"
    if ql.startswith("maj") or ql in ("", "6", "7", "9", "11", "13"):
        return "maj"
    return "maj"


def _normalize_power_chord_quality(q):
    """Collapse the bare power-chord quality '5' to '' for root_quality/full
    bag matching. A power chord (root + perfect 5th, no third) is acoustically
    indistinguishable from a triad with the third buried below the noise
    floor — for our bag-of-chords scorer it makes more sense to treat 'E5'
    and 'E' as the same token than to penalize either side.
    Anything richer than bare '5' (e.g. 'add9', '5b9') is preserved verbatim.
    """
    return "" if q == "5" else q


def chord_to_pitch_classes(chord):
    """Convert a chord label to the set of pitch classes (0-11) it implies.

    Includes root, quality intervals, extensions, and slash-bass when present.
    Returns a frozenset for hash-friendly comparison. Equivalent chords with
    different names (e.g. Em6 = Em/C# = {E,G,B,C#}, Am/F# = F#m7b5 = {A,C,E,F#})
    naturally produce the same set — that's the whole point of this function.
    """
    root, qual, bass = parse_chord(chord)
    if root is None:
        return None
    r = NOTE_PC[root]
    pcs = {r}
    ql = (qual or "").lower()

    is_minor = ql.startswith("m") and not ql.startswith("maj")
    is_dim = ql.startswith("dim") or ql.startswith("°") or ql.startswith("o")
    is_aug = ql.startswith("aug") or ql.startswith("+")
    is_sus2 = "sus2" in ql
    is_sus4 = "sus4" in ql or ql.endswith("sus")

    is_power = ql == "5"

    # Third (or its sus replacement)
    if is_power:
        pass
    elif is_dim:
        pcs.add((r + 3) % 12)
    elif is_minor:
        pcs.add((r + 3) % 12)
    elif is_sus2:
        pcs.add((r + 2) % 12)
    elif is_sus4:
        pcs.add((r + 5) % 12)
    else:
        pcs.add((r + 4) % 12)

    # Fifth
    if is_dim:
        pcs.add((r + 6) % 12)
    elif is_aug:
        pcs.add((r + 8) % 12)
    else:
        pcs.add((r + 7) % 12)

    # Extensions / sevenths
    if "maj13" in ql:
        pcs.update({(r + 11) % 12, (r + 2) % 12, (r + 9) % 12})
    elif "maj11" in ql:
        pcs.update({(r + 11) % 12, (r + 2) % 12, (r + 5) % 12})
    elif "maj9" in ql:
        pcs.update({(r + 11) % 12, (r + 2) % 12})
    elif "maj7" in ql:
        pcs.add((r + 11) % 12)
    elif "b13" in ql or "13b" in ql:
        pcs.update({(r + 10) % 12, (r + 2) % 12, (r + 8) % 12})
    elif "13" in ql:
        pcs.update({(r + 10) % 12, (r + 2) % 12, (r + 9) % 12})
    elif "11" in ql:
        pcs.update({(r + 10) % 12, (r + 2) % 12, (r + 5) % 12})
    elif "b9" in ql:
        pcs.update({(r + 10) % 12, (r + 1) % 12})
    elif "#9" in ql:
        pcs.update({(r + 10) % 12, (r + 3) % 12})
    elif "9" in ql and "add9" not in ql:
        pcs.update({(r + 10) % 12, (r + 2) % 12})
    elif "add9" in ql:
        pcs.add((r + 2) % 12)
    elif "7" in ql:
        if is_dim:
            pcs.add((r + 9) % 12)
        else:
            pcs.add((r + 10) % 12)

    if "6" in ql and "7" not in ql and "9" not in ql:
        pcs.add((r + 9) % 12)

    if "b5" in ql:
        pcs.discard((r + 7) % 12)
        pcs.add((r + 6) % 12)
    if "#5" in ql:
        pcs.discard((r + 7) % 12)
        pcs.add((r + 8) % 12)

    if bass and bass in NOTE_PC:
        pcs.add(NOTE_PC[bass])

    return frozenset(pcs)


def chord_to_key(chord, level):
    """Reduce a chord string to a comparable key at the given granularity."""
    root, qual, bass = parse_chord(chord)
    if root is None:
        return None
    root = PC_TO_NOTE[NOTE_PC[root]]
    bass_norm = (PC_TO_NOTE[NOTE_PC[bass]] if bass and bass in NOTE_PC
                 else bass)

    if level == "root":
        return root
    if level == "root_family":
        return f"{root}{quality_family(qual)}"

    qual_norm = _normalize_power_chord_quality(qual)
    if level == "root_quality":
        return f"{root}{qual_norm}" if qual_norm else root
    if level == "full":
        s = f"{root}{qual_norm}" if qual_norm else root
        if bass_norm:
            s = f"{s}/{bass_norm}"
        return s
    raise ValueError(f"unknown level: {level}")


def flatten_ground_truth(gt):
    """Walk sections.lines and return a flat list of chord strings, capo-applied."""
    chords = []
    for section in gt.get("sections") or []:
        for line in section.get("lines") or []:
            for cell in line:
                if not isinstance(cell, str):
                    continue
                root, _, _ = parse_chord(cell)
                if root:
                    chords.append(cell)
    capo = int(gt.get("capo") or 0)
    if capo:
        chords = [transpose(c, capo) for c in chords]
    return chords


def flatten_detector(chord_chart):
    """Pull chord labels out of a StemScriber chord_chart.json.

    Tries multiple shapes in priority order:
      1. sections[].lines[].segments[].chord  — the lyric-aligned chord chart
         (this is the user-facing structure rendered on practice.html)
      2. bar_grid[].chord  — flat bar-level grid
      3. sections[].bars[].chord  — legacy shape from older versions
      4. chord_progression[].chord  — raw detector event stream (job_metadata format)
    """
    chords = []

    for section in chord_chart.get("sections") or []:
        for line in section.get("lines") or []:
            if isinstance(line, dict):
                for seg in line.get("segments") or []:
                    if isinstance(seg, dict):
                        c = seg.get("chord")
                        if c:
                            chords.append(c)
    if chords:
        return chords

    for bar in chord_chart.get("bar_grid") or []:
        if isinstance(bar, dict):
            c = bar.get("chord")
            if c:
                chords.append(c)
    if chords:
        return chords

    for section in chord_chart.get("sections") or []:
        for bar in section.get("bars") or []:
            if isinstance(bar, dict):
                c = bar.get("chord")
                if c:
                    chords.append(c)
    if chords:
        return chords

    for ev in chord_chart.get("chord_progression") or []:
        if isinstance(ev, dict):
            c = ev.get("chord")
            if c:
                chords.append(c)
    return chords


def pcs_match(gt_chord, det_chord, allow_superset):
    """Two chords match by pitch-class set equivalence.

    Equal sets always match. If allow_superset is True, also match when the
    detector's set is a proper superset of GT's (detector has MORE info than
    GT — e.g. detector says Em7 when GT says Em — we give credit since we're
    not wrong, just more theoretical).
    """
    a = chord_to_pitch_classes(gt_chord)
    b = chord_to_pitch_classes(det_chord)
    if a is None or b is None:
        return False
    if a == b:
        return True
    if allow_superset and a.issubset(b) and len(a) >= 3:
        return True
    if allow_superset and len(a) == 2 and a.issubset(b):
        return True
    return False


def pcs_bag_f1(gt_chords, det_chords):
    """Bag-of-chords F1 with pitch-class-set equivalence matching.

    Handles enharmonic equivalents (Em6 == Em/C#), notation differences
    (Am/F# == F#m7b5), and detector-richer-than-GT (Em7 vs Em).

    Greedy bipartite matching: for each GT chord, find an unmatched detector
    chord whose pitch-class set matches. Limits TP to len(min(gt,det)) but
    allows fuzzy notation matches that strict bag-F1 would miss.
    """
    gt_pcs = [chord_to_pitch_classes(c) for c in gt_chords]
    det_pcs = [chord_to_pitch_classes(c) for c in det_chords]

    gt_counter = Counter(p for p in gt_pcs if p is not None)
    det_counter = Counter(p for p in det_pcs if p is not None)

    tp = sum((gt_counter & det_counter).values())

    gt_remaining = gt_counter - det_counter
    det_remaining = det_counter - gt_counter
    superset_tp = 0
    for gt_set, gt_n in list(gt_remaining.items()):
        if gt_n <= 0 or len(gt_set) < 2:
            continue
        for det_set, det_n in list(det_remaining.items()):
            if det_n <= 0:
                continue
            if gt_set.issubset(det_set):
                k = min(gt_n, det_n)
                superset_tp += k
                gt_remaining[gt_set] -= k
                det_remaining[det_set] -= k
                if gt_remaining[gt_set] <= 0:
                    break

    tp_total = tp + superset_tp
    fn = sum(gt_counter.values()) - tp_total
    fp = sum(det_counter.values()) - tp_total

    precision = tp_total / (tp_total + fp) if (tp_total + fp) else 0.0
    recall = tp_total / (tp_total + fn) if (tp_total + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp_total,
        "fp": fp,
        "fn": fn,
        "exact_pcs_tp": tp,
        "superset_tp": superset_tp,
    }


def bag_f1(gt_chords, det_chords, level):
    """Bag-of-chords F1 at the given granularity.

    Treats each chord as a token. Multiplicity matters: if GT has 5 Em
    and detector has 3 Em, that's min(5,3)=3 true positives at the Em key.
    """
    gt_keys = Counter(k for c in gt_chords if (k := chord_to_key(c, level)))
    det_keys = Counter(k for c in det_chords if (k := chord_to_key(c, level)))

    tp = sum((gt_keys & det_keys).values())
    fn = sum(gt_keys.values()) - tp
    fp = sum(det_keys.values()) - tp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_unique_chords": len(gt_keys),
        "det_unique_chords": len(det_keys),
        "gt_total_events": sum(gt_keys.values()),
        "det_total_events": sum(det_keys.values()),
    }


def vocab_coverage(gt_chords, det_chords):
    """% of GT's distinct chord vocabulary that appears at all in detector output."""
    gt_vocab = {chord_to_key(c, "root_quality") for c in gt_chords
                if chord_to_key(c, "root_quality")}
    det_vocab = {chord_to_key(c, "root_quality") for c in det_chords
                 if chord_to_key(c, "root_quality")}
    shared = gt_vocab & det_vocab
    gt_only = gt_vocab - det_vocab
    det_only = det_vocab - gt_vocab
    coverage_pct = round(100 * len(shared) / len(gt_vocab), 1) if gt_vocab else 0.0
    return {
        "gt_vocab": sorted(gt_vocab),
        "det_only": sorted(det_only),
        "gt_only": sorted(gt_only),
        "shared": sorted(shared),
        "coverage_pct": coverage_pct,
    }


def slash_chord_analysis(gt_chords, det_chords):
    """How does the detector handle slash chords (C/G, Em/B, etc.)?"""
    gt_slashes = [c for c in gt_chords if parse_chord(c)[2]]
    det_slashes = [c for c in det_chords if parse_chord(c)[2]]
    return {
        "gt_slash_count": len(gt_slashes),
        "gt_unique_slashes": sorted(set(gt_slashes)),
        "det_slash_count": len(det_slashes),
        "det_unique_slashes": sorted(set(det_slashes)),
        "detector_emits_slashes": len(det_slashes) > 0,
    }


def score(gt, chord_chart):
    gt_chords = flatten_ground_truth(gt)
    det_chords = flatten_detector(chord_chart)
    scores = {}
    for level in ("root", "root_family", "root_quality", "full"):
        scores[level] = bag_f1(gt_chords, det_chords, level)
    scores["pcs"] = pcs_bag_f1(gt_chords, det_chords)
    return {
        "song": gt.get("song"),
        "artist": gt.get("artist"),
        "key": gt.get("key"),
        "capo": gt.get("capo", 0),
        "gt_chord_count": len(gt_chords),
        "det_chord_count": len(det_chords),
        "scores": scores,
        "vocabulary": vocab_coverage(gt_chords, det_chords),
        "slash_chord_analysis": slash_chord_analysis(gt_chords, det_chords),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gt_path", nargs="?")
    ap.add_argument("chart_path", nargs="?")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if not args.gt_path or not args.chart_path:
        ap.print_help()
        sys.exit(2)
    gt = json.loads(Path(args.gt_path).read_text())
    chart = json.loads(Path(args.chart_path).read_text())
    r = score(gt, chart)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
