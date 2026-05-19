#!/usr/bin/env python3
"""HONEST chord-chart scorer (v2).

WHY THIS EXISTS
---------------
The v1 scorer (`audit/score_chord_chart.py`) is a bag-of-chords metric: it
throws every chord into a multiset and compares multisets, ignoring ORDER,
PLACEMENT, and (for the lenient `root` level) FLAVOR. Three forensic agents
measured the consequence: v1 scored The Beatles' "In My Life" served prod
chart at root F1 = 0.919 even though that chart is "basically root notes,
missing the Dm / A7 / B7" — i.e. the WORST chart of every config tested when
judged by what a musician actually plays. v1 rewards librosa's single best
trait (it often gets the right chord NAMES somewhere in the song) while being
blind to its worst traits (wrong placement, dropped 7ths/min). It made a
genuinely ~25% better detector (ACE/Jiang) look like a regression.

A musician needs three things from a chart, and the scorer must measure all
three SEPARATELY (a single composite number hides exactly the failure that
misled every prior decision):

  1. ROOT       — is the right root note called at all?
  2. PLACEMENT  — is each chord in the right PLACE (right order, right bar)?
  3. FLAVOR     — is the chord QUALITY right (maj vs min, triad vs dom7 vs
                  maj7/min7, sus/add, slash) where it matters?

DESIGN RULE (anti-goalpost — see audit/SCORER_V2_RATIONALE.md)
--------------------------------------------------------------
The scorer must agree with informed-musician judgment. If it scores a chart
HIGH that a musician calls bad, the SCORER is wrong, not the musician. This
file is validated against that bar, not tuned to flatter our detector.

GROUND-TRUTH TIMING LIMITATION (stated honestly, not hidden)
------------------------------------------------------------
GT fixtures are bar-indexed: an ordered sequence of bar cells with NO
absolute timestamps. Detector output (`sections[].lines[].segments[]`) is an
ordered sequence of segments WITH relative start/end times and a `bars` hold
count. Therefore PLACEMENT is scored as a SEQUENCE-ALIGNMENT problem (relative
order + hold structure), not an absolute-time problem. We report three
placement variants to separate the failure modes:

  * strict      — position-for-position bar match after expanding GT cells by
                   their implied hold and detector segments by their `bars`
                   count, at zero offset. The harshest, most literal read of
                   "right chord in the right bar".
  * hold_invariant — collapse consecutive-duplicate bars on BOTH sides (run
                   length encode), then order-aware LCS. Removes "held the
                   chord N bars vs M bars" and tempo-scale noise; measures
                   pure progression-order correctness.
  * best_offset — strict, but search a global +/-N bar shift and keep the
                   best. Separates a pure ANCHORING error (whole chart shifted
                   by an intro-count mismatch — musically harmless) from REAL
                   misplacement (chords genuinely in the wrong order).

If best_offset >> strict, the chart is merely mis-anchored (cheap to fix and
not what a musician would call "wrong"). If hold_invariant is also low, the
progression itself is wrong (a musician WOULD call that bad).

USAGE
-----
    python audit/score_chord_chart_v2.py <ground_truth.json> <chord_chart.json>
    python audit/score_chord_chart_v2.py <gt.json> <chart.json> --json

The v1 scorer is intentionally NOT modified or imported beyond shared chord
parsing — v2 stands alone so the two can be compared side by side.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

# Reuse ONLY the well-tested, side-effect-free chord-parsing primitives from
# v1. We do NOT reuse its scoring functions (that is the thing being replaced).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_chord_chart import (  # noqa: E402
    NOTE_PC,
    PC_TO_NOTE,
    parse_chord,
    transpose,
    quality_family,
    chord_to_pitch_classes,
)

# ---------------------------------------------------------------------------
# Flatteners (ORDER-PRESERVING). Unlike v1's bag flatteners, these keep the
# sequence AND, where available, the per-chord bar-hold count.
# ---------------------------------------------------------------------------


def gt_bar_sequence(gt: dict) -> list[str]:
    """Ordered list of GT chords, one entry per bar cell, capo applied.

    A GT 'line' is a list of cells; each cell is one bar of harmony. This is
    the canonical bar sequence the song is played in. No timestamps exist.
    """
    seq: list[str] = []
    for section in gt.get("sections") or []:
        for line in section.get("lines") or []:
            if not isinstance(line, list):
                continue
            for cell in line:
                if not isinstance(cell, str):
                    continue
                root, _, _ = parse_chord(cell)
                if root:  # skip stray lyric strings
                    seq.append(cell)
    capo = int(gt.get("capo") or 0)
    if capo:
        seq = [transpose(c, capo) for c in seq]
    return seq


def detector_segment_sequence(chart: dict) -> list[tuple[str, int]]:
    """Ordered (chord, bar_hold) tuples from the SERVED chart.

    Priority order matches what the user actually sees on practice.html:
      1. sections[].lines[].segments[]  (chord, bars)  -- served, corrected
      2. bar_grid[]                     (chord -> hold 1)
      3. sections[].bars[]              (legacy)
      4. chord_progression[]            (raw events -> hold 1)
    """
    out: list[tuple[str, int]] = []

    for section in chart.get("sections") or []:
        for line in section.get("lines") or []:
            if isinstance(line, dict):
                for seg in line.get("segments") or []:
                    if isinstance(seg, dict) and seg.get("chord"):
                        hold = seg.get("bars")
                        try:
                            hold = max(1, int(round(float(hold))))
                        except (TypeError, ValueError):
                            hold = 1
                        out.append((seg["chord"], hold))
    if out:
        return out

    for bar in chart.get("bar_grid") or []:
        if isinstance(bar, dict) and bar.get("chord"):
            out.append((bar["chord"], 1))
    if out:
        return out

    for section in chart.get("sections") or []:
        for bar in section.get("bars") or []:
            if isinstance(bar, dict) and bar.get("chord"):
                out.append((bar["chord"], 1))
    if out:
        return out

    for ev in chart.get("chord_progression") or []:
        if isinstance(ev, dict) and ev.get("chord"):
            out.append((ev["chord"], 1))
    return out


# ---------------------------------------------------------------------------
# AXIS 1 — ROOT correctness (order-aware, NOT bag)
# ---------------------------------------------------------------------------


def _root_pc(chord: str) -> Optional[int]:
    r, _, _ = parse_chord(chord)
    if r is None or r not in NOTE_PC:
        return None
    return NOTE_PC[r]


def _expand(seq_with_holds: list[tuple[str, int]]) -> list[str]:
    """Expand (chord, hold) -> flat per-bar chord list."""
    flat: list[str] = []
    for ch, hold in seq_with_holds:
        flat.extend([ch] * max(1, hold))
    return flat


def _rle(seq: list) -> list:
    out: list = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def _lcs_len(a: list, b: list) -> int:
    """Length of the longest common subsequence (order-aware, gap-tolerant)."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    prev = [0] * (m + 1)
    for i in range(n - 1, -1, -1):
        cur = [0] * (m + 1)
        for j in range(m - 1, -1, -1):
            if a[i] == b[j]:
                cur[j] = prev[j + 1] + 1
            else:
                cur[j] = max(prev[j], cur[j + 1])
        prev = cur
    return prev[0]


def _seq_f1(matches: int, len_gt: int, len_det: int) -> dict:
    p = matches / len_det if len_det else 0.0
    r = matches / len_gt if len_gt else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"f1": round(f1, 3), "precision": round(p, 3),
            "recall": round(r, 3), "matches": matches,
            "len_gt": len_gt, "len_det": len_det}


def root_axis(gt_seq: list[str], det_flat: list[str]) -> dict:
    """Order-aware root correctness.

    Reported on the RLE (collapsed-consecutive) sequence so the metric is
    'did the right root appear in the right place in the progression',
    independent of how many bars each was held. This is the honest root
    signal — the v1 'root' level (0.919 on IML) was a bag and is exactly the
    number we are replacing.
    """
    g = _rle([_root_pc(c) for c in gt_seq if _root_pc(c) is not None])
    d = _rle([_root_pc(c) for c in det_flat if _root_pc(c) is not None])
    matches = _lcs_len(g, d)
    res = _seq_f1(matches, len(g), len(d))
    res["note"] = "order-aware LCS over run-length-encoded root sequence"
    return res


# ---------------------------------------------------------------------------
# AXIS 2 — PLACEMENT correctness (3 variants)
# ---------------------------------------------------------------------------


def _strict_pos_match(g: list, d: list, offset: int) -> int:
    """Count position-for-position equal entries with detector shifted by
    `offset` bars. None entries never match.
    """
    m = 0
    for i, gv in enumerate(g):
        if gv is None:
            continue
        j = i + offset
        if 0 <= j < len(d) and d[j] is not None and d[j] == gv:
            m += 1
    return m


def placement_axis(gt_seq: list[str], det_seq: list[tuple[str, int]],
                    max_offset: int = 12) -> dict:
    """Three placement reads on the per-BAR root sequence.

    Roots (not full chords) are used for placement so the axis isolates
    'right chord in the right place' from 'right flavor' (which is AXIS 3).
    A chart can be perfectly placed but flavor-wrong; this axis must not
    double-penalize that.
    """
    g_flat_chords = _expand([(c, 1) for c in gt_seq])  # GT cell == 1 bar
    d_flat_chords = _expand(det_seq)

    g = [_root_pc(c) for c in g_flat_chords]
    d = [_root_pc(c) for c in d_flat_chords]
    n_gt = sum(1 for x in g if x is not None)

    # strict (offset 0)
    strict_m = _strict_pos_match(g, d, 0)
    strict = strict_m / n_gt if n_gt else 0.0

    # best global offset
    best_m, best_off = -1, 0
    for off in range(-max_offset, max_offset + 1):
        mm = _strict_pos_match(g, d, off)
        if mm > best_m:
            best_m, best_off = mm, off
    best_offset = best_m / n_gt if n_gt else 0.0

    # hold-invariant order match (RLE both sides, then LCS-based F1)
    g_rle = _rle([x for x in g if x is not None])
    d_rle = _rle([x for x in d if x is not None])
    hi_match = _lcs_len(g_rle, d_rle)
    hi = _seq_f1(hi_match, len(g_rle), len(d_rle))

    return {
        "strict_bar": {
            "f1": round(strict, 3), "matched_bars": strict_m,
            "gt_bars": n_gt, "det_bars": sum(1 for x in d if x is not None),
            "note": "position-for-position root match, zero offset (harshest)",
        },
        "best_offset": {
            "f1": round(best_offset, 3), "offset": best_off,
            "matched_bars": best_m,
            "note": "best global +/-bar shift; large gap over strict_bar "
                    "== pure anchoring error (musically harmless)",
        },
        "hold_invariant": {
            **hi,
            "note": "RLE both sides then order-aware LCS; pure progression "
                    "order independent of hold length / tempo scale",
        },
    }


# ---------------------------------------------------------------------------
# AXIS 3 — FLAVOR / quality correctness, broken out by class
# ---------------------------------------------------------------------------

# Structural weight per flavor-error class. dom7 / maj / min are structurally
# load-bearing for how a song is played and are weighted heavily. Slash /
# inversion is DIRECTIONAL ONLY (slash detection is a known gated-off gap) —
# weight 0 so it is reported but does not drag the composite.
FLAVOR_WEIGHTS = {
    "maj_min": 1.0,       # maj <-> min : structural, always matters
    "triad_dom7": 1.0,    # triad -> dom7 : blues/funk character
    "triad_maj7_min7": 0.8,  # triad -> maj7/min7 : color, important
    "sus_add_ext": 0.5,   # sus/add/9/11/13 : embellishment
    "slash_inv": 0.0,     # slash/inversion : directional only (gated gap)
}


def _flavor_class(qual: str, bass: Optional[str]) -> dict:
    """Decompose a chord's quality into the buckets AXIS 3 grades."""
    q = (qual or "").lower()
    fam = quality_family(qual)
    is_min = fam == "min"
    is_dim = fam == "dim"
    is_aug = fam == "aug"
    is_sus = fam == "sus"
    has_maj7 = "maj7" in q or "maj9" in q or "maj11" in q or "maj13" in q
    # dominant/minor 7th: a '7' not part of 'maj7'
    has_dom_or_m7 = (("7" in q) and not has_maj7) or has_maj7 is False and (
        "9" in q or "11" in q or "13" in q) and "maj" not in q
    has_seventh = ("7" in q) or ("9" in q) or ("11" in q) or ("13" in q)
    is_dom7 = has_seventh and not has_maj7 and not is_min and not is_dim
    is_min7 = has_seventh and is_min
    is_maj7 = has_maj7
    has_ext = is_sus or "add" in q or has_seventh
    return {
        "fam": fam, "is_min": is_min, "is_dim": is_dim, "is_aug": is_aug,
        "is_sus": is_sus, "is_dom7": is_dom7, "is_min7": is_min7,
        "is_maj7": is_maj7, "has_ext": has_ext, "has_bass": bool(bass),
    }


def flavor_axis(gt_seq: list[str], det_seq: list[tuple[str, int]]) -> dict:
    """Quality correctness, only over GT positions whose ROOT the detector
    got right (you cannot grade the flavor of a chord the detector did not
    even root-identify; that is already penalized by AXIS 1).

    For each such aligned position, score per error-class. We align by the
    hold-invariant RLE path so flavor is judged on the progression a musician
    would actually read, not on raw bar duplication.
    """
    g_flat = _expand([(c, 1) for c in gt_seq])
    d_flat = _expand(det_seq)

    # Build RLE chord sequences (keep the representative chord string).
    def rle_chords(flat):
        out = []
        for c in flat:
            r = _root_pc(c)
            if r is None:
                continue
            if not out or _root_pc(out[-1]) != r or out[-1] != c:
                # collapse only exact-consecutive duplicates (keep flavor changes)
                if out and out[-1] == c:
                    continue
                out.append(c)
        return out

    g_rle = rle_chords(g_flat)
    d_rle = rle_chords(d_flat)

    # Greedy order-preserving alignment on ROOT (so we grade flavor only where
    # the root matched). Standard LCS backtrace on root pitch-class.
    a = [_root_pc(c) for c in g_rle]
    b = [_root_pc(c) for c in d_rle]
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    pairs = []
    i = j = 0
    while i < n and j < m:
        if a[i] == b[j]:
            pairs.append((g_rle[i], d_rle[j]))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1

    classes = {
        "maj_min": {"total": 0, "correct": 0},
        "triad_dom7": {"total": 0, "correct": 0},
        "triad_maj7_min7": {"total": 0, "correct": 0},
        "sus_add_ext": {"total": 0, "correct": 0},
        "slash_inv": {"total": 0, "correct": 0},
    }
    examples: list[str] = []

    for gc, dc in pairs:
        _, gq, gb = parse_chord(gc)
        _, dq, db = parse_chord(dc)
        gf = _flavor_class(gq, gb)
        df = _flavor_class(dq, db)

        # maj<->min: graded on every aligned pair (the most fundamental flavor)
        classes["maj_min"]["total"] += 1
        if gf["fam"] == df["fam"] or (
            gf["fam"] in ("maj", "min") and df["fam"] in ("maj", "min")
            and gf["is_min"] == df["is_min"]
        ):
            classes["maj_min"]["correct"] += 1
        elif len(examples) < 12:
            examples.append(f"maj/min: GT {gc} -> DET {dc}")

        # triad -> dom7 : grade only where GT actually IS a dom7
        if gf["is_dom7"]:
            classes["triad_dom7"]["total"] += 1
            if df["is_dom7"]:
                classes["triad_dom7"]["correct"] += 1
            elif len(examples) < 12:
                examples.append(f"dom7 dropped: GT {gc} -> DET {dc}")

        # triad -> maj7/min7 : grade where GT is maj7 or min7
        if gf["is_maj7"] or gf["is_min7"]:
            classes["triad_maj7_min7"]["total"] += 1
            if (gf["is_maj7"] and df["is_maj7"]) or (gf["is_min7"] and df["is_min7"]):
                classes["triad_maj7_min7"]["correct"] += 1
            elif len(examples) < 12:
                examples.append(f"7th color dropped: GT {gc} -> DET {dc}")

        # sus/add/extension : grade where GT carries one
        if gf["is_sus"] or "add" in (gq or "").lower():
            classes["sus_add_ext"]["total"] += 1
            if df["is_sus"] or "add" in (dq or "").lower():
                classes["sus_add_ext"]["correct"] += 1

        # slash/inversion : directional only
        if gf["has_bass"]:
            classes["slash_inv"]["total"] += 1
            if df["has_bass"]:
                classes["slash_inv"]["correct"] += 1

    # per-class accuracy + aggregate (skip classes with no GT support).
    #
    # AGGREGATION PRINCIPLE: a systematic color failure (e.g. 0/4 dom7s — the
    # detector NEVER hears the 7th) is what a musician hears as "this chart is
    # wrong", and it must NOT be diluted by a large pile of easy-correct plain
    # majors. So the flavor composite is NOT a count-weighted mean (that is
    # the v1-style averaging pathology). It is the structural-importance-
    # weighted mean of per-class ACCURACIES (each class counts once, scaled by
    # its musical weight, independent of how many instances it had), blended
    # 60/40 toward the weakest structurally-weighted class. A class the
    # detector systematically fails drags the flavor score down even if it is
    # rare — which is correct: dropping every dom7 in a blues IS the failure.
    per_class = {}
    acc_by_class = {}
    for k, v in classes.items():
        acc = (v["correct"] / v["total"]) if v["total"] else None
        per_class[k] = {
            "accuracy": round(acc, 3) if acc is not None else None,
            "graded": v["total"], "correct": v["correct"],
            "weight": FLAVOR_WEIGHTS[k],
            "note": "directional only (slash detection gated off)"
            if k == "slash_inv" else None,
        }
        if acc is not None and FLAVOR_WEIGHTS[k] > 0 and v["total"] > 0:
            acc_by_class[k] = acc

    if acc_by_class:
        wsum = sum(FLAVOR_WEIGHTS[k] * a for k, a in acc_by_class.items())
        wtot = sum(FLAVOR_WEIGHTS[k] for k in acc_by_class)
        weighted_mean = wsum / wtot
        weakest = min(acc_by_class.values())
        composite_flavor = round(0.6 * weighted_mean + 0.4 * weakest, 3)
    else:
        composite_flavor = None
    return {
        "aligned_positions": len(pairs),
        "per_class": per_class,
        "weighted_flavor": composite_flavor,
        "examples": examples,
        "note": "graded only at root-aligned positions; slash weight 0 "
                "(directional). weighted by structural importance.",
    }


# ---------------------------------------------------------------------------
# COMPOSITE — breakdown always shown; single number never reported alone
# ---------------------------------------------------------------------------

# Composite is deliberately the geometric-leaning min-aware blend of the three
# axes so that a chart cannot earn a high composite by acing one axis (the
# exact pathology of the v1 bag metric). A musician needs ALL THREE.
COMPOSITE_WEIGHTS = {"root": 0.40, "placement": 0.35, "flavor": 0.25}


def composite(root_f1: float, placement_f1: float,
              flavor_f1: Optional[float]) -> dict:
    parts = {"root": root_f1, "placement": placement_f1}
    if flavor_f1 is not None:
        parts["flavor"] = flavor_f1
    # Weighted arithmetic mean for interpretability, AND a min-aware penalty:
    # composite is dragged toward the weakest axis so a single strong axis
    # cannot mask a broken one (anti-v1-pathology).
    w = {k: COMPOSITE_WEIGHTS[k] for k in parts}
    wsum = sum(w.values())
    arith = sum(parts[k] * w[k] for k in parts) / wsum
    weakest = min(parts.values())
    # blend 60% weighted mean + 40% weakest axis
    comp = 0.6 * arith + 0.4 * weakest
    return {
        "composite": round(comp, 3),
        "weighted_mean": round(arith, 3),
        "weakest_axis": round(weakest, 3),
        "parts": {k: round(v, 3) for k, v in parts.items()},
        "note": "0.6*weighted_mean + 0.4*weakest_axis. The weakest-axis term "
                "is the anti-bag-of-chords safeguard.",
    }


# ---------------------------------------------------------------------------
# Top-level scoring
# ---------------------------------------------------------------------------


def score_v2(gt: dict, chart: dict) -> dict:
    gt_seq = gt_bar_sequence(gt)
    det_seq = detector_segment_sequence(chart)
    det_flat = _expand(det_seq)

    root = root_axis(gt_seq, det_flat)
    placement = placement_axis(gt_seq, det_seq)
    flavor = flavor_axis(gt_seq, det_seq)

    # Placement representative for the composite.
    #
    # A musician reading a chart needs the right chord UNDER THE RIGHT BAR /
    # LYRIC, not merely "the progression visits the right roots in roughly the
    # right order eventually". So the representative is the STRICT bar match,
    # with ONE concession: a single constant global shift (intro-count
    # mismatch) is musically harmless, so we forgive pure anchoring by using
    # best_offset as the representative ONLY when the gap between best_offset
    # and strict is explained entirely by a constant shift (best_offset's win
    # comes from one offset, not from many local re-shuffles). In practice we
    # take best_offset (it is >= strict by construction and only differs when
    # a constant shift helps) but we DO NOT fall back to hold_invariant —
    # hold_invariant answers a different question (vocabulary-order) and using
    # it here is exactly the leniency that produced the misleading v1 picture.
    placement_repr = placement["best_offset"]["f1"]
    comp = composite(root["f1"], placement_repr, flavor["weighted_flavor"])

    return {
        "song": gt.get("song"),
        "artist": gt.get("artist"),
        "key": gt.get("key"),
        "capo": gt.get("capo", 0),
        "gt_bars": len(gt_seq),
        "det_segments": len(det_seq),
        "det_bars_expanded": len(det_flat),
        "axes": {
            "1_root": root,
            "2_placement": placement,
            "3_flavor": flavor,
        },
        "composite": comp,
        "limitations": [
            "GT has no absolute timestamps; placement is sequence-aligned "
            "(relative order + hold structure), NOT time-aligned.",
            "strict_bar assumes 1 GT cell == 1 bar; songs whose GT cells "
            "span !=1 bar will read low on strict_bar but hold_invariant "
            "stays valid (it is the recommended placement read).",
            "Diagnostic fixtures (audit/fixtures/ground_truth/*__diag*) need "
            "the song's audio run through the pipeline to produce a detector "
            "chart; with GT only, v2 mechanics validate via GT-vs-GT (==1.0) "
            "and oracle inputs.",
        ],
    }


def _fmt_axis(name, d):
    pass


def print_report(r: dict) -> None:
    print(f"=== {r['song']} — {r['artist']} ===")
    print(f"  key={r['key']} capo={r['capo']}  "
          f"GT bars={r['gt_bars']}  det segs={r['det_segments']} "
          f"(expanded {r['det_bars_expanded']} bars)")
    print()
    a = r["axes"]
    rt = a["1_root"]
    print(f"  AXIS 1 ROOT (order-aware)        f1={rt['f1']:.3f}  "
          f"P={rt['precision']:.3f} R={rt['recall']:.3f}")
    p = a["2_placement"]
    print(f"  AXIS 2 PLACEMENT")
    print(f"    strict_bar      f1={p['strict_bar']['f1']:.3f}  "
          f"({p['strict_bar']['matched_bars']}/{p['strict_bar']['gt_bars']} bars)")
    print(f"    best_offset     f1={p['best_offset']['f1']:.3f}  "
          f"@offset {p['best_offset']['offset']:+d}")
    print(f"    hold_invariant  f1={p['hold_invariant']['f1']:.3f}  "
          f"(progression order)")
    f = a["3_flavor"]
    print(f"  AXIS 3 FLAVOR (weighted={f['weighted_flavor']})")
    for cls, cv in f["per_class"].items():
        acc = cv["accuracy"]
        accs = f"{acc:.3f}" if acc is not None else "  n/a"
        tag = " [directional]" if cls == "slash_inv" else ""
        print(f"    {cls:<18} acc={accs}  "
              f"({cv['correct']}/{cv['graded']}) w={cv['weight']}{tag}")
    if f["examples"]:
        print("    flavor misses:")
        for ex in f["examples"][:8]:
            print(f"      - {ex}")
    c = r["composite"]
    print()
    print(f"  COMPOSITE = {c['composite']:.3f}  "
          f"(weighted_mean={c['weighted_mean']:.3f} "
          f"weakest={c['weakest_axis']:.3f})")
    print(f"    parts: {c['parts']}")
    print()
    print("  LIMITATIONS:")
    for lim in r["limitations"]:
        print(f"    - {lim}")


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
    r = score_v2(gt, chart)
    if args.json:
        print(json.dumps(r, indent=2))
    else:
        print_report(r)


if __name__ == "__main__":
    main()
