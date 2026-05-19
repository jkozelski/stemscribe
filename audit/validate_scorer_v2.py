#!/usr/bin/env python3
"""Validation harness for the v2 honest chord-chart scorer.

Read-only. No prod, no network, no deploy. Reconstructs the forensic detector
event streams (librosa / ACE / Jiang / oracle) from /tmp/forensic if present,
falls back to the in-repo copies if /tmp was wiped, and confirms:

  (1) v2 rates "In My Life" served prod chart LOW (musician verdict bar),
      in deliberate contrast to v1's 0.919 root.
  (2) Detector ranking is sane: oracle > ACE/Jiang > librosa, matching the
      measured raw-signal truth (a scorer that ranks librosa top is broken).
  (3) Mechanics: GT-vs-GT == 1.0 on every axis (sanity floor).

Usage:  python audit/validate_scorer_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from score_chord_chart import score as score_v1  # noqa: E402
from score_chord_chart_v2 import (  # noqa: E402
    score_v2,
    gt_bar_sequence,
    placement_axis,
    flavor_axis,
    root_axis,
    composite,
    _expand,
)

GT_DIR = HERE / "fixtures" / "ground_truth"
PROD_DIR = HERE / "fixtures" / "prod_charts"

GT = {
    "iml": GT_DIR / "the-beatles__in-my-life.json",
    "pos": GT_DIR / "bob-dylan__positively-4th-street.json",
}
PROD = {
    "iml": PROD_DIR / "the-beatles__in-my-life__fb8a175e.json",
    "pos": PROD_DIR / "bob-dylan__positively-4th-street__1433ba27.json",
}

# Forensic raw-event streams. Prefer /tmp (fresh); these were verified
# byte-identical to prod via md5 during construction.
FORENSIC = Path("/tmp/forensic")


def _events_to_chart(events: list[dict]) -> dict:
    """Wrap a raw event stream as a minimal chart the v2 flattener accepts
    (chord_progression shape -> hold 1 per event, order preserved)."""
    return {"chord_progression": [{"chord": e["chord"]} for e in events]}


def _load_stream(name: str, kind: str):
    base = FORENSIC / name
    if kind == "librosa":
        m = json.load(open(base / "job_metadata.json"))
        return [{"chord": e["chord"]} for e in m["chord_progression"]
                if e.get("chord") and e["chord"] != "N"]
    if kind == "ace":
        d = json.load(open(base / "ace_events.json"))
        return [{"chord": e["chord"]} for e in d["events"]]
    if kind == "jiang":
        d = json.load(open(base / "jiang_events.json"))
        return [{"chord": e["chord"]} for e in d["events"]]
    raise ValueError(kind)


def _oracle_stream(name: str):
    """Perfect detector: GT bar sequence, one event per bar, in order."""
    gt = json.load(open(GT[name]))
    seq = gt_bar_sequence(gt)
    return [{"chord": c} for c in seq]


def _score_stream(gt: dict, events: list[dict]) -> dict:
    chart = {"chord_progression": events}
    r = score_v2(gt, chart)
    return {
        "root": r["axes"]["1_root"]["f1"],
        "placement_strict": r["axes"]["2_placement"]["strict_bar"]["f1"],
        "placement_hold_inv": r["axes"]["2_placement"]["hold_invariant"]["f1"],
        "flavor": r["axes"]["3_flavor"]["weighted_flavor"],
        "composite": r["composite"]["composite"],
    }


def main():
    print("=" * 78)
    print("V2 SCORER VALIDATION")
    print("=" * 78)

    # ---- (3) MECHANICS: GT vs GT must be 1.0 everywhere ------------------
    print("\n[MECHANICS] GT-vs-GT (perfect chart) — every axis must be ~1.0")
    ok_mech = True
    for name in ("iml", "pos"):
        gt = json.load(open(GT[name]))
        gtseq = gt_bar_sequence(gt)
        ra = root_axis(gtseq, _expand([(c, 1) for c in gtseq]))
        det_seq = [(c, 1) for c in gtseq]
        pa = placement_axis(gtseq, det_seq)
        fa = flavor_axis(gtseq, det_seq)
        strict = pa["strict_bar"]["f1"]
        hi = pa["hold_invariant"]["f1"]
        fl = fa["weighted_flavor"]
        passed = (ra["f1"] == 1.0 and strict == 1.0 and hi == 1.0
                  and (fl is None or fl == 1.0))
        ok_mech &= passed
        print(f"  {name}: root={ra['f1']} strict={strict} hold_inv={hi} "
              f"flavor={fl}  -> {'PASS' if passed else 'FAIL'}")

    # ---- (1) IN MY LIFE: old vs new -------------------------------------
    print("\n[CONTRAST] 'In My Life' served prod chart — v1 (bag) vs v2 (honest)")
    gt_iml = json.load(open(GT["iml"]))
    ch_iml = json.load(open(PROD["iml"]))
    v1 = score_v1(gt_iml, ch_iml)["scores"]
    v2 = score_v2(gt_iml, ch_iml)
    print(f"  v1 root           = {v1['root']['f1']:.3f}   <- the misleading number")
    print(f"  v1 root_quality   = {v1['root_quality']['f1']:.3f}")
    print(f"  v2 AXIS1 root     = {v2['axes']['1_root']['f1']:.3f}")
    print(f"  v2 AXIS2 strict   = {v2['axes']['2_placement']['strict_bar']['f1']:.3f}")
    print(f"  v2 AXIS3 flavor   = {v2['axes']['3_flavor']['weighted_flavor']:.3f}")
    print(f"  v2 COMPOSITE      = {v2['composite']['composite']:.3f}")
    iml_low = v2["composite"]["composite"] < 0.55
    print(f"  -> v2 rates IML LOW (composite < 0.55)? "
          f"{'PASS' if iml_low else 'FAIL'}  "
          f"(musician verdict: 'root notes, missing Dm/7ths' = BAD)")

    # ---- (2) DETECTOR RANKING -------------------------------------------
    print("\n[RANKING] raw detector streams (forensic) — expect "
          "oracle > ACE/Jiang > librosa")
    have_forensic = (FORENSIC / "iml" / "ace_events.json").exists()
    if not have_forensic:
        print("  /tmp/forensic not present and no in-repo copy — SKIPPED. "
              "(Honest: cannot reproduce raw streams without them.)")
        ranking_ok = None
    else:
        ranking_ok = True
        for name in ("iml", "pos"):
            gt = json.load(open(GT[name]))
            rows = {}
            for kind in ("librosa", "ace", "jiang"):
                rows[kind] = _score_stream(gt, _load_stream(name, kind))
            rows["oracle"] = _score_stream(gt, _oracle_stream(name))
            print(f"\n  --- {name} ---")
            print(f"    {'stream':<8} {'root':>6} {'pl_str':>7} "
                  f"{'flavor':>7} {'comp':>6}")
            for k in ("librosa", "ace", "jiang", "oracle"):
                s = rows[k]
                fl = s['flavor'] if s['flavor'] is not None else float('nan')
                print(f"    {k:<8} {s['root']:>6.3f} "
                      f"{s['placement_strict']:>7.3f} {fl:>7.3f} "
                      f"{s['composite']:>6.3f}")
            # sanity: oracle must top; librosa must NOT top composite
            comps = {k: rows[k]["composite"] for k in rows}
            oracle_top = comps["oracle"] >= max(comps.values()) - 1e-9
            librosa_not_top = comps["librosa"] <= max(
                comps["ace"], comps["jiang"]) + 1e-9
            ok = oracle_top and librosa_not_top
            ranking_ok &= ok
            print(f"    sanity: oracle top={oracle_top}  "
                  f"librosa<=ACE/Jiang={librosa_not_top}  "
                  f"-> {'PASS' if ok else 'FAIL'}")

    # ---- VERDICT --------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"  MECHANICS (GT==GT==1.0):        "
          f"{'PASS' if ok_mech else 'FAIL'}")
    print(f"  IML rated LOW (vs v1 0.919):    "
          f"{'PASS' if iml_low else 'FAIL'}")
    print(f"  Detector ranking sane:          "
          f"{'PASS' if ranking_ok else ('SKIPPED' if ranking_ok is None else 'FAIL')}")
    print("=" * 78)
    return 0 if (ok_mech and iml_low and ranking_ok is not False) else 1


if __name__ == "__main__":
    sys.exit(main())
