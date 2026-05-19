# Detector Signal Research — Anthropic Chord Corrector Gate

**Date:** 2026-05-10
**Audit dataset:** `/tmp/audit-may8-results/` (18 songs, `full` mode, post-corrector)
**Oracle:** `_oracle-final.jsonl` (Claude-API canonical chord sets via `llm_oracle.py`)

## Top finding

**`quality_flips`** — the count of Claude chord_set entries that share a root with a librosa chord BUT have the opposite major/minor quality (e.g., librosa had `D`, Claude returned `Dm`; librosa had `Bm`, Claude returned `B`).

- **Pearson r vs `delta_F1` = -0.649** — strongest of any pre-oracle signal tested.
- Recommended gate: **skip corrector if `quality_flips >= 2`** (or the slightly more aggressive composite below).
- At `quality_flips >= 2`: catches the two worst losses (Sister Golden Hair -0.52, Don't Let Me Be Misunderstood -0.47), **loses zero wins**, lifts mean F1 from 0.768 to 0.823.

### Why this works

The corrector's failure mode is **modal/key confusion**, not random noise. When Claude transposes the song to the wrong relative key (e.g., Bm→Am, F#m→F, D→Dm), it doesn't replace librosa's chords with completely novel ones — it replaces them with the *same root, opposite quality*. The big wins (Bad Company, Man in the Box, Every Rose) all have `quality_flips=0` because Claude returned a chord set that doesn't share roots with librosa's hallucinated key. That structural asymmetry is exactly the signal we need.

`drop_ratio` failed because high `drop_ratio` is correlated with the corrector being NEEDED (librosa was in a wrong key). `key_distance` failed because the wrong-key bug spans 0–5 semitones with no clean threshold. `quality_flips` is the only signal where wins and losses cleanly separate.

## Per-signal evaluation

(Wins to preserve: Bad Company F1=0.91, Man in the Box 0.89, Every Rose 0.75, Highway 0.67.
Losses to catch: Wildest Dreams F1=0.62, Misunderstood F1=0.15, plus Sister Golden Hair F1=0.40 and Hells Bells F1=0.67 — all are real `delta_F1 < -0.05` losses.)

| Signal | Pearson r vs delta_F1 | Catches Wildest Dreams? | Catches Misunderstood? | Preserves Bad Company / Man in the Box / Every Rose / Highway? | Mean F1 at proposed threshold |
|---|---|---|---|---|---|
| `quality_flips >= 2` | **−0.649** | No (qf=1) | **Yes** (qf=2) | **All preserved** (all qf=0) | **0.823** |
| `quality_flips >= 1` | −0.649 | Yes | Yes | All preserved (qf=0) but Dream On, Paint It Black, Day After Day lost | 0.818 |
| `quality_flips >= 2 OR (drop_ratio < 0.4 AND quality_flips >= 1)` | n/a (composite) | **Yes** (drop=0.33, qf=1) | **Yes** | **All preserved** | **0.840** |
| `drop_ratio >= 0.7` (the rejected gate) | +0.463 (wrong sign!) | No (drop=0.33) | Yes | **Bad Company, Man in Box, Every Rose lost** (drop=1.0) | 0.633 (regression) |
| `key_distance > 2` semitones | −0.045 | No (kd=0) | No (kd=2) | Bad Company kd=1 preserved; Highway kd=5 LOST | n/a (anti-correlated) |
| `overlap_C_L` (librosa chords kept by Claude) | −0.463 | n/a | n/a | Big wins all have overlap=0 (gate would skip them) | n/a (wrong direction) |
| `replaced_frac` (bars overwritten) | +0.435 | n/a | n/a | Bad Company/Man in the Box/Every Rose all 1.0 | n/a (wrong direction) |
| `n_claude` (Claude chord-set size) | −0.370 | n/a | n/a | n/a | n/a (weak) |
| Claude `notes` hedging language | n/a | No hedge text found in any loss | No | n/a | n/a (zero songs had hedge keywords — `claude_notes` is consistently confident prose regardless of correctness) |
| Subset compression (Claude ⊂ Librosa, shrinkage < 0.7) | +0.019 | n/a | n/a | Take On Me, House of Rising Sun, Free Fallin', Hey Joe all subsets (would gate big wins) | n/a (wrong direction) |
| `jac_claude_oracle` | +0.981 vs post-F1 | n/a | n/a | n/a | **Tautological** — `chords_used` after `full` mode IS Claude's chord_set, so this is a restatement of F1. Not usable as a pre-gate without calling the oracle. |

### Detail on the recommended gate (Gate D in the analysis)

```
Gate D: skip corrector if
  quality_flips >= 2
  OR (drop_ratio < 0.4 AND quality_flips >= 1)
```

Per-song result (18 songs, average F1):
- Baseline (corrector always on): **0.768**
- Corrector always off: **0.611**
- Gate D applied: **0.840** (+7.2 percentage points)

Gate D outcomes:
- Losses caught: Sister Golden Hair (−0.52), Don't Let Me Be Misunderstood (−0.47), Your Wildest Dreams (−0.38)
- Loss missed: Hells Bells (−0.17, qf=0, drop=0.43 — structurally indistinguishable from wins)
- Wins lost: Dream On (+0.07 — marginal win, net cost across 18 songs is 0.004 F1)

If you want a strictly safer gate that **never loses any win** in this audit, use the simpler `quality_flips >= 2` (Gate A): mean F1 = 0.823, catches 2 of 4 losses, zero wins lost.

## Computing `quality_flips`

`quality_flips` is computed inside the corrector AFTER Claude returns but BEFORE writing the bar_grid. Both `detected_norm` (librosa, already on the hot path) and `canonical_set` (Claude) are in scope at `chord_corrector_anthropic.py:518`.

```python
import re

def _root_pc(c: str):
    """Return integer pitch class 0..11 of chord root, or None."""
    m = re.match(r"^([A-G])([#b]?)", c or "")
    if not m:
        return None
    base = {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[m.group(1)]
    a = m.group(2)
    if a == '#': base += 1
    elif a == 'b': base -= 1
    return base % 12

def _is_minor(c: str) -> bool:
    # After normalize_chord: "m", "m7", "m9" etc. = minor; "maj7" = NOT minor
    return 'm' in c and 'maj' not in c.lower()

def _quality_flips(librosa_set: set[str], claude_set: set[str]) -> int:
    """Count Claude chords that share a root with a librosa chord but flip
    major<->minor. Excludes exact matches (root + quality both equal)."""
    lib_by_root: dict[int, list[str]] = {}
    for c in librosa_set:
        rp = _root_pc(c)
        if rp is not None:
            lib_by_root.setdefault(rp, []).append(c)
    flips = 0
    for cn in claude_set:
        rp = _root_pc(cn)
        if rp not in lib_by_root:
            continue
        candidates = lib_by_root[rp]
        if any(o == cn for o in candidates):
            continue  # exact match — not a flip
        if any(_is_minor(o) != _is_minor(cn) for o in candidates):
            flips += 1
    return flips
```

## Recommended gate (concrete)

Insert at `chord_corrector_anthropic.py:519` (right after `drop_ratio` is computed, before any of the three mode branches):

```python
# Quality-flip gate: corrector frequently fails by transposing to the
# wrong relative key (major<->minor swap on the same root). Skip when
# librosa was probably right.
quality_flips = _quality_flips(set(detected_norm.keys()), canonical_set)
qflip_threshold = int(os.environ.get("ANTHROPIC_CORRECTION_QFLIP_GATE", "2"))
qflip_aggressive_drop = float(os.environ.get("ANTHROPIC_CORRECTION_QFLIP_DROP_GATE", "0.4"))

skip_for_qflip = (
    quality_flips >= qflip_threshold
    or (quality_flips >= 1 and drop_ratio < qflip_aggressive_drop)
)

if skip_for_qflip:
    meta.update({
        "status": "skipped_quality_flip",
        "drop_ratio": round(drop_ratio, 2),
        "quality_flips": quality_flips,
        "detector_chord_set": sorted(detected_norm.keys()),
    })
    chord_chart["anthropic_correction"] = meta
    logger.warning(
        f"[chord_corrector:qflip] {title!r}: quality_flips={quality_flips}, "
        f"drop_ratio={drop_ratio:.0%}; likely wrong-key correction, leaving "
        f"detector output untouched"
    )
    return chord_chart
```

Env vars to add:
- `ANTHROPIC_CORRECTION_QFLIP_GATE` — minimum quality_flips to trigger skip. Default `2`. Set to `1` for max-aggressive (catches Wildest Dreams + the 3 worst losses, at cost of Dream On and Day After Day).
- `ANTHROPIC_CORRECTION_QFLIP_DROP_GATE` — drop_ratio ceiling below which a single quality flip is enough to skip. Default `0.4`. Models the intuition: "if Claude only differs from librosa in a small way (low drop_ratio) AND one of those differences is a major/minor flip, prefer librosa."

Also document at the top of the file: this signal is empirically validated against the 18-song May 8 audit (`/tmp/audit-may8-results/`). Re-run the audit before relaxing it.

## Open questions for tomorrow

1. **Hells Bells (-0.17 loss, F1=0.67) is invisible to this gate.** `quality_flips=0`, `drop_ratio=0.43`, Claude is a strict subset of librosa (`{A,D,E,G} ⊂ {A,Am,C,D,E,Em,G}`). Claude *correctly* dropped extras but also dropped `Am` and `C` which are real chords. Structurally indistinguishable from Take On Me (+0.19 win) which also has Claude ⊂ Librosa. **Possible fix: when Claude omits chords that appear in a high fraction of bars (e.g., librosa Am=27% of bars, C=7%), promote them back.** This is a different feature ("librosa chord retention by bar-weight"), not a gate. Worth a follow-up.

2. **Sample size n=18 is small.** Pearson r=-0.649 has a wide confidence interval. Re-run with the next batch of audit songs before lowering the threshold to 1. The 2 threshold is conservative and matches the cleanest empirical separation in this dataset.

3. **Confidence variance hypothesis (signal #3 in the brief) couldn't be tested** — `bar_grid[i].source_meta` only contains `replaced_from` and `reason`. No per-bar librosa confidence is plumbed through into the final chord chart. If we want to use librosa-confidence as a gate signal in the future, we'd need to surface it from `chord_decoder.py` (or wherever the K-K detection happens) into the bar grid first.

4. **Claude `notes` hedging hypothesis (signal #4) is dead.** None of the 18 songs — wins or losses — contain hedge keywords ("appears to be", "likely", "not entirely sure", "possibly", etc.). Claude returns confident prose for everything. Don't bother building on this.

5. **Should `quality_flips` only count flips for chords that appear frequently in librosa?** Currently a single-bar `Fm` in librosa that Claude rewrites to `F#m` counts as a flip. In practice the wrong-key losses all involve flips of high-frequency librosa chords (>15% of bars), so a bar-weight refinement might tighten the gate. Easy follow-up.

6. **Mode interaction.** This gate was derived in `full` mode (which both rewrites bar_grid AND applies format corrections). The same `quality_flips` signal should also fire in `replace` mode (same bar_grid logic). In `drop` mode, the failure profile is different — drop only ever removes; it can't introduce a quality flip. Probably safe to skip the gate in `drop` mode. Code as written applies in all modes — if you want it mode-scoped, wrap the block in `if use_mode != "drop":`.

7. **Hells Bells second-look.** Worth re-prompting Claude with the actual librosa chord histogram ("librosa says Am=27%, A=23%, C=7%, ...") and asking "of these, which are real?". This is a different architecture (Claude as a re-ranker, not a from-scratch generator). The current corrector ignores librosa's output entirely except to fill in the bar_grid AFTER Claude returns. That throw-away might be the root cause of the Hells Bells class of failures.
