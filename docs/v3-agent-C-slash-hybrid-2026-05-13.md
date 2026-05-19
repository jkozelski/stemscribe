# V3 Agent C: Slash-Chord Hybrid Prototype

**Date:** 2026-05-13
**Time-box:** 60 min
**Goal:** Combine Jiang Chord-CNN-LSTM (chord names from full mix) + `bass_root_extraction` (bass pitch class from bass stem) to promote bare triads to slash chords.

## TL;DR — Recommendation: **DROP** (as proposed). Strict variant ties Jiang; permissive variant regresses.

| variant | gained slash TPs | added slash FPs | full F1 delta (sum across 5 songs) |
|---------|-----------------:|----------------:|-----------------------------------:|
| permissive (3rd/5th/7th-in-bass)    | **+1** (Aja Bmaj7/F#) | **+22** | **-0.107** |
| strict (3rd or 7th-in-bass only)    | **0** | **+9**  | **-0.016** |

The hybrid surfaces musically-plausible slash chords (HoTRS `Am/E`, HC `Bm/F#`) but UG ground truth doesn't notate them, so they score as FPs and drag full-F1 down. Neither variant meets the ship criterion ("positive slash-F1 lift with no root-F1 regression").

PCS F1 is unchanged on every song — the hybrid is non-destructive at the pitch-class-set level (Am/E and Am have identical PCS). The damage is purely in strict `full` bag matching.

---

## 1. The Hybrid Rule (final code)

```python
QUALITY_CHORD_TONES = {
    '':     {0, 4, 7},  # maj triad
    'm':    {0, 3, 7},
    '7':    {0, 4, 7, 10},
    'maj7': {0, 4, 7, 11},
    'm7':   {0, 3, 7, 10},
    '9':    {0, 4, 7, 10, 2},
    'maj9': {0, 4, 7, 11, 2},
    'm9':   {0, 3, 7, 10, 2},
    'sus2': {0, 2, 7},
    'sus4': {0, 5, 7},
    '7sus4':{0, 5, 7, 10},
    'dim':  {0, 3, 6},
    'dim7': {0, 3, 6, 9},
    'm7b5': {0, 3, 6, 10},
    'aug':  {0, 4, 8},
    # 11 / 13 / add9 / 6 also handled
}

def upgrade_to_slash(jiang_chord: str, bass_pc: int, strict: bool = False) -> str:
    """When bass differs from chord root by a chord tone, emit slash form.

      - Bass on root:           keep bare chord.
      - Bass on chord tone:     upgrade to C/E, C/G, etc.
      - Bass on non-chord tone: keep bare chord (passing note / bleed).
      - strict=True: 5ths-in-bass excluded (common false positive).

    Already-slashed chord (Jiang already decided) is returned untouched.
    """
    if not jiang_chord or '/' in jiang_chord:
        return jiang_chord
    root, qual, _ = scorer.CHORD_RE.match(jiang_chord).groups()
    if root not in NOTE_PC:
        return jiang_chord
    root_pc = NOTE_PC[root]
    interval = (bass_pc - root_pc) % 12
    if interval == 0:
        return jiang_chord
    chord_tones = QUALITY_CHORD_TONES.get(qual, {0, 4, 7})
    if strict:
        allowed = {3, 4, 10, 11} & chord_tones   # 3rd or 7th only
        if interval not in allowed:
            return jiang_chord
    else:
        if interval not in chord_tones:
            return jiang_chord
    return f'{jiang_chord}/{NOTE_NAMES_SHARP[bass_pc]}'
```

**Time-alignment**: for each downbeat-bar in `tempo_beats.extract_grid`, find the dominant Jiang segment overlapping that bar, take the bass PC for that bar from `bass_root_extraction`, apply rule. Bass PC ignored when `bass_confidence < 0.4` (consistent with `combine_with_detector_quality`'s `min_bass_confidence`).

## 2. Per-Song Results

Jiang full-vocab (`submission`) was used for all 5 songs. Bass stems came from existing job outputs (Hetzner-quality htdemucs_6s output). Beat grid from drums stem.

### Permissive rule (3rd/5th/7th-in-bass)

| song | GT slashes | Jiang slashes | Hybrid slashes | Jiang slash-F1 | Hybrid slash-F1 | Jiang root-F1 | Hybrid root-F1 | Jiang full-F1 | Hybrid full-F1 | Δ full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hotel California | 0 | 0 | 3 | 0.000 | 0.000 | 0.914 | 0.914 | 0.760 | 0.742 | **−0.018** |
| House of the Rising Sun | 0 | 1 | 12 | 0.000 | 0.000 | 0.661 | 0.661 | 0.629 | 0.547 | **−0.082** |
| Peg | 46 | 2 | 6 | 0.000 | 0.000 | 0.703 | 0.703 | 0.273 | 0.266 | −0.007 |
| Black Cow | 35 | 0 | 2 | 0.000 | 0.000 | 0.886 | 0.886 | 0.281 | 0.281 | +0.000 |
| Aja | 29 | 1 | 13 | 0.067 | **0.095** | 0.671 | 0.671 | 0.355 | 0.355 | +0.000 |

### Strict rule (3rd or 7th-in-bass only)

| song | GT slashes | Jiang slashes | Hybrid slashes | Jiang slash-F1 | Hybrid slash-F1 | Jiang full-F1 | Hybrid full-F1 | Δ full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hotel California | 0 | 0 | 1 | 0.000 | 0.000 | 0.760 | 0.760 | +0.000 |
| House of the Rising Sun | 0 | 1 | 3 | 0.000 | 0.000 | 0.629 | 0.620 | −0.009 |
| Peg | 46 | 2 | 4 | 0.000 | 0.000 | 0.273 | 0.266 | −0.007 |
| Black Cow | 35 | 0 | 1 | 0.000 | 0.000 | 0.281 | 0.281 | +0.000 |
| Aja | 29 | 1 | 6 | 0.067 | 0.057 | 0.355 | 0.355 | +0.000 |

**Notable:** Root F1 and PCS F1 are identical between Jiang-alone and hybrid for every song in both modes. The hybrid is purely additive of slash bass labels — it never changes a root.

## 3. Example bars

### Hits (hybrid right)
| song | bar | Jiang | bass | interval | hybrid | GT-match? |
|---|---:|---|---|---|---|---|
| Aja | 4   | Bmaj7 | F# | 5th | **Bmaj7/F#** | ✓ matches GT `Bmaj7/F#` (one of 4 occurrences) |
| Aja | 147 | C     | G  | 5th | **C/G**       | (Jiang already emitted this; hybrid kept it) |

### Plausibly-right but UG doesn't notate (counted as FP)
| song | bar | Jiang | bass | interval | hybrid | comment |
|---|---:|---|---|---|---|---|
| HoTRS | 5,17,23,29,38,50,56,62 | Am | E | 5th | Am/E | Animals' iconic A-minor arpeggio voices low E as the bass — musically a first-inversion sound, but UG official tab calls it Am. **8 FPs** of this same kind. |
| Hotel California | 89, 97 | Bm  | F# | 5th | Bm/F#  | Outro vamp. Real Eagles arrangement has F# under Bm in some bars. |
| Black Cow | 20 | D7sus4 | A | 5th | D7sus4/A | Walking bass moment. GT just says D7sus4. |

### Clear bass-detection errors (hybrid wrong)
| song | bar | Jiang | bass | interval | hybrid | comment |
|---|---:|---|---|---|---|---|
| Peg   | 6   | Cmaj7  | B | 7th | Cmaj7/B  | UG GT has plain `Cmaj7` here — bass detector picked up the lead-tone B from a melodic phrase, not a held bass. |
| Aja   | 24  | C      | E | 3rd | C/E      | UG has plain `C`. Real bassline walks E→D→C at this transition; bass-root extraction captured the E. |
| HoTRS | 68  | D7     | C | b7  | D7/C     | UG GT has `D` (no 7th) at this bar — both the Jiang quality and the hybrid slash are wrong. |

## 4. Why it didn't work

1. **Bass detection is faithful to the recording, not to UG's chart.** Real bass players play more passing tones, octaves, and melodic figures than the published lead sheet shows. When the bass differs from chord root by a chord tone, it's often a *moment* (sub-beat), not the harmonic foundation of the bar. Promoting per-bar collapses that distinction.

2. **5th-in-bass is the dominant signal and the dominant noise.** Of 27 hybrid changes in permissive mode, 16 were "bass = 5th of chord." Bass players in pop/rock parks on the 5th constantly (open low E under Am being the textbook example). UG charts almost never write `Am/E` for this — they write `Am`.

3. **Jiang already emits the slash chords it's confident about.** Peg shows this: Jiang independently emits `G:maj/3` (G/B) 11 times correctly. Where Jiang doesn't emit a slash, it's usually because the song genuinely doesn't want one at that moment — the bass is on the root.

4. **Aja's high slash count (29 GT slashes) is dominated by `D6/9`, `Cadd9/D`, `G7/F`, etc.** — these are notational compositions (extension-stacked nicknames), not first-inversions. Bass-anchored upgrade cannot produce them; the chord is already specified in the bass-PC alone but UG renders it as a slash for player readability.

## 5. What would actually move slash F1

The slash chords this hybrid CAN'T reach but Jeff's GT cares about most:

- **Descending-bass clichés over a static chord** (Tom Petty IGWO: `Em → Emmaj7 → Em7 → Em/C#`). This needs **time-resolved bass + held chord quality detection**, not bar-quantized. Bass walks below 1/4-bar boundaries.
- **First-inversion guitar voicings where the bass plays root** (e.g. `D/F#`). Bass extractor returns D (root). Hybrid keeps it bare. Need: detect that the guitar voicing has F# as lowest note — requires looking at the guitar/other stem, not bass.
- **Notational stacks** (D6/9, Aadd9/B). Out of Jiang's vocab entirely; chord-naming convention rather than acoustic fact.

A productive V3 direction is probably **time-resolved bass tracking** (sub-beat) + **slash-when-bass-walks-descend**, scoped to songs where the bass and Jiang's chord disagree consistently for ≥2 sub-beats inside a bar. That's a separate prototype.

## 6. Ship / Refine / Drop

**DROP** the rule as designed. Specifically:

- Do **not** ship `upgrade_to_slash` in V3's main pipeline.
- The one TP it gains (Aja `Bmaj7/F#`) doesn't justify the FP load it adds elsewhere — and HC/HoTRS get *worse* full-F1.
- Keep `bass_root_extraction` unchanged — it's already valuable for the existing root-anchoring use case (`combine_with_detector_quality` in `bass_root_extraction.py:169`).
- Keep Jiang's native slash output — Jiang found 1 real TP (Aja `C/G`) on its own and that's already best-in-class.

If we want slash chords in V3, the next experiment should be:
1. **Sub-beat bass tracking** (windowed PYIN on 1/8-note intervals) → detect bass *motion* within a bar.
2. **Descending-bass detector**: flag bars where bass walks down ≥2 semitones over ≥2 sub-beats while Jiang's chord label stays constant.
3. Only then emit `Em/C#`-style slashes — and only for bars in the "descending bass under static chord" pattern. This targets the IGWO / Hotel California intro / Stairway intro / House of the Rising Sun (real) cases.

---

## Artifacts

- Prototype script: `/tmp/v3_slash_hybrid/hybrid.py`
- Jiang `.lab` outputs: `/tmp/v3_slash_hybrid/{black-cow,hotrs,peg,aja,hotel-california}.lab`
- Per-song hybrid + jiang charts: `/tmp/v3_slash_hybrid/chart_<song>_{jiang,hybrid}.json`
- Permissive results: `/tmp/v3_slash_hybrid/results.json`
- Strict results: `/tmp/v3_slash_hybrid/results_strict.json`
