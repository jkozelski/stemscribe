# V3 Agent 2 — Bass-stem-aware post-processing for ACE
**Date:** 2026-05-13
**Author:** v3 swarm agent #2
**Time-box:** 60 minutes

## Question
Can a bass-stem-aware post-processor collapse ACE's hallucinated slash/extension
chords on Hotel California (the one cohort regression vs current prod) without
hurting the other songs ACE wins?

## TL;DR — VERDICT

**Drop both rules.** Net root-F1 change across the 4 songs is **0.000**. Neither
rule helps Hotel California meaningfully. Rule A almost never fires (because
ACE's slash chords are actually bass-truthful, not hallucinations). Rule B *hurts*
Hotel California by destroying the F#7 chords UG also lists.

**Headline finding:** the original hypothesis ("ACE hallucinates G/D when bass is
on G") is **falsified by the data**. On Hotel California, every single one of
ACE's 13 slash-chord events was confirmed by the bass-stem: the bass *is*
playing the 5th under those G chords. ACE is hearing the correct musical
relationship; the disagreement vs UG's bare `G` is a notational-convention
gap (UG collapses voicings, ACE doesn't), not a detection error.

---

## The two rules (code)

Both live in `/tmp/v3pp/bass_postproc.py`. Sampler runs once per song
(`librosa.pyin` + `chroma_cqt` on the bass stem).

### Rule A — slash drop
```python
def apply_rule_a_slash_drop(events, sampler, debug=None):
    out = []
    for ev in events:
        root, qual, bass = parse_chord(ev["chord"])
        if not bass or root is None:
            out.append(ev); continue
        pc, conf = sampler.pc_at(ev["start_time"], ev["end_time"])
        slash_pc = NOTE_PC[bass]; root_pc = NOTE_PC[root]
        if pc is None:            # no voiced bass -> drop slash (no evidence)
            new = f"{root}{qual}"
        elif pc == slash_pc:      # bass-stem confirms the slash -> keep
            new = ev["chord"]
        elif pc == root_pc:       # bass is on the root -> drop slash
            new = f"{root}{qual}"
        else:                     # bass disagrees with both -> drop slash
            new = f"{root}{qual}"
        ev2 = dict(ev); ev2["chord"] = new
        out.append(ev2)
    return out
```

### Rule B — extension collapse (RISKY)
```python
def apply_rule_b_extension_collapse(events, sampler, threshold=0.07):
    out = []
    for ev in events:
        root, qual, bass = parse_chord(ev["chord"])
        if not is_seventh_chord(qual):     # only mess with 7/9/11/13/maj7
            out.append(ev); continue
        seventh_pc = (NOTE_PC[root] + (11 if 'maj' in qual.lower() else 10)) % 12
        energy = sampler.chroma_energy_at(ev["start_time"], ev["end_time"], seventh_pc)
        if energy < threshold:
            triad = quality_root_to_triad(qual)
            new = f"{root}{triad}" + (f"/{bass}" if bass else "")
        else:
            new = ev["chord"]
        ev2 = dict(ev); ev2["chord"] = new
        out.append(ev2)
    return out
```

---

## Per-song before/after

Scored against UG ground truth via `audit/score_chord_chart.py` (bag-of-chords
F1). All ACE labs are from the consonance-ACE outputs in `/tmp/ace_outputs/`
except IGWO, which was generated via Jiang chord-CNN-LSTM today (caveat below).
Bass stems from `demucs --two-stems=bass` (htdemucs).

| Song | Variant | root F1 | r+q F1 | full F1 | det slash # |
|---|---|---|---|---|---|
| **Hotel California** | ACE-default | 0.805 | 0.745 | 0.745 | 13 |
| | +Rule A | 0.805 | 0.745 | 0.745 | 13 |
| | +Rule B | 0.805 | **0.718** | 0.718 | 13 |
| | +Rule A+B | 0.805 | 0.718 | 0.718 | 13 |
| **Paint It Black** | ACE-default | 0.746 | 0.541 | 0.533 | 1 |
| | +Rule A | 0.746 | 0.541 | **0.541** | 0 |
| | +Rule B | 0.746 | 0.541 | 0.533 | 1 |
| | +Rule A+B | 0.746 | 0.541 | 0.541 | 0 |
| **Mary Jane's** | ACE-default | 0.888 | 0.853 | 0.853 | 0 |
| | +Rule A | 0.888 | 0.853 | 0.853 | 0 |
| | +Rule B | 0.888 | **0.860** | 0.860 | 0 |
| | +Rule A+B | 0.888 | 0.860 | 0.860 | 0 |
| **IGWO** (Jiang lab) | default | 0.770 | 0.762 | 0.762 | 9 |
| | +Rule A | 0.770 | 0.762 | 0.762 | 9 |
| | +Rule B | 0.770 | 0.762 | 0.762 | 9 |
| | +Rule A+B | 0.770 | 0.762 | 0.762 | 9 |

**Net root F1 delta across the 4 songs: +0.000 (Rule A), +0.000 (Rule B),
+0.000 (both).** All movement is in r+q / full F1, where Rule B hurts Hotel
California (−0.027) and barely helps Mary Jane's (+0.007).

### Slash chord preservation
Does Rule A wrongly remove slash chords UG actually has? **No.** Hotel California
GT has zero slashes; Paint It Black GT has zero; Mary Jane's GT has zero. IGWO
GT has 15 slashes (`Em/C#`, `C/G`, `Am/F#`) and the Jiang detector emitted 9
slashes (`C/G`, `D/A`, `G/D`) — Rule A preserved **all 9** because the bass
stem confirmed every one of them. So Rule A is *theoretically* safe in this
direction too.

---

## Why the hypothesis failed

I dumped the per-event Rule A debug log for Hotel California
(`/tmp/v3pp/eagles__hotel-california__ruleA_debug.txt`). Every single one of
ACE's 13 slash chords gets `keep-confirmed`:

```
[60.00-61.88]  A/E    -> A/E    (keep-confirmed, bass-pc=E, conf=0.80)
[93.11-94.32]  G/D    -> G/D    (keep-confirmed, bass-pc=D, conf=0.82)
[100.00-100.83] Em7/B -> Em7/B  (keep-confirmed, bass-pc=B, conf=0.65)
[144.85-146.33] G/D   -> G/D    (keep-confirmed, bass-pc=D, conf=0.83)
[171.53-172.32] Gmaj7/D -> Gmaj7/D (keep-confirmed, bass-pc=D, conf=0.68)
[192.78-193.46] Dmaj7/C# -> Dmaj7/C# (keep-confirmed, bass-pc=C#, conf=0.97)
[249.42-250.28] G/D   -> G/D    (keep-confirmed, bass-pc=D, conf=0.74)
... (all 13 confirmed)
```

The bass really *is* playing D under those G chords (it's the famous descending
bassline B → A → G/D → D → Em → G/B → F# of the chorus). ACE is correct
musically. UG just doesn't bother writing the slash because rhythm-guitar
players ignore the bass voicing. Same story for IGWO — every one of `C/G`,
`D/A`, `G/D` is confirmed by the bass stem.

So **Rule A is essentially a no-op on songs where the detector and bass agree
on the slash bass** — which appears to be most of the time, because both are
listening to the same audio source.

## Why Rule B regresses Hotel California

Rule B looks for energy at the 7th-of-root pitch class in the **bass-stem
chroma**. The problem: in most rock arrangements, the 7th of a m7 chord is
played by the *guitar/keyboard/vocals*, not the bass. The bass plays root +
maybe 5th. So Rule B sees ~0.02–0.06 chroma energy at the 7th PC for legit
Em7/F#7 events and collapses them.

Example (`/tmp/v3pp/eagles__hotel-california__ruleB_debug.txt`):
```
[29.42-32.65] F#7 -> F#  (collapse-7th, energy=0.021)   # F#7 is in GT!
[97.56-100.00] Em7 -> Em (collapse-7th, energy=0.039)   # Em7 has 12 events in GT
[171.53-172.32] Gmaj7/D -> G/D (collapse-7th, energy=0.043)
```

Mary Jane's got a tiny win (+0.007 r+q F1) because the song really is straight
triads and the only 3 7th-emissions ACE made (2× D7, 1× Em7) all collapsed
correctly. But the same logic destroys Hotel California's F#7s.

## Unexpected behavior

- **Rule B on Aja-style jazz would be catastrophic.** I didn't run Aja in this
  bake-off (no ACE lab for it), but extrapolating: Aja has 226 extension events
  in GT, the bass stem won't carry the 7th, and Rule B would strip all of them.
  This rule cannot ship anywhere near jazz songs.
- **Jiang chord-CNN-LSTM on IGWO emits zero 7th-chords** despite UG GT having
  24 events (12× Em7 + 12× Emmaj7). The detector under-detects extensions on
  IGWO entirely — a separate problem that a "collapse extensions" rule cannot
  fix (it can only delete extensions, not add them).
- **The ACE-emitted slash bass-PCs aren't hallucinations.** The single false
  positive (Paint It Black's spurious `Em/B` at 91.37s) is dropped correctly
  by Rule A — it gained `full_f1` 0.533 → 0.541. So Rule A is *correct* but its
  positive-impact surface area is tiny.

---

## Verdict (rationale)

- **Drop Rule A.** Logically sound; almost never fires; net F1 delta = 0.000.
  Maintenance cost > benefit.
- **Drop Rule B.** Actively harms Hotel California (the song this whole
  experiment was aimed at). Even where it helps (Mary Jane's), the +0.007 gain
  is below noise. Will destroy jazz cohort if shipped.
- **Real fix for Hotel California**: this isn't a bass-stem problem. ACE's
  decorated chords (G/D, Em7) are *correct hearings* of the audio. The
  Hotel-California regression vs current prod is a **scoring artifact** of
  comparing decorated ACE output to bare UG chords. If we want better Hotel
  Cal numbers, the right move is either:
  1. **Strip slash chords AND 7ths from the bag-of-chords scorer** before
     comparing to UG. This is a scorer change, not a detector change.
  2. **A separate "UG-normalize" post-processor** that drops *all* slash
     chords and reduces m7→m / maj7→maj unconditionally. No bass stem needed.
     But this would also hurt every song where UG actually wrote the slash
     (IGWO loses 9 correct slash matches).

The deeper finding worth flagging to Jeff: **ACE is doing the chord-detection
job correctly on Hotel California**. The 0.805 root F1 is bag-of-chords ceiling
without normalizing notation differences. A useful next sprint would be a
"UG-normalize → score" pass to measure ACE's *musically correct* F1 vs its
*UG-literal* F1; current prod's higher number on Hotel Cal is probably just
because the simpler detector happens to emit UG-style bare triads, not because
it's hearing the song better.

---

## Artifacts

- Code: `/tmp/v3pp/bass_postproc.py` (rules), `/tmp/v3pp/run_bake.py` (driver)
- Bass stems: `/tmp/v3pp/bass/htdemucs/<song>/bass.wav`
- Results: `/tmp/v3pp/scores/results.json`
- Per-song rule debug logs: `/tmp/v3pp/<slug>__rule{A,B}_debug.txt`
- IGWO Jiang lab (generated today): `/tmp/v3pp/tom-petty__into-the-great-wide-open.lab`
