# Black Cow maj7 detection — diagnosis (deferred fix)

**Date:** 2026-04-25 evening session
**Status:** Investigated, fix NOT shipped. Deferred for careful handling.

## What we know

Black Cow's ground truth is a jazz-pop progression in A major / F# minor:
- Verse: Amaj7 / G#m7 / F#m7 / Bm7 / E7
- Chorus: Amaj7 / Dmaj7

After today's family-aware + per-root + m3-priority fixes, Black Cow's chord
chart is `A9 (37 bars), D (25), C9 (17), E (15), D# (7)` — **57 of 116 bars
have extensions but they're the wrong family** (dominant 9 instead of maj7).

## Diagnostic

Ran `diag_blackcow_maj7.py` (saved at `/tmp/`) which traces interval
detection per segment. Aggregate over 222 bass-detected segments:

| Interval signal | Count | % |
|-----------------|-------|---|
| maj7 ONLY (11)  | 38    | 17.1% |
| b7 ONLY (10)    | 107   | 48.2% |
| BOTH ambiguous  | 35    | 15.8% |
| NEITHER         | 42    | 18.9% |

**Interval 11 (the maj7 note, e.g. G# from A) IS being detected** — just less
often than interval 10 (b7, e.g. G).

## Why a simple post-processing fix isn't enough

Two compounding bugs:

1. **Wrong bass roots on opening bars.** Sample first 10 segments show
   `bass=C` for ~9 of 10 segments. Black Cow opens on A. The bass stem
   detection is identifying C as the dominant pitch class even though
   the bassist is playing A. This may be due to A's overtones colliding
   with C in basic_pitch's piano-range view, or stem-separation bleed
   from another instrument.

2. **Maj7 vs dom7 confusion on correct-root bars.** Even on the bars
   where bass is correctly identified as A, interval 10 (G) is detected
   2× as often as interval 11 (G#). The detector picks dominant-9 over
   maj9 because the b7 signal dominates.

Bug 1 affects 40-60% of bars. Bug 2 affects the rest. A post-processing
maj7 fix only addresses bug 2.

## Risks of shipping a key-aware maj7 promotion now

- Could degrade Cosmic Girl (min7s correctly detected — those bars
  also show interval 11 sometimes when the m9 sounds, but we want min7
  not maj7)
- Could break Aja's coherent min7/min9 output (224 bars of clean jazz)
- Needs careful test coverage with multiple positive and negative cases
- Late evening, solo dev, fatigued

## Plan for next session

1. **First fix bass-root detection on Black Cow** — this is upstream and
   affects the whole pipeline. Diagnostic needed: why does bass detection
   pick C on a song where the bass plays A?
2. **Then add key-aware maj7 promotion** — when detected key is major
   AND a bar's bass root matches I or IV of the key AND interval 11 is
   in pitch_classes at all, prefer maj7 over dom7.
3. **Regression suite** — tests must cover:
   - Cosmic Girl: m7s preserved (don't flip to maj7)
   - Alright: m7/m9 preserved
   - Aja: dense Bm jazz preserved
   - Black Cow: maj7s now appear on Amaj7/Dmaj7 sections
   - A blues song (if available): dominant 7ths preserved on V chord

Estimated work: 3-4 hours including bass-root investigation.

---

## UPDATE 2026-04-26: bass-root WAS NOT broken — it's slash-chord detection

Re-ran pyin diagnostic over the full 308s of Black Cow's bass stem.
Pitch-class distribution: A 22.3%, D 16.2%, E 16.2%, C 11.6%, ... that's
EXACTLY what you'd expect for a song in A major where the bassist plays
the famous descending chromatic intro line (A-G-F#-E-D-C-B-A). The "wrong"
C in early segments was correct — the bassist really does play C as a
passing tone.

**The real bug is slash-chord detection.** When the bassist walks down
through the scale beneath a sustained chord (Amaj7 with bass A→G→F#→E
→D→C→B→A), the harmony stays Amaj7 but the bass note changes per beat.
The detector currently treats bass=C as "chord root is C" → outputs C9.
The musically correct output is `Amaj7/C` (slash chord: Amaj7 chord with
C in bass).

### Slash-chord detection approach

Compare bars over a moving window:
- If `bass_root` changes from one bar to the next BUT `pitch_classes`
  (the harmony notes detected from guitar/piano stems) are mostly the
  same, then the harmony is stable and only the bass moved → slash chord
- Hold the previous chord's quality, change only the slash-bass label
- Output: `Amaj7/C` instead of `C9`

This requires multi-bar analysis in `combine_with_detector_quality` (or a
new pass after it). It's a meaningfully larger architectural change than
yesterday's per-root family override — touches the bar_grid construction
loop, not just smoothing.

### Revised priority

Slash-chord detection is the bigger fix and probably the right one. But
it's harder than the maj7 promotion. Realistic order:
1. Try the key-aware maj7 promotion FIRST (smaller, may improve the
   non-walk-bass bars on Black Cow). Test against Cosmic Girl/Alright/Aja.
2. Slash-chord detection separately, larger scope.
3. Regression suite covers both.

### What didn't I deploy today

Stopped after the diagnostic — the slash-chord finding changes the fix
shape, and the right move is to think it through, not ship blind.
Marketing draft + sections-empty bug investigation are next instead.

---

## UPDATE 2026-04-26 (afternoon): maj7 promotion shipped, didn't fire

Shipped commit `bb64448` — `promote_diatonic_maj7()` in
`bass_root_extraction.py`. Promotes I/IV dom-7 chords to maj-7 when key
is major and song doesn't look like a 12-bar blues. 17 tests cover
positive cases + 4 categories of regression protection.

**Verification result:**
- Black Cow: 0 promotions fired. Detected key was "D" (wrong — Black
  Cow is in A major). With key=D, A9 is the V chord, and my safety
  check correctly skips V.
- Alright (regression): 92/97 extensions, 0 maj-family bars, 0
  promoted. Pattern unchanged. Fix is safe in production.

**Real bottleneck for Black Cow:** Krumhansl-Kessler key detection in
`detect_key_from_chords()` is fooled by the slash-chord moments. Black
Cow's chord weights: A=42, D=25, E=15, C=16. C is high because slash-
chord bars output as "C9" (when really Amaj7/C). C as bIII isn't part
of A major, so the K-K correlation drifts toward D major.

**Either fix unblocks Black Cow:**
1. **Slash-chord detection** (planned) — fixes the C9 → Amaj7/C
   relabeling. As a side effect, fixes key detection too because the
   noise C-root chords disappear.
2. **Key detection over pitch-classes** — instead of weighting only
   chord roots, also weight the implied notes of each chord (M3, 5,
   etc.). A song with A I, D IV, E V chords should add A/C#/E (from A
   major), D/F#/A (from D), E/G#/B (from E) to the histogram. The
   resulting profile would clearly fit A major even with passing C
   chords.

Either fix is ~2-4 hours of careful work. Slash-chord detection is the
cleaner architectural fix; key-detection-over-pitch-classes is faster
to ship.

## Files

- Diagnostic script: `/tmp/diag_blackcow_maj7.py` on local + VPS
- Last Black Cow chord chart: `~/stemscribe/docs/audit-2026-04-25/blackcow-postfix-v4.json`
- Source audio still on VPS: `/opt/stemscribe/uploads/c21719a4-5c26-4785-9d96-81839b5212e3/Steely_Dan_-_Black_Cow.mp3`
