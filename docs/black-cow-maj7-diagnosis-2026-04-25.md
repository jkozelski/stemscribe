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

## Files

- Diagnostic script: `/tmp/diag_blackcow_maj7.py` on local + VPS
- Last Black Cow chord chart: `~/stemscribe/docs/audit-2026-04-25/blackcow-postfix-v4.json`
- Source audio still on VPS: `/opt/stemscribe/uploads/c21719a4-5c26-4785-9d96-81839b5212e3/Steely_Dan_-_Black_Cow.mp3`
