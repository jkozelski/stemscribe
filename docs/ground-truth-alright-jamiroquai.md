# Ground Truth — Jamiroquai, "Alright"

Reference chord chart supplied by Jeff on 2026-04-23 for regression comparison against StemScriber's output.

## Structure

| Section | Lyric Open | Chord Pattern |
|---|---|---|
| Intro | [instrumental + "I need your love" repeats] | Cm7 / Gm7 / Dm7 / Am7 (x2+) |
| Verse 1 | "You, give me light" | Cm7 / Gm7 / Dm7 / Am7 |
| Chorus | "Yeah hey / Alright (right now)" | Cm7 / Gm7 / Dm7 / Am7 |
| Verse 2 | "I see your eyes / Hold the key, to my paradise" | Cm7 / Gm7 / Dm7 / Am7 |
| Chorus (x2) | same as above | same |
| Bridge | "I need your touch / I want your love, so much" | Cm7 / Gm7 / Dm7 / Am7 |
| Chorus (final) | same as above | same |

## Key observations for StemScriber QA

1. **Chord quality: all 7ths** (Cm7, Gm7, Dm7, Am7). StemScriber is outputting plain triads (Cm, Gm, Dm, Am). The `_simplify_bleed_extensions()` pass in `stem_chord_detector.py` is likely simplifying real 7th extensions out along with bleed artifacts. Investigate threshold.
2. **"I need your love" hook is Intro material**, not a separate Verse. StemScriber was labeling it Verse 1 / Verse 2.
3. **Verse 1 entrance marker: the word "You"** — Jeff's domain heuristic. First line: "You, give me light".
4. **Chorus ("Yeah hey / Alright / right now / We'll spend the night together...")** — Whisper has trouble with the "Alright" fill and renders the Chorus as starting mid-word ("right we'll spend the night together"), which caused the chord-fingerprint labeler to call it "Bridge 2" instead of "Chorus".
5. **StemScriber's current mis-mapping (as of commit `706db57`):**
   - Our "Verse 1" / "Verse 2" = Intro hook
   - Our "Chorus" = Verse (e.g., "See your eyes")
   - Our "Bridge 1" = Bridge
   - Our "Bridge 2" = Chorus

## Upstream fixes to prioritize

- Chord detector: preserve 7ths in the `_simplify_bleed_extensions` pass.
- Section labeling: the stem-RMS labeler's chord-fingerprint approach can't distinguish vocal content types on identical chord progressions. Post-launch, use lyric-phrase analysis or user-edit data to bias labeling.
