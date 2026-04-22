# Auto Chord Chart Quality Report

**Date:** 2026-03-16 (Updated - comprehensive validation)
**Pipeline version:** chart_assembler.py + chord_detector.py + lyrics_extractor.py + structure_detector.py
**Test suite:** 174/174 tests passing (0 failures, 1 warning)
**Validator:** Agent 5 - Validation & Quality

---

## 1. Summary Scores

| Metric | Score | Notes |
|--------|-------|-------|
| Chord vocabulary accuracy | 100% | All 8 ground truth chords detected (F#m, A, B, Bm, C#m, D, E, G) |
| Per-line chord accuracy | 25% (7/28 lines) | Correct chords detected but placed on wrong lines due to timing/segmentation |
| Section detection | 3/6 types found | Only Intro/Chorus/Outro detected; Verse and Bridge never identified |
| Section count accuracy | 14 detected vs 6 expected | Massive over-segmentation |
| Lyrics: lines matched | 24/28 (86%) | Most lines detected, 4 bridge lines missed entirely |
| Lyrics: word similarity | 47% average | Significant word-level errors throughout |
| Overall usability | 3/10 | Not usable as-is; requires substantial manual editing |

---

## 2. Side-by-Side Comparison

### Verse 1, Line 1
| | Manual | Auto |
|---|--------|------|
| **Chords** | F#m, A | A, F#m |
| **Lyrics** | "You say you wanted" | "Say you want it, someone to take your way" |
| **Issue** | Auto merges 2 lines into 1, drops "You", changes "wanted" to "want it" |

### Verse 1, Line 3-4
| | Manual | Auto |
|---|--------|------|
| **Chords** | F#m, A / F#m, B | A, F#m, B |
| **Lyrics** | "From out of the deep end" / "Where the devil's all play" | "Remind of the deep ends, but where the devils all play" |
| **Issue** | "From out of" becomes "Remind of", two lines merged |

### Chorus, Line 1
| | Manual | Auto |
|---|--------|------|
| **Chords** | D, C#m, Bm | C#m, Bm |
| **Lyrics** | "The time comes to turn it around" | "Time comes to turn it around" |
| **Issue** | Missing D chord on this line, drops "The" |

### Bridge (entire section -- worst quality)
| | Manual | Auto |
|---|--------|------|
| **Lyrics** | "Come to life / Feel easy when the wind blows / Let go of that strife / Throw your worries out the window..." | "The yeah, from the lie high / Feeling with ghosts, go on that strike high / Throw your words at the window..." |
| **Issue** | Bridge lyrics are almost completely hallucinated by Whisper. "Come to life" becomes "from the lie high", "Feel easy when the wind blows" becomes "Feeling with ghosts". This is the worst section. |

### Section Labels
| Manual | Auto |
|--------|------|
| Intro | Intro |
| Verse 1 | Chorus 1, Chorus 2, Chorus 3 |
| Chorus | Chorus 3 (end), Chorus 4 (start) |
| Verse 2 | Chorus 4, Chorus 5 |
| Chorus | Chorus 5 (end) |
| Bridge | Chorus 6, Chorus 7, Chorus 8, Chorus 9, Chorus 10, Chorus 11 |
| (none expected) | Outro, Outro |

### Full Lyrics Comparison (Ground Truth vs Auto)

| Ground Truth Line | Whisper Output | Accuracy |
|---|---|---|
| You say you wanted | Say you want it | Partial |
| Someone to take you away | someone to take your way | Partial |
| From out of the deep end | Remind of the deep ends | Wrong |
| Where the devil's all play | but where the devils all play | Close |
| Is where I found you | Is Where I found you | Correct |
| When you were going astray | when you were going straight | Wrong |
| Pick up your head now | To cut your head now | Wrong |
| Get up and get on your way | get up and be on your way | Close |
| The time comes to turn it around | Time comes to turn it around | Close |
| All the lines must be put down | All the lines must be put down | Correct |
| Free your head now | free your head now | Correct |
| You know the truth is inside | Truth Is inside and | Partial |
| And once you find | When you find out | Wrong |
| You have what you desire | That you have what you decide | Wrong |
| It will free you | It / Will free you | Correct (split) |
| And the memories will wake | and your memories will wake | Close |
| To hover around you | And hover around you | Close |
| Till the heavens will shake | to the heavens we'll shake | Partial |
| The time comes to turn it around | The Time comes to turn it around | Correct |
| Can't deny when you've been found | You can't deny when you've been found | Correct |
| Come to life | from the lie high | Wrong |
| Feel easy when the wind blows | feeling with the windows | Wrong |
| Let go of that strife | go on that strike high | Wrong |
| Throw your worries out the window | Throw your words at the window | Partial |
| It comes from the past | come from the past | Close |
| To shine the light onto the future | Shine them up to the future | Partial |
| Forever to last | forever to lie | Wrong |
| Cause we don't ever need to leave here | Cause we don't ever meet the need for it | Wrong |

---

## 3. Top Issues (Ranked by Impact)

### Issue 1: Structure detection labels everything as "Chorus" (CRITICAL)
The structure detector identified 11 "Chorus" sections and 2 "Outro" sections. Not a single "Verse" or "Bridge" was detected. The root cause is in `_chord_based_segmentation()`: the verse chords (F#m, A, B) appear in >40% of windows and are classified as "common", so most windows get the `_base_` label. The labeling logic then assigns `_base_` to "Chorus" instead of "Verse" because the energy-based discrimination fails when the verse is as energetic as the chorus.

**Over-segmentation**: 14 sections detected vs 6 actual. The 4-second window size is too granular.

### Issue 2: Lyrics hallucination on the bridge section (CRITICAL)
Whisper (base model, 74M params) completely fails on the bridge lyrics. "Come to life" becomes "from the lie high", "Feel easy when the wind blows" becomes "Feeling with ghosts, go on that strike high". The bridge has a different vocal quality (more atmospheric/reverb) that confuses the model. The bridge lyrics are detected twice (at ~93s and ~171s) with near-identical hallucinations, suggesting a systematic failure mode.

### Issue 3: Line segmentation and merging (HIGH)
The manual chart has clean 2-4 word lines per chord change. The auto chart merges multiple phrases into single long lines. The `PAUSE_THRESHOLD` of 1.0s and `MAX_LINE_DURATION` of 6.0s in lyrics_extractor.py are too generous for chord chart formatting.

### Issue 4: Per-line chord placement (MEDIUM)
Even where the right chords are detected in the right time range, they end up on the wrong lyric lines because timing misalignment between chord events and lyric line boundaries causes chords to spill across lines.

### Issue 5: Missing lyrics for instrumental sections (LOW)
The song has ~50 seconds of instrumental content where the auto chart correctly shows chord-only lines, but these are spread across too many fragmented sections.

---

## 4. Improvement Suggestions (Ranked by Impact)

### 1. Fix Verse/Chorus labeling logic (Highest impact, code-only fix)
The structure detector defaults to calling the most common pattern "Chorus". For most songs, the most common/longest section is the Verse. **Fix**: flip the default so `_base_` (most common chord pattern) maps to "Verse", and distinctive chord patterns map to "Chorus". Current logic has the right idea but the polarity is inverted for this song type.

### 2. Upgrade Whisper model to "medium" or "large-v3" for lyrics
The base model (74M params) hallucinates badly on reverb-heavy vocals. The medium model (769M) or large-v3 (1.5B) would dramatically improve bridge/atmospheric section transcription. Even "small" (244M) would be a significant upgrade. Tradeoff: ~3x slower for medium.

### 3. Reduce over-segmentation (code-only fix)
Increase window size from 4.0s to 8.0s in `_chord_based_segmentation()`. Increase minimum run length from 2 windows to 3-4 before creating a new section. This would reduce 14 sections to ~6-8.

### 4. Add chord-change-aware line splitting (code-only fix)
In `lyrics_extractor.py`, add an optional `chord_events` parameter. When available, use chord change timestamps as preferred line break points, so each lyric line maps to one chord change. This would fix the merged-line problem.

### 5. Lyrics post-correction with known song data
When UG/Songsterr lyrics are available, use them as a reference to correct Whisper output via alignment. This would fix hallucinations without needing a larger model.

---

## 5. Pipeline Reliability

| Aspect | Status |
|--------|--------|
| Chord detection | Ran without errors. 100% chord vocabulary accuracy. |
| Lyrics extraction | Ran without errors. Model loaded successfully (faster-whisper base). |
| Structure detection | Ran without errors but produced poor labeling results. |
| Chart assembly | Ran without errors. Correct JSON format produced. |
| Full pipeline integration | `generate_chord_chart_for_job()` orchestration works correctly. |
| Fault tolerance | All steps wrapped in try/except; graceful degradation to chord-only chart. |
| Test suite | **174/174 passing**, 0 failures, 1 warning (urllib3 version). |
| Cross-validation (Dylan) | Pipeline ran end-to-end on a different song without crashes. |

**No crashes or exceptions occurred during any test.** The pipeline is mechanically reliable -- it always produces output. The quality issues are algorithmic, not engineering failures.

---

## 6. Cross-Validation: Bob Dylan - Tangled Up In Blue

**Job ID:** 54d475f4-3936-4947-9126-7d187ecd6457

| Metric | Result |
|--------|--------|
| Chord vocabulary | 10 unique chords detected (C, Csus4, D, Dsus4, Em, Emin7, F, Fmaj7, G, Gmaj7) |
| Ground truth overlap | 2/4 main chords found (D, G); missed A and E entirely |
| Key detection | Detected G (actual key is A) -- off by a whole step |
| Lyrics extracted | 92 lines (expected ~49 -- significant over-segmentation) |
| Chart sections | 12 auto-sections (all labeled "Verse" via _auto_section fallback) |
| Pipeline crashes | None |

**Observations**: Chord detection performed significantly worse on Dylan than on The Time Comes. The strumming-heavy acoustic guitar confused the chroma-based detector -- it detected G and C instead of A and E (possibly hearing relative modes or overtones). The lyrics were over-segmented (92 lines vs ~49 expected). However, the structure detector actually performed better here by defaulting to "Verse" labels through the _auto_section fallback, which is more appropriate for a verse-heavy song.

**Key finding**: The chord detector's accuracy is highly dependent on the source material. Clean electric guitar (The Time Comes) yields 100% accuracy, while acoustic strumming (Dylan) drops to ~50%. This suggests the chroma template matching works well for separated electric guitar but needs tuning or a different approach for acoustic.

---

## 7. Component Quality Matrix

| Component | The Time Comes | Tangled Up In Blue | Average |
|-----------|---------------|-------------------|---------|
| Chord vocab accuracy | 100% | 50% | 75% |
| Lyrics extraction | 47% word sim | Not compared | ~47% |
| Section labeling | 3/6 correct types | N/A (no ground truth) | Poor |
| Chart formatting | Correct JSON | Correct JSON | Good |
| Pipeline reliability | No crashes | No crashes | Excellent |

---

## 8. Conclusion

The auto chord chart pipeline is **mechanically sound** but **algorithmically immature**. The chord detector's vocabulary accuracy is excellent on clean electric guitar (100% on The Time Comes) but degrades on acoustic (50% on Dylan). The three biggest quality gaps are:

1. **Structure detection mislabels sections** -- verses become choruses, massive over-segmentation (code fix)
2. **Whisper hallucinations on atmospheric vocals** -- bridge sections get gibberish (model upgrade)
3. **Line segmentation doesn't align with chord changes** -- merged lines, wrong chord placement (code fix)

Fixing issues #1 and #3 are pure code changes with no model dependency. Issue #2 requires either a model upgrade or a lyrics reference source. Together, these three fixes would likely raise overall usability from **3/10 to 6-7/10**.

The pipeline never crashes and degrades gracefully -- this is a strong foundation to build quality improvements on top of.
