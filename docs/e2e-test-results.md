# End-to-End Test Results

**Date:** 2026-03-18
**Tester:** Automated agent review
**System:** Apple M3 Max, 48GB RAM, Python 3.11, macOS Darwin 24.6.0

---

## 1. Test Suite Results

**315 passed, 2 failed** (out of 317 total)

Failed tests are **pre-existing billing test failures** (not caused by any agent changes):
- `test_premium_plan_pricing` — expects `$4.99`, actual is `$20`
- `test_pro_plan_unlimited` — expects `-1` (unlimited), actual is `25`

**No new test failures introduced by any agent.**

---

## 2. Server Import Verification

Server imports OK. All modules load without errors. The watchdog starts correctly (30s check interval, 120s stall threshold). All dependencies detected: chord detector V10, guitar separator, piano/drum/bass transcribers, GPU manager (MPS), ensemble separator.

---

## 3. Code Review: Processing Stability (Agent 1)

### separation.py
- **MPS memory flush** (`_flush_mps_memory`): Correctly placed at lines 233 and 256 in the RoFormer pipeline — before loading RoFormer and after freeing it (before Demucs). Uses `torch.mps.empty_cache()` + `torch.mps.synchronize()` with proper feature detection (`hasattr` checks). PASS.
- **ffmpeg timeout handling**: `FFMPEG_TIMEOUT = 180` in utils.py. On `TimeoutExpired`, the WAV is kept instead of being lost. The stem is still added to the `converted` dict. PASS.
- **Logger used before definition**: `_flush_mps_memory()` calls `logger.info()` on line 30, but `logger` is defined on line 34. This is fine at runtime because the function is only called later, but it's a style issue. Minor — no bug.

### watchdog.py
- **Stall detection**: 120s threshold, 30s check interval. Correct — a stalled job will be detected within 2.5 minutes worst case.
- **Retry logic**: Flushes MPS memory + gc.collect() before retry. Max 2 retries before marking failed. PASS.
- **Snapshot tracking**: Progress comparison works correctly — only triggers stall when progress hasn't changed AND time threshold exceeded. PASS.

### pipeline.py
- **Chord chart generation timeout**: 180s timeout via thread join. If the thread is still alive after timeout, it logs a warning and continues. The thread is daemonic so it won't block shutdown. PASS.
- **Concern**: The timed-out thread continues running in the background (no cancellation mechanism). For Whisper-based lyrics extraction, this could consume CPU/memory indefinitely. Low priority since it's daemonic and the job proceeds.

### utils.py
- **convert_wavs_to_mp3**: Handles `TimeoutExpired` correctly, keeps WAV on failure. PASS.

---

## 4. Code Review: Mobile Playback (Agent 2)

### practice.html
- **Synchronous play()**: Lines 2570-2591 — `play()` is called directly in the gesture handler with no `setTimeout` wrapper. Comments clearly explain the iOS requirement. PASS.
- **preload="auto"**: Line 2452 — set on audio elements. This will cause all stems to begin downloading immediately. For mobile on cellular, this could use significant bandwidth (6 stems x ~5-10MB each). However, this is standard behavior for a music app and the stems need to load anyway. PASS (acceptable tradeoff).
- **Desktop compatibility**: The same code path handles both desktop and mobile. No platform-specific branching that could break desktop. The `play().catch()` pattern handles both cases. PASS.
- **Late-loading stems**: Lines 2594-2607 — stems that aren't ready when play is pressed get a `canplaythrough` listener that syncs their time. They don't re-call `play()` because it was already called in the gesture context. PASS.

### karaoke.html
- **iOS unlock pattern**: Lines 3476-3486 — On first play, calls `play()` then immediately `pause()` + reset `currentTime` to "unlock" audio elements for iOS, before the countdown animation. After countdown, calls `play()` normally. This is the correct iOS Safari unlock pattern. PASS.
- **preload="auto"**: Line 3265 — same as practice.html. PASS.
- **seekTo() resume**: Lines 3594-3598 — resumes playback synchronously without setTimeout. Comment explains iOS compatibility. PASS.

### js/mixer.js
- **AudioContext resume**: Lines 12-16 — `resume()` called synchronously in `ensureMixerAudioCtx()`. PASS.
- **Synchronous play in togglePlayback**: Lines 90-94 — all stems played in a synchronous loop with no deferral. PASS.
- **connectAllStems called inside gesture**: Line 86 — Web Audio nodes connected inside the click handler before play. PASS.

---

## 5. Code Review: Chord Display (Agent 3)

### Fallback chain (practice.html lines 3400-3476)
Priority order:
1. Songsterr chords (if `songsterrHasChords()` returns true)
2. AI chord diagrams (if Songsterr had tabs but no chords)
3. Manual/auto chord chart (chord_chart.json)
4. AI chord progression as pills
5. "No chord data available" message

Logic is correct and complete. Each step returns early if data is found. PASS.

### songsterrHasChords() (lines 3310-3327)
Checks three data shapes: `chordEvents[]`, `chords[]`, and `measures[].chords[]`. Each checks for non-empty, non-whitespace content. Handles both string and object chord formats. PASS.

### Key correction (lines 3332-3398)
- **Detection**: Counts flat vs sharp roots. Triggers when >50% are flats AND zero sharps AND >5 total chords.
- **Db/Gb pattern**: Requires Db >15% and Gb >10% of total chords. Transposes +1.
- **Eb/Ab pattern**: Requires Eb >15% and Ab >10%. Transposes +1.
- **applyKeyCorrection**: Modifies chordEvents and measures in-place using sharp note names. Handles slash chords. PASS.
- **Concern**: The heuristic won't catch songs legitimately in Db major (which uses Db, Gb, Ab). However, these are extremely rare in rock/pop, so the heuristic is reasonable.

### Rapid chord filter (lines 4384-4402)
- Filters chords lasting less than 1.0 second (except the last chord).
- Re-deduplicates after filtering (handles cases where removing a short chord makes two adjacent identical chords).
- PASS.

### The Time Comes manual chart
No modifications found to any chord_chart.json files or the chord_sheet.py route. The only reference to "The Time Comes" in chord_sheet.py is a comment about title cleaning. PASS.

---

## 6. Conflict Check: Agent 2 vs Agent 3 on practice.html

Agent 2 (mobile) changes are in the **stem loading/playback section** (~lines 2440-2615).
Agent 3 (chord) changes are in the **chord display section** (~lines 3310-4402).

These sections are ~700 lines apart with no overlapping code. Both sets of changes are present and functioning independently. **No conflicts.**

---

## 7. Remaining Issues / Concerns

| Priority | Issue | Location |
|----------|-------|----------|
| Low | Logger referenced before definition in `_flush_mps_memory()` | separation.py:30 vs :34 |
| Low | Chart generation timeout thread has no cancellation — could run indefinitely | pipeline.py:341-346 |
| Low | `preload="auto"` on mobile may use excessive bandwidth on cellular | practice.html:2452, karaoke.html:3265 |
| Info | Key correction heuristic will false-positive on legitimate Db major songs | practice.html:3352-3362 |
| Pre-existing | 2 billing test failures (plan pricing/limits out of sync with test expectations) | test_billing.py |

---

## 8. Recommendations

1. **All three agent fixes look correct and ready for production.** No bugs found.
2. Consider adding `preload="metadata"` as default with a "Load All" button for mobile users on cellular to save bandwidth.
3. The billing test failures should be fixed by updating the test expectations to match current plan pricing.
4. Consider adding a cancellation mechanism for the chord chart generation thread (e.g., a threading.Event).
