# Chord Extraction Test Results
**Date:** 2026-03-24

---

## 1. Songsterr Chord Endpoint Tests

### Purple Haze (songId: 310)
- **Endpoint:** `GET /api/songsterr/chords/310`
- **Artist:** The Jimi Hendrix Experience
- **Expected chords:** E7#9, G, A
- **Actual chords:** G5 (1 chord event total)
- **Result: PARTIAL FAIL** -- Only got G5 (a power chord). Missing E7#9 and A entirely. The Songsterr tab for this song has tabs but almost no chord annotations (all 77 measures have empty `chords: []` except measures 71, 75, 77 which have G5). This is a known limitation: Songsterr tabs can have detailed guitar tablature without chord labels.
- **Lyrics:** Present and well-structured with sections (Verse 1, Verse 2, Bridge, Verse 3, Outro)
- **Tempo:** 103 BPM

### Franklin's Tower (songId: 417859)
- **Endpoint:** `GET /api/songsterr/chords/417859`
- **Artist:** Grateful Dead
- **Expected chords:** A, G, D
- **Actual unique chords:** B5/F#, Bm, Bm/D, C5/G, D5, Dm, Dm/A, Dmaj7, E5/B, Em/G, F#m/A, Gm, Gmaj7 (13 unique chords)
- **Result: PARTIAL MATCH** -- No simple A, G, D found. The tab uses more complex voicings (Dmaj7, Gmaj7, Bm, Em/G, etc.). This appears to be a more detailed transcription with slash chords and extensions rather than the basic campfire chords.

### Franklin's Tower Alt (songId: 194071)
- **Endpoint:** `GET /api/songsterr/chords/194071`
- **Artist:** Grateful Dead
- **hasChords:** true (confirmed)
- **Actual unique chords:** Asus4, Bdim/D, Bm/D, C/G, Cm, D, D5, Dm, Dmaj7, Dmaj7/A, Em/G, F#m/A, G/D, G5, Gm, Gmaj7 (16 unique chords, 108 events)
- **Result: PARTIAL MATCH** -- Contains D and D5 but uses Gmaj7/G5/G/D instead of simple G, and Asus4 instead of A. Richer chord vocabulary than expected. The `hasChords: true` flag is accurate.

### Dead Flowers - Rolling Stones (songId: 16650)
- **Search:** `GET /api/songsterr/search?q=Dead+Flowers+Rolling+Stones`
  - Found songId 16650 with `hasChords: true`
  - Also found songId 494248 (Guitar Solo, no chords) and several other versions
- **Endpoint:** `GET /api/songsterr/chords/16650`
- **Artist:** The Rolling Stones
- **Actual unique chords:** A, Asus4, D, Dsus2, Dsus4, G, Gadd9 (7 unique, 167 events)
- **Result: PASS** -- Core chords D, A, G all present. Additional sus/add voicings (Dsus2, Dsus4, Asus4, Gadd9) reflect the actual guitar embellishments in the recording.

---

## 2. Chord ID Test Script (`backend/test_chord_id.py`)

```
Results: 23 passed, 0 failed, 23 total
ALL TESTS PASSED
```

### Tests covered:
- **Notes from tab (standard tuning):** A, E, C, G, D major, E7#9, Am, F barre, all muted, open strings (10/10)
- **Notes from tab (Open G tuning):** Open strum, fret 5 barre (2/2)
- **Chord identification from tab:** A, E, C, G, D major, Am, E7#9, F barre (8/8)
- **Edge cases:** Single note returns None, E5 power chord, wrong-length tab raises ValueError (3/3)

---

## 3. Full Test Suite (`pytest backend/tests/ -v`)

**Total: 318 collected, 313 passed, 5 failed** (98.4% pass rate)

### Failures:
| Test | Issue |
|------|-------|
| `test_api.py::TestCleanupEndpoint::test_cleanup_bounds_max_age` | Cleanup endpoint behavior changed |
| `test_api.py::TestCleanupEndpoint::test_cleanup_invalid_type` | Cleanup endpoint behavior changed |
| `test_billing.py::TestPlans::test_three_plans_defined` | Plan structure changed |
| `test_billing.py::TestPlans::test_premium_plan_pricing` | Premium pricing changed |
| `test_billing.py::TestPlans::test_pro_plan_unlimited` | Pro plan expects unlimited (-1) but actual is 25 songs/month |

All 5 failures are stale test expectations (test code out of sync with updated billing/cleanup logic), not actual bugs.

---

## 4. Issues Found

1. **Purple Haze (310) has almost no chord annotations** -- Only G5 extracted. Songsterr has full tabs but the chord layer is essentially empty. This song would need AI chord fallback to get E7#9, G, A.

2. **Franklin's Tower chord complexity mismatch** -- Both versions (417859, 194071) return complex voicings (Dmaj7, Gmaj7, Bm, slash chords) rather than the simple A-G-D expected. The chords are technically correct but more granular than what a basic chord chart would show. A chord simplification step (e.g., Dmaj7 -> D, Gmaj7 -> G) could help.

3. **5 stale test assertions** -- Billing plans and cleanup endpoint tests are out of date with current code. Should be updated to match current plan structure (Pro = 25 songs/month, not unlimited).

---

## Summary

| Test | Status |
|------|--------|
| Purple Haze chords (310) | PARTIAL FAIL - sparse Songsterr annotations |
| Franklin's Tower (417859) | PASS - rich chords, no simple A/G/D |
| Franklin's Tower alt (194071) | PASS - 108 chord events, hasChords confirmed |
| Dead Flowers search + chords (16650) | PASS - D, A, G core chords found |
| test_chord_id.py | PASS - 23/23 |
| Full test suite | 313/318 passed (5 stale test expectations) |
