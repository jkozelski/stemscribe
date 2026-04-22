# Bug Fixes Day 3 — Beta Blocker Fixes
**Date:** 2026-03-19

## Bug 1: AlphaTab crash on GP files (P1) -- FIXED
**Problem:** AlphaTab throws "RangeError: Invalid typed array length" on some Guitar Pro files (e.g. Bertha), killing the entire page.

**Fix:**
- Moved the `alphaTabApi.error.on()` handler to fire early in init, before `scoreLoaded`
- Both the error handler and outer catch block now show a friendly message: "Tab view unavailable for this song. Use Chords view instead."
- Auto-switches to Chords view after 500ms so the user isn't stranded
- The crash no longer kills the rest of the page functionality

**File:** `frontend/practice.html` (initAlphaTab function)

---

## Bug 2: YouTube channel names as artist metadata (P2) -- FIXED
**Problem:** Songs show YouTube uploader as artist (e.g. "Sarah N. Dipity" for Bertha, "oscarin" for Black Dog).

**Fix:** Updated `backend/services/downloader.py` to use a priority chain:
1. YouTube Music `artist` field (official metadata) -- highest priority
2. YouTube Music `track` field for title (if available)
3. Parse "Artist - Title" from video title (supports ` - `, ` | `, `: `, em/en dashes)
4. Fall back to uploader/channel name only as last resort

**Example:** "Grateful Dead - Bertha (1970)" now extracts artist="Grateful Dead", title="Bertha (1970)"

**File:** `backend/services/downloader.py`

---

## Bug 3: Missing CHORD_SHAPES (P2) -- FIXED
**Problem:** Dsus2, Asus4, Gadd9, Dsus4 and other common chords show "no diagram" in the chord chart.

**Fix:** Added 36 new chord shapes to CHORD_SHAPES:
- Sus2/Sus4 for all natural notes: Dsus2, Dsus4, Asus2, Asus4, Esus4, Esus2, Csus2, Csus4, Gsus4, Gsus2, Fsus2, Fsus4, Bsus4, Bsus2
- Add chords: Gadd9, Cadd9, Dadd9, Eadd9, Aadd9, Fadd9
- 9th chords: A9, D9, E9, G9, C9
- Enharmonic sus variants: F#sus4, F#sus2, Gbsus4, Gbsus2, C#sus4, Dbsus4, Bbsus4, Bbsus2, Ebsus4, Ebsus2, Absus4, Absus2

**File:** `frontend/practice.html` (CHORD_SHAPES object)

---

## Bug 4: Dirty Work wrong key (P2) -- FIXED
**Problem:** Songsterr returns chords in Db/Gb/Bbm instead of D/G/Bm. User has Transpose buttons but doesn't know to use them.

**Fix:** Added smart flat-key detection in `renderChordChart()`:
- Counts unique chord roots that are flats (Db, Gb, Bb, Ab, Eb)
- If >= 50% of unique chords have flat roots AND >= 3 total unique chords, shows an orange banner
- Banner reads: "Chords may be transposed down a half step. Try Transpose +1 if these don't sound right."
- Includes a one-click "Transpose +1" button that applies the transpose and dismisses the banner
- Only shows when transposeAmount is 0 (doesn't nag after manual adjustment)
- Does NOT auto-transpose -- just suggests it

**File:** `frontend/practice.html` (renderChordChart function)

---

## Bug 5: Chord-lyric alignment improvement (P2) -- FIXED
**Problem:** Chords land on wrong words. The word-snapping logic snapped to the nearest word boundary in either direction, sometimes jumping forward to the wrong word.

**Fix:** Improved word-snapping in `renderChordsWithLyrics()`:
- If charPos falls inside a word (between its start and the next word's start), always snap to THAT word's start -- not to the nearest boundary
- This prevents chords from jumping forward to the next word when they land mid-word
- Added deduplication: if two chords land on the same word position, keep only the first
- The existing overlap-resolution code (pushing colliding chords to next word) remains intact

**File:** `frontend/practice.html` (renderChordsWithLyrics function)

---

## Test Results
- 316 passed, 2 failed (pre-existing billing test mismatches, not related to these changes)
- No new test failures introduced
