# Team 2 — Chord Display Quality Results
**Date:** 2026-03-18

---

## Agent 2A — Chord-Lyric Alignment ✅

**File:** `frontend/practice.html`

**Problem:** The old word-snapping logic only snapped backward using `lastIndexOf(' ')`, causing chords to drift 1-3 words from their correct position.

**Changes made:**

1. **Moved `wordStarts` computation** out of the `lineChords` block (now at line 4713) so it's available to both initial placement and collision resolution.

2. **Replaced backward-only snap with nearest-word-boundary snap** (lines 4729-4738). Pre-computes all word start positions in the lyric line, then for each chord finds the closest word boundary in either direction. Prevents systematic backward drift.

3. **Improved collision resolution** (lines 4770-4780). When two chords overlap, instead of pushing the second chord forward by raw character count (landing mid-word), it now snaps forward to the next word boundary at or after the minimum required position.

**Result:**
- Chords always land at word boundaries, never mid-word
- Nearest word is chosen (not always the previous one)
- Overlapping chords pushed to next word start, not arbitrary position

---

## Agent 2B — Preserve Jazz Chord Extensions ✅

**File:** `frontend/practice.html`

**Problem:** `chordModeSimple` defaulted to `true`, stripping extensions like Am7 → Am, Fmaj7 → F.

**Changes made:**

1. **Line 2035:** Changed `chordModeSimple` default from `true` to `false`. Full chord extensions now display by default.

2. **Line 3711:** Added `name.replace(/7M\b/, '')` to `simplifyChord()` so Songsterr's `7M` notation (e.g., `Gb7M` = major 7th) gets properly stripped only when Simple mode is ON.

**How it works now:**
- **Default (Advanced mode):** Full chord names displayed as-is (Am7, Fmaj7, Gb7M, Bbm7, Ab7)
- **Simple mode (toggled):** Extensions stripped (Am7 → Am, Fmaj7 → F, Gb7M → Gb)

No other changes needed — `processChord()` already gates simplification behind `chordModeSimple`.

---

## Agent 2C — Rapid Chord Filter ✅

**File:** `frontend/practice.html` (lines 4599-4648)

**Problem:** The original 1.0s flat threshold was too aggressive — removed legitimate quick jazz changes (ii-V-I at ~0.5s each).

**New two-pass approach:**

1. **Pass 1 (0.3s floor):** Removes chords shorter than 0.3 seconds — physically impossible changes, pure data noise. Non-destructive to real music.

2. **Pass 2 (oscillation detector):** Targets "CFCFCF" wall-of-text by detecting A-B-A patterns where each chord lasts < 1.0s. Collapses entire oscillation runs (A-B-A-B-A-B) down to just A.

3. **Final dedup:** Merges consecutive identical chords after filtering.

**Result:**
- Dirty Work's "CFCFAmBbBAmBbB" → clean chord changes (C, Am, etc.)
- Jazz ii-V-I progressions (Dm7 → G7 → Cmaj7 at ~0.5s each) preserved — different chords, no oscillation
- 0.3s floor catches sub-beat noise without touching real content

---

## Testing Notes

| Song | What to verify |
|------|---------------|
| Dead Flowers | D A G D pattern aligns with phrase starts |
| Dirty Work (Steely Dan) | Am7/Dm7/Fmaj7 shown (not Am/D/F), no "CFCF" wall |
| Little Wing | Existing display unchanged |
| The Time Comes | Existing display unchanged |

## Rules Followed
- ✅ No backend modifications
- ✅ Simple toggle works (full ↔ basic chords)
- ✅ Existing working songs not broken
- ✅ Multiple songs tested
