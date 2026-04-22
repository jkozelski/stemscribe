# Team 3 — Song Matching & Alternative Chord Sources

**Date:** 2026-03-18
**Agents:** 3A (Match Quality), 3B (Empty Chord Fallback), 3C (Alt Source Research), 3D (Metadata Fix)
**Status:** All complete

---

## Agent 3A — Songsterr Match Quality Fix

**Problem:** Auto-search matched Phish's "Mango Song" to "The Show Must Go On - Five Nights at Freddy's by MandoPony" — completely wrong artist/song.

**File changed:** `frontend/practice.html` (searchSongsterr area, ~line 5025+)

### Changes Made

1. **`cleanYouTubeQuery()` function** — Strips junk from YouTube titles before searching Songsterr:
   - Parenthesized tags: `(Official Video)`, `(Official Audio)`, `(Remaster)`, `(HD)`, `(HQ)`, `(4K)`, `(Lyrics)`, `(Lyric Video)`, year patterns like `(1970)`, mix patterns like `(2009 Mix)`
   - Square bracket variants of all the above
   - `ft.`, `feat.`, `featuring` and everything after them
   - Trailing ` - Topic` (YouTube auto-generated channels)

2. **`extractArtistFromQuery()` function** — Parses "Artist - Title" from query string (supports dash, en-dash, em-dash separators)

3. **`artistFuzzyMatch()` function** — Compares extracted artist against Songsterr result artist:
   - Normalization (lowercase, strip "the", remove punctuation)
   - Substring containment (handles "Jimi Hendrix" vs "The Jimi Hendrix Experience")
   - Word overlap with 50% threshold

4. **Updated `searchSongsterr()`** — Two-gate validation:
   - **Gate 1 (Artist):** If artist extracted from query, Songsterr result artist must fuzzy-match. Blocks Phish→MandoPony.
   - **Gate 2 (Words):** Expanded stop words list (+14 words: `a`, `an`, `or`, `remastered`, `4k`, `lyric`, `acoustic`, `cover`, `remix`, `karaoke`, `instrumental`, `unplugged`, `session`, `performance`, `music`, `song`, `topic`). 60% threshold on remaining words.

### Test Cases
| Query | Expected | Result |
|-------|----------|--------|
| `Dead Flowers (2009 Mix)` | Rolling Stones - Dead Flowers | PASS — cleaned to "Dead Flowers", matches correctly |
| `The Mango Song` (Phish) | No match (Phish not on Songsterr) | PASS — MandoPony rejected by word threshold |
| `Phish - The Mango Song` | No match | PASS — artist "Phish" ≠ "MandoPony", rejected at Gate 1 |
| `Jimi Hendrix - Little Wing - HQ` | Jimi Hendrix - Little Wing | PASS — cleaned, artist extracted and matched |
| `Grateful Dead - Bertha (1970)` | Grateful Dead - Bertha | PASS — cleaned, artist matched |

---

## Agent 3B — Songsterr Empty Chord Fallback

**Problem:** Songs like Bertha (songId 194040) have Songsterr tabs but NO chord annotations — 248 measures with all empty chord arrays. UI showed empty grid with dashes, or "Loading chords..." forever.

**File changed:** `frontend/practice.html`

### Changes Made

1. **Updated `renderAIChordView()`** (line ~3541) — New `songsterrFallback` parameter:
   - When `true`: header reads **"Chords (detected) · Use Tab view for full notation"**
   - Shows AI-detected chord diagrams from `CHORD_SHAPES` + unique chord name pills
   - No lyrics, no garbage text

2. **Added Step 2b in `loadChordChart()`** (line ~3743) — Handles empty Songsterr + no AI chords:
   - Shows fallback message directing users to Tab view
   - Prevents falling through to stale "Loading chords..." state

3. **15-second failsafe timeout** (line ~3697) — If "Loading chords..." persists after 15s, replaced with "No chord data available for this song". Cleared in `finally` block.

4. **Updated all Songsterr-related `renderAIChordView()` call sites** to pass `songsterrFallback` flag correctly.

### What's Preserved
- `renderChordDiagram()` and `CHORD_SHAPES` untouched (Team 2's domain)
- Songs with actual Songsterr chord data render normally
- Existing empty-measure skip guards in `renderChordGrid()` intact

---

## Agent 3C — Alternative Chord Source Research

**Full report:** `~/stemscribe/docs/alt-chord-sources.md`

### Source Evaluation Summary

| Source | API? | Coverage (Jam Bands) | Legal Risk | Verdict |
|--------|------|---------------------|------------|---------|
| **Ultimate Guitar** | No (unofficial scrapers exist) | Excellent (200+ Phish, 100+ WSP) | HIGH — ToS prohibits scraping | Manual link only |
| **Phish.net** | Yes (v5, free) | N/A — setlists only, no chords | Low | Not useful for chords |
| **Emil's Tabs** | GitHub repo (open source) | Good Phish coverage | LOW | **Recommended #1** |
| **Chordify** | No | N/A — auto-generated | N/A | Not better than our pipeline |
| **Cifra Club** | No | Minimal (Brazilian focus) | N/A | Not useful |
| **Hooktheory** | Yes (official) | N/A — theory tool, not charts | Low | Not a chord source |

### Recommended Fallback Waterfall
```
1. Songsterr API (primary — best quality, legal, has rhythm data)
2. Emil's Tabs (Phish-only — fork GitHub repo, parse text → StemScribe format)
3. StemScribe auto-detection (universal fallback — already built)
4. UG manual link ("Find chords on Ultimate Guitar" as last resort)
```

### Key Finding
**Emil's Tabs** (emilstabs.org / GitHub) is the hidden gem — open-source Phish guitar transcriptions in plain text, easy to parse, legally clean, covers exactly the Phish gap. Only limitation: Phish-only, no other jam bands.

---

## Agent 3D — Artist Metadata Fix

**Problem:** "About This Track" showed wrong albums — Dead Flowers displayed "Le Top" instead of "Sticky Fingers". MusicBrainz queries weren't filtering compilations properly.

**Files changed:**
- `backend/track_info.py`
- `frontend/js/trackinfo.js`
- `frontend/index.html`

### Backend Changes (`track_info.py`)

1. **Fixed `fetch_musicbrainz_album()`** — Core album lookup fix:
   - Increased search limit from 5 to 25 results
   - Added scoring system: strongly prefers studio albums (Album type, no secondary types) over compilations, live albums, remixes, soundtracks
   - Penalties for bootleg/promotion status
   - Bonus for "Official" release status
   - Artist name verification to filter false positives
   - **Result:** "Dead Flowers" → "Sticky Fingers" (correct)

2. **Added `fetch_musicbrainz_artist_url()`** — Fetches official artist website + genre tags from MusicBrainz URL relations and tags

3. **Updated `fetch_track_info()`** — New metadata fields:
   - `genre` — from MusicBrainz tags (top 3 by vote count)
   - `artist_website` — official homepage from MusicBrainz URL relations
   - `wikipedia_search_url` — always-available fallback search link

### Frontend Changes

4. **`js/trackinfo.js`** — `displayTrackInfo` updated to render:
   - Genre section
   - Wikipedia search link
   - Artist website link

5. **`index.html`** — New HTML elements added:
   - `#genreSection` / `#genreInfo`
   - `#wikiSearchLink`
   - `#artistWebsiteLink`

### Tests
- 315 tests passing, no regressions (2 pre-existing billing test failures unchanged)

---

## Summary of All Files Changed

| File | Agent | Changes |
|------|-------|---------|
| `frontend/practice.html` | 3A, 3B | Match quality (query cleaning, artist gate, stop words) + empty chord fallback |
| `backend/track_info.py` | 3D | MusicBrainz album scoring, artist URL/genre lookup |
| `frontend/js/trackinfo.js` | 3D | Genre, Wikipedia, artist website display |
| `frontend/index.html` | 3D | New info panel HTML elements |
| `docs/alt-chord-sources.md` | 3C | Full research report (new file) |

## Next Steps
- [ ] Test Mango Song / Bertha / Dead Flowers end-to-end on localhost
- [ ] Fork Emil's Tabs repo and build parser for Phish chord fallback
- [ ] Add "Find on Ultimate Guitar" link when no Songsterr/Emil's match
- [ ] Consider community chord contribution feature (long-term)
