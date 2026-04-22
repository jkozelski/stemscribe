# Songsterr Chord Data Sources Research

**Date:** 2026-03-24
**Status:** Research complete

## Summary

Songsterr has **three distinct chord data sources**, two of which StemScriber is not currently using. The biggest win is the **ChordPro CDN endpoint** -- a dedicated service that returns full chord charts with lyrics in ChordPro format, complete with sections and chord-lyric alignment.

---

## 1. Current StemScriber Approach

StemScriber currently uses two methods (in `backend/routes/songsterr.py`):

### Method A: Native `beat.chord` annotations (lines 290-318)
- Looks for `beat.chord.text` in the per-track JSON from the CloudFront CDN
- Format: `{"text": "Em7", "width": 29}`
- This is the **most accurate** source when present -- human-curated chord names placed at exact beat positions
- **Problem:** Many songs lack these annotations. Purple Haze (song 310) has **zero** chord annotations across all 6 tracks, despite `hasChords=true` on the API.

### Method B: Fret-to-chord inference (lines 323-374)
- Falls back to analyzing simultaneous notes (fret positions) to identify chords
- Uses interval analysis to detect major, minor, sus, dim, aug, power chords, etc.
- **Problem:** Only works when the tab has chord strumming patterns. Lead guitar parts with single-note riffs produce no chords.

### Current API endpoints used:
- `https://www.songsterr.com/api/songs?size=N&pattern=QUERY` -- search
- `https://www.songsterr.com/api/meta/{songId}` -- metadata (revisionId, image hash, tracks)
- `https://dqsljvtekg760.cloudfront.net/{songId}/{revisionId}/{imageHash}/{trackIndex}.json` -- per-track tab data

---

## 2. NEW: ChordPro CDN Endpoint (Not Currently Used)

### Discovery

Songsterr has a **separate ChordPro system** served from dedicated CDN subdomains. Found by analyzing the JS bundle (`appClient-*.js`).

### How to access it:

**Step 1:** Get the chordpro hash from the chords API:
```
GET https://www.songsterr.com/api/chords/{songId}
```
Response:
```json
{
    "artist": "Oasis",
    "artistId": 2,
    "chordpro": "rRafLLBSu7RoPJm9rqPyM",   // <-- hash needed
    "chordsRevisionId": 2,                      // <-- revision ID needed
    "hasPlayer": true,
    "songId": 2,
    "title": "Wonderwall"
}
```

**Step 2:** Fetch the ChordPro file from the CDN:
```
GET https://chordpro1.songsterr.com/{songId}/{chordsRevisionId}/{chordpro_hash}.chordpro
Accept: text/plain
```

CDN subdomains (load balanced):
- `chordpro1.songsterr.com`
- `chordpro2.songsterr.com`
- `chordpro3.songsterr.com`
- Fallback: `www.songsterr.com/cdn/chordpro/{songId}/{chordsRevisionId}/{hash}.chordpro`

**Note:** Response is gzip/brotli compressed -- use `--compressed` with curl or `Accept-Encoding` header.

### ChordPro format example (Wonderwall):

```
{capo: 2}
{tuning: Standard (EADGBE)}
{section: Intro}
[Em7]   [G]   [Dsus4]   [A7sus4]

{section: Verse}
[Em7] Today is g[G]onna be the day
That they're g[Dsus4]onna throw it back to y[A7sus4]ou,

{section: Bridge}
And [Cadd9]all the roads we h[Dsus4]ave to walk are w[Em7]inding

{section: Chorus}
Because m[Cadd9]aybe - [Em7] - [G]
[Em7] You're gonna be the one that s[Cadd9]aves me
```

### What it includes:
- **Chord names** in square brackets, positioned inline with lyrics
- **Section markers** (`{section: Verse}`, `{section: Chorus}`, etc.)
- **Capo information** (`{capo: 2}`)
- **Tuning** (`{tuning: Standard (EADGBE)}`)
- **Lyrics** with chords aligned to specific words/syllables

### Coverage:

| Song | songId | hasChords | ChordPro Available | ChordPro Content |
|------|--------|-----------|-------------------|-----------------|
| Wonderwall - Oasis | 2 | true | Yes | Full (chords + lyrics + sections) |
| Sweet Child O' Mine - GNR | 23 | true | Yes | Full |
| Stairway to Heaven - Led Zep | 27 | true | Yes | Full |
| Purple Haze - Hendrix | 310 | true | Yes | **Minimal** (tuning only) |
| Smoke On The Water - Deep Purple | 329 | true | Yes | Full |
| Hotel California - Eagles | 447 | true | Yes | Full |
| Bach Prelude | various | false | N/A | N/A |

**Key finding:** Purple Haze has `hasChords=true` and a chordpro hash, but the ChordPro file only contains `{tuning: Standard (EADGBE)}` -- no actual chord data. Most popular rock songs have full ChordPro content.

---

## 3. NEW: `/api/chords/{songId}` Endpoint (Not Currently Used)

This is the gateway to the ChordPro system:

```
GET https://www.songsterr.com/api/chords/{songId}
```

Returns the chordpro hash and chordsRevisionId needed to fetch the actual ChordPro file. Also confirms whether the song has a player (`hasPlayer`).

The `chordsRevisionId` is typically the same as the `songId` for the initial revision, but may differ for updated chord charts.

---

## 4. Beat-Level Chord Annotations in Track JSON

### Where they exist
The per-track CDN JSON (`{trackIndex}.json`) can contain `chord` objects on individual beats:
```json
{
    "notes": [...],
    "velocity": "mf",
    "type": 8,
    "chord": {"text": "Em7", "width": 29}
}
```

### When they're present
- Typically found on **rhythm guitar tracks** for well-curated tabs
- Wonderwall rhythm guitar (track 3): chord annotations on every chord change
- Purple Haze: **none** across all 6 tracks (lead guitar, bass, drums, vocals, octaver)

### Important: beat.chord is track-specific
Different tracks for the same song may or may not have chord annotations. The current code checks all tracks (good), but could prioritize rhythm guitar tracks that are more likely to have them.

---

## 5. `guitarpro` Library Chord Support

The `guitarpro` Python library's `Chord` model has rich fields:
- `name` (string) -- e.g., "Em7"
- `root`, `bass` -- root and bass note
- `type` -- ChordType enum: major, minor, seventh, diminished, augmented, sus2, sus4, power, etc.
- `extension` -- ChordExtension enum: none, ninth, eleventh, thirteenth
- `strings` -- fret positions per string (e.g., `[0, 2, 2, 0, 0, 0]` for Em)
- `firstFret` -- position for chord diagram
- `barres`, `fingerings` -- for chord diagram rendering
- `sharp`, `tonality`, `fifth`, `ninth`, `eleventh` -- detailed voicing info

The current `songsterr_to_gp.py` already creates `Chord` objects and attaches them to beats when converting to GP5, but only from fret inference -- it does NOT use the native `beat.chord.text` or ChordPro data.

---

## 6. Embedded Page State

The Songsterr HTML page embeds a large JSON state (`<script id="state" type="application/json">`). Key findings:

### Relevant state keys:
- `state.chords` -- Contains `current`, `songId`, `chordsRevisionId` (null on initial page load, loaded lazily)
- `state.chordpro` -- Same structure, loaded when user switches to chord view
- `state.leadSheet` -- Lead sheet display mode (`on`, `supportByPart`)
- `state.chordDiagram` -- Chord diagram data (empty on initial load)
- `state.chordsSimplify` -- Boolean for simplified chord names
- `state.chordsTransposeNotation` -- Transpose semitones for chord display

### Lazy loading behavior:
Chord/ChordPro data is NOT included in the initial page HTML. It's loaded via XHR when the user clicks the "Chords" tab in the Songsterr player. This is why `state.chords.current` and `state.chordpro.current` are null on initial page load.

---

## 7. Other Endpoints Found

From JS analysis:
- `api/chords/{songId}` -- Get chordpro hash (confirmed working)
- `api/chords/{songId}/{chordsRevisionId}` -- Returns same as above (NOT separate endpoint)

From page state, the data also includes:
- `meta.current.lyrics` -- Boolean indicating lyrics availability
- `meta.current.audioV4` / `audioV4Meta` -- Audio metadata
- `meta.current.tags` -- Song tags
- `track.newLyrics` -- Lyrics with syllable splitting embedded in track JSON

---

## 8. Lyrics in Track Data

Track JSON includes a `newLyrics` array with syllable-split lyrics:
```json
{
    "line": 1,
    "offset": 1,
    "text": "[Verse 1]\nPur-ple ha-ze all in my bra-in\nLate-ly things..."
}
```

Currently used by StemScriber (line 472-479 in `routes/songsterr.py`), with lrclib.net as fallback for synced lyrics.

---

## Recommendations

### Priority 1: Integrate ChordPro CDN (High Impact, Low Effort)

Add a new function to fetch ChordPro data:
1. Call `GET /api/chords/{songId}` to get the `chordpro` hash and `chordsRevisionId`
2. Fetch `https://chordpro1.songsterr.com/{songId}/{chordsRevisionId}/{hash}.chordpro`
3. Parse the ChordPro format (simple text parsing: `[Chord]` markers + `{section:}` directives)
4. Use this as the **primary chord source** when available (better than fret inference)
5. Fall back to beat.chord annotations, then fret inference, then AI detection

**Why:** This gives us human-curated chord charts with lyrics + sections for the majority of popular songs. The format is already standard (ChordPro) and easy to parse. This would eliminate the chord-lyric alignment issues mentioned in MEMORY.md.

### Priority 2: Better beat.chord Extraction

The current code already handles `beat.chord.text` but could be improved:
- Prioritize rhythm guitar tracks (instrumentId 25-29 = acoustic/clean guitar)
- Check multiple tracks and merge chord data
- Use beat timing data to produce more accurate chord timestamps

### Priority 3: Use ChordPro Data for Chord Sheet Generation

The ChordPro format from Songsterr maps directly to the chord sheet format StemScriber generates. Instead of building chord sheets from separate chord events + lyrics + word timestamps, parse the ChordPro directly -- it already has chords aligned to lyrics.

### Priority 4: Cache Strategy

The ChordPro data is small (< 2KB per song) and rarely changes. Cache aggressively:
- Store by `songId + chordsRevisionId` key
- Check revision on each request but serve from cache if unchanged

### Not Recommended

- Scraping the HTML page state for chords -- data is lazy-loaded and not in initial HTML
- Using the GP5 CDN (d12drcwhcokzqv.cloudfront.net) -- this is for legacy GP5 downloads and may not contain chord annotations
- Relying solely on `hasChords` flag -- Purple Haze has `hasChords=true` but empty ChordPro content

---

## API Reference Summary

| Endpoint | Purpose | Auth Required |
|----------|---------|---------------|
| `GET /api/songs?size=N&pattern=QUERY` | Search songs | No |
| `GET /api/meta/{songId}` | Song metadata + tracks | No |
| `GET /api/chords/{songId}` | ChordPro hash + revision | No |
| `GET https://chordpro{1,2,3}.songsterr.com/{songId}/{chordsRevisionId}/{hash}.chordpro` | ChordPro file | No |
| `GET https://dqsljvtekg760.cloudfront.net/{songId}/{revisionId}/{imageHash}/{trackIdx}.json` | Per-track tab data (may contain beat.chord) | No |
