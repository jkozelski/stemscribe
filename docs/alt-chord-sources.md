# Alternative Chord/Tab Sources for StemScribe

**Research Date:** 2026-03-18
**Purpose:** Identify fallback chord sources when Songsterr lacks coverage (especially jam bands)

---

## 1. Ultimate Guitar (UG)

### API Availability
- **No official public API.** UG has never released a developer API.
- **Unofficial scrapers exist:** Multiple open-source projects on GitHub:
  - [Pilfer/ultimate-guitar-scraper](https://github.com/Pilfer/ultimate-guitar-scraper) — Go, uses UG's mobile API endpoints
  - [joncardasis/ultimate-api](https://github.com/joncardasis/ultimate-api) — Python scraper
  - [seanfhear/tab-scraper](https://github.com/seanfhear/tab-scraper) — Node.js, downloads tabs
- The mobile API endpoints these scrapers use are undocumented and can change without notice.

### Legal Considerations
- **ToS explicitly prohibits scraping.** UG's Terms of Service contain anti-scraping clauses.
- User-submitted tabs are classified as "User Generated Content" — UG claims rights over them.
- UG has licensing agreements with publishers (Harry Fox Agency, etc.) for chord/lyric display.
- **Risk level: HIGH.** Commercial use of scraped UG data could trigger legal action.
- No known cease-and-desist cases against scrapers found, but the ToS exposure is real.

### Data Quality & Format
- Plain text chord/tab format (not ChordPro or MusicXML)
- User-rated content — quality varies wildly, but popular songs have well-vetted versions
- Multiple versions per song (chords, tabs, guitar pro, bass, ukulele)
- Ratings system helps surface the best transcriptions

### Coverage
- **Excellent for jam bands.** Phish: 200+ songs. Widespread Panic: 100+. moe.: 50+. Umphrey's: 50+.
- Best overall coverage of any tab site — millions of tabs across all genres.

### Pricing
- Free to scrape (but violates ToS). No API to purchase.

### Verdict
Best coverage by far, but highest legal risk. Not recommended as an automated source. Could potentially be used as a manual reference/fallback link.

---

## 2. Phish.net

### API Availability
- **Official API (v5)** — well-documented, free API keys granted instantly.
- Base URL: `https://api.phish.net/v5/`
- Docs: https://docs.phish.net
- Python wrapper: [pysh](https://github.com/readicculus/pysh)

### What Data Is Available
- **Setlists** — complete show history with songs, dates, venues
- **Song metadata** — song names, debut dates, times played, gaps
- **Show data** — dates, venues, ratings, reviews
- **NO chord/tab data.** The API is purely setlist/metadata. No musical notation.

### Related Phish Tab Sources
- **Emil's Tabs** (emilstabs.org) — community-maintained Phish guitar transcriptions
  - Plain text format, hosted on [GitHub](https://github.com/ehedaya/emilstabs.org)
  - Good coverage of major Phish songs
  - Could be scraped/forked legally (open source, MIT-like)
- **StrangeDesign.org** — Phish tabs, chords, lyrics (may be partially offline)

### Verdict
Phish.net API is great for setlist data but has zero chord content. Emil's Tabs is the real gem here — open source, scrapable, good Phish coverage. However, only covers Phish, not other jam bands.

---

## 3. Chordify

### API Availability
- **No public API.** Community has requested one; Chordify has not released it.
- No known B2B/enterprise API program.
- No unofficial scrapers of note.

### Data Quality
- Auto-generates chords from audio using deep neural networks + beat tracking.
- **Accuracy: Moderate.** Good for simple pop/rock progressions, struggles with:
  - Complex jazz voicings
  - Extended chords (7ths, 9ths, 13ths)
  - Fast chord changes
  - Jam band improvisation sections
- Quality similar to what StemScribe already does with its own chord detection pipeline.

### Pricing (Consumer)
- Free tier with ads
- Premium: ~$7/month (capo, transpose, MIDI export, etc.)

### Verdict
No API access, and the auto-generated quality wouldn't be better than StemScribe's own pipeline. **Not useful as a fallback.**

---

## 4. Cifra Club

### API Availability
- **No official API.**
- One unofficial GitHub scraper exists: [lucis/cifra-club-api](https://github.com/lucis/cifra-club-api)
- 400k+ chords/tabs on the platform.

### Coverage
- **Primarily Brazilian/Portuguese music.** Latin America's biggest music platform.
- Has some English-language artists (Beatles, Oasis, Elvis), but coverage of American jam bands is minimal to nonexistent.
- Phish, Widespread Panic, moe. — unlikely to have meaningful coverage.

### Data Format
- Standard chord-over-lyrics text format
- Supports guitar, bass, piano, ukulele, drums

### Legal Considerations
- ToS likely prohibits scraping (standard for these sites).

### Verdict
**Not useful for StemScribe's needs.** Poor coverage of target genres (jam bands, American indie/classic rock). Would only add value for Brazilian music.

---

## 5. Other Sources

### Hooktheory (hooktheory.com)
- **Has an official API** — https://www.hooktheory.com/api/trends/docs
- OAuth 2.0 bearer token auth
- Rate limit: 10 requests per 10 seconds
- **Data available:** Chord probability data, songs containing specific chord progressions
- **NOT a tab source.** It's a music theory analysis database — tells you what chords are common in songs, not the actual chord chart for a specific song.
- Interesting for chord suggestion features, but not a fallback tab source.

### Uberchord API (api.uberchord.com)
- Free API for chord diagrams/voicings
- Endpoint: `https://api.uberchord.com/v1/chords/{chordName}`
- **Chord shapes only** — not song-specific chord progressions.
- Useful for rendering chord diagrams (which StemScribe already handles).

### Scales-Chords API (scales-chords.com/api/)
- Free, no activation required
- 500k+ chord images and sound files
- **Chord reference only** — not song chord charts.

### Open Chord Charts (GitHub)
- [open-chord-charts/web-api](https://github.com/open-chord-charts/web-api)
- Collaborative chord chart database
- Very small catalog — not practical as a source.

### tombatossals/chords-db (GitHub)
- Static JSON database of chord voicings for guitar/ukulele
- Useful for chord diagram rendering, not for song charts.

### autochord (PyPI)
- Python library for automatic chord recognition from audio
- Open source alternative to Chordify
- StemScribe already has its own chord detection — redundant.

---

## Recommendations

### Best Fallback #1: Emil's Tabs (for Phish specifically)
- **Why:** Open source on GitHub, plain text format, easy to parse, covers the exact gap (Phish songs Songsterr doesn't have).
- **How:** Fork/clone the GitHub repo, parse text files into StemScribe's chord format, serve as a fallback when Songsterr returns no results for Phish songs.
- **Legal risk:** LOW — community-contributed, open source.
- **Limitation:** Only covers Phish. No help for Widespread Panic, moe., Umphrey's, etc.

### Best Fallback #2: UG Data via Manual Curation (not scraping)
- **Why:** UG has the best coverage for everything Songsterr misses, including all jam bands.
- **How:** Instead of automated scraping (too legally risky), consider:
  1. Manual entry of key jam band songs from UG into StemScribe's own chord database
  2. Linking out to UG as a "can't find chords? try here" reference
  3. Building a community contribution feature where users can submit/correct chords (like UG's model)
- **Legal risk:** LOW if manually curated or user-submitted. HIGH if automated scraping.

### Long-term Strategy: StemScribe's Own Chord Pipeline
- The best long-term play is improving StemScribe's own audio-to-chord detection.
- For songs with no external source, run the chord detection pipeline on the separated stems.
- Combine multiple sources: Songsterr (primary) → Emil's Tabs (Phish) → StemScribe auto-detection (fallback).

### Not Recommended
- **Chordify** — No API, quality no better than StemScribe's own detection
- **Cifra Club** — Wrong genre focus (Brazilian music)
- **Hooktheory** — Music theory tool, not a chord chart source
- **Phish.net API** — Setlists only, no chords

---

## Priority Waterfall for Chord Lookup

```
1. Songsterr API (current primary — best quality, legal, has rhythm data)
2. Emil's Tabs (Phish-only fallback — open source, free)
3. StemScribe auto-detection (universal fallback — already built)
4. UG manual link (show "Find chords on Ultimate Guitar" link as last resort)
```
