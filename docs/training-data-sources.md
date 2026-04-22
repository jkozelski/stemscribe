# Training Data Sources for Stem-Aware Chord Detection

Last updated: 2026-04-04

We need sources where we have ALL THREE: audio, correct chord annotations, and
the legal right to use them commercially for model training. This document
catalogs what we have, what we can get, and what the gaps are.

---

## TIER 1 — Ready to Use (Have Audio + Chords + Legal Clearance)

### 1. Jeff's Own Music (Kozelski / The Outervention)

- **License:** Jeff owns all rights. No restrictions whatsoever.
- **Songs:** 20 chord charts digitized, audio for ~18 confirmed on disk
- **Audio location:**
  - `~/Desktop/MUSIC/Kozelski/Kozelski _ Collector/` — 9 songs (m4a)
  - `~/Desktop/MUSIC/Kozelski/Kozelski_ Systematic Static/` — 9 songs (wav)
  - `~/Desktop/MUSIC/The Outervention/ACCLIMATOR/` — 6 songs (wav)
- **Chord charts:** `~/stemscribe/backend/chord_library/kozelski/` — 20 JSON files
- **Chart format:** Section-based JSON with chord symbols per line + lyrics. NOT beat-aligned.
- **Songs with both audio AND charts:**
  - Collector (9): Thunderhead, The City's Gonna Teach You, Living Water, The Blinding Glow, Cold Dice, Mystified, Clever Devil, The Time Comes, Solar Connection
  - Systematic Static (9): Born In Zen, Clandestine Sam, Climbing The Bars, Eye To Eye, Lady In The Stars, Natural Grow, Silent Life, Simpler Understanding
  - Note: "Haitian Divorce" and "Hard Feelings" have charts but audio location needs to be confirmed
- **Gap:** Charts are section/line-level, not beat-aligned timestamps. Will need alignment step (use onset detection or manual beat annotation).
- **Priority:** HIGHEST. Perfect training data — artist-verified ground truth, full copyright ownership.

### 2. GuitarSet

- **License:** CC BY 4.0 — fully clear for commercial training
- **Location:** `~/stemscribe/train_tab_model/trimplexx/python/_mir_datasets_storage/`
- **Songs:** 360 clips (30-second excerpts, solo acoustic guitar)
- **Audio:** `audio_mono-pickup_mix/audio_mono-pickup_mix.zip` — needs unzipping
- **Annotations:** 361 JAMS files in `annotation/` directory
- **Annotation contents (per JAMS file):**
  - `pitch_contour` — continuous f0 pitch tracking (note-level)
  - `chord` — time-aligned chord symbols (e.g., "D#:maj", "G#:maj") with start time + duration
  - Also has note-level MIDI annotations
- **Format:** JAMS (JSON Annotated Music Specification) — standard MIR format
- **Strengths:** Has ALL THREE requirements (audio + chords + pitch/note data). Beat-aligned chord annotations with timestamps. Clean solo guitar — ideal for guitar-specific chord detection.
- **Weakness:** Solo guitar only (no full band mixes). 30-second clips, not full songs. Limited harmonic complexity.
- **Priority:** HIGH. Unzip audio and this is immediately usable.

---

## TIER 2 — Have Annotations, Need Audio

### 3. JAAH (Jazz Audio-Aligned Harmony) Dataset

- **License:** Annotations are open (academic). Audio is copyrighted jazz recordings — must be obtained separately.
- **Location:** `~/stemscribe/backend/training_data/jaah/`
- **Songs:** 113 jazz standards
- **What we have:**
  - `annotations/` — 113 JSON files with beat-aligned chord annotations + beat timestamps
  - `labs/` — 113 .lab files (start_time, end_time, chord_symbol format — standard MIREX)
  - `features/` — pre-extracted chroma features (.csv.gz)
  - MusicBrainz IDs for each track (in JSON metadata)
- **Chord format:** Very detailed — uses extended chord vocabulary (e.g., "F:(b3,5,b7,11)", "C:(3,5,b7,#9)", "Bb:min7"). Beat-aligned with precise timestamps.
- **Audio availability:** NOT included. These are copyrighted recordings (Tito Puente, Miles Davis, etc.). Audio must be purchased or accessed via MusicBrainz/streaming APIs.
- **How to get audio:** Each annotation has an `mbid` (MusicBrainz Recording ID). Could potentially match with a personal music library or use the Spotify/YouTube Audio Features API for feature extraction (but NOT for training directly).
- **Priority:** MEDIUM-HIGH. Excellent annotations but audio acquisition is a legal and practical hurdle. If Jeff owns any of these jazz albums on CD/vinyl, we can rip them.

### 4. McGill Billboard Dataset

- **License:** Annotations are CC (academic use). Audio is copyrighted Billboard hits.
- **Location:** `~/stemscribe/backend/training_data/billboard/`
- **Songs:** 890 entries in the index (not all have annotations)
- **What we have:**
  - `McGill-Billboard/` — 890 numbered directories, each containing a `full.lab` file
  - `billboard-2.0-index.csv` — maps entry IDs to song title, artist, chart date, rank
  - Lab format: `start_time  end_time  chord_symbol` (e.g., "3.53  5.26  A:min")
- **Chord format:** Time-aligned, uses standard chord notation (root:quality). Excellent for training.
- **Audio availability:** NOT included. These are copyrighted Billboard hits (James Brown, Billy Joel, Bette Midler, etc.).
- **How to get audio:** Would need to purchase or rip from owned CDs. NOT legal to scrape.
- **Priority:** MEDIUM. Great annotations, but audio is a blocker. Could cross-reference with Jeff's personal music library.

---

## TIER 3 — Available Externally, Needs Acquisition

### 5. Schubert & Muller Chord Recognition Datasets (ISMIR Benchmarks)

- **License:** Annotations are publicly available for research. Audio is copyrighted (Beatles, Queen, Zweieck).
- **What exists:**
  - Beatles: 180 songs with beat-aligned chord annotations (Chris Harte annotations)
  - Queen: ~50 songs with chord annotations
  - Zweieck: ~18 songs (German band, some may be CC-licensed)
- **Annotation format:** .lab files (start, end, chord) — same as Billboard
- **Audio:** Copyrighted. Beatles/Queen albums must be purchased.
- **How to acquire:** Annotations downloadable from ISMIR datasets page. Audio requires CD rips.
- **Priority:** LOW for now. Great gold standard but audio requires purchases.

### 6. RWC Music Database

- **License:** Available for research under license agreement from AIST Japan. $100-200 for academic license.
- **What exists:** 100 pop songs (original compositions, NOT covers of copyrighted songs)
- **Key advantage:** The songs themselves were COMPOSED for the dataset. No copyright issues on the audio itself.
- **Annotations:** Chord labels, beat positions, melody, structure — very comprehensive
- **Audio:** Included with license. Full songs, multi-genre.
- **How to acquire:** Apply at https://staff.aist.go.jp/m.goto/RWC-MDB/
- **Priority:** MEDIUM-HIGH. This is one of the few datasets where audio + annotations are both obtainable legally. Worth the license fee.

### 7. Jamendo / Free Music Archive (FMA)

- **License:** Various CC licenses. Many tracks are CC BY or CC BY-SA — usable for commercial training.
- **Jamendo:** ~600K+ CC-licensed tracks. Full audio downloadable.
- **FMA:** ~106K tracks, 161 genres, CC-licensed. Audio downloadable.
- **Chord annotations:** NEITHER has chord annotations included.
- **How to create annotations:**
  - Run our own chord detection (creates a chicken-and-egg problem for training)
  - Crowdsource annotations (expensive/slow)
  - Cross-reference with community chord sites (Ultimate Guitar, Chordify) — but those have their own IP issues
- **Priority:** LOW for chord training specifically. Useful as unlabeled audio for self-supervised pre-training or for stem separation training.

### 8. iReal Pro Charts (Jeff's iPad)

- **License:** The chord charts in iReal Pro are user-contributed. The FORMAT is exportable but the charts themselves may have unclear IP status (jazz standards' chord progressions are generally not copyrightable, but arrangements may be).
- **What exists:** Thousands of jazz/pop/rock chord charts with:
  - Chord symbols
  - Time signature
  - Key
  - Style/feel markers
  - Section markers (intro, verse, chorus, etc.)
- **Export format:** iReal Pro can export to:
  - HTML (human-readable)
  - MusicXML
  - Audio (generated MIDI-style backing tracks — NOT real recordings)
  - Shareable URLs (irealb:// protocol with encoded chart data)
- **How to extract:** The irealb:// URL format is well-documented and parseable. Python libraries exist: `ireal-reader` (npm) and `pyRealParser` (Python, pip-installable). These decode the URI-encoded chart data into structured chord progressions.
- **Audio pairing:** iReal Pro charts have NO audio. Would need to pair with separate audio sources. Could pair with JAAH dataset annotations (many overlap — both cover jazz standards).
- **Priority:** MEDIUM. Great structured chord data for jazz standards. Pair with JAAH annotations + purchased jazz recordings for a strong jazz training set.

### 9. Musopen (Public Domain Classical)

- **License:** Public domain performances of public domain compositions. Fully clear for commercial use.
- **What exists:** ~10K recordings of classical music, professionally performed
- **Audio:** Downloadable (some require free account)
- **Chord annotations:** NONE. Classical music uses Roman numeral analysis, not pop chord symbols.
- **Relevance:** Useful for piano chord detection training IF we can generate/find harmonic analyses. Many classical pieces have published harmonic analyses in music theory textbooks.
- **How to create annotations:** Could use existing harmonic analyses from music theory resources (Riemenschneider Bach chorales, Kostka-Payne analyses, etc.)
- **Priority:** LOW. Useful for piano model but requires significant annotation effort.

### 10. Archive.org Live Recordings (Grateful Dead, Phish, etc.)

- **License:** Many bands allow taping and distribution. Grateful Dead recordings on archive.org are explicitly licensed for non-commercial sharing. Phish similarly.
- **"Non-commercial" problem:** These recordings are typically licensed for NON-COMMERCIAL distribution. Using them to train a commercial model is a legal gray area. Would need lawyer review.
- **What exists:** ~15K+ Grateful Dead shows, ~2K+ Phish shows, with known setlists
- **Chord pairing:** Community chord charts exist widely for Dead and Phish songs. Rukind.com (Dead), phish.net (Phish) have comprehensive charts.
- **Audio quality:** Variable (audience recordings). Some are soundboard quality.
- **Priority:** LOW. Legal status for commercial training is unclear. Ask Alexandra about this specifically.

### 11. Synthesized Training Data (Generate Our Own)

- **License:** We own everything we generate. No restrictions.
- **Approach:** Use MIDI chord voicings → synthesize through guitar/piano VSTs → create labeled training pairs
- **Tools:** FluidSynth, SoundFont-based synthesis, or neural audio synthesis
- **Advantages:** Unlimited supply, perfect labels, controllable complexity
- **Disadvantages:** Synthetic audio ≠ real recordings. Models trained on synthetic data may not generalize.
- **Hybrid approach:** Use synthetic data for pre-training, fine-tune on real data (Jeff's songs + GuitarSet)
- **Priority:** MEDIUM. Good supplement but not a replacement for real recordings.

---

## Summary Matrix

| Source | Songs | Audio? | Chords? | Beat-aligned? | License | Priority |
|--------|-------|--------|---------|---------------|---------|----------|
| Jeff's Music | 20 | YES (18 confirmed) | YES (20 charts) | NO (need alignment) | OWNED | HIGHEST |
| GuitarSet | 360 clips | YES (needs unzip) | YES (JAMS) | YES | CC BY 4.0 | HIGH |
| JAAH | 113 | NO (copyrighted) | YES (lab + JSON) | YES | Annotations open | MED-HIGH |
| Billboard | 890 | NO (copyrighted) | YES (lab) | YES | Annotations open | MEDIUM |
| RWC | 100 | YES (with license) | YES | YES | License ~$200 | MED-HIGH |
| iReal Pro | 1000s | NO (charts only) | YES (parseable) | Bars only | Unclear | MEDIUM |
| Synthetic | Unlimited | GENERATE | GENERATE | YES | OWNED | MEDIUM |
| Schubert/ISMIR | ~250 | NO (copyrighted) | YES (lab) | YES | Annotations open | LOW |
| Jamendo/FMA | 100K+ | YES (CC) | NO | N/A | CC variants | LOW |
| Musopen | 10K+ | YES (PD) | NO | N/A | Public domain | LOW |
| Archive.org Live | 15K+ | YES | NO (need pairing) | NO | Non-commercial? | LOW |

---

## Recommended Action Plan

### Phase 1 — Immediate (use what we have)
1. **Unzip GuitarSet audio** and build a data loader for JAMS → our training format
2. **Beat-align Jeff's 20 chord charts** to the audio files using onset detection
3. Combined: ~380 labeled examples (360 GuitarSet clips + 20 Kozelski songs)

### Phase 2 — Quick Wins
4. **Apply for RWC license** ($200) — adds 100 fully-labeled songs
5. **Parse iReal Pro charts** with `pyRealParser` — creates structured chord data for pairing
6. **Cross-reference JAAH + iReal Pro + Jeff's jazz collection** — any jazz albums Jeff owns can be paired with JAAH's beat-aligned annotations

### Phase 3 — Scale Up
7. **Synthetic data pipeline** — generate MIDI chord progressions → render through guitar VSTs
8. **Explore Jamendo/FMA** for self-supervised pre-training (learn audio representations without labels)
9. **Ask Alexandra** about Archive.org live recordings for commercial training use

### Phase 4 — Community
10. **Build a "contribute your chart" feature** in StemScriber — users upload corrections to auto-detected charts, creating a growing labeled dataset over time (with consent/ToS)
