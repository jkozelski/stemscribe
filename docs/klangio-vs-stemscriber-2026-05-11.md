# StemScriber vs. Klangio — Empirical Test (2026-05-11)

**TL;DR:** Klangio splits in two. **Song Search** (songs.klang.io) is a free pre-computed chord-chart database for ~1,500 popular songs; on 8 shared audit songs it edges StemScriber 0.762 to 0.741 mean F1. Their **transcription apps** (Piano2Notes, Guitar2Tabs, Melody Scanner) output sheet music / TAB / MIDI, not chord charts — different product, 20-sec free demo too short to audit. Chord-set quality is roughly a tie within noise; coverage and product overlap are the real story.

## Free Tier Limits

| Surface | Free tier | Output |
|---|---|---|
| **Song Search** | Unlimited, no signup | Chord chart, key, BPM, meter, beat-aligned timeline |
| **Piano2Notes / Guitar2Tabs / Sing/Drum/Violin/Wind2Notes** | First 20 sec | Sheet / TAB / MIDI / MusicXML / .gp |
| **Melody Scanner** | 1 min, 40-bar PDF cap | Lead sheet (melody + chords) |
| Pro | Required for full-song | — |

Song Search is the only free surface giving full-song chord output. 20-sec demos on per-instrument apps are too short for head-to-head.

## Song Search vs. StemScriber — chord-chart head-to-head

Same canonical chord sets and `diff_sets()` scorer as the May 8 audit and the Chordify head-to-head. Two of 18 audit songs were absent (Hey Joe, Bad Company); catalog skews modern pop.

| Song | SS F1 | KL F1 | Win | Note |
|---|---:|---:|---|---|
| House of the Rising Sun (Animals) | 0.91 | **1.00** | KL | SS adds spurious Dm |
| Free Fallin' (Tom Petty)          | 0.86 | **1.00** | KL | SS adds Fm |
| Don't Let Me Be Misunderstood     | 0.71 | **0.92** | KL | SS adds Am+Dm, misses Bm |
| Hotel California (Eagles)         | 0.86 | 0.86     | Tie | Both miss F#7 |
| Back In Black (AC/DC)             | 0.55 | **0.60** | KL | Both over-detect |
| Paint It Black (Stones)           | **0.67** | 0.17 | SS | KL key detector wrong (F vs E) |
| Dream On (Aerosmith)              | 0.67 | **0.78** | KL | KL catches A#m |
| Take On Me (a-ha)                 | 0.71 | **0.77** | KL | KL adds spurious C#m |

**Tally: Klangio 6, SS 1, Tie 1. Means: KL 0.762, SS 0.741, n=8.**

### Where each wins

- **Klangio:** tight conservative chord lists (5–8 chords/song). Doesn't emit Dm-next-to-D or Fm-next-to-F variant noise. Free Fallin' and Rising Sun are perfect.
- **StemScriber:** Paint It Black saves us. Klangio's key detector returned F (song is in E), only C matched canonical — single-song catastrophic failure for them.
- **Coverage:** Hey Joe + Bad Company not in database. SS works on any upload.

0.02 F1 on n=8 is within noise; Chordify (n=18) also landed at 0.77. All three tools cluster in 0.74–0.81 — none dramatically better at chord-set level.

## Transcription apps — different product

Not scored (20-sec free demos can't be compared with full-song output). Confirmed:

- **Guitar2Tabs:** YouTube URL → 20s TAB + sheet + MIDI + .gp. Note-level, not chord-chart-first.
- **Melody Scanner:** lead-sheet mode (melody + chords); 1-min + 40-bar PDF cap.
- **Piano/Drum/Sing/Violin/Wind2Notes:** instrument-specific sheet music. No chord focus.

These overlap with StemScriber's MIDI/.gp5 export, not the chord chart. More mature than our note-level export today — but not chord-chart competitors.

## Honest Verdict

**Chord charts:** roughly tied (KL 0.762 vs SS 0.741 within noise). Klangio wins on conservative output; SS wins when their key detector flips and on songs outside their database.

**Sheet music / TAB / note transcription:** Klangio more mature. Real product line vs SS's MIDI side output.

**Coverage:** Klangio is a closed ~1,500-song catalog, modern-pop weighted. StemScriber takes any upload — structural advantage on long-tail (indie, covers, live recordings, user demos).

## Strategic Implication

Partially competitor, partially sister product, not a partner candidate.

1. **Song Search is a database play, not a real-time engine.** Chart-toppers belong to them, Chordify, and Ultimate Guitar. SS's defense is anything-you-can-upload — own recordings, indie, covers, pre-2010 deep cuts.
2. **Don't reframe SS as a Klangio competitor.** Different audience (catalog browser vs upload tool), different output (in-player chord chart vs PDF sheet music). Forced comparisons backfire.
3. **Copy their restraint.** Klangio's 5–8 conservative chord-list output is the right model. SS bleeds F1 on over-detection (Dm-next-to-D, Fm-next-to-F) on songs we should nail. A precision-prune pass — drop minor-of-major when major dominates — would probably close the gap.
4. **No partnership angle.** Their `klang.io/api` is B2B bulk-transcription, not a chord-detection API.

The audit shows we're competitive where it counts. Bigger lever is over-detection cleanup, not chasing their catalog.

## Source Data

- Extraction: in-browser fetch of `/api/search/{q}` then `/en/s/{artist}/{title}/{ytid}` HTML scrape
- Per-song data: `/tmp/klangio-audit/data.json`
- Scoring: `/tmp/klangio-audit/score.py` (uses `backend/audit/llm_oracle.py::diff_sets`)
- Scored output: `/tmp/klangio-audit/scored.json`
- Canonical reference (shared with Chordify test): `/tmp/audit-may8-results/_oracle-qflip.jsonl`
