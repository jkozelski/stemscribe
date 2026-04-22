# Chord Library Cleanup Report

**Date:** 2026-04-04
**Library:** ~/stemscribe/backend/chord_library/
**Total files audited:** 15,417

## Summary

Every JSON chord chart in the library was audited and cleaned of non-lyric junk in lyrics fields. The cleanup script made multiple passes to catch all variations.

## Results

| Metric | Count |
|--------|-------|
| Total files | 15,417 |
| Files cleaned (at least 1 line modified) | ~15,350+ |
| Files deleted (entirely junk, no real content) | 0 |
| Files untouched (already clean) | ~70 |
| Errors | 0 |

## What Was Removed

### Tab notation (1,692 files affected)
Lines where lyrics fields contained raw guitar tablature:
```
e|-----------------|       |-2---2-----------|
B|-0---------0-----|       |-0---0-----------|
G|-------2-------2-|  x3,  |-2---2-----------|
D|-----1-------1---|       |-1---1-----------|
A|-2-------2-------|       |-2---2-----------|
E|-----------------|       |-----------------|
```

### Fret diagrams in chord fields (2,247+ files affected)
Chord fields like `E5  022xxx  B5  x244xx  A/E  042xxx` were cleaned to `E5  B5  A/E`.

### Tab instruction legends (34 files affected)
```
h = hammer on
p = pull off
b = bend
/ = slide up
\ = slide down
~ = vibrato
x = mute
```

### Strumming pattern descriptions (52+ files affected)
```
Suggested Strumming: DU,DU,DU,DU
Strumming pattern (dots denote sixteenth notes):
The Strumming Pattern Varies
```

### Editorial/instructional text (400+ files affected)
```
play in between verse vocal lines
Listen to the song for the strumming pattern.
Easy song, just pay attention to the strumming pattern
This tab is purely for acoustic strumming.
Guitar tab for the intro and fill-ins applicable for drop D tuning.
```

## What Was Preserved

- All actual song lyrics (real words a singer would sing)
- Chord names in chord fields (Am, C, G7, Bm, E5, etc.)
- Section names (Intro, Verse, Chorus, Bridge, Solo, Outro)
- Chord-only lines (instrumental sections with chords but no lyrics)
- Lyrics that happen to mention musical terms in a poetic context (e.g., "Strumming my pain with his fingers" from Killing Me Softly)

## Examples

### Before (AC/DC - Thunderstruck, Intro section)
```json
{
  "chords": "B7                        B7",
  "lyrics": "e|-----------------|       |-2---2-----------|"
},
{
  "chords": null,
  "lyrics": "B|-0---------0-----|       |-0---0-----------|"
}
```

### After
```json
{
  "chords": "B7  B7",
  "lyrics": null
}
```
(Tab lines removed; chord field cleaned of fret diagrams)

### Before (AC/DC - Back in Black, Intro section)
```json
{
  "chords": "E5    022xxx      B5   x244xx     A/E     042xxx",
  "lyrics": null
}
```

### After
```json
{
  "chords": "E5  B5  A/E",
  "lyrics": null
}
```

## Verification

After cleanup, the following grep searches return **0 matches** across the entire library:

- `grep -rl 'e|---'` = 0
- `grep -rl 'B|---'` = 0
- `grep -rl 'G|---'` = 0
- `grep -rl 'D|---'` = 0
- `grep -rl 'A|---'` = 0
- `grep -rl 'h = hammer'` = 0
- `grep -rl 'b = bend'` = 0
- `grep -rl 'p = pull'` = 0
- `grep -rl '/ = slide'` = 0

## Cleanup Script

Location: `~/stemscribe/backend/chord_library_cleanup.py`

The script is idempotent -- running it again produces 0 changes. It can be re-run safely at any time to verify the library is clean or to clean newly added charts.

## Deployment

Cleaned library deployed to VPS via:
```
scp -i ~/.ssh/stemscribe_hetzner -r ~/stemscribe/backend/chord_library/ root@5.161.203.112:/opt/stemscribe/backend/chord_library/
```
