# Practice Mode — Definitive Spec (April 2026)

**This is the final word. Stop changing the order of operations.**

## What Each Instrument Gets

### Guitar & Bass — TABLATURE
- **Source:** Songsterr API (real human-transcribed tabs)
- **Display:** Guitar/bass tablature (6-string or 4-string tab notation via AlphaTab)
- **Filter:** ONLY show tracks that are actual guitar or bass instruments. No sax tabs, no piano tabs, no vocal tabs.
- **Fallback:** If Songsterr doesn't have the song, show chord chart only. Do NOT show AI-generated GP5 tabs — they're garbage.

### Chords & Lyrics — CHORD CHART
- **Source:** Local chord library (~15,400 charts in chord_library/)
- **Display:** Chord names above lyrics, section headers (Verse, Chorus, etc.)
- **Fallback:** Songsterr chord data if no local chart exists
- **Quality:** Library is being cleaned — no raw tab notation in lyrics fields

### Piano/Keys — STANDARD NOTATION
- **Source:** Custom CRNN piano transcription model (best_piano_model.pt, 145MB, Jeff spent $100 training it)
- **Display:** Standard notation — treble + bass clef, proper note heads on staff lines. NOT guitar tab.
- **Renderer:** OSMD (OpenSheetMusicDisplay) from MusicXML output
- **Toggle:** User opt-in via Piano button in notation toggles (default OFF)
- **Future:** Look for open-source piano sheet music databases to supplement AI transcription

### Drums — DRUM NOTATION
- **Source:** Custom CRNN drum transcription model (best_drum_model.pt, 114MB, Jeff spent $100 training it)
- **Display:** Standard drum notation on percussion staff. NOT guitar tab.
- **Renderer:** OSMD from MusicXML output
- **Toggle:** User opt-in via Drums button in notation toggles (default OFF)
- **Quality:** Drum transcription rated highly — this is a real feature, not junk

### Vocals — LYRICS ONLY
- **Source:** LRCLIB (synced lyrics) or Whisper fallback
- **Display:** Synced lyrics in karaoke mode, plain lyrics in practice mode
- **No notation needed** — vocals are for singing along, not reading notation

### Sax/Horns/Other — STANDARD NOTATION (future)
- **Source:** Would need transcription from the "other" stem
- **Display:** Standard notation on treble clef
- **Status:** Not implemented yet. For now, these instruments get no notation.

## Songsterr Track Filtering

When loading Songsterr tabs, ONLY show these instrument types as selectable tracks:
- Guitar (any variant: lead, rhythm, acoustic, clean, distortion)
- Bass (any variant: electric, acoustic, fretless)

HIDE these from the track selector:
- Vocals / Voice
- Piano / Keyboard / Organ / Synthesizer
- Drums / Percussion (Songsterr drum tab is on a guitar staff — useless)
- Saxophone / Trumpet / Trombone / any horn
- Strings (violin, cello, etc.)

## View Toggle

- **Chords** view: Chord chart with lyrics (default view)
- **Tab** view: Songsterr guitar/bass tabs only
- **Notation toggles** (below chord chart): Piano and Drums buttons for standard notation

## What We Are NOT Doing

- No AI-generated guitar GP5 tabs (Basic Pitch quality is too low)
- No guitar tab notation for non-guitar instruments
- No OSMD rendering of guitar/bass (Songsterr handles those)
- No AI chord detection by default (toggle is off, Songsterr/library handles it)
