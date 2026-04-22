# Pipeline Integration Summary
**Date:** 2026-04-12

## What Was Connected

Every piece of the processing pipeline is now wired end-to-end. When a user uploads a song, the full flow is:

### 1. Stem Separation (already working)
- BS-RoFormer on Modal cloud GPU (A10G)
- Produces 6-8 stems: vocals, bass, drums, guitar, piano, other
- Auto-splits vocals into lead/backing (BS-Roformer Karaoke model)
- Auto-splits guitar into lead/rhythm (MelBand-RoFormer, when available)

### 2. Chord Detection -- SWITCHED TO V8
- **Before:** V10 (BTC, 170 chord classes) was primary
- **After:** V8 Transformer (93.6% accuracy, 337 chord classes) is primary
- V8 includes: Am7, Cm7, E9, Bm6, mMaj7, dim7, hdim7, aug, sus2, sus4, add9, inversions (C/E, Am/C)
- V10 remains as fallback if V8 fails to import
- V8's `ChordProgression` now has `tuning_info` field for pipeline compatibility
- V8's `detect()` now accepts `artist`/`title` kwargs for pipeline compatibility

### 3. Chart Formatter (was built, already wired)
- `chart_formatter.py` takes chord events + Whisper word timestamps
- Produces `chord_chart.json` in UG-style format (chords above lyrics, sections)
- **Fix:** Whisper model downgraded from `large-v3` to `medium` on CPU servers (large-v3 was OOMing/stalling on 8GB VPS)

### 4. Lead Sheet Generator -- NEWLY WIRED
- `lead_sheet_generator.py` generates MusicXML with slash notation + chord symbols
- Produces `lead_sheet.musicxml` for rendering via OSMD (OpenSheetMusicDisplay)
- Requires music21 (installed on VPS: `pip install music21`)
- Runs after chart formatter, before MIDI transcription

### 5. Guitar Transcription -- VERIFIED ACTIVE
- Guitar CRNN model (`best_guitar_model.pt`, 75MB) is highest priority
- Kong-style architecture: 4 ConvBlocks + BiGRU, 48 pitches (E2-Eb6)
- Falls back to Basic Pitch guitar tab model if quality < 0.3
- Confirmed working: Thunderhead transcribed with `guitar_v3_nn` mode

### 6. Bass Transcription -- FIXED AND ACTIVE
- **Bug fixed:** Checkpoint path was `best_bass_v3_model.pt` but file is `best_bass_model.pt`
- **Bug fixed:** Checkpoint embeds CNN weights; code was trying to load separate piano CNN first
- **Fix:** Auto-detects whether checkpoint has embedded CNN weights
- `MODEL_AVAILABLE` no longer requires piano checkpoint (bass model is self-contained)
- Confirmed working: Thunderhead transcribed with `bass_v3_nn` mode

### 7. Drums/Piano (already working)
- Neural CRNN drum model (114MB, 8 classes)
- Piano CRNN model (145MB, 88 keys)

## Files Modified

| File | Change |
|------|--------|
| `backend/dependencies.py` | V8 chord detector is now primary (was V10) |
| `backend/chord_detector_v8.py` | Added `tuning_info` to ChordProgression, `artist`/`title` kwargs to `detect()` |
| `backend/bass_nn_transcriber.py` | Fixed checkpoint path, auto-detect embedded CNN weights |
| `backend/processing/pipeline.py` | Wired lead sheet generator after chart formatter |
| `backend/word_timestamps.py` | Auto-select Whisper model size based on hardware (medium on CPU, large-v3 on GPU) |

## Test Results (Thunderhead - job 9752ac67)

- **Status:** completed
- **Key:** Am (detected by V8)
- **Chords:** 80 events including Cm7, Bm7, Am7, Em7, E7, Gmaj7, Dm7
- **Stems:** 8 (bass, drums, guitar, other, piano, vocals, vocals_backing, vocals_lead)
- **MIDI:** 7 files (all instruments except 'other')
- **Transcription modes:**
  - guitar: `guitar_v3_nn` (CRNN)
  - bass: `bass_v3_nn` (CRNN)
  - drums: `drum_nn` (CRNN)
  - piano: `piano_nn` (CRNN)
  - vocals/vocals_lead: `melody`
  - vocals_backing: `enhanced`
- **Chord chart:** Generated (9 sections)
- **Lead sheet:** Requires music21 (now installed on VPS)

## VPS Deployment

All files deployed to `root@5.161.203.112:/opt/stemscribe/backend/`.
Service restarted: `systemctl restart stemscribe`.
music21 installed: `pip install music21`.
