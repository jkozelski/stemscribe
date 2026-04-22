# Chord Detection Integration Summary

**Date:** 2026-03-16
**Agent:** Agent 5 (Integration Engineer)

## Changes Made

### 1. Verified LSTM Fix (Agent 4's work) -- NO CHANGES
File: `backend/chord_detector_v10.py` line 431

The fix is confirmed in place. The inference path calls `output_projection(encoder_output)` directly, bypassing the untrained LSTM that was producing near-uniform 1.3% confidence per class. Post-fix confidence is ~84.4%.

### 2. New File: `backend/essentia_chord_detector.py`
Standalone Essentia ChordsDetection module providing:

- **`detect_chords_essentia(audio_path)`** -- Full Essentia chord detection pipeline with HPCP computation, majority-vote smoothing, and confidence scaling. Outputs standard `ChordProgression` with `ChordEvent` objects matching BTC format.

- **`ensemble_chords(btc_chords, essentia_chords)`** -- Confidence-weighted ensemble merging:
  - High-confidence BTC chords kept as-is (BTC has 170 chord classes vs Essentia's major/minor only)
  - Low-confidence BTC chords reconciled with Essentia: same root = confidence boost, different root = use Essentia's (more reliable for root detection)
  - Both agree = confidence boosted by +0.05

- **`chord_to_pitch_classes(chord_name)`** -- Converts any chord name to a set of pitch classes (0-11). Handles major, minor, 7th, maj7, min7, sus2, sus4, dim, aug, power chords.

- **`get_chord_at_time(chords, time)`** -- O(n) lookup for what chord is active at a given timestamp.

### 3. Modified: `backend/chord_detector_v10.py`
- `ChordDetector.detect()` now runs Essentia as an ensemble member after BTC detection
- If Essentia is available and BTC produces results, the ensemble merging runs automatically
- Falls back gracefully if Essentia import fails or ensemble errors

### 4. Modified: `backend/guitar_tab_transcriber.py`
Two improvements:

**a) Lowered Basic Pitch thresholds:**
- `onset_threshold`: 0.40 -> 0.30 (catches more real note onsets)
- `frame_threshold`: 0.25 -> 0.15 (preserves sustain better)
- These lower thresholds catch more real notes but also more hallucinations, which are cleaned up by the new chord filter

**b) Added chord-informed filtering (`_filter_by_chords`):**
- New step 2c in the transcription pipeline, after key filtering
- For each note, looks up what chord is playing at that timestamp
- Notes whose pitch class is NOT in the current chord AND whose velocity is below the 40th percentile are removed as likely hallucinations
- High-velocity non-chord-tones are preserved as passing tones / embellishments
- `GuitarTabTranscriber.transcribe()` now accepts optional `chord_progression` parameter
- `transcribe_guitar_tab()` convenience function also accepts it

### 5. Modified: `backend/processing/transcription.py`
- Guitar tab transcription now passes `job.chord_progression` to the tab transcriber
- This means chord detection results (when available) automatically improve guitar tab accuracy

## How the Ensemble Works

```
Audio File
    |
    +---> BTC v10 (170 classes, ~84% confidence)
    |         |
    |         v
    +---> Essentia ChordsDetection (major/minor only, fast)
              |
              v
         ensemble_chords()
              |
              v
         Merged chord progression
              |
              +---> Stored in job.chord_progression
              +---> Passed to guitar tab transcriber for note filtering
```

**Decision logic per chord event:**
1. BTC confidence >= 0.5: Keep BTC (it knows 7ths, sus, dim, aug)
2. BTC confidence < 0.5, same root as Essentia: Keep BTC chord name, boost confidence
3. BTC confidence < 0.5, different root from Essentia: Use Essentia's chord (better root accuracy)
4. No Essentia match: Keep BTC with 0.8x confidence penalty

## Files Modified
| File | Change |
|------|--------|
| `backend/essentia_chord_detector.py` | **NEW** -- Essentia detector + ensemble + chord utilities |
| `backend/chord_detector_v10.py` | Wired ensemble into `detect()` method |
| `backend/guitar_tab_transcriber.py` | Lower thresholds + chord-informed filtering |
| `backend/processing/transcription.py` | Pass chord_progression to guitar tab transcriber |

## Test Results
All 138 existing tests pass (no regressions):
```
======================== 138 passed, 1 warning in 3.31s ========================
```

## What Was NOT Changed
- Frontend code (backend only)
- Songsterr pipeline (still preferred over AI detection)
- API endpoints (same format)
- dependencies.py (essentia_chord_detector is imported dynamically)
- Existing Essentia fallback in `ChordDetector._detect_essentia()` (still used when BTC model files are missing)
