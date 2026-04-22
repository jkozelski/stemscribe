# Stem-Aware Chord Detection: Test Results

**Date:** 2026-04-12
**Module:** `backend/stem_chord_detector.py`
**Status:** Deployed to VPS, integrated into pipeline

---

## Test 1: Local (M3 Max) - Thunderhead Stems

**Stems:** `backend/outputs/modal_test/` (guitar.mp3, bass.mp3, piano.mp3)
**Processing time:** ~35 seconds
**Chords detected:** 397 (before post-processing merge), consolidated to major segments

### Known Correct Chords (Thunderhead)
Am7, Bm7, Cm7, E9, Bm6, Dm7, Gmaj7, Em7, G#7, G#maj7

### Detection Results (unique chords detected, mapped to V8 vocabulary)

| Known Chord | Detected? | V8 Mapping | Notes |
|------------|-----------|------------|-------|
| Am7        | NO        | -          | Missed -- detected as Am, A7, or A6 variants |
| Bm7        | YES       | Bmin7      | Correctly detected |
| Cm7        | NO        | -          | Not found in unique chords |
| E9         | YES       | E9         | Correctly detected -- the 9th (F#) was identified |
| Bm6        | YES       | Bmin6      | Correctly detected -- G# (6th) distinguished |
| Dm7        | NO        | -          | Detected as D variants (D7, Dadd9, Dsus2) |
| Gmaj7      | YES       | Gmaj7      | Correctly detected |
| Em7        | YES       | Emin7      | Correctly detected |
| G#7        | NO        | -          | Detected as G#min7, G#hdim7 variants |
| G#maj7     | NO        | -          | Not detected |

**Accuracy: 5/10 (50%) on unique chord identification**

### Key Findings

1. **E9 DETECTED** -- This is the big win. The current BTC/V8 system labels E9 as E7 because it cannot distinguish the F# 9th from overtone noise in the spectrogram. The stem-aware system correctly identified the F# note in the guitar stem and assembled E9.

2. **Bm6 DETECTED** -- Another win. The distinction between Bm6 (G#) and Bm7 (A) was correctly resolved by detecting the actual G# pitch class in the guitar voicing.

3. **Over-segmentation** -- 397 chord events before merge is too many. The onset detection is too sensitive, creating micro-segments. The minimum segment duration (0.15s) needs to be increased to ~0.3s for production, or the pitch-class voting threshold needs tuning.

4. **64 unique chord types** -- Too many unique chords. The system detects passing tones and voicing artifacts as separate chord changes. Post-processing needs stronger consolidation.

5. **Bass root detection working** -- pyin detected 16,773 voiced frames out of 18,909, providing bass root information for most of the song.

---

## Test 2: VPS (Hetzner 4-CPU, 8GB RAM) - "One True Love" Stems

**Song:** One True Love BOUNCE 8-2.mp3
**Job:** 058e96c9-b015-4e3d-8621-ccce26f598bf
**Processing time:** ~47 seconds (CPU only)
**Backend:** TFLite + Basic Pitch on Python 3.10 (system Python with numpy 1.26.4)

### VPS-Specific Notes

- Basic Pitch required `numpy<2` for tflite-runtime compatibility (downgraded from 2.2.6 to 1.26.4 on system Python)
- The venv311 on VPS uses TensorFlow backend (works with its existing numpy)
- Processing time acceptable: ~47s for a 3.5min song on 4 CPU cores
- Memory usage well within 8GB limit

### Detection Results

47 unique chord types detected. Matched 5/10 Thunderhead reference chords (note: this comparison is imperfect since "One True Love" is a different song than Thunderhead -- the reference chords are for Thunderhead only).

The actual detected progression for "One True Love" shows:
- Primary chords: D, Bm, Em, A, G (consistent with key of D major)
- Extended chords detected: A9, Emin7, Bmin7, Gmaj7, Dmin7
- Bass root tracking working (pyin: 7,292 voiced frames out of 17,164)

---

## Pipeline Integration

### How It Works Now

1. When `detect_chords_for_job()` is called, it first checks if separated stems (guitar, bass, piano) are available
2. If stems exist, it runs **stem-aware detection** (Basic Pitch multi-pitch on each stem -> note assembly -> chord identification)
3. If stem-aware detection produces >= 3 chords, those are used (tagged as `detector_version: 'stem_aware'`)
4. If stem-aware detection fails or produces too few chords, it **falls back** to the BTC/V8 pattern-matching ensemble (existing system)
5. Results flow into the same `job.chord_progression` format used by chart_formatter and lead_sheet_generator

### Files Modified

| File | Change |
|------|--------|
| `backend/stem_chord_detector.py` | **NEW** -- Full stem-aware chord detection module |
| `backend/processing/transcription.py` | Modified `detect_chords_for_job()` to try stem-aware first, fall back to BTC/V8 |

### Deployed to VPS

Both files deployed via scp. Service restarted and running. New uploads will automatically use stem-aware detection when stems are available.

---

## Comparison: Stem-Aware vs Current BTC/V8

| Feature | BTC/V8 (current) | Stem-Aware (new) |
|---------|------------------|------------------|
| Input | Harmonic mix spectrogram | Individual separated stems |
| Method | Pattern match to chord labels | Multi-pitch note detection + assembly |
| E9 vs E7 | Cannot distinguish | CAN distinguish (detects F# 9th) |
| Bm6 vs Bm7 | Cannot distinguish | CAN distinguish (detects G# vs A) |
| Time resolution | 372ms/frame | 11.6ms/frame (32x finer) |
| Passing chords | Often missed | Detected via onset segmentation |
| Inversions | V8 has 144 slash chords | Bass root detection for inversions |
| Fallback | Essentia | BTC/V8 (full chain preserved) |
| VPS time | ~10s | ~47s (acceptable) |

---

## Known Issues / Next Steps

1. **Over-segmentation** -- Too many micro-chord-changes. Need to increase minimum segment duration and add stronger temporal smoothing.
2. **Chord vocabulary explosion** -- 64 unique types for one song is too many. Need a "rare chord filter" that collapses infrequent chords to simpler versions.
3. **Missing Am7, Cm7** -- These may be getting classified as major or as other minor variants. The pitch-class voting threshold may need tuning.
4. **G#7 / G#maj7 missed** -- Detected as G#min7 instead. The major/minor 3rd distinction needs more weight in the scoring.
5. **numpy compatibility on VPS** -- System Python needed numpy<2 for tflite-runtime. The venv311 with TensorFlow backend works fine.
