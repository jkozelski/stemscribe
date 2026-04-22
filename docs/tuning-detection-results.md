# Tuning Detection & Correction — Results

## Algorithm

### Primary: Vocabulary-Based Comparison
When a UG/Songsterr chord vocabulary is available (via `_lookup_vocab`), the detector tests all 12 transpositions of the detected chords against the reference vocabulary. A duration-weighted score combines root match (70%) and full chord match (30%). The transposition with the highest score is selected.

This is the most reliable method because it directly compares what the detector found against what the song's chords are known to be.

### Fallback: Pitch Analysis (librosa.pyin)
When no reference vocabulary is available, the detector uses `librosa.pyin` on the guitar stem to track pitch contours and measure systematic deviation from 440Hz-based equal temperament. If the median deviation exceeds 0.35 semitones, a tuning offset is inferred.

**Important finding:** Pitch analysis on the stem-separated guitar of Little Wing showed standard tuning (median deviation = 0.6 cents, effective A4 = 440.2Hz). This is because the stem separation preserves the sounding pitch, and the BTC model correctly detects the sounding pitch. The "problem" is not wrong detection -- it's that Eb-tuned guitars produce different sounding pitches than what guitarists call the chords. The vocabulary comparison method handles this correctly.

## Little Wing Test Results

### Detected A4 Frequency
- **Effective A4:** 440.2 Hz (essentially standard)
- **Tuning offset detected:** +1 semitone (via vocabulary comparison)
- **Tuning name:** Half-step down (Eb)
- **Detection method:** vocabulary
- **Vocab match score:** 0.97

### Before/After Chord Comparison

| Time | Before (sounding pitch) | After (guitarist notation) |
|------|------------------------|---------------------------|
| 1.1s | D#m | Em |
| 3.8s | B | C |
| 4.6s | F# | G |
| 8.1s | G#m | Am |
| 9.4s | B | C |
| 11.7s | D#m | Em |
| 14.1s | C# | D |
| 15.3s | A#m | Bm |
| 20.3s | B | C |
| 22.2s | F# | G |
| 26.3s | B | C |
| 27.4s | G#m | Am |
| 28.1s | C# | D |
| 30.9s | F# | G |
| 34.9s | D#m | Em |
| 38.7s | F# | G |
| 42.1s | G#m | Am |
| 44.0s | B | C |
| 45.6s | D#m | Em |

Expected chords for Little Wing (from UG): Em, G, Am, Bm, Bbm7, C, F, D

After correction, the output matches the expected chord vocabulary perfectly -- Em, G, Am, C, D, Bm all appear correctly.

## Test Results

All 138 existing tests pass with no regressions:
```
======================== 138 passed, 1 warning in 2.53s ========================
```

## Integration Points

- **Module:** `/backend/tuning_detector.py` -- standalone tuning detection & correction
- **Pipeline integration:** `/backend/processing/transcription.py` in `detect_chords_for_job()`
- Tuning correction runs AFTER chord detection + ensemble but BEFORE storing results
- Tuning metadata stored on `job.tuning_info` (offset, name, method, effective A4)
- Each chord dict gets `tuning_offset` field when correction was applied

## Limitations

1. **Vocabulary required for reliability** -- Without UG/Songsterr vocab, the pitch analysis fallback is unreliable because stem separation preserves the original sounding pitch (the guitar IS tuned down, so pitches ARE lower).
2. **Songs without UG data** will only get pitch-based detection, which may miss deliberate alternate tunings.
3. **Capo songs** are the inverse problem (capo raises pitch) and would need separate handling.
