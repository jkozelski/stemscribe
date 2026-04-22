# Vocab Validation Results

## Problem
UG (Ultimate Guitar) scraper sometimes returns chords for the WRONG SONG. When this happens, the BTC chord detector gets constrained to garbage vocabulary and outputs wrong chords.

**Example:** "The Time Comes" by Kozelski
- UG scraped: C, G, F (wrong song)
- Actual chords: F#m, A, B, D, E, G, C#m

## Solution: Vocab Validation

Added a validation step in `chord_detector_v10.py` that:

1. Runs BTC **unconstrained** (no vocab filter) to get raw chord predictions
2. Compares unconstrained chord roots against scraped vocab roots (duration-weighted)
3. If overlap < 30%, **discards the scraped vocab** and uses unconstrained BTC results
4. Logs a WARNING when vocab is discarded for monitoring
5. Also prevents discarded vocab from being passed to tuning detector (which would compute bogus offsets)

### Key Design Decisions
- **Songsterr chords bypass validation** -- they are more reliable than UG and go directly to constrained mode
- **Vocab validation runs BEFORE tuning detection** -- tuning detector uses vocab as reference, so bad vocab = bad tuning offset
- **When vocab is discarded, `vocab` is set to `None`** so downstream steps (tuning correction) don't re-lookup the bad vocab
- **30% overlap threshold** -- conservative enough to catch wrong-song scenarios (typically < 5% overlap) while not rejecting legitimate vocab (typically > 60%)

## Test Results

### "The Time Comes" (Guitar Stem)
```
Vocab validation: overlap=3.0%
  BTC roots: {'F#': 70.2s, 'B': 50.2s, 'A': 37.1s, 'E': 27.8s, 'D': 15.6s, 'G': 6.3s}
  vocab roots: ['C', 'F', 'G']
VOCAB MISMATCH DETECTED: overlap=3.0% < 30% threshold
Discarding scraped vocab -- using unconstrained BTC results
```

**Before fix:** C, G, F (wrong -- constrained to bad UG vocab, then tuning-shifted)
**After fix:** F#, B, A, E, D, G, C# (correct -- key=F#m)

### Full Test Suite
```
138 passed, 0 failed
```

## Files Modified
- `backend/chord_detector_v10.py`:
  - Added `_validate_vocab()` method on `ChordDetector` class
  - Restructured `detect()` to validate vocab before tuning detection
  - Added `vocab_discarded` flag to prevent re-lookup of bad vocab in `_apply_tuning_correction()`

## Implementation Location
- `_validate_vocab()`: Lines ~636-697 in chord_detector_v10.py
- `detect()` restructured flow: Lines ~798-882
- `_apply_tuning_correction()` updated signature: `vocab_discarded` parameter
