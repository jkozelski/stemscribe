# FretNet Integration Research

**Date:** 2026-04-05
**Status:** Research complete — no code changes made
**Repo:** [github.com/cwitkowitz/guitar-transcription-continuous](https://github.com/cwitkowitz/guitar-transcription-continuous)
**Paper:** ICASSP 2023 — [arXiv:2212.03023](https://arxiv.org/abs/2212.03023)
**License:** MIT

---

## 1. Current State: midi_to_gp.py

The existing pipeline: **Basic Pitch → MIDI → midi_to_gp.py → GP5 file**

`midi_to_gp.py` (697 lines) contains `FretMapper`, a heuristic scorer that takes MIDI note numbers and guesses string+fret positions by weighing:

- **Position zone** — penalizes high frets (13+ costs +15), favors open and low positions
- **String preference** — penalizes outer strings (distance from middle × 2)
- **Fret continuity** — rewards staying near previous fret, penalizes jumps >7
- **String continuity** — rewards same/adjacent strings

A separate **lead mode** activates on CC#20 markers — uses 6-note position history, stronger same-string preference, and bend awareness.

### Key limitations
1. No actual string/fret awareness — pure heuristic on MIDI pitches
2. `map_chord` exists but is **never called** by the main converter
3. No dotted/triplet durations, no ties, hardcoded 4/4
4. Articulations (bends, slides) only work when CC#20 markers exist (rare from Basic Pitch)
5. Single voice per measure, no polyphonic voice separation

---

## 2. What FretNet Does

FretNet is an end-to-end **audio → tablature** CNN that produces three simultaneous outputs:

| Output | Description |
|--------|-------------|
| **Discrete tablature** | Per-frame string (1-6) + fret (0-19) assignments |
| **Continuous pitch deviation** | Float value per activation showing bend/vibrato offset from nominal fret pitch (±1 semitone default) |
| **Onset detection** | Binary onset predictions per string/pitch |

This is the **only open-source model** that outputs continuous pitch contours with string+fret anchoring. This is exactly what's needed for bend/slide/vibrato detection in GP5 output.

### Architecture
- 3 CNN blocks (16/32/48 filters) + BatchNorm + ReLU + MaxPool + Dropout
- 3 linear prediction heads (tablature, pitch deviation, onset)
- Input: HCQT features — 6 harmonics, 144 bins (4 octaves from E2), 36 bins/octave
- Audio: mono WAV @ 22050 Hz, hop length 512 (~23ms/frame)
- Small model: ~2-5 MB parameters

### Dependencies
```
Python >= 3.8
PyTorch >= 1.11.0
amt-tools >= 0.3.1        (Cwitkowitz's AMT framework)
guitar-transcription-with-inhibition  (his prior TabCNN+ work)
librosa >= 0.9.1
sacred >= 0.8.2            (experiment management)
jams >= 0.3.4, muda >= 0.4.1
```

### GPU Requirements
- **Inference:** Runs fine on CPU. Any GPU speeds it up but is not required.
- **Training:** A single consumer GPU (GTX 1060+) is sufficient. Batch size 30, 2500 iterations.

---

## 3. Critical Blockers

### No pre-trained weights available
The repo has no releases, no hosted checkpoints, no download links. The inference example hardcodes `~/Downloads/FretNet/models/fold-0/model-2000.pt`. **You must train from scratch.**

### Trained only on GuitarSet
- 360 clips of solo acoustic guitar, 6 players
- Clean mic'd recordings — no electric guitar, no effects, no separation artifacts
- Domain gap to separated stems (BS-RoFormer output) is significant
- Fine-tuning or data augmentation would be needed for production use

### Dependency chain is heavy
`amt-tools` and `guitar-transcription-with-inhibition` are research-grade packages from the same author. Not widely used or maintained. Integration risk.

---

## 4. Integration Options

### Option A: Replace midi_to_gp.py entirely (NOT recommended)
- FretNet handles audio → tablature end-to-end
- But: no pre-trained weights, GuitarSet-only training, lower discrete accuracy than trimplexx
- Would lose all existing articulation/GP5 infrastructure

### Option B: Post-processor for articulation data (RECOMMENDED)
Per the upgrade plan, use FretNet **after** trimplexx CRNN for note detection:

```
Guitar stem (audio)
  ├── trimplexx CRNN → discrete string+fret notes (high accuracy)
  └── FretNet → continuous pitch contours per string
       ↓
  Merge: anchor FretNet pitch deviations to trimplexx note events
       ↓
  midi_to_gp.py → GP5 with real bend/slide/vibrato data
```

This leverages FretNet's unique strength (pitch contours) while using trimplexx for what it's better at (discrete tab accuracy).

### Option C: Pitch contour extraction only (PRAGMATIC)
Strip FretNet down to just the continuous pitch head. Use existing transcription for note detection, run FretNet's HCQT + pitch deviation head on identified note regions to detect:
- Bends (sustained deviation > threshold)
- Vibrato (oscillating deviation)
- Slides (monotonic deviation change)

This could work even with approximate weights since you're looking for relative pitch movement, not absolute accuracy.

---

## 5. Comparison to Alternative Approaches

| Model | Input | Output | Tab F1 (GuitarSet) | Pitch Contours | Weights Available |
|-------|-------|--------|---------------------|----------------|-------------------|
| **FretNet** | Audio | String+fret+pitch | 0.727 | Yes | No (train from scratch) |
| **trimplexx CRNN** | Audio | String+fret | 0.857 | No | No (train from scratch) |
| **TabCNN** | Audio | String+fret | ~0.65 | No | No |
| **A\*-Guitar** | MIDI | String+fret | N/A (search) | No | N/A (algorithm) |
| **MIDI-to-Tab** (ISMIR 2024) | MIDI | String assignments | User-study preferred over A\* | No | Unknown |
| **Fretting-Transformer** (2025) | MIDI | String+fret | Outperforms A\* on GuitarToday | No | Unknown |

### Key insights
- **A\*-Guitar** is a graph-search algorithm, not a neural model. Takes MIDI as input and optimizes fret assignments for playability (minimize hand movement, respect span constraints). Two-stage: needs AMT first. Could replace FretMapper heuristic with zero training data needed.
- **Fretting-Transformer** uses T5 encoder-decoder on MIDI → tab. Best discrete accuracy in recent benchmarks. Two-stage like A\*. No published weights found.
- **MIDI-to-Tab** uses masked language modeling for string assignment. Interesting but also two-stage and no weights.
- For StemScriber, the two-stage approaches (A\*, Fretting-Transformer) are worth considering as a **FretMapper replacement** — they optimize playability from MIDI, which is exactly what FretMapper tries to do heuristically.

---

## 6. Recommendations

### Priority order for StemScriber tab quality

1. **Phase 1 (highest ROI):** Replace FretMapper heuristic with A\*-Guitar search
   - Algorithm-based, no training needed, no weights to find
   - Directly improves fret position quality from existing MIDI output
   - Respects hand span constraints and minimizes position jumps
   - Could be implemented as a drop-in replacement for `midi_note_to_fret()`

2. **Phase 2:** Deploy trimplexx CRNN (per upgrade plan)
   - Eliminates Basic Pitch → MIDI → fret-guess pipeline for guitar
   - Direct audio → string+fret, much higher accuracy (0.857 vs FretMapper heuristic)
   - Requires training on GuitarSet (~3-4 hours)

3. **Phase 3:** Add FretNet pitch contour layer
   - Only after Phase 2 is working — FretNet adds articulation data on top
   - Train on GuitarSet, evaluate on separated stems
   - Feed pitch deviations into GP5 converter for bend/slide/vibrato marks

4. **Phase 4:** Evaluate Fretting-Transformer
   - If/when weights become available, could be the best discrete tab model
   - T5-based, likely higher accuracy than trimplexx on complex passages

### What NOT to do
- Don't try to use FretNet as the primary transcription model (lower discrete accuracy)
- Don't skip trimplexx in favor of FretNet (wrong tool for note detection)
- Don't ignore A\*-Guitar — it's the fastest path to better fret positions with zero ML overhead

---

## 7. FretNet Integration Sketch (Phase 3)

If proceeding with FretNet as an articulation layer:

```python
# Conceptual flow — not production code

# 1. Transcribe with trimplexx
notes = trimplexx_transcribe(guitar_stem)  # → list of (time, string, fret, duration)

# 2. Run FretNet pitch contour extraction
hcqt = compute_hcqt(guitar_stem, sr=22050)
tablature, pitch_deviations, onsets = fretnet.predict(hcqt)

# 3. For each trimplexx note, sample FretNet pitch deviation
for note in notes:
    frames = time_to_frames(note.start, note.end)
    deviations = pitch_deviations[note.string, note.fret, frames]
    
    if is_bend(deviations):      # sustained offset > 0.3 semitones
        note.articulation = 'bend'
        note.bend_amount = max(deviations)
    elif is_vibrato(deviations):  # oscillation > 3Hz
        note.articulation = 'vibrato'
    elif is_slide(deviations):    # monotonic change > 0.5 semitones
        note.articulation = 'slide'

# 4. Convert to GP5 with articulation data
convert_to_gp5(notes)  # existing midi_to_gp.py already supports these effects
```

### Training requirements
- Download GuitarSet (~1.6 GB)
- 6-fold cross-validation, ~2500 iterations per fold
- Estimated training time: 1-2 hours on A10G (Modal)
- Consider data augmentation with separated stems for domain adaptation

---

## References

- FretNet paper: Cwitkowitz et al., "Guitar Tablature Transcription with a Continuous Representation of Pitch" (ICASSP 2023)
- trimplexx CRNN: [github.com/trimplexx/music-transcription](https://github.com/trimplexx/music-transcription)
- A*-Guitar: Yazawa et al., "Audio-to-Score Alignment for Guitar Based on A* Search"
- MIDI-to-Tab: Toyama et al. (ISMIR 2024) — [arXiv:2408.05024](https://arxiv.org/abs/2408.05024)
- Fretting-Transformer: (2025) — [arXiv:2506.14223](https://arxiv.org/html/2506.14223v1)
- GuitarSet: Xi et al. (ISMIR 2018)
- TabCNN: Wiggins (ISMIR 2019)
