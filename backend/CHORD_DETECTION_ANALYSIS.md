# StemScriber Chord Detection Analysis

**Date:** 2026-03-09
**Test case:** "The Time Comes" by Kozelski
**Ground truth (songwriter's chart):** F#m, A, Bm, E, B, D, C#m, G — key of F#m / A major
**StemScriber output:** Key of G, chords mostly G and C with one F — completely wrong

---

## 1. Which Detector Is Actually Running

The import chain in `dependencies.py` (lines 179-206) tries detectors in this order:

1. **v10 (BTC)** — preferred, loads `chord_detector_v10.py`
2. v8 (Transformer, 337 classes) — fallback
3. v7 (Transformer, 25 classes) — fallback
4. basic (template matching) — last resort

**v10 is the active detector.** It loads the BTC (Bi-directional Transformer for Chord Recognition) model from `~/stemscribe/btc_chord/`. A fine-tuned checkpoint exists at `backend/training_data/btc_finetune/checkpoints/btc_finetuned_best.pt` and is loaded preferentially over the original.

The BTC model outputs 170 chord classes (12 roots x 14 qualities + N + X) using large vocabulary mode. This covers: min, maj, dim, aug, min6, maj6, min7, minmaj7, maj7, 7, dim7, hdim7, sus2, sus4.

---

## 2. How Chord Detection Runs in the Pipeline

`detect_chords_for_job()` in `processing/transcription.py` (line 656):

1. **Mixes harmonic stems** (guitar, piano, bass, other) into a single WAV file at 22050 Hz — this is a good idea in theory (removes drums/vocals that confuse detection).
2. **Passes artist/title** to the detector for vocabulary-constrained mode (UG scrape).
3. Calls `detector.detect(analyze_path, artist=artist, title=title)`.
4. Key is detected from the chord histogram using Krumhansl-Kessler profiles.

For "The Time Comes" by Kozelski — this song is almost certainly NOT on Ultimate Guitar, so vocabulary constraint mode would not activate. The detector runs unconstrained.

---

## 3. Root Cause Analysis: Why F#m Becomes G

### Problem 1: CQT Frequency Resolution and Bin Quantization

The BTC model uses CQT with 144 bins / 24 bins-per-octave / hop_length 2048 at 22050 Hz. With 24 bins per octave, each bin spans 50 cents (half a semitone). This is adequate resolution in theory, but:

- **F# and G are adjacent semitones** (only 100 cents apart). If the CQT energy distribution slightly favors one bin over the other due to intonation, overtone structure, or tuning, the wrong root is picked.
- **Guitar tuning matters.** If the guitar is tuned even 10-20 cents flat or uses non-standard tuning, F# energy bleeds into the G bin. Many live recordings and home demos have imperfect tuning.
- The BTC model was trained on the Billboard/McGill datasets which are predominantly commercial, well-tuned recordings. A home demo or indie recording may have tuning characteristics the model hasn't seen.

### Problem 2: Key Detection from Chord Histogram Is Fragile

The `_detect_key_from_chords()` method (line 478 of v10) builds a duration-weighted root histogram from detected chords, then correlates against Krumhansl-Kessler profiles. This creates a **circular dependency**:

1. If chord detection already gets roots wrong (F# → G, C# → C), the histogram is wrong.
2. A wrong histogram produces a wrong key.
3. The wrong key reinforces the wrong chord interpretation.

The Krumhansl-Kessler correlation approach also has known weaknesses:
- It was derived from psychological experiments on Western tonal music, not from signal analysis.
- F#m and G major have very different pitch class distributions, but if the roots are already shifted by one semitone, the correlation picks up the shifted key.

### Problem 3: BTC Model Class Imbalance

The BTC model was trained primarily on the Billboard and McGill datasets. Research shows severe class imbalance: **major and minor triads dominate the training data** (C, G, D, Am, Em are among the most frequent). Chords like F#m, C#m, Bm are significantly underrepresented. The model has a statistical bias toward "simpler" sharp-free chords.

This is consistent with the observed failure: the model collapses F#m → G (major), C#m → C (major), Bm → B or omits it. It systematically drops sharps/flats and converts minor to major.

### Problem 4: No Tuning Reference / A440 Assumption

Neither the BTC model nor any of the fallback detectors perform tuning estimation before chord detection. The CQT is computed assuming A4 = 440 Hz. If the recording's reference pitch is even slightly off (A = 437 Hz is common for older recordings, some artists tune to A = 432 Hz), every detected root shifts.

This is the single most likely explanation for a systematic one-semitone shift (all sharps become naturals): **the recording is tuned slightly flat, causing F# energy to land in the F or G bins.**

### Problem 5: Harmonic Stem Mix May Be Noisy

The stem mixing in `detect_chords_for_job()` sums guitar + piano + bass + "other" stems. The "other" stem is explicitly skipped in MIDI transcription because it's mixed content, but it IS included in the chord detection mix. This can introduce noise, synth pads, or other non-harmonic content that muddies the chroma/CQT features.

### Problem 6: No Temporal Smoothing or Musical Heuristics

The BTC model outputs per-frame predictions but the v10 code does minimal post-processing — it just consolidates consecutive identical frames into regions. There is no:
- Transition probability model (certain chord changes are far more likely than others)
- Key-aware filtering (once a key is estimated, reject chords that don't fit the key's diatonic set)
- Beat-aligned chord change detection (chords almost always change on beats, not between them)

---

## 4. State of the Art in Chord Detection (2025-2026)

### Current Best Approaches

| Approach | Accuracy (MIREX) | Notes |
|----------|-----------------|-------|
| **BTC (original, 2019)** | ~80% majmin | Baseline transformer model |
| **BTC-FDAA-FGF (2025)** | ~83% majmin | Feature fusion improvement on BTC |
| **ChordFormer (2025)** | +2% frame, +6% class | Conformer blocks (CNN + Transformer) |
| **LLM Chain-of-Thought (2025)** | +1-2.7% MIREX | GPT-4o post-processing of BTC output |
| **Deep pitch-class CTC (2021)** | Competitive | Multi-label CTC loss, better feature learning |

### Key Insights from Recent Research

1. **Tuning estimation is critical.** Top-performing MIREX systems pre-estimate tuning and compensate before feature extraction.
2. **CQT > Chroma for neural models.** Raw CQT features (like BTC uses) preserve more harmonic information than collapsed chroma vectors. StemScriber's fallback detectors (v7, v8, basic) use chroma — this is strictly worse.
3. **Source separation helps.** A 2020 study showed that running chord detection on separated stems (specifically harmonic stems without drums/vocals) significantly improves accuracy. StemScriber already does this, which is good.
4. **Class imbalance is the #1 unsolved problem.** Rare chords (F#m, C#m, Bbm, etc.) are systematically misclassified. Data augmentation via pitch shifting during training partially addresses this.
5. **Post-processing matters.** HMM/CRF temporal smoothing, key-constrained filtering, and beat-aligned chord changes all provide 2-5% accuracy gains on top of any neural model.

### Recommended Libraries/Models

- **BTC (current, already integrated):** Solid foundation. The fine-tuned version should be better than vanilla.
- **Autochord (Python, NNLS-Chroma + Bi-LSTM-CRF):** Only 25 classes, not an upgrade.
- **madmom:** Strong beat/downbeat detection but no dedicated chord model. Useful for beat-aligned post-processing.
- **Essentia ChordsDetection:** Already integrated as fallback in v10. Decent but not state-of-the-art.
- **ChordFormer:** Not yet open-sourced as of this writing. Worth watching.
- **Pitch-shift augmented BTC:** Re-training BTC with aggressive pitch augmentation (-6 to +6 semitones) would directly address the class imbalance and tuning sensitivity issues.

---

## 5. Specific Recommendations

### Quick Wins (days, not weeks)

#### A. Add Tuning Estimation Before CQT (HIGH IMPACT)
In `chord_detector_v10.py`, before computing CQT features, estimate the recording's tuning offset using `librosa.estimate_tuning()` and compensate:

```python
# In _detect_btc(), after loading audio:
tuning = librosa.estimate_tuning(y=original_wav, sr=sr)
logger.info(f"Estimated tuning offset: {tuning:.2f} semitones")
# Shift audio to compensate
if abs(tuning) > 0.05:
    original_wav = librosa.effects.pitch_shift(original_wav, sr=sr, n_steps=-tuning)
```

This single change would likely fix the systematic semitone-shift problem for "The Time Comes" and similar recordings.

#### B. Remove "other" Stem from Harmonic Mix
In `detect_chords_for_job()` (transcription.py line 674), remove `'other'` from the harmonic stems list:

```python
# Change this:
for stem_key in ('guitar', 'guitar_left', 'piano', 'bass', 'other'):
# To this:
for stem_key in ('guitar', 'guitar_left', 'piano', 'bass'):
```

#### C. Add Key-Constrained Post-Processing
After BTC detection, if chords outside the detected key's diatonic set appear, check if shifting them by one semitone puts them in-key. This catches systematic tuning errors:

```python
def _correct_tuning_drift(self, events, key):
    """If most chords are one semitone off from a diatonic set, shift them."""
    # Build expected diatonic roots for the key
    # Check if shifting all roots by +1 or -1 semitone improves diatonic fit
    # If so, apply the correction
```

#### D. Lower Confidence Threshold for Constrained Mode
Currently, unconstrained BTC outputs get confidence=0.8 hardcoded (line 441-442 of v10). This is misleading — actual model confidence should come from softmax probabilities, not a fixed value. Computing real confidence would help downstream consumers know when to trust the output.

### Medium-Term Improvements (weeks)

#### E. Beat-Aligned Chord Detection
Use `librosa.beat.beat_track()` or madmom's beat tracker to find beat positions, then snap chord changes to the nearest beat. Chords virtually never change mid-beat in popular music.

#### F. Retrain BTC with Pitch Augmentation
The original BTC training used -5 to +6 semitone augmentation. Retrain the fine-tuned model with:
- More aggressive augmentation (include quarter-tone shifts: -0.5, -0.25, +0.25, +0.5 semitones)
- Oversampling of underrepresented chord classes (F#, C#, Bb, Eb roots)
- Include the UG-scraped chord database as weak supervision labels

#### G. HMM/CRF Temporal Smoothing
Add a Hidden Markov Model or CRF layer after BTC prediction to enforce musically plausible chord transitions. The BTC repo already has `crf_model.py` and `train_crf.py` — these could be leveraged.

### Longer-Term (months)

#### H. Ensemble Detection
Run BTC + Essentia in parallel, compare outputs, and use agreement as a confidence signal. When they disagree, use music-theory heuristics to break ties.

#### I. LLM Post-Processing
Following the 2025 research, use an LLM to review detected chord sequences and correct implausible progressions. For example, "G C F in key of G" could be flagged as unusual (F is not diatonic to G major) and corrected to "G C F#m" or similar.

#### J. User Correction Feedback Loop
Allow users to correct detected chords in the UI. Store corrections and use them to fine-tune the model over time. This is the highest-ROI long-term investment for accuracy.

---

## 6. Summary of the "The Time Comes" Failure

The most likely failure chain:

1. **Recording tuning is slightly off A440** (even 15 cents flat shifts F# toward F/G in CQT bins)
2. **BTC model has class imbalance bias** toward natural-note chords (G, C, F are far more common in training data than F#, C#)
3. **No tuning compensation** — the CQT assumes perfect A440
4. **Wrong chords produce wrong key** — duration-weighted histogram of {G, C, F} correlates to G major
5. **No post-processing catches the error** — no key-aware filtering, no beat alignment, no transition probability model

**The single highest-impact fix is adding `librosa.estimate_tuning()` before CQT feature extraction.** This would likely correct the systematic semitone shift that turns F#m → G and C#m → C.

---

## References

- [BTC: A Bi-directional Transformer for Musical Chord Recognition (2019)](https://arxiv.org/abs/1907.02698)
- [BTC-FDAA-FGF: Feature Fusion Based Automatic Chord Recognition (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0045790625004987)
- [Enhancing ACR through LLM Chain-of-Thought Reasoning (2025)](https://arxiv.org/abs/2509.18700)
- [Training Chord Recognition Models on Artificially Generated Audio (2025)](https://arxiv.org/html/2508.05878)
- [MIREX 2025 Audio Chord Estimation Results](https://music-ir.org/mirex/wiki/2025:Audio_Chord_Estimation_Results)
- [librosa.estimate_tuning documentation](https://librosa.org/doc/main/generated/librosa.estimate_tuning.html)
- [Improving Balance in ACR with Random Forests (2022)](https://eurasip.org/Proceedings/Eusipco/Eusipco2022/pdfs/0000244.pdf)
