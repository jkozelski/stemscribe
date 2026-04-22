# Automatic Chord Recognition & Guitar Transcription Research
**Date:** 2026-03-16
**Purpose:** Survey state-of-the-art for StemScribe chord detection improvement

---

## Summary Comparison Table

| Library/Model | Chord Vocabulary | Accuracy (MajMin) | Stems OK? | pip install? | GPU Required? | MPS? | Active? | License |
|---|---|---|---|---|---|---|---|---|
| **CREMA** | 602 classes (maj/min/7th/ext/inv) | ~80% MIREX | Yes | `pip install crema` (or from GitHub) | No (CPU fine) | N/A | Low (last significant update ~2020) | BSD-like |
| **BTC (ISMIR19)** | Maj/min/7th (170 classes) | ~82% MajMin | Yes | No (clone repo) | Yes (CUDA) | Unlikely | Low (2019, no updates) | MIT |
| **Harmony Transformer v2** | Maj/min/7th + functional | ~83-85% MajMin | Yes | No (clone repo) | Yes (PyTorch) | Possible | Low (research code) | Not specified |
| **madmom** | Maj/min only (24+N) | ~78% MajMin | Yes | `pip install madmom` | No | N/A | Moderate (v0.16.1, issues active 2025) | BSD |
| **autochord** | 25 classes (maj/min only) | 67-70% | Yes | `pip install autochord` | No | N/A | Dead (inactive) | MIT |
| **chord-extractor** | Depends on Chordino dict | ~75% | Yes | `pip install chord-extractor` | No | N/A | Low (2021-22) | GPL-2.0+ |
| **Chordino/NNLS** | Configurable via chord.dict | ~75% | Yes | VAMP plugin (not pip) | No | N/A | Low | GPL |
| **Omnizart** | Harmony Transformer-based | ~80% | Yes | `pip install omnizart` | Optional (helps) | Unknown | Moderate (GitHub active) | MIT |
| **Basic Pitch** | N/A (MIDI, not chords) | 79% Fno (GuitarSet) | Yes | `pip install basic-pitch` | No | Yes | Active (Spotify) | Apache-2.0 |
| **MT3** | N/A (MIDI multi-track) | SOTA on Slakh2100 | Yes | Colab/T5X | Yes (TPU/GPU) | No | Moderate (Google Magenta) | Apache-2.0 |
| **MR-MT3** | N/A (MIDI, reduced leakage) | 66.4% onset F1 | Yes | Clone repo | Yes (GPU) | Unknown | Recent (2024) | Apache-2.0 |

---

## Detailed Notes

### 1. CREMA (Convolutional and Recurrent Estimators for Music Analysis)
- **Repo:** https://github.com/bmcfee/crema
- **Docs:** https://crema.readthedocs.io/
- **Paper:** McFee & Bello, ISMIR 2017 - "Structured Training for Large-Vocabulary Chord Recognition"
- **What it detects:** 602 chord classes including major, minor, 7ths, diminished, augmented, sus, slash/inversion chords. Based on extended Harte grammar. `N` = no-chord, `X` = out-of-gamut (power chords).
- **How it works:** Uses librosa for audio features, structured prediction with deep learning. Operates on CQT features.
- **Install:** `pip install crema` or `pip install -e git+https://github.com/bmcfee/crema.git`
- **GPU:** Not required, runs on CPU. Uses TensorFlow/Keras internally.
- **Stems:** Can process any audio input (files or numpy arrays), so isolated stems work fine.
- **Maintenance:** Last significant development ~2020. Version 0.2.0. Brian McFee (librosa author) is the maintainer but project appears low-activity.
- **Strengths:** Largest chord vocabulary of any easy-to-install library. Supports inversions. Well-documented API.
- **Weaknesses:** Aging codebase, TF dependency version conflicts possible. Not the highest accuracy on benchmarks.

### 2. BTC (Bi-directional Transformer for Chord Recognition)
- **Repo:** https://github.com/jayg996/BTC-ISMIR19
- **Paper:** Park et al., ISMIR 2019
- **What it detects:** Large vocabulary including 7ths. Uses CQT features from 10-second windows.
- **How it works:** Bi-directional multi-head self-attention, position-wise convolutional blocks, positional encoding, layer norm, dropout, FC layers.
- **Install:** Clone from GitHub, manual setup. Not on PyPI.
- **GPU:** Yes, CUDA required for practical use.
- **MPS:** Unlikely without code changes (older PyTorch).
- **Maintenance:** Last updated 2019. Research code only.
- **Note:** Despite the name, NOT affiliated with ByteDance. The ByteTransformer is a separate project.

### 3. Harmony Transformer v2
- **Repo:** https://github.com/Tsung-Ping/Harmony-Transformer-v2
- **Paper:** Chen & Su, TISMIR 2021 - "Attend to Chords"
- **What it detects:** Chord symbol recognition + functional harmony. Supports maj/min/7th chords.
- **How it works:** Encoder-decoder architecture. Encoder does chord segmentation, decoder recognizes progression. More accurate than BTC in 8/10 evaluation measures.
- **Install:** Clone from GitHub. Research code.
- **GPU:** PyTorch-based, GPU recommended.
- **MPS:** Potentially possible with PyTorch MPS backend (untested).
- **Maintenance:** Research code, minimal updates.
- **Strength:** Best accuracy among standalone transformer models for chord recognition. Also does functional harmony (Roman numeral analysis).

### 4. madmom
- **Repo:** https://github.com/CPJKU/madmom
- **Docs:** https://madmom.readthedocs.io/
- **PyPI:** https://pypi.org/project/madmom/
- **What it detects:** Major and minor chords only (24 + no-chord). Uses:
  - `DeepChromaChordRecognitionProcessor` - deep chroma + CRF
  - `CNNChordFeatureProcessor` - learned CNN features
  - `CRFChordRecognitionProcessor` - CNN features + CRF
- **Install:** `pip install madmom`
- **GPU:** Not required. NumPy-based with some Cython.
- **Stems:** Works on any audio input.
- **Maintenance:** v0.16.1 on PyPI. GitHub issues active through 2025. Python 3.10+ compatibility fork exists.
- **Strengths:** Well-established, good ecosystem, also does beat/tempo/onset detection. Reliable for basic chord detection.
- **Weaknesses:** Major/minor only -- no 7ths, no extensions. Not suitable for complex chord detection.

### 5. autochord
- **Repo:** https://github.com/cjbayron/autochord
- **Paper:** ISMIR 2021 Late-Breaking Demo
- **PyPI:** https://pypi.org/project/autochord/
- **What it detects:** 25 classes: 12 major, 12 minor, N (no-chord).
- **How it works:** NNLS-Chroma VAMP plugin for chroma features -> Bi-LSTM-CRF in TensorFlow.
- **Accuracy:** 67.33% test accuracy. 70.62% WCSR on major/minor triads (Billboard dataset).
- **Install:** `pip install autochord`
- **GPU:** Not required.
- **Maintenance:** DEAD. Inactive project, no longer maintained.
- **Verdict:** Too low accuracy, too limited vocabulary. Not suitable for StemScribe.

### 6. chord-extractor
- **Repo:** https://github.com/ohollo/chord-extractor
- **Docs:** https://ohollo.github.io/chord-extractor/
- **PyPI:** https://pypi.org/project/chord-extractor/
- **What it detects:** Whatever Chordino's chord.dict supports (configurable).
- **How it works:** Python wrapper around Chordino C++ VAMP plugin. Supports multiprocessing for batch extraction.
- **Install:** `pip install chord-extractor` (requires numpy pre-installed, VAMP plugin binaries).
- **GPU:** Not required.
- **License:** GPL-2.0+
- **Maintenance:** Last update 2021-22. Low activity.
- **Strengths:** Easy batch processing, multiprocessing support, extensible.
- **Weaknesses:** Inherits Chordino's limitations. GPL license is restrictive.

### 7. Chordino / NNLS Chroma
- **Repo:** https://github.com/c4dm/nnls-chroma
- **Website:** http://www.isophonics.net/nnls-chroma
- **What it detects:** Configurable via `chord.dict` file. Default includes common chord types.
- **How it works:** NNLS Chroma features -> chord profiles -> HMM/Viterbi smoothing.
- **Install:** VAMP plugin binary, not a Python package. Accessed through `vamp` Python bindings or `chord-extractor`.
- **Maintenance:** Low. v1.0 was reported as less accurate than previous versions.
- **Note:** The underlying engine for both `autochord` and `chord-extractor`.

### 8. Omnizart
- **Repo:** https://github.com/Music-and-Culture-Technology-Lab/omnizart
- **Docs:** https://music-and-culture-technology-lab.github.io/omnizart-doc/
- **Paper:** Wu et al., JOSS 2021
- **What it detects:** Chords (via Harmony Transformer), plus vocal, drums, beat, instruments.
- **How it works:** Harmony Transformer encoder-decoder. Encoder segments chords, decoder recognizes progression.
- **Install:** `pip install omnizart` then `omnizart download-checkpoints`
- **CLI:** `omnizart chord transcribe <audio.wav>`
- **GPU:** Optional but helpful. Docker image available with GPU support.
- **Maintenance:** Active-ish on GitHub. Last release may be dated but repo has activity.
- **License:** MIT
- **Strengths:** All-in-one toolkit. CLI is very convenient. Chord + beat + instruments in one package.
- **Weaknesses:** Heavy dependency chain. May have compatibility issues with newer Python versions. Chord vocabulary details unclear from docs.

### 9. ChordNet (Voice Leading, not ACR)
- **Paper:** Tsang & Aucouturier, JNMR 2004
- **What it is:** NOT a chord recognition model. It's for voice leading and harmonization -- learns to produce chord sequences, not detect them from audio.
- **Verdict:** Not relevant for StemScribe.

### 10. HarmonyNet
- **Status:** No specific model by this name found in chord recognition literature.
- **Verdict:** Does not exist as a standalone chord recognition tool.

---

## Cutting-Edge Research (2024-2026)

### LLM-Enhanced Chord Recognition (2025)
- **Paper:** "Enhancing Automatic Chord Recognition through LLM Chain-of-Thought Reasoning" (arXiv:2509.18700)
- **Approach:** 5-stage chain-of-thought framework using GPT-4o to coordinate multiple MIR tools (source separation, key detection, chord recognition, beat tracking).
- **Results:** +1-2.77% improvement on MIREX metric across three datasets.
- **Key insight:** LLMs as integrative bridges between specialized MIR tools. Converts audio-derived info to text for reasoning.
- **Relevance to StemScribe:** This approach is directly applicable. StemScribe already does stem separation. Could pipe separated stems + key detection + multiple chord detectors through an LLM for consensus/correction.

### Training on Artificially Generated Audio (2025)
- **Paper:** arXiv:2508.05878
- **Approach:** Transformer-based models trained on combinations of artificial audio multitracks, Schubert's Winterreise, and McGill Billboard datasets.
- **Finding:** Artificially generated training data can supplement human-composed music datasets for chord recognition.

### MARBLE Benchmark v2 (2025)
- **Website:** https://marble-bm.shef.ac.uk/
- **Repo:** https://github.com/a43992899/MARBLE-Benchmark
- **Chord task:** 421-class classification (35 chord types x 12 roots + none). Evaluates root, majmin, mirex, thirds, triads, sevenths metrics.
- **Datasets:** 1,217 songs from Isophonics, Billboard, MARL collections.
- **Relevance:** Best current benchmark for comparing chord recognition models.

### MIREX 2025 Audio Chord Estimation
- **Results:** https://music-ir.org/mirex/wiki/2025:Audio_Chord_Estimation_Results
- **Top system:** Wu et al. "Ensemble 2+4" transformer-based system with consistently highest CSR.
- **Evaluation vocabularies:** Root, MajMin, Sevenths (with inversions), Tetrads, MIREX.

---

## Guitar-Specific Transcription Models

### TabCNN
- **Repo:** https://github.com/andywiggins/tab-cnn
- **Paper:** Wiggins & Kim, ISMIR 2019
- **What it does:** Audio -> guitar tablature (fret positions per string).
- **Architecture:** CNN on CQT spectrogram. Outputs 6-string fret predictions (silence, open, frets 1-19).
- **Dataset:** GuitarSet (solo acoustic guitar).
- **Limitation:** Solo acoustic guitar only. No electric guitar, no full-mix.

### SynthTab (ICASSP 2024)
- **Repo:** https://github.com/yongyizang/SynthTab
- **Website:** https://synthtab.dev/
- **What it does:** Large-scale synthesized guitar tablature dataset from DadaGP (GuitarPro files). Pre-training improves TabCNN and TabCNNx4 models.
- **Key result:** Pre-training on SynthTab then fine-tuning on GuitarSet significantly improves cross-dataset generalization.
- **Relevance:** Best available approach for training guitar tab models with limited real data.

### FretNet (ICASSP 2023)
- **Repo:** https://github.com/cwitkowitz/guitar-transcription-continuous
- **Paper:** Cwitkowitz et al., ICASSP 2023
- **What it does:** Continuous-valued pitch contour streaming for polyphonic guitar tablature. Handles pitch bends, slides, vibrato.
- **Architecture:** Deepened TabCNN backbone with continuous pitch output layer.
- **Strength:** Can represent playing techniques (bends, slides) that discrete tab models miss.

### CRNN Guitar Transcription (2025)
- **Repo:** https://github.com/trimplexx/music-transcription
- **What it does:** Guitar audio -> tablature using Convolutional Recurrent Neural Network.
- **Performance:** 0.87 MPE F1-Score on GuitarSet (SOTA for models trained only on GuitarSet).
- **Architecture:** Multi-task learning predicting note onsets and fret positions simultaneously.

### GAPS Dataset + Model (August 2024)
- **Paper:** arXiv:2408.08653
- **What it is:** 14 hours of classical guitar audio-score aligned pairs from 200+ performers.
- **Key result:** Benchmark model achieves SOTA on GuitarSet in both supervised and zero-shot settings.

### Guitar-TECHS Dataset (ICASSP 2025)
- **Paper:** arXiv:2501.03720
- **What it is:** 5+ hours of electric guitar covering techniques, excerpts, chords, scales with diverse hardware.
- **Key result:** Augmenting GuitarSet training with Guitar-TECHS improved TabCNN disambiguation rate by 8+ points.
- **License:** CC-BY 4.0

### High Resolution Guitar Transcription (2024)
- **Paper:** arXiv:2402.15258
- **Approach:** Domain adaptation from high-resolution piano transcription model to guitar.
- **Result:** SOTA on GuitarSet in zero-shot context.

### MT3 (Google Magenta)
- **Repo:** https://github.com/magenta/mt3
- **Paper:** Gardner et al., ICLR 2022
- **What it does:** Multi-task multitrack music transcription. Arbitrary instrument combinations -> MIDI.
- **Architecture:** T5-based sequence-to-sequence transformer.
- **Guitar performance:** Dramatic improvement for "low-resource" instruments like guitar when using dataset mixing.
- **Limitation:** Outputs MIDI, not tablature. Requires TPU/GPU. Uses T5X/JAX framework (complex setup).
- **Still relevant:** Referenced in 2025 AMT Challenge as baseline.

### MR-MT3 (2024)
- **Repo:** https://github.com/gudgud96/MR-MT3
- **Paper:** arXiv:2403.10024
- **Improvement over MT3:** Memory retention mechanism reduces "instrument leakage" (notes assigned to wrong instrument). F1 improved from 61.6% to 66.4%, leakage ratio from 1.65 to 1.05.

### Basic Pitch (Spotify)
- **Repo:** https://github.com/spotify/basic-pitch
- **What it does:** Audio -> MIDI with pitch bend detection.
- **Guitar performance:** 79% Fno on GuitarSet.
- **Strengths:** Lightweight, runs faster than real-time, no GPU needed, MPS compatible.
- **Weaknesses:** MIDI output only (no tab). Known to produce poor results on distorted electric guitar. "Garbage for guitar" per user experience.
- **Note:** Community forks in 2025 add real-time DAW plugins, WASM ports, mobile apps.

---

## Key Questions Answered

### 1. Best open-source chord detection for complex chords (7ths, 9ths, dim, aug)?

**CREMA is the clear winner for vocabulary breadth** with 602 chord classes covering 7ths, extensions, inversions, and slash chords. No other easy-to-install library comes close.

For highest accuracy, the **Harmony Transformer v2** likely outperforms CREMA but requires manual setup and GPU.

**Recommended approach for StemScribe:**
1. Use CREMA as primary detector (large vocabulary, pip installable, CPU-only)
2. Cross-validate with madmom (for maj/min confidence)
3. Consider the LLM-ensemble approach from the 2025 paper: run multiple detectors, use an LLM to reconcile disagreements using music theory reasoning

### 2. Does running chord detection on isolated stems vs full mix improve accuracy?

**Mixed results.** Research from UW-Madison found that for deep neural network models, source separation may not significantly improve accuracy because the models are already deep enough to handle full mixes. However, for shallower models, separated stems can help by reducing spectral complexity.

**For StemScribe specifically:** Since we already have separated stems, it's worth running chord detection on:
- The harmonic stem (guitar/piano) -- best signal for chords
- The full mix as a secondary input
- Compare results; the stem-based detection should have fewer false positives from drums/vocals

The 2025 LLM paper explicitly used source separation as one of the MIR tools feeding into their ensemble, suggesting it adds value in a multi-tool pipeline.

### 3. Best approach for guitar-specific transcription in 2026?

**For tablature (fret positions):**
- SynthTab pre-training + fine-tuning on GuitarSet is the current best approach
- CRNN architecture (trimplexx/music-transcription) achieves 0.87 F1 on GuitarSet
- FretNet for continuous pitch (bends, slides)
- Guitar-TECHS dataset for electric guitar augmentation

**For chord detection from guitar stems:**
- CREMA on isolated guitar stem
- Consider fine-tuning a transformer model on guitar-specific data

### 4. End-to-end audio -> guitar tab models?

Yes, several exist but all have limitations:
- **TabCNN** -- solo acoustic only, simple architecture
- **SynthTab/TabCNNx4** -- improved with synthetic pre-training, still GuitarSet-focused
- **FretNet** -- handles pitch bends but still GuitarSet-based
- **CRNN (trimplexx)** -- current SOTA on GuitarSet alone
- **Klangio Guitar2Tabs** -- commercial API, not open source
- **AnthemScore 5** -- commercial desktop app

**No open-source model currently handles full-mix electric guitar -> tab reliably.** All academic models are trained on GuitarSet (solo acoustic) and degrade significantly on real-world recordings with distortion, effects, and multiple instruments.

---

## Recommendations for StemScribe (Ranked)

### Tier 1: Implement Now (High Value, Low Effort)

1. **Replace current chord detection with CREMA**
   - 602 chord classes vs whatever we have now
   - pip installable, CPU-only, works on stems
   - Immediate improvement in chord vocabulary
   - `pip install crema` -> call `crema.analyze(audio_file)`

2. **Run chord detection on isolated harmonic stems**
   - We already separate stems with Demucs
   - Feed guitar/piano stem to CREMA instead of full mix
   - Should reduce noise from drums/vocals

### Tier 2: Medium-Term (High Value, Medium Effort)

3. **Multi-detector ensemble with LLM reconciliation**
   - Run CREMA + madmom + Chordino on same audio
   - Use Claude/GPT to reconcile disagreements using music theory
   - The 2025 paper showed +2.77% improvement with this approach
   - StemScribe already has key detection -- feed that context to the LLM

4. **Integrate Omnizart for chord + beat alignment**
   - Chord detection + beat tracking in one package
   - Better temporal alignment of chord changes to beats

### Tier 3: Longer-Term (High Value, High Effort)

5. **Fine-tune a transformer chord model on stem-separated data**
   - Use Harmony Transformer v2 as base
   - Train on stem-separated audio from datasets
   - Could significantly improve accuracy on isolated instruments

6. **Guitar tablature from SynthTab pipeline**
   - Pre-train on SynthTab, fine-tune on GuitarSet
   - Add Guitar-TECHS for electric guitar coverage
   - Would give actual fret positions, not just chord labels

### Not Recommended

- **autochord** -- dead project, low accuracy
- **Basic Pitch for chords** -- outputs MIDI, not chords; poor on guitar
- **MT3** -- too complex to deploy (TPU/JAX), outputs MIDI not chords
- **BTC** -- superseded by Harmony Transformer v2

---

## Links & References

### Libraries & Repos
- CREMA: https://github.com/bmcfee/crema
- BTC: https://github.com/jayg996/BTC-ISMIR19
- Harmony Transformer v2: https://github.com/Tsung-Ping/Harmony-Transformer-v2
- madmom: https://github.com/CPJKU/madmom
- autochord: https://github.com/cjbayron/autochord
- chord-extractor: https://github.com/ohollo/chord-extractor
- Chordino: https://github.com/c4dm/nnls-chroma
- Omnizart: https://github.com/Music-and-Culture-Technology-Lab/omnizart
- Basic Pitch: https://github.com/spotify/basic-pitch
- MT3: https://github.com/magenta/mt3
- MR-MT3: https://github.com/gudgud96/MR-MT3
- TabCNN: https://github.com/andywiggins/tab-cnn
- SynthTab: https://github.com/yongyizang/SynthTab
- FretNet: https://github.com/cwitkowitz/guitar-transcription-continuous
- CRNN Guitar: https://github.com/trimplexx/music-transcription
- Awesome AGT: https://github.com/lucasgris/awesome-agt
- MARBLE Benchmark: https://github.com/a43992899/MARBLE-Benchmark

### Key Papers
- McFee & Bello, "Structured Training for Large-Vocabulary Chord Recognition" (ISMIR 2017) -- CREMA
- Park et al., "A Bi-Directional Transformer for Musical Chord Recognition" (ISMIR 2019) -- BTC
- Chen & Su, "Attend to Chords" (TISMIR 2021) -- Harmony Transformer v2
- Wiggins & Kim, "Guitar Tablature Estimation with a CNN" (ISMIR 2019) -- TabCNN
- Zang et al., "SynthTab: Leveraging Synthesized Data for GTT" (ICASSP 2024)
- Cwitkowitz et al., "FretNet" (ICASSP 2023)
- "Enhancing ACR through LLM Chain-of-Thought Reasoning" (arXiv:2509.18700, 2025)
- Gardner et al., "MT3: Multi-Task Multitrack Music Transcription" (ICLR 2022)
- "GAPS: A Large and Diverse Classical Guitar Dataset" (arXiv:2408.08653, 2024)
- "Guitar-TECHS: An Electric Guitar Dataset" (arXiv:2501.03720, 2025)
- "High Resolution Guitar Transcription via Domain Adaptation" (arXiv:2402.15258, 2024)
- "Training chord recognition models on artificially generated audio" (arXiv:2508.05878, 2025)

### Benchmarks & Competitions
- MARBLE: https://marble-bm.shef.ac.uk/
- MIREX 2025 ACE Results: https://music-ir.org/mirex/wiki/2025:Audio_Chord_Estimation_Results
- MIREX 2025 ACE Results (GitHub): https://github.com/ismir-mirex/ace-results
- 2025 AMT Challenge: https://ai4musicians.org/transcription/2025transcription.html
- Papers with Code - Chord Recognition: https://paperswithcode.com/task/chord-recognition
