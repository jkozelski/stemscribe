# Polyphonic Pitch → Chord Recognition Research — Findings

**Date:** 2026-05-25
**Researcher:** Fresh Claude Code session (no prior project context outside the two memory files cited in the brief)
**Brief:** `~/stemscribe/docs/polyphonic-chord-detection-research-brief-2026-05-25.md`

---

## TL;DR (3 sentences)

The hypothesis — feeding the **guitar stem's Basic Pitch MIDI into a chord inferer** — is *unlikely* to bump F1 meaningfully and is the wrong shape; the published 1–3% gains from "stem-aware" ACR (ChordCoT 2025, APSIPA 2025) come from a different shape: using the **bass stem to correct roots** and using **drums/vocals-removed mixes** as alternate ACR inputs, not from running poly transcription on an isolated guitar stem. The single cheapest, most-precedented experiment with a real chance to move headline accuracy 1–2 points is a **bass-stem root corrector** (htdemucs already produces the bass stem in your pipeline; monophonic pitch tracking on bass is robust; override V1 librosa roots when the bass disagrees and key context confirms). However, this overlaps your `project_chord_dead_ends_ledger.md` "accuracy chase permanently closed" verdict: even a perfect detector caps at ~0.70 on your honest v2 scorer, so the strategic recommendation is **don't ship this for launch**; revisit only as a post-launch "Stage 2" polish, scoped exactly like ChordCoT's bass-correction step.

---

## Per-question answers

### Q1 — State of the art in ACR (2024–2026)

**Top line: the field has plateaued.** Headline gains in 2024–2025 papers are 1–3% on MIREX metrics, not the step-changes of 2017–2020.

Current ceilings on standard benchmarks (Isophonics, UsPop2002, Billboard, JAAH):

| Metric | Approximate ceiling | Notes |
|---|---|---|
| **Root only** | ~83–84% | BTC, ChordFormer, ChordCoT |
| **Maj-Min** | ~83% (WCSR) | BTC reports 83.1 maj-min |
| **MIREX (frame-wise)** | ~80–86% | dataset-dependent |
| **Sevenths / large-vocab** | ~72% | Drop-off when 170+ chord classes |
| **Tetrads** | ~60–65% | Sparse training labels |

**Approach families:**
- **Full-mix harmonic (librosa CQT + templates / HMM)** — what you have in prod. Cheap, F1 ≈ 0.71 on your audit set, ~0.83 framewise on Isophonics-class benchmarks. This is the floor reference.
- **End-to-end neural** —
  - **BTC** (Park et al. 2019, ISMIR) — bi-directional transformer, MIT code, established baseline. https://github.com/jayg996/BTC-ISMIR19
  - **Jiang large-vocab structured** (ISMIR 2019) — chord-structure decomposition for 301-class output. Pretrained models at https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition
  - **ChordFormer** (Akram et al., arXiv:2502.11840, Feb 2025) — Conformer architecture, 2% framewise / 6% class-wise gain on large-vocab. Code not yet released on Papers with Code as of search date.
  - **BTC-FDAA-FGF** (2025) — HCQT features + frequency-domain adaptive attention. 1.2–2.2% MIREX gain.
  - **ChordMini** (arXiv:2602.19778) — pseudo-labeled training with BTC teacher → student. MIT license. 2E1D encoder. https://github.com/ptnghia-j/ChordMini
- **LLM-coordinated multi-tool (newest direction):**
  - **ChordCoT** (Chang et al., arXiv:2509.18700, Sept 2025) — GPT-4o as music-theoretical coordinator. **5 stages: (1) MSS-aware ACR on full-mix + drums-removed + drums-vocals-removed, (2) bass-stem root correction, (3) key correction, (4) anomaly detection, (5) beat alignment.** Gains: 1.06–2.77% MIREX over already-strong baselines (79.5% → 81.2% / 80.1% → 81.0% / 83.3% → 86.1%). Code at https://github.com/WildHoneyPie/ChordCoT (license not visible in paper).
- **Synthetic-data training:**
  - **AAM (Artificial Audio Multitracks)** + Transformer — Majchrzak & Mańdziuk, Aug 2025 (arXiv:2508.05878). Synth-only training is viable for pop on Root/MajMin/CCM. Relevant because your own training data legal file already approves Slakh2100 (CC BY 4.0) — same family.

**Reframe (consistent with your dead-ends ledger conclusion):** the published ceiling is a *labeling problem*, not an audio problem. "Dsus2 vs D" disagreements between models and ground truth are the bulk of the residual error — they read as "AI wrong" to a guitarist but as "essentially correct" to the benchmark. Your "guitar-voicing identity" framing matches the field literature exactly.

**Sources:**
- ChordCoT: https://arxiv.org/abs/2509.18700 / https://arxiv.org/html/2509.18700v1
- ChordFormer: https://arxiv.org/abs/2502.11840
- BTC: https://arxiv.org/abs/1907.02698 / https://github.com/jayg996/BTC-ISMIR19
- ChordMini: https://github.com/ptnghia-j/ChordMini / https://www.chordmini.me/about
- BTC-FDAA-FGF: https://www.sciencedirect.com/science/article/abs/pii/S0045790625004987
- Synthetic training: https://arxiv.org/abs/2508.05878
- Field review baseline: https://archives.ismir.net/ismir2016/paper/000178.pdf

---

### Q2 — Stem-based chord recognition (the core hypothesis)

Two recent papers tested it directly. They reach **opposite conclusions** depending on *how* the stems are used.

**Result A — separation hurts (the cautionary tale):**
- Daniel Ko, UW-Madison, "Automatic Chord Recognition by Music Source Separation" (~2020). Used Demucs to separate, combined "other + bass" stems into one audio track, fed to a Jiang-style CRNN. **Demucs-separated input performed *worse* than the original full-mix** across all metrics (Root, Thirds, MajMin, Triads, Sevenths, Tetrads, MIREX). Cause inferred: separation artifacts in the spectrogram-based input degraded the recognizer more than the cleaner harmonic content helped. https://ko28.github.io/chord-transcription/

**Result B — selective stem use helps:**
- "Accuracy Improvement of Automatic Chord Recognition with Source Separation Preprocessing" (APSIPA 2025) — uses MSS to separate, **amplifies the "other" stem volume**, then recombines into one track. Reports error reduction in "over 2,000 frames"; specific F1 not extractable from the proceedings PDF (encoding issue). http://www.apsipa.org/proceedings/2025/papers/APSIPA2025_P307.pdf / https://ieeexplore.ieee.org/document/11249321/
- **ChordCoT** (cited in Q1) — uses **drums-removed and drums+vocals-removed mixes** as alternate ACR inputs in Stage 1, then GPT-4o picks the most confident one. Plus Stage 2 uses **isolated bass stem** for root correction. Net: 1.06–2.77% MIREX gain.

**The pattern: nobody publishes "isolated guitar stem → poly transcription → MIDI → chord inference"** as a productive ACR signal. The winning recipes are:
1. Bass stem (monophonic-friendly) → robust root estimate → correct the polyphonic recognizer's root errors.
2. Drum/vocal-removed mix (still polyphonic, less noisy) → feed to the *same* recognizer → ensemble the result.

**Why guitar-stem-MIDI-as-chord-signal is unattractive:**
- htdemucs guitar stem is acoustically noisy (you already know this — Jeff's project state notes "other stems too noisy" for MusicXML and only ships bass MusicXML).
- Basic Pitch on a noisy isolated stem will produce a noisier MIDI stream than on full-mix (where the model was trained to handle multi-instrument mixtures).
- The resulting MIDI's pitch-class histogram is no richer than the librosa chroma already extracted on the full mix — you've changed the input medium, not added information.
- The note onset times from Basic Pitch don't align with chord boundaries, so they don't help with placement (which is your *real* error mode per the dead-ends ledger).

**Sources:**
- Ko (negative result): https://ko28.github.io/chord-transcription/
- APSIPA 2025 (positive result): http://www.apsipa.org/proceedings/2025/papers/APSIPA2025_P307.pdf
- ChordCoT (positive, bass-driven): https://arxiv.org/html/2509.18700v1

---

### Q3 — MIDI → chord inference algorithms

If you *did* feed MIDI to a chord labeler, the options:

| Library | License | Approach | Notes |
|---|---|---|---|
| **music-x-lab/midi-chord-recognition** | Unspecified ⚠️ | Rule-based dynamic programming, uses beat + downbeat info | ~80% maj/min on RWC Pop. Requires correct beat/downbeat. No license file — assume restrictive until clarified. https://github.com/music-x-lab/midi-chord-recognition |
| **pychord** (yuma-m) | MIT ✅ | `find_chords_from_notes()` — exact set-match against named chord vocabulary | Stateless. No timing/duration weighting. Good for "what chord is this set of notes" but not for time-series. https://github.com/yuma-m/pychord |
| **chorder** (joshuachang2311) | Unspecified ⚠️ | DeChorder class, scoring-based against pitch sets in time ranges | Needs miditoolkit objects. Algorithm not well-documented. https://github.com/joshuachang2311/chorder |
| **chord-extractor** (ohollo) | **GPL ❌** | Python wrapper around Chordino (NNLS chroma) | GPL incompatible with commercial closed-source deployment. https://github.com/ohollo/chord-extractor |
| **NNLS Chroma / Chordino** | **GPL ❌** | Vamp plugin. NNLS-based chroma → HMM chord smoother | C4DM/QMUL. Excellent quality but GPL — would force your backend GPL. https://github.com/c4dm/nnls-chroma |
| **Custom (recommended)** | N/A | Window MIDI notes by bar → duration-weighted pitch-class histogram → cosine similarity vs. 24+ chord templates | Trivial to implement (50–100 LOC). No license risk. Same approach librosa V1 uses on chroma — just sourced from MIDI instead of CQT. |

**Algorithm sketch for the custom approach (what you'd actually build):**
1. Take MIDI note events from Basic Pitch on the bass (or guitar) stem.
2. For each bar (using your existing bar grid): sum note durations per pitch class (0–11).
3. Normalize to a 12-d "pitch class profile."
4. Cosine-distance match against a template library (C maj = [1,0,0,0,1,0,0,1,0,0,0,0], C min, C7, etc.).
5. Output the best match.

This is structurally identical to what librosa V1 does — you're just changing the chroma source from CQT-on-audio to histogram-of-MIDI. **Whether it helps depends entirely on whether Basic Pitch's MIDI on a separated stem is cleaner than librosa's CQT chroma on the full mix.** No published evidence says it is.

**Sources:**
- midi-chord-recognition: https://github.com/music-x-lab/midi-chord-recognition
- pychord: https://github.com/yuma-m/pychord / https://pypi.org/project/pychord/
- chorder: https://github.com/joshuachang2311/chorder
- chord-extractor (GPL): https://github.com/ohollo/chord-extractor / https://pypi.org/project/chord-extractor/
- NNLS Chroma (GPL): https://github.com/c4dm/nnls-chroma / http://www.isophonics.net/nnls-chroma

---

### Q4 — Polyphonic transcription on stems (and licensing)

| Model | License (code) | Training data license | Commercial-safe for you? | Notes |
|---|---|---|---|---|
| **Spotify Basic Pitch** | Apache 2.0 | Mixed (instrument-agnostic, includes GuitarSet CC BY 4.0 + MedleyDB etc.) | ✅ **Yes — you already use it** | Polyphonic. F1 ≈ 79% on isolated GuitarSet guitar; ~52% F-measure no-offset on harder polyphonic test sets. https://github.com/spotify/basic-pitch |
| **Magenta MT3** | Apache 2.0 (code) | **MAESTRO is CC BY-NC-SA 4.0** ❌ | ❌ **No** — same legal posture as SynthTab in your blocklist. Pretrained weights are tainted by NC training data. | Multi-instrument transcription. https://github.com/magenta/mt3/blob/main/LICENSE / https://github.com/magenta/magenta/issues/1915 |
| **Banquet** (Watcharasupat, ISMIR 2024) | MIT ✅ | MoisesDB-licensed for non-comm research; check checkpoint terms before shipping | ⚠️ **Verify weights provenance** before commercial deploy | **Outperforms htdemucs_6s on guitar and piano stems.** 24.9M params. Already on your radar in `project_model_survey_may2026.md`. Single-decoder, query-based — could replace htdemucs_6s for guitar+piano specifically. https://arxiv.org/abs/2406.18747 / https://github.com/kwatcharasupat/query-bandit |
| **PESTO** (Sony CSL) | (check repo; likely permissive) | Self-supervised, no labeled music data | Likely ✅ | **Monophonic only** — not useful for polyphonic chord, useful only for the per-string tuner idea (Q6). https://github.com/SonyCSLParis/pesto |
| **CREPE** (NYU MARL) | MIT ✅ | Monophonic training set | ✅ but **monophonic only** | https://github.com/marl/crepe |
| **CREPE Notes** (xavriley) | MIT (code) | n/a | ⚠️ **depends on madmom (non-commercial license)** | Don't introduce madmom into your stack. https://github.com/xavriley/crepe_notes |
| **PolyScribe** (2024) | Unclear | Unclear | ⚠️ verify | Newer poly transcription. https://aiqiliu.github.io/polyscribe/ |

**Key finding for your stack:** Basic Pitch is your best legal option for polyphonic guitar/piano stem transcription. **Banquet is the only realistic upgrade path** — it's MIT, separates guitar/piano better than htdemucs_6s, and matches the post-launch A/B you already flagged in `project_model_survey_may2026.md`. MT3 is permanently off-limits for the same reason SynthTab is.

**Sources:**
- Basic Pitch: https://engineering.atspotify.com/2022/6/meet-basic-pitch / https://github.com/spotify/basic-pitch
- MT3 license issue: https://github.com/magenta/mt3/blob/main/LICENSE / https://magenta.tensorflow.org/datasets/maestro
- Banquet paper: https://arxiv.org/abs/2406.18747
- Banquet repo (MIT confirmed): https://github.com/kwatcharasupat/query-bandit
- PESTO: https://github.com/SonyCSLParis/pesto / https://arxiv.org/abs/2309.02265
- htdemucs (MIT): https://github.com/facebookresearch/demucs
- BS-RoFormer (MIT): https://github.com/lucidrains/BS-RoFormer

---

### Q5 — Confidence-fusion approaches

If you ran V1 librosa + a stem-derived second source, the standard combiners:

1. **Per-bar majority vote** — N detectors emit one chord per bar; vote. Tied if 2/2. Cheap.
2. **Weighted vote by per-detector confidence** — each detector outputs (chord, confidence ∈ [0,1]); pick argmax of summed weighted confidence. Requires calibrated confidence (librosa V1 doesn't emit this natively, but you can approximate via best-template-distance ratio).
3. **Gating model** — train a small classifier on (per-bar features) → which detector to trust. Needs labeled data — you have your honest-scorer audit set (small) but it's not big enough to train this without overfitting.
4. **LLM-as-judge** — exactly what ChordCoT does with GPT-4o. **You already have this in your stack** (V1 librosa + Anthropic correction). Adding a second detector under it is the natural extension; the cost is API spend per song.
5. **Stacking with music-theory priors** — given two candidates, prefer the one consistent with detected key + neighboring chords + bass note. ChordCoT Stage 2 (bass-driven root override) is the most concrete recipe.

**Honest evaluation against your situation:** your "V1 + Anthropic corrector" architecture is already a fusion system — the LLM is the combiner. The most leveraged change here is **giving the LLM a second source of ground truth to weigh against V1**, specifically the **bass note** (which is the *one* signal that's both reliable and corrective). Adding a noisy guitar-MIDI source would more likely confuse the LLM than help it.

**Sources:**
- Majority Vote for Late Fusion (PAC-Bayesian): https://arxiv.org/pdf/1207.1019 / https://link.springer.com/chapter/10.1007/978-3-662-44415-3_16
- Late fusion overview: https://apxml.com/courses/intro-to-multimodal-ai/chapter-3-techniques-integrating-modalities/late-fusion
- ChordCoT (LLM as combiner): https://arxiv.org/html/2509.18700v1

---

### Q6 — Polyphonic tuner (secondary "Guitar Coach" feature)

PolyTune's algorithm is proprietary (TC Electronic) — no open-source clone of MonoPoly exists. But the building blocks are available:

| Tool | License | What it does |
|---|---|---|
| **aubio** | GPL-3.0 ❌ | Onset, pitch, beat. Lists polyphonic chord detection as a *future idea* — not implemented. https://github.com/aubio/aubio |
| **sevagh/chord-detection** | MIT ✅ | DSP algorithms for chord + key detection from academic papers. Could be repurposed for per-string analysis. https://github.com/sevagh/chord-detection |
| **CREPE / PESTO** | MIT / permissive ✅ | Monophonic pitch tracking. Combine with bandpass-per-string-frequency-range to get crude per-string output. |
| **librosa** | ISC ✅ | Already in your stack. CQT + per-bin peak picking gives you a crude "which strings are sounding" map. |

**Architecture sketch for a Guitar Coach polyphonic tuner:**
1. User plays an open chord into the mic.
2. Run CQT (librosa) → identify the 3–6 strongest peaks in the lower octaves.
3. For each peak, map to nearest expected string-pitch given the tuning (default E A D G B E).
4. Compute cent-offset per string; render a string-by-string display.

This is **cleanly separated from chord detection** — no audio-engineering risk to existing detector. License-clean (all permissive). Effort: ~1–2 days for a working prototype. Honest expected accuracy: solid on clean DI/clean amp signal; degrades with distortion (harmonics confuse peak picking).

aubio's GPL is the **only license trap** in this space — avoid it.

**Sources:**
- PolyTune algorithm description: https://toneprints.com/media/218185/tc_electronic_polytune_manual_english.pdf (proprietary)
- aubio (GPL): https://github.com/aubio/aubio
- sevagh/chord-detection (MIT): https://github.com/sevagh/chord-detection
- PESTO: https://github.com/SonyCSLParis/pesto

---

### Q7 — Competitor analysis

| Competitor | Public architecture info | Key inference |
|---|---|---|
| **Chordify** | None published. Founder team has ISMIR papers on HMM + chord templates from late 2000s; reasonable to assume the prod stack is chroma + HMM-style smoothing + commercial post-processing. | Same ceiling as everyone else (~80% MIREX). Their moat is UX + library scale, not detection accuracy. |
| **Moises** | None published. Recent (2024–2025) feature updates added "easy/medium/advanced" chord vocabularies + bars/beats sync. | The "advanced" mode supports sevenths — suggests a large-vocab recognizer (BTC/Jiang/Chordformer family) under the hood. Quality is the perceived market leader; field cap means they have ~2–3% on you at most on raw F1. |
| **Klang.io** | Public engineering posts: spectrogram → CNN (image-recognition methods) → instrument-specific transcription model → "language model that depicts musical 'grammar'" for output cleanup. Closed source. | The "musical grammar" language-model step is structurally similar to your "V1 librosa + Anthropic" stack. Instrument-specific models match Banquet's direction. |

**Strategic read:** the competitor stack is converging on the same architecture: separation → per-instrument transcription → language-model-style refinement. **You are not behind on architecture.** Your gap is dataset scale + UX polish, neither of which is closed by changing the chord detector.

**Sources:**
- Moises chord-detection updates: https://moises.ai/blog/latest/advanced-chord-detection/ / https://help.moises.ai/hc/en-us/articles/6569274648220-How-do-I-use-Chord-Detection
- Klang.io engineering description: https://karlsruhe.digital/en/2025/09/klang-io-ki-musik/ / https://klang.io/about-us/research/
- Chordify (no published architecture — inferred from ISMIR back-catalogue of co-founder Bas de Haas)

---

### Q8 — Dead-ends check

**Read of `project_chord_dead_ends_ledger.md` (dated 2026-05-19):**

The hypothesis "stem-MIDI → chord inference" is **not literally in the ledger**, but the ledger's central conclusion directly bears on it:

> **🔒 ACCURACY CHASE PERMANENTLY CLOSED — 2026-05-19 (proven by math, do NOT reopen)**
> ... PERFECT GT labels through the real bar grid score only composite ~0.70 → a perfect detector caps at ~0.70 on this honest metric ... Nothing can beat perfect → no detector/API/fine-tune/architecture yields a materially-better chart by launch.

**The closest analogue in the ledger:**
> Quality-only enricher (librosa roots + ACE adds 7th/maj7/sus when roots agree) — BUILT+MEASURED DEAD: predicted +.09–.12 full F1 didn't materialize, slight regression; missing colors sit on bars where librosa's ROOT is also wrong.

That dead-end is a structurally similar idea: *take librosa V1 as the base, layer a richer signal on top.* It failed because **librosa's errors aren't just missing colors — they're wrong roots in the first place**, and a quality-enricher on top of a wrong root doesn't help.

**The stem-MIDI hypothesis would face the same trap if applied to quality.** It might *escape* the trap if applied to **roots** specifically (which the quality-enricher didn't touch). Hence the Q5 / top-recommendation framing: bass-stem root correction is the only angle within this hypothesis space that hasn't been measured-dead.

**Other relevant dead-ends (don't re-litigate):**
- ACE/Jiang detector swap as primary — superseded; the path is now "fix metric + disarm smooth_qualities + ship via router," not a fresh detector evaluation.
- Madmom downbeat tracker — DEAD, formatter is glued to old librosa quirks. Don't re-introduce madmom (also bonus: it has a non-commercial license).
- Placement fixes inside the formatter — CLOSED CATEGORY. Don't propose a 5th formatter tweak.
- BTC fine-tune (your in-house v10) — orphan, 79.92% ceiling, trained on legal-landmine data, don't ship.

**Flag for the user:** if you read this report and you're tempted to chase headline F1, **stop and re-read the ledger's "trial verdict" section.** This research confirms — does not contradict — the conclusion that the accuracy ceiling is structural to the labeling problem, not detector-side.

---

### Q9 — Legal / license constraints

Cross-reference against `project_training_data_legal.md`:

**Models / repos that are commercial-safe (code + weights):**
- htdemucs (MIT) ✅ — already in prod
- BS-RoFormer (MIT) ✅ — already in prod
- Basic Pitch (Apache 2.0) ✅ — already in prod, training data is permissive-mixed
- Banquet (MIT, code+weights on Zenodo) ✅ — verify checkpoint provenance before shipping but the published weights are released under an MIT-spirit posture
- pychord (MIT) ✅
- sevagh/chord-detection (MIT) ✅
- CREPE (MIT, monophonic) ✅
- PESTO (permissive, monophonic) ✅

**Trained-data poisoned (DO NOT ship, even if code is open):**
- MT3 — code Apache 2.0 BUT trained on MAESTRO (CC BY-NC-SA 4.0). Same legal posture as SynthTab in your blocklist. **Reject.**
- Any model trained on RWC, Billboard, JAAH, Beatles/Queen/Schubert sets, Archive.org Live, mySongBook — your blocklist already covers these.
- music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition pretrained weights — likely trained on Isophonics/RWC/Billboard (mixed); **provenance unclear, treat as tainted**.
- BTC pretrained weights — same training data class (Isophonics, Robbie Williams, UsPop2002; audio scraped from "online music service providers" per authors). **Tainted; don't ship.** Code itself is fine to use for your own training on clean data.

**Copyleft traps (avoid in your stack):**
- Chordino / NNLS Chroma — **GPL** ❌
- chord-extractor (Python wrapper around Chordino) — **GPL** ❌
- aubio — **GPL-3.0** ❌
- madmom — **non-commercial license** ❌ (also cited in the dead-ends ledger as a blocker for the downbeat-tracker fix)

**Unclear / verify-before-use:**
- music-x-lab/midi-chord-recognition — no license file → **assume restrictive**
- joshuachang2311/chorder — no clear license → **assume restrictive**
- ChordCoT code repo — license not visible in paper; check before relying on
- ChordMini pretrained weights — MIT code, FMA/DALI/MAESTRO **unlabeled** for pseudo-labeling (acceptable for self-supervised), but the labeled chord dataset has "restricted access"; **safe for inference if weights are released, risky if redistributing**

---

## Ranked next experiments

Ranking is **what's most likely to deliver real F1 lift × cheapest to test × lowest license risk**. All proposed only as **post-launch** work given the dead-ends ledger's conclusion.

| # | Experiment | What it tests | Effort | Expected F1 lift | License risk |
|---|---|---|---|---|---|
| 1 | **Bass-stem root corrector** — Basic Pitch on htdemucs bass stem → per-bar dominant pitch class → if disagrees with V1 librosa root AND key-consistent, override the V1 root. Measure on your 8-song bench with v2 honest scorer. | Replicates ChordCoT Stage 2 in isolation. Tests whether bass-driven root correction beats V1's chroma-derived roots. | **1 day** (build) + 0.5 day (measure) | **+1–2% MIREX** (ChordCoT got 1.06–2.77% with this + 3 other stages combined; this alone should be the dominant component since bass=root is the cleanest of the four steps) | **Low** — Basic Pitch + htdemucs both in prod |
| 2 | **Drum/vocal-removed full-mix ACR** — feed V1 librosa the htdemucs "no_drums" stem (sum of bass+other+guitar+piano) and the "no_drums_no_vocals" stem, output 3 chord streams, let your Anthropic corrector pick most-confident per bar. | Tests Result A vs Result B from Q2 directly on your stack. Cheaper alternative to "audio amplification then recombine" from APSIPA 2025. | **2–3 days** (separator outputs already exist; need pipeline changes to run librosa on subsets and feed 3 streams to LLM) | **+1% MIREX** (within published variance) — but could regress (Ko's negative result) | **Low** |
| 3 | **Banquet A/B vs htdemucs_6s** for guitar+piano stems | Already in `project_model_survey_may2026.md`. Better guitar/piano stems → better Basic Pitch MIDI (→ better Guitar Pro export, not necessarily better chord F1). | **~1 day** training-free A/B (pretrained weights) + measure | F1 lift on chord detection: **~0** (chord det uses full mix). Quality lift on Guitar Pro export: **possibly meaningful**. | **Low–medium** (verify Banquet checkpoint license) |
| 4 | **Sevagh chord-detection multipitch as 2nd source** | Run sevagh/chord-detection's polyphonic harmonic algorithm in parallel with V1 librosa on full mix → late-fuse via majority vote per bar. | **2 days** | **±0.5% MIREX** (likely noise) | **Low** (MIT) |
| 5 | **Per-string polyphonic tuner ("Guitar Coach")** — separate feature, not chord detection | Standalone retention feature. Doesn't touch ACR. | **2 days** for prototype | n/a (not a chord-detection experiment) | **Low** |
| 6 | **ChordCoT replication** — port the full 5-stage GPT-4o pipeline | Tests whether the whole published 1–2.77% gain stack reproduces on your audit set. | **1 week+** plus ongoing API spend | **+1–3% MIREX** if it replicates | **Medium** (code license unclear; needs GPT-4o ≠ Anthropic which is what you have wired) |
| 7 | **Train a fresh chord recognizer on Slakh2100** | Synthetic-data-only training per arXiv:2508.05878. Fully license-clean. | **3–4 weeks** + GPU spend (you already audited Modal cost discipline) | Unknown, likely matches BTC class (~80% MajMin) — **doesn't beat your ceiling** | **Very low** (Slakh = CC BY 4.0) |

---

## Top recommendation (one thing to try first)

**If you decide to revisit chord accuracy post-launch (and only then), build experiment #1: the bass-stem root corrector.**

**Rationale:**
1. **It's the *one* shape of the hypothesis that has published positive evidence** — ChordCoT Stage 2 reports the bass-stem root-correction step as a meaningful contributor to its overall 1–2.77% gain.
2. **It's the cheapest** — Basic Pitch and htdemucs bass stem are already in your pipeline. You're discarding the bass stem's note information today. Picking it back up costs one Python module.
3. **It targets the *right* error mode** — your dead-ends ledger says librosa V1 has root-level errors (bad roots, bad placement) more than quality-level errors. A bass-driven root override hits exactly the failure mode the quality-enricher couldn't.
4. **It composes cleanly with your existing Anthropic corrector** — bass-derived candidate root becomes a third input to the LLM correction prompt, not a replacement for V1.
5. **It's license-clean** — entirely within tools you already ship.
6. **It validates or refutes the hypothesis cheaply** — if a 1-day build gives +1% on your audit set, the broader hypothesis has legs. If it gives 0%, the ledger's "accuracy chase closed" verdict gets one more confirming data point.

**What to expect:** 1–2% MIREX lift on root-related metrics in the best case (matching ChordCoT's component contribution). Probably 0% on placement (which is your real product-feel problem). This is consistent with the ledger: not a launch-changer.

**What NOT to do based on this research:**
- Don't build the guitar-stem-poly-transcription chord inferer described in the hypothesis. No published evidence supports it; the more careful negative result (Ko, UW-Madison) is closer to that shape than the positive results.
- Don't introduce MT3 — license-blocked by training data.
- Don't introduce Chordino / NNLS / chord-extractor / aubio / madmom — all copyleft or non-commercial.
- Don't re-do detector swap experimentation generally — the ledger is right; the ceiling is structural to the label space, not the detector.

**The deeper recommendation (consistent with your ledger):** the highest-EV use of post-launch engineering hours is **not** in this research area. It's in (a) confidence-faded chord rendering, (b) one-tap chord edit, (c) the practice room as a standalone value prop — exactly the path the dead-ends ledger and the trial verdict converged on. This research confirms that conclusion from the literature side.

---

## Sources (consolidated, no fabricated citations)

**Papers:**
- ChordCoT — https://arxiv.org/abs/2509.18700 / https://arxiv.org/html/2509.18700v1 / https://arxiv.org/pdf/2509.18700
- ChordFormer — https://arxiv.org/abs/2502.11840 / https://arxiv.org/html/2502.11840v1
- BTC (Park et al. 2019) — https://arxiv.org/abs/1907.02698 / https://arxiv.org/pdf/1907.02698 / https://archives.ismir.net/ismir2019/paper/000075.pdf
- BTC-FDAA-FGF (2025) — https://www.sciencedirect.com/science/article/abs/pii/S0045790625004987
- Synthetic-data training (Majchrzak & Mańdziuk 2025) — https://arxiv.org/abs/2508.05878 / https://link.springer.com/article/10.1007/s00521-026-12069-0
- Banquet (Watcharasupat & Lerch, ISMIR 2024) — https://arxiv.org/abs/2406.18747
- APSIPA 2025 source-separation preprocessing — https://ieeexplore.ieee.org/document/11249321/ / http://www.apsipa.org/proceedings/2025/papers/APSIPA2025_P307.pdf
- Daniel Ko UW-Madison stem ACR (negative result) — https://ko28.github.io/chord-transcription/
- PESTO (Riou & Lattner 2024/2025) — https://arxiv.org/abs/2309.02265 / https://arxiv.org/abs/2508.01488
- Basic Pitch paper (Bittner et al. 2022, ICASSP) — https://engineering.atspotify.com/2022/6/meet-basic-pitch
- Deep Chroma (Korzeniowski & Widmer 2016) — https://archives.ismir.net/ismir2016/paper/000178.pdf / https://arxiv.org/pdf/1612.05065
- Late-fusion majority vote (PAC-Bayesian) — https://arxiv.org/pdf/1207.1019

**Repositories (with license status as found in this research):**
- htdemucs (MIT) — https://github.com/facebookresearch/demucs
- BS-RoFormer (MIT) — https://github.com/lucidrains/BS-RoFormer
- Banquet / query-bandit (MIT) — https://github.com/kwatcharasupat/query-bandit
- Basic Pitch (Apache 2.0) — https://github.com/spotify/basic-pitch
- BTC code (MIT) — https://github.com/jayg996/BTC-ISMIR19
- ChordMini (MIT code, training data caveats) — https://github.com/ptnghia-j/ChordMini / https://www.chordmini.me/about
- ChordCoT (license unclear) — https://github.com/WildHoneyPie/ChordCoT
- MT3 (Apache 2.0 code, NC training data) — https://github.com/magenta/mt3
- music-x-lab large-vocab — https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition
- music-x-lab midi-chord-recognition — https://github.com/music-x-lab/midi-chord-recognition
- pychord (MIT) — https://github.com/yuma-m/pychord / https://pypi.org/project/pychord/
- chorder — https://github.com/joshuachang2311/chorder
- chord-extractor (GPL) — https://github.com/ohollo/chord-extractor / https://pypi.org/project/chord-extractor/
- NNLS Chroma / Chordino (GPL) — https://github.com/c4dm/nnls-chroma / http://www.isophonics.net/nnls-chroma
- aubio (GPL) — https://github.com/aubio/aubio
- CREPE (MIT) — https://github.com/marl/crepe
- PESTO — https://github.com/SonyCSLParis/pesto
- CREPE Notes — https://github.com/xavriley/crepe_notes
- sevagh/chord-detection (MIT) — https://github.com/sevagh/chord-detection

**Competitor pages:**
- Moises chord detection update — https://moises.ai/blog/latest/advanced-chord-detection/
- Moises chord-detection help — https://help.moises.ai/hc/en-us/articles/6569274648220-How-do-I-use-Chord-Detection
- Klang.io engineering writeup — https://karlsruhe.digital/en/2025/09/klang-io-ki-musik/
- Klang.io research page — https://klang.io/about-us/research/

**License references:**
- MT3 license — https://github.com/magenta/mt3/blob/main/LICENSE
- MAESTRO commercial-use issue — https://github.com/magenta/magenta/issues/1915
- MAESTRO dataset page — https://magenta.tensorflow.org/datasets/maestro
- BS-RoFormer LICENSE — https://github.com/lucidrains/BS-RoFormer/blob/main/LICENSE
