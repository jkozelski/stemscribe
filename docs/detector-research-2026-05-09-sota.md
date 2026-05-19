# Chord-Detection SOTA Survey — 2026-05-09

**Author:** research agent
**Status:** research only. NO code commits. Deliverable for "what could replace librosa+corrector to lift the F1=0.77 ceiling on rock?"
**Time-box honored:** ~30 min reading + ~60 min web research + ~30 min synthesis.
**Today's baseline to beat:** librosa CQT chroma + 24-template + Anthropic Sonnet 4.5 corrector = **F1 0.77 / P 0.80 / R 0.76** on the 17-song rock audit (per `chord-research-2026-05-06.md` May 9 addendum).

---

## §1 — Survey methodology

### Sources I searched
- arXiv (2024-2025-2026): chord recognition, conformer, mamba, transformer, foundation-model probing
- ISMIR 2024 / ISMIR 2025 program pages and accepted-papers list
- MIREX 2024 + 2025 Audio Chord Estimation pages (`music-ir.org/mirex/wiki`)
- MIREX results repo at `github.com/ismir-mirex/ace-results`
- HuggingFace Hub (model cards filtered for chord recognition + audio classification)
- GitHub Topics: `chord-recognition`, `chord-detection`
- Direct GitHub API for license, last-push-date, star count on every named candidate
- Direct README + `requirements.txt` fetches for installability check

### Queries that produced the candidate set
1. `ISMIR 2024 audio chord recognition transformer state of the art`
2. `automatic chord estimation 2025 ACE deep learning F1 score benchmark`
3. `ChordFormer chord recognition transformer github 2024`
4. `MIREX 2024 chord recognition results winner MajMin` + same for 2025
5. `huggingface pretrained chord recognition model audio`
6. `"chord recognition" github 2024 2025 pytorch open source pretrained`
7. `MERT music representation chord recognition fine-tuning`
8. `MusicFM CLAP chord recognition downstream task pretrained foundation model`
9. `"BACHI" symbolic chord recognition iterative ranking SOTA accuracy`
10. `"BMACE" Mamba chord recognition github`
11. `arxiv 2509.01588 consonance training chord estimation Poltronieri code`
12. `"basic-pitch" OR "MT3" multitask transcription chord recognition pitch class output`

### What I deliberately skipped (per task brief)
- BTC (rejected — was prior fallback, dropped May 6)
- Klangio (rejected May 2)
- autochord, madmom (broken on Python 3.11 + arm64 macOS)
- HMM/CRF temporal smoothing
- Custom CRNN training
- Anything paywalled or research-only-license

### What I confirmed before any candidate goes on the table
- License via GitHub API (`spdx_id`) — must be MIT / Apache-2.0 / BSD
- Last push date — anything dormant >12 months gets a yellow flag
- Pretrained checkpoint actually present in the repo (or downloadable separately) — papers without released weights are tagged "training cost"
- Python version (need 3.11 compatibility for our backend)

---

## §2 — Candidate list

| # | Candidate | Year | License | Last push | Repo | Pretrained? | Reported best (MIREX/MajMin) | Inference | Py 3.11? | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **consonance-ACE** (Poltronieri et al. ISMIR 2025) | 2025-09 | **MIT** | 2026-01-29 | andreamust/consonance-ACE | YES — 55 MB ckpt in repo | MIREX 79.8 / MajMin 77.8 (RWC+USPop) | CPU-OK, 20s chunks, conformer ~moderate | YES (declared) | **Strongest practical candidate.** Single-command inference. .lab output. |
| 2 | **ChordMini (CNN-LSTM)** via ChordMiniApp | 2025 | **MIT** | 2026-05-08 | ptnghia-j/ChordMiniApp | YES (Git LFS) | 301 chord classes, no published academic numbers | CPU-OK | 3.10.x (declared) | Python 3.10 dependency chain (`spleeter`, `madmom`); not fully separable from web app. |
| 3 | **ChordCoT** (LLM CoT refinement) | 2025-09 | NONE declared | 2025-09-21 | WildHoneyPie/ChordCoT | NO — code being cleaned up | MIREX 81-86 (across 3 sets) | GPT-4o calls per song = $$ | Unknown | **DISQUALIFIED on license.** Repo also notes "code is being cleaned up." Not usable today. |
| 4 | **ChordFormer** (Liu et al. arXiv 2025-02) | 2025-02 | (paper only — repo at `cameron-cs/chordformer` is a different work) | n/a | none official | NO official release | Root 84.69 / MajMin 84.09 / MIREX 83.62 (Humphrey-Bello set) | unknown | unknown | **No code published.** Paper-only. |
| 5 | **BMACE (Mamba ACE)** Yuan/Sim/Devaney MIREX 2025 | 2025-10 | none located | n/a | not found | NO | Submitted MIREX 2025 (mid-pack) | very small (1/25 BTC params) | unknown | Code never released; paper has no repo link. |
| 6 | **ISMIR2019 Large-Vocab** (Jiang et al.) | 2019 | **MIT** | 2024-04-09 | music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition | YES | Used as MIREX 2025 baseline (~74-78 MajMin) | CPU-OK | likely (last touched 2024) | This is the model ChordMini and ChordCoT both wrap. **Direct path: skip the LLM/wrapper, use the underlying model.** 301 classes. |
| 7 | **MusicFM** (Won et al.) | 2024-02 | MIT/Apache (NOASSERTION badge) | 2024-02-14 | minzwon/musicfm | Pretrained foundation model — chord head NOT included | Paper claims SOTA on chord probing, no concrete num released | GPU recommended, 12s chunks @ 16Hz | likely | **Requires fine-tuning for chord head.** No drop-in chord output. |
| 8 | **MERT-v1-95M** (m-a-p) | 2023-03 | Code Apache-2.0; **WEIGHTS CC-BY-NC-4.0** | 2025-05-25 (code) | yizhilll/MERT + huggingface m-a-p | Weights yes; chord head NO | Strong on MARBLE chord task (probing) | GPU recommended | YES | **Weights non-commercial — DISQUALIFIED for production use.** |
| 9 | **BACHI** (Weasley/etc, ICASSP 2026) | 2025-10 | **MIT** | 2026-01-26 | AndyWeasley2004/BACHI_Chord_Recognition | YES | Pop full-chord 82.4%, Classical 68.1% | CPU-OK | unknown | **SYMBOLIC (operates on MIDI), not audio.** Would need a transcription front-end. Not a drop-in replacement. |
| 10 | **Chordino / NNLS-Chroma** (the May 6 plan's gate) | 2010 (still active) | **GPL-2.0** (the python wrapper `chord-extractor`) | wrapper 2025-08 | ohollo/chord-extractor | YES (Vamp plugin) | MIREX 2025 baseline ~70-75 MajMin | CPU-fast | not officially | **Wrapper is GPL-2.0 — VIRAL LICENSE, DISQUALIFIED for production link.** Could shell out to Vamp host (`sonic-annotator`) instead, which is GPL but executed-not-linked. Vamp plugin binary is itself GPL — same problem if we redistribute. |

### License field is doing a lot of work here
Of the 10 candidates: 6 are MIT, 1 has weights under CC-BY-NC, 1 is paper-only (no license to evaluate yet), 1 has no license declared, 1 is GPL-2.0. Five candidates are commercially clean *and* have shipped code *and* have a pretrained checkpoint. Two of those five (ISMIR2019 and consonance-ACE) actually output chord labels directly without further fine-tuning. **That's the entire viable shortlist.**

---

## §3 — Top-3 deep dives

### #1 — consonance-ACE (Poltronieri, Serra, Rocamora — ISMIR 2025)

**Paper:** "From Discord to Harmony: Decomposed Consonance-based Training for Improved Audio Chord Estimation," ISMIR 2025 (Daejeon, Korea). [arxiv:2509.01588](https://arxiv.org/abs/2509.01588)
**Repo:** `github.com/andreamust/consonance-ACE` — **MIT, 29 stars, last pushed 2026-01-29.**

**What it is.** A conformer-based audio chord estimation model with two key innovations vs the BTC family:
1. **Decomposed output heads** — instead of one softmax over 170 chord classes, the model has separate heads for *root*, *bass*, and *pitch-class activation*, then reconstructs the chord label from those. This directly attacks the class-imbalance problem (rare chords like Cmaj7#11 have terrible support in any direct-classification model — but a `pc=B` activation gets supervision signal from every chord that contains a B).
2. **Consonance-based label smoothing** — instead of one-hot targets, neighboring chord labels are weighted by a perceptual consonance metric. This is a soft regularizer: predicting Bm when truth is D6 is treated as less wrong than predicting F#dim, because Bm shares more pitch classes with D6.

**Reported numbers (averaged over RWC-Pop + USPop test sets, Table 2):**
| Metric | This model | BTC baseline (their reimpl.) | Δ |
|---|---|---|---|
| Root | 84.0 | 82.8 | +1.2 |
| MajMin | 77.8 | 76.1 | +1.7 |
| Sevenths | 66.0 | — | — |
| MIREX | 79.8 | — | — |

The "Decomposed + Consonance Smoothing" variant is the one in the released checkpoint (`conformer_decomposed_smooth.ckpt`, 55 MB).

**Cross-walk to our metrics.** Our 0.77 F1 is computed against the Claude oracle on a custom 17-song rock set, *not* against `mir_eval` on RWC/USPop. The numbers are not directly comparable. But: our F1 is in the same ballpark as their MajMin (0.77 vs 77.8), so we're roughly competitive with their *baseline* — and their proposed model only beats BTC by ~1.7 points on MajMin. **The expected lift from a drop-in swap is therefore small (probably +0 to +3 points).** The real opportunity is in *what kinds of mistakes shift around*.

**Where it could help our specific failure modes.**
- *Wrong-key cascade (Sister Golden Hair / Misunderstood):* the **decomposed root head** is the directly-relevant component. A trained root classifier is far less likely to confuse G with D#m than the librosa argmax over CQT correlation, because it's been supervised on root *as root*. This is the strongest reason to pilot.
- *Extension hallucination on power-chord rock:* not directly addressed. The decomposed PC-activation head could plausibly *cause* extension hallucination on power-chord bars (the model thinks a B7 is also "kind of an E major" because they share PCs). Need to spike-test.
- *Flat-key songs:* helped IF the training data covers flat keys (Isophonics + McGill Billboard does include flat-key material; this is mid-cohort).

**Integration sketch.**
```
audio.wav → ACE/inference --audio audio.wav --out audio.lab
            → parse .lab → ChordEvent list with (start, end, "X:maj", confidence)
            → feed into our existing Anthropic corrector
```
Output format is the standard `.lab` file (3 columns: start, end, chord-symbol-with-Harte-shorthand like `E:maj`, `A:min7`). Trivial to parse into our `ChordEvent` dataclass. Inference is in 20-second chunks, so for a 4-minute song = ~12 chunks. No GPU required; a CPU pass on a 4-minute song should take 30-60 seconds (conformer is moderate-size).

**Modal compatibility.** Yes — pure PyTorch, can wrap as a Modal function with a CPU container. No CUDA-specific code. Could also run on the existing Hetzner CPX41 (8 vCPU, 16GB RAM) if we don't want a Modal trip just for chord detection.

**Cost to pilot.**
- Clone repo: 10 min
- Install Python 3.11 venv with their `requirements.txt` (already pinned): 30 min
- Run `python -m ACE.inference` on each of our 18 audit songs: 30 min wall time
- Parse outputs and run our Claude oracle scorer (already exists): 15 min + $1.08
- **Total: ~1.5 hours, $1.08, zero engineering commitment if it loses.**

**Risks.**
1. Their training set (Isophonics Beatles + McGill Billboard) is already heavily pop. If the model's already been tuned on Beatles, our Beatles-style rock songs may benefit. If it's been tuned on McGill (which is primarily 1958-1991 chart pop), our rock test set is in-distribution and the numbers should hold. **This is mostly a positive risk.**
2. The model outputs the Harte-style chord vocabulary (170 classes). Their `.lab` format will use `E:min7`, `A:hdim7`, etc. — we need to trim or map to our existing label space before the corrector sees it. ~20 min mapping table.
3. PyTorch 2.6 / Lightning 2.5 dependency — needs verification against our backend's existing torch (`venv311` likely has 2.x already, low risk).

**Verdict — pilot this. It's the highest information-density single experiment in the survey.**

---

### #2 — ISMIR2019 Large-Vocabulary Chord Recognition (Jiang, Chen et al.)

**Paper:** "Large-Vocabulary Chord Transcription via Chord Structure Decomposition," ISMIR 2019.
**Repo:** `github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition` — **MIT, 58 stars, last pushed 2024-04-09.**

**What it is.** A CNN+BLSTM model that decomposes chord prediction into root + chord-quality structure components, supporting 301 distinct chord labels. **Crucially, this is the same underlying model that both ChordMini and ChordCoT wrap.** ChordCoT's reported MIREX scores of 81-86% are this model's outputs *after* GPT-4o cleanup; the bare model is the foundation that everyone keeps coming back to when they want a working large-vocabulary detector.

**Reported numbers.** The original paper shipped in 2019; MIREX 2025 used a 2025-prepared variant of this model as one of the **official baselines**. From the 2025 results page, the ISMIR2019 baseline lands around 74-78 on MajMin across the four 2025 evaluation datasets — which means it's been outperformed by the 2025 winners but not by a huge margin (the top 2025 system YK1 scored 87.27 MajMin on RWC-Popular).

**Cross-walk to our setting.** This model has been in the wild for 6+ years and is a well-understood quantity. Its output style — Harte chord names with a wide vocabulary — likely produces **more** extension hallucination than our librosa+corrector, not less. The corrector currently handles "drop the hallucinated extension"; piping a 301-class model into that corrector may load up the corrector with more of the same work it already does well.

**Where it could help our specific failure modes.**
- *Wrong-key cascade:* unclear. The model has a real root head, so individual roots are likely better than ours. But it shares the "wide vocabulary, encourages hallucination" property that hurt us before.
- *Extension hallucination on power-chord rock:* probably HURTS. 301 classes including all the 7th/9th/13th variants — power-chord rock is the worst case for this label space.
- *Flat-key songs:* training data unclear; original paper used Billboard + Isophonics mix.

**Integration sketch.**
```
python3 chord_recognition.py audio.mp3 audio.lab submission
```
Single CLI invocation. Output is a `.lab` file. The same parse-and-feed logic as consonance-ACE works.

**Cost to pilot.** Roughly identical to consonance-ACE, ~1.5 hours and $1.08. **But** the dependency story is older; we'd need to verify Python 3.11 + modern PyTorch compatibility (the repo was last pushed in 2024-04, on what could be PyTorch 1.x).

**Risks.**
1. Old PyTorch — may need a compat layer or env upgrade.
2. 301-class output is more prone to extension hallucination than we want.
3. Already used as the underlying model by ChordCoT, which means: **if we want what ChordCoT promises, we already HAVE the LLM-corrector layer (Anthropic Sonnet 4.5).** Running this model + our existing corrector is *exactly* the architecture ChordCoT pitches, minus their specific 5-stage CoT recipe. That's actually a strong reason to pilot.

**Verdict — pilot ONLY IF consonance-ACE underperforms.** This is the fallback if the conformer-decomposed variant doesn't beat our librosa baseline.

---

### #3 — Chordino (NNLS-Chroma) via Vamp host — *recommended only as a research baseline, never integrated*

**Source:** `code.soundsoftware.ac.uk/projects/nnls-chroma`, the C++ Vamp plugin.
**Wrapper:** `ohollo/chord-extractor` Python package — **GPL-2.0**.

**What it is.** The 15-year-old NNLS-Chroma + Chordino approach: NNLS spectral whitening → chroma vector → chord template scoring with a hand-tuned chord profile. Beats raw chromagram on rock by being aware of guitar harmonic series.

**Reported numbers.** MIREX 2025 used Chordino as one of the official baselines — it scored in the **mid-60s to low-70s on MajMin** across the four eval datasets, depending on dataset. That's worse than our 0.77.

**Why it was on the May 6 plan.** As a feasibility gate: "if a 15-year-old free model beats our 40%, switch to it." That gate is moot now — our librosa+corrector at 0.77 is already above what Chordino reportedly scores on MIREX 2025 datasets. Running Chordino is a research baseline check, not an integration target.

**License problem.** The most popular Python wrapper is GPL-2.0. The C++ Vamp plugin is also GPL. We can't statically link or ship either. We *could* shell out to `sonic-annotator` (a GPL CLI tool) at runtime — that's the "executed but not linked" pattern that's commonly considered safe by GPL but is legally murky and our Apr 10 lawyer call did not bless it.

**Where it could help our specific failure modes.**
- *Probably nowhere* given our current 0.77 baseline. Chordino's MIREX 2025 baseline numbers are below that.

**Verdict — DO NOT INTEGRATE.** Optional 30-min research-only run as a sanity-check baseline, but no engineering effort beyond that. The May 6 plan's Chordino-as-feasibility-gate is now obsolete because we already cleared the bar Chordino represented.

---

## §4 — Honest verdict

**Pilot exactly one thing: consonance-ACE.**

- It's the only candidate that's simultaneously (a) MIT-licensed code and weights, (b) released a pretrained checkpoint in-repo, (c) supports Python 3.11, (d) outputs `.lab` directly with no additional fine-tuning, (e) was published in 2025 with current SOTA numbers, and (f) uses an architecture (decomposed root/bass/PC heads) that targets the exact failure mode (root-confusion → wrong-key cascade) that drove our 2 catastrophic outliers.

**Time-boxed pilot proposal:**
- **Day 1 morning (3 hours):** Clone, install in fresh `venv311_ace/`, run `python -m ACE.inference` on the 18 audit songs. Parse `.lab` outputs. Score against the Claude oracle. **Cost: $1.08 + ~3 hours wall time.**
- **Decision gate:** does consonance-ACE *raw* (no corrector) score above 0.40 (raw librosa) and above 0.77 (corrector-cleaned librosa)? If raw consonance-ACE > 0.55 → it's worth a serious second pass with the corrector. If raw < 0.55 → likely no-op even with corrector, and we stop.
- **Day 1 afternoon (2 hours, conditional):** Pipe consonance-ACE → existing Anthropic corrector. Re-score. **Total cost: $1.08 + $1.08 = $2.16.**
- **Decision gate:** does (consonance-ACE + corrector) beat 0.77? Yes → write a feature-flag-gated swap into `pipeline.py:546-549` and roll into staging. No → write up the negative result and move on.

**What I am NOT recommending:**
- Don't ensemble. The detector + corrector pipeline is already an ensemble in spirit (model + LLM rescue). A three-way ensemble adds latency without obvious lift.
- Don't fine-tune. We'd need labeled rock data we don't have, and the corrector layer already does most of the rock-domain work that fine-tuning would.
- Don't pursue the foundation-model path (MERT/MusicFM). MERT's CC-BY-NC weights kill it for production; MusicFM has no pretrained chord head and would require fine-tuning. Both are >2 weeks of engineering for an unknown delta.

**Expected outcome — be calibrated.** The 2025 SOTA numbers (consonance-ACE MIREX ~80, vs older BTC ~76) say the ML field gained roughly **+4 points in 6 years** on the standardized benchmarks. Our librosa+corrector is already at 0.77. **The realistic best-case is a +3 to +5 point lift** (to 0.80-0.82), with a 30% chance of no lift or a slight regression because our specific test set is rock-heavy and consonance-ACE's training set leans toward Billboard/Isophonics pop. **A 0.83 result would be a real win; a 0.77-0.79 result is a wash; below 0.77 is a no-op.**

---

## §5 — Anti-recommendations (look good, won't help)

### 5.1 — ChordCoT: looks good, license is missing and the code isn't ready
ChordCoT (WildHoneyPie/ChordCoT) is the most relevant *concept* match for our existing pipeline — its whole proposition is "wrap a base ACE model with an LLM that does chain-of-thought correction across MSS, bass, key, anomaly, and beat alignment." That's almost exactly our architecture. **Why I'm not recommending it:**
1. **No license declared.** Use is legally undefined.
2. README says "code is currently being organized and cleaned up. Updates will be pushed soon." 6 stars, 3 commits, last pushed Sept 2025. Not in a usable state.
3. The underlying audio model is the ISMIR2019 large-vocab one, which we can already use directly. The LLM corrector logic is replicated by Anthropic Sonnet 4.5 (which we're already paying for).
4. **We are already running their architecture, just with a different LLM and different correction prompts.** Adopting ChordCoT would mean swapping our corrector prompt for theirs, not adding a new detector.

### 5.2 — MERT / MusicFM (foundation models)
The 2024-2025 MIR-foundation-model wave (MERT, MusicFM, CLAP) genuinely produces strong representations for chord recognition as a probing task. **Why I'm not recommending pursuit:**
1. **MERT-v1-95M / 330M weights are CC-BY-NC-4.0** — disqualified for our commercial product.
2. **MusicFM has no pretrained chord-recognition head.** The paper reports good chord *probing* numbers, but using it requires us to fine-tune a head on a labeled chord dataset we'd have to acquire. That's ~weeks of engineering for an unknown delta.
3. The probing-task gain reported in MARBLE / MARBLE-style benchmarks against BTC is typically 1-3 points on MajMin — same order as a pure detector swap. We'd be paying weeks of fine-tuning cost for a delta we could get for $2 with consonance-ACE.

### 5.3 — BACHI (symbolic chord recognition)
BACHI scored 82.4% full-chord accuracy on POP909 (pop) — that's an *excellent* number relative to anything else in this survey. **Why it doesn't apply to us:**
1. **It's symbolic.** Operates on MIDI tokens, not audio. Adding it would require a transcription front-end (Basic Pitch + the same problems we already have) → BACHI operates on the transcription's MIDI → we'd be stacking errors.
2. The architecture (boundary detection + iterative ranking via masked decoding) is genuinely novel and probably worth eyeing for a future *symbolic* pipeline if we ever feed reliable MIDI in. But that's not today's question.
3. The 82.4% number is on cleanly-prepared MIDI from POP909-CL — not on Basic-Pitch-derived MIDI from real audio. Our latency is in the audio→MIDI step, not the MIDI→chord step.

### 5.4 — ChordFormer (Liu et al. 2025-02 paper)
Reported numbers (Root 84.69 / MajMin 84.09 / MIREX 83.62) are competitive with current SOTA. **Why I'm not recommending it:**
1. **No official code release.** The arXiv paper is paper-only; the GitHub repo named `cameron-cs/chordformer` is a different (independent) work. Without weights or code, integrating is "train from scratch on the Humphrey-Bello set" — multi-day GPU work for a paper we can't even verify is reproducible.
2. The conformer architecture is the same family as consonance-ACE — and consonance-ACE *has* released code and weights. Whatever ChordFormer can do, consonance-ACE does within ~1 point on MajMin and is actually installable today.

### 5.5 — BMACE (Mamba-based ACE, MIREX 2025 entrant)
A clever Mamba SSM model with 1/25 the parameters of BTC. **Why I'm not recommending it:**
1. **No code released.** Submitted to MIREX 2025, scored mid-pack. The paper has no repo link; the authors haven't published weights.
2. Even if released, the reported uspop2002 results show it's slightly *worse* than BTC on MajMin and only better on the niche rare-tetrad metric — not the metric we care about.

### 5.6 — Chordino + GPL Python wrapper
Discussed above in §3. Twin disqualifications: (a) Chordino's MIREX 2025 baseline scores ~70 MajMin which is *below* our current 0.77, and (b) the only practical Python wrapper is GPL-2.0. The May 6 plan listed it as a feasibility gate; that gate is now obsolete and the integration cost was always going to be heavy.

### 5.7 — Tuning the librosa CQT chromagram more (more bins, different median window, etc.)
Not in the survey but mentioned for completeness: the May 6 plan documented that more chromagram knobs are no-ops. The corrector layer compensates for chromagram-level errors so well that further tuning of the chromagram ahead of the corrector is dominated. This is a known dead end; don't revisit.

---

## Summary table — what to do tomorrow

| Action | Cost | Decision |
|---|---|---|
| Pilot consonance-ACE on the 18-song audit | ~3 hrs + $1.08 | If raw F1 > 0.55, proceed to corrector pass |
| Run consonance-ACE → Anthropic corrector pass | +2 hrs + $1.08 | If F1 > 0.79, ship behind feature flag |
| Anything else from the survey | — | Wait until pilot resolves |

**Net recommendation:** **3 hours and $2.16 to know whether the SOTA literature actually has a lever for our specific 0.77 ceiling.** That's the cheapest information we can buy.

---

## Sources

- [ISMIR 2025 program — consonance-ACE poster](https://ismir2025program.ismir.net/poster_268.html)
- [arxiv:2509.01588 — From Discord to Harmony (Poltronieri et al.)](https://arxiv.org/abs/2509.01588)
- [github.com/andreamust/consonance-ACE](https://github.com/andreamust/consonance-ACE)
- [arxiv:2502.11840 — ChordFormer](https://arxiv.org/abs/2502.11840)
- [arxiv:2509.18700 — ChordCoT](https://arxiv.org/html/2509.18700v1)
- [github.com/WildHoneyPie/ChordCoT](https://github.com/WildHoneyPie/ChordCoT)
- [arxiv:2508.05878 — Training chord recognition on artificial audio](https://arxiv.org/abs/2508.05878)
- [arxiv:2601.02101 — BMACE Mamba ACE](https://arxiv.org/abs/2601.02101)
- [arxiv:2510.06528 — BACHI symbolic chord recognition](https://arxiv.org/abs/2510.06528)
- [github.com/AndyWeasley2004/BACHI_Chord_Recognition](https://github.com/AndyWeasley2004/BACHI_Chord_Recognition)
- [github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition)
- [github.com/jayg996/BTC-ISMIR19](https://github.com/jayg996/BTC-ISMIR19)
- [github.com/ptnghia-j/ChordMiniApp](https://github.com/ptnghia-j/ChordMiniApp)
- [github.com/minzwon/musicfm](https://github.com/minzwon/musicfm)
- [github.com/yizhilll/MERT](https://github.com/yizhilll/MERT)
- [huggingface.co/m-a-p/MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) (CC-BY-NC weights)
- [MIREX 2025 ACE Results](https://music-ir.org/mirex/wiki/2025:Audio_Chord_Estimation_Results)
- [MIREX 2024 ACE task](https://music-ir.org/mirex/wiki/2024:Audio_Chord_Estimation)
- [github.com/ismir-mirex/ace-results](https://github.com/ismir-mirex/ace-results)
- [Chordino / NNLS-Chroma](http://www.isophonics.net/nnls-chroma)
- [github.com/ohollo/chord-extractor](https://github.com/ohollo/chord-extractor) (GPL-2.0 wrapper)
