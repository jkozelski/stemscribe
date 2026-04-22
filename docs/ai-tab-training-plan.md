# AI Tab Training Plan: Replacing Songsterr Dependency

**Date:** 2026-04-04
**Status:** Research/planning -- no code changes made
**Goal:** Train an AI model to generate guitar and bass tablature from audio, eliminating the Songsterr API dependency

---

## 1. Current State

### What we get from Songsterr

The Songsterr integration (`backend/routes/songsterr.py`) fetches tab data via three API endpoints:

1. **Search:** `https://www.songsterr.com/api/songs?pattern=QUERY` -- returns song list with songId, artist, title, tracks (instrument, tuning, difficulty, views)
2. **Metadata:** `https://www.songsterr.com/api/meta/{songId}` -- returns revisionId, image hash, track list
3. **Track JSON:** `https://dqsljvtekg760.cloudfront.net/{songId}/{revisionId}/{imageHash}/{trackIdx}.json` -- the actual tab data

The track JSON format (`songsterr_to_gp.py` reveals the full structure) contains:

```
{
  "measures": [
    {
      "signature": [4, 4],         // time signature
      "marker": {"text": "Verse"}, // section markers
      "voices": [
        {
          "beats": [
            {
              "type": 8,           // duration denominator (1=whole, 2=half, 4=quarter, 8=eighth...)
              "duration": [3, 4],  // dotted: [numerator, denominator]
              "tuplet": {"enters": 3, "times": 2},
              "rest": true/false,
              "velocity": "mf",    // ppp/pp/p/mp/mf/f/ff/fff
              "letRing": true/false,
              "notes": [
                {
                  "string": 2,     // 0-indexed string number
                  "fret": 5,       // fret number
                  "tie": true/false,
                  "dead": true/false,
                  "hammerOn": true/false,
                  "pullOff": true/false,
                  "vibrato": true/false,
                  "palmMute": true/false,
                  "staccato": true/false,
                  "harmonic": "natural"/"artificial"/"pinch",
                  "slide": "shiftSlide"/"legSlide"/"inFromAbove"/"inFromBelow"/"outDown"/"outUp",
                  "bend": {
                    "tone": 100,   // bend amount (100 = half step)
                    "points": [{"position": 0, "tone": 0}, {"position": 30, "tone": 100}]
                  }
                }
              ],
              "chord": {"text": "Am7"}  // native chord annotations
            }
          ]
        }
      ]
    }
  ],
  "automations": {
    "tempo": [{"measure": 0, "beat": 0, "bpm": 120}]
  },
  "tuning": [64, 59, 55, 50, 45, 40],  // MIDI note per string (high E to low E)
  "strings": 6,
  "instrumentId": 27,
  "frets": 24,
  "name": "Electric Guitar",
  "newLyrics": [{"text": "..."}]
}
```

**Key insight:** Songsterr's data is note-by-note JSON with exact string, fret, duration, articulations, dynamics, and timing. This is effectively a Guitar Pro file in JSON form -- much richer than MIDI. It is **ideal training data** for a tab transcription model.

### What we have locally

- **15,417 songs** in the chord library (`backend/chord_library/`) -- artist/title/chord progressions only (no note-level tab data, no Songsterr IDs stored)
- **BS-RoFormer-SW** for clean stem separation (deployed on Modal, +2-3 dB over htdemucs_6s for guitar)
- **A\*-Guitar** fret optimizer deployed in `midi_to_gp.py`
- **trimplexx CRNN** researched (87% F1 on GuitarSet, native string+fret output, no pretrained weights)
- **FretNet** researched (continuous pitch contours for bends/vibrato, no pretrained weights)
- **Basic Pitch** for MIDI transcription (current production, ~79% F1)

---

## 2. Training Data Pipeline

### 2a. Can we bulk-fetch tab data from Songsterr?

**Technically yes, practically risky.**

The Songsterr API has no published rate limits or bulk-access terms. The CDN pattern is predictable:
```
https://dqsljvtekg760.cloudfront.net/{songId}/{revisionId}/{imageHash}/{trackIdx}.json
```

To fetch tab data for our 15,400 songs we would need to:
1. Search Songsterr for each artist+title to get the songId
2. Hit `/api/meta/{songId}` to get revisionId + imageHash
3. Fetch each track's JSON from the CDN

That is **~15,400 search requests + ~15,400 meta requests + ~30,000-60,000 track requests** (2-4 tracks per song average). At a conservative 1 request/second with politeness delays, that is ~24-40 hours of crawling.

**Legal/ethical concerns:**
- Songsterr's ToS likely prohibits bulk scraping (we have not reviewed it, but standard for any API)
- This is exactly the kind of thing Alexandra warned about -- do NOT frame this as scraping in any lawyer communications
- The tab data itself is user-contributed transcriptions, not Songsterr's original work, but the compilation/hosting is theirs
- **Recommendation: discuss with Alexandra before any bulk fetching.** Frame it as "we want to build our own transcription capability to stop depending on a third-party API"

**Alternative legitimate data sources:**

| Source | Size | Format | Quality | License |
|--------|------|--------|---------|---------|
| **GuitarSet** | 360 clips (3 hours) | Hexaphonic audio + JAMS annotations | Gold standard (per-string ground truth) | CC BY 4.0 |
| **GuitarPro files (online)** | 800K+ files exist | GP3/4/5 binary | Variable (crowd-sourced) | Grey area |
| **DadaGP** | 26,181 tracks | GuitarPro token sequences | High (curated from Ultimate Guitar) | Research use |
| **IDMT-SMT-Guitar** | 4,700 notes | Isolated plucked notes | Perfect for single-note models | Academic |
| **Slakh2100** | 2,100 MIDI-synth tracks | MIDI + rendered audio | Synthetic but large | CC BY 4.0 |
| **MusicNet** | 330 pieces | Audio + MIDI | Classical (not ideal for guitar) | CC BY-SA 4.0 |

**DadaGP is the most promising large dataset.** It contains 26K GuitarPro tracks parsed into a token format by researchers at Queen Mary University of London. Paper: "DadaGP: A Dataset of Tokenized GuitarPro Songs" (ISMIR 2021). The tokens include note pitch, string, fret, duration, articulations -- essentially everything Songsterr provides. Licensed for research use.

### 2b. Songsterr tab data format

As documented above: JSON with per-beat, per-note granularity including:
- **Pitch:** string index (0-5) + fret (0-24) -- no ambiguity
- **Rhythm:** duration type (1/2/4/8/16/32/64), dotted, tuplet
- **Articulation:** bends (with point curves), slides (6 types), hammer-on/pull-off, vibrato, palm mute, staccato, harmonics (3 types), dead notes, ties, let-ring
- **Dynamics:** per-beat velocity (ppp through fff)
- **Structure:** time signatures, tempo automations, section markers, chord annotations, tuning

This is richer than GuitarPro binary because it is already parsed to clean JSON. One song's guitar track is typically 50-200 KB of JSON.

### 2c. Pairing tab data with separated guitar stems

The training pipeline needs (audio, tablature) pairs. Here is how to create them:

**Option A: Synthetic rendering (recommended for initial training)**
1. Take GuitarPro/DadaGP tab data
2. Render to audio using a guitar synthesizer (FluidSynth + SoundFont, or Neural Amp Modeler)
3. Pair: rendered audio + ground truth tab
4. Advantage: perfect alignment, unlimited augmentation
5. Disadvantage: synthetic audio sounds different from real recordings

**Option B: Real audio + matched tabs**
1. Take songs where we have both the original audio AND a correct tab
2. Run BS-RoFormer stem separation to isolate guitar
3. Pair: separated guitar stem + Songsterr/DadaGP tab
4. Advantage: real-world audio distribution
5. Disadvantage: separation artifacts, tab may not perfectly match the recording (different arrangement, timing drift)

**Option C: GuitarSet (gold standard, small)**
1. Hexaphonic recordings with per-string annotations
2. Already used by trimplexx and FretNet
3. 360 clips, 3 hours total -- enough for initial training, not for robust generalization

**Recommended approach: Train on GuitarSet first (Option C), then fine-tune on synthetic renders (Option A), then adapt to separated stems (Option B).**

---

## 3. Model Training Approach

### Strategy Overview

Three architectures are worth considering, in priority order:

| Approach | Input | Output | Training Data | Key Advantage |
|----------|-------|--------|---------------|---------------|
| **A: trimplexx CRNN** | CQT spectrogram | String+fret per frame | GuitarSet (gold), then fine-tune | Proven 87% F1, end-to-end |
| **B: MIDI-to-Tab network** | MIDI (from Basic Pitch) + audio features | String+fret assignments | DadaGP + any MIDI corpus | Leverages existing AMT, focuses on tab assignment |
| **C: Fretting-Transformer** | MIDI note sequence | String+fret sequence | DadaGP tokens | T5-based, highest reported accuracy, two-stage |

### Approach A: trimplexx CRNN (Primary)

**Phase A1: Baseline on GuitarSet**
1. Clone `github.com/trimplexx/music-transcription`
2. Download GuitarSet via mirdata (~2 GB)
3. Train using documented best config:
   - GRU, hidden=768, 2 layers, bidirectional
   - lr=0.0003, onset_loss_weight=9.0
   - Data augmentation: time stretch, noise, reverb, EQ, SpecAugment
   - 300 epochs with early stopping (patience=25)
4. Target: TDR F1 > 0.85 on held-out test
5. **Time:** ~4 hours on Modal A10G, **cost:** ~$5

**Phase A2: Fine-tune on synthetic tab renders**
1. Take DadaGP guitar tracks (26K available)
2. Render audio using FluidSynth + guitar SoundFont (free, automation-friendly)
3. Convert DadaGP tokens back to JAMS-like annotation format (string, fret, onset, duration)
4. Fine-tune the GuitarSet-trained CRNN on this synthetic data
5. Use aggressive audio augmentation to bridge the synth-to-real gap:
   - Random amp sim (overdrive, distortion, clean)
   - Random reverb/delay
   - Random EQ curves
   - Background noise (crowd noise, room ambience)
   - Pitch shifting (simulates alternate tunings)
6. **Time:** ~8-12 hours training on A10G (larger dataset), **cost:** ~$10-15

**Phase A3: Domain adaptation with separated stems**
1. Process a set of songs through BS-RoFormer to get guitar stems
2. Where we have matched tab data (from Songsterr or DadaGP), create (stem, tab) pairs
3. Fine-tune or continue training on these real-world separated stems
4. This teaches the model to handle separation artifacts
5. **Time:** depends on dataset size, ~4-8 hours

### Approach B: MIDI-to-Tab Network (Complementary)

This approach separates AMT (audio -> MIDI) from tab assignment (MIDI -> string+fret). It is useful because:
- Basic Pitch already gives decent MIDI output
- The tab assignment problem is simpler than full audio-to-tab
- We can use a much larger training dataset (any MIDI + tab pair works)

**Architecture: Sequence-to-sequence with context**
```
Input: MIDI note sequence [pitch, onset, duration, velocity, ...]
       + audio features (CQT embedding per note for timbre context)
Output: per-note [string, fret] assignment
```

This is essentially what the Fretting-Transformer does, but we can build a simpler version:

1. **Input encoder:** Embed each MIDI note as a vector [pitch_embed, duration_embed, context_embed]
2. **Context:** CQT spectrogram features around each note onset (tells the model about timbre/string resonance)
3. **Decoder:** Bidirectional GRU or Transformer that outputs (string, fret) per note
4. **Loss:** Cross-entropy on string (6 classes) + fret (22 classes), weighted by playability score

**Training data:** DadaGP provides MIDI + tab pairs directly. 26K tracks = massive dataset.

**Advantage over A\*-Guitar:** A\*-Guitar optimizes for hand movement but has no acoustic awareness. A learned model can recognize that a note played on a thin string sounds different from the same pitch on a thick string (timbre cues from the audio).

**Time:** ~2-3 days to build the data pipeline + model + train. **Cost:** ~$10-20 on Modal.

### Approach C: Full Fretting-Transformer (Longer-term)

The Fretting-Transformer (2025) uses a T5 encoder-decoder architecture to map MIDI sequences to tab sequences. It reportedly outperforms A\* on the GuitarToday dataset. No published weights.

**Decision:** Monitor for released weights. If none by mid-April, skip and rely on Approaches A+B.

---

## 4. Self-Evaluation: How Does the AI Check Its Own Output?

### Automated metrics (run on every song)

| Metric | What it measures | How to compute |
|--------|-----------------|----------------|
| **Tab Detection Rate (TDR)** | % of notes with correct string+fret assignment | Compare predicted (string, fret) against ground truth for each onset |
| **Multi-Pitch F1 (MPE)** | Note-level pitch accuracy (ignoring string assignment) | mir_eval note-level metrics |
| **Onset F1** | Timing accuracy of detected note onsets | mir_eval onset metrics (50ms tolerance) |
| **Playability score** | How physically natural the tab is | A\*-Guitar cost function: total hand movement, span violations, impossible stretches |
| **Chord voicing accuracy** | Do multi-note groups form recognizable chord shapes? | Compare voicings against known chord dictionary |
| **Articulation accuracy** | Are bends/slides/hammer-ons detected correctly? | Requires annotated test set with articulation ground truth |

### Comparison pipeline against Songsterr (A/B testing)

For any song where we have the Songsterr tab available:

```
Song Audio
  ├── BS-RoFormer → Guitar Stem → AI Tab Model → Predicted Tab
  └── Songsterr API → Reference Tab
       ↓
  Compare: predicted vs reference
    - Note-level matching (onset ± 50ms, pitch exact)
    - String assignment agreement (how often same string chosen)
    - Fret assignment agreement (how often same fret chosen)
    - Duration accuracy (within 1 quantization step)
    - Articulation precision/recall
```

This gives us a song-level quality score. We can rank songs by difficulty and identify where the model struggles.

### Confidence scoring for production

The model outputs logits for each fret class. We can compute per-note confidence as:
```python
confidence = softmax(fret_logits).max(dim=-1)
song_confidence = confidence.mean()  # or percentile
```

When confidence drops below a threshold, fall back to the existing pipeline (Basic Pitch + A\*-Guitar) or flag the song for manual review.

### Human evaluation loop

For launch, manually review the first 50-100 transcriptions:
- Load GP5 in the AlphaTab viewer (already built into StemScriber practice mode)
- Play along with the audio -- does it feel right?
- Flag specific failure modes: wrong position, missed notes, phantom notes, bad rhythm

---

## 5. Cost and Timeline Estimates

### Training data preparation

| Task | Time | Cost |
|------|------|------|
| Download GuitarSet via mirdata | 30 min | Free |
| Download DadaGP dataset | 1 hour | Free |
| Build DadaGP-to-JAMS converter | 4-6 hours dev | Free |
| Render DadaGP tracks with FluidSynth | 8-12 hours compute | ~$5 (Modal CPU) |
| Build data augmentation pipeline | 2-3 hours dev | Free |
| **Total data prep** | **~2-3 days** | **~$5** |

### Model training

| Task | Time | Cost |
|------|------|------|
| trimplexx on GuitarSet (baseline) | 4 hours GPU | ~$5 |
| trimplexx on DadaGP synthetic (fine-tune) | 8-12 hours GPU | ~$15 |
| MIDI-to-Tab network training | 4-8 hours GPU | ~$10 |
| Hyperparameter sweeps (3-5 runs) | 20-40 hours GPU | ~$30-50 |
| **Total training** | **~3-5 days** | **~$60-80** |

### Integration and testing

| Task | Time | Cost |
|------|------|------|
| Build trimplexx wrapper (`trimplexx_transcriber.py`) | 4-6 hours dev | Free |
| Wire into transcription.py fallback chain | 2 hours dev | Free |
| Build evaluation pipeline (predicted vs reference) | 4-6 hours dev | Free |
| A/B test on 20-50 songs | 1 day | ~$5 (Modal compute) |
| Production deploy to Modal | 2-4 hours | Free |
| **Total integration** | **~2-3 days** | **~$5** |

### Grand total

| | Time | Cost |
|---|------|------|
| **Data prep** | 2-3 days | $5 |
| **Training** | 3-5 days | $60-80 |
| **Integration** | 2-3 days | $5 |
| **Total** | **7-11 days** | **$70-90** |

---

## 6. MVP by May 5 vs Full Version

### MVP (ship by May 5) -- 10 days from now

**Scope:** Deploy trimplexx CRNN trained on GuitarSet only, as highest-priority guitar transcriber.

| Component | Status | Work remaining |
|-----------|--------|----------------|
| Download GuitarSet | Not started | 30 min |
| Train trimplexx on GuitarSet | Not started | 4 hours (Modal A10G) |
| Build `trimplexx_transcriber.py` wrapper | Not started | 4-6 hours dev |
| Wire into `transcription.py` fallback chain | Not started | 2 hours dev |
| Deploy checkpoint to Modal Volume | Not started | 1 hour |
| A/B test with 5 songs | Not started | 2 hours |
| **Total** | | **~2 days** |

**What this gives you:**
- 87% F1 guitar tab transcription (up from ~79% Basic Pitch + FretMapper heuristic)
- Native string+fret output (no more FretMapper guessing)
- Bypasses A\*-Guitar for guitar tracks (A\* still used for MIDI-sourced bass)
- Falls back to existing pipeline when confidence is low

**What it does NOT give you:**
- No benefit from large-scale training data (GuitarSet = 3 hours of clean acoustic only)
- Poor generalization to distorted electric guitar, metal, heavy effects
- No articulation detection improvement (still no bend/slide from audio)
- No bass tab improvement (trimplexx is guitar-only)
- Still uses Songsterr for "professional" reference tabs when available

**This MVP is realistic for May 5** and is a meaningful quality improvement. It is also Phase 2 from the existing transcription-upgrade-plan.md.

### Full Version (post-launch, 4-8 weeks)

| Phase | Timeline | What it adds |
|-------|----------|-------------|
| **Phase 1: DadaGP fine-tuning** | Week 1-2 | Train on 26K synthetic guitar tracks -- massive generalization improvement for electric, distorted, varied styles |
| **Phase 2: MIDI-to-Tab network** | Week 2-3 | Separate model for string+fret assignment from MIDI -- improves bass tab, works with any AMT front-end |
| **Phase 3: Domain adaptation** | Week 3-4 | Fine-tune on real separated stems -- model learns to handle BS-RoFormer artifacts |
| **Phase 4: FretNet articulations** | Week 4-6 | Add continuous pitch contours -- enables bend/slide/vibrato detection from audio |
| **Phase 5: Self-evaluation loop** | Week 5-7 | Automated A/B testing against reference tabs, confidence-based routing, quality dashboards |
| **Phase 6: Songsterr removal** | Week 7-8 | Remove Songsterr API calls entirely once AI quality exceeds Songsterr on blind A/B tests |

**Full version cost estimate:** $150-250 total (mostly Modal GPU hours for training iterations).

---

## 7. Recommended Execution Plan

### Week of April 7 (now)

1. **Train trimplexx on GuitarSet** (4 hours Modal, $5)
   - Clone repo, download GuitarSet, run training
   - Validate TDR F1 > 0.85 on test set
2. **Build `trimplexx_transcriber.py`** (half day)
   - Wrapper that loads model, runs CQT, produces note list
   - Output format compatible with `midi_to_gp.py`
3. **Wire into transcription.py** (2 hours)
   - Highest priority in guitar fallback chain
   - Quality score threshold for auto-fallback

### Week of April 14

4. **Download DadaGP dataset** (free, research license)
5. **Build DadaGP-to-training-data converter**
   - Parse GuitarPro tokens to JAMS annotation format
   - Render audio with FluidSynth + guitar SoundFonts
6. **Start DadaGP fine-tuning** (12 hours Modal, $15)

### Weeks of April 21-28

7. **Train MIDI-to-Tab network** for bass tab improvement
8. **Domain adaptation** with real separated stems
9. **Build A/B evaluation pipeline** against Songsterr reference tabs

### Post-May 5 launch

10. **FretNet articulation layer**
11. **Confidence-based routing** (AI vs Songsterr per song)
12. **Songsterr removal** when AI wins blind A/B tests

---

## 8. Legal Considerations

- **DO NOT bulk-scrape Songsterr** without legal guidance
- **DadaGP** is explicitly licensed for research -- check if commercial fine-tuning is covered
- **GuitarSet** is CC BY 4.0 -- commercial use OK with attribution
- **GuitarPro files** found online are user-generated transcriptions of copyrighted songs -- training on these is legally grey (similar to training on book text)
- **Discuss with Alexandra:** Frame as "building our own transcription technology to replace third-party dependency" -- the training data question is about what constitutes fair use for model training
- **Safest path:** Train on GuitarSet (CC BY 4.0) + DadaGP (research license) + our own synthetic renders. Avoid Songsterr bulk fetch entirely until legal clarity.

---

## 9. Key Files Reference

### Existing StemScriber code
- `backend/routes/songsterr.py` -- Songsterr API integration (search, meta, track fetch, chord extraction)
- `backend/songsterr_to_gp.py` -- Songsterr JSON to GP5 converter (reveals full data format)
- `backend/processing/transcription.py` -- transcription dispatcher with fallback chains
- `backend/guitar_tab_transcriber.py` -- Basic Pitch + FretMapper (current production)
- `backend/midi_to_gp.py` -- MIDI to GP5 with A\*-Guitar fret optimizer

### Research docs
- `docs/trimplexx-integration.md` -- trimplexx CRNN audit (architecture, benchmarks, integration plan)
- `docs/fretnet-integration.md` -- FretNet audit (continuous pitch, articulation detection)
- `docs/astar-fret-results.md` -- A\*-Guitar implementation results
- `docs/transcription-upgrade-plan.md` -- full pipeline upgrade roadmap

### External resources
- trimplexx CRNN: `github.com/trimplexx/music-transcription` (MIT)
- FretNet: `github.com/cwitkowitz/guitar-transcription-continuous` (MIT)
- DadaGP: `github.com/dada-bots/dadaGP` (research dataset, ISMIR 2021)
- GuitarSet: via mirdata (`mirdata.initialize('guitarset')`)

---

## 10. Decision Matrix

| Question | Answer |
|----------|--------|
| Can we ship an AI tab model by May 5? | **Yes** -- trimplexx on GuitarSet only, 2 days of work, $5 |
| Should we bulk-fetch from Songsterr? | **No** -- legal risk, ask Alexandra first. Use DadaGP instead. |
| What is the biggest accuracy win? | **Phase A2** -- DadaGP fine-tuning (26K tracks vs 360 GuitarSet clips) |
| When can we fully replace Songsterr? | **6-8 weeks** after launch, once AI wins blind A/B tests on diverse songs |
| Total cost to get there? | **$150-250** in Modal GPU compute |
| What about bass tabs? | MIDI-to-Tab network (Approach B) covers bass. trimplexx is guitar-only. |
| What about articulations (bends/slides)? | FretNet Phase 4 (post-launch). Current pipeline has no audio-based articulation detection. |
