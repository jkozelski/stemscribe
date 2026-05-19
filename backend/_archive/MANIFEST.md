# Archived chord-detection modules

Archived 2026-05-13 per `docs/v3-agent-D-cleanup-deploy-2026-05-13.md`. These files
are kept for git history reference but are NOT imported by the live pipeline. Do not
modify or restore without reading the rationale below.

## V3.1 architecture context

The live chord-detection chain is:
1. `processing/detector_router.py` — Claude classifies title/artist into "jazz" or "general"
2. **Jazz path** → `stem_chord_detector.py` (legacy stem-aware family-aware-consistency — nailed Aja 226/226 in Apr 25 sprint)
3. **General path** → `processing/chord_router.py` → ACE (`backend/external/consonance-ACE/`) + Jiang (`backend/external/chord_cnn_lstm/`) per-bar router

Empirical baseline: avg root F1 0.869 on 13-song UG benchmark (+0.156 over the old librosa+V1 chain).

## Why each file was archived

| File | Reason archived |
|---|---|
| `chord_detector.py` | Original librosa-chroma 24-template matcher. Dead since `chord_detector_v8.py` became the primary fallback in `dependencies.py`. The active librosa-style detector now lives at `processing/chord_detector_librosa.py` (different module, full-mix-focused). |
| `chord_detector_v7.py` | Transformer NN with 25 classes. 3rd-tier fallback in the old `dependencies.py` chain, never reached in practice. v8 supersedes. |
| `btc_validator.py` | Standalone CLI for validating BTC predictions against a held-out set. Only consumer was `chord_detector_v10.py` ad-hoc; not in the live pipeline. |
| `btc_stress_test.py` | Standalone BTC inference stress test. Same as above. |
| `essentia_chord_detector.py` | Wraps Essentia's `ChordsDetection`. Essentia is not installed on the prod VPS; the import in `chord_detector_v10.py` always failed at runtime. Removed from both call sites (`chord_detector_v10.py:1057` and `guitar_tab_transcriber.py:265`). |
| `chord_retrain.py` | Training-only CLI for V8 retraining. Outside the pipeline. |
| `chord_training_pipeline.py` | Training-data builder CLI. Outside the pipeline. |
| `chord_library_cleanup.py` | One-shot cleanup tool for the killed scraped chord library. Library was deleted Apr 16 per legal; this tool has no remaining purpose. |
| `chord_eval.py` | Offline CLI for evaluating detector output against Songsterr ground truth. Imported `chord_detector_v8` and `chord_detector` — both still present. Tool itself is unused. |
| `chord_analysis.py` | Standalone MIDI chord-analysis CLI using `music21.chordify()`. Not in the pipeline. |
| `chord_recall_rag.py` | RAG build/query CLI for the killed lyric-embedding feature (15K-song lyric corpus). Lawyer killed the data product Apr 10; index already archived to Desktop `stemscriber-archive/chord_recall_index-2026-04-24/`. |
| `calibrate_chord_detector.py` | Offline calibration tool. Not in the pipeline. |
| `test_chord_id.py` | One-off chord-ID smoke test at backend root (not under `tests/`). Replaced by unit tests in `tests/test_seventh_preservation.py`, `tests/test_smooth_qualities.py`, etc. |

## Files NOT archived (still load-bearing)

- `chord_detector_v8.py` — current `dependencies.py` primary detector.
- `chord_detector_v10.py` — 2nd-tier fallback in `dependencies.py`. Holds the K-K key detector + tuning-comp reference pattern. The Essentia ensemble code was removed (essentia_chord_detector archived) but the BTC + V8 fallback hybrid is intact.
- `stem_chord_detector.py` — the V3.1 jazz path. Apr 25 sprint code.
- `processing/chord_detector_librosa.py` — pre-V3.1 librosa template detector. Default fallback when `USE_ACE_ROUTER_DETECTOR=false`.
- `processing/chord_corrector_anthropic.py` — Anthropic correction. Gated by `ENABLE_ANTHROPIC_CORRECTION`. V3.1 default is OFF on the ACE path.
- `chord_lookup.py` — filesystem-based chord library lookup. Still used for the 20 Kozelski-original charts that survived the Apr 16 legal cleanup.
- `chord_pattern_analyzer.py` — song-structure detection via chord pattern repetition.
- `chord_theory.py` — practice-mode scale/mode mapping.
- `chord_accuracy.py` — imported by `tests/test_self_healing.py`.
- `tuning_detector.py` — tuning offset helpers. Currently NOT wired into V3.1; available for future use (the Mar 9 root-cause doc identified tuning as the #1 issue for the F#m→G shift, but Agent B's May 13 experiment showed `librosa.estimate_tuning` net-no-op or worse with corrector).
- `midi_chord_detector.py` — gated behind `ENABLE_MIDI_DETECTOR`; post-launch tuning task.

## Restoration

If a future session needs to restore any of these:
```bash
cd ~/stemscribe
git mv backend/_archive/<filename> backend/<filename>
# Re-add the import edits in dependencies.py / guitar_tab_transcriber.py / chord_detector_v10.py
```

Reconsider the v10 archive in the **post-V3.1 retrospective** (Jul 1, ~10 days post-launch)
once Jiang+ACE has been on for 30+ days without falling back to V8.
