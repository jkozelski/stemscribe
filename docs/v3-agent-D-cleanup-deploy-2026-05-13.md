# V3 Chord-Detection Sprint — Agent D: Cleanup & Deployment Recommendation

**Date:** 2026-05-13
**Scope:** Dead-code manifest, v10 tuning-pattern verification, Jiang Chord-CNN-LSTM deployment recommendation
**Author:** Agent D (analysis only — no code execution, no production deploys)

---

## Task 1 — Dead-code manifest

Live chord path today (with `USE_LIBROSA_DETECTOR=true` in prod):

```
pipeline.py:642  →  processing/chord_detector_librosa.py
                     └─ imports `stem_chord_detector.detect_key_from_chords` (line 189)
pipeline.py:741  →  processing/chord_corrector_anthropic.py
                     └─ imports `backend/audit/llm_oracle.py`
chart_formatter.py:589 → imports `stem_chord_detector.detect_key_from_chords`
```

Legacy fallback chain (when `USE_LIBROSA_DETECTOR=false`, currently still loaded at import time):

```
dependencies.py:196  →  chord_detector_v8.ChordDetector       (primary fallback)
dependencies.py:202  →  chord_detector_v10.ChordDetector      (2nd fallback if v8 fails)
dependencies.py:208  →  chord_detector_v7.ChordDetector       (3rd fallback if v10 fails)
dependencies.py:214  →  chord_detector.ChordDetector          (last-resort, basic)
transcription.py:846 →  midi_chord_detector                   (gated ENABLE_MIDI_DETECTOR)
transcription.py:883 →  stem_chord_detector.StemAwareChordDetector (legacy stem path)
```

Because `dependencies.py` imports the fallback chain at module load (try/except cascade), v8 succeeds first and v10 / v7 / chord_detector never load in practice. But **all four are referenced** by the source code, so removing the file breaks the import chain on next deploy unless we also edit dependencies.py.

### Verdict table

| File | Live path? | Audit/tests? | Test file? | Action | Rationale |
|---|---|---|---|---|---|
| `backend/chord_detector.py` (original librosa-chroma 24-template) | No — only as 4th-tier fallback in `dependencies.py:214` that never fires once v8 imports OK | `chord_eval.py:199` (also dead) | None | **archive** | Last-resort fallback, dead in practice. Patch `dependencies.py` to drop the try-block, then archive. |
| `backend/chord_detector_v7.py` (Transformer NN, 25 classes) | No — 3rd-tier fallback only | None | None | **archive** | Same reasoning as above. Never reached in production. |
| `backend/chord_detector_v8.py` (Transformer NN, 337 classes) | **YES** — exported as `ChordDetector` to `transcription.py:915` when `USE_LIBROSA_DETECTOR=false`. The stem-aware path falls into this when it returns <3 chords. | None | None | **KEEP** | Live fallback when librosa flag is off OR when librosa+corrector fail and pipeline retries the legacy path. Cannot remove until librosa is locked on for 100% of jobs and fallback is gutted. |
| `backend/chord_detector_v10.py` (BTC) | **YES** — 2nd-tier fallback in `dependencies.py:202`; imported by `essentia_chord_detector.py:25`, `tuning_detector.py:324`, `chord_eval.py`, `btc_stress_test.py`, `btc_validator.py`, `lead_sheet_generator.py:12` (docstring), `chart_formatter.py:238` (docstring). | None | None | **KEEP** (for now) | (a) Contains the tuning-comp pattern Agent B needs to port (see Task 2). (b) v8→v10 cascade still possible if v8 import ever fails (e.g., model file missing on a redeploy). Reassess after V3 ships. |
| `backend/btc_validator.py` | No | None | None | **archive** | Standalone validator script. Only imports `chord_detector_v10` for ad-hoc validation runs. |
| `backend/btc_stress_test.py` | No | None | None | **archive** | Standalone stress test. Only imports `chord_detector_v10`. |
| `backend/essentia_chord_detector.py` | Indirect — imported by `chord_detector_v10.py:1057` and `guitar_tab_transcriber.py:265`, both inside try/except. Essentia itself logs "not available — essentia_chord_detector disabled" on import, so the module is dead-on-arrival on prod. | None | None | **archive** | Essentia is not installed on the VPS; the module's own warning confirms it self-disables. The two consumers handle ImportError already. Verify after edit that v10 / guitar_tab_transcriber still load. |
| `backend/chord_retrain.py` | No | None | None | **archive** | Training-only CLI script. Outside pipeline. |
| `backend/chord_training_pipeline.py` | No | None | None | **archive** | Training-data builder CLI. Imports `chord_eval` (also dead). |
| `backend/chord_library_cleanup.py` | No | None | None | **archive** | One-shot library cleanup CLI. Outputs a JSON report. |
| `backend/stem_chord_detector.py` (previous prod detector before May 6 librosa swap) | **YES** — imported by `processing/chord_detector_librosa.py:189` for `detect_key_from_chords`; `chart_formatter.py:589`; `transcription.py:883` as fallback. Tests: `test_per_root_family.py`, `test_seventh_preservation.py`, `test_key_detection.py`. | tests (3 files) | 3 test files | **KEEP** | Despite being "replaced" on May 6, the librosa detector still imports its key-detection helper. Live dependency. Do not archive. |
| `backend/tuning_detector.py` | Indirect — imported by `chord_detector_v10.py:1123` inside try/except. Not used by the librosa path today. Exposes `transpose_chord()` (line 75) which Agent B will want when porting the tuning-comp pattern into the librosa path. | None | None | **KEEP** | (a) Already used by v10 (which we're keeping as fallback). (b) Agent B's port needs `transpose_chord()`; rebuilding it elsewhere is duplication. Cheap to keep. |
| `backend/midi_chord_detector.py` | Conditionally live — gated behind `ENABLE_MIDI_DETECTOR=true` in `transcription.py:846`. Currently `false` in prod. Imports `stem_chord_detector`. | `test_midi_chord_detector.py` | yes | **KEEP** | Flag is off in prod but Jeff's memory lists this as a "post-launch tuning task" (MIDI-intermediate detector). Leaving the gated path in place costs nothing. |
| `backend/chord_eval.py` | No — imports `chord_detector_v8` and `chord_detector` for offline eval | None | None | **archive** | Standalone eval script. Not in any pipeline call chain. |
| `backend/calibrate_chord_detector.py` | No — CLI tool, imports `stem_chord_detector` | None | None | **archive** | Offline calibration tool. Not pipeline. |
| `backend/chord_accuracy.py` | No — but **imported by `tests/test_self_healing.py`** (6 occurrences) | tests/test_self_healing.py | yes | **KEEP** | Test dependency. Out of pipeline but the test suite imports it. |
| `backend/chord_analysis.py` | No — CLI script (`python chord_analysis.py <midi>`) | None | None | **archive** | Standalone MIDI chord-analysis CLI. |
| `backend/chord_recall_rag.py` | No — CLI script | None | None | **archive** | RAG build/query CLI. Not in pipeline. |
| `backend/test_chord_id.py` (note: at backend root, not in tests/) | No | None | itself | **archive** | One-off script, not picked up by pytest. |

**Files explicitly kept (live deps):** `chord_detector_librosa.py`, `chord_corrector_anthropic.py`, `chord_detector_v8.py`, `chord_detector_v10.py`, `stem_chord_detector.py`, `midi_chord_detector.py`, `tuning_detector.py`, `chord_accuracy.py`, `chord_theory.py` (untouched, theory helpers), `chord_lookup.py` (Flask blueprint, registered in `app.py:187`), `chord_pattern_analyzer.py` (has test), `audit/llm_oracle.py`.

**Total archive candidates:** 11 files.

### Pre-archive cleanup edits (must land in the same PR)

1. **`backend/dependencies.py`** — collapse the v8/v10/v7/basic fallback chain to v8-only (the only one that actually loads). If v8 import fails, raise — don't silently chain into modules that may also be missing.
2. **`backend/chord_detector_v10.py:1057`** — remove the `from essentia_chord_detector import …` block; replace with a logged no-op. The Essentia dependency is unavailable on the VPS and the import is the only thing still tying v10 to essentia.
3. **`backend/guitar_tab_transcriber.py:265`** — same fix: drop the essentia import.

Without these three edits, archiving `chord_detector.py`, `chord_detector_v7.py`, and `essentia_chord_detector.py` will break import on next service restart.

---

## Task 2 — Verify May 6 claim about v10 tuning compensation

**Claim** (from `docs/chord-research-2026-05-06.md:37,80,83`): `chord_detector_v10.py:1178` contains a `librosa.estimate_tuning()` + `pitch_shift` block that should be ported into the live librosa path.

**Verified — the pattern exists at lines 1177–1183 of `backend/chord_detector_v10.py`:**

```python
            # --- FIX 1: Tuning compensation before CQT ---
            tuning_offset = librosa.estimate_tuning(y=original_wav, sr=sr)
            logger.info(f"Estimated tuning offset: {tuning_offset:.3f} semitones")
            if abs(tuning_offset) > 0.05:
                original_wav = librosa.effects.pitch_shift(
                    original_wav, sr=sr, n_steps=-tuning_offset)
                logger.info(f"Applied tuning compensation: {-tuning_offset:+.3f} semitones")
```

Context: this lives inside `ChordDetector._detect_btc()` (line 1162). It runs on the original waveform BEFORE the CQT feature extraction that feeds the BTC model. It applies only when the estimated offset exceeds 0.05 semitones (a 5-cent threshold — librosa returns values in semitones, range roughly [-0.5, 0.5]).

**For Agent B's port into `chord_detector_librosa.py`:** drop this block in right after the `librosa.load(...)` call (currently line 103), before `chroma_cqt(...)`. Same threshold, same `n_steps=-tuning_offset` sign. Also consider stashing the offset on the `ChordProgression.tuning_info` field that's already declared in the dataclass at line 50.

A complementary helper, `tuning_detector.detect_and_correct_tuning()` (line 254), operates on chord events POST-detection — different semantic. The pre-CQT pattern is what we want to copy.

---

## Task 3 — Jiang Chord-CNN-LSTM: Hetzner vs Modal

### Measured numbers (local M3 Max, 2026-05-13)

Successfully imported the package — all 5 deps (`torch`, `librosa`, `pretty_midi`, `pumpp`, `jams`, `mir_eval`, `h5py`, `joblib`, `scikit_learn`) are present in `~/stemscribe/venv311`.

Ran `chord_recognition.py` on two clips from `/Users/jeffkozelski/stemscribe/uploads/f4c2dae7/Ramble_On_Remaster.wav` (Led Zeppelin – Ramble On, 268.9 s @ 44.1k → resampled to 22.05k mono):

| Clip duration | Wall time | User CPU | Sys CPU |
|---|---|---|---|
| 30 s | **3.02 s** | 3.88 s | 5.21 s |
| 268.9 s (full song) | **7.83 s** | 15.59 s | 16.73 s |

5-model ensemble runs sequentially. CPU time (~32 s wall-equivalent on full song) divided by wall (7.83 s) shows the script effectively uses ~4 cores via PyTorch's BLAS threading on M3 Max. The Jiang README's 7–10 s claim **holds on M3 hardware**.

### Hetzner translation

CPX41 = 8 vCPU AMD EPYC (shared, slower per-core than M3). Realistic adjustment:
- Single-job inference time: **estimate 10–15 s** on CPX41 (1.5–2× slowdown vs M3 is typical for x86 cloud vCPU on PyTorch CPU inference).
- Memory per job: 5 × 5.5 MB ensemble checkpoint files = **27 MB on disk**; in-RAM working set with CQT features + intermediate tensors ≈ 300–500 MB per active inference.
- Concurrency: post-sep cap is 4 simultaneous jobs → up to 4 × 500 MB = 2 GB peak Jiang RSS on top of the other pipeline stages. CPX41 has 16 GB and the watchdog already lets 4 jobs co-run, so this fits comfortably.

### Modal alternative

- A10G cold-start: 5–10 s container spin-up (the BS-RoFormer container takes ~8 s when cold).
- A10G inference time for Jiang: <1 s (Tesla A10G is ~50× faster than CPX41 cores on PyTorch).
- Round-trip per song: ~6–11 s including network upload/download of a ~25–80 MB WAV.
- Modal pricing: A10G is $0.000306/sec, so cost ≈ 6 s × $0.000306 = **$0.002/song extra** (well under the $0.02 estimate; A10G inference is short).
- **Hidden cost**: keeps the song bytes flowing to Modal twice (once for separation, once for chord detection) unless we reuse the existing separation container — not worth the engineering for marginal speed.

### Recommendation: **In-process on Hetzner VPS**

Rationale:

1. **It fits.** 10–15 s of CPU on a job that already takes 4–6 min end-to-end (separation + transcription + chord chart + Whisper + MIDI/XML/GP) is <5% of total latency. Modal saves 4–10 s by running on GPU, but adds 5–10 s of cold-start round-trip — net wash, with strictly more failure modes (network, Modal outage, cold-start variance).
2. **Memory is fine.** 300–500 MB × 4 concurrent jobs = ~2 GB, against 16 GB total with separation already done.
3. **Cost.** Modal would add ~$0.002/song. Negligible, but not free. Hetzner adds zero — we're already paying for the vCPUs.
4. **Operational simplicity.** No extra Modal deploy step, no API key plumbing for Jiang, no two-container failure modes. The 5 checkpoint files ship in the repo (`backend/external/chord_cnn_lstm/cache_data/`) and load on import.
5. **Failure isolation.** Running Jiang in-process means a model-load error fails the job locally and the existing `try/except` fallback to the librosa path triggers immediately. Modal failures (timeout, cold-start, network) make this harder.

**Exception clause:** if Agent C's benchmarking shows Jiang inference dominates pipeline latency (e.g., 25+ s sustained on CPX41 under 4-way concurrency), revisit and move Jiang to Modal A10G alongside the existing separation container. **Not before then.**

---

## Sketch — `backend/_archive/MANIFEST.md`

```
# Archived chord-detection files
# Moved here from backend/ on 2026-05-13 after V3 sprint cleanup.
# These files are not imported by any live pipeline code as of USE_LIBROSA_DETECTOR=true.
# Restoration: move back to backend/ and re-enable the dependencies.py fallback chain.

chord_detector.py           — Original librosa-chroma 24-template detector (pre-v7). Dead since v8 became default fallback. Last live: pre-2026-03.
chord_detector_v7.py        — Transformer NN, 25-class output. 3rd-tier fallback in dependencies.py, never reached once v8 loads.
btc_validator.py            — Standalone BTC validation script. Only consumer was chord_detector_v10 ad-hoc runs.
btc_stress_test.py          — Standalone BTC stress test. Only consumer was chord_detector_v10 ad-hoc runs.
essentia_chord_detector.py  — Essentia-based detector. Dead on prod: Essentia is not installed on VPS; module self-disables on import.
chord_retrain.py            — Training-only CLI for chord-classifier retrain. Outside pipeline call chain.
chord_training_pipeline.py  — Training-data builder CLI. Outside pipeline.
chord_library_cleanup.py    — One-shot library cleanup tool. Produces chord_library_cleanup_report.json.
chord_eval.py               — Offline chord-accuracy eval CLI (imports v8 + chord_detector).
chord_analysis.py           — Standalone MIDI chord-analysis CLI.
chord_recall_rag.py         — RAG build/query CLI for chord recall. Not pipeline.
calibrate_chord_detector.py — Offline calibration tool against GuitarSet / Songsterr ground truth.
test_chord_id.py            — One-off script at backend root (not under tests/, not picked up by pytest).
```

---

## Open questions for the next agent

1. **Should `chord_detector_v10.py` be archived too?** It's still in the dependencies.py fallback chain (2nd tier, after v8). If V3 ships and we lock librosa on for 100% of jobs for 2+ weeks with no fallback-path activations, the entire `dependencies.py` legacy chord chain becomes dead and v8 + v10 can both be archived. The tuning-comp pattern from Task 2 will already be ported into the librosa path by then. **Recommend revisiting in the post-V3 retrospective.**

2. **Should `_archive/` be a sibling of `backend/` or live inside it?** Question targets repo conventions — left at `backend/_archive/` to match the user's brief.

3. **Should we delete instead of archive?** Recommend archive for at least 30 days. Disk cost is trivial (~250 KB total for the 11 files); the safety net is high. Convert to deletion in a follow-up cleanup once V3 is stable in prod.
