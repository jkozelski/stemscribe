# MIDI-Intermediate Chord Detector — Architecture Spec

**Date:** 2026-04-23
**Phase:** 1 (spec only — no code)
**Gate to enter Phase 2:** Jeff sign-off on this document
**Phase 0 result:** PASS (see `midi-prototype-results-2026-04-23.md`) — 7ths present in 100% of eligible bars on *Alright*.

---

## 1. Summary

Replace the CQT→BTC→pitch-class-pruning path's matching layer with a note-based matcher fed by Basic Pitch MIDI. The upstream reliable components (BS-RoFormer stems, `tempo_beats.extract_grid`, `bass_root_extraction.extract_bass_roots`) are unchanged. The existing detector stays in place as the fallback when the new path fails or its feature flag is off.

**New file:** `backend/midi_chord_detector.py`
**New test file:** `backend/tests/test_midi_chord_detector.py`
**Integration edits:** `backend/processing/transcription.py` (one branch), `backend/processing/pipeline.py` (one reorder), `backend/dependencies.py` (flag), `.env.example` (flag).

**Everything else is untouched.** No frontend edits. No chart_formatter edits. No changes to bass_root_extraction / tempo_beats / BTC / V8 / Essentia.

---

## 2. What's kept as-is

| Component | File | Role |
|---|---|---|
| Stem separation | `backend/processing/separation.py` + Modal BS-RoFormer | Produces `guitar.mp3`, `bass.mp3`, `piano.mp3`, etc. |
| Bar grid | `backend/processing/tempo_beats.py::extract_grid` | Returns `{tempo_bpm, downbeat_times, bar_count, ...}`. |
| Per-bar bass root | `backend/processing/bass_root_extraction.py::extract_bass_roots` | Returns `[{bar, root, confidence, source}, ...]`. |
| Chord chart formatter | `backend/chart_formatter.py` | Consumes `ChordEvent` list + grid + bass_roots. |
| Legacy detector | `backend/stem_chord_detector.py` | Fallback when flag off or Basic Pitch fails. |
| Legacy BTC/V8 hybrid | `backend/chord_detector_v10.py` | Second-line fallback (unchanged behavior). |

**Note:** `detect_bass_root` inside `stem_chord_detector.py` is not the same function as `processing/bass_root_extraction.py::extract_bass_roots`. The new detector uses the latter (per-bar, aligned to grid).

## 3. Pipeline reorder

Today's order inside `pipeline.py::process_job` (for context):

1. `detect_chords_for_job` (line ~359)
2. `extract_grid` (line ~378, guarded by `if job.chord_progression`)
3. `extract_bass_roots` (line ~404, guarded by same)
4. `format_chart` (line ~482)

**New order (required by Phase 2):**

1. `extract_grid` — grid depends only on drums stem + mix, not on chord detection. Move up.
2. `extract_bass_roots` — depends on grid + bass stem, independent of chord detection. Move up.
3. `detect_chords_for_job` — now has `job.metadata['grid']` and `job.metadata['bass_roots']` available as inputs when they exist.
4. `format_chart` — unchanged.

**Backward compat:** legacy detectors ignore the new kwargs they receive; they still work. The reorder is safe with the flag OFF because the existing path never depended on chord detection having run first.

**Guards:** steps 1 and 2 drop the `if job.chord_progression` guard — we want the grid and bass roots computed even if chord detection later fails, so downstream rendering has something to hang on. This matches the spirit of the audit doc's comment that "the grid is correct, the render just wasn't using it."

## 4. New module — `backend/midi_chord_detector.py`

### 4.1 Public interface

```python
@dataclass
class MidiNote:
    start: float          # seconds
    end: float            # seconds
    pitch: int            # MIDI pitch 0-127
    amplitude: float      # 0.0-1.0
    stem: str             # "guitar" | "piano" (bass excluded from harmony set)

@dataclass
class MidiChordDetectorResult:
    chord_progression: ChordProgression   # same type as existing detectors
    per_bar: List[Dict]                    # diagnostic: the matched tones per bar
    basic_pitch_ok: bool                   # whether BP produced usable output on all stems
    notes_by_stem: Dict[str, int]          # note counts per stem, for logs

def detect_chords_from_midi(
    guitar_path: Optional[str],
    piano_path: Optional[str],
    bass_path: Optional[str],
    grid: Dict,                            # from tempo_beats.extract_grid
    bass_roots: List[Dict],                # from bass_root_extraction.extract_bass_roots
    # Tunables (bounded, to avoid cargo-culting defaults forever)
    min_note_length_ms: int = 58,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    bar_energy_floor: float = 0.05,        # pc weight < 5% of bar peak = "noise"
) -> MidiChordDetectorResult: ...
```

`ChordEvent` is imported from `stem_chord_detector` so the return type is identical to the legacy path — no formatter changes required.

### 4.2 Internal steps

1. **Stem → MIDI notes.** Call `basic_pitch.inference.predict(path, ...)` on each provided harmonic stem (guitar, piano). Parse note events into `MidiNote`. **Bass stem is NOT sent through Basic Pitch here** — bass root per bar already comes from `extract_bass_roots` (pyin, which is the right tool for a monophonic bass line). Bass is optionally used for verification only.
2. **Index notes by bar.** Given `grid['downbeat_times']`, each bar is `[downbeat[i], downbeat[i+1]]`. For each bar, collect every `MidiNote` whose `[start, end]` overlaps the window. No `max_notes` cap. No `min_score` floor at this stage.
3. **Per-bar pitch-class weights.** For each of the 12 pitch classes, sum `amplitude × overlap_duration_seconds` across the bar. Normalize by the max pc weight in that bar. Result: `pc_weights ∈ [0, 1]^12`.
4. **Root selection.** If `bass_roots[bar_index]` has confidence ≥ 0.3 and `source ∈ {"pyin", "inherit"}`, use that root. Otherwise, pick the pc with highest `pc_weight` in the bar as a fallback root.
5. **Template match.** For each chord quality template below, score as:
   - `+2` for every template interval whose `pc_weight >= bar_energy_floor` (0.05).
   - `+1` for every template interval below the floor but present at all (`pc_weight > 0`).
   - `−0.5` for every non-template pc with `pc_weight >= bar_energy_floor` (penalty for spurious strong tones).
   - Normalize by `(number of template intervals × 2)` so templates of different sizes compare fairly.
   Pick the highest-scoring template. On ties, prefer the template with more intervals (prefer m7 over m, maj7 over maj, etc.) — this is the crucial anti-triad-bias that the current detector lacks.
6. **Build `ChordEvent`.** `time = downbeat[i]`, `duration = downbeat[i+1] - downbeat[i]`, `chord = root + quality_name`, `confidence = normalized_score`.
7. **Per-bar diagnostics.** Keep a dict per bar with `root, quality, score, matched_tones, pc_weights` so debugging a misfit chord is a log-inspection, not a re-run.

### 4.3 Templates

Initial set (extend later only if regression demonstrates a gap):

```
maj      = (0, 4, 7)          # C, E, G
min      = (0, 3, 7)          # C, Eb, G
7        = (0, 4, 7, 10)      # dom 7
maj7     = (0, 4, 7, 11)      # major 7
m7       = (0, 3, 7, 10)      # minor 7     ← missing today, the whole point
6        = (0, 4, 7, 9)
m6       = (0, 3, 7, 9)
m7b5     = (0, 3, 6, 10)      # half-dim
dim7     = (0, 3, 6, 9)
sus2     = (0, 2, 7)
sus4     = (0, 5, 7)
add9     = (0, 4, 7, 14 mod 12 = 2)
maj9     = (0, 4, 7, 11, 2)
m9       = (0, 3, 7, 10, 2)
9        = (0, 4, 7, 10, 2)
aug      = (0, 4, 8)
```

Output naming matches existing convention: `Cm7`, `Cmaj7`, `C7`, `Csus4`, etc. — what `chart_formatter.py` and `practice.html` already render.

### 4.4 Failure modes and returns

| Condition | Action |
|---|---|
| Basic Pitch raises or returns empty on any of the provided stems | Set `basic_pitch_ok = False`, return empty `ChordProgression`. Caller falls back. |
| `grid` is empty or has < 4 downbeats | Return empty progression. Caller falls back. |
| A bar has zero notes across stems | Emit `ChordEvent(chord="N.C.", quality="none", confidence=0.0)`. (Silence / rest.) |
| `bass_roots[bar]` missing AND no harmonic note in bar | Same as above — N.C. |
| Score for best template < 0.4 (normalized) | Keep the chord but stamp `confidence` low; downstream min-confidence filter may drop it. |

The detector does NOT do key detection, smoothing, or merging — those are downstream responsibilities handled by `postprocess_chords` and `chart_formatter`. Separation of concerns.

## 5. Integration — `transcription.py::detect_chords_for_job`

Adds a new primary branch *before* the existing stem-aware call. Shape (pseudocode, for clarity):

```python
if os.environ.get("ENABLE_MIDI_DETECTOR", "false").lower() == "true":
    grid = job.metadata.get("grid")
    bass_roots = job.metadata.get("bass_roots")
    if grid and bass_roots and (has_guitar or has_piano):
        try:
            from midi_chord_detector import detect_chords_from_midi
            result = detect_chords_from_midi(
                guitar_path=job.stems.get("guitar") if has_guitar else None,
                piano_path=job.stems.get("piano") if has_piano else None,
                bass_path=job.stems.get("bass") if has_bass else None,
                grid=grid,
                bass_roots=bass_roots,
            )
            if result.basic_pitch_ok and len(result.chord_progression.chords) >= 3:
                _store_progression_on_job(job, result.chord_progression, "midi_intermediate")
                logger.info(f"✅ MIDI detector: {len(result.chord_progression.chords)} chords")
                return
            logger.info("MIDI detector produced insufficient output, falling back")
        except Exception as e:
            logger.warning(f"MIDI detector crashed, falling back: {e}")

# ---- existing stem-aware path (UNCHANGED) ----
# ---- existing BTC/V8 path (UNCHANGED) ----
```

**Flag semantics:**
- `ENABLE_MIDI_DETECTOR=false` (default) → new detector is inert; exact current behavior.
- `ENABLE_MIDI_DETECTOR=true` → try new detector first; fall through to legacy on failure.

Flag lives in `.env.example` as a new line. Read via `os.environ` at call time (not import time) so `.env` reloads don't require a restart on VPS.

**`detector_version` label on jobs:** `"midi_intermediate"` when the new path wins, `"stem_aware"` / `"btc_v10"` / `"essentia"` as today when legacy wins. This lets regression diff old vs new outputs per-job trivially.

## 6. Testing

### 6.1 Unit tests — `backend/tests/test_midi_chord_detector.py`

Deterministic, no audio. Each constructs a fake `List[MidiNote]` and a fake `grid` with one bar, calls the internal template-match function, asserts the chord name.

| Test | Notes (pitch, overlapping bar) | Bass root | Expected |
|---|---|---|---|
| `test_cm7_clean` | `{C4, Eb4, G4, Bb4}` each 60-800ms, amplitude 0.6 | `C` | `"Cm7"` |
| `test_fmaj7_clean` | `{F3, A3, C4, E4}` | `F` | `"Fmaj7"` |
| `test_cm_no_seventh` | `{C4, Eb4, G4}` | `C` | `"Cm"` |
| `test_cmaj_with_passing_b7` | `{C4, E4, G4, Bb4@0.02 amp}` | `C` | `"C"` (b7 below energy floor) |
| `test_cmaj_with_real_b7` | `{C4, E4, G4, Bb4@0.5 amp}` | `C` | `"C7"` |
| `test_messy_extras` | Cm7 tones + random stray `F#3@0.2`, `A3@0.15` | `C` | `"Cm7"` (extras penalized but chord tones dominate) |
| `test_ambiguous_partial` | `{C4, G4}` (power chord, no 3rd) | `C` | `"C"` (maj beats min on tie via confidence threshold rule — document behavior) |
| `test_no_notes` | `[]` | `C` | `"N.C."` |
| `test_missing_bass_root` | `{C4, Eb4, G4, Bb4}` strong | none | `"Cm7"` with fallback root = highest-weight pc |
| `test_am7_jeffs_case` | `{A3, C4, E4, G4}` | `A` | `"Am7"` |

Plus the **existing pipeline regression** (no new harness): `./venv311/bin/python -m pytest backend/tests/` must stay green with the flag off (backwards compat) and with the flag on (new tests pass).

### 6.2 End-to-end regression in Phase 4 (not part of Phase 2)

The Phase 4 harness (to be written later) runs the flag on/off against the 20 Kozelski chord library JSONs and one golden 7ths-heavy song (*Alright*, ground truth in `docs/ground-truth-alright-jamiroquai.md`). Gate: the new detector must win on at least 15 of 20 Kozelski songs AND nail Alright's m7 quality on ≥ 80% of bars. That's NOT a Phase 2 task — it's called out here only to show the validation exists.

## 7. Runtime / cost

Phase 0 measurement: **5.8 s guitar + 5.1 s bass + 6.0 s piano = 17 s total CPU time** on an M3 Max laptop for a 4-minute song. VPS is slower (Hetzner CPX31, 4 AMD vCPU), expect roughly 2-3× — call it **35-50 s per song, CPU**. Current stem-aware detector runs in comparable wall time on VPS. No Modal GPU needed.

If VPS wall time becomes a concern post-launch (high traffic), the hook for a Modal offload is one function call — Basic Pitch already runs via stateless `predict()`. Not in scope for Phase 2.

## 8. What could go wrong (risks + mitigations)

| Risk | Probability | Mitigation |
|---|---|---|
| Basic Pitch false-positives pollute some pc sets badly enough to misidentify chord quality | Medium | `bar_energy_floor` + penalty term in scoring. Ties broken by richer quality. Regression against Kozelski catches overt failures. |
| Key-of-F songs where E (maj7 of F) is also the common passing note → false `Fmaj7` promotion | Low-Medium | Energy floor (≥ 5% of bar peak) rejects passing notes. If this shows up in regression, tighten floor for maj7 specifically (7→maj7 promotion conservative). |
| Bars with bass_root missing (intro / rests) produce wrong roots | Low | Fallback to dominant pc. `confidence` is reduced so downstream min-confidence filter can drop. |
| Bar grid mis-detection propagates — wrong bar boundaries → wrong chord windows | Medium (existing issue) | Out of scope. If tempo_beats misfires, every detector fails; not a regression caused by this change. |
| Basic Pitch runtime spikes on noisy / long songs | Low | Wrap `predict()` in try/except; time limit at caller level (existing watchdog already catches). On failure, fall back. |
| New detector produces different chord vocabulary that breaks formatter assumptions | Low | Output uses same `ChordEvent` + same quality strings (`m7`, `maj7`, etc.) the formatter already handles. Regression will surface any surprise. |

**No model training** anywhere. Basic Pitch is shipped as-is (MIT).

## 9. What this spec does NOT touch

- `stem_chord_detector.py` — existing detector, unchanged; remains the fallback.
- `chord_detector_v10.py` — BTC + V8 hybrid, unchanged; remains the second-line fallback.
- `bass_root_extraction.py` — unchanged.
- `tempo_beats.py` — unchanged.
- `chart_formatter.py` — unchanged.
- `practice.html` / frontend — unchanged.
- `btc_chord/` model files — unchanged.
- Any trained model — we add none.

## 10. Ship order (Phase 2 implementation plan, for reference — NOT code yet)

1. Scaffolding: create `midi_chord_detector.py` with the interface from §4.1, stub implementation that returns empty progression.
2. Unit tests for the template-match function (synthetic note lists, §6.1). These pass before any real-audio invocation exists.
3. Real implementation: Basic Pitch call, per-bar note windowing, pc-weighted matching.
4. Integration shim in `transcription.py` behind the flag.
5. Pipeline reorder in `pipeline.py` (grid + bass_roots first).
6. `.env.example` flag addition.
7. Run full pytest with flag off (baseline, must stay green).
8. Run full pytest with flag on (new tests pass).
9. Single end-to-end smoke test: upload *Alright* locally with flag on, confirm bar_grid chord field contains m7s.

Commit after each of steps 2, 3, 5, 7, 9. Nothing pushed to origin until Jeff reviews.

---

**Open questions for Jeff (nothing blocks spec approval; answers inform Phase 2 defaults):**

1. Keep `ENABLE_MIDI_DETECTOR=false` default through launch, or flip to `true` after Phase 4 regression passes?
2. `bar_energy_floor=0.05` is a conservative 5%. If regression shows 7th-heavy songs missing promotions, should Phase 2 expose this as an env tunable (`MIDI_DETECTOR_ENERGY_FLOOR`), or hold at one value?
3. For the "ambiguous no-3rd" case (power chord, sus-like voicings) — prefer `maj` default, `min` default, or keep both as valid with a `5` (power chord) template added?

My defaults if no response: 1 = keep false through launch (safer), 2 = hardcode 0.05 (fewer knobs), 3 = add `5 = (0, 7)` template so power chords are recognized explicitly.
