# Separation Mode Consolidation Plan

## Date: 2026-02-27
## Author: Pipeline Agent
## Task: #17

---

## 1. Current Separation Modes Inventory

| # | Function | Line | Semaphore | Active Runner Tracking | Used From Frontend | Used From API |
|---|----------|------|-----------|----------------------|-------------------|---------------|
| 1 | `separate_stems()` | 856 | YES | YES | As fallback only | As fallback only |
| 2 | `separate_stems_roformer()` | 963 | YES | YES | YES (default) | YES (default) |
| 3 | `separate_stems_mdx()` | 1121 | **NO** | **NO** | **NO** | Yes (mdx_model=true) |
| 4 | `separate_stems_hq_vocals()` | 1266 | **NO** | **NO** | **NO** | Yes (hq_vocals=true) |
| 5 | `separate_stems_vocal_focus()` | 1370 | **NO** | **NO** | **NO** | Yes (vocal_focus=true) |
| 6 | `separate_stems_ensemble()` | 1475 | **NO** | **NO** | **NO** | Yes (ensemble=true) |

### Frontend Exposure
The frontend (`index.html`) has **ZERO** UI controls for selecting separation modes. No checkboxes, dropdowns, or toggles for hq_vocals, vocal_focus, mdx_model, or ensemble. The only way to trigger these modes is through direct API calls with the corresponding flags.

### API Exposure
All 3 API endpoints (`/api/upload`, `/api/url`, `/api/archive`) accept the mode flags as optional parameters. The `/api/capabilities` endpoint advertises `separation_modes: ['standard', 'hq_vocals', 'vocal_focus', 'mdx', 'ensemble']`.

---

## 2. Reachability Analysis

### Default Flow (99%+ of all usage)
```
process_audio()
  -> if ENHANCED_SEPARATOR_AVAILABLE:   (true -- audio-separator is installed)
       separate_stems_roformer()        <-- THE DEFAULT
     else:
       separate_stems()                 <-- FALLBACK ONLY (if audio-separator missing)
```

### Non-Default Modes (API-only, no frontend)
These are only reachable if someone passes explicit flags via the JSON API:
- `ensemble_mode=true` -> `separate_stems_ensemble()`
- `mdx_model=true` -> `separate_stems_mdx()`
- `vocal_focus=true` -> `separate_stems_vocal_focus()`
- `hq_vocals=true` -> `separate_stems_hq_vocals()`

**Priority cascade:** ensemble > mdx > vocal_focus > hq_vocals > roformer > standard

---

## 3. Dead Code Classification

### KEEP (Active, Production Code)

**`separate_stems_roformer()`** (line 963, ~157 lines)
- Default pipeline, handles 99%+ of requests
- Proper semaphore, active runner tracking, checkpoint saves
- BS-RoFormer vocals + Demucs instruments ensemble
- Falls back to `separate_stems()` on failure

**`separate_stems()`** (line 856, ~106 lines)
- Pure Demucs fallback when audio-separator unavailable or RoFormer fails
- Proper semaphore, active runner tracking, MPS->CPU fallback
- Essential safety net

### REMOVE (Dead Code)

**`separate_stems_hq_vocals()`** (line 1266, ~103 lines)
- Uses raw `subprocess.run` (not DemucsRunner) -- missing progress-as-error fix
- No semaphore -- concurrent jobs would OOM
- No active runner tracking -- not cancellable
- The approach (2-stem Demucs -> 6-stem on accompaniment) is **strictly worse** than the RoFormer approach (RoFormer gives SDR 12.9 vs Demucs SDR ~10 for vocals)
- **Verdict: REMOVE** -- superseded by `separate_stems_roformer()`

**`separate_stems_vocal_focus()`** (line 1370, ~104 lines)
- Uses raw `subprocess.run` (not DemucsRunner) -- same bugs
- No semaphore, no active runner tracking
- Uses `htdemucs_ft` with `--shifts 2` -- slightly better vocals than standard Demucs but still worse than RoFormer
- Then runs 6-stem Demucs on instrumental -- same as roformer approach
- **Verdict: REMOVE** -- superseded by `separate_stems_roformer()`

### EVALUATE (Potentially Useful but Needs Repair)

**`separate_stems_mdx()`** (line 1121, ~145 lines)
- Uses `audio_separator.separator.Separator` directly (not `EnhancedSeparator`)
- MDX23C-InstVoc_HQ for vocals, then htdemucs_6s via audio-separator for instruments
- No semaphore, no active runner tracking
- **Functionally similar to RoFormer mode** but uses MDX23C instead of BS-RoFormer
- MDX23C is a different architecture (MDX-Net) -- may be better for some material
- Has stereo guitar split option built in
- **Verdict: EVALUATE** -- could be an alternate strategy, but needs the semaphore/runner fixes. If kept, should go through EnhancedSeparator.

**`separate_stems_ensemble()`** (line 1475, ~~120 lines)
- Uses `EnsembleSeparator` (multi-model voting)
- Uses `GPUManager` for memory management
- Runs htdemucs_ft + htdemucs, votes on best per-stem
- Post-processing (bleed reduction, noise removal, phase alignment)
- No semaphore, no active runner tracking
- **Highest potential quality** but 2-3x slower
- **Verdict: EVALUATE** -- good concept, but needs operational fixes. Could be a "premium" tier.

---

## 4. Consolidation Plan

### Phase 1: Remove Dead Code (Safe, Immediate)

Delete these functions and all references:
1. `separate_stems_hq_vocals()` -- lines 1266-1368 (~103 lines)
2. `separate_stems_vocal_focus()` -- lines 1370-1472 (~104 lines)

Update `process_audio()` mode selection:
```python
# BEFORE (6 modes)
if ensemble_mode and ENSEMBLE_SEPARATOR_AVAILABLE:
    separation_success = separate_stems_ensemble(job, audio_path)
elif mdx_model and ENHANCED_SEPARATOR_AVAILABLE:
    separation_success = separate_stems_mdx(job, audio_path, stereo_split_guitar=stereo_split)
elif vocal_focus:
    separation_success = separate_stems_vocal_focus(job, audio_path)  # REMOVE
elif hq_vocals:
    separation_success = separate_stems_hq_vocals(job, audio_path)  # REMOVE
elif ENHANCED_SEPARATOR_AVAILABLE:
    separation_success = separate_stems_roformer(job, audio_path)
else:
    separation_success = separate_stems(job, audio_path)

# AFTER (4 modes, 2 actively used)
if ensemble_mode and ENSEMBLE_SEPARATOR_AVAILABLE:
    separation_success = separate_stems_ensemble(job, audio_path)
elif mdx_model and ENHANCED_SEPARATOR_AVAILABLE:
    separation_success = separate_stems_mdx(job, audio_path, stereo_split_guitar=stereo_split)
elif ENHANCED_SEPARATOR_AVAILABLE:
    separation_success = separate_stems_roformer(job, audio_path)
else:
    separation_success = separate_stems(job, audio_path)
```

Remove `hq_vocals` and `vocal_focus` parameters from:
- `process_audio()` signature (line 2534)
- `process_url()` signature (line 2760)
- Upload endpoint parsing (lines 2915-2916)
- URL endpoint parsing (lines 3011-3012)
- Archive endpoint parsing (lines 3976-3977)
- Thread creation args (lines 2939, 3037, 3998-4009)
- Capabilities endpoint (line 2815)

**Lines saved:** ~207 lines of separation code + ~30 lines of parameter threading

### Phase 2: Fix Remaining Modes (If Keeping)

If keeping `separate_stems_mdx()` and `separate_stems_ensemble()`:

1. Add `_separation_semaphore` acquire/release
2. Add `_active_runners` tracking for DemucsRunner instances
3. Add `save_job_checkpoint()` calls at stage transitions
4. Refactor `separate_stems_mdx()` to use `EnhancedSeparator` instead of raw `Separator`

### Phase 3: Strategy Pattern (Optional Refactor)

If we want to formalize this as a strategy pattern:

```python
class SeparationStrategy:
    """Base class for stem separation strategies."""

    def separate(self, job: ProcessingJob, audio_path: Path) -> bool:
        """Run separation. Returns True on success."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

class RoFormerStrategy(SeparationStrategy):
    """Default: BS-RoFormer vocals + Demucs instruments."""
    name = 'roformer'
    ...

class StandardDemucsStrategy(SeparationStrategy):
    """Fallback: Pure Demucs htdemucs_6s."""
    name = 'standard'
    ...

class MDXHybridStrategy(SeparationStrategy):
    """MDX23C vocals + Demucs instruments."""
    name = 'mdx'
    ...

class EnsembleStrategy(SeparationStrategy):
    """Multi-model voting with post-processing."""
    name = 'ensemble'
    ...
```

Common logic extracted to base class:
- Semaphore acquisition/release
- Active runner tracking
- Output directory creation
- WAV-to-MP3 conversion
- Checkpoint saves
- Error handling with fallback

**This is a bigger refactor** and should be coordinated with Task #12 (extract app.py into modular route files). Don't do both simultaneously.

---

## 5. Shared Code Analysis (Duplication)

Common patterns across all 6 separation functions:

| Pattern | Occurrences | Lines Each |
|---------|------------|-----------|
| Create output directory | 6 | 3 |
| Progress callback definition | 3 | 8 |
| Convert WAV to MP3 | 5 | 2-10 |
| Collect stems from output | 5 | 10-20 |
| Error handling with fallback | 4 | 5-10 |
| Semaphore acquire/release | 2 | 8 |
| Active runner tracking | 2 | 8 |

Total duplicate code: ~150-200 lines that could be shared.

---

## 6. Risk Assessment

| Action | Risk | Mitigation |
|--------|------|-----------|
| Remove hq_vocals mode | Low -- no frontend usage, superseded | Keep API params but log deprecation warning for 1 release |
| Remove vocal_focus mode | Low -- same as above | Same |
| Fix mdx semaphore | Low -- additive change | Test with concurrent jobs |
| Fix ensemble semaphore | Low -- additive change | Test with concurrent jobs |
| Strategy pattern refactor | Medium -- touches core pipeline | Do after Task #12 modularization |

---

## 7. Recommendation

**Immediate action:** Remove `separate_stems_hq_vocals()` and `separate_stems_vocal_focus()`. They are strictly worse than the RoFormer default, have known bugs (missing DemucsRunner, no semaphore), and are unreachable from the frontend.

**Short term:** Fix `separate_stems_mdx()` and `separate_stems_ensemble()` with semaphore + runner tracking. These have legitimate use cases as alternate strategies.

**Deferred:** Strategy pattern refactor after app.py modularization (Task #12).

---

*Analysis compiled: 2026-02-27*
