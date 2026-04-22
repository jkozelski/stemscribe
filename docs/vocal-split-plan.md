# Lead/Backing Vocal Separation — Implementation Plan

**Date:** 2026-04-04
**Status:** Prototype complete, NOT deployed
**Files changed:** `backend/modal_separator.py`

---

## Summary

Added a second separation pass to the Modal GPU function that splits the vocals stem into lead and backing vocals using `UVR_MDXNET_KARA_2.onnx`. The output changes from 6 stems to 8 stems while maintaining full backward compatibility.

## Approach: Two-Pass Separation

1. **Pass 1 (existing):** BS-RoFormer-SW extracts 6 stems: vocals, guitar, bass, drums, piano, other
2. **Pass 2 (new):** UVR-MDX-NET Karaoke 2 takes the vocals WAV and splits it into lead vocals + backing vocals

This is the community-standard approach. Running KARA on pre-extracted vocals produces much better results than running on the full mix.

## Model Details

| Property | Value |
|----------|-------|
| Model file | `UVR_MDXNET_KARA_2.onnx` |
| Architecture | MDX-NET |
| Size | ~150 MB |
| Auto-download | Yes (audio-separator handles this) |
| Cached at | `/model-cache/models/` (Modal Volume) |
| Additional processing time | ~30-60 seconds |
| Additional VRAM | Minimal (~2-4 GB, well within A10G 24GB budget) |

## Output Schema

### With `split_vocals=True` (new default) — 8 stems:

| Key | Description |
|-----|-------------|
| `vocals` | Original mixed vocals (kept for backward compat) |
| `vocals_lead` | Lead vocals only (from KARA_2 primary output) |
| `vocals_backing` | Backing/harmony vocals (from KARA_2 secondary output) |
| `guitar` | Guitar |
| `bass` | Bass |
| `drums` | Drums |
| `piano` | Piano |
| `other` | Everything else |

### With `split_vocals=False` — 6 stems (original behavior):

Same as before: vocals, guitar, bass, drums, piano, other.

## API Changes

```python
# New parameter added (backward compatible — defaults to True)
separate_stems_gpu(audio_bytes, filename="input.mp3", split_vocals=True)
```

The `split_vocals` parameter defaults to `True`. Existing callers that don't pass it will automatically get the vocal split. The original `vocals` key is always present regardless, so no downstream code breaks.

## How KARA_2 Output Mapping Works

The KARA_2 model is a "Vocals vs Instrumental" MDX model. When fed pre-extracted vocals:

- **Primary output** (filename contains "Vocals"): Lead vocals — the main vocal line
- **Secondary output** (filename contains "Instrumental"): Backing vocals — harmonies, ad-libs, backing tracks

Detection logic:
1. Check filename for "instrumental" -> backing vocals
2. Check filename for "vocals" -> lead vocals
3. Fallback by position: first file = lead, second file = backing

## Failure Handling

The vocal split is wrapped in a try/except. If KARA_2 fails for any reason:
- The original `vocals` stem is still present
- `vocals_lead` and `vocals_backing` keys are simply absent
- A warning is logged but separation does not fail
- No impact on the 6 base stems

## Test Entrypoint

```bash
# Full 8-stem test (default)
modal run modal_separator.py --file /path/to/song.mp3

# 6-stem only (skip vocal split)
modal run modal_separator.py --file /path/to/song.mp3 --no-vocal-split
```

The test entrypoint now validates:
- All 6 base stems present
- `vocals_lead` and `vocals_backing` present (when split enabled)
- Reports PASS/WARN with expected stem count

## Downstream Integration Needed (not yet done)

These changes are needed before the vocal split is useful in the UI:

1. **`backend/processing/separation.py`** (`separate_stems_modal`): Currently saves all stems from the dict. No change needed — `vocals_lead.mp3` and `vocals_backing.mp3` will be saved automatically alongside the other stems.

2. **Frontend mixer UI** (`frontend/index.html`): Add track lanes for vocals_lead and vocals_backing. Consider a toggle to show combined vocals vs split view.

3. **Practice mode** (`frontend/practice.html`): Lead-only or backing-only playback for vocal practice.

4. **Supabase/R2 storage**: The storage layer saves whatever stems exist in the job dict, so no schema change needed.

## Cost Impact

- KARA_2 adds ~30-60s of GPU time per song
- At Modal A10G pricing (~$0.000463/s), this adds ~$0.01-0.03 per song
- Total cost goes from ~$0.06 to ~$0.07-0.09 per song
- Negligible impact on the budget

## Deployment Checklist

- [ ] Review this plan
- [ ] Deploy to Modal: `cd ~/stemscribe/backend && ../venv311/bin/python -m modal deploy modal_separator.py`
- [ ] Test with a real song: `modal run modal_separator.py --file /path/to/song.mp3`
- [ ] Verify 8 stems in `~/stemscribe/outputs/modal_test/`
- [ ] Listen to vocals_lead.mp3 and vocals_backing.mp3 for quality
- [ ] Update frontend mixer to show lead/backing tracks
- [ ] Update practice mode for vocal-specific playback

---

*Research sources: [python-audio-separator GitHub](https://github.com/nomadkaraoke/python-audio-separator), [UVR discussion on lead/backing separation](https://github.com/Anjok07/ultimatevocalremovergui/discussions/1250), [audio-separator model recommendations](https://github.com/nomadkaraoke/python-audio-separator/discussions/133)*
