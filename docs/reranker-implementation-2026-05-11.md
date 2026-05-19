# Re-ranker Implementation — Day 1

**Date:** 2026-05-11
**Spec:** `docs/reranker-design-2026-05-11.md`
**Status:** Local prototype + tests. NOT deployed.

---

## 1. Files changed

| File | Lines | Change |
|---|---|---|
| `backend/processing/chord_detector_librosa.py` | 17-43, 79-89 (config), 117-156 (loop), 197-201 (dict) | Added `candidates: List[Dict]` to `ChordEvent`, top-K extraction post-argmax, plumb-through to `job.chord_progression` |
| `backend/chart_formatter.py` | 701-770 inside `_quantize_chords_to_bars` | Aggregate per-beat `candidates` into per-bar `source_meta.top_k`, weighted by overlap duration, re-sorted and clipped to K |
| `backend/processing/chord_corrector_anthropic.py` | New block ~530-870 (before existing `apply_correction`) | New `apply_correction_reranker()` + helpers + strategy dispatch inside `apply_correction()` |
| `backend/tests/test_chord_corrector_reranker.py` | New file, 15 tests | Tests for env routing, valid response handling, abstain, no-top-K fallback, malformed JSON, validation rejection, exception fall-through, and the chart_formatter aggregation step |
| `/tmp/audit_reranker_run.py` | New file | Audit driver with `--strategy {generator,reranker}` flag |

No edits to `apply_correction` signature. No edits to existing tests. The
generator-strategy code path is byte-identical to before once the new
dispatcher detects `strategy != "reranker"`.

---

## 2. How to enable

Three env vars, all read at request time (no restart needed):

```bash
ENABLE_ANTHROPIC_CORRECTION=true            # required gate (unchanged)
ANTHROPIC_CORRECTION_STRATEGY=reranker      # NEW. defaults to "generator"
ANTHROPIC_CORRECTION_RERANKER_K=4           # NEW. defaults to 4
ANTHROPIC_CORRECTION_RERANKER_MIN_SCORE=0.55  # NEW. defaults to 0.55
ANTHROPIC_CORRECTION_MODEL=claude-sonnet-4-5-20250929  # unchanged
```

`ANTHROPIC_CORRECTION_STRATEGY` is the on/off switch. Unset or `generator`
gives the current production behavior. `reranker` enables the new path.

API key sourcing is unchanged: `ANTHROPIC_API_KEY` env first, macOS keychain
`anthropic-api-key` second. Both paths use `_api_key()`.

---

## 3. How to test locally

```bash
cd ~/stemscribe
./venv311/bin/python -m pytest backend/tests/test_chord_corrector_reranker.py -v
```

Or run the whole suite:

```bash
./venv311/bin/python -m pytest backend/tests/
```

Baseline before this change: 493 tests.
After this change: 508 tests (493 + 15 new). Zero regressions.

---

## 4. Test results

```
============================= test session starts ==============================
collected 15 items

backend/tests/test_chord_corrector_reranker.py::test_apply_correction_disabled_returns_input_unchanged PASSED
backend/tests/test_chord_corrector_reranker.py::test_apply_correction_default_strategy_is_generator PASSED
backend/tests/test_chord_corrector_reranker.py::test_apply_correction_strategy_reranker_routes_to_reranker PASSED
backend/tests/test_chord_corrector_reranker.py::test_strategy_dispatcher_case_insensitive PASSED
backend/tests/test_chord_corrector_reranker.py::test_reranker_applies_picks_to_bar_grid PASSED
backend/tests/test_chord_corrector_reranker.py::test_reranker_respects_abstain PASSED
backend/tests/test_chord_corrector_reranker.py::test_reranker_handles_found_false PASSED
backend/tests/test_chord_corrector_reranker.py::test_reranker_no_top_k_returns_unchanged PASSED
backend/tests/test_chord_corrector_reranker.py::test_reranker_no_api_key_returns_unchanged PASSED
backend/tests/test_chord_corrector_reranker.py::test_reranker_malformed_json_returns_unchanged PASSED
backend/tests/test_chord_corrector_reranker.py::test_reranker_invalid_pick_rejects_response PASSED
backend/tests/test_chord_corrector_reranker.py::test_reranker_missing_bar_rejects_response PASSED
backend/tests/test_chord_corrector_reranker.py::test_apply_correction_reranker_exception_falls_through PASSED
backend/tests/test_chord_corrector_reranker.py::test_quantize_chords_to_bars_aggregates_top_k PASSED
backend/tests/test_chord_corrector_reranker.py::test_quantize_chords_to_bars_no_candidates_no_source_meta PASSED

============================== 15 passed in 0.42s ==============================
```

Full suite:

```
======================== 508 passed, 1 warning in 5.28s ========================
```

---

## 5. The actual Anthropic prompt being sent

### System prompt (cached via `cache_control: ephemeral`)

```
You are a chord-chart re-ranker. For each bar of a song, you receive 3-5 chord candidates from a librosa template detector, each with a cosine similarity score in [0,1]. Pick the single best chord per bar.

Hard rules:
  - You MUST pick one of the provided candidates OR output "abstain".
  - "abstain" means "candidates look noisy, keep librosa's top pick" — use it for bars where the top score is < 0.65 or all candidates look musically implausible given the song key.
  - DO NOT invent chord names not in the candidate list.
  - DO NOT add extensions (7, maj7, sus, etc.) — the candidate vocabulary is maj/min triads only. Note them in `notes` if a chord SHOULD be a 7 chord, but pick the triad in the candidates.
  - Prefer chords diatonic to the provided key and consistent with the song's most-frequent chord set.
  - When two candidates are within 0.05 score of each other, prefer the one that maintains chord continuity with the previous bar (avoid one-bar outliers between two identical neighbors).
  - Within a bar's candidates, if `Xm` and `X` (same root, different quality) both appear, prefer the one diatonic to the declared key — but if neither is, prefer the higher-score one.

Output a JSON object, no fences:
{
  "found": true | false,
  "bars": [
    {"bar": 1, "pick": "Am", "abstain": false},
    {"bar": 2, "pick": null, "abstain": true},
    ...
  ],
  "notes": "one-line summary"
}

Every bar in the input must appear exactly once in "bars". `pick` must be null when `abstain` is true and non-null otherwise.
```

I added the family-aware rule from §7.4 of the design spec (the
"`Xm` and `X` same root" line) directly into the system prompt — that
mitigation was flagged as recommended in the spec and costs nothing.

### User message template

```
Song: "<title>" by <artist>
Librosa key: <key or 'unknown'>
Librosa tempo: <tempo:.0f> BPM
Top chords across the whole song (by bar weight): <chord:pct list, top 8>

Bars (top-K candidates per bar, score in 0..1):
[{"bar":1,"candidates":[{"c":"Am","s":0.91},{"c":"C","s":0.74},...]},...]

Return ONLY the JSON object.
```

Compact JSON (no spaces, `c`/`s` keys) keeps input tokens low. Per
spec §5, expected ~3,600 input + ~1,200 output tokens at 100 bars and
K=4, or ~$0.03/song uncached and ~$0.028/song cached.

The call uses `temperature=0` to make per-song output deterministic
(spec §6 recommendation; reduces audit variance noise).

---

## 6. Validation contract

`_validate_reranker_response` enforces:

1. `bars` is a list.
2. Every bar entry has an integer `bar` field that appears in the input
   bars set, and appears exactly once.
3. The set of `bar` numbers in the response equals the set in the input
   (no missing, no extra).
4. `abstain=true` → `pick` must be `null`.
5. `abstain=false` → `pick` must be a non-empty string that is in that
   bar's candidate list (no inventing chords not surfaced by librosa).

If ANY of these fail, the whole response is rejected and `bar_grid` is
left untouched — `anthropic_correction.status = "reranker_validation_failed"`.
This is "fail closed" per spec §3.

---

## 7. Failure handling matrix

| Condition | Result | `anthropic_correction.status` |
|---|---|---|
| Env var unset → strategy=generator | Falls to existing generator code path | (whatever generator wrote) |
| `bar_grid` lacks `source_meta.top_k` on every bar | No-op, no Claude call | `skipped_no_top_k` |
| No API key (env + keychain miss) | No-op | `skipped_no_response` |
| `anthropic` SDK missing | No-op | `skipped_no_response` |
| Network error / API exception | No-op | `skipped_no_response` |
| Claude returns malformed JSON | No-op | `skipped_no_response` |
| Claude `found=false` | No-op, claude_notes preserved | `skipped_unrecognized` |
| Response fails validation (wrong bars, invalid pick, etc.) | No-op | `reranker_validation_failed` |
| Reranker function raises any exception | Falls through to generator strategy | (whatever generator wrote, with a warning log) |
| Valid response, all abstain | bar_grid unchanged, counters recorded | `applied` |
| Valid response, some/all rewritten | bar_grid mutated, chords_used recomputed | `applied` |

Pipeline-level safety: `processing/pipeline.py:740-744` already wraps
`apply_correction` in `try/except`, so even a worst-case crash bubbles up
as a non-fatal warning.

---

## 8. Known limitations / TODOs

1. **Triads-only ceiling.** Per spec §7.1, the librosa template bank only
   produces maj/min over 12 roots. Anything dom7 / maj7 / sus / dim cannot
   be in `top_k` and therefore cannot be recovered by the re-ranker. Aja's
   `Bm7` becomes `Bm`. This caps the predicted +0.04 to +0.10 lift at the
   low end. Expanding `_build_templates` is out of scope here but is the
   next move if re-ranker beats generator at the triad-only level.

2. **bass-aware detector path is not wired.** The librosa detector goes
   through `chart_formatter._quantize_chords_to_bars` (where I added
   top-K aggregation). The bass-root path uses
   `bass_root_extraction.combine_with_detector_quality`, which I left
   unchanged — that detector doesn't produce candidates yet, so top-K
   wouldn't be meaningful. Result: when the bass-aware detector runs,
   `apply_correction_reranker` will hit `skipped_no_top_k` and the chart
   passes through unchanged. This is intentional — re-ranker is only
   meaningful for the librosa-detector path.

3. **`apply_correction_reranker` doesn't honor `ANTHROPIC_CORRECTION_MODE`.**
   The mode flag (`drop`/`replace`/`full`) is a generator-strategy concept.
   The reranker has a single mode (per-bar pick). The `full`-mode format
   pass (section relabeling) is not invoked from the reranker path either,
   per spec §4 ("Orthogonal — section relabeling is independent of chord
   choice and can layer on top of re-ranker output"). I haven't wired
   that layering; if you want it, it's a 5-line addition right after the
   reranker `_apply_reranker_picks` call. Flagging here so the omission is
   intentional and easy to flip.

4. **Audit driver assumes prod accepts per-request strategy override.**
   `/tmp/audit_reranker_run.py` posts `anthropic_correction_strategy` as a
   form field on `/api/upload`. The prod Flask route doesn't read this
   field yet. Per design spec §6, the alternatives are:
   - (a) add ~10 lines to `routes/api.py` to read the form field when
     `X-Audit-Token` is present and export it into the job's env, OR
   - (b) flip `ANTHROPIC_CORRECTION_STRATEGY` on the VPS env file between
     audit runs (slower, but no code change in prod).
   I went with (a) on the script side so it's ready, but the route hook
   needs adding before that path is live. Until then, use (b).

5. **Token budget not enforced.** The current implementation sends ALL
   bars in one call. For a 200-bar song that's ~6K input tokens — still
   well inside the $0.10 ceiling. If we encounter a 500-bar outlier, we
   may want to batch (e.g. 100 bars per call, then stitch). Not built;
   add if anyone hits the ceiling.

---

## 9. Deviation from spec

Where I deviated from `docs/reranker-design-2026-05-11.md`:

1. **`temperature=0` in the Claude call.** Spec §6 says this is needed for
   the audit ("Determinism (temperature=0) must land first"), but doesn't
   explicitly bake it into the implementation. I bolted it on at the API
   call site. If you want this configurable, add an env var.

2. **Family-aware rule in the system prompt.** Spec §7.4 calls it out as
   a recommended mitigation but doesn't pre-write the prompt line. I
   pulled it into the system prompt verbatim from the spec.

3. **`anthropic_correction` meta uses `strategy: "reranker"` key.** The
   generator path doesn't write a `strategy` key (just `mode`). I added
   `strategy` to the reranker path's meta so analytics / debugging can
   tell them apart at a glance.

4. **`_recompute_chords_used` is ordered, deduped, derived from final
   bar_grid.** Spec §4 says "Recompute `chords_used` from the final
   bar_grid (de-dup ordered)" — that's exactly what I did. No deviation,
   just confirming intent.

5. **`_validate_reranker_response` returns `None` on ANY violation, not a
   partial-apply.** Spec §3 says "Fail closed on any violation (fallback
   to argmax, log `status: 'reranker_validation_failed'`)." Implemented
   exactly that — but worth flagging that partial-apply would also be
   defensible (apply the valid picks, skip the invalid ones). Picked
   fail-closed for safety; happy to relax if the audit reveals Claude
   often returns one bad bar in a sea of good ones.

---

## 10. Next steps (for Jeff)

1. Read this doc + the prompt (§5) and approve the contract.
2. If approved, deploy: scp the 3 modified files + the new test file to
   the VPS, restart `stemscribe.service`. Verify with `pytest` on the VPS.
3. Set `ANTHROPIC_CORRECTION_STRATEGY=reranker` in
   `/opt/stemscribe/backend/.env`, restart.
4. Run `/tmp/audit_reranker_run.py --strategy reranker` and
   `/tmp/audit_reranker_run.py --strategy generator` (need the prod
   route hook from §8.4 first, or use VPS env flip).
5. Compare paired F1 deltas. Ship/iterate per spec §8 step 7.
