# Re-ranker Design — Claude on librosa top-K candidates

**Date:** 2026-05-11
**Motivates:** `docs/detector-quality-comprehensive-audit-2026-05-11.md` (§1, recommended architecture). Replaces the generator contract in `backend/processing/chord_corrector_anthropic.py:148-200`.
**Predicted lift:** +0.04 to +0.10 mean F1 over current `full`-mode corrector (audit baseline F1=0.80).

---

## 1. Data flow diagram

```
Current (generator):
  audio -> librosa detector (argmax per beat)            chord_detector_librosa.py:118-127
        -> chord_events                                  chord_detector_librosa.py:133-141
        -> bar_grid build (downstream)
        -> chord_chart {title, artist, chords_used, bar_grid}
        -> apply_correction()                            chord_corrector_anthropic.py:531
             -> _query_canonical_chords(title, artist)   chord_corrector_anthropic.py:148
                  [Claude sees ONLY title + artist; invents chord_set]
             -> drop / replace / full mutates bar_grid

New (re-ranker):
  audio -> librosa detector
        -> per-beat top-K (chord_str, score)             NEW: chord_detector_librosa.py:118-127
        -> chord_events[i].candidates: List[Candidate]   NEW: dataclass field
        -> bar_grid build (downstream preserves candidates per bar)
        -> chord_chart {..., bar_grid[i].candidates}
        -> apply_correction_reranker()                   NEW
             -> _query_reranker(title, artist, key, tempo, bar_candidates, summary)
                  [Claude sees librosa evidence; CHOOSES from candidates or "abstain"]
             -> bar_grid[i].chord <- chosen | (abstain ? argmax : ...)
             -> chords_used recomputed from final bar_grid
```

**Key change:** Claude's contract narrows from "invent a canonical chord_set" to "vote per bar from this evidence." The wrong-key failure mode (`docs/detector-signal-research-2026-05-10.md` §"Why this works") becomes structurally impossible when no librosa bar surfaces the wrong-key chord as a candidate.

---

## 2. Top-K extraction from librosa

**Current argmax site:** `chord_detector_librosa.py:122-127`

```
scores = _TEMPLATES @ v_norm   # (24,) cosine-ish, line 123
best = int(np.argmax(scores))  # line 124
```

`_TEMPLATES` (line 67) is 24 rows: 12 major + 12 minor triads. The scores vector is already a full ranking — we throw away 23 of 24 ranks today.

**Change:** after line 123, also compute `top_idx = np.argsort(-scores)[:K]` with K=4. Build `candidates: List[{chord_name, score}]` from those indices via `_label_to_chord_str` (line 70-78). Drop candidates whose score < 0.55 (below template-match floor — these are noise on silence/percussion-only beats).

**Plumb-through:**

- Add `candidates: List[Dict]` to `ChordEvent` (`chord_detector_librosa.py:28-37`).
- Add to the dict emitted at `chord_detector_librosa.py:183-193` so `job.chord_progression[i]["candidates"]` carries through.
- Downstream bar-grid builder (in `chart_formatter` per `chord_detector_librosa.py:150` comment) must propagate `candidates` from chord events onto each bar. When multiple beats share a bar, aggregate by summing per-chord scores and re-sort. Re-clip to K=4.

**Candidate shape:**
```
{"chord": "Am", "score": 0.91, "root": "A", "quality": "min"}
```

K=4 is chosen because:
- The current detector only emits maj/min over 12 roots = 24 chords total. Anything beyond K=4 dredges noise.
- Token budget §5 stays cheap at K=4.

**Caveat:** the librosa template bank is maj+min only (`chord_detector_librosa.py:50-52`). The re-ranker cannot recover dom7, maj7, sus, dim chords because they are **not in any template's top-K**. Aja's `Bm7` would surface as `Bm`, not `Bm7`. Extension recovery requires expanding `_build_templates` (`chord_detector_librosa.py:55-64`) — out of scope here, flagged in §7.

---

## 3. Claude re-ranker prompt

System prompt (cached via `cache_control: ephemeral`, same pattern as `chord_corrector_anthropic.py:170-174`):

```
You are a chord-chart re-ranker. For each bar of a song, you receive 3-5
chord candidates from a librosa template detector, each with a cosine
similarity score in [0,1]. Pick the single best chord per bar.

Hard rules:
  - You MUST pick one of the provided candidates OR output "abstain".
  - "abstain" means "candidates look noisy, keep librosa's top pick" —
    use it for bars where the top score is < 0.65 or all candidates look
    musically implausible given the song key.
  - DO NOT invent chord names not in the candidate list.
  - DO NOT add extensions (7, maj7, sus, etc.) — the candidate vocabulary
    is maj/min triads only. Note them in `notes` if a chord SHOULD be a 7
    chord, but pick the triad in the candidates.
  - Prefer chords diatonic to the provided key and consistent with the
    song's most-frequent chord set.
  - When two candidates are within 0.05 score of each other, prefer the
    one that maintains chord continuity with the previous bar (avoid
    one-bar outliers between two identical neighbors).

Output a JSON object, no fences:
{
  "found": true | false,           // false if you can't identify the song / candidates unusable
  "bars": [
    {"bar": 1, "pick": "Am", "abstain": false},
    {"bar": 2, "pick": null, "abstain": true},
    ...
  ],
  "notes": "one-line summary"
}

Every bar in the input must appear exactly once in "bars".
```

User message:

```
Song: "{title}" by {artist}
Librosa key: {key}
Librosa tempo: {tempo} BPM
Top chords across the whole song (by total time): {chords_used_with_pct}

Bars (top-K candidates per bar, score in 0..1):
[
  {"bar": 1, "candidates": [{"c":"Am","s":0.91},{"c":"C","s":0.74},{"c":"E","s":0.62},{"c":"F","s":0.55}]},
  {"bar": 2, "candidates": [{"c":"C","s":0.88},{"c":"Am","s":0.71},{"c":"G","s":0.66},{"c":"Em","s":0.59}]},
  ...
]

Return ONLY the JSON object.
```

Output schema enforced post-parse: `bars[i].bar` is a permutation of the input bar numbers; `pick` is null iff `abstain` is true; non-null `pick` is in that bar's candidate list. Fail closed on any violation (fallback to argmax, log `status: "reranker_validation_failed"`).

---

## 4. Integration points

**New function:** `apply_correction_reranker(chord_chart, *, model=None) -> dict` in `chord_corrector_anthropic.py`, alongside the existing `apply_correction` at `chord_corrector_anthropic.py:531`.

**Dispatcher:** `apply_correction` (line 531) reads `ANTHROPIC_CORRECTION_STRATEGY in {"generator", "reranker"}` (default `generator`) and delegates. This preserves the public surface — callers (Flask route) don't change.

**Stays:**
- `_api_key` (line 129), `normalize_chord` import, `_PITCH_CLASS` (line 62), `_root_pc` (line 65), `_is_minor` (line 79). All reusable for cross-strategy gating later.
- `_scrub_bar_grid` (line 203) and `_replace_in_bar_grid` (line 351). Not used by re-ranker (no chord-set delta) but no reason to delete.
- The full-mode format pass (`_query_format_correction` line 275, `_apply_format_corrections` line 428). Orthogonal — section relabeling is independent of chord choice and can layer on top of re-ranker output. If `ANTHROPIC_CORRECTION_MODE=full`, run the format pass after the re-ranker pass.

**New:**
- `_query_reranker(title, artist, key, tempo, bar_candidates, chord_set_summary, model) -> Optional[dict]`. Same pattern as `_query_canonical_chords` (line 148): API key check, model call with cached system prompt, tolerant JSON parse.
- `_apply_reranker_picks(chord_chart, picks) -> int`. Walks `bar_grid`, sets `bar["chord"] = pick.pick` when `abstain=false`. Records `source_meta.replaced_from` and `source_meta.reason = "reranker-rerank"`. Returns count rewritten.
- Recompute `chords_used` from the final bar_grid (de-dup ordered).

**Env vars:**
- `ANTHROPIC_CORRECTION_STRATEGY` — `generator` (default) | `reranker`.
- `ANTHROPIC_CORRECTION_RERANKER_K` — top-K size, default 4.
- `ANTHROPIC_CORRECTION_RERANKER_MIN_SCORE` — abstain floor, default 0.55.

The qflip gate (`chord_corrector_anthropic.py:652-671`) does NOT apply to re-ranker output — qflip detects whole-key transposition, which re-ranker can't produce. Keep the gate in the generator code path only.

---

## 5. Token + cost estimate

Typical song: 70-130 bars (audit samples: Day After Day 78, House of Rising Sun 131). K=4 candidates per bar.

**Input:** ~30 tokens per bar (compact JSON: `{"bar":N,"candidates":[{"c":"Am","s":0.91},...]}`) × 100 bars = ~3,000 tokens, plus ~200 tokens of header + ~400 tokens system prompt = **~3,600 input tokens**.

**Output:** ~10 tokens per bar (`{"bar":N,"pick":"Am","abstain":false}`) × 100 = ~1,000 tokens, plus framing = **~1,200 output tokens**.

Sonnet 4.5 pricing: $3 / Mtok input, $15 / Mtok output. With ephemeral cache hit on system prompt (cache reads $0.30 / Mtok per `chord_corrector_anthropic.py:170-174` precedent):
- First call per song: ~$0.0108 (3.6k × $3/M) + $0.018 (1.2k × $15/M) = **~$0.029 / song**
- Cache-hit calls: $0.001 cached + ~$0.0096 fresh input + $0.018 output = **~$0.028 / song**

Compare to current generator: ~$0.01-$0.02 per `chord_corrector_anthropic.py:30`. **Re-ranker is ~$0.01 more per song, ~$10/month at 1000-song scale.** Cheap.

---

## 6. Eval harness changes

`/tmp/audit_qflip_run.py` currently bakes one strategy into the prod request path. Two changes:

1. **Per-request strategy override.** The upload call (`audit_qflip_run.py:30-38`) sets form fields but the strategy is read from the prod server's env. Either:
   - (a) Add `ANTHROPIC_CORRECTION_STRATEGY` form override accepted only when `X-Audit-Token` is present (~10 lines in the Flask route reading the form post `chord_detection`).
   - (b) Run two separate audit passes — flip the env var on the VPS between runs. Slower but no code change.
2. **Two output dirs + comparison.** Add `--strategy {generator,reranker}` CLI flag to `audit_qflip_run.py:61`, mkdir `/tmp/audit-may8-results.{strategy}/`. Run both. New summary step: paired diff per song (mean delta F1, per-song delta, win/loss/tie count, sign test p-value). Re-use `backend.audit.llm_oracle` (`audit_qflip_run.py:100-103`) unchanged.

Sample size note: per `docs/detector-quality-comprehensive-audit-2026-05-11.md` §2, per-song noise std is 0.135 — at n=18 the SEM is 0.032, so detecting a +0.04 mean lift is marginal. Either run with temperature=0 (Agent C's recommendation) or expand n to 30+ before treating any A/B result as conclusive.

---

## 7. Risks and failure modes

1. **Right chord not in top-K.** When librosa's template bank fundamentally can't produce a chord (any 7, sus, dim — only 24 maj/min exist per `chord_detector_librosa.py:50-52`), the re-ranker cannot recover it. Aja's `Bm7` would never surface. **Mitigation:** add `Bm7` etc. to `_build_templates` (line 55) post-hoc, OR keep generator path for songs Claude is highly confident need extensions (signal: title contains jazz cues, or Claude's free-form `notes` requests escalation). **+0.04-0.10 prediction assumes vocab parity** — if reranker is stuck at triads-only on Aja-class songs, the lift caps at ~+0.04.

2. **Claude abstains everywhere.** Fallback is librosa argmax (current ground truth). Net result: re-ranker is a no-op, equivalent to disabling the corrector. F1 floor = "corrector always off" = 0.611 per `docs/detector-signal-research-2026-05-10.md` Gate D table. Floor is real but not catastrophic — current corrector also no-ops on `skipped_quality_flip` (line 660).

3. **No-chord bars (silence/intro).** Today's bar_grid emits a chord per bar regardless; silence is glossed over (`chord_detector_librosa.py:120-121` skips silence at the chord-event level but the bar-grid builder downstream fills with previous chord). The re-ranker should accept `"pick": null, "abstain": true` and the apply step writes `""` (empty chord) when all candidate scores < `RERANKER_MIN_SCORE`. UI already handles empty chord strings (chord-held rendering).

4. **Family-aware regression risk.** Apr 25's family-aware fixes live upstream of this corrector (in `stem_chord_detector.py` per `MEMORY.md`). The librosa detector at `chord_detector_librosa.py` is the alternate path that **doesn't have family-aware logic** — it's pure template argmax. The re-ranker operates on librosa output, so it inherits whichever detector ran. If the librosa path produces same-root family-confused candidates (`Am` and `A` both in top-K), Claude could flip family. **Mitigation:** in the system prompt, add: "Within a bar's candidates, if `Xm` and `X` (same root, different quality) both appear, prefer the one diatonic to the declared key — but if neither is, prefer the higher-score one." This is exactly the qflip-gate intuition, applied per-bar.

5. **Where +0.04-0.10 might not materialize:**
   - Songs whose argmax is already correct gain nothing (top-1 = top-K[0]).
   - The audit's worst losses (Sister Golden Hair, Misunderstood) are wrong-key transpositions — re-ranker cannot transpose, so the qflip-gate already catches these (lift to 0.823 per `detector-signal-research-2026-05-10.md`). Re-ranker's lift comes from the *Hells Bells class* (`detector-signal-research-2026-05-10.md` open question #1), where qflip is blind. If only Hells Bells and 1-2 similar songs benefit, the audit-set lift is ~+0.02, not +0.04.
   - Variance noise (per-song std 0.135) will swamp small lifts. Determinism (temperature=0) must land first.

---

## 8. Concrete rollout plan

| Step | Files | Time | Parallel? |
|---|---|---|---|
| 1. Top-K extraction in detector + plumb candidates through | `chord_detector_librosa.py:118-141, 183-193`; bar-grid builder in chart_formatter | 4h | No — gates everything else |
| 2. Implement `_query_reranker`, `_apply_reranker_picks`, `apply_correction_reranker` | `chord_corrector_anthropic.py` (new ~150 LOC) | 6h | Parallel with step 3 once step 1's data shape is fixed |
| 3. Env-var dispatcher in `apply_correction` (line 531); update CLAUDE.md / module docstring | `chord_corrector_anthropic.py:1-31, 531-549` | 1h | Parallel with step 2 |
| 4. Audit harness `--strategy` flag + paired summary | `/tmp/audit_qflip_run.py:61, 100-126` | 2h | Parallel with steps 2-3 |
| 5. Deploy behind flag, default OFF in prod (`ANTHROPIC_CORRECTION_STRATEGY=generator`) | VPS `/opt/stemscribe/backend/.env`, restart systemd | 0.5h | After 1-4 |
| 6. Run audit with `ANTHROPIC_CORRECTION_STRATEGY=reranker` at temperature=0 | n/a, just env override + run | 2h (mostly wait) | After step 5 |
| 7. Compare paired results; decide ship/iterate | analysis only | 1h | After step 6 |

Total: ~16.5h serial, ~10h with parallelization. Sits inside the audit's stated "16-40h" estimate for this architecture (`detector-quality-comprehensive-audit-2026-05-11.md` §1).

Default-OFF deploy in step 5 is critical — re-ranker is unproven against prod traffic, generator is the known-good fallback. Flip the flag only after step 7 ship-decision.

wrote to /Users/jeffkozelski/stemscribe/docs/reranker-design-2026-05-11.md
