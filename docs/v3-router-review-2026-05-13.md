# V3.1 Detector Router — Review & Hardening Report

**Date:** 2026-05-13
**Module:** `backend/processing/detector_router.py`
**Scope:** Pre-prod review of the Claude-as-traffic-cop router prototyped this morning. Decides per-upload routing to either the legacy stem-aware "jazz" detector (Aja-style extension-heavy material) or the V3.1 "general" detector (ACE + Jiang per-bar router).
**Reviewers:** 4 parallel agents — A (empirical breadth, 100 songs), B (code review), C (cost/SLA/latency), D (architecture critique).

---

## Verdict: **SHIP** — with the P0/P1 patches already landed and three follow-up items for Week 1.

The Claude-title-+-artist routing approach is well-fit for the actual problem. Empirically it routes at ~96% accuracy on a 100-song breadth test (Agent A); none of the painful failure modes (jazz standard → general) occurred. Cost is ~$4/month at the 50K-songs/month projection (Agent C). The original prototype shipped with two genuine P0 bugs and a P1 cluster around concurrency, Unicode, and cache invalidation (Agent B). All P0/P1 fixes have been merged into `backend/processing/detector_router.py` as part of this review. Architecture is correct for V3.1; deferred levers (audio features, more categories, Sonnet tier, per-bar rerank) are correctly out of scope for the June 20 launch window (Agent D).

The patched router went through one additional pass on top of Agent B's draft: Agent C's outage allowlist was integrated so Steely-Dan-class jazz uploads during an Anthropic outage still get the jazz path instead of silently degrading to general.

---

## 1. Empirical results (Agent A)

Full table + per-genre breakdown: `/tmp/router-review/A-breadth-test.md`. Raw data: `/tmp/router-review/results.json`. Production cache (`backend/data/detector_router_cache.json`) now has **124 entries** pre-warmed from the breadth run — free seed for prod.

### Headline numbers

| Metric | Value |
|---|---:|
| Songs tested | 100 |
| API calls (real) | 99 + 1 short-circuit fallback |
| Total spend | $0.297 (under $0.30 cap) |
| Mean wall-clock | 1.2 s / call |
| Cache verified | ✅ duplicate "Anti-Hero" call returned `source=cache` |

### Routing distribution

84 → `general`, 16 → `jazz`. The 16 jazz routings:
- **10 / 10 jazz standards** routed correctly (Misty, Autumn Leaves, Giant Steps, etc.) at confidence 0.98–0.99
- **5 / 9 fusion songs** routed jazz (Snarky Puppy "Lingus", BBNG "Lavender", Glasper, Thundercat × 2)
- **1 R&B song** (Stevie Wonder "Sir Duke") routed jazz — soft miss

### Confidence histogram

Median 0.96, mode 0.98 (29 of 100). 92 / 100 calls ≥ 0.85 confident. Only 8 land in 0.70–0.85 — and those are exactly the borderline cases (fusion, neo-soul) where you'd want low confidence.

### Correctness assessment

**~96% ± 3pp.**

- 1 clear miss: "Sir Duke" by Stevie Wonder → jazz (per rule 3, not a standard — should be general)
- 3 soft concerns: Thundercat × 2 → jazz, Snarky Puppy "What About Me?" inconsistent within the Snarky cohort
- **No jazz standards were misrouted to general** — the more painful failure mode (jazz song silently downgraded to ACE+Jiang) did not occur in this 100-song sample.

### Edge-case wins

- Typo "Hotle California" → general @ 0.95 (Claude recognized through the misspelling)
- Café Tacvba accent handled cleanly (after the P1 NFKD normalization fix)
- Dylan "Watchtower" + Hendrix "Watchtower" both → general (correct per rule 2)
- Empty / empty → fallback, no API call (cost-safe)

### Borderline behavior

Haiku is slightly aggressive on instrumental fusion (Snarky/Thundercat/Glasper) but confidence drops to 0.72–0.96 vs 0.99 for standards. The signal is there if a future iteration wants to threshold-gate the jazz path with a higher confidence floor for borderline genres.

---

## 2. Code review (Agent B)

Full review: `/tmp/router-review/B-code-review.md`. Patched router (already landed): `backend/processing/detector_router.py`.

### Severity table

| Issue | Severity | Lines (original) | Fix landed |
|---|---|---:|---|
| No HTTP timeout on Anthropic call | **P0** | 161 | ✅ `timeout=8.0` in `_build_client` |
| Non-atomic cache writes (concurrent overwrite + corrupt-on-SIGKILL) | **P0** | 117–128 | ✅ `_atomic_write` (tmp + `os.replace`) + `fcntl.flock` in `_save_entry` |
| ASCII-only normalization mangles diacritics + collapses Japanese to empty | **P1** | 108–114 | ✅ `_norm` now does NFKD + strip combining marks + preserves CJK/Hiragana/Katakana |
| Cache poisoning — no TTL/version stamp on entries | **P1** | 117 / 200 | ✅ entries stamped with `_model` + `_prompt_hash` + `_cached_at`; `_entry_is_fresh` discards stale on read |
| Caches low-confidence decisions forever | **P1** | 200 | ✅ `_CACHE_CONFIDENCE_FLOOR = 0.7` — low-conf returned but not persisted |
| Brittle JSON regex `\{.*?\}` non-greedy + DOTALL | **P1** | 182 | ✅ tolerant parser: direct → fence-strip → greedy regex |
| No retry on transient 429 / 5xx | **P1** | 161–173 | ✅ `_RETRY_ATTEMPTS = 2` with 1.5s sleep on `RateLimitError` / `APIStatusError` / `APITimeoutError` / `APIConnectionError` |
| No recovery on rotated API key (401) | **P1** | 161 | ✅ on `AuthenticationError`, re-resolve key from keychain once and retry |
| `client.messages.create` kwargs leak inside a single broad `except` | P2 | 171 | Covered by typed exception handlers above |

### Top-3 findings reflected in the SMS-worthy headline

1. **8s HTTP timeout on Anthropic call** — without this, 4 concurrent stalled requests deadlocked the entire post-separation semaphore. Single line, single most-important fix in the review.
2. **Atomic cache writes + file locking** — silent total-cache-wipe under interrupted writes was a real risk in production.
3. **NFKD Unicode normalization** — "Mötley Crüe" was being collapsed to "m tley cr e"; Japanese-only titles to the empty string. Real-world collision risk.

### Tests still owed (not blocking)

A test file `backend/tests/test_detector_router.py` is not yet written. Recommended cases:
- Known-jazz cache hit + cache miss
- Empty title + empty artist (early-return fallback)
- Voice-memo-like title (heuristic short-circuit)
- Malformed JSON response (tolerant parser)
- `AuthenticationError` 401 re-resolution path
- Concurrent write safety (spawn 4 threads calling `_save_entry` on different keys)
- Diacritics + CJK normalization (`Mötley Crüe`, `Sigur Rós`, `はちみつぱい`)
- Cache-version mismatch invalidates entry

These should land in Week 1 alongside the integration into `pipeline.py`.

---

## 3. Cost + SLA (Agent C)

Full report: `/tmp/router-review/C-cost-sla.md`. Spend during analysis: $0.036 (used 20 of 50 budgeted calls).

### Latency

| Regime | p50 | p95 | p99 |
|---|---:|---:|---:|
| Cache hits (local JSON read) | **0.028 ms** | 0.049 ms | 0.131 ms |
| Fresh Anthropic calls | **1,049 ms** | 1,572 ms | 1,998 ms |

Calls 2–20 of the uncached batch hit warm Anthropic prompt cache (ran inside the 5-minute TTL). Real cold-start p99 may be slightly higher; 2 s is the conservative ceiling.

**Chosen timeout = 8.0 s** (Agent B's recommendation; Agent C recommended 3.0 s as 1.5× p99). I kept 8 s for headroom on Anthropic latency spikes — still bounds the semaphore exposure without breaking on the long tail.

### Cost projections (steady state, Zipf 60% / 1000-song popularity assumption)

| Monthly volume | With current behavior (no pre-seed) | With top-10K cache pre-seed |
|---|---:|---:|
| 5 K songs / month (launch) | $1.24 | ~$0.50 |
| 50 K songs / month (6-month projection) | **$4.28** | ~$1.80 |
| 500 K songs / month (aspirational) | $26.55 | ~$11 |

Pre-seeding the top 10K most-uploaded songs costs **$1.57 one-time** (Agent C measured actual tokens — much cheaper than my earlier $30 ballpark) and pays back in < 1 day at 50K/month.

### Outage cascade (landed in this PR)

Original behavior: API failure → fall through to `general` for all jobs. Jazz songs uploaded during the outage got the worse detector silently.

Patched behavior:
1. Cache hit (model + prompt-hash match) → use cached decision
2. API call (with retry) → use Claude decision
3. **NEW: Outage allowlist** — if title or artist matches a small canonical-jazz allowlist (Steely Dan, Jamiroquai, Pat Metheny, Miles Davis, Coltrane, Bill Evans, Wayne Shorter, Chick Corea, Monk, Parker, Brubeck, Chet Baker, Ella, Sarah Vaughan, Sonny Rollins, Stan Getz, Cannonball, Wes Montgomery, plus 14 jazz-standard titles), route jazz at confidence 0.6.
4. Otherwise → general fallback.

The allowlist is intentionally narrow (≤ 35 entries). False-positives (rock → jazz) hurt the cohort F1 more than false-negatives (jazz → general) hurt user experience, so the bar is high. Edits go in `backend/processing/detector_router.py` at the `_OUTAGE_JAZZ_ARTISTS` and `_OUTAGE_JAZZ_TITLES` frozensets.

### Batching note

Top-10K seeding via Anthropic Message Batches API at ~50% off would cost ~$0.80 vs $1.57 sequential. Not worth the engineering for V3.1 — the absolute cost is too low. Worth revisiting if we ever do a 100K seed.

---

## 4. Architecture critique (Agent D)

Full critique: `/tmp/router-review/D-architecture.md`.

### Summary table

| Question | Verdict | Urgency for V3.1 |
|---|---|---|
| Q1 — title+artist enough? Or add audio features? | Title+artist is sufficient for the dominant signal. Audio features only matter when Claude confidence < 0.6, and the safe-default-general behavior already covers that case. | **Defer** to V3.2 if confidence telemetry reveals a real gap. |
| Q2 — more than 2 categories (jazz / metal / classical / ...)? | Each new category needs a specialist detector to be worth it. We don't have metal-specific or classical-specific detectors. 2 categories is correct. | **Defer** indefinitely. |
| Q3 — Haiku vs Sonnet? | Haiku at 96% accuracy is fine for the budget. Sonnet's 3.75× cost would buy ≤ 2-3 pp accuracy. Not worth it. | **Defer**. |
| Q4 — Pipeline placement (post-sep vs parallel-with-sep vs upload-time)? | **Move to parallel-with-separation** in `pipeline.py`. Hides the 1–3 s latency inside the 30 s Modal separation. ~30-line pipeline.py change. | **Land in Week 1.** |
| Q5 — Per-bar rerank in the same trip? | Real scope: 2.5–3 weeks of work. Breaks the June 20 timeline. | **Defer to V3.2** per V3 plan's existing post-launch lever. |
| Q6 — Cache seeding strategy? | **Seed from existing `outputs/` directory** at deploy time. Higher signal than Spotify charts; no legal grey zone like Last.fm scrobbles. Cost < $10 one-time. | **Land in Week 1.** |

### One prompt tweak from Agent D, deferred to V3.2

Rename the `jazz` path to `extension_rich` to capture neo-soul (D'Angelo, Anderson .Paak, Bill Withers ballads) and reduce Agent A's observed jazz over-routing on R&B. The rename would invalidate the 124 entries in the production cache — minor cost, but worth queuing rather than rushing. Land it post-launch when we have telemetry on how many R&B uploads hit the jazz path in prod.

### Agent D's bottom line

> The Claude title+artist router is the right architecture for V3.1 — fail-safe, cheap (< $30/mo at 50K songs), and addresses the acknowledged Aja-flattening regression. **Verdict: ship it.**

---

## 5. Recommendation

### SHIP — with three Week-1 follow-ups

The router is ready to wire into the V3.1 pipeline. All P0 and P1 bugs identified by Agent B are fixed in `backend/processing/detector_router.py`. The outage cascade from Agent C is landed. Empirical accuracy is ~96% on a 100-song breadth test with no jazz-standard misroutes.

**Required for Week-1 integration PR (alongside the ACE/Jiang router wiring already in the V3.1 plan):**

1. **Move the routing call to parallel-with-separation** in `backend/processing/pipeline.py` (Agent D's Q4). Currently the V3.1 plan places it in the post-sep worker (~30 s after upload). Moving it to fire in parallel with the Modal separation job hides the 1–3 s router latency inside the 30 s separation latency. Implementation: kick off `route_detector()` as a `concurrent.futures.ThreadPoolExecutor.submit()` right after metadata is extracted; `.result(timeout=10)` it just before chord detection runs. Falls back cleanly to `general` if the future times out. ~30 LOC change in `pipeline.py`.

2. **Pre-seed the cache from existing `outputs/` at deploy time** (Agent D's Q6). Write `scripts/seed_router_cache.py` that walks `/opt/stemscribe/outputs/*/job_metadata.json`, extracts `(title, artist)` pairs, deduplicates, and calls `route_detector()` for each. Budget: cap at 5,000 unique songs and $30 spend (whichever hits first). One-time cost; eliminates cold-start non-determinism on launch day.

3. **Write `backend/tests/test_detector_router.py`** with the case list from §2 of this report. Targeted at preventing regressions of the P0/P1 fixes when someone touches this module in 3 months and doesn't remember why the cache uses `fcntl.flock`.

**Nice-to-have (not blocking):**

- Add a Week-3 telemetry pass: log every routing decision to a sidecar JSONL so we can review the long tail of low-confidence cases pre-launch. The data tells us whether the `_CACHE_CONFIDENCE_FLOOR = 0.7` threshold is well-calibrated.
- Document the privacy posture in the user-facing privacy policy: "Song title and artist metadata are sent to Anthropic to select the best chord-detection model. Audio content is never sent." Minor copy update.

### Not for V3.1 (explicitly deferred to V3.2)

- Audio-feature fallback when Claude confidence < 0.6
- More than 2 routing categories (metal / classical / electronic specialists)
- Haiku → Sonnet two-tier escalation
- Per-bar Claude reranking (separate 2.5–3 week project)
- `jazz` → `extension_rich` rename (cache-invalidation cost; queue for the V3.2 telemetry review)

---

## 6. Updated `detector_router.py`

The patched module is already in place at `/Users/jeffkozelski/stemscribe/backend/processing/detector_router.py`. Summary of what changed from the original 226-line prototype:

- **Imports added:** `hashlib`, `time`, `unicodedata`
- **New helpers:** `_norm` (NFKD-aware), `_atomic_write`, `_save_entry` (fcntl-locked), `_entry_is_fresh`, `_parse_router_json` (tolerant), `_build_client` (timeout-bounded), `_call_claude` (retries + 401 re-resolve), `_outage_allowlist_lookup`, `_fallback` (allowlist-aware), `_sanitize_for_prompt`, `invalidate_cache_entry`, `clear_cache`
- **New module config:** `_REQUEST_TIMEOUT_S = 8.0`, `_RETRY_ATTEMPTS = 2`, `_CACHE_CONFIDENCE_FLOOR = 0.7`, `_PROMPT_HASH` (sha256 of `_SYSTEM_PROMPT`), `_VOICE_MEMO_HINT` regex, outage allowlist frozensets
- **Cache key format:** unchanged shape, but `_norm` now correctly handles diacritics and CJK
- **Cache entry shape:** decisions stamped with `_model`, `_prompt_hash`, `_cached_at`; stale entries discarded on read
- **CLI:** added `--clear` and `--invalidate <title> <artist>` admin commands

Lines: 226 → 500. Doubled, but every added LOC corresponds to a P0/P1 fix or a new operational affordance (admin commands).

---

## 7. Pointers

- This synthesis: `docs/v3-router-review-2026-05-13.md`
- Empirical breadth (Agent A): `/tmp/router-review/A-breadth-test.md`
- Code review (Agent B): `/tmp/router-review/B-code-review.md` (full patch at `/tmp/router-review/detector_router_patched.py`)
- Cost / SLA (Agent C): `/tmp/router-review/C-cost-sla.md`
- Architecture critique (Agent D): `/tmp/router-review/D-architecture.md`
- Production module (patched, with Agent C outage allowlist merged): `backend/processing/detector_router.py`
- V3.1 plan (the plan this router is being integrated into): `docs/v3-ace-tuning-2026-05-13.md`
- V3 plan (original, overridden by V3.1): `docs/v3-plan-2026-05-13.md`

---

## 8. One-line summary for the Week-1 standup

**Router reviewed by 4 agents, ~96% accuracy on a 100-song breadth test, P0/P1 fixes landed, cost is $4/mo at 50K/mo — ship in Week 1 with parallel-with-separation pipeline placement + outputs-based cache pre-seed.**
