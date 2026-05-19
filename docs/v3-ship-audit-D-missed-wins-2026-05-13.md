# V3.1 Ship Audit — Missed Wins Sweep

**Date:** 2026-05-13 (T-38 days to Jun 20 soft launch)
**Auditor:** Claude Opus 4.7 (1M context), analytical sweep, no code execution
**Time-box:** 60 min, deliverable cap = 2 hr of follow-up dev work
**Inputs reviewed:**
- `/tmp/router-review/{A-breadth-test, B-code-review, C-cost-sla, D-architecture}.md`
- `docs/v3-router-review-2026-05-13.md` (synthesis)
- `docs/v3-plan-2026-05-13.md` (Jiang V3)
- `docs/v3-ace-tuning-2026-05-13.md` (V3.1 ACE+router)
- V3 + V3.1 agent reports (skim)
- Current state of `backend/processing/detector_router.py`, `pipeline.py`, `backend/data/detector_router_cache.json`, `backend/tests/test_detector_router.py`

---

## Method

I cross-referenced each prior review's recommendations against what's actually merged in code today, then filtered to items that fit in ≤2hr of dev work, require no fresh empirical validation, and are not already done. I also inspected the live cache and pipeline to spot regressions and follow-throughs that slipped between writeup and merge.

---

## Candidate inventory

Listed in the rough order they came up in the source documents, not by priority. Top-3 picks are in the next section.

| # | Item | Source | Effort | Impact | Recommendation |
|---:|---|---|---|---|---|
| 1 | **Cache backfill: 119 of 128 entries lack `_model` / `_prompt_hash` stamps and will be silently invalidated by `_entry_is_fresh`** on next upload. Re-Claude-calls all 119 — wastes ~$0.36 in spend, more importantly causes a cold-call latency spike + non-determinism for songs we've already routed (Sir Duke, every jazz standard from Agent A's breadth run, etc.). Fix: walk cache, stamp each entry with current `_model` + `_PROMPT_HASH` + `int(time.time())` if missing. 15 LOC script. | Live state of `backend/data/detector_router_cache.json` (128 entries, only 9 stamped); B §2.1; D §Q6 "version key" recommendation | **15 min** | **HIGH** — eliminates $0.36 + ~150 s of aggregate cold-call latency at first prod use of pre-warmed songs. Also restores the 100-song breadth-test cache as the "pre-seed" we already paid for. | (a) MUST DO before launch |
| 2 | **Pipeline routing is post-separation, not parallel-with-separation** — Agent D's Q4 said move to parallel (kick off router thread when stems start, `.result(timeout=10)` before chord detection). Currently `pipeline.py:643-661` synchronously waits for routing AFTER separation finishes. Hides 1–3s of router latency inside the 30s Modal window. | D §Q4, router-review §5 follow-up #1 "Required for Week-1 integration PR" | **60 min** (ThreadPoolExecutor.submit + result with timeout; one new helper, one existing block refactored) | **MEDIUM** — saves 1–3 s p50 / 2 s p99 of perceived upload latency. Cold-cache uploads at the Refinery soft launch will hit this path most often (small audience = sparse cache hits). | (b) NICE TO HAVE pre-launch |
| 3 | **Seed cache from `outputs/job_metadata.json`** — Agent D's Q6 + router-review §5 follow-up #2 "scripts/seed_router_cache.py". `outputs/` has 158 jobs, 111 unique title+artist pairs already on disk. Walk, dedupe, call `route_detector()`. Cost ≤$0.17. | D §Q6 "Phase 1 internal seed"; router-review §5 follow-up #2 (explicitly listed as Week-1 task) | **45 min** (script + run; output JSON parser already trivial — saw `metadata.title` / `metadata.artist` keys present) | **MEDIUM** — boosts launch-day cache hit rate from ~5% (9 stamped entries) to ~50%+ on the actual upload distribution. Eliminates launch-day cold-call cost spike. | (b) NICE TO HAVE pre-launch |
| 4 | **Invalidate the one known-wrong cache entry: Sir Duke / Stevie Wonder → jazz** — Agent A's 100-song breadth flagged this as the only clear miss (rule 3 says borderline R&B-with-extensions → general). It IS cached today. The router has a `invalidate_cache_entry(title, artist)` CLI; one line to fix. Re-routes correctly next upload. | A §"Looks-wrong list"; B §2.4 (rollback story) | **5 min** | **LOW** dollar/F1 but **HIGH operational signal** — proves the cache invalidation tooling works before launch, and removes the one known false-positive. | (b) NICE TO HAVE pre-launch |
| 5 | **Unit tests for the P0/P1 fixes that don't exist yet:** concurrency-safety (4-thread spawn), voice-memo heuristic short-circuit, outage allowlist rescuing Steely Dan, 401 re-resolve, cache version-mismatch invalidates. 12 tests exist; B's recommended list is 15+. | B §6 "Tests still owed (not blocking)"; router-review §2 same list | **90 min** | **MEDIUM** — protects launch-week P0/P1 fixes from accidental regression. No immediate user impact. | (b) NICE TO HAVE pre-launch — but losing to other items on impact-per-hour |
| 6 | **Telemetry: log routing decisions to a sidecar JSONL** — D's "build the dataset for V3.2 fallback heuristics" + router-review §5 nice-to-have. Append `(timestamp, title, artist, decision, confidence, source)` to `outputs/<job_id>/routing.jsonl` per call. ~10 LOC in pipeline.py at the same site where `decision` is set. | D §Q1 "10-min telemetry"; router-review §5 nice-to-have | **20 min** | **LOW** pre-launch (no user impact) **MEDIUM** post-launch (Week-3 review enables threshold-tuning of `_CACHE_CONFIDENCE_FLOOR` and seeds the V3.2 audio-feature fallback). | (b) NICE TO HAVE pre-launch (data starts accruing now) |
| 7 | **User-facing FAQ note about jazz/extension flattening** — V3.1 plan §10 Week 2 "Aja regression callout: document in user-facing FAQ that jazz-extension songs may lose some extension detail in V3.1; recommend `📋 My Chart` paste for users who care." | V3.1 plan §10 Week 2 explicit; D §Q1 mentions Aja as named regression | **20 min** | **LOW** pre-launch, real **post-launch** when first jazz-pianist user hits it. No general user-facing FAQ exists today (only `frontend/billing-faq.html` and `docs/legal-faq.md`). Either inline copy on practice.html "Report Issue" tooltip OR a new section in billing-faq. | (a) MUST DO before launch — this is Jeff's first defense when Tidepool's jazz player asks why Aja is flat |
| 8 | **`X5 → maj` scorer fix** — V3 plan Week 2 + V3.1 plan §1 + §9 explicit. Iron Man + Hells Bells quality F1 currently 0.000 due to GT-vs-detector vocab mismatch. Fix is one line in `audit/score_chord_chart.py:96` per V3.1 §9. Both plans flag this; status of whether it's landed not confirmed in router-review sweep. | V3.1 §1, §9; V3 plan §Week 2 candidate 1 | **15 min** | **HIGH** for the regression-watch table on admin dashboard (Iron Man jumps 0.000 → ~0.83 quality F1 without touching the detector). Without it, the Jun 5 decision-gate scoreboard reads worse than reality. | (a) MUST DO before launch — already in both plans, just needs verification it landed |
| 9 | **Outage allowlist tested end-to-end?** The frozenset exists in `detector_router.py` (Agent C's recommendation merged) but the test file's docstring mentions it without a dedicated test. A simulated API-down + Steely Dan upload should hit `outage_allowlist` source. ~15 LOC. | C §"Outage cascade"; router-review §2 "Outage allowlist rescuing canonical jazz artists" listed in test docstring but no test method | **15 min** | **MEDIUM** — Agent C's most-cited safety feature is currently unverified by automated test. | (b) NICE TO HAVE pre-launch |
| 10 | **Sentry/error tracker hook on `source="fallback"` rate ≥ 25%** — B §7.7 "no fallback-rate alerting." Increment a counter on each route_detector call, fire an alert when fallback rate exceeds threshold over 30 min. Mostly a one-line `error_tracker.incr("router.fallback")` + admin-dashboard widget. | B §7.7, router-review §2 last row | **45 min** (need to find existing `error_tracker` plumbing + widget) | **LOW** for Jun 20 (Anthropic uptime SLA is good); meaningful by month 2. | (c) confirmed post-launch |
| 11 | **Memoize `_load_cache()` per-process with mtime check** — C §"Read-on-every-call" + B §4.3. At launch cache size (~140 entries) it's 36 μs overhead per call, real noise — but ~10 LOC to fix and removes a footgun as the cache grows. | B §4.3, C §1 last paragraph | **30 min** | **LOW** at launch volume — text-book premature optimization unless `outputs/` seed grows it ≥10K. | (c) confirmed post-launch |
| 12 | **Cache size cap / LRU eviction** — B §2.5. Pure post-launch item; size at launch will be < 200 entries. | B §2.5 | n/a | n/a | (c) confirmed post-launch |
| 13 | **Two-tier Haiku → Sonnet escalation on confidence < 0.7** | D §Q3 — explicit defer | n/a | n/a | (c) confirmed post-launch |
| 14 | **`jazz` → `extension_rich` prompt rename** — D §Q2 "queue rather than rush." Would invalidate the cache (different prompt hash). Net negative pre-launch (loses pre-warmed entries) unless paired with the cache-backfill above. | D §Q2 last paragraph | n/a | n/a | (c) confirmed post-launch |
| 15 | **Audio-feature fallback for low-confidence routing** — D §Q1. Architectural change, > 2hr. | D §Q1 explicit defer | n/a | n/a | (c) confirmed post-launch |
| 16 | **Anthropic data-retention disclosure in consent + Alexandra review of new third-party LLM disclosure** | B §5.1 | n/a (legal review) | n/a | (c) confirmed post-launch — but flag for Alexandra in next standing call |

---

## Top 3 ranked by impact-per-hour

### #1 — Cache backfill of 119 unstamped entries (15 min, blocking)

**Why this is the right call now.** This is a regression I caught by inspecting the live cache file rather than by re-reading docs — only 9 of 128 entries carry `_model` / `_prompt_hash`, so the freshness check `_entry_is_fresh()` will treat 119 entries (Aja, Hotel California, Stairway, every jazz standard from Agent A's breadth test, etc.) as misses and re-call Claude on first prod upload. The breadth-test cohort we already paid $0.30 to populate becomes worthless at launch unless we backfill. The fix is a single short script that loads the JSON, walks entries lacking `_model`, stamps them with the current model + prompt hash + timestamp, and atomic-writes back. 15 LOC, ≤15 min including verification. Highest impact-per-hour in the whole sweep: it converts $0.36 of waste + a launch-day cold-call latency spike into zero work for the user. **Must do before launch.**

### #2 — `X5 → maj` scorer fix (15 min, blocking if not already in)

**Why this is the right call now.** Both V3 and V3.1 plans flag this as Week-1 work. It's one line in `audit/score_chord_chart.py:96`. Iron Man and Hells Bells currently score quality F1 = 0.000 against the X5-form GT fixtures because the detectors emit plain triads while UG uses power-chord notation. Without this scorer fix, the Jun 5 decision-gate scoreboard misreads two songs as catastrophic regressions when they're actually performing fine. We can't responsibly take Jeff into the decision gate with a known-wrong number on the regression-watch list. This is also the cheapest credibility move for the admin dashboard going live in Week 2. **Verify it's landed; if not, land it.**

### #3 — User-facing Aja/jazz-flattening note + `📋 My Chart` callout (20 min, blocking)

**Why this is the right call now.** V3.1 plan §10 Week 2 names it as the launch-week task. We accepted Aja's quality F1 drop (0.41 from the legacy stem-aware's 226/226) as a documented regression — the contract is "users who care get the `📋 My Chart` paste escape hatch." That contract is invisible to users today because no FAQ surface mentions it. The Refinery soft launch will include musicians with jazz tastes (Tidepool's roster has at least two jazz-leaning artists per memory). When the first jazz player uploads Aja and the chart comes back flat, Jeff's only recovery is a manual email reply unless this is documented. A short paragraph either on `practice.html` (next to the existing "Report Issue" mailto) or in `billing-faq.html`'s neighbor doc gives Jeff a URL to point at. Costs 20 minutes; saves a half-hour customer-service interaction in the first week of public exposure. **Must do before launch.**

---

## Total time-box: 50 min for the top 3

Leaves 70 min of the 2hr cap for a stretch. If extra time is available, the highest-leverage stretch picks are #2 (pipeline parallel-with-separation, 60 min) and #3 (seed from `outputs/`, 45 min), in that order. Doing both within the 2hr cap is feasible if there are no merge conflicts.

---

## Honest notes on what I deliberately left off

- **Reranker / per-bar Claude over top-K** is the V3.2 lever both plans correctly defer. Pulling it forward breaks the 38-day timeline (D §Q5 estimated 2.5–3 weeks honestly). Not in scope.
- **Audio-feature fallback on `confidence < 0.6`** (D §Q1) genuinely belongs in V3.2 — D's reasoning is right: the default-to-general behavior already handles low-confidence cases safely, so the fallback's upside is invisible most of the time.
- **Fine-tuning ACE** (V3.1 §8) requires hand-written training data + license-clean corpus + significant GPU time. Multi-week project, post-launch only.
- **Sir Duke cache invalidation** (#4 above) is intellectually satisfying because Agent A flagged it explicitly, but the worst-case outcome is "Sir Duke routes through stem-aware detector tuned on Aja/Peg" — which is funk-soul music with extensions, so the routing isn't actually painful. Skipping it is fine; doing it costs 5 minutes if the muscle memory is there.
- **Sentry-style fallback-rate alerting** (#10) only matters if Anthropic has a multi-hour outage in launch month. Anthropic's SLA history says this is sub-1%-probability event in our specific window. Post-launch is right.
- **Cache LRU eviction / SQLite migration** (#11, #12) are pure scale items. Launch cache size is ≤500 entries; the JSON file approach is genuinely fine to month 6.

I considered and rejected a "chunk_dur=10 special-case on Hotel California for the demo song" suggestion (V3.1 §5 mentions it as a one-line override). It's cosmetic and applies to a song where ACE already scores 0.822 root F1 — not worth the special-case maintenance burden.

---

## Pointers

- Live cache state: `backend/data/detector_router_cache.json` (128 entries, 9 stamped, 119 needing backfill)
- Live router module: `backend/processing/detector_router.py` (550 lines, P0/P1 patches merged)
- Live test file: `backend/tests/test_detector_router.py` (12 tests, ~5 more recommended)
- Pipeline integration site (post-sep, should be parallel-with-sep): `backend/processing/pipeline.py:633-661`
- V3.1 plan: `docs/v3-ace-tuning-2026-05-13.md`
- Router review synthesis: `docs/v3-router-review-2026-05-13.md`
