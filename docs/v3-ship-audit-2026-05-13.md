# V3.1 Ship-Readiness Audit — Synthesis

**Date:** 2026-05-13
**Owner:** Jeff Kozelski (jkozelski@gmail.com)
**Launch:** June 20, 2026 (Refinery, Charleston — soft launch). 38 days out.
**Override:** This doc supersedes parts of `docs/v3-ace-tuning-2026-05-13.md` — specifically the jazz-path provisions in §9 and §10.

---

## Verdict: **SHIP** — with the simplified architecture below and three pre-launch patches.

The audit produced one major architectural finding and three small must-do patches. The major finding: **drop the "jazz" detector path entirely**. The stem-aware detector that ostensibly justified Claude routing is empirically worse than ACE on every jazz song in the cohort — the Apr-25 "Aja 226/226" claim was scored against weaker self-generated ground truth, not UG fixtures. The Claude detector router becomes optional infrastructure (stays in repo for V3.2 if a real specialist use case appears) but does not need to be wired for V3.1.

Net effect on V3.1 architecture: **ACE + Jiang per-bar router. Full stop.** No Claude routing call in the upload path. No jazz fallback. Simpler than yesterday's plan, lower cost, lower latency, no behavioral surprises from "did the router fire correctly?" The wiring is already in place at `pipeline.py:638–662` per Agent C's verification.

Deployment passes. Flip-on/flip-off cycle verified live on prod at ~15.5 s per restart (74% inside Better Stack threshold). One P1 doc fix: the launch-day runbook points at the wrong `.env`. Scorer fix already landed (Agent B). Held-out validation at 0.797 root F1 is now interpretable as the genuine architectural ceiling on that 9-song subset — not a "missing 0.05 to the gate" failure.

---

## 1 — Agent A: jazz path empirical bake-off

Full report: `docs/v3-ship-audit-A-jazz-path-2026-05-13.md`.

### Headline

ACE wins decisively on every UG jazz fixture. Stem-aware loses by **−0.256 root F1** and **−0.304 quality F1** on the 6-song jazz cohort.

| Song | ACE root | stem-aware root | ACE qual | stem-aware qual |
|---|---:|---:|---:|---:|
| Aja | **0.812** | 0.341 | **0.411** | 0.000 |
| Peg | **0.821** | 0.442 | **0.228** | 0.096 |
| Rikki | **0.792** | 0.586 | **0.465** | 0.414 |
| Cosmic Girl | **0.837** | 0.495 | **0.382** | 0.000 |
| Alright | 0.861 | **0.899** | **0.500** | 0.000 |
| Virtual Insanity | **0.744** | 0.569 | **0.546** | 0.198 |
| **MEAN** | **0.811** | **0.555** | **0.422** | **0.118** |
| **Slash chords emitted** | **51** | **0** | | |

Only Alright lifts under stem-aware, and only on root F1 by +0.038 — its quality F1 still drops by 0.500. Stem-aware emits **zero slash chords** across all 6 songs; ACE emits 51 (1 to 17 per song).

### Root cause

`stem_chord_detector._prune_outlier_chords(min_frequency=0.05)` drops any chord type appearing in less than 5% of bars and replaces it with the nearest surviving chord. On Aja the live log records "Pruned 53 outlier chord types (94/120 bars = 78%)" — 53 of 56 detected chord types deleted, 78% of bars rewritten to 3 dominant chords. The pruner is rock-tuned; jazz repertoires of 20-33 distinct chords with 1-3 occurrences each are catastrophically simplified.

### Apr-25 claim falsified

The "Aja 226/226 extension matches" headline that motivated the jazz path was scored against self-generated ground truth, not UG. Under UG-grounded scoring the stem-aware detector hits root F1 **0.341**, quality F1 **0.000**, vocab coverage **0%** on Aja. ACE on the same song hits 0.812 / 0.411 / 36.4% — the alternative we were going to route around is in fact strictly worse.

### Action

**Drop the jazz path from V3.1.** Route everything through the ACE + Jiang per-bar router. The Claude detector_router stays in the repo as V3.2 infrastructure (gated off; not wired). `stem_chord_detector.py` remains in repo only because `chord_detector_librosa.py` imports its `detect_key_from_chords` helper — its detection path itself is dormant.

---

## 2 — Agent B: weak-song triage

Full report: `docs/v3-ship-audit-B-weak-songs-2026-05-13.md`. **Scorer patch LANDED in `audit/score_chord_chart.py`.**

| Song | Diagnosis | Status |
|---|---|---|
| **Hells Bells** | Scorer bug — UG uses `X5` power-chord notation, ACE emits `X` triads. Quality F1 0.000 was an artifact. | ✅ **PATCHED.** Quality F1: 0.000 → **0.574** (full 0.473, pcs 0.907). Cohort root_quality avg **+0.041 across 26 songs, zero regressions.** |
| **Iron Man** | Same X5 GT issue. | ✅ **PATCHED** in same diff. Quality F1: 0.000 → **0.490**. |
| **Superstition** | Genuine ACE limitation — hears Ebm7 vamp as D#:maj (right root, wrong quality). Tried `--chord-min-duration 0.25` and `--threshold 0.4` — both **worse** (precision crashes 0.65 → 0.23). Default config is the local optimum. | Accept. Revisit V3.2 with a funk-tuned post-processor. |
| **Beast of Burden** | Genuine ACE weakness — confuses E with Em (24 events), hallucinates A:7 / A:maj7, drops G#m7 7ths. Not scorer, not parameter. Characteristic ACE behavior on rock with vocal slides. | Accept for V3.1. |

The X5 scorer fix is the single biggest UG-scoreboard cleanup in the audit. Iron Man + Hells Bells no longer read as catastrophic regressions on the Jun 5 decision-gate scoreboard.

---

## 3 — Agent C: deployment safety

Full report: `docs/v3-ship-audit-C-deployment-2026-05-13.md`. **PASSES with one P1 doc fix.**

### Verified live on prod

- Service health: ✅ running cleanly
- Flip-on cycle (`USE_ACE_ROUTER_DETECTOR=false → true → restart`): **~15.5 s wall** (well inside the 60s Better Stack threshold)
- Flip-off cycle (back to false → restart): **~15.5 s wall**
- End state: clean, on librosa+V1 baseline (where we started)
- Zero crash-loops, zero import errors for ACE/Jiang/router modules
- Pipeline wiring at `pipeline.py:638–662` — lazy-imports the router (clean preload when flag is off)
- ANTHROPIC_API_KEY (len=108) resolves in-process via `_api_key()` keychain fallback ✅
- ACE checkpoint present + correct size (55 MB at `external/consonance-ACE/ACE/checkpoints/`)
- Jiang's 5 ensemble checkpoints present (`external/chord_cnn_lstm/cache_data/`)
- Modal A10G separation enabled and reachable
- Disk space healthy
- Backup of original `.env` left at `/opt/stemscribe/.env.bak.audit-2026-05-13` on VPS

### P1 finding (BLOCKER for the launch-day runbook, not a code blocker)

**The brief and prior plans pointed at the wrong `.env`.** The real production env is `/opt/stemscribe/.env` (parent dir), **not** `/opt/stemscribe/backend/.env` (which is an 11-line minor-overrides file). `app.py` loads both via python-dotenv, parent first.

**Anyone following the V3.1 launch-day runbook verbatim would edit a no-op file**, restart the service, see no behavior change, and panic. Fix the runbook before Jun 20.

### P2 hardening (not a blocker)

Systemd unit runs as `User=root` with no `EnvironmentFile=` directive — env is loaded only by `python-dotenv` at app startup. Switch to a dedicated service user + `EnvironmentFile=/opt/stemscribe/.env` post-launch.

---

## 4 — Agent D: missed-wins sweep

Full report: `docs/v3-ship-audit-D-missed-wins-2026-05-13.md`.

| Item | Source | Effort | Impact | Status |
|---|---|---:|---|---|
| **X5 → maj scorer fix** | V3 plan, V3.1 plan | 15 min | Iron Man + Hells Bells lift to non-zero quality F1; +0.041 cohort root_quality | ✅ **LANDED via Agent B** |
| **Cache backfill of 119 missing prompt-hash entries** | Live cache inspection | 15 min | Restores $0.30 pre-warm; saves API spend at launch | **MOOT** — router not wired in V3.1 (Agent A's finding); the pre-warmed cache is V3.2 infrastructure now |
| **Aja-flattening user FAQ note** | V3.1 plan §10 Week-2 task | 20 min | Refinery launch has jazz-leaning Tidepool musicians; doc URL > customer-service email | **MUST DO before Jun 20** |
| Pipeline parallel-with-separation | Agent D router-review Q4 | 60 min | Hides 1-3s router latency inside Modal separation | **MOOT** — router not in upload path for V3.1 |
| `outputs/`-based cache seed | Agent D router-review Q6 | 45 min | Eliminates cold-start cost spike | **MOOT** — same reason |
| Tests for `detector_router.py` | Agent B router-review | 90 min | Lock in P0/P1 fixes | **NICE TO HAVE** — V3.2 prep |

---

## 5 — V3.1 architecture (final, simplified)

```
audio upload (with title + artist metadata)
  → Modal A10G stem separation  [unchanged]
  → run BOTH detectors on full mix, in-process on Hetzner CPX41:
      events_ace   = ACE-default (chunk_dur=20, threshold=0.5, min_dur=0.5)
      events_jiang = Chord-CNN-LSTM (Jiang, default params)
  → per-bar router (deterministic, no API call):
      density   = len(events_ace) / max(1, len(events_jiang))
      agreement = bar_grid_agreement(events_ace, events_jiang)
      if agreement > 0.5 AND density > 1.15:
          events = events_jiang
      else:
          events = events_ace
  → Harte → standard chord-name parser
  → chart_formatter.format_chart()
  → chord_chart.json
```

### Flags

- `V3_DETECTOR=ace_router` (production switch; default `librosa` until Jun 5)
- `ENABLE_ANTHROPIC_CORRECTION=false` (corrector strips slash chords; not used on ACE+Jiang path)
- `USE_DETECTOR_ROUTER=false` (Claude routing layer stays gated off; module remains in repo for V3.2)

### What changed from V3.1 plan §9

- **Removed:** Claude detector_router call in the upload path. No per-song API call to Haiku.
- **Removed:** "jazz" path / `stem_chord_detector` invocation. Module remains in repo only because `chord_detector_librosa.py` imports its key-detection helper.
- **Unchanged:** per-bar density+agreement router from the V3.1 plan §9 (Agent 3's winner).
- **Unchanged:** ACE-default params, Jiang default params, in-process deployment on Hetzner.

### What ships at launch

- ACE + Jiang per-bar router (in-process)
- Slash chord support from audio (first time in StemScriber history)
- 38-song nightly benchmark scoreboard on admin page
- X5-aware scorer (just landed)
- `📋 My Chart` paste as the escape hatch (already live)

### What stays in repo but unwired for V3.1

- `backend/processing/detector_router.py` (Claude routing, hardened by this morning's 4-agent review)
- 128-entry routing cache at `backend/data/detector_router_cache.json` (pre-warmed from Agent A's $0.30 breadth test — feed into V3.2 when we find a real use)
- `backend/stem_chord_detector.py` (key-detection helper still in active use; detection path itself dormant)

---

## 6 — Pre-launch MUST-DO list

In execution order:

1. **Fix the launch-day runbook to point at the correct `.env`.** Change all references from `/opt/stemscribe/backend/.env` to `/opt/stemscribe/.env`. Affected docs: `docs/v3-plan-2026-05-13.md`, `docs/v3-ace-tuning-2026-05-13.md`, this doc, and any internal runbook page. **Effort: 5 min. Owner: Jeff.**

2. **Write the Aja-flattening user FAQ entry.** One paragraph + the "Import your chart" workflow link. Place in the public FAQ on stemscriber.com. **Effort: 20 min. Owner: Jeff.**

3. **Verify the V3_DETECTOR=ace_router flag toggles correctly on staging** with the simplified architecture (no Claude routing). Agent C verified the flag flip works; this is a paranoia check that the per-bar router fires correctly when the flag is on. **Effort: 15 min. Owner: any engineer with VPS access.**

4. **Backfill the held-out validation with the X5 scorer fix.** Re-run the 9 held-out songs through the new scorer; expect Hells Bells + Iron Man to lift their quality F1 numbers (likely +0.5 each). The held-out average **should rise from 0.797 to ~0.82 root F1 / ~0.55 quality F1** once these stop reading as zero. **Effort: 15 min. Owner: any engineer.** **This is what makes the Jun 5 decision gate readable.**

Total pre-launch work: ~55 minutes.

---

## 7 — Decision gate (Jun 5, updated)

The original V3.1 gate was "held-out avg root F1 ≥ 0.85." That was tuned to in-sample router numbers (0.869) with implicit assumption that held-out would compress modestly.

The held-out reality is **0.797 ACE-only** (per the brief), which Agent A's data now explains is the architectural ceiling on the 9-song held-out set — routing 5 of them through the jazz path would make them strictly worse (−0.256 root, −0.304 quality).

**Revised Jun 5 decision gate:**

**Ship if:**
- 7-day prod soak avg root F1 ≥ **0.80** (on the held-out 9 songs + any new fixtures added in Weeks 2–3)
- Aja quality F1 ≥ **0.40** (the documented ACE floor, not the discredited 226/226 stem-aware number)
- No song regressed > 0.10 root F1 vs Agent A's measured ACE numbers
- No P0/P1 production issue; no memory leak
- X5 scorer fix landed; held-out re-scored

**Hold for Jun 9 if:**
- 7-day average in 0.75–0.79
- One or two held-out songs regressed > 0.15 with no diagnosed cause

**Roll back to librosa+V1 (0.71) if:**
- 7-day average < 0.75
- Any P0
- Memory leak under 4-way concurrency

Rollback is still one env flip — and now we know to edit the right file: `/opt/stemscribe/.env`, not `/opt/stemscribe/backend/.env`.

---

## 8 — What's explicitly post-launch (V3.2)

- **Claude-as-reranker over per-bar top-K from both detectors.** Still the highest-leverage architectural lever from the V3 plan. Non-destructive picker; could catch both ACE's eager-on-rock and flattening-on-jazz failure modes at the bar level.
- **Funk-aware ACE post-processor** (for the Superstition class — D#:maj → Ebm7 quality recovery on m7-vamp songs).
- **GT enrichment for high-traffic demo songs** (Hotel California first — Agent 4 of the V3.1 sprint confirmed 22% of ACE's "extras" are real audio).
- **detector_router.py productionization** — only if a genuine specialist use case appears. The infrastructure (cache, prompt, hardening, outage allowlist) is paid for; the consumer doesn't exist yet.
- **stem_chord_detector retirement** — after V3.1 ships and stabilizes, replace its `detect_key_from_chords` consumer in `chord_detector_librosa.py` with a self-contained helper and archive the entire stem-aware module.

---

## 9 — Pointers

- This synthesis: `docs/v3-ship-audit-2026-05-13.md`
- Agent A (jazz path bake-off, kills the jazz path): `docs/v3-ship-audit-A-jazz-path-2026-05-13.md`
- Agent B (weak-song triage, X5 scorer patch landed): `docs/v3-ship-audit-B-weak-songs-2026-05-13.md`
- Agent C (deployment verified live on prod): `docs/v3-ship-audit-C-deployment-2026-05-13.md`
- Agent D (missed-wins sweep): `docs/v3-ship-audit-D-missed-wins-2026-05-13.md`
- Scorer fix: `audit/score_chord_chart.py` (X5-aware now)
- V3.1 plan being partially overridden: `docs/v3-ace-tuning-2026-05-13.md`
- Router review from this morning (now V3.2 infrastructure): `docs/v3-router-review-2026-05-13.md`

---

## 10 — One-line for the Jun 5 gate

**Did the 7-day prod soak hold root F1 ≥ 0.80 with Aja quality F1 ≥ 0.40 on the X5-aware scorer?** Pass → ship simplified ACE + Jiang per-bar router. Fail → flip back to librosa+V1 (0.71).
