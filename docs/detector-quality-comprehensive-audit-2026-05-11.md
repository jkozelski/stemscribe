# Detector Quality Comprehensive Audit — 2026-05-11

**Author:** Multi-agent audit (B/C/D + memory). Agent A (chronological timeline) did not complete; timeline content drawn from `stemscriber_full_state.md` and `project_detector_apr25_sprint.md`.
**Inputs:**
- Agent B (architectures): `chord_corrector_anthropic.py`, `chord_detector_librosa.py`, May-9 and May-10 research docs
- Agent C (variance): `/tmp/audit-may8-results.pre-qflip-gate/_oracle-final.jsonl` (N=18, mean F1 0.7676) vs `/tmp/audit-may8-results/_oracle-qflip.jsonl` (N=18, mean F1 0.8041)
- Agent D (SOTA): MIREX 2025, ChordFormer, ChordCoT, Moises, Chordify, autochord

---

## TL;DR

F1 0.80 on the internal 18-song audit is **inside the commercial shipping band** (Chordify-grade, ~5–7 points below Moises's marketing claim and 2025 academic SOTA at 84–87 MajMin). The +0.037 mean lift from the May-10 quality_flips gate is **statistically indistinguishable from Claude's stochastic noise floor** (per-song noise std 0.135, four songs swung >0.20 between identical-config runs). The corrector has a real ceiling around **F1 0.82–0.83** with current architecture once determinism is added; getting to 0.90 requires a different architecture (Claude-as-re-ranker on librosa top-K candidates, or replacing the librosa template front-end with a learned model). **Ship now.** The single most useful next action is a 2-hour temperature=0 re-run to verify the launch number is real, not lucky.

---

## 1. Have we missed an obvious architectural alternative?

**Yes — one.** The current corrector (`chord_corrector_anthropic.py:148-200`) generates a chord_set from scratch given only title+artist, then drops/replaces librosa's output against it. **Claude never sees librosa's actual per-bar candidates.** This was flagged as an open question in `detector-signal-research-2026-05-10.md` (question #7) and never built.

**The high-leverage untried architecture is Claude-as-re-ranker on librosa top-K:**
- Keep top-3 templates per beat (`chord_detector_librosa.py:118-127` currently does `argmax` only)
- Pass per-bar candidates + scores to Claude; Claude *picks*, never invents
- Eliminates the wholesale-rewrite failure mode that produces the worst regressions (Sister Golden Hair −0.52, Day After Day −0.23 between runs)

**Estimated impact:** +0.04 to +0.10 F1. Effort: 16–40h. **Post-launch, not pre-launch.**

**Sub-4-hour quick win** (could ship pre-launch if Jeff wants): Agent B's #5 — chord-vocabulary verification gate. After `_query_canonical_chords` returns, validate the returned `chord_set` is consistent with the returned `key` (≥70% diatonic/secondary-dominant/modal-interchange). Catches the exact Wildest Dreams failure pattern (Claude returns chord_set in a different key than its declared `key`). Pure validation, never adds chords, near-zero regression risk. +0.01 to +0.03.

**Architectures explicitly rejected for pre-launch by Agent B (with reasons):**
- Per-bar verification (#7): API cost balloon, latency at scale
- MusicXML retrieval (#8): needs Alexandra-time the team doesn't have
- Multi-config librosa ensemble (#3) and HMM/CRF smoothing (#2): real upside (+0.02 to +0.06) but lower per-hour return than Claude-as-re-ranker

**Front-end replacement (Agent D's lever):** The librosa CQT+24-template detector is the 2010 baseline. 2025 SOTA systems (ChordFormer, consonance-ACE, ISMIR2019-large-vocab) use conformer/transformer over CQT and score 4–7 points higher on standardized benchmarks. **This is the structurally biggest ceiling lift available**, but it's a multi-week project and out of scope pre-launch.

---

## 2. What's the realistic ceiling for this corrector approach?

**Agent C's variance analysis is the load-bearing number here.** Two runs on essentially the same config produced:
- Mean |ΔF1| per song = **0.116**
- Std dev of ΔF1 = **0.190** (≈ √2 × per-run noise → single-run noise std ≈ 0.135)
- **4 of 18 songs swung >0.20 F1 between identical-config runs** (Sister Golden Hair, Your Wildest Dreams, Hells Bells, Day After Day)
- One of those four moved *backward* (Day After Day: 0.80 → 0.57). This is the smoking gun that qflip is not uniformly beneficial — it's a stochastic resample.

**Decomposition of the 0.80 → 0.90 gap:**
- **~25–40% is fixable noise** (temperature=0 + prompt caching + k=3 self-consistency voting → realistic mean lift to **0.82–0.83**)
- **~60–75% is real signal needed** — either better detector input (Black Cow key-detection bug, Rikki zero-extensions bug, both documented) or a fundamentally different corrector contract (Agent B's #1)

**Critical implication:** the +0.037 mean lift from quality_flips is **inside the SEM** of a single 18-song A/B (SEM 0.032). Without a second deterministic run, the team **cannot statistically distinguish "qflip helped" from "Claude got lucky on Sister Golden Hair this time."**

**Absolute ceiling for current architecture: F1 ≈ 0.83**, reached by adding determinism + voting on top of the current quality_flips config. Beyond that requires architectural change.

---

## 3. Should we stop iterating on the corrector and ship?

**Yes. Ship. Three independent agents converged on this:**

- **Agent B** (architectures): *"Ship as-is. Don't build any of these pre-launch."*
- **Agent C** (variance): The current 0.037 F1 gain from the most recent gate is within noise; further gate-tuning rounds will produce stochastic ±0.02–0.04 swings indistinguishable from real signal.
- **Agent D** (SOTA): 0.80 is competitive with Chordify, a decade-long market leader. The bar for launch is "useful enough that users come back," not academic SOTA.

**Incremental detector work will not move F1 more than ±0.03 in the launch window.** That engineering time is better spent on UI polish, marketing prep (the drafts in `docs/marketing-drafts-2026-04-26.md`), or Tidepool/testimonial outreach — all of which have higher launch-day ROI than another corrector gate.

**Caveat on the methodology gap (Agent D):** F1 0.80 is on a curated 18-song rock set scored by a same-family Claude oracle. This biases up vs an industry benchmark (mir_eval on Isophonics/Billboard). Post-launch, run one honest comparable number — that's the only way to know where StemScriber actually sits relative to MIREX 2025.

---

## Concrete next action

**Re-run today's qflip config with temperature=0. ~2 hours total.**

This single change does three things at once:
1. Verifies whether the +0.037 quality_flips lift is real signal or a lucky stochastic draw (Agent C's #1 recommendation).
2. Locks the corrector output so launch users get reproducible charts — a UX win independent of F1.
3. Closes ~60% of the variance gap "for free" (no algorithm change), establishing the honest launch F1 number.

If the deterministic re-run lands near 0.80 → the launch number is real, ship. If it drops back toward 0.77 → revert qflip and ship the simpler pre-gate config (still in the commercial band). Either way, the question gets answered, and the answer is the launch-readiness number.

Everything else (Claude-as-re-ranker, chord-vocab verification gate, front-end replacement) goes on the post-launch backlog.

---

## Appendix: known-failing songs (do not waste pre-launch cycles on these)

- **Black Cow** — upstream key-detection bug (Krumhansl-Kessler fooled by slash-chord moments). Doc: `black-cow-maj7-diagnosis-2026-04-25.md`. Real fix is ~2h post-launch.
- **Rikki Don't Lose That Number** — detector outputs zero extensions, different upstream pitch-detection issue, untouched.
- **Day After Day** — variance-dominated; deterministic re-run likely fixes.
