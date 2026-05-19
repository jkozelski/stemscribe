# Detector Quality Audit — Comprehensive Prompt

**Purpose:** Independent agent team audit of months of detector-quality work on StemScriber. Look for missed approaches, fundamental architectural alternatives, and whether the current ~F1=0.80 ceiling is real or fixable.

**How to use:** Paste this whole document into a fresh Claude Code session in a different terminal. The receiving agent should spawn 4-5 parallel sub-agents and synthesize their findings.

---

## Context

StemScriber is a music-stem-separation + chord-detection app on Hetzner VPS, going to launch in the next few weeks. The chord detection pipeline is:

1. **librosa CQT chromagram + 24 templates** detects chord events from the bass stem
2. **Anthropic Sonnet 4.5 corrector** post-processes — returns a canonical chord_set + key, optionally rewrites the bar_grid

Over the past ~6 weeks the team has run multiple audits (18-song "may8 audit", various 10-song spot checks) trying to lift mean F1 from baselines around 0.55 → 0.78. Current state: ~F1 0.80 with the "quality_flips" gate. Multiple gates were tried; most rejected.

**The question:** Have we missed a fundamentally different approach? Is F1 0.80 the real ceiling, or is there a path to 0.90+ we haven't explored?

---

## Sources for the audit

### Primary docs (read all)
- `/Users/jeffkozelski/stemscribe/docs/detector-signal-research-2026-05-10.md` — quality_flips signal discovery
- `/Users/jeffkozelski/stemscribe/docs/detector-research-2026-05-09-corrector.md` — original gate proposals (some rejected)
- `/Users/jeffkozelski/stemscribe/docs/detector-research-2026-05-09-failure.md` — failure analysis
- `/Users/jeffkozelski/stemscribe/docs/detector-research-2026-05-09-sota.md` — state-of-the-art survey
- `/Users/jeffkozelski/stemscribe/docs/chord-research-2026-05-06.md` — earlier round of research (note the May 10 addendum at top)
- `/Users/jeffkozelski/stemscribe/docs/black-cow-maj7-diagnosis-2026-04-25.md` — specific song failure root cause
- Any other `docs/*detector*` or `docs/*chord*` files

### Memory state
- `/Users/jeffkozelski/.claude/projects/-Users-jeffkozelski/memory/stemscriber_full_state.md` — full project timeline including all detector quality work
- `/Users/jeffkozelski/.claude/projects/-Users-jeffkozelski/memory/project_detector_apr25_sprint.md` — Apr 25-26 detector sprint summary

### Actual code
- `/Users/jeffkozelski/stemscribe/backend/processing/chord_detector_librosa.py` — the detector itself
- `/Users/jeffkozelski/stemscribe/backend/processing/chord_corrector_anthropic.py` — the corrector with all current gates (qflip, retention, drop_ratio)
- `/Users/jeffkozelski/stemscribe/backend/audit/llm_oracle.py` — the scoring code (Claude-as-oracle)

### Audit data
- `/tmp/audit-may8-results/` — latest audit charts + scores
- `/tmp/audit-may8-results.pre-qflip-gate/` — pre-qflip audit (baseline 0.77)
- `/tmp/audit-may8-results.qflip-credit-contaminated/` — credit-exhaustion contaminated run
- `/tmp/audit-retention-mini/` — today's 4-song retention experiment
- The chord_chart.json files have `anthropic_correction` meta with all gate decisions

### Conversation history
- Transcripts at `/Users/jeffkozelski/.claude/projects/-Users-jeffkozelski/*.jsonl` — the actual decision logs

---

## Approaches tried (synthesized — verify in sources, don't trust this summary)

| # | Approach | When | Outcome | Documented in |
|---|---|---|---|---|
| 1 | librosa-only (no corrector) | Pre-corrector baseline | F1 ~0.55 | Apr 25 sprint memory |
| 2 | librosa + Apr 25 triple-fix (family-aware, per-root, m3-priority) | Apr 25-26 | 8/10 at A-grade | project_detector_apr25_sprint.md |
| 3 | librosa + maj7 promotion on I/IV | Apr 26 | Mixed | same |
| 4 | Anthropic corrector (drop mode) | Late Apr | Added some lift | chord-research doc |
| 5 | Anthropic corrector (replace/full mode) | Early May | F1 0.77 mean (May 9 audit) | detector-research-2026-05-09-corrector.md |
| 6 | drop_ratio >= 0.7 gate | May 10 night | Rejected — regresses wholesale-rewrite wins | conversation transcripts |
| 7 | key_distance > 2 semitones gate | May 10 night | Rejected — anti-correlated | detector-research-2026-05-10 |
| 8 | quality_flips gate | May 10-11 | Shipped, F1 0.768 → 0.804 | detector-signal-research-2026-05-10.md |
| 9 | qflip composite (qf>=1 AND dr<0.4) | May 11 | Tuned off — false positive on Day After Day | conversation |
| 10 | Bar-weight retention (>15% bars) | May 11 (today) | Tested, regresses Hells Bells | conversation + audit-retention-mini |
| 11 | MIDI-intermediate detector | Apr 23, gated off | Phase 4 regression, never shipped | memory |
| 12 | Trimplexx CRNN guitar tab | Apr | Gated off | memory |

Always-rejected: Premium Opus model upgrade ($1,200/mo cost ceiling per Jeff).

---

## Task

Spawn **4 parallel agents** (in background), each with a specific lens. Then synthesize their reports into a single recommendation doc.

### Agent A — Timeline & pattern audit
Read every doc, memory file, and audit result in the Sources list. Map the full timeline of detector-quality work over the past 6 weeks. For each approach: hypothesis, what was measured, what failed, what was learned. Output a single table sorted chronologically. Identify recurring patterns: songs that fail across multiple approaches, songs that swing wildly between runs.

### Agent B — Architectural alternatives we haven't tried
Read the corrector code in `chord_corrector_anthropic.py`. The current architecture is: **Claude generates a chord_set from scratch given title+artist, then we replace or drop**. List all FUNDAMENTALLY DIFFERENT architectures the team has NOT yet tried. Examples to evaluate (not prescribe — find more):
- Claude as a re-ranker on librosa's top-K candidates per bar (not from-scratch)
- Ensemble voting across multiple librosa template configurations
- Chord-vocabulary verification (does Claude's chord_set even make sense for this key?)
- Multi-model ensemble (Sonnet + Haiku + Opus consensus)
- Templating from known-good covers (search MusicXML datasets for songs in same key)
- Per-bar verification instead of whole-song
- Asymmetric trust (use Claude only for key detection, librosa for individual chords)
For each, estimate: implementation effort, expected F1 impact (with reasoning), why it might or might not work.

### Agent C — Variance & ceiling analysis
Read `/tmp/audit-may8-results/_oracle-qflip.jsonl` and `/tmp/audit-may8-results.pre-qflip-gate/_oracle-final.jsonl` (today's clean audit vs May 9 baseline). Same songs, same prompts, mostly same code. Compute per-song variance between runs. Identify songs that swing >0.20 F1 between identical runs — these are dominated by Claude's stochastic output. Quantify: of the gap between current F1=0.80 and a hypothetical F1=0.90, how much is "fixable noise" (variance reduction via temperature=0 + caching) vs "real signal needed" (fundamental quality gap). End with a defensible estimate of the absolute ceiling.

### Agent D — SOTA reality check
Read `detector-research-2026-05-09-sota.md` and search the web for current (2025-2026) state-of-the-art in chord recognition. What are real production systems (Chordify, Hookpad, Soundslice, AudioJam, Moises) actually achieving? What's their architecture? What benchmarks (HookTheory, McGill Billboard, Isophonics) are people using? Is F1=0.80 on a curated 18-song set actually competitive, or is the industry already at 0.90+? Cite concrete numbers from published papers or product reviews.

---

## Synthesis (after all 4 agents finish)

Write a single document `docs/detector-quality-comprehensive-audit-2026-05-11.md` answering these three questions:

1. **Have we missed an obvious architectural alternative?** If yes, which one is highest leverage and what's the path to validate it?
2. **What's the realistic ceiling for this corrector approach?** Cite Agent C's variance analysis. Is F1 0.80 the actual ceiling or is there ~0.05-0.10 of "fixable noise" via determinism (temperature=0 + caching)?
3. **Should we stop iterating on the corrector and ship?** Pre-launch time is finite. If incremental detector work won't move F1 more than ±0.03, that engineering time is better spent on UI, marketing, or other launch prep.

Be brutally honest. The team has spent months on this. The right answer might be "you've hit the ceiling, ship it" — say so if that's what the data shows.

End the doc with a single concrete next action.

---

## Constraints

- Do not fabricate. If a doc/file doesn't exist, say so.
- Cite file:line for every claim about code behavior.
- Don't propose anything the team has already rejected (drop_ratio gate, $1,200/mo Opus, key_distance gate). The docs list these explicitly.
- The user wants definitive answers, not more research-loop suggestions.
- Total time budget: 30-45 min for all agents + synthesis.

End with the path to the synthesis doc and a one-paragraph TL;DR.
