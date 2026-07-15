# Research Brief — Polyphonic Pitch → Chord Recognition for StemScriber

**Date:** 2026-05-25
**For:** A fresh Claude Code session in another terminal
**Mode:** Research only — no code changes, no deploys. Output is a written analysis at the end.

---

## Who you are working for

StemScriber (https://stemscriber.com) — a web app that uploads a song, separates it into stems (vocals/guitar/bass/drums/piano/other), and produces a chord chart for practicing along to the original recording. Public soft launch **June 20, 2026** at Refinery (Charleston). Hobbyist musicians are the primary audience.

## Current chord-detection state (as of 2026-05-25)

- **In prod:** V1 librosa on the FULL MIX + Anthropic LLM correction layer. Held-out root F1 ≈ **0.71** on a 14-song audit set vs Ultimate Guitar ground truth.
- **Gated behind flag:** V3.1 = ACE + Jiang per-bar router. Held-out F1 = 0.797. Decision gate June 5.
- **Already in pipeline but NOT used for chord detection:** the guitar stem (from htdemucs) is run through Basic Pitch → MIDI for the Guitar Pro export. Those MIDI notes are **discarded** before chord detection runs (which operates on the full mix).
- **Stem separator:** htdemucs_6s primary, BS-RoFormer secondary, both via Modal A10G GPU.
- **Tab transcription model:** a CRNN ("trimplexx") was trained but is gated off (`ENABLE_CRNN_TRANSCRIPTION=false`).

## The hypothesis to test

> The chord detector is running with one hand tied behind its back. The guitar stem already gets polyphonic pitch transcription (Basic Pitch → MIDI). If we feed those simultaneous notes into chord inference — alone, or as a confidence booster on top of the full-mix detector — we could meaningfully bump F1.

The user described this as analogous to a polyphonic tuner (TC Electronic PolyTune) that tells you string-by-string which notes are sounding. Same idea, applied to chord recognition from already-separated stems.

## Questions to answer

Numbered so the response can address each:

1. **State of the art (2024–2026):** What are the best-performing approaches to automatic chord recognition right now? Compare full-mix harmonic analysis (librosa CQT, chord templates, HMMs) vs stem-separation-aware approaches vs end-to-end neural (ACE, BTC, Jiang, Chordify-style). Include held-out F1 numbers where published.

2. **Stem-based chord recognition:** Is anyone doing chord detection on separated guitar/piano stems specifically? Published papers, open-source implementations, blog posts, GitHub repos. What's the F1 gain vs full-mix on the same datasets?

3. **From MIDI → chord:** Given a stream of MIDI notes (with onset/offset times) from the guitar stem, what's the best published algorithm for inferring chord labels? Template matching, NMF, harmonic-rule-based, learned? Open-source implementations?

4. **Polyphonic transcription quality on isolated stems:** Basic Pitch is what we use. Are there better options for transcribing a separated guitar/piano stem to MIDI? Specifically evaluate: Spotify Basic Pitch, Magenta MT3, Demucs+CREPE chain, Banquet (already on our radar — see model_survey memory). Need commercial-license-compatible only.

5. **Confidence-fusion approach:** If we keep V1 librosa as one source AND add a stem-MIDI-based detector as a second source, what's the right way to combine them? Late fusion via voting? Weighted by per-bar confidence? Trained classifier on top? Examples from published systems.

6. **Real-time polyphonic-tuner angle (secondary):** Open-source algorithms that, given live mic input of someone strumming an open chord, identify which of the 6 strings are in/out of tune simultaneously. Could become a separate "Guitar Coach" practice feature. Note license carefully.

7. **Competitor analysis:** What does Chordify do under the hood? Moises? Klang.io? Anything they've published about chord-detection architecture. Don't scrape; use published papers + their public docs.

8. **Dead-ends check:** Read `/Users/jeffkozelski/.claude/projects/-Users-jeffkozelski/memory/project_chord_dead_ends_ledger.md` BEFORE proposing anything. The user has a strong "don't re-litigate" preference. Flag any of your proposals that overlap with the ledger.

9. **Legal/license constraints:** Anything you recommend must be commercial-license-compatible. Read `/Users/jeffkozelski/.claude/projects/-Users-jeffkozelski/memory/project_training_data_legal.md` for what's already verified OK vs BLOCKED. If a model was trained on blocked data (mySongBook, RWC, Billboard, JAAH, SynthTab), reject it.

## Output expected

A written analysis (markdown), saved to `~/stemscribe/docs/polyphonic-chord-research-findings-2026-05-25.md`, containing:

- **TL;DR** (3 sentences): Is the hypothesis (stem MIDI → chord inference) likely to bump F1? By how much? What's the cheapest experiment to validate?
- **Per-question answers** (numbered to match above)
- **A ranked list of concrete next experiments**: each with (a) what it tests, (b) effort estimate in hours/days, (c) expected F1 lift, (d) license risk
- **Top recommendation**: ONE thing to try first, with rationale
- **Sources**: real URLs, paper titles, repo links — no fabricated citations

## Constraints

- **No code changes**, no model training, no deploys. Pure research.
- **No proposing scraped chord libraries** — that was settled by counsel (Alexandra Mayo) in April 2026. Project ripped one out (chart-recall LoRA) for exactly this reason.
- **Plain language** in the TL;DR and recommendations. The user is a musician, not an ML engineer. Jargon is OK in per-question detail sections.
- **Cite real sources** — papers, GitHub, blog posts with URLs. Don't make up references.
- **Time budget:** spend the context you need. This is a strategic investigation; depth matters more than speed.
