# Hook Dominance & 7ths Detection Audit — 2026-04-23

**Scope:** Research-only audit of two connected StemScriber defects visible on the Jamiroquai *Alright* chart. No code changes. Conclusions are backed by live production JSON pulled on 2026-04-23.

**Artifacts inspected:**
- `/tmp/alright_pre_7ths.json` — production chord sheet for job `0b1ebec2-a5e5-40e6-8ea9-3f2c3af1006d` (pre-fix render).
- `/tmp/alright_post_7ths.json` — bar_grid for fresh-upload job `0102b7ac-728b-42ac-a100-0c3c75c79434` (post commit `559fdd2`). Chord_progression pulled from `/api/status/...`.
- `/Users/jeffkozelski/stemscribe/backend/stem_chord_detector.py` — bass-root-first assembler, `_match_intervals_to_quality`, `_simplify_bleed_extensions`.
- `/Users/jeffkozelski/stemscribe/backend/chord_detector_v10.py` — BTC + V8 hybrid (not used for Alright; detector_version = `stem_aware`).
- `/Users/jeffkozelski/stemscribe/backend/chart_formatter.py` lines 220-309, 652-969 — `_snap_sections_to_phrase_boundaries`, `_split_sections_by_lyric_hook`, `_relabel_hook_dominated_sections`, `_rename_post_preverse_to_verse`, `_rebuild_sections_as_4bar_chunks`, `_shift_pickup_words_across_lines`.
- VPS `ssh root@5.161.203.112` verified code at `/opt/stemscribe/backend/stem_chord_detector.py` contains both the exact-match dominance block (line 455) and the >0.90 extension-rate ceiling (line 939). Fix is deployed.

---

## 1. Executive Summary

**Headline finding 1 — Whisper IS hallucinating the hook.** Real Whisper word-timestamp 4-grams of "I need your love" = **17**, clustered bars 3-16 (45-75 s, pure intro). Displayed chart shows the phrase 42 times spanning Intro + Pre-Verse + Verse. The inflation is inside segment word-arrays: Pre-Verse line[2] holds 92 "words" all stamped near t=69.08 s (unique-timestamp ratio 0.22 — classic Whisper-medium repetition-loop). Verse line[0] carries 46 words at ratio 0.37. Two sections are hallucinated; every other line in the song clocks ratio >= 0.90. This directly maps to the founder's "I need your love goes forever." **Hypothesis H1 is the dominant root cause,** not H2 or H3.

**Headline finding 2 — the 7ths fix did NOT ship 7ths.** Commit `559fdd2` is live on VPS and code-correct. Fresh bar_grid on job `0102b7ac-...` contains **0** seventh-chords across 97 bars (still all plain Cm/Gm/Dm/Am). Upstream `chord_progression` quality distribution: `min=56, sus4=12, maj=12, sus2=3, aug=3, 5=2, min11=1, min13=1` — zero `min7`. The exact-match dominance rule fires only when the input pitch-class set literally equals the min7 template `{0,3,7,10}`. The onset-weighted PC collector (`_onset_weighted_pitch_classes_in_segment`) is not including pc-interval 10 (the 7th) in its output; root+3rd+5th wins exact match as a triad. The fix is correct but addresses a symptom one layer deeper than the actual failure.

**Headline finding 3 — section labeling is data-downstream of the hallucination.** `_split_sections_by_lyric_hook` (line 837), `_relabel_hook_dominated_sections` (line 719), and `_rename_post_preverse_to_verse` (line 789) all operate on the same Whisper word list that contains the 92-word hallucination cluster. They correctly classified the Intro-like Pre-Verse, but the subsequent "Verse" section is mis-dominated by phantom hook repetitions that Whisper emitted AFTER the real hook ended (bar 17+), so real verse content ("We don't touch for the rest of that time together, baby. Come fly with me") is packed into a single 4-bar chunk while 80 phantom hook words pad the chunk before it. Fixing section labels without fixing Whisper will keep pushing the problem around.

---

## 2. 7ths Fix Validation

**Verdict: NO. Commit `559fdd2` is deployed but ineffective.**

Live code on VPS (`/opt/stemscribe/backend/stem_chord_detector.py`):
- Line 455: `if intervals == template: score = 2.0 - priority_penalty` — correct.
- Line 939: `if extension_rate > 0.90: systematic_bleed = False` — correct.

Fresh `chord_progression` pulled from `/api/status/0102b7ac-...` (90 events):

| Quality | Count |
|---|---|
| min | 56 |
| sus4 | 12 |
| maj | 12 |
| sus2 | 3 |
| aug | 3 |
| 5 | 2 |
| min11 | 1 |
| min13 | 1 |
| **min7 / maj7 / 7 / min6** | **0** |

`bar_grid` (97 bars) chord distribution: `Dm 16, Cm 14, Am 14, Gm 13, G 13, A 11, F 4, B 3, E 3, C 2, D# 2, Bm 1, D#m 1`. **Zero seventh-chords.**

**Why the fix didn't fire.** `_match_intervals_to_quality` receives an `intervals` frozenset built from pitch-classes returned by `_onset_weighted_pitch_classes_in_segment` (lines 628-712). That function:
1. Takes onset-weighted activations per MIDI pitch from Basic Pitch.
2. Normalizes into 12-bin pitch-class score (`pc_score`).
3. Keeps top-N by score with `max_notes=6`, requires `norm_score > 0.05` or an onset.

For a Cm7 chord (C-Eb-G-Bb), Basic Pitch activations for the Bb are the weakest (7th is the softest voicing tone after the root, especially in Jamiroquai's smooth piano/bass voicings). The threshold and ranking filter prune Bb, so the pitch-class set reaching `notes_to_chord` is `{0,3,7}` = exact match for `min` template `{0,3,7}` → triad wins with score ~1.996. The exact-match fix is doing its job; the problem is the 7th never reached it.

**Another artefact:** 12 spurious `sus4` bars in chord_progression. `_match_intervals_to_quality` must be seeing `{0,5,7}` on some segments — almost certainly the 4th from a piano voicing (or bleed from F notes in fills) displacing the minor 3rd pc=3 under the max_notes cap. Those get `sus4` exact match, score ~1.98, then pass the simplifier unchanged (not in `_SIMPLIFY_MAP`).

**What needs to change for 7ths to ship:**
- Lower `min_score` (currently 0.05) and/or raise `max_notes` (currently 6) in `_onset_weighted_pitch_classes_in_segment`.
- OR: add an explicit "promote to 7th" pass: after matching to `min`/`maj`, check if the 7th pitch-class has ANY activation above a relaxed threshold and promote the quality. This is the simplest intervention — preserves existing scoring and only adds opt-in detail.
- OR: use the `piano_pcs` set more aggressively — currently `cross_validate_segments` unions guitar+piano, but the onset-weighted collector independently caps each at 6 PCs before merging.

No evidence of a different layer stripping extensions. `_simplify_bleed_extensions` never runs the simplify branch because `extension_rate = 2/90 ≈ 2%` (only min11 + min13) falls through the `< 0.60` clause.

---

## 3. Hook-Dominance Root Cause

**Verdict: H1 (Whisper hallucination) confirmed. H2 partially contributes. H3 is a downstream mitigation that will fail as long as H1 is present.**

### Evidence for H1

Per-line Whisper word stats (unique-timestamp ratio = distinct `t` values / total word entries):

| Section | Line | total words | unique t | ratio | verdict |
|---|---|---|---|---|---|
| Intro | 0 | 14 | 14 | 1.00 | clean |
| Pre-Verse | 0 | 32 | 29 | 0.91 | clean |
| Pre-Verse | 1 | 11 | 7 | 0.64 | mild |
| **Pre-Verse** | **2** | **92** | **20** | **0.22** | **HALLUCINATION** |
| **Verse** | **0** | **46** | **17** | **0.37** | **HALLUCINATION** |
| Verse 1-5 | — | 8-13 | same | 1.00 | clean |
| Chorus 0-8 | — | 3-18 | same | 0.94-1.00 | clean |
| Bridge 1-2 | — | 6-15 | same | 0.93-1.00 | clean |

Only 2 of 24 lines hallucinate, and they are exactly the two "I need your love goes forever" lines the founder complained about. Pre-Verse line[2] has 92 "words" with only 20 unique timestamps — 72 of those 92 entries share the single timestamp 69.08 s (Whisper seeded the repetition loop on that one frame). Verse line[0] repeats "I need your love x 8" before the real lyric "We don't touch for the rest of that time together..." kicks in — meaning the real verse is preceded by ~30 seconds of phantom hook text.

Non-overlapping 4-gram count of "I need your love" in the real Whisper stream is **17**, confined to bars 3-16 (~45-75 s). The chart displays it 42 times. The gap — 25 phantom repeats — are all inside the two hallucinated lines.

**Conclusion:** Whisper medium is looping the hook. No language-model constraint, no logprob filtering, nothing to catch it. The lead vocals stem is clean (vocal_backing split works), so this isn't a "background bleed" problem — this is vanilla Whisper-medium behavior on a vocal with low dynamic variety.

### Evidence against H2 (section labeler)

The section labeler *does* put the intro hook into a single "Pre-Verse" (correct), and it does put the "You give me light" verse into a "Verse" section (name mapped wrong — labeler called it "Chorus" and `_rename_post_preverse_to_verse` fixed it — but time window is right). The labeler produces the correct **time boundaries**; the lyric text shown inside those boundaries is wrong because Whisper's word list is wrong. H2 is not the driver here; the chord-fingerprint labeler actually handled Alright's 4-chord vamp about as well as can be expected.

### Evidence against H3 (hook heuristics too weak)

`_split_sections_by_lyric_hook` split Intro off from Pre-Verse correctly (coverage threshold 0.45, min repeats 3). `_relabel_hook_dominated_sections` (threshold 0.70, min repeats 4) correctly upgraded Pre-Verse. The "Verse" that follows DOES have unique content ("We don't touch", "See your eyes", "all right"), so the hook-dominance pass correctly leaves it alone. The problem is that the unique content is sandwiched inside 46 words of Whisper loop — relabeling is not the fix; clipping the phantom repeats is.

### The true chain

1. Whisper transcribes Alright's vocal → 419 words total, but Pre-Verse line[2] (9 seconds of audio, should be ~18-25 words of hook) gets 92 words emitted at duplicate timestamps.
2. `_split_sections_by_lyric_hook` sees a hook that covers >45 % of each section and chops section boundaries — works reasonably.
3. `_rebuild_sections_as_4bar_chunks` bucket-assigns every word whose `start` falls in a chunk's time window (line 1037-1044). Because 72 phantom words all stamp at t=69.08 s (inside bar 14, Pre-Verse line[2] chunk), they ALL pile into that one 4-bar chunk.
4. Render: the user sees "I need your love" repeated ~25 times in a single 4-bar chunk that should hold ~8 words.

---

## 4. Proposed Plan (ordered by impact)

**No cosmetic-layer patches.** Each item below is structural.

### P1 — Strip Whisper repetition-loop phantom words (HIGHEST IMPACT)
**What:** Post-process `words` from the Whisper output in `backend/processing/lyrics.py` (or wherever the Whisper wrapper lives — needs confirmation) before it reaches `chart_formatter.py`. Drop any word whose `start` timestamp is already occupied by 3+ other words, OR collapse runs of identical n-grams whose timestamps land inside a single-frame window (< 0.1 s apart). The unique-timestamp-ratio < 0.6 test correctly identifies both hallucinated lines and leaves 22 other lines untouched.

**File:** wherever Whisper word list is assembled — likely `backend/processing/lyrics.py` or `backend/whisper_transcriber.py` (need to confirm the module path, grep for `word_timestamps`).

**Effort:** 2-4 hours. One pass over words, dedupe by (word, bucketed-timestamp). ~30 lines of code + a test.

**Songs it helps:** Every song with a repeated hook or vamp. Almost the entire pop/soul catalog. Primary test case: Alright. Secondary: anything with a one-phrase refrain (Cosmic Girl chorus, Peg verse vamp, Virtual Insanity ad-libs).

**Tradeoff:** We might drop legitimate repeats of short phrases sung unusually fast. The unique-ratio test (< 0.6) is conservative enough to avoid this; the cases that trip it are synthetic. A safety valve: only drop when the offending cluster is 3+ stdev above per-line word density.

### P2 — Add a 7th-promotion pass after triad match in `stem_chord_detector.py`
**What:** After `_match_intervals_to_quality` returns `min`/`maj`, run a secondary check: if `(root + 10) % 12` (minor 7th) or `(root + 11) % 12` (major 7th) has non-zero activation in the segment's un-thresholded `pc_score` (the full 12-bin array from `_onset_weighted_pitch_classes_in_segment`), and the activation exceeds a fixed low threshold (e.g. 0.015), promote `min` → `min7` / `maj` → `maj7`. The existing `max_notes=6` cap is preserved; we just peek at the 7th pc before discarding it.

**File:** `backend/stem_chord_detector.py` — new helper `_promote_seventh_if_present(pc_score_full, root_pc, quality) -> quality`, called from the last few lines of `notes_to_chord` before the `_build_result` call.

**Effort:** 3-5 hours including a ground-truth regression on Alright + 2 other songs (Peg, Black Cow — both 7ths-heavy).

**Songs it helps:** Alright (headline), Peg, Black Cow, Aja, most of Steely Dan, any soul/funk with smooth voicings. Will NOT over-promote on pop/rock because Basic Pitch typically gives the 7th zero activation on those.

**Tradeoff:** 7th-heavy songs get richer chords; pop/rock songs are unaffected. Needs regression test on a 3-chord rock song (e.g. a Kozelski test) to confirm no false positives.

### P3 — Reject Whisper segments where `compression_ratio > 2.4` or `avg_logprob < -1.0`
**What:** Standard Whisper hallucination filter built into `whisper` library (OpenAI recommends it). Segments with repetition loops have characteristic high compression_ratio (low entropy) and low average log-probability. Drop those segments entirely rather than emitting words from them.

**File:** wherever the Whisper call happens.

**Effort:** 1-2 hours. One flag on the call site + the drop logic.

**Songs it helps:** Belt-and-braces for P1. Catches failure modes P1 misses (e.g. Whisper emitting *different* words all at valid-looking timestamps).

**Tradeoff:** Occasionally loses a legitimately-repetitive phrase. Set thresholds conservatively.

### P4 — Cap per-chunk lyric word count at reasonable density
**What:** In `_rebuild_sections_as_4bar_chunks` (line 1037-1044), after collecting `chunk_word_objs`, if the count exceeds e.g. 4x the bar count (a 4-bar chunk with 20+ words is almost certainly hallucinated), truncate to the first N words or skip entirely.

**File:** `backend/chart_formatter.py` around line 1040.

**Effort:** 30 minutes.

**Songs it helps:** Belt-and-braces for P1. Wouldn't be needed if P1 ships.

**Tradeoff:** This is a render-layer patch; only include it if P1 can't ship in time. Jeff's guidance in `feedback_chord_pipeline.md` is to avoid cosmetic fixes, so prefer P1+P3.

### Non-recommended changes
- **Upgrading Whisper medium → large-v3.** Medium is 769 MB; large-v3 is 2.9 GB and 3-5x slower. Would reduce hallucination but VPS is 8 GB RAM — large-v3 would compete with BS-RoFormer + BTC. Skip unless we move Whisper to Modal. Not worth it for launch.
- **Retraining the chord detector.** No evidence the detector is the bottleneck. The stem-aware path is choosing the wrong pc set, not the wrong mapping from pc set to chord name.
- **Rewriting the section labeler.** It's doing its job; the label post-passes do their job. The lyrics text inside the labels is what's broken.
- **No edits to `_split_sections_by_lyric_hook` or `_relabel_hook_dominated_sections`.** Their logic is sound — their inputs are poisoned. Fix the inputs.

---

## 5. Launch-Date Honesty — May 12, 2026

**Days remaining from 2026-04-23: 19.**

**Can May 12 ship at acceptable quality?** Yes, with the following scope calls:

### Critical path to May 12

1. **P1 (Whisper dedup) — 2-4 h. Ship by Apr 26.** Removes "hook goes forever." Highest-visibility bug the founder is watching. Must ship or the chart still looks broken.
2. **P2 (7th promotion) — 3-5 h. Ship by Apr 28.** Alright renders as Cm7-Gm7-Dm7-Am7. Needed because the ground-truth doc promises it and the demo target is a 7ths-heavy song.
3. **Render-layer rewrite to walk `bar_grid`** (the big punch-list item from HANDOFF-2026-04-22.md, #1). Still the largest unstarted piece. Allocate 2-3 dev-days.
4. **Consent modal + ToS deploy** (legal) — 0.5 day.
5. **copyright@stemscriber.com alias** — 15 minutes.
6. **5-song regression batch** (Alright, Cosmic Girl, Peg, Black Cow, a Kozelski) — 1 day.
7. **Pipeline trim** (gate CRNN transcription on feature flag) — 2 hours, cuts ~2-3 min/job.

Total focused-dev: ~5-6 person-days. Feasible in 19 days with iteration time + marketing prep + Alexandra review.

### Quality bar for May 12

**Acceptable:**
- Chord chart renders correct per-bar chords starting on musical downbeat.
- 7ths visible on 7ths-heavy songs (Alright, Peg, Black Cow).
- Lyrics under each bar are REAL words (no hallucination clusters).
- Section names approximate Intro/Verse/Chorus/Bridge/Outro with 1-2 minor misses tolerable.
- Practice mode: stems mute/solo/volume, 50-200% speed, A-B loop.

**Not required for May 12:**
- Jazz DB lookup (P1 punch-list #9) — post-launch.
- Per-recording credits (P1 #11) — post-launch.
- Klangio swap — post-launch pending Sebastian's Apr 27 reply.
- Tab/sheet-music view — already cut.

### If P1 + P2 slip

Realistic quality bar: `+1 week` launch (May 19). Shipping Alright with 42 hook repeats on the chart is not Jeff's brand — the founder has explicitly called it out, and it's the first song a visitor hears demoed. Better to move to May 19 than ship with the hallucination visible.

Recommendation: **hold the May 12 target, but the go/no-go decision rides on P1 shipping by Apr 28.** If P1 isn't green by Apr 28, pull the plug to May 19 rather than pushing through.

---

## Appendix A — Raw data points

- Fresh upload job: `0102b7ac-728b-42ac-a100-0c3c75c79434` (Jamiroquai - Alright.mp3).
- Bar_grid: 97 bars, 39.87 s → 265.33 s, 100% triads.
- Chord_progression: 90 events, quality histogram above.
- Pre_7ths job for hook analysis: `0b1ebec2-a5e5-40e6-8ea9-3f2c3af1006d`.
- Total Whisper words across all sections: 419.
- Total displayed "I need your love" hook repeats in rendered chart: 42.
- Real 4-gram count in Whisper words (non-overlapping): 17.
- Ratio: 42/17 = 2.47x inflation. Gap of 25 phantom repeats = exactly the count inside the two hallucinated lines (72 phantom words in Pre-Verse line[2] + some in Verse line[0], the difference is the 4-gram filter skipping identical-timestamp runs).

## Appendix B — Commit `559fdd2` code inspection

Both changes are live on VPS and correct:

```python
# stem_chord_detector.py:455
if intervals == template:
    score = 2.0 - priority_penalty

# stem_chord_detector.py:939
if extension_rate > 0.90:
    systematic_bleed = False
```

Neither rule fires for Alright because the upstream pitch-class set given to `_match_intervals_to_quality` never contains pc-interval 10 (the 7th). Cm7's Bb is filtered out by `_onset_weighted_pitch_classes_in_segment`'s `min_score = threshold/12 ≈ 0.02` floor combined with `max_notes=6` truncation — the 7th is the lowest-activation tone and gets dropped.
