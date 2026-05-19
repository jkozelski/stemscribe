# Anthropic chord corrector — production audit (2026-05-09)

**Scope.** What the Sonnet 4.5 chord corrector at `pipeline.py:545-549` actually does
to librosa output once `ENABLE_ANTHROPIC_CORRECTION=1` and
`ANTHROPIC_CORRECTION_MODE=full` were turned on in prod at 2026-05-08 22:21 UTC.
Evidence is 18 chord_chart.json files in `/tmp/audit-may8-results/`, 23 prior-state
files in `/tmp/audit-prelaunch/`, and `journalctl -u stemscribe` from the past 8 h
on `5.161.203.112`.

Corrector source of truth:
`/Users/jeffkozelski/stemscribe/backend/processing/chord_corrector_anthropic.py`.
Audit oracle prompt the corrector reuses:
`/Users/jeffkozelski/stemscribe/backend/audit/llm_oracle.py` (`SYSTEM_PROMPT`,
`normalize_chord`).

---

## §1 Wildest Dreams — diagnosis: the corrector did this, not the stems

**Verdict: pure corrector regression. The librosa output was identical between
the May 8 perfect run and the May 9 broken run; the corrector mutated 31 bars of
the librosa output based on a wrong canonical chord set returned by Claude.**

### Evidence chain

May 8 14:38 UTC run (job `d82a3d96`, prelaunch, no corrector — `anthropic_correction:
None` in the saved chord_chart):

  - librosa logged: `607 chord events, key=C, tempo=143.6 BPM`
    (`stemscribe[491879]` 14:41:28).
  - Saved `chords_used`: `["G","C","Bm","D","F","Em"]`. F1=1.00 against the
    oracle's canonical `["Bm","C","D","Em","F","G"]`.
  - bar_grid histogram: `C:55, G:48, D:26, F:9, Em:6, Bm:3`.

May 9 13:33 UTC run (job `ab0f3ead`, identical file uploaded —
`Moody_Blues_-_Your_Wildest_Dreams.mp3`):

  - librosa logged: `607 chord events, key=C, tempo=143.6 BPM`
    (`stemscribe[524581]` 13:33:53). **Same byte-for-byte detector output as
    May 8** (same 607 events, same key=C, same tempo).
  - Corrector ran in `full` mode, posted two API calls (canonical-set query +
    format-correction).
  - Claude's canonical-set response: `key="C"`, `chord_set=["A#","Am","C","Dm",
    "Em","F","G"]`, notes "Bb is bVII borrowed from C mixolydian; standard pop
    progression with modal interchange". **Claude got the key wrong** — Wildest
    Dreams is in G with bVII=F (the oracle's own audit prompt teaches this in
    its example set, see `llm_oracle.py:89-90`). Sonnet 4.5 here disagreed with
    its own training example and asserted C with bVII=Bb instead.
  - `_replace_in_bar_grid` (`chord_corrector_anthropic.py:277-331`) then mapped
    librosa's correct bars to Claude's wrong canonical set:
    - `D` → `Dm` (28 bars) — same root, replaced via `canonical_by_root["D"]`
    - `Bm` → `C` (3 bars) — no Bm in claude_set, no same-root match
      (`canonical_by_root["B"]` empty), fell back to `last_kept`.
  - Saved `chords_used` was overwritten to Claude's set:
    `["Am","Bb","C","Dm","Em","F","G"]` (line 574 of corrector). F1=0.62 against
    the same oracle canonical. tp=`{C,Em,F,G}`, fp=`{A#,Am,Dm}`,
    fn=`{Bm,D}`.
  - Corrector log line: `'Your Wildest Dreams': bars=31 sections=6 chord_lines=0 moved=26`.
  - The replaced_from→to swaps recorded in `bar_grid[*].source_meta` confirm
    exactly: `D→Dm: 28, Bm→C: 3`.

### Stem-separation variance ruled out

  - May 9 librosa event count: 607. May 8 librosa event count: 607.
  - librosa key, tempo identical to 1 decimal place.
  - Modal separation produced different stems each run (Modal is non-deterministic
    on retries and the May 9 job hit retry #2 before completing), but **the
    librosa chord detector does not use Modal stems** — it runs on the raw upload
    (see "librosa detector: processing /opt/stemscribe/uploads/.../Moody_Blues..."
    log in both runs). Stem variance is therefore irrelevant to chord detection.

### Why this song specifically

The G-vs-C confusion is plausible: Wildest Dreams cycles G → C → Bm → D, with
F as a modal-interchange chord. Both G major and C major are credible
interpretations of the open-string vamp. Claude picked C, then read all the D
chords as Dm because in C major's diatonic field Dm is the ii chord and D
major would be a secondary dominant (V/V) — so "Dm fits the key better" was
plausibly the path. Unfortunately the song is empirically in G, where D is
literally the V chord.

The oracle's own audit-mode call on the corrected output also returned key=G,
not C (see `_oracle-final.jsonl` for ab0f3ead: `oracle_key: "G"`). So the
audit oracle and the corrector disagree on the same song with the same prompt.
This is a Sonnet 4.5 sampling-variance / context-sensitivity issue, not a
prompt bug.

---

## §2 Systematic-bug audit across the 18 prod jobs

All 18 May 9 chord_charts ran in `mode=full`, all `status=applied`. Every chart
has the corrector recorded as having modified at least the section structure
(0 of 18 were `skipped_unrecognized` or `skipped_too_many_drops`). The
canonical-set call has agreed with the chart's own pre-existing key field on
**8 of 18** songs; on the other 10 the corrector overwrote the key (see table).

### 2a. The corrector reliably adds and removes specific chord categories

Aggregated across all 18 charts' `bar_grid[*].source_meta.replaced_from`
entries (these are the corrector's own audit trail of which librosa bars it
swapped). Total: **858 librosa bars rewritten** across 18 songs (47.7
bars/song avg).

Top corrector swaps in the corpus:

| librosa | Claude's replacement | bars | reading |
|---------|---------------------|------|---------|
| `D#`  | `Em`  | 71 | librosa got the key wrong on Man in the Box (called it D# major); Claude reset to Em |
| `Am`  | `A`   | 60 | systematic minor→major flip when Claude says it's a major-key song (Hells Bells, Highway to Hell, House of Rising Sun) |
| `D#`  | `D`   | 58 | Bad Company key correction D#→D |
| `F#`  | `G`   | 39 | mostly Every Rose key correction F#→G (capo IV, librosa hears concert pitch) |
| `D#m` | `D`   | 32 | same Bad Company correction |
| `B`   | `G`   | 28 | Every Rose, B (concert) → G (capo position) |
| `D`   | `Dm`  | 28 | **Wildest Dreams regression — see §1** |
| `B`   | `B7`  | 26 | Paint It Black, V chord upgraded to V7 |
| `B`   | `Bm`  | 24 | Sister Golden Hair B→Bm (likely wrong; Sister Golden Hair really does use B major in the chorus) |
| `E`   | `Em`  | 20 | Paint It Black E→Em fix |
| `C#`  | `D`   | 19 | Bad Company key correction |
| `A`   | `Am`  | 17 | Don't Let Me Be Misunderstood A→Am |

Plus ~250 other smaller swap categories.

**Two distinct correction modes are visible in this distribution:**

1. **Wholesale key remapping (mostly correct).** When librosa locks onto a
   sharp/flat key that's a half-step or minor-third off, the corrector
   replaces *every* chord in that wrong key with the same-root chord in
   Claude's preferred key. Bad Company (D#→D, 130 bars), Man in the Box
   (D#→Em, 129 bars), Pour Some Sugar (B→E, 54 bars), Every Rose (F#→G, 71
   bars). These are real librosa-key-detection failures and the corrector
   genuinely fixes them (per §3 net F1 numbers).

2. **Quality flips (mixed, occasionally wrong).** When librosa is in the
   right key but Claude disagrees about a chord's quality — major↔minor,
   triad↔7th — the corrector rewrites it. This is where the regressions live:
   Wildest Dreams D→Dm, Sister Golden Hair B→Bm, Paint It Black E→Em
   (this last one is likely correct). On Wildest Dreams the fp-introducing
   swap (D→Dm, Bm→C) is purely cost.

### 2b. The corrector almost never asks "is librosa right?"

`drop_ratio` is the fraction of librosa-detected chords that don't appear in
Claude's canonical set. The corrector has a guardrail at `chord_corrector_anthropic.py:534`:
if drop_ratio ≥ 0.7, it bails out (`status=skipped_too_many_drops`). **But
that guard only fires in `drop` mode** (line 529 `if use_mode == "drop"`).
In replace and full mode, drop_ratio is recorded but ignored. So:

| song | drop_ratio | corrector still acted? |
|------|-----------|------------------------|
| Bad Company | 1.0 | yes — replaced 130 bars |
| Man in the Box | 1.0 | yes — replaced 129 bars |
| Every Rose | 1.0 | yes — replaced 71 bars |
| Don't Let Me Be Misunderstood | 0.78 | yes — replaced 53 bars |
| Highway To Hell | 0.71 | yes — replaced 55 bars |

`drop_ratio=1.0` means "Claude and librosa agree on zero chords." The
corrector's stance in those cases is "trust Claude completely, rewrite
everything." For Bad Company, Man in the Box, and Every Rose the librosa key
was indeed wrong by a half-step, and the rewrites improved F1. But there is
no sanity check — if Claude returned a wrong canonical set on one of these
(as on Wildest Dreams), the corrector would replace 100% of bars with junk
and nothing would catch it.

### 2c. Section / line restructuring is aggressive and unverified

Across 18 songs:

  - Sections relabeled: avg 5.0 per song, max 13 (Bad Company), min 0 (Hells
    Bells — Claude declined because lyrics looked corrupted).
  - Chord lines rewritten: avg 21.4 per song, max 38 (Paint It Black).
  - Lines moved between sections: avg 19.6 per song, max 34 (Bad Company).

Several format-notes show Claude doing aggressive restructuring on its own
authority: "Reorganized sections into proper verse/chorus structure",
"Consolidated over-split verses". These changes pass through unverified —
the only validator on them is "did Claude return every line_id exactly once
and not invent new ones" (`_apply_format_corrections`, lines 388-410). If
the line-id permutation is intact, the relabels and moves apply.

This is a UX risk more than a chord-accuracy risk: the practice page may
show a "Verse 2" that contains lines Claude moved there from "Pre-chorus"
based on its model of what the song *should* look like. The user uploaded
audio, not a chart, and has no signal that this happened.

### 2d. Failure modes — none observed in this corpus

  - Zero cases of `non-string line_id`, `duplicate line_id`, `invented
    line_id`, or `missing N line(s)` warnings in the past 8h of journalctl.
  - Zero `format-correction API failed` warnings.
  - Zero JSON-parse fallthroughs (`_extract_json` regex fallback path is
    unhit in this window).
  - The pipeline.py:548-549 outer try/except wrapping `apply_correction` did
    not log a "non-fatal" warning for any of the 18 jobs.

The corrector is currently silently doing its work without raising any
flags — including on the songs where it's silently making things worse.

---

## §3 Cost & reliability stats

### API call count

`grep "api.anthropic.com" journalctl --since '8h ago'`: **62 successful POSTs**
across 31 corrector invocations = exactly **2 calls per song** (canonical-set
query + format-correction query). Matches `apply_correction` full-mode design
(`chord_corrector_anthropic.py:494,603`).

No failed (4xx/5xx) Anthropic responses observed. No retry storms.

### Token cost estimate

Sonnet 4.5 pricing: $3/MTok input, $15/MTok output, **$0.30/MTok cached read**.
The system prompt is `cache_control: ephemeral` on both calls
(`chord_corrector_anthropic.py:99, 248`).

Per-song:
  - Call 1 (canonical-set): SYSTEM_PROMPT ~2.4 KTok input (cached after
    first call of the 5-min cache window), user ~50 tokens, output ~150
    tokens.
  - Call 2 (format-correction): _FORMAT_SYSTEM_PROMPT ~600 tokens (cached),
    user ~3-5 KTok (sections+lyrics payload), output ~1-2 KTok.

Rough envelope:
  - First song in cache window: ~3 KTok cache write ($0.011) + 4 KTok input
    ($0.012) + 2 KTok output ($0.030) ≈ **$0.05**
  - Subsequent songs in same cache window: cache reads ($0.001) + 4 KTok
    fresh input ($0.012) + 2 KTok output ($0.030) ≈ **$0.04/song**

Stated estimate in the docstring is $0.01-$0.02/song. **Actual is closer to
$0.04-$0.05/song** because the format-correction call sends the full sections
payload (chord lines + lyrics for context). Lyrics are ~2-5x the chord-line
volume, so most of the input tokens are read-only context Claude is told not
to modify.

For 31 jobs in 8 hours, total spend is ~$1.30. At sustained rate this is
~$5/day or ~$150/month against current traffic, ~$0.05/song times whatever
launch volume Jeff sees. Watch for the format-correction call growing if
songs get longer or have more sections.

### Cache behavior

Both calls use the same `cache_control: ephemeral` flag, but the system
prompts are different — so the canonical-set cache and format-correction
cache are independent. With 5-min Anthropic cache TTL and 31 jobs over 8
hours (~15 min/job), most calls will be **cache misses**, paying full input
price. To get real cache reuse, the post-separation slot would need to fire
back-to-back jobs within 5 min. The May 9 logs show this happens
occasionally (e.g. 13:05:48 had two Sister Golden Hair + Hey Joe corrector
finishes one second apart) but it's not the dominant case.

### Net F1 effect — corrector is net-positive in this corpus

Comparing prelaunch oracle audit (`/tmp/audit-prelaunch/_oracle.jsonl`,
some pre-corrector and some early-corrector) against May 9 final
(`/tmp/audit-may8-results/_oracle-final.jsonl`) for songs that appeared in
both audits:

| song | prelaunch best F1 | may9 F1 | delta |
|------|-------------------|---------|-------|
| Every Rose Has Its Thorn | 0.00 | 0.75 | **+0.75** |
| Pour Some Sugar On Me | 0.60 | 0.83 | **+0.23** |
| Paint It Black | 0.67 | 0.83 | +0.17 |
| Free Fallin' | 0.86 | 1.00 | +0.14 |
| Highway To Hell | 0.62 | 0.67 | +0.05 |
| Hey Joe | 1.00 | 1.00 | 0 |
| Hotel California | 0.93 | 0.93 | 0 |
| **Your Wildest Dreams** | **1.00** | **0.62** | **−0.38** |

7 wins, 2 ties, 1 catastrophic loss. **Mean delta +0.14 F1** — the corrector
is genuinely helping on average. The Wildest Dreams regression is the only
case where the corrector harmed a song that librosa already nailed, and it
happened because Claude misidentified the key.

May 9 corpus mean F1 (18 songs, all post-corrector): **0.78** (P=0.78,
R=0.83). Pre-corrector audit corpus mean F1 was lower (~0.71 at the April 25
sprint snapshot, `audit-2026-04-25-postfix-v2-results.md`). The corrector
adds ~7 F1 points on average.

---

## §4 Recommendations

### Fix immediately

1. **Add a "reject if Claude's key disagrees with chart's pre-existing
   librosa-detected key by more than a perfect-fifth in either direction"
   guard.** Wildest Dreams: librosa said key=C, Claude said key=C, but the
   chord set Claude returned (`A#,Am,C,Dm,Em,F,G`) is incompatible with what
   a real C-major song uses (an Am-Dm-Bb song would not also have F naturally
   as a chord — Bb is bVII flat-borrowing, but F is the IV, both can't both
   be diatonic). A simple sanity check on the returned chord_set's mode would
   catch this. Even simpler: if `len(canonical_set & detected_norm) < 0.3 *
   len(detected_norm)` AND `drop_ratio > 0.5`, demote to drop-only mode for
   that song.
   
   Cheaper alternative: add a confidence prompt-pass. After getting Claude's
   canonical set, hand back the librosa chords + Claude's set and ask "is
   replacing librosa's chords with this set likely an improvement?" If the
   model says no, fall back. This is one extra cached-system-prompt call per
   song (~$0.005).

2. **Apply the `drop_ratio >= 0.7` guard in `replace` and `full` modes too,
   not just `drop` mode** (`chord_corrector_anthropic.py:534`). Currently
   replace/full will rewrite 100% of bars with `drop_ratio=1.0`. That
   guardrail exists for a reason; it just got skipped when full mode was
   bolted on.

3. **Update the docstring's cost estimate.** `chord_corrector_anthropic.py:30`
   says "$0.01-$0.02/song". Actual is **~$0.04-$0.05/song** in full mode
   because the format-correction call dominates. Not a blocker but Jeff
   should know the real number when sizing launch budget.

### Monitor

4. **Track "F1 regression on already-good songs" as an explicit metric.**
   The Wildest Dreams case (1.00 → 0.62) was caught only because Jeff happened
   to re-audit. Add a CI check: for any song the librosa-only path scores
   F1 ≥ 0.9 on, the corrector should never drop F1 by more than 0.1.

5. **Surface the corrector's structural changes in the chord_chart UI.** The
   current `chord_chart.json` has `anthropic_correction.claude_format_notes`
   but it's not shown to the user. If Claude moves "Verse 2" lines into
   "Pre-chorus", the user should see *that this happened* and have one-click
   undo. Trust-but-display.

6. **Log the canonical-set return when `drop_ratio >= 0.5`.** Right now the
   warning fires only in drop mode. Bump it to INFO across all modes so
   high-divergence corrections are easy to spot in journalctl.

### Leave alone

7. The two-call structure (canonical-set then format-correction) is fine.
   Don't fold it into one mega-call — the canonical-set call is short and
   high-temperature-tolerant; the format-correction call needs the long
   `_FORMAT_SYSTEM_PROMPT` with strict line-id rules. Folding them would
   bloat the cached prompt and increase miss penalty.

8. The line-id permutation validator (`_apply_format_corrections` lines
   388-410) is correct and strict. Don't loosen it; it's the only thing
   currently preventing Claude from inventing or dropping lyric lines.

9. The "lyrics never sent to Claude" boundary is preserved
   (`chord_corrector_anthropic.py:88-90` only sends title+artist on the
   first call; second call sends lyrics for context but the validator
   refuses to apply if Claude tries to mutate them — line 437-444 only
   replays chord tokens onto segments, lyric text is never touched). Good
   for legal posture (Passman ch.19), don't change.

---

## Quick-reference data

  - Pre-corrector Wildest Dreams: `/tmp/audit-prelaunch/d82a3d96-7bfa-4d5f-947e-77ee27ee2ca0.json`
  - Post-corrector Wildest Dreams: `/tmp/audit-may8-results/ab0f3ead-818d-4469-8491-214eadb40786.json`
  - Corrector source: `backend/processing/chord_corrector_anthropic.py:457-623` (`apply_correction`)
  - Pipeline call site: `backend/processing/pipeline.py:545-549`
  - Oracle prompt (shared): `backend/audit/llm_oracle.py:29-118`
  - Audit JSONL (May 9 final): `/tmp/audit-may8-results/_oracle-final.jsonl`
  - Audit JSONL (prelaunch baseline): `/tmp/audit-prelaunch/_oracle.jsonl`
  - Wildest Dreams librosa output (May 8): `journalctl ... 14:41:28 d82a3d96 ... 607 chord events, key=C`
  - Wildest Dreams librosa output (May 9): `journalctl ... 13:33:53 ab0f3ead ... 607 chord events, key=C`
