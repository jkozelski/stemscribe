# Detector Research — Why Sister Golden Hair (.40) and Don't Let Me Be Misunderstood (.15) Still Fail

**Date:** 2026-05-09
**Scope:** Diagnosis of the 2 still-failing songs from the May 9 17-song audit run through the librosa + Anthropic-corrector pipeline. Research only — no code changes.
**Reading:** `stemscriber_full_state.md` (May 9 audit table), `docs/chord-research-2026-05-06.md` + May 9 addendum, `backend/processing/chord_detector_librosa.py`, `backend/processing/chord_corrector_anthropic.py`, `backend/audit/llm_oracle.py`, the four `chord_chart.json` files in `/tmp/audit-may8-results/`, and prod `journalctl` (May 8–9).

**Headline.** The May 6 plan predicted these 2 would fail because of an upstream "wrong-key cascade" inside librosa. **That prediction is wrong.** On both songs librosa actually identifies the correct key center (or close to it) and recovers most canonical chords from the chromagram. The catastrophic F1 numbers come from the **chord_corrector overwriting librosa's correct output with chord names from the wrong key**, because (a) the corrector's `replace`-style logic is structurally lossy when the canonical chord set spans accidentals librosa already got right, and (b) the corrector and the audit oracle are pinned to different Claude models that disagree about what the canonical chord set even is. Detector engineering will not fix this. The fix is upstream of the detector — in the corrector's contract with librosa and in the disagreement between the two LLMs.

---

## §1. What librosa actually outputs on these 2 songs

### Reconstruction method

The corrector preserves librosa's original chord name in `bar_grid[i].source_meta.replaced_from` for every bar it rewrites, and leaves untouched bars without that key. Concatenating those gives the true raw librosa pre-corrector chord stream. Cross-checked against `journalctl -u stemscribe` logs which print librosa's detected key directly (`chord_detector_librosa.py:158-161`).

### Sister Golden Hair (de09e50d…)

| Source | Key | Chord histogram (top) |
|---|---|---|
| **librosa raw** (logs + `replaced_from`) | **B** | A=25, B=24, E=23, G#m=18, F#m=9, C#m=7 |
| **after corrector** (`bar_grid[i].chord`) | _bar_grid uses_ `claude_key=G` | A=37, E=37, Bm=32 |
| **audit oracle (`claude-opus-4-7`)** | **E** | canonical set = `[A, B, C#m, D, E, F#m, G#m]` |
| **corrector oracle (`claude-sonnet-4-5`)** | **G** | claude_chord_set = `[A, Am, Bm, C, D, E, Em, G]` |

Prod log line (May 9 13:00:02):
```
librosa detector: 427 chord events, key=B, tempo=136.0 BPM
```

**Librosa was substantially right.** Its 6 most-frequent roots — A, B, E, G#m, F#m, C#m — match 6 of the 7 canonical chords (only D missing). Librosa's detected key "B" is one of the diatonic options (B is the V of E major, the canonical key); the chromatic energy mass is in the right neighborhood. The 24-template detector cannot distinguish E major from B major from A major when the songs uses all three — it picks whichever has the highest correlation across the song, and B happened to win.

### Don't Let Me Be Misunderstood (1a73825b…)

| Source | Key | Chord histogram (top) |
|---|---|---|
| **librosa raw** (logs + `replaced_from`) | **Bm** | Bm=18, A=9, F#=9, Em=8, G=7, Dm=6, Am=4, B=2, F#m=1 |
| **after corrector** | _bar_grid uses_ `claude_key=Am` | Am=21, G=17, E7=17, D7=9 |
| **audit oracle (`claude-opus-4-7`)** | **Bm** | canonical set = `[A, Bm, D, Em, F#, F#7, G]` |
| **corrector oracle (`claude-sonnet-4-5`)** | **Am** | claude_chord_set = `[Am, C, D7, E7, F, G]` |

Prod log line (May 9 13:00ish):
```
librosa detector: ~263 chord events, key=Bm, tempo=112.3 BPM
```

**Librosa was correct on key and largely correct on chord identity.** Its top 5 roots — Bm, A, F#, Em, G — match 5 of 7 canonical chords (`[A, Bm, D, Em, F#, F#7, G]`); only D and F#7 are missing, and F#7 is a 7th-quality refinement of F# that the 24-template detector cannot distinguish by construction. The detected key "Bm" is exactly the canonical key. **Librosa's raw F1 against the audit oracle's canonical set would be roughly 5/7 hit, ~0.71 — solid B-grade.** The post-corrector F1 is 0.15.

### Contrast with passing songs

For grounding, here's what well-aligned cases look like:

**Hey Joe (perfect 1.00):** librosa key=E (canonical E), raw histogram E=37/A=15/G=13/C=3/Bm=2/D=1, **only 2 of 71 bars replaced**. The corrector's Claude (sonnet-4-5) returned key=E with chord_set [A,C,D,E,G] — agrees with librosa. Same key + small chord-set delta = corrector barely intervenes.

**Take On Me (.91):** librosa key=F#m (canonical A — relative major of F#m), raw histogram F#m=42/E=37/D=22/Bm=21/Fm=10/G=7/A=5/Am=3, **20 of 147 bars replaced**. Corrector's claude returned key=A and chord_set [A,Bm,D,E,F#m] — recognized the relative-key relationship and kept most librosa names. The corrector rewrites Fm→F#m and Am→A (transposing the 10 false-flat hits up a semitone), drops C/G hallucinations, leaves the rest. **Librosa's key error was relative-minor-vs-relative-major, not random — and the corrector recognized that and rewrote correctly.**

So the contrast is clear: when librosa and the corrector's Claude land on the same (or relative) key, the corrector helps. When they disagree, the corrector destroys correct work.

---

## §2. What the corrector does to that output

Reading `chord_corrector_anthropic.py` end-to-end:

### The corrector's contract — what it CAN and CANNOT do

- **`drop` mode** (`:528-561`): only removes hallucinated chords (those not in Claude's canonical set). Refuses to act if drop ratio ≥ 70% (`:534-545`). Safe — strictly subtractive.
- **`replace` mode** (`:566-585`): rewrites chord names in `bar_grid` to the closest canonical chord (same root if possible, else previous-bar carry). Resets `chords_used` to `claude_chord_set` wholesale (`:574`).
- **`full` mode** (`:587-622`): does `replace`-mode bar rewriting AND additionally sends sections to a second Claude call (`_query_format_correction:201-274`) for section relabeling and per-line chord rewrites. Lyrics never sent.

**Prod is on `full` mode** (per `anthropic_correction.mode` in both failing JSONs, line 2999 of Sister Golden Hair, line 2020 of Misunderstood).

### The replace-mode mechanism (`_replace_in_bar_grid:277-331`)

```
For each bar:
  if bar.chord ∈ canonical_set:                       keep
  elif bar.root_letter ∈ canonical_by_root:           rewrite to canonical chord at same root
  else:                                                replace with previous-kept chord
  always: set chords_used = canonical_chord_list      (line 574 in replace, 598 in full)
```

This is structurally lossy in two ways:

1. **Same-root rewriting collapses quality information** — and the canonical_by_root map keeps only the FIRST canonical chord per root encountered in `canonical_chord_list` (`:299-301`). For Sister Golden Hair the corrector's claude returned `[A, Am, Bm, C, D, E, Em, G]`. The map gets `A→A, B→Bm, C→C, D→D, E→E, G→G` (Am gets shadowed by A since A came first; Em similarly). When librosa says "G#m" — a chord NOT in the canonical set — root letter "G" matches, so it's rewritten to `G`. But "G#m" is one of the actual canonical roots (G#m IS in the audit-oracle's chord set). The corrector silently strips the sharp.

2. **`chords_used` is fully replaced** by the corrector's claude_chord_set (`:574`, `:598`). So even if `bar_grid` retained correct chords, the audit's set-based F1 is computed against `chords_used`, which is now whatever the corrector's Claude returned. The audit can never see librosa's original answer.

### The structural failure on Sister Golden Hair

Librosa correctly outputs G#m (18 bars), C#m (7), F#m (9), B (24). Corrector's Claude returns `[A, Am, Bm, C, D, E, Em, G]` — **no G#m, no C#m, no F#m, no B**. So:

- 18 G#m bars → root "G" matches → rewritten to **G** (wrong: should stay G#m)
- 7 C#m bars → root "C" matches → rewritten to **C** (wrong: should stay C#m)
- 9 F#m bars → root "F" not in map → carry-previous (varies)
- 24 B bars → root "B" matches → rewritten to **Bm** (wrong: should stay B)

Final post-replace `chords_used` = `[A, Am, Bm, C, D, E, Em, G]` — exactly the corrector's `claude_chord_set`. Audit oracle's canonical = `[A, B, C#m, D, E, F#m, G#m]`. Set intersection = `{A, D, E}` — exactly the 3 "hits" in the audit. **The 4 misses (B, C#m, F#m, G#m) and 5 extras (Am, Bm, C, Em, G) are all directly explained by the corrector's chord_set being in the wrong key relative to the song.**

`bars_replaced=58/106`, `drop_ratio=0.67` (corrector log line in JSON line 3018). Notably **the drop ratio is BELOW the 0.70 safety gate** so the corrector proceeded; if it had been a hair higher it would have skipped (`:534-545`). The safety gate exists exactly for this case but the threshold is tuned wrong.

### The structural failure on Misunderstood

Same mechanism, more catastrophic:
- librosa raw `[Bm, A, F#, Em, G, Dm, Am, B, F#m]`
- corrector's claude_chord_set = `[Am, C, D7, E7, F, G]` (key=Am)
- audit oracle canonical = `[A, Bm, D, Em, F#, F#7, G]` (key=Bm)

The two LLMs disagree by a tritone-ish amount. The corrector's claude believes the song is in A minor; the audit's claude believes it's in B minor. They are scoring against **different ground truths**.

Post-corrector `chords_used` = `[Am, C, D7, E7, F, G]`. Audit canonical `[A, Bm, D, Em, F#, F#7, G]`. Intersection = `{G}` — exactly the 1 hit. **All 6 misses and 5 extras are explained by the two Claudes disagreeing about the song's key.**

`bars_replaced=53/64` (line 2033), `drop_ratio=0.78` (line 2037). **`0.78 ≥ 0.70` would block in `drop` mode (line 534-545), but THAT GATE IS NOT CHECKED in `replace`-mode or `full`-mode** (the gate code at `:534-545` is only inside the `if use_mode == "drop":` branch). So full-mode happily replaced 78% of bars based on a chord set the corrector itself flagged with a high drop ratio.

### The format-correction layer (full mode only)

`_query_format_correction` sends a SECOND Claude call (`:201-274`) with section names + per-line `chords` strings + lyrics (lyrics for context only — NOT modifiable by hard rule at `_FORMAT_SYSTEM_PROMPT:170-174`). Claude returns chord_overrides per line; the apply step (`_apply_format_corrections:354-454`) rewrites both the line `chords` field and the per-segment chord names.

For Sister Golden Hair this rewrote 27 chord lines and moved 17 lines (line 3015-3017). For Misunderstood: 11 chord lines, 14 lines moved (line 2034-2036). These edits compound the wrong-key replacement in `bar_grid` — both surfaces (the bar_grid chart view and the section/lyric tab view) end up in the corrector-Claude's wrong key.

Importantly: the `sections` view in the JSON appears to retain librosa's original chord names in some places (e.g. Sister Golden Hair sections still show G#m, C#m, F#m, B in many lines — see the file at JSON lines 30-100). This is because `_apply_format_corrections` only rewrites lines that appear in Claude's `chord_overrides` map — Claude can return overrides for some lines and not others. So the artifact has TWO inconsistent chord views: `bar_grid` (in corrector-Claude's wrong key) and `sections` (mixed librosa + corrector-Claude). The frontend renders sections; the audit scores `chords_used` (the wrong-key set). So the audit grade is far worse than what a user sees on screen — but a user still sees a chimera.

---

## §3. Specific failure mechanism (one paragraph)

**Librosa is not the problem.** Librosa correctly identifies the key (Bm for Misunderstood, plausibly-E-via-B for Sister Golden Hair) and detects 5–6 of 7 canonical chord roots from the chromagram on both songs. The chord_corrector_anthropic `full` mode then calls Claude Sonnet 4.5 (`_query_canonical_chords:74-126`) which returns a `claude_chord_set` and `claude_key` for the song. On these 2 songs Sonnet 4.5 returns a chord_set in the **wrong key relative to the song the audit oracle (Claude Opus 4.7 in `llm_oracle.py`) believes is canonical** — for Sister Golden Hair, Sonnet says key=G with set `[A, Am, Bm, C, D, E, Em, G]` while Opus says key=E with set `[A, B, C#m, D, E, F#m, G#m]`; for Misunderstood, Sonnet says key=Am with set `[Am, C, D7, E7, F, G]` while Opus says key=Bm with set `[A, Bm, D, Em, F#, F#7, G]`. The corrector's `_replace_in_bar_grid` then forcibly rewrites every librosa chord to the nearest same-root canonical chord, overwrites `chords_used` to Sonnet's set wholesale (`:574`/`:598`), and the audit scores Sonnet's set against Opus's set. The 0.40 and 0.15 F1 scores are not measuring detector quality — they are measuring **the disagreement between two LLM-as-oracle models on what the canonical chord set is**. The `drop_ratio ≥ 0.70` safety gate (`:534-545`) that exists exactly to prevent this only fires in `drop` mode, not `full` mode where it's needed most — and Misunderstood's drop_ratio of 0.78 should have triggered a bailout but didn't.

---

## §4. Ranked fix recommendations

Each ranked by (impact on these 2 songs) × (regression risk on the other 16) ÷ (effort).

### Fix 1 — Extend the `drop_ratio ≥ 0.70` safety gate to replace + full modes

**Hypothesis:** The safety gate at `chord_corrector_anthropic.py:534-545` was designed exactly for the wrong-key case ("`{drop_ratio:.0%}` flagged on `{title}` — likely wrong-key bug"). It was correctly placed in `drop` mode but never extended when `replace` and `full` modes were added. Misunderstood's drop_ratio is 0.78 (line 2037 of its JSON) — would have bailed out and left librosa's output untouched, scoring ~0.71 instead of 0.15. Sister Golden Hair's drop_ratio is 0.67 — just below the threshold, so this alone wouldn't help that song; but lowering the threshold to e.g. 0.60 catches both with very low risk because at that drop ratio the corrector's Claude almost certainly has the wrong key anyway.

**Effort:** 30 minutes — move the gate check to before the mode switch, lower threshold to 0.60.

**Expected impact:** Misunderstood F1 0.15 → ~0.71 (librosa raw). Sister Golden Hair F1 0.40 → ~0.86 (librosa raw, 6 of 7 canonical chords). Total +52 points across the 2 songs.

**Regression risk on the other 16:** LOW. Hey Joe drop ratio is ~0 (only 2 of 71 bars replaced). Take On Me drop ratio is 0.38. None of the passing songs are near 0.60. Worst case a song that was bailed out simply gets librosa-raw output, which is the May 5 baseline — known performance, not a new failure mode.

**Validation:** $1.02 audit re-run.

### Fix 2 — Pin both LLM oracles to the same model

**Hypothesis:** Half the failure on these 2 songs is "the corrector and the auditor disagree on the canonical chord set." That's a methodology bug, not a detector bug. The audit uses `claude-opus-4-7`; the corrector uses `claude-sonnet-4-5-20250929`. Pinning both to opus-4-7 (or both to sonnet-4-5) eliminates the within-Anthropic disagreement.

**Effort:** 5 minutes — change `_DEFAULT_MODEL` at `chord_corrector_anthropic.py:52` from `claude-sonnet-4-5-20250929` to `claude-opus-4-7`. Also bump `ANTHROPIC_CORRECTION_MODEL` env on prod.

**Expected impact:** Material on Sister Golden Hair (Opus likely returns the canonical [A,B,C#m,D,E,F#m,G#m] set since it returned that to the audit). Less certain on Misunderstood — Opus returned key=Bm in audit with chord_set `[A, Bm, D, Em, F#, F#7, G]`; if Opus also returns that as correction set, librosa's [Bm, A, F#, Em, G] survives same-root rewriting cleanly → F1 ~0.71+. Could easily lift the 2 failing songs to 0.7–0.85.

**Regression risk on the other 16:** LOW-MEDIUM. Opus is more conservative and may return `found: false` more often, in which case the corrector skips and you get librosa-raw — still the May 5 baseline. Cost increases ~5–10× per call but it's still <$0.10/song.

**Operational consideration:** Opus is slower (~1–3 sec extra per song) — adds to the post-separation semaphore time per `docs/scaling-pipeline-2026-05-09.md`. Acceptable.

**Validation:** $1.02 audit re-run + cost check on a 50-song sample.

### Fix 3 — Add a "key-disagreement detector" to the corrector and bail out on disagreement

**Hypothesis:** When librosa's detected key (logged at `chord_detector_librosa.py:158-161`) and the corrector-Claude's `claude_key` are NEITHER the same NOR a relative-major/minor pair, the corrector should not run replace mode. This is a content check: librosa picks key from the actual chromagram; if Claude's key matches the chromagram-implied key (modulo enharmonic spelling and relative major/minor), the corrector is on solid ground; if not, librosa is more reliable than Claude on "what notes are actually in this audio."

**Effort:** 1–2 hours. Add a `_keys_compatible(librosa_key, claude_key)` helper that returns True if they're equal, enharmonic, or relative-major-minor (e.g. F#m ↔ A, Bm ↔ D). Add a check in `apply_correction` after `_query_canonical_chords`: if not compatible AND drop_ratio > 0.50, set status to "skipped_key_disagreement" and return chord_chart unchanged.

**Expected impact:** Misunderstood — librosa says Bm, Claude-corrector says Am. Bm and Am are NOT relative — bails out, F1 0.15 → ~0.71. Sister Golden Hair — librosa says B, Claude-corrector says G. Not relative either — bails out, F1 0.40 → ~0.86. Take On Me — librosa F#m, Claude-corrector A. F#m ↔ A IS relative-minor-major — corrector runs as today, F1 stays 0.91.

**Regression risk on the other 16:** LOW. The "compatible-key" relation is permissive; only catches the catastrophic disagreements.

**Validation:** $1.02 audit + manual check that the 18 songs each match correctly under `_keys_compatible`. Combine with Fix 1 for belt + suspenders.

### Fix 4 — Stop overwriting `chords_used` wholesale in replace/full modes

**Hypothesis:** `chord_corrector_anthropic.py:574` and `:598` reset `chord_chart["chords_used"] = list(canonical_chord_list)` — discarding the actually-detected chords entirely. Even when bar replacement is reasonable, this overwrite means the audit (which scores `chords_used` set-comparison) only ever sees the corrector-Claude's set, never librosa's. Replace these lines with `list({c for c in chords_used if normalize_chord(c) in canonical_set} | set(canonical_chord_list))` — the union of librosa-detected chords that survived corrector validation plus any canonical chords. This way the audit sees both signals.

**Effort:** 30 minutes.

**Expected impact:** Modest on these 2 songs (the wrong-set is still mixed in). But makes future audits more honest — the `chords_used` field actually reflects what the system believes is in the song, not just what one Claude call asserted. Pairs well with Fix 1 (when the gate fires, you don't lose the chords_used signal).

**Regression risk on the other 16:** Could hurt precision on songs where librosa hallucinated and the corrector correctly dropped — those hallucinations would now show up in the audit. Likely 2–5 point F1 drop on the strong songs. Net might be neutral.

### Fix 5 — Use librosa's detected key as a hint to the corrector's Claude

**Hypothesis:** The corrector's Claude has no information about the actual audio — it only sees the song title + artist. Adding "the audio chromagram suggests key={librosa_key}" to the user prompt (`_query_canonical_chords:88-91`) lets Claude resolve title-disambiguation cases (live versions, alternate keys, capo'd recordings that don't match the published chart's key). For Sister Golden Hair: Claude was told "Sister Golden Hair by America" with no key hint; America actually has multiple recordings in different keys and Claude landed on G (the published-chart key, capo IV). If we'd told Claude "audio is in B," Claude could have transposed mentally and returned [B, C#m, F#m, A, E, G#m, D] — the actual song-as-recorded key.

**Effort:** 1 hour. Thread librosa's key out through the corrector entry point (already in `chord_chart["detected_key"]` — confirm). Append "Audio analysis suggests the recording is in key {key}; if your canonical chord set is in a different key, transpose to match the audio key." to the user message.

**Expected impact:** Likely fixes Sister Golden Hair (Claude's Sonnet returns capo'd key by default). Less likely on Misunderstood (the discrepancy is Bm vs Am, a third — unusual transposition). Probably +30 points on Sister Golden Hair, neutral on Misunderstood.

**Regression risk on the other 16:** LOW-MEDIUM. Some songs librosa gets the key wrong (e.g. relative-minor confusion); telling Claude an incorrect key could flip a currently-passing song into the wrong key. Need to validate against the full audit.

---

### Recommended sequencing

1. **Fix 1** (extend safety gate to all modes) + **Fix 3** (key-disagreement bailout) — both are < 2 hr combined, low risk, high impact. **Do these first.** Ship as a single cherry — if drop_ratio > 0.60 OR key disagreement, leave librosa untouched.
2. **Fix 2** (pin model) — 5 min change but uncertain magnitude. Run after #1 to see what marginal value pinning has.
3. **Fix 5** (key hint) — only if 1+3 don't get Sister Golden Hair to A-grade.
4. **Fix 4** (chords_used union) — only if you want the audit signal to be more faithful. Lower priority.

Skip the May 6 plan's Top-3 (Chordino, PC-count gating, m3-vs-M3 energy). The May 9 evidence is that **the detector is fine on these 2 songs**. The fix lives in the corrector's coupling logic, not in any new detector engineering.

---

## §5. Honest verdict

**Fixable in <1 week.** Confidently. This is not a structural detector limitation. The May 6 plan's "wrong-key cascade" diagnosis was for the OLD stem-aware detector (where the cascade truly was structural — minor-template exact match → polarity-gate lockout → maj7 promotion blocked). That cascade was eliminated when librosa replaced the stem-aware detector. The librosa detector has different failure modes (24 templates means no quality information beyond major/minor; key picked by global PC histogram K-K so multi-tonal songs can land on a non-tonic) — but on these 2 songs librosa is mostly right.

The remaining 0.15 and 0.40 F1 numbers are 90% explained by **a corrector that overwrites correct librosa output with the wrong-key chord_set returned by Claude Sonnet 4.5**, with the safety gate that exists for exactly this purpose left disabled in the most-aggressive mode. Two-line code change to the `apply_correction` function fixes the worst case; pinning the model and adding a key-disagreement bailout fixes the rest.

**Confidence in this diagnosis is high** because the evidence is fully reconstructable from the artifacts — every bar's pre-corrector chord is in `bar_grid[i].source_meta.replaced_from`, librosa's logged key matches what `replaced_from` implies, and the corrector's output is verbatim the corrector-Claude's `chord_set`. There's no opacity in the pipeline; the failure is mechanical.

**Confidence in the fix** is medium-high. Fix 1 is mechanically certain (the gate exists; we're moving it). Fix 2 and Fix 3 are content-dependent — Opus may or may not return better sets, and the relative-key compatibility check could miss edge cases. But the lower bound of "do nothing in the wrong-key case and ship librosa-raw" is the May 5 baseline, which on these 2 songs would be ~0.71 and ~0.86 — both A-grade. **There is no realistic scenario where these 2 songs stay at 0.15 / 0.40 after a 2-hour corrector fix.**

The only thing structural here is the 24-template detector's inability to distinguish major-7 from minor-7 (relevant to F#7 in Misunderstood) — that's a 1-point recall ceiling, irrelevant compared to the 50-point loss to the corrector.

**Bottom line:** spend the half-day, ship the fixes, expect both songs to land in the A/B+ band. Then turn back to the launch sprint.
