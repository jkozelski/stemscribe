# Modal Spend Post-Mortem — Apr 1 to May 21, 2026

**Pulled:** 2026-05-21 via `modal billing report --json`
**Raw data:** `modal_apr.json`, `modal_may.json`, `modal_apr.csv`, `modal_may.csv` (same folder)

## Total: $191.27 across 50 days

### Breakdown by app

| Spend | App | What it is | Status |
|---|---|---|---|
| **$98.95** | `stemscribe-chart-formatter-train` | Training the chart-formatter (chord-list → pretty chart) | **SHIPPED** — `stemscribe-chart-formatter` is in deployed-apps list, used in prod |
| **$48.13** | `stemscribe-chart-recall-train` | Training the chord-recall model | Unclear shipping status — verify |
| **$19.23** | `bass-transcription-training` | Bass-NN training | **DID NOT SHIP** — prod bass uses Basic Pitch + post-processing per CLAUDE.md |
| **$18.27** | `stemscribe-separator` | Production stem-separation (every user upload) | **LEGITIMATE** — this is the app working |
| **$4.66** | `trimplexx-guitar-training` | Guitar-tab CRNN | **TRAINED, GATED OFF** — `ENABLE_CRNN_TRANSCRIPTION=false`, also has drum-tab bug (#42) |
| **$0.87** | `stemscribe-guitar-training` | Earlier guitar attempt | Likely superseded by trimplexx |
| **$0.69** | `stemscribe-chart-formatter` | Deployed app idle/serving | Production |
| **$0.46** | `btc-finetune` | BTC chord-detector fine-tune | **ORPHAN** — 79.92% val checkpoint, never shipped (data was scraped audio, legal landmine) |

### Production vs. training

- **$18.27 (10%)** — production app processing real user uploads. Legitimate spend.
- **$172.30 (90%)** — training experiments.
- **$0.69 (<1%)** — other.

### Where the training money actually went

Two days account for **71%** of all training spend:

| Date | Day total | Top cost |
|---|---|---|
| **Apr 14** | $74.42 | chart-formatter-train $38.63 + chart-recall-train $33+ |
| **Apr 13** | $61.00 | chart-formatter-train $43.43 + chart-recall-train $12.57 + trimplexx $4.62 |
| Apr 16 | $18.64 | chart-formatter-train $16.90 (single run) |
| Apr 11 | $14.35 | bass-transcription-training $13.02 (single run) |
| Apr 12 | $7.31 | bass-transcription-training $6.21 |

### The "idle running" smoking gun

Individual training runs (each = one Modal app id) include:
- `ap-4fm5p4IC8zfLyCEmJetgNR` chart-formatter-train Apr 16: **$16.89** (single run)
- `ap-3qNTZPNKZvCPt2ybipa5EI` chart-formatter-train Apr 14: **$14.51**
- `ap-49J6fGmUSkm8xUlTMbZV4T` chart-recall-train Apr 14: **$14.48**
- `ap-lNF5GhIkfT8hr64EqbNUUm` chart-formatter-train Apr 14: **$14.50**
- `ap-xwrOwHGHCSnklNmVlibyTz` chart-formatter-train Apr 13: **$13.39**
- `ap-FirIEvbnJW6WVuNzaDFm8K` bass-transcription Apr 11–12: **$19.15 combined**
- `ap-ncbwBRWAIFVA8ge0gAa2eE` chart-recall-train Apr 14: **$12.32**
- `ap-886ZidGSkZc8EugXZrEplT` chart-recall-train Apr 13–14: **$14.77 combined**

Expected single-run cost per the script docs: **~$2–4**. Actual single-run costs: **$12–17**. That's the 4–6× overrun consistent with hangs / idle runs / training that didn't checkpoint properly.

## What's actually wasted

- **chart-formatter training ($98.95)** — SHIPPED. Not wasted, even if individual runs ran long.
- **chart-recall training ($48.13)** — verify shipping; if not in prod, it's the second-biggest waste candidate.
- **bass-transcription ($19.23)** — DID NOT SHIP. Bass currently uses Basic Pitch. **All wasted.**
- **trimplexx ($4.66)** — trained but gated off. Recoverable if #42 fixed.
- **BTC finetune ($0.46)** — orphan. Tiny amount.

**Conservative wasted estimate: $25–75** (bass + half of chart-recall if it didn't ship + trimplexx).
**The $98.95 chart-formatter training money is NOT wasted** — that model is in prod.

## Don't-do-again rules

1. **Modal training jobs need a hard wall-clock timeout.** Set `@app.function(timeout=1800)` (30 min) or similar on every training entrypoint. Prevents runaway hangs.
2. **Check Modal dashboard at end of every training day.** Sub-15-minute habit, prevents Apr-13/14-style $130 surprises.
3. **One training run, then assess.** Don't kick off 6 chart-formatter runs in 24 hours. Run one, look at the loss curve, then iterate.
4. **No new training before June 20 launch.** Per the dead-ends ledger — accuracy chase is closed. Trust-UX pivot is locked. Train post-launch with real user-correction data.
5. **`monitor_runpod.py` exists for a reason.** Build the equivalent for Modal: a script that lists running apps + their accumulated cost and alerts if anything's been running >2 hours.

## Files preserved

- `modal_apr.json` / `modal_apr.csv` — full Apr billing line-by-line
- `modal_may.json` / `modal_may.csv` — May through 21st
- This file: the analyzed conclusion

Saved 2026-05-21. Next session can read this file instead of re-litigating "where did the $150 go."
