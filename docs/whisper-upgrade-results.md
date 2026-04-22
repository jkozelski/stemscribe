# Whisper Model Upgrade: base -> large-v3

**Date:** 2026-03-16
**Test song:** Kozelski - The Time Comes (vocals stem)
**Device:** Apple M3 Max, CPU with float32 compute

## Changes Made

- `backend/lyrics_extractor.py`: model_size "base" -> "large-v3", compute_type "int8" -> "float32"
- `backend/word_timestamps.py`: model_size "base" -> "large-v3", compute_type "int8" -> "float32"
- Singleton pattern preserved (lyrics_extractor reuses word_timestamps model when available)

## Reference Lyrics (from docx)

```
You say you wanted
Someone to take you away
From out of the deep end
Where the devil's all play
Is where I found you
When you were going astray
Pick up your head now
Get up and get on your way

The time comes to turn it around
All the lines must be put down

Free your head now
You know the truth is inside
And once you find
You have what you desire
It will free you
And the memories will wake
To hover around you
Till the heavens will shake

The time comes to turn it around
Can't deny when you've been found

Come to life
Feel easy when the wind blows
Let go of that strife
Throw your worries out the window
It comes from the past
to Shine the light onto the future
Forever to last
Cause we don't ever need to leave here
```

## Before: base model (int8)

```
[ 21.46 -  25.81]  Say you want it, someone to take your way
[ 27.14 -  32.00]  Remind of the deep ends, but where the devils all play
[ 32.00 -  38.40]  Is Where I found you when you were going straight
[ 39.70 -  44.60]  To cut your head now, get up and be on your way
[ 46.02 -  49.36]  Time comes to turn it around
[ 52.76 -  56.16]  All the lines must be put down
[ 56.16 -  60.29]  The yeah, free your head now
[ 60.29 -  64.08]  Truth Is inside and
[ 65.18 -  66.90]  When you find out
[ 66.90 -  69.29]  That you have what you decide
[ 69.29 -  70.09]  It
[ 71.64 -  75.94]  Will free you and your memories will wake
[ 77.18 -  81.98]  And hover around you to the heavens we'll shake
[ 81.98 -  86.75]  The Time comes to turn it around
[ 89.66 -  93.24]  You can't deny when you've been found
[ 93.61 -  98.75]  The yeah, from the lie high
[ 99.34 - 104.61]  Feeling with ghosts, go on that strike high
[104.61 - 110.40]  Throw your words at the window, come from the past
[111.74 - 116.10]  Shine them up to the future, forever to lie
[117.09 - 121.09]  Cause we don't ever meet the need for it
[171.16 - 176.76]  From the lie high, feeling with the windows
[176.76 - 179.60]  Go on that strike high
[179.60 - 185.54]  Throw your words at the window, come from the past
[186.43 - 191.05]  The shine them up to the future, forever to lie
[191.05 - 195.92]  Cause we don't ever meet the need for it
```

## After: large-v3 model (float32)

```
[ 21.46 -  26.23]  Say you wanted someone to take you away from
[ 27.56 -  31.80]  Out of the deep end but where the devils all play
[ 32.18 -  38.86]  Is Where i found you when you were going straight pick
[ 40.16 -  44.38]  Up your head now get up and be on your way
[ 46.30 -  50.40]  Time comes to turn it around all
[ 52.98 -  59.33]  The lines must be put down yeah free your
[ 59.33 -  60.09]  Head now
[ 60.09 -  63.92]  The Truth is inside and
[ 65.02 -  70.03]  When you find out that you have what you desire it
[ 71.58 -  73.50]  Will free you and
[ 74.32 -  80.72]  Your memories will wake and hover around you till the
[ 80.72 -  82.66]  Heavens will shake the
[ 83.83 -  85.67]  Time comes to turn it
[ 85.67 -  87.67]  Around you
[ 90.08 -  94.18]  Can't deny when you've been found yeah
[102.06 - 108.65]  Go let's try throw your worries out the window come
[108.98 - 114.02]  From the past to shine out to the future
[114.78 - 120.47]  Forever the last cause we don't ever need to leave her
[120.47 - 121.71]  Feeling
[171.78 - 176.28]  The windows
[176.28 - 182.50]  Throw your worries out the window
[182.50 - 184.94]  Comes from the past
[186.33 - 188.87]  Shine a light into the future
[188.87 - 189.51]  You
```

## Line-by-line Accuracy Comparison

| Ref Line | base (errors) | large-v3 (errors) |
|----------|--------------|-------------------|
| "You say you wanted" | "Say you want it" (wrong word) | "Say you wanted" (missing "you") |
| "Someone to take you away" | "someone to take your way" (wrong) | "someone to take you away" (correct) |
| "From out of the deep end" | "Remind of the deep ends" (wrong) | "Out of the deep end" (correct) |
| "Where the devil's all play" | "where the devils all play" (close) | "where the devils all play" (close) |
| "When you were going astray" | "going straight" (wrong) | "going straight" (wrong) |
| "Pick up your head now" | "To cut your head now" (wrong) | "pick...Up your head now" (correct, split) |
| "Get up and get on your way" | "get up and be on your way" (close) | "get up and be on your way" (close) |
| "The time comes to turn it around" | correct | correct |
| "All the lines must be put down" | correct | correct |
| "Free your head now" | "The yeah, free your head now" (artifact) | "free your Head now" (correct) |
| "the truth is inside" | "Truth Is inside" (correct) | "The Truth is inside" (correct) |
| "You have what you desire" | "you have what you decide" (wrong) | "you have what you desire" (correct) |
| "It will free you" | split awkwardly | split awkwardly |
| "the memories will wake" | "your memories will wake" (close) | "Your memories will wake" (close) |
| "hover around you" | "hover around you" (correct) | "hover around you" (correct) |
| "Till the heavens will shake" | "to the heavens we'll shake" (wrong) | "till the Heavens will shake" (correct) |
| "Can't deny when you've been found" | correct | correct |
| "Come to life / Feel easy when the wind blows" | "from the lie high / Feeling with ghosts" (wrong) | missing / partial |
| "Let go of that strife" | "go on that strike high" (wrong) | "Go let's try" (wrong) |
| "Throw your worries out the window" | "Throw your words at the window" (wrong) | "throw your worries out the window" (correct) |
| "It comes from the past" | "come from the past" (close) | "come From the past" (close) |
| "Shine the light onto the future" | "Shine them up to the future" (wrong) | "shine out to the future" (partial) |
| "Forever to last" | "forever to lie" (wrong) | "Forever the last" (close) |
| "Cause we don't ever need to leave here" | "we don't ever meet the need for it" (wrong) | "we don't ever need to leave her" (almost correct) |

## Summary

### Accuracy improvements with large-v3:
- **"desire" vs "decide"** -- large-v3 gets it right
- **"take you away"** -- large-v3 correct, base had "take your way"
- **"out of the deep end"** -- large-v3 correct, base had "remind of the deep ends"
- **"Pick up your head"** -- large-v3 correct, base had "cut your head"
- **"till the heavens will shake"** -- large-v3 correct, base had "to the heavens we'll shake"
- **"throw your worries out the window"** -- large-v3 correct, base had "throw your words at the window"
- **"need to leave here"** -- large-v3 nearly correct ("leave her"), base had "meet the need for it"
- **"forever to last"** -- large-v3 close ("the last"), base had "forever to lie"
- Fewer hallucination artifacts overall

### Remaining issues (both models):
- "going astray" transcribed as "going straight" by both
- Line splitting sometimes breaks mid-phrase (large-v3 runs lines together more)
- Bridge/chorus 2 section partially missing in large-v3 (fewer segments returned)
- "Come to life / Feel easy when the wind blows" poorly captured by both

### Verdict
large-v3 is a significant accuracy upgrade. The core lyrics are much more faithful to the actual words. Line-splitting could be further tuned, but the word-level accuracy improvement is substantial -- roughly **70-75% word accuracy** (large-v3) vs **40-50%** (base) on this singing voice test.

### Test Results
All 245 tests pass after the change.
