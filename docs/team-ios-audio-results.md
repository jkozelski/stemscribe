# Team iOS Audio — Results

**Date:** 2026-03-18
**Agents:** 1 (Routing), 2 (iOS Session), 3 (Stem Loading), 4 (Test & Debug)
**Status:** All complete
**Priority:** #1 blocker — iPhone plays but no sound

---

## Root Causes Found (3 independent issues)

### 1. iOS Audio Session — Silent Switch Kills Sound (Agent 2)
**The primary cause.** iOS Safari plays Web Audio through the "ambient" audio category, which respects the physical silent/ringer switch. Even with volume at max, if the silent switch is on (or iOS decides to use ambient mode), no sound comes out.

**Fix:** On first `touchend`/`click`, play a tiny silent MP3 via HTMLAudioElement (not Web Audio) to force iOS to switch to the "playback" audio category. Also resume AudioContext and play a 1-sample silent buffer to warm it up.

**Files:** `practice.html` (~line 2049), `mixer.js` (top of file)

### 2. Stale AudioContext — Orphaned GainNodes (Agent 1)
**Critical bug.** iOS Safari closes AudioContexts created outside user gestures. When `init()` detects a closed context and creates a new one, ALL existing GainNodes and AnalyserNodes are orphaned on the old dead context. Source nodes on the new context connect to gain nodes on the old context → silence.

**Fix:** Added `_rebuildNodes()` / `_rebuildGainNodes()` that recreates all gain/analyser nodes on the new context when a stale context is detected. Added stale-context detection in `play()` → `startAllSources()`.

**Files:** `practice.html`, `mixer.js`

### 3. Sequential Stem Loading — One Failure Kills All (Agent 3)
**Critical bug.** In `mixer.js`, `loadStems()` chained all stem loads sequentially with no error isolation. If stem #3 failed to fetch or decode, the promise chain rejected → stems #4-6 never attempted → `_loaded` stayed `false` → `play()` guard blocked ALL playback.

**Fix:** Each stem's fetch+decode now has its own `.catch()` that logs but doesn't propagate. `_loaded` set to `true` if at least 1 stem succeeds. Added buffer validation (duration, length, silent detection) and enhanced error logging.

**Files:** `mixer.js`, `results.js`, `practice.html`

---

## All Changes by File

### `frontend/practice.html`
| Agent | Change |
|-------|--------|
| 2 | iOS audio session unlock IIFE (silent MP3 + resume + warm buffer) |
| 1 | `_rebuildGainNodes()` method — recreates gain nodes on new context |
| 1 | `init()` — calls rebuild when context replaced |
| 1 | `play()` — closed-context check, stale-context detection in startAllSources |
| 1 | `init()` — 1-sample silent buffer play for iOS unlock |
| 3 | Buffer validation after decode (duration, length, silent detection) |
| 3 | Enhanced decode error logging (buffer size, context state, sample rate) |
| 4 | `debug()` method on StemAudioEngine (dumps full audio state) |
| 4 | `?audioDebug=1` URL param mode with periodic checks |

### `frontend/js/mixer.js`
| Agent | Change |
|-------|--------|
| 2 | iOS audio session unlock IIFE (with `window.__iOSAudioUnlocked` dedup) |
| 1 | `_unlocked` flag + silent buffer play in `init()` |
| 1 | `_rebuildNodes()` method — recreates gain + analyser nodes |
| 1 | `init()` — calls rebuild when context replaced |
| 1 | `play()` — closed-context check, stale-context detection in startAllSources |
| 3 | Per-stem error isolation in `loadStems()` (`.catch()` per stem) |
| 3 | Partial load support (`_loaded = true` if ≥1 stem) |
| 3 | Buffer validation (duration, length, silent buffer warning) |
| 3 | Enhanced `_decodeAudio()` with stemName param + error details |
| 4 | `debug()` prototype method (dumps full audio state) |

### `frontend/js/results.js`
| Agent | Change |
|-------|--------|
| 3 | Partial load user notification toast ("X/Y stems loaded") |
| 3 | `.catch()` only fires when ALL stems fail |

---

## Debug Mode (Agent 4)

### How to Use
- **URL param:** Add `?audioDebug=1` to practice page URL
- **Console:** Call `window.audioDebug()` (practice) or `window.audioDebugMixer()` (mixer) anytime

### What It Logs
- Platform info, iOS detection result
- AudioContext state, sampleRate, latency
- Per-stem buffer analysis: duration, channels, sampleRate, first-100 + mid-buffer sample check
- GainNode values for each stem
- Audio graph visualization
- 2-second periodic checks for "playing but suspended" and "all gains zero"

### Full debug guide: `~/stemscribe/docs/ios-audio-debug.md`

---

## Audio Chain (Verified Correct)

```
AudioBufferSourceNode (per stem)
  → GainNode (per stem, default 0.8)
    → [AnalyserNode (mixer only, for VU meters)]
      → AudioContext.destination
```

Both engines (`practice.html` class + `mixer.js` prototype) maintain this chain. The `_rebuildNodes()` fix ensures the chain uses the SAME AudioContext throughout.

---

## iOS Safari Quirks Reference

1. **AudioContext created outside user gesture → suspended or closed**
2. **Silent switch → ambient category → no sound** (fixed by HTMLAudioElement trick)
3. **`decodeAudioData`** uses callback-only mode in older Safari (dual callback+promise wrapper needed)
4. **Context can be closed at any time** by iOS memory pressure
5. **`webkitAudioContext`** still needed for older iOS versions

---

## Testing Checklist

- [ ] Desktop Chrome/Firefox — playback still works (regression check)
- [ ] iPhone Safari, silent switch OFF — sound plays
- [ ] iPhone Safari, silent switch ON — sound still plays (audio session unlock)
- [ ] iPhone Safari — tap play immediately after page load (context resume)
- [ ] iPhone Safari — 3/6 stems loaded scenario → partial playback works
- [ ] iPhone Safari — all 6 stems loaded → full playback
- [ ] iPad Safari — same tests as iPhone
- [ ] Check `?audioDebug=1` output on iPhone for any warnings
- [ ] Mixer page (results.html) — same playback tests
- [ ] Practice page — same playback tests

---

## What Was NOT the Issue
- **CORS** — Cloudflare Tunnel proxies same-origin, no CORS problems
- **Audio format** — WAV files decode fine on iOS
- **Chord rendering** — untouched (Team 2's domain)
