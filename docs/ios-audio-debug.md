# iOS Audio Debug Guide

## Summary of iOS Audio Issues

### Root Problem
iPhone Safari: user taps play, the timer advances and the button shows pause, but **no sound** comes out. The UI suggests playback is happening but the AudioContext is either suspended or gain nodes are zeroed.

### iOS Safari Web Audio Constraints
1. **AudioContext starts suspended.** iOS Safari creates every `AudioContext` in the `suspended` state. It will only transition to `running` when `context.resume()` is called **inside a user gesture call stack** (tap, click).
2. **One-shot resume.** If `resume()` is called outside a gesture (e.g., in a `setTimeout`, `fetch().then()`, or `DOMContentLoaded`), it silently fails or stays suspended.
3. **`webkitAudioContext` vs `AudioContext`.** Older iOS versions only expose `webkitAudioContext`. Both engines use `window.AudioContext || window.webkitAudioContext`.
4. **`interrupted` state.** Safari has a non-standard `interrupted` state (phone call, Siri, switching apps). The context must be resumed again after this.
5. **`decodeAudioData` callback form.** Safari sometimes only supports the callback form of `decodeAudioData`, not the promise form. Both engines handle this with a dual callback+promise wrapper.
6. **Silent audio buffers.** If stems were decoded while the context was in a bad state, the buffers could theoretically be empty. Debug mode checks for this.

### What Each Agent Fixed

| Agent | Focus | Changes |
|-------|-------|---------|
| Agent 1 | Audio routing | Ensures gain nodes connect to the correct `ctx.destination` after context rebuild. Adds `_rebuildNodes()` in mixer.js to reconnect the graph if the context was closed and recreated. |
| Agent 2 | iOS session unlock | Ensures `init()` (which calls `resume()`) is called in the direct user tap call stack in `togglePlayback()`. Handles the `interrupted` state with auto-resume on `statechange`. Mixer engine plays a silent buffer on first init to fully unlock iOS audio output. |
| Agent 3 | Stem loading | Separates `_ensureContext()` (creates context for decoding, no resume) from `init()` (creates + resumes). Ensures stems can be decoded while context is suspended. |
| Agent 4 | Debug tooling | Added `debug()` method and `?audioDebug=1` mode to both engines (practice.html and mixer.js). No functional code changes. |

## How to Use Debug Mode

### Method 1: URL Parameter
Append `?audioDebug=1` to the page URL:
```
https://stemscribe.io/practice.html?job=abc123&audioDebug=1
```
This enables:
- Verbose logging on every `init()`, `play()`, and `loadStem()` call
- Automatic full debug dump 500ms after each `play()`
- Capture-phase logging on play button clicks
- Periodic (2s) checks for "playing but suspended" or "all gains zero" conditions

### Method 2: Console Commands
Available on any page load (no URL param needed):

**Practice page:**
```js
window.audioDebug()
```

**Mixer page (index.html):**
```js
window.audioDebugMixer()
```

Both return a formatted string and log it to console.

### Debug Output Sections

| Section | What It Shows |
|---------|---------------|
| **Platform** | userAgent, iOS detection, AudioContext/webkitAudioContext availability |
| **AudioContext** | state, sampleRate, currentTime, baseLatency, outputLatency, destination channel info |
| **Engine State** | isPlaying, offset, startTime, playbackRate, duration |
| **Stems** | Per-stem: loaded, volume, muted, gain value, source node status, buffer stats (duration, channels, sampleRate, sample data check) |
| **Audio Graph** | Visual chain: BufferSource -> GainNode(value) -> [Analyser ->] destination |
| **Resume Test** | If suspended, attempts resume and logs result |

### Key Things to Look For

1. **`ctx.state: suspended`** while `isPlaying: true` -- the context never resumed, audio is silent
2. **`gain.gain.value: 0.0000`** on all stems -- volumes are zeroed (mute/solo logic bug)
3. **`first100samples.hasNonZero: false`** -- the decoded buffer is empty/silent
4. **`source: null`** while `isPlaying: true` -- source nodes were never created
5. **`ctx: NULL`** -- AudioContext was never created
6. **`buffer: NULL`** on a stem -- fetch or decode failed silently

## Known iOS Safari Quirks

### AudioContext Lifecycle
```
Created (outside gesture) -> state: "suspended"
resume() in gesture       -> state: "running"     -> AUDIO WORKS
resume() outside gesture  -> state: "suspended"   -> SILENT
Phone call / Siri         -> state: "interrupted"
statechange event fires   -> must resume() again
App backgrounded          -> may become "suspended" or "interrupted"
```

### Common Failure Modes

1. **Context created in `DOMContentLoaded`** -- will be suspended, `resume()` there does nothing
2. **`resume()` in a `.then()` chain** -- loses the user gesture call stack
3. **Creating a new context per play** -- wastes resources and each one starts suspended
4. **Not handling `interrupted`** -- after a phone call, audio stays dead

### StemScriber's Approach
- Context is created lazily (in `_ensureContext()` for decoding, in `init()` for playback)
- `init()` is called directly in the `togglePlayback()` function, which is in the user tap call stack
- `play()` checks for suspended state and calls `resume().then(startAllSources)`
- `statechange` listener auto-resumes from `interrupted` state

## Testing Checklist for iOS Audio

### Setup
- [ ] Use a physical iPhone/iPad (simulators don't reproduce audio policy)
- [ ] Test in Safari (not Chrome-on-iOS, which uses the same WebKit but may differ)
- [ ] Have the Safari Web Inspector connected for console logs
- [ ] Add `?audioDebug=1` to the URL

### Basic Playback
- [ ] Load a song, wait for stems to finish loading
- [ ] Tap play -- sound should come out
- [ ] Check console: `ctx.state` should be `running` after play
- [ ] Check console: gains should be non-zero
- [ ] Tap pause, tap play again -- sound resumes

### Edge Cases
- [ ] Load page, wait 30+ seconds, then tap play (delayed first interaction)
- [ ] Play, receive a phone call, dismiss it, tap play again
- [ ] Play, lock the phone, unlock, tap play again
- [ ] Play, switch to another app, come back, tap play again
- [ ] Play, invoke Siri, dismiss Siri, tap play again
- [ ] Seek while playing -- audio continues from new position
- [ ] Change tempo while playing -- audio adjusts without cutting out
- [ ] Mute/solo stems while playing -- gain changes take effect

### Volume Controls
- [ ] Master volume slider works
- [ ] Individual stem volume sliders work
- [ ] Mute button silences the stem
- [ ] Solo button isolates the stem
- [ ] Un-solo restores all stems

### Debug Verification
- [ ] `window.audioDebug()` returns a complete report
- [ ] All stems show `loaded: true`
- [ ] All stems show `first100samples.hasNonZero: true`
- [ ] Audio graph shows `BufferSource -> GainNode -> destination` for each stem
- [ ] No warnings about "PLAYING but ctx.state is suspended" in periodic check

## File Locations

| File | What |
|------|------|
| `/Users/jeffkozelski/stemscribe/frontend/practice.html` | Practice page audio engine (StemAudioEngine class + debug mode) |
| `/Users/jeffkozelski/stemscribe/frontend/js/mixer.js` | Mixer page audio engine (StemAudioEngine prototype + debug method) |
| `/Users/jeffkozelski/stemscribe/frontend/js/results.js` | Results display, creates engine instance, loads stems |
