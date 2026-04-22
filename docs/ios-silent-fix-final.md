# iOS Silent Switch Bypass — Final Implementation

**Date:** 2026-03-18

## Problem
iOS Safari plays Web Audio through the "ambient" audio category by default. When the physical silent/ringer switch is ON, ambient audio is muted. This means StemScribe stems and karaoke play no sound on iPhones with the silent switch engaged.

## Solution (belt-and-suspenders)

Two strategies are used together for maximum iOS version coverage:

### Strategy 1 — `navigator.audioSession.type = 'playback'` (iOS 16.4+)
The official Web Audio Session API. One-line fix that tells Safari to use the "playback" audio category, which ignores the silent switch. Set before any AudioContext creation/resume.

### Strategy 2 — Silent looping MP3 via HTMLAudioElement (legacy fallback)
A silent MP3 data URI plays in a loop via an `<audio>` element. This forces iOS from the "ambient" to "playback" audio channel even on older iOS versions. Key details:
- Uses `x-webkit-airplay="deny"` to hide from Control Center
- `loop=true` keeps the media channel active
- Volume stays at default 1.0 (setting to 0 defeats the purpose)
- Element does NOT need to be in the DOM
- Paused on `visibilitychange` (page hidden) to save battery, resumed when visible

Both strategies are triggered on the FIRST user interaction (touchend/click), inside the gesture handler as required by iOS.

## Files Modified

### 1. `frontend/index.html`
- **Lines 711-755 (approx):** Replaced old one-shot silent MP3 unlock IIFE with enhanced version using both strategies + visibility change cleanup.

### 2. `frontend/js/mixer.js`
- **IIFE at top of file:** Replaced old `unlockiOSAudio()` with `initIOSAudioSession()` using both strategies + visibility change cleanup.
- **`StemAudioEngine.prototype.init()`:** Added `navigator.audioSession.type = 'playback'` before AudioContext creation.

### 3. `frontend/practice.html`
- **Old one-shot IIFE (was lines 1997-2015):** Removed (replaced with comment pointing to enhanced version below).
- **Second IIFE (was lines 2069-2124):** Replaced with enhanced `initIOSAudioSession()` using both strategies + visibility change cleanup.
- **`StemAudioEngine.init()` method:** Added `navigator.audioSession.type = 'playback'` before AudioContext creation.

## What Was NOT Changed
- No backend code modified
- No changes to stem loading, decoding, or playback logic
- No changes to karaoke.js (it runs inside index.html which already has the fix)
- Desktop Chrome/Firefox behavior unaffected (feature detection guards all iOS-specific code)

## Silent MP3 Data URI
The looping silent MP3 uses this data URI (valid MPEG audio, ~192 bytes decoded):
```
data:audio/mpeg;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA/+M4wAAAAAAAAAAAAEluZm8AAAAPAAAAAwAAAbAA...
```

## Debugging
All steps are logged to console with `[iOS Audio]` prefix:
- `[iOS Audio] initIOSAudioSession — unlocking on first user interaction`
- `[iOS Audio] navigator.audioSession.type set to "playback"`
- `[iOS Audio] Silent looping MP3 playing — audio channel forced to playback`
- `[iOS Audio] Paused silent MP3 (page hidden)`
- `[iOS Audio] Resumed silent MP3 (page visible)`
