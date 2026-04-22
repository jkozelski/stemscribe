# iOS Safari Multi-Track Audio Playback Research

**Date:** 2026-03-18
**Problem:** StemScribe uses 6 separate `<audio>` elements (vocals, drums, bass, guitar, other, full mix). On iOS Safari, the play button shows pause state but audio stays at 0:00, no sound plays.

**Root cause:** iOS Safari does not allow multiple `<audio>` elements to play simultaneously. Only one HTMLMediaElement can be active at a time. This is a longstanding WebKit restriction that has never been lifted.

---

## Table of Contents

1. [Approach 1: Web Audio API with AudioBufferSourceNode](#approach-1-web-audio-api-with-audiobuffersourcenode)
2. [Approach 2: Web Audio API with MediaElementSource](#approach-2-web-audio-api-with-mediaelementsource)
3. [Approach 3: Howler.js / Tone.js](#approach-3-howlerjs--tonejs)
4. [Approach 4: Single Merged Audio](#approach-4-single-merged-audio)
5. [Approach 5: Competitor Analysis](#approach-5-competitor-analysis)
6. [iOS Safari Constraints (Verified)](#ios-safari-constraints-verified)
7. [RECOMMENDED APPROACH](#recommended-approach)
8. [Implementation Code Skeleton](#implementation-code-skeleton)

---

## Approach 1: Web Audio API with AudioBufferSourceNode

**How it works:** Fetch each stem as an ArrayBuffer, decode to AudioBuffer via `decodeAudioData()`, create an `AudioBufferSourceNode` per stem, route each through a `GainNode` for volume/mute control, connect all to the AudioContext destination.

### Pros
- Full iOS Safari support -- multiple AudioBufferSourceNodes CAN play simultaneously through a single AudioContext
- Precise synchronization via `audioContext.currentTime`
- GainNode per stem gives smooth volume/mute control
- No dependency on `<audio>` elements at all
- Works across all modern browsers

### Cons
- AudioBufferSourceNode is "one-shot" -- cannot pause/resume, must stop and recreate
- All audio data must be decoded into RAM (uncompressed PCM)
- Memory-heavy for long songs with 6 stems

### Seek Implementation
AudioBufferSourceNode has no seek capability. You must:
1. Stop the current source node (`source.stop()`)
2. Create a new AudioBufferSourceNode
3. Connect it to the same GainNode
4. Call `source.start(0, offsetInSeconds)` where offset is the seek position

### Pause/Resume Implementation
```javascript
// Pause: record elapsed time, stop source
function pause() {
  elapsedTime += audioContext.currentTime - startTimestamp;
  sources.forEach(s => s.stop());
  isPlaying = false;
}

// Resume: create new sources, start from offset
function resume() {
  stems.forEach((stem, i) => {
    const source = audioContext.createBufferSource();
    source.buffer = stem.buffer;
    source.connect(stem.gainNode);
    sources[i] = source;
    source.start(0, elapsedTime);
  });
  startTimestamp = audioContext.currentTime;
  isPlaying = true;
}
```

### Memory Concerns
- A 5MB MP3 decodes to ~55MB of uncompressed PCM (Float32, 44.1kHz, stereo)
- 6 stems x 55MB = ~330MB for a 5-minute song
- iOS Safari has a ~300-500MB WebKit process memory limit before it kills the tab
- **This is the primary risk** -- long songs with 6 stems can OOM on older iPhones
- Mitigation: use mono stems where possible (halves memory), consider lower sample rates

### Verdict
**Best approach for iOS compatibility.** Memory is the only real concern, and it's manageable for typical 3-5 minute songs, especially if stems are mono.

---

## Approach 2: Web Audio API with MediaElementSource

**How it works:** Keep `<audio>` elements but route them through `createMediaElementSource()` into GainNodes via a single AudioContext.

### The Problem
This does NOT fix iOS Safari's core limitation. `createMediaElementSource()` still requires the underlying `<audio>` element to play, and iOS Safari still only allows one HTMLMediaElement to be active at a time. Routing through AudioContext doesn't bypass this restriction.

### Known Issues
- WebKit bug #211394: `createMediaElementSource` has been broken or unreliable on iOS Safari across many versions
- Apple Developer Forums confirm this approach does not work for multi-track
- Safari limits to 4 AudioContexts total per page
- Choppy playback reported even when it partially works (wavesurfer.js issue #2210)

### Verdict
**Do not use this approach.** It does not solve the fundamental problem and introduces additional Safari-specific bugs.

---

## Approach 3: Howler.js / Tone.js

### Howler.js
- Defaults to Web Audio API, falls back to HTML5 Audio
- Has built-in iOS unlock handling
- **Issue:** Documented problems with simultaneous multi-track playback on iOS (GitHub issues #698, #1148)
- When using Web Audio mode (which it does by default), it uses AudioBufferSourceNode internally -- same as Approach 1
- Bundle size: ~10KB gzipped
- **Verdict:** Adds a dependency layer but underneath it's doing the same thing as Approach 1. The abstraction adds complexity without solving additional problems. Not recommended for our use case since we need fine-grained control over sync/seek.

### Tone.js
- Full music framework built on Web Audio API
- Has `Tone.Player` and `Tone.Players` for multi-track
- Known iOS issues: playbackRate not supported as signal on Safari, Tone.Offline broken on iOS
- AudioContext "interrupted" state handling is problematic (GitHub issue #995)
- Bundle size: ~150KB gzipped -- very heavy for just playback
- **Verdict:** Overkill. We don't need synthesis, scheduling, or effects. Too large, too many iOS edge cases.

### standardized-audio-context (notable mention)
- Cross-browser Web Audio API ponyfill
- Smooths over Safari inconsistencies without patching globals
- Worth considering as a thin compatibility layer
- Confirms Safari limit: only 4 AudioContexts simultaneously

### Verdict
**Libraries don't solve the fundamental problem.** They all use Web Audio API internally. Better to implement directly with AudioBufferSourceNode and have full control.

---

## Approach 4: Single Merged Audio

**Concept:** Use ffmpeg on the backend to merge 6 stems into a single file.

### Options
- **Multi-channel WAV/FLAC:** 12 channels (6 stereo stems) -- browsers don't support >2 channel playback natively
- **Interleaved approach:** Multiplex stems into time segments -- destroys sync
- **Downmix subsets:** Pre-render common combinations (e.g., "everything minus vocals") -- exponential combinations (2^6 = 64)

### Verdict
**Not practical.** Can't independently control 6 stems from a stereo audio file. Would require server-side rendering of every possible combination.

---

## Approach 5: Competitor Analysis

### stemplayer-js (open source)
- Web component for stem playback
- Uses HLS streaming with segmented audio chunks
- Relies on Web Audio API for synchronized playback
- Addresses memory by streaming segments rather than loading entire files
- Source: https://github.com/stemplayer-js/stemplayer-js

### simple-mixer (open source)
- React + Web Audio API stem mixer
- Uses AudioBufferSourceNode with SoundTouchJS for pitch/speed
- Explicitly notes: "OfflineAudioContext does not work on iOS devices"
- Uses AudioWorklet where available, ScriptProcessorNode as fallback
- Source: https://github.com/goto920/simple-mixer

### waveform-playlist (open source)
- Most mature multi-track Web Audio editor
- Uses AudioBufferSourceNode approach
- Canvas waveform visualization
- Tone.js integration for effects
- Source: https://github.com/naomiaro/waveform-playlist

### Moises App
- Native iOS/Android apps (not web-based for playback)
- Web version exists for editing but relies on native app for real-time stem playback
- Uses server-side stem separation, client-side mixing

### Common Pattern Across All
Every working implementation uses **Web Audio API with AudioBufferSourceNode** routed through GainNodes. None use multiple `<audio>` elements. The HLS streaming variant (stemplayer-js) is the most memory-efficient but adds significant complexity.

---

## iOS Safari Constraints (Verified)

### AudioContext must be created/resumed on user gesture
**YES, confirmed.** AudioContext starts in "suspended" state. Must call `audioContext.resume()` inside a user-initiated event handler (click, touchend). Best practice: create AudioContext on first user tap, call resume(), and reuse it for the page lifetime.

### Can multiple AudioBufferSourceNodes play simultaneously?
**YES, confirmed.** Multiple AudioBufferSourceNodes connected to the same AudioContext destination play simultaneously without issue. This is the fundamental architecture of the Web Audio API -- mixing is handled by the audio graph. This works on iOS Safari.

### Memory limit for decoded audio buffers?
- iOS Safari WebKit process limit: ~300-500MB before tab crash
- Each minute of stereo 44.1kHz audio = ~21MB decoded PCM
- 6 stereo stems x 5 minutes = ~630MB (OVER LIMIT)
- 6 mono stems x 5 minutes = ~315MB (borderline)
- **Mitigation:** Use mono stems (our stems are already mono-source, just encoded as stereo). Strip to mono on backend = 50% memory savings. Also consider: lazy-load only active stems, or use lower sample rate (22050Hz) for non-critical stems.

### Does decodeAudioData() work with large files (5+ min)?
**YES, but with caveats:**
- Safari uses callback API, not promise API (must handle both)
- Decoding blocks the main thread briefly -- decode in sequence, not all 6 at once
- No streaming decode support -- entire file must be fetched first
- For very long files (10+ min), consider chunked loading approach

### Audio session interruptions (phone call, notification)?
- AudioContext state changes to "interrupted" (Safari-specific, non-standard)
- Must listen for `statechange` event on AudioContext
- After interruption ends, call `audioContext.resume()` on next user gesture
- All source nodes are destroyed -- must recreate and restart from tracked position
- This is the most fragile part of iOS audio and requires careful state management

### Silent mode / ringer switch?
- Web Audio API respects the iOS silent mode switch
- If ringer is on vibrate, Web Audio produces no sound
- `<video>` and `<audio>` elements with user interaction CAN play in silent mode
- **No workaround exists** -- this is an OS-level restriction for Web Audio

---

## RECOMMENDED APPROACH

### Web Audio API with AudioBufferSourceNode + GainNode per stem

This is the only approach that reliably works on iOS Safari for multi-track synchronized playback. Every competitor and open-source project uses this pattern.

### Architecture

```
User Tap (unlock) → AudioContext.resume()
                          │
  fetch(stem1.mp3) → decodeAudioData() → AudioBuffer ─→ AudioBufferSourceNode ─→ GainNode ─┐
  fetch(stem2.mp3) → decodeAudioData() → AudioBuffer ─→ AudioBufferSourceNode ─→ GainNode ─┤
  fetch(stem3.mp3) → decodeAudioData() → AudioBuffer ─→ AudioBufferSourceNode ─→ GainNode ─┤
  fetch(stem4.mp3) → decodeAudioData() → AudioBuffer ─→ AudioBufferSourceNode ─→ GainNode ─┼→ ctx.destination
  fetch(stem5.mp3) → decodeAudioData() → AudioBuffer ─→ AudioBufferSourceNode ─→ GainNode ─┤
  fetch(stem6.mp3) → decodeAudioData() → AudioBuffer ─→ AudioBufferSourceNode ─→ GainNode ─┘
```

### Key Design Decisions

1. **Single AudioContext** -- create once, reuse forever, never create more than one
2. **Persistent GainNodes** -- create once per stem, reconnect new sources to them
3. **Mono stems on backend** -- cut memory usage in half, our stems are mono-source anyway
4. **Sequential decode** -- decode one buffer at a time to avoid blocking the main thread
5. **Track elapsed time manually** -- use `audioContext.currentTime` delta tracking for pause/resume/seek
6. **Handle "interrupted" state** -- listen for statechange, re-resume on user gesture

### Memory Budget (mono stems, 44.1kHz)
| Song Length | Per Stem | 6 Stems | Safe? |
|-------------|----------|---------|-------|
| 3 min       | 15.9 MB  | 95 MB   | YES   |
| 4 min       | 21.2 MB  | 127 MB  | YES   |
| 5 min       | 26.5 MB  | 159 MB  | YES   |
| 7 min       | 37.0 MB  | 222 MB  | YES   |
| 10 min      | 52.9 MB  | 317 MB  | RISKY |

For songs under 7 minutes (vast majority), this is safe on all modern iPhones.

---

## Implementation Code Skeleton

```javascript
/**
 * StemScribe iOS-Compatible Multi-Track Audio Engine
 * Uses Web Audio API with AudioBufferSourceNode per stem
 */
class StemAudioEngine {
  constructor() {
    this.audioContext = null;
    this.stems = {};          // { name: { buffer, source, gainNode } }
    this.isPlaying = false;
    this.startTimestamp = 0;   // audioContext.currentTime when playback started
    this.elapsedTime = 0;      // accumulated playback time across pauses
    this.duration = 0;         // total song duration in seconds
    this._onEndedCallback = null;
    this._unlocked = false;
  }

  // ─── INITIALIZATION ───────────────────────────────────────────

  /**
   * Must be called from a user gesture (click/tap handler).
   * Creates and unlocks the AudioContext.
   */
  async init() {
    if (this.audioContext) return;

    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    this.audioContext = new AudioContextClass();

    // iOS Safari: resume from suspended state
    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }

    this._unlocked = true;

    // Handle iOS "interrupted" state (phone calls, Siri, tab switch)
    this.audioContext.addEventListener('statechange', () => {
      if (this.audioContext.state === 'interrupted') {
        // Audio was interrupted -- will need user gesture to resume
        this._unlocked = false;
        if (this.isPlaying) {
          // Track where we were
          this.elapsedTime += this.audioContext.currentTime - this.startTimestamp;
          this.isPlaying = false;
          this._notifyUI('interrupted');
        }
      }
      if (this.audioContext.state === 'running' && !this._unlocked) {
        this._unlocked = true;
        this._notifyUI('resumed');
      }
    });
  }

  /**
   * Attempt to resume after interruption. Call from user gesture.
   */
  async tryResume() {
    if (this.audioContext && this.audioContext.state !== 'running') {
      await this.audioContext.resume();
      this._unlocked = true;
    }
  }

  // ─── LOADING ──────────────────────────────────────────────────

  /**
   * Load and decode all stems. Call after init().
   * @param {Object} stemUrls - { vocals: '/url', drums: '/url', ... }
   * @param {Function} onProgress - Called with (loaded, total) counts
   */
  async loadStems(stemUrls, onProgress) {
    const stemNames = Object.keys(stemUrls);
    const total = stemNames.length;
    let loaded = 0;

    for (const name of stemNames) {
      const url = stemUrls[name];

      // Fetch the audio file
      const response = await fetch(url);
      const arrayBuffer = await response.arrayBuffer();

      // Decode to AudioBuffer (sequential to avoid main thread overload)
      const audioBuffer = await this._decodeAudio(arrayBuffer);

      // Create persistent GainNode for this stem
      const gainNode = this.audioContext.createGain();
      gainNode.connect(this.audioContext.destination);

      this.stems[name] = {
        buffer: audioBuffer,
        source: null,
        gainNode: gainNode,
        muted: false,
        volume: 1.0,
      };

      // Track duration from first stem
      if (this.duration === 0) {
        this.duration = audioBuffer.duration;
      }

      loaded++;
      if (onProgress) onProgress(loaded, total);
    }
  }

  /**
   * Cross-browser decodeAudioData (Safari uses callback API)
   */
  _decodeAudio(arrayBuffer) {
    return new Promise((resolve, reject) => {
      // Try promise-based first (Chrome, Firefox, modern Safari)
      const result = this.audioContext.decodeAudioData(
        arrayBuffer,
        (buffer) => resolve(buffer),    // Success callback (Safari fallback)
        (err) => reject(err)            // Error callback
      );
      // If it returns a promise, use that instead
      if (result && typeof result.then === 'function') {
        result.then(resolve).catch(reject);
      }
    });
  }

  // ─── PLAYBACK CONTROLS ────────────────────────────────────────

  /**
   * Start or resume playback from current position.
   */
  play() {
    if (this.isPlaying) return;
    if (!this._unlocked) {
      console.warn('AudioContext not unlocked. Call tryResume() from user gesture.');
      return;
    }

    // Create new source nodes for each stem (they are one-shot)
    for (const name of Object.keys(this.stems)) {
      const stem = this.stems[name];
      this._createAndStartSource(stem, this.elapsedTime);
    }

    this.startTimestamp = this.audioContext.currentTime;
    this.isPlaying = true;
  }

  /**
   * Pause playback, preserving position.
   */
  pause() {
    if (!this.isPlaying) return;

    // Accumulate elapsed time
    this.elapsedTime += this.audioContext.currentTime - this.startTimestamp;

    // Stop all source nodes
    for (const name of Object.keys(this.stems)) {
      const stem = this.stems[name];
      if (stem.source) {
        try { stem.source.stop(); } catch (e) { /* already stopped */ }
        stem.source = null;
      }
    }

    this.isPlaying = false;
  }

  /**
   * Seek to a specific time in seconds.
   */
  seek(timeInSeconds) {
    const wasPlaying = this.isPlaying;

    // Stop current playback
    if (wasPlaying) {
      for (const name of Object.keys(this.stems)) {
        const stem = this.stems[name];
        if (stem.source) {
          try { stem.source.stop(); } catch (e) {}
          stem.source = null;
        }
      }
    }

    // Update position
    this.elapsedTime = Math.max(0, Math.min(timeInSeconds, this.duration));

    // Restart if was playing
    if (wasPlaying) {
      for (const name of Object.keys(this.stems)) {
        const stem = this.stems[name];
        this._createAndStartSource(stem, this.elapsedTime);
      }
      this.startTimestamp = this.audioContext.currentTime;
      this.isPlaying = true;
    }
  }

  /**
   * Get current playback position in seconds.
   */
  getCurrentTime() {
    if (this.isPlaying) {
      return this.elapsedTime + (this.audioContext.currentTime - this.startTimestamp);
    }
    return this.elapsedTime;
  }

  // ─── STEM VOLUME / MUTE CONTROL ──────────────────────────────

  /**
   * Set volume for a specific stem (0.0 to 1.0).
   */
  setVolume(stemName, volume) {
    const stem = this.stems[stemName];
    if (!stem) return;
    stem.volume = volume;
    if (!stem.muted) {
      stem.gainNode.gain.setValueAtTime(volume, this.audioContext.currentTime);
    }
  }

  /**
   * Toggle mute for a specific stem.
   */
  toggleMute(stemName) {
    const stem = this.stems[stemName];
    if (!stem) return;
    stem.muted = !stem.muted;
    const targetGain = stem.muted ? 0 : stem.volume;
    // Use setTargetAtTime for smooth transition (avoid clicks)
    stem.gainNode.gain.setTargetAtTime(targetGain, this.audioContext.currentTime, 0.015);
  }

  /**
   * Set mute state explicitly.
   */
  setMute(stemName, muted) {
    const stem = this.stems[stemName];
    if (!stem) return;
    stem.muted = muted;
    const targetGain = muted ? 0 : stem.volume;
    stem.gainNode.gain.setTargetAtTime(targetGain, this.audioContext.currentTime, 0.015);
  }

  /**
   * Solo a stem (mute all others).
   */
  solo(stemName) {
    for (const name of Object.keys(this.stems)) {
      this.setMute(name, name !== stemName);
    }
  }

  /**
   * Unsolo (unmute all stems).
   */
  unsolo() {
    for (const name of Object.keys(this.stems)) {
      this.setMute(name, false);
    }
  }

  // ─── INTERNAL HELPERS ─────────────────────────────────────────

  /**
   * Create a new AudioBufferSourceNode and start it at the given offset.
   */
  _createAndStartSource(stem, offset) {
    const source = this.audioContext.createBufferSource();
    source.buffer = stem.buffer;
    source.connect(stem.gainNode);

    // Handle natural end of playback
    source.onended = () => {
      if (this.isPlaying && this.getCurrentTime() >= this.duration - 0.1) {
        this.isPlaying = false;
        this.elapsedTime = 0;
        if (this._onEndedCallback) this._onEndedCallback();
      }
    };

    // Clamp offset to valid range
    const safeOffset = Math.max(0, Math.min(offset, stem.buffer.duration - 0.01));
    source.start(0, safeOffset);
    stem.source = source;
  }

  /**
   * Notify UI of state changes (implement as needed).
   */
  _notifyUI(event) {
    // Override this or use an event emitter
    console.log('StemAudioEngine:', event);
  }

  // ─── CLEANUP ──────────────────────────────────────────────────

  /**
   * Release all audio buffers and close the context.
   */
  dispose() {
    this.pause();
    for (const name of Object.keys(this.stems)) {
      this.stems[name].buffer = null;
      this.stems[name].gainNode.disconnect();
    }
    this.stems = {};
    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close();
    }
    this.audioContext = null;
  }
}
```

### Integration Example

```javascript
// In your existing mixer UI code:

const engine = new StemAudioEngine();

// On first user tap (play button, or a dedicated "Load Song" button):
async function onPlayButtonClick() {
  // Init AudioContext on first tap (iOS requirement)
  await engine.init();

  // Load stems if not loaded
  if (Object.keys(engine.stems).length === 0) {
    await engine.loadStems({
      vocals: `/api/stems/${songId}/vocals.mp3`,
      drums:  `/api/stems/${songId}/drums.mp3`,
      bass:   `/api/stems/${songId}/bass.mp3`,
      guitar: `/api/stems/${songId}/guitar.mp3`,
      other:  `/api/stems/${songId}/other.mp3`,
      full:   `/api/stems/${songId}/full.mp3`,
    }, (loaded, total) => {
      updateLoadingBar(loaded / total * 100);
    });
  }

  // Toggle play/pause
  if (engine.isPlaying) {
    engine.pause();
  } else {
    engine.play();
  }
  updatePlayButton(engine.isPlaying);
}

// Seek bar handler
function onSeekBarChange(e) {
  const seekTime = (e.target.value / 100) * engine.duration;
  engine.seek(seekTime);
}

// Volume slider handler
function onVolumeChange(stemName, value) {
  engine.setVolume(stemName, value / 100);
}

// Mute button handler
function onMuteToggle(stemName) {
  engine.toggleMute(stemName);
}

// Update playback position display (call in requestAnimationFrame loop)
function updateTimeDisplay() {
  if (engine.isPlaying) {
    const current = engine.getCurrentTime();
    updateProgressBar(current / engine.duration * 100);
    updateTimeLabel(formatTime(current));
    requestAnimationFrame(updateTimeDisplay);
  }
}

// Handle interruption recovery
document.addEventListener('touchend', async () => {
  await engine.tryResume();
}, { once: false });
```

### Backend Optimization: Mono Stems

Add this to your stem processing pipeline to cut iOS memory usage in half:

```python
# In your ffmpeg stem processing, force mono output:
# Instead of: -ac 2 (stereo)
# Use: -ac 1 (mono)

ffmpeg_cmd = [
    'ffmpeg', '-i', input_stem,
    '-ac', '1',           # Force mono -- halves decoded memory on iOS
    '-ar', '44100',       # Standard sample rate
    '-b:a', '128k',       # Good quality for mono
    '-f', 'mp3',
    output_path
]
```

---

## Migration Checklist

1. [ ] Create `StemAudioEngine` class (based on skeleton above)
2. [ ] Update backend to serve mono stems (add `-ac 1` to ffmpeg)
3. [ ] Replace `<audio>` element creation with `engine.loadStems()`
4. [ ] Replace `audio.play()/pause()` with `engine.play()/pause()`
5. [ ] Replace volume slider handlers to use `engine.setVolume()`
6. [ ] Replace mute button handlers to use `engine.toggleMute()`
7. [ ] Replace seek bar to use `engine.seek()` and `engine.getCurrentTime()`
8. [ ] Add `touchend` listener for interruption recovery
9. [ ] Add loading spinner/progress during stem decode phase
10. [ ] Test on iOS Safari (iPhone), iPad Safari, Chrome iOS, desktop browsers
11. [ ] Test with 3-min, 5-min, and 7-min songs for memory stability
12. [ ] Call `engine.dispose()` when navigating away from mixer page

---

## Sources

- [MDN Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
- [MDN AudioBufferSourceNode](https://developer.mozilla.org/en-US/docs/Web/API/AudioBufferSourceNode)
- [Apple: Playing Sounds with Web Audio API](https://developer.apple.com/library/archive/documentation/AudioVideo/Conceptual/Using_HTML5_Audio_Video/PlayingandSynthesizingSounds/PlayingandSynthesizingSounds.html)
- [Matt Montag: Unlock Web Audio in Safari](https://www.mattmontag.com/web/unlock-web-audio-in-safari-for-ios-and-macos)
- [WebKit Bug #211394: createMediaElementSource not working](https://bugs.webkit.org/show_bug.cgi?id=211394)
- [AudioContext "interrupted" state issue](https://github.com/WebAudio/web-audio-api/issues/2585)
- [stemplayer-js (streaming stem player)](https://github.com/stemplayer-js/stemplayer-js)
- [simple-mixer (React stem mixer)](https://github.com/goto920/simple-mixer)
- [waveform-playlist (multi-track editor)](https://github.com/naomiaro/waveform-playlist)
- [standardized-audio-context (cross-browser ponyfill)](https://github.com/chrisguttandin/standardized-audio-context)
- [Howler.js](https://github.com/goldfire/howler.js)
- [Tone.js iOS issues](https://github.com/Tonejs/Tone.js/issues/511)
- [Hans Garon: Pause/Resume Web Audio](https://hansgaron.com/articles/web_audio/enabling_pause_and_resume/)
- [AudioWorklet replacing AudioBufferSourceNode](https://blog.kenrick95.org/2022/04/how-i-use-audio-worklet-to-replace-audiobuffersourcenode/)
- [Kenrick: fetch-stream-audio (chunked decode)](https://github.com/anthumchris/fetch-stream-audio)
