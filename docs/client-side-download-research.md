# Client-Side YouTube Audio Extraction Research

**Date:** 2026-03-26
**Purpose:** Evaluate whether the user's browser can download/process YouTube audio WITHOUT the server touching YouTube, shifting legal liability away from StemScribe.

---

## Table of Contents

1. [Direct Client-Side YouTube Audio Extraction](#1-direct-client-side-youtube-audio-extraction)
2. [YouTube IFrame Player API + Web Audio API](#2-youtube-iframe-player-api--web-audio-api)
3. [MediaSource / captureStream() Approach](#3-mediasource--capturestream-approach)
4. [getDisplayMedia / Tab Audio Capture](#4-getdisplaymedia--tab-audio-capture)
5. [Browser Extension Approach](#5-browser-extension-approach)
6. [Service Worker Interception](#6-service-worker-interception)
7. [User Upload from Local Music Library](#7-user-upload-from-local-music-library)
8. [Third-Party Proxy Services (Cobalt, etc.)](#8-third-party-proxy-services-cobalt-etc)
9. [Legal Analysis](#9-legal-analysis)
10. [Recommended Approaches](#10-recommended-approaches)

---

## 1. Direct Client-Side YouTube Audio Extraction

**Feasibility: Not Feasible**

### The Problem

True client-side YouTube audio extraction from a browser is fundamentally blocked by CORS (Cross-Origin Resource Sharing) restrictions. YouTube does not include `Access-Control-Allow-Origin` headers on its media resources. This means:

- JavaScript running on `stemscribe.io` cannot make `fetch()` or `XMLHttpRequest` calls to YouTube's media servers
- The browser's same-origin policy enforces this at the engine level -- it cannot be bypassed by client-side code
- YouTube actively obfuscates and rotates its media URLs, requiring signature decryption logic (which is why yt-dlp needs a JavaScript runtime)

### What About yt-dlp in the Browser?

There is no working WebAssembly port of yt-dlp. A GitHub issue (yt-dlp/yt-dlp#9820) proposed using Pyodide to compile yt-dlp's Python to WASM, but this remains unimplemented. Even if it were compiled, the CORS restriction would still block the actual HTTP requests to YouTube's servers from the browser.

### What About ytdl-core in the Browser?

The npm package `ytdl-core` (and its forks like `@distube/ytdl-core`) are Node.js libraries. They make HTTP requests server-side. Running them in the browser hits the same CORS wall.

### Bottom Line

There is no way for JavaScript running in a user's browser on your domain to directly fetch audio data from YouTube's servers. The browser will block it. Period.

**Sources:**
- [YouTube CORS issue - ytdl-core](https://github.com/fent/node-ytdl-core/issues/75)
- [yt-dlp WASM discussion](https://github.com/yt-dlp/yt-dlp/issues/9820)
- [yt-dlp now requires external JS runtime](https://github.com/yt-dlp/yt-dlp/issues/15012)

---

## 2. YouTube IFrame Player API + Web Audio API

**Feasibility: Not Feasible**

### What the API Offers

The YouTube IFrame Player API lets you embed a YouTube video player and control it via JavaScript:
- Play/pause/seek
- Volume control
- Get video metadata (title, duration)
- Subscribe to state change events

### What It Does NOT Offer

- No access to raw audio data or audio buffers
- No `getAudioData()` or similar method
- No way to route the audio through Web Audio API nodes (AnalyserNode, etc.)
- No way to create a `MediaElementSource` from the iframe's internal `<video>` element

### Why Web Audio API Can't Connect

Even if you could access the `<video>` element inside the YouTube iframe (you cannot, due to cross-origin iframe restrictions), calling `audioContext.createMediaElementSource(videoElement)` would fail because:

1. The iframe is cross-origin (youtube.com vs stemscribe.io)
2. The media resource lacks CORS headers
3. The browser outputs silence: "MediaElementAudioSource outputs zeroes due to CORS access restrictions"

### Practical Use

The IFrame API is useful for building a playback UI (play alongside your stems), but you cannot extract or analyze the audio data from it.

**Sources:**
- [YouTube IFrame Player API Reference](https://developers.google.com/youtube/iframe_api_reference)
- [Web Audio API cross-origin issue #2547](https://github.com/WebAudio/web-audio-api/issues/2547)
- [W3C mediacapture-output iframe audio issue](https://github.com/w3c/mediacapture-output/issues/63)

---

## 3. MediaSource / captureStream() Approach

**Feasibility: Not Feasible**

### HTMLMediaElement.captureStream()

The `captureStream()` method returns a `MediaStream` that captures the output of an `<audio>` or `<video>` element in real time. In theory, you could then record or analyze this stream.

### Cross-Origin Restriction

This is explicitly blocked for cross-origin media:

> "mozCaptureStream() stops cross-origin media data from entering MediaStreams. This was added to block cross-origin audio from entering the MediaStreamGraph system where it could eventually be examined by Web Audio ScriptProcessorNodes."

For YouTube content:
- The media is served from `*.googlevideo.com` (cross-origin)
- No CORS headers are provided
- `captureStream()` returns a stream that outputs **silence** (zeroed audio data)
- `createMediaElementSource()` produces the same result -- zeroed output

### Even If You Had a `<video>` Element

Even if YouTube served content with CORS headers (it never will), you would need the `<video>` element to have `crossOrigin="anonymous"` set, AND the server to respond with `Access-Control-Allow-Origin: *`. YouTube does neither.

**Sources:**
- [MDN: HTMLMediaElement.captureStream()](https://developer.mozilla.org/en-US/docs/Web/API/HTMLMediaElement/captureStream)
- [W3C: captureStream for cross-origin media](https://github.com/w3c/mediacapture-fromelement/issues/21)
- [WebAudio issue: cross-origin restrictions](https://github.com/WebAudio/web-audio-api/issues/2547)

---

## 4. getDisplayMedia / Tab Audio Capture

**Feasibility: Partially Feasible (with severe UX limitations)**

### How It Works

`navigator.mediaDevices.getDisplayMedia()` can capture a browser tab's audio:

```javascript
const stream = await navigator.mediaDevices.getDisplayMedia({
  video: true,  // REQUIRED -- cannot request audio-only
  audio: {
    systemAudio: 'include'
  }
});

// Extract audio track
const audioTrack = stream.getAudioTracks()[0];
const recorder = new MediaRecorder(stream);
// ... record the audio
```

### The UX Problems

1. **User must explicitly share a tab** -- a browser picker dialog appears
2. **The "Share tab audio" checkbox is NOT checked by default** -- user must manually enable it
3. **Video is required** -- you must request video even if you only want audio
4. **Real-time only** -- audio is captured at 1x playback speed (a 4-minute song takes 4 minutes)
5. **User must keep the tab open and playing** for the entire duration

### Browser Support

- Chrome/Edge: Works (tab audio capture supported)
- Firefox: Ignores the audio parameter entirely
- Safari: Same as Firefox -- audio not supported in getDisplayMedia

### Could This Work for StemScribe?

Theoretically: User plays a YouTube video, shares the tab, waits for the song to finish, then the recorded audio blob is sent to the server for stem separation. But the UX is terrible -- multi-step, confusing, slow, and only works on Chrome/Edge.

**Sources:**
- [getDisplayMedia demo with audio](https://addpipe.com/getdisplaymedia-demo/)
- [MDN: getDisplayMedia()](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getDisplayMedia)
- [Chrome screen capture docs](https://developer.chrome.com/docs/extensions/how-to/web-platform/screen-capture)

---

## 5. Browser Extension Approach

**Feasibility: Partially Feasible (with major distribution constraints)**

### Chrome Extensions

#### Technical Capability
A Chrome extension with the `tabCapture` permission can:
- Capture all audio playing in the current tab as a `MediaStream`
- Process it with Web Audio API (AnalyserNode, MediaRecorder, etc.)
- Send the captured audio blob to a web app via `chrome.runtime.sendMessage` or by injecting it into the page

```javascript
// In extension service worker (Chrome 116+):
chrome.tabCapture.getMediaStreamId({ targetTabId: tabId }, (streamId) => {
  // Pass streamId to offscreen document for recording
});
```

#### Chrome Web Store Policy: BLOCKED

Google explicitly prohibits extensions that download from YouTube:

> "Chrome Web Store does not permit extensions that download videos from YouTube, circumvent YouTube's protections, or enable saving videos outside of YouTube Premium features."

Extensions like "Video & Audio Downloader" and "Audio Downloader Prime" explicitly state they do NOT work with YouTube due to these policies.

**An extension that captures YouTube tab audio and sends it to StemScribe would be rejected from the Chrome Web Store.**

### Firefox Add-ons

Firefox is more permissive. Multiple YouTube audio download extensions exist on addons.mozilla.org:
- YouTube Video and Audio Downloader (WebEx)
- YouTube Audio Downloader
- YouTube to MP3

A Firefox extension could capture audio and pass it to StemScribe. However, Firefox's market share is ~3%, making this impractical as a primary solution.

### Sideloaded Extension (No Store)

You could distribute a Chrome extension outside the Web Store (via developer mode), but:
- Requires users to enable developer mode
- Shows security warnings
- Chrome periodically disables sideloaded extensions
- Terrible UX for non-technical users

### Extension-to-Web-App Communication

If an extension captures audio, it can pass it to the web app via:
- `window.postMessage()` from content script to page
- Creating a Blob URL and injecting it into the page DOM
- Using a localhost WebSocket server

**Sources:**
- [Chrome tabCapture API](https://developer.chrome.com/docs/extensions/reference/api/tabCapture)
- [Chrome Web Store troubleshooting violations](https://developer.chrome.com/docs/webstore/troubleshooting)
- [How to build a Chrome recording extension](https://www.recall.ai/blog/how-to-build-a-chrome-recording-extension)
- [Tab audio capture to server](https://www.codestudy.net/blog/chrome-extension-capture-tab-audio/)

---

## 6. Service Worker Interception

**Feasibility: Not Feasible**

### Why It Fails

Service workers can only intercept requests within their own origin scope. A service worker registered on `stemscribe.io` can intercept requests from `stemscribe.io` pages, but:

- **Cannot intercept YouTube's media requests** (different origin)
- **Cannot intercept requests made by YouTube's iframe** (cross-origin)
- YouTube's media is served from `*.googlevideo.com`, completely outside StemScribe's scope

### What Service Workers CAN Do

- Cache audio files that users upload to StemScribe (useful for offline playback of stems)
- Serve cached stems and chord charts
- Handle range requests for audio playback from StemScribe's own resources

### Bottom Line

Service workers are a non-starter for intercepting YouTube audio. They are architecturally limited to same-origin requests by design.

**Sources:**
- [Mux: Service workers are underrated](https://www.mux.com/blog/service-workers-are-underrated)
- [Chrome: Serving cached audio and video](https://developer.chrome.com/docs/workbox/serving-cached-audio-and-video)

---

## 7. User Upload from Local Music Library

**Feasibility: Feasible -- BEST APPROACH**

### Standard File Input

The simplest and most legally defensible approach. Users select audio files from their device:

```html
<input type="file" accept="audio/*" id="audioUpload" />
```

```javascript
const input = document.getElementById('audioUpload');
input.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  const formData = new FormData();
  formData.append('audio', file);

  const response = await fetch('/api/process', {
    method: 'POST',
    body: formData
  });
});
```

Supported formats: MP3, WAV, FLAC, M4A, OGG, AAC, AIFF -- whatever the user has.

### File System Access API (Advanced)

For a richer experience, the File System Access API lets users pick files or entire folders:

```javascript
const [fileHandle] = await window.showOpenFilePicker({
  types: [{
    description: 'Audio Files',
    accept: { 'audio/*': ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac'] }
  }]
});
const file = await fileHandle.getFile();
```

- Works in Chrome, Edge, Opera (Chromium-based browsers)
- NOT supported in Firefox or Safari (falls back to `<input type="file">`)
- Provides persistent access to files (with permission) for re-processing

### Drag and Drop

```javascript
const dropZone = document.getElementById('dropzone');
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  // Process file...
});
```

### Where Users Get Audio Files

Users who want to practice along with songs likely have music from:
- **iTunes/Apple Music purchases** (DRM-free M4A since 2009)
- **Downloaded MP3s** (from Amazon, Bandcamp, etc.)
- **CD rips** (MP3/FLAC/WAV)
- **Bandcamp purchases** (FLAC/MP3/WAV)
- **YouTube Music Premium** (allows offline downloads, but in encrypted format)
- **Spotify downloads** (encrypted OGG, NOT accessible -- see below)

### Streaming Service Limitations

| Service | Can Users Export Audio Files? |
|---------|------------------------------|
| iTunes/Apple Music (purchased) | Yes -- DRM-free M4A/AAC |
| Apple Music (subscription) | No -- DRM protected (FairPlay) |
| Spotify | No -- encrypted OGG, not accessible outside app |
| YouTube Music Premium | No -- encrypted offline downloads |
| Amazon Music (purchased) | Yes -- DRM-free MP3 |
| Bandcamp | Yes -- multiple formats (FLAC, MP3, WAV) |
| Tidal | No -- DRM protected |
| CD Rips | Yes -- whatever format user chooses |

### Legal Advantages

- **StemScribe never touches YouTube** -- the user provides their own legally obtained audio
- **Fair use is strongest** when users process their own purchased music for personal practice
- **No DMCA liability** -- StemScribe is a tool processing user-provided content, similar to how Photoshop processes user-provided images
- **DMCA Safe Harbor applies** if StemScribe implements takedown procedures

**Sources:**
- [MDN: File System API](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API)
- [Chrome: File System Access API](https://developer.chrome.com/docs/capabilities/web-apis/file-system-access)
- [File System Access API + Web Audio API example](https://gist.github.com/Anoesj/7d560cfb51223dfb7120827fe9f153c6)
- [Can I Use: File System Access API](https://caniuse.com/native-filesystem-api)

---

## 8. Third-Party Proxy Services (Cobalt, etc.)

**Feasibility: Partially Feasible (but shifts liability, doesn't eliminate it)**

### How Cobalt Works

Cobalt (cobalt.tools) acts as a server-side proxy:
1. User provides a YouTube URL
2. Cobalt's servers fetch the audio from YouTube
3. Cobalt returns the audio to the user

It "works like a fancy proxy" -- the actual downloading happens on Cobalt's servers, not in the user's browser.

### Could StemScribe Use a Service Like This?

You could direct users to download audio via Cobalt, then upload it to StemScribe. But:
- Adds friction to the workflow (download elsewhere, then upload)
- Cobalt could shut down at any time (these services frequently get taken down)
- You'd be recommending users violate YouTube's ToS, which could create contributory liability

### Self-Hosted Proxy

Running your own CORS proxy (e.g., a Cloudflare Worker that fetches from YouTube on behalf of the client) technically makes the request "client-initiated" but the proxy server is still downloading from YouTube. This is exactly what StemScribe currently does -- it doesn't change the legal picture at all.

**Sources:**
- [Cobalt GitHub](https://github.com/imputnet/cobalt)
- [Cobalt Tools](https://cobalt.tools/)

---

## 9. Legal Analysis

### YouTube's Terms of Service

YouTube ToS Section 5(B) states:

> "You shall not download any Content unless you see a 'download' or similar link displayed by YouTube on the Service for that Content."

This applies to USERS, not just service providers. Both server-side and client-side downloading violate YouTube's ToS. **The legal risk does not fundamentally change based on where the download happens.**

### Does Client-Side Shift Liability?

**Short answer: Not significantly.**

- If StemScribe provides JavaScript code that extracts audio from YouTube (even if it runs in the user's browser), StemScribe is still facilitating the violation of YouTube's ToS
- Courts look at whether the service provider is "inducing" or "contributing to" infringement (see MGM v. Grokster)
- Providing a tool specifically designed to extract YouTube audio signals intent, regardless of where the code executes
- The DMCA's anti-circumvention provisions (Section 1201) may apply if any DRM/protection measures are bypassed

### DMCA Safe Harbor

Safe Harbor (Section 512) protects service providers when:
1. **Users** are the ones providing/uploading content
2. The provider doesn't have **actual knowledge** of infringement
3. The provider **doesn't benefit financially** from the infringement in a way tied to specific infringing material
4. The provider implements a **takedown procedure** and **repeat infringer policy**

**Key distinction:** Safe Harbor applies when StemScribe processes user-UPLOADED files. It does NOT apply when StemScribe (or its client-side code) actively downloads from YouTube, because then StemScribe is the direct actor, not a passive platform.

### The Strongest Legal Position

1. **User uploads their own audio file** (purchased MP3, CD rip, etc.)
2. StemScribe processes it (stem separation, chord detection) as a tool
3. This is analogous to Photoshop, Audacity, or any other media processing tool
4. Fair use for personal practice/education is the strongest argument

### Audio Home Recording Act (AHRA)

The AHRA does NOT protect internet-based downloads. It was written for analog and early digital recording devices. A 2025 Justia legal analysis confirmed: "The Audio Home Recording Act does not give you the legal right to download audio from YouTube -- even for personal use."

**Sources:**
- [YouTube Terms of Service (TLDRLegal)](https://www.tldrlegal.com/license/youtube-terms-of-service)
- [DMCA Safe Harbor overview (Justia)](https://www.justia.com/intellectual-property/copyright/docs/dmca/)
- [DMCA Safe Harbor (Copyright Alliance)](https://copyrightalliance.org/education/copyright-law-explained/the-digital-millennium-copyright-act-dmca/dmca-safe-harbor/)
- [AHRA and YouTube (Justia Q&A)](https://answers.justia.com/question/2025/05/18/does-audio-home-recording-act-allow-down-1061421)
- [Is YouTube to MP3 illegal? (Bridge Legal)](https://bridgelegal.org/is-youtube-mp3-illegal-what-you-need-know/)
- [AI Stem Extraction legal analysis (Northwestern JTIP)](https://jtip.law.northwestern.edu/2022/05/03/ai-stem-extraction-a-creative-tool-or-facilitator-of-mass-infringement/)
- [Protecting against copyright claims (DMLP)](https://www.dmlp.org/legal-guide/protecting-yourself-against-copyright-claims-based-user-content)

---

## 10. Recommended Approaches

### Tier 1: DO THIS (Feasible, Legal, Good UX)

#### A. User File Upload (Primary Path)
- **How:** `<input type="file" accept="audio/*">` + drag-and-drop zone
- **Legal risk:** Minimal -- StemScribe is a tool processing user-provided content
- **UX:** Simple, universally understood
- **Implementation:** Trivial -- standard HTML/JS, works in all browsers
- **Messaging:** "Upload your own music to practice with" / "Bring your own tracks"

#### B. File System Access API (Enhanced UX for Chrome/Edge)
- **How:** `showOpenFilePicker()` with audio file type filters
- **Legal risk:** Same as file upload -- minimal
- **UX:** Slightly better than `<input>` -- native file picker with filters
- **Caveat:** Not supported in Firefox/Safari; needs fallback to `<input>`

### Tier 2: CONSIDER (Feasible, Some Tradeoffs)

#### C. YouTube Link + Server-Side Download (Current Approach)
- **How:** Keep doing what you're doing, but add DMCA Safe Harbor compliance
- **Legal risk:** Moderate, but manageable at current scale
- **Action items:**
  - Register a DMCA agent with the Copyright Office ($6 filing fee)
  - Add a DMCA takedown page to stemscribe.io
  - Add a repeat infringer policy to ToS
  - Consult with Lindsay Spiller or Jesse Morris (already planned)
- **Why keep it:** It's the killer UX feature that differentiates StemScribe

#### D. Hybrid: YouTube Playback + User Upload for Processing
- **How:** Use YouTube IFrame API for reference playback (legal), but require user to upload their own audio file for stem separation
- **Legal risk:** Low -- YouTube embed is sanctioned by YouTube's ToS
- **UX:** "Listen to the song on YouTube, then upload your copy for stems"
- **Tradeoff:** Adds friction, but is the most legally defensible model

### Tier 3: AVOID (Not Worth the Effort)

#### E. Browser Extension
- Rejected from Chrome Web Store (Google policy)
- Firefox-only has ~3% market share
- Sideloading is terrible UX
- Still legally questionable

#### F. getDisplayMedia Tab Audio Capture
- Terrible UX (multi-step, user must check "share audio")
- Real-time only (4-min song = 4-min wait)
- Chrome/Edge only
- Confusing for non-technical users

#### G. Client-Side CORS Proxy / Service Worker
- Does not actually shift liability
- Same as server-side download with extra complexity
- Service workers cannot intercept cross-origin requests

---

## Summary Matrix

| Approach | Technically Feasible | Legally Safe | Good UX | Recommendation |
|----------|---------------------|--------------|---------|----------------|
| Direct browser extraction | No | N/A | N/A | Impossible |
| YouTube IFrame + Web Audio | No | N/A | N/A | Impossible |
| captureStream() | No | N/A | N/A | Impossible |
| getDisplayMedia tab capture | Partially | Gray area | Terrible | Avoid |
| Chrome Extension | Partially | Gray area | Bad | Avoid |
| Firefox Extension | Yes | Gray area | Niche | Avoid |
| Service Worker intercept | No | N/A | N/A | Impossible |
| User file upload | Yes | Safe | Good | **DO THIS** |
| File System Access API | Yes (Chrome) | Safe | Good | **DO THIS** |
| Server-side YT download | Yes (current) | Moderate risk | Great | **Keep + add DMCA compliance** |
| Cobalt/third-party proxy | Partially | Risky | Awkward | Avoid |

---

## Final Verdict

**Client-side YouTube audio extraction is not technically possible** due to CORS restrictions. Every approach that actually works requires either (a) a server-side component that downloads from YouTube, or (b) the user providing their own audio file.

**The best path forward is a dual approach:**
1. **Add user file upload** as the primary, legally safe path -- most musicians own their music
2. **Keep the YouTube URL feature** but add DMCA Safe Harbor compliance (agent registration, takedown page, repeat infringer policy)
3. **Talk to the lawyer** (already planned) about the YouTube feature specifically

The idea that client-side downloading would shift legal liability is a misconception. If your code (running anywhere) facilitates downloading from YouTube, you are contributing to a ToS violation. The only clean legal path is user-provided audio files.
