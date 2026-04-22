# YouTube API & Music Licensing Research for StemScriber

**Date:** 2026-03-26
**Purpose:** Research legal alternatives to yt-dlp for audio sourcing in a music practice app
**Status:** RESEARCH ONLY -- no code changes

---

## Table of Contents
1. [YouTube Data API v3](#1-youtube-data-api-v3)
2. [YouTube Content ID](#2-youtube-content-id)
3. [YouTube Music API](#3-youtube-music-api)
4. [How Shazam & SoundHound Handle Licensing](#4-how-shazam--soundhound-handle-licensing)
5. [Music Licensing Companies & APIs](#5-music-licensing-companies--apis)
6. [Specific Licensing Paths](#6-specific-licensing-paths)
7. [YouTube Embed vs Download Legal Distinction](#7-youtube-embed-vs-download-legal-distinction)
8. [Apple MusicKit API](#8-apple-musickit-api)
9. [Spotify API](#9-spotify-api)
10. [DMCA Safe Harbor & User-Upload Model](#10-dmca-safe-harbor--user-upload-model)
11. [How Moises Does It](#11-how-moises-does-it)
12. [Practical Options for Solo Founder](#12-practical-options-for-solo-founder)

---

## 1. YouTube Data API v3

### What It Actually Allows
The YouTube Data API v3 provides access to **metadata only**: video titles, descriptions, thumbnails, channel info, playlists, comments, captions/subtitles, and search. It does NOT provide access to audio or video streams.

### Key Restrictions (from official YouTube API ToS)

**Explicitly prohibited:**
- Downloading videos or audio for offline playback outside YouTube Premium
- Separating audio tracks from video
- Modifying audio or video portions of content
- Offering MP3 files extracted from YouTube videos
- Accessing content through any means other than official playback pages, the embeddable player, or explicitly authorized methods

**Specific policy language:** "API Clients must not allow users to download videos for offline play outside of the YT Premium experience, nor offer users the ability to download or separate audio tracks or allow users to modify the audio or video portions of a video."

### Cost
- Free tier: 10,000 quota units/day
- Each search costs 100 units, so ~100 searches/day free
- Can request quota increases via audit form

### Verdict for StemScriber
**NOT VIABLE for audio sourcing.** The API is metadata-only. Using yt-dlp alongside the API still violates ToS. There is no legitimate path to get raw audio through YouTube's official API.

### Sources
- [YouTube API ToS](https://developers.google.com/youtube/terms/api-services-terms-of-service)
- [YouTube Developer Policies](https://developers.google.com/youtube/terms/developer-policies)
- [Developer Policies Guide](https://developers.google.com/youtube/terms/developer-policies-guide)
- [Quota and Compliance Audits](https://developers.google.com/youtube/v3/guides/quota_and_compliance_audits)

---

## 2. YouTube Content ID

### What It Is
Content ID is YouTube's rights management system that allows copyright holders to identify and manage their content on YouTube. It scans uploaded videos against a database of reference files submitted by rights holders.

### Access Requirements
- You must own **exclusive rights** to the content
- Access is typically restricted to large companies, networks, and partnered creators
- Individual creators generally need to go through a multi-channel network (MCN) or music distribution service
- Strict eligibility criteria -- YouTube evaluates whether the copyright owner's content can be claimed through Content ID

### Is It Relevant for StemScriber?
**NO.** Content ID is for rights holders to protect their content on YouTube, not for third-party apps to access or process audio. It is designed for content identification and monetization/blocking decisions, not for licensing audio to external services.

### Sources
- [YouTube Content ID API](https://developers.google.com/youtube/partner)
- [Content ID for Music Partners](https://support.google.com/youtube/answer/2822002)
- [Content ID Qualification](https://support.google.com/youtube/answer/1311402)

---

## 3. YouTube Music API

### Official API Status
**YouTube Music does NOT have an official public API.** Google has never released one.

### Unofficial APIs
- **ytmusicapi** (Python): Emulates web client requests using browser cookies for authentication. Open-source, ~1.11.x versions. Can search, get playlists, library content, etc.
- **ytmusicapiJS** (JavaScript): Same concept, different language
- Various npm packages exist

### Legal Risk
All unofficial YouTube Music APIs violate YouTube's ToS because they access YouTube data through means other than the official API or embeddable player. Using these for a commercial product carries significant legal risk.

### Verdict for StemScriber
**NOT VIABLE.** No official API exists. Unofficial ones violate ToS just like yt-dlp does.

### Sources
- [ytmusicapi GitHub](https://github.com/sigma67/ytmusicapi)
- [ytmusicapi Docs](https://ytmusicapi.readthedocs.io/)
- [Musicfetch - Does YouTube Music Have an API?](https://musicfetch.io/services/youtube-music/api)

---

## 4. How Shazam & SoundHound Handle Licensing

### Shazam's Approach
1. **Audio Fingerprinting:** Captures ~20 seconds of audio, creates a spectrogram, finds peaks, generates fingerprints
2. **Database Licensing:** Licensed reference database through a mix of:
   - Direct licensing from record labels (limited, non-exclusive rights to create fingerprints)
   - Aggregation from public and commercial sources
   - Negotiated deals with music industry intermediaries
3. **Revenue Model:** Drives users to streaming services (affiliate revenue), displays ads, premium features
4. **Apple Acquisition:** Simplified some licensing through deeper Apple Music integration
5. **Key Detail:** Labels grant rights to *create fingerprints*, not to download/stream/distribute audio. Streaming previews require additional performance rights licenses.

### SoundHound's Approach
1. **User-contributed fingerprints:** Early Midomi used user-sung samples to build fingerprint database
2. **Direct licensing:** Deals with rights holders for reference database
3. **Revenue model:** Licensing voice AI technology to businesses (automotive, customer service), not music distribution
4. **Key Detail:** SoundHound identifies music; it doesn't distribute it. The legal bar is much lower for recognition-only services.

### Relevance to StemScriber
These companies do NOT process/distribute copyrighted audio. They only:
- Create fingerprints (tiny mathematical representations)
- Link users to licensed streaming services
- Display metadata

StemScriber's use case (downloading full audio, separating into stems, playing back) is fundamentally different and requires much more extensive licensing.

### Sources
- [How Shazam Works - Toptal](https://www.toptal.com/developers/algorithms/shazam-it-music-processing-fingerprinting-and-recognition)
- [Shazam/SoundHound Licensing - Quora](https://www.quora.com/How-did-Shazam-and-SoundHound-license-their-database-s-of-songs)
- [SoundHound Business Model - Vizologi](https://vizologi.com/business-strategy-canvas/soundhound-business-model-canvas/)

---

## 5. Music Licensing Companies & APIs

### Audible Magic
- **What:** Content identification for social networks, labels, publishers, streaming services, fitness/gaming apps
- **Services:** Music identification, DMCA takedown reduction, catalog fulfillment APIs
- **Pricing:** Not publicly listed; enterprise sales model. Contact helpdesk@audiblemagic.com
- **Verdict:** Designed for platforms hosting user content, not for sourcing audio. Could be useful if StemScriber shifts to user-upload model for content identification.
- **Source:** [audiblemagic.com](https://www.audiblemagic.com/)

### Pex (acquired by Vobile, April 2025)
- **What:** AI-powered content recognition (ACR) for identifying music in audio/video
- **Services:** Discovery (copyright search engine), Attribution Engine (free for rightsholders)
- **Pricing:** Starting at $1/file/month for tracking
- **Verdict:** Content identification, not audio sourcing. Not directly useful for StemScriber's needs.
- **Source:** [pex.com](https://pex.com/)

### ACRCloud
- **What:** Audio fingerprinting and recognition API with 150M+ track database
- **Services:** Song recognition, humming recognition, broadcast monitoring
- **Pricing:** 14-day free trial, then tiered by volume. Roughly $32/10K requests based on listed pricing.
- **Verdict:** Could identify songs users upload, but doesn't provide audio sourcing.
- **Source:** [acrcloud.com](https://www.acrcloud.com/)

### Songclip
- **What:** Licensed music clips (5-30 seconds) for integration into apps
- **Partners:** UMG, WMG, Warner Chappell, Kobalt, BMG, 7,500+ rights owners, NMPA partnership
- **API:** RESTful API, ~48 hour integration, 3M+ music clips
- **Pricing:** Not public. Contact api@songclip.com
- **Verdict:** Only provides short clips (5-30 sec), not full songs. Not suitable for practice/stem separation.
- **Source:** [songclip.com/api](https://www.songclip.com/api)

### Songtradr
- **What:** Largest B2B music licensing marketplace
- **Services:** Rights management, monetization, AI-guided music access
- **API:** Available through acquired Tunefind platform (data services)
- **Pricing:** Enterprise; not public
- **Verdict:** Designed for sync licensing (ads, films, games), not for music practice apps.
- **Source:** [songtradr.com](https://www.songtradr.com/)

---

## 6. Specific Licensing Paths

### MLC Blanket License (Becoming a Digital Service Provider)

**What:** The Mechanical Licensing Collective administers blanket mechanical licenses for interactive streaming/downloads in the US under the Music Modernization Act.

**Requirements:**
- Submit a Notice of License (NoL) to The MLC
- Monthly usage reports and mechanical royalty payments
- Fund MLC operating costs proportionally (Administrative Assessment)
- Annual minimum fee (reportedly $5,000/month or $60,000/year as a floor)

**What it covers:** Mechanical rights only (reproduction of musical compositions). You still need:
- **Performance licenses** from PROs (ASCAP, BMI, SESAC) -- typically 1.5-5% of revenue
- **Master recording licenses** from record labels (Universal, Sony, Warner + independents) -- negotiated directly, typically 50-70% of revenue
- **Synchronization licenses** if combining audio with visual elements

**Cost estimate for a small DSP:**
- MLC assessment: ~$60K+/year minimum
- PRO licenses: $500-5,000+/year (based on revenue)
- Label deals: Revenue share (major labels typically want 50-70% of streaming revenue)
- Legal fees to negotiate: $10K-50K+

**Verdict:** Designed for Spotify-scale operations. The minimum costs ($60K+/year before label deals) make this impractical for a solo founder at StemScriber's current scale.

**Sources:**
- [MLC - How It Works](https://www.themlc.com/how-it-works)
- [MLC DSP FAQs](https://www.themlc.com/dsp-faqs)
- [Digital Licensee Coordinator FAQ](https://digitallicenseecoordinator.org/faq/)
- [37 CFR Part 390](https://www.ecfr.gov/current/title-37/chapter-III/subchapter-E/part-390)

### Harry Fox Agency (HFA) / Songfile

**What:** Mechanical licensing for smaller quantities of reproductions.

**Costs:**
- Per-composition processing fee: $15 for first 5, $13 for additional
- Covers physical copies and downloads, not interactive streaming
- Affiliation fee: $125-$150

**Verdict:** Designed for cover song CDs and downloads, not interactive streaming apps. Would not cover StemScriber's use case of on-demand stem playback.

**Sources:**
- [harryfox.com](https://www.harryfox.com/)
- [HFA on Songtrust](https://help.songtrust.com/knowledge/what-is-the-harry-fox-agency)

### PRO Licenses (ASCAP, BMI, SESAC)

**What:** Performance rights -- needed when music is performed publicly (including streaming).

**Cost for small businesses:**
- ASCAP: Starts around $400-700/year for small digital services
- BMI: Similar range, revenue-based
- SESAC: Invitation-only, negotiate directly

**Verdict:** Necessary if you stream audio, but these only cover the performance right. You need mechanical + master recording licenses on top of this.

### Fair Use Analysis for Stem Separation

**Four-factor test applied to StemScriber:**

1. **Purpose and character of use:** Transformative? Arguable -- stems enable practice, not just redistribution. But the user is still listening to the copyrighted recording. Commercial use weighs against fair use.

2. **Nature of the copyrighted work:** Creative musical works get strong copyright protection. Weighs against fair use.

3. **Amount used:** The entire work is downloaded and processed. Weighs heavily against fair use.

4. **Market effect:** Stem separation competes with the original market (users may not buy/stream the song elsewhere). Also competes with official remix/stems markets. Weighs against fair use.

**Legal analysis from Northwestern JTIP:** AI stem extraction tools may face both direct liability (the tool performing the separation) and secondary liability (facilitating user infringement). The creation of stems likely constitutes creation of "derivative works" under Section 106.

**Educational use exemption (Section 110):** Only applies to face-to-face instruction at nonprofit educational institutions, or distance learning under the TEACH Act. A commercial practice app does NOT qualify.

**Bottom line:** Fair use is extremely unlikely to protect StemScriber's current model of downloading copyrighted audio from YouTube and separating it into stems. A lawyer would almost certainly advise against relying on fair use.

**Sources:**
- [Northwestern JTIP - AI Stem Extraction](https://jtip.law.northwestern.edu/2022/05/03/ai-stem-extraction-a-creative-tool-or-facilitator-of-mass-infringement/)
- [17 U.S.C. Section 110](https://www.law.cornell.edu/uscode/text/17/110)
- [Bray & Krais - Stem Separation](https://www.brayandkrais.com/stem-separating-ai-is-revolutionising-the-music-industry/)
- [U.S. Copyright Office Fair Use Index](https://www.copyright.gov/fair-use/)

---

## 7. YouTube Embed vs Download Legal Distinction

### Embedding (Legal)
- YouTube grants a sublicense for embedding through their official IFrame Player API
- Content stays on YouTube's servers -- your site/app just displays the player
- You can control playback (play, pause, seek) via JavaScript
- You can create audio-only players by hiding the video element (technically violates ToS UI requirements)
- **You CANNOT access raw audio data from an embedded player** (DRM + cross-origin restrictions)

### Downloading (Violates ToS)
- YouTube ToS: "You shall not download any Content unless you see a 'download' or similar link displayed by YouTube on the Service for that Content"
- yt-dlp, youtube-dl, and all third-party downloaders violate ToS
- This is a ToS violation (breach of contract) more than a copyright violation per se, but rights holders can pursue copyright claims independently

### The Gray Area: Server-Side Audio Streaming
- Libraries like `youtube-audio-stream` (Node.js) pipe audio server-side
- Still violates YouTube ToS (accessing content through unauthorized means)
- No technical distinction that makes this legal vs. yt-dlp

### Verdict for StemScriber
**Embedding is the only ToS-compliant option, but it doesn't give you audio data to process.** You can use the YouTube player for playback, but you cannot extract audio for stem separation through any YouTube-authorized method.

### Sources
- [YouTube IFrame Player API](https://developers.google.com/youtube/iframe_api_reference)
- [YouTube ToS - TLDRLegal](https://www.tldrlegal.com/license/youtube-terms-of-service)
- [SDNY YouTube Embedding Ruling](https://copyrightlately.com/sdny-social-media-embedding/)

---

## 8. Apple MusicKit API

### What It Offers
- Playback of Apple Music catalog in third-party apps
- Access to metadata, search, playlists, user library
- Requires Apple Music subscription from the user
- Free for developers (part of Apple Developer Program, $99/year)

### Raw Audio Access
- **Standard MusicKit does NOT expose raw audio/PCM data** for DRM-protected songs
- Playback goes through MPMusicPlayerController -- no way to tap into audio samples
- **Exception:** Approved DJ apps (djay, Serato, rekordbox) have special entitlements that appear to grant decoded PCM access
- How to get this entitlement is not publicly documented; likely requires direct relationship with Apple

### Verdict for StemScriber
**Not viable for stem separation.** Even if you could get the special DJ entitlement (unlikely for a solo founder), Apple's terms almost certainly prohibit modifying, separating, or redistributing the audio. The DJ apps use it for real-time mixing, not extraction.

However, MusicKit could be useful for:
- Song search and metadata
- Playback of original tracks alongside stems (if stems are sourced legally)
- User library integration

### Sources
- [MusicKit - Apple Developer](https://developer.apple.com/musickit/)
- [Apple Music API Docs](https://developer.apple.com/documentation/applemusicapi/)
- [Apple Developer Forums - DJ Apps](https://developer.apple.com/forums/thread/803991)
- [Apple Developer Forums - Raw Audio](https://developer.apple.com/forums/thread/782902)

---

## 9. Spotify API

### Current Status (Post-November 2024 Restrictions)
Spotify significantly restricted their Web API in November 2024. New apps lost access to:
- Audio Analysis (track structure, rhythm, tempo details)
- Audio Features (danceability, energy, acoustic characteristics)
- Recommendations
- 30-second preview URLs

### What Remains Available
- Search, metadata, playlists, user library management
- Playback control (requires Spotify Premium + official SDK)
- Basic track/artist/album info

### Raw Audio Access
**No.** Spotify has never provided raw audio access through their API. All playback goes through their DRM-protected player. The recent restrictions made the API even more limited.

### Verdict for StemScriber
**Not viable for audio sourcing.** Spotify's API is metadata/playback-control only. No audio extraction possible.

### Sources
- [Spotify Web API Changes (Nov 2024)](https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api)
- [Spotify API Restrictions - TechCrunch](https://techcrunch.com/2024/11/27/spotify-cuts-developer-access-to-several-of-its-recommendation-features/)

---

## 10. DMCA Safe Harbor & User-Upload Model

### How DMCA Safe Harbor Works
Under Section 512(c), online service providers that host user-uploaded content are protected from copyright liability IF they:
1. Have no actual knowledge of infringement
2. Do not financially benefit directly from infringement
3. Designate a DMCA agent and respond expeditiously to takedown notices
4. Have and enforce a repeat infringer policy

### How This Applies to StemScriber
If StemScriber shifts to a **user-upload model** (users provide their own audio files), the DMCA safe harbor could apply:
- StemScriber processes files that users upload
- StemScriber does not know whether the user has rights to the file
- StemScriber responds to DMCA takedowns
- StemScriber has a copyright policy and repeat infringer policy

**Critical distinction:** The safe harbor only applies when *someone other than the service provider* is the direct infringer. If StemScriber itself downloads from YouTube (via yt-dlp), StemScriber IS the direct infringer and safe harbor does not apply.

### Requirements to Qualify
- Register a DMCA agent with the U.S. Copyright Office (~$6 filing fee)
- Implement takedown/counter-notice procedures
- Have a clear Terms of Service requiring users to have rights to uploaded content
- Enforce a repeat infringer policy

### Sources
- [DMCA Safe Harbor - Copyright Alliance](https://copyrightalliance.org/education/copyright-law-explained/the-digital-millennium-copyright-act-dmca/dmca-safe-harbor/)
- [Section 512 - U.S. Copyright Office](https://www.copyright.gov/512/)
- [DMCA Safe Harbor Overview - Justia](https://www.justia.com/intellectual-property/copyright/copyright-safe-harbor/)

---

## 11. How Moises Does It

### Business Model
Moises is the closest competitor/analog to StemScriber. Here is how they handle licensing:

**User-upload model:**
- Users upload their own audio files (MP3, WAV, FLAC, M4A, etc.)
- Moises does NOT source audio from YouTube or any streaming service
- Users cannot paste streaming URLs -- Moises explicitly blocks Spotify/Apple Music/YouTube URLs
- Terms of Service state: users must not submit content they don't own or aren't authorized to use

**DMCA compliance:**
- Copyright responsibility is placed on the user
- Moises processes files as a tool, similar to how Photoshop processes images
- They maintain DMCA takedown procedures

**Revenue model:**
- Freemium: 5 free files/month (5 min max each), processed tracks expire in 3 days
- Premium subscription tiers
- Raised $40M Series A (Music AI parent company), 50M+ users

**Key insight:** Moises has raised significant VC funding and operates at massive scale, yet still uses the user-upload model rather than licensing music directly. This strongly suggests that direct music licensing is prohibitively expensive even for well-funded startups.

### What Moises Does NOT Do
- Download from YouTube
- Integrate with streaming service APIs for audio
- License music catalogs from labels
- Claim fair use

### Sources
- [Moises Terms & Conditions](https://help.moises.ai/hc/en-us/articles/7401394754962-Terms-Conditions)
- [Moises Streaming URL Policy](https://help.moises.ai/hc/en-us/articles/18322457396508)
- [Music AI $40M Series A](https://www.musicbusinessworldwide.com/music-ai-raises-40m-in-series-a-round-as-its-moises-platform-hits-50m-users/)

---

## 12. Practical Options for Solo Founder

Ranked by feasibility, cost, and legal safety:

### Option 1: User-Upload Model (RECOMMENDED)
**Cost:** ~$6 (DMCA agent filing) + lawyer time
**Legal safety:** HIGH
**How it works:**
- Users upload their own audio files (from their personal music library, CDs they own, etc.)
- StemScriber processes the files as a tool
- Place legal responsibility on users via ToS
- Implement DMCA takedown procedures
- Register DMCA agent with Copyright Office

**Pros:**
- This is exactly what Moises does (50M+ users, $40M funding)
- DMCA safe harbor protection
- No licensing costs
- No dependency on YouTube or any streaming platform

**Cons:**
- Higher friction for users (can't just paste a URL)
- Users need to already have the audio file
- Some users may upload files they don't own (their problem, not yours under DMCA)

**Implementation:** Remove yt-dlp dependency. Add file upload UI. Update ToS. Register DMCA agent.

---

### Option 2: Hybrid -- User Upload + YouTube Embed for Discovery
**Cost:** ~$6 + lawyer time
**Legal safety:** MEDIUM-HIGH
**How it works:**
- YouTube IFrame player for song discovery and playback of originals
- User uploads their own audio file for stem separation
- YouTube embed is for reference/listening only; processing happens on uploaded files

**Pros:**
- Better UX than pure upload (users can preview songs via YouTube)
- YouTube embed is fully ToS-compliant
- Still DMCA safe harbor protected for uploads

**Cons:**
- Two-step process (find on YouTube, then upload separately)
- Can't process YouTube audio

---

### Option 3: Partner with a Licensed Music Service
**Cost:** Revenue share + legal fees ($10K-50K to negotiate)
**Legal safety:** HIGH (if done right)
**How it works:**
- Partner with a service like Songclip, or negotiate directly with a label/distributor
- Access licensed audio through their API
- Pay per-stream or revenue share

**Pros:**
- Fully legal
- Could provide a seamless user experience

**Cons:**
- Extremely expensive for a solo founder
- Major labels won't negotiate with tiny startups
- Revenue share models (50-70% to labels) leave little margin
- Months of legal negotiation

---

### Option 4: Apple MusicKit Integration (Playback Only)
**Cost:** $99/year (Apple Developer Program)
**Legal safety:** HIGH (for playback), N/A (for stems)
**How it works:**
- Use MusicKit for song search, metadata, and original playback
- Still need user-uploaded files for stem separation
- Apple Music subscription required from users

**Pros:**
- Professional integration with massive catalog
- Good for metadata and original track playback

**Cons:**
- Cannot extract audio for stem separation
- Requires users to have Apple Music subscription
- Doesn't solve the core audio sourcing problem

---

### Option 5: Become a Licensed DSP (NOT RECOMMENDED for solo founder)
**Cost:** $60K+/year minimum (MLC) + PRO fees + label deals + legal
**Legal safety:** HIGHEST
**How it works:**
- Register as Digital Service Provider with MLC
- Get blanket mechanical license
- Negotiate master recording licenses with labels
- Get performance licenses from ASCAP/BMI/SESAC

**Cons:**
- Minimum ~$60K/year before any label deals
- Labels want 50-70% revenue share
- Months of negotiation, expensive lawyers
- Designed for Spotify-scale operations

---

### Option 6: Continue Current Approach (NOT RECOMMENDED)
**Cost:** $0 now, potentially catastrophic later
**Legal safety:** VERY LOW
**How it works:** Keep using yt-dlp

**Risks:**
- Violates YouTube ToS (account termination, API key revocation)
- Copyright infringement liability (statutory damages up to $150K per work for willful infringement)
- NMPA has an active app enforcement initiative targeting unlicensed music apps
- No DMCA safe harbor when the service itself is doing the downloading
- Fair use defense extremely unlikely to succeed (see Section 6)

---

## Summary Recommendation

**Switch to the User-Upload Model (Option 1), potentially with YouTube Embed for discovery (Option 2).**

This is what Moises does. This is what every successful stem separation service does. It is the only model that:
- A solo founder can afford
- Provides DMCA safe harbor protection
- Does not require licensing deals with labels/publishers
- Has been validated at scale (Moises: 50M users)

### Immediate Action Items
1. **Lawyer consultation** (already planned -- see lawyer-call-prep.md)
2. Ask lawyer specifically about the user-upload + DMCA safe harbor model
3. Register DMCA agent with U.S. Copyright Office ($6)
4. Draft ToS requiring users to have rights to uploaded content
5. Build file upload UI to replace YouTube URL input
6. Optionally add YouTube IFrame embed for song discovery/preview
7. Remove yt-dlp from production code

### Questions for Lawyer
1. Does the user-upload + DMCA safe harbor model adequately protect StemScriber?
2. Is stem separation itself a "derivative work" that creates additional liability for the tool provider?
3. Does the Moises model (50M users, VC-backed, no label licenses) validate this approach?
4. What ToS language is needed to properly disclaim liability?
5. Should we proactively implement content identification (e.g., ACRCloud) to detect copyrighted uploads?
6. Any risk from previously processed songs via yt-dlp? Should those cached files be deleted?
