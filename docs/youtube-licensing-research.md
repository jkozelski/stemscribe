# YouTube Audio Licensing Research for StemScriber
**Date:** 2026-03-26
**Status:** URGENT -- product viability depends on this
**Budget constraint:** Solo founder, <$1,000/month

---

## Executive Summary

**The hard truth:** There is NO legal way to download/extract audio from YouTube for processing in a third-party app. YouTube's API Terms of Service explicitly prohibit it. Every app in this space either (a) uses embedded YouTube players without downloading, (b) requires users to upload their own files, or (c) has direct licensing deals worth six figures+.

**The realistic path for StemScriber:** A hybrid model combining user-uploaded audio (BYOA) with embedded YouTube playback for discovery, plus free/CC-licensed content to seed the library. This can be built in 30 days.

---

## 1. YouTube API -- What's Actually Allowed

### Prohibited (explicitly in YouTube Developer Policies)
- Downloading or extracting audio from videos
- Separating audio or video components
- Offering MP3 or audio files derived from YouTube content
- Any form of "stream ripping" or audio capture
- Modifying audio/video portions of a video

### Allowed
- Embedded YouTube iframe player (with YouTube branding visible)
- Playback control via iframe API (play, pause, seek, volume)
- Metadata retrieval (title, description, thumbnails)
- Search functionality

### Enforcement
- API key revocation
- Google account termination
- Legal action (see yt-dlp lawsuits below)

### YouTube Content ID API
- Restricted to YouTube Partner Program members
- Designed for rights holders to manage their content
- NOT available to third-party developers
- No paid tier exists for audio access

**Sources:**
- [YouTube Developer Policies](https://developers.google.com/youtube/terms/developer-policies)
- [YouTube API Terms of Service](https://developers.google.com/youtube/terms/api-services-terms-of-service)

---

## 2. Legal Risk of yt-dlp / youtube-dl

### Active Lawsuits
- Sony, Warner, and Universal sued youtube-dl's hosting provider (Uberspace) in Germany -- **won**, site ordered offline
- 2026 lawsuit invokes DMCA Section 1201, arguing YouTube's technological measures qualify as effective access controls
- The RIAA previously issued a DMCA takedown against youtube-dl on GitHub (later reversed by EFF)

### Risk Assessment for StemScriber
- **HIGH RISK** for commercial use. Using yt-dlp in a paid product that processes copyrighted music is exactly the use case major labels are suing over
- Not a gray area -- it's the specific thing being litigated right now
- A solo founder in Charleston SC would be an easy target for a cease-and-desist
- Even if you "win" legally, defending costs $50K-$200K+

**Bottom line: Stop using yt-dlp for the production product immediately.**

---

## 3. How Competitors Handle This

### Chordify (36M+ songs, Netherlands-based)
| Aspect | Detail |
|--------|--------|
| Audio source | Embedded YouTube iframe player -- does NOT download audio |
| Processing | Server-side chord analysis (how they get audio to analyze is unclear -- possibly pre-YouTube-API-restrictions legacy) |
| Output | Chord diagrams, MIDI chords (not original audio) |
| Offline mode | Chord diagrams + metronome only, NO original audio |
| Legal model | YouTube API ToS compliant for playback; chord progressions are not copyrightable |
| Pricing | Free tier + Premium subscription |

### Moises ($10M+ raised, VC-backed)
| Aspect | Detail |
|--------|--------|
| Audio source | **User uploads their own audio files** (BYOA model) |
| Streaming URLs | Explicitly rejected -- "streaming platforms use DRM to prevent extraction" |
| Processing | AI stem separation on user-provided files |
| Legal model | User responsible for having rights to uploaded content |
| ToS | Users must own or have permission to use uploaded content |

### Yousician ($28M+ raised, Finland-based)
| Aspect | Detail |
|--------|--------|
| Audio source | **Licensed directly from publishers and labels** |
| History | Removed user upload feature in 2018, replaced with licensed songs |
| Legal model | Direct licensing deals with major publishers |
| Cost | Massive -- requires significant funding |
| Pricing | $29.99/mo or $179.99/yr |

### Songsterr (Guitar Tabs LLC, US-based)
| Aspect | Detail |
|--------|--------|
| Content | Guitar tabs and chord charts (NOT audio) |
| Legal model | Licensed from music publishers, pays ~50% of revenue as royalties |
| Pricing | $9.90/mo for Plus features |

### Key Takeaway
- **Funded companies** (Yousician) license directly -- costs millions
- **Smart bootstrapped companies** (Moises, Chordify) either use BYOA or embedded players
- **Nobody** legally downloads YouTube audio for processing

---

## 4. Realistic Options for StemScriber

### Option A: BYOA (Bring Your Own Audio) -- RECOMMENDED
**How it works:** Users upload MP3/WAV/FLAC files they already own. StemScriber processes them.

| Aspect | Detail |
|--------|--------|
| Legal basis | Users are responsible for having rights to their files |
| DMCA protection | Register as DMCA agent, implement takedown process |
| Cost to implement | $0 (already have stem separation working) |
| User friction | Medium -- users need files on their device |
| Precedent | Moises, LALAL.AI, iZotope all use this model |

**Requirements for legal protection:**
1. Terms of Service stating users must own/have rights to uploaded content
2. DMCA designated agent registered with US Copyright Office ($6 one-time fee)
3. Repeat infringer policy in ToS
4. Takedown process (respond within 24-48 hours)
5. Don't store/share processed files beyond the user's account

**This is what Moises does. It works. It's legally defensible.**

### Option B: Embedded YouTube Player + Chord Analysis Only
**How it works:** Use YouTube iframe to play audio in-browser. Display chord charts synced to playback. No stem separation from YouTube content.

| Aspect | Detail |
|--------|--------|
| Legal basis | YouTube iframe API is explicitly allowed |
| What you CAN do | Play video, sync chord display, show lyrics |
| What you CANNOT do | Stem separation, audio download, tempo change |
| Cost | $0 (YouTube API is free up to quota) |
| User experience | Good for chord practice, bad for stem practice |
| Precedent | Chordify does exactly this |

### Option C: Apple Music / MusicKit Integration
**How it works:** Use Apple's MusicKit to stream Apple Music content in your app.

| Aspect | Detail |
|--------|--------|
| Audio access | Streaming only -- cannot extract or process audio server-side |
| Requirement | Users need Apple Music subscription ($10.99/mo) |
| Stem separation | NOT possible -- MusicKit doesn't expose raw audio buffers |
| Advanced features | Tempo control available only to approved partners |
| Cost | Free API, but limited to Apple ecosystem |

**Verdict: Not viable for stem separation.**

### Option D: Spotify Web Playback SDK
**How it works:** Stream Spotify content in your web app.

| Aspect | Detail |
|--------|--------|
| Audio access | Streaming only via Spotify Connect |
| Requirement | Users need Spotify Premium |
| Stem separation | NOT possible |
| Recent changes | Spotify cut developer access to recommendations (Nov 2024) |
| Extended access | Must apply, only 25 users in dev mode |

**Verdict: Not viable for stem separation. Spotify is actively restricting API access.**

### Option E: Direct Music Licensing
**How it works:** License songs directly from publishers/labels.

| Aspect | Detail |
|--------|--------|
| Sync license cost | $50-$1,000 per indie song, $1,000-$10,000+ per known artist |
| Mechanical license | 9.1 cents per unit per song + $15 HFA fee |
| Revenue share model | ~15-50% of revenue to publishers (varies by deal) |
| Who to contact | Harry Fox Agency, NMPA, individual publishers |
| Minimum viable | Could license ~50-100 indie songs for $2,500-$10,000 |
| Songsterr model | Pays ~50% of revenue as royalties |

**For 1,000 songs at even $100/song/year = $100,000/year. Not viable for solo founder.**

### Option F: Free/CC-Licensed Content Library
**How it works:** Build a library from legal free sources.

| Source | Content | License |
|--------|---------|---------|
| ccMixter | Stems already separated, CC-licensed | CC (varies by track) |
| Musopen | Classical music, public domain | Public domain / CC |
| Free Music Archive | Wide variety, CC-licensed | CC (varies) |
| Archive.org (live shows) | Grateful Dead, Phish, etc. (taper-friendly bands) | Artist-approved |
| Jamendo | Indie artists, some CC | CC / commercial license |
| Indie artist partnerships | Artists who WANT exposure | Direct deal |

**Could seed library with 500-1,000 tracks for $0.**

---

## 5. The Recommended Hybrid Model

### Phase 1: Ship in 30 Days
1. **BYOA Upload** -- Users upload their own MP3/WAV files for stem separation
   - Add file upload UI (drag-and-drop, file picker)
   - Process uploaded files with existing stem separation pipeline
   - Store processed stems in user's account
   - Implement DMCA agent registration and takedown process
   - **Cost: $6 (DMCA registration)**

2. **YouTube Embedded Player** for chord practice (no stems)
   - Keep YouTube URL input for chord-only mode
   - Embed YouTube player, sync chord charts to playback
   - No audio download, no stem separation
   - **Cost: $0**

3. **Free Content Library** seed
   - Add 50-100 CC-licensed songs from ccMixter/FMA
   - Include taper-friendly live recordings from Archive.org
   - **Cost: $0**

### Phase 2: 90 Days
4. **Indie Artist Partnerships**
   - Reach out to local Charleston musicians
   - Offer: "Get your songs on StemScriber, fans practice along"
   - Artists provide stems directly (many already have them from recording)
   - Simple agreement: non-exclusive, artist retains rights, can withdraw anytime
   - **Cost: $0 (mutual benefit)**

5. **Songsterr-style licensing** for chord charts
   - Contact HFA about licensing chord/tab display
   - Revenue share model (~50% of subscription revenue)
   - Start with indie publisher catalog
   - **Cost: % of revenue (only pay when you earn)**

### Phase 3: 6 Months
6. **Scale with revenue**
   - As paying users grow, reinvest in licensing
   - Consider Loudr API for mechanical license management
   - Explore NMPA deals once at scale (1,000+ paying users)

---

## 6. The Math

### Break-Even Analysis

**Scenario: BYOA + Free Content (Phase 1)**
| Item | Monthly Cost |
|------|-------------|
| Hetzner VPS | $8 |
| Modal GPU | ~$30 (at current usage) |
| Domain/Cloudflare | $0 |
| DMCA registration | $0.50 (amortized) |
| **Total** | **~$39/month** |

**At $10/month per user: Need 4 paying users to break even.**

**Scenario: With Indie Licensing (Phase 2-3)**
| Item | Monthly Cost |
|------|-------------|
| Infrastructure | $39 |
| Revenue share to publishers | 50% of subscription revenue |
| Licensing admin (HFA/Loudr) | ~$50/month |
| **Total** | **~$89 + 50% of revenue** |

**At $10/month per user (keeping $5 after royalties): Need 18 paying users to break even.**

### Revenue Projections
| Users | Monthly Revenue | After 50% Royalty | After Costs ($89) | Net |
|-------|----------------|-------------------|-------------------|-----|
| 10 | $100 | $50 | -$39 | Loss |
| 25 | $250 | $125 | $36 | Positive |
| 50 | $500 | $250 | $161 | Healthy |
| 100 | $1,000 | $500 | $411 | Good |
| 500 | $5,000 | $2,500 | $2,411 | Great |

---

## 7. Contact Information

### Licensing Services
| Service | Contact | What They Do |
|---------|---------|--------------|
| Harry Fox Agency | harryfox.com, Songfile.com | Mechanical licenses, $15/song + royalties |
| NMPA | nmpa.org | Publisher trade group, licensing deals |
| Loudr (Rock Paper Scissors) | loudr.rockpaperscissors.biz | Licensing API for apps, used by DistroKid |
| Songtradr | songtradr.com | B2B music licensing, subscription model |

### Legal Counsel (from lawyer-call-prep.md)
| Lawyer | Contact | Notes |
|--------|---------|-------|
| Lindsay Spiller | spillerlaw.com | Free consult, entertainment law |
| Jesse Morris | morrismusiclaw.com | Free consult, music law |

### Free Music Sources
| Source | URL | Content Type |
|--------|-----|-------------|
| ccMixter | ccmixter.org | CC-licensed stems and remixes |
| Free Music Archive | freemusicarchive.org | CC-licensed full songs |
| Musopen | musopen.org | Public domain classical |
| Jamendo | jamendo.com | Indie CC-licensed music |
| Archive.org | archive.org/details/etree | Taper-friendly live recordings |

---

## 8. What to Tell the Lawyer

When you call Lindsay Spiller or Jesse Morris, here's what to ask:

1. **"We switched to a user-upload model (BYOA). Users upload their own audio files, we separate stems. Are we protected under DMCA safe harbor?"**
   - Expected answer: Yes, if you follow DMCA requirements

2. **"Can we embed YouTube's iframe player for chord-chart-only features without downloading audio?"**
   - Expected answer: Yes, this is what Chordify does

3. **"We want to license chord chart display from publishers. What's the minimum viable deal?"**
   - This is where the lawyer adds real value

4. **"What's our exposure for the YouTube downloading we did during development/beta?"**
   - Important to understand retroactive risk

5. **"Can we partner with indie artists who voluntarily provide their stems?"**
   - Expected answer: Yes, with a simple licensing agreement

---

## 9. Immediate Action Items

### This Week
- [ ] Remove yt-dlp / YouTube download from production code path
- [ ] Add file upload UI (drag-and-drop MP3/WAV/FLAC)
- [ ] Register DMCA designated agent at copyright.gov ($6)
- [ ] Update Terms of Service with DMCA policy and user upload terms
- [ ] Add repeat infringer policy to ToS

### Next 2 Weeks
- [ ] Build YouTube embedded player chord-only mode
- [ ] Add 50+ CC-licensed songs from ccMixter/FMA
- [ ] Reach out to 10 local Charleston artists about stem partnerships
- [ ] Call Lindsay Spiller or Jesse Morris (free consult)

### Next 30 Days
- [ ] Ship BYOA + embedded YouTube + free library
- [ ] Test with existing beta users
- [ ] Get legal sign-off from entertainment lawyer

---

## 10. Comparison Table: All Options

| Option | Legal Risk | Cost | Time to Ship | Stem Separation | Song Library Size |
|--------|-----------|------|-------------|----------------|-------------------|
| **A: BYOA (user upload)** | LOW | $0 | 1-2 weeks | YES | Unlimited (user files) |
| **B: YouTube embed (chords only)** | LOW | $0 | 1 week | NO | 36M+ (YouTube catalog) |
| **C: Apple MusicKit** | LOW | $0 | 2-4 weeks | NO | 100M+ (Apple Music) |
| **D: Spotify SDK** | LOW | $0 | 2-4 weeks | NO | 100M+ (Spotify) |
| **E: Direct licensing** | NONE | $$$$ | 3-6 months | YES | 50-1000 (what you license) |
| **F: CC/free content** | NONE | $0 | 1 week | YES | 500-1000 |
| **G: yt-dlp (current)** | **EXTREME** | $0 | Already built | YES | Unlimited |
| **RECOMMENDED: A+B+F** | LOW | ~$6 | 2-4 weeks | YES (user files) | User files + YouTube chords + CC library |

---

## Bottom Line

The product is NOT dead. It just needs to change HOW users get songs into it.

**Instead of:** Paste YouTube URL -> download -> separate stems
**Switch to:** Upload your own MP3 -> separate stems (+ YouTube embed for chord-only practice)

Moises built a $10M+ company on exactly this model. Chordify built a 36M-song service on embedded YouTube + chord analysis. StemScriber can combine both approaches.

The BYOA model actually has advantages:
- Higher audio quality (user's own files vs. YouTube compression)
- No dependency on YouTube API changes
- Users who own music files are more serious musicians (better customers)
- Legal clarity -- DMCA safe harbor is well-established law

**Ship the upload feature this week. Call the lawyer next week. Stop downloading from YouTube today.**
