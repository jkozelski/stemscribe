# Chordify Legal Research: How They Handle YouTube URLs Legally

**Research Date:** 2026-03-26
**Purpose:** Understand how Chordify (chordify.net) legally operates with YouTube URLs to inform StemScribe's legal strategy.

---

## 1. Company Overview

**Chordify B.V.** is a Dutch limited liability corporation based in Groningen, Netherlands (Stationsweg 3G, 9726 AC). Registered at Dutch Chamber of Commerce #60387238. VAT: NL853888310B01.

- **Founded:** ~2011, spun out of Utrecht University research (Music Information Retrieval group)
- **Founder/CEO:** Bas de Haas (PhD in Music Informatics)
- **Revenue:** ~$11M/year (2026 estimate)
- **Users:** 10+ million musicians monthly, 100M+ unique visitors since 2013 launch
- **Funding:** Equity crowdfunding round (exact amount not public)

Sources:
- [Chordify Terms and Conditions](https://chordify.net/pages/terms-and-conditions/)
- [Chordify Crunchbase](https://www.crunchbase.com/organization/chordify)
- [Grokipedia - Chordify](https://grokipedia.com/page/chordify)

---

## 2. Terms of Service -- Key Excerpts (Verbatim)

The following are direct quotes from Chordify's Terms and Conditions as of March 2026:

### Section 2.1 -- Personal Use Only
> "The Chordify Service is for your own personal non-commercial use only. The Chordify Services are licensed to you, and not sold, assigned or transferred."

### Section 2.5 -- User Responsibility for Data
> "You shall be responsible for the data and other results processed and created by you through the use of the Chordify Services. You guarantee that such data and results are not illegal and will not infringe rights of third parties. You indemnify us against claims from third parties, of whatever nature, in relation to this data, results or your use of the Chordify Services."

### Section 2.8 -- YouTube API Services
> "Chordify uses YouTube API Services to play embedded videos, which are subject to YouTube's Terms of Service."

### Section 8 -- Lyrics (Licensed Content)
> "Usage of Lyrics is limited to your personal, noncommercial use in accordance with the terms of this Agreement. You may not reproduce (other than as authorized for your own personal usage), publish, transmit, distribute, publicly display, rent or lend, modify, create derivative works from, sell or participate in the sale of or exploit in any way, in whole or in part, directly or indirectly, any of the Lyrics so provided."

> "You agree that you are not granted any so-called 'karaoke' or 'sing-along' rights to Lyrics and you shall not seek to or remove any vocal track from a sound recording that shall be associated with a Lyric provided to you."

### Section 9 -- Rights (Intellectual Property)
> "The Chordify Service contains material which is owned by us. This material includes, but is not limited to, the design, layout, look, sounds, appearance, graphics and software. All rights of use not expressly granted to you are reserved by us."

> "It is not allowed to make the Chordify Services or any contents thereof public or to reproduce it, including but not limited to disclosing, publishing, (re)distributing or (re)producing, whether in whole or a part thereof, in any form, unless explicitly allowed in these terms and conditions."

### Section 12 -- Notice and Takedown Procedure
> "If any person or legal entity is of the opinion that any content available through the Chordify Services infringes its rights in or to such content, it can submit a complaint at retract@chordify.net, specifying the title of the content and the rights which he or she has in or to the content. If we find the complaint founded, we will remove the content as soon as possible."

### Section 13 -- Applicable Law
> "Your use of the Chordify Service, these terms and conditions, the relationship between you and us and any dispute arising therefrom is subject to the laws of the Netherlands."

Source: [Chordify Terms and Conditions](https://chordify.net/pages/terms-and-conditions/) (retrieved via Playwright browser, March 2026)

---

## 3. How Chordify Handles Copyright -- Their Official Position

From their support article "Does this not infringe copyright?" (verbatim):

> "Chordify automatically extracts chords from songs. Unlike lyrics or melody, chord progressions are not considered innovative or sufficiently unique enough to be copyrighted on their own. In fact, some chord progressions are particularly popular and are used in exactly the same shape by hundreds of popular songs!"

> "In any case, copyright owners who do not want the chords of their works to be displayed by Chordify can request the retraction of individual songs by emailing retract@chordify.net."

**Key legal arguments Chordify relies on:**
1. Chord progressions alone are not copyrightable (established music copyright law)
2. They only output chords -- not melodies, not lyrics, not audio
3. They provide a takedown mechanism (retract@chordify.net)
4. They operate under Dutch/EU law, not US law

Sources:
- [Does this not infringe copyright? -- Chordify Support](https://support.chordify.net/hc/en-us/articles/360001420738-Does-this-not-infringe-copyright)
- [Can you copyright a chord progression? -- Local 802 AFM](https://www.local802afm.org/allegro/articles/can-you-copyright-a-chord-progression/)

---

## 4. Technical Approach -- How Audio Is Processed

### Audio Sources
Chordify accepts audio from:
- YouTube (embedded player via YouTube API)
- SoundCloud
- Deezer (currently unavailable)
- User-uploaded files (MP3, MP4, OGG, up to 20 minutes -- Premium feature)

### YouTube Integration -- The Critical Detail
**Chordify uses the YouTube IFrame Player API to EMBED YouTube videos.** The video plays in the bottom-right corner of the Chordify interface. This is the crucial legal distinction:

- They do NOT download YouTube audio to their servers for playback
- The YouTube video is embedded and plays through YouTube's official embedded player
- Audio playback comes from YouTube's servers, not Chordify's
- Chordify syncs their chord display with the embedded video playback

### Chord Extraction Pipeline
1. Audio is converted into a **spectrogram** (time-frequency representation)
2. Deep neural networks analyze the spectrogram for chord recognition and beat tracking
3. The neural networks were trained on thousands of songs with known chord labels
4. The HarmTrace harmony model (developed during Bas de Haas's PhD at Utrecht University) applies music theory rules to improve accuracy
5. Accuracy is 75-90% depending on music style and recording quality
6. Community edits improve accuracy over time via information fusion model

### What Chordify Stores vs. What It Doesn't
**Stores:**
- Chord data (the extracted chord progression with timing)
- Beat positions
- User edits and corrections
- Metadata (song title, source URL)

**Does NOT store (strong evidence):**
- Original audio files from YouTube/SoundCloud/Deezer
- Offline mode explicitly "doesn't include the original audio of the song (unless it is a song you have uploaded yourself), but does include the chord diagrams, metronome and chord audio"
- MIDI exports contain "only the generated on-screen chords" -- not full instrumentation

Sources:
- [AI-technology behind the chords of Chordify](https://chordify.net/pages/technology-algorithm-explained/)
- [How can I use Chordify offline? -- Chordify Support](https://support.chordify.net/hc/en-us/articles/360020907938-How-can-I-use-Chordify-offline)
- [What is Chordify? -- Chordify Support](https://support.chordify.net/hc/en-us/articles/360002221018-What-is-Chordify)
- [CHORDIFY: CHORD TRANSCRIPTION FOR THE MASSES -- ISMIR 2012](https://ismir2012.ismir.net/event/papers/LBD2.pdf)
- [CHORDIFY: THREE YEARS AFTER THE LAUNCH -- ISMIR 2015](https://ismir2015.ismir.net/LBD/LBD42.pdf)

---

## 5. Legal Discussions -- Lawsuits, DMCA, Cease and Desist

### No Known Lawsuits
Extensive searching found **zero lawsuits, DMCA actions, or cease-and-desist letters** publicly documented against Chordify. This is remarkable for a service with 10M+ monthly users and 100M+ total visitors since 2013.

### Why They Haven't Been Sued (Analysis)
1. **They don't distribute copyrighted audio** -- playback is via YouTube's embedded player
2. **Chord progressions aren't copyrightable** -- well-established legal principle
3. **They're in the Netherlands** -- Dutch/EU law applies, not US law (harder for US labels to pursue)
4. **They have a takedown mechanism** -- retract@chordify.net provides a path for rights holders
5. **They license what IS copyrightable** -- lyrics come from LyricFind (licensed)
6. **They add value, not compete** -- they drive traffic TO YouTube, not away from it
7. **EU safe harbor protections** -- Article 14 of the EU E-commerce Directive limits intermediary liability

### The "Warner Music Group" Connection
Chordify celebrated with "friends from Warner Music Group" at a company event, suggesting an informal or formal relationship with at least one major label. This is not confirmed as a licensing deal, but indicates they are not adversarial with the majors.

Sources:
- [Chordify Crunchbase](https://www.crunchbase.com/organization/chordify)
- [Copyright in the Netherlands -- Lexology](https://www.lexology.com/library/detail.aspx?g=560596f0-069d-49bc-ae51-1feba07b3d4b)

---

## 6. Licensing Deals and Partnerships

### Confirmed Licensing: LyricFind (Lyrics)
Chordify partnered with **LyricFind** for licensed lyrics display. LyricFind:
- Has licenses with 50,000+ publishers
- Pays royalties to songwriters
- Facilitated $25M+ in payments to rightsholders in 2024
- Provides legal cover for lyrics display

This is why Chordify's ToS has a specific "Section 8: Lyrics" with strict usage restrictions -- because lyrics ARE copyrightable and they need to comply with LyricFind's licensing terms.

### Partnership: Rocket Songs
Chordify partnered with Rocket Songs (music licensing startup) to provide chords for songs available for licensing on that platform. This is a B2B partnership, not a copyright license.

### No Confirmed Major Label Licensing Deals
Unlike services such as Spotify, Klay, or Musixmatch that have explicit licensing deals with Universal, Sony, and Warner, **there is no public evidence that Chordify pays licensing fees to major labels or publishers for chord extraction.** Their argument is that they don't need to -- because chord progressions aren't copyrightable.

Sources:
- [Chordify Lyrics -- The Feature You've Been Screaming For!](https://chordify.net/pages/chordify-lyrics/)
- [Chordify lyrics feature -- MusicRadar](https://www.musicradar.com/news/chordify-lyrics-live-chord-detection)
- [Rocket Songs partnership](https://chordify.net/pages/rocket-songs-partnership/)

---

## 7. Business Model

### Freemium SaaS
- **Free tier:** Access to chord detection for any song, basic playback
- **Premium subscription:** Monthly or yearly
  - MIDI download (time-aligned and simplified versions)
  - Transpose/capo
  - Speed adjustment
  - PDF export
  - Offline mode (chords only, no original audio)
  - Song upload (personal files)
  - Lyrics display (via LyricFind)
- **Premium + Toolkit:** Additional tools including live chord detection via microphone

### Revenue Streams
1. Premium subscriptions (primary)
2. Freemium ad-supported tier
3. Chordify Embed (for third-party websites)
4. Chordify API

Sources:
- [Chordify Premium](https://chordify.net/premium)
- [What are the subscription options?](https://support.chordify.net/hc/en-us/articles/360002273238-What-are-the-subscription-options)

---

## 8. Their Legal Positioning Strategy (Summary)

Chordify's legal defense rests on **five pillars:**

1. **"We only show chords"** -- Chord progressions are not copyrightable. This is the bedrock of their legal position. They carefully avoid displaying copyrightable elements (melody, full lyrics without license, audio recordings).

2. **"We use YouTube's official embed"** -- They don't download or redistribute audio. YouTube handles the audio playback through its own embedded player. This keeps them compliant with YouTube's ToS and means they're not creating unauthorized copies.

3. **"We license what's copyrightable"** -- When they DO display copyrightable content (lyrics), they use a licensed provider (LyricFind) that pays royalties to publishers.

4. **"We have a takedown process"** -- retract@chordify.net allows rights holders to request removal. This is analogous to DMCA safe harbor provisions but under EU law.

5. **"We're a Dutch company under Dutch law"** -- Operating under Netherlands/EU jurisdiction gives them distance from aggressive US copyright enforcement and benefits from EU safe harbor protections.

---

## 9. Key Takeaways for StemScribe

### What StemScribe Can Replicate from Chordify's Model

#### A. YouTube Audio Playback -- Use the YouTube IFrame Embed API
- **DO:** Use YouTube's official embedded player for audio playback
- **DO:** Sync your chord/lyric display with the embedded player's playback position
- **DON'T:** Download YouTube audio for playback purposes
- **RATIONALE:** This is the single most important legal distinction. Chordify doesn't serve YouTube audio -- YouTube does, through its own player. StemScribe should adopt this same approach for any YouTube-sourced content that users play back.

#### B. Chord Display -- Lean on "Chords Aren't Copyrightable"
- **DO:** Display chord progressions -- they are not copyrightable
- **DO:** Generate chords algorithmically from audio analysis (this is what Chordify does)
- **DON'T:** Display full melodies or tablature of copyrighted works without a license
- **RATIONALE:** Chordify has operated for 13 years on this principle without legal challenge

#### C. Stem Separation -- This Is Where StemScribe Differs (and Has More Risk)
- Chordify does NOT separate stems -- they only analyze the full mix for chords
- StemScribe separates audio into stems (vocals, drums, bass, etc.), which is a MORE transformative process
- **The question:** Does stem separation create a derivative work of the original recording?
- **Mitigation strategies:**
  - Process audio ephemerally (don't permanently store separated stems)
  - Use stems only for practice/educational purposes (stronger fair use argument)
  - Don't allow downloading/exporting of separated stems
  - Use YouTube embed for the original audio playback, only process for analysis

#### D. Lyrics -- Use a Licensed Provider
- **DO:** Use LyricFind or Musixmatch for lyrics (they handle publisher licensing)
- **DON'T:** Scrape lyrics from unlicensed sources or generate them with Whisper and display them
- **RATIONALE:** Chordify licenses lyrics through LyricFind specifically because lyrics ARE copyrightable. StemScribe should do the same.
- **NOTE:** Whisper-generated lyrics for personal/internal use (not displayed to users) is a grayer area

#### E. Takedown Process -- Implement a Notice-and-Takedown System
- **DO:** Create a dedicated email (e.g., retract@stemscribe.io or takedown@stemscribe.io)
- **DO:** Document your takedown policy in your Terms of Service
- **DO:** Act promptly on takedown requests
- **RATIONALE:** This demonstrates good faith and provides safe harbor protection

#### F. Jurisdiction -- Consider Where You Incorporate
- Chordify benefits from being a Dutch company under EU law
- StemScribe (US-based) faces stricter US copyright enforcement
- **Consider:** US DMCA safe harbor provisions (Section 512) if you register as a service provider
- **Consider:** Emphasizing educational/transformative use for fair use arguments

#### G. Terms of Service -- Model After Chordify's
- Personal, non-commercial use only
- Users responsible for their own data/results
- Users indemnify the service against third-party claims
- Reference YouTube's ToS for embedded content
- Separate section for licensed content (lyrics) with usage restrictions
- Clear notice-and-takedown procedure

### What StemScribe Should NOT Do (Based on Chordify's Approach)
1. **Don't store YouTube audio permanently** -- Process ephemerally, discard after analysis
2. **Don't allow users to download copyrighted audio** -- Only export derived data (chords, MIDI, etc.)
3. **Don't display copyrighted content without a license** -- Lyrics need LyricFind/Musixmatch
4. **Don't frame it as a "downloader"** -- Frame it as an educational/practice tool

### The Biggest Risk Unique to StemScribe
Chordify's output is **non-copyrightable data** (chord symbols). StemScribe's output includes **separated audio stems** which are derived from copyrighted recordings. This is the key difference and the key risk. Mitigations:
- Ephemeral processing (don't cache stems permanently)
- Educational/practice framing
- No stem download/export
- YouTube embed for original audio playback
- Strong fair use argument (transformative educational purpose)

---

## 10. Source URLs (Complete List)

- https://chordify.net/pages/terms-and-conditions/
- https://chordify.net/pages/privacy-policy/
- https://support.chordify.net/hc/en-us/articles/360001420738-Does-this-not-infringe-copyright
- https://support.chordify.net/hc/en-us/articles/360002221018-What-is-Chordify
- https://support.chordify.net/hc/en-us/articles/360001416658-When-I-upload-my-songs-to-your-service-do-I-keep-my-rights
- https://support.chordify.net/hc/en-us/articles/360020907938-How-can-I-use-Chordify-offline
- https://support.chordify.net/hc/en-us/articles/360016231697-My-YouTube-video-won-t-play
- https://chordify.net/pages/technology-algorithm-explained/
- https://chordify.net/pages/chordify-lyrics/
- https://chordify.net/pages/rocket-songs-partnership/
- https://chordify.net/premium
- https://techround.co.uk/interviews/a-chat-with-bas-de-haas-chordify/
- https://grokipedia.com/page/chordify
- https://ismir2012.ismir.net/event/papers/LBD2.pdf
- https://ismir2015.ismir.net/LBD/LBD42.pdf
- https://www.crunchbase.com/organization/chordify
- https://www.musicradar.com/news/chordify-lyrics-live-chord-detection
- https://www.local802afm.org/allegro/articles/can-you-copyright-a-chord-progression/
- https://www.lexology.com/library/detail.aspx?g=560596f0-069d-49bc-ae51-1feba07b3d4b
- https://news.ycombinator.com/item?id=7429282
- https://developers.google.com/youtube/terms/developer-policies
- https://developers.google.com/youtube/iframe_api_reference
- https://tools.aiformusic.org/knowledgebase/articles/chordify-ai-powered-chord-extraction-interactive-jam-sessions
