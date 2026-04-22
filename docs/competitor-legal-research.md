# Competitor Legal Research: Stem Separation & Music Practice Apps

**Date:** 2026-03-26 (updated 2026-04-04 with Chordify, SoundSlice, fair use analysis, and lawyer call talking points)
**Purpose:** Understand how competitors handle the legal question of getting copyrighted audio into their apps, and identify the safest model for StemScriber.

---

## Quick Comparison Table

| App | Audio Input | Accepts URLs? | YouTube? | Stores Audio? | ToS on Copyright | Licensing Deals? | Been Sued? |
|-----|------------|---------------|----------|---------------|-------------------|-----------------|------------|
| **Moises** | Upload files, cloud URLs (Dropbox/iCloud/GDrive) | Yes (cloud storage only) | NO | Yes (user library) | User responsible; must own or have rights | No content licenses; $40M VC funded | No |
| **Capo** | Apple Music library (DRM-free only) | No | No | Local only | Only works with purchased/owned music | No; Apple ecosystem only | No |
| **LALAL.AI** | Upload files only | No (user must download first) | No (blog shows manual workaround) | Deleted after 1 day | User solely responsible | No content licenses | No |
| **Splitter.ai** | Upload, URL, live recording | Yes (including song URLs) | Unclear | Deleted after 1 hour | User responsible | No content licenses | No |
| **PhonicMind** | Upload files only | No | No | Not stored/republished | User takes full responsibility | No content licenses | No |
| **iZotope RX** | Local files in DAW/standalone | No (desktop software) | No | Local only | "Within boundaries of copyright law" | N/A (pro audio tool) | No |
| **Spleeter** | Local files (open source library) | N/A (CLI tool) | N/A | Local only | MIT license; "get authorization from right owners" | N/A (open source) | No |
| **SongDonkey** | Upload files | No | No | Temporary | Minimal ToS | No content licenses | No |
| **VocalRemover.org** | Upload files, YouTube URLs | Yes | YES | 30 days max | User confirms legal authorization | No content licenses | No |
| **Chordify** | YouTube/Deezer/SoundCloud streams + user upload | Yes (streaming embeds) | YES (embed) | NO (streams from YouTube) | Chords aren't copyrightable; personal use only | No formal deals found | No |
| **SoundSlice** | User upload (notation + audio) | No | No | Yes (user library) | User certifies rights; DMCA process | No content licenses | No |
| **MashApp** | In-app licensed catalog | No (walled garden) | No | Licensed server-side | Fully licensed | YES - all 3 majors + Kobalt | No |
| **AudioShake** | B2B API (labels/publishers) | N/A (enterprise) | N/A | Enterprise | Licensed by rights holders | YES - labels & publishers | No |

---

## 1. Moises App

### Overview
- 50+ million users worldwide
- $16.1M revenue (July 2025)
- $40M Series A (January 2025) led by Connect Ventures (CAA + NEA)
- Apple iPad App of the Year 2024

### Audio Input Methods
- Upload audio/video files directly (MP3, WAV, FLAC, AAC, OGG, WMA, AIFF, MP4, MOV, etc.)
- Import from iTunes (iOS)
- Import from cloud storage: Google Drive, Dropbox, iCloud, OneDrive
- Import from Files app
- Import from other apps (e.g., WhatsApp)
- Public URLs (cloud storage links only)
- **NO streaming service URLs** (no YouTube, Spotify, Apple Music, SoundCloud, Tidal, Amazon Music, Deezer, TikTok, Twitter)

### Terms of Service Summary
- Users are "solely responsible for ensuring they have all necessary authorizations, consents, permissions, and rights" to upload content
- "Moises strongly advises users to only process music they have the legal right to modify"
- Users must "provide an audio file that they own or have legal permission to use"
- Moises complies with DMCA; designated copyright agent at copyright@moises.ai
- Users retain ownership of uploads and outputs
- No content licensing deals with labels

### Why They Block Streaming URLs
Moises explicitly explains: "Streaming platforms have terms of service and legal restrictions that prevent external tools from modifying their content. Attempting to extract or modify audio from a streaming service without permission can lead to legal consequences, including copyright infringement claims."

### Key Takeaway
Moises is the gold standard model: upload-only, user responsibility, no YouTube, no streaming URLs, DMCA compliance. They have VC backing and legal counsel. This is the model to follow.

**Sources:**
- [Moises Upload Help](https://help.moises.ai/hc/en-us/articles/8583454469276)
- [Moises Terms & Conditions](https://help.moises.ai/hc/en-us/articles/7401394754962)
- [Moises Streaming URL Policy](https://help.moises.ai/hc/en-us/articles/18322457396508)
- [Moises Acceptable Use Policy](https://help.moises.ai/hc/en-us/articles/5968943445404)
- [Moises Content Ownership](https://help.moises.ai/hc/en-us/articles/23877386138140)
- [Music AI $40M Series A](https://www.musicbusinessworldwide.com/music-ai-raises-40m-in-series-a-round-as-its-moises-platform-hits-50m-users/)

---

## 2. Capo App (SuperMegaUltraGroovy)

### Overview
- iOS/Mac app for learning songs by ear
- Chord detection, tempo adjustment, looping
- One-time purchase / subscription model
- Much smaller than Moises

### Audio Input Methods
- Accesses user's local music library (iOS Music app)
- ONLY works with DRM-free audio (purchased from iTunes Store)
- **Cannot access Apple Music streaming tracks** (DRM-protected)
- If user tries to load a streaming track, Capo offers a "Buy on iTunes" button to purchase a DRM-free copy
- No URL input, no cloud import, no upload

### Terms of Service Summary
- Specific ToS not publicly indexed
- App is designed around the assumption that users own the music they analyze
- Built-in DRM check enforces this technically

### Key Takeaway
Capo is the most conservative model possible: it technically prevents processing any music the user doesn't own by checking for DRM. This is extremely safe legally but limits the user base to people who purchase music from iTunes.

**Sources:**
- [Capo Apple Music Support](https://supermegaultragroovy.com/products/capo/support/apple-music.html)
- [Capo iOS Getting Started](https://supermegaultragroovy.com/products/capo/help/ios/2.9/getting-started/)
- [Capo App Store](https://apps.apple.com/us/app/capo-learn-music-by-ear/id887497388)

---

## 3. LALAL.AI

### Overview
- Popular vocal remover and stem splitter
- Web-based + desktop apps + VST plugin
- Freemium model with minute-based pricing

### Audio Input Methods
- File upload only (drag and drop or browse)
- Supports audio: MP3, WAV, FLAC, OGG, etc.
- Supports video: MP4, MKV, AVI
- API available (POST /api/upload/)
- **NO URL input** on the platform itself
- Blog post suggests downloading YouTube videos first with external tools, then uploading

### Terms of Service Summary
- "You are solely responsible for the usage and distribution of uploaded audio files and 'end-result' audio files"
- Does NOT store or distribute copyrighted content
- Does NOT use user files for AI training
- Does NOT share copyrighted materials with third parties
- Files deleted after 1 day
- Voluntarily complies with DMCA (despite not being US-based)

### Key Takeaway
Upload-only, auto-delete, user responsibility. Clean model. They explicitly distance themselves from any role in how users obtain their files.

**Sources:**
- [LALAL.AI Terms and Conditions](https://www.lalal.ai/terms-and-conditions/)
- [LALAL.AI FAQ](https://www.lalal.ai/guides/faq/)
- [LALAL.AI Privacy Policy](https://www.lalal.ai/privacy-policy/)
- [LALAL.AI YouTube Blog Post](https://www.lalal.ai/blog/how-to-remove-vocals-from-a-youtube-video/)

---

## 4. Splitter.ai

### Overview
- Free web-based stem separation tool
- Simple drag-and-drop interface

### Audio Input Methods
- File upload (drag and drop)
- Song URL input
- Live recording
- Files expire and auto-delete after 1 hour

### Terms of Service Summary
- Users agree to compliance with applicable local laws
- Materials protected by copyright and trademark law
- Minimal ToS compared to competitors

### Key Takeaway
Splitter.ai is more permissive than most -- it accepts URLs and even live recording. Minimal ToS. This is a riskier model but they operate with very low profile.

**Sources:**
- [Splitter.ai](https://splitter.ai/)
- [Splitter.ai Terms of Service](https://splitter.ai/terms-of-service)
- [Splitter.ai Privacy Policy](https://www.splitter.ai/help)

---

## 5. PhonicMind

### Overview
- AI vocal remover, "Hi-Fi" stems
- Web-based, subscription model

### Audio Input Methods
- File upload only
- No URL input

### Terms of Service Summary
- "We do not take any responsibility or authorship rights over what users upload"
- "We do not review or republish audio tracks uploaded by users"
- "Users take full responsibility for not breaching any copyright or authorship laws"
- Only processes and returns results to user
- No refund policy (high computational costs)

### Key Takeaway
Upload-only, zero involvement in content, full user responsibility. Very standard model.

**Sources:**
- [PhonicMind Legal Disclaimer](https://phonicmind.com/legal-disclaim/)
- [PhonicMind](https://phonicmind.com/)

---

## 6. iZotope RX (Music Rebalance)

### Overview
- Professional audio repair suite ($129-$1,199)
- Desktop software (plugin + standalone)
- Used by audio engineers, producers, post-production

### Audio Input Methods
- Local files only (loaded in DAW or standalone app)
- No web interface, no URLs, no uploads

### Terms of Service Summary
- Users can use stems "within reason (and within the boundaries of copyright law)"
- Professional tool marketed to legitimate audio professionals

### Key Takeaway
Desktop-only pro tool. No server involvement. The safest possible model since everything is local and there's no server-side processing of user content.

**Sources:**
- [iZotope RX Music Rebalance](https://www.izotope.com/en/learn/stem-separation-music-rebalance)
- [iZotope RX Features](https://www.izotope.com/en/products/rx/features/music-rebalance)

---

## 7. Deezer Spleeter (Open Source)

### Overview
- Open-source stem separation library (MIT license)
- Released by Deezer Research
- CLI/Python tool, no web interface from Deezer directly
- Powers many third-party tools (including SongDonkey)

### Audio Input Methods
- Local files via command line
- N/A for web (it's a library)

### License & Copyright
- Code is MIT-licensed (free for any use including commercial)
- **Critical caveat:** "If you plan to use Spleeter on copyrighted material, make sure you get proper authorization from right owners beforehand"
- Responsibility falls entirely on the user/developer

### Key Takeaway
The tool itself is clearly legal. Using it on copyrighted material without authorization is the user's/developer's problem. This is the model most open-source tools follow.

**Sources:**
- [Spleeter GitHub](https://github.com/deezer/spleeter)
- [Spleeter LICENSE](https://github.com/deezer/spleeter/blob/master/LICENSE)
- [Deezer Spleeter Blog Post](https://deezer.io/releasing-spleeter-deezer-r-d-source-separation-engine-2b88985e797e)

---

## 8. VocalRemover.org

### Overview
- Free web-based vocal remover
- Simple upload interface

### Audio Input Methods
- File upload
- **Accepts YouTube URLs** (users confirm legal authorization)

### Terms of Service Summary
- Users "must confirm that their use of the Service will not infringe proprietary rights"
- Users confirm they "have legal authorization to modify files and YouTube videos they submit"
- Content stored for maximum 30 days
- Users retain rights to their content

### Key Takeaway
One of the few that explicitly accepts YouTube URLs. Higher risk profile, but they place full responsibility on users via ToS language about confirming authorization.

**Sources:**
- [VocalRemover Terms of Use](https://vocalremover.com/terms)
- [VocalRemover.org](https://vocalremover.org/)

---

## 9. MashApp (Fully Licensed Model)

### Overview
- Music mashup creation tool
- iPhone app, launched US February 2025
- Took 4 years to secure licensing

### Audio Input Methods
- In-app catalog only (walled garden)
- NO user uploads
- NO URLs
- Users browse and select from licensed tracks

### Licensing
- Licensed by ALL three major labels: Universal Music Group, Sony Music Entertainment, Warner Music Group
- Also licensed by Warner Chappell Music, Universal Music Publishing Group (UMPG), and Kobalt Music
- Proper attribution to artists, songwriters, and rightsholders
- Freemium model (free users get limited mashups; premium unlocks more)

### Key Takeaway
The "gold standard" for legal compliance but the hardest to replicate. It took 4 years and significant resources to secure these deals. Not feasible for a solo founder at launch, but this is the direction the industry is heading.

**Sources:**
- [MashApp Launch - MusicTech](https://musictech.com/news/gear/mashapp-licensing-deals-major-labels/)
- [MashApp - RouteNote](https://routenote.com/blog/mashapp-the-new-music-mashup-app-backed-by-major-labels/)
- [MashApp - Music Business Worldwide](https://www.musicbusinessworldwide.com/mashapp-founded-by-a-former-spotify-exec-debuts-music-mixing-platform-licensed-by-all-3-major-labels/)
- [Fan Remixing Licensing - Billboard](https://blackpromoterscollective.com/fan-remixing-is-the-latest-music-tech-trend-could-licensing-stop-its-growth/)

---

## 10. AudioShake (B2B Licensed Model)

### Overview
- Enterprise AI stem separation company
- B2B API for labels, publishers, sync licensing
- Not consumer-facing

### Business Model
- Works directly with rights holders (labels, publishers)
- Creates stems for authorized remixes, sync licensing, immersive mixes
- Partners with platforms like Bushido for white-label licensing flows
- "Trusted by labels and publishers worldwide"

### Key Takeaway
The B2B model sidesteps the consumer copyright question entirely because the clients ARE the rights holders. Interesting but not applicable to a consumer practice app.

**Sources:**
- [AudioShake Licensing Blog](https://www.audioshake.ai/post/licensing-derivative-works-how-ai-is-opening-the-door-to-legal-remixes-and-edits)
- [AudioShake + Bushido Partnership](https://www.audioshake.ai/press-releases/bushido-partners-with-audioshake-to-unlock-stem-creation-for-remixes-edits-and-derivative-works)

---

## 11. Chordify (chordify.net) — ADDED 2026-04-04

### Overview
- Netherlands-based company (Groningen), founded 2013 by Bas de Haas
- Shows chord progressions for any song in real-time
- Small company — raised ~720k-1M EUR via equity crowdfunding (2016)
- Freemium model

### How It Works (Legally Critical)
- **Streams audio from YouTube/Deezer/SoundCloud embeds** — does NOT host or store the audio
- Chord sheets move in real-time synced to the embedded YouTube player
- Users can also upload their own files, but those CANNOT be shared with other users
- Uses deep neural networks trained on spectrograms to detect chords and beats

### Their Copyright Defense (Two-Pronged)
1. **Chord progressions are not copyrightable** — this is well-established music copyright law. Chordify's position: chords are not "innovative or sufficiently unique enough to be copyrighted on their own"
2. **Audio never leaves YouTube's servers** — they piggyback on YouTube's existing blanket licenses with publishers (via Content ID). Chordify just displays analysis alongside the stream

### Terms of Service
- Service is for "personal non-commercial use only"
- Restrictions on lyrics and content distribution
- Copyright owners can request retraction of individual songs via retract@chordify.net

### Business Model
| Plan | Price | Features |
|------|-------|----------|
| Free | $0 | Basic chord display, YouTube streaming |
| Premium | $6.99/mo ($3.49 annual) | Loop, capo, transpose |
| Premium + Toolkit | From $2.25/mo | Metronome, drumbeat, chord virtuoso |

### Licensing Deals
- **No confirmed licensing deals with major labels/publishers**
- Partnership with Rocket Songs (allows users to license songs from songwriters)
- Operates under EU copyright law (may have different requirements than US)

### Key Takeaway for StemScriber
Chordify proves that **showing chord progressions for copyrighted songs is viable** without licensing deals, because chords themselves aren't copyrightable. Their YouTube embed model is clever — they never touch the audio, YouTube handles all the licensing. StemScriber's chord chart feature has solid precedent here.

**Sources:**
- [Does this not infringe copyright? — Chordify Support](https://support.chordify.net/hc/en-us/articles/360001420738)
- [What is Chordify? — Chordify Support](https://support.chordify.net/hc/en-us/articles/360002221018)
- [Terms and Conditions — Chordify](https://chordify.net/pages/terms-and-conditions/)
- [AI technology behind Chordify](https://chordify.net/pages/technology-algorithm-explained/)

---

## 12. SoundSlice (soundslice.com) — ADDED 2026-04-04

### Overview
- Interactive notation/tab player synced to audio
- Has a store where users sell lessons/courses
- Licensing option for businesses to embed tech

### Copyright Approach
- Users retain IP in uploaded content
- **Grant SoundSlice broad license**: "worldwide, non-exclusive, royalty-free license (with the right to sublicense) to use, copy, reproduce, process, adapt, modify, publish, transmit, display and distribute"
- **Sellers must certify they own copyright** or have obtained necessary rights
- Printed arrangements require a special "print" license from the song's publisher

### DMCA Process
- Notices go to feedback@soundslice.com
- Upon valid DMCA notice: slice is unpublished (no longer publicly accessible)
- Both parties notified
- Repeat infringers: "we reserve the right to disable their public sharing ability"
- References fair use exceptions in 17 U.S.C. Section 107

### Business Model
- One-time purchase for store content (not subscription for buyers)
- Licensing option for businesses to embed SoundSlice tech
- Creator/publisher subscription plans

### Key Takeaway for StemScriber
SoundSlice shows how a notation/tab platform handles copyright — standard DMCA safe harbor, user responsibility, repeat infringer policy. Their "print license" requirement for arrangements is worth asking the lawyer about: does auto-generated notation from audio analysis require a print license?

**Sources:**
- [SoundSlice Terms of Service](https://www.soundslice.com/terms/)
- [SoundSlice Copyright and DMCA](https://www.soundslice.com/help/en/channels/posting/330/copyright-and-dmca/)
- [SoundSlice Content Policy](https://www.soundslice.com/help/en/store-selling/policies/153/content-policy/)

---

## LEGAL ANALYSIS: AI STEM SEPARATION & FAIR USE — ADDED 2026-04-04

### Four-Factor Fair Use Test Applied to Stem Separation

Per Northwestern Journal of Technology and IP analysis:

**Factor 1 — Transformative Purpose:**
- More transformation = stronger defense
- Using 2 seconds of extracted strings vs. entire vocal stem = very different legal positions
- Stem separation for practice/learning is more defensible than redistribution
- However: stem extraction itself may not be "transformative" — it's isolating existing content, not creating new meaning

**Factor 2 — Nature of the Copyrighted Work:**
- Music is "close to the core of intended copyright protection" and "far removed from more factual or descriptive work"
- This factor **disfavors** stem users

**Factor 3 — Amount and Substantiality:**
- Stems "typically represent a considerably smaller portion of a work" than full copies
- Potentially **favorable** compared to traditional sampling cases

**Factor 4 — Market Effect (Most Important):**
- Courts examine "whether unrestricted and widespread conduct would result in substantially adverse impact"
- A musician practicing at home differs vastly from someone selling isolated vocals
- Personal practice use = low market harm = favorable

### Derivative Works Question
- Copyright protection extends to components (stems) of a song
- "Any modification of that work, such as extracting a stem and using it elsewhere, likely qualifies as a derivative work"
- Only copyright owners may authorize derivative works
- BUT: the tool that enables the extraction is different from the act of creating a derivative work

### The Tool vs. The Use Distinction (Sony Betamax Defense)
- The tool itself is not infringing — it's a "staple article of commerce" with substantial non-infringing uses
- Sony v. Universal (1984): manufacturers of copying tech are not liable for infringement by users if the tech is "capable of substantial noninfringing uses"
- Stem separation has clear non-infringing uses: practice, education, remixing own music, accessibility

### Copyright Office Position (2025)
- GenAI training on large, diverse datasets "will often be transformative"
- BUT: use of copyrighted materials for AI model training is alone insufficient to justify fair use
- Where AI outputs are substantially similar to training data, there's a "strong argument" the models themselves infringe
- Note: this relates to MODEL TRAINING, not to using a model to process user-supplied audio

### Key Legal Distinction for StemScriber
Stem separation tools are **fundamentally different** from generative AI music tools:
- Suno/Udio: trained ON copyrighted music to GENERATE new music → clear infringement risk
- StemScriber/Moises: trained on licensed/synthetic data, processes USER-SUPPLIED audio → tool/service provider

**Sources:**
- [AI Stem Extraction: Creative Tool or Mass Infringement? — Northwestern Law](https://jtip.law.northwestern.edu/2022/05/03/ai-stem-extraction-a-creative-tool-or-facilitator-of-mass-infringement/)
- [Copyright Office on Fair Use in GenAI Training — Skadden](https://www.skadden.com/insights/publications/2025/05/copyright-office-report)
- [Copyright Office on Fair Use — Wiley](https://www.wiley.law/alert-Copyright-Office-Issues-Key-Guidance-on-Fair-Use-in-Generative-AI-Training)
- [AI Music Copyright 2026 — Jam.com](https://jam.com/resources/ai-music-copyright-2026)

---

## LICENSING FRAMEWORKS REFERENCE — ADDED 2026-04-04

| License Type | What It Covers | Relevant to StemScriber? |
|-------------|----------------|--------------------------|
| Mechanical license | Reproducing/distributing a song (covers, recordings) | Maybe — if stems are considered reproductions |
| Sync license | Pairing music with visual media | No — unless karaoke mode with lyrics is considered sync |
| Master use license | Using a specific recording | Potentially — separated stems derive from masters |
| Performance license | Playing music publicly | No — StemScriber is personal use |
| Print license | Sheet music/notation | **Ask lawyer** — does auto-generated chord chart/tab require this? |

### The Print License Question
- SoundSlice notes that producing printed arrangements requires a "print" license
- Chordify's defense: chord progressions alone aren't copyrightable
- But full tab transcriptions (melody + rhythm) may cross into copyrightable territory
- **Key question for lawyer**: If StemScriber auto-generates guitar tabs from audio analysis, is that a "print" arrangement requiring a license? Or is it analogous to Chordify showing chords?

---

## TALKING POINTS FOR APRIL 4 LAWYER CALL — ADDED 2026-04-04

### Point 1: Industry Precedent
"Moises has 50M+ users, $40M in VC funding, and has operated since ~2020 with an upload-only model and no licensing deals. No lawsuits. They put all copyright responsibility on users via ToS. We follow the same model."

### Point 2: Chord Charts Are Safe
"Chordify has operated since 2013 showing chord progressions for copyrighted songs with no licensing deals and no lawsuits. Their defense: chord progressions aren't copyrightable. Our chord chart feature has solid precedent."

### Point 3: No Stem Separation App Has Been Sued
"As of April 2026, zero stem separation tools have been the target of copyright litigation. The major AI music lawsuits (Suno/Udio) target generative AI that creates new music from training data — fundamentally different from processing a user's own file."

### Point 4: Our Legal Safeguards
"We use an upload-only model (no YouTube/Spotify URLs). Users must have rights to what they upload. We comply with DMCA. We auto-delete processed audio. We restrict to personal, non-commercial use."

### Point 5: The Tool Defense
"Under Sony v. Universal (Betamax), manufacturers of technology capable of substantial non-infringing uses aren't liable for user infringement. Stem separation has clear non-infringing uses: practicing your own music, education, accessibility, remixing original compositions."

### Point 6: Questions for the Lawyer
1. Do we need to register a DMCA agent with the Copyright Office? (It's $6 online)
2. What specific ToS language do you recommend for user upload responsibility?
3. Do auto-generated guitar tabs/chord charts require a print license?
4. Should we add an upload consent checkbox ("I confirm I have rights to this audio")?
5. What's our exposure if a user uploads copyrighted music for personal practice?
6. Should we proactively limit any features (e.g., stem downloads, tab exports)?
7. Do we need any specific insurance for IP/copyright claims?

---

## RIAA & Major Label Positions

### RIAA Stance (as of March 2026)
- The RIAA has NOT specifically targeted stem separation tools
- Their enforcement focus has been on **generative AI** (Suno, Udio) that creates new music from copyrighted training data
- RIAA Chairman Mitch Glazier: "The music community has embraced AI and we are already partnering and collaborating with responsible developers to build sustainable AI tools centered on human creativity"
- The RIAA supports California legislation requiring AI developers to disclose training data sources
- They advocate for federal deepfake protection laws

### Suno/Udio Lawsuit Status (March 2026)
- **Filed:** June 2024 by UMG, Sony, WMG against both Suno and Udio
- **Udio:** UMG settled October 2025; launching joint licensed AI platform in 2026
- **Suno:** WMG settled November 2025; Sony and UMG still litigating; dispositive motions due March 13, 2026
- **Key allegation:** "Stream-ripping" from YouTube to train AI models
- **Potential damages:** Up to $150,000 per infringed song
- No final court ruling on merits yet

### Universal Music Group
- Settled with Udio (October 2025), partnering on licensed AI music platform launching 2026
- "Walled garden" model: no downloads or external posting of AI creations
- Opt-in only for artists
- Fingerprinting and filter technologies for attribution
- Also partnered with Stability AI for "next-generation" AI music tools
- Still suing Suno (as of March 2026)

### Warner Music Group
- Settled with both Suno (November 2025) and Udio
- Suno replacing current models with licensed alternatives in 2026
- Download restrictions and caps for paid users
- Artists can opt in/out of having their content used

### Sony Music Group
- Sent formal letters to 700+ AI companies prohibiting unauthorized use of content
- Developing AI fingerprinting technology to detect copyrighted material in AI-generated content
- Can operate in "non-cooperative" mode (analyze output without developer access)
- Still suing Suno (as of March 2026)

### NMPA (National Music Publishers' Association)
- Called generative AI "the greatest risk to the human creative class that has ever existed"
- Not opposed to AI broadly, but demands proper licensing
- NMPA CEO David Israelite: AI could "rewrite the playbook on music licensing"
- Believes "the song is just as valuable, if not more, than the sound recording in the AI model"

**Sources:**
- [RIAA v Suno/Udio](https://www.riaa.com/record-companies-bring-landmark-cases-for-responsible-ai-againstsuno-and-udio-in-boston-and-new-york-federal-courts-respectively/)
- [UMG Settles with Udio - Variety](https://variety.com/2025/music/news/universal-music-settles-udio-lawsuit-partners-with-stability-ai-1236565616/)
- [UMG + Udio - Rolling Stone](https://www.rollingstone.com/music/music-features/ai-music-universal-music-group-settlement-udio-1235457945/)
- [WMG + Suno Settlement - Rolling Stone](https://www.rollingstone.com/music/music-features/suno-warner-music-group-ai-music-settlement-lawsuit-1235472868/)
- [WMG + Suno - Hollywood Reporter](https://www.hollywoodreporter.com/music/music-industry-news/warner-music-group-settles-ai-infringement-suit-with-suno-1236435516/)
- [Sony Warns 700 AI Companies - Billboard](https://www.billboard.com/business/tech/sony-music-artificial-intelligence-training-opt-out-1235684192/)
- [Sony AI Detection Tech - TechRepublic](https://www.techrepublic.com/article/news-sony-ai-music-detector/)
- [NMPA on AI](https://www.nmpa.org/nmpa-generative-ai-is-the-greatest-risk-to-the-human-creative-class-that-has-ever-existed/)
- [Suno Lawsuit Updates - McKool Smith](https://www.mckoolsmith.com/newsroom-ailitigation-46)
- [Google/Lyria 3 Lawsuit - MBW](https://www.musicbusinessworldwide.com/indie-artists-sue-google-claiming-it-used-youtubes-own-catalog-to-train-lyria-3-ai-music-tool/)

---

## Legal Analysis: Has Any Stem Separation App Been Sued?

### Short Answer: No.

As of March 2026, **no stem separation or vocal removal app has been sued** by any record label, publisher, or the RIAA. The legal actions have been focused entirely on:

1. **Generative AI music tools** (Suno, Udio) -- for using copyrighted recordings to train models that generate new music
2. **Stream-ripping services** -- for circumventing access controls to download from YouTube/streaming services
3. **Google** -- for allegedly training Lyria 3 on YouTube's own catalog

### Why Stem Separation Tools Haven't Been Targeted

1. **They're tools, not content:** Stem separators are analogous to audio editors, EQs, or DAWs. They process audio the user provides. DAWs already include tools that could facilitate infringement, and this hasn't led to lawsuits.

2. **No training on copyrighted data:** Unlike Suno/Udio, stem separators don't ingest copyrighted music into training datasets. They use models trained on licensed/synthetic data to perform separation.

3. **Personal use defense:** Most users are musicians practicing, creating karaoke tracks, or learning songs. While not legally airtight, personal, non-commercial use is a strong fair use factor.

4. **Enforcement difficulty:** As noted by Bray & Krais (UK music law firm), "it will become increasingly difficult to hear when a sample is used if it has been taken from an individual stem."

5. **Secondary liability risk is low:** Tool providers who don't encourage infringement, don't store/redistribute content, and comply with DMCA are in a strong position.

### The Real Legal Risk Areas

The legal risk for a stem separation app comes from:

1. **Accepting YouTube/streaming URLs** -- this involves downloading copyrighted content from platforms that prohibit it, potentially circumventing access controls (DMCA anti-circumvention)
2. **Storing/caching copyrighted audio** -- creating a library of processed copyrighted content on your servers
3. **Redistributing stems** -- making separated stems available to users other than the uploader
4. **Scraping lyrics/chords from copyrighted sources** -- separate IP issues (Songsterr, Ultimate Guitar, etc.)

**Sources:**
- [AI Stem Extraction: Creative Tool or Mass Infringement? - Northwestern Law](https://jtip.law.northwestern.edu/2022/05/03/ai-stem-extraction-a-creative-tool-or-facilitator-of-mass-infringement/)
- [Stem-Separating AI - Bray & Krais](https://www.brayandkrais.com/stem-separating-ai-is-revolutionising-the-music-industry/)
- [DMCA Safe Harbor - Copyright Alliance](https://copyrightalliance.org/education/copyright-law-explained/the-digital-millennium-copyright-act-dmca/dmca-safe-harbor/)
- [DMCA Safe Harbor Guide - Fenwick](https://assets.fenwick.com/legacy/FenwickDocuments/DMCA-QA.pdf)

---

## YouTube Downloading: Legal Status

### Current Legal Position (March 2026)
- A US court ruled in early 2026 that downloading YouTube videos using third-party sites **bypasses copyright protections under the DMCA**
- YouTube's Terms of Service explicitly prohibit downloading except via YouTube's own features
- The software tools themselves (yt-dlp, youtube-dl) are legal; **how you use them** determines compliance
- YouTube's standard stream delivery (HLS/DASH) is NOT DRM-protected like Netflix Widevine, creating legal ambiguity around DMCA anti-circumvention
- Personal offline backup is "legally grey" and unlikely to result in action
- Redistributing, monetizing, or republishing is clearly infringement

### Implications for StemScribe
Accepting YouTube URLs puts StemScribe in the position of being the tool that downloads copyrighted content from a platform that prohibits it. This is the highest-risk feature from a legal standpoint.

**Sources:**
- [Third-Party YouTube Downloads Create Copyright Risks - MediaNama](https://www.medianama.com/2026/02/223-dmca-ruling-third-party-youtube-downloads-legal-risks-creators/)
- [How to Download YouTube Videos Legally 2026](https://www.bestvideodownloader.net/how-to-download-youtube-videos-legally-2026/)
- [youtube-dl Wikipedia](https://en.wikipedia.org/wiki/Youtube-dl)

---

## DMCA Safe Harbor: What StemScribe Needs

To qualify for DMCA safe harbor protection as an online service provider:

1. **Designate and register a DMCA agent** with the U.S. Copyright Office
2. **Adopt a repeat infringer policy** and terminate accounts of repeat offenders
3. **Implement a takedown notice mechanism** (accept and act on DMCA notices)
4. **Don't have actual knowledge** of infringing material, or remove it promptly when notified
5. **Don't financially benefit directly** from infringing activity you have the ability to control
6. **Don't be the direct infringer** -- safe harbor only applies when someone OTHER than the service provider is the infringer

**Critical distinction:** If StemScribe downloads YouTube videos server-side, StemScribe itself is the entity performing the download (potential direct infringement). If users upload files, users are the ones who obtained the files (StemScribe is the service provider, eligible for safe harbor).

---

## Safest Model for StemScribe: Specific Recommendations

### Tier 1: Minimum Viable Legal Protection (Do Immediately)

1. **Remove YouTube URL input as the primary/default flow.** The user should upload a file. This is what Moises, LALAL.AI, and PhonicMind all do. This single change eliminates the biggest legal risk.

2. **Add clear ToS language** modeled on Moises/LALAL.AI:
   - "You are solely responsible for ensuring you have all necessary rights and permissions to upload and process content"
   - "You must only process music you own or have legal permission to modify"
   - "StemScribe does not store, distribute, or claim rights to your uploaded content"
   - "StemScribe does not use uploaded content for AI training"

3. **Register a DMCA agent** with the U.S. Copyright Office ($6 filing fee). This is required for safe harbor protection.

4. **Add a DMCA takedown process** -- an email address (e.g., copyright@stemscribe.io) and a procedure for handling notices.

5. **Auto-delete processed audio** after a reasonable period (1 hour like Splitter.ai, or 24 hours like LALAL.AI). Don't store a permanent library of separated copyrighted songs on your server.

6. **Add a repeat infringer policy** to your ToS -- accounts that receive multiple DMCA notices get terminated.

### Tier 2: Stronger Protection (Do Before Scaling)

7. **If keeping URL input, restrict to cloud storage URLs only** (Dropbox, Google Drive, iCloud, OneDrive) -- this is the Moises model. The user already has the file; you're just making it easier to get it to your server.

8. **Add a copyright acknowledgment checkbox** on upload: "I confirm I have the right to process this audio" (similar to VocalRemover.org).

9. **Don't cache or store processed stems permanently.** Process, deliver, delete. The URL cache system (SQLite) should be reconsidered -- caching copyrighted audio on your server creates liability.

10. **Remove or restructure the chord library** if it contains scraped copyrighted chord charts. This is a separate IP risk area (see legal brief).

### Tier 3: Future-Proofing (For Growth Phase)

11. **Consider the MashApp/AudioShake licensing path** if you grow large enough. At 50M+ users, Moises still hasn't needed licensing deals, so this isn't urgent at your scale.

12. **Monitor the Suno/Udio case outcomes** -- dispositive motions due March 13, 2026. Any rulings on fair use, stream-ripping, or AI tool liability will shape the landscape.

13. **Explore partnerships with rights holders** for a "practice mode" with licensed content -- this is where the industry is heading but only makes sense at scale.

14. **Keep the personal practice / educational use framing** prominent in marketing and ToS. Fair use weighs educational and transformative purposes more favorably.

### What NOT to Do

- **Do NOT accept YouTube URLs server-side** -- this makes StemScribe the direct downloader, not a neutral tool
- **Do NOT build a library of pre-separated copyrighted songs** -- this turns you from a tool into a content distributor
- **Do NOT scrape Songsterr/Ultimate Guitar content at scale** without a license
- **Do NOT advertise the ability to "rip" or "download" from any streaming service
- **Do NOT train AI models on copyrighted music** without licensing

### The Bottom Line

Every major competitor in this space uses the same model: **users upload files they already have, the app processes them, and responsibility for copyright compliance falls on the user.** No stem separation app has been sued. The RIAA's focus is on generative AI, not processing tools. The risk for StemScribe at its current scale (11 testers, 24 songs) is negligible, but building the right legal foundation now avoids problems at scale.

The single most important change: **shift from YouTube URL input to file upload as the primary flow.** Everything else is icing.

---

*Research conducted 2026-03-26, updated 2026-04-04 with Chordify, SoundSlice, fair use analysis, licensing frameworks, and lawyer call talking points. This is not legal advice. Consult an attorney (see ~/stemscribe/docs/lawyer-call-prep.md) before making final decisions.*
