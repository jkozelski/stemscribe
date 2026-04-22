# StemScriber ToS Research — Competitive Analysis & Draft Clauses
**Date:** 2026-04-21 (21 days to launch)
**Purpose:** Survey industry ToS language for audio-processing / chord-detection / stem-separation services, and draft matching clauses for StemScriber.
**Action required:** Alexandra Mayo review before deployment.

---

## Part 1: Comparison Table

| Company (ToS URL) | a) Input methods | b) User warranty | c) Hosting / caching | d) Takedown / DMCA | e) Commercial use by user | f) Training rights grant | g) YouTube / third-party platform |
|---|---|---|---|---|---|---|---|
| **Chordify** (chordify.net/pages/terms-and-conditions/) | YouTube / Deezer / SoundCloud URLs + file upload. "Audio files uploaded by users are owned by them, and are presented only to them." | "You guarantee that such data and results are not illegal and will not infringe rights of third parties." User indemnifies Chordify. | Does not claim to host source audio. Retention not quantified. | retract@chordify.net | Not directly addressed for output; user grants Chordify license to "data and results" for the purpose of the Chordify Service. | Not explicitly addressed. | "Chordify uses YouTube API Services to play embedded videos, which are subject to YouTube's Terms of Service." |
| **Moises** (help.moises.ai/hc/en-us/articles/7401394754962) | File upload + Dropbox/iCloud/OneDrive URLs. Explicitly rejects streaming URLs (YouTube/Spotify/Apple Music/SoundCloud). | "you are the creator and owner... or have the necessary licenses, rights, consents, and permissions." ALL-CAPS: YOU ARE SOLELY RESPONSIBLE FOR ENSURING YOU HAVE ALL NECESSARY RIGHTS TO UPLOAD CONTENT FROM ANY THIRD-PARTY SERVICE. | Limited license "to host, store, transfer, reproduce, modify... solely to process the User Content and generate Output on your behalf." Output is confidential to user. | copyright@moises.ai. Moises Systems, Inc. d/b/a Music AI, Attn: Copyright Agent, 136 South Main Street, Suite 400, Salt Lake City, Utah 84101. | User retains rights to Output. May NOT: redistribute as samples/loops/libraries; compete with the Service; train AI. | **"WE DO NOT USE YOUR USER CONTENT OR OUTPUT TO TRAIN AI MODELS WITHOUT YOUR PERMISSION."** ALL-CAPS, ironclad. | Streaming URLs rejected per external policy (DRM + derivative works concerns). |
| **Klangio** (klang.io/terms/) | File upload + YouTube URLs. | "you have legal authorization to modify files and YouTube videos you submit." | **"Klangio does not host audio or video content of any kind."** 30-day max retention. | retract@klangio.com | Not addressed. | Not addressed. | **"Klangio uses Google's YouTube API... you agree to be bound by the YouTube Terms of Service."** |
| **LALAL.AI** (lalal.ai/terms-and-conditions/) | File upload (no YouTube URL). | Not explicitly addressed as warranty clause. | "The Service doesn't store or distribute any copyrighted content." | notice@incorporatenow.com (OmniSale GmbH's agent). | "You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the Service." | Not addressed. | Not addressed. |
| **Music.ai** (music.ai/terms/, B2B arm of Moises) | "Customer Content" — no YouTube URL language. | "You represent and warrant that you possesses all rights necessary for us to receive, process, and create the Customer Content." | "we are under no obligation to store, maintain, or provide you with copies"; may retain "indefinitely in anonymized form." | copyright@moises.ai | User may use Service to generate Output — subject to training + competitive restrictions. | **"DOES NOT INCLUDE ANY RIGHTS FOR US TO USE YOUR CUSTOMER CONTENT TO TRAIN OR FINE-TUNE ARTIFICIAL INTELLIGENCE OR MACHINE LEARNING MODELS UNLESS YOU HAVE EXPRESSLY AUTHORIZED SUCH USE IN WRITING."** | Not addressed. |
| **Songsterr** (songsterr.com/a/wa/terms) | Generic "Content" — user-posted tabs. | "the Content is yours (you own it) and/or you have the right to use it." | Not addressed. | support@songsterr.com (subject: "Copyright Infringement") | "You may not distribute, modify, transmit, reuse, download, repost, copy, or use said Content... for commercial purposes." | Not addressed. | YouTube embeds — user bound by YouTube ToS. |
| **Ultimate Guitar** (ultimate-guitar.com/about/tos.htm) | User-submitted tabs/audio/video. | "you have obtained all necessary rights, licenses or clearances." | "reasonable period of time for backup, archival, or audit purposes" after termination. | copyright@ultimate-guitar.com. Ultimate Guitar USA LLC, 268 Bush Street, #3044, San Francisco, CA 94104; phone (415) 599-4620 Ext. 2. | "Content may be accessed for your personal, non-commercial use only." Licensed Compositions carve-out. | Not addressed. | YouTube-hosted content: UG does not have ability to remove from YouTube's servers. Directs takedowns to YouTube. |
| **Yousician** (yousician.com/terms-of-service) | Generic User Content — library is pre-licensed. | "you own or have the necessary licenses, rights, consents, and permissions to submit." | Not addressed. | copyrightagent@yousician.com. Yousician Oy, Siltasaarenkatu 16, FI-00530 Helsinki, Finland. | "prohibited from using the Services for public performances." | Not addressed. | Not addressed. |

**Note on Chordify:** Full ToS page returns 403 to automated fetches. Quotes assembled from support articles + Google-indexed excerpts. Pull verbatim from browser for Alexandra's review.

---

## Part 2: Recommended ToS Clauses for StemScriber

### 1. Service description & input methods
*(pattern: Moises §1 + Klangio)*

> StemScriber provides an audio processing service that accepts audio files uploaded by you and public YouTube URLs that you submit. StemScriber uses these inputs to produce stem separations, chord charts, and practice-mode output ("Output").

### 2. User warranty (covers files AND YouTube URLs)
*(pattern: Moises §11.7 + Klangio)*

> By submitting any audio file or YouTube URL to StemScriber, you represent and warrant that:
>
> (a) you are the creator or copyright owner of the underlying recording and composition, OR you have the necessary licenses, rights, consents, and permissions to submit that content for processing;
>
> (b) your submission will not infringe, violate, misappropriate, or otherwise breach any copyright, trademark, publicity right, contract right, or other right of any person or entity;
>
> (c) for YouTube URLs specifically: you are solely responsible for ensuring that you have all necessary authorizations, consents, permissions, and rights to submit that video to StemScriber for processing, including compliance with YouTube's Terms of Service.
>
> You indemnify StemScriber against any third-party claim arising from content you submit.

### 3. No-hosting / transient processing
*(pattern: Klangio + Moises §11.3 + Music.ai retention)*

> StemScriber does not host user audio or video content. Source audio submitted to the Service is processed transiently and deleted after Output is generated. Processed Output (stems, chord charts) is cached briefly on our infrastructure solely to deliver your results to you, and may be purged at any time without notice. We have no obligation to store, maintain, or provide you with copies of your submitted content or Output.

### 4. Limited license to operate the Service
*(pattern: Moises §11.3 — verbatim structure)*

> By submitting content to the Service, you grant StemScriber a worldwide, non-exclusive, royalty-free, fully-paid license to host, store, transfer, reproduce, and modify your submitted content solely to process it and generate Output on your behalf, and to provide the Service to you. No other rights are granted, express or implied.

### 5. No training rights — headline protection
*(pattern: Moises §11.4)*

> **STEMSCRIBER DOES NOT USE YOUR SUBMITTED CONTENT OR OUTPUT TO TRAIN, FINE-TUNE, OR OTHERWISE DEVELOP ARTIFICIAL INTELLIGENCE OR MACHINE LEARNING MODELS. NO LICENSE SET FORTH IN THESE TERMS INCLUDES ANY SUCH RIGHT. WE WILL NOT USE YOUR CONTENT FOR AI TRAINING UNLESS YOU HAVE EXPRESSLY AUTHORIZED SUCH USE IN WRITING OR VIA AN ELECTRONIC OPT-IN MECHANISM.**

### 6. User ownership of Output + carve-outs
*(pattern: Moises §11.2)*

> You retain all rights, title, and interest in Output you generate using the Service, to the extent allowed by applicable law. However, you may not:
>
> (a) redistribute, sublicense, or package the Output (stems, chord charts, or practice tracks) as samples, loops, sound libraries, or similar content collections;
>
> (b) use the Output in a manner that competes with or is intended to displace the market for the Service; or
>
> (c) use the Output to train or fine-tune artificial intelligence or machine learning models, either as the sole input or as part of a larger dataset.
>
> You remain responsible for ensuring your downstream use of Output complies with the rights of any third party in the underlying recording or composition.

### 7. Copyright takedown — dedicated address
*(pattern: Music.ai / Yousician `copyright@` alias. Adopt instead of `support@`.)*

> If you are a copyright owner and believe your work has been made available through the Service in a way that infringes your rights, contact our Copyright Agent at **copyright@stemscriber.com**. Please include: (i) identification of the work, (ii) identification of the location on the Service, (iii) your contact information, (iv) a good-faith statement that the use is not authorized, and (v) a statement under penalty of perjury that the information is accurate and you are authorized to act for the copyright owner.
>
> Copyright Agent — StemScriber, [physical address]. We will respond to valid notices in accordance with the DMCA.

### 8. YouTube / third-party platform clause
*(pattern: Klangio + Songsterr. IMPORTANT: do NOT copy Klangio's "uses Google's YouTube API" language — we use yt-dlp, not the official API. Language below stays neutral on retrieval method.)*

> When you submit a YouTube URL, StemScriber retrieves the audio stream for the sole purpose of processing it on your behalf and generating Output. You are responsible for ensuring your submission complies with YouTube's Terms of Service and with the rights of any person appearing in or owning rights in the video. StemScriber is not affiliated with, endorsed by, or sponsored by YouTube or Google.

### 9. Consent modal replacement
*(pattern: hybrid current modal + Moises §10.1 + Klangio)*

**Current text:**
> By uploading audio to StemScriber, you confirm that you own this content or have the legal right to use it. StemScriber provides an audio processing service only. We do not claim ownership of your content.

**Proposed replacement:**
> By submitting audio or a YouTube URL, you confirm that you own this content or have the legal right to submit it for processing. StemScriber is an audio processing service — we do not host your source audio, we do not claim ownership of your content, and we do not use your content to train AI models. You remain responsible for how you use the Output.

---

## Gap-closing summary

| Gap in current StemScriber posture | Fix | Source pattern |
|---|---|---|
| No explicit warranty covering YouTube URL submissions | Add §2(c) | Moises §10.1 |
| No explicit "no-hosting" claim | Add §3 | Klangio |
| No explicit no-training clause in ToS | Add §5 | Moises §11.4 (verbatim template) |
| No dedicated copyright email alias | Add copyright@stemscriber.com | Music.ai / Yousician |
| No commercial-use carve-outs on Output | Add §6 | Moises §11.2 |
| Consent modal is file-upload only, silent on YouTube + AI training | Replace per §9 | — |

## Key flags for Alexandra

1. **yt-dlp vs. YouTube API:** Klangio and Chordify both say they use YouTube's official API. We use yt-dlp. Language in §8 is deliberately neutral — confirm this is acceptable or require architecture change.
2. **Output carve-outs (§6):** Moises restricts commercial redistribution. This is the industry standard but it limits what musicians can do with their own chord charts. Confirm this is OK for our positioning.
3. **No-training clause (§5):** Ironclad — but we must actually honor it. If post-launch we want to train on user content, we need opt-in UX, not a retroactive ToS change.
4. **copyright@stemscriber.com alias:** Needs Cloudflare email routing added before this language goes live.
5. **Physical address for DMCA agent (§7):** Need Jeff's business address.

---

**Sources:**
- [Moises Terms & Conditions](https://help.moises.ai/hc/en-us/articles/7401394754962-Terms-Conditions)
- [Moises streaming-URL policy](https://help.moises.ai/hc/en-us/articles/18322457396508)
- [Music.ai Terms of Service](https://music.ai/terms/)
- [Klangio Terms](https://klang.io/terms/)
- [LALAL.AI Terms and Conditions](https://www.lalal.ai/terms-and-conditions/)
- [Songsterr Terms](https://www.songsterr.com/a/wa/terms)
- [Ultimate Guitar Terms of Service](https://www.ultimate-guitar.com/about/tos.htm)
- [Yousician Terms of Service](https://yousician.com/terms-of-service)
- [Chordify Terms and Conditions](https://chordify.net/pages/terms-and-conditions/) (403 to bots)
