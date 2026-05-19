# StemScriber — Pre-Launch Legal Review Package

**Prepared:** 2026-04-30
**Prepared by:** Jeff Kozelski (founder), with research support from a Claude AI assistant
**Audience:** Alexandra Mayo, Esq. — Morris Music Law, retained counsel
**Version:** 1.0

---

## Cover note

Alexandra —

This package consolidates everything I've decided, built, and gated since our last call. I've done substantial research on my end (six legal/business texts cited herein, plus current case law) so that your time on this can be **focused review and sign-off, not exploratory consultation**.

I am not asking you to research from scratch. I'm asking you to:
1. **Confirm** the positions I've taken on routine matters where I've cited authority — flag anything that's wrong
2. **Weigh in** on a small number of genuinely novel questions where I'd value your professional judgment
3. **Identify** anything I haven't thought of that should be on the radar before launch

If everything in here looks defensible to you, a one-line "looks good with the noted caveats" reply lets us launch with documented professional sign-off. If you see issues, I'd rather know now than after launch.

— Jeff

---

## 1. Executive summary

**StemScriber** is a consumer web app at stemscriber.com (currently BETA-tagged, pre-launch) that takes a user-uploaded audio file (or, since today, a user-pasted URL) and returns: separated stems, an auto-detected chord chart, a practice mode (loop / slow / mute / solo), and Guitar Pro export. Solo founder; no employees.

**Legal posture (current state):**

| Layer | Posture | Status |
|-------|---------|--------|
| Sound recording reproduction | User-warranty shift via ToS §4.3 | **Live** |
| URL-pasted content (new today) | User-warranty shift via ToS §4.4 + per-URL attestation modal | **Live** |
| Composition rights (chord-chart output) | Per-user processing only, no public chart library, no lyrics | **Live** (lyrics gated off Apr 10; chart library deleted Apr 16) |
| AI training data | We don't train on user content; output is transformative (audio → notation) | **Live** |
| AI-generated output ownership | Per current Copyright Office stance: uncopyrightable (no human authorship) | **Live (defensive only)** |
| DMCA §512 safe harbor | Registered (Reg # DMCA-1070849) with `support@stemscriber.com` as agent | **Live** |
| §230 user-directed content | User pastes URL / uploads file; service is user-directed | **Live** |
| Karaoke / lyrics distribution | Permanently disabled per your Apr 10 directive | **Gated off** |

**Three matters where I'd value your specific input** (covered in §6):
- (a) **Confirmation** that today's YouTube URL acceptance feature, with the architecture and ToS clause described herein, is consistent with your prior guidance.
- (b) **Trademark filing strategy** for "StemScriber" before public launch.
- (c) **Active AI litigation watch list** — confirm my re-evaluation triggers cover the cases that would actually shift our posture.

---

## 2. Background research and citations

**Texts consulted on my end** (full text retained as plain text; passages cited inline below):

| Source | Edition / date | Used for |
|--------|----------------|----------|
| Donald S. Passman, *All You Need to Know About the Music Business* | 11th ed., Simon & Schuster 2023 | Composition vs. sound-recording rights; transcription license terminology; AI authorship; mechanical licensing |
| Harry Borovick, *AI and the Law: A Practical Guide to Using Artificial Intelligence Safely* | Apress 2024 | AI training-data jurisprudence; UK/EU comparison; current AI-copyright cases |
| Eric Goldman, *Internet Law Casebook* | 2025 ed., Santa Clara Law (released July 2025) | Section 230 doctrine; DMCA §512 safe harbor; current platform-liability case law (Moody v. NetChoice 2024, Anderson v. TikTok 2024, Twitter v. Taamneh 2023) |
| Stephen Fishman, *Legal Guide for Starting & Running a Small Business* | 18th ed., Nolo 2023 | Trademark, contracts, electronic ToS for online services |
| NYC Bar Association Committee on Entertainment Law, *Music Rights Primer* | v.4, 2003 | Foundational §106 exclusive rights; composition vs. master distinction |

I have full plain-text copies of all six locally and can pull additional citations on demand.

**Cases tracked** (current as of 2026-04-30): *Bartz v. Anthropic*, *Concord v. Anthropic*, *NYT v. OpenAI*, *Getty v. Stability AI*, *Anderson v. TikTok* (3d Cir. 2024), *Twitter v. Taamneh* (SCOTUS 2023), *Moody v. NetChoice* (SCOTUS 2024), *Andy Warhol Foundation v. Goldsmith* (SCOTUS 2023), *Naruto v. Slater* (9th Cir. 2018), *Thaler v. Perlmutter* (Copyright Office refusal + appeal), *Kashtanova / Zarya of the Dawn* (Copyright Office partial refusal).

---

## 3. Decisions made independently — please confirm

Each of these is a position I've taken without escalating to you, with my cited reasoning. I'm asking you to confirm each is defensible. If any aren't, please flag.

### 3.1 The legal term "transcription license" does not apply to StemScriber

**Position:** In music-industry usage (Passman ch 17), "transcription license" means an *audio-only sync license* — licensing a song for use with audio media (radio, podcasts, audio commercials). It has nothing to do with deriving notation from a recording. What StemScriber does (audio → chord chart / MIDI) is governed by sound recording reproduction rights, derivative work rights, and fair use / transformative use doctrine — not transcription licensing.

**Practical implication:** We do not need transcription licenses. We do not need ASCAP/BMI/SESAC public-performance licenses (no public performance occurs). We do not need mechanical licenses (we are not making cover recordings).

**Source:** Passman, *All You Need to Know About the Music Business*, 11th ed., Ch 17 ("Synchronization and Transcription Licenses").

---

### 3.2 AI-generated chord charts have no copyright owner under current US law

**Position:** Chord charts produced by StemScriber's ML pipeline from user-uploaded audio are not copyrightable because there is insufficient human authorship in the output itself. Per the US Copyright Office's 2018 + 2022 + 2023 rulings, AI-generated content without sufficient human creative input is not registrable. This applies to:
- The chord chart itself (machine-generated derivative of the recording)
- The MIDI output
- The Guitar Pro export

**What this gives us (defensively):**
- Competitors copying our chart output have no copyright claim against us — neither do we
- Strengthens "we are a tool, not a publisher" framing in ToS
- The user owns nothing they didn't already own; we own nothing of theirs

**What this does NOT give us:**
- Permission to ingest copyrighted *inputs* without authorization (separate question — handled by user-warranty ToS shift)
- Right to host a public library of generated charts (the underlying *compositions* remain copyrighted; this is why we deleted the stored chart library Apr 16 per your direction)

**Sources:** Passman 11th ed. Ch 19 ("Artificial Intelligence and Monkey Selfies"), citing *Naruto v. Slater* No. 16-15469 (9th Cir. 2018), *Thaler* Copyright Office refusal, *Kashtanova/Zarya of the Dawn* partial refusal. Borovick (2024), p. ~178: "human authorship is a prerequisite to copyright protection."

---

### 3.3 User-warranty shift via Terms of Service §4.3 (file uploads) and §4.4 (URLs)

**Position:** StemScriber relies on the user's contractual representation that they hold all necessary rights to audio they upload (§4.3) or URLs they submit (§4.4, new today). We do not verify rights independently. User attestation is collected in-product (consent popup for uploads; per-URL attestation modal for YouTube URLs as of today).

**Architecture:**
- ToS §4.3 (existing) — file upload warranty
- ToS §4.4 (new today, attached as Appendix A) — URL-paste warranty mirroring §4.3
- Consent popup — every-session for uploads
- Per-URL attestation modal — checkbox confirmation + audit-trail persistence (timestamp, attestation type, user agent, SHA256 hash of IP) before any URL is processed

**Source basis:** Standard industry posture per *Goldman Internet Law Casebook* 2025 ed. (§230 user-directed content discussion + §512 DMCA safe harbor mechanics). Industry-standard among peer services (Klangio, Moises, Chordify, AudioShake all use comparable user-warranty postures).

---

### 3.4 YouTube URL acceptance with architectural guardrails (shipped today)

**Position:** Effective 2026-04-30, StemScriber accepts YouTube URLs through a deliberately-secondary "Don't have the file?" UI fallback under the primary file-upload flow. Server-side, the URL is processed via yt-dlp (already wired via the pre-existing `/api/url` route, supporting SoundCloud, Bandcamp, Vimeo, Archive.org since prior versions; YouTube was already in the supported-URL allowlist but had no frontend exposure until today).

**Risk decomposition (four legal layers):**

| Layer | Analysis | Posture |
|-------|----------|---------|
| (1) Copyright (label/publisher) | User audio is copyrighted regardless of source. Same risk profile as direct uploads. ToS §4.4 user-warranty + no public hosting + transformative output cover this. | No incremental risk vs. status quo |
| (2) YouTube ToS | yt-dlp violates YT's contract terms. **Contract dispute with Google, not copyright.** Worst case: cease-and-desist or IP block. | Operational, not existential |
| (3) DMCA §512 safe harbor | §512 protects hosted user content. yt-dlp = user-directed processing, not hosting. With pass-through architecture (no caching past session, immediate processing), safe harbor preserved. | Preserved if architected as below |
| (4) §230 user-directed content | User pastes URL (provides the content). Per *Goldman 2025*, search-engine analogy applies. Defensible but unsettled. | Likely applies |

**Architectural guardrails (all in place):**
1. yt-dlp runs server-side, retention matches existing 48h upload retention. Pass-through architecture (no public cache of YouTube-extracted audio).
2. Per-URL attestation flow — modal with explicit "I confirm I have the right to use this content for personal music learning" checkbox before any processing. Required, no default-checked.
3. Server-side audit trail — `attestation_at`, `attestation_type`, `attestation_user_agent`, SHA256 IP hash all persisted on the job for any later good-faith inquiry.
4. UX framing — file upload is primary, YouTube URL is secondary fallback with italic "Don't have the file?" copy. The deliberate-secondary UI strengthens both the §230 user-directed-content posture and the rights-attestation (user must consciously opt in).
5. Fallback help page at `/youtube-fallback.html` with manual capture instructions for when yt-dlp fails on a specific video (mirroring Klangio's escape hatch).
6. ToS §4.4 explicitly covers URL-pasted content with rights warranty + acknowledgment that YT ToS compliance is the user's responsibility.

**Industry comparison:** Klangio (Germany), Moises, Chordify, and AudioShake all accept YouTube URLs. Klangio's posture has been undisturbed by litigation despite YouTube ToS exposure. Klangio's Germany jurisdiction gives them additional EU TDM cover (Borovick ch 6) that we don't have, but the architectural posture (user-directed, transformative output, no public storage) translates.

**Re-evaluation triggers:**
- A peer competitor (Klangio, Moises, Chordify, AudioShake) is sued for yt-dlp-based ingestion specifically
- *Bartz v. Anthropic* or *Concord v. Anthropic* produces a definitive ruling on AI training-data ingestion that changes the transformative-use calculus
- YouTube changes its enforcement posture against transcription tools (e.g., a public statement, a high-profile cease-and-desist)
- Our scale changes materially (e.g., 10x ARR or visible national press)

**Sources:** Passman 11th ed. Ch 17 (transcription license terminology), Ch 19 (AI authorship + transformative-use precedent including *Authors Guild v. Google*, *Andy Warhol Foundation v. Goldsmith*); Goldman 2025 ed. §230 chapter (incl. *Zeran v. AOL*, *Twitter v. Taamneh*, *Anderson v. TikTok*) and §512 chapter (DMCA safe harbor mechanics); Borovick 2024 ch 6 (AI training data + transformative use, EU TDM Article 4 comparison).

---

### 3.5 No public chord library, no stored transcriptions of others' compositions

**Position:** All processing is per-user, ephemeral. We do not host a searchable public library of chord charts for popular songs. The pre-existing 15,417-chart library was deleted Apr 16 per your direct instruction; only 20 Kozelski-original charts (compositions I own) remain. This avoids the *Ultimate Guitar / Genius* litigation pattern that has consistently produced rulings against public hosting of derivatives.

**Source basis:** Your direct guidance, Apr 10 2026 call. Goldman 2025 ed. covers analogous platform cases.

---

### 3.6 Karaoke / lyrics features remain disabled

**Position:** Karaoke output and any feature that distributes synchronized lyrics remains disabled per your Apr 10 directive. Whisper-generated lyric strings appear in the user's session-only chord chart but are not stored centrally or distributed. This avoids the *NMPA / LyricFind / Concord v. Anthropic* lyrics-licensing surface area.

**Source basis:** Your direct guidance, Apr 10 2026 call.

---

### 3.7 No on-platform redistribution of separated stems for commercial use

**Position:** ToS §5 (Acceptable Use) prohibits users from redistributing, reselling, or sublicensing separated stems or transcriptions of copyrighted works they do not own; from creating competing commercial products using our output (karaoke tracks, sample packs from copyrighted recordings); and from using the Service to train ML models without our written consent.

**Source basis:** Standard SaaS ToS pattern; Fishman 18th ed. Ch 20 (electronic contracts for online sales).

---

### 3.8 DMCA §512 agent registration is current and operative

**Position:** StemScriber is registered with the US Copyright Office as a DMCA agent (Registration # DMCA-1070849) with `support@stemscriber.com` as the designated agent address (cited on `/dmca.html`). Cloudflare email routing forwards to my Gmail; takedown notices reach me typically within minutes. DMCA notice-and-takedown procedures are documented on `/dmca.html`. Three-strike repeat-infringer policy is documented in ToS.

**Operational note:** Email-routing reliability is the single point of failure for §512 protection. I monitor this; current SLO is <24h response.

---

## 4. Architectural diagram

(Text representation for review.)

```
                ┌────────────────────────────────┐
USER ──────────▶│  upload.html (primary)         │
                │  - file drop zone              │
                │  - "Don't have the file?"      │──┐
                │    YouTube URL field (secondary)│  │
                └────────────────────────────────┘  │
                       │                            │
                       │ file upload                │ URL paste
                       ▼                            ▼
                ┌─────────────┐          ┌──────────────────────┐
                │ /api/upload │          │  Per-URL attestation │
                │  (existing) │          │  modal (REQUIRED     │
                └─────────────┘          │  checkbox before     │
                       │                  │  processing)         │
                       │                  └──────────────────────┘
                       │                              │
                       │                              │ POST /api/url
                       │                              │ + attestation_at
                       │                              │ + attestation_type
                       │                              │ + IP hash + UA
                       │                              ▼
                       │                  ┌──────────────────────┐
                       │                  │  yt-dlp (server-side,│
                       │                  │  pass-through to R2; │
                       │                  │  no public caching)  │
                       │                  └──────────────────────┘
                       │                              │
                       └──────────────┬───────────────┘
                                      ▼
                       ┌─────────────────────────────┐
                       │ Stem separation (Modal A10G)│
                       │ Chord detection (CPU/local) │
                       │ Chart formatter             │
                       └─────────────────────────────┘
                                      │
                                      ▼
                       ┌─────────────────────────────┐
                       │ Per-user output:            │
                       │ - 6 stems (downloadable)    │
                       │ - chord chart (JSON+UI)     │
                       │ - MIDI / GP5 export         │
                       │ Retention: 48h uploads,     │
                       │            7d outputs       │
                       │ NO public chord library     │
                       └─────────────────────────────┘
```

**Key invariants:**
- All processing is per-user (no cross-user reuse of others' uploaded content)
- No public hosting of derivatives (the chart-library deletion Apr 16 enforced this)
- All URL submissions require attestation; audit trail persisted
- DMCA §512 takedown is the cleanup mechanism if any user-warranty proves false

---

## 5. Re-evaluation triggers

The following events would, in my view, require revisiting the positions above. I am tracking them via Court Listener and music-industry press; please add anything I'm missing.

| Trigger event | What it would change |
|---------------|----------------------|
| Any of the active AI cases (*Bartz*, *Concord*, *NYT v. OpenAI*, *Getty v. Stability*) produces a ruling adverse to transformative-use defenses for AI training data | May force revisiting our characterization of audio-to-notation as transformative |
| A peer competitor (Klangio, Moises, Chordify, AudioShake) is sued for yt-dlp ingestion specifically | Would prompt immediate suspension of YouTube URL acceptance and reassessment |
| YouTube publicly changes enforcement posture against transcription tools, or sends StemScriber a cease-and-desist | Would prompt immediate suspension of URL acceptance pending counsel review |
| Major shift in our scale (10x ARR, national press coverage, B2B enterprise contract) | Would prompt revisit of the "below-radar" component of our risk posture |
| Copyright Office or court ruling that AI-generated derivatives ARE copyrightable | Would change our framing but likely not our practical operations |
| New EU AI Act enforcement actions affecting consumer-facing music AI | Would matter if we expand to EU users |

---

## 6. Items where I want your specific input

These are the small number of questions where I'm not confident enough to act without your sign-off, or where I want professional judgment with malpractice insurance backing.

### 6.1 Confirmation of the YouTube URL acceptance launch

**Question:** Given the architecture in §3.4 (deliberate-secondary UI, per-URL attestation, ToS §4.4, server-side audit trail, no public caching, fallback page), are you comfortable with the YouTube URL feature as launched today?

**My analysis:** Defensible per the four-layer breakdown. Industry-standard among peer services. ToS shift + attestation chain + transformative output + DMCA §512 + §230 give layered defense. The novel exposure (YT ToS) is operational, not existential.

**Asking for:** Yes / yes-with-tweaks / no, with specifics if anything in the architecture concerns you.

---

### 6.2 Trademark filing strategy for "StemScriber"

**Question:** Pre-launch trademark filing for "StemScriber" — recommend?

**My analysis (per Fishman 18th ed. Ch 6):** The mark is in active commercial use, has been for several months. Pre-launch federal registration (USPTO) prevents future squatting and gives nationwide priority. ITU (intent-to-use) was probably appropriate months ago; at this point a use-based application makes more sense. Cost: ~$350 USPTO filing fee + your hourly time, OR a flat-fee service like LegalZoom or Nolo's trademark filing service.

**Asking for:** (a) Recommend filing now vs. post-launch? (b) Is this work I'd hire you to do, or is a flat-fee online service reasonable for a routine word mark? (c) Does StemScriber's mark conflict with anything you're aware of?

---

### 6.3 Active AI litigation watch list — am I tracking the right cases?

**Question:** My re-evaluation trigger list (§5) names *Bartz v. Anthropic*, *Concord v. Anthropic*, *NYT v. OpenAI*, *Getty v. Stability* as the cases whose outcomes would meaningfully shift our posture. Am I missing any that you think are more directly relevant to a consumer audio-transcription tool?

**Asking for:** Add cases I should be tracking. Remove ones that are noise.

---

### 6.4 Anything else you'd flag

**Question:** What am I not thinking about that I should be?

I'd particularly value your read on:
- Whether the per-URL attestation flow's wording (Appendix B) is sufficient or needs strengthening
- Whether ToS §4.4 (Appendix A) covers what it needs to or should be expanded
- Anything about the launch timeline that you'd insist on closing before public announcement

---

## 7. Out of scope for this package

For clarity on what's NOT included here:

- **Privacy policy / GDPR / CCPA** — handled separately via Termly auto-generation; please flag if you want to review independently
- **Stripe / payment processing** — operating under standard Stripe ToS and 1099-K reporting; no novel issues
- **Employment law** — no employees yet; will revisit when hiring
- **Tax / entity** — handled by my CPA; structure is single-member LLC
- **Insurance** — separate matter, not yet in place; flagging that I should probably have professional liability + cyber

---

## Appendix A — Terms of Service §4 (Your Content and Responsibilities)

(Text of ToS §4.1 through §4.5 as currently live on stemscriber.com/terms.html. §4.4 is new today. Full text follows.)

> **§4.1 We Do Not Own Your Content** — You retain all rights to the audio files you upload ("User Content") and to the outputs the Service generates from your uploads (separated stems, chord charts, transcriptions, etc.). StemScriber does not claim any ownership of your content or the processed outputs.
>
> **§4.2 Processing License** — By uploading audio, you grant us a limited, non-exclusive, royalty-free license to process, temporarily store, cache, and transmit your content solely to provide the Service to you. This license ends when your content is deleted from our systems.
>
> **§4.3 You Must Have Rights to What You Upload** — You represent and warrant that you own or have obtained all necessary rights, licenses, and permissions for any audio you upload to StemScriber. This includes the right to make copies and create derivative works for personal practice purposes. You are solely responsible for ensuring that your use of the Service does not violate the rights of any third party, including copyright holders, recording artists, music publishers, and record labels. StemScriber relies on this representation and is not in a position to verify the copyright status of uploaded files.
>
> **§4.4 URLs You Submit** [NEW] — The Service offers an optional fallback that lets you paste a URL (such as a YouTube link) instead of uploading a file. When you do, the Service downloads the audio at that URL on your direction and processes it the same way as a direct upload. The same rights warranty in 4.3 applies to URLs. By submitting a URL, you represent and warrant that you have all necessary rights to use the audio at that URL for personal music learning. You acknowledge that the Service does not host the audio at the URL, does not control whether your access to it complies with the source platform's terms of service, and acts solely as a user-directed processing tool. You are responsible for any third-party terms of service or copyright issues that arise from URLs you submit. Before processing any URL, we require you to confirm this rights warranty through an in-product prompt.
>
> **§4.5 Intended Uses** — StemScriber is designed for lawful personal and educational use, such as: separating stems from recordings you own or have licensed for personal practice; generating chord charts and transcriptions from your own audio to learn songs; creating practice tracks for rehearsal or performance preparation; analyzing musical arrangements for educational purposes.

---

## Appendix B — Per-URL attestation modal copy (live in product as of today)

(Exact text shown to the user when they submit a YouTube URL. Modal blocks submission until checkbox is checked.)

> **Confirm rights**
>
> Before we process this YouTube link, please confirm:
>
> - You have the right to use this audio (you own it, it's licensed to you, or it's public domain)
> - StemScriber processes per-user only — we don't store or republish the audio
> - You're responsible for any copyright issues with the source material
>
> ☐ I confirm I have the right to use this content for personal music learning.
>
> [Cancel] [Get Chords]

Server-side, every URL submission persists: `attestation_at` (ISO timestamp), `attestation_type` (`youtube_user_rights_confirmation`), `attestation_user_agent` (browser UA string), `attestation_ip_hash` (SHA256 of remote IP, first 16 hex chars). Stored on the job metadata; retained for the job's retention period (currently 7d outputs, 48h uploads) plus access logs.

---

## Appendix C — Internal legal-decision log

The full internal `docs/legal-faq.md` is available on request. It documents each decision with citation, reasoning, and re-evaluation triggers — written contemporaneously as decisions were made. Maintained as a living document; new entries added whenever a question is resolved.

---

## Appendix D — How I prepared this

For transparency: this package was assembled from approximately 8 hours of research across the six texts cited in §2, plus active case-law tracking via Court Listener and music-industry trade press. An AI assistant (Claude) helped me search, synthesize, and structure the writing, but the positions taken are mine and the decisions to ship features are mine. I reviewed every citation against the source text. If you spot anything where the citation doesn't support the position, please flag — that's exactly the quality-control I'm asking for.

---

*End of package. Please return signed-off, with any caveats, at your convenience. No urgency unless you see a red flag.*
