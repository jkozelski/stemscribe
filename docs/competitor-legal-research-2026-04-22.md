# Competitor Legal Page Research — 2026-04-22

**Launch:** 20 days out (2026-05-12). Research-only doc; feeds a later edit of terms.html / privacy.html after Alexandra signs off.

Sources were fetched live today where possible. Klangio, Moises (help center), Yousician, Songsterr returned clean content. Ultimate Guitar, Chordify, and the main Moises terms page returned 403 to the fetch tool; those rows rely on yesterday's `tos-research-2026-04-21.md` extracts (which were pulled from browser) plus today's web-search confirmation.

---

## Executive summary

- **Biggest single gap:** our ToS has no explicit "no-AI-training" clause. Moises/Music.ai put this in ALL-CAPS at the headline level ("WE DO NOT USE YOUR USER CONTENT OR OUTPUT TO TRAIN AI MODELS"). We only mention it under §10.1 and in privacy. Move it up and harden it.
- **We do not yet cover YouTube URL submissions in the user warranty.** Every peer that accepts URLs (Klangio, VocalRemover, Chordify) makes the user affirm authority to process that URL. We accept YouTube URLs in product but our §4.3 warranty only talks about "audio you upload."
- **No arbitration / class-action waiver.** Yousician has one (NY law, individual-only claims). Ultimate Guitar is famously litigated — they rely heavily on this posture. We default to SC state/federal courts with a 1-year claim window, which is OK but weaker than an AAA arbitration + class waiver combo.
- **DMCA agent listing is incomplete.** We list an email and a city. Ultimate Guitar lists full street address + phone; Moises lists Salt Lake City street address; Yousician lists Helsinki. A full mailing address is effectively required for a registered DMCA agent under 17 USC §512(c)(2).
- **Indemnification is one-way (user → us), which matches the industry.** No competitor we reviewed indemnifies the user. We're aligned here.

---

## Per-competitor extracts

### Klangio (klang.io/terms, fetched 2026-04-22)

- **AI training:** not addressed in ToS; privacy policy also silent. Gap in their own posture.
- **User warranty:** "your use of the Service will not infringe the proprietary rights... of any third party; you have legal authorization to modify files and YouTube videos you submit." (verbatim)
- **YouTube risk-shift:** "by using our service, you agree to be bound by the YouTube Terms of Service." Explicitly names the YouTube API. We use yt-dlp so we can't use this exact language.
- **DMCA:** no formal DMCA agent. Uses a retraction alias: `retract@klangio.com`. No three-strike policy. This is weak — they're a German co. leaning on GDPR, not DMCA safe harbor.
- **Retention:** "Files uploaded to the Service are temporarily stored, optimized and deleted within 30 days." Log entries 31 days.
- **Indemnification:** user indemnifies Klangio.
- **Arbitration:** none.
- **Governing law:** Germany; venue Karlsruhe.

### Moises (help.moises.ai/7401394754962 + sibling policy docs; partially fetched today, rest from 2026-04-21 notes)

- **AI training:** "WE DO NOT USE YOUR USER CONTENT OR OUTPUT TO TRAIN AI MODELS WITHOUT YOUR PERMISSION." (ALL-CAPS in original.)
- **User warranty:** "you are the creator and owner... or have the necessary licenses, rights, consents, and permissions." Also ALL-CAPS: "YOU ARE SOLELY RESPONSIBLE FOR ENSURING YOU HAVE ALL NECESSARY RIGHTS TO UPLOAD CONTENT FROM ANY THIRD-PARTY SERVICE."
- **YouTube / URL ingestion:** Moises explicitly **rejects** streaming URLs (YouTube, Spotify, Apple Music, SoundCloud, Tidal) citing DRM and derivative-work concerns. They accept Dropbox / iCloud / OneDrive links only. So their risk-shift is: refuse the risk entirely.
- **DMCA:** `copyright@moises.ai`. Physical agent: "Moises Systems, Inc. d/b/a Music AI, Attn: Copyright Agent, 136 South Main Street, Suite 400, Salt Lake City, Utah 84101."
- **Retention:** limited license "to host, store, transfer, reproduce, modify... solely to process the User Content and generate Output on your behalf." Music.ai sibling: may retain "indefinitely in anonymized form."
- **Indemnification:** user indemnifies Moises.
- **Arbitration:** present per 2026-04-21 notes (not re-verified today).
- **Karaoke:** no lyrics product, avoids composition-copyright exposure by not offering lyric/karaoke output at all.

### Chordify (chordify.net/pages/terms-and-conditions/; 403 to fetch, extracts from 2026-04-21 notes + today's web search)

- **AI training:** not explicitly addressed in ToS.
- **User warranty:** "You guarantee that such data and results are not illegal and will not infringe rights of third parties."
- **YouTube risk-shift:** embeds YouTube player via YouTube API — "Chordify uses YouTube API Services to play embedded videos, which are subject to YouTube's Terms of Service." They never re-host the audio — they show YouTube's own player above the chords. That is their core legal shield.
- **DMCA:** `retract@chordify.net`. No three-strike policy published.
- **Retention:** not quantified; don't claim to host source audio.
- **Karaoke:** no karaoke; just synchronized chord boxes. Position: "chord progressions are not considered innovative or sufficiently unique enough to be copyrighted on their own" (public support article). This is their affirmative defense — argue chords aren't copyrightable, don't host audio, don't display lyrics.
- **Governing law:** Netherlands (Groningen), per their support docs.

### Songsterr (songsterr.com/a/wa/terms, fetched 2026-04-22)

- **AI training:** not addressed.
- **User warranty:** "the Content is yours (you own it) and/or you have the right to use it." Users also warrant no IP / privacy / publicity violations.
- **Platform license:** Songsterr grants itself "use, modify, publicly perform, publicly display, reproduce, and distribute" rights over user tab submissions — much broader than Moises/StemScriber take.
- **DMCA:** `support@songsterr.com` with subject "Copyright Infringement." No dedicated agent, no three-strike.
- **Indemnification:** user indemnifies Songsterr; user covers attorneys' fees + court costs.
- **Arbitration / class waiver:** none. Wisconsin small-claims court, $10,000 cap on monetary judgments. Distinctive and aggressive forum-selection move — drives plaintiffs to small claims rather than federal court.
- **Governing law:** Wisconsin.
- **Retention:** not addressed.

### Yousician (yousician.com/terms-of-service, fetched 2026-04-22)

- **AI training:** not addressed.
- **User warranty:** users affirm necessary licenses, no law/third-party violation, no false affiliation.
- **DMCA:** `copyrightagent@yousician.com`. Mailing: "Yousician Oy, Siltasaarenkatu 16, FI-00530 Helsinki, Finland, Attn: Legal."
- **Arbitration / class waiver:** "YOU AGREE THAT YOU MAY BRING CLAIMS AGAINST YOUSICIAN ONLY IN YOUR INDIVIDUAL CAPACITY AND NOT AS A PLAINTIFF OR CLASS MEMBER IN ANY PURPORTED CLASS OR REPRESENTATIVE PROCEEDING." (verbatim)
- **Indemnification:** user indemnifies Yousician; survives termination.
- **Retention:** inactive free accounts terminated after 180 days.
- **Karaoke/lyrics:** not specifically addressed — Yousician's library is pre-licensed, so the composition-copyright risk sits with their licensors, not their ToS.
- **Governing law:** US users → New York law / NY courts. Others → Finnish law / Helsinki courts.

### Ultimate Guitar (ultimate-guitar.com/about/tos.htm + dmca.htm; 403 to fetch, extracts from 2026-04-21 notes + public dmca page via search)

- **AI training:** not addressed.
- **User warranty:** "you have obtained all necessary rights, licenses or clearances."
- **DMCA:** `copyright@ultimate-guitar.com`. Full agent: "Ultimate Guitar USA LLC, 268 Bush Street, #3044, San Francisco, CA 94104; phone (415) 599-4620 Ext. 2." This is the model we should emulate.
- **YouTube content:** UG doesn't try to remove YouTube-hosted content — they direct takedowns to YouTube directly. Clean separation of responsibility.
- **Indemnification:** user indemnifies UG.
- **Retention after termination:** "reasonable period of time for backup, archival, or audit purposes" — deliberately vague.
- **Lyrics/tab:** personal, non-commercial use only; licensed-composition carve-out acknowledging UG has deals with NMPA member publishers for some tabs. This is post-2017 NMPA settlement posture.
- **Arbitration / class waiver:** not confirmed today; worth re-verifying from a browser — UG has been sued multiple times (latest: auto-renew class action per topclassactions.com hit) so they likely have updated arbitration language.

---

## Gap analysis vs StemScriber current (terms.html + privacy.html)

| Area | StemScriber today | Peer norm | Gap? |
|---|---|---|---|
| No-AI-training clause in ToS | §10.1 one sentence, buried | Moises: ALL-CAPS at headline. Music.ai: ALL-CAPS §11.4 | **Yes — promote and harden** |
| YouTube URL user warranty | §4.3 says "audio you upload" — doesn't mention URLs | Klangio, VocalRemover name URLs explicitly | **Yes — add URL prong** |
| DMCA agent listing | Name + email + "Charleston, SC, USA" | Full street address + phone (UG, Moises) | **Yes — add full address** |
| Three-strike policy | §9.4 present and well-drafted | Most peers don't even have this | **Strength — keep** |
| Arbitration / class waiver | Not present; SC courts, 1-yr claim window | Yousician has explicit class waiver | **Gap — lawyer call** |
| Indemnification direction | User → us only (§13) | Industry standard | Aligned |
| Retention (audio 48h, output 7d) | Clear, in both ToS and privacy | Better than Klangio's 30-day vague, matches LALAL's 1-day | **Strength — keep** |
| Commercial-use carve-outs on Output | §5 prohibits repackaging as sample packs | Moises §11.2 also bans AI-training-as-customer | Mostly aligned; could add "don't train AI on our Output" |
| Karaoke / lyrics | Disabled entirely per Apr 10 lawyer call | Chordify avoids by chords-only; Moises avoids by no-lyrics | Aligned |
| Governing law | South Carolina / Charleston County | Varies; US peers use CA/NY/WI | OK for solo LLC |
| Limitation of liability cap | $100 or 12 months of fees | Industry standard | Aligned |
| User-content grant scope | "limited, non-exclusive, royalty-free... solely to provide the Service" | Moises §11.3 uses nearly identical wording | Aligned |
| Upload consent modal | Silent on YouTube + AI training per launch prep | Should mirror the ToS representations | **Gap — update modal** |

---

## Recommended clauses to add

**1. Promote AI-training clause to headline (new §6.5 or new §10 header).** Modeled on Moises §11.4:

> STEMSCRIBER DOES NOT USE YOUR SUBMITTED CONTENT OR OUTPUT TO TRAIN, FINE-TUNE, OR OTHERWISE DEVELOP ARTIFICIAL INTELLIGENCE OR MACHINE LEARNING MODELS. NO LICENSE IN THESE TERMS INCLUDES ANY SUCH RIGHT.

Put it in ALL-CAPS and pull it out of §10. Move the sentence currently at §10.1 up to the top of §4 so users see it before they click "Accept."

**2. Extend §4.3 user warranty to cover YouTube URLs.** Add a subsection:

> When you submit a YouTube or third-party URL for processing, you represent and warrant that you have all necessary authorizations to submit that video to StemScriber, including compliance with the source platform's Terms of Service. StemScriber is not affiliated with, endorsed by, or sponsored by YouTube or Google.

Language kept neutral on retrieval mechanism (we use yt-dlp, not the YouTube API — Alexandra flag).

**3. Full DMCA agent listing per UG model.** Replace §9.5 with name + street address + phone + email. Need Jeff's registered business address. Also file with the USCO DMCA Designated Agent directory ($6 fee, takes ~2 weeks).

**4. Add arbitration + class-action waiver (new §15.5).** Modeled on Yousician:

> YOU AGREE THAT YOU MAY BRING CLAIMS AGAINST KOZELSKI ENTERPRISES LLC ONLY IN YOUR INDIVIDUAL CAPACITY AND NOT AS A PLAINTIFF OR CLASS MEMBER IN ANY PURPORTED CLASS OR REPRESENTATIVE PROCEEDING.

Flag for Alexandra: pair with AAA consumer arbitration? Opt-out window (30 days)?

**5. Add explicit §5 ban on using StemScriber Output to train AI.** We prohibit "using the Service to train ML models" but not "using Output we generated to train ML models." Moises §11.2 does both.

**6. Update upload-consent modal** to mirror ToS representations (current text doesn't mention URL authorization or AI training).

---

## Open questions for Alexandra

1. **yt-dlp vs YouTube API:** Klangio and Chordify both rely on "you agreed to YouTube ToS" language because they use the official API. We don't. Is neutral language ("third-party URL") defensible, or does the retrieval mechanism itself expose us? Architectural decision.
2. **Arbitration + class-action waiver:** worth the added complexity + AAA fees for a solo LLC at launch? Or is SC courts + 1-year claim window enough for now, revisit at Series A?
3. **DMCA agent address:** Jeff wants to keep the home address off public legal docs. Is a registered agent service (e.g., Northwest, ~$125/yr) the right answer, or use the Charleston LLC address on file with the SC SOS?
4. **Output training ban scope:** if a user trains their own personal model on stems they generated (not distributed), should we actually police that? Current prohibition on "using Service to train models" already covers it loosely.
5. **Songsterr third-party tab embed (§7):** we display Songsterr tabs. Do we need our own copy of Songsterr's user warranty flowing through to our users, or is "subject to Songsterr's terms" disclaimer enough?
6. **Class waiver enforceability in SC:** some state courts have pushed back on standalone class waivers without arbitration (per Dentons July 2025 alert). Pair with arbitration or skip entirely?
7. **Retention on deleted-account data:** our privacy says 30 days post-deletion for account data. OK for GDPR Art. 17? Some EU regulators push for shorter.

---

**Sources consulted (2026-04-22 fetch unless noted):**

- [Klangio Terms](https://klang.io/terms/) — fetched today
- [Klangio Privacy](https://klang.io/privacy/) — fetched today
- [Yousician Terms](https://yousician.com/terms-of-service) — fetched today
- [Songsterr Terms](https://www.songsterr.com/a/wa/terms) — fetched today
- [Moises T&C](https://help.moises.ai/hc/en-us/articles/7401394754962-Terms-Conditions) — 403 today; extracts from 2026-04-21 browser pull
- [Ultimate Guitar ToS](https://www.ultimate-guitar.com/about/tos.htm) — 403 today; extracts from 2026-04-21 browser pull
- [Chordify T&C](https://chordify.net/pages/terms-and-conditions/) — 403 today; extracts from 2026-04-21 browser pull
- Prior research: `~/stemscribe/docs/tos-research-2026-04-21.md`, `~/stemscribe/docs/competitor-legal-research.md`, `~/stemscribe/docs/chordify-legal-research.md`
- [Ultimate Guitar auto-renew class action](https://topclassactions.com/lawsuit-settlements/consumer-products/subscriptions/lawsuit-says-ultimate-guitar-traps-users-in-unwanted-auto-renewals/) — context on why UG hardened their ToS
- [Dentons on standalone class waivers, July 2025](https://www.dentons.com/en/insights/alerts/2025/july/15/enforceability-of-stand-alone-class-action-waivers) — relevant to #6 above
