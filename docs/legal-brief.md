# StemScribe Legal Brief

**Prepared:** March 25, 2026
**For:** Jeff Kozelski, Kozelski Enterprises LLC (South Carolina)
**Product:** StemScribe (stemscribe.io) — AI-powered music practice tool
**Status:** Live with paying customers (Free / Pro $10/mo / Premium $20/mo)

---

## 1. Executive Summary

StemScribe is an AI-powered music practice tool that:
- Accepts user-uploaded audio files or YouTube URLs
- Separates audio into individual stems (vocals, guitar, bass, drums, piano) using AI on cloud GPU (Modal)
- Displays chord charts and guitar tablature sourced from Songsterr API
- Maintains a library of 2,665 chord charts scraped from Ultimate Guitar
- Caches processed results server-side by URL for performance
- Charges subscription fees (Pro $10/mo, Premium $20/mo) via Stripe

**Current legal protections in place:**
- Kozelski Enterprises LLC (SC)
- USPTO trademark filed (Class 9 + Class 42)
- DMCA agent registered (DMCA-1070849)
- 6 legal pages deployed (ToS, Privacy, Security, Cookies, DMCA, Refund)
- Stripe payments live with refund policy

**Bottom line:** StemScribe has several significant legal vulnerabilities that need to be addressed before scaling further. Three areas are HIGH risk and require immediate action. The good news: competitors like Moises (70M users, $50M raised) have solved these problems — StemScribe can follow their playbook.

---

## 2. Risk Assessment

### HIGH RISK

#### 2a. YouTube URL Downloading (yt-dlp)
**Risk: HIGH** | Priority: IMMEDIATE

- YouTube's ToS explicitly prohibits downloading content
- Federal court rulings have found circumventing YouTube's protections can violate DMCA anti-circumvention provisions
- Under MGM v. Grokster (2005), providing a YouTube URL input field could constitute "inducement" of infringement
- **Moises explicitly REFUSES streaming URLs** — their docs state: "Attempting to extract or modify audio from a streaming service without permission can lead to legal consequences"
- The commercial nature of StemScribe weakens any personal-use defense
- This feature also undermines DMCA safe harbor eligibility (inducement disqualifies safe harbor)

**Action:** Remove YouTube URL feature. Require users to upload their own audio files only.

#### 2b. Scraped UG Chord Library (2,665 charts)
**Risk: HIGH** | Priority: IMMEDIATE

- UG has explicit anti-scraping clauses in their ToS
- UG has proper licensing deals with publishers and pays royalties — StemScribe scraped this licensed content without authorization
- Attribution was removed, making it worse — eliminates any fair use or citation argument
- Under hiQ v. LinkedIn, ToS violations can support breach of contract claims ($500K+ judgments)
- UG has previously pursued legal action against unauthorized tab services
- Chord progressions themselves are NOT copyrightable (Spirit v. Page/Plant confirmed this), but the specific arrangements, voicings, and notation in tabs may be

**Action:** Delete the scraped library entirely. Use AI-generated chord detection only (which StemScribe already has). Alternatively, license content from publishers directly.

#### 2c. Server-Side Stem Caching
**Risk: HIGH** | Priority: 30 DAYS

- **UMG v. MP3.com (2000) is directly on point:** Server-side copies of copyrighted music = infringement, even for user convenience. MP3.com paid $53M.
- Stem-separated audio qualifies as a derivative work under 17 U.S.C. Section 103
- URL-based caching means one user's upload could serve cached stems to another user — this transforms StemScribe from a "processing tool" into a "distribution platform"
- Every music tech service that stored/cached copyrighted content server-side without licenses was eventually sued

**Action:** Process ephemerally — deliver stems to client, delete from server. If caching is needed, make it per-user only with 24-48 hour auto-expiration. Never serve one user's cached stems to another user.

---

### MEDIUM RISK

#### 2d. Processing Copyrighted Music (Fair Use Question)
**Risk: MEDIUM** | Priority: 60 DAYS

- No definitive ruling yet on AI stem separation — RIAA v. Suno/Udio (filed June 2024) expected to produce guidance by summer 2026
- StemScribe's strongest argument: stem separation is *transformative* (changes purpose from entertainment to educational practice)
- However, Northwestern JTIP analysis concluded stem extraction "likely qualifies as a derivative work" requiring copyright owner authorization
- Audio Home Recording Act does NOT apply — it covers physical consumer hardware, not cloud-based processing
- **Moises's approach (70M users, not sued):** Users upload own files, ToS places all copyright liability on users, no caching/redistribution

**Action:** Strengthen ToS to follow Moises model — users warrant they own or have rights to uploaded content. StemScribe is a processing tool, not a content provider.

#### 2e. Songsterr API Usage
**Risk: MEDIUM** | Priority: 60 DAYS

- Songsterr's API is publicly accessible and they encourage third-party apps
- However, their ToS prohibits commercial use without "express advance written permission"
- Songsterr has legitimate licensing deals with publishers — they have standing to enforce
- StemScribe charges subscription fees, making this clearly commercial use

**Action:** Contact Songsterr about a commercial API license, or ensure data is used within their permitted guidelines with proper attribution.

#### 2f. DMCA Safe Harbor Compliance
**Risk: MEDIUM** | Priority: 30 DAYS

DMCA-1070849 is registered, but safe harbor requires ALL of these:
1. Designated agent registered ✅
2. Agent contact info published on website — **verify this is complete** (name, address, phone, email)
3. Repeat infringer policy — **must have AND implement**
4. No actual knowledge of infringement — problematic if caching known copyrighted songs
5. Expeditious removal upon DMCA notice — **must have working takedown procedure**
6. No direct financial benefit from infringement you can control
7. Renewal every 3 years

**Critical issue:** Safe harbor protects passive hosts of user content. StemScribe actively processes and transforms content — it may not qualify as a passive host at all. The YouTube URL feature especially undermines safe harbor eligibility.

**Action:** Ensure all 6 requirements are met. Publish clear DMCA policy, repeat infringer policy, and takedown procedure on the website. Removing the YouTube feature significantly strengthens safe harbor arguments.

---

### LOW RISK

#### 2g. Chord Progressions (Copyrightability)
**Risk: LOW**

- Case law is clear: chord progressions are "part of the common stock of musical raw material" and are NOT copyrightable
- Spirit v. Page/Plant (Stairway to Heaven) reinforced this
- AI-generated chord charts based on audio analysis are the safest approach
- The risk is only HIGH when the charts are scraped from licensed sources (see 2b above)

---

## 3. What Happened to Companies Like StemScribe

| Company | What They Did | Outcome |
|---------|--------------|---------|
| **MP3.com** | Server-side copies of CDs | **$53M damages** |
| **Grooveshark** | Streaming without licenses | **$75M penalty, shut down, forfeited all assets** |
| **LimeWire** | P2P with inducement | **$105M settlement, shut down** |
| **Jammit** | Licensed multitrack play-along (closest to StemScribe) | Shut down — even WITH licenses, couldn't sustain financially |
| **Napster** | P2P file sharing | Shut down, bankruptcy |
| **Moises** (70M users) | User-uploads-only stem separation | **NOT sued** — no streaming URLs, no caching, users bear liability |
| **Chordify** | Auto-extract chords | Operating — retraction process for rights holders, personal use only |
| **Yousician** (10K+ songs) | Licensed play-along | Operating — **full licensing deals with publishers** |
| **AudioShake** | B2B stem separation | Operating — **licensed by Warner, Universal, Sony, Disney** |

**Pattern:** Services without licenses that stored/distributed copyrighted content were all shut down or paid $30M-$105M. Survivors either (1) have licenses or (2) carefully avoid doing what StemScribe currently does.

---

## 4. How Competitors Handle This

| App | Accepts URLs? | Caches Stems? | Has Licenses? | Scraped Content? |
|-----|:---:|:---:|:---:|:---:|
| **Moises** | NO (upload only) | NO (ephemeral) | No (user liability) | NO |
| **Capo** | NO (user's library) | Local only | No (user liability) | NO |
| **Yousician** | N/A | Licensed | YES (publishers) | NO |
| **AudioShake** | N/A (B2B) | Licensed | YES (labels) | NO |
| **Chordify** | Analyzes URLs | No audio storage | Retraction process | NO |
| **StemScribe** | YES (YouTube) | YES (server) | NO | YES (2,665 UG charts) |

StemScribe is the only app doing ALL four risky things simultaneously.

---

## 5. Recommended Legal Actions (Priority Order)

| # | Action | Priority | Est. Cost | Timeline |
|---|--------|----------|-----------|----------|
| 1 | Remove YouTube URL downloading | IMMEDIATE | $0 (code change) | This week |
| 2 | Delete scraped UG chord library | IMMEDIATE | $0 (code change) | This week |
| 3 | Switch to ephemeral stem processing (no server cache) | HIGH | $0 (code change) | 30 days |
| 4 | Update ToS to Moises model (user bears liability) | HIGH | $500-1,500 (lawyer review) | 30 days |
| 5 | Complete DMCA compliance (policy, repeat infringer, takedown) | HIGH | $500-1,000 | 30 days |
| 6 | Get Songsterr commercial API license | MEDIUM | $0-5,000/yr (depends on terms) | 60 days |
| 7 | Initial attorney consultation | HIGH | $0-500 (many offer free) | 30 days |
| 8 | Trademark monitoring/prosecution | MEDIUM | $1,500-3,000/yr | 60 days |
| 9 | Patent evaluation (audio pipeline, chord detection) | LOW | $5,000-15,000 per patent | 90 days |
| 10 | Terms review by entertainment attorney | MEDIUM | $2,000-5,000 | 60 days |

**Estimated total first-year legal spend:** $5,000-15,000 (solo founder budget)

---

## 6. Top Lawyer Recommendations

### For Music Tech / Copyright / DMCA (Primary Attorney)

#### 1. Rosenberg, Giger & Perala P.C. — TOP PICK
- **Location:** New York City (nationwide)
- **Website:** rglawpc.com
- **Why:** They literally advise music tech startups on structuring platforms to avoid DMCA litigation. Clients include "some of the most significant and innovative entrepreneurs involved in the digital delivery of online content." This is exactly what StemScribe needs.
- **Free consult:** Contact via website

#### 2. Daniel Schacht — Donahue Fitzgerald LLP
- **Location:** Oakland/LA
- **Website:** donahue.com/attorneys/daniel-j-schacht/
- **Why:** Billboard "Top Music Lawyer" 2024 & 2025. Led the "Happy Birthday" public domain case. Co-counsel in Wixen v. Spotify (major mechanical licensing case). Teaches at UC Berkeley Law. Deep music + tech IP expertise.
- **Free consult:** Not advertised (boutique firm, likely accessible)

#### 3. Lindsay Spiller — Spiller Law
- **Location:** San Francisco / LA / Silicon Valley
- **Website:** spillerlaw.com
- **Why:** 20+ years advising founders. Clients include online music platforms, labels, YouTube channels. Combines startup law with entertainment expertise.
- **Free consult:** YES

#### 4. Jesse E. Morris — Morris Music Law
- **Location:** Marina del Rey / LA (Silicon Beach)
- **Website:** morrismusiclaw.com
- **Why:** Specifically serves music tech startups and new media businesses. Cost-effective, indie-friendly approach. Perfect for early-stage budget.
- **Free consult:** YES

#### 5. Cassandra Spangler, Esq.
- **Location:** New York City (nationwide)
- **Website:** cspanglermusiclaw.com
- **Why:** Indie-focused solo practitioner. Emphasis on independent artists and startup music companies. Affordable ongoing counsel.
- **Free consult:** Contact via website

### For IP / Trademark / Patents

#### 6. Miller IP Law (Devin Miller) — BEST FOR TRADEMARK
- **Location:** Utah (remote nationwide)
- **Website:** lawwithmiller.com
- **Why:** 1,000+ startups served. EE degree + MBA/JD — can understand your audio pipeline at a technical level. Flat fees: trademarks ~$750, patents $1,700-$5,500. 95% success rate, 100+ five-star reviews.
- **Free consult:** YES — free strategy session, responds within 30 minutes

#### 7. Schmeiser, Olsen & Watts LLP — BEST FOR PATENTS
- **Location:** NY (main), also NC office
- **Website:** schmeiserolsen.com
- **Why:** 40+ years of patent experience. Have prosecuted music tech patents (instrument control devices). Founding partner is a former USPTO Patent Examiner. NC office = closer to SC.
- **Free consult:** Contact via website

### Bonus: Local SC Option

#### Edward Fenno — Fenno Law (Charleston, SC)
- **Website:** fennolaw.com
- **Why:** SC bar member, entertainment/IP/tech law. Can handle SC-specific business matters while a specialist handles music tech specifics.

---

## 7. Questions for the First Consultation

### Ask Every Attorney:
1. "Given that Moises has 70M users and has never been sued, can we adopt their exact legal model? What would that look like for StemScribe?"
2. "Does stem separation of user-provided audio constitute fair use under current case law?"
3. "What's our exposure if we remove YouTube downloading, delete the UG library, and switch to ephemeral processing — does that reduce risk to an acceptable level?"
4. "Do we need music publisher licenses to operate, or can we rely on user-provides-own-content model like Moises and Capo?"
5. "Is our DMCA safe harbor registration sufficient, or do we need additional compliance steps?"
6. "What does the RIAA v. Suno/Udio ruling (expected summer 2026) mean for us?"

### Ask IP/Trademark Attorneys:
7. "Can you take over management of our existing USPTO filing (Class 9 + Class 42)?"
8. "Is our audio processing pipeline patentable? What about the chord detection algorithm?"
9. "What trade secret protections should we have for our proprietary technology?"
10. "What's the realistic cost to maintain trademark protection for a solo-founder startup?"

### Ask About Pricing:
11. "Do you offer flat fees or monthly retainers for startups?"
12. "What's a realistic annual legal budget for a solo-founder music tech startup at our stage?"
13. "Can you do a one-time legal audit of our platform and ToS for a fixed fee?"

---

## 8. Timeline: 30 / 60 / 90 Days

### Days 1-30: ELIMINATE HIGH RISKS
- [ ] **Week 1:** Remove YouTube URL downloading feature (code change, $0)
- [ ] **Week 1:** Delete scraped UG chord library (code change, $0)
- [ ] **Week 2:** Switch to ephemeral stem processing — no server-side cache, or per-user cache with 24-48hr expiration
- [ ] **Week 2:** Book free consultations with Lindsay Spiller (spillerlaw.com) and Jesse Morris (morrismusiclaw.com)
- [ ] **Week 3:** Book consultation with Rosenberg, Giger & Perala (rglawpc.com) for DMCA/platform structuring
- [ ] **Week 3:** Audit DMCA compliance — verify agent contact info on website, publish repeat infringer policy, implement takedown procedure
- [ ] **Week 4:** Book free strategy session with Miller IP (lawwithmiller.com) for trademark management

### Days 31-60: STRENGTHEN LEGAL FOUNDATION
- [ ] Hire primary music tech attorney (Rosenberg Giger or Spiller recommended)
- [ ] Have attorney review and update ToS, Privacy Policy, DMCA Policy
- [ ] Adopt Moises-model ToS: users warrant they own or have rights to content, user bears all copyright liability
- [ ] Contact Songsterr about commercial API license terms
- [ ] Begin trademark prosecution with Miller IP or chosen IP attorney
- [ ] Evaluate patent potential for audio pipeline and chord detection

### Days 61-90: SCALE SAFELY
- [ ] Complete all legal page updates based on attorney review
- [ ] Finalize Songsterr API license (or replace with alternative data source)
- [ ] File provisional patent application if attorney recommends (preserves priority date for $1,700-3,000)
- [ ] Set up trademark monitoring ($500-1,500/yr)
- [ ] Create internal compliance checklist for new features
- [ ] Evaluate whether to pursue music publisher licensing for premium features

---

## Sources

### Legal Analysis
- [AI Stem Extraction: Creative Tool or Facilitator of Mass Infringement?](https://jtip.law.northwestern.edu/2022/05/03/ai-stem-extraction-a-creative-tool-or-facilitator-of-mass-infringement/) — Northwestern JTIP
- [UMG Recordings v. MP3.com](https://en.wikipedia.org/wiki/UMG_Recordings,_Inc._v._MP3.com,_Inc.) — Key precedent on server-side copies
- [DMCA Safe Harbor Requirements](https://copyrightalliance.org/education/copyright-law-explained/the-digital-millennium-copyright-act-dmca/dmca-safe-harbor/) — Copyright Alliance
- [17 U.S.C. Section 512](https://www.law.cornell.edu/uscode/text/17/512) — DMCA Safe Harbors statute
- [Federal Court Ruling on YouTube Ripping Tools](https://www.webpronews.com/federal-court-ruling-on-youtube-ripping-tools-reshapes-digital-copyright-enforcement-under-dmca/)
- [Can You Copyright a Chord Progression?](https://www.local802afm.org/allegro/articles/can-you-copyright-a-chord-progression/) — Local 802 AFM

### Competitor Analysis
- [Moises Terms & Conditions](https://help.moises.ai/hc/en-us/articles/7401394754962-Terms-Conditions)
- [Moises: Why Streaming URLs Are Not Accepted](https://help.moises.ai/hc/en-us/articles/18322457396508)
- [AudioShake Licensing Derivative Works](https://www.audioshake.ai/post/licensing-derivative-works-how-ai-is-opening-the-door-to-legal-remixes-and-edits)
- [Disney Music Group x AudioShake](https://www.businesswire.com/news/home/20240715870771/en/)
- [Songsterr Terms of Service](https://www.songsterr.com/terms)
- [UG Tabs Legal & Licensing](https://www.ultimate-guitar.com/news/ug_news/are_ug_tabs_legal_and_does_ug_pay_money_to_artists_for_hosting_tabs_yes_and_yes.html)
- [Chordify Copyright FAQ](https://support.chordify.net/hc/en-us/articles/360001420738)

### Lawyer Sources
- [Rosenberg, Giger and Perala P.C.](https://www.rglawpc.com/intellectual-property-litigation/new-media-startups/)
- [Daniel Schacht — Donahue Fitzgerald](https://donahue.com/attorneys/daniel-j-schacht/)
- [Spiller Law](https://www.spillerlaw.com/)
- [Morris Music Law](https://morrismusiclaw.com/)
- [Cassandra Spangler Music Law](https://www.cspanglermusiclaw.com/)
- [Miller IP Law](https://lawwithmiller.com/)
- [Schmeiser, Olsen & Watts](https://schmeiserolsen.com/)
- [One LLP](https://onellp.com/)
- [Fenno Law](https://fennolaw.com/)

---

*This brief is for informational purposes only and does not constitute legal advice. Consult a licensed attorney for advice specific to your situation.*
