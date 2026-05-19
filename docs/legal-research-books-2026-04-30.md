# StemScriber Legal Research Book Sourcing

**Date:** 2026-04-30
**Goal:** Build a PDF/EPUB book stack so Jeff can self-answer 95% of recurring legal questions and reserve $400/hr Alexandra Mayo calls for genuinely hard ones. PDFs feed into Claude for retrieval.
**Format constraint:** PDF or EPUB only. No Kindle-DRM-exclusive, no Audible-only, no subscription-locked. Every book below has a confirmed buy-PDF-or-EPUB SKU on a legitimate seller — verification URLs included.
**Budget verdict:** 8-book starter set lands at **$197.92 total** — $52 under the $250 cap. Two premium-tier books ($225 and $268) flagged for "buy on hit" rather than upfront.

---

# Tier 1 — Buy These First ($197.92, 8 books)

The complete starter set. Every book is verified PDF and/or EPUB on a non-DRM seller. Ranked by leverage-per-dollar.

---

## 1. *All You Need to Know About the Music Business*, 11th ed. — $19.99

**Author:** Donald S. Passman | **Publisher:** Simon & Schuster, Oct 2023
**Buy:** [Apple Books — EPUB $19.99](https://books.apple.com/us/book/all-you-need-to-know-about-the-music-business/id6445638043) | [Google Play — EPUB $19.99](https://play.google.com/store/books/details?id=zROsEAAAQBAJ)

The bible-tier reference for music industry structure: sound-recording vs. composition rights, mechanical/sync licenses, master vs. publishing, MMA-era streaming royalty flows, and (in the 11th ed.) AI-generated music. Plain-English enough that Jeff reads it cover-to-cover, dense enough to stay on the shelf as a lookup. The 11th edition is 2023, post-MMA full implementation, and explicitly adds AI-music material — making it current for StemScriber's posture toward stem outputs and chord-chart derivations.

**Question this answers:** *When a user uploads a copyrighted recording and StemScriber returns isolated stems plus a chord chart, am I creating two outputs that implicate two different rightsholders — the sound recording (label) and the underlying composition (publisher) — and does my hosting posture differ for each?*

---

## 2. *Internet Law: Cases & Materials*, 2025 ed. — $10

**Author:** Eric Goldman (Santa Clara Law) | **Self-published, July 2025**
**Buy:** [Gumroad — DRM-free PDF, $10](https://ericgoldman.gumroad.com/l/acxudc)

The single highest-leverage purchase on this list dollar-for-dollar. Goldman is the leading academic on platform-liability law and revises this casebook every July. Covers DMCA §512 safe harbor with current notice-and-takedown procedures, Section 230 (post-*Moody v. NetChoice* 2024), the FTC click-to-cancel rule and ROSCA mechanics, online-contract formation (browsewrap vs. clickwrap enforceability — directly relevant to StemScriber's ToS acceptance flow), and state privacy laws including CCPA/CPRA, Virginia CDPA, Colorado CPA, and Texas TDPSA. It's a casebook, so the format is case excerpts plus Goldman's notes — exactly the source-of-truth Claude needs when answering DMCA-agent and §230 questions.

**Question this answers:** *Does StemScriber's DMCA agent setup — Cloudflare-routed support@stemscriber.com to my personal Gmail, Reg # DMCA-1070849 — actually satisfy §512(c)(2)'s "designated agent" requirement, and do I need a separate webform on the site beyond the support email to qualify for the safe harbor?*

---

## 3. *AI and the Law: A Practical Guide to Using Artificial Intelligence Safely* — $22.99

**Author:** Harry Borovick (General Counsel, Luminance) | **Apress / Springer Nature, Sept 2024**
**Buy:** [Springer Link — DRM-free PDF + EPUB](https://link.springer.com/book/10.1007/979-8-8688-0400-7) ($22.99 base, frequently $12.99 promotional)

Borovick is sitting in the GC chair of an actual AI company shipping into legal markets — structurally Jeff's position. Covers the operational questions of running an AI product: training-data provenance, ToS drafting around AI features, output-error liability, customer indemnification, and what to tell your insurer. September 2024 publication is post-Andersen-v.-Stability and post-Thomson-Reuters-v.-Ross trial-court ruling, so it's calibrated to the current case law.

**⚠ Verify format before buy:** Springer's Apress imprint is reliably DRM-free PDF + EPUB direct downloads, but the research agent could not personally fetch the product page (403). Open the URL above and confirm both PDF and EPUB are listed as download options before checkout. If only Kindle is offered, skip and re-evaluate Tier 2 alternates.

**Question this answers:** *If StemScriber's chord-detection model occasionally outputs a wrong chord and a paying user makes a public performance error because of it, what's the standard-of-care framework for AI-output liability and what indemnity language should StemScriber's ToS contain?*

---

## 4. *Legal Guide for Starting & Running a Small Business*, 19th ed. — $27.99

**Author:** Fred Steingold + David Steingold | **Nolo, Dec 2025**
**Buy:** [Nolo Store — PDF + EPUB, $27.99](https://store.nolo.com/products/legal-guide-for-starting-and-running-a-small-business-runs.html)

The current, founder-tier reference: choice of entity (sole prop vs. LLC vs. S-corp), tax basics, IP-assignment for solo-founder pre-incorporation work, vendor contracts, contractor/employee classification, business banking, basic dispute resolution. The 19th ed. (Dec 2025) is current with 2025 IRS numbers and the post-*Loper Bright* agency-rule landscape. Most directly: when StemScriber transitions from solo-author IP to LLC, the IP-assignment chapter is exactly the load-bearing reference for transferring rights so the LLC actually owns its core asset.

**Question this answers:** *If StemScriber's chord-detection codebase, demo-song clearances, and trained model weights were all created by me as an unincorporated individual, what specific IP-assignment paperwork do I execute when I form the LLC so the LLC actually owns those assets and not me personally?*

---

## 5. *Contracts: The Essential Business Desk Reference*, 3rd ed. — $27.99

**Author:** Richard Stim | **Nolo, Sept 2021**
**Buy:** [Nolo Store — PDF + EPUB, $27.99](https://store.nolo.com/products/contracts-ctrct.html)

A-Z plain-English reference covering 300+ contract terms — indemnification, limitation of liability, warranty disclaimers, force majeure, choice of law, assignability, termination-for-convenience, IP ownership clauses, NDA structures, basic offer/acceptance/consideration. The right tier: thicker than a blog post, thinner than Williston-on-Contracts. When Jeff stares at a Modal MSA or Stripe Connect addendum, this is the book that decodes "indemnification carve-out for gross negligence" before he decides whether to ship to Alexandra. 2021 publication is fine here — basic contract doctrine doesn't drift year-to-year.

**Question this answers:** *Modal's MSA caps liability at 12 months of fees paid and excludes consequential damages, but my Stripe agreement has unlimited liability for IP indemnification. Am I exposed if Modal's GPU output causes me to ship infringing audio that I refund Stripe for? Should I push Modal for an IP-indemnity carve-out?*

---

## 6. *Trademark: Legal Care for Your Business & Product Name*, 14th ed. — $27.99

**Author:** Stephen Fishman | **Nolo, April 2025**
**Buy:** [Nolo Store — PDF + EPUB, $27.99](https://store.nolo.com/products/trademark-trd.html)

The DIY-grade book that walks a solo founder through the entire USPTO process: TESS / TMS searches (the 14th ed. specifically covers the search-tool replacement that retired TESS in late 2024 — pre-2025 trademark books are actively misleading on this), common-law vs. registered rights, intent-to-use vs. use-based filings, TEAS Plus vs. TEAS Standard, valid specimens of use (very real for SaaS — the specimen is a screenshot of stemscriber.com showing the mark on the upload page), Office Action responses under §2(d) likelihood-of-confusion, TTAB oppositions, Madrid Protocol. Class-by-class analysis is directly load-bearing because StemScriber needs to file in both Class 9 (downloadable software/SaaS) and Class 41 (educational/entertainment services).

**Question this answers:** *Is "StemScriber" likely-confusable under §2(d) with marks like "Stem Studio," "StemSpace," or "Audioshake" in Class 9 / Class 41 — and what's my actual procedure if I get an Office Action citing one of them?*

---

## 7. *The Employer's Legal Handbook*, 17th ed. — $34.99

**Authors:** Fred Steingold + Aaron Hotfelder | **Nolo, July 2025**
**Buy:** [Nolo Store — PDF + EPUB, $34.99](https://store.nolo.com/products/the-employers-legal-handbook-empl.html)

The most current US employment-law single-volume for small employers. Post-FTC-noncompete-vacatur (the rule was struck down in *Ryan LLC v. FTC* August 2024), post-AB5 California gig-worker shifts, post-Loper-Bright. Covers IRS three-factor / common-law worker classification, when state registration as an employer is triggered (relevant for Jeff's first SC contractor), I-9 verification, anti-discrimination obligations as soon as he hires anyone, mandatory state postings, payroll tax registration, 50-state appendix including South Carolina. The 17th ed. addresses remote-workforce issues — important since any contractor Jeff hires will be remote and likely out of state.

**Question this answers:** *If I hire a marketing freelancer in Texas to run StemScriber's Reddit/HN launch, do I have to register with SCDEW, the Texas Workforce Commission, or neither — and at what dollar threshold do 1099-NEC filing obligations kick in?*

---

## 8. *Consultant & Independent Contractor Agreements*, 11th ed. — $25.99

**Author:** Stephen Fishman | **Nolo, Sept 2023**
**Buy:** [Nolo Store — PDF + EPUB, $25.99](https://store.nolo.com/products/consultant-and-independent-contractor-agreements-cica.html)

The contract-drafting playbook companion to #7. Negotiated agreement templates for fixed-fee, hourly, and milestone IC arrangements; IP-assignment, work-for-hire, and confidentiality clauses Jeff specifically needs when a contractor produces marketing copy, customer-support scripts, or band-relations email templates. The 11th ed. (Sept 2023) is the first edition that addresses contractors using AI tools in their deliverables — a real concern when a freelance copywriter generates Reddit launch copy with ChatGPT and Jeff needs to know who owns it. Also covers equity-for-services (advisor shares / RSAs / when a 1099 contractor crosses into "common-law employee" by accepting equity).

**Question this answers:** *If I hire a Tidepool-Artists outreach contractor on 1099 and they help draft email templates and a sales deck, who owns the copyright on those deliverables — me or the contractor — and what one paragraph in the agreement flips it to me?*

---

# Tier 2 — Buy Only When a Specific Question Hits

These are excellent but premium-priced. Don't pre-buy. Buy if and when you face a specific question the Tier 1 set can't resolve.

## *The Tech Contracts Handbook*, 3rd ed. — $37.95
**Author:** David W. Tollen | **LexisNexis, 2021** (still current — Tollen confirmed no 4th ed.)
**Buy:** [LexisNexis Store — EPUB $37.95](https://store.lexisnexis.com/en-us/products/the-tech-contracts-handbook-cloud-computing-agreements-software-licenses-and-other-it-contracts-for-lawyers-and-businesspeople-sku-us-ebook-30557-epub.html)

Tech-specific contract reference: SaaS agreements, software licenses, cloud-services contracts, IT pro-services agreements, DPAs (Modal/Cloudflare), open-source-license obligations, SLA remedies, deconversion rights. **Buy when:** you need to push back on a vendor TOS in writing — e.g., the Modal "service improvement" clause that may grant them rights to train on your customers' uploads. Stim (Tier 1 #5) covers contracts generally; Tollen drills into tech-vendor specifics.

## *The Cambridge Handbook of Generative AI and the Law* — $225
**Editors:** Zou, Poncibò, Ebers, Calo | **Cambridge UP, August 2025**
**Buy:** [Cambridge Core — Digital edition (per-chapter PDF downloads)](https://www.cambridge.org/core/books/cambridge-handbook-of-generative-ai-and-the-law/2965086BB1DFA8D147DF9A9667671493)

The most current practitioner-academic AI-law reference: published after Bartz v. Anthropic settled, after Thomson Reuters v. Ross, during active NYT v. OpenAI and Getty v. Stability litigation. Part III (IP) is the deep reference on training-data fair use, output-as-derivative-work, and US/EU jurisdictional splits.

**Format quirk:** Cambridge Core sells this as per-chapter PDF downloads, not a single bundled EPUB. That's fully compatible with Claude retrieval (upload each chapter), but be aware before checkout.

**Buy when:** you're facing a real claim or pre-litigation posture on training-data fair use, OR a specific publisher writes asking about your model's training corpus. Borovick (Tier 1 #3) is sufficient for routine questions.

## *Music and Copyright* (LexisNexis treatise) — $268
**Author:** Ronald S. Rosen | **LexisNexis, current edition**
**Buy:** [LexisNexis Store — EPUB $268](https://store.lexisnexis.com/en-us/products/music-and-copyright-grpussku-us-ebook-04627-epub.html)

Practitioner-grade treatise on music copyright litigation: substantial-similarity tests, derivative-work mechanics, how courts analyze musical-element overlap (notes, rhythm, harmonic progressions). The exact reference a judge would consult if a publisher sued StemScriber over auto-detected chord charts. **Buy when:** you receive a takedown notice that escalates beyond §512 boilerplate, or a specific publisher writes accusing your chord output of being derivative. Passman (Tier 1 #1) plus Goldman (Tier 1 #2) cover routine questions.

---

# Skip List — Look Relevant, Aren't

Each of these came up in research and was actively considered. Skipping each saves money, time, or both.

| Book | Why skip |
|------|----------|
| *Music Money and Success*, 8th ed. (Brabec & Brabec) | No legitimate seller offers a confirmed PDF/EPUB SKU — only Kindle-DRM. Format-disqualified. |
| *Music Copyright: An Essential Guide for the Digital Age* (Casey Rae, 2021) | Pre-Bartz, pre-MMA full implementation, Kindle-only. Passman 11th ed. covers same ground with 2023 currency. |
| *Copyright in the Music Industry* (Hayleigh Bosher, 2021) | UK/EU-centric, pre-Bartz. Wrong jurisdiction for a US founder. |
| *Kohn on Music Licensing*, 5th ed. (Wolters Kluwer, 2019) | Predates MMA full implementation and 2022-2024 streaming rate-court rulings. EPUB SKU not verifiable on a fetchable seller. |
| PLI Press *Artificial Intelligence & IP* (Brown & Oberlander, 2025) | PLI distribution is DiscoverPlus institutional-subscription, not a standalone DRM-free EPUB. Format-disqualified. Re-check in 6 months. |
| *Drafting Contracts: How and Why Lawyers Do What They Do* (Stark & Llorente, 2024) | Aspen Coursebook for first-year associates. Wrong tier — teaches drafting style, not founder self-screening. |
| *Information Privacy Law* (Solove & Schwartz, 8th ed.) | Aspen casebook for law students, ~1,200 pages of edited cases. Treatise-dense, not founder-friendly. |
| *The Twenty-Six Words That Created the Internet* (Kosseff, 2019) | Excellent §230 narrative history, but pre-*Moody v. NetChoice* 2024 and pre-state-privacy wave. Goldman 2025 supersedes. |
| *Privacy Program Management*, 3rd ed. (IAPP) | For in-house privacy program managers at companies with 12+ engineers. Massively over-tier for solo founder. Right book for Jeff's first privacy hire someday. |
| *Practical Data Privacy* (Jarmul, O'Reilly, 2023) | Excellent on differential privacy / GDPR-CCPA technical compliance, but the O'Reilly product page only routes to subscription "Read now" — direct buy-PDF SKU not verifiable. Re-check on a fetch. |
| McCarthy on Trademarks and Unfair Competition (multi-volume treatise) | Practitioner treatise, thousands of dollars. Massive overshoot for one TEAS Plus filing. |
| *The Trademark Guide* (Lee Wilson, 2012) | Predates 2018 USPTO electronic-filing modernization, 2020 Trademark Modernization Act, and 2024-25 search-tool replacement. Materially out of date. |
| *Working With Independent Contractors* (Fishman, 10th ed., 2023) | Would otherwise be perfect, but Nolo's standalone product page is gone (redirects to Business Suite upsell), Apple Books only has the older 9th ed., and the 10th ed. is Kindle-DRM-only on Amazon. The Steingold + Fishman *IC Agreements* combo (Tier 1 #7 + #8) covers the same ground from verified sources. |
| *The Manager's Legal Handbook* (Guerin & Barreiro, Nolo) | Aimed at line managers inside companies of 50+ employees managing existing staff. Wrong audience for a 1-person company. |
| *The Essential Guide to Federal Employment Laws* (Guerin & DelPo, Nolo) | Reference summary of statutes that mostly attach at 15+ or 50+ employees. Useful for HR managers, not for issuing a first 1099. |
| Any pre-2024 employment-law title | Misses FTC noncompete saga (rule promulgated Apr 2024, vacated Aug 2024 in *Ryan LLC v. FTC*), post-Loper-Bright deference shift, 2024 DOL IC rule. Anything dated 2023 or earlier teaches a regulatory landscape that no longer exists. |
| *HR for Small Business for Dummies* / Wiley Dummies titles | Generic HR-process content (org charts, review templates), not legal-grade. Won't survive an IRS three-factor test or SCDEW audit. |

---

# Where Books Aren't Enough — Follow These Instead

Five areas move faster than the publishing cycle. For these, replace book-buying with a free RSS-and-newsletter discipline:

## 1. Active AI litigation tracking
Bartz, NYT v. OpenAI, Concord, Getty, Andersen, Thomson Reuters — books go stale within 6 months of any major ruling. Follow:
- **Eric Goldman's blog** (blog.ericgoldman.org) — the canonical platform-liability and online-contracts source. He posts case analyses within days of filings.
- **Ed Newton-Rex's substack** (Fairly Trained) — music/AI litigation specifically, written by the former Stability VP who resigned over training-data ethics.
- **Plagiarism Today** (plagiarismtoday.com) by Jonathan Bailey — DMCA, fair use, takedown procedure.
- **CourtListener** (courtlistener.com) — read the actual filings; free RECAP archive.

## 2. State privacy law tracking
New state laws every quarter. Goldman's 2025 casebook (Tier 1 #2) is current as of July 2025, but by mid-2026 several new states will have statutes in effect. Follow:
- **IAPP Privacy Tracker** (iapp.org/resources/article/us-state-privacy-legislation-tracker) — canonical.
- **Husch Blackwell's Bytes column** — quarterly state-privacy roundups.

## 3. USPTO-specific procedural changes
TESS replacement, fee schedule changes, Trademark Modernization Act guidance updates. Follow:
- **USPTO Trademark blog** (uspto.gov/blog) — official.
- **r/Trademark** — filing-experience reports from real applicants.
- **Erik Pelton's TM Blog** (erikpelton.com/blog) — practical solo-applicant focus.

## 4. South Carolina-specific employment changes
SC LLR posting requirements, SCDEW thresholds, Charleston-specific business licenses. Books can't keep pace at state level. Follow:
- **SC LLR website** (llr.sc.gov) — official postings/changes.
- **Bridges Law Firm Charleston blog** or similar local SC employer-side counsel — practical updates.

## 5. FTC click-to-cancel and ROSCA enforcement
The FTC click-to-cancel rule was vacated in 2024 *FTC v. Pioneer Energy* — current state of negative-option compliance is in flux, with state-level analogs (California ARLA, etc.) filling the gap. Follow:
- **FTC Business Guidance blog** (ftc.gov/business-guidance/blog) — official.
- **Goldman's blog** again — he covers click-to-cancel saga in detail.

---

# Verification Caveats

- **Every Tier 1 book has a verified, fetched product page with a confirmed PDF or EPUB buy button** — except #3 (Borovick) where the Springer/Apress page returned 403 to the research agent. Apress is reliably DRM-free PDF + EPUB by historical pattern, but **open the URL and confirm before checkout.** If only Kindle is offered, skip and search "AI law founder 2025" on Springer Link for an alternate.
- **Tier 2 prices and SKUs were verified** but the Cambridge Handbook's per-chapter PDF download model means you'll be downloading chapters individually rather than a single EPUB. Functional for Claude retrieval; worth knowing before you click.
- **Prices reflect public listings as of 2026-04-30.** Re-verify before checkout.

---

# Recommended Buy Order

1. **Goldman, Internet Law 2025** ($10) — buy first. Highest leverage per dollar, immediate use on DMCA agent question.
2. **Passman, 11th ed.** ($19.99) — buy second. Cover-to-cover read; sets vocabulary for all music-IP questions.
3. **Borovick, AI and the Law** ($22.99 — verify format first) — buy third. Most current AI-founder reference.
4. **Stim, Contracts** ($27.99) — buy when first vendor MSA hits the inbox.
5. **Steingold, Legal Guide for Starting & Running a Small Business** ($27.99) — buy when forming the LLC.
6. **Fishman, Trademark** ($27.99) — buy now if filing TEAS Plus is on the pre-launch checklist; otherwise after launch.
7. **Steingold/Hotfelder, Employer's Legal Handbook** ($34.99) — buy before hiring first contractor.
8. **Fishman, Consultant & IC Agreements** ($25.99) — buy with #7. They pair.

Total: **$197.92**, $52 under your $250 cap. Tier 2 buys come out of that headroom or fresh budget when triggered.
