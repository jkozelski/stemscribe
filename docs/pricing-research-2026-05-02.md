# StemScriber Pricing Research & Recommendation

**Date:** 2026-05-02
**Soft launch:** June 20, 2026 (Refinery cohort) — pricing must be locked before pre-show social arc starts
**Current pricing:** Free (3 songs/mo) | Pro $10/mo | Premium $20/mo | **no annual plan**
**Triggering question:** Audio Jam at $39/yr makes StemScriber's annualized $120 look 3× expensive. Is that a real problem or perceived?

**TL;DR:**
- StemScriber's $10/mo monthly is fine — it's the missing annual plan that creates the perceived gap, not the monthly price.
- Audio Jam's $39 is supported by an iOS-IAP-inflated monthly anchor; their effective web-monthly annualizes to ~$84/yr, not $120.
- Recommended structure: Free (3 songs) | Pro $10/mo OR **$89/yr** | drop the Premium tier for now, reintroduce as "Studio" post-launch when usage data is real | optional Lifetime $199 LTD capped at 100 launch buyers.
- 🔴 The biggest risk is dropping the monthly price below $10 to chase Audio Jam. Race-to-bottom doesn't end well when your COGS isn't zero.

---

# 1. Pricing Matrix 🟢

Verified 2026-05-02. "FETCHED" = pricing page returned full data; "PARTIAL" = some tiers visible, others gated/secondary; "BLOCKED" = pricing page 403/404, numbers from credible third-party 2026 reviews.

| Company | Free Tier | Monthly Pro | Annual Pro | Annual = $/mo | Premium / Top Tier | Lifetime | Verification |
|---------|-----------|-------------|------------|----------------|---------------------|----------|--------------|
| **Audio Jam** | 3 projects, 1-min chord cap, 0.9-1.1× speed | $6.99 web / $12.99 iOS IAP | $39.99/yr (also $29.99 IAP) | **$3.33/mo** | none | none | PARTIAL |
| **Moises** | limited uploads + ads | ~$3.99/mo¹ | ~$35.88/yr¹ | $2.99/mo¹ | $9.99/mo Pro (API) | none | BLOCKED (login wall) |
| **LALAL.AI** | 10 min Relaxed Queue, 200MB | $7.50 Lite / $15 Pro | $90 Lite / $180 Pro | **$7.50/$15 (0% off)** | minute top-up packs | none | FETCHED |
| **AudioShake** | limited credits | ~$19.99/mo¹ | hidden | hidden | "Contact Sales" | none | BLOCKED (404 on /pricing) |
| **Klangio** (per app) | 20-sec demo, unlimited uses | none | $24.99-$49.99/yr | **~$2-$4/mo** | $29.99/mo Studio Pro | none | PARTIAL |
| **Vocali.se** | unlimited, no signup | donation only | — | — | — | — | FETCHED |
| **Chordify** | ad-supported viewer | $6.99/mo | $41.88/yr | **$3.49/mo** | +Toolkit $4.50/mo | none | BLOCKED (403, support docs cited) |
| **Soundslice** | editor + 2 PDF/mo | $5/mo | $50/yr | **$4.17/mo** | Teacher $20/mo | none | FETCHED |
| **Songsterr** | tabs + ads + 10-bar pause | ~$9.95/mo¹ | ~$59.95/yr¹ | **$5/mo** | none | none | PARTIAL (price hidden) |
| **Ultimate Guitar Pro** | static tabs + ads | ~$24.99/mo¹ | ~$99.99/yr¹ | **$8.33/mo** | none | none | BLOCKED (UG hides prices) |
| **Hookpad** | limited free tier | $4.99/mo | $49.99/yr | **$4.17/mo** | +Aria AI $14.99/mo | **$149 lifetime¹** | BLOCKED (third-party) |
| **Anytune** | base IAP + à la carte | none | none | — | — | **$14.99 iOS / $34.99 Mac one-time** | FETCHED |
| **Yousician** | time-gated daily play | $19.99/mo | ~$120/yr | **$9.99/mo** | $29.99/mo Premium+ | none | PARTIAL |
| **Simply Piano** | trial lessons | ~$17.90-$24.86/mo | ~$169.90/yr | **$14.16/mo** | Family $23.90/mo | none | PARTIAL |
| **Splice** | none — paid only | $12.99 Sounds+ / $19.99 Creator | ~$200/yr Creator ($120 promo) | **$16.67/mo** | Creator+ $39.99/mo | none | FETCHED |
| **StemScriber (current)** | 3 songs/mo, full features | $10/mo | **(none)** | — | $20/mo | none | — |

¹ = secondary source (third-party 2026 review); pricing not directly fetchable from competitor's own page.

---

# 2. Annual-vs-Monthly Discount 🟡

The category is **bimodal**, not normalized:

| Discount tier | Examples | Rough discount |
|---------------|----------|----------------|
| **Conservative ("10 months for 12")** | Soundslice, Hookpad | ~16-17% |
| **Standard SaaS** | Yousician, Splice, Simply Piano | ~25-50% |
| **Aggressive consumer-app** | Audio Jam, Chordify, Songsterr, UG Pro | 50-67% (or "annual = ~3× cheaper than 12 monthlies") |
| **No discount** | LALAL Pro/Lite | 0% (annual = 12 × monthly exactly) |

**StemScriber's gap is the missing plan, not the price.** Without an annual option at all, comparison shoppers see $10/mo × 12 = $120/yr against Audio Jam's $39 — a 3:1 ratio. Add a $79-89 annual and the comparison flips: now it's a roughly competitive offer with one comparable competitor (Audio Jam) and clearly better than mid-tier (Soundslice $50, Chordify $42, Hookpad $50).

**Audio Jam's $39 is partly billing arbitrage:**
- $12.99/mo iOS IAP (which includes Apple's 30% cut) makes their headline monthly look high
- $6.99/mo web is the "real" monthly → annualizes to $84/yr
- $39 is 53% off their *web* monthly, not 74% off their *real* monthly
- Their annual price is anchored against an inflated IAP rate

This matters: **StemScriber doesn't need to drop to $39 to neutralize the comparison. It needs an annual plan that frames a credible discount against the $10 web monthly.**

---

# 3. Free-Tier Patterns 🟡

Five archetypes:

| Archetype | Examples | What's gated |
|-----------|----------|--------------|
| **Hard cap on quantity** | Audio Jam (3 projects), Klangio (20 sec/file), LALAL (10 min) | Volume — once you hit it, you stop |
| **Watermark / quality degrade** | (none in this set use this overtly) | — |
| **Time-gate** | Yousician (minutes/day), Simply Piano (intro lessons only) | Daily/total session length |
| **Feature lock** | Chordify (no MIDI/transpose), Soundslice (export gated), Hookpad (instrument library locked) | Specific features behind paywall |
| **No free tier** | Splice (sample subscription is paid-only) | Everything |
| **Fully free** | Vocali.se | Nothing — donation model |

**StemScriber's "3 songs/mo, full features" is more generous than category norm.** Most direct competitors (Audio Jam 3 projects, Klangio 20 seconds, LALAL 10 minutes) cap volume harder. StemScriber giving away unlimited features at low volume is a marketing differentiator and a margin risk: heavy free users churn through 3 songs/mo at $0.18 in Modal COGS, then leave. This is fine if conversion is healthy (>5%), painful otherwise.

**The category quietly abandoned generous free tiers in 2024-2025.** Splice killed its free tier entirely, Yousician/Simply gate everything behind 7-day trials, Chordify/Songsterr increasingly degrade free with ads. StemScriber's instinct to keep free generous reads as 2022-vintage product strategy, not 2026. Worth holding (small operator + word-of-mouth-driven launch can afford it), but with eyes open.

---

# 4. Price-Quality Curve 🟡

Rough plot of consumer-music-tool category, X-axis = annual price, Y-axis = quality of stem-separation + chord-detection (combined). Stem-only and chord-only tools graded on their available scope:

```
QUALITY ↑
   |                                            • LALAL Pro ($180)
HIGH                          • Soundslice ($50) — chords/notation only, deep
   |        • Audio Jam ($39)        • StemScriber ($120 monthly only) ← here now
MID|                                 • LALAL Lite ($90)
   |    • Moises ($36)    • Hookpad ($50)  • Songsterr ($60)
LOW|                          • Chordify ($42)
   |    • UG Pro ($100) — catalog scale, low transcription quality
   +———————————————————————————————————————————————————————————————→
        $30        $50        $80        $120        $180        PRICE/yr
```

**StemScriber's quality position (post-Apr-26 detector sprint):**
- Stem separation: 95%+ on real audio (BS-RoFormer-SW + vocal split) → matches LALAL Pro
- Chord detection: 8 of 10 audit songs at A-grade (post family-aware + maj7 promotion) → above Audio Jam, above Chordify, comparable to Soundslice on its narrower scope
- Practice mode: full feature set (mute/solo/speed/loop) → comparable to Songsterr Plus, Soundslice
- Guitar Pro export + bass MusicXML: differentiator — none of the competitors bundle all three artifacts

**StemScriber's actual quality justifies a price between Soundslice ($50) and LALAL Pro ($180).** The current $120-annualized-monthly slots reasonably; the absence of an annual plan is what looks bad.

---

# 5. Closest-Priced Peer 🟢

**Yousician at $9.99/mo annualized ($120/yr).** Identical effective monthly to StemScriber's annualized cost. What you get for $120/yr at Yousician: gamified instrument lessons for ONE instrument family (guitar OR piano OR bass), curated song library, real-time feedback via mic, structured progression curriculum.

**Does StemScriber under- or over-deliver vs. Yousician at $120?** Different category — Yousician is *learn an instrument* and StemScriber is *learn a song*. But on raw tool-value: a working musician who already plays gets meaningfully more from StemScriber (any song → stems + chords + practice tools) than from Yousician's structured curriculum. **StemScriber over-delivers vs. Yousician at the same price** — but only for the gigging-musician audience, not for total-beginners.

The closer apples-to-apples peer at StemScriber's price doesn't really exist. The category is bimodal: ~$50/yr utilities (Soundslice/Chordify/Hookpad/Audio Jam) and ~$120-200/yr learning platforms (Yousician/Simply/Splice). StemScriber sits in the gap between them.

---

# 6. Closest-Feature Peer 🟢

**Audio Jam.** They are the only competitor that bundles stem separation + chord detection + practice tools (slow/loop) + cross-platform delivery in one product. Feature surface is nearly identical to StemScriber. Their price: $39.99/yr.

**Second-closest:** Moises (stems + chords + AI vocal/voice features), at ~$36-48/yr depending on tier. Heavier on AI-music-generation, lighter on practice.

**Quality difference vs. Audio Jam:**
- Audio Jam has 2.6K App Store ratings @ 4.6 stars — real social proof
- Audio Jam ships native iOS/Android/Mac/Win — not just web
- StemScriber has materially better stem separation (BS-RoFormer-SW vs. their unknown but likely older Demucs-based engine), better chord detection (post-Apr-26 family-aware + maj7), and Guitar Pro export + bass MusicXML as bundled artifacts Audio Jam doesn't surface

**This means: StemScriber is ~20-30% better than Audio Jam on actual output quality, while charging ~3× more annualized. The price gap isn't justified at the current ratio.** Either StemScriber's annual price needs to come down (toward $79-89) OR the differentiation needs to be loud enough to support $120 (fold "Guitar Pro export + bass written down + 95% stem quality" into the launch pitch).

---

# 7. Stay at $10/mo + Add an Annual Plan 🟢

The recommended path. Math on three candidate annual prices:

| Annual price | Effective monthly | % off monthly | Comparison to category | Modal COGS coverage* |
|--------------|-------------------|---------------|------------------------|---------------------|
| **$79/yr** | $6.58/mo | 34% off | Closes Audio Jam gap to ~2:1 (AJ $39 vs SS $79). Above Soundslice/Chordify/Hookpad ($42-50). | Covers ~110 songs/yr per subscriber |
| **$89/yr** | $7.42/mo | 26% off | Closes gap to ~2.3:1. Reads "premium but reasonable." Anchors $10/mo as convenience tax. | Covers ~123 songs/yr |
| **$99/yr** | $8.25/mo | 17.5% off | Soundslice-pattern ("10 months for 12"). Most conservative discount. | Covers ~137 songs/yr |

*At Modal $0.06/song COGS. A heavy user doing 50 songs/mo = 600 songs/yr = $36 COGS, leaving $43-63 gross per subscriber across the three options.

**Recommended: $89/yr.** Reasoning:
- **Closes the Audio Jam comparison enough.** $89 vs $39 is 2.3× — defensible with quality differentiation. $79 vs $39 is 2× — stronger but $10 less margin per annual buyer.
- **Above Soundslice/Chordify/Hookpad ($42-50) by ~80%** — signals "more than a chord tool" without entering the Yousician/Splice tier ($120-200) where curriculum + content licenses justify price.
- **Round-friendly framing:** "$89/year — saves $31, that's 3+ months free." The "3 months free" hook is more compelling than "26% off" in landing copy.
- **Aligns with the "$15.99 elsewhere vs $10 here" existing message:** an $89 annual = ~$7.42/mo, well under the implied competitor-stack price.

---

# 8. Drop the Monthly Price 🔴

**Don't.** Going to $7/mo or $5/mo to chase Audio Jam is a 🔴 RED move:

- **The race-to-bottom dynamics.** Audio Jam at $39/yr can sustain that price because they have 2.6K App Store ratings and presumably scale-driven economics. StemScriber pre-launch has neither. Joining their price floor without their unit economics is margin suicide.
- **Modal COGS isn't zero.** $0.06/song × heavy-user volumes makes $5/mo subscribers borderline negative-margin once support time is factored. At $7/mo it's marginal.
- **Premium positioning is sticky.** Once you drop, you can't easily raise without alienating early adopters. The current "$10/mo" already anchors below Yousician/Splice; going to $5/mo recasts StemScriber as a Soundslice/Chordify-tier utility, which is *under*-positioning given the actual quality.
- **The Audio Jam gap is solved by the annual plan, not by dropping monthly.** See #7.

The only scenario where dropping monthly makes sense: if conversion data after launch shows <2% free→paid and the bottleneck is sticker shock at $10. Then drop to $7. But don't pre-emptively cut before launch when the diagnosis is hypothetical.

---

# 9. Raise the Price (Premium Positioning) 🟡

**$15/mo or $20/mo Pro is defensible but only with stronger differentiation messaging.** Who's at this tier:
- Yousician $19.99/mo — but bundles content licenses + curriculum
- Simply Piano $17.90-$24.86/mo — same
- Splice Creator $19.99/mo — but bundles 200 sample downloads/mo
- AudioShake ~$19.99/mo — but B2B-positioned

**StemScriber doesn't bundle content or curriculum**, so $15-20/mo would need to be carried by tool-quality alone. That's harder. The quality is genuinely there (better stems than Audio Jam, real chord detection, Guitar Pro export), but the marketing burden to make $20/mo feel justified — vs. just opening Audio Jam — is heavy.

**$15/mo would be defensible if** the soft-launch testimonials from Refinery musicians ("nailed our bridge," "saved our setlist") are strong enough to anchor it. Worth A/B testing post-launch, not pre-launch.

**$10/mo is the right launch price.** Raising it post-launch with usage data is easier than launching high and explaining the cut.

---

# 10. Tier Restructuring 🟡

Three patterns worth considering:

### Option A: Keep 3 tiers (Free, Pro, Premium), add annuals to both
- Free: 3 songs/mo
- Pro: $10/mo or $89/yr — full features, generous song cap (e.g., 50/mo)
- Premium: $20/mo or $179/yr — unlimited songs, batch processing, priority queue, API access
- Pros: tier ladder; pulls power users to higher LTV
- Cons: $20/mo is hard to justify pre-launch when usage data doesn't yet support specific Premium features

### Option B: Collapse to 2 tiers (Free, Pro)
- Free: 3 songs/mo
- Pro: $10/mo or $89/yr — unlimited songs, full features
- Pros: simpler messaging; no "what does Premium add?" confusion
- Cons: no upsell path for power users; leaves money on the table

### Option C: 2 tiers + Lifetime LTD
- Free: 3 songs/mo
- Pro: $10/mo or $89/yr
- **Lifetime: $199 LTD, capped at 100 launch buyers, 50-songs/mo cap built in**
- Pros: cash up front from soft-launch enthusiasts; neutralizes Audio Jam's annual as a comparison ("you can pay them $39/yr forever, or pay us $199 once"); creates scarcity narrative for Refinery cohort
- Cons: forfeits MRR from those 100; precedent for asking again later; needs careful wording to avoid being viewed as "you'll go out of business" signal

### Recommended: Option C
- Pre-launch: launch with Free + Pro ($10/mo or $89/yr) + Lifetime $199 LTD capped at 100
- 90 days post-launch: when real usage data is in, evaluate adding a Studio tier ($25/mo or $199/yr) for the top 5-10% of users who want batch processing + API access. Don't ship Premium pre-launch — ship Studio post-launch when you know what to put in it.

The current $20/mo Premium tier is a "founder vanity" tier — it exists because Stripe makes it easy to add, not because there's product-shaped demand for it. Drop it, build conviction with usage data, reintroduce intentionally.

---

# 11. Single Best Pricing Recommendation 🟢

```
┌────────────────────────────────────────────────────────────┐
│ FREE                                                       │
│   3 songs / month                                          │
│   Full features (no watermark, no quality gate)            │
│   Conversion lever: hard song cap                          │
├────────────────────────────────────────────────────────────┤
│ PRO                                                        │
│   $10 / month  •  OR  •  $89 / year (3+ months free)       │
│   50 songs / month                                         │
│   All features: stems + chord chart + practice mode        │
│   + Guitar Pro export + bass MusicXML                      │
├────────────────────────────────────────────────────────────┤
│ LIFETIME (launch promo, 100 codes only)                    │
│   $199 one-time                                            │
│   50 songs / month forever                                 │
│   "REFINERY-XXXX" codes for soft-launch cohort             │
│   Wave goes live June 20, expires when 100 sold or         │
│   September 1, whichever first                             │
└────────────────────────────────────────────────────────────┘
```

**Drop the current $20/mo Premium tier.** Reintroduce as "Studio" 60-90 days post-launch with batch processing + API access + priority queue, priced $25/mo or $199/yr, once usage data tells you what power users actually want.

**Why this structure works:**
- Pro $89/yr is **2.3× Audio Jam** — gap is defensible given StemScriber's better stem quality + chord detection + Guitar Pro export
- Pro $89/yr is **~80% above Soundslice/Chordify/Hookpad** — signals broader scope (stems + chords + practice, not just one of the three)
- Pro $89/yr is **27% below Yousician's $120** — reads "premium tool, not curriculum platform"
- Lifetime $199 is in the same neighborhood as Hookpad's $149 lifetime — proven willingness-to-pay precedent
- Lifetime cap (100 buyers, 50 songs/mo) protects margin: worst case = 100 × 50 songs/mo × $0.06 × 36 months = $10,800 in COGS over 3 years vs. $19,900 collected up front. Positive even in heavy-use 3-year scenario.

---

# 12. Launch-Day Pricing Message 🟡

**Current message:** "Stop Paying for Two Tools — $15.99 for three tools vs $10 for StemScriber."

**Recommended replacement:**

> **One upload. Stems + chord chart + practice tools.**
> **$10/mo or $89/year — half what the three-tool stack costs.**

Why this is sharper:
- **Leads with the product** ("one upload"), not the price comparison — the product framing is the differentiator, the price is the proof
- **Names what's bundled** ("stems + chord chart + practice tools") so the comparison is concrete
- **Uses "half" instead of a precise dollar comparison** — easier to say, harder to argue against, doesn't rot when competitor prices shift
- **Includes the annual price** — current message hides it because it doesn't exist yet

Even sharper alternative for the Refinery cohort specifically:
> **Six stems. One chord chart. Slow it down. Loop it. Get it right.**
> **$89/year. Or $199 once, for 100 musicians at this show only.**

The Refinery framing creates urgency without manufactured scarcity — there are literally 100 codes, the show literally has musicians in the audience.

---

# 13. Single Biggest Pricing Risk 🔴

**Dropping the monthly price below $10 to chase Audio Jam.**

Why this is the highest-stakes wrong move:
- **Sticky precedent.** Once $7 or $5/mo is the launch price, raising it post-launch alienates everyone who bought at the lower price. You'd need to grandfather them, then explain why new users pay more. This is operationally and reputationally expensive.
- **Margin compression at scale.** Current Modal COGS at $0.06/song is fine at $10/mo (~167 songs/mo of break-even). At $5/mo it's 83 songs/mo break-even, and the heavy-user tail (50+ songs/mo regulars) goes negative once support time is added.
- **Position collapse.** $10/mo aligns with Yousician $9.99 effective monthly — premium-but-accessible. $5/mo aligns with Soundslice $4.17 — utility tier. The actual product is closer to the former; positioning it as the latter under-sells the quality.
- **Audio Jam isn't actually winning at $39.** They have 2.6K App Store ratings after multiple years; that's modest. Their price is anchoring, not converting at the rate the price implies. Don't optimize against an anchor that isn't converting.

The second-biggest risk is **shipping Premium at $20/mo without product-shaped demand**. That tier signals greed without justification. Drop it; reintroduce post-launch as Studio with real features.

---

# Final Recommendation

**Lock these prices for soft launch (June 20):**
- Free: 3 songs/mo, full features
- Pro: **$10/mo or $89/yr**
- Lifetime LTD: **$199 one-time, capped at 100 buyers, expires Sep 1 or sellout**
- Drop the current $20/mo Premium tier; reintroduce as Studio post-launch

**Math summary:**
- Annual saves $31/yr vs monthly = 26% off = "3+ months free" framing
- $89/yr vs Audio Jam $39 = 2.3× ratio, defensible on quality
- $89/yr vs Soundslice/Chordify/Hookpad $42-50 = 1.8-2.1× ratio, defensible on scope (3 tools vs 1)
- $89/yr vs Yousician $120 = 0.74× ratio, reads as "premium tool, not curriculum"
- Lifetime $199 covers Modal COGS for any plausible use pattern over 3+ years; positive even at 50-song/mo heavy use
- 100 lifetime sales = $19,900 upfront cash to fund post-launch ad spend

**Severity summary:**
- 🟢 GREEN (settled): Pricing matrix, closest peers, recommended structure, current monthly price ($10), launch pitch
- 🟡 YELLOW (defensible either way): Annual discount %, free-tier generosity, raising price post-launch, tier restructuring details
- 🔴 RED (high-stakes): Dropping monthly below $10 (don't); shipping unjustified $20/mo Premium (drop it)

---

# The One Experiment Before Locking In

**A/B test the annual price during the 4-week pre-launch teaser arc** (May 20 – June 20). Two pricing landing pages, identical except for the annual number:

- **Variant A:** Pro $10/mo or **$79/yr**
- **Variant B:** Pro $10/mo or **$89/yr**

Split traffic 50/50 from the @stemscriber social posts. Measure:
1. Click-through rate from "View pricing" → "Select annual"
2. Email-capture rate from each variant's checkout-attempt flow
3. Direct feedback from the Refinery cohort during pre-show conversations ("does $89 feel right?")

Even if total volume is low (say, 200 visitors / 20 email captures), the relative differential is informative. If A shows materially higher annual-select rate, lock at $79. If they're tied, lock at $89 to capture the extra $10/sub margin.

**The variant NOT to test:** $69/yr or below. Anything under $79 erodes the "premium tool" position you're trying to hold. Test within a defensible band; don't test the floor.

---

# Sources

Direct stem/chord competitors:
- [Audio Jam homepage + App Store IAP](https://audiojam.app/)
- [LALAL.AI Pricing](https://www.lalal.ai/pricing/) (FETCHED)
- [Soundslice Plans](https://www.soundslice.com/plans/) (FETCHED)
- [Vocali.se](https://vocali.se) (FETCHED)
- [Klangio Piano2Notes](https://klang.io/piano2notes/) + sale pages
- [Moises pricing (login-gated)](https://moises.ai/pricing) — secondary
- [AudioShake homepage](https://www.audioshake.ai/) — pricing 404
- [Chordify support docs](https://support.chordify.net/) — secondary

Practice/learning + adjacent:
- [Songsterr Plus](https://www.songsterr.com/a/wa/plus) (FETCHED, $ hidden)
- [UG Pro Help Center](https://help.ultimate-guitar.com/) — secondary
- [Hookpad pricing](https://www.hooktheory.com/hookpad/pricing) — third-party
- [Anytune Pro App Store](https://apps.apple.com/us/app/anytune-pro-music-practice/id415365180) (FETCHED)
- [Yousician account/plans](https://account.yousician.com/plans) — gated
- [Simply Piano](https://www.hellosimply.com) — gated
- [Splice Plans](https://splice.com/plans) (FETCHED)

All prices verified or sourced 2026-05-02. Re-verify before checkout if the launch slips past July 2026.
