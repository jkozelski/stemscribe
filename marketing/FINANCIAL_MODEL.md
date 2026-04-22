# StemScribe Financial Model
**Last Updated:** 2026-03-03

---

## 1. Unit Economics — Cost Per Song Processed

### Variable Costs Per Song

| Cost Component | Free Tier | Premium Tier | Pro Tier | Notes |
|---|---|---|---|---|
| **GPU Processing (Replicate/Demucs)** | $0.017 | $0.017 | $0.017 | ~78s run on Nvidia T4, $0.017/run |
| **R2 Storage (stems + MIDI + GP)** | $0.0003 | $0.0005 | $0.0008 | ~20MB free (MP3 128k), ~35MB premium, ~55MB pro (WAV lossless) |
| **R2 Class A Ops** | $0.000018 | $0.000027 | $0.000036 | ~4-8 write ops per song at $4.50/M |
| **Stripe Fees** | $0.00 | $0.00 | $0.00 | Per-song: $0 (Stripe charged on subscription, not per song) |
| **Total Per Song** | **$0.0173** | **$0.0175** | **$0.0178** |

### Stripe Fees Per Subscription Payment

| Tier | Monthly Price | Stripe Fee (2.9% + $0.30) | Net Revenue |
|---|---|---|---|
| Free | $0.00 | $0.00 | $0.00 |
| Premium Monthly | $4.99 | $0.44 | $4.55 |
| Premium Annual | $39.99 | $1.46 | $38.53 |
| Pro Monthly | $14.99 | $0.73 | $14.26 |
| Pro Annual | $119.99 | $3.78 | $116.21 |

### Cost Per User Per Month (at expected usage)

| Tier | Songs/Mo (Expected) | GPU Cost | Storage | Total Variable | Revenue | Margin |
|---|---|---|---|---|---|---|
| Free | 1.5 avg | $0.026 | $0.0005 | **$0.027** | $0.00 | **-$0.027** |
| Premium | 15 avg | $0.255 | $0.008 | **$0.263** | $4.55 | **$4.29 (94%)** |
| Pro | 40 avg | $0.680 | $0.032 | **$0.712** | $14.26 | **$13.55 (95%)** |

**Key Insight:** Variable costs are negligible. Even a heavy Pro user processing 100 songs/month costs us only $1.78 in GPU — the margins are excellent.

---

## 2. Monthly Fixed Costs

| Item | Monthly Cost | Notes |
|---|---|---|
| **Railway Hosting (Hobby)** | $5.00 | Flask API + Gunicorn, included credits cover light traffic |
| **Railway Hosting (Pro, scaling)** | $20.00 | Needed at ~500+ active users for higher RAM/CPU limits |
| **Supabase Postgres** | $0.00 | Free tier: 500MB, 50K rows — sufficient to ~5,000 users |
| **Supabase Postgres (Pro)** | $25.00 | Needed at ~5,000+ users or if free tier limits hit |
| **Cloudflare Pages** | $0.00 | Free tier, more than sufficient |
| **Cloudflare R2 Storage** | $0.00 | 10GB free. ~$1.50/mo per 100GB after |
| **Domain** | $1.00 | ~$12/yr |
| **Sentry Monitoring** | $0.00 | Free tier sufficient for early stage |
| **Email (transactional)** | $0.00 | Resend free tier: 100 emails/day |
| **Stripe** | $0.00 | No monthly fee, only per-transaction |
| **Total (Launch)** | **$6.00** | Minimal viable infrastructure |
| **Total (Growth, 500+ users)** | **$46.00** | Railway Pro + Supabase Pro |
| **Total (Scale, 5000+ users)** | **$75.00+** | + R2 storage growth |

---

## 3. Break-Even Analysis

### At Launch (Fixed Costs: $6/month)

| Scenario | Users Needed | Calculation |
|---|---|---|
| All Premium Monthly | **2 paid users** | 2 x $4.55 net = $9.10 > $6.00 |
| All Pro Monthly | **1 paid user** | 1 x $14.26 net = $14.26 > $6.00 |
| Realistic Mix (70% Prem / 30% Pro) | **2 paid users** | 1.4 x $4.55 + 0.6 x $14.26 = $14.93 |

### At Growth Stage (Fixed Costs: $46/month)

| Scenario | Users Needed | Calculation |
|---|---|---|
| All Premium Monthly | **11 paid users** | 11 x $4.29 margin = $47.19 |
| All Pro Monthly | **4 paid users** | 4 x $13.55 margin = $54.20 |
| Realistic Mix (70/30) | **6 paid users** | Revenue covers fixed + variable |

### At Scale (Fixed Costs: $75/month)

| Scenario | Users Needed | Calculation |
|---|---|---|
| Realistic Mix (70/30) | **10 paid users** | Comfortably covers all costs |

**Verdict:** StemScribe has an extremely low break-even point. The infrastructure costs are minimal, and even a small number of paying customers makes this profitable. The real cost is customer acquisition, not infrastructure.

---

## 4. Revenue Projections

### Assumptions
- Free-to-paid conversion: 4% (industry avg for freemium SaaS tools)
- Tier split of paid: 70% Premium, 30% Pro
- Monthly churn: 8% (early-stage consumer SaaS benchmark)
- Annual subscribers: 25% of paid (lower churn, better LTV)
- Growth modeled as new free signups per month

### Conservative (Organic Only — Reddit, SEO, Word of Mouth)

| Month | New Free Users | Total Free | Paid Users | MRR | Cumulative Revenue |
|---|---|---|---|---|---|
| 1 | 100 | 100 | 4 | $29 | $29 |
| 3 | 150 | 380 | 14 | $101 | $218 |
| 6 | 200 | 780 | 28 | $203 | $723 |
| 12 | 300 | 1,800 | 58 | $420 | $2,784 |

### Moderate (Organic + YouTube Tutorials + Community)

| Month | New Free Users | Total Free | Paid Users | MRR | Cumulative Revenue |
|---|---|---|---|---|---|
| 1 | 300 | 300 | 12 | $87 | $87 |
| 3 | 500 | 1,200 | 44 | $319 | $688 |
| 6 | 800 | 3,200 | 105 | $762 | $3,126 |
| 12 | 1,200 | 8,500 | 250 | $1,814 | $14,016 |

### Optimistic (Viral Moment / Feature in Major Music Pub)

| Month | New Free Users | Total Free | Paid Users | MRR | Cumulative Revenue |
|---|---|---|---|---|---|
| 1 | 1,000 | 1,000 | 40 | $290 | $290 |
| 3 | 2,000 | 5,000 | 180 | $1,306 | $2,802 |
| 6 | 3,000 | 14,000 | 460 | $3,338 | $14,448 |
| 12 | 5,000 | 40,000 | 1,200 | $8,706 | $68,736 |

**MRR Calculation:** (Paid * 0.70 * $4.55) + (Paid * 0.30 * $14.26) = blended $7.26 ARPU per paid user

---

## 5. Competitive Pricing Analysis

### Competitor Landscape (March 2026)

| Competitor | Free Tier | Entry Price | Pro Price | Model | Primary Feature |
|---|---|---|---|---|---|
| **StemScribe** | 3 songs/mo | $4.99/mo | $14.99/mo | Subscription | Full Guitar Pro transcription |
| **Moises.ai** | 2 songs/mo | $3.99/mo | $9.99/mo | Subscription | Stem separation + practice tools |
| **LALAL.AI** | Limited | $7.50/mo or $20 pack | $15/mo or $70 pack | Hybrid | Stem separation only |
| **AnthemScore** | 30s trial | $19.97 one-time | $39.99 one-time | One-time | Desktop transcription |
| **Splitter.ai** | Basic free | ~$5-10/mo (est.) | — | Freemium | Browser-based splitting |
| **Guitar2Tabs** | Free basic | — | — | Free | Basic tab generation |

### Pricing Validation

**Our Premium ($4.99/mo) vs. Market:**
- Slightly above Moises.ai ($3.99) but we offer MIDI export + chord detection, which Moises doesn't at that tier
- Below LALAL.AI's subscription ($7.50/mo) while offering more features
- **Verdict: Well-positioned.** $4.99 is the sweet spot — affordable enough for hobbyists, signals quality

**Our Pro ($14.99/mo) vs. Market:**
- Same as LALAL.AI Pro ($15/mo) but we include Guitar Pro export (unique differentiator)
- Above Moises Pro ($9.99) but our tab output is a distinct value prop they don't offer
- Below professional-grade tools (Melodyne, etc.)
- **Verdict: Competitive.** Guitar Pro export justifies the premium

**Our Free Tier (3 songs/mo) vs. Market:**
- More generous than Moises (2 songs/mo)
- Less generous than Splitter.ai (unlimited basic)
- **Verdict: Good.** 3 songs lets users genuinely try the product before converting

### Recommendation: Keep Current Pricing

The current tiers are well-calibrated. No changes recommended at launch. Key reasons:
1. $4.99 Premium undercuts LALAL.AI and is only $1 more than Moises — easy upsell
2. $14.99 Pro is justified by the unique Guitar Pro export feature (no direct competitor offers this)
3. Annual discounts (33% off) are standard and effective for reducing churn

---

## 6. Should We Add a One-Time Purchase Option?

### Case FOR (LALAL.AI Model)

- LALAL.AI's pay-per-minute packs ($20 for 90 min) are popular with casual users
- Removes subscription fatigue for infrequent users
- Could capture users who only need 5-10 songs transcribed total
- No recurring billing overhead

### Case AGAINST

- Recurring revenue is far more valuable for the business (higher LTV, predictable MRR)
- Adds billing complexity
- Most competitors are moving toward subscription models
- Our per-song GPU cost ($0.017) means a one-time pack would need careful pricing to be profitable

### Recommendation: Consider Adding "Credit Packs" Post-Launch

A la carte option to test:
- **Starter Pack:** $9.99 for 10 songs (no expiry)
- **Power Pack:** $24.99 for 30 songs (no expiry)
- Includes Premium-tier features (MIDI, chord detection)
- Guitar Pro export as a $2.99 per-song add-on or included at Power Pack level

**Rationale:** Low-risk addition that captures a market segment (casual users, one-time projects) that subscription-only misses. But deprioritize until you have data on user behavior post-launch.

---

## 7. Customer Acquisition Cost (CAC) Estimates

### By Channel

| Channel | Estimated CAC | Time to Impact | Notes |
|---|---|---|---|
| **Organic/SEO** | $2-5 per signup | 3-6 months | Blog posts, landing page optimization. Long-term lowest CAC |
| **Reddit (r/Guitar, r/musicproduction)** | $1-3 per signup | Immediate | Share demos, answer questions. CPCs 50-70% below Facebook. Highest ROI at launch |
| **YouTube Tutorials** | $3-8 per signup | 1-3 months | "How to transcribe any song" style content. Compounds over time |
| **Product Hunt Launch** | $0.50-2 per signup | One-time spike | Free but requires preparation. Good for initial user burst |
| **Music Forum Presence** | $1-4 per signup | 1-2 months | Ultimate Guitar, TalkBass, Drummerworld. Authentic participation |
| **Paid Ads (Google)** | $15-30 per signup | Immediate | Expensive for broad terms. Only viable for high-intent keywords |
| **Paid Ads (Reddit/Instagram)** | $8-15 per signup | Immediate | Better than Google for niche targeting |
| **Referral Program** | $1-3 per signup | 2-3 months | "Give a friend 3 free songs, get 5" — lowest CAC long-term |

### Blended CAC Target

- **Launch phase (Mo 1-3):** $3-5 per free signup, $75-125 per paid conversion (at 4% conversion)
- **Growth phase (Mo 4-12):** $2-4 per free signup, $50-100 per paid conversion
- **Mature (12+ months):** $1-3 per free signup, $25-75 per paid conversion

**Industry benchmark:** SaaS companies average $1,200 CAC across all channels, but that's B2B enterprise. Consumer music tools are much lower — $50-150 per paying customer is realistic.

---

## 8. Lifetime Value (LTV) by Tier

### Assumptions
- Monthly churn: 8% (industry standard for early-stage consumer SaaS)
- Average customer lifetime = 1 / churn rate = 12.5 months
- Annual subscribers have ~50% lower churn (4% monthly) = 25-month lifetime

| Tier | Monthly Revenue (net) | Avg Lifetime | LTV | LTV (Annual Sub) |
|---|---|---|---|---|
| **Premium Monthly** | $4.55 | 12.5 months | **$56.88** | — |
| **Premium Annual** | $3.21/mo effective | 25 months | **$80.25** | **$80.25** |
| **Pro Monthly** | $14.26 | 12.5 months | **$178.25** | — |
| **Pro Annual** | $9.68/mo effective | 25 months | **$242.00** | **$242.00** |

### Blended LTV (70% Premium / 30% Pro, 75% Monthly / 25% Annual)

**Blended LTV = $93.20 per paid customer**

### LTV:CAC Ratio

| Scenario | CAC | LTV | LTV:CAC | Verdict |
|---|---|---|---|---|
| Organic-heavy launch | $100 | $93 | 0.93:1 | Borderline — needs churn reduction or ARPU lift |
| Reddit + community focus | $75 | $93 | 1.24:1 | Acceptable for early stage |
| Mature (6+ months) | $50 | $93 | 1.86:1 | Healthy |
| With annual push | $50 | $120+ | 2.4:1 | Target — push annual subscriptions |

**Industry Target:** LTV:CAC > 3:1 is ideal. At launch, 1.2-2:1 is acceptable. Key levers:
1. **Reduce churn** — improve onboarding, add sticky features (practice mode, song library)
2. **Push annual subscriptions** — 33% discount but 2x lifetime
3. **ARPU expansion** — consider a higher Studio tier ($24.99/mo) once product-market fit is proven

---

## 9. Cash Flow Timeline — When Does This Become Self-Sustaining?

### Monthly Burn Rate (Pre-Revenue)

| Phase | Monthly Fixed | Variable (GPU/Storage) | Total Burn |
|---|---|---|---|
| **Pre-Launch** | $6 | $0 | **$6/mo** |
| **Launch (Mo 1-3)** | $6 | $5-15 | **$11-21/mo** |
| **Growth (Mo 4-6)** | $46 | $15-50 | **$61-96/mo** |
| **Scale (Mo 7-12)** | $75 | $50-150 | **$125-225/mo** |

### Conservative Cash Flow (No External Funding)

| Month | MRR | Monthly Costs | Net Cash Flow | Cumulative |
|---|---|---|---|---|
| 0 (Pre-launch) | $0 | $6 | -$6 | -$6 |
| 1 | $29 | $15 | +$14 | +$8 |
| 3 | $101 | $25 | +$76 | +$158 |
| 6 | $203 | $70 | +$133 | +$578 |
| 12 | $420 | $150 | +$270 | +$2,028 |

### Moderate Cash Flow

| Month | MRR | Monthly Costs | Net Cash Flow | Cumulative |
|---|---|---|---|---|
| 0 | $0 | $6 | -$6 | -$6 |
| 1 | $87 | $20 | +$67 | +$61 |
| 3 | $319 | $55 | +$264 | +$544 |
| 6 | $762 | $100 | +$662 | +$2,648 |
| 12 | $1,814 | $200 | +$1,614 | +$12,000+ |

### When Self-Sustaining?

**Conservative:** Cash-flow positive from Month 1 (fixed costs are so low that even 2 paying customers covers them). Self-sustaining at $6/month is nearly immediate.

**At growth infrastructure ($46/mo fixed):** Need ~6 paid users = achievable by Month 2-3.

**At scale infrastructure ($75/mo fixed):** Need ~10 paid users = achievable by Month 3-4.

**This is the most attractive aspect of StemScribe's financial model:** The extremely low fixed costs mean there is virtually no "cash runway" risk. You are not burning $5K-50K/month like typical SaaS startups. The question is not "can we survive?" but "how fast can we grow?"

---

## 10. Summary and Recommendations

### Key Numbers at a Glance

| Metric | Value |
|---|---|
| Cost per song processed | $0.017 |
| Monthly fixed costs (launch) | $6 |
| Break-even (paid users) | 2 |
| Blended ARPU (paid) | $7.26/mo |
| Blended LTV | $93.20 |
| Target CAC | <$75 |
| Gross margin | 94-95% |

### Strategic Recommendations

1. **Keep current pricing as-is for launch.** $4.99/$14.99 is well-positioned competitively.
2. **Push annual subscriptions hard** — they double LTV and halve churn. Offer first-month trial + 33% annual discount prominently.
3. **Consider credit packs post-launch** ($9.99/10 songs, $24.99/30 songs) to capture one-time users.
4. **Invest in Reddit and community marketing first** — lowest CAC channel for music tools.
5. **Focus on churn reduction over acquisition** — getting churn from 8% to 5% increases LTV by 60%.
6. **Monitor GPU costs closely** — if Replicate pricing changes or volume spikes, self-hosting Demucs on a $50-80/mo GPU server becomes worthwhile at ~3,000+ songs/month.
7. **Do NOT raise prices until you have 100+ paying users** — price discovery needs real data.
8. **Consider a Studio tier ($24.99/mo)** after product-market fit is proven — batch processing, API access, commercial license.

### Risk Factors

| Risk | Likelihood | Mitigation |
|---|---|---|
| Replicate price increase | Low | Self-host Demucs at scale (break-even ~3K songs/mo) |
| Moises.ai adds Guitar Pro export | Medium | First-mover advantage, deeper tab features, practice mode |
| Free tools improve (e.g., open-source) | Medium | UX polish, cloud convenience, speed |
| Low conversion rate (<2%) | Medium | A/B test onboarding, improve free tier value demo |
| High churn (>10%) | Medium | Add sticky features (song library, practice tools, learning paths) |

---

*Model built March 2026. Revisit monthly as real usage data becomes available.*
