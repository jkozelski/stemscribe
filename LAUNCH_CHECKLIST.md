# StemScribe Launch Checklist

## Phase 1: Legal Foundation (Week 1)
- [ ] Form LLC in your state (or Delaware)
  - [ ] File articles of organization
  - [ ] Get EIN from IRS (free, online at irs.gov)
  - [ ] Open business bank account
- [ ] Register DMCA agent at copyright.gov ($6)
- [ ] Finalize Terms of Service (have attorney review - see `legal/TERMS_OF_SERVICE.md`)
- [ ] Finalize Privacy Policy (see `legal/PRIVACY_POLICY.md`)
- [ ] Finalize DMCA Policy (see `legal/DMCA_POLICY.md`)
- [ ] Verify third-party licenses (see `legal/THIRD_PARTY_LICENSES.md`)
- [ ] Register domain: stemscribe.app (or .com)

## Phase 2: Product Ready (Week 2-3)
- [ ] Fix Demucs async processing (see MANUS_AUDIT_BRIEFING.md)
- [ ] Add user authentication (JWT or session-based)
- [ ] Add rate limiting by tier (free: 3 songs/mo, premium: unlimited)
- [ ] Integrate Stripe billing
  - [ ] Create products (Premium $4.99/mo, Pro $14.99/mo)
  - [ ] Create annual plans (Premium $39.99/yr, Pro $119.99/yr)
  - [ ] Set up webhook handler
- [ ] Deploy frontend to Cloudflare Pages
- [ ] Deploy API to Railway
- [ ] Set up GPU processing (Replicate API or Modal)
- [ ] Set up file storage (Cloudflare R2)
- [ ] Set up database (Supabase)
- [ ] Add legal pages to website (ToS, Privacy, DMCA)
- [ ] Set up error monitoring (Sentry free tier)
- [ ] Set up uptime monitoring (UptimeRobot free tier)

## Phase 3: AI Agent Teams (Week 3-4)
- [ ] Set up CrewAI agents (see `marketing/crewai/`)
  - [ ] Get Anthropic API key
  - [ ] Get Serper API key (free 2,500 searches)
  - [ ] Run `./run_agents.sh setup`
  - [ ] Run research team: `./run_agents.sh research`
  - [ ] Review competitor analysis report
  - [ ] Review pricing strategy recommendations
- [ ] Run marketing team: `./run_agents.sh marketing`
  - [ ] Review and edit blog posts
  - [ ] Review social media calendar
  - [ ] Review email sequences
  - [ ] Set up Brevo for email (free 300 emails/day)
  - [ ] Set up Buffer for social scheduling ($5/mo)

## Phase 4: Pre-Launch (Week 4-5)
- [ ] Create landing page with email capture
- [ ] Record 5 demo videos
  - [ ] "Separate vocals from any song in 60 seconds"
  - [ ] "AI chord recognition for jazz standards"
  - [ ] "Learn Grateful Dead songs - isolate Jerry/Bob guitars"
  - [ ] "Free stem separation vs Moises vs LALAL.AI"
  - [ ] "How worship teams use StemScribe for chord charts"
- [ ] Set up YouTube channel
- [ ] Set up TikTok account
- [ ] Write 3 launch blog posts
- [ ] Prepare Product Hunt launch assets
  - [ ] Logo, screenshots, tagline
  - [ ] Launch day GIF/video
  - [ ] Find a hunter (or self-hunt)

## Phase 5: Launch (Week 5-6)
- [ ] Post to Product Hunt
- [ ] Post to Hacker News (Show HN)
- [ ] Post to relevant subreddits (provide value, not spam)
  - [ ] r/musictheory
  - [ ] r/WeAreTheMusicMakers
  - [ ] r/Guitar
  - [ ] r/jazz
- [ ] Send launch email to waitlist
- [ ] Start social media posting (follow the calendar)
- [ ] Reach out to music bloggers/YouTubers for reviews
- [ ] Offer free Pro trials to music educators

## Phase 6: Iterate (Ongoing)
- [ ] Monitor analytics (signups, conversions, churn)
- [ ] Run AI research team weekly for competitor updates
- [ ] Run AI marketing team weekly for content generation
- [ ] Collect user feedback
- [ ] Ship multi-genre chord models (Bluegrass, Rock, Country)
- [ ] A/B test pricing
- [ ] Build API for B2B (follow LALAL.AI's model)

---

## Monthly AI Agent Schedule

| Week | Research Team | Marketing Team |
|------|--------------|----------------|
| 1st Mon | Competitor analysis | Blog post + social calendar |
| 2nd Mon | User pain points scan | Email campaigns |
| 3rd Mon | Pricing review | SEO audit + content |
| 4th Mon | Market trends | Community outreach |

## Key Metrics to Track

| Metric | Target (Month 1) | Target (Month 6) |
|--------|-------------------|-------------------|
| Website visitors | 1,000 | 10,000 |
| Free signups | 100 | 2,000 |
| Paid conversions | 10 | 200 |
| MRR (Monthly Revenue) | $100 | $2,000 |
| Churn rate | < 10% | < 5% |
| Songs processed | 500 | 20,000 |

## Budget Summary (Monthly)

| Item | Cost |
|------|------|
| Infrastructure (hosting, GPU, storage) | $30-150 |
| AI Agent tools (Claude API, Serper) | $25-50 |
| Marketing tools (Buffer, Brevo) | $5-30 |
| Domain + misc | $5 |
| **Total operating cost** | **$65-235/mo** |
| **Break-even at** | **~15-50 paid users** |

---

## File Structure Reference

```
~/stemscribe/
├── backend/                    # Flask API (existing)
├── frontend/                   # UI (existing)
├── legal/                      # NEW: Legal documents
│   ├── TERMS_OF_SERVICE.md
│   ├── PRIVACY_POLICY.md
│   ├── DMCA_POLICY.md
│   └── THIRD_PARTY_LICENSES.md
├── marketing/                  # NEW: AI marketing system
│   ├── requirements.txt
│   └── crewai/
│       ├── agents.py           # Research + Marketing agent teams
│       ├── config.yaml         # Team configuration
│       ├── run_agents.sh       # Runner script
│       └── .env.example        # API key template
├── deploy/                     # NEW: Deployment guides
│   └── DEPLOYMENT_GUIDE.md
├── LAUNCH_CHECKLIST.md         # NEW: This file
└── ...existing files...
```
