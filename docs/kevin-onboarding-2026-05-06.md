# Kevin Hanley — Onboarding Pack

**Created:** 2026-05-06 · **Role:** Creative Director, brand & marketing
**Comp framing:** barter partnership — see `marketing-prep-kevin-2026-05-05.md` for full terms

---

## 1. Social media channels — current state

All channels are owned under the handle `@stemscriber`. Confirmed Apr 26 2026.

| Platform | Handle | URL | State |
|---|---|---|---|
| **Instagram** | @stemscriber | https://www.instagram.com/stemscriber | live |
| **X (Twitter)** | @stemscriber | https://x.com/stemscriber | live |
| **TikTok** | @stemscriber | https://www.tiktok.com/@stemscriber | live |
| **Facebook** | (pending vanity) | https://www.facebook.com/profile.php?id=61582164772137 | live but no vanity URL claimed — Kevin to set `/stemscriber` once eligible |
| **YouTube** | @Stemscriber | https://www.youtube.com/@stemscriber | **Brand new channel — created May 2026.** Vanity handle URL is active. Channel ID URL is https://www.youtube.com/channel/UCn2X2H87Nb_p4TQRkmCE4PQ for reference. **Day 1 setup needed:** banner art (default placeholder is YouTube's red+white TV image — 2048×1152 px min, 6MB max), profile picture is in place. **Inaugural upload candidate:** the full walkthrough video (task #78). |

**Footer link gaps to fix on stemscriber.com:**
- YouTube channel URL is NOT currently in the site footer — needs adding (use vanity URL `https://www.youtube.com/@stemscriber`).
- Facebook footer link uses the numeric profile ID — should swap to vanity URL once Kevin claims it.

---

## 2. Brand assets — already in repo

### Voice + visual direction
- `marketing/BRAND_GUIDE.md` — voice principles, visual aesthetic ("record store meets control room"), tagline rationale
- `marketing/POSITIONING.md` — competitive positioning, what we are / aren't
- `marketing/SOCIAL_STRATEGY.md` — channel strategy, posting cadence framework
- `marketing/content_calendar.md` — 2-week beta launch calendar (template, needs updating)
- `marketing/social_posts.md` — drafted posts (reference voice, not necessarily for use)
- `marketing/LANDING_PAGE_COPY.md` — landing page copy
- `marketing/legal_analysis.md` — copyright/fair-use posture
- `marketing/FINANCIAL_MODEL.md` — financial model (private — Kevin needs only if equity discussion goes forward)

### Image assets (in `marketing/social-images/`)
- `facebook_cover.png`
- `facebook_profile.png`
- `instagram_profile.png`
- `twitter_banner.png`
- `twitter_profile.png`
- **GAP:** no TikTok profile image yet
- **GAP:** no YouTube banner yet (2560×1440 px recommended)

### Logo
- `frontend/images/logomark.png` — 32×32 mark used in site header

### Domain assets
- Site: https://stemscriber.com
- Email: anything@stemscriber.com routes via Cloudflare email forwarding (currently `support@` is set up). Can add `kevin@stemscriber.com` if Kevin wants a branded address.

---

## 3. Tooling access — Jeff needs to invite Kevin

These are NOT credentials to share via this doc. Jeff invites Kevin via the platform's normal user-invitation flow (email-based).

| Tool | Purpose | Action for Jeff |
|---|---|---|
| **Plausible Analytics** | Site traffic dashboard | Add Kevin as a team member (Plausible → Settings → People → Invite) |
| **Cloudflare** | DNS, email routing, tunnel | Optional — only if Kevin needs to manage the website domain |
| **Stripe** | Subscriber metrics | Skip unless Kevin needs revenue dashboards (probably not for marketing) |
| **GitHub** | Repo access | Skip — Kevin doesn't need code access |
| **n8n** | Automation flows | Skip unless social-post automation is in scope |
| **Google Drive** | Shared docs | Add Kevin to any folders Jeff already shares with Tidepool / collaborators |

---

## 4. Credentials handoff — DO THIS SEPARATELY, NOT IN THIS DOC

Social media platform login credentials (FB, IG, X, TikTok, YouTube/Google) **must NOT** be put in this file or any other plaintext doc.

**Recommended channels:**
1. **1Password Family** or similar password manager — share each login as a vault item with Kevin's email. Lets you revoke access cleanly later.
2. **Apple Passwords share** if you both use Apple devices (iOS 17+ has a sharing feature).
3. **Last resort:** in-person from your laptop, Kevin types them into his password manager directly. Slow but rock-solid.

**What Kevin needs login access to (5 logins):**
- Facebook (the StemScriber Page — Kevin should be added as a Page admin via Meta Business Suite, not by sharing your personal FB password)
- Instagram (linked to the FB Page — same admin add through Meta)
- X (@stemscriber)
- TikTok (@stemscriber)
- YouTube (which is the StemScriber Google account)

**Important — FB and IG:** don't share your personal Facebook password. Use **Meta Business Suite** to add Kevin as a Page Admin / Editor on both FB Page and IG Business Account. That gives him posting + analytics access without exposing your personal Meta login.

**X / TikTok / YouTube:** these don't have proper team-management at the free tier — you'll need to share the actual account login. Use a password manager for clean revocation.

---

## 5. What to give Kevin first

**Must-haves at handoff:**
1. This doc (`kevin-onboarding-2026-05-06.md`)
2. The lunch-prep doc with role + barter terms (`marketing-prep-kevin-2026-05-05.md`)
3. `marketing/BRAND_GUIDE.md` — set the voice
4. `marketing/POSITIONING.md` — set the competitive frame
5. `marketing/social-images/` folder — current profile/cover assets
6. The 5 social URLs above
7. Login credentials via password manager (separate channel)
8. Plausible team invite

**Nice-to-haves for context:**
- `docs/competitor-landing-research-2026-04-28.md` — recent competitor audit
- `docs/marketing-drafts-2026-04-26.md` — draft posts/threads (reference, not gospel)
- `marketing/SOCIAL_STRATEGY.md` — strategy framework
- `marketing/content_calendar.md` — calendar structure (current dates are stale)

**Hold back until first deliverable lands:**
- `marketing/FINANCIAL_MODEL.md` — share only if/when equity discussion goes forward
- Code repository access — not needed for marketing

---

## 6. Open items for Jeff before the handoff

- [ ] Decide whether to give Kevin a `kevin@stemscriber.com` email forward
- [ ] Add Kevin as Page Admin / Editor on Facebook and Instagram via Meta Business Suite
- [ ] Add Kevin to Plausible team
- [ ] Share X / TikTok / YouTube logins via password manager (NOT plaintext)
- [ ] Create TikTok profile image (matching the existing IG / FB / X aesthetic)
- [ ] Replace YouTube banner — current is a red+white TV/laptop placeholder, not on brand (2048×1152 px minimum, 6MB max per YouTube)
- [ ] Add YouTube link to website footer alongside FB/IG/X/TikTok
- [ ] Once Kevin sets vanity URLs for FB Page and YouTube, swap the footer URLs to match
