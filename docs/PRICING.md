# StemScriber Pricing — Single Source of Truth

**Locked 2026-06-19.** If you change a price, change it in **every** numbered surface below, in the same sitting. Prices must be identical across **website, Stripe, Apple IAP, and code**. There is **no "Premium" tier** (retired 2026-06-19) and **no $100 / $200** pricing anywhere.

## Canonical prices
| Tier | Price | Songs/mo | Notes |
|---|---|---|---|
| **Free** | $0 | 3 | 4 stems, 5-min cap |
| **Pro** | **$10/mo** or **$89/yr** | 30 | 6 stems, chords, tab/MIDI export, priority |
| **Lifetime Founder** | **$199 one-time** | 50 | Everything in Pro + founder badge |
| **Song Pack** | **$5 / 10 songs** | — | Overage add-on, not a tier |

## Stripe live price IDs (the anchor)
- Pro monthly $10 → `price_1TBzzCAEwUQPqC7VUfQegQe3`
- Pro annual $89 → `price_1ThujgAEwUQPqC7V6xX4A9KR`
- Lifetime $199 → `price_1TbkNXAEwUQPqC7VzQ3gC4TL`
- Song Pack $5 → `price_1TbkNZAEwUQPqC7VZ1vy3HOc`
- ~~Premium $20/mo `price_1TBzzD…`~~ **ARCHIVED 2026-06-19**
- ~~Premium $200/yr `price_1TBzzE…`~~ **ARCHIVED 2026-06-19**

## Every surface that states a price (change ALL together)
1. **`backend/billing/plans.py`** — the code source of truth (`PLANS` + `get_stripe_prices`). ⭐ Start here.
2. **`backend/auth/decorators.py`** — `PLAN_LIMITS` + `PLAN_HIERARCHY` (must match plans.py).
3. **`backend/middleware/rate_limit.py`** — doc comment of limits.
4. **Stripe Dashboard** — products + prices (archive, don't delete, when retiring).
5. **Prod env** — `STRIPE_PRICE_PRO_MONTHLY / PRO_ANNUAL / LIFETIME_FOUNDER / SONG_PACK_10` (in BOTH `backend/.env` and root `.env` — backend overrides root).
6. **`frontend/landing.html`** — the public pricing section.
7. **`frontend/help.html`** — Help Center pricing answers.
8. **`frontend/billing-faq.html`** — Billing FAQ tables.
9. **Apple App Store Connect** — IAP subscription products (when IAP ships; must match).

## The "never again" rule
Code prices live in `plans.py`. The long-term fix (TODO) is a `/api/plans` endpoint so the frontend pulls prices instead of hardcoding them — then surfaces 6–8 above stop being manual. Until then, this checklist is the guardrail.
