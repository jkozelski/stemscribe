"""
Plan definitions, canonical pricing, and Stripe price-ID mapping.

⚠️  SINGLE SOURCE OF TRUTH FOR PRICING.
If you change a price, change it HERE, then update the matching Stripe price
and Apple IAP product, and walk the checklist in docs/PRICING.md so no surface
drifts. Canonical prices (locked 2026-06-19):

    Free       —  $0
    Pro        —  $10/mo   or   $89/yr   (30 songs/mo)
    Lifetime   —  $199 one-time          (50 songs/mo)
    Song Pack  —  $5 / 10 songs          (overage add-on, not a tier)

There is NO "Premium" tier (retired 2026-06-19) and NO $100/$200 pricing.
"""

import os

# ============ PLAN TIERS ============

PLANS = {
    'free': {
        'name': 'Free',
        'monthly_price': 0,
        'annual_price': 0,
        'one_time_price': None,
        'songs_per_month': 3,
        'max_duration_sec': 300,       # 5 minutes
        'stems': 4,
        'features': ['Basic 4-stem separation', 'MP3 output (128kbps)'],
    },
    'pro': {
        'name': 'Pro',
        'monthly_price': 10,
        'annual_price': 89,
        'one_time_price': None,
        'songs_per_month': 30,
        'max_duration_sec': 900,       # 15 minutes
        'stems': 6,
        'features': [
            '6-stem separation (+ guitar, piano)',
            'Chord detection + key analysis',
            'Chord chart export (PDF + ChordPro)',
            'Guitar Pro tab export',
            'Priority processing',
        ],
    },
    'lifetime': {
        'name': 'Lifetime Founder',
        'monthly_price': 0,
        'annual_price': 0,
        'one_time_price': 199,
        'songs_per_month': 50,
        'max_duration_sec': 1800,      # 30 minutes
        'stems': 6,
        'features': [
            'Everything in Pro',
            '50 songs per month',
            'Lifetime access — never billed again',
            'Founding-customer badge',
        ],
    },
    # 'beta' is retained ONLY for the transition off the invite gate. It grants
    # Pro-equivalent access so existing beta-code redeemers are not downgraded.
    # Remove once the beta gate is fully gone.
    'beta': {
        'name': 'Beta',
        'monthly_price': 0,
        'annual_price': 0,
        'one_time_price': None,
        'songs_per_month': 30,
        'max_duration_sec': 900,
        'stems': 6,
        'features': ['Pro-equivalent access (beta)'],
    },
}


def get_stripe_prices():
    """Return mapping of plan+interval to Stripe Price IDs from env vars.

    Env-backed so test-mode and live-mode price IDs can differ without code
    changes. Canonical live IDs are documented in docs/PRICING.md.
    """
    return {
        'pro_monthly': os.environ.get('STRIPE_PRICE_PRO_MONTHLY', ''),
        'pro_annual': os.environ.get('STRIPE_PRICE_PRO_ANNUAL', ''),
        'lifetime': os.environ.get('STRIPE_PRICE_LIFETIME_FOUNDER', ''),
        'song_pack': os.environ.get('STRIPE_PRICE_SONG_PACK_10', ''),
    }


def get_price_id(plan: str, interval: str = 'monthly') -> str | None:
    """Look up the Stripe Price ID for a plan (+ interval for subscriptions).

    Args:
        plan: 'pro' or 'lifetime'
        interval: 'monthly' or 'annual' (ignored for one-time 'lifetime')

    Returns:
        Stripe price_xxx string, or None if not configured.
    """
    prices = get_stripe_prices()
    if plan == 'lifetime':
        return prices.get('lifetime') or None
    key = f'{plan}_{interval}'
    return prices.get(key) or None


def plan_from_price_id(price_id: str) -> str | None:
    """Reverse lookup: given a Stripe Price ID, return the plan name.

    Returns 'pro' or 'lifetime', or None if the price ID is not recognized.
    """
    prices = get_stripe_prices()
    for key, pid in prices.items():
        if pid and pid == price_id:
            return key.split('_')[0]   # 'pro_monthly' -> 'pro', 'lifetime' -> 'lifetime'
    return None
