"""
Plan definitions and Stripe price ID mapping.

Stripe price IDs are loaded from environment variables so that
test-mode and live-mode keys can differ without code changes.
"""

import os

# ============ PLAN TIERS ============

PLANS = {
    'free': {
        'name': 'Free',
        'monthly_price': 0,
        'annual_price': 0,
        'songs_per_month': 5,
        'max_duration_sec': 300,       # 5 minutes
        'stems': 6,
        'features': ['6-stem separation', 'MP3 output (128kbps)'],
    },
    'pro': {
        'name': 'Pro',
        'monthly_price': 10,
        'annual_price': 89,
        'songs_per_month': 30,
        'max_duration_sec': 1800,      # 30 minutes
        'stems': 6,
        'features': [
            '6-stem separation (+ guitar, piano)',
            'Chord detection + key analysis',
            'Guitar Pro tab export',
            'Priority processing',
        ],
    },
    'lifetime': {
        'name': 'Lifetime Founder',
        'monthly_price': 0,
        'annual_price': 0,
        'one_time_price': 199,
        'qty_cap': 100,
        'songs_per_month': 50,         # 50/mo cap (margin) — NOT unlimited
        'max_duration_sec': 1800,      # 30 minutes
        'stems': 6,
        'features': [
            'Everything in Pro, 50 songs per month',
            'Lifetime access — never billed again',
            'Founding-customer badge',
            'First-look on every new feature',
            'Limited to first 100 customers',
        ],
    },
    'beta': {
        'name': 'Beta',
        'monthly_price': 0,
        'annual_price': 0,
        'songs_per_month': -1,         # unlimited (matches pro)
        'max_duration_sec': 1800,      # 30 minutes
        'stems': 6,
        'features': [
            '6-stem separation (+ guitar, piano)',
            'Chord detection + key analysis',
            'Guitar Pro tab export',
            'Priority processing',
            'WAV lossless output',
        ],
    },
}


def get_stripe_prices():
    """Return mapping of plan+interval to Stripe Price IDs from env vars."""
    return {
        'premium_monthly': os.environ.get('STRIPE_PRICE_PREMIUM_MONTHLY', ''),  # legacy
        'premium_annual': os.environ.get('STRIPE_PRICE_PREMIUM_ANNUAL', ''),    # legacy
        'pro_monthly': os.environ.get('STRIPE_PRICE_PRO_MONTHLY', ''),
        'pro_annual': os.environ.get('STRIPE_PRICE_PRO_ANNUAL', ''),
        'lifetime_founder': os.environ.get('STRIPE_PRICE_LIFETIME_FOUNDER', ''),
    }


def get_price_id(plan: str, interval: str = '') -> str | None:
    """Look up the Stripe Price ID for a plan + interval combo.

    Args:
        plan: 'pro', 'premium' (legacy), or 'lifetime'
        interval: 'monthly' or 'annual' for subscription plans, ignored for lifetime

    Returns:
        Stripe price_xxx string, or None if not configured.
    """
    prices = get_stripe_prices()
    if plan == 'lifetime':
        return prices.get('lifetime_founder') or None
    key = f'{plan}_{interval}'
    price_id = prices.get(key)
    return price_id if price_id else None


def plan_from_price_id(price_id: str) -> str | None:
    """Reverse lookup: given a Stripe Price ID, return the plan name.

    Returns 'pro', 'premium' (legacy), 'lifetime', or None.
    """
    prices = get_stripe_prices()
    for key, pid in prices.items():
        if pid == price_id:
            if key == 'lifetime_founder':
                return 'lifetime'
            # key is like 'premium_monthly' or 'pro_annual'
            return key.split('_')[0]
    return None


# Hard cap on Lifetime Founder seats. Enforced in create_checkout_session.
LIFETIME_FOUNDER_CAP = 100
