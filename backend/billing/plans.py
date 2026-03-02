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
        'songs_per_month': 3,
        'max_duration_sec': 300,       # 5 minutes
        'stems': 4,
        'features': ['Basic 4-stem separation', 'MP3 output (128kbps)'],
    },
    'premium': {
        'name': 'Premium',
        'monthly_price': 4.99,
        'annual_price': 39.99,
        'songs_per_month': 50,
        'max_duration_sec': 900,       # 15 minutes
        'stems': 6,
        'features': [
            '6-stem separation (+ guitar, piano)',
            'Chord recognition',
            'MIDI export',
            '320kbps MP3 output',
        ],
    },
    'pro': {
        'name': 'Pro',
        'monthly_price': 14.99,
        'annual_price': 119.99,
        'songs_per_month': -1,         # unlimited
        'max_duration_sec': 1800,      # 30 minutes
        'stems': 6,
        'features': [
            'Everything in Premium',
            'Unlimited songs',
            'Guitar Pro / tab export',
            'WAV lossless output',
            'Priority processing queue',
            'Stereo split (dual guitars)',
        ],
    },
}


def get_stripe_prices():
    """Return mapping of plan+interval to Stripe Price IDs from env vars."""
    return {
        'premium_monthly': os.environ.get('STRIPE_PRICE_PREMIUM_MONTHLY', ''),
        'premium_annual': os.environ.get('STRIPE_PRICE_PREMIUM_ANNUAL', ''),
        'pro_monthly': os.environ.get('STRIPE_PRICE_PRO_MONTHLY', ''),
        'pro_annual': os.environ.get('STRIPE_PRICE_PRO_ANNUAL', ''),
    }


def get_price_id(plan: str, interval: str) -> str | None:
    """Look up the Stripe Price ID for a plan + interval combo.

    Args:
        plan: 'premium' or 'pro'
        interval: 'monthly' or 'annual'

    Returns:
        Stripe price_xxx string, or None if not configured.
    """
    prices = get_stripe_prices()
    key = f'{plan}_{interval}'
    price_id = prices.get(key)
    return price_id if price_id else None


def plan_from_price_id(price_id: str) -> str | None:
    """Reverse lookup: given a Stripe Price ID, return the plan name.

    Returns 'premium' or 'pro', or None if the price ID is not recognized.
    """
    prices = get_stripe_prices()
    for key, pid in prices.items():
        if pid == price_id:
            # key is like 'premium_monthly' or 'pro_annual'
            return key.split('_')[0]
    return None
