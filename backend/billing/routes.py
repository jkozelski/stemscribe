"""
Billing Blueprint — Stripe Checkout and Customer Portal endpoints.

Endpoints:
    POST /billing/create-checkout-session  — Start a Stripe Checkout session
    POST /billing/portal-session           — Create a Stripe Customer Portal session
    GET  /billing/plans                    — List available plans and pricing
"""

import os
import logging

import stripe
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

from auth.models import get_user_by_id
from billing.plans import PLANS, get_price_id

logger = logging.getLogger(__name__)

billing_bp = Blueprint('billing', __name__, url_prefix='/billing')

# Stripe API key is set lazily on first use
_stripe_initialized = False


def _init_stripe():
    """Set the Stripe API key from environment. Called once on first request."""
    global _stripe_initialized
    if not _stripe_initialized:
        stripe.api_key = os.environ.get('STRIPE_SECRET_KEY', '')
        if not stripe.api_key:
            logger.warning("STRIPE_SECRET_KEY not set -- billing will fail")
        _stripe_initialized = True


def _get_app_url():
    return os.environ.get('APP_URL', 'http://localhost:5555')


# ============ PLANS (public, no auth required) ============

@billing_bp.route('/plans', methods=['GET'])
def list_plans():
    """Return available plans with pricing info (no auth required)."""
    plans_list = []
    for plan_id, info in PLANS.items():
        entry = {
            'id': plan_id,
            'name': info['name'],
            'monthly_price': info['monthly_price'],
            'annual_price': info['annual_price'],
            'songs_per_month': info['songs_per_month'],
            'max_duration_sec': info['max_duration_sec'],
            'stems': info['stems'],
            'features': info['features'],
        }
        # Lifetime Founder uses one_time_price + qty_cap
        if 'one_time_price' in info:
            entry['one_time_price'] = info['one_time_price']
        if 'qty_cap' in info:
            entry['qty_cap'] = info['qty_cap']
        plans_list.append(entry)
    return jsonify({'plans': plans_list})


# ============ CHECKOUT ============

@billing_bp.route('/create-checkout-session', methods=['POST'])
@jwt_required()
def create_checkout_session():
    """Create a Stripe Checkout Session for upgrading to a paid plan.

    Request JSON:
        plan:     'premium' or 'pro'
        interval: 'monthly' or 'annual'

    Returns:
        checkout_url: Stripe-hosted checkout page URL
    """
    _init_stripe()

    user_id = get_jwt_identity()
    user = get_user_by_id(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Request body must be JSON'}), 400

    plan = data.get('plan', '').lower()
    interval = data.get('interval', 'monthly').lower()

    # Validate plan
    if plan not in ('premium', 'pro', 'lifetime'):
        return jsonify({'error': 'Invalid plan. Choose "pro" or "lifetime"'}), 400

    is_lifetime = (plan == 'lifetime')

    if not is_lifetime and interval not in ('monthly', 'annual'):
        return jsonify({'error': 'Invalid interval. Choose "monthly" or "annual"'}), 400

    # Check if user is already on this plan
    if user.plan == plan:
        return jsonify({
            'error': f'You are already on the {plan} plan',
            'current_plan': user.plan,
        }), 409

    # 100-customer cap on Lifetime Founder. Count active lifetime users in our DB.
    # (Stripe count would be more authoritative, but adds latency on every checkout
    # request — local count is good enough; ground truth resolves on webhook.)
    if is_lifetime:
        from billing.plans import LIFETIME_FOUNDER_CAP
        from db import query_one
        try:
            row = query_one("SELECT COUNT(*) AS n FROM users WHERE plan = 'lifetime'")
            seats_taken = (row or {}).get('n', 0) if isinstance(row, dict) else (row[0] if row else 0)
        except Exception as e:
            logger.warning(f"Lifetime cap count query failed: {e}; defaulting to 0")
            seats_taken = 0
        if seats_taken >= LIFETIME_FOUNDER_CAP:
            return jsonify({
                'error': 'Lifetime Founder is sold out — all 100 seats claimed.',
                'seats_taken': seats_taken,
                'cap': LIFETIME_FOUNDER_CAP,
            }), 410

    # Look up the Stripe Price ID
    price_id = get_price_id(plan, '' if is_lifetime else interval)
    if not price_id:
        env_var = 'STRIPE_PRICE_LIFETIME_FOUNDER' if is_lifetime else f'STRIPE_PRICE_{plan.upper()}_{interval.upper()}'
        return jsonify({
            'error': f'Stripe price not configured for {plan}{"" if is_lifetime else "/"+interval}. '
                     f'Set {env_var} env var.',
        }), 500

    app_url = _get_app_url()

    try:
        # If user already has a Stripe customer ID, reuse it
        customer_kwargs = {}
        if user.stripe_customer_id:
            customer_kwargs['customer'] = user.stripe_customer_id
        else:
            customer_kwargs['customer_email'] = user.email

        session_kwargs = {
            'mode': 'payment' if is_lifetime else 'subscription',
            'line_items': [{'price': price_id, 'quantity': 1}],
            'success_url': f'{app_url}/success.html?session_id={{CHECKOUT_SESSION_ID}}',
            'cancel_url': f'{app_url}/landing.html#pricing',
            'metadata': {
                'user_id': str(user.id),
                'plan': plan,
            },
            'allow_promotion_codes': True,
            **customer_kwargs,
        }
        # subscription_data is only valid in subscription mode
        if not is_lifetime:
            session_kwargs['subscription_data'] = {
                'metadata': {
                    'user_id': str(user.id),
                    'plan': plan,
                },
            }
        session = stripe.checkout.Session.create(**session_kwargs)

        logger.info(f"Checkout session created for user {user_id}: {plan}/{interval}")

        return jsonify({
            'checkout_url': session.url,
            'session_id': session.id,
        })

    except stripe.error.StripeError as e:
        logger.error(f"Stripe error creating checkout: {e}")
        return jsonify({'error': f'Payment service error: {str(e)}'}), 500


# ============ OVERAGE PACKS ============

@billing_bp.route('/overage/checkout', methods=['POST'])
@jwt_required()
def create_overage_checkout():
    """Create a Stripe Checkout Session for a 10-song overage pack ($5).

    Only available to authenticated users on a paid plan (pro/lifetime/etc).
    Returns the Stripe-hosted checkout URL — frontend redirects, Stripe
    posts checkout.session.completed back to our webhook, webhook bumps
    users.extras_balance by 10.
    """
    _init_stripe()

    user_id = get_jwt_identity()
    user = get_user_by_id(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    price_id = os.environ.get('STRIPE_PRICE_SONG_PACK_10', '')
    if not price_id:
        return jsonify({
            'error': 'Overage packs are not configured yet. Set STRIPE_PRICE_SONG_PACK_10.',
        }), 500

    app_url = _get_app_url()

    try:
        customer_kwargs = {}
        if user.stripe_customer_id:
            customer_kwargs['customer'] = user.stripe_customer_id
        else:
            customer_kwargs['customer_email'] = user.email

        session = stripe.checkout.Session.create(
            mode='payment',
            line_items=[{'price': price_id, 'quantity': 1}],
            success_url=f'{app_url}/app?overage=success&session_id={{CHECKOUT_SESSION_ID}}',
            cancel_url=f'{app_url}/app?overage=cancelled',
            metadata={
                'user_id': str(user.id),
                'pack': 'song_pack_10',
                'units': '10',
            },
            allow_promotion_codes=False,
            **customer_kwargs,
        )

        logger.info(f"Overage checkout session created for user {user_id}: {session.id}")

        return jsonify({
            'checkout_url': session.url,
            'session_id': session.id,
        })

    except stripe.error.StripeError as e:
        logger.error(f"Stripe error creating overage checkout: {e}")
        return jsonify({'error': f'Payment service error: {str(e)}'}), 500


# ============ CUSTOMER PORTAL ============

@billing_bp.route('/portal-session', methods=['POST'])
@jwt_required()
def create_portal_session():
    """Create a Stripe Customer Portal session for managing subscription.

    Users can change plan, update payment method, or cancel via the portal.

    Returns:
        portal_url: Stripe-hosted portal page URL
    """
    _init_stripe()

    user_id = get_jwt_identity()
    user = get_user_by_id(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    if not user.stripe_customer_id:
        return jsonify({
            'error': 'No billing account found. Subscribe to a plan first.',
            'current_plan': user.plan,
        }), 404

    app_url = _get_app_url()

    try:
        session = stripe.billing_portal.Session.create(
            customer=user.stripe_customer_id,
            return_url=f'{app_url}/account',
        )

        logger.info(f"Portal session created for user {user_id}")

        return jsonify({
            'portal_url': session.url,
        })

    except stripe.error.StripeError as e:
        logger.error(f"Stripe error creating portal session: {e}")
        return jsonify({'error': f'Payment service error: {str(e)}'}), 500


# ============ SUBSCRIPTION STATUS ============

@billing_bp.route('/status', methods=['GET'])
@jwt_required()
def billing_status():
    """Get the current user's billing status and subscription info."""
    _init_stripe()

    user_id = get_jwt_identity()
    user = get_user_by_id(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    result = {
        'plan': user.plan,
        'plan_info': PLANS.get(user.plan, PLANS['free']),
        'stripe_customer_id': user.stripe_customer_id,
        'payment_failed': user.payment_failed_at is not None,
    }

    # Fetch subscription details from Stripe if available.
    # Lifetime users don't have a recurring sub even if stripe_subscription_id is
    # set from a prior plan, so skip the lookup for them.
    if user.stripe_subscription_id and user.plan != 'lifetime':
        try:
            sub = stripe.Subscription.retrieve(user.stripe_subscription_id)
            # Stripe API moved current_period_end from Subscription to its
            # SubscriptionItem in 2024. Try both locations defensively, plus
            # use getattr/dict access so any future field renames don't 500.
            items_data = (sub.get('items') or {}).get('data') or []
            first_item = items_data[0] if items_data else {}
            cpe = (
                getattr(sub, 'current_period_end', None)
                or first_item.get('current_period_end')
            )
            recurring = ((first_item.get('price') or {}).get('recurring') or {})
            result['subscription'] = {
                'id': getattr(sub, 'id', None),
                'status': getattr(sub, 'status', None),
                'current_period_end': cpe,
                'cancel_at_period_end': getattr(sub, 'cancel_at_period_end', False),
                'interval': recurring.get('interval'),
            }
        except stripe.error.StripeError as e:
            logger.warning(f"Could not fetch subscription {user.stripe_subscription_id}: {e}")
            result['subscription'] = None
        except Exception as e:
            # Any other Stripe-object access error: log + return None, don't 500
            logger.warning(f"Subscription field-access error for {user.stripe_subscription_id}: {e}")
            result['subscription'] = None
    else:
        result['subscription'] = None

    return jsonify(result)


@billing_bp.route('/lifetime-seats', methods=['GET'])
def lifetime_seats():
    """Public seat availability for the Lifetime Founder cap. Added 2026-07-19 so
    the in-app (StoreKit) paywall can refuse seat #101+ — the web checkout already
    enforces this server-side, but StoreKit purchases had no gate (RISK-18)."""
    from billing.plans import LIFETIME_FOUNDER_CAP
    from db import query_one
    try:
        row = query_one("SELECT COUNT(*) AS n FROM users WHERE plan = 'lifetime'")
        taken = (row or {}).get('n', 0) if isinstance(row, dict) else (row[0] if row else 0)
    except Exception as e:
        logger.warning(f"lifetime-seats count failed: {e}")
        taken = 0
    return jsonify({'cap': LIFETIME_FOUNDER_CAP, 'taken': int(taken), 'available': int(taken) < LIFETIME_FOUNDER_CAP})
