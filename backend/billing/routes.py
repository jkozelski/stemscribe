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

from auth.models import get_user_by_id, update_user_plan
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
        plans_list.append({
            'id': plan_id,
            'name': info['name'],
            'monthly_price': info['monthly_price'],
            'annual_price': info['annual_price'],
            'songs_per_month': info['songs_per_month'],
            'max_duration_sec': info['max_duration_sec'],
            'stems': info['stems'],
            'features': info['features'],
        })
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
    if plan not in ('premium', 'pro'):
        return jsonify({'error': 'Invalid plan. Choose "premium" or "pro"'}), 400

    if interval not in ('monthly', 'annual'):
        return jsonify({'error': 'Invalid interval. Choose "monthly" or "annual"'}), 400

    # Check if user is already on this plan
    if user.plan == plan:
        return jsonify({
            'error': f'You are already on the {plan} plan',
            'current_plan': user.plan,
        }), 409

    # Look up the Stripe Price ID
    price_id = get_price_id(plan, interval)
    if not price_id:
        return jsonify({
            'error': f'Stripe price not configured for {plan}/{interval}. '
                     f'Set STRIPE_PRICE_{plan.upper()}_{interval.upper()} env var.',
        }), 500

    app_url = _get_app_url()

    try:
        # If user already has a Stripe customer ID, reuse it
        customer_kwargs = {}
        if user.stripe_customer_id:
            customer_kwargs['customer'] = user.stripe_customer_id
        else:
            customer_kwargs['customer_email'] = user.email

        session = stripe.checkout.Session.create(
            mode='subscription',
            line_items=[{'price': price_id, 'quantity': 1}],
            success_url=f'{app_url}/billing/success?session_id={{CHECKOUT_SESSION_ID}}',
            cancel_url=f'{app_url}/pricing',
            metadata={
                'user_id': str(user.id),
                'plan': plan,
            },
            subscription_data={
                'metadata': {
                    'user_id': str(user.id),
                    'plan': plan,
                },
            },
            allow_promotion_codes=True,
            **customer_kwargs,
        )

        logger.info(f"Checkout session created for user {user_id}: {plan}/{interval}")

        return jsonify({
            'checkout_url': session.url,
            'session_id': session.id,
        })

    except stripe.error.StripeError as e:
        logger.error(f"Stripe error creating checkout: {e}")
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

    # Fetch subscription details from Stripe if available
    if user.stripe_subscription_id:
        try:
            sub = stripe.Subscription.retrieve(user.stripe_subscription_id)
            result['subscription'] = {
                'id': sub.id,
                'status': sub.status,
                'current_period_end': sub.current_period_end,
                'cancel_at_period_end': sub.cancel_at_period_end,
                'interval': sub['items']['data'][0]['price']['recurring']['interval']
                            if sub['items']['data'] else None,
            }
        except stripe.error.StripeError as e:
            logger.warning(f"Could not fetch subscription {user.stripe_subscription_id}: {e}")
            result['subscription'] = None
    else:
        result['subscription'] = None

    return jsonify(result)
