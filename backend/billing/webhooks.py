"""
Stripe webhook handler — processes subscription lifecycle events.

Endpoint:
    POST /webhooks/stripe

Events handled:
    checkout.session.completed      — User completed checkout, activate plan
    customer.subscription.updated   — Plan changed (upgrade/downgrade)
    customer.subscription.deleted   — Subscription cancelled, revert to free
    invoice.payment_failed          — Payment failed, flag user
    invoice.paid                    — Payment succeeded, clear failure flag

Security:
    All events are verified against the Stripe webhook signing secret.
    Raw request body is used for signature verification (not parsed JSON).
"""

import os
import logging

import stripe
from flask import Blueprint, request, jsonify

from auth.models import (
    get_user_by_id,
    update_user_plan,
    set_payment_failed,
    clear_payment_failed,
)
from billing.plans import plan_from_price_id
from db import query_one, execute

logger = logging.getLogger(__name__)

webhooks_bp = Blueprint('webhooks', __name__, url_prefix='/webhooks')


def _get_webhook_secret():
    return os.environ.get('STRIPE_WEBHOOK_SECRET', '')


def _verify_stripe_signature(payload: bytes, sig_header: str):
    """Verify Stripe webhook signature. Raises on failure."""
    secret = _get_webhook_secret()
    if not secret:
        raise ValueError("STRIPE_WEBHOOK_SECRET not configured")
    return stripe.Webhook.construct_event(payload, sig_header, secret)


def _find_user_by_stripe_customer(customer_id: str):
    """Look up a user by their Stripe customer ID."""
    row = query_one(
        "SELECT * FROM users WHERE stripe_customer_id = %s",
        (customer_id,),
    )
    if row:
        from auth.models import User
        return User(row)
    return None


# ============ WEBHOOK ENDPOINT ============

@webhooks_bp.route('/stripe', methods=['POST'])
def stripe_webhook():
    """Handle incoming Stripe webhook events.

    IMPORTANT: This endpoint must receive the raw request body for
    signature verification. Flask's request.get_data() provides this.
    """
    payload = request.get_data()
    sig_header = request.headers.get('Stripe-Signature', '')

    try:
        event = _verify_stripe_signature(payload, sig_header)
    except stripe.error.SignatureVerificationError:
        logger.warning("Stripe webhook: invalid signature")
        return jsonify({'error': 'Invalid signature'}), 400
    except ValueError as e:
        logger.error(f"Stripe webhook config error: {e}")
        return jsonify({'error': 'Webhook not configured'}), 500

    event_type = event['type']
    logger.info(f"Stripe webhook received: {event_type}")

    try:
        if event_type == 'checkout.session.completed':
            _handle_checkout_completed(event['data']['object'])
        elif event_type == 'customer.subscription.updated':
            _handle_subscription_updated(event['data']['object'])
        elif event_type == 'customer.subscription.deleted':
            _handle_subscription_deleted(event['data']['object'])
        elif event_type == 'invoice.payment_failed':
            _handle_payment_failed(event['data']['object'])
        elif event_type == 'invoice.paid':
            _handle_payment_succeeded(event['data']['object'])
        else:
            logger.debug(f"Unhandled Stripe event type: {event_type}")
    except Exception as e:
        logger.error(f"Error handling {event_type}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Return 200 anyway to prevent Stripe from retrying
        # (we log the error and can investigate manually)

    return jsonify({'status': 'ok'}), 200


# ============ EVENT HANDLERS ============

def _handle_checkout_completed(session):
    """User completed Stripe Checkout -- activate their plan.

    The session metadata contains user_id and plan, which we set
    when creating the checkout session in routes.py.
    """
    user_id = session.get('metadata', {}).get('user_id')
    plan = session.get('metadata', {}).get('plan')
    customer_id = session.get('customer')
    subscription_id = session.get('subscription')

    if not user_id or not plan:
        logger.error(f"checkout.session.completed missing metadata: {session.get('id')}")
        return

    user = get_user_by_id(user_id)
    if not user:
        logger.error(f"checkout.session.completed: user {user_id} not found")
        return

    update_user_plan(
        user_id=user_id,
        plan=plan,
        stripe_customer_id=customer_id,
        stripe_subscription_id=subscription_id,
    )
    logger.info(f"User {user_id} upgraded to {plan} (customer={customer_id}, sub={subscription_id})")


def _handle_subscription_updated(subscription):
    """Subscription changed -- update plan if price changed (upgrade/downgrade)."""
    customer_id = subscription.get('customer')
    subscription_id = subscription.get('id')

    user = _find_user_by_stripe_customer(customer_id)
    if not user:
        logger.warning(f"subscription.updated: no user for customer {customer_id}")
        return

    # Determine the new plan from the subscription's price
    items = subscription.get('items', {}).get('data', [])
    if not items:
        logger.warning(f"subscription.updated: no items in subscription {subscription_id}")
        return

    price_id = items[0].get('price', {}).get('id')
    new_plan = plan_from_price_id(price_id) if price_id else None

    if not new_plan:
        # Check subscription metadata as fallback
        new_plan = subscription.get('metadata', {}).get('plan')

    if new_plan and new_plan != user.plan:
        update_user_plan(
            user_id=str(user.id),
            plan=new_plan,
            stripe_customer_id=customer_id,
            stripe_subscription_id=subscription_id,
        )
        logger.info(f"User {user.id} plan changed: {user.plan} -> {new_plan}")
    elif new_plan:
        logger.debug(f"subscription.updated for user {user.id} but plan unchanged ({new_plan})")


def _handle_subscription_deleted(subscription):
    """Subscription cancelled or expired -- revert to free plan."""
    customer_id = subscription.get('customer')

    user = _find_user_by_stripe_customer(customer_id)
    if not user:
        logger.warning(f"subscription.deleted: no user for customer {customer_id}")
        return

    update_user_plan(
        user_id=str(user.id),
        plan='free',
        stripe_customer_id=customer_id,
        stripe_subscription_id=None,
    )
    logger.info(f"User {user.id} subscription cancelled, reverted to free plan")


def _handle_payment_failed(invoice):
    """Payment failed -- flag the user so we can show a grace period notice."""
    customer_id = invoice.get('customer')

    user = _find_user_by_stripe_customer(customer_id)
    if not user:
        logger.warning(f"invoice.payment_failed: no user for customer {customer_id}")
        return

    set_payment_failed(str(user.id))
    logger.warning(f"Payment failed for user {user.id} (customer={customer_id})")


def _handle_payment_succeeded(invoice):
    """Payment succeeded -- clear any payment failure flags."""
    customer_id = invoice.get('customer')

    user = _find_user_by_stripe_customer(customer_id)
    if not user:
        # This fires for every successful invoice, including first payment.
        # If user doesn't exist by customer_id yet, it's handled by checkout.session.completed.
        return

    if user.payment_failed_at:
        clear_payment_failed(str(user.id))
        logger.info(f"Payment succeeded for user {user.id}, cleared failure flag")
