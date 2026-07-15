"""
Stripe webhook handler — processes subscription lifecycle events.

Endpoint:
    POST /webhooks/stripe

Events handled:
    checkout.session.completed      — User completed checkout, activate plan
    customer.subscription.updated   — Plan changed (upgrade/downgrade)
    customer.subscription.deleted   — Subscription cancelled, revert to free
    charge.refunded                 — Charge fully refunded, revoke entitlement
    invoice.payment_failed          — Payment failed, flag user
    invoice.paid                    — Payment succeeded, clear failure flag

Security:
    All events are verified against the Stripe webhook signing secret.
    Raw request body is used for signature verification (not parsed JSON).
"""

import os
import json
import logging
import threading
import traceback

import stripe
import requests as http_requests
from flask import Blueprint, request, jsonify

from auth.models import (
    get_user_by_id,
    update_user_plan,
    set_payment_failed,
    clear_payment_failed,
    increment_user_extras_balance,
)
from billing.plans import plan_from_price_id
from db import query_one, execute

logger = logging.getLogger(__name__)

webhooks_bp = Blueprint('webhooks', __name__, url_prefix='/webhooks')

# Events that affect revenue/customer entitlement. A failure here means the
# customer paid but their account state is wrong — alert immediately and let
# Stripe retry.
REVENUE_EVENTS = frozenset({
    'checkout.session.completed',
    'customer.subscription.updated',
    'customer.subscription.deleted',
    'charge.refunded',
    'invoice.payment_failed',
    'invoice.paid',
})


def _alert_admin_async(subject: str, body: str) -> None:
    """Best-effort SMS to Jeff. Runs in a background thread so a slow Twilio
    response doesn't hold the webhook return open. Never raises."""
    def _send():
        try:
            sid = os.environ.get('TWILIO_ACCOUNT_SID')
            token = os.environ.get('TWILIO_AUTH_TOKEN')
            from_num = os.environ.get('TWILIO_FROM_NUMBER', '+18447915323')
            to_num = os.environ.get('ADMIN_ALERT_PHONE')
            if not (sid and token and to_num):
                logger.warning("SMS alert skipped — Twilio env not configured")
                return
            msg = (subject + "\n" + body)[:1500]
            r = http_requests.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
                auth=(sid, token),
                data={'From': from_num, 'To': to_num, 'Body': msg},
                timeout=8,
            )
            if not r.ok:
                logger.warning(f"Twilio alert failed: {r.status_code} {r.text[:200]}")
        except Exception as e:
            logger.warning(f"Twilio alert exception: {e}")
    threading.Thread(target=_send, daemon=True).start()


def _record_webhook_failure(event_id, event_type, obj, error):
    """Persist the failure so Jeff can replay it after fixing the root cause."""
    try:
        # Best-effort extraction of who this affected
        customer_id = (obj or {}).get('customer') if isinstance(obj, dict) else None
        meta = (obj or {}).get('metadata') if isinstance(obj, dict) else None
        user_id = (meta or {}).get('user_id') if isinstance(meta, dict) else None
        execute(
            "INSERT INTO webhook_failures "
            "(event_id, event_type, payload, error_message, error_traceback, user_id, customer_id) "
            "VALUES (%s, %s, %s::jsonb, %s, %s, %s, %s)",
            (
                event_id,
                event_type,
                json.dumps(obj, default=str)[:500_000],
                str(error)[:5000],
                traceback.format_exc()[:10000],
                user_id,
                customer_id,
            ),
        )
    except Exception as db_e:
        # If we can't even record the failure, at least log it
        logger.error(f"Could not persist webhook_failures row: {db_e}")


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
    event_id = event.get('id')
    obj = event['data']['object']
    logger.info(f"Stripe webhook received: {event_type} (id={event_id})")

    try:
        if event_type == 'checkout.session.completed':
            _handle_checkout_completed(obj)
        elif event_type == 'customer.subscription.updated':
            _handle_subscription_updated(obj)
        elif event_type == 'customer.subscription.deleted':
            _handle_subscription_deleted(obj)
        elif event_type == 'charge.refunded':
            _handle_charge_refunded(obj)
        elif event_type == 'invoice.payment_failed':
            _handle_payment_failed(obj)
        elif event_type == 'invoice.paid':
            _handle_payment_succeeded(obj)
        else:
            logger.debug(f"Unhandled Stripe event type: {event_type}")
    except Exception as e:
        logger.error(f"Error handling {event_type}: {e}")
        logger.error(traceback.format_exc())

        # Persist for replay regardless of severity
        _record_webhook_failure(event_id, event_type, obj, e)

        if event_type in REVENUE_EVENTS:
            # Customer-money-affecting failure. Alert Jeff and tell Stripe to
            # retry with its exponential backoff (up to ~3 days). Once Jeff
            # fixes the root cause, the retry will succeed and the customer
            # gets the right state without manual intervention.
            customer = obj.get('customer') if isinstance(obj, dict) else None
            amount = obj.get('amount_total') if isinstance(obj, dict) else None
            amount_str = f" ${amount/100:.2f}" if isinstance(amount, int) else ""
            _alert_admin_async(
                subject=f"🚨 StemScriber webhook FAILED: {event_type}{amount_str}",
                body=(
                    f"Event {event_id}\n"
                    f"Customer: {customer or '?'}\n"
                    f"Error: {str(e)[:300]}\n"
                    f"Stripe will retry. Run: SSH and check `webhook_failures` table."
                ),
            )
            return jsonify({'error': 'handler_failed_will_retry'}), 500

        # Non-revenue event — log + 200 so we don't generate retry storms on
        # unhandled side-effect events.
        return jsonify({'status': 'logged_error'}), 200

    return jsonify({'status': 'ok'}), 200


# ============ EVENT HANDLERS ============

def _handle_checkout_completed(session):
    """User completed Stripe Checkout.

    Two flavors:
      1. Plan upgrade (metadata.plan set) -- update users.plan.
      2. Overage pack (metadata.pack == 'song_pack_10') -- bump
         users.extras_balance by `units` (default 10).

    Raises on any condition that means we can't fulfill. Parent handler
    catches, alerts Jeff, persists to webhook_failures, returns 500 so
    Stripe retries.
    """
    meta = session.get('metadata') or {}
    user_id = meta.get('user_id')
    customer_id = session.get('customer')

    if not user_id:
        raise ValueError(
            f"checkout.session.completed missing user_id metadata "
            f"(session={session.get('id')})"
        )

    user = get_user_by_id(user_id)
    if not user:
        raise LookupError(
            f"checkout.session.completed: user_id {user_id} not in DB "
            f"(session={session.get('id')}, customer={customer_id})"
        )

    # ---- Overage pack branch ----
    pack = meta.get('pack')
    if pack == 'song_pack_10':
        try:
            units = int(meta.get('units', '10') or 10)
        except (TypeError, ValueError):
            units = 10
        new_balance = increment_user_extras_balance(user_id, units)
        logger.info(
            f"Overage pack credited: user={user_id} +{units} songs "
            f"(new balance={new_balance}, session={session.get('id')})"
        )
        return

    # ---- Plan upgrade branch (original behavior) ----
    plan = meta.get('plan')
    subscription_id = session.get('subscription')
    if not plan:
        raise ValueError(
            f"checkout.session.completed missing plan metadata "
            f"(session={session.get('id')}, user_id={user_id!r})"
        )

    update_user_plan(
        user_id=user_id,
        plan=plan,
        stripe_customer_id=customer_id,
        stripe_subscription_id=subscription_id,
    )
    logger.info(f"User {user_id} upgraded to {plan} (customer={customer_id}, sub={subscription_id})")

    # ---- Attribution funnel: if this user was a student on any open share
    # attribution rows, mark them converted. Lets us trace conversions back
    # to the teacher/band/coach who shared the link. See migration 005.
    if plan in ('pro', 'lifetime'):
        try:
            from db import execute
            execute(
                """
                UPDATE share_attribution
                   SET converted_at = NOW(),
                       conversion_plan = %s
                 WHERE student_user_id = %s
                   AND converted_at IS NULL
                """,
                (plan, user_id),
            )
        except Exception as e:
            logger.warning(f"share_attribution: conversion update failed for user {user_id}: {e}")


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


def _cancel_stripe_subscription(subscription_id: str) -> None:
    """Cancel a Stripe subscription immediately so it stops billing.

    Tolerant of an already-cancelled or missing subscription (a refund may
    arrive after the sub was separately cancelled). Never raises for the
    "nothing to cancel" case; re-raises only on unexpected Stripe errors so
    the parent handler can alert + let Stripe retry.
    """
    api_key = os.environ.get('STRIPE_SECRET_KEY', '')
    if api_key:
        stripe.api_key = api_key
    try:
        # stripe-python moved cancel from .delete() to .cancel(); support both.
        canceller = getattr(stripe.Subscription, 'cancel', None) or stripe.Subscription.delete
        canceller(subscription_id)
        logger.info(f"Cancelled Stripe subscription {subscription_id} after refund")
    except stripe.error.InvalidRequestError as e:
        # Already cancelled / no such subscription — nothing to do.
        logger.info(f"Subscription {subscription_id} not active at refund time ({e}); skipping cancel")


def _handle_charge_refunded(charge):
    """A charge was refunded -- if fully refunded, revoke the customer's
    entitlement (revert to free) and cancel any active subscription so it
    stops billing.

    This closes the gap where refunding money did NOT remove paid access:
    Stripe never fires subscription.deleted on a refund, and a refunded
    one-time Lifetime payment otherwise had no revocation path at all.

    Only acts on a FULL refund. Partial refunds (e.g. goodwill credit) keep
    access. Idempotent: a user already on free is a no-op.
    """
    customer_id = charge.get('customer')
    amount = charge.get('amount')
    amount_refunded = charge.get('amount_refunded')
    fully_refunded = bool(charge.get('refunded')) or (
        isinstance(amount, int) and isinstance(amount_refunded, int)
        and amount_refunded >= amount > 0
    )

    if not fully_refunded:
        logger.info(
            f"charge.refunded for {charge.get('id')} is partial "
            f"({amount_refunded}/{amount}) — entitlement unchanged"
        )
        return

    if not customer_id:
        logger.warning(
            f"charge.refunded {charge.get('id')} has no customer — cannot map to a user"
        )
        return

    user = _find_user_by_stripe_customer(customer_id)
    if not user:
        logger.warning(f"charge.refunded: no user for customer {customer_id}")
        return

    if user.plan == 'free':
        logger.debug(
            f"charge.refunded: user {user.id} already on free — no-op (idempotent)"
        )
        return

    prior_plan = user.plan
    if user.stripe_subscription_id:
        _cancel_stripe_subscription(user.stripe_subscription_id)

    update_user_plan(
        user_id=str(user.id),
        plan='free',
        stripe_customer_id=customer_id,
        stripe_subscription_id=None,
    )
    logger.info(
        f"User {user.id} fully refunded (charge={charge.get('id')}, "
        f"was {prior_plan!r}) — entitlement revoked, reverted to free"
    )


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
