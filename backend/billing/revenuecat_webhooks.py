"""
RevenueCat webhook handler — Apple In-App Purchase (StoreKit) lifecycle events.

Endpoint:
    POST /webhooks/revenuecat

RevenueCat fires these whenever an Apple purchase happens in the iOS app. The
purchase was made with appUserID = the StemScriber user id (the JWT `sub`), so
`event.app_user_id` maps directly to users.id — which is how Apple purchases and
Stripe purchases land on the same account/plan (unified entitlements).

Security:
    RevenueCat sends a fixed Authorization header value that you configure in the
    RevenueCat dashboard. We compare it (constant-time) against
    REVENUECAT_WEBHOOK_AUTH. No shared-secret => reject (so nobody can forge a
    "give me Pro" call).
"""

import os
import hmac
import logging

from flask import Blueprint, request, jsonify

from auth.models import get_user_by_id, set_user_plan

logger = logging.getLogger(__name__)

rc_webhooks_bp = Blueprint('revenuecat_webhooks', __name__, url_prefix='/webhooks')

# RevenueCat product id -> StemScriber plan
PRODUCT_PLAN = {
    'stemscriber_pro_monthly': 'pro',
    'stemscriber_pro_annual': 'pro',
    'stemscriber_pro_monthly_v2': 'pro',   # live Apple IAP ids (old ones burned by Apple)
    'stemscriber_pro_annual_v2': 'pro',
    'stemscriber_lifetime': 'lifetime',
}

# Events that GRANT/refresh access (set the paid plan)
GRANT_EVENTS = {
    'INITIAL_PURCHASE', 'RENEWAL', 'UNCANCELLATION',
    'PRODUCT_CHANGE', 'NON_RENEWING_PURCHASE',
}
# Events that REVOKE access (back to free)
REVOKE_EVENTS = {'EXPIRATION', 'REFUND'}
# Everything else (CANCELLATION = auto-renew off but still active until EXPIRATION,
# BILLING_ISSUE, SUBSCRIPTION_PAUSED, TRANSFER, TEST, SUBSCRIBER_ALIAS) is logged
# and left alone — we do not downgrade until access actually lapses.


def _authorized(req) -> bool:
    expected = os.environ.get('REVENUECAT_WEBHOOK_AUTH', '')
    if not expected:
        logger.error("RevenueCat webhook: REVENUECAT_WEBHOOK_AUTH not configured")
        return False
    provided = req.headers.get('Authorization', '')
    return hmac.compare_digest(provided, expected)


def _plan_for(event) -> str:
    """Resolve the plan from product_id, falling back to the Pro entitlement."""
    pid = event.get('product_id') or ''
    if pid in PRODUCT_PLAN:
        return PRODUCT_PLAN[pid]
    # defensive substring match in case the store appends a suffix
    for key, plan in PRODUCT_PLAN.items():
        if key in pid:
            return plan
    # last resort: any active Pro entitlement -> pro
    ents = event.get('entitlement_ids') or []
    if 'Stemscriber Pro' in ents:
        return 'pro'
    return ''


@rc_webhooks_bp.route('/revenuecat', methods=['POST'])
def revenuecat_webhook():
    if not _authorized(request):
        logger.warning("RevenueCat webhook: unauthorized")
        return jsonify({'error': 'unauthorized'}), 401

    body = request.get_json(silent=True) or {}
    event = body.get('event') or {}
    etype = event.get('type', '')
    app_user_id = event.get('app_user_id') or ''

    logger.info(f"RevenueCat webhook: type={etype} app_user_id={app_user_id} product={event.get('product_id')}")

    # Ignore RevenueCat test pings and anonymous/unattributed purchases gracefully.
    if etype == 'TEST':
        return jsonify({'status': 'ok', 'note': 'test event'}), 200
    if not app_user_id or app_user_id.startswith('$RCAnonymousID'):
        logger.warning(f"RevenueCat webhook: no resolvable app_user_id ({app_user_id}) — skipping")
        return jsonify({'status': 'ok'}), 200

    try:
        user = get_user_by_id(app_user_id)
        if not user:
            logger.warning(f"RevenueCat webhook: user {app_user_id} not found")
            return jsonify({'status': 'ok'}), 200

        if etype in GRANT_EVENTS:
            plan = _plan_for(event)
            if not plan:
                logger.warning(f"RevenueCat {etype}: could not resolve plan for product {event.get('product_id')}")
            else:
                set_user_plan(str(user.id), plan)
                logger.info(f"RevenueCat {etype}: user {user.id} -> {plan}")
        elif etype in REVOKE_EVENTS:
            cur = getattr(user, 'plan', None)
            # comped/beta accounts are manual — an Apple event must never touch them
            if cur == 'beta':
                logger.info(f"RevenueCat {etype}: user {user.id} is comped (beta) — NOT downgrading")
            # Lifetime is a NON-CONSUMABLE: subscriptions EXPIRE, a one-time purchase
            # does not. Only a real REFUND claws back lifetime. An EXPIRATION must NOT.
            elif cur == 'lifetime' and etype != 'REFUND':
                logger.info(f"RevenueCat {etype}: user {user.id} is lifetime — NOT downgrading on {etype} (only REFUND revokes)")
            else:
                set_user_plan(str(user.id), 'free')
                logger.info(f"RevenueCat {etype}: user {user.id} -> free")
        else:
            logger.info(f"RevenueCat {etype}: no plan change for user {user.id}")
    except Exception as e:
        import traceback
        logger.error(f"RevenueCat webhook error on {etype}: {e}\n{traceback.format_exc()}")
        # 200 so RevenueCat doesn't hammer retries; we have the log.

    return jsonify({'status': 'ok'}), 200
