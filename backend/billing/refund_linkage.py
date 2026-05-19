"""
Deterministic charge <-> customer <-> job-count linkage for refund evaluation.

THIS MODULE NEVER MOVES MONEY.

It is a READ-ONLY data module. Given a Stripe charge id OR a Stripe customer
id, it assembles a single verified "refund-evaluation packet" so a human (or
a deterministic non-LLM script outside this module) can judge a refund request
against *real data* instead of the customer's prose claim.

Why this exists
----------------
The support-agent red-team flagged the exact gap this closes: the original
auto-refund design had no charge<->job linkage, so the agent would fall back
to whatever the customer claimed ("I never used it"). Tier-1 silent
auto-process of policy-clear refunds is explicitly blocked until this linkage
exists (see stemscriber-support-agent-spec-2026-05-16.md, NOTIFICATION &
ESCALATION ROUTING). Job count here comes from the usage DB via the existing
`get_monthly_usage` helper -- never from a customer-supplied number.

Hard constraints (enforced by construction)
-------------------------------------------
- Stripe access is READ-ONLY: only `.retrieve` / `.list` (GET) calls. This
  module imports no Stripe write surface and calls none.
- It performs NO refund, cancel, subscription edit, or any write of any kind.
- It is NOT wired into any auto-refund path. It only returns data.
- The app loads `sk_live`; this module uses that key strictly for reads.

Usage
-----
    from billing.refund_linkage import build_refund_evaluation_packet

    packet = build_refund_evaluation_packet(charge_id="ch_123")
    # or
    packet = build_refund_evaluation_packet(customer_id="cus_123")

`packet.ok` is True when the linkage resolved cleanly. `packet.warnings`
collects every non-fatal ambiguity (e.g. customer not matched to a local
user, multiple charges) so the human reviewer sees exactly what is and isn't
verified. The packet is a plain dataclass with `.to_dict()` for serialization
into an evidence packet / digest.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import stripe

from db import query_one
from auth.models import get_monthly_usage

logger = logging.getLogger(__name__)


# ---- Stripe error classes (test-environment safe) --------------------------
# In production `stripe.error.*` are real Exception subclasses. The test suite
# replaces `stripe` with a MagicMock whose error attributes are NOT real
# exception classes (conftest.py only makes SignatureVerificationError real),
# which would make `except _InvalidRequestError` a TypeError. We
# resolve the classes defensively: use them if they are real exception
# classes, otherwise fall back to the base `Exception` so the handlers still
# behave (and tests can raise a plain Exception to exercise error paths).

def _resolve_exc(name: str):
    cls = getattr(getattr(stripe, 'error', None), name, None)
    if isinstance(cls, type) and issubclass(cls, BaseException):
        return cls
    return Exception


_StripeError = _resolve_exc('StripeError')
_InvalidRequestError = _resolve_exc('InvalidRequestError')


# ---- Read-only Stripe key handling -----------------------------------------

def _ensure_stripe_key() -> bool:
    """Set stripe.api_key from env for READ-ONLY use. Returns False if absent.

    We deliberately do not raise on a missing key: the packet should still be
    returned (with a warning) so the human reviewer is not blocked, they just
    won't have Stripe-side facts.
    """
    key = os.environ.get('STRIPE_SECRET_KEY', '')
    if not key:
        logger.warning("refund_linkage: STRIPE_SECRET_KEY not set -- "
                        "Stripe-side facts will be unavailable")
        return False
    stripe.api_key = key
    return True


# ---- Result type -----------------------------------------------------------

@dataclass
class RefundEvaluationPacket:
    """A verified, read-only snapshot for judging a single refund request.

    This object is *evidence*, not a decision. It deliberately does NOT
    contain a "should_refund" boolean -- that judgment belongs to a human or
    a separate deterministic policy script, never to this data module and
    never to an LLM.
    """

    # Inputs / linkage
    stripe_customer_id: Optional[str] = None
    user_id: Optional[str] = None
    email: Optional[str] = None
    plan: Optional[str] = None

    # signup (local account) OR purchase (Stripe charge) date, ISO-8601 UTC
    signup_or_purchase_date: Optional[str] = None

    # Job count for the CURRENT usage period, from the usage DB (NOT the
    # customer's claim). None means "could not be determined" (e.g. no local
    # user) -- the reviewer must treat None as unverified, not as zero.
    job_count_current_period: Optional[int] = None

    prior_refund: bool = False
    open_dispute: bool = False

    # Charge facts (only populated when a specific charge was resolved)
    charge_id: Optional[str] = None
    charge_amount: Optional[int] = None      # minor units (cents), as Stripe reports
    currency: Optional[str] = None
    livemode: Optional[bool] = None

    # Verification surface
    ok: bool = False
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---- Internal helpers ------------------------------------------------------

def _find_user_row_by_customer(customer_id: str) -> Optional[dict]:
    """Local DB lookup: user by Stripe customer id. Read-only."""
    if not customer_id:
        return None
    return query_one(
        "SELECT id, email, plan, created_at FROM users "
        "WHERE stripe_customer_id = %s",
        (customer_id,),
    )


def _find_user_row_by_email(email: str) -> Optional[dict]:
    """Fallback local lookup by email (Stripe customer may predate the
    stripe_customer_id write). Read-only."""
    if not email:
        return None
    return query_one(
        "SELECT id, email, plan, created_at FROM users WHERE email = %s",
        (email.lower().strip(),),
    )


def _iso(ts) -> Optional[str]:
    """Stripe unix ts (int) or datetime -> ISO-8601 UTC string."""
    if ts is None:
        return None
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts.isoformat()
        # already a string
        return str(ts)
    except (ValueError, OSError, OverflowError):
        return None


def _has_prior_refund(customer_id: str, exclude_charge_id: Optional[str]) -> bool:
    """READ-ONLY: any prior refund on this customer's charges.

    Uses Stripe Refund.list filtered by charge where possible, and falls back
    to scanning the customer's charges for `amount_refunded > 0`. Never writes.
    """
    try:
        charges = stripe.Charge.list(customer=customer_id, limit=100)
    except _StripeError as e:
        logger.warning(f"refund_linkage: Charge.list failed for {customer_id}: {e}")
        # Unknown -> conservatively report True so a human looks (safer than
        # falsely asserting "no prior refund" and auto-clearing).
        return True

    for ch in getattr(charges, 'data', []) or []:
        cid = ch.get('id') if isinstance(ch, dict) else getattr(ch, 'id', None)
        amt_refunded = (ch.get('amount_refunded') if isinstance(ch, dict)
                        else getattr(ch, 'amount_refunded', 0)) or 0
        refunded_flag = (ch.get('refunded') if isinstance(ch, dict)
                         else getattr(ch, 'refunded', False))
        if cid == exclude_charge_id:
            # The charge under evaluation: a refund on IT is the request
            # itself, not a "prior" refund. Skip.
            continue
        if refunded_flag or amt_refunded > 0:
            return True
    return False


def _has_open_dispute(customer_id: str) -> bool:
    """READ-ONLY: any open/unresolved dispute on this customer. Never writes."""
    try:
        disputes = stripe.Dispute.list(limit=100)
    except _StripeError as e:
        logger.warning(f"refund_linkage: Dispute.list failed: {e}")
        return True  # unknown -> flag for human review

    # Dispute.list is not filterable by customer in all API versions, so we
    # match on the customer of each dispute's charge defensively.
    open_states = {'warning_needs_response', 'warning_under_review',
                   'needs_response', 'under_review'}
    for d in getattr(disputes, 'data', []) or []:
        status = d.get('status') if isinstance(d, dict) else getattr(d, 'status', None)
        d_charge = d.get('charge') if isinstance(d, dict) else getattr(d, 'charge', None)
        if status not in open_states:
            continue
        # Resolve the charge's customer to confirm it's this customer.
        try:
            ch = stripe.Charge.retrieve(d_charge) if d_charge else None
        except _StripeError:
            ch = None
        ch_cust = None
        if ch is not None:
            ch_cust = ch.get('customer') if isinstance(ch, dict) else getattr(ch, 'customer', None)
        if ch_cust == customer_id:
            return True
    return False


# ---- Public API ------------------------------------------------------------

def build_refund_evaluation_packet(
    charge_id: Optional[str] = None,
    customer_id: Optional[str] = None,
) -> RefundEvaluationPacket:
    """Assemble a verified refund-evaluation packet (READ-ONLY).

    Provide exactly one of `charge_id` or `customer_id`. If a charge id is
    given, the charge's customer is resolved from Stripe and full charge
    facts (amount/currency/livemode/purchase date) are included.

    This function performs ONLY Stripe GET/list calls and DB reads. It never
    refunds, cancels, edits, or writes anything. The returned packet is
    evidence for a human / deterministic policy script -- it intentionally
    carries no refund decision.
    """
    packet = RefundEvaluationPacket()

    if not charge_id and not customer_id:
        packet.warnings.append("No charge_id or customer_id provided.")
        return packet
    if charge_id and customer_id:
        packet.warnings.append(
            "Both charge_id and customer_id provided; using charge_id as the "
            "anchor and ignoring the passed customer_id."
        )

    have_key = _ensure_stripe_key()

    resolved_customer_id = customer_id

    # ---- Step 1: resolve charge facts + its customer (if a charge given) ----
    if charge_id:
        if not have_key:
            packet.warnings.append(
                "Stripe key unavailable; cannot resolve charge facts.")
            return packet
        try:
            charge = stripe.Charge.retrieve(charge_id)
        except _InvalidRequestError as e:
            packet.warnings.append(f"Charge {charge_id} not found in Stripe: {e}")
            return packet
        except _StripeError as e:
            packet.warnings.append(f"Stripe error retrieving charge {charge_id}: {e}")
            return packet

        packet.charge_id = charge.get('id') if isinstance(charge, dict) else getattr(charge, 'id', None)
        packet.charge_amount = (charge.get('amount') if isinstance(charge, dict)
                                else getattr(charge, 'amount', None))
        packet.currency = (charge.get('currency') if isinstance(charge, dict)
                           else getattr(charge, 'currency', None))
        packet.livemode = (charge.get('livemode') if isinstance(charge, dict)
                            else getattr(charge, 'livemode', None))
        created = (charge.get('created') if isinstance(charge, dict)
                   else getattr(charge, 'created', None))
        packet.signup_or_purchase_date = _iso(created)

        resolved_customer_id = (charge.get('customer') if isinstance(charge, dict)
                                else getattr(charge, 'customer', None))
        if not resolved_customer_id:
            packet.warnings.append(
                f"Charge {charge_id} has no associated Stripe customer; "
                "cannot link to a local account.")
            # Still return charge facts -- a human can act on them.
            return packet

    packet.stripe_customer_id = resolved_customer_id

    # ---- Step 2: pull customer email from Stripe (read-only) ---------------
    customer_email = None
    if have_key and resolved_customer_id:
        try:
            cust = stripe.Customer.retrieve(resolved_customer_id)
            if getattr(cust, 'deleted', False) or (
                isinstance(cust, dict) and cust.get('deleted')
            ):
                packet.warnings.append(
                    f"Stripe customer {resolved_customer_id} is deleted.")
            else:
                customer_email = (cust.get('email') if isinstance(cust, dict)
                                  else getattr(cust, 'email', None))
        except _StripeError as e:
            packet.warnings.append(
                f"Stripe error retrieving customer {resolved_customer_id}: {e}")
    elif not have_key:
        packet.warnings.append(
            "Stripe key unavailable; customer email / refund / dispute facts "
            "could not be verified.")

    # ---- Step 3: link to the local user (DB read-only) ---------------------
    user_row = _find_user_row_by_customer(resolved_customer_id)
    if not user_row and customer_email:
        user_row = _find_user_row_by_email(customer_email)
        if user_row:
            packet.warnings.append(
                "Local user matched by EMAIL, not stripe_customer_id "
                "(customer id not yet written to the user row). Verify the "
                "match before relying on the job count.")

    if user_row:
        packet.user_id = str(user_row.get('id'))
        packet.email = user_row.get('email') or customer_email
        packet.plan = user_row.get('plan')
        # Prefer the local account creation date as signup date when we did
        # NOT resolve a specific charge (customer-id-only path). When a charge
        # was resolved, keep the purchase date (more relevant to the refund).
        if not charge_id:
            packet.signup_or_purchase_date = _iso(user_row.get('created_at'))

        # ---- Step 4: JOB COUNT FROM THE DB, never the customer's claim ----
        try:
            packet.job_count_current_period = get_monthly_usage(packet.user_id)
        except Exception as e:
            logger.warning(f"refund_linkage: get_monthly_usage failed for "
                           f"{packet.user_id}: {e}")
            packet.warnings.append(
                f"Job count lookup failed ({e}); treat as UNVERIFIED, not zero.")
    else:
        packet.email = customer_email
        packet.warnings.append(
            "No local user linked to this Stripe customer. Job count is "
            "UNVERIFIED -- do NOT silently auto-process; route to human review.")

    # ---- Step 5: prior refund + open dispute (Stripe read-only) ------------
    if have_key and resolved_customer_id:
        packet.prior_refund = _has_prior_refund(resolved_customer_id, packet.charge_id)
        packet.open_dispute = _has_open_dispute(resolved_customer_id)
    else:
        packet.warnings.append(
            "prior_refund / open_dispute not verified (no Stripe key or "
            "customer). Defaulting both to False -- treat as UNVERIFIED.")

    # ---- ok flag: linkage is clean enough for data-driven judgment ---------
    packet.ok = bool(
        packet.user_id
        and packet.job_count_current_period is not None
        and have_key
    )
    return packet
