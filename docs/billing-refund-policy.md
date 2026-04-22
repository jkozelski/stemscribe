# StemScriber Billing & Refund Policy — Internal Reference

**Last updated:** 2026-03-16

## Overview

This document covers internal processes for handling refunds via Stripe. The customer-facing policy lives in:
- `/frontend/terms.html` (Section 14 — Refund Policy)
- `/frontend/billing-faq.html` (Billing FAQ page)

---

## Refund Categories

| Category | When | Amount | How |
|----------|------|--------|-----|
| **30-day money-back** | Within 30 days of first subscription | Full refund | Stripe Dashboard or API |
| **Annual pro-rated** | Annual plan, past 30-day window | Unused full months (at monthly rate) | Stripe Dashboard or API |
| **Failed processing credit** | Song fails due to system error | Credit restored automatically | Automatic (no Stripe refund needed) |
| **Goodwill/discretionary** | Edge cases, customer service | Varies | Stripe Dashboard |

---

## Issuing Refunds via Stripe Dashboard

1. Log in to [Stripe Dashboard](https://dashboard.stripe.com)
2. Go to **Payments** and find the relevant charge
3. Click the payment, then click **Refund payment**
4. Choose **Full** or **Partial** refund
5. Enter the amount (for pro-rated refunds, see calculation below)
6. Add an internal note with the reason (e.g., "30-day guarantee", "annual pro-rate", "goodwill")
7. Click **Refund**

Refunds take 5-10 business days to appear on the customer's statement.

---

## Issuing Refunds via Stripe API (Python)

Use these examples within the existing billing blueprint context. The `stripe` library and API key are already configured in `backend/billing/routes.py`.

### Full Refund

```python
import stripe
import os

stripe.api_key = os.environ.get('STRIPE_SECRET_KEY')

def issue_full_refund(payment_intent_id: str, reason: str = "requested_by_customer") -> dict:
    """Issue a full refund for a payment.

    Args:
        payment_intent_id: The Stripe PaymentIntent ID (pi_xxx)
        reason: One of 'duplicate', 'fraudulent', 'requested_by_customer'

    Returns:
        Stripe Refund object as dict
    """
    refund = stripe.Refund.create(
        payment_intent=payment_intent_id,
        reason=reason,
    )
    return refund
```

### Partial / Pro-Rated Refund

```python
def issue_prorated_refund(payment_intent_id: str, months_unused: int,
                          monthly_rate: float, reason: str = "requested_by_customer") -> dict:
    """Issue a pro-rated refund for an annual subscription.

    Args:
        payment_intent_id: The Stripe PaymentIntent ID (pi_xxx)
        months_unused: Number of full months remaining
        monthly_rate: Monthly equivalent price (e.g., 4.99 for Premium, 14.99 for Pro)
        reason: Refund reason string

    Returns:
        Stripe Refund object as dict
    """
    refund_amount_cents = int(months_unused * monthly_rate * 100)

    refund = stripe.Refund.create(
        payment_intent=payment_intent_id,
        amount=refund_amount_cents,
        reason=reason,
    )
    return refund
```

### Cancel Subscription + Refund

```python
def cancel_and_refund(subscription_id: str, payment_intent_id: str,
                      refund_amount_cents: int = None) -> dict:
    """Cancel a subscription and optionally issue a refund.

    Args:
        subscription_id: Stripe Subscription ID (sub_xxx)
        payment_intent_id: The PaymentIntent to refund (pi_xxx)
        refund_amount_cents: Amount in cents, or None for full refund
    """
    # Cancel the subscription immediately
    stripe.Subscription.cancel(subscription_id)

    # Issue refund
    refund_kwargs = {
        'payment_intent': payment_intent_id,
        'reason': 'requested_by_customer',
    }
    if refund_amount_cents:
        refund_kwargs['amount'] = refund_amount_cents

    refund = stripe.Refund.create(**refund_kwargs)
    return refund
```

### Look Up a Customer's Payment History

```python
def get_customer_payments(stripe_customer_id: str, limit: int = 10) -> list:
    """Retrieve recent payments for a customer.

    Useful for finding the payment_intent_id needed for refunds.
    """
    charges = stripe.Charge.list(
        customer=stripe_customer_id,
        limit=limit,
    )
    return [
        {
            'charge_id': c.id,
            'payment_intent': c.payment_intent,
            'amount': c.amount / 100,
            'currency': c.currency,
            'created': c.created,
            'status': c.status,
            'refunded': c.refunded,
        }
        for c in charges.data
    ]
```

---

## Pro-Rated Refund Calculation

For annual plans cancelled after the 30-day money-back window:

```
Monthly equivalent rates:
  Premium: $4.99/mo
  Pro:     $14.99/mo

Refund = months_unused * monthly_rate

Example: Pro annual ($119.99), cancelled after 4 months
  Months unused = 8
  Refund = 8 * $14.99 = $119.92
```

Note: Pro-rated refunds use the monthly rate (not the discounted annual rate divided by 12). This is intentional -- the customer got the annual discount for the months they used.

---

## Processing Credit Restoration

When a song fails to process due to a server-side error, the processing credit is restored automatically in the backend. No Stripe refund is needed because no additional charge was made -- credits are part of the subscription allotment.

The credit restoration happens in the processing pipeline's error handler. If a user reports that their credit wasn't restored after a failure, manually check:

1. The job status in the database
2. Whether the failure was server-side vs. user-side (bad file, too long, etc.)
3. Restore the credit manually if needed via the admin tools

---

## Beta Period Policy

**Recommendation: Beta is FREE.**

During the 10-person beta:
- All features unlocked at no cost
- No billing integration active
- Beta invite codes grant full Pro-tier access
- No refund scenarios to handle

**Post-beta transition plan:**
1. Give beta testers 30 days notice before billing starts
2. Offer beta testers a loyalty discount (e.g., 50% off first 3 months)
3. Beta testers who provided significant feedback could get extended free access
4. Grandfather early pricing if rates change later

This approach builds goodwill, gets honest feedback without the "I paid for this" pressure, and avoids billing headaches during a period when bugs are expected.

---

## Tracking Refunds

For now, track refunds via:
1. **Stripe Dashboard** -- all refunds are logged automatically with timestamps and reasons
2. **Email thread** -- keep the support@stemscribe.io conversation as a paper trail
3. **Internal notes** -- add a note to the Stripe payment with the reason category

As volume grows, consider adding a `refunds` table to the database to track:
- user_id
- stripe_refund_id
- amount
- reason category
- requested_at
- processed_at
- processed_by
