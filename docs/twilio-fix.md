# Twilio SMS Fix — Error 30034 (Carrier Blocked)

**Date:** 2026-04-15
**Status:** Research complete — action needed from Jeff

---

## Diagnosis

### Account Status: CONFIRMED OK
- Account `AC61b4ba568a01c65bf90d98655261161b` is **active** (Full, not trial)
- Balance: **$49.99** — plenty of funds
- Number `+18438748999` is **SMS-capable** (sms: true, mms: true, voice: true)
- Number type: **local** (10-digit long code, area code 843)
- Number is NOT in a Messaging Service (`messaging_service_sid: null`)

### The Problem: No A2P 10DLC Registration

Every single message to +18034149454 has `error_code: 30034` and `status: undelivered`. This is **100% a carrier blocking issue** — the number is not registered for A2P 10DLC.

Since late 2023, US carriers (T-Mobile, AT&T, Verizon) **block** SMS from unregistered local numbers sending application-generated (A2P) messages. Your number sends via the Twilio API → carriers see it as A2P → it's unregistered → blocked.

You're being charged per message ($0.0083/segment) even though nothing delivers. The 5 most recent messages alone cost ~$0.06 for zero delivery.

### Messages Confirmed Blocked (last 5)
| Date | Body (truncated) | Segments | Cost | Status |
|------|-------------------|----------|------|--------|
| Apr 14 03:49 | "Test from Claude - both models training..." | 1 | $0.0083 | undelivered |
| Apr 14 02:13 | "Training update: Formatter epoch 2 done..." | 1 | $0.0083 | undelivered |
| Apr 13 18:08 | "Test from StemScriber..." | 1 | $0.0083 | undelivered |
| Apr 10 21:55 | "All 4 teams done..." (short) | 1 | $0.0083 | undelivered |
| Apr 10 21:52 | "All 4 teams done. Results..." (long) | 3 | $0.0249 | undelivered |

---

## Fix Options (Ranked by Speed & Simplicity)

### Option A: Buy a Toll-Free Number (RECOMMENDED)

**Why:** Toll-Free numbers bypass 10DLC entirely. They have their own simpler verification process, no monthly campaign fees, and work immediately for low-volume use (verification required but may take only days).

**Steps:**
1. Log into [Twilio Console](https://console.twilio.com)
2. Go to **Phone Numbers → Manage → Buy a Number**
3. Search for a Toll-Free number (check "Toll Free" under Type)
4. Buy it (~$2.15/month)
5. Go to **Messaging → Toll-Free → Verification** and submit verification:
   - Business name: Kozelski Enterprises LLC
   - Use case: "Application status notifications sent to the business owner"
   - Sample message: "Training update: Model epoch 5 complete, loss 0.12. -StemScriber"
   - Opt-in: "Business owner is the sole recipient, consent is implicit"
   - Volume: <100 messages/month
6. Update StemScriber code: change the `from` number to the new Toll-Free number
7. While verification is pending (1-7 business days), messages may still be blocked — wait for approval

**Cost:**
| Item | Cost |
|------|------|
| Toll-Free number | $2.15/month |
| Verification | Free |
| Monthly campaign fee | None |
| Per-message | $0.0079 + $0.003 surcharge = ~$0.011/segment |
| **Total monthly (low use)** | **~$2.50** |

**Timeline:** Number available instantly. Verification: 1-7 business days typical.

---

### Option B: Register for A2P 10DLC (Keep Current Number)

**Why:** Keeps your existing 843 number. Required if you want to use a local number for A2P messaging.

**Steps:**
1. Log into [Twilio Console](https://console.twilio.com)
2. Go to **Messaging → Trust Hub → A2P 10DLC**
3. **Register Brand:**
   - Brand type: **Sole Proprietor** (simplest for LLC with one person)
   - Business name: Kozelski Enterprises LLC
   - State: SC
   - Email: jkozelski@gmail.com
   - No EIN required for Sole Proprietor type (SSN-based)
   - Cost: $4 one-time
4. **Register Campaign:**
   - Use case: "Low Volume Mixed" or auto-assigned "Sole Proprietor" campaign
   - Description: "Application status notifications sent to the business owner for a music transcription SaaS"
   - Sample messages: "Training complete: guitar model accuracy 92%. -StemScriber"
   - Opt-in: "Business owner is sole recipient"
   - Cost: $15 one-time + $2/month
5. **Assign Number:**
   - After campaign is approved, assign +18438748999 to the campaign
   - This requires creating a Messaging Service and adding the number to it
6. Wait for carrier approval (1-4 weeks, sometimes faster)

**Cost:**
| Item | Cost |
|------|------|
| Brand registration | $4 (one-time) |
| Campaign registration | $15 (one-time) |
| Campaign monthly fee | $2/month |
| Number | $1.15/month (existing) |
| Per-message | $0.0079 + $0.003-0.005 surcharge |
| **Total setup** | **$19** |
| **Total monthly** | **~$3.50** |

**Timeline:** Brand: instant to 1 day. Campaign: 1-4 weeks (carrier review).

---

### Option C: Alternative Notification Channels (Immediate, Free)

If SMS isn't critical and you just need Claude/StemScriber to notify you:

1. **Email via SendGrid or SMTP** — send to jkozelski@gmail.com (free tier: 100 emails/day)
2. **Slack webhook** — create a #stemscriber-alerts channel, post via webhook URL
3. **Discord webhook** — same idea, even simpler setup
4. **Pushover** ($5 one-time) — push notifications to your phone
5. **ntfy.sh** — free, open-source push notifications, no account needed

Any of these work immediately with zero registration.

---

## Recommendation

**Do Option A (Toll-Free) + Option C (backup channel) in parallel:**

1. Buy a Toll-Free number right now ($2.15) and submit verification
2. Set up email or Pushover as an immediate backup notification channel
3. Once Toll-Free is verified, update the StemScriber `from` number
4. Optionally release the current 843 number to stop paying $1.15/month for a number that can't deliver

**Do NOT bother with 10DLC** unless you specifically need a local number. Toll-Free is cheaper, simpler, and faster for your use case.

---

## What NOT to Do

- Don't keep sending from the current number — you're paying $0.0083/message for zero delivery
- Don't try to "trick" the carrier filter with different message content — the block is on the number registration, not the content
- Don't switch to a different local number without 10DLC — same problem will happen

---

## Code Changes Needed (After Fix)

In your StemScriber backend (wherever Twilio sends are triggered), update the `from` number:

```python
# Old (blocked)
from_number = "+18438748999"

# New (after buying Toll-Free)
from_number = "+1XXXXXXXXXX"  # Your new Toll-Free number
```

Also consider adding delivery status checking so you know when messages fail:

```python
# Add a status callback URL to your send calls
message = client.messages.create(
    body="...",
    from_=TOLL_FREE_NUMBER,
    to="+18034149454",
    status_callback="https://stemscriber.com/api/twilio/status"
)
```

---

## Also Noted

- The current number has SMS webhook pointing to **Voiceflow**: `https://runtime-api.voiceflow.com/v1/twilio/webhooks/...` — this handles inbound SMS. If you switch numbers, you'll need to move this webhook to the new number.
- Voice webhook also points to Voiceflow (different flow). Same applies.
- The number has `bundle_sid: null` — no regulatory bundle attached, which is fine for US numbers.
