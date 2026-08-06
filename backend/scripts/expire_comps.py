#!/usr/bin/env python3
"""Sweep expired comp plans back to free. Daily cron.

Only ever touches rows where users.plan_expires_at IS NOT NULL — that column is set
ONLY by apply_pending_comps.py when a comp has pending_comps.days. Paying customers
never have it set, and anyone who starts paying is skipped by the stripe guard below.

Also texts Jeff a 14-day heads-up so a comp never expires on someone by surprise.
Policy (Jeff, 2026-08-03): musicians get pro for 365 days; lifetime is reserved for
people who already signed up, or who Jeff personally told they'd have lifetime.
"""
import base64, datetime, urllib.parse, urllib.request
import psycopg2

JEFF = "+18034149454"


def envval(key):
    for p in ("/opt/stemscribe/backend/.env", "/opt/stemscribe/.env"):
        try:
            for line in open(p):
                if line.strip().startswith(key + "="):
                    return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
        except FileNotFoundError:
            pass
    return None


def sms(body):
    sid, tok = envval("TWILIO_ACCOUNT_SID"), envval("TWILIO_AUTH_TOKEN")
    frm = envval("TWILIO_FROM_NUMBER") or "+18447915323"
    if not (sid and tok):
        return
    data = urllib.parse.urlencode({"From": frm, "To": JEFF, "Body": body}).encode()
    req = urllib.request.Request(
        "https://api.twilio.com/2010-04-01/Accounts/%s/Messages.json" % sid, data=data)
    req.add_header("Authorization",
                   "Basic " + base64.b64encode(("%s:%s" % (sid, tok)).encode()).decode())
    try:
        urllib.request.urlopen(req, timeout=15)
    except Exception:
        pass


c = psycopg2.connect(envval("DATABASE_URL"))
cur = c.cursor()
now = datetime.datetime.utcnow().isoformat()

# 14-day warning (Jeff only — we never auto-email the user)
cur.execute("""SELECT email, plan, plan_expires_at FROM users
               WHERE plan_expires_at IS NOT NULL
                 AND plan_expires_at BETWEEN now() AND now() + interval '14 days'
                 AND stripe_subscription_id IS NULL""")
soon = cur.fetchall()
for e, pl, exp in soon:
    print("%sZ WARN %s %s expires %s" % (now, e, pl, exp))
if soon:
    sms("StemScriber: %d comp plan(s) expire within 14 days: %s"
        % (len(soon), ", ".join(s[0] for s in soon)))

# Anyone who started paying keeps their plan; just clear the expiry flag.
cur.execute("""UPDATE users SET plan_expires_at=NULL, updated_at=now()
               WHERE plan_expires_at IS NOT NULL AND stripe_subscription_id IS NOT NULL
               RETURNING email""")
for (e,) in cur.fetchall():
    print("%sZ KEPT %s — now a paying subscriber, expiry cleared" % (now, e))

cur.execute("""UPDATE users SET plan='free', plan_expires_at=NULL, updated_at=now()
               WHERE plan_expires_at IS NOT NULL AND plan_expires_at < now()
                 AND plan <> 'free'
               RETURNING email, plan""")
expired = cur.fetchall()
for e, _ in expired:
    print("%sZ EXPIRED %s -> free" % (now, e))
if expired:
    sms("StemScriber: %d comp plan(s) expired back to free: %s"
        % (len(expired), ", ".join(e for e, _ in expired)))
c.commit()
