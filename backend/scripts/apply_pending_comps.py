#!/usr/bin/env python3
"""Grant pending comp plans the moment a flagged email signs up. Idempotent; cron every 2 min.

pending_comps.days: NULL = permanent comp. An integer = comp expires that many days after
the grant (users.plan_expires_at), and expire_comps.py sweeps it back to free.
Policy (Jeff, 2026-08-03): musicians get pro/365. Lifetime is reserved, not the default.
"""
import psycopg2, datetime
url=None
for p in ["/opt/stemscribe/backend/.env","/opt/stemscribe/.env"]:
    try:
        for line in open(p):
            if line.startswith("DATABASE_URL"):
                url=line.split("=",1)[1].strip().strip('"').strip("'"); break
    except Exception: pass
    if url: break
c=psycopg2.connect(url); cur=c.cursor()
cur.execute("""UPDATE users u
     SET plan=pc.plan,
         plan_expires_at = CASE WHEN pc.days IS NULL THEN NULL
                                ELSE now() + (pc.days || ' days')::interval END,
         updated_at=now()
  FROM pending_comps pc
  WHERE lower(u.email)=lower(pc.email) AND pc.granted_at IS NULL
    AND u.plan IS DISTINCT FROM pc.plan
  RETURNING u.email, pc.plan, pc.days""")
granted=cur.fetchall()
if granted:
    lows=[g[0].lower() for g in granted]
    cur.execute("UPDATE pending_comps SET granted_at=now() WHERE lower(email)=ANY(%s)",(lows,))
    for e,pl,d in granted:
        span="permanent" if d is None else f"{d}d"
        print(f"{datetime.datetime.utcnow().isoformat()}Z granted {pl} ({span}) -> {e}")
c.commit()
