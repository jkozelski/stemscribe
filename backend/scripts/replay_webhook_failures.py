"""
Replay failed Stripe webhooks after the root cause has been fixed.

Usage on prod (from /opt/stemscribe/backend):
    ../venv311/bin/python scripts/replay_webhook_failures.py            # list pending
    ../venv311/bin/python scripts/replay_webhook_failures.py --replay   # re-run all unresolved
    ../venv311/bin/python scripts/replay_webhook_failures.py --replay --id 42
    ../venv311/bin/python scripts/replay_webhook_failures.py --replay --type checkout.session.completed

Each successful replay marks the row resolved_at=now(). Failures stay open and
print the new error so you can fix and retry.
"""
import argparse
import json
import sys
import os
import traceback

# Make sibling backend modules importable when run from the scripts/ subdir
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv('/opt/stemscribe/.env')

from db import query_all, execute  # noqa: E402
from billing import webhooks as wh  # noqa: E402


HANDLERS = {
    'checkout.session.completed':     wh._handle_checkout_completed,
    'customer.subscription.updated':  wh._handle_subscription_updated,
    'customer.subscription.deleted':  wh._handle_subscription_deleted,
    'invoice.payment_failed':         wh._handle_payment_failed,
    'invoice.paid':                   wh._handle_payment_succeeded,
}


def list_pending(filter_type=None, filter_id=None):
    sql = "SELECT id, event_id, event_type, customer_id, user_id, error_message, created_at FROM webhook_failures WHERE resolved_at IS NULL"
    params = []
    if filter_type:
        sql += " AND event_type = %s"
        params.append(filter_type)
    if filter_id:
        sql += " AND id = %s"
        params.append(filter_id)
    sql += " ORDER BY created_at"
    return query_all(sql, tuple(params))


def replay_one(row):
    handler = HANDLERS.get(row['event_type'])
    if not handler:
        return False, f"no handler registered for {row['event_type']}"
    # Pull the payload back out of the DB
    payload_row = query_all("SELECT payload FROM webhook_failures WHERE id = %s", (row['id'],))
    if not payload_row:
        return False, "row vanished"
    obj = payload_row[0]['payload']
    if isinstance(obj, str):
        obj = json.loads(obj)
    try:
        handler(obj)
        execute("UPDATE webhook_failures SET resolved_at = now() WHERE id = %s", (row['id'],))
        return True, "resolved"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replay', action='store_true', help='actually re-run handlers (default: list only)')
    ap.add_argument('--id', type=int, default=None)
    ap.add_argument('--type', default=None)
    args = ap.parse_args()

    rows = list_pending(filter_type=args.type, filter_id=args.id)
    if not rows:
        print("No unresolved webhook failures.")
        return

    print(f"{len(rows)} unresolved failure(s):")
    for r in rows:
        when = r['created_at'].isoformat()[:19] if r['created_at'] else '?'
        print(f"  #{r['id']:3}  {when}  {r['event_type']:36}  cust={r['customer_id'] or '-':<22}  {(r['error_message'] or '')[:70]}")

    if not args.replay:
        print("\n(Add --replay to actually re-run them.)")
        return

    print("\nReplaying...")
    ok = 0
    fail = 0
    for r in rows:
        success, msg = replay_one(r)
        marker = "✓" if success else "✗"
        print(f"  {marker} #{r['id']:3}  {r['event_type']:36}  →  {msg}")
        if success:
            ok += 1
        else:
            fail += 1
    print(f"\n{ok} resolved, {fail} still failing.")


if __name__ == '__main__':
    main()
