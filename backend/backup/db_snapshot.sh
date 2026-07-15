#!/bin/bash
# Nightly Supabase DB snapshot → Cloudflare R2.
# Run via systemd timer (stemscribe-db-backup.timer) once daily.
#
# Rotation: 30 daily snapshots kept (R2 lifecycle rule, not implemented in
# this script — R2 doesn't natively support TTL yet; manual cleanup TODO).
#
# Exit non-zero on any failure so systemd marks the unit failed and you
# get a journal entry. No customer money is at stake from a single skipped
# snapshot; the alert is so we notice if backups silently stop.

set -euo pipefail

# Load env the same way app.py does (root first, backend overrides)
set -a
source /opt/stemscribe/.env
[ -f /opt/stemscribe/backend/.env ] && source /opt/stemscribe/backend/.env
set +a

: "${DATABASE_URL:?DATABASE_URL not set}"
: "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID not set}"
: "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID not set}"
: "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY not set}"
: "${R2_BUCKET:=stemscriber-backups}"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
TMP_DIR=$(mktemp -d -t ss-dbsnap-XXXXXX)
trap 'rm -rf "$TMP_DIR"' EXIT

DUMP_FILE="$TMP_DIR/stemscribe-$STAMP.sql.gz"

echo "[$(date -u +%FT%TZ)] pg_dump starting -> $DUMP_FILE"

# Custom format would be faster for restore but plain SQL is more portable
# and the size delta is small for our DB (<100MB even at scale).
pg_dump --no-owner --no-acl --clean --if-exists "$DATABASE_URL" | gzip -9 > "$DUMP_FILE"

DUMP_BYTES=$(stat -c %s "$DUMP_FILE" 2>/dev/null || stat -f %z "$DUMP_FILE")
echo "[$(date -u +%FT%TZ)] dump complete: ${DUMP_BYTES} bytes"

# Use python r2_client to upload (already wired with retries + correct endpoint)
/opt/stemscribe/venv311/bin/python <<PYEOF
import sys
sys.path.insert(0, '/opt/stemscribe/backend')
from backup.r2_client import upload_file, r2_enabled
if not r2_enabled():
    print("R2 not configured", file=sys.stderr); sys.exit(1)
key = f"db-snapshots/stemscribe-${STAMP}.sql.gz"
ok = upload_file("$DUMP_FILE", key, content_type="application/gzip")
if not ok:
    print(f"upload failed: {key}", file=sys.stderr); sys.exit(2)
print(f"uploaded: {key}")
PYEOF

echo "[$(date -u +%FT%TZ)] snapshot complete"
