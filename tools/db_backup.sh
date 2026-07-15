#!/bin/bash
# Nightly Supabase Postgres backup (the DB is NOT on this box — Supabase free
# tier has no backups, so these dumps are the only safety net). Keep 14 days.
set -euo pipefail
URL=$(grep "^DATABASE_URL=" /opt/stemscribe/.env | cut -d= -f2- | tr -d "\"'")
OUT=/opt/stemscribe/backups/nightly/stemscribe-$(date +%Y%m%d-%H%M).sql.gz
/usr/lib/postgresql/17/bin/pg_dump --no-owner --no-privileges "$URL" | gzip > "$OUT.tmp"
# hard verification: gzip integrity + a real dump is never tiny
gunzip -t "$OUT.tmp"
SIZE=$(stat -c%s "$OUT.tmp")
if [ "$SIZE" -lt 20000 ]; then
    echo "$(date -Is) BACKUP FAILED: dump only ${SIZE}B" >> /opt/stemscribe/backups/nightly/backup.log
    rm -f "$OUT.tmp"; exit 1
fi
mv "$OUT.tmp" "$OUT"
find /opt/stemscribe/backups/nightly -name "*.sql.gz" -mtime +14 -delete
echo "$(date -Is) backup ok: $OUT ($(du -h "$OUT" | cut -f1))" >> /opt/stemscribe/backups/nightly/backup.log

# ---- offsite copy to Cloudflare R2 (3-2-1 offsite leg; added 2026-07-14) ----
# R2 failure is logged loudly but does NOT fail the run (verified local dump already exists).
if /opt/stemscribe/venv311/bin/python3 /opt/stemscribe/tools/db_backup_r2push.py "$OUT" >> /opt/stemscribe/backups/nightly/backup.log 2>&1; then
    :
else
    echo "$(date -Is) R2 UPLOAD FAILED for $OUT" >> /opt/stemscribe/backups/nightly/backup.log
fi
