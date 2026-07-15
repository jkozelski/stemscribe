#!/usr/bin/env python3
"""Safety-net backup: mirror /opt/stemscribe/outputs/ -> R2 jobs/ (UPLOAD-ONLY).

Runs on a timer. Catches anything the per-job async hook missed, and never
deletes from R2 -- so once a job is backed up it stays backed up even after the
local reaper clears it. This is the durable backstop against losing libraries.
"""
import os
from dotenv import load_dotenv
load_dotenv('/opt/stemscribe/.env'); load_dotenv('/opt/stemscribe/backend/.env', override=True)
import boto3

OUTPUTS = '/opt/stemscribe/outputs'
acct = os.environ['R2_ACCOUNT_ID']
s3 = boto3.client('s3', endpoint_url='https://' + acct + '.r2.cloudflarestorage.com',
                  aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
                  aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'])
bucket = os.environ['R2_BUCKET']

# Map existing R2 objects (key -> size) so we only upload new/changed files.
existing = {}
paginator = s3.get_paginator('list_objects_v2')
for page in paginator.paginate(Bucket=bucket, Prefix='jobs/'):
    for o in page.get('Contents', []):
        existing[o['Key']] = o['Size']

uploaded = 0; skipped = 0; failed = 0; up_bytes = 0
for root, _dirs, files in os.walk(OUTPUTS):
    for fn in files:
        local = os.path.join(root, fn)
        rel = os.path.relpath(local, OUTPUTS)        # {job_id}/...
        key = 'jobs/' + rel
        try:
            sz = os.path.getsize(local)
        except OSError:
            continue
        if existing.get(key) == sz:
            skipped += 1
            continue
        try:
            s3.upload_file(local, bucket, key)
            uploaded += 1; up_bytes += sz
        except Exception as e:
            failed += 1
            print('FAIL %s: %s' % (key, e))
print('backup_sync: uploaded %d files (%.1f MB), %d current, %d failed'
      % (uploaded, up_bytes / 1e6, skipped, failed))
