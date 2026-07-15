#!/usr/bin/env python3
"""Push a DB dump to Cloudflare R2 under db-snapshots/ and prune >30d.
Called by db_backup.sh after a verified local dump. Reads R2_* from env files."""
import sys, os
from datetime import datetime, timezone, timedelta

def load_env(path):
    d={}
    try:
        for line in open(path):
            line=line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            k,v=line.split("=",1); d[k]=v.strip().strip("\"").strip("\x27")
    except FileNotFoundError: pass
    return d

env={}
for p in ("/opt/stemscribe/.env","/opt/stemscribe/backend/.env"):
    env.update(load_env(p))   # backend/.env has the R2_* keys

import boto3
acct=env["R2_ACCOUNT_ID"]; bucket=env["R2_BUCKET"]
s3=boto3.client("s3",
    endpoint_url=f"https://{acct}.r2.cloudflarestorage.com",
    aws_access_key_id=env["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"],
    region_name="auto")

path=sys.argv[1]
key="db-snapshots/"+os.path.basename(path)
s3.upload_file(path, bucket, key)
print(f"{datetime.now(timezone.utc).isoformat()} R2 upload ok: {key} ({os.path.getsize(path)} B) -> {bucket}")

# retention: keep 30 days in R2
cutoff=datetime.now(timezone.utc)-timedelta(days=30); deleted=0
for pg in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="db-snapshots/"):
    for obj in pg.get("Contents",[]):
        if obj["LastModified"]<cutoff:
            s3.delete_object(Bucket=bucket, Key=obj["Key"]); deleted+=1
if deleted: print(f"  R2 retention: pruned {deleted} snapshot(s) >30d")
