#!/usr/bin/env python3
"""
Import Jeff's cleaned OnSong chart library into the chart_library table.

Idempotent: upserts on (user_id, source, source_file) — re-running never
duplicates, it refreshes rows in place.

Usage:
    cd /opt/stemscribe/backend
    /opt/stemscribe/venv311/bin/python scripts/import_chart_library.py \
        --owner-email jkozelski@gmail.com \
        --root /opt/stemscribe/chart_import

Expects <root>/cleaned, <root>/top200_charts, <root>/top300_charts full of
.txt files with a header block:

    TITLE: ...
    ARTIST: ...
    KEY: ...
    <blank line>
    <chart body>

Charts containing a "verify" note or the ⚠ marker are flagged_for_review.
The note line stays in the body — it's part of the chart.
"""

import os
import re
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv('/opt/stemscribe/.env')
load_dotenv('/opt/stemscribe/backend/.env', override=True)

from db import query_one, get_db  # noqa: E402

SOURCES = {
    'cleaned': 'onsong',
    'top200_charts': 'top200',
    'top300_charts': 'top300',
}

HEADER_RE = re.compile(r'^(TITLE|ARTIST|KEY):\s*(.*)$')
FLAG_RE = re.compile(r'verify', re.IGNORECASE)


def parse_chart(text, fallback_title):
    """Split header block from body. Returns (title, artist, key, body)."""
    lines = text.split('\n')
    meta = {}
    body_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            body_start = i + 1
            break
        m = HEADER_RE.match(stripped)
        if m:
            meta[m.group(1)] = m.group(2).strip()
            body_start = i + 1
        else:
            # Non-header line before a blank — body starts here
            body_start = i
            break
    body = '\n'.join(lines[body_start:]).strip('\n')
    title = meta.get('TITLE') or fallback_title
    artist = meta.get('ARTIST') or None
    key = meta.get('KEY') or None
    return title, artist, key, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--owner-email', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    owner = query_one('SELECT id, email FROM users WHERE email = %s',
                      (args.owner_email.lower(),))
    if not owner:
        print(f'ERROR: no user with email {args.owner_email}', file=sys.stderr)
        sys.exit(1)
    user_id = str(owner['id'])
    print(f'Owner: {owner["email"]} ({user_id})')

    root = Path(args.root)
    totals = {}
    flagged_total = 0
    skipped = []

    for folder, source in SOURCES.items():
        d = root / folder
        if not d.is_dir():
            print(f'WARNING: missing folder {d}', file=sys.stderr)
            continue
        files = sorted(d.glob('*.txt'))
        inserted = 0
        updated = 0
        for fp in files:
            text = fp.read_text(encoding='utf-8', errors='replace')
            if not text.strip():
                skipped.append(f'{folder}/{fp.name} (empty)')
                continue
            fallback = re.sub(r'^\d+-', '', fp.stem).replace('-', ' ')
            title, artist, key, body = parse_chart(text, fallback)
            if not body:
                # Header-only chart: the KEY line carries the whole chart
                # (e.g. "Em – D6add9/F# (two chords, whole song)")
                if key:
                    body = key
                else:
                    skipped.append(f'{folder}/{fp.name} (no body)')
                    continue
            flagged = bool('⚠' in text or FLAG_RE.search(text))
            if flagged:
                flagged_total += 1
            if args.dry_run:
                inserted += 1
                continue
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO chart_library
                            (user_id, title, artist, song_key, body, source,
                             source_file, flagged_for_review)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (user_id, source, source_file) DO UPDATE SET
                            title = EXCLUDED.title,
                            artist = EXCLUDED.artist,
                            song_key = EXCLUDED.song_key,
                            body = EXCLUDED.body,
                            flagged_for_review = EXCLUDED.flagged_for_review,
                            updated_at = NOW()
                        RETURNING (xmax = 0) AS is_insert
                        """,
                        (user_id, title, artist, key, body, source,
                         fp.name, flagged),
                    )
                    if cur.fetchone()[0]:
                        inserted += 1
                    else:
                        updated += 1
        totals[source] = (len(files), inserted, updated)

    print('\n=== Import summary ===')
    grand = 0
    for source, (nfiles, ins, upd) in totals.items():
        print(f'{source:8s} files={nfiles:5d} inserted={ins:5d} updated={upd:5d}')
        grand += ins + upd
    print(f'total imported/refreshed: {grand}')
    print(f'flagged_for_review: {flagged_total}')
    if skipped:
        print(f'skipped ({len(skipped)}):')
        for s in skipped:
            print(f'  - {s}')

    if not args.dry_run:
        row = query_one(
            'SELECT count(*) AS c FROM chart_library WHERE user_id = %s',
            (user_id,))
        print(f'rows in chart_library for owner: {row["c"]}')


if __name__ == '__main__':
    main()
