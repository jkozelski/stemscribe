#!/usr/bin/env python3
"""Pull every genuine typed message Jeff has sent, across all session transcripts.

Why: notes go stale and compaction drops things. Jeff's own words are the
highest-signal, lowest-volume slice of 1.3 GB of transcripts — every decision,
preference and correction he has ever given lives there. Tool results also
arrive as role=user records, so those are filtered out.

Streams line by line; never loads a transcript into memory.
"""
import glob
import json
import os
import re

D = os.path.expanduser("~/.claude/projects/-Users-jeffkozelski")
OUT = "/private/tmp/claude-501/-Users-jeffkozelski/6389173b-d548-477f-b7d3-67dd5088abd0/scratchpad/all_user_messages.md"

# Records that are machine-generated even though role=user
NOISE = re.compile(
    r"^\s*(<system-reminder>|<task-notification>|<command-name>|<local-command|"
    r"Caveat: The messages below|\[Request interrupted|<bash-input>|<bash-stdout>)",
    re.I,
)

msgs = []
for path in glob.glob(f"{D}/*.jsonl"):
    sid = os.path.basename(path)[:8]
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or '"type":"user"' not in line.replace(" ", ""):
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("type") != "user" or d.get("isSidechain"):
                continue
            m = d.get("message") or {}
            content = m.get("content")
            # genuine typed input arrives as a plain string; tool results are lists
            if not isinstance(content, str):
                continue
            text = content.strip()
            if not text or NOISE.match(text):
                continue
            msgs.append((d.get("timestamp") or "", sid, text))

msgs.sort(key=lambda x: x[0])

# de-dupe exact repeats (same text within the same minute)
seen, uniq = set(), []
for ts, sid, text in msgs:
    key = (ts[:16], text)
    if key in seen:
        continue
    seen.add(key)
    uniq.append((ts, sid, text))

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write(f"# Every message Jeff typed, all sessions\n\n{len(uniq)} messages\n\n")
    day = None
    for ts, sid, text in uniq:
        d = ts[:10]
        if d != day:
            fh.write(f"\n\n## {d}\n\n")
            day = d
        fh.write(f"- `{ts[11:16]}` [{sid}] {text}\n")

print(f"  transcripts scanned : {len(glob.glob(f'{D}/*.jsonl'))}")
print(f"  raw user messages   : {len(msgs)}")
print(f"  after de-dupe       : {len(uniq)}")
print(f"  written to          : {OUT}")
print(f"  size                : {os.path.getsize(OUT)/1024:.0f} KB")
if uniq:
    print(f"  date range          : {uniq[0][0][:10]} .. {uniq[-1][0][:10]}")
