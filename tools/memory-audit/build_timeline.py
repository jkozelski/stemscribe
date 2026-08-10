#!/usr/bin/env python3
"""Build a day-by-day timeline of what Jeff pushed on, from his own prompts.

SOURCE: ~/.claude/history.jsonl — every prompt typed into Claude Code, back to
2026-02-26. Use THIS, not the .jsonl session transcripts.

Why it matters: session transcripts under ~/.claude/projects/ are auto-deleted
after ~30 days (cleanupPeriodDays). On 2026-08-09 only 13 survived, covering 44
days, and an earlier version of this script read that as "44 days of history
exist." It is not. history.jsonl is not on that cleanup schedule and held 8,697
prompts across 147 days — 3.3x the days and 4.5x the messages. If a future
version of this file looks thin at the front, suspect retention before
concluding the work did not happen.

What this gives per day: volume, topics (keyword hit), and the most substantive
things he actually wrote. It contains only HIS side — no assistant replies, no
record of what was built or broke. Pair it with stemscriber_full_state.md.
"""
import datetime
import glob
import json
import os
import re
from collections import Counter, defaultdict

SRC = os.path.expanduser("~/.claude/history.jsonl")
OUT = os.path.expanduser("~/Desktop/StemScriber/Docs/SESSION-TIMELINE.md")

TOPICS = {
    "ads/marketing": r"\b(ad|ads|meta|facebook|instagram|campaign|pixel|market|cpm|ctr|audience|koozie|sticker|newsletter|beehiiv)\b",
    "iOS/App Store": r"\b(ios|apple|testflight|app store|xcode|ipad|iphone|asc|review)\b",
    "Android/Play": r"\b(android|play store|play console|aab|google play|keystore)\b",
    "charts/binder": r"\b(chart|binder|chord|onsong|setlist|set list|lyric|tab|key|transpose)\b",
    "console/practice": r"\b(console|practice|stem|separat|mixer|tuner|vocal|piano|fretboard|demucs|model|train)\b",
    "infra/prod": r"\b(server|prod|deploy|hetzner|cloudflare|backup|r2|database|supabase|crash|down|502|modal)\b",
    "money/pricing": r"\b(price|pricing|pay|paid|revenue|stripe|subscription|plan|lifetime|comp|refund|invoice)\b",
    "Linda": r"\b(linda|irs|ssa|medicare|poa|tiaa|social security|mychart)\b",
    "band/music": r"\b(band|gig|spare kings|reckoning|poster|song|album|logic|record)\b",
    "video/content": r"\b(video|clip|contributor|youtube|reel|tiktok|script|film|footage)\b",
    "memory/process": r"\b(memory|notes|stale|context window|session|audit|organize)\b",
    "legal": r"\b(copyright|dmca|licen|lawyer|legal|trademark|llc|terms|privacy)\b",
}

SKIP = re.compile(r"^\s*(ok|okay|yes|yeah|no|nope|thanks|got it|sure|cool|nice|great|perfect|"
                  r"do it|go|yep|k|done|right|correct|good|hi|hey|morning|continue|next)\b[\s.!?]*$", re.I)

# Length-ranking otherwise surfaces text he PASTED IN (handoffs from other
# terminals, context-continuation boilerplate, error dumps). Those are the
# loudest messages of a day and say the least about what he wanted.
PASTED = re.compile(r"^\s*(this session is being continued|from other terminal|"
                    r"from another terminal|this is from another terminal|caveat:|"
                    r"<command-|traceback|\{|\[\{|https?://\S+$|/[a-z-]+\s*$)", re.I)
ATTACH = re.compile(r'@"?/Users/\S*?/uploads/[^"\s]+"?\s*')


def clean(t):
    t = ATTACH.sub("[screenshot] ", t)          # keep the signal, drop the path
    return re.sub(r"\s+", " ", t).strip()


days = defaultdict(list)
seen = set()
total = 0
for ln in open(SRC, encoding="utf-8", errors="replace"):
    try:
        d = json.loads(ln)
    except Exception:
        continue
    text, ts = d.get("display", ""), d.get("timestamp")
    if not text or not ts:
        continue
    dt = datetime.datetime.fromtimestamp(ts / 1000)
    key = (dt.strftime("%Y-%m-%d"), text.strip()[:200])
    if key in seen:                              # same prompt resent / retried
        continue
    seen.add(key)
    total += 1
    days[dt.strftime("%Y-%m-%d")].append((dt.strftime("%H:%M"), clean(text)))

out = [
    "# StemScriber — session timeline\n",
    f"Built from **{total:,} prompts Jeff typed** into Claude Code across "
    f"{len(days)} days, {min(days)} to {max(days)}.\n",
    "**What this is:** what Jeff pushed on each day, in his own words. "
    "**What it is not:** a record of what got built or what broke — that lives in "
    "`stemscriber_full_state.md`. Use them together.\n",
    "**Source:** `~/.claude/history.jsonl`. Do **not** rebuild this from the session "
    "transcripts in `~/.claude/projects/` — those auto-delete after ~30 days and will "
    "make the early months look empty when they were not.\n",
    "Regenerate: `~/stemscribe/tools/memory-audit/build_timeline.py`\n",
    # Hand-written, unlike everything below it. The per-day entries are mechanical
    # extraction; this arc is interpretation and should be re-read sceptically.
    """
## The arc

**This was near-daily work.** 147 active days out of the 165 between first and last.
March 26/31, April 27/30, **May 31/31**, June 24/30, July 28/31. Any account of this
period that describes a quiet month is wrong.

**Before Feb 26 — Cowork, not the terminal.** The terminal record starts mid-story. Earlier
work lived in Cowork and claude.ai chats; the session titles are indexed in the Prologue at
the bottom, going back to **Jan 17**, and they start with **poster generation, not StemScriber**.

**Feb 26 — the first terminal prompts.** Three days, mostly setup and band material.

**March to May — building the thing.** The console and the separation pipeline are the
constant, with charts and the binder close behind, and band material threaded all the
way through. May is the only month with a perfect attendance record.

**June — the store turn.** iOS and App Store jump to sit level with the console. The
work shifts from making it to shipping it.

**July — the heaviest technical month.** Console and pipeline density roughly doubles,
App Store review runs alongside it, and Linda's IRS matter starts taking real space.

**August — shipping, then selling, then doubting the instruments.** Both stores
published. The centre of gravity moves to ads, money and *measurement*. The repeated
theme is not "build more" but **"something is broken and was never set up right"** —
running through the ad spend, the tracking stack, and the memory notes themselves, and
ending 8/9 with the free call-to-action found to be a dead end.

⚠️ **A previous version of this arc claimed May was mostly Linda and June was too
sparse to call a phase.** Both were false. They were artifacts of building from the 13
surviving session transcripts instead of `history.jsonl`, and Jeff caught it
immediately because he had been there. Sampling bias reads exactly like history.
""",
    "\n---\n",
]

month = None
for day in sorted(days):
    msgs = days[day]
    if day[:7] != month:
        month = day[:7]
        mdays = [k for k in days if k.startswith(month)]
        out.append(f"\n# {month}  ·  {sum(len(days[k]) for k in mdays):,} messages "
                   f"over {len(mdays)} days\n")

    blob = " ".join(m[1] for m in msgs).lower()
    counts = Counter({n: len(re.findall(p, blob)) for n, p in TOPICS.items()})
    top = [n for n, c in counts.most_common(4) if c > 0]

    subst = [m for m in msgs
             if 60 < len(m[1]) < 900 and not SKIP.match(m[1]) and not PASTED.match(m[1])]
    subst.sort(key=lambda m: -len(m[1]))
    picks = sorted(subst[:5], key=lambda m: m[0])

    out.append(f"\n## {day}  ·  {len(msgs)} messages")
    if top:
        out.append(f"*{' · '.join(top)}*\n")
    for t, text in picks:
        if len(text) > 300:
            text = text[:300].rsplit(" ", 1)[0] + "…"
        out.append(f"- `{t}` {text}")
    if not picks:
        out.append("- (short exchanges only)")

# ── Prologue: Cowork sessions ───────────────────────────────────────────────
# Jeff worked in Cowork and claude.ai chats before the terminal, so history.jsonl
# alone starts mid-story. These files carry a title and createdAt per session but
# NO message bodies, so this is a rundown of what was worked on, not a record of
# what was said. Jeff's own read (8/9): "most of that is just gonna be the
# beginnings... nice to have a rundown but not as important." Kept proportionate.
COWORK = os.path.expanduser("~/Library/Application Support/Claude")
sess = []
for sub in ("local-agent-mode-sessions", "claude-code-sessions"):
    for f in glob.glob(os.path.join(COWORK, sub, "**", "*.json"), recursive=True):
        try:
            d = json.load(open(f, errors="replace"))
        except Exception:
            continue
        if isinstance(d, dict) and d.get("createdAt") and d.get("title"):
            sess.append((datetime.datetime.fromtimestamp(d["createdAt"] / 1000), d["title"]))

if sess:
    sess.sort()
    bym = defaultdict(list)
    for dt, title in sess:
        bym[dt.strftime("%Y-%m")].append((dt.strftime("%d"), title))
    pro = ["\n\n---\n",
           "\n# Prologue — Cowork sessions\n",
           f"**{len(sess)} sessions, {sess[0][0].date()} to {sess[-1][0].date()}**, from the Claude "
           "desktop app's session store. This is where the work lived before the terminal.\n",
           "**Titles only.** These files record what each session was *about*, not what was said — "
           "there are no message bodies to recover. Treat it as an index, not a transcript.\n",
           "Note the start: **January is poster generation, not StemScriber.** The product does not "
           "appear until later.\n"]
    for mo in sorted(bym):
        seen_t, uniq = set(), []
        for dd, t in bym[mo]:
            if t.lower() not in seen_t:
                seen_t.add(t.lower())
                uniq.append((dd, t))
        pro.append(f"\n**{mo}** · {len(bym[mo])} sessions\n")
        for dd, t in uniq[:40]:
            pro.append(f"- `{dd}` {t}")
        if len(uniq) > 40:
            pro.append(f"- *…and {len(uniq) - 40} more*")
    out += pro

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write("\n".join(out) + "\n")
print(f"  messages : {total:,}")
print(f"  days     : {len(days)}  ({min(days)} .. {max(days)})")
print(f"  written  : {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB)")
