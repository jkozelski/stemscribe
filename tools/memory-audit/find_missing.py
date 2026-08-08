#!/usr/bin/env python3
"""Find durable instructions Jeff gave that never made it into memory.

Heuristic: messages carrying directive language ("never", "always", "make
sure", "I want", "don't", "remember", "from now on"...) are the ones that
should have become feedback/project notes. Cross-check their distinctive
terms against the memory directory and surface the ones with no coverage.
"""
import os
import re
import glob
import collections

SRC = "/private/tmp/claude-501/-Users-jeffkozelski/6389173b-d548-477f-b7d3-67dd5088abd0/scratchpad/all_user_messages.md"
MEM = os.path.expanduser("~/.claude/projects/-Users-jeffkozelski/memory")

DIRECTIVE = re.compile(
    r"\b(never|always|make sure|from now on|remember|don'?t (ever|forget|use|do)|"
    r"i want|i need|i'?d like|stop (doing|using)|prefer|going forward|"
    r"every time|each time|has to|must)\b", re.I)

# throwaway conversational stuff that matches but carries nothing durable
SKIP = re.compile(r"^\s*[-`\d:\[\]]*\s*(ok|okay|yes|yeah|no|nope|thanks|got it|sure|"
                  r"cool|nice|great|perfect|do it|go|yep|k)\b[\s.!?]*$", re.I)

memory_text = ""
for p in glob.glob(f"{MEM}/*.md"):
    memory_text += open(p, encoding="utf-8", errors="replace").read().lower()

lines = open(SRC, encoding="utf-8", errors="replace").read().splitlines()
cur_day = None
hits = []
for ln in lines:
    if ln.startswith("## "):
        cur_day = ln[3:].strip()
        continue
    if not ln.startswith("- `"):
        continue
    m = re.match(r"- `(\d\d:\d\d)` \[([0-9a-f]{8})\] (.*)", ln)
    if not m:
        continue
    time_, sid, text = m.groups()
    if len(text) < 25 or SKIP.match(text):
        continue
    if not DIRECTIVE.search(text):
        continue
    hits.append((cur_day, time_, text))

print(f"  directive-sounding messages: {len(hits)}")

# score coverage: distinctive words from the message present in memory?
STOP = set("""the a an and or but if then that this these those to of in on for with
without is are was were be been being it its as at by from into over under about
you your we our i me my he she they them do does did doing done have has had
can could should would will shall may might must not no yes just like get got
make made want need use using used out up down all any some more most other
than when where which who whom what how why so such very really thing things
let lets go going went too also new old first last next back over again once
know knows knew think thought see seen say said tell told put puts run runs
work works working time times day days now today one two three""".split())

def distinctive(text):
    ws = re.findall(r"[a-z][a-z0-9_.-]{3,}", text.lower())
    return [w for w in ws if w not in STOP]

uncovered = []
for day, time_, text in hits:
    ws = distinctive(text)
    if not ws:
        continue
    missing = [w for w in ws if w not in memory_text]
    ratio = len(missing) / len(ws)
    if ratio >= 0.5 and len(ws) >= 4:
        uncovered.append((ratio, day, time_, text, missing[:6]))

uncovered.sort(key=lambda x: (-x[0], x[1]))
print(f"  of those, poorly covered by memory: {len(uncovered)}\n")

by_month = collections.Counter(u[1][:7] for u in uncovered)
print("  uncovered by month:")
for k, v in sorted(by_month.items()):
    print(f"    {k}: {v}")

OUT = "/private/tmp/claude-501/-Users-jeffkozelski/6389173b-d548-477f-b7d3-67dd5088abd0/scratchpad/uncovered_instructions.md"
with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("# Directive messages with little or no coverage in memory\n\n")
    fh.write(f"{len(uncovered)} candidates, newest first.\n\n")
    for ratio, day, time_, text, missing in uncovered:
        fh.write(f"## {day} {time_}  (uncovered {ratio:.0%})\n\n> {text}\n\n")
        fh.write(f"missing terms: {', '.join(missing)}\n\n")
print(f"\n  written: {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)")
