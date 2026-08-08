# Memory audit tools

Built 2026-08-08 after Jeff: *"every time I get to 1 million tokens the session
is swollen and some of the more important stuff gets left out."*

The problem: context handoffs lose things, and notes silently go stale. The
transcripts on disk are the ground truth and nobody was reading them.

## extract_user_msgs.py
Streams every `~/.claude/projects/-Users-jeffkozelski/*.jsonl` transcript and
pulls out **only the messages Jeff actually typed**, chronologically. Filters
out tool results (which also arrive as role=user) and system-reminder noise.

First run: 1.3 GB of transcripts -> 1,866 messages, 546 KB, 2026-05-09..08-08.

## find_missing.py
Flags directive-sounding messages ("never", "make sure", "I want"...) and
scores them against the memory directory.

**Known weakness, do not trust its zero:** it tests whether words appear
anywhere in ~200 KB of memory text, so almost everything looks "covered" by
chance. Treat it as a way to shrink 1,866 messages down to ~113 worth READING,
then read them. Two real gaps were found that way (band-share model, Ultimate
Guitar import stripping lyrics) after the automated score said zero.

## When to run
At every context handoff, before assuming compaction kept what mattered.
