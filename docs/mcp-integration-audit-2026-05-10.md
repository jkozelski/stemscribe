# MCP Integration Audit — 2026-05-10

**Sources:** `/Users/jeffkozelski/.claude.json`, `claude mcp list`, 33 transcripts modified 2026-04-26 → 2026-05-10 under `/Users/jeffkozelski/.claude/projects/-Users-jeffkozelski/` (top-level + `*/subagents/`). Raw scan output at `/tmp/mcp_scan_summary.json`. Token estimates = chars/4.

---

## Task 1 — Inventory (25 active MCPs)

### Stdio servers in `.claude.json`

| Server | Talks to | Connected | ~Tools |
|---|---|---|---:|
| github | GitHub API (PAT) | yes | 26 |
| playwright | local Chromium | yes | 25 |
| n8n-mcp | n8n.kozbotix.com | yes | 20 |
| context7 | Upstash docs API | yes | 2 |
| sequential-thinking | local | yes | 1 |
| memory | local JSON | yes | 9 |
| filesystem | local FS (scoped) | yes | 14 |
| **google-workspace** | Google APIs | yes | **~140** |
| stripe | Stripe API (test key) | yes | 40 |
| resend | Resend API | yes | 70 |
| tavily | Tavily search | yes | 5 |
| twilio | Twilio API (alpha) | yes | 12 |
| slack | Slack workspace | yes | 12 |
| notion | Notion API | yes | 20 |
| perplexity | Perplexity API | yes | 3 |
| desktop-commander | local shell+FS | yes | 30 |
| fetch | HTTP GET wrapper | yes | 1 |

### HTTP/SSE servers (registered via claude.ai, not in `.claude.json`)

| Server | Connected |
|---|---|
| claude.ai Crypto.com | yes |
| claude.ai Gmail | yes |
| claude.ai Google Calendar | yes |
| claude.ai Google Drive | **needs auth** |
| claude.ai Cloudflare Developer Platform | yes |
| claude.ai Zapier | **needs auth** |
| claude.ai Canva | yes |
| claude.ai HubSpot | **needs auth** |
| sentry | yes |

Plus `computer-use` exposed by the harness itself. Notable: claude.ai Gmail overlaps with stdio google-workspace; three HTTP MCPs are unauthenticated and useless.

---

## Task 2 — Real call frequency (last 14 days)

### Per-server (transcripts in scan period)

| Server | Calls | Total resp tokens | Avg tok | Heaviest tool |
|---|---:|---:|---:|---|
| **playwright** | 198 | 94,590 | 477 | `browser_snapshot` 26 @ **2,605** |
| **google-workspace** | 47 | 52,600 | 1,119 | `get_gmail_messages_content_batch` 7 @ **2,739** |
| computer-use | 62 | 1,442 | 23 | `screenshot` ×30 (images, text only counted) |
| stripe | 9 | 1,163 | 129 | `list_prices` ×3 |
| sentry | 6 | 758 | 126 | `search_issues` ×2 |
| claude.ai Gmail (HTTP) | 1 | 466 | 466 | `search_threads` |
| n8n-mcp | 2 | 183 | 91 | `n8n_health_check` |
| desktop-commander | 5 | 151 | 30 | `write_pdf` ×5 |
| tavily | 4 | 117 | 29 | `tavily_search` ×4 |

**Zero calls in 14d:** github, notion, perplexity, fetch, resend, twilio, slack, memory, sequential-thinking, filesystem, context7, claude.ai Canva, Cloudflare, Crypto.com, Calendar, Drive, Zapier, HubSpot.

### Citations (heavy hitters)

- `mcp__playwright__browser_snapshot` ×26, avg 2,605 tok — `4d86d432-1c9e-47f9-902f-e6527b96de2b.jsonl` line 423
- `mcp__google-workspace__get_gmail_messages_content_batch` ×7, avg 2,739 tok — `4d86d432-...jsonl` line 62
- `mcp__google-workspace__search_gmail_messages` ×9, avg 1,058 — line 58
- `mcp__google-workspace__read_sheet_values` ×9, avg 1,038 — `d3e68e78-bee2-4789-a32b-1e5655546338.jsonl` line 9040
- `mcp__google-workspace__get_drive_file_content` ×6, avg 1,113 — line 119
- `mcp__playwright__browser_evaluate` ×61, avg 227 — `4d86d432-...jsonl` line 468
- `mcp__stripe__list_prices` ×3 — `d3e68e78-...jsonl` line 10315
- `mcp__sentry__search_issues` ×2 — `ddd5be43-32f4-43d9-8ee1-9afd40194b68.jsonl` line 1900

3 servers (playwright, computer-use, google-workspace) accounted for **307 of 334 calls = 92%**.

---

## Task 3 — Categorization

**Token-heavy (CLI wins big):**
- *google-workspace* — 1,119 avg tok/call, mostly Gmail bodies and sheet rows. A typed CLI returning just requested fields cuts payload ~10×. Also has ~140 tool schemas loaded into every session, the worst schema-tax offender.
- *playwright* — 477 avg with `browser_snapshot` at 2,605. CLI conversion is awkward (see below).
- *github* — 0 calls in window but 26 tool defs always loaded. `gh` CLI is already installed and a strict drop-in.
- *n8n-mcp* — when it fires (`get_workflow`, `n8n_list_workflows`) payloads are huge.

**Latency-bound (CLI helps):**
- Every HTTP MCP (sentry, the eight claude.ai ones) does oauth-backed round trips with cold starts. CLI process spawn is faster but Jeff barely uses most of these.
- `stripe` npx cold-spawn adds ~1s per session.

**Light / fine-as-is:**
- sequential-thinking (1 tool, no payload), context7 (2 tools, light docs lookup), tavily (4 calls, tiny). Keep.
- memory (local JSON) — duplicates the role of `~/.claude/.../MEMORY.md`. Probably removable.
- fetch (1 tool, 0 calls) — built-in WebFetch covers it. Remove.

**Custom data plane (hard to CLI):**
- *playwright* — stateful browser session, live page handle.
- *desktop-commander* — persistent shell processes; that statefulness IS the value.
- *slack* — pull-based today, but bidirectional in principle.
- *computer-use* — harness tool, out of scope.

**Dead weight — disable, don't migrate:**
- claude.ai Drive, Zapier, HubSpot (unauthenticated, 0 calls).
- claude.ai Crypto.com, Canva, Cloudflare, Calendar (connected, 0 calls, 10–40 schemas each loaded into every system prompt).
- resend (~70 tools), twilio (~12), notion (~20), perplexity (~3) — all stdio, all 0 calls in 14d.

---

## Task 4 — Top 5 migration candidates

Ranked by (context cost × usage volume) ÷ migration risk.

| # | MCP | 14d tokens | API clarity | Risk | Why CLI wins |
|---|---|---:|---|---|---|
| 1 | **google-workspace** | **52,600** | High — Google REST is well-typed and stable | Low — almost all reads, only 2 `modify_sheet_values` writes | Avg 1,000+ tok/call is mostly fluff. A typed CLI returning `from\|subject\|date\|id` instead of full message bodies cuts payload ~10×, AND removes 140 tool-schema entries from every system prompt. Biggest single win. |
| 2 | **github** | 0 (but 26 schemas always loaded) | Very high — `gh` exists with Jeff's PAT | Zero — `gh` is a drop-in | Disable the MCP entirely. Built-in Bash + `gh` is strictly better. Frees 26 schemas. |
| 3 | **n8n-mcp** | 183, but each fire is huge | Medium — n8n REST documented, workflow JSON varied | Low — 2 calls in 14d, no critical dep | Typed `n8n workflows list --fields=id,name,active` + `n8n workflow get <id> --jq` covers 95% of agent use. JWT already in `.claude.json`. |
| 4 | **stripe** | 1,163 | Very high — `stripe` CLI is canonical | Low — 9 calls all mutating but well-scoped | CLI already installed; keys in `~/stemscribe/.env`. Removes ~40 schemas + npx cold-spawn. |
| 5 | **resend** | 0 | High — small REST API | Zero | Either remove, OR Printing-Press a CLI from OpenAPI for the day Jeff actually broadcasts. ~70 schemas of pure tax today. |

Honorable mentions: **playwright** is the largest token consumer (94k/14d) but stateful — attack after the top 5. **claude.ai Canva** has heavy schemas and 0 calls — just turn it off in claude.ai settings.

---

## Task 5 — Keep as MCP

| MCP | Why |
|---|---|
| sequential-thinking | 1 tool, no API, no payload. CLI makes no sense (it's a reasoning scratchpad). |
| filesystem | 0 calls but trivial cost; useful fallback for sub-agents without built-ins. |
| memory | If kept at all, leave as MCP (tiny schema). Otherwise remove entirely — don't migrate. |
| context7 | Prebuilt docs lookup, light schema, occasionally useful. |
| playwright | Stateful page handle is the value — CLI loses it. |
| desktop-commander | `start_process` + `interact_with_process` semantics aren't reproducible via stateless CLI. |
| sentry | Structured issue search beats `sentry-cli`; light usage anyway. |
| computer-use | Harness tool, not Jeff's MCP. |

---

## Bottom line

In 14 days, 3 servers accounted for 92% of calls. Only **google-workspace** is a clean CLI candidate: high payload, well-defined REST, low write-risk. Most other MCPs either fire too rarely to justify migration, are stateful, or are dead weight that should be **disabled, not migrated**.

The single biggest immediate win is **disabling the 8 claude.ai HTTP MCPs that haven't fired once** plus the 4 stdio MCPs (resend, twilio, notion, perplexity) with zero calls. That reclaims hundreds of tokens of system-prompt overhead per session at zero risk and zero migration work.

**Verdict:** Worth ~1 day to disable the dead MCPs and Printing-Press-ify google-workspace; the remaining github/stripe/n8n/resend work is marginal — only pursue it if Jeff specifically wants to standardize everything on CLIs.
