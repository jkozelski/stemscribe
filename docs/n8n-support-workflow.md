# n8n Support Ticket Workflow — StemScribe

**Version:** 1.0
**Last updated:** 2026-03-16
**Scale:** 10 beta testers

---

## Workflow Diagram

```
                        POST /api/support/ticket
                                 |
                                 v
                    +------------------------+
                    |   Webhook Trigger      |
                    |   (receives ticket)    |
                    +------------------------+
                                 |
                    +------------+------------+
                    |            |            |
                    v            v            v
          +-----------+  +-----------+  +-----------+
          | Log to    |  | SMS to    |  | Email to  |
          | Google    |  | Jeff      |  | Customer  |
          | Sheet     |  | (Twilio)  |  | (Gmail)   |
          +-----------+  +-----------+  +-----------+
                |                |
                v                |
       +----------------+       |
       | IF billing/    |       |
       | refund ticket  |       |
       +----------------+       |
          |          |          |
         YES         NO        |
          |          |          |
          v          v          |
  +------------+ (done)        |
  | Append to  |               |
  | Billing    |               |
  | tab + send |               |
  | HIGH PRIO  |               |
  | SMS        |               |
  +------------+               |
                               |
                               v
                    +------------------------+
                    |   Respond to Webhook   |
                    |   { status: "ok" }     |
                    +------------------------+


--- SEPARATE WORKFLOW ---

          Admin responds (webhook or manual trigger)
                         |
                         v
              +---------------------+
              | Update Google Sheet |
              | (status, response)  |
              +---------------------+
                         |
                         v
              +---------------------+
              | Email response to   |
              | customer (Gmail)    |
              +---------------------+
```

---

## Credentials Needed

Set these up in n8n **Settings > Credentials** before building the workflow.

| Credential Name | Type | Details |
|---|---|---|
| `Twilio StemScribe` | Twilio API | Account SID: `AC61b4ba568a01c65bf90d98655261161b`, Auth Token from Twilio console |
| `Gmail StemScribe` | Gmail OAuth2 | Account: `stemscribe.io@gmail.com`, enable Gmail API in Google Cloud Console |
| `Google Sheets StemScribe` | Google Sheets OAuth2 | Same Google account, enable Sheets API |

### Google Cloud Setup (one-time)
1. Go to https://console.cloud.google.com
2. Create project "StemScribe n8n"
3. Enable APIs: Gmail API, Google Sheets API
4. Create OAuth2 credentials (Desktop app type)
5. Download client_id and client_secret
6. In n8n, create Google OAuth2 credential with those values
7. Authorize when prompted

---

## Google Sheet Template

**Spreadsheet name:** `StemScribe Support Tickets`

### Tab 1: "All Tickets"

| Column | Header | Example |
|---|---|---|
| A | ID | `tkt_1710547200_abc123` |
| B | Date | `2026-03-16T10:00:00Z` |
| C | Name | `Dave Grohl` |
| D | Email | `dave@example.com` |
| E | Subject | `Audio Processing` |
| F | Message | `My stems sound weird on track 3...` |
| G | Priority | `normal` |
| H | Status | `new` |
| I | Response | *(empty until replied)* |
| J | Resolved Date | *(empty until resolved)* |

### Tab 2: "Billing"

Same columns. Only billing/refund tickets get copied here.

**Create the spreadsheet manually**, then grab the spreadsheet ID from the URL:
`https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit`

---

## Node-by-Node Setup

### Node 0: Webhook Trigger

- **Type:** Webhook
- **HTTP Method:** POST
- **Path:** `support-ticket`
- **Response Mode:** Last Node (so we can return a response)
- **Authentication:** Header Auth
  - Header Name: `x-api-key`
  - Header Value: generate a random key, e.g. `sk_support_a1b2c3d4e5f6` — store this in your backend `.env`

**Full webhook URL (local):**
```
http://localhost:5678/webhook/support-ticket
```

**Production (if tunneled):**
```
https://n8n.yourdomain.com/webhook/support-ticket
```

**Expected payload:**
```json
{
  "id": "tkt_1710547200_abc123",
  "timestamp": "2026-03-16T10:00:00Z",
  "name": "Dave Grohl",
  "email": "dave@example.com",
  "subject": "Audio Processing",
  "message": "My stems sound weird when I separate track 3. The vocals bleed into the guitar stem.",
  "priority": "normal"
}
```

---

### Node 1: Log to Google Sheet — All Tickets

- **Type:** Google Sheets
- **Operation:** Append Row
- **Credential:** `Google Sheets StemScribe`
- **Document ID:** `{your spreadsheet ID}`
- **Sheet Name:** `All Tickets`
- **Mapping Mode:** Map Each Column

**Column mappings (use expressions):**

| Sheet Column | Value (Expression) |
|---|---|
| ID | `{{ $json.body.id }}` |
| Date | `{{ $json.body.timestamp }}` |
| Name | `{{ $json.body.name }}` |
| Email | `{{ $json.body.email }}` |
| Subject | `{{ $json.body.subject }}` |
| Message | `{{ $json.body.message }}` |
| Priority | `{{ $json.body.priority }}` |
| Status | `new` |
| Response | *(leave empty)* |
| Resolved Date | *(leave empty)* |

---

### Node 2: SMS Notification to Jeff (Twilio)

- **Type:** Twilio
- **Operation:** Send SMS
- **Credential:** `Twilio StemScribe`

**Parameters:**

| Field | Value |
|---|---|
| From | `+18447915323` |
| To | `+18034149454` |
| Message | See expression below |

**Message expression:**
```
{{ $json.body.priority === "high" || $json.body.subject.toLowerCase().includes("billing") || $json.body.subject.toLowerCase().includes("refund") ? "⚠️ HIGH PRIORITY\n" : "" }}🎫 New StemScribe ticket from {{ $json.body.name }}: {{ $json.body.subject }} — {{ $json.body.message.substring(0, 100) }}{{ $json.body.message.length > 100 ? "..." : "" }}
```

**Example SMS output (normal):**
```
🎫 New StemScribe ticket from Dave Grohl: Audio Processing — My stems sound weird when I separate track 3. The vocals bleed into the guitar stem.
```

**Example SMS output (high priority):**
```
⚠️ HIGH PRIORITY
🎫 New StemScribe ticket from Karen Manager: Billing/Refund — I was charged twice for my subscription last month and need a refund processed immediately...
```

**Twilio API equivalent (for reference):**
```bash
curl -X POST "https://api.twilio.com/2010-04-01/Accounts/AC61b4ba568a01c65bf90d98655261161b/Messages.json" \
  --data-urlencode "Body=🎫 New StemScribe ticket from Dave Grohl: Audio Processing — My stems sound weird..." \
  --data-urlencode "From=+18447915323" \
  --data-urlencode "To=+18034149454" \
  -u "AC61b4ba568a01c65bf90d98655261161b:{AUTH_TOKEN}"
```

---

### Node 3: Confirmation Email to Customer (Gmail)

- **Type:** Gmail
- **Operation:** Send Email
- **Credential:** `Gmail StemScribe`

**Parameters:**

| Field | Value |
|---|---|
| To | `{{ $json.body.email }}` |
| Subject | `We got your message — StemScribe Support` |
| Email Format | HTML |
| Body | See HTML template below |

**Email HTML Template:**

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin:0; padding:0; background-color:#1a1a2e; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#1a1a2e;">
    <tr>
      <td align="center" style="padding:40px 20px;">
        <table role="presentation" width="600" cellpadding="0" cellspacing="0" style="background-color:#16213e; border-radius:12px; overflow:hidden;">

          <!-- Header -->
          <tr>
            <td style="background: linear-gradient(135deg, #e94560 0%, #ff6b35 100%); padding:30px 40px; text-align:center;">
              <h1 style="margin:0; color:#ffffff; font-size:24px; font-weight:700; letter-spacing:1px;">
                StemScribe
              </h1>
              <p style="margin:8px 0 0; color:rgba(255,255,255,0.85); font-size:13px; letter-spacing:2px; text-transform:uppercase;">
                Support
              </p>
            </td>
          </tr>

          <!-- Body -->
          <tr>
            <td style="padding:40px;">
              <p style="color:#e0e0e0; font-size:16px; line-height:1.6; margin:0 0 20px;">
                Hey {{ $json.body.name }}, thanks for reaching out!
              </p>
              <p style="color:#b0b0b0; font-size:15px; line-height:1.6; margin:0 0 25px;">
                We got your message about <strong style="color:#ff6b35;">{{ $json.body.subject }}</strong> and we'll get back to you within 24 hours.
              </p>

              <!-- Ticket ID Box -->
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 25px;">
                <tr>
                  <td style="background-color:#0f3460; border-left:4px solid #ff6b35; border-radius:6px; padding:16px 20px;">
                    <p style="color:#808080; font-size:12px; text-transform:uppercase; letter-spacing:1px; margin:0 0 4px;">
                      Your Ticket ID
                    </p>
                    <p style="color:#ffffff; font-size:16px; font-family:'Courier New', monospace; margin:0;">
                      {{ $json.body.id }}
                    </p>
                  </td>
                </tr>
              </table>

              <p style="color:#b0b0b0; font-size:14px; line-height:1.6; margin:0 0 10px;">
                If it's urgent, text us at <strong style="color:#e0e0e0;">(843) 874-8999</strong>
              </p>
              <p style="color:#b0b0b0; font-size:14px; line-height:1.6; margin:0;">
                Keep making music,<br>
                <strong style="color:#e0e0e0;">The StemScribe Team</strong>
              </p>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="background-color:#0f3460; padding:20px 40px; text-align:center; border-top:1px solid #1a3a6e;">
              <p style="color:#606080; font-size:12px; margin:0;">
                StemScribe &mdash; Stems, chords, and tabs from any song.
              </p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>
```

> **n8n tip:** Paste the HTML into the "HTML" body field. Use `{{ }}` expressions inline — n8n will resolve them before sending.

---

### Node 4: IF — Billing/Refund Check

- **Type:** IF
- **Conditions (OR):**
  - `{{ $json.body.subject }}` contains `Billing`
  - `{{ $json.body.subject }}` contains `Refund`
  - `{{ $json.body.priority }}` equals `high`

**True branch:**

#### Node 4a: Append to Billing Tab

- **Type:** Google Sheets
- **Operation:** Append Row
- **Sheet Name:** `Billing`
- Same column mappings as Node 1

#### Node 4b: High Priority SMS (optional second alert)

- **Type:** Twilio
- **From:** `+18447915323`
- **To:** `+18034149454`
- **Message:**
```
🚨 BILLING/REFUND TICKET — {{ $json.body.name }} ({{ $json.body.email }})
{{ $json.body.message.substring(0, 200) }}
Reply needed ASAP.
```

**False branch:** No operation (connect to Respond node directly).

---

### Node 5: Respond to Webhook

- **Type:** Respond to Webhook
- **Response Code:** 200
- **Response Body:**
```json
{
  "status": "ok",
  "ticketId": "{{ $json.body.id }}",
  "message": "Ticket received. Confirmation email sent."
}
```

---

## Separate Workflow: Admin Response

This is a second n8n workflow triggered when you respond to a ticket.

### Trigger: Webhook

- **Path:** `support-respond`
- **Method:** POST
- **Payload:**
```json
{
  "ticketId": "tkt_1710547200_abc123",
  "customerEmail": "dave@example.com",
  "customerName": "Dave Grohl",
  "subject": "Audio Processing",
  "response": "Hey Dave! That bleed issue is usually fixed by...",
  "status": "resolved"
}
```

### Node R1: Update Google Sheet

- **Type:** Google Sheets
- **Operation:** Update Row
- **Sheet:** `All Tickets`
- **Lookup Column:** `ID`
- **Lookup Value:** `{{ $json.body.ticketId }}`
- **Update columns:**
  - Status: `{{ $json.body.status }}`
  - Response: `{{ $json.body.response }}`
  - Resolved Date: `{{ $now.toISO() }}` (if status is "resolved")

### Node R2: Email Response to Customer

- **Type:** Gmail
- **To:** `{{ $json.body.customerEmail }}`
- **Subject:** `Re: {{ $json.body.subject }} — StemScribe Support`
- **Body (HTML):**

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin:0; padding:0; background-color:#1a1a2e; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#1a1a2e;">
    <tr>
      <td align="center" style="padding:40px 20px;">
        <table role="presentation" width="600" cellpadding="0" cellspacing="0" style="background-color:#16213e; border-radius:12px; overflow:hidden;">

          <!-- Header -->
          <tr>
            <td style="background: linear-gradient(135deg, #e94560 0%, #ff6b35 100%); padding:24px 40px; text-align:center;">
              <h1 style="margin:0; color:#ffffff; font-size:22px; font-weight:700;">
                StemScribe Support
              </h1>
            </td>
          </tr>

          <!-- Body -->
          <tr>
            <td style="padding:40px;">
              <p style="color:#e0e0e0; font-size:16px; line-height:1.6; margin:0 0 20px;">
                Hey {{ $json.body.customerName }},
              </p>
              <p style="color:#b0b0b0; font-size:15px; line-height:1.7; margin:0 0 25px; white-space:pre-line;">
                {{ $json.body.response }}
              </p>

              <!-- Ticket Reference -->
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 25px;">
                <tr>
                  <td style="background-color:#0f3460; border-left:4px solid #ff6b35; border-radius:6px; padding:12px 20px;">
                    <p style="color:#808080; font-size:11px; text-transform:uppercase; letter-spacing:1px; margin:0;">
                      Ticket: {{ $json.body.ticketId }}
                    </p>
                  </td>
                </tr>
              </table>

              <p style="color:#b0b0b0; font-size:14px; line-height:1.6; margin:0 0 8px;">
                Still need help? Just reply to this email or text <strong style="color:#e0e0e0;">(843) 874-8999</strong>.
              </p>
              <p style="color:#b0b0b0; font-size:14px; margin:0;">
                <strong style="color:#e0e0e0;">The StemScribe Team</strong>
              </p>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="background-color:#0f3460; padding:16px 40px; text-align:center; border-top:1px solid #1a3a6e;">
              <p style="color:#606080; font-size:12px; margin:0;">
                StemScribe &mdash; Stems, chords, and tabs from any song.
              </p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>
```

---

## Backend Integration

Add this to the StemScribe backend to call the webhook. Drop it wherever your support form POST handler lives.

```python
import httpx

N8N_WEBHOOK_URL = "http://localhost:5678/webhook/support-ticket"
N8N_API_KEY = "sk_support_a1b2c3d4e5f6"  # set in .env

async def send_to_n8n(ticket: dict):
    """Fire-and-forget to n8n. Don't block the user if n8n is down."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                N8N_WEBHOOK_URL,
                json=ticket,
                headers={"x-api-key": N8N_API_KEY}
            )
    except Exception as e:
        print(f"[n8n] Failed to send ticket: {e}")
```

---

## Environment Variables

Add to your `.env` (backend):

```bash
# n8n support workflow
N8N_WEBHOOK_URL=http://localhost:5678/webhook/support-ticket
N8N_SUPPORT_API_KEY=sk_support_a1b2c3d4e5f6

# n8n admin response workflow
N8N_RESPOND_WEBHOOK_URL=http://localhost:5678/webhook/support-respond
```

---

## Testing Instructions

### 1. Test the webhook directly

```bash
curl -X POST http://localhost:5678/webhook-test/support-ticket \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_support_a1b2c3d4e5f6" \
  -d '{
    "id": "tkt_test_001",
    "timestamp": "2026-03-16T10:00:00Z",
    "name": "Test User",
    "email": "jkozelski@gmail.com",
    "subject": "Audio Processing",
    "message": "Testing the support ticket workflow. This is a normal priority ticket about stem separation quality.",
    "priority": "normal"
  }'
```

### 2. Test high priority / billing

```bash
curl -X POST http://localhost:5678/webhook-test/support-ticket \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_support_a1b2c3d4e5f6" \
  -d '{
    "id": "tkt_test_002",
    "timestamp": "2026-03-16T10:05:00Z",
    "name": "Billing Test",
    "email": "jkozelski@gmail.com",
    "subject": "Billing/Refund",
    "message": "I was charged twice for my subscription. Please refund the duplicate charge.",
    "priority": "high"
  }'
```

### 3. Test the admin response workflow

```bash
curl -X POST http://localhost:5678/webhook-test/support-respond \
  -H "Content-Type: application/json" \
  -d '{
    "ticketId": "tkt_test_001",
    "customerEmail": "jkozelski@gmail.com",
    "customerName": "Test User",
    "subject": "Audio Processing",
    "response": "Hey! Thanks for reporting that. The stem bleed issue is usually resolved by re-running separation with the quality slider set to High. Give that a try and let us know!",
    "status": "resolved"
  }'
```

### 4. Verify checklist

- [ ] Google Sheet: row appears in "All Tickets" tab
- [ ] Google Sheet: billing ticket also appears in "Billing" tab
- [ ] SMS: Jeff receives notification on +18034149454
- [ ] SMS: billing tickets have HIGH PRIORITY prefix
- [ ] Email: customer receives confirmation with correct name, subject, ticket ID
- [ ] Email: renders correctly in Gmail (dark theme, orange accents)
- [ ] Webhook: returns `{ "status": "ok" }` with 200
- [ ] Response workflow: Sheet row updated with response text and resolved date
- [ ] Response workflow: customer receives response email

---

## n8n Import (Node Connections)

When building in n8n, connect nodes in this order:

```
Webhook --> [3 parallel branches]
  Branch 1: Google Sheets (All Tickets) --> IF (billing?) --> [true] Google Sheets (Billing) + Twilio (high prio SMS)
  Branch 2: Twilio (standard SMS to Jeff)
  Branch 3: Gmail (confirmation to customer)

All branches --> Respond to Webhook
```

**To create parallel branches:** Drag three connections from the Webhook node output to each of the three target nodes. n8n runs them concurrently.

**To merge back:** Connect all terminal nodes to the "Respond to Webhook" node. Set it to wait for all inputs (n8n default behavior with multiple inputs is to trigger on the first — change the "Respond to Webhook" to use "Respond to All" or just place it after the slowest branch).

> **Practical tip:** For 10 beta testers, the simpler approach is to just chain the nodes sequentially: Webhook -> Sheet -> SMS -> Email -> IF -> (billing extras) -> Respond. Total execution time will still be under 3 seconds and avoids merge complexity.

---

## SMS Message Templates Reference

| Scenario | Template |
|---|---|
| Normal ticket | `🎫 New StemScribe ticket from {name}: {subject} — {message[:100]}` |
| High priority | `⚠️ HIGH PRIORITY\n🎫 New StemScribe ticket from {name}: {subject} — {message[:100]}` |
| Billing alert | `🚨 BILLING/REFUND TICKET — {name} ({email})\n{message[:200]}\nReply needed ASAP.` |

---

## Cost Estimate (10 beta testers)

| Service | Cost |
|---|---|
| Twilio SMS (outbound) | ~$0.0079/msg |
| Gmail | Free (Google Workspace) |
| Google Sheets | Free |
| n8n (self-hosted) | Free |
| **Per ticket** | **~$0.01-0.02** |

Even at 50 tickets/month, total cost is under $1/month.

---
---

# Workflow 2: Beta Tester Activity Monitor

**Import file:** `~/stemscribe/n8n/beta-monitor-workflow.json`

## Purpose

Polls the StemScribe backend every 30 minutes and sends Jeff an SMS if there is new beta tester activity (code redemptions or songs processed).

## Node-by-Node

### Node 0: Schedule Trigger

- **Type:** Schedule Trigger
- **Interval:** Every 30 minutes

### Node 1: GET Beta Codes

- **Type:** HTTP Request
- **Method:** GET
- **URL:** `http://localhost:5555/api/beta/codes`
- **Timeout:** 10s
- Runs in parallel with Node 2.

### Node 2: GET Library Songs

- **Type:** HTTP Request
- **Method:** GET
- **URL:** `http://localhost:5555/api/library`
- **Timeout:** 10s
- Runs in parallel with Node 1.

### Node 3: Merge Results

- **Type:** Merge
- **Mode:** Combine (by position)
- Combines the beta codes response and library response into a single item.

### Node 4: Compare With Previous Run (Code node)

- **Type:** Code (JavaScript)
- Uses `$getWorkflowStaticData('global')` to persist previous values across runs.
- Calculates deltas: `newRedemptions` and `newSongs`.
- Sets `hasActivity = true` if either delta > 0.
- Saves current counts back to static data for next run.

### Node 5: IF New Activity

- **Type:** IF
- **Condition:** `hasActivity` equals `true`
- **True branch:** Send SMS
- **False branch:** No action (workflow ends silently)

### Node 6: SMS Activity Summary (Twilio)

- **Type:** Twilio
- **From:** `+18447915323`
- **To:** `+18034149454`
- **Message template:**

```
📊 StemScribe Beta Activity
🆕 2 new code redemption(s) (total: 5)
🎵 3 new song(s) processed (total: 12)
Checked: 2026-03-17T10:30:00Z
```

Only lines with non-zero deltas appear.

---

## Connection Map

```
Schedule (30min) --> [parallel]
  --> GET /api/beta/codes --\
  --> GET /api/library -----+--> Merge --> Code (compare) --> IF activity?
                                                              |        |
                                                             YES       NO
                                                              |       (end)
                                                              v
                                                         SMS to Jeff
```

---
---

# Workflow 3: Error Alert

**Import file:** `~/stemscribe/n8n/error-alert-workflow.json`

## Purpose

Receives error events via webhook from the StemScribe error tracker. Maintains a rolling 1-hour window of errors using n8n static data. If the count exceeds 3 in that window, sends an SMS alert to Jeff.

## Node-by-Node

### Node 0: Webhook — Error Event

- **Type:** Webhook
- **HTTP Method:** POST
- **Path:** `error-alert`
- **Response Mode:** Last Node

**Full webhook URL:**
```
http://localhost:5678/webhook/error-alert
```

**Expected payload:**
```json
{
  "error_type": "TranscriptionError",
  "message": "MIDI conversion failed for job abc123",
  "endpoint": "/api/process",
  "timestamp": "2026-03-17T10:00:00Z"
}
```

### Node 1: Count Errors (Code node)

- **Type:** Code (JavaScript)
- Uses `$getWorkflowStaticData('global')` to maintain a rolling array of error timestamps.
- Adds the incoming error, prunes entries older than 1 hour.
- Sets `shouldAlert = true` if `errorCount > 3`.
- Outputs: `errorCount`, `errorTypes`, `latestMessage`, `latestEndpoint`.

### Node 2: IF > 3 Errors in 1hr

- **Type:** IF
- **Condition:** `shouldAlert` equals `true`
- **True branch:** SMS alert
- **False branch:** Respond OK silently

### Node 3: SMS Error Alert (Twilio)

- **Type:** Twilio
- **From:** `+18447915323`
- **To:** `+18034149454`
- **Message template:**

```
🚨 StemScribe alert: 5 errors in the last hour.
Types: TranscriptionError, SeparationError
Latest: MIDI conversion failed for job abc123
Endpoint: /api/process
Check /api/errors/patterns
```

### Node 4: Respond OK

- **Type:** Respond to Webhook
- **Response Code:** 200
- **Body:** `{ "status": "received", "errorCount": N, "alerted": true/false }`

---

## Connection Map

```
Webhook (POST /error-alert) --> Code (count + prune) --> IF > 3?
                                                         |       |
                                                        YES      NO
                                                         |       |
                                                    SMS Alert    |
                                                         |       |
                                                         v       v
                                                      Respond OK
```

---

## Backend Integration for Error Alerts

To send errors to this workflow, add a call in your error handler or logging system:

```python
import requests
import os

N8N_ERROR_WEBHOOK = os.environ.get('N8N_ERROR_WEBHOOK_URL', 'http://localhost:5678/webhook/error-alert')

def report_error_to_n8n(error_type, message, endpoint='unknown'):
    """Fire-and-forget error report to n8n."""
    try:
        requests.post(N8N_ERROR_WEBHOOK, json={
            'error_type': error_type,
            'message': message,
            'endpoint': endpoint,
        }, timeout=3)
    except Exception:
        pass  # Don't let monitoring failures cascade
```

---
---

# Importable n8n JSON Files

All three workflows are ready to import via **n8n UI > Workflows > Import from File**:

| Workflow | File | Trigger |
|---|---|---|
| Support Ticket Handler | `~/stemscribe/n8n/support-ticket-workflow.json` | Webhook POST |
| Beta Tester Activity Monitor | `~/stemscribe/n8n/beta-monitor-workflow.json` | Schedule (30min) |
| Error Alert | `~/stemscribe/n8n/error-alert-workflow.json` | Webhook POST |

### After Import Checklist

1. Replace `YOUR_SPREADSHEET_ID_HERE` in the Google Sheets nodes with your actual spreadsheet ID
2. Replace `CREDENTIAL_ID` references — n8n will prompt you to select credentials on first open
3. Activate each workflow (they import as inactive by default)
4. Test with the curl commands in the Testing section above

---

## Backend Integration (Updated)

The support endpoint at `backend/routes/support.py` now forwards ticket data to n8n automatically when `N8N_WEBHOOK_URL` is set.

### How It Works

- On ticket creation, `_forward_to_n8n(ticket)` fires in a background thread
- Non-blocking: the user gets their 201 response immediately
- If n8n is down, the ticket is still saved locally (graceful degradation)
- Optionally sends `x-api-key` header if `N8N_SUPPORT_API_KEY` is set

### Environment Variables

```bash
# Add to .env
N8N_WEBHOOK_URL=http://localhost:5678/webhook/support-ticket
N8N_SUPPORT_API_KEY=sk_support_a1b2c3d4e5f6

# For admin response workflow
N8N_RESPOND_WEBHOOK_URL=http://localhost:5678/webhook/support-respond

# For error alerts
N8N_ERROR_WEBHOOK_URL=http://localhost:5678/webhook/error-alert
```
