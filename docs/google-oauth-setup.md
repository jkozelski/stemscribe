# Google OAuth Setup for StemScriber

## 1. Create Google Cloud Project

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a new project named **StemScriber** (or select existing)
3. Enable **Google Identity Services** (no specific API needed for Sign-In with ID tokens)

## 2. Create OAuth Credentials

1. Go to **APIs & Services > Credentials**
2. Click **Create Credentials > OAuth Client ID**
3. Application type: **Web application**
4. Name: **StemScriber Web**
5. **Authorized JavaScript origins:**
   - `https://stemscribe.io`
   - `http://localhost:5555`
6. **Authorized redirect URIs:**
   - `https://stemscribe.io/auth/google/callback`
   - `http://localhost:5555/auth/google/callback`
7. Click **Create**

## 3. Configure OAuth Consent Screen

1. Go to **APIs & Services > OAuth consent screen**
2. User type: **External**
3. App name: **StemScriber**
4. Support email: jeff@tidepoolartist.com
5. Scopes: `email`, `profile`, `openid`
6. Add test users if in testing mode

## 4. Add to .env

Copy the Client ID and Client Secret from the credentials page and add to `~/stemscribe/.env`:

```
GOOGLE_CLIENT_ID=your-client-id-here.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-client-secret-here
```

## 5. Run the Database Migration

```bash
cd ~/stemscribe/backend
psql "$DATABASE_URL" -f migrations/002_google_oauth.sql
```

## 6. Frontend Integration

Add the Google Sign-In button to your login page. The frontend sends the credential token to `POST /auth/google`:

```html
<script src="https://accounts.google.com/gsi/client" async></script>

<div id="g_id_onload"
     data-client_id="YOUR_GOOGLE_CLIENT_ID"
     data-callback="handleGoogleLogin">
</div>
<div class="g_id_signin" data-type="standard"></div>

<script>
function handleGoogleLogin(response) {
    fetch('/auth/google', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({credential: response.credential})
    })
    .then(r => r.json())
    .then(data => {
        // data.access_token, data.refresh_token, data.user
        // Store tokens and redirect to app
    });
}
</script>
```

## API Endpoint

**POST /auth/google**

Request:
```json
{"credential": "google_id_token_string"}
```

Response (same as /auth/login):
```json
{
    "access_token": "...",
    "refresh_token": "...",
    "user": {
        "id": "...",
        "email": "...",
        "display_name": "...",
        "plan": "free",
        "avatar_url": "https://lh3.googleusercontent.com/...",
        "created_at": "..."
    }
}
```

Status codes:
- **200** — Existing user logged in (or existing email linked to Google)
- **201** — New user created via Google
- **401** — Invalid Google token
- **500** — GOOGLE_CLIENT_ID not configured
