// StemScriber — Authentication Module (Google Sign-In + JWT)
window.StemScriber = window.StemScriber || {};

// Capacitor native app (iOS/Android shell): apply iOS-specific tweaks.
//   - API_BASE points to prod (window.location.origin is capacitor://localhost otherwise)
//   - viewport-fit=cover + safe-area-inset padding so the status bar / Dynamic Island /
//     home-indicator don't overlap content
//   - Hide the heavy web footer (irrelevant inside a native shell)
//   - Hide the Google sign-in button (Google blocks OAuth in WKWebViews, and offering it
//     would force Apple to require Sign in with Apple too — magic link only here)
//   - Modal CTA rows get safe-area-inset padding so Cancel/Process clear the home indicator
//   - Treat a beta-redeemed user as authenticated for UI purposes — reveals the nav
//     items (Library/Practice/Demo/Settings) that are hidden by .nav-authed-only
if (window.Capacitor && window.Capacitor.isNativePlatform && window.Capacitor.isNativePlatform()) {
    // FORCE-override, do NOT fallback. Other scripts (js/config.js) auto-derive
    // API_BASE from window.location.origin, which is capacitor://localhost inside
    // the native shell and points to the local bundle, not the backend. Must win.
    window.StemScriber.API_BASE = 'https://stemscriber.com/api';

    // ---- Audio + download auth token injector ----------------------------
    // HTMLAudioElement and <a download> can't send Authorization headers; in
    // Capacitor we also can't share cookies with stemscriber.com. So append
    // the access token as ?token=... on every stemscriber.com URL the page
    // reaches (audio src, image src, fetch calls, download anchors). The
    // backend accepts query_string tokens via flask-jwt-extended (see
    // auth/jwt_setup.py — JWT_TOKEN_LOCATION includes 'query_string'). Web
    // users never run this block; they keep using the Authorization header.
    (function() {
        function appendToken(url) {
            if (typeof url !== 'string') return url;
            if (url.indexOf('https://stemscriber.com/') !== 0) return url;
            if (url.indexOf('token=') >= 0) return url; // already signed
            var tok;
            try { tok = localStorage.getItem('access_token'); } catch (e) {}
            if (!tok) return url;
            return url + (url.indexOf('?') >= 0 ? '&' : '?') + 'token=' + encodeURIComponent(tok);
        }
        window.__ssAppendToken = appendToken; // exposed for tests + retroactive patching

        // Patch window.fetch — covers SS.authHeaders() callers AND any bare fetch().
        var origFetch = window.fetch;
        window.fetch = function(input, init) {
            if (typeof input === 'string') input = appendToken(input);
            return origFetch.call(this, input, init);
        };

        // Patch HTMLMediaElement.src + HTMLImageElement.src setter (and getter
        // is a pass-through). HTMLAudioElement inherits from HTMLMediaElement.
        function patchSrc(proto) {
            var desc = Object.getOwnPropertyDescriptor(proto, 'src');
            if (!desc || !desc.set) return;
            var origSet = desc.set;
            Object.defineProperty(proto, 'src', {
                get: desc.get,
                set: function(v) { origSet.call(this, appendToken(v)); },
                configurable: true
            });
        }
        if (window.HTMLMediaElement) patchSrc(HTMLMediaElement.prototype);
        if (window.HTMLImageElement) patchSrc(HTMLImageElement.prototype);

        // Patch Element.setAttribute — catches innerHTML-built nodes
        // (e.g. results.js builds <audio src="..."> via string concatenation).
        var origSetAttr = Element.prototype.setAttribute;
        Element.prototype.setAttribute = function(name, value) {
            if (name === 'src' || name === 'href') value = appendToken(value);
            return origSetAttr.call(this, name, value);
        };

        // Retroactive sweep — innerHTML sets attributes via the parser, which
        // bypasses setAttribute. After any DOM mutation, rewrite stale audio /
        // image / anchor URLs that point at stemscriber.com without a token.
        function sweep(root) {
            (root || document).querySelectorAll('audio[src*="stemscriber.com"],img[src*="stemscriber.com"],a[href*="stemscriber.com"]').forEach(function(el) {
                var attr = el.tagName === 'A' ? 'href' : 'src';
                var v = el.getAttribute(attr);
                var v2 = appendToken(v);
                if (v2 !== v) el.setAttribute(attr, v2);
            });
        }
        if (document.body) sweep();
        document.addEventListener('DOMContentLoaded', function() { sweep(); });
        try {
            new MutationObserver(function(muts) {
                for (var i = 0; i < muts.length; i++) {
                    var m = muts[i];
                    if (m.addedNodes && m.addedNodes.length) sweep(m.target);
                }
            }).observe(document.documentElement, { childList: true, subtree: true });
        } catch (e) {}
    })();

    (function() {
        var vp = document.querySelector('meta[name="viewport"]');
        if (vp && vp.content.indexOf('viewport-fit') < 0) {
            vp.content = vp.content + ', viewport-fit=cover';
        }

        var s = document.createElement('style');
        s.textContent = [
            // Safe-area padding around the whole page
            'body{padding-top:env(safe-area-inset-top);padding-bottom:env(safe-area-inset-bottom);box-sizing:border-box;}',
            '.beta-card{margin-top:env(safe-area-inset-top);}',
            // Hide the heavy web footer inside the native app (every variant)
            '.landing-footer,.site-footer,footer.landing-footer,footer.site-footer,footer[class*="footer"]{display:none!important;}',
            // Defensive: hide the upload-options modal when aria-hidden (in case
            // upload-options-modal.js fails to load — left it always-visible on iOS once).
            '.upload-options-modal[aria-hidden="true"]{display:none!important;}',
            // Fixed-position side panels (Library, Settings, sign-in chooser, etc.)
            // sit at top:0 and ignore body padding — push them below the status bar.
            '.library-panel,.settings-panel,.share-panel,.save-panel,[class*="side-panel"]{padding-top:env(safe-area-inset-top)!important;padding-bottom:env(safe-area-inset-bottom)!important;box-sizing:border-box!important;}',
            // Landing nav (the top header with logo + hamburger + signin) — same
            // problem as the side panels: sits at top:0 and overlaps the Dynamic
            // Island/notch. Pad it down so the logo and the menu trigger are tappable.
            '.landing-nav,nav.landing-nav,header.landing-nav{padding-top:calc(env(safe-area-inset-top) + 0.6rem)!important;}',
            // Account profile dropdown — its parent sits near horizontal center
            // (under the StemScriber logo) so the default right:0 anchor pulls the
            // menu off the LEFT edge of the screen on iPhone. Pin it to the viewport
            // top-right so it always lands fully on screen.
            '.auth-profile-menu{position:fixed!important;top:calc(env(safe-area-inset-top) + 64px)!important;right:0.75rem!important;left:auto!important;max-width:calc(100vw - 1.5rem)!important;min-width:220px!important;}',
            // Profile button on mobile — un-hide the user name + chevron so the
            // entry has a visible label like the other nav items, not just a bare
            // avatar circle. The mobile media query in auth-ui.css hides them.
            '.auth-profile-btn .auth-profile-name{display:inline!important;}',
            '.auth-profile-btn .auth-profile-chevron{display:inline!important;}',
            // Static fallback label when the JS-bound .auth-profile-name is empty —
            // ensures the button reads "Account" even before currentUser is fetched.
            '.auth-profile-btn .auth-profile-name:empty::before{content:"Account";opacity:0.9;}',
            // Normalize padding across .library-btn / .settings-btn / .theme-toggle-btn.
            // Demo and Practice are <a> tags (browser default padding=0) but Library/
            // Settings/Theme are <button>s (default padding ~2px 6px) — same class but
            // different element = inconsistent left inset. Force the layout explicitly.
            'a.library-btn,button.library-btn,.library-btn,.settings-btn,.theme-toggle-btn{display:flex!important;align-items:center!important;padding:0.7rem 1rem!important;box-sizing:border-box!important;}',
            // Their close (X) buttons also need to clear the status bar
            '.library-panel .close-btn,.settings-panel .close-btn,.share-panel .close-btn,.save-panel .close-btn,.library-header .close-btn,.settings-header .close-btn{margin-top:env(safe-area-inset-top)!important;}',
            // Modal CTA rows clear the home indicator
            '.upload-options-content,.upload-options-actions{padding-bottom:calc(env(safe-area-inset-bottom) + 1rem)!important;}',
            '.modal-actions,.modal-buttons{padding-bottom:env(safe-area-inset-bottom)!important;}',
            // Toast/error notifications sit at bottom:2rem — lift them above the
            // home indicator so they are not clipped on notched devices.
            '.toast{bottom:calc(2rem + env(safe-area-inset-bottom))!important;}',
            // Magic-link sign-in modal (iOS only — replaces Google sign-in)
            '.ss-mlink-backdrop{position:fixed;inset:0;background:rgba(0,0,0,.78);z-index:99999;display:none;align-items:center;justify-content:center;padding:1.2rem;font-family:Space Grotesk,-apple-system,BlinkMacSystemFont,sans-serif;}',
            '.ss-mlink-backdrop.open{display:flex;}',
            '.ss-mlink-card{background:#1a1a24;border:1px solid #2a2a35;border-radius:18px;padding:1.8rem 1.6rem;max-width:380px;width:100%;color:#e8e4df;box-sizing:border-box;}',
            '.ss-mlink-title{font-family:Righteous,cursive;font-size:1.5rem;color:#ff7b54;margin:0 0 .4rem;text-align:center;}',
            '.ss-mlink-sub{color:#7a7a85;font-size:.92rem;margin:0 0 1.4rem;text-align:center;line-height:1.4;}',
            '.ss-mlink-input{width:100%;padding:.85rem 1rem;background:#0d0d12;border:1px solid #2a2a35;border-radius:10px;color:#fff;font-size:1rem;font-family:inherit;box-sizing:border-box;margin-bottom:.8rem;}',
            '.ss-mlink-input:focus{outline:none;border-color:#ff7b54;}',
            '.ss-mlink-input.code{text-align:center;font-family:SF Mono,Menlo,monospace;font-size:1.6rem;letter-spacing:.4rem;}',
            '.ss-mlink-btn{width:100%;padding:.95rem;background:linear-gradient(135deg,#ff7b54,#ff6b9d);color:#fff;border:none;border-radius:10px;font-size:1rem;font-weight:600;font-family:inherit;cursor:pointer;}',
            '.ss-mlink-btn:disabled{opacity:.55;cursor:default;}',
            '.ss-mlink-link{display:block;margin-top:1rem;text-align:center;color:#7a7a85;font-size:.85rem;text-decoration:underline;cursor:pointer;background:none;border:none;width:100%;font-family:inherit;}',
            '.ss-mlink-msg{margin-top:.8rem;font-size:.85rem;text-align:center;min-height:1.2em;}',
            '.ss-mlink-msg.err{color:#ff6b9d;}',
            '.ss-mlink-msg.ok{color:#00ff88;}'
        ].join('');
        (document.head || document.documentElement).appendChild(s);

        // ---- Magic-link sign-in modal (iOS / Android only) ---------------
        // Build the modal lazily on first call. Two states: "email" and "code".
        function ensureModal() {
            if (document.getElementById('ssMlinkBackdrop')) return;
            var bd = document.createElement('div');
            bd.id = 'ssMlinkBackdrop';
            bd.className = 'ss-mlink-backdrop';
            // Inject extra CSS for tabs + pw eye toggle (idempotent)
            if (!document.getElementById('ssMlinkExtraStyle')) {
                var xs = document.createElement('style');
                xs.id = 'ssMlinkExtraStyle';
                xs.textContent = [
                    '.ss-mlink-tabs{display:flex;gap:.25rem;background:#0d0d12;border-radius:10px;padding:.25rem;margin-bottom:1.2rem;}',
                    '.ss-mlink-tab{flex:1;padding:.55rem .4rem;background:transparent;border:none;color:#7a7a85;font-family:inherit;font-size:.88rem;font-weight:600;cursor:pointer;border-radius:8px;}',
                    '.ss-mlink-tab.active{background:#1a1a24;color:#ff7b54;}',
                    '.ss-mlink-pw-wrap{position:relative;margin-bottom:.7rem;}',
                    '.ss-mlink-pw-wrap .ss-mlink-input{margin-bottom:0;padding-right:2.4rem;}',
                    '.ss-mlink-eye{position:absolute;right:.55rem;top:50%;transform:translateY(-50%);background:none;border:0;padding:.3rem;cursor:pointer;color:#7a7a85;display:flex;align-items:center;justify-content:center;}',
                    '.ss-mlink-eye:hover{color:#ff7b54;}',
                    '.ss-mlink-hint{font-size:.72rem;color:#5a5a65;margin-top:-.3rem;margin-bottom:.7rem;padding-left:.2rem;}',
                ].join('');
                document.head.appendChild(xs);
            }
            bd.innerHTML = ''
                + '<div class="ss-mlink-card" role="dialog" aria-modal="true" aria-labelledby="ssMlinkTitle">'
                +   '<div class="ss-mlink-tabs" id="ssMlinkTabs">'
                +     '<button class="ss-mlink-tab active" data-mode="signin" type="button">Sign in</button>'
                +     '<button class="ss-mlink-tab" data-mode="signup" type="button">Create account</button>'
                +   '</div>'
                +   '<h2 id="ssMlinkTitle" class="ss-mlink-title">Welcome back</h2>'
                +   '<p id="ssMlinkSub" class="ss-mlink-sub">Sign in with your email and password.</p>'
                +   '<div id="ssMlinkStepEmail">'
                +     '<input id="ssMlinkEmail" class="ss-mlink-input" type="email" inputmode="email" autocomplete="email" autocapitalize="off" autocorrect="off" placeholder="you@example.com">'
                +     '<input id="ssMlinkName" class="ss-mlink-input" type="text" autocomplete="name" placeholder="Your name" style="display:none">'
                +     '<div class="ss-mlink-pw-wrap">'
                +       '<input id="ssMlinkPw" class="ss-mlink-input" type="password" autocomplete="current-password" placeholder="Password">'
                +       '<button type="button" class="ss-mlink-eye" data-target="ssMlinkPw" aria-label="Show or hide password" tabindex="-1"></button>'
                +     '</div>'
                +     '<div class="ss-mlink-pw-wrap" id="ssMlinkPwConfirmWrap" style="display:none">'
                +       '<input id="ssMlinkPwConfirm" class="ss-mlink-input" type="password" autocomplete="new-password" placeholder="Confirm password">'
                +       '<button type="button" class="ss-mlink-eye" data-target="ssMlinkPwConfirm" aria-label="Show or hide password" tabindex="-1"></button>'
                +     '</div>'
                +     '<div id="ssMlinkPwHint" class="ss-mlink-hint" style="display:none">At least 8 characters.</div>'
                +     '<button id="ssMlinkSubmitBtn" class="ss-mlink-btn" type="button">Sign in</button>'
                +     '<button id="ssMlinkSendBtn" class="ss-mlink-link" type="button" style="display:none">Send code</button>'
                +     '<button id="ssMlinkForgotBtn" class="ss-mlink-link" type="button">Forgot password? Get a code by email</button>'
                +   '</div>'
                +   '<div id="ssMlinkStepCode" style="display:none">'
                +     '<input id="ssMlinkCode" class="ss-mlink-input code" type="text" inputmode="numeric" autocomplete="one-time-code" pattern="[0-9]*" maxlength="6" placeholder="000000">'
                +     '<button id="ssMlinkVerifyBtn" class="ss-mlink-btn" type="button" disabled>Verify & sign in</button>'
                +     '<button id="ssMlinkBackBtn" class="ss-mlink-link" type="button">Use a different email</button>'
                +   '</div>'
                +   '<div id="ssMlinkMsg" class="ss-mlink-msg" aria-live="polite"></div>'
                +   '<button id="ssMlinkCloseBtn" class="ss-mlink-link" type="button">Cancel</button>'
                + '</div>';
            document.body.appendChild(bd);

            var API = window.StemScriber.API_BASE.replace(/\/api$/, '');
            var emailInput = bd.querySelector('#ssMlinkEmail');
            var nameInput = bd.querySelector('#ssMlinkName');
            var pwInput = bd.querySelector('#ssMlinkPw');
            var pwConfirmInput = bd.querySelector('#ssMlinkPwConfirm');
            var pwConfirmWrap = bd.querySelector('#ssMlinkPwConfirmWrap');
            var pwHint = bd.querySelector('#ssMlinkPwHint');
            var submitBtn = bd.querySelector('#ssMlinkSubmitBtn');
            var forgotBtn = bd.querySelector('#ssMlinkForgotBtn');
            var codeInput = bd.querySelector('#ssMlinkCode');
            var verifyBtn = bd.querySelector('#ssMlinkVerifyBtn');
            var backBtn = bd.querySelector('#ssMlinkBackBtn');
            var closeBtn = bd.querySelector('#ssMlinkCloseBtn');
            var msg = bd.querySelector('#ssMlinkMsg');
            var title = bd.querySelector('#ssMlinkTitle');
            var sub = bd.querySelector('#ssMlinkSub');
            var stepEmail = bd.querySelector('#ssMlinkStepEmail');
            var stepCode = bd.querySelector('#ssMlinkStepCode');
            var tabs = bd.querySelectorAll('.ss-mlink-tab');
            var currentEmail = '';
            var _mode = 'signin';

            function setMsg(text, kind) {
                msg.textContent = text || '';
                msg.className = 'ss-mlink-msg' + (kind ? ' ' + kind : '');
            }
            function close() {
                bd.classList.remove('open');
                setMsg('');
                emailInput.value = '';
                nameInput.value = '';
                pwInput.value = ''; pwInput.type = 'password';
                pwConfirmInput.value = ''; pwConfirmInput.type = 'password';
                codeInput.value = '';
                stepEmail.style.display = '';
                stepCode.style.display = 'none';
                verifyBtn.disabled = true;
                setMode('signin');
            }
            closeBtn.addEventListener('click', close);
            bd.addEventListener('click', function(e) { if (e.target === bd) close(); });

            // ---- Eye toggle on every password field --------------------------
            var EYE_OPEN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="18" height="18"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>';
            var EYE_CLOSED = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="18" height="18"><path d="M17.94 17.94A10.06 10.06 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>';
            bd.querySelectorAll('.ss-mlink-eye').forEach(function(btn) {
                btn.innerHTML = EYE_CLOSED;
                btn.addEventListener('click', function() {
                    var input = bd.querySelector('#' + btn.getAttribute('data-target'));
                    if (!input) return;
                    var showing = input.type === 'text';
                    input.type = showing ? 'password' : 'text';
                    btn.innerHTML = showing ? EYE_CLOSED : EYE_OPEN;
                });
            });

            function setMode(mode) {
                _mode = mode;
                setMsg('');
                tabs.forEach(function(t) { t.classList.toggle('active', t.getAttribute('data-mode') === mode); });
                if (mode === 'signin') {
                    title.textContent = 'Welcome back';
                    sub.textContent = 'Sign in with your email and password.';
                    nameInput.style.display = 'none';
                    pwInput.placeholder = 'Password';
                    pwInput.autocomplete = 'current-password';
                    pwConfirmWrap.style.display = 'none';
                    pwHint.style.display = 'none';
                    submitBtn.textContent = 'Sign in';
                    forgotBtn.style.display = '';
                } else {
                    title.textContent = 'Create your account';
                    sub.textContent = 'A password lets you sign in instantly next time.';
                    nameInput.style.display = '';
                    pwInput.placeholder = 'Choose a password';
                    pwInput.autocomplete = 'new-password';
                    pwConfirmWrap.style.display = '';
                    pwHint.style.display = '';
                    submitBtn.textContent = 'Create account';
                    forgotBtn.style.display = 'none';
                }
            }
            tabs.forEach(function(t) {
                t.addEventListener('click', function() { setMode(t.getAttribute('data-mode')); });
            });

            async function storeTokensAndGo(data) {
                try {
                    localStorage.setItem('access_token', data.access_token);
                    if (data.refresh_token) localStorage.setItem('refresh_token', data.refresh_token);
                    if (data.user) localStorage.setItem('ss_user_cache', JSON.stringify(data.user));
                    localStorage.setItem('ss_ios_signin_prompt', '1');
                } catch (e) {}
                try { document.body.classList.add('is-authed'); } catch (e) {}
                setMsg('Signed in! Loading…', 'ok');
                setTimeout(function() {
                    try { sessionStorage.setItem('ss_ios_landed', '1'); } catch (e) {}
                    try { window.location.href = 'index.html'; } catch (e) { try { window.location.reload(); } catch (e2) {} }
                }, 400);
            }

            // ---- Submit (Sign in / Create account) ---------------------------
            submitBtn.addEventListener('click', async function() {
                var email = (emailInput.value || '').trim().toLowerCase();
                var password = pwInput.value || '';
                if (!email || email.indexOf('@') < 0) { setMsg('Please enter a valid email.', 'err'); return; }
                if (!password) { setMsg('Please enter a password.', 'err'); return; }
                if (_mode === 'signup' && password.length < 8) { setMsg('Password must be at least 8 characters.', 'err'); return; }
                if (_mode === 'signup' && pwConfirmInput.value !== password) { setMsg("Passwords don't match. Try again.", 'err'); pwConfirmInput.focus(); return; }
                submitBtn.disabled = true;
                setMsg(_mode === 'signup' ? 'Creating account…' : 'Signing in…');
                try {
                    var body = { email: email, password: password };
                    var path;
                    if (_mode === 'signup') {
                        path = '/auth/register';
                        var name = (nameInput.value || '').trim();
                        if (name) body.display_name = name;
                    } else {
                        path = '/auth/login';
                    }
                    var res = await fetch(API + path, {
                        method: 'POST', cache: 'no-store',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(body),
                    });
                    var data = await res.json().catch(function() { return {}; });
                    if (res.ok && data.access_token) { await storeTokensAndGo(data); }
                    else { setMsg(data.error || (_mode === 'signup' ? 'Could not create account.' : 'Invalid email or password.'), 'err'); submitBtn.disabled = false; }
                } catch (e) {
                    setMsg('Network error. Try again.', 'err');
                    submitBtn.disabled = false;
                }
            });

            // ---- Forgot password (drops into code flow) ----------------------
            forgotBtn.addEventListener('click', async function() {
                var email = (emailInput.value || '').trim().toLowerCase();
                if (!email || email.indexOf('@') < 0) { emailInput.focus(); setMsg('Enter your email above first, then tap Forgot.', 'err'); return; }
                forgotBtn.disabled = true;
                setMsg('Sending reset code…');
                try {
                    await fetch(API + '/auth/forgot-password', {
                        method: 'POST', cache: 'no-store',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ email: email }),
                    });
                    currentEmail = email;
                    stepEmail.style.display = 'none';
                    stepCode.style.display = '';
                    title.textContent = 'Check your email';
                    sub.textContent = 'We sent a 6-digit code to ' + email + '.';
                    setMsg('');
                    codeInput.value = '';
                    verifyBtn.disabled = true;
                    codeInput.focus();
                } catch (e) {
                    setMsg('Could not send. Try again.', 'err');
                } finally {
                    forgotBtn.disabled = false;
                }
            });

            codeInput.addEventListener('input', function() {
                codeInput.value = codeInput.value.replace(/\D/g, '').slice(0, 6);
                verifyBtn.disabled = codeInput.value.length !== 6;
            });
            codeInput.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !verifyBtn.disabled) verifyBtn.click();
            });

            verifyBtn.addEventListener('click', async function() {
                var code = codeInput.value.trim();
                if (code.length !== 6) return;
                verifyBtn.disabled = true;
                setMsg('Verifying…');
                var verifyUrl = API + '/auth/magic-link/verify-code';
                try {
                    var res = await fetch(verifyUrl, {
                        method: 'POST',
                        cache: 'no-store',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ email: currentEmail, code: code })
                    });
                    var bodyText = await res.text();
                    var data = {};
                    try { data = JSON.parse(bodyText); } catch (e) {}
                    if (res.ok && data.access_token) {
                        try {
                            localStorage.setItem('access_token', data.access_token);
                            localStorage.setItem('refresh_token', data.refresh_token);
                            if (data.user) localStorage.setItem('ss_user_cache', JSON.stringify(data.user));
                            // Flag a one-step beta-code prompt to fire after reload
                            // (only shown to free-plan users — see initIosAccount).
                            localStorage.setItem('ss_ios_signin_prompt', '1');
                        } catch (e) {}
                        setMsg('Signed in! Loading your library…', 'ok');
                        // Navigate to the app dashboard (index.html) rather than
                        // reloading the current page — otherwise a magic-link sign-in
                        // initiated from the landing page leaves the user on the
                        // landing page after sign-in instead of in their library.
                        setTimeout(function() {
                            try {
                                // Clear the "first iOS load" flag so index.html's
                                // landing-redirect doesn't bounce them back.
                                sessionStorage.setItem('ss_ios_landed', '1');
                            } catch (e) {}
                            try {
                                window.location.href = 'index.html';
                            } catch (e) {
                                try { window.location.reload(); } catch (e2) {}
                            }
                        }, 600);
                    } else {
                        setMsg(data.error || 'Invalid or expired code.', 'err');
                        verifyBtn.disabled = false;
                    }
                } catch (e) {
                    setMsg('Network error. Try again.', 'err');
                    verifyBtn.disabled = false;
                }
            });

            backBtn.addEventListener('click', function() {
                stepCode.style.display = 'none';
                stepEmail.style.display = '';
                sub.textContent = 'Enter your email — we\'ll send you a 6-digit code.';
                setMsg('');
                emailInput.focus();
            });
        }

        function openModal() {
            ensureModal();
            var bd = document.getElementById('ssMlinkBackdrop');
            // Reset to sign-in tab + clear all fields on every open.
            var stepE = document.getElementById('ssMlinkStepEmail');
            var stepC = document.getElementById('ssMlinkStepCode');
            var title = document.getElementById('ssMlinkTitle');
            var sub = document.getElementById('ssMlinkSub');
            var msg = document.getElementById('ssMlinkMsg');
            var ei = document.getElementById('ssMlinkEmail');
            var ni = document.getElementById('ssMlinkName');
            var pw = document.getElementById('ssMlinkPw');
            var pwc = document.getElementById('ssMlinkPwConfirm');
            var pwcw = document.getElementById('ssMlinkPwConfirmWrap');
            var hint = document.getElementById('ssMlinkPwHint');
            var sb = document.getElementById('ssMlinkSubmitBtn');
            var fb = document.getElementById('ssMlinkForgotBtn');
            var ci = document.getElementById('ssMlinkCode');
            var vb = document.getElementById('ssMlinkVerifyBtn');
            var tabs = bd.querySelectorAll('.ss-mlink-tab');
            if (stepE) stepE.style.display = '';
            if (stepC) stepC.style.display = 'none';
            if (title) title.textContent = 'Welcome back';
            if (sub) sub.textContent = 'Sign in with your email and password.';
            if (msg) { msg.textContent = ''; msg.className = 'ss-mlink-msg'; }
            if (ei) ei.value = '';
            if (ni) { ni.value = ''; ni.style.display = 'none'; }
            if (pw) { pw.value = ''; pw.type = 'password'; pw.placeholder = 'Password'; pw.autocomplete = 'current-password'; }
            if (pwc) { pwc.value = ''; pwc.type = 'password'; }
            if (pwcw) pwcw.style.display = 'none';
            if (hint) hint.style.display = 'none';
            if (sb) { sb.textContent = 'Sign in'; sb.disabled = false; }
            if (fb) fb.style.display = '';
            if (ci) ci.value = '';
            if (vb) vb.disabled = true;
            tabs.forEach(function(t) { t.classList.toggle('active', t.getAttribute('data-mode') === 'signin'); });
            // Reset all eye toggles to "hidden" SVG
            bd.querySelectorAll('.ss-mlink-eye').forEach(function(b) {
                b.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="18" height="18"><path d="M17.94 17.94A10.06 10.06 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>';
            });
            bd.classList.add('open');
            setTimeout(function() {
                if (ei) ei.focus();
            }, 50);
        }

        // Expose globally
        window.triggerMagicLinkSignIn = openModal;

        // ---- Native Google Sign-In via @capgo/capacitor-social-login --------
        // On iOS we use the native Google SDK (system Safari handoff) instead
        // of the magic-link modal. Result is exchanged for our backend JWT via
        // POST /auth/google with the idToken. Falls back to the magic-link
        // modal if the plugin errors or the user cancels.
        var _socialLoginReady = false;
        async function _ensureSocialLoginInit() {
            if (_socialLoginReady) return true;
            try {
                var mod = window.Capacitor && window.Capacitor.Plugins && window.Capacitor.Plugins.SocialLogin;
                if (!mod) {
                    // Fallback: dynamically import the bundled plugin
                    var imp = await import('@capgo/capacitor-social-login');
                    mod = imp && imp.SocialLogin;
                    if (mod) window.SocialLoginPlugin = mod;
                }
                if (!mod) return false;
                await mod.initialize({
                    google: {
                        iOSClientId: '737338395840-h94ofut1e02amijpbhgrdlha99g8aujn.apps.googleusercontent.com',
                        // Removed iOSServerClientId — Google rejects when the
                        // iOSClientId and iOSServerClientId are in different
                        // GCP projects ("invalid_audience" runtime error). Our
                        // backend already accepts the iOS audience directly via
                        // GOOGLE_CLIENT_ID_EXTRAS, so no server-side audience
                        // exchange is needed.
                        mode: 'online',
                    },
                    apple: {
                        // iOS uses ASAuthorizationAppleIDProvider — no client ID
                        // needed here. Web (when we ship it) will use the Services
                        // ID `com.kozelski.stemscriber.signin` and redirect URL
                        // `https://stemscriber.com/auth/apple/callback`.
                        clientId: 'com.kozelski.stemscriber',
                    },
                });
                window.SocialLoginPlugin = mod;
                _socialLoginReady = true;
                return true;
            } catch (e) {
                console.warn('[Auth iOS] SocialLogin init failed:', e && e.message);
                return false;
            }
        }

        async function triggerNativeGoogleSignIn() {
            var ok = await _ensureSocialLoginInit();
            if (!ok) {
                alert('Google sign-in plugin failed to load. Falling back to email sign-in.');
                openModal();
                return;
            }
            var res, data, resp;
            try {
                res = await window.SocialLoginPlugin.login({
                    provider: 'google',
                    options: { scopes: ['email', 'profile'] },
                });
            } catch (e) {
                if (e && (e.message || '').toLowerCase().indexOf('cancel') < 0) {
                    alert('Google plugin error: ' + (e && (e.message || e.errorMessage || JSON.stringify(e))));
                }
                return;
            }
            var idToken = res && res.result && res.result.idToken;
            if (!idToken) {
                alert('Google returned no idToken. Response: ' + JSON.stringify(res));
                return;
            }
            try {
                var apiBase = (window.StemScriber && window.StemScriber.API_BASE) || 'https://stemscriber.com/api';
                resp = await fetch(apiBase.replace(/\/api$/, '') + '/auth/google', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ credential: idToken }),
                });
                data = await resp.json().catch(function() { return {}; });
            } catch (e) {
                alert('Network error calling backend /auth/google: ' + (e && e.message));
                return;
            }
            if (!resp.ok || !data.access_token) {
                alert('Backend rejected Google token (HTTP ' + resp.status + '): ' + (data && data.error || 'no error message'));
                return;
            }
            try {
                localStorage.setItem('access_token', data.access_token);
                if (data.refresh_token) localStorage.setItem('refresh_token', data.refresh_token);
                if (data.user) localStorage.setItem('ss_user_cache', JSON.stringify(data.user));
                sessionStorage.setItem('ss_ios_landed', '1');
            } catch (e) {}
            try { document.body.classList.add('is-authed'); } catch (e) {}
            window.location.href = 'index.html';
        }
        window.triggerGoogleSignIn = triggerNativeGoogleSignIn;

        // ---- Sign in with Apple (native, iOS) -------------------------------
        // Apple's identity token comes back as a JWT we forward to POST /auth/apple
        // for verification + JWT issuance. First sign-in includes the user's name
        // and email; subsequent sign-ins do NOT — the backend already has them
        // stored against the apple_id `sub`.
        async function triggerNativeAppleSignIn() {
            var ok = await _ensureSocialLoginInit();
            if (!ok) {
                alert('Apple sign-in plugin failed to load. Falling back to email sign-in.');
                openModal();
                return;
            }
            var res, data, resp;
            try {
                res = await window.SocialLoginPlugin.login({
                    provider: 'apple',
                    options: { scopes: ['email', 'name'] },
                });
            } catch (e) {
                if (e && (e.message || '').toLowerCase().indexOf('cancel') < 0) {
                    alert('Apple plugin error: ' + (e && (e.message || e.errorMessage || JSON.stringify(e))));
                }
                return;
            }
            var r = (res && res.result) || {};
            var identityToken = r.identityToken || r.idToken;
            if (!identityToken) {
                alert('Apple returned no identityToken. Response: ' + JSON.stringify(res));
                return;
            }
            var body = { identity_token: identityToken };
            if (r.givenName || r.familyName || r.email) {
                body.user = {
                    name: { firstName: r.givenName || '', lastName: r.familyName || '' },
                    email: r.email || '',
                };
            }
            try {
                var apiBase = (window.StemScriber && window.StemScriber.API_BASE) || 'https://stemscriber.com/api';
                resp = await fetch(apiBase.replace(/\/api$/, '') + '/auth/apple', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                });
                data = await resp.json().catch(function() { return {}; });
            } catch (e) {
                alert('Network error calling backend /auth/apple: ' + (e && e.message));
                return;
            }
            if (!resp.ok || !data.access_token) {
                alert('Backend rejected Apple token (HTTP ' + resp.status + '): ' + (data && data.error || 'no error message'));
                return;
            }
            try {
                localStorage.setItem('access_token', data.access_token);
                if (data.refresh_token) localStorage.setItem('refresh_token', data.refresh_token);
                if (data.user) localStorage.setItem('ss_user_cache', JSON.stringify(data.user));
                sessionStorage.setItem('ss_ios_landed', '1');
            } catch (e) {}
            try { document.body.classList.add('is-authed'); } catch (e) {}
            window.location.href = 'index.html';
        }
        window.triggerAppleSignIn = triggerNativeAppleSignIn;

        // Reveal .nav-authed-only items (Library, Practice, Demo, Settings) ONLY
        // when the user has a real backend session (access_token present).
        // Beta-code redemption alone does NOT count — those endpoints need a JWT,
        // and showing the buttons without working backend = "Failed to load library"
        // for every tap. Magic-link sign-in is the gate.
        function applyAuthedClass() {
            try {
                var hasAccessToken = !!localStorage.getItem('access_token');
                if (hasAccessToken && document.body) {
                    document.body.classList.add('is-authed');
                } else if (document.body) {
                    document.body.classList.remove('is-authed');
                }
            } catch (e) {}
        }

        // Rewire the Google sign-in button to call the native Google plugin.
        // Also inject a smaller "Sign in with Email" link next to it so users
        // who don't want to go through Google can still use magic-link.
        var GOOGLE_ICON = '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" focusable="false" style="flex-shrink:0;"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 01-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>';
        var EMAIL_ICON = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false" style="flex-shrink:0;"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="m2 7 10 6 10-6"/></svg>';
        function relabelSignInButtons() {
            var btns = document.querySelectorAll('.auth-signin-btn');
            for (var i = 0; i < btns.length; i++) {
                var btn = btns[i];
                if (btn.getAttribute('data-ss-relabeled') === '1') continue;
                btn.innerHTML = GOOGLE_ICON + '<span>Sign in with Google</span>';
                btn.style.display = 'inline-flex';
                btn.style.alignItems = 'center';
                btn.style.gap = '0.4rem';
                btn.onclick = function(e) { e && e.preventDefault && e.preventDefault(); triggerNativeGoogleSignIn(); };
                btn.setAttribute('data-ss-relabeled', '1');

                // Inject Sign-in-with-Apple + Email buttons alongside Google.
                // Order is Apple → Email → Google (Apple first per HIG guidance
                // when offering social sign-in on iOS).
                if (!btn.parentNode.querySelector('[data-ss-apple-btn]')) {
                    var appleBtn = document.createElement('button');
                    appleBtn.type = 'button';
                    appleBtn.setAttribute('data-ss-apple-btn', '1');
                    appleBtn.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true" focusable="false" style="flex-shrink:0;"><path d="M17.05 12.04c-.03-2.71 2.22-4.02 2.32-4.08-1.27-1.85-3.24-2.1-3.94-2.13-1.68-.17-3.27 1-4.12 1-.86 0-2.18-.98-3.58-.95-1.84.03-3.54 1.07-4.48 2.71-1.91 3.32-.49 8.22 1.37 10.9.91 1.31 2 2.78 3.4 2.73 1.36-.06 1.88-.88 3.53-.88 1.65 0 2.11.88 3.55.86 1.46-.03 2.39-1.34 3.29-2.66 1.04-1.53 1.47-3.01 1.49-3.09-.03-.01-2.85-1.09-2.88-4.31zM14.4 4.07c.74-.9 1.24-2.16 1.1-3.41-1.07.04-2.36.71-3.13 1.61-.69.8-1.29 2.07-1.13 3.3 1.19.09 2.41-.6 3.16-1.5z"/></svg><span>Sign in with Apple</span>';
                    appleBtn.style.cssText = 'display:inline-flex;align-items:center;gap:0.4rem;margin-left:.5rem;padding:.5rem 0.9rem;background:#000;border:1px solid #000;color:#fff;border-radius:8px;font-family:"Space Grotesk",sans-serif;font-size:.85rem;font-weight:600;cursor:pointer;';
                    appleBtn.onclick = function(e) { e && e.preventDefault && e.preventDefault(); triggerNativeAppleSignIn(); };
                    btn.parentNode.insertBefore(appleBtn, btn.nextSibling);
                }
                if (!btn.parentNode.querySelector('[data-ss-email-link]')) {
                    var emailBtn = document.createElement('button');
                    emailBtn.type = 'button';
                    emailBtn.setAttribute('data-ss-email-link', '1');
                    emailBtn.innerHTML = EMAIL_ICON + '<span>Email</span>';
                    emailBtn.style.cssText = 'display:inline-flex;align-items:center;gap:0.35rem;margin-left:.5rem;padding:.45rem .8rem;background:rgba(255,123,84,0.12);border:1px solid rgba(255,123,84,0.4);color:#ff7b54;border-radius:8px;font-family:"Space Grotesk",sans-serif;font-size:.85rem;font-weight:600;cursor:pointer;';
                    emailBtn.onclick = openModal;
                    // Place email link AFTER the Apple button
                    var appleNeighbour = btn.parentNode.querySelector('[data-ss-apple-btn]');
                    btn.parentNode.insertBefore(emailBtn, appleNeighbour ? appleNeighbour.nextSibling : btn.nextSibling);
                }
            }
        }

        // ---- Focused sign-in screen (iOS only) ------------------------------
        // When the user is NOT signed in, show ONLY a clean centered card with
        // the three sign-in options (Apple / Email / Google). Hide the
        // marketing hero, the demo card, the upload area, the nav — all of it.
        // Once signed in, the overlay hides and the normal app UI shows.
        function ensureSigninOverlay() {
            if (document.getElementById('ssSigninOverlay')) return;
            var style = document.createElement('style');
            style.textContent = [
                '#ssSigninOverlay{position:fixed;inset:0;background:linear-gradient(180deg,#0d0d12 0%,#1a1a24 100%);z-index:9998;display:none;flex-direction:column;align-items:center;justify-content:center;padding:calc(env(safe-area-inset-top) + 1.5rem) 1.5rem calc(env(safe-area-inset-bottom) + 1.5rem);font-family:"Space Grotesk",-apple-system,BlinkMacSystemFont,sans-serif;overflow-y:auto;}',
                // Removed `body:not(.is-authed) #ssSigninOverlay{display:flex}`
                // The app now opens to the landing page (with Sign in / nav
                // buttons up top) and the overlay only shows when the user
                // explicitly taps a sign-in button — same flow as web.
                '#ssSigninOverlay .sso-logo{font-family:Righteous,cursive;font-size:2.4rem;color:#ff7b54;margin:0 0 .25rem;text-align:center;letter-spacing:-.02em;}',
                '#ssSigninOverlay .sso-tag{display:inline-block;padding:2px 10px;background:linear-gradient(135deg,#ff7b54,#ff6b9d);color:#0d0d12;font-weight:700;font-size:.7rem;letter-spacing:.12em;border-radius:4px;margin:0 auto 2rem;}',
                '#ssSigninOverlay .sso-sub{color:#a8a8b3;font-size:1rem;margin:0 0 2.2rem;text-align:center;line-height:1.4;max-width:340px;}',
                '#ssSigninOverlay .sso-btns{width:100%;max-width:340px;display:flex;flex-direction:column;gap:.7rem;}',
                '#ssSigninOverlay .sso-btn{display:flex;align-items:center;justify-content:center;gap:.6rem;width:100%;padding:1rem;border-radius:12px;font-family:inherit;font-size:1rem;font-weight:600;cursor:pointer;border:1px solid transparent;}',
                '#ssSigninOverlay .sso-btn.apple{background:#fff;color:#000;}',
                '#ssSigninOverlay .sso-btn.google{background:#fff;color:#3c4043;}',
                '#ssSigninOverlay .sso-btn.email{background:transparent;color:#ff7b54;border-color:#ff7b54;}',
                '#ssSigninOverlay .sso-foot{margin-top:1.8rem;color:#5a5a65;font-size:.78rem;text-align:center;line-height:1.5;max-width:300px;}',
                // While unauthed, hide everything outside the overlay so the
                // marketing hero / demo card / upload area / nav don't peek
                // through.
                // Also removed the "hide everything outside overlay when not
                // authed" rule. Users now see the full landing page (hero +
                // demo card + features) before signing in.
            ].join('');
            document.head.appendChild(style);

            var overlay = document.createElement('div');
            overlay.id = 'ssSigninOverlay';
            overlay.innerHTML = ''
                + '<h1 class="sso-logo">StemScriber</h1>'
                + '<span class="sso-tag">BETA</span>'
                + '<p class="sso-sub">Sign in to upload songs, save your library, and edit chord charts.</p>'
                + '<div class="sso-btns">'
                +   '<button type="button" class="sso-btn apple" id="ssoApple">'
                +     '<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor" aria-hidden="true" focusable="false"><path d="M17.05 12.04c-.03-2.71 2.22-4.02 2.32-4.08-1.27-1.85-3.24-2.1-3.94-2.13-1.68-.17-3.27 1-4.12 1-.86 0-2.18-.98-3.58-.95-1.84.03-3.54 1.07-4.48 2.71-1.91 3.32-.49 8.22 1.37 10.9.91 1.31 2 2.78 3.4 2.73 1.36-.06 1.88-.88 3.53-.88 1.65 0 2.11.88 3.55.86 1.46-.03 2.39-1.34 3.29-2.66 1.04-1.53 1.47-3.01 1.49-3.09-.03-.01-2.85-1.09-2.88-4.31zM14.4 4.07c.74-.9 1.24-2.16 1.1-3.41-1.07.04-2.36.71-3.13 1.61-.69.8-1.29 2.07-1.13 3.3 1.19.09 2.41-.6 3.16-1.5z"/></svg>'
                +     '<span>Sign in with Apple</span>'
                +   '</button>'
                +   '<button type="button" class="sso-btn email" id="ssoEmail">'
                +     '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="m2 7 10 6 10-6"/></svg>'
                +     '<span>Sign in with Email</span>'
                +   '</button>'
                +   '<button type="button" class="sso-btn google" id="ssoGoogle">'
                +     '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" focusable="false"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 01-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>'
                +     '<span>Sign in with Google</span>'
                +   '</button>'
                + '</div>'
                + '<p class="sso-foot">By signing in you agree to StemScriber’s terms of service and privacy policy.</p>';
            document.body.appendChild(overlay);

            overlay.querySelector('#ssoApple').addEventListener('click', function() { triggerNativeAppleSignIn(); });
            overlay.querySelector('#ssoEmail').addEventListener('click', function() { openModal(); });
            overlay.querySelector('#ssoGoogle').addEventListener('click', function() { triggerNativeGoogleSignIn(); });
        }

        // ---- iOS account UI: Settings account section + beta-code modal -----
        // Self-contained so it doesn't depend on the main auth IIFE having run
        // (load order on iOS is not guaranteed). Talks to the backend directly.
        var API_ROOT = window.StemScriber.API_BASE.replace(/\/api$/, '');
        function getToken() { try { return localStorage.getItem('access_token'); } catch (e) { return null; } }

        var _userCache = null;
        async function fetchUser(force) {
            if (_userCache && !force) return _userCache;
            var tok = getToken();
            if (!tok) return null;
            try {
                var res = await fetch(API_ROOT + '/auth/me', { headers: { 'Authorization': 'Bearer ' + tok } });
                if (res.ok) {
                    var data = await res.json();
                    _userCache = data.user || data;
                    try { localStorage.setItem('ss_user_cache', JSON.stringify(_userCache)); } catch (e) {}
                    return _userCache;
                }
            } catch (e) {}
            // Network/offline fallback: last known user
            try { var c = localStorage.getItem('ss_user_cache'); if (c) { _userCache = JSON.parse(c); return _userCache; } } catch (e) {}
            return null;
        }

        async function doSignOut() {
            var tok = getToken();
            try { if (tok) await fetch(API_ROOT + '/auth/logout', { method: 'POST', headers: { 'Authorization': 'Bearer ' + tok } }); } catch (e) {}
            try {
                localStorage.removeItem('access_token');
                localStorage.removeItem('refresh_token');
                localStorage.removeItem('ss_user_cache');
                localStorage.removeItem('stemscribe-beta');
            } catch (e) {}
            try { window.location.reload(); } catch (e) {}
        }

        // Beta-code modal — reuses the .ss-mlink-* styles. Used by the Settings
        // "Redeem beta code" button and the one-step post-sign-in prompt.
        function ensureBetaModal() {
            if (document.getElementById('ssBetaBackdrop')) return;
            var bd = document.createElement('div');
            bd.id = 'ssBetaBackdrop';
            bd.className = 'ss-mlink-backdrop';
            bd.innerHTML = ''
                + '<div class="ss-mlink-card" role="dialog" aria-modal="true" aria-labelledby="ssBetaTitle">'
                +   '<h2 id="ssBetaTitle" class="ss-mlink-title">Got a beta code?</h2>'
                +   '<p class="ss-mlink-sub">Enter it to unlock Lifetime access.</p>'
                +   '<input id="ssBetaInput" class="ss-mlink-input" type="text" autocapitalize="characters" autocomplete="off" autocorrect="off" spellcheck="false" placeholder="YOUR CODE" style="text-align:center;text-transform:uppercase;letter-spacing:.15rem;">'
                +   '<button id="ssBetaRedeemBtn" class="ss-mlink-btn" type="button">Redeem</button>'
                +   '<div id="ssBetaMsg" class="ss-mlink-msg" aria-live="polite"></div>'
                +   '<button id="ssBetaSkipBtn" class="ss-mlink-link" type="button">Maybe later</button>'
                + '</div>';
            document.body.appendChild(bd);

            var input = bd.querySelector('#ssBetaInput');
            var redeemBtn = bd.querySelector('#ssBetaRedeemBtn');
            var skipBtn = bd.querySelector('#ssBetaSkipBtn');
            var msg = bd.querySelector('#ssBetaMsg');
            function setBetaMsg(t, k) { msg.textContent = t || ''; msg.className = 'ss-mlink-msg' + (k ? ' ' + k : ''); }
            function closeBeta() { bd.classList.remove('open'); setBetaMsg(''); input.value = ''; redeemBtn.disabled = false; }
            skipBtn.addEventListener('click', closeBeta);
            bd.addEventListener('click', function(e) { if (e.target === bd) closeBeta(); });
            input.addEventListener('keydown', function(e) { if (e.key === 'Enter') redeemBtn.click(); });

            redeemBtn.addEventListener('click', async function() {
                var code = (input.value || '').trim();
                if (!code) { setBetaMsg('Enter a code.', 'err'); return; }
                var user = await fetchUser();
                if (!user || !getToken()) { setBetaMsg('Please sign in first.', 'err'); return; }
                redeemBtn.disabled = true;
                setBetaMsg('Redeeming…');
                try {
                    var res = await fetch(API_ROOT + '/api/beta/redeem', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + getToken() },
                        body: JSON.stringify({ code: code, email: user.email })
                    });
                    var data = await res.json().catch(function() { return {}; });
                    if (res.ok && data.valid) {
                        setBetaMsg('Unlocked! Refreshing…', 'ok');
                        // Keep stemscribe-beta in sync so upload.js sees the new plan
                        try { var b = JSON.parse(localStorage.getItem('stemscribe-beta') || '{}'); b.plan = data.plan || 'lifetime'; localStorage.setItem('stemscribe-beta', JSON.stringify(b)); } catch (e) {}
                        setTimeout(function() { try { window.location.reload(); } catch (e) {} }, 800);
                    } else {
                        setBetaMsg(data.error || data.message || 'Invalid or used code.', 'err');
                        redeemBtn.disabled = false;
                    }
                } catch (e) {
                    setBetaMsg('Network error. Try again.', 'err');
                    redeemBtn.disabled = false;
                }
            });
        }

        function openBetaModal() {
            ensureBetaModal();
            var bd = document.getElementById('ssBetaBackdrop');
            bd.classList.add('open');
            setTimeout(function() { var i = document.getElementById('ssBetaInput'); if (i) i.focus(); }, 50);
        }
        window.ssOpenBetaModal = openBetaModal;

        // Inject Account section at the top of the Settings panel (authed only).
        function injectAccountSection(user) {
            var panel = document.getElementById('settingsPanel');
            if (!panel || !user) return;
            var planRaw = (user.plan || 'free');
            var planLabel = planRaw.charAt(0).toUpperCase() + planRaw.slice(1);
            var isFree = (planRaw === 'free');
            var section = document.getElementById('ssAccountSection');
            if (!section) {
                section = document.createElement('div');
                section.className = 'settings-section';
                section.id = 'ssAccountSection';
                var header = panel.querySelector('.settings-header');
                if (header && header.nextSibling) panel.insertBefore(section, header.nextSibling);
                else panel.insertBefore(section, panel.firstChild);
            }
            section.innerHTML = ''
                + '<h4>👤 Account</h4>'
                + '<div class="status-card">'
                +   '<div class="status-row"><span class="status-label">Email</span><span class="status-value" id="ssAcctEmail"></span></div>'
                +   '<div class="status-row"><span class="status-label">Plan</span><span class="status-value" id="ssAcctPlan"></span></div>'
                + '</div>'
                + (isFree ? '<button class="settings-btn-action primary" id="ssRedeemBtn" type="button">🎟️ Redeem beta code</button>' : '')
                + '<button class="settings-btn-action secondary" id="ssSignOutBtn" type="button" style="margin-top:.5rem;">Sign out</button>';
            // Set user text via textContent (never innerHTML) to avoid injection.
            var em = section.querySelector('#ssAcctEmail'); if (em) em.textContent = user.email || '';
            var pl = section.querySelector('#ssAcctPlan'); if (pl) pl.textContent = planLabel;
            var rb = section.querySelector('#ssRedeemBtn'); if (rb) rb.onclick = openBetaModal;
            var so = section.querySelector('#ssSignOutBtn'); if (so) so.onclick = doSignOut;
        }

        function initIosAccount() {
            if (!getToken()) return;
            fetchUser().then(function(user) {
                if (!user) return;
                injectAccountSection(user);
                // One-step post-sign-in beta prompt: only for free users, once.
                try {
                    if (localStorage.getItem('ss_ios_signin_prompt') === '1') {
                        localStorage.removeItem('ss_ios_signin_prompt');
                        if ((user.plan || 'free') === 'free') openBetaModal();
                    }
                } catch (e) {}
            });
        }

        if (document.body) { applyAuthedClass(); relabelSignInButtons(); initIosAccount(); ensureSigninOverlay(); }
        document.addEventListener('DOMContentLoaded', function() {
            applyAuthedClass();
            relabelSignInButtons();
            initIosAccount();
            ensureSigninOverlay();
        });
    })();
}

(function(SS) {
    'use strict';

    // ---- Auth State ----
    SS.currentUser = null;
    SS.accessToken = localStorage.getItem('access_token');
    SS.refreshToken = localStorage.getItem('refresh_token');
    var _googleClientId = null;
    var _googleInitialized = false;

    // ---- Fetch Google Client ID from backend ----
    async function fetchConfig() {
        try {
            var base = SS.API_BASE || '/api';
            var res = await fetch(base + '/config');
            if (res.ok) {
                var data = await res.json();
                _googleClientId = data.google_client_id || null;
            }
        } catch (e) {
            console.log('[Auth] Config endpoint unavailable:', e.message);
        }
    }

    // ---- Initialize Google Identity Services ----
    function initGoogleSignIn() {
        if (_googleInitialized || !_googleClientId) return;
        if (typeof google === 'undefined' || !google.accounts || !google.accounts.id) {
            // GSI library not loaded yet, retry shortly
            setTimeout(initGoogleSignIn, 200);
            return;
        }
        google.accounts.id.initialize({
            client_id: _googleClientId,
            callback: handleGoogleCredential,
            auto_select: true,
            cancel_on_tap_outside: true,
            context: 'signin',
            ux_mode: 'popup'
        });
        _googleInitialized = true;
        console.log('[Auth] Google Sign-In initialized');
    }

    // ---- Handle Google credential response ----
    async function handleGoogleCredential(response) {
        try {
            var base = SS.API_BASE || '/api';
            var res = await fetch(base.replace('/api', '') + '/auth/google', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ credential: response.credential })
            });
            var data = await res.json();
            if (data.access_token) {
                SS.accessToken = data.access_token;
                localStorage.setItem('access_token', data.access_token);
                if (data.refresh_token) {
                    SS.refreshToken = data.refresh_token;
                    localStorage.setItem('refresh_token', data.refresh_token);
                }
                SS.currentUser = data.user;
                updateAuthUI();
                console.log('[Auth] Signed in as', SS.currentUser.email);

                // If there was a pending save prompt, close it
                closeSavePrompt();
            } else {
                console.error('[Auth] Google sign-in failed:', data.error || 'Unknown error');
            }
        } catch (e) {
            console.error('[Auth] Google sign-in request failed:', e);
        }
    }

    // ---- Check auth state on page load ----
    SS.checkAuth = async function() {
        if (!SS.accessToken) {
            updateAuthUI();
            return;
        }
        try {
            var base = SS.API_BASE || '/api';
            var res = await fetch(base.replace('/api', '') + '/auth/me', {
                headers: { 'Authorization': 'Bearer ' + SS.accessToken }
            });
            if (res.ok) {
                var data = await res.json();
                SS.currentUser = data.user;
                updateAuthUI();
            } else if (res.status === 401) {
                // Token expired, try refresh
                var refreshed = await refreshAccessToken();
                if (!refreshed) {
                    clearAuthState();
                    updateAuthUI();
                }
            } else {
                updateAuthUI();
            }
        } catch (e) {
            console.log('[Auth] Auth check failed:', e.message);
            updateAuthUI();
        }
    };

    // ---- Refresh access token ----
    async function refreshAccessToken() {
        if (!SS.refreshToken) return false;
        try {
            var base = SS.API_BASE || '/api';
            var res = await fetch(base.replace('/api', '') + '/auth/refresh', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + SS.refreshToken
                }
            });
            if (res.ok) {
                var data = await res.json();
                SS.accessToken = data.access_token;
                localStorage.setItem('access_token', data.access_token);
                if (data.refresh_token) {
                    SS.refreshToken = data.refresh_token;
                    localStorage.setItem('refresh_token', data.refresh_token);
                }
                SS.currentUser = data.user || SS.currentUser;
                updateAuthUI();
                return true;
            }
        } catch (e) {
            console.log('[Auth] Token refresh failed:', e.message);
        }
        return false;
    }

    // ---- Clear auth state ----
    function clearAuthState() {
        SS.accessToken = null;
        SS.refreshToken = null;
        SS.currentUser = null;
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
    }

    // ---- Logout ----
    SS.logout = async function() {
        try {
            var base = SS.API_BASE || '/api';
            if (SS.accessToken) {
                await fetch(base.replace('/api', '') + '/auth/logout', {
                    method: 'POST',
                    headers: { 'Authorization': 'Bearer ' + SS.accessToken }
                });
            }
        } catch (e) {
            // Logout request failed, clear local state anyway
        }
        clearAuthState();

        // Revoke Google session
        if (_googleInitialized && google && google.accounts && google.accounts.id) {
            google.accounts.id.disableAutoSelect();
        }

        updateAuthUI();
        console.log('[Auth] Signed out');
    };

    // ---- Trigger Google Sign-In ----
    SS.triggerGoogleSignIn = function() {
        if (!_googleInitialized || !_googleClientId) {
            console.warn('[Auth] Google Sign-In not initialized yet');
            return;
        }

        // On mobile, prompt() is often blocked by Safari/Chrome popup blockers.
        // Use a hidden rendered button as fallback.
        var isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

        if (isMobile) {
            // Render a temporary Google button and click it programmatically
            _renderAndClickGoogleButton();
        } else {
            google.accounts.id.prompt(function(notification) {
                if (notification.isNotDisplayed()) {
                    console.log('[Auth] Prompt blocked, falling back to rendered button');
                    _renderAndClickGoogleButton();
                }
                if (notification.isSkippedMoment()) {
                    console.log('[Auth] Google prompt skipped:', notification.getSkippedReason());
                    _renderAndClickGoogleButton();
                }
            });
        }
    };

    // ---- Fallback: render a real Google button and show it ----
    function _renderAndClickGoogleButton() {
        // Create or reuse a container for the Google button
        var container = document.getElementById('googleSignInFallback');
        if (!container) {
            container = document.createElement('div');
            container.id = 'googleSignInFallback';
            container.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.7);display:flex;align-items:center;justify-content:center;z-index:10000;';
            // Close on backdrop click
            container.addEventListener('click', function(e) {
                if (e.target === container) container.remove();
            });
            var inner = document.createElement('div');
            inner.style.cssText = 'background:#1a1a2e;border-radius:16px;padding:32px;text-align:center;max-width:320px;width:90%;';
            inner.innerHTML = '<p style="color:#fff;margin:0 0 20px;font-family:Outfit,sans-serif;font-size:1.1rem;">Sign in with Google</p><div id="googleBtnTarget"></div><p style="color:#888;margin:16px 0 0;font-size:0.85rem;cursor:pointer;" onclick="this.closest(\'#googleSignInFallback\').remove()">Cancel</p>';
            container.appendChild(inner);
            document.body.appendChild(container);
        } else {
            container.style.display = 'flex';
        }

        // Render the official Google button into the target div
        var target = document.getElementById('googleBtnTarget');
        if (target) {
            target.innerHTML = '';
            google.accounts.id.renderButton(target, {
                theme: 'filled_black',
                size: 'large',
                width: 260,
                text: 'signin_with',
                shape: 'pill'
            });
        }
    }

    // ---- Auth headers helper ----
    SS.authHeaders = function() {
        var headers = {};
        if (SS.accessToken) {
            headers['Authorization'] = 'Bearer ' + SS.accessToken;
        }
        return headers;
    };

    // ---- Beta code redemption ----
    SS.redeemBetaCode = async function(code) {
        if (!SS.currentUser) {
            SS.triggerGoogleSignIn();
            return;
        }
        try {
            var base = SS.API_BASE || '/api';
            var res = await fetch(base + '/beta/redeem', {
                method: 'POST',
                headers: Object.assign({ 'Content-Type': 'application/json' }, SS.authHeaders()),
                body: JSON.stringify({ code: code, email: SS.currentUser.email })
            });
            var data = await res.json();
            if (data.success) {
                // Refresh user info to get updated plan
                await SS.checkAuth();
                showToast('Beta code redeemed successfully!');
            } else {
                showToast(data.error || 'Invalid beta code', 'error');
            }
        } catch (e) {
            showToast('Failed to redeem code', 'error');
        }
    };

    // ---- Show toast notification (if available) ----
    function showToast(message, type) {
        var toastEl = document.getElementById('toast');
        var toastMsg = document.getElementById('toastMessage');
        var toastIcon = document.getElementById('toastIcon');
        if (toastEl && toastMsg) {
            toastMsg.textContent = message;
            if (toastIcon) toastIcon.textContent = type === 'error' ? '!' : '>';
            toastEl.classList.add('show');
            setTimeout(function() { toastEl.classList.remove('show'); }, 3000);
        }
    }

    // ---- Update UI based on auth state ----
    function updateAuthUI() {
        var signInBtns = document.querySelectorAll('.auth-signin-btn');
        var profileDropdowns = document.querySelectorAll('.auth-profile-dropdown');

        if (SS.currentUser) {
            // Reveal header nav items gated behind .nav-authed-only (Library,
            // Practice, Demo, Settings). Without this body class the CSS at the
            // top of index.html keeps them hidden even after sign-in.
            try { document.body.classList.add('is-authed'); } catch (e) {}
            // Hide sign-in buttons, show profile dropdowns.
            // Also hide the injected Apple + Email buttons (they're not
            // .auth-signin-btn but get added next to it in the iOS Capacitor
            // block — without this fix they keep showing after sign-in).
            signInBtns.forEach(function(btn) { btn.style.display = 'none'; });
            document.querySelectorAll('[data-ss-apple-btn],[data-ss-email-link]').forEach(function(el) { el.style.display = 'none'; });
            profileDropdowns.forEach(function(dd) {
                dd.style.display = 'flex';
                // Update avatar — letter-initial fallback when no profile picture or image fails
                var avatar = dd.querySelector('.auth-profile-avatar');
                var gIcon = dd.querySelector('.auth-google-profile-icon');
                var letterAvatar = dd.querySelector('.auth-letter-avatar');

                function _showLetterAvatar() {
                    if (avatar) avatar.style.display = 'none';
                    if (gIcon) gIcon.style.display = 'none';
                    if (!letterAvatar) {
                        letterAvatar = document.createElement('span');
                        letterAvatar.className = 'auth-letter-avatar';
                        letterAvatar.style.cssText = 'display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:50%;background:linear-gradient(135deg,#ff7b54,#ff6b9d);color:#fff;font-weight:700;font-size:0.85rem;flex-shrink:0;';
                        if (avatar) avatar.parentNode.insertBefore(letterAvatar, avatar);
                    }
                    letterAvatar.textContent = (SS.currentUser.display_name || SS.currentUser.email || '?')[0].toUpperCase();
                    letterAvatar.style.display = 'inline-flex';
                }

                if (avatar && SS.currentUser.avatar_url) {
                    avatar.src = SS.currentUser.avatar_url;
                    avatar.style.display = 'block';
                    if (gIcon) gIcon.style.display = 'none';
                    if (letterAvatar) letterAvatar.style.display = 'none';
                    avatar.onerror = function() { _showLetterAvatar(); };
                } else {
                    _showLetterAvatar();
                }
                // Update name
                var nameEl = dd.querySelector('.auth-profile-name');
                if (nameEl) nameEl.textContent = SS.currentUser.display_name || SS.currentUser.email.split('@')[0];
                // Update email in menu
                var emailEl = dd.querySelector('.auth-profile-email');
                if (emailEl) emailEl.textContent = SS.currentUser.email;
                // Update plan badge
                var planEl = dd.querySelector('.auth-profile-plan');
                if (planEl) {
                    var plan = (SS.currentUser.plan || 'free').charAt(0).toUpperCase() + (SS.currentUser.plan || 'free').slice(1);
                    planEl.textContent = plan;
                    planEl.className = 'auth-profile-plan plan-' + (SS.currentUser.plan || 'free');
                }
            });
        } else {
            // Logged out — re-hide nav-authed-only items
            try { document.body.classList.remove('is-authed'); } catch (e) {}
            // Show sign-in buttons, hide profile dropdowns
            signInBtns.forEach(function(btn) { btn.style.display = ''; });
            profileDropdowns.forEach(function(dd) { dd.style.display = 'none'; });
        }
    }

    // ---- Profile dropdown toggle ----
    SS.toggleProfileMenu = function(e) {
        if (e) e.stopPropagation();
        var menu = document.querySelector('.auth-profile-menu.active') ||
                   (e && e.currentTarget && e.currentTarget.parentElement.querySelector('.auth-profile-menu'));
        if (!menu) return;
        menu.classList.toggle('open');
    };

    // Close profile menu when clicking elsewhere
    document.addEventListener('click', function(e) {
        if (!e.target.closest('.auth-profile-dropdown')) {
            document.querySelectorAll('.auth-profile-menu').forEach(function(menu) {
                menu.classList.remove('open');
            });
        }
    });

    // ---- Save prompt modal (shown after processing if not signed in) ----
    SS.showSavePrompt = function() {
        if (SS.currentUser) return; // Already signed in
        var modal = document.getElementById('savePromptModal');
        if (modal) modal.style.display = 'flex';
    };

    function closeSavePrompt() {
        var modal = document.getElementById('savePromptModal');
        if (modal) modal.style.display = 'none';
    }
    // Expose to window for onclick handlers
    window.closeSavePrompt = closeSavePrompt;
    window.triggerGoogleSignIn = SS.triggerGoogleSignIn;
    window.logout = SS.logout;

    // ---- Initialize auth on page load ----
    SS.initAuth = async function() {
        await fetchConfig();
        initGoogleSignIn();
        await SS.checkAuth();
    };

    // Kick off auth initialization. Without this, Google sign-in stays
    // un-initialized and /auth/me never runs — leaving the user signed in
    // server-side but the page stuck in the logged-out shell (no library,
    // marketing hero, Sign In button visible).
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() { SS.initAuth(); });
    } else {
        SS.initAuth();
    }

})(window.StemScriber);
