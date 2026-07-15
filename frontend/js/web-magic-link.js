// StemScriber — Web auth modal (Sign In / Sign Up / Forgot password)
// Renders an "Sign in with Email" button next to the Google one and opens a
// modal with three flows:
//   1. Sign In  — email + password -> POST /auth/login
//   2. Sign Up  — email + password (+ optional name) -> POST /auth/register
//   3. Forgot   — email -> POST /auth/forgot-password (magic-code reset)
// On success: stores access + refresh tokens, caches user, redirects to /app.
// Skipped inside Capacitor (iOS app uses its own magic-link modal in auth.js).
(function() {
    if (window.Capacitor && window.Capacitor.isNativePlatform && window.Capacitor.isNativePlatform()) return;

    var s = document.createElement('style');
    s.textContent = [
        '.wml-backdrop{position:fixed;inset:0;background:rgba(0,0,0,.78);z-index:99999;display:none;align-items:center;justify-content:center;padding:1.2rem;font-family:"Space Grotesk",-apple-system,BlinkMacSystemFont,sans-serif;}',
        '.wml-backdrop.open{display:flex;}',
        '.wml-card{background:#1a1a24;border:1px solid #2a2a35;border-radius:18px;padding:1.6rem 1.5rem;max-width:400px;width:100%;color:#e8e4df;box-sizing:border-box;}',
        '.wml-tabs{display:flex;gap:.25rem;background:#0d0d12;border-radius:10px;padding:.25rem;margin-bottom:1.2rem;}',
        '.wml-tab{flex:1;padding:.55rem .4rem;background:transparent;border:none;color:#7a7a85;font-family:inherit;font-size:.88rem;font-weight:600;cursor:pointer;border-radius:8px;transition:background .15s,color .15s;}',
        '.wml-tab.active{background:#1a1a24;color:#ff7b54;}',
        '.wml-title{font-family:Righteous,cursive;font-size:1.4rem;color:#ff7b54;margin:0 0 .4rem;text-align:center;}',
        '.wml-sub{color:#7a7a85;font-size:.88rem;margin:0 0 1.1rem;text-align:center;line-height:1.4;}',
        '.wml-input{width:100%;padding:.8rem 1rem;background:#0d0d12;border:1px solid #2a2a35;border-radius:10px;color:#fff;font-size:1rem;font-family:inherit;box-sizing:border-box;margin-bottom:.7rem;}',
        '.wml-input:focus{outline:none;border-color:#ff7b54;}',
        '.wml-input.code{text-align:center;font-family:"SF Mono",Menlo,monospace;font-size:1.5rem;letter-spacing:.4rem;}',
        '.wml-btn{width:100%;padding:.9rem;background:linear-gradient(135deg,#ff7b54,#ff6b9d);color:#fff;border:none;border-radius:10px;font-size:1rem;font-weight:600;font-family:inherit;cursor:pointer;margin-top:.2rem;}',
        '.wml-btn:disabled{opacity:.55;cursor:default;}',
        '.wml-link{display:block;margin-top:.85rem;text-align:center;color:#7a7a85;font-size:.82rem;text-decoration:underline;cursor:pointer;background:none;border:none;width:100%;font-family:inherit;}',
        '.wml-link:hover{color:#ff7b54;}',
        '.wml-msg{margin-top:.7rem;font-size:.84rem;text-align:center;min-height:1.2em;}',
        '.wml-msg.err{color:#ff6b9d;}',
        '.wml-msg.ok{color:#00ff88;}',
        '.wml-hint{font-size:.72rem;color:#5a5a65;margin-top:-.3rem;margin-bottom:.7rem;padding-left:.2rem;}',
        // Method-chooser buttons (Apple/Email/Google big buttons at the top
        // of the modal). One full-width pill per method, distinct backgrounds.
        '#wmlChooser{display:flex;flex-direction:column;gap:.6rem;margin-top:.5rem;}',
        '.wml-choice{display:flex;align-items:center;justify-content:center;gap:.55rem;width:100%;padding:.85rem 1rem;border-radius:10px;font-family:inherit;font-size:.95rem;font-weight:600;cursor:pointer;border:1px solid transparent;}',
        '.wml-choice.apple{background:#fff;color:#000;}',
        '.wml-choice.apple[disabled]{cursor:default;opacity:.6;}',
        '.wml-choice.google{background:#fff;color:#3c4043;}',
        '.wml-choice.email{background:transparent;color:#ff7b54;border-color:#ff7b54;}',
        '.wml-choice em{font-style:normal;font-size:.78rem;}',
        // Permanently hide the original Google sign-in button once we've
        // paired it with our single "Sign in" entry point. Inline style won't
        // stick because auth.js updateAuthUI() does `btn.style.display = ""`
        // which wipes inline assignments. External CSS with !important wins.
        '.auth-signin-btn[data-wml-paired="1"]{display:none!important;}',
        // Hide our injected "Sign in" entry-point button when the user is
        // signed in (body.is-authed is added by auth.js).
        'body.is-authed .auth-email-signin-btn{display:none!important;}',
        // Password field + show/hide eye toggle. The wrapper is position:relative
        // so the eye button can sit absolutely over the right edge of the input.
        '.wml-pw-wrap{position:relative;margin-bottom:.7rem;}',
        '.wml-pw-wrap .wml-pw{margin-bottom:0;padding-right:2.4rem;}',
        '.wml-eye{position:absolute;right:.55rem;top:50%;transform:translateY(-50%);background:none;border:0;padding:.3rem;cursor:pointer;color:#7a7a85;display:flex;align-items:center;justify-content:center;}',
        '.wml-eye:hover{color:#ff7b54;}',
        '.wml-eye svg{width:18px;height:18px;}',
        '.auth-email-signin-btn{display:inline-flex;align-items:center;gap:0.4rem;padding:0.45rem 0.9rem;background:rgba(255,123,84,0.12);border:1px solid rgba(255,123,84,0.4);color:#ff7b54;border-radius:8px;font-family:"Space Grotesk",sans-serif;font-size:0.85rem;font-weight:600;cursor:pointer;text-decoration:none;}',
        '.auth-email-signin-btn:hover{background:rgba(255,123,84,0.2);border-color:rgba(255,123,84,0.6);}',
        '.auth-email-signin-btn svg{flex-shrink:0;}'
    ].join('');
    document.head.appendChild(s);

    var _modalBuilt = false;
    var _mode = 'signin'; // 'signin' | 'signup' | 'forgot'
    var _resetEmail = '';

    function buildModal() {
        if (_modalBuilt) return;
        _modalBuilt = true;
        var bd = document.createElement('div');
        bd.id = 'wmlBackdrop';
        bd.className = 'wml-backdrop';
        bd.innerHTML = ''
            + '<div class="wml-card" role="dialog" aria-modal="true" aria-labelledby="wmlTitle">'
            +   '<h2 id="wmlTitle" class="wml-title">Sign in</h2>'
            +   '<p id="wmlSub" class="wml-sub">Choose how you’d like to sign in.</p>'
            +   '<div id="wmlChooser">'
            +     '<button id="wmlChApple" type="button" class="wml-choice apple">'
            +       '<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor" aria-hidden="true"><path d="M17.05 12.04c-.03-2.71 2.22-4.02 2.32-4.08-1.27-1.85-3.24-2.1-3.94-2.13-1.68-.17-3.27 1-4.12 1-.86 0-2.18-.98-3.58-.95-1.84.03-3.54 1.07-4.48 2.71-1.91 3.32-.49 8.22 1.37 10.9.91 1.31 2 2.78 3.4 2.73 1.36-.06 1.88-.88 3.53-.88 1.65 0 2.11.88 3.55.86 1.46-.03 2.39-1.34 3.29-2.66 1.04-1.53 1.47-3.01 1.49-3.09-.03-.01-2.85-1.09-2.88-4.31zM14.4 4.07c.74-.9 1.24-2.16 1.1-3.41-1.07.04-2.36.71-3.13 1.61-.69.8-1.29 2.07-1.13 3.3 1.19.09 2.41-.6 3.16-1.5z"/></svg>'
            +       '<span>Sign in with Apple</span>'
            +     '</button>'
            +     '<button id="wmlChEmail" type="button" class="wml-choice email">'
            +       '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="m2 7 10 6 10-6"/></svg>'
            +       '<span>Sign in with Email</span>'
            +     '</button>'
            +     '<button id="wmlChGoogle" type="button" class="wml-choice google">'
            +       '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 01-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>'
            +       '<span>Sign in with Google</span>'
            +     '</button>'
            +     '<div id="wmlGoogleHost" style="display:flex;justify-content:center;"></div>'
            +   '</div>'
            +   '<div id="wmlEmailFlow" style="display:none">'
            +   '<div class="wml-tabs" id="wmlTabs">'
            +     '<button class="wml-tab active" data-mode="signin" type="button">Sign in</button>'
            +     '<button class="wml-tab" data-mode="signup" type="button">Create account</button>'
            +   '</div>'
            +   '<div id="wmlFormPassword">'
            +     '<input id="wmlEmail" class="wml-input" type="email" inputmode="email" autocomplete="email" autocapitalize="off" autocorrect="off" placeholder="you@example.com">'
            +     '<input id="wmlName" class="wml-input" type="text" autocomplete="name" placeholder="Your name" style="display:none">'
            +     '<div class="wml-pw-wrap">'
            +       '<input id="wmlPassword" class="wml-input wml-pw" type="password" autocomplete="current-password" placeholder="Password">'
            +       '<button type="button" class="wml-eye" data-target="wmlPassword" aria-label="Show or hide password" tabindex="-1"></button>'
            +     '</div>'
            +     '<div class="wml-pw-wrap" id="wmlConfirmWrap" style="display:none">'
            +       '<input id="wmlPasswordConfirm" class="wml-input wml-pw" type="password" autocomplete="new-password" placeholder="Confirm password">'
            +       '<button type="button" class="wml-eye" data-target="wmlPasswordConfirm" aria-label="Show or hide password" tabindex="-1"></button>'
            +     '</div>'
            +     '<div id="wmlPwHint" class="wml-hint" style="display:none">At least 8 characters.</div>'
            +     '<button id="wmlSubmitBtn" class="wml-btn" type="button">Sign in</button>'
            +     '<button id="wmlForgotBtn" class="wml-link" type="button">Forgot password? Get a code by email</button>'
            +   '</div>'
            +   '<div id="wmlFormCode" style="display:none">'
            +     '<input id="wmlCode" class="wml-input code" type="text" inputmode="numeric" autocomplete="one-time-code" pattern="[0-9]*" maxlength="6" placeholder="000000">'
            +     '<button id="wmlVerifyBtn" class="wml-btn" type="button" disabled>Verify & sign in</button>'
            +     '<button id="wmlBackBtn" class="wml-link" type="button">Use password instead</button>'
            +   '</div>'
            +   '<button id="wmlChooserBackBtn" class="wml-link" type="button" style="display:none">← Choose a different method</button>'
            +   '</div>'   // end #wmlEmailFlow
            +   '<div id="wmlMsg" class="wml-msg" aria-live="polite"></div>'
            +   '<button id="wmlCloseBtn" class="wml-link" type="button">Cancel</button>'
            + '</div>';
        document.body.appendChild(bd);

        var apiBase = (window.StemScriber && window.StemScriber.API_BASE) || '/api';
        var API = apiBase.replace(/\/api$/, '');

        var $ = function(id) { return bd.querySelector('#' + id); };
        var tabs = bd.querySelectorAll('.wml-tab');
        var title = $('wmlTitle'), sub = $('wmlSub'), msg = $('wmlMsg');
        var emailInput = $('wmlEmail'), nameInput = $('wmlName'), pwInput = $('wmlPassword');
        var pwHint = $('wmlPwHint'), submitBtn = $('wmlSubmitBtn'), forgotBtn = $('wmlForgotBtn');
        var formPw = $('wmlFormPassword'), formCode = $('wmlFormCode');
        var codeInput = $('wmlCode'), verifyBtn = $('wmlVerifyBtn'), backBtn = $('wmlBackBtn');
        var closeBtn = $('wmlCloseBtn');

        function setMsg(text, kind) {
            msg.textContent = text || '';
            msg.className = 'wml-msg' + (kind ? ' ' + kind : '');
        }
        function close() { bd.classList.remove('open'); }
        closeBtn.addEventListener('click', close);
        bd.addEventListener('click', function(e) { if (e.target === bd) close(); });

        // ---- Method chooser wiring ----------------------------------------
        var chooser = $('wmlChooser');
        var emailFlow = $('wmlEmailFlow');
        var chooserBackBtn = $('wmlChooserBackBtn');
        function showChooser() {
            chooser.style.display = '';
            emailFlow.style.display = 'none';
            chooserBackBtn.style.display = 'none';
            title.textContent = 'Sign in';
            sub.textContent = 'Choose how you’d like to sign in.';
            setMsg('');
            renderWmlGoogleButton();
        }
        function showEmailFlow() {
            chooser.style.display = 'none';
            emailFlow.style.display = '';
            chooserBackBtn.style.display = '';
            setMode('signin');
        }
        $('wmlChEmail').addEventListener('click', showEmailFlow);
        chooserBackBtn.addEventListener('click', showChooser);

        // Google's OFFICIAL button (redirect mode) is rendered into #wmlGoogleHost
        // by the module-level renderWmlGoogleButton() — one tap → full-page
        // redirect → signed in, no nested modal. The custom button below is a
        // fallback if Google's button fails to render.
        // Direct Google OAuth redirect (OpenID implicit, form_post). Full-page
        // navigation to Google → Google POSTs the id_token to /auth/google/callback
        // → backend verifies, signs in, lands you in /app. No GIS iframe, no popup,
        // no third-party-cookie dependency. One tap.
        $('wmlChGoogle').addEventListener('click', function() {
            try {
                var nonce = Date.now().toString(36) + Math.random().toString(36).slice(2);
                var params = new URLSearchParams({
                    client_id: '1079765524201-tc68snnqqfe3ubf7q4vac0meis948cub.apps.googleusercontent.com',
                    redirect_uri: window.location.origin + '/auth/google/callback',
                    response_type: 'id_token',
                    scope: 'openid email profile',
                    nonce: nonce,
                    response_mode: 'form_post',
                    prompt: 'select_account'
                });
                window.location.href = 'https://accounts.google.com/o/oauth2/v2/auth?' + params.toString();
            } catch (e) {}
        });

        // Apple: load Apple's JS SDK on demand, call AppleID.auth.signIn(),
        // forward identity token + user (first-time-only payload) to /auth/apple.
        var _appleSdkPromise = null;
        function loadAppleSdk() {
            if (_appleSdkPromise) return _appleSdkPromise;
            _appleSdkPromise = new Promise(function(resolve, reject) {
                if (window.AppleID && window.AppleID.auth) return resolve(window.AppleID);
                var s = document.createElement('script');
                s.src = 'https://appleid.cdn-apple.com/appleauth/static/jsapi/appleid/1/en_US/appleid.auth.js';
                s.async = true;
                s.onload = function() {
                    try {
                        window.AppleID.auth.init({
                            clientId: 'com.kozelski.stemscriber.signin',
                            scope: 'name email',
                            redirectURI: 'https://stemscriber.com/auth/apple/callback',
                            usePopup: true,
                        });
                        resolve(window.AppleID);
                    } catch (e) { reject(e); }
                };
                s.onerror = function() { reject(new Error('Apple SDK failed to load')); };
                document.head.appendChild(s);
            });
            return _appleSdkPromise;
        }

        $('wmlChApple').addEventListener('click', async function() {
            setMsg('Opening Apple sign-in…');
            try {
                var Apple = await loadAppleSdk();
                var res = await Apple.auth.signIn();
                // res.authorization.id_token = JWT; res.user (first sign-in only) has name/email
                var idToken = res && res.authorization && res.authorization.id_token;
                if (!idToken) { setMsg('No token from Apple. Try again.', 'err'); return; }
                var body = { identity_token: idToken };
                if (res.user) body.user = res.user;
                var apiBase = (window.StemScriber && window.StemScriber.API_BASE) || '/api';
                var API = apiBase.replace(/\/api$/, '');
                var resp = await fetch(API + '/auth/apple', {
                    method: 'POST', cache: 'no-store',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                });
                var data = await resp.json().catch(function() { return {}; });
                if (resp.ok && data.access_token) {
                    try {
                        localStorage.setItem('access_token', data.access_token);
                        if (data.refresh_token) localStorage.setItem('refresh_token', data.refresh_token);
                        if (data.user) localStorage.setItem('ss_user_cache', JSON.stringify(data.user));
                    } catch (e) {}
                    setMsg('Signed in! Loading…', 'ok');
                    setTimeout(function() { window.location.href = '/app'; }, 400);
                } else {
                    setMsg(data.error || 'Apple sign-in failed.', 'err');
                }
            } catch (e) {
                // User cancelled, or popup blocked — be quiet about it
                if (e && (e.error === 'popup_closed_by_user' || e.error === 'user_cancelled_authorize')) {
                    setMsg('');
                } else {
                    setMsg((e && e.message) || 'Apple sign-in unavailable.', 'err');
                }
            }
        });

        var confirmInput = $('wmlPasswordConfirm');
        var confirmWrap = $('wmlConfirmWrap');

        // ---- Show/hide eye toggle on every .wml-eye button ------------------
        var EYE_OPEN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>';
        var EYE_CLOSED = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.94 17.94A10.06 10.06 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>';
        bd.querySelectorAll('.wml-eye').forEach(function(btn) {
            btn.innerHTML = EYE_CLOSED;
            btn.addEventListener('click', function() {
                var input = $(btn.getAttribute('data-target'));
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
                pwHint.style.display = 'none';
                pwInput.placeholder = 'Password';
                pwInput.autocomplete = 'current-password';
                confirmWrap.style.display = 'none';
                submitBtn.textContent = 'Sign in';
                forgotBtn.style.display = '';
                formPw.style.display = '';
                formCode.style.display = 'none';
            } else if (mode === 'signup') {
                title.textContent = 'Create your account';
                sub.textContent = 'A password lets you sign in instantly next time.';
                nameInput.style.display = '';
                pwHint.style.display = '';
                pwInput.placeholder = 'Choose a password';
                pwInput.autocomplete = 'new-password';
                confirmWrap.style.display = '';
                submitBtn.textContent = 'Create account';
                forgotBtn.style.display = 'none';
                formPw.style.display = '';
                formCode.style.display = 'none';
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
            } catch (e) {}
            setMsg('Signed in! Loading…', 'ok');
            setTimeout(function() { window.location.href = '/app'; }, 400);
        }

        submitBtn.addEventListener('click', async function() {
            var email = (emailInput.value || '').trim().toLowerCase();
            var password = pwInput.value || '';
            if (!email || email.indexOf('@') < 0) { setMsg('Please enter a valid email.', 'err'); return; }
            if (!password) { setMsg('Please enter a password.', 'err'); return; }
            if (_mode === 'signup' && password.length < 8) { setMsg('Password must be at least 8 characters.', 'err'); return; }
            if (_mode === 'signup' && (confirmInput.value || '') !== password) { setMsg('Passwords don\'t match. Try again.', 'err'); confirmInput.focus(); return; }
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
                if (res.ok && data.access_token) {
                    await storeTokensAndGo(data);
                } else {
                    setMsg(data.error || (_mode === 'signup' ? 'Could not create account.' : 'Invalid email or password.'), 'err');
                    submitBtn.disabled = false;
                }
            } catch (e) {
                setMsg('Network error. Try again.', 'err');
                submitBtn.disabled = false;
            }
        });

        // Forgot-password → switch to magic-code form, send the code
        forgotBtn.addEventListener('click', async function() {
            var email = (emailInput.value || '').trim().toLowerCase();
            if (!email || email.indexOf('@') < 0) {
                emailInput.focus();
                setMsg('Enter your email above first, then click Forgot.', 'err');
                return;
            }
            forgotBtn.disabled = true;
            setMsg('Sending reset code…');
            try {
                var res = await fetch(API + '/auth/forgot-password', {
                    method: 'POST', cache: 'no-store',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email: email }),
                });
                await res.json().catch(function() { return {}; });
                _resetEmail = email;
                formPw.style.display = 'none';
                formCode.style.display = '';
                title.textContent = 'Check your email';
                sub.textContent = 'We sent a 6-digit code to ' + email + '. Enter it below to sign in, then set a new password.';
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
            try {
                var res = await fetch(API + '/auth/magic-link/verify-code', {
                    method: 'POST', cache: 'no-store',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email: _resetEmail, code: code }),
                });
                var data = await res.json().catch(function() { return {}; });
                if (res.ok && data.access_token) {
                    await storeTokensAndGo(data);
                } else {
                    setMsg(data.error || 'Invalid or expired code.', 'err');
                    verifyBtn.disabled = false;
                }
            } catch (e) {
                setMsg('Network error. Try again.', 'err');
                verifyBtn.disabled = false;
            }
        });

        backBtn.addEventListener('click', function() { setMode('signin'); });

        setMode('signin');
    }

    // Render Google's official sign-in button (configured for redirect mode in
    // auth.js) into the chooser. One tap → full-page redirect → signed in.
    // Module-scoped so both showChooser() and openModal() can call it.
    // Google's official GIS button renders 0×0 under modern third-party-cookie
    // rules, so we DON'T use it. We keep our own styled button (#wmlChGoogle)
    // visible and send it through a direct OAuth redirect (see its click handler).
    function renderWmlGoogleButton() {
        var host = document.getElementById('wmlGoogleHost');
        if (host) host.innerHTML = '';
        var c = document.getElementById('wmlChGoogle');
        if (c) c.style.display = '';   // ensure the custom Google button stays visible
    }

    function openModal() {
        buildModal();
        var bd = document.getElementById('wmlBackdrop');
        document.getElementById('wmlEmail').value = '';
        document.getElementById('wmlName').value = '';
        document.getElementById('wmlPassword').value = '';
        document.getElementById('wmlPassword').type = 'password';
        var _confirm = document.getElementById('wmlPasswordConfirm');
        if (_confirm) { _confirm.value = ''; _confirm.type = 'password'; }
        // Reset all eye toggles to "hidden" state visually
        bd.querySelectorAll('.wml-eye').forEach(function(b) {
            // EYE_CLOSED is the default — we set the SVG inline to keep openModal
            // independent of the inner closure. Cheap to re-render.
            b.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="18" height="18"><path d="M17.94 17.94A10.06 10.06 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>';
        });
        var _confirmWrap = document.getElementById('wmlConfirmWrap');
        if (_confirmWrap) _confirmWrap.style.display = 'none'; // signin default
        document.getElementById('wmlCode').value = '';
        document.getElementById('wmlMsg').textContent = '';
        // Reset to chooser on every open. Email + tabs are hidden until the
        // user picks "Sign in with Email" from the chooser.
        var tabs = bd.querySelectorAll('.wml-tab');
        tabs.forEach(function(t) { t.classList.toggle('active', t.getAttribute('data-mode') === 'signin'); });
        document.getElementById('wmlFormPassword').style.display = '';
        document.getElementById('wmlFormCode').style.display = 'none';
        document.getElementById('wmlName').style.display = 'none';
        document.getElementById('wmlPwHint').style.display = 'none';
        document.getElementById('wmlPassword').placeholder = 'Password';
        document.getElementById('wmlSubmitBtn').textContent = 'Sign in';
        document.getElementById('wmlSubmitBtn').disabled = false;
        document.getElementById('wmlForgotBtn').style.display = '';
        // Show chooser (hide email flow + chooser-back button)
        document.getElementById('wmlChooser').style.display = '';
        document.getElementById('wmlEmailFlow').style.display = 'none';
        document.getElementById('wmlChooserBackBtn').style.display = 'none';
        document.getElementById('wmlTitle').textContent = 'Sign in';
        document.getElementById('wmlSub').textContent = 'Choose how you’d like to sign in.';
        bd.classList.add('open');
        renderWmlGoogleButton();
    }
    window.openEmailSignIn = openModal;

    function injectEmailButton() {
        // Replace the existing Google sign-in button with a single "Sign in"
        // button that opens the chooser modal (Apple / Email / Google). The
        // original Google button gets hidden — its onclick handler still
        // works because SS.triggerGoogleSignIn calls the GIS popup directly,
        // not via the button itself.
        var googleBtns = document.querySelectorAll('.auth-signin-btn');
        for (var i = 0; i < googleBtns.length; i++) {
            var g = googleBtns[i];
            if (g.getAttribute('data-wml-paired') === '1') continue;
            // auth.js's updateAuthUI sets `btn.style.display = ''` when logged
            // out, which would un-hide this Google button and show two
            // mismatched sign-in buttons. setProperty with `important` wins
            // against that inline reset.
            g.style.setProperty('display', 'none', 'important');
            var signinBtn = document.createElement('button');
            signinBtn.className = 'auth-email-signin-btn';
            signinBtn.type = 'button';
            signinBtn.setAttribute('data-wml-signin', '1');
            signinBtn.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4"/><polyline points="10 17 15 12 10 7"/><line x1="15" y1="12" x2="3" y2="12"/></svg><span>Sign in</span>';
            signinBtn.addEventListener('click', openModal);
            g.parentNode.insertBefore(signinBtn, g);
            g.setAttribute('data-wml-paired', '1');
        }
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', injectEmailButton);
    } else {
        injectEmailButton();
    }
    setTimeout(injectEmailButton, 500);
    setTimeout(injectEmailButton, 1500);
})();
