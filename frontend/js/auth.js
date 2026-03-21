// StemScribe — Authentication Module (Google Sign-In + JWT)
window.StemScribe = window.StemScribe || {};

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
            // Hide sign-in buttons, show profile dropdowns
            signInBtns.forEach(function(btn) { btn.style.display = 'none'; });
            profileDropdowns.forEach(function(dd) {
                dd.style.display = 'flex';
                // Update avatar
                var avatar = dd.querySelector('.auth-profile-avatar');
                if (avatar && SS.currentUser.avatar_url) {
                    avatar.src = SS.currentUser.avatar_url;
                    avatar.style.display = 'block';
                } else if (avatar) {
                    avatar.style.display = 'none';
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

})(window.StemScribe);
