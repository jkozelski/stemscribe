"""
StemScriber - Audio Stem Separation & Transcription API

Thin app factory — all logic lives in routes/, processing/, models/, services/.
"""

import os
import sys
import time
import logging
import signal
import atexit
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS

# Load .env from project root, then backend directory
load_dotenv(Path(__file__).parent.parent / '.env')
load_dotenv(Path(__file__).parent / '.env')

# Add homebrew to path
os.environ['PATH'] = os.environ.get('PATH', '') + ':/opt/homebrew/bin:/usr/local/bin'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Frontend directory (relative to backend)
FRONTEND_DIR = Path(__file__).parent.parent / 'frontend'


def create_app():
    app = Flask(__name__)
    allowed_origins = os.environ.get(
        'CORS_ORIGINS',
        'http://localhost:5555,http://localhost:3000'
    ).split(',')
    CORS(app, origins=allowed_origins)

    # Limit upload size to 500 MB
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

    # ---- Session cookie hardening ----
    app.config['SESSION_COOKIE_SECURE'] = True
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

    # ---- Error handlers ----
    from routes import register_error_handlers
    register_error_handlers(app)

    # ---- Security headers ----
    @app.after_request
    def add_security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Permissions-Policy'] = 'camera=(), microphone=(self), geolocation=()'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' cdn.jsdelivr.net cdnjs.cloudflare.com unpkg.com accounts.google.com; "
            "style-src 'self' 'unsafe-inline' cdn.jsdelivr.net fonts.googleapis.com cdnjs.cloudflare.com accounts.google.com; "
            "font-src 'self' fonts.gstatic.com cdn.jsdelivr.net cdnjs.cloudflare.com; "
            "img-src 'self' data: blob: *.googleusercontent.com *.ytimg.com i.scdn.co *.mzstatic.com; "
            "media-src 'self' blob:; "
            "connect-src 'self' accounts.google.com; "
            "frame-src 'self' accounts.google.com; "
            "worker-src 'self' blob: cdn.jsdelivr.net; "
        )
        return response

    # ---- No-cache headers for API responses only (not file downloads) ----
    @app.after_request
    def add_no_cache_headers(response):
        content_type = response.content_type or ''
        if 'application/json' in content_type or 'text/html' in content_type or 'javascript' in content_type or 'text/css' in content_type:
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
        return response

    # ---- Static file serving ----
    @app.route("/robots.txt")
    def serve_robots():
        return send_from_directory(FRONTEND_DIR, "robots.txt", mimetype="text/plain")

    @app.route("/sitemap.xml")
    def serve_sitemap():
        return send_from_directory(FRONTEND_DIR, "sitemap.xml", mimetype="application/xml")

    @app.route("/")
    def serve_index():
        return send_from_directory(FRONTEND_DIR, 'landing.html')

    @app.route('/app')
    def serve_app():
        return send_from_directory(FRONTEND_DIR, 'index.html')

    @app.route('/karaoke')
    @app.route('/karaoke.html')
    def serve_karaoke():
        return send_from_directory(FRONTEND_DIR, 'karaoke.html')

    @app.route('/practice.html')
    def serve_practice():
        resp = send_from_directory(FRONTEND_DIR, 'practice.html')
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
        return resp

    @app.route('/<path:filename>')
    def serve_static(filename):
        if (FRONTEND_DIR / filename).exists():
            resp = send_from_directory(FRONTEND_DIR, filename)
            # Prevent caching of HTML files
            if filename.endswith('.html'):
                resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
                resp.headers['Pragma'] = 'no-cache'
                resp.headers['Expires'] = '0'
            return resp
        return jsonify({'error': 'Not found'}), 404

    # ---- JWT Authentication ----
    try:
        from auth.jwt_setup import init_jwt
        init_jwt(app)
        logger.info("JWT authentication initialized")
    except Exception as e:
        logger.warning(f"JWT auth not available: {e}")

    # ---- Rate Limiting ----
    try:
        from middleware.rate_limit import (
            init_limiter, limiter,
            UPLOAD_LIMIT, SONGSTERR_LIMIT, LIBRARY_LIMIT, BETA_LIMIT, SMS_LIMIT,
        )
        init_limiter(app)
        _rate_limiter_ready = True
    except Exception as e:
        logger.warning(f"Rate limiter not available: {e}")
        _rate_limiter_ready = False

    # ---- Register blueprints ----
    from routes.api import api_bp
    from routes.library import library_bp
    from routes.stems import stems_bp
    from routes.tabs import tabs_bp
    from routes.theory import theory_bp
    from routes.info import info_bp
    from routes.archive import archive_bp
    from routes.drive import drive_bp
    from routes.health import health_bp
    from routes.songsterr import songsterr_bp
    from routes.ug import ug_bp
    from routes.chord_sheet import chord_sheet_bp
    from routes.lyrics import lyrics_bp
    from routes.beta import beta_bp
    from routes.sms import sms_bp
    from routes.support import support_bp
    from routes.feedback import feedback_bp
    from routes.accuracy import accuracy_bp
    from routes.jazz_chords import jazz_bp

    app.register_blueprint(api_bp)
    app.register_blueprint(library_bp)
    app.register_blueprint(stems_bp)
    app.register_blueprint(tabs_bp)
    app.register_blueprint(theory_bp)
    app.register_blueprint(info_bp)
    app.register_blueprint(archive_bp)
    app.register_blueprint(drive_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(songsterr_bp)
    app.register_blueprint(ug_bp)
    app.register_blueprint(chord_sheet_bp)
    app.register_blueprint(lyrics_bp)
    app.register_blueprint(beta_bp)
    app.register_blueprint(sms_bp)
    app.register_blueprint(support_bp)
    app.register_blueprint(feedback_bp)
    app.register_blueprint(accuracy_bp)
    app.register_blueprint(jazz_bp)

    # Auth blueprint (JWT-protected endpoints)
    try:
        from auth.routes import auth_bp
        app.register_blueprint(auth_bp)
        # Apply rate limiting to auth endpoints (brute-force protection)
        try:
            from middleware.rate_limit import limiter, AUTH_LIMIT
            limiter.limit(AUTH_LIMIT)(app.view_functions.get('auth.login', lambda: None))
            limiter.limit(AUTH_LIMIT)(app.view_functions.get('auth.register', lambda: None))
            limiter.limit(AUTH_LIMIT)(app.view_functions.get('auth.forgot_password', lambda: None))
            logger.info("Auth blueprint registered with rate limiting (/auth/*)")
        except Exception:
            logger.info("Auth blueprint registered (/auth/*)")
    except Exception as e:
        logger.warning(f"Auth blueprint not available: {e}")

    # Billing blueprint (Stripe checkout + portal)
    try:
        from billing.routes import billing_bp
        from billing.webhooks import webhooks_bp
        app.register_blueprint(billing_bp)
        app.register_blueprint(webhooks_bp)
        logger.info("Billing blueprints registered (/billing/*, /webhooks/*)")
    except Exception as e:
        logger.warning(f"Billing blueprints not available: {e}")

    # ---- Endpoint-specific rate limits ----
    if _rate_limiter_ready:
        try:
            # Exempt health endpoints from rate limiting
            limiter.exempt(app.view_functions.get('health.health', lambda: None))
            limiter.exempt(app.view_functions.get('api.health', lambda: None))

            # Exempt static file serving — JS/CSS/images must never be rate-limited
            static_fn = app.view_functions.get('serve_static')
            if static_fn:
                limiter.exempt(static_fn)

            # Processing endpoints: 5/min (expensive GPU work)
            for ep in ('api.upload_audio', 'api.process_url_endpoint'):
                fn = app.view_functions.get(ep)
                if fn:
                    limiter.limit(UPLOAD_LIMIT)(fn)

            # Songsterr: 30/min
            for ep_name, fn in app.view_functions.items():
                if ep_name.startswith('songsterr.'):
                    limiter.limit(SONGSTERR_LIMIT)(fn)

            # Library: 60/min
            fn = app.view_functions.get('library.get_library')
            if fn:
                limiter.limit(LIBRARY_LIMIT)(fn)

            # Beta endpoints: 10/min
            for ep_name, fn in app.view_functions.items():
                if ep_name.startswith('beta.'):
                    limiter.limit(BETA_LIMIT)(fn)

            # SMS endpoints: 10/min
            for ep_name, fn in app.view_functions.items():
                if ep_name.startswith('sms.'):
                    limiter.limit(SMS_LIMIT)(fn)

            logger.info("Endpoint-specific rate limits applied (upload=5/min, songsterr=30/min, library=60/min, beta=10/min, sms=10/min)")
        except Exception as e:
            logger.warning(f"Failed to apply endpoint rate limits: {e}")

    # ---- Graceful shutdown: wait for active jobs, then cancel ----
    from processing.separation import (
        shutdown_active_runners as _shutdown_handler,
        _active_runners, _active_runners_lock,
    )

    def _graceful_shutdown(signum=None, frame=None):
        """Wait up to 60s for active separation jobs to finish, then force-cancel."""
        with _active_runners_lock:
            active_count = len(_active_runners)

        if active_count > 0:
            logger.info(f"Graceful shutdown: waiting for {active_count} active job(s) to finish (up to 60s)...")
            deadline = time.time() + 60
            while time.time() < deadline:
                with _active_runners_lock:
                    if len(_active_runners) == 0:
                        break
                time.sleep(1)

            with _active_runners_lock:
                remaining = len(_active_runners)
            if remaining > 0:
                logger.warning(f"Graceful shutdown: {remaining} job(s) still running after 60s, force-cancelling")
                _shutdown_handler()
            else:
                logger.info("Graceful shutdown: all jobs completed cleanly")
        else:
            logger.info("Graceful shutdown: no active jobs")

        sys.exit(0)

    atexit.register(_shutdown_handler)
    for _sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(_sig, _graceful_shutdown)

    # ---- DB connection pool cleanup on shutdown ----
    try:
        from db import close_pool
        atexit.register(close_pool)
    except Exception:
        pass

    # ---- Load dependencies (triggers all try/except imports) ----
    import dependencies  # noqa: F401

    # ---- Load saved jobs from library ----
    from models.job import load_all_jobs_from_disk
    loaded = load_all_jobs_from_disk()
    logger.info(f"Library: {loaded} songs available")

    # ---- Start processing watchdog (background thread) ----
    try:
        from processing.watchdog import start_watchdog
        start_watchdog(app)
        logger.info("Processing watchdog started")
    except Exception as e:
        logger.warning(f"Watchdog not available: {e}")

    return app


# Allow running as: python app.py
app = create_app()

if __name__ == '__main__':
    import shutil
    logger.info("Starting StemScriber API server...")
    logger.info(f"yt-dlp available: {shutil.which('yt-dlp') is not None}")

    port = int(os.environ.get('PORT', 5555))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
