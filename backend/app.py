"""
StemScribe - Audio Stem Separation & Transcription API

Thin app factory — all logic lives in routes/, processing/, models/, services/.
"""

import os
import sys
import logging
import signal
import atexit
from pathlib import Path
from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS

# Add homebrew to path
os.environ['PATH'] = os.environ.get('PATH', '') + ':/opt/homebrew/bin:/usr/local/bin'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Frontend directory (relative to backend)
FRONTEND_DIR = Path(__file__).parent.parent / 'frontend'


def create_app():
    app = Flask(__name__)
    CORS(app)

    # Limit upload size to 500 MB
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

    # ---- Error handlers ----
    from routes import register_error_handlers
    register_error_handlers(app)

    # ---- No-cache headers for development ----
    @app.after_request
    def add_no_cache_headers(response):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response

    # ---- Static file serving ----
    @app.route('/')
    def serve_index():
        return send_from_directory(FRONTEND_DIR, 'index.html')

    @app.route('/practice.html')
    def serve_practice():
        return send_from_directory(FRONTEND_DIR, 'practice.html')

    @app.route('/<path:filename>')
    def serve_static(filename):
        if (FRONTEND_DIR / filename).exists():
            return send_from_directory(FRONTEND_DIR, filename)
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
        from middleware.rate_limit import init_limiter
        init_limiter(app)
    except Exception as e:
        logger.warning(f"Rate limiter not available: {e}")

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

    app.register_blueprint(api_bp)
    app.register_blueprint(library_bp)
    app.register_blueprint(stems_bp)
    app.register_blueprint(tabs_bp)
    app.register_blueprint(theory_bp)
    app.register_blueprint(info_bp)
    app.register_blueprint(archive_bp)
    app.register_blueprint(drive_bp)
    app.register_blueprint(health_bp)

    # Auth blueprint (JWT-protected endpoints)
    try:
        from auth.routes import auth_bp
        app.register_blueprint(auth_bp)
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

    # ---- Shutdown handler for active runners ----
    from processing.separation import shutdown_active_runners as _shutdown_handler
    atexit.register(_shutdown_handler)
    for _sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(_sig, lambda s, f: (_shutdown_handler(), sys.exit(0)))

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

    return app


# Allow running as: python app.py
app = create_app()

if __name__ == '__main__':
    import shutil
    logger.info("Starting StemScribe API server...")
    logger.info(f"yt-dlp available: {shutil.which('yt-dlp') is not None}")

    port = int(os.environ.get('PORT', 5555))
    app.run(host='0.0.0.0', port=port, debug=False)
