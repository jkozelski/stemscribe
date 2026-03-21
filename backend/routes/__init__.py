"""StemScriber route blueprints."""

from flask import jsonify


def register_error_handlers(app):
    """Register common error handlers on the Flask app."""

    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({'error': 'File too large. Maximum upload size is 500 MB.'}), 413

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({'error': str(error.description) if hasattr(error, 'description') else 'Bad request'}), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Resource not found'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
