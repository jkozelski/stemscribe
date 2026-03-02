"""
Health check endpoint for Railway / Docker / monitoring.
"""

from flask import Blueprint, jsonify

health_bp = Blueprint('health', __name__)


@health_bp.route('/health')
def health():
    return jsonify({'status': 'ok', 'service': 'stemscribe-api'}), 200
