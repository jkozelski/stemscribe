"""
StemScribe Models Module
========================
Handles ML model management, downloading, and inference.
"""

from .model_manager import (
    ModelManager,
    get_model_manager,
    ensure_model,
    list_available_models,
    MODELS,
    DEFAULT_MODEL_DIR
)

__all__ = [
    'ModelManager',
    'get_model_manager',
    'ensure_model',
    'list_available_models',
    'MODELS',
    'DEFAULT_MODEL_DIR'
]
