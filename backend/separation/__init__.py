"""
StemScribe Advanced Audio Separation System
Moises.ai-quality separation using open-source ensemble models
"""

from .ensemble import EnsembleSeparator
from .gpu_manager import GPUManager
from .quality_metrics import QualityMetrics
from .post_processor import PostProcessor
from .voting_system import VotingSystem

__all__ = [
    'EnsembleSeparator',
    'GPUManager',
    'QualityMetrics',
    'PostProcessor',
    'VotingSystem'
]

__version__ = '2.0.0'
