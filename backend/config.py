"""
StemScribe v2.0 Configuration
Centralized settings for ensemble separation system
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR.parent / 'outputs'
MODEL_CACHE_DIR = Path.home() / '.cache' / 'stemscribe' / 'models'

# Ensure directories exist
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Model configurations
SEPARATION_CONFIG = {
    'models': {
        'primary': [
            {
                'name': 'htdemucs_ft',
                'enabled': True,
                'memory_required_gb': 8,
                'processing_time_factor': 1.0,  # Baseline
                'quality_tier': 1,
                'stem_count': 4,
                'stems': ['drums', 'bass', 'other', 'vocals']
            },
            {
                'name': 'htdemucs',
                'enabled': True,
                'memory_required_gb': 6,
                'processing_time_factor': 0.7,
                'quality_tier': 2,
                'stem_count': 4,
                'stems': ['drums', 'bass', 'other', 'vocals']
            },
            {
                'name': 'htdemucs_6s',
                'enabled': True,
                'memory_required_gb': 8,
                'processing_time_factor': 1.2,
                'quality_tier': 1,
                'stem_count': 6,
                'stems': ['drums', 'bass', 'guitar', 'piano', 'other', 'vocals']
            }
        ],
        'fallback': [
            {
                'name': 'mdx_extra',
                'enabled': True,
                'memory_required_gb': 4,
                'processing_time_factor': 0.5,
                'quality_tier': 3,
                'stem_count': 4
            }
        ]
    },

    'quality_thresholds': {
        'min_snr_db': 6.0,
        'max_cross_correlation': 0.5,
        'min_spectral_entropy': 2.5,
        'bleed_penalty_threshold': 0.4
    },

    'post_processing': {
        'phase_alignment': True,
        'bleed_reduction': {
            'enabled': True,
            'aggressiveness': 0.5  # 0-1, higher = more aggressive
        },
        'noise_reduction': {
            'enabled': True,
            'threshold_db': -40
        },
        'artifact_removal': True,
        'loudness_normalization': {
            'enabled': True,
            'target_lufs': -14.0
        }
    },

    'output': {
        'sample_rate': 44100,
        'bit_depth': 24,
        'format': 'wav'
    }
}

# Hardware profiles optimized for different systems
HARDWARE_PROFILES = {
    'apple_m3_max_48gb': {
        'device': 'mps',
        'enabled_models': ['htdemucs_ft', 'htdemucs', 'htdemucs_6s'],
        'strategy': 'parallel',  # M3 Max can handle parallel!
        'max_parallel_models': 2,
        'use_fp16': True,  # MPS supports fp16
        'batch_size': 1
    },
    'apple_m1_16gb': {
        'device': 'mps',
        'enabled_models': ['htdemucs', 'htdemucs_ft'],
        'strategy': 'sequential',
        'max_parallel_models': 1,
        'use_fp16': True,
        'batch_size': 1
    },
    'nvidia_rtx_24gb': {
        'device': 'cuda',
        'enabled_models': ['htdemucs_ft', 'htdemucs', 'htdemucs_6s'],
        'strategy': 'parallel',
        'max_parallel_models': 3,
        'use_fp16': True,
        'batch_size': 1
    },
    'nvidia_rtx_8gb': {
        'device': 'cuda',
        'enabled_models': ['htdemucs_ft', 'htdemucs'],
        'strategy': 'sequential',
        'max_parallel_models': 1,
        'use_fp16': True,
        'batch_size': 1
    },
    'cpu_only': {
        'device': 'cpu',
        'enabled_models': ['htdemucs'],
        'strategy': 'sequential',
        'max_parallel_models': 1,
        'use_fp16': False,
        'batch_size': 1
    }
}

def get_hardware_profile():
    """Auto-detect hardware and return appropriate profile"""
    import torch

    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Detect memory (approximate based on system)
        try:
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'hw.memsize'],
                                  capture_output=True, text=True)
            total_mem_gb = int(result.stdout.strip()) / (1024**3)

            if total_mem_gb >= 32:
                return HARDWARE_PROFILES['apple_m3_max_48gb']
            else:
                return HARDWARE_PROFILES['apple_m1_16gb']
        except:
            return HARDWARE_PROFILES['apple_m1_16gb']

    # Check for CUDA
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb >= 20:
            return HARDWARE_PROFILES['nvidia_rtx_24gb']
        else:
            return HARDWARE_PROFILES['nvidia_rtx_8gb']

    return HARDWARE_PROFILES['cpu_only']
