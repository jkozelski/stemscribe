"""
GPU/MPS Memory Manager for StemScribe
Optimized for Apple Silicon M3 Max and NVIDIA GPUs
"""

import logging
import asyncio
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    CPU = 'cpu'
    CUDA = 'cuda'
    MPS = 'mps'


@dataclass
class DeviceInfo:
    device_type: DeviceType
    device_name: str
    total_memory_gb: float
    available_memory_gb: float
    compute_capability: Optional[str] = None


class GPUManager:
    """
    Intelligent GPU/MPS memory management for ensemble inference.
    Optimized for Apple M3 Max with 48GB unified memory.
    """

    def __init__(self):
        self.device_info = self._detect_device()
        self.loaded_models: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

        logger.info(f"GPUManager initialized: {self.device_info}")

    def _detect_device(self) -> DeviceInfo:
        """Detect available compute device and capabilities"""
        import torch

        # Check MPS first (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Get system memory as proxy for unified memory
            try:
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'hw.memsize'],
                                      capture_output=True, text=True)
                total_mem = int(result.stdout.strip()) / (1024**3)

                # Get chip info
                chip_result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                            capture_output=True, text=True)
                chip_name = chip_result.stdout.strip() if chip_result.returncode == 0 else 'Apple Silicon'

            except Exception as e:
                logger.warning(f"Could not detect Mac specs: {e}")
                total_mem = 16.0
                chip_name = 'Apple Silicon'

            return DeviceInfo(
                device_type=DeviceType.MPS,
                device_name=chip_name,
                total_memory_gb=total_mem,
                available_memory_gb=total_mem * 0.7,  # Conservative estimate
                compute_capability='MPS'
            )

        # Check CUDA
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_mem = props.total_memory / (1024**3)

            return DeviceInfo(
                device_type=DeviceType.CUDA,
                device_name=props.name,
                total_memory_gb=total_mem,
                available_memory_gb=total_mem * 0.9,
                compute_capability=f"{props.major}.{props.minor}"
            )

        # CPU fallback
        import psutil
        total_mem = psutil.virtual_memory().total / (1024**3)

        return DeviceInfo(
            device_type=DeviceType.CPU,
            device_name='CPU',
            total_memory_gb=total_mem,
            available_memory_gb=total_mem * 0.5
        )

    @property
    def device(self) -> str:
        """Get torch device string"""
        return self.device_info.device_type.value

    @property
    def is_mps(self) -> bool:
        return self.device_info.device_type == DeviceType.MPS

    @property
    def is_cuda(self) -> bool:
        return self.device_info.device_type == DeviceType.CUDA

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        import torch

        if self.is_cuda:
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'free_gb': self.device_info.total_memory_gb - reserved
            }

        elif self.is_mps:
            # MPS doesn't have direct memory query, estimate from system
            try:
                import psutil
                mem = psutil.virtual_memory()
                used = mem.used / (1024**3)
                available = mem.available / (1024**3)
                return {
                    'allocated_gb': used,
                    'reserved_gb': used,
                    'free_gb': available
                }
            except:
                return {
                    'allocated_gb': 0,
                    'reserved_gb': 0,
                    'free_gb': self.device_info.available_memory_gb
                }

        return {'allocated_gb': 0, 'reserved_gb': 0, 'free_gb': 8.0}

    def can_load_model(self, memory_required_gb: float) -> bool:
        """Check if we have enough memory to load a model"""
        mem = self.get_memory_usage()
        return mem['free_gb'] >= memory_required_gb * 1.2  # 20% buffer

    def clear_memory(self):
        """Clear GPU/MPS memory cache"""
        import torch
        import gc

        gc.collect()

        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        elif self.is_mps:
            # MPS memory management
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

        logger.info("Memory cache cleared")

    async def run_with_memory_management(self,
                                         func: Callable,
                                         memory_required_gb: float,
                                         *args, **kwargs) -> Any:
        """
        Run a function with memory management.
        Clears cache before and after if needed.
        """
        async with self._lock:
            # Check if we need to clear memory first
            if not self.can_load_model(memory_required_gb):
                logger.info(f"Clearing memory before loading (need {memory_required_gb}GB)")
                self.clear_memory()

            try:
                # Run the function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                return result

            finally:
                # Clear cache after processing large models
                if memory_required_gb > 6:
                    self.clear_memory()

    def get_optimal_batch_size(self, model_memory_gb: float) -> int:
        """Calculate optimal batch size based on available memory"""
        mem = self.get_memory_usage()
        available = mem['free_gb']

        # Leave 2GB headroom
        usable = max(0, available - 2)

        # Batch size = usable memory / model memory
        batch_size = int(usable / model_memory_gb)

        return max(1, min(batch_size, 4))  # Clamp between 1 and 4

    def __repr__(self):
        return f"GPUManager({self.device_info.device_name}, {self.device_info.total_memory_gb:.1f}GB)"
