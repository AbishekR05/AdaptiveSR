"""
adaptive_sr.benchmarking.adapters.real_esrgan
=============================================
Step 5.2 — Real-ESRGAN Adapter Wrapper.
"""

import sys
import types
from typing import List, Optional
import numpy as np

# Apply torchvision compatibility patch prior to importing basicsr/realesrgan
try:
    import torchvision.transforms.functional as T_F
    functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
    functional_tensor.rgb_to_grayscale = T_F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor
except ImportError:
    pass

# Safe check for Real-ESRGAN packages availability
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    import torch
    HAS_REAL_ESRGAN = True
except ImportError:
    HAS_REAL_ESRGAN = False

from adaptive_sr.benchmarking.adapters.base import BaseSRAdapter


class RealESRGANAdapter(BaseSRAdapter):
    """Adapter wrapping the Real-ESRGAN upsampler execution backend."""

    def __init__(self) -> None:
        self._device: Optional[str] = None
        self._scale: Optional[int] = None
        self._initialized: bool = False
        self._backend_module = None

    @property
    def model_id(self) -> str:
        return "real_esrgan"

    @property
    def display_name(self) -> str:
        return "Real-ESRGAN"

    @property
    def backend(self) -> str:
        return "realesrgan"

    @property
    def scale_factors(self) -> List[int]:
        return [2, 4]

    @property
    def temporal_or_spatial(self) -> str:
        return "spatial"

    @property
    def precision(self) -> str:
        return "fp32"

    def is_available(self) -> bool:
        return HAS_REAL_ESRGAN

    def get_unavailable_reason(self) -> Optional[str]:
        if not HAS_REAL_ESRGAN:
            return "basicsr / realesrgan packages are not installed or are incompatible."
        return None

    def initialize(self, device: str, scale: int) -> None:
        if not self.is_available():
            raise RuntimeError(
                f"[{self.model_id}] Backend is unavailable: {self.get_unavailable_reason()}"
            )

        if device not in ["cpu", "cuda"]:
            raise ValueError(f"[{self.model_id}] Unsupported device: {device}")
        if scale not in self.scale_factors:
            raise ValueError(f"[{self.model_id}] Unsupported scale factor: {scale}")

        # Fail clearly if CUDA is requested but unavailable
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"[{self.model_id}] CUDA requested but not supported/available in this environment.")

        # Deferred import to prevent ModuleNotFoundError when package is missing
        from src.modules.backends import realesrgan_backend
        self._backend_module = realesrgan_backend

        self._device = device
        self._scale = scale
        # Force model load/weight download internally
        self._backend_module.load_model(device, scale=scale)
        self._initialized = True

    def _run_inference(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if not self._initialized or self._backend_module is None:
            raise RuntimeError(f"[{self.model_id}] Adapter has not been initialized.")

        enhanced_frames = []
        for frame in frames:
            # We call backend infer with BGR frame
            out = self._backend_module.infer(frame, self._device, scale=self._scale)
            enhanced_frames.append(out)
        return enhanced_frames

    def close(self) -> None:
        # Caches locally; nothing to release
        pass
