"""
adaptive_sr.benchmarking.adapters.fsrcnn
========================================
Step 5.2 — FSRCNN FP32 Adapter Wrapper.
"""

from typing import List, Optional
import numpy as np
import torch

from adaptive_sr.benchmarking.adapters.base import BaseSRAdapter
from src.modules.backends.fsrcnn_backend import infer, load_model


class FSRCNNAdapter(BaseSRAdapter):
    """Adapter wrapping the PyTorch FSRCNN FP32 execution backend."""

    def __init__(self) -> None:
        self._device: Optional[str] = None
        self._scale: Optional[int] = None
        self._initialized: bool = False

    @property
    def model_id(self) -> str:
        return "tinysr"

    @property
    def display_name(self) -> str:
        return "FSRCNN (lightweight)"

    @property
    def backend(self) -> str:
        return "pytorch"

    @property
    def scale_factors(self) -> List[int]:
        return [2, 3, 4]

    @property
    def temporal_or_spatial(self) -> str:
        return "spatial"

    @property
    def precision(self) -> str:
        return "fp32"

    def is_available(self) -> bool:
        """PyTorch backend is always available since torch is pinned in requirements."""
        return True

    def get_unavailable_reason(self) -> Optional[str]:
        return None

    def initialize(self, device: str, scale: int) -> None:
        if device not in ["cpu", "cuda"]:
            raise ValueError(f"[{self.model_id}] Unsupported device: {device}")
        if scale not in self.scale_factors:
            raise ValueError(f"[{self.model_id}] Unsupported scale factor: {scale}")

        # Fail clearly if CUDA is requested but unavailable
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"[{self.model_id}] CUDA requested but not supported/available in this environment.")

        self._device = device
        self._scale = scale
        # Force model load/weight download internally
        load_model(device, scale=scale)
        self._initialized = True

    def _run_inference(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if not self._initialized:
            raise RuntimeError(f"[{self.model_id}] Adapter has not been initialized.")

        enhanced_frames = []
        for frame in frames:
            out = infer(frame, self._device, scale=self._scale)
            enhanced_frames.append(out)
        return enhanced_frames

    def close(self) -> None:
        # Pytorch model caches locally in backend dict; nothing to release
        pass
