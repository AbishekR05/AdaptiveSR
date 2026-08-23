"""
adaptive_sr.benchmarking.adapters.fsrcnn_int8
=============================================
Step 5.2 — FSRCNN INT8 (ONNX Runtime) Adapter Wrapper.
"""

import os
from typing import List, Optional
import numpy as np

from adaptive_sr.benchmarking.adapters.base import BaseSRAdapter

# Safe check for ONNX Runtime package availability
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


class FSRCNNInt8Adapter(BaseSRAdapter):
    """Adapter wrapping the quantized FSRCNN INT8 ONNX Runtime execution backend."""

    def __init__(self) -> None:
        self._device: Optional[str] = None
        self._scale: Optional[int] = None
        self._initialized: bool = False
        self._backend_module = None

    @property
    def model_id(self) -> str:
        return "tinysr_int8"

    @property
    def display_name(self) -> str:
        return "FSRCNN (INT8 Quantized CPU)"

    @property
    def backend(self) -> str:
        return "onnxruntime"

    @property
    def scale_factors(self) -> List[int]:
        return [2]

    @property
    def temporal_or_spatial(self) -> str:
        return "spatial"

    @property
    def precision(self) -> str:
        return "int8"

    def is_available(self) -> bool:
        if not HAS_ORT:
            return False
        weights_path = os.path.join("models/tinysr", "fsrcnn_x2_int8.onnx")
        if not os.path.exists(weights_path):
            return False
        return True

    def get_unavailable_reason(self) -> Optional[str]:
        if not HAS_ORT:
            return "ONNX Runtime (onnxruntime) package is not installed."
        weights_path = os.path.join("models/tinysr", "fsrcnn_x2_int8.onnx")
        if not os.path.exists(weights_path):
            return f"INT8 ONNX model weights not found at: {weights_path}. Please run 'python benchmark/quantize_tinysr.py'."
        return None

    def initialize(self, device: str, scale: int) -> None:
        if not self.is_available():
            raise RuntimeError(
                f"[{self.model_id}] Backend is unavailable: {self.get_unavailable_reason()}"
            )

        if device != "cpu":
            # Expose the fact that INT8 dynamic quantization session in this project is CPU-only
            raise ValueError(f"[{self.model_id}] ONNX INT8 quantized session only supports execution on device='cpu'")
        if scale != 2:
            raise ValueError(f"[{self.model_id}] INT8 FSRCNN only supports scale=2 currently.")

        # Deferred import to prevent ModuleNotFoundError when package is missing
        from src.modules.backends import fsrcnn_backend_int8
        self._backend_module = fsrcnn_backend_int8

        self._device = device
        self._scale = scale
        # Force session initialization
        self._backend_module.load_model(device, scale=scale)
        self._initialized = True

    def _run_inference(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if not self._initialized or self._backend_module is None:
            raise RuntimeError(f"[{self.model_id}] Adapter has not been initialized.")

        enhanced_frames = []
        for frame in frames:
            out = self._backend_module.infer(frame, self._device, scale=self._scale)
            enhanced_frames.append(out)
        return enhanced_frames

    def close(self) -> None:
        # ONNX sessions cached locally in backend module; nothing to release
        pass
