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

    def initialize(
        self,
        device: str,
        scale: int,
        num_threads: Optional[int] = None
    ) -> None:
        if not self.is_available():
            raise RuntimeError(
                f"[{self.model_id}] Backend is unavailable: {self.get_unavailable_reason()}"
            )

        if device != "cpu":
            # Expose the fact that INT8 dynamic quantization session in this project is CPU-only
            raise ValueError(f"[{self.model_id}] ONNX INT8 quantized session only supports execution on device='cpu'")
        if scale != 2:
            raise ValueError(f"[{self.model_id}] INT8 FSRCNN only supports scale=2 currently.")
        if num_threads is not None and num_threads < 1:
            raise ValueError(f"[{self.model_id}] num_threads must be >= 1, got {num_threads}")

        # Deferred import to prevent ModuleNotFoundError when package is missing
        from src.modules.backends import fsrcnn_backend_int8
        self._backend_module = fsrcnn_backend_int8

        self._device = device
        self._scale = scale
        self._num_threads = num_threads

        # Construct and cache customized ONNX Session to apply num_threads settings
        import onnxruntime as ort
        weights_path = os.path.join("models/tinysr", f"fsrcnn_x{scale}_int8.onnx")
        providers = ["CPUExecutionProvider"]
        
        opts = ort.SessionOptions()
        if num_threads is not None:
            opts.intra_op_num_threads = num_threads
        else:
            import multiprocessing
            opts.intra_op_num_threads = max(1, multiprocessing.cpu_count() // 2)
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        session = ort.InferenceSession(weights_path, opts, providers=providers)
        self._backend_module._session_cache[scale] = session
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
