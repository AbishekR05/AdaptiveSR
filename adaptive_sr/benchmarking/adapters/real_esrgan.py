"""
adaptive_sr.benchmarking.adapters.real_esrgan
=============================================
Step 5.2 — Real-ESRGAN Adapter Wrapper.
"""

import sys
import types
from typing import List, Optional, Dict, Any
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
        self._last_crop_metadata: Dict[str, Any] = {
            "crop_applied": False,
            "pre_crop_width": None,
            "pre_crop_height": None,
            "final_width": None,
            "final_height": None,
            "crop_pixels_if_available": None
        }

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

        if device not in ["cpu", "cuda"]:
            raise ValueError(f"[{self.model_id}] Unsupported device: {device}")
        if scale not in self.scale_factors:
            raise ValueError(f"[{self.model_id}] Unsupported scale factor: {scale}")

        # Fail clearly if CUDA is requested but unavailable
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"[{self.model_id}] CUDA requested but not supported/available in this environment.")

        if num_threads is not None:
            if device != "cpu":
                raise ValueError(
                    f"[{self.model_id}] Custom thread configuration is only supported "
                    f"on device='cpu', got device='{device}'"
                )
            if num_threads < 1:
                raise ValueError(f"[{self.model_id}] num_threads must be >= 1, got {num_threads}")
            torch.set_num_threads(num_threads)

        # Deferred import to prevent ModuleNotFoundError when package is missing
        from src.modules.backends import realesrgan_backend
        self._backend_module = realesrgan_backend

        self._device = device
        self._scale = scale
        self._num_threads = num_threads
        # Force model load/weight download internally
        self._backend_module.load_model(device, scale=scale)
        self._initialized = True

    def _run_inference(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if not self._initialized or self._backend_module is None:
            raise RuntimeError(f"[{self.model_id}] Adapter has not been initialized.")

        self._last_crop_metadata = {
            "crop_applied": False,
            "pre_crop_width": None,
            "pre_crop_height": None,
            "final_width": None,
            "final_height": None,
            "crop_pixels": None
        }

        enhanced_frames = []
        for frame in frames:
            # We call backend infer with BGR frame
            out = self._backend_module.infer(frame, self._device, scale=self._scale)
            
            # Crop the reconstructed output to the exact expected upscaled dimensions
            h_in, w_in, _ = frame.shape
            expected_h = h_in * self._scale
            expected_w = w_in * self._scale
            
            h_out, w_out, _ = out.shape
            width_diff = w_out - expected_w
            height_diff = h_out - expected_h
            
            # Sanity check: Real-ESRGAN padding/tiling output size delta is normally <= 64 pixels.
            # If the output size exceeds the expected scale size by more than 64 pixels per dimension,
            # this represents an unexpectedly large crop operation.
            CROP_THRESHOLD_PX = 64
            if width_diff > CROP_THRESHOLD_PX or height_diff > CROP_THRESHOLD_PX or width_diff < 0 or height_diff < 0:
                raise ValueError(
                    f"Unexpectedly large crop detected during Real-ESRGAN upscaling. "
                    f"Expected shape: {expected_h}x{expected_w}, got shape: {h_out}x{w_out}. "
                    f"Width difference: {width_diff}px, Height difference: {height_diff}px. "
                    f"Maximum allowable boundary padding is {CROP_THRESHOLD_PX}px."
                )

            if h_out != expected_h or w_out != expected_w:
                out = out[:expected_h, :expected_w, :]
                crop_pixels = (h_out * w_out) - (expected_h * expected_w)
                self._last_crop_metadata = {
                    "crop_applied": True,
                    "pre_crop_width": w_out,
                    "pre_crop_height": h_out,
                    "final_width": expected_w,
                    "final_height": expected_h,
                    "crop_pixels": crop_pixels
                }
            else:
                self._last_crop_metadata = {
                    "crop_applied": False,
                    "pre_crop_width": w_out,
                    "pre_crop_height": h_out,
                    "final_width": expected_w,
                    "final_height": expected_h,
                    "crop_pixels": 0
                }
                
            enhanced_frames.append(out)
        return enhanced_frames

    def get_last_inference_metadata(self) -> Dict[str, Any]:
        return self._last_crop_metadata

    def close(self) -> None:
        # Caches locally; nothing to release
        pass
