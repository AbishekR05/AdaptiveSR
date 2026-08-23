"""
adaptive_sr.benchmarking.adapters.base
======================================
Step 5.2 — Base SR Model Adapter Contract.

Defines the Abstract Base Class (ABC) for all Super-Resolution model adapters.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional, Tuple
import numpy as np


class BaseSRAdapter(ABC):
    """Abstract Base Class defining the contract for all Super-Resolution adapters in AdaptiveSR."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique identifier of the model (e.g., 'tinysr')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable display name of the model."""
        pass

    @property
    @abstractmethod
    def backend(self) -> str:
        """Backend technology used (e.g., 'pytorch', 'onnxruntime')."""
        pass

    @property
    @abstractmethod
    def scale_factors(self) -> List[int]:
        """List of integer scale factors supported by this model (e.g., [2, 4])."""
        pass

    @property
    @abstractmethod
    def temporal_or_spatial(self) -> str:
        """Returns 'spatial' (independent frames) or 'temporal' (sequence-based)."""
        pass

    @property
    @abstractmethod
    def precision(self) -> str:
        """Model precision category, e.g., 'fp32', 'fp16', 'int8'."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the backend dependencies and weight files are present."""
        pass

    @abstractmethod
    def get_unavailable_reason(self) -> Optional[str]:
        """Returns the reason string if is_available() is False, else None."""
        pass

    @abstractmethod
    def initialize(self, device: str, scale: int) -> None:
        """Initialize the model on the requested execution device and scale factor.

        Parameters
        ----------
        device : str
            Target execution device: 'cpu' or 'cuda'.
        scale : int
            Upscaling factor (must be in scale_factors).
        """
        pass

    @abstractmethod
    def _run_inference(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Performs the actual model inference execution on a list of raw frames.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Release resources, sessions, or clear caches."""
        pass

    # -----------------------------------------------------------------------
    # Common Validation Engine
    # -----------------------------------------------------------------------

    def validate_inputs(
        self,
        inputs: Union[np.ndarray, List[np.ndarray]],
        scale: int
    ) -> Tuple[List[np.ndarray], bool]:
        """Validates input frame format, dimensions, and sequence requirements.

        Parameters
        ----------
        inputs : np.ndarray or List[np.ndarray]
            Single frame (H, W, 3) or sequence of frames.
        scale : int
            Scale factor used for execution.

        Returns
        -------
        Tuple[List[np.ndarray], bool]
            A tuple of (list of validated frames, is_single_frame_input).
        """
        if scale not in self.scale_factors:
            raise ValueError(
                f"[{self.model_id}] Unsupported scale factor: {scale}. "
                f"Supported: {self.scale_factors}"
            )

        # Normalize single numpy array to a list
        if isinstance(inputs, np.ndarray):
            frames = [inputs]
            is_single = True
        elif isinstance(inputs, list):
            if not inputs:
                raise ValueError(f"[{self.model_id}] Input frame list cannot be empty.")
            frames = inputs
            is_single = False
        else:
            raise TypeError(
                f"[{self.model_id}] Expected np.ndarray or List[np.ndarray], got {type(inputs)}"
            )

        # Enforce temporal sequence checks for temporal models
        if self.temporal_or_spatial == "temporal" and is_single:
            raise ValueError(
                f"[{self.model_id}] Temporal model requires a list of frames as sequence input."
            )

        # Verify dimensions and formats for each frame
        ref_shape = None
        for i, frame in enumerate(frames):
            if not isinstance(frame, np.ndarray):
                raise TypeError(f"[{self.model_id}] Frame index {i} is not a numpy array.")
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                raise ValueError(
                    f"[{self.model_id}] Frame index {i} must have shape (H, W, 3), got {frame.shape}"
                )
            if frame.dtype != np.uint8:
                raise TypeError(
                    f"[{self.model_id}] Frame index {i} must be uint8 type, got {frame.dtype}"
                )

            # Spatial models processed as sequence must have identical sizes
            if ref_shape is None:
                ref_shape = frame.shape
            elif frame.shape != ref_shape:
                raise ValueError(
                    f"[{self.model_id}] Mismatched frame dimensions inside input list at index {i}. "
                    f"Expected {ref_shape}, got {frame.shape}"
                )

        return frames, is_single

    def validate_outputs(
        self,
        outputs: List[np.ndarray],
        input_shape: Tuple[int, int, int],
        scale: int
    ):
        """Validates that output dimensions match scale factor scaling laws exactly.
        No silent resizing is permitted.
        """
        in_h, in_w, in_c = input_shape
        expected_h = in_h * scale
        expected_w = in_w * scale

        for i, out_frame in enumerate(outputs):
            if not isinstance(out_frame, np.ndarray):
                raise TypeError(f"[{self.model_id}] Inference output index {i} is not a numpy array.")
            if out_frame.shape != (expected_h, expected_w, in_c):
                raise ValueError(
                    f"[{self.model_id}] Output dimension mismatch at index {i}. "
                    f"Expected ({expected_h}, {expected_w}, {in_c}), got {out_frame.shape}. "
                    "Silent upscaling/resizing is prohibited."
                )
            if out_frame.dtype != np.uint8:
                raise TypeError(
                    f"[{self.model_id}] Output frame index {i} must be uint8 type, got {out_frame.dtype}"
                )

    def process(
        self,
        inputs: Union[np.ndarray, List[np.ndarray]],
        scale: int = 2
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Processes the input frame or sequence through the SR inference engine.

        Parameters
        ----------
        inputs : np.ndarray or List[np.ndarray]
            Single frame (H, W, 3) or sequence of frames (uint8).
        scale : int
            Upscaling factor (default: 2).

        Returns
        -------
        np.ndarray or List[np.ndarray]
            Single upscaled frame or sequence of upscaled frames.
        """
        if not self.is_available():
            raise RuntimeError(
                f"[{self.model_id}] Model is currently unavailable: {self.get_unavailable_reason()}"
            )

        # 1. Validate Inputs
        frames, is_single = self.validate_inputs(inputs, scale)
        input_shape = frames[0].shape

        # 2. Execute Inference
        try:
            enhanced_frames = self._run_inference(frames)
        except Exception as e:
            raise RuntimeError(f"[{self.model_id}] Inference failed: {e}") from e

        # 3. Validate Outputs
        if len(enhanced_frames) != len(frames):
            raise ValueError(
                f"[{self.model_id}] Output sequence size mismatch: "
                f"expected {len(frames)}, got {len(enhanced_frames)}"
            )

        self.validate_outputs(enhanced_frames, input_shape, scale)

        return enhanced_frames[0] if is_single else enhanced_frames

    def get_metadata(self) -> Dict[str, Any]:
        """Retrieves static metadata definitions."""
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "backend": self.backend,
            "scale_factors": self.scale_factors,
            "temporal_or_spatial": self.temporal_or_spatial,
            "precision": self.precision,
            "is_available": self.is_available(),
            "unavailable_reason": self.get_unavailable_reason()
        }


# Type definition helper
from typing import Tuple
