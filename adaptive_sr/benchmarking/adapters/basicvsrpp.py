"""
adaptive_sr.benchmarking.adapters.basicvsrpp
============================================
Step 5.2 — BasicVSR++ Adapter Wrapper.
"""

from typing import List, Optional
import numpy as np

from adaptive_sr.benchmarking.adapters.base import BaseSRAdapter


class BasicVSRppAdapter(BaseSRAdapter):
    """Adapter wrapping the BasicVSR++ temporal sequence-aware execution backend.
    
    Since MMCV compilation dependencies are blocked on Windows development environment,
    this adapter acts as a capability stub and reports unavailable status.
    """

    @property
    def model_id(self) -> str:
        return "basicvsr++"

    @property
    def display_name(self) -> str:
        return "BasicVSR++"

    @property
    def backend(self) -> str:
        return "basicvsr_backend"

    @property
    def scale_factors(self) -> List[int]:
        return [4]

    @property
    def temporal_or_spatial(self) -> str:
        return "temporal"

    @property
    def precision(self) -> str:
        return "fp32"

    def is_available(self) -> bool:
        """MMCV is blocked on Windows environment."""
        return False

    def get_unavailable_reason(self) -> Optional[str]:
        return (
            "BasicVSR++ is currently unavailable due to MMCV compilation dependency "
            "mismatches on Windows development machines."
        )

    def initialize(self, device: str, scale: int) -> None:
        raise NotImplementedError(
            f"[{self.model_id}] MMCV backend is unavailable: {self.get_unavailable_reason()}"
        )

    def _run_inference(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError(
            f"[{self.model_id}] MMCV backend is unavailable: {self.get_unavailable_reason()}"
        )

    def close(self) -> None:
        pass
