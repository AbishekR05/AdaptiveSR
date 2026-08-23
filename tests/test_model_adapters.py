"""
tests/test_model_adapters.py
============================
Step 5.2 — SR Model Runner Adapter Interface unit and integration tests.
"""

import os
import sys
from typing import List, Dict, Any, Tuple, Union, Optional
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adaptive_sr.benchmarking.adapters.base import BaseSRAdapter
from adaptive_sr.benchmarking.adapters.registry import (
    get_adapter,
    list_registered_models,
    list_available_models,
    get_model_status_report
)
from src.modules.video_loader import VideoLoader

# ---------------------------------------------------------------------------
# Mock Adapter for testing base validation logic
# ---------------------------------------------------------------------------

class MockSpatialAdapter(BaseSRAdapter):
    """Mock spatial adapter to verify BaseSRAdapter contract and validations."""

    def __init__(self, simulate_bad_output: bool = False) -> None:
        self.simulate_bad_output = simulate_bad_output
        self._device = None
        self._scale = None
        self._initialized = False

    @property
    def model_id(self) -> str:
        return "mock_spatial"

    @property
    def display_name(self) -> str:
        return "Mock Spatial Adapter"

    @property
    def backend(self) -> str:
        return "mock"

    @property
    def scale_factors(self) -> List[int]:
        return [2, 3]

    @property
    def temporal_or_spatial(self) -> str:
        return "spatial"

    @property
    def precision(self) -> str:
        return "fp32"

    def is_available(self) -> bool:
        return True

    def get_unavailable_reason(self) -> Optional[str]:
        return None

    def initialize(self, device: str, scale: int) -> None:
        if device not in ["cpu", "cuda"]:
            raise ValueError("Unsupported device")
        if scale not in self.scale_factors:
            raise ValueError("Unsupported scale factor")
        self._device = device
        self._scale = scale
        self._initialized = True

    def _run_inference(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if not self._initialized:
            raise RuntimeError("Not initialized")
        
        enhanced = []
        for frame in frames:
            h, w, c = frame.shape
            if self.simulate_bad_output:
                # Return incorrect height (doesn't scale properly)
                bad_frame = np.zeros((h * self._scale + 5, w * self._scale, c), dtype=np.uint8)
                enhanced.append(bad_frame)
            else:
                good_frame = np.zeros((h * self._scale, w * self._scale, c), dtype=np.uint8)
                enhanced.append(good_frame)
        return enhanced

    def close(self) -> None:
        pass


class MockTemporalAdapter(BaseSRAdapter):
    """Mock temporal adapter to verify sequence validation logic."""

    def __init__(self) -> None:
        self._device = None
        self._scale = None
        self._initialized = False

    @property
    def model_id(self) -> str:
        return "mock_temporal"

    @property
    def display_name(self) -> str:
        return "Mock Temporal Adapter"

    @property
    def backend(self) -> str:
        return "mock"

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
        return True

    def get_unavailable_reason(self) -> Optional[str]:
        return None

    def initialize(self, device: str, scale: int) -> None:
        self._device = device
        self._scale = scale
        self._initialized = True

    def _run_inference(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        # Simple upscaling copy
        enhanced = []
        for frame in frames:
            h, w, c = frame.shape
            enhanced.append(np.zeros((h * self._scale, w * self._scale, c), dtype=np.uint8))
        return enhanced

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# 1. Base validation and interface tests
# ---------------------------------------------------------------------------

def test_mock_adapter_successful_flow():
    """Verifies that the mock spatial adapter works cleanly under standard inputs."""
    adapter = MockSpatialAdapter()
    adapter.initialize(device="cpu", scale=2)
    
    # Test single frame input
    frame = np.zeros((100, 150, 3), dtype=np.uint8)
    out = adapter.process(frame, scale=2)
    assert isinstance(out, np.ndarray)
    assert out.shape == (200, 300, 3)
    assert out.dtype == np.uint8

    # Test list input
    frames = [np.zeros((100, 150, 3), dtype=np.uint8) for _ in range(3)]
    outs = adapter.process(frames, scale=2)
    assert isinstance(outs, list)
    assert len(outs) == 3
    assert outs[0].shape == (200, 300, 3)


def test_mock_adapter_input_type_validation():
    """Verifies that invalid input formats/types raise appropriate errors."""
    adapter = MockSpatialAdapter()
    adapter.initialize(device="cpu", scale=2)

    # Test wrong dtype (float instead of uint8)
    bad_frame = np.zeros((100, 150, 3), dtype=np.float32)
    with pytest.raises(TypeError, match="must be uint8 type"):
        adapter.process(bad_frame, scale=2)

    # Test non-numpy input
    with pytest.raises(TypeError, match="Expected np.ndarray or List"):
        adapter.process("not_a_numpy_array", scale=2)

    # Test empty list
    with pytest.raises(ValueError, match="Input frame list cannot be empty"):
        adapter.process([], scale=2)


def test_mock_adapter_input_dimension_validation():
    """Verifies that frames of invalid shapes or dimensions raise errors."""
    adapter = MockSpatialAdapter()
    adapter.initialize(device="cpu", scale=2)

    # Test wrong dimensions (2D grayscale instead of 3D BGR)
    bad_dims = np.zeros((100, 150), dtype=np.uint8)
    with pytest.raises(ValueError, match="must have shape"):
        adapter.process(bad_dims, scale=2)

    # Test mismatched frames in list input
    frames = [
        np.zeros((100, 150, 3), dtype=np.uint8),
        np.zeros((120, 150, 3), dtype=np.uint8) # Mismatched height
    ]
    with pytest.raises(ValueError, match="Mismatched frame dimensions"):
        adapter.process(frames, scale=2)


def test_mock_adapter_output_size_validation():
    """Verifies that output shape validation prevents silent scaling/dimension bugs."""
    adapter = MockSpatialAdapter(simulate_bad_output=True)
    adapter.initialize(device="cpu", scale=2)
    
    frame = np.zeros((100, 150, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Output dimension mismatch"):
        adapter.process(frame, scale=2)


def test_mock_adapter_uninitialized_execution():
    """Verifies that executing process() before initialize() raises appropriate errors."""
    adapter = MockSpatialAdapter()
    frame = np.zeros((100, 150, 3), dtype=np.uint8)
    
    with pytest.raises(RuntimeError, match="Not initialized"):
        adapter.process(frame, scale=2)


def test_mock_adapter_device_and_scale_validation():
    """Verifies that requesting unsupported devices or scale factors fails clearly."""
    adapter = MockSpatialAdapter()
    
    with pytest.raises(ValueError, match="Unsupported device"):
        adapter.initialize(device="invalid_gpu", scale=2)

    with pytest.raises(ValueError, match="Unsupported scale factor"):
        adapter.initialize(device="cpu", scale=4)


# ---------------------------------------------------------------------------
# 2. Temporal vs Spatial validation checks
# ---------------------------------------------------------------------------

def test_temporal_adapter_rejects_single_frame_input():
    """Checks that a temporal adapter rejects a single numpy array frame input."""
    adapter = MockTemporalAdapter()
    adapter.initialize(device="cpu", scale=4)

    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Temporal model requires a list of frames"):
        adapter.process(frame, scale=4)


def test_temporal_adapter_accepts_sequence():
    """Checks that a temporal adapter accepts frame sequences correctly."""
    adapter = MockTemporalAdapter()
    adapter.initialize(device="cpu", scale=4)

    frames = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(5)]
    out = adapter.process(frames, scale=4)
    assert isinstance(out, list)
    assert len(out) == 5
    assert out[0].shape == (200, 200, 3)


# ---------------------------------------------------------------------------
# 3. Discovery / Registry Integration
# ---------------------------------------------------------------------------

def test_registry_integration():
    """Verifies that all standard model_ids resolve to valid registered adapters."""
    registered = list_registered_models()
    assert "tinysr" in registered
    assert "tinysr_int8" in registered
    assert "real_esrgan" in registered
    assert "basicvsr++" in registered

    # Verify resolution
    fsrcnn = get_adapter("tinysr")
    assert fsrcnn.model_id == "tinysr"
    assert fsrcnn.temporal_or_spatial == "spatial"

    basicvsr = get_adapter("basicvsr++")
    assert basicvsr.model_id == "basicvsr++"
    assert basicvsr.temporal_or_spatial == "temporal"


def test_basicvsr_reports_unavailable_on_windows():
    """Ensures BasicVSR++ adapter reports itself as unavailable on Windows development setup."""
    adapter = get_adapter("basicvsr++")
    assert not adapter.is_available()
    assert adapter.get_unavailable_reason() is not None
    assert "MMCV" in adapter.get_unavailable_reason()

    # Verify initialization fails cleanly
    with pytest.raises(NotImplementedError):
        adapter.initialize("cpu", 4)


def test_discovery_helpers():
    """Verifies available and status reports helpers are consistent."""
    available = list_available_models()
    assert "tinysr" in available # Should be always runnable

    report = get_model_status_report()
    assert "tinysr" in report
    assert report["tinysr"]["is_available"] is True

    assert "basicvsr++" in report
    assert report["basicvsr++"]["is_available"] is False


# ---------------------------------------------------------------------------
# 4. FSRCNN FP32 execution smoke test (actual model)
# ---------------------------------------------------------------------------

def test_fsrcnn_fp32_cpu_execution():
    """Smoke test running FSRCNN FP32 model on CPU with mock inputs."""
    adapter = get_adapter("tinysr")
    assert adapter.is_available()
    
    adapter.initialize(device="cpu", scale=2)
    
    # Run spatial SR on a tiny mock frame
    input_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    output_frame = adapter.process(input_frame, scale=2)
    
    assert output_frame.shape == (128, 128, 3)
    assert output_frame.dtype == np.uint8


# ---------------------------------------------------------------------------
# 5. Integration Smoke Test: Step 5.1 Dataset Input -> Adapter -> valid SR output
# ---------------------------------------------------------------------------

def test_smoke_integration_with_step5_1_dataset():
    """Integration smoke test validating the flow:

    dataset input chunk frame -> adapter -> valid scale output.
    """
    manifest_path = "data/benchmarks/sr/manifests/benchmark_manifest.json"
    if not os.path.exists(manifest_path):
        pytest.skip("Step 5.1 benchmark manifest is not present, skipping integration test.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f) if 'json' in globals() else json.loads(f.read())

    # Get the first video and its first chunk
    videos = manifest.get("videos", [])
    if not videos:
        pytest.skip("No videos found inside manifest.")

    first_video = videos[0]
    chunks = first_video.get("chunks", [])
    if not chunks:
        pytest.skip("No chunks associated with the first video.")

    chunk_rel_path = chunks[0]["file_path"]
    base_dir = os.path.dirname(os.path.dirname(manifest_path))
    abs_chunk_path = os.path.join(base_dir, chunk_rel_path)

    # 1. Load frame from Step 5.1 chunk using VideoLoader
    assert os.path.exists(abs_chunk_path), f"Chunk file not found: {abs_chunk_path}"
    loader = VideoLoader(abs_chunk_path)
    
    # Extract first frame
    import cv2
    cap = cv2.VideoCapture(abs_chunk_path)
    ret, frame = cap.read()
    cap.release()
    
    assert ret, "Failed to read frame from chunk."
    assert frame.dtype == np.uint8

    # 2. Feed to FSRCNN adapter (tinysr)
    adapter = get_adapter("tinysr")
    adapter.initialize(device="cpu", scale=2)
    
    # 3. Execute and verify valid output dimensions
    upscaled = adapter.process(frame, scale=2)
    
    h_in, w_in, _ = frame.shape
    assert upscaled.shape == (h_in * 2, w_in * 2, 3)
    assert upscaled.dtype == np.uint8
    print(f"\n[smoke_integration] Scaled {w_in}x{h_in} frame to {upscaled.shape[1]}x{upscaled.shape[0]} using tinysr.")


# Helper import json safely inside tests if not already loaded
import json
