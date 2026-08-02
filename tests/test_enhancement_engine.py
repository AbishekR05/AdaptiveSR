import os
import time
import pytest
import numpy as np
import cv2
import torch
from src.utils.state_types import Decision
from src.modules.enhancement_engine import EnhancementEngine, get_inference_device
from src.modules.backends.fsrcnn_backend import infer as fsrcnn_infer
from src.modules.backends.realesrgan_backend import infer as realesrgan_infer

@pytest.fixture(scope="module")
def sample_frame():
    # Generate a simple 64x64 color checkerboard pattern
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[0:32, 0:32, :] = [255, 0, 0]    # Blue in BGR
    frame[32:64, 32:64, :] = [0, 0, 255]   # Red in BGR
    frame[0:32, 32:64, :] = [0, 255, 0]    # Green in BGR
    return frame

def test_independent_backends_smoke_test(sample_frame):
    # Ensure test outputs folder exists
    os.makedirs("test_outputs", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. FSRCNN Test
    out_fsrcnn = fsrcnn_infer(sample_frame, device, scale=2)
    assert out_fsrcnn.shape == (128, 128, 3)
    cv2.imwrite("test_outputs/fsrcnn_x2_smoke.png", out_fsrcnn)
    
    # 2. Real-ESRGAN Test (scale 2)
    out_realesrgan = realesrgan_infer(sample_frame, device, scale=2)
    assert out_realesrgan.shape == (128, 128, 3)
    cv2.imwrite("test_outputs/realesrgan_x2_smoke.png", out_realesrgan)

def test_caching_verification(sample_frame):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Import and clear global caches to ensure we benchmark fresh weight loading vs cached inference
    from src.modules.backends.fsrcnn_backend import _model_cache as fsrcnn_cache
    from src.modules.backends.realesrgan_backend import _model_cache as realesrgan_cache
    fsrcnn_cache.clear()
    realesrgan_cache.clear()
    
    # Verify FSRCNN Caching
    t0_fsrcnn_1 = time.perf_counter()
    _ = fsrcnn_infer(sample_frame, device, scale=2)
    t_fsrcnn_1 = time.perf_counter() - t0_fsrcnn_1
    
    t0_fsrcnn_2 = time.perf_counter()
    _ = fsrcnn_infer(sample_frame, device, scale=2)
    t_fsrcnn_2 = time.perf_counter() - t0_fsrcnn_2
    
    print(f"\nFSRCNN: Call 1 = {t_fsrcnn_1:.4f}s | Call 2 = {t_fsrcnn_2:.4f}s")
    # Call 2 must be significantly faster due to cached weights
    assert t_fsrcnn_2 < t_fsrcnn_1

    # Verify Real-ESRGAN Caching
    t0_realesrgan_1 = time.perf_counter()
    _ = realesrgan_infer(sample_frame, device, scale=2)
    t_realesrgan_1 = time.perf_counter() - t0_realesrgan_1
    
    t0_realesrgan_2 = time.perf_counter()
    _ = realesrgan_infer(sample_frame, device, scale=2)
    t_realesrgan_2 = time.perf_counter() - t0_realesrgan_2
    
    print(f"Real-ESRGAN: Call 1 = {t_realesrgan_1:.4f}s | Call 2 = {t_realesrgan_2:.4f}s")
    assert t_realesrgan_2 < t_realesrgan_1

def test_cpu_vs_gpu_dispatch(sample_frame):
    # Test FSRCNN on CPU explicitly
    out_cpu = fsrcnn_infer(sample_frame, "cpu", scale=2)
    assert out_cpu.shape == (128, 128, 3)
    
    if torch.cuda.is_available():
        # Test FSRCNN on CUDA explicitly
        out_gpu = fsrcnn_infer(sample_frame, "cuda", scale=2)
        assert out_gpu.shape == (128, 128, 3)

def test_enhancement_engine_dispatch(sample_frame):
    device = get_inference_device()
    engine = EnhancementEngine(device)
    
    # 1. Routing to tinysr (Case 4 Decision)
    dec_tiny = Decision(model="tinysr", scale=2, reason="flat scene, low battery")
    out_tiny = engine.enhance(sample_frame, dec_tiny)
    assert out_tiny.shape == (128, 128, 3)
    
    #  routing to real_esrgan (Case 2 Decision)
    dec_real = Decision(model="real_esrgan", scale=2, reason="medium complexity, normal battery")
    out_real = engine.enhance(sample_frame, dec_real)
    assert out_real.shape == (128, 128, 3)

def test_vram_stress_check():
    # Only run full 480p -> 4x upscale Real-ESRGAN test on GPU if available to verify VRAM limits and tiling
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is not available, skipping VRAM stress test.")
        
    device = "cuda"
    large_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(large_frame, (320, 240), 100, (255, 255, 0), -1) # Draw cyan circle
    
    torch.cuda.reset_peak_memory_stats()
    
    # Run Real-ESRGAN scale 4 with tiling enabled (tile=400 in backend)
    t0 = time.perf_counter()
    out_enhanced = realesrgan_infer(large_frame, device, scale=4)
    dt = time.perf_counter() - t0
    
    peak_vram_bytes = torch.cuda.max_memory_allocated(device)
    peak_vram_mb = peak_vram_bytes / (1024 * 1024)
    
    print(f"\nReal-ESRGAN x4 Stress Test: Time = {dt:.2f}s | Output Shape = {out_enhanced.shape} | Peak VRAM = {peak_vram_mb:.2f} MB")
    
    assert out_enhanced.shape == (1920, 2560, 3)
    assert peak_vram_mb < 4096.0  # Must not exceed 4GB VRAM
    cv2.imwrite("test_outputs/realesrgan_x4_stress.png", out_enhanced)
