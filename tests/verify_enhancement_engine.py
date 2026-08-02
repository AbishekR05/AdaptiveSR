import time
import numpy as np
import cv2
import torch
from src.modules.backends.fsrcnn_backend import infer as fsrcnn_infer
from src.modules.backends.realesrgan_backend import infer as realesrgan_infer

def main():
    print("=== Phase 5a Validation: Model Latency Benchmarking ===")
    
    # Generate 640x480 input frame (target 480p resolution)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (540, 380), (255, 0, 0), -1)
    
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
        
    results = {}
    
    for device in devices:
        results[device] = {}
        print(f"\nBenchmarking on device: {device}...")
        
        # 1. FSRCNN Benchmarking
        print("Benchmarking FSRCNN (tinysr)...")
        # Warmup
        _ = fsrcnn_infer(frame, device, scale=2)
        
        t0 = time.perf_counter()
        runs = 5
        for _ in range(runs):
            _ = fsrcnn_infer(frame, device, scale=2)
        avg_fsrcnn = (time.perf_counter() - t0) / runs
        results[device]["tinysr"] = avg_fsrcnn * 1000.0  # ms
        
        # 2. Real-ESRGAN Benchmarking
        print("Benchmarking Real-ESRGAN...")
        # Warmup
        _ = realesrgan_infer(frame, device, scale=2)
        
        t0 = time.perf_counter()
        runs = 3
        for _ in range(runs):
            _ = realesrgan_infer(frame, device, scale=2)
        avg_realesrgan = (time.perf_counter() - t0) / runs
        results[device]["real_esrgan"] = avg_realesrgan * 1000.0  # ms
        
    print("\nMeasured Model Latencies (640x480 frame, scale=2):")
    print(f"{'Model':<12} | {'Device':<6} | {'Avg Latency (ms)':<16}")
    print("-" * 42)
    for device in devices:
        for model in ["tinysr", "real_esrgan"]:
            print(f"{model:<12} | {device:<6} | {results[device][model]:<16.2f}")
            
    # Print note if GPU is missing from PyTorch
    if "cuda" not in devices:
        print("\nNote: PyTorch CUDA support is not installed; GPU latencies skipped.")

if __name__ == "__main__":
    main()
