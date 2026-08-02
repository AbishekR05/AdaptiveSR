import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.modules.backends.fsrcnn_backend import infer as infer_fp32
from src.modules.backends.fsrcnn_backend_int8 import infer as infer_int8

def main():
    print("====================================================")
    print("Benchmarking FSRCNN Optimizations: FP32 CPU vs INT8 CPU")
    print("====================================================\n")
    
    input_path = "benchmark_data/futbol_lr.mp4"
    if not os.path.exists(input_path):
        print(f"Error: Input video not found at: {input_path}")
        return
        
    cap = cv2.VideoCapture(input_path)
    frames = []
    # Read first 30 frames for benchmark
    for _ in range(30):
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    
    num_frames = len(frames)
    if num_frames == 0:
        print("Error: No frames extracted from video.")
        return
        
    print(f"Extracted {num_frames} frames from '{input_path}' for benchmark.")
    
    # 1. Benchmark FP32 FSRCNN on CPU
    print("[RUNNING] Evaluating FSRCNN FP32 CPU baseline...")
    t_start_fp32 = time.perf_counter()
    fp32_outputs = []
    for frame in frames:
        out_fp32 = infer_fp32(frame, device="cpu", scale=2)
        fp32_outputs.append(out_fp32)
    dt_fp32 = time.perf_counter() - t_start_fp32
    avg_latency_fp32_ms = (dt_fp32 / num_frames) * 1000.0
    print(f"[COMPLETED] FP32 CPU: Total time = {dt_fp32:.2f}s | Avg Latency = {avg_latency_fp32_ms:.1f} ms\n")
    
    # 2. Benchmark INT8 FSRCNN on CPU
    print("[RUNNING] Evaluating FSRCNN INT8 CPU optimized session...")
    t_start_int8 = time.perf_counter()
    int8_outputs = []
    for frame in frames:
        out_int8 = infer_int8(frame, device="cpu", scale=2)
        int8_outputs.append(out_int8)
    dt_int8 = time.perf_counter() - t_start_int8
    avg_latency_int8_ms = (dt_int8 / num_frames) * 1000.0
    print(f"[COMPLETED] INT8 CPU: Total time = {dt_int8:.2f}s | Avg Latency = {avg_latency_int8_ms:.1f} ms\n")
    
    # 3. Calculate Speedup
    speedup = (avg_latency_fp32_ms / avg_latency_int8_ms)
    speedup_pct = (speedup - 1.0) * 100.0
    
    # 4. Compute Quantization Quality Cost (FP32 vs INT8 output frames)
    psnr_scores = []
    ssim_scores = []
    for out_fp32, out_int8 in zip(fp32_outputs, int8_outputs):
        p_val = psnr(out_fp32, out_int8, data_range=255)
        psnr_scores.append(p_val)
        s_val = ssim(out_fp32, out_int8, channel_axis=2, data_range=255)
        ssim_scores.append(s_val)
        
    avg_psnr_diff = np.mean(psnr_scores)
    avg_ssim_diff = np.mean(ssim_scores)
    
    # 5. Output Summary Results
    print("====================================================")
    print("FSRCNN Optimization Summary Results")
    print("====================================================")
    print(f"FP32 CPU Latency: {avg_latency_fp32_ms:.1f} ms/frame")
    print(f"INT8 CPU Latency: {avg_latency_int8_ms:.1f} ms/frame")
    print(f"Speedup Factor:   {speedup:.2f}x ({speedup_pct:.1f}% faster)")
    print(f"Quantization Cost (PSNR delta vs FP32): {avg_psnr_diff:.2f} dB")
    print(f"Quantization Cost (SSIM delta vs FP32): {avg_ssim_diff:.4f}")
    print("====================================================")

if __name__ == "__main__":
    main()
