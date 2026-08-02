import os
import sys
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips

# Load LPIPS model globally
device = "cuda" if torch.cuda.is_available() else "cpu"
lpips_model = lpips.LPIPS(net='alex').to(device)

def to_tensor(img_bgr):
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Normalize to [0, 1] and then map to [-1, 1]
    img_float = img_rgb.astype(np.float32) / 255.0
    img_normalized = (img_float - 0.5) * 2.0
    # Reshape to (1, 3, H, W) PyTorch Tensor on device
    tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor

def compute_metrics_for_videos(gt_path, enhanced_path):
    cap_gt = cv2.VideoCapture(gt_path)
    cap_enh = cv2.VideoCapture(enhanced_path)
    
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    
    frame_idx = 0
    while True:
        ret_gt, frame_gt = cap_gt.read()
        ret_enh, frame_enh = cap_enh.read()
        
        if not ret_gt or not ret_enh:
            break
            
        # Ensure dimensions match.
        # Resize ground truth to match the enhanced frame size
        h_enh, w_enh, _ = frame_enh.shape
        h_gt, w_gt, _ = frame_gt.shape
        if h_gt != h_enh or w_gt != w_enh:
            frame_gt = cv2.resize(frame_gt, (w_enh, h_enh), interpolation=cv2.INTER_LANCZOS4)
            
        # 1. Compute PSNR
        p_val = psnr(frame_gt, frame_enh, data_range=255)
        psnr_scores.append(p_val)
        
        # 2. Compute SSIM (RGB image with channel_axis=2)
        s_val = ssim(frame_gt, frame_enh, channel_axis=2, data_range=255)
        ssim_scores.append(s_val)
        
        # 3. Compute LPIPS
        with torch.no_grad():
            t_gt = to_tensor(frame_gt)
            t_enh = to_tensor(frame_enh)
            l_val = lpips_model(t_enh, t_gt).item()
        lpips_scores.append(l_val)
        
        frame_idx += 1
        
    cap_gt.release()
    cap_enh.release()
    
    if frame_idx == 0:
        return {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
        
    return {
        "psnr": float(np.mean(psnr_scores)),
        "ssim": float(np.mean(ssim_scores)),
        "lpips": float(np.mean(lpips_scores))
    }

def main():
    results_dir = Path("benchmark_results")
    categories = ["simple", "complex", "mixed"]
    configs = ["baseline_tinysr", "baseline_real_esrgan", "adaptive"]
    
    print("====================================================")
    print("Computing Visual Quality Metrics (PSNR, SSIM, LPIPS)")
    print("====================================================\n")
    
    for category in categories:
        gt_path = f"benchmark_data/{category}_gt.mp4"
        if not os.path.exists(gt_path):
            print(f"Ground truth video not found: {gt_path}")
            continue
            
        for config_name in configs:
            enhanced_path = results_dir / f"{category}__{config_name}.mp4"
            out_json_path = results_dir / f"{category}__{config_name}_quality.json"
            
            if not enhanced_path.exists():
                print(f"[SKIP] Output video missing for {config_name} on category '{category}'")
                continue
                
            print(f"[CALCULATING] Processing visual metrics for {config_name} on category '{category}'...")
            t0 = time.time()
            metrics = compute_metrics_for_videos(gt_path, str(enhanced_path))
            
            # Save metrics to JSON file
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4)
                
            dt = time.time() - t0
            print(f"[COMPLETED] Avg PSNR: {metrics['psnr']:.2f} | Avg SSIM: {metrics['ssim']:.4f} | Avg LPIPS: {metrics['lpips']:.4f} (took {dt:.2f}s)\n")

if __name__ == "__main__":
    import time
    main()
