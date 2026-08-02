# Workaround for compatibility issues in older BasicSR/RealESRGAN imports on Python 3.10+ and Torchvision >= 0.15
import sys
import types
import collections

# Fix 1: collections.Container deprecation patch
if not hasattr(collections, 'Container'):
    import collections.abc
    collections.Container = collections.abc.Container

# Fix 2: torchvision.transforms.functional_tensor module removal patch
try:
    import torchvision.transforms.functional as T_F
    functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
    functional_tensor.rgb_to_grayscale = T_F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor
except ImportError:
    pass

import os
import urllib.request
import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Model caching by (device, scale)
_model_cache = {}

def load_model(device: str, scale: int = 2) -> RealESRGANer:
    global _model_cache
    cache_key = (device, scale)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    if scale not in [2, 4]:
        raise ValueError("Real-ESRGAN backend only supports scale factors 2 and 4.")

    weights_dir = "models/real_esrgan"
    model_name = f"RealESRGAN_x{scale}plus.pth"
    weights_path = os.path.join(weights_dir, model_name)

    # Autonomous Weights Downloading if missing
    if not os.path.exists(weights_path):
        os.makedirs(weights_dir, exist_ok=True)
        if scale == 2:
            url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
        else:
            url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            
        print(f"Downloading Real-ESRGAN x{scale} weights from: {url}")
        try:
            # Download file
            urllib.request.urlretrieve(url, weights_path)
            print(f"Real-ESRGAN weights saved to: {weights_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download Real-ESRGAN weights from {url}: {e}")

    # Initialize RRDBNet architecture
    model = RRDBNet(
        num_in_ch=3, 
        num_out_ch=3, 
        num_feat=64, 
        num_block=23, 
        num_grow_ch=32, 
        scale=scale
    )

    # Disable FP16 (half) precision on CUDA to prevent underflow/black screen issues on GTX 1650
    half_precision = False

    # Initialize upsampler wrapper
    upsampler = RealESRGANer(
        scale=scale,
        model_path=weights_path,
        model=model,
        tile=0,           # Disabled tiling (tile=0) to run in a single forward pass for speed
        tile_pad=10,
        pre_pad=0,
        half=half_precision,
        device=device
    )

    print(f"Real-ESRGAN {model_name} initialized. Model device parameter: {next(upsampler.model.parameters()).device}")

    _model_cache[cache_key] = upsampler
    return upsampler


def infer(frame_bgr: np.ndarray, device: str, scale: int = 2, device_state = None) -> np.ndarray:
    upsampler = load_model(device, scale=scale)
    
    # Dynamic Tiling: Set tile size based on GPU workload constraints
    if device_state is not None:
        gpu_load = device_state.gpu if device_state.gpu is not None else 0.0
        threshold = 0.60
        if os.path.exists("configs/decision_config.yaml"):
            try:
                import yaml
                with open("configs/decision_config.yaml", "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    threshold = cfg.get("thresholds", {}).get("tile_size_healthy_gpu_threshold", 0.60)
            except Exception:
                pass
        
        if gpu_load >= threshold:
            upsampler.tile = 400
        else:
            upsampler.tile = 0
            
    # RealESRGANer expects BGR input and returns enhanced BGR output
    enhanced_frame, _ = upsampler.enhance(frame_bgr, outscale=scale)
    return enhanced_frame
