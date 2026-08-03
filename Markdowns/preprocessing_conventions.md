# Preprocessing Conventions

This document summarizes the project's preprocessing conventions for dataset creation, model inputs, metrics, and export/quantization.

## Dataset generation
- Script: [benchmark/generate_dataset.py](benchmark/generate_dataset.py)
- LR generation: downscale HR frames using `cv2.resize(..., interpolation=cv2.INTER_AREA)`.
- Outputs: `benchmark_data/{name}_gt.mp4` and `benchmark_data/{name}_lr.mp4`.

## Model input preprocessing (inference time)
- Files: [src/modules/backends/fsrcnn_backend.py](src/modules/backends/fsrcnn_backend.py) and [src/modules/backends/fsrcnn_backend_int8.py](src/modules/backends/fsrcnn_backend_int8.py)
- Steps (applies to both PyTorch and ONNX INT8 backends):
  1. Color: BGR -> RGB (`cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`).
  2. Normalize: divide by 255.0 to convert to float in [0, 1].
  3. Shape: convert HWC -> CHW, add batch dim -> shape `(1, 3, H, W)`.
  4. Device: move tensor to target device (PyTorch) or provide float32 numpy input (ONNX).

Postprocess (both backends): clip or round values, convert to uint8, transpose back to HWC, RGB -> BGR.

## Metrics preprocessing
- File: [benchmark/compute_quality_metrics.py](benchmark/compute_quality_metrics.py)
- PSNR/SSIM: frames are compared in 8-bit uint range (0-255). The code resizes GT to match enhanced output using `cv2.INTER_LANCZOS4` when needed.
- LPIPS: inputs are converted to float tensors and mapped to [-1, 1] using `(img/255 - 0.5) * 2.0` (see `to_tensor`).

## Quantization and model export
- File: [benchmark/quantize_tinysr.py](benchmark/quantize_tinysr.py)
- Exports PyTorch FSRCNN to ONNX with dynamic axes, then applies dynamic INT8 quantization (output saved to `models/tinysr/fsrcnn_x{scale}_int8.onnx`).

## Conventions & tips
- Default color order throughout the pipeline is BGR for images on disk, converted to RGB for model input.
- Use `/255.0` normalization for model inference; use `(-1,1)` mapping only where LPIPS is required.
- Typical frame size for export/ONNX dummy input: `(1, 3, 480, 640)`.
- Interpolation choices:
  - Downscaling (dataset): `cv2.INTER_AREA`.
  - Resize for metrics (GT->enhanced): `cv2.INTER_LANCZOS4`.

## Quick commands
```bash
python benchmark/generate_dataset.py   # create synthetic benchmark_data
python benchmark/quantize_tinysr.py  # export and quantize tinysr to INT8
python benchmark/compute_quality_metrics.py  # compute PSNR/SSIM/LPIPS
```

If you want this doc moved to a different location or expanded with code references, tell me where/how.
