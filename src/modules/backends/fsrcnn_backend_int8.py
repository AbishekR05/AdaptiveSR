import os
import cv2
import numpy as np
import onnxruntime as ort

# Cache for the loaded ONNX sessions
_session_cache = {}

def load_model(device: str = "cpu", scale: int = 2) -> ort.InferenceSession:
    global _session_cache
    if scale != 2:
        raise ValueError("Quantized INT8 FSRCNN backend only supports scale=2 currently.")
        
    cache_key = scale
    if cache_key in _session_cache:
        return _session_cache[cache_key]
        
    weights_path = os.path.join("models/tinysr", f"fsrcnn_x{scale}_int8.onnx")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"INT8 ONNX model weights not found at: {weights_path}. "
            "Please run 'python benchmark/quantize_tinysr.py' to generate them."
        )
        
    # We explicitly force CPU execution for INT8 dynamic quantization
    providers = ["CPUExecutionProvider"]
    
    opts = ort.SessionOptions()
    import multiprocessing
    opts.intra_op_num_threads = max(1, multiprocessing.cpu_count() // 2)
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    session = ort.InferenceSession(weights_path, opts, providers=providers)
    _session_cache[cache_key] = session
    return session

def infer(frame_bgr: np.ndarray, device: str = "cpu", scale: int = 2) -> np.ndarray:
    session = load_model(device, scale=scale)
    
    # 1. Preprocess: BGR -> RGB, normalization, transpose to BCHW (1, 3, H, W)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = frame_rgb.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    # 2. Run ORT Inference
    ort_inputs = {session.get_inputs()[0].name: input_tensor}
    ort_outs = session.run(None, ort_inputs)
    output_tensor = ort_outs[0]
    
    # 3. Postprocess: squeeze batch, transpose back to HWC (H, W, 3), convert to uint8, RGB -> BGR
    output_tensor = np.squeeze(output_tensor, axis=0)
    output_tensor = np.transpose(output_tensor, (1, 2, 0))
    output_np = np.clip(output_tensor * 255.0, 0, 255).astype(np.uint8)
    enhanced_frame_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
    
    return enhanced_frame_bgr
