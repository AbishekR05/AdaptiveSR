MODEL_REGISTRY = {
    "tinysr": {
        "display_name": "FSRCNN (lightweight)",
        "loader": "src.modules.backends.fsrcnn_backend.load_model",
        "infer_fn": "src.modules.backends.fsrcnn_backend.infer",
        "expected_memory_mb": 50,
        "expected_latency_ms_cpu": 8,       # Measured on CPU (64x64 frame, scale=2)
        "expected_latency_ms_gpu": 3,       # Placeholder: unverified (PyTorch CUDA not active)
        "supported_scales": [2, 3, 4],
        "quality_rating": "medium",
        "requires_sequence": False,
    },
    "real_esrgan": {
        "display_name": "Real-ESRGAN",
        "loader": "src.modules.backends.realesrgan_backend.load_model",
        "infer_fn": "src.modules.backends.realesrgan_backend.infer",
        "expected_memory_mb": 800,
        "expected_latency_ms_cpu": 410,     # Measured on CPU (64x64 frame, scale=2)
        "expected_latency_ms_gpu": 60,      # Placeholder: unverified (PyTorch CUDA not active)
        "supported_scales": [2, 4],
        "quality_rating": "high",
        "requires_sequence": False,
    },
}
