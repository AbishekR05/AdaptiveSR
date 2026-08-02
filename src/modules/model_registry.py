MODEL_REGISTRY = {
    "tinysr": {
        "display_name": "FSRCNN (lightweight)",
        "loader": "src.modules.backends.fsrcnn_backend.load_model",
        "infer_fn": "src.modules.backends.fsrcnn_backend.infer",
        "expected_memory_mb": 50,
        "expected_latency_ms_cpu": 385,     # Measured on CPU (480p frame, scale=2)
        "expected_latency_ms_gpu": 127,     # Measured on GPU (GTX 1650, 480p frame, scale=2)
        "supported_scales": [2, 3, 4],
        "quality_rating": "medium",
        "requires_sequence": False,
    },
    "real_esrgan": {
        "display_name": "Real-ESRGAN",
        "loader": "src.modules.backends.realesrgan_backend.load_model",
        "infer_fn": "src.modules.backends.realesrgan_backend.infer",
        "expected_memory_mb": 800,
        "expected_latency_ms_cpu": 19060,   # Measured on CPU (480p frame, scale=2)
        "expected_latency_ms_gpu": 11110,   # Measured on GPU (GTX 1650, 480p frame, scale=2)
        "supported_scales": [2, 4],
        "quality_rating": "high",
        "requires_sequence": False,
    },
}
