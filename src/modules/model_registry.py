MODEL_REGISTRY = {
    "tinysr": {
        "display_name": "FSRCNN (lightweight)",
        "loader": "src.modules.backends.fsrcnn_backend.load_model",
        "infer_fn": "src.modules.backends.fsrcnn_backend.infer",
        "expected_memory_mb": 50,
        "expected_latency_ms_cpu": 400,     # Measured on CPU (480p frame, scale=2)
        "expected_latency_ms_gpu": 118,     # Measured on GPU (GTX 1650, 480p frame, scale=2)
        "supported_scales": [2, 3, 4],
        "quality_rating": "medium",
        "requires_sequence": False,
    },
    "real_esrgan": {
        "display_name": "Real-ESRGAN",
        "loader": "src.modules.backends.realesrgan_backend.load_model",
        "infer_fn": "src.modules.backends.realesrgan_backend.infer",
        "expected_memory_mb": 800,
        "expected_latency_ms_cpu": 18080,   # Measured on CPU (480p frame, scale=2)
        "expected_latency_ms_gpu": 8920,    # Measured on GPU (GTX 1650, 480p frame, scale=2)
        "supported_scales": [2, 4],
        "quality_rating": "high",
        "requires_sequence": False,
    },
    "basicvsr++": {
        "display_name": "BasicVSR++",
        "loader": "src.modules.backends.basicvsr_backend.load_model",
        "infer_fn": "src.modules.backends.basicvsr_backend.infer_sequence",
        "expected_memory_mb": 1500,         # Estimated minimum overhead (OOM ceiling risk)
        "expected_latency_ms_cpu": 35000,   # Estimated (CPU-only recurrent propagation is extremely slow)
        "expected_latency_ms_gpu": 450,     # Estimated (Recurrent temporal alignment GPU cost)
        "supported_scales": [4],
        "quality_rating": "very_high",
        "requires_sequence": True,
        "sequence_window": 5,
        "available": False,                 # Deferred due to MMCV Windows build dependency mismatch
    },
    "skip": {
        "display_name": "Skip (Passthrough)",
        "loader": None,
        "infer_fn": None,
        "expected_memory_mb": 0,
        "expected_latency_ms_cpu": 0,
        "expected_latency_ms_gpu": 0,
        "supported_scales": [1],
        "quality_rating": "low",
        "requires_sequence": False,
        "available": True,
    },
    "tinysr_int8": {
        "display_name": "FSRCNN (INT8 Quantized CPU)",
        "loader": "src.modules.backends.fsrcnn_backend_int8.load_model",
        "infer_fn": "src.modules.backends.fsrcnn_backend_int8.infer",
        "expected_memory_mb": 30,
        "expected_latency_ms_cpu": 100,      # Estimated 2x-4x speedup over FP32 (400ms)
        "expected_latency_ms_gpu": 999999,  # CPU only
        "supported_scales": [2],
        "quality_rating": "medium_low",
        "requires_sequence": False,
        "available": True,
    },
}
