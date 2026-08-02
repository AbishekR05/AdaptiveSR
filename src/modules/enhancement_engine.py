import importlib
import torch
from src.modules.model_registry import MODEL_REGISTRY
from src.utils.state_types import Decision

def get_inference_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

class EnhancementEngine:
    def __init__(self, device: str):
        self.device = device
        self._resolved_cache = {}

    def _resolve(self, model_name: str):
        if model_name in self._resolved_cache:
            return self._resolved_cache[model_name]
            
        if model_name not in MODEL_REGISTRY:
            raise KeyError(f"Model '{model_name}' is not registered in the Model Registry.")
            
        entry = MODEL_REGISTRY[model_name]
        module_path, fn_name = entry["infer_fn"].rsplit(".", 1)
        
        # Dynamically import module and retrieve function
        module = importlib.import_module(module_path)
        infer_fn = getattr(module, fn_name)
        
        self._resolved_cache[model_name] = infer_fn
        return infer_fn

    def enhance(self, frame_bgr, decision: Decision):
        infer_fn = self._resolve(decision.model)
        return infer_fn(frame_bgr, self.device, scale=decision.scale)
