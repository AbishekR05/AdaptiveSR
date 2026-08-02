import os
import yaml

DEFAULT_WEIGHTS = {"motion": 0.25, "texture": 0.25, "edges": 0.25, "blur_clarity": 0.25}

def load_weights_from_config(config_path="configs/decision_config.yaml") -> dict:
    """Loads complexity weights from the decision config file, with graceful fallbacks."""
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                if cfg and "complexity_weights" in cfg:
                    return cfg["complexity_weights"]
        except Exception:
            pass
    return DEFAULT_WEIGHTS

def estimate_complexity(scene_metrics: dict, weights: dict | None = None) -> float:
    """
    Estimates the final complexity score (0-1) based on a weighted sum of individual metrics.
    Note: Lower blur_clarity (blurrier image) increases the overall complexity contribution 
    via the (1 - blur_clarity) term.
    
    scene_metrics: dict with keys motion, texture, edges, blur_clarity
    weights: optional override; defaults to loading weights from config or equal weights.
    """
    w = weights or load_weights_from_config()
    
    # Fill in defaults if any keys are missing from the configuration
    w_motion = w.get("motion", 0.25)
    w_texture = w.get("texture", 0.25)
    w_edges = w.get("edges", 0.25)
    w_blur_clarity = w.get("blur_clarity", 0.25)

    complexity = (
        w_motion * scene_metrics.get("motion", 0.0)
        + w_texture * scene_metrics.get("texture", 0.0)
        + w_edges * scene_metrics.get("edges", 0.0)
        + w_blur_clarity * (1.0 - scene_metrics.get("blur_clarity", 1.0))
    )
    
    return min(max(float(complexity), 0.0), 1.0)
