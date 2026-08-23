"""
adaptive_sr.benchmarking.adapters.registry
==========================================
Step 5.2 — SR Adapter Registry and Discovery.

Maps model_ids to their respective adapter implementations and supports
discovery of available/unavailable backends.
"""

from typing import List, Dict, Any, Type, Optional

from adaptive_sr.benchmarking.adapters.base import BaseSRAdapter
from adaptive_sr.benchmarking.adapters.fsrcnn import FSRCNNAdapter
from adaptive_sr.benchmarking.adapters.fsrcnn_int8 import FSRCNNInt8Adapter
from adaptive_sr.benchmarking.adapters.real_esrgan import RealESRGANAdapter
from adaptive_sr.benchmarking.adapters.basicvsrpp import BasicVSRppAdapter

# Registration map linking model_id to its Adapter implementation class
ADAPTER_MAP: Dict[str, Type[BaseSRAdapter]] = {
    "tinysr": FSRCNNAdapter,
    "tinysr_int8": FSRCNNInt8Adapter,
    "real_esrgan": RealESRGANAdapter,
    "basicvsr++": BasicVSRppAdapter
}


def get_adapter(model_id: str) -> BaseSRAdapter:
    """Resolves a model_id to its instantiated adapter class.

    Parameters
    ----------
    model_id : str
        The unique ID of the model to retrieve (e.g., 'tinysr').

    Returns
    -------
    BaseSRAdapter
        An instance of the resolved adapter class.
    """
    if model_id not in ADAPTER_MAP:
        raise ValueError(
            f"Model ID '{model_id}' is not registered. "
            f"Available registered models: {list(ADAPTER_MAP.keys())}"
        )
    return ADAPTER_MAP[model_id]()


def list_registered_models() -> List[str]:
    """Returns a list of all model IDs currently registered in the system."""
    return list(ADAPTER_MAP.keys())


def list_available_models() -> List[str]:
    """Returns a list of model IDs whose backends are available and runnable."""
    available = []
    for model_id in ADAPTER_MAP.keys():
        adapter = get_adapter(model_id)
        if adapter.is_available():
            available.append(model_id)
    return available


def get_model_status_report() -> Dict[str, Dict[str, Any]]:
    """Generates a capability discovery dictionary for all registered models.
    Useful for future benchmarking summary reports.
    """
    report = {}
    for model_id in ADAPTER_MAP.keys():
        adapter = get_adapter(model_id)
        report[model_id] = {
            "display_name": adapter.display_name,
            "backend": adapter.backend,
            "scale_factors": adapter.scale_factors,
            "temporal_or_spatial": adapter.temporal_or_spatial,
            "precision": adapter.precision,
            "is_available": adapter.is_available(),
            "unavailable_reason": adapter.get_unavailable_reason()
        }
    return report
