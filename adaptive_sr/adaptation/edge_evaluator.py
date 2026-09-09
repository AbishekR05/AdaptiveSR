"""
adaptive_sr.adaptation.edge_evaluator
=====================================
Step 9 — Edge Selection & Resource Allocation Layer.

Evaluates whether available Edge nodes can support a requested SR workload under their
current compute and network conditions, producing machine-readable candidate/resource signals
(EdgeResourceSignal) for Step 10.

This layer evaluates hardware capability, load state, network telemetry, and workload feasibility.
It does NOT make final adaptive decisions or calculate arbitrary scalar utility scores.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field


@dataclass
class EdgeResourceState:
    edge_id: str = "edge_01"
    cluster_id: str = "cluster_01"
    # Hardware capability
    cpu_cores_total: Optional[int] = None
    gpu_available: bool = False
    gpu_device_name: Optional[str] = None
    gpu_memory_total_bytes: Optional[int] = None
    supported_devices: List[str] = field(default_factory=lambda: ["cpu"])
    supported_models: List[str] = field(default_factory=lambda: ["tinysr", "real_esrgan"])
    # Current resource state (load state)
    cpu_utilization_percent: Optional[float] = None
    memory_utilization_percent: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    gpu_memory_free_bytes: Optional[int] = None
    active_requests: int = 0
    queue_depth: int = 0
    # Network telemetry
    cloud_edge_rtt_ms: Optional[float] = None
    client_edge_rtt_ms: Optional[float] = None
    measured_bandwidth_mbps: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EdgeResourceSignal:
    edge_id: str
    cluster_id: str
    model_id: str
    scale: int
    device: str
    base_representation_id: str
    target_resolution: str
    resource_feasible: bool
    network_feasible: Optional[bool]
    feasibility_status: str  # "feasible", "infeasible", "unknown"
    hardware_capability: Dict[str, Any]
    resource_availability: Dict[str, Any]
    network_telemetry: Dict[str, Any]
    sr_telemetry: Dict[str, Any]
    measurement_provenance: str
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EdgeResourceEvaluator:
    """
    Edge Resource Evaluator.
    Determines whether an Edge node can support a requested SR workload under current
    compute and network conditions.
    """

    @classmethod
    def evaluate_node(
        cls,
        state: EdgeResourceState,
        model_id: str = "tinysr",
        scale: int = 2,
        device: str = "cpu",
        base_representation_id: str = "360p",
        target_resolution: str = "1280x720",
        source_fps: float = 30.0,
        required_gpu_memory_bytes: Optional[int] = None,
        sr_processing_time_ms: Optional[float] = None,
        measurement_provenance: str = "direct_measurement"
    ) -> EdgeResourceSignal:
        """
        Evaluates resource capability and feasibility for a single Edge node.
        """
        warnings: List[str] = []
        req_device = str(device).lower()

        # 1. Hardware Capability Check
        device_supported = False
        if req_device in [d.lower() for d in state.supported_devices]:
            device_supported = True
        elif req_device.startswith("cuda") and state.gpu_available:
            device_supported = True

        model_supported = model_id in state.supported_models

        # 2. Resource Feasibility Assessment
        resource_feasible = True

        # Requirement 9.4: NO silent GPU -> CPU fallback. If CUDA requested but unavailable, infeasible.
        if req_device.startswith("cuda"):
            if not state.gpu_available:
                resource_feasible = False
                warnings.append(f"CUDA requested on Edge node '{state.edge_id}', but GPU is unavailable (no silent CPU fallback allowed).")
            elif required_gpu_memory_bytes is not None and state.gpu_memory_free_bytes is not None:
                if state.gpu_memory_free_bytes < required_gpu_memory_bytes:
                    resource_feasible = False
                    warnings.append(
                        f"Insufficient free GPU memory on '{state.edge_id}': required {required_gpu_memory_bytes / 1e6:.1f} MB, free {state.gpu_memory_free_bytes / 1e6:.1f} MB."
                    )
            elif state.gpu_utilization_percent is not None and state.gpu_utilization_percent >= 98.0:
                resource_feasible = False
                warnings.append(f"GPU on Edge node '{state.edge_id}' is overloaded ({state.gpu_utilization_percent:.1f}% utilization).")

        if not device_supported:
            resource_feasible = False
            warnings.append(f"Requested device '{device}' is not supported by Edge node '{state.edge_id}'.")

        if not model_supported:
            resource_feasible = False
            warnings.append(f"Requested SR model '{model_id}' is not supported by Edge node '{state.edge_id}'.")

        # CPU Utilization Overload Check
        if state.cpu_utilization_percent is not None and state.cpu_utilization_percent >= 95.0:
            resource_feasible = False
            warnings.append(f"CPU on Edge node '{state.edge_id}' is overloaded ({state.cpu_utilization_percent:.1f}% utilization).")

        # 3. Network Feasibility Assessment
        network_feasible: Optional[bool] = None
        if state.cloud_edge_rtt_ms is not None:
            network_feasible = state.cloud_edge_rtt_ms < 5000.0  # basic sanity check
            warnings.append("Cloud-to-Edge RTT measured; Cloud RTT does not represent end-to-end streaming latency.")
        else:
            warnings.append("Network telemetry (Cloud/Client RTT) unavailable or missing.")

        # 4. Feasibility Status Determination
        if not resource_feasible:
            feasibility_status = "infeasible"
        elif state.cpu_utilization_percent is None and state.gpu_utilization_percent is None and req_device.startswith("cuda"):
            feasibility_status = "unknown"
            warnings.append("GPU telemetry missing; feasibility status marked as unknown.")
        else:
            feasibility_status = "feasible"

        # Hardware capability dict
        hardware_capability = {
            "cpu_cores_total": state.cpu_cores_total,
            "gpu_available": state.gpu_available,
            "gpu_device_name": state.gpu_device_name,
            "gpu_memory_total_bytes": state.gpu_memory_total_bytes,
            "supported_devices": state.supported_devices,
            "supported_models": state.supported_models,
        }

        # Resource availability dict (current load)
        resource_availability = {
            "cpu_utilization_percent": state.cpu_utilization_percent,
            "memory_utilization_percent": state.memory_utilization_percent,
            "gpu_utilization_percent": state.gpu_utilization_percent,
            "gpu_memory_free_bytes": state.gpu_memory_free_bytes,
            "active_requests": state.active_requests,
            "queue_depth": state.queue_depth,
        }

        # Network telemetry dict
        network_telemetry = {
            "cloud_edge_rtt_ms": state.cloud_edge_rtt_ms,
            "client_edge_rtt_ms": state.client_edge_rtt_ms,
            "measured_bandwidth_mbps": state.measured_bandwidth_mbps,
        }

        # SR performance telemetry dict
        est_fps = (1000.0 / sr_processing_time_ms) if (sr_processing_time_ms and sr_processing_time_ms > 0.0) else None
        sr_telemetry = {
            "sr_processing_time_ms": sr_processing_time_ms,
            "estimated_processing_fps": est_fps,
        }

        return EdgeResourceSignal(
            edge_id=state.edge_id,
            cluster_id=state.cluster_id,
            model_id=str(model_id),
            scale=int(scale),
            device=str(device),
            base_representation_id=str(base_representation_id),
            target_resolution=str(target_resolution),
            resource_feasible=bool(resource_feasible),
            network_feasible=network_feasible,
            feasibility_status=feasibility_status,
            hardware_capability=hardware_capability,
            resource_availability=resource_availability,
            network_telemetry=network_telemetry,
            sr_telemetry=sr_telemetry,
            measurement_provenance=str(measurement_provenance),
            warnings=warnings
        )

    @classmethod
    def evaluate_candidates(
        cls,
        candidates: List[EdgeResourceState],
        model_id: str = "tinysr",
        scale: int = 2,
        device: str = "cpu",
        base_representation_id: str = "360p",
        target_resolution: str = "1280x720",
        source_fps: float = 30.0
    ) -> List[EdgeResourceSignal]:
        """
        Evaluates a list of Edge candidate nodes.
        Feasible nodes are returned first, ordered by resource availability.
        No arbitrary weighted utility function is applied (Step 10 performs final decision).
        """
        signals = [
            cls.evaluate_node(
                state=cand,
                model_id=model_id,
                scale=scale,
                device=device,
                base_representation_id=base_representation_id,
                target_resolution=target_resolution,
                source_fps=source_fps,
                measurement_provenance="edge_candidate_eval"
            )
            for cand in candidates
        ]

        # Deterministic ordering: feasible nodes first, then by CPU utilization ascending
        def sort_key(s: EdgeResourceSignal) -> Tuple[int, float]:
            is_feas = 0 if s.resource_feasible else 1
            cpu_util = s.resource_availability.get("cpu_utilization_percent")
            cpu_val = cpu_util if cpu_util is not None else 50.0
            return (is_feas, cpu_val)

        return sorted(signals, key=sort_key)

    @classmethod
    def evaluate_from_step6_telemetry(
        cls,
        telemetry: Dict[str, Any],
        edge_id: str = "edge_01",
        cluster_id: str = "cluster_01"
    ) -> EdgeResourceSignal:
        """
        Ingests Step 6 Edge runtime telemetry / headers and creates an EdgeResourceSignal.
        """
        norm = {str(k).lower(): v for k, v in telemetry.items()}

        sr_model = telemetry.get("sr_model_id") or norm.get("x-sr-model") or "tinysr"
        try:
            scale = int(telemetry.get("sr_scale") or norm.get("x-sr-scale") or 2)
        except (ValueError, TypeError):
            scale = 2

        device = telemetry.get("sr_device") or norm.get("x-sr-device") or "cpu"
        base_rep = telemetry.get("representation_id") or norm.get("base_representation_id") or "360p"
        rtt_val = telemetry.get("rtt") or norm.get("x-edge-cloud-rtt")

        rtt_ms = None
        if rtt_val is not None and str(rtt_val).upper() != "N/A":
            try:
                rf = float(rtt_val)
                rtt_ms = rf * 1000.0 if rf < 10.0 else rf
            except (ValueError, TypeError):
                rtt_ms = None

        sr_time_sec = telemetry.get("sr_processing_time") or norm.get("x-sr-processing-time")
        sr_time_ms = None
        if sr_time_sec is not None:
            try:
                stf = float(sr_time_sec)
                sr_time_ms = stf * 1000.0 if stf < 10.0 else stf
            except (ValueError, TypeError):
                sr_time_ms = None

        import torch
        gpu_avail = torch.cuda.is_available()

        state = EdgeResourceState(
            edge_id=edge_id,
            cluster_id=cluster_id,
            gpu_available=gpu_avail,
            supported_devices=["cpu", "cuda"] if gpu_avail else ["cpu"],
            cloud_edge_rtt_ms=rtt_ms
        )

        return cls.evaluate_node(
            state=state,
            model_id=sr_model,
            scale=scale,
            device=device,
            base_representation_id=base_rep,
            sr_processing_time_ms=sr_time_ms,
            measurement_provenance="step6_edge_telemetry"
        )
