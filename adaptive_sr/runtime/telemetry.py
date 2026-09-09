"""
adaptive_sr.runtime.telemetry
=============================
Step 11 — Hardened per-chunk machine-readable telemetry schema and runtime state tracking.

Defines standardized, machine-readable telemetry records and state tracking objects
for the end-to-end AdaptiveSR streaming runtime.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List, Tuple


@dataclass
class ChunkIdentity:
    video_id: str
    chunk_id: str
    request_id: str
    edge_id: str
    cluster_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConfigurationTuple:
    representation_id: str
    target_resolution: str
    model_id: Optional[str]
    scale: Optional[int]
    device: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionTelemetry:
    decision: str  # "selected", "no_suitable_candidate", "no_feasible_candidates", "execution_failed"
    fuzzy_suitability: Optional[float]
    suitability_tier: Optional[str]
    rejection_reason: Optional[str] = None
    fallback_reason: Optional[str] = None
    min_suitability_threshold: float = 35.0
    decision_eligible: Optional[bool] = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TimingTelemetry:
    request_start_unix_timestamp: float
    completion_unix_timestamp: float
    client_elapsed_seconds: float
    download_transfer_time: float
    sr_processing_time: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NetworkTelemetry:
    measured_bandwidth_mbps: float
    rtt_seconds: float
    bytes_received: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResourceTelemetry:
    cpu_utilization_pct: float
    gpu_utilization_pct: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BufferTelemetry:
    buffer_before: float
    buffer_after: float
    stall_count: int
    stall_duration: float
    chunk_delivered: bool = True
    delivered_chunk_duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProvenanceTelemetry:
    fps_signal_provenance: Dict[str, Any] = field(default_factory=dict)
    bitrate_signal_provenance: Dict[str, Any] = field(default_factory=dict)
    resource_signal_provenance: Dict[str, Any] = field(default_factory=dict)
    decision_signal_provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ErrorTelemetry:
    error_code: str
    error_message: str
    traceback_available: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChunkTelemetry:
    identity: ChunkIdentity
    delivery_mode: str  # "sr", "native", "execution_failed"
    requested_configuration: Optional[Dict[str, Any]]
    executed_configuration: Optional[Dict[str, Any]]
    decision: DecisionTelemetry
    timing: TimingTelemetry
    network: NetworkTelemetry
    resource: ResourceTelemetry
    buffer: BufferTelemetry
    provenance: ProvenanceTelemetry
    is_injected_test_condition: bool = False
    injected_fields: List[str] = field(default_factory=list)
    error: Optional[ErrorTelemetry] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "identity": self.identity.to_dict(),
            "delivery_mode": self.delivery_mode,
            "requested_configuration": self.requested_configuration,
            "executed_configuration": self.executed_configuration,
            "decision": self.decision.to_dict(),
            "timing": self.timing.to_dict(),
            "network": self.network.to_dict(),
            "resource": self.resource.to_dict(),
            "buffer": self.buffer.to_dict(),
            "provenance": self.provenance.to_dict(),
            "is_injected_test_condition": self.is_injected_test_condition,
            "injected_fields": self.injected_fields,
        }
        if self.error:
            d["error"] = self.error.to_dict()
        return d


@dataclass
class RuntimeState:
    video_id: str
    current_chunk_id: Optional[str] = None
    buffer_seconds: float = 0.0
    stall_count: int = 0
    total_stall_duration: float = 0.0
    decision_count: int = 0
    configuration_switch_count: int = 0
    previous_executed_state: Optional[Tuple[Any, ...]] = None
    current_executed_state: Optional[Tuple[Any, ...]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "current_chunk_id": self.current_chunk_id,
            "buffer_seconds": self.buffer_seconds,
            "stall_count": self.stall_count,
            "total_stall_duration": self.total_stall_duration,
            "decision_count": self.decision_count,
            "configuration_switch_count": self.configuration_switch_count,
            "previous_executed_state": list(self.previous_executed_state) if self.previous_executed_state else None,
            "current_executed_state": list(self.current_executed_state) if self.current_executed_state else None,
        }
