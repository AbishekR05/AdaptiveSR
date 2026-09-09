"""
adaptive_sr.runtime.telemetry
=============================
Step 11 — Per-chunk machine-readable telemetry schema and runtime state tracking.

Defines standardized, machine-readable telemetry records and state tracking objects
for the end-to-end AdaptiveSR streaming runtime.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List


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
class SelectedConfiguration:
    representation_id: str
    target_resolution: str
    model_id: str
    scale: int
    device: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionTelemetry:
    decision: str  # "selected", "no_suitable_candidate", "no_feasible_candidates", "execution_failed"
    fuzzy_suitability: Optional[float]
    suitability_tier: Optional[str]
    rejection_reason: Optional[str] = None
    min_suitability_threshold: float = 35.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TimingTelemetry:
    request_start_time: float
    completion_time: float
    download_transfer_time: float
    sr_processing_time: float
    total_chunk_completion_time: float

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
class ChunkTelemetry:
    identity: ChunkIdentity
    selected_configuration: Optional[SelectedConfiguration]
    decision: DecisionTelemetry
    timing: TimingTelemetry
    network: NetworkTelemetry
    resource: ResourceTelemetry
    buffer: BufferTelemetry
    provenance: ProvenanceTelemetry
    is_injected_test_condition: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "identity": self.identity.to_dict(),
            "selected_configuration": self.selected_configuration.to_dict() if self.selected_configuration else None,
            "decision": self.decision.to_dict(),
            "timing": self.timing.to_dict(),
            "network": self.network.to_dict(),
            "resource": self.resource.to_dict(),
            "buffer": self.buffer.to_dict(),
            "provenance": self.provenance.to_dict(),
            "is_injected_test_condition": self.is_injected_test_condition,
        }
        if self.error:
            d["error"] = self.error
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
    previous_selected_configuration: Optional[Dict[str, Any]] = None
    current_selected_configuration: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
