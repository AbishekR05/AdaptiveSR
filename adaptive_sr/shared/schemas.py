from pydantic import BaseModel
from typing import List, Optional

class Representation(BaseModel):
    representation_id: str
    resolution: str
    bitrate: int

class ChunkMetadata(BaseModel):
    chunk_id: str
    duration: float

class VideoManifest(BaseModel):
    video_id: str
    duration: float
    representations: List[Representation]
    chunks: List[ChunkMetadata]

class ChunkRequest(BaseModel):
    video_id: str
    chunk_id: str
    representation_id: str
    target_representation_id: Optional[str] = None
    base_representation_id: Optional[str] = None

class ClientTelemetry(BaseModel):
    request_start_time: float
    response_time: float
    download_duration: float
    bytes_received: int
    measured_throughput_mbps: float
    RTT: Optional[float] = None  # Client -> Edge RTT
    buffer_before: float
    buffer_after: float
    cache_hit: Optional[bool] = None
    stall_duration_seconds: float = 0.0

class EdgeTelemetry(BaseModel):
    request_id: str
    video_id: str
    chunk_id: str
    representation_id: str
    target_representation_id: Optional[str] = None
    base_representation_id: Optional[str] = None
    cache_hit: bool
    cloud_fetch_time: float
    edge_processing_time: float
    response_time: float
    bytes_sent: int
    rtt: Optional[float] = None  # Edge -> Cloud RTT
    sr_processing_time: Optional[float] = 0.0  # Placeholder for future VSR


from typing import Union, Literal
from pydantic import Field, field_validator, model_validator
from datetime import datetime, timezone

class VideoRepresentation(BaseModel):
    representation_id: str
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    resolution_label: str
    bitrate_kbps: int = Field(..., gt=0)
    codec: str
    fps: Union[int, Literal["source"]]

    @field_validator("codec")
    @classmethod
    def validate_codec(cls, v: str) -> str:
        supported_codecs = {"h264", "h265", "hevc", "vp9", "av1"}
        if v.lower() not in supported_codecs:
            raise ValueError(f"Unsupported codec: {v}. Supported codecs are: {supported_codecs}")
        return v.lower()

    @field_validator("fps")
    @classmethod
    def validate_fps(cls, v: Union[int, str]) -> Union[int, str]:
        if isinstance(v, int):
            if v <= 0:
                raise ValueError("FPS must be a positive integer.")
            if v not in {30, 60, 120}:
                raise ValueError("FPS must be one of 30, 60, 120.")
        elif isinstance(v, str):
            if v != "source":
                raise ValueError("FPS string value must be 'source'.")
        return v

    def materialize(self, source_fps: int) -> 'VideoRepresentation':
        """Materializes the representation by resolving 'source' FPS to the actual source FPS."""
        resolved_fps = source_fps if self.fps == "source" else self.fps
        return VideoRepresentation(
            representation_id=self.representation_id,
            width=self.width,
            height=self.height,
            resolution_label=self.resolution_label,
            bitrate_kbps=self.bitrate_kbps,
            codec=self.codec,
            fps=resolved_fps
        )

class RepresentationConfig(BaseModel):
    representations: List[VideoRepresentation]

    @model_validator(mode="after")
    def validate_unique_representations(self) -> 'RepresentationConfig':
        # Unique representation IDs
        ids = [r.representation_id for r in self.representations]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate representation IDs are not allowed.")

        # No duplicate resolution + FPS variants (width, height, fps)
        variants = [(r.width, r.height, r.fps) for r in self.representations]
        if len(variants) != len(set(variants)):
            raise ValueError("Duplicate variant configurations (width, height, fps) are not allowed.")
        return self

    def materialize(self, source_fps: int) -> 'RepresentationConfig':
        """
        Materializes the config by resolving 'source' FPS for all representations 
        and validating uniqueness of the materialized variants.
        """
        materialized_reps = [rep.materialize(source_fps) for rep in self.representations]
        
        # Verify representation IDs remain unique
        ids = [r.representation_id for r in materialized_reps]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate representation IDs are not allowed in materialized config.")
            
        # Verify resolved (width, height, fps) variants are unique
        variants = [(r.width, r.height, r.fps) for r in materialized_reps]
        if len(variants) != len(set(variants)):
            raise ValueError(
                f"Duplicate materialized variants (width, height, fps) detected: "
                f"after resolving source_fps={source_fps}."
            )
            
        return RepresentationConfig(representations=materialized_reps)


class RepresentationChunk(BaseModel):
    chunk_id: str
    representation_id: str
    frame_start: int = Field(..., ge=0)
    frame_end: int = Field(..., ge=0)
    start_time_seconds: float = Field(..., ge=0.0)
    end_time_seconds: float = Field(..., ge=0.0)
    duration_seconds: float = Field(..., gt=0.0)
    file_path: str
    size_bytes: int = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_frame_range(self) -> 'RepresentationChunk':
        if self.frame_start > self.frame_end:
            raise ValueError(f"frame_start ({self.frame_start}) cannot exceed frame_end ({self.frame_end})")
        if self.start_time_seconds > self.end_time_seconds:
            raise ValueError("start_time_seconds cannot exceed end_time_seconds")
        return self


class RepresentationChunkMapping(BaseModel):
    representation_chunks: List[RepresentationChunk]

    def validate_invariants(self, config: RepresentationConfig, source_metadata: dict, profile_chunks: List[dict]):
        """
        Validates mapping invariants against the configured representations, 
        source video metadata, and the authoritative Step 1 profile chunks.
        """
        valid_rep_ids = {r.representation_id for r in config.representations}
        
        # Build map of representation ID to resolved/materialized FPS
        rep_id_to_fps = {}
        for r in config.representations:
            resolved_fps = source_metadata["fps"] if r.fps == "source" else r.fps
            rep_id_to_fps[r.representation_id] = resolved_fps
        
        rep_to_chunks = {}
        for rc in self.representation_chunks:
            if rc.representation_id not in valid_rep_ids:
                raise ValueError(
                    f"Representation ID '{rc.representation_id}' referenced in mapping "
                    f"does not exist in representation config."
                )
            rep_to_chunks.setdefault(rc.representation_id, []).append(rc)
            
        seen_pairs = set()
        for rc in self.representation_chunks:
            pair = (rc.representation_id, rc.chunk_id)
            if pair in seen_pairs:
                raise ValueError(f"Duplicate mapping entry for (representation, chunk): {pair}")
            seen_pairs.add(pair)

        logical_chunk_ids = [c["chunk_id"] for c in profile_chunks]
        if not logical_chunk_ids:
            raise ValueError("Authoritative timeline has no chunks.")

        for rep_id in valid_rep_ids:
            chunks_for_rep = rep_to_chunks.get(rep_id, [])
            mapped_chunk_ids = [c.chunk_id for c in chunks_for_rep]
            
            missing = set(logical_chunk_ids) - set(mapped_chunk_ids)
            if missing:
                raise ValueError(
                    f"Missing chunks for representation '{rep_id}': {missing}"
                )
                
            extra = set(mapped_chunk_ids) - set(logical_chunk_ids)
            if extra:
                raise ValueError(
                    f"Unknown chunk IDs mapped for representation '{rep_id}': {extra}"
                )
                
        # A. Logical temporal boundaries are identical across representations and match Step 1
        for p_chunk in profile_chunks:
            c_id = p_chunk["chunk_id"]
            expected_start_time = p_chunk["start_time_seconds"]
            expected_end_time = p_chunk["end_time_seconds"]
            expected_duration = p_chunk["duration_seconds"]
            
            for rep_id, chunks_for_rep in rep_to_chunks.items():
                matching_rc = next((c for c in chunks_for_rep if c.chunk_id == c_id), None)
                if matching_rc is None:
                    raise ValueError(f"Logical chunk '{c_id}' not found in representation '{rep_id}'.")
                    
                if matching_rc.start_time_seconds != expected_start_time or matching_rc.end_time_seconds != expected_end_time:
                    raise ValueError(
                        f"Timestamp range mismatch for chunk '{c_id}' under representation '{rep_id}': "
                        f"expected ({expected_start_time}, {expected_end_time}), "
                        f"got ({matching_rc.start_time_seconds}, {matching_rc.end_time_seconds})."
                    )
                    
                if matching_rc.duration_seconds != expected_duration:
                    raise ValueError(
                        f"Duration mismatch for chunk '{c_id}' under representation '{rep_id}': "
                        f"expected {expected_duration}, got {matching_rc.duration_seconds}."
                    )

        # Validate representation-local frame ranges and chronological sequences
        for rep_id, chunks_for_rep in rep_to_chunks.items():
            rep_fps = rep_id_to_fps[rep_id]
            sorted_chunks = sorted(chunks_for_rep, key=lambda x: x.chunk_id)
            
            # C. Each representation has a valid ordered frame range: first chunk starts at local frame 0
            if sorted_chunks[0].frame_start != 0:
                raise ValueError(
                    f"First chunk '{sorted_chunks[0].chunk_id}' for representation '{rep_id}' "
                    f"does not start at local frame 0 (starts at {sorted_chunks[0].frame_start})."
                )
                
            # I. First logical chunk begins at the source timeline start (0.0s)
            if sorted_chunks[0].start_time_seconds != 0.0:
                raise ValueError(
                    f"First chunk '{sorted_chunks[0].chunk_id}' for representation '{rep_id}' "
                    f"does not start at time 0.0s (starts at {sorted_chunks[0].start_time_seconds})."
                )
                
            # J. Final logical chunk ends at the source timeline end
            expected_total_duration = source_metadata["frame_count"] / source_metadata["fps"]
            # Allow minor floating point tolerances
            if abs(sorted_chunks[-1].end_time_seconds - expected_total_duration) > 0.01:
                raise ValueError(
                    f"Final chunk '{sorted_chunks[-1].chunk_id}' for representation '{rep_id}' "
                    f"does not end at final timeline time {expected_total_duration:.3f}s "
                    f"(ends at {sorted_chunks[-1].end_time_seconds:.3f}s)."
                )
                
            for idx in range(len(sorted_chunks)):
                rc = sorted_chunks[idx]
                
                if rc.duration_seconds <= 0:
                    raise ValueError(f"Chunk '{rc.chunk_id}' duration must be positive.")
                if rc.frame_start > rc.frame_end:
                    raise ValueError(f"Chunk '{rc.chunk_id}' frame_start > frame_end.")
                    
                # B. Representation-local frame ranges are internally consistent with the representation's materialized FPS and logical duration
                local_frame_count = rc.frame_end - rc.frame_start + 1
                expected_local_frame_count = int(round(rc.duration_seconds * rep_fps))
                # Validate that local frame count matches expected count derived from duration and rep FPS (with a tiny rounding tolerance)
                if abs(local_frame_count - expected_local_frame_count) > 1:
                    raise ValueError(
                        f"Frame count inconsistency for chunk '{rc.chunk_id}' under representation '{rep_id}': "
                        f"frame count ({local_frame_count}) does not match expected ({expected_local_frame_count}) "
                        f"for duration {rc.duration_seconds}s at {rep_fps} FPS."
                    )
                    
                if idx < len(sorted_chunks) - 1:
                    next_rc = sorted_chunks[idx + 1]
                    
                    # F. Logical chunk ordering remains monotonic (by chunk_id)
                    if next_rc.chunk_id <= rc.chunk_id:
                        raise ValueError(f"Non-monotonic chunk IDs: {rc.chunk_id} -> {next_rc.chunk_id}")
                        
                    # H. Logical chunks contain no temporal overlaps
                    if next_rc.start_time_seconds < rc.end_time_seconds:
                        raise ValueError(
                            f"Overlap detected between chunk '{rc.chunk_id}' (ends at {rc.end_time_seconds}s) "
                            f"and chunk '{next_rc.chunk_id}' (starts at {next_rc.start_time_seconds}s) "
                            f"under representation '{rep_id}'."
                        )
                        
                    # G. Logical chunks contain no temporal gaps
                    if next_rc.start_time_seconds != rc.end_time_seconds:
                        raise ValueError(
                            f"Gap detected between chunk '{rc.chunk_id}' (ends at {rc.end_time_seconds}s) "
                            f"and chunk '{next_rc.chunk_id}' (starts at {next_rc.start_time_seconds}s) "
                            f"under representation '{rep_id}'."
                        )
                        
                    # C. Check no overlaps/gaps in representation-local frame ranges
                    if next_rc.frame_start <= rc.frame_end:
                        raise ValueError(
                            f"Local frame overlap detected between chunk '{rc.chunk_id}' (ends at {rc.frame_end}) "
                            f"and chunk '{next_rc.chunk_id}' (starts at {next_rc.frame_start}) "
                            f"under representation '{rep_id}'."
                        )
                    if next_rc.frame_start != rc.frame_end + 1:
                        raise ValueError(
                            f"Local frame gap detected between chunk '{rc.chunk_id}' (ends at {rc.frame_end}) "
                            f"and chunk '{next_rc.chunk_id}' (starts at {next_rc.frame_start}) "
                            f"under representation '{rep_id}'."
                        )


class NetworkMeasurement(BaseModel):
    request_id: str
    network_path: Literal["client_edge", "edge_cloud"]
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z")
    chunk_id: Optional[str] = None
    representation_id: Optional[str] = None
    bytes_transferred: Optional[int] = None
    rtt_ms: Optional[float] = None
    transfer_duration_seconds: Optional[float] = None
    measured_throughput_mbps: Optional[float] = None

    @model_validator(mode="after")
    def validate_measurement(self) -> 'NetworkMeasurement':
        # Zero-byte RTT probes do not produce a fake throughput value
        if self.bytes_transferred == 0 or self.bytes_transferred is None:
            if self.measured_throughput_mbps is not None and self.measured_throughput_mbps > 0.0:
                raise ValueError("Zero-byte RTT probes must not fabricate a non-zero throughput.")
                
        # Throughput validation
        if self.bytes_transferred is not None and self.transfer_duration_seconds is not None:
            if self.transfer_duration_seconds > 0.0:
                expected_mbps = (self.bytes_transferred * 8) / (self.transfer_duration_seconds * 1_000_000.0)
                if self.measured_throughput_mbps is not None and abs(self.measured_throughput_mbps - expected_mbps) > 0.01:
                    raise ValueError(
                        f"Throughput mismatch: got {self.measured_throughput_mbps} Mbps, "
                        f"expected {expected_mbps} Mbps."
                    )
        return self


class EdgeResourceTelemetry(BaseModel):
    """Step 4 — Edge Resource Telemetry.

    Represents a single real-time snapshot of compute resource state on an
    Edge node.  All measurements are taken from the live system via psutil.

    This schema is SEPARATE from NetworkMeasurement by design:
      - NetworkMeasurement answers: "What is happening to the network?"
      - EdgeResourceTelemetry answers: "What is happening to the Edge compute?"

    FIELDS
    ------
    timestamp : str
        ISO-8601 UTC timestamp of the snapshot, e.g.
        "2026-08-20T12:00:00.000000Z".  Always timezone-aware.

    cluster_id : str
        Cluster identity from Edge service configuration.

    edge_id : str
        Node identity from Edge service configuration.

    cpu_cores_total : int
        Number of logical CPU cores exposed to the process by the OS.
        (psutil.cpu_count(logical=True))

    cpu_utilization : float
        System-wide CPU utilisation averaged across all logical cores,
        expressed as a percentage [0, 100].
        Measured from a real psutil call — NOT synthesised.

    cpu_cores_available : float
        ESTIMATED number of logical CPU cores not currently consumed by
        active workloads.
        Formula: cpu_cores_total × (1 − cpu_utilization / 100)
        IMPORTANT: This is an estimation, not an OS-level reservation.
        The Step 4 Edge implementation has no resource reservation
        mechanism.  "Available" here means "currently not accounted for
        by observed utilization" — it does NOT mean cores are reserved or
        guaranteed for a specific workload.
        CPU = PRIMARY resource dimension for AdaptiveSR.

    memory_total_bytes : int
        Total physical RAM in bytes (psutil.virtual_memory().total).
        RAM = SECONDARY observed resource (not yet used for allocation).

    memory_used_bytes : int
        Currently used physical RAM in bytes (psutil.virtual_memory().used).

    memory_utilization : float
        memory_used_bytes / memory_total_bytes × 100, expressed as a
        percentage [0, 100].

    active_requests : int
        Number of requests currently being processed by the Edge service.
        This value is caller-supplied — the request handler passes the
        current concurrency count at the time of the snapshot.

    queue_depth : int
        Number of requests pending after admission but before execution.
        The synchronous FastAPI Edge implementation has NO application-level
        work queue.  Callers pass 0 and this must be documented as:
            "No explicit application-level work queue exists in the current
             synchronous Edge implementation."
        A real SR scheduling queue belongs to a later step.
    """
    timestamp: str
    cluster_id: str
    edge_id: str
    cpu_cores_total: int
    cpu_utilization: float          # [0, 100]
    cpu_cores_available: float      # Estimated; see field docstring
    memory_total_bytes: int
    memory_used_bytes: int
    memory_utilization: float       # [0, 100]
    active_requests: int
    queue_depth: int
