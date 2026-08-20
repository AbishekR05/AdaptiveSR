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

