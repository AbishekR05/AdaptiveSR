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

