import os
import sys
import pytest
from pydantic import ValidationError

# Ensure root of repository is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adaptive_sr.shared.schemas import (
    VideoRepresentation,
    RepresentationConfig,
    RepresentationChunk,
    RepresentationChunkMapping
)

@pytest.fixture
def base_config():
    return RepresentationConfig(
        representations=[
            VideoRepresentation(
                representation_id="360p",
                width=640,
                height=360,
                resolution_label="360p",
                bitrate_kbps=800,
                codec="h264",
                fps=30
            ),
            VideoRepresentation(
                representation_id="720p",
                width=1280,
                height=720,
                resolution_label="720p",
                bitrate_kbps=2500,
                codec="h264",
                fps=30
            )
        ]
    )

@pytest.fixture
def source_meta_30fps():
    return {"frame_count": 90, "fps": 30.0}

@pytest.fixture
def logical_timeline_30fps():
    # 3 chunks of 1.0s (30 frames each at 30 FPS source)
    return [
        {
            "chunk_id": "0000",
            "start_time_seconds": 0.0,
            "end_time_seconds": 1.0,
            "duration_seconds": 1.0,
            "start_frame": 0,
            "end_frame": 29
        },
        {
            "chunk_id": "0001",
            "start_time_seconds": 1.0,
            "end_time_seconds": 2.0,
            "duration_seconds": 1.0,
            "start_frame": 30,
            "end_frame": 59
        },
        {
            "chunk_id": "0002",
            "start_time_seconds": 2.0,
            "end_time_seconds": 3.0,
            "duration_seconds": 1.0,
            "start_frame": 60,
            "end_frame": 89
        }
    ]

def test_valid_mapping_multiple_representations_accepted(base_config, source_meta_30fps, logical_timeline_30fps):
    """Verifies that a valid mapping containing all chunks and representations passes checks."""
    chunks = []
    # 2 representations x 3 chunks = 6 mapping entries
    for rep in ["360p", "720p"]:
        for c in logical_timeline_30fps:
            chunks.append(
                RepresentationChunk(
                    chunk_id=c["chunk_id"],
                    representation_id=rep,
                    frame_start=c["start_frame"],
                    frame_end=c["end_frame"],
                    start_time_seconds=c["start_time_seconds"],
                    end_time_seconds=c["end_time_seconds"],
                    duration_seconds=c["duration_seconds"],
                    file_path=f"chunks/{rep}/chunk_{c['chunk_id']}.mp4",
                    size_bytes=10000
                )
            )
            
    mapping = RepresentationChunkMapping(representation_chunks=chunks)
    mapping.validate_invariants(base_config, source_meta_30fps, logical_timeline_30fps)

def test_one_representation_multiple_chunks_accepted(source_meta_30fps, logical_timeline_30fps):
    """Verifies that a configuration with only one representation maps and validates correctly."""
    single_config = RepresentationConfig(
        representations=[
            VideoRepresentation(
                representation_id="360p",
                width=640,
                height=360,
                resolution_label="360p",
                bitrate_kbps=800,
                codec="h264",
                fps=30
            )
        ]
    )
    chunks = []
    for c in logical_timeline_30fps:
        chunks.append(
            RepresentationChunk(
                chunk_id=c["chunk_id"],
                representation_id="360p",
                frame_start=c["start_frame"],
                frame_end=c["end_frame"],
                start_time_seconds=c["start_time_seconds"],
                end_time_seconds=c["end_time_seconds"],
                duration_seconds=c["duration_seconds"],
                file_path=f"chunks/360p/chunk_{c['chunk_id']}.mp4",
                size_bytes=10000
            )
        )
    mapping = RepresentationChunkMapping(representation_chunks=chunks)
    mapping.validate_invariants(single_config, source_meta_30fps, logical_timeline_30fps)

def test_missing_representation_chunk_rejected(base_config, source_meta_30fps, logical_timeline_30fps):
    """Verifies that if any configured representation is missing a mapped chunk, validation fails."""
    chunks = []
    for c in logical_timeline_30fps:
        chunks.append(
            RepresentationChunk(
                chunk_id=c["chunk_id"],
                representation_id="360p",
                frame_start=c["start_frame"],
                frame_end=c["end_frame"],
                start_time_seconds=c["start_time_seconds"],
                end_time_seconds=c["end_time_seconds"],
                duration_seconds=c["duration_seconds"],
                file_path="chunks/chunk.mp4",
                size_bytes=10000
            )
        )
    for c in logical_timeline_30fps[:2]:  # Omit chunk 0002 for 720p
        chunks.append(
            RepresentationChunk(
                chunk_id=c["chunk_id"],
                representation_id="720p",
                frame_start=c["start_frame"],
                frame_end=c["end_frame"],
                start_time_seconds=c["start_time_seconds"],
                end_time_seconds=c["end_time_seconds"],
                duration_seconds=c["duration_seconds"],
                file_path="chunks/chunk.mp4",
                size_bytes=10000
            )
        )
    mapping = RepresentationChunkMapping(representation_chunks=chunks)
    with pytest.raises(ValueError) as exc_info:
        mapping.validate_invariants(base_config, source_meta_30fps, logical_timeline_30fps)
    assert "Missing chunks for representation '720p'" in str(exc_info.value)

def test_unknown_representation_id_rejected(base_config, source_meta_30fps, logical_timeline_30fps):
    """Verifies that mappings referencing representation IDs not in the config are rejected."""
    chunks = []
    for c in logical_timeline_30fps:
        chunks.append(
            RepresentationChunk(
                chunk_id=c["chunk_id"],
                representation_id="1080p",  # Non-existent in base_config
                frame_start=c["start_frame"],
                frame_end=c["end_frame"],
                start_time_seconds=c["start_time_seconds"],
                end_time_seconds=c["end_time_seconds"],
                duration_seconds=c["duration_seconds"],
                file_path="chunks/chunk.mp4",
                size_bytes=10000
            )
        )
    mapping = RepresentationChunkMapping(representation_chunks=chunks)
    with pytest.raises(ValueError) as exc_info:
        mapping.validate_invariants(base_config, source_meta_30fps, logical_timeline_30fps)
    assert "referenced in mapping does not exist" in str(exc_info.value)

def test_duplicate_representation_chunk_pairs_rejected(base_config, source_meta_30fps, logical_timeline_30fps):
    """Verifies that duplicate (representation_id, chunk_id) pairs are rejected."""
    chunks = []
    for rep in ["360p", "720p"]:
        for c in logical_timeline_30fps:
            chunks.append(
                RepresentationChunk(
                    chunk_id=c["chunk_id"],
                    representation_id=rep,
                    frame_start=c["start_frame"],
                    frame_end=c["end_frame"],
                    start_time_seconds=c["start_time_seconds"],
                    end_time_seconds=c["end_time_seconds"],
                    duration_seconds=c["duration_seconds"],
                    file_path="chunks/chunk.mp4",
                    size_bytes=10000
                )
            )
    # Add duplicate mapping
    chunks.append(
        RepresentationChunk(
            chunk_id="0001",
            representation_id="360p",
            frame_start=30,
            frame_end=59,
            start_time_seconds=1.0,
            end_time_seconds=2.0,
            duration_seconds=1.0,
            file_path="chunks/chunk.mp4",
            size_bytes=10000
        )
    )
    mapping = RepresentationChunkMapping(representation_chunks=chunks)
    with pytest.raises(ValueError) as exc_info:
        mapping.validate_invariants(base_config, source_meta_30fps, logical_timeline_30fps)
    assert "Duplicate mapping entry for (representation, chunk)" in str(exc_info.value)

def test_mismatched_frame_count_rejected(base_config, source_meta_30fps, logical_timeline_30fps):
    """Verifies that frame ranges inconsistent with logical duration and representation FPS are rejected."""
    chunks = []
    for rep in ["360p", "720p"]:
        for c in logical_timeline_30fps:
            f_end = c["end_frame"]
            if rep == "720p" and c["chunk_id"] == "0001":
                f_end += 5  # Deviates from duration * 30 FPS!
            chunks.append(
                RepresentationChunk(
                    chunk_id=c["chunk_id"],
                    representation_id=rep,
                    frame_start=c["start_frame"],
                    frame_end=f_end,
                    start_time_seconds=c["start_time_seconds"],
                    end_time_seconds=c["end_time_seconds"],
                    duration_seconds=c["duration_seconds"],
                    file_path="chunks/chunk.mp4",
                    size_bytes=10000
                )
            )
    mapping = RepresentationChunkMapping(representation_chunks=chunks)
    with pytest.raises(ValueError) as exc_info:
        mapping.validate_invariants(base_config, source_meta_30fps, logical_timeline_30fps)
    assert "Frame count inconsistency for chunk '0001'" in str(exc_info.value)

def test_mismatched_timestamps_rejected(base_config, source_meta_30fps, logical_timeline_30fps):
    """Verifies that logical timestamp ranges deviating from the authoritative timeline are rejected."""
    chunks = []
    for rep in ["360p", "720p"]:
        for c in logical_timeline_30fps:
            t_end = c["end_time_seconds"]
            if rep == "720p" and c["chunk_id"] == "0001":
                t_end += 0.5
            chunks.append(
                RepresentationChunk(
                    chunk_id=c["chunk_id"],
                    representation_id=rep,
                    frame_start=c["start_frame"],
                    frame_end=c["end_frame"],
                    start_time_seconds=c["start_time_seconds"],
                    end_time_seconds=t_end,
                    duration_seconds=c["duration_seconds"],
                    file_path="chunks/chunk.mp4",
                    size_bytes=10000
                )
            )
    mapping = RepresentationChunkMapping(representation_chunks=chunks)
    with pytest.raises(ValueError) as exc_info:
        mapping.validate_invariants(base_config, source_meta_30fps, logical_timeline_30fps)
    assert "Timestamp range mismatch for chunk '0001'" in str(exc_info.value)

def test_local_frame_gaps_rejected(base_config, source_meta_30fps, logical_timeline_30fps):
    """Verifies that frame gaps inside representation-local indexes are rejected."""
    chunks = []
    for rep in ["360p", "720p"]:
        for c in logical_timeline_30fps:
            f_start = c["start_frame"]
            if rep == "360p" and c["chunk_id"] == "0001":
                f_start += 2 # Creates frame gap on 360p local indexing
            chunks.append(
                RepresentationChunk(
                    chunk_id=c["chunk_id"],
                    representation_id=rep,
                    frame_start=f_start,
                    frame_end=c["end_frame"],
                    start_time_seconds=c["start_time_seconds"],
                    end_time_seconds=c["end_time_seconds"],
                    duration_seconds=c["duration_seconds"],
                    file_path="chunks/chunk.mp4",
                    size_bytes=10000
                )
            )
    patched_timeline = [
        dict(c, start_frame=(c["start_frame"] + 2 if c["chunk_id"] == "0001" else c["start_frame"]))
        for c in logical_timeline_30fps
    ]
    # For 720p to match the timeline checks, apply frame offset shift
    for rc in chunks:
        if rc.representation_id == "720p" and rc.chunk_id == "0001":
            rc.frame_start += 2
            
    mapping = RepresentationChunkMapping(representation_chunks=chunks)
    with pytest.raises(ValueError) as exc_info:
        mapping.validate_invariants(base_config, source_meta_30fps, patched_timeline)
    assert "Local frame gap detected between chunk '0000'" in str(exc_info.value)

def test_local_frame_overlaps_rejected(base_config):
    """Verifies that frame overlaps inside representation-local indexes are rejected."""
    # Use timeline with 2.5s duration total:
    # chunk 0000: 1.0s, chunk 0001: 1.0s, chunk 0002: 0.5s
    custom_source_meta = {"frame_count": 60, "fps": 24.0} # 60 / 24 = 2.5 seconds
    timeline = [
        {
            "chunk_id": "0000",
            "start_time_seconds": 0.0,
            "end_time_seconds": 1.0,
            "duration_seconds": 1.0,
            "start_frame": 0,
            "end_frame": 29
        },
        {
            "chunk_id": "0001",
            "start_time_seconds": 1.0,
            "end_time_seconds": 2.0,
            "duration_seconds": 1.0,
            "start_frame": 15,  # Overlap starting at 15!
            "end_frame": 44   # 30 frames total (internally consistent for 1.0s at 30 FPS)
        },
        {
            "chunk_id": "0002",
            "start_time_seconds": 2.0,
            "end_time_seconds": 2.5,
            "duration_seconds": 0.5,
            "start_frame": 45,
            "end_frame": 59   # 15 frames total (consistent for 0.5s at 30 FPS)
        }
    ]
    
    chunks = []
    for rep in ["360p", "720p"]:
        for c in timeline:
            chunks.append(
                RepresentationChunk(
                    chunk_id=c["chunk_id"],
                    representation_id=rep,
                    frame_start=c["start_frame"],
                    frame_end=c["end_frame"],
                    start_time_seconds=c["start_time_seconds"],
                    end_time_seconds=c["end_time_seconds"],
                    duration_seconds=c["duration_seconds"],
                    file_path="chunks/chunk.mp4",
                    size_bytes=10000
                )
            )
            
    mapping = RepresentationChunkMapping(representation_chunks=chunks)
    with pytest.raises(ValueError) as exc_info:
        mapping.validate_invariants(base_config, custom_source_meta, timeline)
    assert "Local frame overlap detected between chunk '0000'" in str(exc_info.value)

def test_non_monotonic_chunk_ids_rejected(base_config, source_meta_30fps, logical_timeline_30fps):
    """Verifies that non-monotonic chunk sequencing raises validation errors."""
    chunks = []
    for rep in ["360p", "720p"]:
        for i, c in enumerate(logical_timeline_30fps):
            t_start = c["start_time_seconds"]
            t_end = c["end_time_seconds"]
            if c["chunk_id"] == "0001":
                t_start = 2.5
                t_end = 3.5
            chunks.append(
                RepresentationChunk(
                    chunk_id=c["chunk_id"],
                    representation_id=rep,
                    frame_start=c["start_frame"],
                    frame_end=c["end_frame"],
                    start_time_seconds=t_start,
                    end_time_seconds=t_end,
                    duration_seconds=c["duration_seconds"],
                    file_path="chunks/chunk.mp4",
                    size_bytes=10000
                )
            )
    patched_timeline = [
        dict(c, start_time_seconds=(2.5 if c["chunk_id"] == "0001" else c["start_time_seconds"]),
                end_time_seconds=(3.5 if c["chunk_id"] == "0001" else c["end_time_seconds"]))
        for c in logical_timeline_30fps
    ]
    for rc in chunks:
        if rc.representation_id == "720p" and rc.chunk_id == "0001":
            rc.start_time_seconds = 2.5
            rc.end_time_seconds = 3.5
            
    mapping = RepresentationChunkMapping(representation_chunks=chunks)
    with pytest.raises(ValueError) as exc_info:
        mapping.validate_invariants(base_config, source_meta_30fps, patched_timeline)
    assert "Overlap detected" in str(exc_info.value) or "Gap detected" in str(exc_info.value)

def test_variable_final_chunk_duration_accepted(base_config):
    """Verifies that variable final chunk durations and non-2-second target chunk assumptions are accepted."""
    custom_source_meta = {"frame_count": 83, "fps": 83 / 2.75} # 2.75s expected duration
    timeline = [
        {
            "chunk_id": "0000",
            "start_time_seconds": 0.0,
            "end_time_seconds": 2.0,
            "duration_seconds": 2.0,
            "start_frame": 0,
            "end_frame": 59
        },
        {
            "chunk_id": "0001",
            "start_time_seconds": 2.0,
            "end_time_seconds": 2.75,
            "duration_seconds": 0.75,
            "start_frame": 60,
            "end_frame": 82
        }
    ]
    chunks = []
    for rep in ["360p", "720p"]:
        for c in timeline:
            chunks.append(
                RepresentationChunk(
                    chunk_id=c["chunk_id"],
                    representation_id=rep,
                    frame_start=c["start_frame"],
                    frame_end=c["end_frame"],
                    start_time_seconds=c["start_time_seconds"],
                    end_time_seconds=c["end_time_seconds"],
                    duration_seconds=c["duration_seconds"],
                    file_path="chunks/chunk.mp4",
                    size_bytes=10000
                )
            )
    mapping = RepresentationChunkMapping(representation_chunks=chunks)
    mapping.validate_invariants(base_config, custom_source_meta, timeline)

def test_fps_combinations():
    """
    Verifies that the mapping validates correctly under all specified source and representation FPS scenarios:
    1. 60 FPS source + 60 FPS representation.
    2. 60 FPS source + 30 FPS representation.
    3. 60 FPS source + 120 FPS representation.
    4. 120 FPS source + 30 FPS representation.
    5. 120 FPS source + 60 FPS representation.
    6. 120 FPS source + 120 FPS representation.
    7. 30 FPS source + 30 FPS representation.
    """
    # Define representation configuration containing 30, 60, and 120 FPS streams
    config = RepresentationConfig(
        representations=[
            VideoRepresentation(representation_id="rep_30", width=640, height=360, resolution_label="360p", bitrate_kbps=800, codec="h264", fps=30),
            VideoRepresentation(representation_id="rep_60", width=1280, height=720, resolution_label="720p", bitrate_kbps=2500, codec="h264", fps=60),
            VideoRepresentation(representation_id="rep_120", width=1920, height=1080, resolution_label="1080p", bitrate_kbps=6000, codec="h264", fps=120)
        ]
    )

    # Scenarios: (source_fps, representation_id, rep_fps)
    scenarios = [
        # 60 FPS Source
        (60, "rep_60", 60),
        (60, "rep_30", 30),
        (60, "rep_120", 120),
        # 120 FPS Source
        (120, "rep_30", 30),
        (120, "rep_60", 60),
        (120, "rep_120", 120),
        # 30 FPS Source
        (30, "rep_30", 30)
    ]

    for src_fps, rep_id, rep_fps in scenarios:
        # Create authoritative timeline of 3.0s duration
        source_meta = {"frame_count": src_fps * 3, "fps": float(src_fps)}
        timeline = [
            {
                "chunk_id": "0000",
                "start_time_seconds": 0.0,
                "end_time_seconds": 2.0,
                "duration_seconds": 2.0,
                "start_frame": 0,
                "end_frame": (src_fps * 2) - 1
            },
            {
                "chunk_id": "0001",
                "start_time_seconds": 2.0,
                "end_time_seconds": 3.0,
                "duration_seconds": 1.0,
                "start_frame": src_fps * 2,
                "end_frame": (src_fps * 3) - 1
            }
        ]

        # Map representation chunks. Note that the frame boundaries for the representation
        # chunks are local to that representation's FPS (duration * rep_fps).
        chunks = [
            RepresentationChunk(
                chunk_id="0000",
                representation_id=rep_id,
                frame_start=0,
                frame_end=(rep_fps * 2) - 1,
                start_time_seconds=0.0,
                end_time_seconds=2.0,
                duration_seconds=2.0,
                file_path=f"chunks/{rep_id}_0000.mp4",
                size_bytes=20000
            ),
            RepresentationChunk(
                chunk_id="0001",
                representation_id=rep_id,
                frame_start=rep_fps * 2,
                frame_end=(rep_fps * 3) - 1,
                start_time_seconds=2.0,
                end_time_seconds=3.0,
                duration_seconds=1.0,
                file_path=f"chunks/{rep_id}_0001.mp4",
                size_bytes=10000
            )
        ]

        # Use a scoped representation config containing only the tested representation for isolation
        scoped_config = RepresentationConfig(
            representations=[next(r for r in config.representations if r.representation_id == rep_id)]
        )

        mapping = RepresentationChunkMapping(representation_chunks=chunks)
        
        # Verify logical temporal range remains identical while allowing frame ranges to differ
        mapping.validate_invariants(scoped_config, source_meta, timeline)
        
        # Verify specific frame count bounds
        assert chunks[0].frame_end - chunks[0].frame_start + 1 == rep_fps * 2
        assert chunks[1].frame_end - chunks[1].frame_start + 1 == rep_fps
        assert chunks[0].start_time_seconds == timeline[0]["start_time_seconds"]
        assert chunks[0].end_time_seconds == timeline[0]["end_time_seconds"]
        assert chunks[1].start_time_seconds == timeline[1]["start_time_seconds"]
        assert chunks[1].end_time_seconds == timeline[1]["end_time_seconds"]
