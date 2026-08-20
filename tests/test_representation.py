import os
import sys
import pytest
from pydantic import ValidationError

# Ensure root of repository is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adaptive_sr.shared.schemas import VideoRepresentation, RepresentationConfig

def test_valid_representation_accepted():
    """Verifies that a fully valid representation and configuration is accepted."""
    data = {
        "representations": [
            {
                "representation_id": "360p",
                "width": 640,
                "height": 360,
                "resolution_label": "360p",
                "bitrate_kbps": 800,
                "codec": "h264",
                "fps": "source"
            },
            {
                "representation_id": "720p",
                "width": 1280,
                "height": 720,
                "resolution_label": "720p",
                "bitrate_kbps": 2500,
                "codec": "h264",
                "fps": 60
            }
        ]
    }
    config = RepresentationConfig(**data)
    assert len(config.representations) == 2
    assert config.representations[0].representation_id == "360p"
    assert config.representations[0].fps == "source"
    assert config.representations[1].fps == 60

def test_same_resolution_different_fps_accepted():
    """Verifies that identical resolutions are accepted if they have different FPS values."""
    data = {
        "representations": [
            {
                "representation_id": "720p_30",
                "width": 1280,
                "height": 720,
                "resolution_label": "720p",
                "bitrate_kbps": 1500,
                "codec": "h264",
                "fps": 30
            },
            {
                "representation_id": "720p_60",
                "width": 1280,
                "height": 720,
                "resolution_label": "720p",
                "bitrate_kbps": 2500,
                "codec": "h264",
                "fps": 60
            },
            {
                "representation_id": "720p_120",
                "width": 1280,
                "height": 720,
                "resolution_label": "720p",
                "bitrate_kbps": 4000,
                "codec": "h264",
                "fps": 120
            }
        ]
    }
    config = RepresentationConfig(**data)
    assert len(config.representations) == 3
    assert config.representations[0].fps == 30
    assert config.representations[1].fps == 60
    assert config.representations[2].fps == 120

def test_different_resolutions_same_fps_accepted():
    """Verifies that different resolutions at the same FPS are accepted."""
    data = {
        "representations": [
            {
                "representation_id": "360p_60",
                "width": 640,
                "height": 360,
                "resolution_label": "360p",
                "bitrate_kbps": 800,
                "codec": "h264",
                "fps": 60
            },
            {
                "representation_id": "720p_60",
                "width": 1280,
                "height": 720,
                "resolution_label": "720p",
                "bitrate_kbps": 2500,
                "codec": "h264",
                "fps": 60
            }
        ]
    }
    config = RepresentationConfig(**data)
    assert len(config.representations) == 2

def test_duplicate_representation_ids_rejected():
    """Verifies that duplicate representation IDs are rejected."""
    data = {
        "representations": [
            {
                "representation_id": "360p",
                "width": 640,
                "height": 360,
                "resolution_label": "360p",
                "bitrate_kbps": 800,
                "codec": "h264",
                "fps": "source"
            },
            {
                "representation_id": "360p",  # Duplicate ID!
                "width": 1280,
                "height": 720,
                "resolution_label": "720p",
                "bitrate_kbps": 2500,
                "codec": "h264",
                "fps": "source"
            }
        ]
    }
    with pytest.raises(ValidationError) as exc_info:
        RepresentationConfig(**data)
    assert "Duplicate representation IDs are not allowed" in str(exc_info.value)

def test_duplicate_materialized_variants_rejected():
    """Verifies that identical width, height, and FPS combinations are rejected."""
    data = {
        "representations": [
            {
                "representation_id": "360p_h264",
                "width": 640,
                "height": 360,
                "resolution_label": "360p",
                "bitrate_kbps": 800,
                "codec": "h264",
                "fps": 30
            },
            {
                "representation_id": "360p_h265",
                "width": 640,  # Duplicate Resolution and FPS variant!
                "height": 360,
                "resolution_label": "360p",
                "bitrate_kbps": 600,
                "codec": "h265",
                "fps": 30
            }
        ]
    }
    with pytest.raises(ValidationError) as exc_info:
        RepresentationConfig(**data)
    assert "Duplicate variant configurations (width, height, fps) are not allowed" in str(exc_info.value)

def test_invalid_resolution_rejected():
    """Verifies that non-positive widths or heights are rejected."""
    # Invalid width
    with pytest.raises(ValidationError):
        VideoRepresentation(
            representation_id="360p",
            width=0,
            height=360,
            resolution_label="360p",
            bitrate_kbps=800,
            codec="h264",
            fps="source"
        )
    # Invalid height
    with pytest.raises(ValidationError):
        VideoRepresentation(
            representation_id="360p",
            width=640,
            height=-10,
            resolution_label="360p",
            bitrate_kbps=800,
            codec="h264",
            fps="source"
        )

def test_invalid_bitrate_rejected():
    """Verifies that non-positive bitrates are rejected."""
    with pytest.raises(ValidationError):
        VideoRepresentation(
            representation_id="360p",
            width=640,
            height=360,
            resolution_label="360p",
            bitrate_kbps=0,
            codec="h264",
            fps="source"
        )

def test_invalid_fps_rejected():
    """Verifies that invalid FPS settings (not 30/60/120 or 'source') are rejected."""
    # Positive integer but not in supported set
    with pytest.raises(ValidationError):
        VideoRepresentation(
            representation_id="360p",
            width=640,
            height=360,
            resolution_label="360p",
            bitrate_kbps=800,
            codec="h264",
            fps=24
        )
    # Negative integer
    with pytest.raises(ValidationError):
        VideoRepresentation(
            representation_id="360p",
            width=640,
            height=360,
            resolution_label="360p",
            bitrate_kbps=800,
            codec="h264",
            fps=-30
        )
    # Unsupported string literal
    with pytest.raises(ValidationError):
        VideoRepresentation(
            representation_id="360p",
            width=640,
            height=360,
            resolution_label="360p",
            bitrate_kbps=800,
            codec="h264",
            fps="adaptive"
        )

def test_multiple_fps_values_accepted():
    """Verifies that 30, 60, 120 FPS are accepted."""
    for fps in [30, 60, 120]:
        rep = VideoRepresentation(
            representation_id=f"360p_{fps}fps",
            width=640,
            height=360,
            resolution_label="360p",
            bitrate_kbps=800,
            codec="h264",
            fps=fps
        )
        assert rep.fps == fps

def test_fps_source_materialization():
    """Verifies that 'fps: source' correctly materializes to actual source FPS."""
    rep = VideoRepresentation(
        representation_id="360p",
        width=640,
        height=360,
        resolution_label="360p",
        bitrate_kbps=800,
        codec="h264",
        fps="source"
    )
    materialized = rep.materialize(source_fps=120)
    assert materialized.fps == 120
    assert materialized.representation_id == "360p"
    assert materialized.width == 640

def test_source_and_explicit_fps_collision_on_materialization():
    """Verifies that 'source' + explicit FPS detects a duplicate variant collision AFTER materialization."""
    data = {
        "representations": [
            {
                "representation_id": "720p_source",
                "width": 1280,
                "height": 720,
                "resolution_label": "720p",
                "bitrate_kbps": 2000,
                "codec": "h264",
                "fps": "source"
            },
            {
                "representation_id": "720p_60",
                "width": 1280,
                "height": 720,
                "resolution_label": "720p",
                "bitrate_kbps": 2500,
                "codec": "h264",
                "fps": 60
            }
        ]
    }
    config = RepresentationConfig(**data)  # Valid pre-materialization since "source" != 60
    
    # Resolving source_fps=60 causes collision!
    with pytest.raises(ValueError) as exc_info:
        config.materialize(source_fps=60)
    assert "Duplicate materialized variants" in str(exc_info.value)
    
    # Resolving source_fps=30 remains valid!
    materialized = config.materialize(source_fps=30)
    assert len(materialized.representations) == 2

def test_target_base_decision_not_required():
    """Verifies that the representation schema does not require or declare base/target representation scheduling variables."""
    fields = VideoRepresentation.model_fields
    assert "target_representation_id" not in fields
    assert "base_representation_id" not in fields
