import os
import sys
import json
import shutil
import pytest
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import patch

# Ensure root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adaptive_sr.profiling.profile_video import run_profiler, get_file_sha256, profile_chunk
from src.modules.video_loader import VideoLoader

def generate_synthetic_video(path: Path, fps: int, duration_seconds: float = 2.2):
    """Generates a synthetic testing video with a moving circle to produce motion."""
    w, h = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    total_frames = int(fps * duration_seconds)
    for i in range(total_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Draw a moving white circle
        x = int(w / 2 + (w / 4) * np.sin(2 * np.pi * i / fps))
        cv2.circle(frame, (x, h // 2), 20, (255, 255, 255), -1)
        out.write(frame)
    out.release()

@pytest.fixture
def temp_test_dir(tmp_path):
    """Fixture to handle temp directories for tests."""
    test_dir = tmp_path / "test_profiling"
    test_dir.mkdir(parents=True, exist_ok=True)
    yield test_dir
    # Cleanup
    if test_dir.exists():
        shutil.rmtree(test_dir)

def test_video_metadata_extraction(temp_test_dir):
    """Verifies VideoLoader extracts metadata accurately."""
    video_path = temp_test_dir / "meta_test.mp4"
    generate_synthetic_video(video_path, fps=30, duration_seconds=1.5)
    
    loader = VideoLoader(str(video_path))
    meta = loader.get_metadata()
    
    assert meta["width"] == 320
    assert meta["height"] == 240
    assert meta["fps"] == pytest.approx(30.0)
    assert meta["frame_count"] == 45
    assert meta["duration"] == pytest.approx(1.5)

def test_fps_frame_counts_and_temporal_offsets(temp_test_dir):
    """Verifies that frame counts and temporal offsets scale correctly across 30, 60, and 120 FPS."""
    # Test cases for FPS validation
    fps_cases = [30, 60, 120]
    
    for fps in fps_cases:
        video_path = temp_test_dir / f"test_{fps}fps.mp4"
        generate_synthetic_video(video_path, fps=fps, duration_seconds=2.2)
        
        # Output directory for this test case
        out_dir = temp_test_dir / f"out_{fps}fps"
        
        profile_path, manifest_path = run_profiler(
            input_video=str(video_path),
            output_dir=str(out_dir),
            chunk_duration=2.0,
            temporal_window_s=0.0333
        )
        
        with open(profile_path, "r", encoding="utf-8") as f:
            profile = json.load(f)
            
        assert profile["source"]["fps"] == pytest.approx(fps)
        assert len(profile["chunks"]) == 2  # 2.2 seconds should be split into 2 chunks
        
        # Verify chunk 0 frame count and boundaries
        chunk0 = profile["chunks"][0]
        assert chunk0["chunk_id"] == "0000"
        assert chunk0["start_frame"] == 0
        assert chunk0["end_frame"] == (fps * 2) - 1
        assert chunk0["frame_count"] == fps * 2  # exactly 2.0s duration at given FPS
        
        # Verify chunk 1 frame count
        chunk1 = profile["chunks"][1]
        assert chunk1["chunk_id"] == "0001"
        assert chunk1["frame_count"] == int(round(fps * 0.2))  # remaining 0.2s duration
        
        # Verify temporal offset selection
        # N = round(fps * temporal_window_s)
        expected_N = max(1, int(round(fps * 0.0333)))
        if fps == 30:
            assert expected_N == 1
        elif fps == 60:
            assert expected_N == 2
        elif fps == 120:
            assert expected_N == 4

def test_chunk_aggregation(temp_test_dir):
    """Verifies that mean, p95, and max metrics are correctly aggregated."""
    video_path = temp_test_dir / "agg_test.mp4"
    generate_synthetic_video(video_path, fps=30, duration_seconds=1.0)
    
    out_dir = temp_test_dir / "out_agg"
    profile_path, _ = run_profiler(
        input_video=str(video_path),
        output_dir=str(out_dir),
        chunk_duration=1.0,
        temporal_window_s=0.0333
    )
    
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)
        
    chunk = profile["chunks"][0]
    
    # Verify aggregation structure
    for feature in ["motion", "spatial_complexity"]:
        assert "mean" in chunk[feature]
        assert "p95" in chunk[feature]
        assert "max" in chunk[feature]
        assert chunk[feature]["mean"] <= chunk[feature]["p95"] <= chunk[feature]["max"]
        
    for feature in ["texture_density", "edge_density", "blur"]:
        assert "mean" in chunk[feature]
        assert "p95" in chunk[feature]
        assert "max" not in chunk[feature]
        assert chunk[feature]["mean"] <= chunk[feature]["p95"]

def test_profile_schema_and_integrity(temp_test_dir):
    """Verifies required fields are present in profile JSON and manifest JSON."""
    video_path = temp_test_dir / "schema_test.mp4"
    generate_synthetic_video(video_path, fps=30, duration_seconds=1.0)
    
    out_dir = temp_test_dir / "out_schema"
    profile_path, manifest_path = run_profiler(
        input_video=str(video_path),
        output_dir=str(out_dir),
        chunk_duration=1.0,
        temporal_window_s=0.0333
    )
    
    # 1. Profile JSON schema check
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)
        
    assert "schema_version" in profile
    assert "video_id" in profile
    assert "source" in profile
    assert "profiling_config" in profile
    assert "chunks" in profile
    
    # Video source fields
    src = profile["source"]
    for field in ["filename", "duration_seconds", "fps", "width", "height", "frame_count", "codec", "has_audio"]:
        assert field in src
        
    # Chunk fields
    chunk = profile["chunks"][0]
    for field in ["chunk_id", "start_time_seconds", "end_time_seconds", "duration_seconds", "start_frame", "end_frame", "frame_count", "motion", "texture_density", "edge_density", "blur", "spatial_complexity"]:
        assert field in chunk
        
    # Check that file hashes are NOT in the content features
    assert "file_hash" not in chunk
    
    # 2. Manifest JSON check
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    assert "schema_version" in manifest
    assert "video_id" in manifest
    assert "source_file_path" in manifest
    assert "source_file_hash" in manifest
    assert "generated_profile_path" in manifest
    assert "chunks" in manifest
    
    chunk_manifest = manifest["chunks"][0]
    assert "chunk_id" in chunk_manifest
    assert "file_path" in chunk_manifest
    assert "file_hash" in chunk_manifest

def test_repeatability(temp_test_dir):
    """Verifies that running twice on the same video with same config produces identical profiles."""
    video_path = temp_test_dir / "repeat_test.mp4"
    generate_synthetic_video(video_path, fps=30, duration_seconds=2.5)
    
    out_dir_1 = temp_test_dir / "out_repeat_1"
    out_dir_2 = temp_test_dir / "out_repeat_2"
    
    p1, m1 = run_profiler(str(video_path), str(out_dir_1), chunk_duration=2.0, temporal_window_s=0.0333)
    p2, m2 = run_profiler(str(video_path), str(out_dir_2), chunk_duration=2.0, temporal_window_s=0.0333)
    
    with open(p1, "r", encoding="utf-8") as f:
        profile1 = json.load(f)
    with open(p2, "r", encoding="utf-8") as f:
        profile2 = json.load(f)
        
    # Compare source and configuration metadata
    assert profile1["schema_version"] == profile2["schema_version"]
    assert profile1["video_id"] == profile2["video_id"]
    assert profile1["source"]["frame_count"] == profile2["source"]["frame_count"]
    assert profile1["profiling_config"]["chunk_duration_seconds"] == profile2["profiling_config"]["chunk_duration_seconds"]
    
    # Compare exact chunk boundaries and aggregated values
    for c1, c2 in zip(profile1["chunks"], profile2["chunks"]):
        assert c1["chunk_id"] == c2["chunk_id"]
        assert c1["start_frame"] == c2["start_frame"]
        assert c1["end_frame"] == c2["end_frame"]
        assert c1["frame_count"] == c2["frame_count"]
        assert c1["motion"]["mean"] == pytest.approx(c2["motion"]["mean"])
        assert c1["spatial_complexity"]["mean"] == pytest.approx(c2["spatial_complexity"]["mean"])

def test_data_leakage_protection(temp_test_dir):
    """Verifies that the profiler does not depend on post-SR quality metrics or Edge telemetry."""
    video_path = temp_test_dir / "leakage_test.mp4"
    generate_synthetic_video(video_path, fps=30, duration_seconds=1.0)
    
    out_dir = temp_test_dir / "out_leakage"
    profile_path, _ = run_profiler(str(video_path), str(out_dir), chunk_duration=2.0, temporal_window_s=0.0333)
    
    with open(profile_path, "r", encoding="utf-8") as f:
        profile_content = f.read()
        
    # Confirm no reference to post-SR metrics or edge parameters exists
    leakage_terms = ["PSNR", "SSIM", "LPIPS", "VMAF", "telemetry", "throughput", "RTT", "cache_hit"]
    for term in leakage_terms:
        assert term not in profile_content
        assert term.lower() not in profile_content
