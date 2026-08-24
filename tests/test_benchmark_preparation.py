"""
tests/test_benchmark_preparation.py
====================================
Step 5.1 — Benchmark Dataset / Test-Video Preparation unit and integration tests.
"""

import os
import sys
import json
import shutil
import tempfile
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.modules.video_loader import VideoLoader
from adaptive_sr.benchmarking.prepare_dataset import (
    prepare_dataset,
    validate_dataset,
    get_file_sha256,
    generate_synthetic_video,
    MOTION_SPEEDS,
    FPS_VALUES,
    SCHEMA_VERSION
)


@pytest.fixture(scope="module")
def temp_dataset_dir():
    """Fixture to generate a small test dataset in a temporary directory."""
    temp_dir = tempfile.mkdtemp(suffix="_benchmark_test_data")
    # Generate 4-second videos at 320x180 to keep profiling and tests extremely fast
    prepare_dataset(
        output_dir=temp_dir,
        duration=4.0,
        width=320,
        height=180,
        seed=42,
        overwrite=True
    )
    yield temp_dir
    # Cleanup directory
    shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 1. Dataset preparation succeeds
# ---------------------------------------------------------------------------

def test_dataset_preparation_succeeds(temp_dataset_dir):
    """Verifies that the dataset files are created and the global manifest exists."""
    manifest_path = os.path.join(temp_dataset_dir, "manifests", "benchmark_manifest.json")
    assert os.path.exists(manifest_path), "Global benchmark manifest was not created."

    # Validate that videos folder contains the generated mp4 files
    videos_dir = os.path.join(temp_dataset_dir, "videos")
    assert os.path.exists(videos_dir), "Videos directory does not exist."
    video_files = os.listdir(videos_dir)
    assert len(video_files) == 9, f"Expected 9 video files, found {len(video_files)}"


# ---------------------------------------------------------------------------
# 2. 30 FPS synthetic video is generated/validated
# ---------------------------------------------------------------------------

def test_30fps_video_generated_properly(temp_dataset_dir):
    """Verifies that 30 FPS videos are correctly generated and contain expected frames."""
    video_path = os.path.join(temp_dataset_dir, "videos", "synthetic_lowmotion_30fps.mp4")
    assert os.path.exists(video_path), "30 FPS video file is missing."

    loader = VideoLoader(video_path)
    assert abs(loader.metadata["fps"] - 30.0) < 0.01, f"Expected 30 FPS, got {loader.metadata['fps']}"
    assert loader.metadata["frame_count"] == 120, f"Expected 120 frames, got {loader.metadata['frame_count']}"


# ---------------------------------------------------------------------------
# 3. 60 FPS synthetic video is generated/validated
# ---------------------------------------------------------------------------

def test_60fps_video_generated_properly(temp_dataset_dir):
    """Verifies that 60 FPS videos are correctly generated and contain expected frames."""
    video_path = os.path.join(temp_dataset_dir, "videos", "synthetic_lowmotion_60fps.mp4")
    assert os.path.exists(video_path), "60 FPS video file is missing."

    loader = VideoLoader(video_path)
    assert abs(loader.metadata["fps"] - 60.0) < 0.01, f"Expected 60 FPS, got {loader.metadata['fps']}"
    assert loader.metadata["frame_count"] == 240, f"Expected 240 frames, got {loader.metadata['frame_count']}"


# ---------------------------------------------------------------------------
# 4. 120 FPS synthetic video is generated/validated
# ---------------------------------------------------------------------------

def test_120fps_video_generated_properly(temp_dataset_dir):
    """Verifies that 120 FPS videos are correctly generated and contain expected frames."""
    video_path = os.path.join(temp_dataset_dir, "videos", "synthetic_lowmotion_120fps.mp4")
    assert os.path.exists(video_path), "120 FPS video file is missing."

    loader = VideoLoader(video_path)
    assert abs(loader.metadata["fps"] - 120.0) < 0.01, f"Expected 120 FPS, got {loader.metadata['fps']}"
    assert loader.metadata["frame_count"] == 480, f"Expected 480 frames, got {loader.metadata['frame_count']}"


# ---------------------------------------------------------------------------
# 5. FPS is detected from actual video metadata
# ---------------------------------------------------------------------------

def test_fps_detected_from_video_metadata(temp_dataset_dir):
    """Verifies that VideoLoader extracts real FPS rather than hardcoding it."""
    for fps in FPS_VALUES:
        path = os.path.join(temp_dataset_dir, "videos", f"synthetic_lowmotion_{fps}fps.mp4")
        loader = VideoLoader(path)
        assert abs(loader.metadata["fps"] - fps) < 0.01


# ---------------------------------------------------------------------------
# 6. Frame counts are plausible for the declared duration/FPS
# ---------------------------------------------------------------------------

def test_frame_counts_plausible_for_duration_and_fps(temp_dataset_dir):
    """Checks that frame_count == duration * fps for all generated videos."""
    for fps in FPS_VALUES:
        for case in MOTION_SPEEDS.keys():
            path = os.path.join(temp_dataset_dir, "videos", f"synthetic_{case}_{fps}fps.mp4")
            loader = VideoLoader(path)
            expected_count = int(round(fps * 4.0))
            assert loader.metadata["frame_count"] == expected_count, (
                f"Mismatched frames for {fps}fps {case}. Expected {expected_count}, got {loader.metadata['frame_count']}"
            )


# ---------------------------------------------------------------------------
# 7. Benchmark video IDs are unique
# ---------------------------------------------------------------------------

def test_benchmark_video_ids_are_unique(temp_dataset_dir):
    """Verifies that the generated manifest lists unique video IDs."""
    manifest_path = os.path.join(temp_dataset_dir, "manifests", "benchmark_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    video_ids = [video["benchmark_video_id"] for video in manifest["videos"]]
    assert len(video_ids) == len(set(video_ids)), "Duplicate benchmark video IDs found in manifest."
    assert len(video_ids) == 9, f"Expected 9 video entries, got {len(video_ids)}"


# ---------------------------------------------------------------------------
# 8. Dataset manifest is valid
# ---------------------------------------------------------------------------

def test_dataset_manifest_is_valid(temp_dataset_dir):
    """Runs the validation function against the generated manifest."""
    manifest_path = os.path.join(temp_dataset_dir, "manifests", "benchmark_manifest.json")
    success = validate_dataset(manifest_path)
    assert success, "Manifest failed automated validation checks."


# ---------------------------------------------------------------------------
# 9. SHA-256 hashes are generated
# ---------------------------------------------------------------------------

def test_sha256_hashes_are_generated_for_all_files(temp_dataset_dir):
    """Validates that hashes are present in the manifest and match the actual files."""
    manifest_path = os.path.join(temp_dataset_dir, "manifests", "benchmark_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for video in manifest["videos"]:
        # Video file hash check
        video_rel = video["file_path"]
        abs_video_path = os.path.join(temp_dataset_dir, video_rel)
        assert video["file_hash"] == get_file_sha256(abs_video_path)

        # Chunk hashes check
        for chunk in video["chunks"]:
            chunk_rel = chunk["file_path"]
            abs_chunk_path = os.path.join(temp_dataset_dir, chunk_rel)
            assert chunk["file_hash"] == get_file_sha256(abs_chunk_path)


# ---------------------------------------------------------------------------
# 10. Hash validation detects a modified/corrupted file
# ---------------------------------------------------------------------------

def test_hash_validation_detects_corrupted_file(temp_dataset_dir):
    """Checks that tampering with a video file causes validate_dataset to fail."""
    # Copy manifest to avoid corrupting the module-wide fixture
    test_dir = tempfile.mkdtemp(suffix="_tamper_test")
    try:
        shutil.copytree(temp_dataset_dir, test_dir, dirs_exist_ok=True)
        manifest_path = os.path.join(test_dir, "manifests", "benchmark_manifest.json")
        
        # Modify/corrupt one video file slightly
        target_video = os.path.join(test_dir, "videos", "synthetic_lowmotion_30fps.mp4")
        with open(target_video, "ab") as f:
            f.write(b"CORRUPTING_HASH_APPEND")

        # Validation must fail
        success = validate_dataset(manifest_path)
        assert not success, "Validator passed even though a video file was tampered with."
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 11. Re-running preparation with same config produces equivalent metadata
# ---------------------------------------------------------------------------

def test_rerunning_preparation_produces_equivalent_manifests(temp_dataset_dir):
    """Ensures that dataset manifests match exactly when run twice with identical config."""
    temp_dir_2 = tempfile.mkdtemp(suffix="_equivalent_test")
    try:
        prepare_dataset(
            output_dir=temp_dir_2,
            duration=4.0,
            width=320,
            height=180,
            seed=42,
            overwrite=True
        )
        
        manifest_1 = os.path.join(temp_dataset_dir, "manifests", "benchmark_manifest.json")
        manifest_2 = os.path.join(temp_dir_2, "manifests", "benchmark_manifest.json")
        
        with open(manifest_1, "r", encoding="utf-8") as f:
            m1 = json.load(f)
        with open(manifest_2, "r", encoding="utf-8") as f:
            m2 = json.load(f)
            
        # Ignore creation timestamps in comparison
        m1.pop("created_at", None)
        m2.pop("created_at", None)
        
        assert m1 == m2, "Re-running dataset generation produced a different manifest schema or values."
    finally:
        shutil.rmtree(temp_dir_2, ignore_errors=True)


# ---------------------------------------------------------------------------
# 12. Chunk references consistent with Step 1 profiling profile files
# ---------------------------------------------------------------------------

def test_chunk_references_consistent_with_step1_artifacts(temp_dataset_dir):
    """Verifies that chunks listed in the benchmark manifest align with the profile JSONs."""
    manifest_path = os.path.join(temp_dataset_dir, "manifests", "benchmark_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for video in manifest["videos"]:
        profile_rel = video["profile_path"]
        abs_profile_path = os.path.join(temp_dataset_dir, profile_rel)
        assert os.path.exists(abs_profile_path), f"Step 1 profile JSON is missing: {abs_profile_path}"

        with open(abs_profile_path, "r", encoding="utf-8") as f:
            profile = json.load(f)

        profile_chunks = profile["chunks"]
        manifest_chunks = video["chunks"]

        assert len(profile_chunks) == len(manifest_chunks), "Number of chunks in profile and benchmark manifest mismatch."

        for p_chunk, m_chunk in zip(profile_chunks, manifest_chunks):
            assert p_chunk["chunk_id"] == m_chunk["chunk_id"]
            assert p_chunk["start_frame"] == m_chunk["start_frame"]
            assert p_chunk["end_frame"] == m_chunk["end_frame"]
            assert p_chunk["frame_count"] == m_chunk["frame_count"]


# ---------------------------------------------------------------------------
# 13. Chunk IDs are not duplicated
# ---------------------------------------------------------------------------

def test_chunk_ids_are_not_duplicated(temp_dataset_dir):
    """Checks that no duplicate chunk_id exists inside a benchmark video."""
    manifest_path = os.path.join(temp_dataset_dir, "manifests", "benchmark_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for video in manifest["videos"]:
        chunk_ids = [chunk["chunk_id"] for chunk in video["chunks"]]
        assert len(chunk_ids) == len(set(chunk_ids)), f"Duplicate chunk IDs found in video {video['benchmark_video_id']}"


# ---------------------------------------------------------------------------
# 14. 30/60/120 FPS cases do not share incorrect frame counts
# ---------------------------------------------------------------------------

def test_fps_cases_do_not_share_incorrect_frame_counts(temp_dataset_dir):
    """Verifies that frame counts scale correctly with FPS (i.e. not cloned from a single preset)."""
    manifest_path = os.path.join(temp_dataset_dir, "manifests", "benchmark_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    fps_frame_counts = {}
    for video in manifest["videos"]:
        fps = video["source_fps"]
        count = video["frame_count"]
        fps_frame_counts[fps] = count

    assert fps_frame_counts[30.0] == 120
    assert fps_frame_counts[60.0] == 240
    assert fps_frame_counts[120.0] == 480


# ---------------------------------------------------------------------------
# 15. Missing/corrupt video files are detected
# ---------------------------------------------------------------------------

def test_validation_detects_missing_video_files(temp_dataset_dir):
    """Deletes one video file and checks that validation flags it immediately."""
    test_dir = tempfile.mkdtemp(suffix="_missing_video_test")
    try:
        shutil.copytree(temp_dataset_dir, test_dir, dirs_exist_ok=True)
        manifest_path = os.path.join(test_dir, "manifests", "benchmark_manifest.json")

        target_video = os.path.join(test_dir, "videos", "synthetic_lowmotion_30fps.mp4")
        os.remove(target_video)

        # Validation must fail
        success = validate_dataset(manifest_path)
        assert not success, "Validator passed even though a video file was missing."
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 16. The validator rejects a manifest whose file hash does not match
# ---------------------------------------------------------------------------

def test_validator_rejects_altered_manifest_chunk_hash(temp_dataset_dir):
    """Rejects validation when a chunk's file hash is manually tampered inside the manifest."""
    test_dir = tempfile.mkdtemp(suffix="_tamper_manifest_test")
    try:
        shutil.copytree(temp_dataset_dir, test_dir, dirs_exist_ok=True)
        manifest_path = os.path.join(test_dir, "manifests", "benchmark_manifest.json")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Tamper with first video's first chunk hash
        manifest["videos"][0]["chunks"][0]["file_hash"] = "TAMPERED_HASH_12345"

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)

        # Validation must fail
        success = validate_dataset(manifest_path)
        assert not success, "Validator passed despite the manifest hash being manually altered."
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 17. Generator and schema version metadata validation
# ---------------------------------------------------------------------------

def test_manifest_version_metadata(temp_dataset_dir):
    """Verifies that generator_version and dataset_schema_version exist and are validated."""
    manifest_path = os.path.join(temp_dataset_dir, "manifests", "benchmark_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest.get("dataset_schema_version") == SCHEMA_VERSION
    assert manifest.get("generator_version") == "1.0.0"

    # Verify that changing them fails validation
    test_dir = tempfile.mkdtemp(suffix="_version_test")
    try:
        shutil.copytree(temp_dataset_dir, test_dir, dirs_exist_ok=True)
        m_path = os.path.join(test_dir, "manifests", "benchmark_manifest.json")

        # Tamper generator_version
        with open(m_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["generator_version"] = "99.0.0"
        with open(m_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        assert not validate_dataset(m_path), "Validator should fail on invalid generator_version"

        # Tamper dataset_schema_version
        data["generator_version"] = "1.0.0"
        data["dataset_schema_version"] = "99.0.0"
        with open(m_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        assert not validate_dataset(m_path), "Validator should fail on invalid dataset_schema_version"
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 18. Non-keyframe chunk boundary support
# ---------------------------------------------------------------------------

def test_non_keyframe_chunk_boundary_compatibility(temp_dataset_dir):
    """Verifies that validation passes even when chunk boundaries do not align with GOP/keyframes."""
    test_dir = tempfile.mkdtemp(suffix="_gop_test")
    try:
        shutil.copytree(temp_dataset_dir, test_dir, dirs_exist_ok=True)
        manifest_path = os.path.join(test_dir, "manifests", "benchmark_manifest.json")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Manually alter chunk boundaries of first video to represent non-keyframe chunk bounds
        # Original: chunk 0 is frames 0-59 (60 frames), chunk 1 is frames 60-119 (60 frames)
        # Modify to: chunk 0 ends at 37 (non-keyframe), chunk 1 starts at 38
        chunks = manifest["videos"][0]["chunks"]
        chunks[0]["end_frame"] = 37
        chunks[0]["frame_count"] = 38
        chunks[1]["start_frame"] = 38
        chunks[1]["frame_count"] = 82  # 119 - 38 + 1

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)

        # Validation must pass, proving the logic does not assume keyframe alignment
        success = validate_dataset(manifest_path)
        assert success, "Validator failed on non-keyframe chunk boundaries."
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)

