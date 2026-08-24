"""
adaptive_sr.benchmarking.prepare_dataset
========================================
Step 5.1 — Benchmark Dataset / Test-Video Preparation.

PURPOSE
-------
Prepares a deterministic, reproducible benchmark dataset (manifest, videos,
chunks, and metadata) to support Super-Resolution (SR) model evaluation.
This script does NOT perform SR inference, model loading, or adapter mapping.

CLI USAGE
---------
Generate benchmark dataset:
  python -m adaptive_sr.benchmarking.prepare_dataset --output <directory>

Validate existing benchmark dataset:
  python -m adaptive_sr.benchmarking.prepare_dataset --validate <manifest_path>
"""

import os
import sys
import argparse
import json
import time
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np

# Ensure root of repository is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.modules.video_loader import VideoLoader
from adaptive_sr.profiling.profile_video import run_profiler

# Constants for Step 5.1 design
SCHEMA_VERSION = "1.0.0"
GENERATOR_VERSION = "1.0.0"
DEFAULT_DATASET_ID = "benchmark_corpus_v1"

# Motion speeds in pixels per second (under 30fps baseline, distance covered is constant)
MOTION_SPEEDS = {
    "lowmotion": 20.0,
    "moderatemotion": 100.0,
    "highmotion": 300.0
}

FPS_VALUES = [30, 60, 120]


def get_file_sha256(file_path: str) -> str:
    """Calculates SHA-256 hash of a file for integrity tracking."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def generate_frame(
    frame_idx: int,
    total_frames: int,
    fps: float,
    width: int,
    height: int,
    motion_speed: float
) -> np.ndarray:
    """Generates a deterministic frame with spatial grid and moving crosshairs circle."""
    # Dark blue-gray background (Futuristic theme)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = [24, 18, 16]

    # Draw high-contrast grid lines for spatial frequency
    grid_size = 20
    for x in range(0, width, grid_size):
        cv2.line(frame, (x, 0), (x, height), (40, 35, 30), 1)
    for y in range(0, height, grid_size):
        cv2.line(frame, (0, y), (width, y), (40, 35, 30), 1)

    # Text pattern (high frequency text detail)
    cv2.putText(frame, "AdaptiveSR Benchmark Input", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
    cv2.putText(frame, f"FPS: {fps} | Frame: {frame_idx}/{total_frames}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

    # Compute moving pattern coordinates using speed
    # distance moved = speed (px/sec) * time (sec) = speed * (frame_idx / fps)
    distance = motion_speed * (frame_idx / fps)

    # Diagonal bouncing path
    bounce_x = int(distance) % (2 * (width - 40))
    x = bounce_x if bounce_x < (width - 40) else (2 * (width - 40) - bounce_x)
    x += 20

    bounce_y = int(distance * 0.7) % (2 * (height - 40))
    y = bounce_y if bounce_y < (height - 40) else (2 * (height - 40) - bounce_y)
    y += 20

    # Draw vibrant patterns (cyan/magenta details for high spatial frequency content)
    cv2.circle(frame, (x, y), 15, (255, 0, 255), 2)  # Magenta outer
    cv2.circle(frame, (x, y), 8, (255, 255, 0), 1)   # Cyan inner
    cv2.line(frame, (x - 20, y), (x + 20, y), (0, 255, 255), 1)  # Yellow horizontal
    cv2.line(frame, (x, y - 20), (x, y + 20), (0, 255, 255), 1)  # Yellow vertical

    return frame


def generate_synthetic_video(
    video_id: str,
    fps: int,
    motion_speed: float,
    output_path: str,
    width: int,
    height: int,
    duration: float
):
    """Writes a deterministic video file using cv2.VideoWriter."""
    total_frames = int(round(fps * duration))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise IOError(f"Could not open VideoWriter for path: {output_path}")

    for i in range(total_frames):
        frame = generate_frame(i, total_frames, fps, width, height, motion_speed)
        out.write(frame)
    out.release()


def validate_dataset(manifest_path: str) -> bool:
    """Validates an existing dataset manifest and its associated files."""
    print(f"Validating dataset from manifest: {manifest_path}")
    if not os.path.exists(manifest_path):
        print(f"ERROR: Manifest file does not exist: {manifest_path}")
        return False

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load manifest JSON: {e}")
        return False

    # Check manifest-level fields
    if manifest.get("schema_version") != SCHEMA_VERSION:
        print(f"ERROR: Invalid schema version: {manifest.get('schema_version')}")
        return False
    if manifest.get("dataset_schema_version") != SCHEMA_VERSION:
        print(f"ERROR: Invalid dataset schema version: {manifest.get('dataset_schema_version')}")
        return False
    if manifest.get("generator_version") != GENERATOR_VERSION:
        print(f"ERROR: Invalid generator version: {manifest.get('generator_version')}")
        return False

    base_dir = os.path.dirname(os.path.dirname(manifest_path))

    video_ids = set()
    for video in manifest.get("videos", []):
        video_id = video.get("benchmark_video_id")
        if not video_id:
            print("ERROR: Missing benchmark_video_id entry.")
            return False
        if video_id in video_ids:
            print(f"ERROR: Duplicate benchmark_video_id detected: {video_id}")
            return False
        video_ids.add(video_id)

        rel_path = video.get("file_path")
        abs_video_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(abs_video_path):
            print(f"ERROR: Video file not found: {abs_video_path}")
            return False

        # Verify hash
        calculated_hash = get_file_sha256(abs_video_path)
        if calculated_hash != video.get("file_hash"):
            print(f"ERROR: Hash mismatch for video {video_id}. Expected {video.get('file_hash')}, got {calculated_hash}")
            return False

        # Load video metadata via VideoLoader
        try:
            loader = VideoLoader(abs_video_path)
            meta = loader.metadata
        except Exception as e:
            print(f"ERROR: Failed to read video metadata for {video_id}: {e}")
            return False

        # Compare with manifest values
        if meta["width"] != video.get("width") or meta["height"] != video.get("height"):
            print(f"ERROR: Resolution mismatch for {video_id}. Manifest: {video.get('width')}x{video.get('height')}, actual: {meta['width']}x{meta['height']}")
            return False

        if abs(meta["fps"] - video.get("source_fps")) > 0.01:
            print(f"ERROR: FPS mismatch for {video_id}. Manifest: {video.get('source_fps')}, actual: {meta['fps']}")
            return False

        if meta["frame_count"] != video.get("frame_count"):
            print(f"ERROR: Frame count mismatch for {video_id}. Manifest: {video.get('frame_count')}, actual: {meta['frame_count']}")
            return False

        # Verify chunk boundaries
        chunks = video.get("chunks", [])
        if not chunks:
            print(f"ERROR: Video {video_id} has no chunk references.")
            return False

        chunk_ids = set()
        last_end_frame = -1
        last_end_time = 0.0

        for idx, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                print(f"ERROR: Video {video_id} has a chunk with missing chunk_id.")
                return False
            if chunk_id in chunk_ids:
                print(f"ERROR: Duplicate chunk_id {chunk_id} in video {video_id}")
                return False
            chunk_ids.add(chunk_id)

            chunk_rel = chunk.get("file_path")
            abs_chunk_path = os.path.join(base_dir, chunk_rel)
            if not os.path.exists(abs_chunk_path):
                print(f"ERROR: Chunk file not found: {abs_chunk_path}")
                return False

            # Verify chunk file hash
            calculated_chunk_hash = get_file_sha256(abs_chunk_path)
            if calculated_chunk_hash != chunk.get("file_hash"):
                print(f"ERROR: Hash mismatch for chunk {chunk_id} of video {video_id}")
                return False

            # Logical timeline continuity checks
            start_frame = chunk.get("start_frame")
            end_frame = chunk.get("end_frame")
            frame_count = chunk.get("frame_count")

            if start_frame != last_end_frame + 1:
                print(f"ERROR: Gap or overlap in chunk {chunk_id} start_frame. Expected {last_end_frame + 1}, got {start_frame}")
                return False

            if frame_count != (end_frame - start_frame + 1):
                print(f"ERROR: Mismatched frame count inside chunk metadata {chunk_id}")
                return False

            last_end_frame = end_frame
            last_end_time = chunk.get("end_time_seconds")

        # First chunk starts at 0
        if chunks[0].get("start_frame") != 0:
            print(f"ERROR: Video {video_id} first chunk does not start at frame 0")
            return False

        # Last chunk ends at video final frame
        if last_end_frame != video.get("frame_count") - 1:
            print(f"ERROR: Video {video_id} last chunk ends at frame {last_end_frame}, expected {video.get('frame_count') - 1}")
            return False

    print("SUCCESS: Dataset validation complete. All checks passed.")
    return True


def prepare_dataset(
    output_dir: str,
    duration: float,
    width: int,
    height: int,
    seed: int,
    overwrite: bool
):
    """Generates the synthetic video corpus, profiles each video, and creates the manifest."""
    np.random.seed(seed)
    
    videos_dir = os.path.join(output_dir, "videos")
    manifests_dir = os.path.join(output_dir, "manifests")
    
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(manifests_dir, exist_ok=True)
    
    manifest_path = os.path.join(manifests_dir, "benchmark_manifest.json")
    
    # Check if manifest already exists to prevent redundant generation
    if os.path.exists(manifest_path) and not overwrite:
        print(f"Benchmark manifest already exists at {manifest_path}. Use --overwrite to regenerate.")
        return

    videos_metadata = []
    total_generated_videos = 0
    total_frames = 0
    total_duration = 0.0

    print(f"Preparing dataset with duration={duration}s, resolution={width}x{height}, seed={seed}")

    for motion_label, speed_px in MOTION_SPEEDS.items():
        for fps in FPS_VALUES:
            video_id = f"synthetic_{motion_label}_{fps}fps"
            video_filename = f"{video_id}.mp4"
            video_path = os.path.join(videos_dir, video_filename)

            # Generate video if overwrite is enabled or file doesn't exist
            if not os.path.exists(video_path) or overwrite:
                print(f"Generating synthetic video: {video_filename} (FPS={fps}, Speed={speed_px} px/s)...")
                generate_synthetic_video(
                    video_id=video_id,
                    fps=fps,
                    motion_speed=speed_px,
                    output_path=video_path,
                    width=width,
                    height=height,
                    duration=duration
                )
            else:
                print(f"Reusing existing synthetic video: {video_filename}")

            # Run profiler to segment the video and generate chunk-level metadata
            print(f"Segmenting and profiling video: {video_id}...")
            profile_json, chunk_manifest_json = run_profiler(
                input_video=video_path,
                output_dir=output_dir,
                chunk_duration=2.0,
                temporal_window_s=0.0333
            )

            # Read Step 1 profiling JSON and chunk manifest JSON to build Step 5.1 timeline
            with open(profile_json, "r", encoding="utf-8") as f:
                profile_data = json.load(f)
            with open(chunk_manifest_json, "r", encoding="utf-8") as f:
                chunk_manifest_data = json.load(f)

            # Extract chunks list
            combined_chunks = []
            profile_chunks = {c["chunk_id"]: c for c in profile_data["chunks"]}
            manifest_chunks = {c["chunk_id"]: c for c in chunk_manifest_data["chunks"]}

            for chunk_id in sorted(profile_chunks.keys()):
                p_c = profile_chunks[chunk_id]
                m_c = manifest_chunks[chunk_id]
                
                combined_chunks.append({
                    "chunk_id": chunk_id,
                    "file_path": m_c["file_path"],  # Relative to output_dir
                    "file_hash": m_c["file_hash"],
                    "start_time_seconds": p_c["start_time_seconds"],
                    "end_time_seconds": p_c["end_time_seconds"],
                    "duration_seconds": p_c["duration_seconds"],
                    "start_frame": p_c["start_frame"],
                    "end_frame": p_c["end_frame"],
                    "frame_count": p_c["frame_count"]
                })

            video_hash = get_file_sha256(video_path)
            source_meta = profile_data["source"]

            videos_metadata.append({
                "benchmark_video_id": video_id,
                "filename": video_filename,
                "file_path": os.path.relpath(video_path, output_dir),
                "file_hash": video_hash,
                "content_case": f"{motion_label}",
                "source_fps": float(fps),
                "width": width,
                "height": height,
                "duration_seconds": source_meta["duration_seconds"],
                "frame_count": source_meta["frame_count"],
                "codec": source_meta["codec"],
                "pixel_format": "yuv420p",  # OpenCV default container output format
                "source_bitrate": None,
                "audio_presence": False,
                "profile_path": os.path.relpath(profile_json, output_dir),
                "chunks": combined_chunks
            })

            total_generated_videos += 1
            total_frames += source_meta["frame_count"]
            total_duration += source_meta["duration_seconds"]

    # Write the global benchmark manifest JSON
    benchmark_manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "dataset_id": DEFAULT_DATASET_ID,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
        "config": {
            "target_chunk_duration_seconds": 2.0,
            "seed": seed,
            "duration_seconds": duration,
            "width": width,
            "height": height
        },
        "videos": videos_metadata
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_manifest, f, indent=4)

    # Print summary output exactly as specified by section 20
    print("\n--- Benchmark Dataset Generation Summary ---")
    print(f"Dataset ID: {DEFAULT_DATASET_ID}")
    print(f"Number of videos: {total_generated_videos}")
    print("FPS coverage: 30 FPS, 60 FPS, 120 FPS")
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Total frames: {total_frames}")
    print(f"Manifest path: {manifest_path}")
    print("--------------------------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Step 5.1 Benchmark Dataset Preparation CLI")
    parser.add_argument("--output", default="data/benchmarks/sr", help="Output directory path for dataset assets")
    parser.add_argument("--duration", type=float, default=4.0, help="Duration of generated synthetic videos in seconds")
    parser.add_argument("--width", type=int, default=640, help="Width of generated synthetic videos")
    parser.add_argument("--height", type=int, default=360, help="Height of generated synthetic videos")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic generation")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset and manifest files")
    parser.add_argument("--validate", help="Path to an existing benchmark manifest to validate")

    args = parser.parse_args()

    if args.validate:
        success = validate_dataset(args.validate)
        sys.exit(0 if success else 1)
    else:
        # Convert output path to absolute
        abs_output_dir = os.path.abspath(args.output)
        prepare_dataset(
            output_dir=abs_output_dir,
            duration=args.duration,
            width=args.width,
            height=args.height,
            seed=args.seed,
            overwrite=args.overwrite
        )


if __name__ == "__main__":
    main()
