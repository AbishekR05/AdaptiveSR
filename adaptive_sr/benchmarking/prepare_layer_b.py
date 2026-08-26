"""
adaptive_sr.benchmarking.prepare_layer_b
========================================
Step 5.6 — Layer B Reference Corpus / Video Dataset Preparation.

Generates 3 deterministic procedural natural-like video clips (1280x720, 30 FPS, 4s)
representing low-motion, moderate-motion, and high-motion content cases.
"""

import os
import sys
import argparse
import json
import hashlib
from datetime import datetime, timezone
import cv2
import numpy as np

# Ensure root of repository is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.modules.video_loader import VideoLoader
from adaptive_sr.profiling.profile_video import run_profiler

SCHEMA_VERSION = "1.0.0"
GENERATOR_VERSION = "1.0.0"
DEFAULT_DATASET_ID = "benchmark_layer_b_v1"

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

def generate_natural_frame(
    frame_idx: int,
    total_frames: int,
    fps: float,
    width: int,
    height: int,
    category: str
) -> np.ndarray:
    """Generates a deterministic natural-like frame using multi-frequency plasma noise."""
    t = frame_idx / fps  # Time in seconds

    # Generate coordinate grid
    x = np.linspace(0, 10 * np.pi, width)
    y = np.linspace(0, 10 * np.pi, height)
    X, Y = np.meshgrid(x, y)

    # Set motion displacement parameters based on category
    if category == "lowmotion":
        # Slow translation
        dx = 0.05 * t
        dy = 0.02 * t
        scale = 1.0
        angle = 0.0
    elif category == "moderatemotion":
        # Moderate translation + slow rotation + scaling oscillation
        dx = 0.25 * t
        dy = 0.15 * t
        scale = 1.0 + 0.01 * np.sin(t)
        angle = 0.02 * t
    else:  # "highmotion"
        # High translation + fast rotation + scaling oscillation + camera jitter
        dx = 0.8 * t + 0.05 * np.sin(10 * t)
        dy = 0.6 * t + 0.05 * np.cos(10 * t)
        scale = 1.0 + 0.1 * np.sin(3 * t)
        angle = 0.15 * t

    # Apply rotation and scaling to coordinates
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    X_rot = scale * (cos_a * (X - 5 * np.pi) - sin_a * (Y - 5 * np.pi)) + 5 * np.pi + dx
    Y_rot = scale * (sin_a * (X - 5 * np.pi) + cos_a * (Y - 5 * np.pi)) + 5 * np.pi + dy

    # Multi-frequency plasma fractal noise
    v1 = np.sin(X_rot) + np.sin(Y_rot)
    v2 = np.sin(X_rot + Y_rot) + np.sin(np.sqrt(X_rot**2 + Y_rot**2 + 1.0))
    v3 = 0.5 * np.sin(3 * X_rot) + 0.5 * np.cos(3 * Y_rot)
    v4 = 0.25 * np.sin(10 * X_rot - 5 * Y_rot) + 0.25 * np.cos(5 * X_rot + 10 * Y_rot)

    total = v1 + v2 + v3 + v4

    # Normalize to [0, 255]
    total_min, total_max = total.min(), total.max()
    if total_max - total_min > 1e-5:
        normalized = (total - total_min) / (total_max - total_min) * 255.0
    else:
        normalized = np.zeros_like(total)

    # Frame color mapping to represent natural tones
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    if category == "lowmotion":
        # Moss green/earthy forest tones
        frame[:, :, 0] = (normalized * 0.2 + 20).astype(np.uint8)  # B
        frame[:, :, 1] = (normalized * 0.7 + 40).astype(np.uint8)  # G
        frame[:, :, 2] = (normalized * 0.4 + 30).astype(np.uint8)  # R
    elif category == "moderatemotion":
        # Sand/clay/desert soil tones
        frame[:, :, 0] = (normalized * 0.3 + 15).astype(np.uint8)  # B
        frame[:, :, 1] = (normalized * 0.45 + 25).astype(np.uint8) # G
        frame[:, :, 2] = (normalized * 0.7 + 35).astype(np.uint8)  # R
    else:
        # High motion: dynamic cyan/blue water/rapid tones
        frame[:, :, 0] = (normalized * 0.8 + 40).astype(np.uint8)  # B
        frame[:, :, 1] = (normalized * 0.6 + 30).astype(np.uint8)  # G
        frame[:, :, 2] = (normalized * 0.3 + 10).astype(np.uint8)  # R

    # Add simulated fine-grain sensor noise
    np.random.seed(frame_idx + 1000)
    noise = np.random.normal(0, 3.0, (height, width, 3)).astype(np.float32)
    noisy_frame = cv2.add(frame.astype(np.float32), noise)
    noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)

    return noisy_frame

def generate_natural_video(
    video_id: str,
    fps: int,
    output_path: str,
    width: int,
    height: int,
    duration: float,
    category: str
):
    """Writes a deterministic naturalistic video file using cv2.VideoWriter."""
    total_frames = int(round(fps * duration))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise IOError(f"Could not open VideoWriter for path: {output_path}")

    for i in range(total_frames):
        frame = generate_natural_frame(i, total_frames, fps, width, height, category)
        out.write(frame)
    out.release()

def prepare_layer_b(output_dir: str, duration: float = 4.0, overwrite: bool = False):
    """Generates the Layer B dataset, profiles each video, and writes the Layer B manifest."""
    videos_dir = os.path.join(output_dir, "videos", "layer_b")
    manifests_dir = os.path.join(output_dir, "manifests")

    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(manifests_dir, exist_ok=True)

    manifest_path = os.path.join(manifests_dir, "layer_b_manifest.json")

    if os.path.exists(manifest_path) and not overwrite:
        print(f"Layer B manifest already exists at {manifest_path}. Use overwrite=True to regenerate.")
        return

    videos_metadata = []
    total_generated = 0
    total_frames = 0
    total_duration = 0.0

    print(f"Preparing Layer B dataset with duration={duration}s, resolution=1280x720")

    categories = ["lowmotion", "moderatemotion", "highmotion"]
    for idx, cat in enumerate(categories):
        clip_num = idx + 1
        video_id = f"clip_{clip_num:03d}_{cat}_30fps"
        video_filename = f"{video_id}.mp4"
        video_path = os.path.join(videos_dir, video_filename)

        if not os.path.exists(video_path) or overwrite:
            print(f"Generating Layer B video: {video_filename}...")
            generate_natural_video(
                video_id=video_id,
                fps=30,
                output_path=video_path,
                width=1280,
                height=720,
                duration=duration,
                category=cat
            )
        else:
            print(f"Reusing existing Layer B video: {video_filename}")

        # Segment and profile
        print(f"Segmenting and profiling Layer B video: {video_id}...")
        profile_json, chunk_manifest_json = run_profiler(
            input_video=video_path,
            output_dir=output_dir,
            chunk_duration=2.0,
            temporal_window_s=0.0333
        )

        with open(profile_json, "r", encoding="utf-8") as f:
            profile_data = json.load(f)
        with open(chunk_manifest_json, "r", encoding="utf-8") as f:
            chunk_manifest_data = json.load(f)

        combined_chunks = []
        profile_chunks = {c["chunk_id"]: c for c in profile_data["chunks"]}
        manifest_chunks = {c["chunk_id"]: c for c in chunk_manifest_data["chunks"]}

        for chunk_id in sorted(profile_chunks.keys()):
            p_c = profile_chunks[chunk_id]
            m_c = manifest_chunks[chunk_id]

            combined_chunks.append({
                "chunk_id": chunk_id,
                "file_path": m_c["file_path"],
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

        # Calculate relative path from output_dir to matching Step 5.1/5.5 conventions
        rel_video_path = os.path.relpath(video_path, output_dir)
        rel_profile_path = os.path.relpath(profile_json, output_dir)

        videos_metadata.append({
            "benchmark_video_id": video_id,
            "filename": video_filename,
            "file_path": rel_video_path,
            "file_hash": video_hash,
            "content_case": cat,
            "source_fps": 30.0,
            "width": 1280,
            "height": 720,
            "duration_seconds": source_meta["duration_seconds"],
            "frame_count": source_meta["frame_count"],
            "codec": source_meta["codec"],
            "pixel_format": "yuv420p",
            "source_bitrate": None,
            "audio_presence": False,
            "profile_path": rel_profile_path,
            "gt_crop_applied": False,
            "gt_h_before": 720,
            "gt_w_before": 1280,
            "downsampling": {
                "interpolation": "cubic",
                "scale": 2
            },
            "chunks": combined_chunks
        })

        total_generated += 1
        total_frames += source_meta["frame_count"]
        total_duration += source_meta["duration_seconds"]

    layer_b_manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "dataset_id": DEFAULT_DATASET_ID,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
        "config": {
            "target_chunk_duration_seconds": 2.0,
            "seed": 42,
            "duration_seconds": duration,
            "width": 1280,
            "height": 720
        },
        "videos": videos_metadata
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(layer_b_manifest, f, indent=4)

    print("\n--- Layer B Dataset Generation Summary ---")
    print(f"Dataset ID: {DEFAULT_DATASET_ID}")
    print(f"Number of videos: {total_generated}")
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Total frames: {total_frames}")
    print(f"Manifest path: {manifest_path}")
    print("------------------------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Layer B Dataset Generator")
    parser.add_argument("--output", default="data/benchmarks/sr", help="Output directory path for dataset assets")
    parser.add_argument("--duration", type=float, default=4.0, help="Duration of generated videos in seconds")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset and manifest files")
    args = parser.parse_args()

    abs_output = os.path.abspath(args.output)
    prepare_layer_b(abs_output, duration=args.duration, overwrite=args.overwrite)

if __name__ == "__main__":
    main()
