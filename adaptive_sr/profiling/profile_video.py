import os
import sys
import argparse
import json
import time
import logging
import hashlib
import subprocess
from collections import deque
import numpy as np

# Ensure root of repository is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Import legacy components
from src.modules.video_loader import VideoLoader
from src.modules.frame_extractor import FrameExtractor
from src.modules.scene_analyzer import analyze_frame
from src.modules.complexity_estimator import estimate_complexity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AdaptiveSR.profiler")

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

def segment_video(input_path: str, output_dir: str, chunk_duration: float, video_id: str) -> list:
    """
    Segment the input video into keyframe-aligned MP4 chunks using FFmpeg.
    Returns sorted list of generated chunk file paths.
    """
    chunks_dir = os.path.join(output_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    
    # Target output pattern
    output_pattern = os.path.join(chunks_dir, f"{video_id}_%04d.mp4")
    
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-c", "copy",
        "-f", "segment",
        "-segment_time", str(chunk_duration),
        "-reset_timestamps", "1",
        "-loglevel", "warning",
        output_pattern
    ]
    
    logger.info(f"Segmenting video with command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # Collect generated chunk files
    chunk_files = sorted([
        os.path.join(chunks_dir, f)
        for f in os.listdir(chunks_dir)
        if f.startswith(f"{video_id}_") and f.endswith(".mp4")
    ])
    
    return chunk_files

def run_profiler(input_video: str, output_dir: str, chunk_duration: float, temporal_window_s: float):
    """
    Runs the video profiler pipeline.
    Performs a continuous profiling pass on the source video to maintain temporal continuity
    across chunk boundaries, and assigns frame metrics using cumulative frame offset buckets.
    """
    logger.info(f"Initializing continuous profiler on: {input_video}")
    loader = VideoLoader(input_video)
    meta = loader.get_metadata()
    
    video_id = os.path.splitext(os.path.basename(input_video))[0]
    source_hash = get_file_sha256(input_video)
    
    # 1. Segment the video physically using copy-mode FFmpeg to define dynamic keyframe boundaries
    logger.info(f"Segmenting video into chunks of target {chunk_duration} seconds...")
    chunk_files = segment_video(input_video, output_dir, chunk_duration, video_id)
    
    # 2. Extract exact metadata from each physical chunk file to align boundaries deterministically
    chunk_boundaries = []
    running_time = 0.0
    running_frame = 0
    for idx, chunk_path in enumerate(chunk_files):
        chunk_loader = VideoLoader(chunk_path)
        chunk_meta = chunk_loader.get_metadata()
        chunk_dur = chunk_meta["duration"]
        chunk_frame_count = chunk_meta["frame_count"]
        
        chunk_id_str = f"{idx:04d}"
        chunk_boundaries.append({
            "chunk_id": chunk_id_str,
            "start_time_seconds": running_time,
            "end_time_seconds": running_time + chunk_dur,
            "duration_seconds": chunk_dur,
            "start_frame": running_frame,
            "end_frame": running_frame + chunk_frame_count - 1 if chunk_frame_count > 0 else running_frame,
            "frame_count": chunk_frame_count,
            "file_path": chunk_path,
            "file_hash": get_file_sha256(chunk_path)
        })
        
        running_time += chunk_dur
        running_frame += chunk_frame_count

    # 3. Continuous Profiling Pass
    # Pre-allocate buckets for each chunk
    chunk_buckets = {boundary["chunk_id"]: [] for boundary in chunk_boundaries}
    
    # Setup temporal comparison offset N = round(fps * temporal_window_s)
    fps = meta["fps"]
    temporal_offset = max(1, int(round(fps * temporal_window_s)))
    logger.info(f"Continuous profiling pass starting. Temporal comparison frame offset N={temporal_offset} (FPS={fps:.2f})")
    
    # Circular buffer to store recent frames across the entire source video
    frame_buffer = deque(maxlen=temporal_offset + 1)
    extractor = FrameExtractor(input_video)
    
    for idx, ts_ms, frame in extractor.extract():
        frame_buffer.append(frame)
        
        # If N frames have not been observed yet, comparison frame is None (motion=0.0)
        if len(frame_buffer) <= temporal_offset:
            prev_frame = None
        else:
            prev_frame = frame_buffer[0]
            
        metrics = analyze_frame(frame, prev_frame)
        complexity = estimate_complexity(metrics)
        
        # Bucket assignment by frame range
        assigned_chunk_id = None
        for boundary in chunk_boundaries:
            if boundary["start_frame"] <= idx <= boundary["end_frame"]:
                assigned_chunk_id = boundary["chunk_id"]
                break
                
        # Fallback to prevent edge-case frame dropping
        if assigned_chunk_id is None and chunk_boundaries:
            assigned_chunk_id = chunk_boundaries[-1]["chunk_id"]
            
        if assigned_chunk_id is not None:
            chunk_buckets[assigned_chunk_id].append({
                "motion": metrics["motion"],
                "texture": metrics["texture"],
                "edges": metrics["edges"],
                "blur": metrics["blur_clarity"],
                "complexity": complexity
            })

    # 4. Frame-Consistency Validation
    continuous_frame_count = idx + 1
    physical_frame_count = sum(boundary["frame_count"] for boundary in chunk_boundaries)
    
    if continuous_frame_count != physical_frame_count:
        raise RuntimeError(
            f"Frame count mismatch: continuous_frame_count ({continuous_frame_count}) "
            f"!= physical_frame_count ({physical_frame_count}) for source '{input_video}'."
        )
        
    if not chunk_boundaries:
        raise RuntimeError(f"No chunk boundaries generated for source '{input_video}'.")
        
    if chunk_boundaries[0]["start_frame"] != 0:
        raise RuntimeError(
            f"First chunk does not begin at frame 0 (starts at {chunk_boundaries[0]['start_frame']}) "
            f"for source '{input_video}'."
        )
        
    final_end = chunk_boundaries[-1]["end_frame"]
    if final_end != continuous_frame_count - 1:
        raise RuntimeError(
            f"Final chunk does not end at last frame ({continuous_frame_count - 1}) (ends at {final_end}) "
            f"for source '{input_video}'."
        )
        
    assigned_frames = set()
    for i, boundary in enumerate(chunk_boundaries):
        c_id = boundary["chunk_id"]
        c_start = boundary["start_frame"]
        c_end = boundary["end_frame"]
        c_count = boundary["frame_count"]
        
        if c_count != (c_end - c_start + 1):
            raise RuntimeError(
                f"Chunk {c_id} frame range {c_start}-{c_end} count mismatch: "
                f"frame_count ({c_count}) != expected ({c_end - c_start + 1}) for source '{input_video}'."
            )
            
        for f in range(c_start, c_end + 1):
            if f in assigned_frames:
                raise RuntimeError(
                    f"Frame overlap detected: Frame {f} is assigned to multiple chunks "
                    f"for source '{input_video}'."
                )
            assigned_frames.add(f)
            
        if i < len(chunk_boundaries) - 1:
            next_boundary = chunk_boundaries[i + 1]
            next_start = next_boundary["start_frame"]
            if next_start != c_end + 1:
                raise RuntimeError(
                    f"Gap detected between chunk {c_id} (ends at {c_end}) and chunk {next_boundary['chunk_id']} "
                    f"(starts at {next_start}) for source '{input_video}'."
                )
                
    all_source_frames = set(range(continuous_frame_count))
    unassigned = all_source_frames - assigned_frames
    if unassigned:
        raise RuntimeError(
            f"Unassigned frames detected: {sorted(list(unassigned))} not covered "
            f"for source '{input_video}'."
        )

    # 5. Perform chunk aggregations (mean, p95, max)
    chunks_profile_list = []
    chunks_manifest_list = []
    
    for boundary in chunk_boundaries:
        chunk_id_str = boundary["chunk_id"]
        bucket = chunk_buckets[chunk_id_str]
        
        if not bucket:
            # Fallback if empty bucket
            chunk_metrics = {
                "motion": {"mean": 0.0, "p95": 0.0, "max": 0.0},
                "texture_density": {"mean": 0.0, "p95": 0.0},
                "edge_density": {"mean": 0.0, "p95": 0.0},
                "blur": {"mean": 0.0, "p95": 0.0},
                "spatial_complexity": {"mean": 0.0, "p95": 0.0, "max": 0.0}
            }
        else:
            def calc_stats(key):
                vals = [f[key] for f in bucket]
                arr = np.array(vals)
                return {
                    "mean": float(np.mean(arr)),
                    "p95": float(np.percentile(arr, 95)),
                    "max": float(np.max(arr))
                }
                
            motion_stats = calc_stats("motion")
            texture_stats = calc_stats("texture")
            edge_stats = calc_stats("edges")
            blur_stats = calc_stats("blur")
            complexity_stats = calc_stats("complexity")
            
            chunk_metrics = {
                "motion": motion_stats,
                "texture_density": {
                    "mean": texture_stats["mean"],
                    "p95": texture_stats["p95"]
                },
                "edge_density": {
                    "mean": edge_stats["mean"],
                    "p95": edge_stats["p95"]
                },
                "blur": {
                    "mean": blur_stats["mean"],
                    "p95": blur_stats["p95"]
                },
                "spatial_complexity": complexity_stats
            }
            
        chunks_profile_list.append({
            "chunk_id": chunk_id_str,
            "start_time_seconds": boundary["start_time_seconds"],
            "end_time_seconds": boundary["end_time_seconds"],
            "duration_seconds": boundary["duration_seconds"],
            "start_frame": boundary["start_frame"],
            "end_frame": boundary["end_frame"],
            "frame_count": boundary["frame_count"],
            "motion": chunk_metrics["motion"],
            "texture_density": chunk_metrics["texture_density"],
            "edge_density": chunk_metrics["edge_density"],
            "blur": chunk_metrics["blur"],
            "spatial_complexity": chunk_metrics["spatial_complexity"]
        })
        
        chunks_manifest_list.append({
            "chunk_id": chunk_id_str,
            "file_path": os.path.relpath(boundary["file_path"], output_dir),
            "file_hash": boundary["file_hash"]
        })
        
    # 5. Compile output structures
    content_profile = {
        "schema_version": "1.0.0",
        "video_id": video_id,
        "source": {
            "filename": os.path.basename(input_video),
            "duration_seconds": meta["duration"],
            "fps": meta["fps"],
            "width": meta["width"],
            "height": meta["height"],
            "frame_count": meta["frame_count"],
            "codec": meta["codec"],
            "pixel_format": None,
            "bitrate": None,
            "has_audio": meta["has_audio"]
        },
        "profiling_config": {
            "target_chunk_duration_seconds": chunk_duration,
            "motion_temporal_window_seconds": temporal_window_s,
            "aggregation": {
                "motion": ["mean", "p95", "max"],
                "texture": ["mean", "p95"],
                "edge_density": ["mean", "p95"],
                "blur": ["mean", "p95"],
                "complexity": ["mean", "p95", "max"]
            }
        },
        "chunks": chunks_profile_list
    }
    
    profile_path = os.path.join(output_dir, "profiles", f"{video_id}_profile.json")
    manifest_path = os.path.join(output_dir, "manifests", f"{video_id}_manifest.json")
    
    os.makedirs(os.path.dirname(profile_path), exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    
    # Save files
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(content_profile, f, indent=4)
        
    manifest_data = {
        "schema_version": "1.0.0",
        "video_id": video_id,
        "source_file_path": os.path.abspath(input_video),
        "source_file_hash": source_hash,
        "generated_profile_path": os.path.abspath(profile_path),
        "chunks": chunks_manifest_list
    }
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=4)
        
    logger.info(f"Profiling complete. Profile saved to: {profile_path}")
    logger.info(f"Manifest saved to: {manifest_path}")
    
    return profile_path, manifest_path

def main():
    parser = argparse.ArgumentParser(description="Step 1 Video Profiler Pipeline")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Output directory path for profiling assets")
    parser.add_argument("--chunk-duration", type=float, default=2.0, help="Target chunk duration in seconds")
    parser.add_argument("--motion-temporal-window", type=float, default=0.0333, help="Temporal window for motion comparison in seconds")
    
    args = parser.parse_args()
    
    run_profiler(
        input_video=args.input,
        output_dir=args.output,
        chunk_duration=args.chunk_duration,
        temporal_window_s=args.motion_temporal_window
    )

if __name__ == "__main__":
    main()
