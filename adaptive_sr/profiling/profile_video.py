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

def profile_chunk(chunk_path: str, chunk_idx: int, temporal_window_s: float, fps: float) -> dict:
    """
    Profiles an individual physical chunk file.
    Uses the temporal comparison window frames = round(fps * temporal_window_s) for motion logic.
    """
    loader = VideoLoader(chunk_path)
    meta = loader.get_metadata()
    
    extractor = FrameExtractor(chunk_path)
    
    # Calculate N frame offset for the motion comparison
    temporal_offset = max(1, int(round(fps * temporal_window_s)))
    logger.info(f"Profiling chunk {chunk_idx:04d} with temporal offset N={temporal_offset} frames (FPS={fps:.2f})")
    
    # Circular buffer to store recent frames
    frame_buffer = deque(maxlen=temporal_offset + 1)
    
    motions = []
    textures = []
    edges = []
    blurs = []
    complexities = []
    
    for idx, ts_ms, frame in extractor.extract():
        frame_buffer.append(frame)
        
        # If we do not have enough frames in buffer yet, comparison is against None (motion=0.0)
        if len(frame_buffer) <= temporal_offset:
            prev_frame = None
        else:
            prev_frame = frame_buffer[0]
            
        metrics = analyze_frame(frame, prev_frame)
        complexity = estimate_complexity(metrics)
        
        motions.append(metrics["motion"])
        textures.append(metrics["texture"])
        edges.append(metrics["edges"])
        blurs.append(metrics["blur_clarity"])
        complexities.append(complexity)
        
    if not complexities:
        # Fallback if empty chunk
        return {
            "frame_count": 0,
            "motion": {"mean": 0.0, "p95": 0.0, "max": 0.0},
            "texture": {"mean": 0.0, "p95": 0.0},
            "edges": {"mean": 0.0, "p95": 0.0},
            "blur": {"mean": 0.0, "p95": 0.0},
            "complexity": {"mean": 0.0, "p95": 0.0, "max": 0.0}
        }
        
    # Helper to calculate mean, p95, and max
    def calc_stats(vals):
        arr = np.array(vals)
        return {
            "mean": float(np.mean(arr)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr))
        }
        
    motion_stats = calc_stats(motions)
    texture_stats = calc_stats(textures)
    edge_stats = calc_stats(edges)
    blur_stats = calc_stats(blurs)
    complexity_stats = calc_stats(complexities)
    
    return {
        "frame_count": len(complexities),
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

def run_profiler(input_video: str, output_dir: str, chunk_duration: float, temporal_window_s: float):
    logger.info(f"Initializing profiler on: {input_video}")
    loader = VideoLoader(input_video)
    meta = loader.get_metadata()
    
    video_id = os.path.splitext(os.path.basename(input_video))[0]
    source_hash = get_file_sha256(input_video)
    
    # 1. Segment the video physically
    logger.info(f"Segmenting video into chunks of {chunk_duration} seconds...")
    chunk_files = segment_video(input_video, output_dir, chunk_duration, video_id)
    
    chunks_profile_list = []
    chunks_manifest_list = []
    
    running_time = 0.0
    running_frame = 0
    
    # 2. Profile each segment
    for idx, chunk_path in enumerate(chunk_files):
        logger.info(f"Processing chunk {idx}: {chunk_path}")
        chunk_metrics = profile_chunk(chunk_path, idx, temporal_window_s, meta["fps"])
        chunk_hash = get_file_sha256(chunk_path)
        
        # Read exact segment duration and frame count
        chunk_loader = VideoLoader(chunk_path)
        chunk_meta = chunk_loader.get_metadata()
        
        chunk_frame_count = chunk_metrics["frame_count"]
        chunk_dur = chunk_meta["duration"]
        
        chunk_id_str = f"{idx:04d}"
        
        chunks_profile_list.append({
            "chunk_id": chunk_id_str,
            "start_time_seconds": running_time,
            "end_time_seconds": running_time + chunk_dur,
            "duration_seconds": chunk_dur,
            "start_frame": running_frame,
            "end_frame": running_frame + chunk_frame_count - 1 if chunk_frame_count > 0 else running_frame,
            "frame_count": chunk_frame_count,
            "motion": chunk_metrics["motion"],
            "texture_density": chunk_metrics["texture_density"],
            "edge_density": chunk_metrics["edge_density"],
            "blur": chunk_metrics["blur"],
            "spatial_complexity": chunk_metrics["spatial_complexity"]
        })
        
        chunks_manifest_list.append({
            "chunk_id": chunk_id_str,
            "file_path": os.path.relpath(chunk_path, output_dir),
            "file_hash": chunk_hash
        })
        
        running_time += chunk_dur
        running_frame += chunk_frame_count
        
    # 3. Compile output structures
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
            "pixel_format": None,  # Optional placeholder
            "bitrate": None,       # Optional placeholder
            "has_audio": meta["has_audio"]
        },
        "profiling_config": {
            "chunk_duration_seconds": chunk_duration,
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
