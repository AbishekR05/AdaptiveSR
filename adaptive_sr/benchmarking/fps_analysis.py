"""
adaptive_sr.benchmarking.fps_analysis
======================================
Step 5.7 — FPS / Real-Time Feasibility Analysis.
Performs read-only analysis over Step 5.5 benchmark outputs.
"""

import os
import sys
import json
import logging
import argparse
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure repository root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Discovered model limits (§6)
MODEL_LIMITS = {
    "real_esrgan": [2, 4],
    "tinysr_int8": [2],
    "tinysr": [2, 3, 4]
}


def load_manifests_maps(manifests_dir: str) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Loads all manifests in manifests_dir and maps benchmark_video_id to source_fps and frame_count."""
    fps_map = {}
    frame_count_map = {}
    if not os.path.exists(manifests_dir):
        logger.warning(f"Manifests directory does not exist: {manifests_dir}")
        return fps_map, frame_count_map

    for fname in os.listdir(manifests_dir):
        if fname.endswith(".json"):
            fpath = os.path.join(manifests_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for v in data.get("videos", []):
                    vid = v.get("benchmark_video_id")
                    if vid:
                        if "source_fps" in v:
                            fps_map[vid] = float(v["source_fps"])
                        chunks = v.get("chunks", [])
                        if chunks and "frame_count" in chunks[0]:
                            frame_count_map[vid] = int(chunks[0]["frame_count"])
            except Exception as e:
                logger.error(f"Error reading manifest {fname}: {e}")

    return fps_map, frame_count_map


def parse_device_id(device: str) -> Optional[int]:
    """Extracts GPU device ID if device format is cuda:N, else returns None."""
    device_lower = device.lower()
    if device_lower.startswith("cuda"):
        if ":" in device_lower:
            try:
                return int(device_lower.split(":")[1])
            except ValueError:
                pass
        return 0  # default to 0 if only 'cuda' is specified
    return None


def analyze_record(
    record: Dict[str, Any],
    fps_map: Dict[str, float],
    frame_count_map: Dict[str, int]
) -> Dict[str, Any]:
    """Analyzes a single Step 5.5 BenchmarkResult or MultiSessionResult record."""
    # Verify metadata fields are present
    if "config" not in record:
        raise ValueError("Invalid Step 5.5 record: missing config section.")

    config = record["config"]
    model_id = config.get("model_id")
    scale = config.get("scale")
    device = config.get("device")
    input_id = config.get("input_id")

    if not model_id or scale is None or not device or not input_id:
        raise ValueError("Invalid config: model_id, scale, device, and input_id are required.")

    # Check unsupported model+scale limits (§6, §12)
    if model_id in MODEL_LIMITS:
        allowed = MODEL_LIMITS[model_id]
        if scale not in allowed:
            raise ValueError(f"Unsupported model+scale combination: {model_id} x{scale}")

    # Extract latency statistics
    # Check if this is a MultiSessionResult or single BenchmarkResult
    is_multi = "sessions" in record
    sessions = record.get("sessions", [])

    all_latencies = []
    if is_multi:
        for s in sessions:
            all_latencies.extend(s.get("trial_latencies", []))
    else:
        all_latencies.extend(record.get("trial_latencies", []))

    if not all_latencies:
        raise ValueError(f"No trial latencies found in record for model {model_id}.")

    # Calculate statistics in milliseconds
    median_latency_seconds = float(np.median(all_latencies))
    p95_latency_seconds = float(np.percentile(all_latencies, 95))

    # §12: Zero/negative latency raises explicit error
    if median_latency_seconds <= 0 or p95_latency_seconds <= 0:
        raise ValueError(f"Invalid non-positive latency observed: median={median_latency_seconds}s, p95={p95_latency_seconds}s")

    latency_ms = median_latency_seconds * 1000.0
    p95_latency_ms = p95_latency_seconds * 1000.0

    # Resolve source_fps
    source_fps = fps_map.get(input_id)
    source_fps_gap = source_fps is None

    # Resolve latency interpretation (§9)
    # Spatial models processed as per-frame. Temporal models processed as per-sequence.
    spatial_models = ["tinysr", "tinysr_int8", "real_esrgan"]
    if model_id in spatial_models:
        latency_interpretation = "per_frame"
        estimated_processing_fps = 1000.0 / latency_ms
    else:
        latency_interpretation = "per_sequence"
        sequence_frame_count = frame_count_map.get(input_id, 60)
        estimated_processing_fps = sequence_frame_count / median_latency_seconds

    # Frame budget calculations
    if not source_fps_gap:
        frame_budget_ms = 1000.0 / source_fps
        real_time_ratio = frame_budget_ms / latency_ms
        real_time_feasible = latency_ms <= frame_budget_ms
        real_time_feasible_p95_exploratory = p95_latency_ms <= frame_budget_ms
        budget_utilization_percent = (latency_ms / frame_budget_ms) * 100.0
    else:
        frame_budget_ms = None
        real_time_ratio = None
        real_time_feasible = None
        real_time_feasible_p95_exploratory = None
        budget_utilization_percent = None

    # CPU configurations
    cpu_ids = None
    num_threads = None
    cpu_config = config.get("cpu_config")
    if cpu_config:
        cpu_ids = cpu_config.get("cpu_ids")
        num_threads = cpu_config.get("num_threads")

    # GPU device configuration
    gpu_device_id = parse_device_id(device)

    # Session/variance handling variables
    decision_eligible = record.get("decision_eligible", True)
    # If explicitly defined as False in multi session output
    if is_multi:
        session_count = len(sessions)
    else:
        session_count = 1

    # Extract p95 confidence metric
    p95_confidence = "exploratory"
    if is_multi and sessions:
        stats = sessions[0].get("latency_statistics", {})
        p95_confidence = stats.get("p95_confidence", "exploratory")
    elif "latency_statistics" in record:
        stats = record.get("latency_statistics", {})
        p95_confidence = stats.get("p95_confidence", "exploratory")

    # Extract caveats and warnings
    caveats = []
    # Preserve errors/tracebacks
    for fail in record.get("failures", []):
        caveats.append(f"Trial failure: {fail}")
    if record.get("flagged"):
        caveats.append(f"Flagged session: {record['flagged']}")

    # Formulate output structure conforming to §10
    analysis = {
        "benchmark_video_id": input_id if input_id else None,
        "model_id": model_id,
        "scale": scale,
        "device": device,
        "cpu_ids": cpu_ids,
        "num_threads": num_threads,
        "gpu_device_id": gpu_device_id,
        "source_fps": source_fps,
        "frame_budget_ms": frame_budget_ms,
        "latency_ms": latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "p95_exploratory": True,
        "latency_interpretation": latency_interpretation,
        "estimated_processing_fps": estimated_processing_fps,
        "real_time_ratio": real_time_ratio,
        "real_time_feasible": real_time_feasible,
        "real_time_feasible_p95_exploratory": real_time_feasible_p95_exploratory,
        "budget_utilization_percent": budget_utilization_percent,
        "decision_eligible": bool(decision_eligible),
        "session_count": int(session_count),
        "p95_confidence": p95_confidence,
        "caveats": caveats,
        "source_fps_gap": source_fps_gap
    }

    return analysis


def generate_markdown_report(analyzed: List[Dict[str, Any]]) -> str:
    """Generates the step 5.7 human-readable feasibility report conforming to §5 and §11."""
    # Verbatim disclosure block (§5)
    disclosure = (
        "> [!IMPORTANT]\n"
        "> **SR Inference-Only Feasibility Disclosure:**\n"
        "> These results measure SR model inference latency only. Decoding, preprocessing, encoding, and "
        "network transfer are NOT included. End-to-end streaming real-time feasibility cannot be "
        "established from Step 5.5/5.7 data alone.\n"
    )

    # 1. Split into categories
    feasible_eligible = []
    feasible_measured_only = []
    failed_configs = []

    for item in analyzed:
        is_feasible = item["real_time_feasible"]
        is_eligible = item["decision_eligible"]

        if is_feasible is True:
            if is_eligible:
                feasible_eligible.append(item)
            else:
                feasible_measured_only.append(item)
        elif is_feasible is False:
            failed_configs.append(item)

    # Find fastest config (median latency, decision-eligible only)
    eligible_items = [x for x in analyzed if x["decision_eligible"] and x["real_time_feasible"]]
    if eligible_items:
        fastest_item = min(eligible_items, key=lambda x: x["latency_ms"])
        fastest_str = (
            f"**Fastest Real-Time Eligible Configuration:** "
            f"`{fastest_item['model_id']}` (x{fastest_item['scale']}) on `{fastest_item['device']}` "
            f"with a median latency of **{fastest_item['latency_ms']:.2f} ms** "
            f"({fastest_item['estimated_processing_fps']:.2f} FPS vs target {fastest_item['source_fps']} FPS).\n"
        )
    else:
        fastest_str = "**Fastest Real-Time Eligible Configuration:** None found.\n"

    # Main comparison table
    table_lines = [
        "| Model | Scale | Device | Latency (median, ms) | Est. FPS | Source FPS | Budget (ms) | Ratio | Real-Time (measured) | Decision-Eligible |",
        "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for item in analyzed:
        model = item["model_id"]
        scale = item["scale"]
        dev = item["device"]
        lat = f"{item['latency_ms']:.2f}"
        est_fps = f"{item['estimated_processing_fps']:.2f}"
        src_fps = f"{item['source_fps']:.1f}" if item["source_fps"] is not None else "N/A"
        budget = f"{item['frame_budget_ms']:.2f}" if item["frame_budget_ms"] is not None else "N/A"
        ratio = f"{item['real_time_ratio']:.2f}" if item["real_time_ratio"] is not None else "N/A"
        feasible = "YES" if item["real_time_feasible"] is True else ("NO" if item["real_time_feasible"] is False else "N/A")
        eligible = "YES" if item["decision_eligible"] else "NO"

        table_lines.append(
            f"| {model} | x{scale} | {dev} | {lat} | {est_fps} | {src_fps} | {budget} | {ratio} | {feasible} | {eligible} |"
        )

    comparison_table = "\n".join(table_lines)

    # Scale-degradation analysis (§6)
    scale_lines = []
    # Group by model_id, device
    groups = {}
    for item in analyzed:
        key = (item["model_id"], item["device"])
        if key not in groups:
            groups[key] = []
        groups[key].append(item)

    for key, items in groups.items():
        model, device = key
        sorted_items = sorted(items, key=lambda x: x["scale"])
        scale_lines.append(f"### {model} on {device}")
        for item in sorted_items:
            feasible_str = "FEASIBLE" if item["real_time_feasible"] else "UNFEASIBLE"
            scale_lines.append(
                f"- **Scale x{item['scale']}**: Latency **{item['latency_ms']:.2f} ms** "
                f"({item['estimated_processing_fps']:.2f} FPS), {feasible_str}."
            )

    scale_summary = "\n".join(scale_lines)

    # CPU vs GPU side-by-side comparison (§7)
    cpu_gpu_table_lines = [
        "| Model | Scale | CPU Threads | CPU Latency (ms) | GPU Latency (ms) | Ratio (CPU/GPU) | Feasibility Verdict |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |"
    ]

    # Match CPU and GPU rows
    cpu_rows = [x for x in analyzed if "cpu" in x["device"].lower()]
    gpu_rows = [x for x in analyzed if "cuda" in x["device"].lower()]

    for cpu in cpu_rows:
        # Find matching GPU row
        match_gpu = None
        for gpu in gpu_rows:
            if gpu["model_id"] == cpu["model_id"] and gpu["scale"] == cpu["scale"]:
                match_gpu = gpu
                break

        if match_gpu:
            ratio_val = cpu["latency_ms"] / match_gpu["latency_ms"]
            verdict = f"CPU: {'Feasible' if cpu['real_time_feasible'] else 'Fail'} / GPU: {'Feasible' if match_gpu['real_time_feasible'] else 'Fail'}"
            cpu_gpu_table_lines.append(
                f"| {cpu['model_id']} | x{cpu['scale']} | {cpu['num_threads']} | {cpu['latency_ms']:.2f} | {match_gpu['latency_ms']:.2f} | {ratio_val:.2f}x | {verdict} |"
            )

    cpu_gpu_table = "\n".join(cpu_gpu_table_lines)

    # Formatting sections
    report = (
        f"# Step 5.7 — FPS / Real-Time Feasibility Analysis: Results & Conclusions\n\n"
        f"**Freeze Status:** PIPELINE VALIDATED. Real-world quality evidence deferred pending genuine Layer B natural-video corpus (tracked as future work).\n\n"
        f"{disclosure}\n"
        f"## 1. Executive Summary\n\n"
        f"{fastest_str}\n"
        f"## 2. Quantitative Benchmark Comparisons\n\n"
        f"{comparison_table}\n\n"
        f"## 3. Real-Time Feasibility Classifications\n\n"
        f"### 3.1 Measured + Decision-Eligible Configurations (Safe for Production)\n"
    )

    if feasible_eligible:
        for item in feasible_eligible:
            report += f"- `{item['model_id']}` x{item['scale']} on `{item['device']}` (Latency: **{item['latency_ms']:.2f} ms**)\n"
    else:
        report += "- None.\n"

    report += "\n### 3.2 Measured Feasible Only (Not Decision-Eligible due to high variance)\n"
    if feasible_measured_only:
        for item in feasible_measured_only:
            report += f"- `{item['model_id']}` x{item['scale']} on `{item['device']}` (Latency: **{item['latency_ms']:.2f} ms**)\n"
    else:
        report += "- None.\n"

    report += "\n### 3.3 Failing Configurations (Unfeasible for Real-Time)\n"
    if failed_configs:
        for item in failed_configs:
            report += f"- `{item['model_id']}` x{item['scale']} on `{item['device']}` (Latency: **{item['latency_ms']:.2f} ms**)\n"
    else:
        report += "- None.\n"

    report += (
        f"\n## 4. Scale-Degradation Performance Trend\n\n"
        f"{scale_summary}\n\n"
        f"## 5. CPU vs. GPU Performance Comparison\n\n"
        f"{cpu_gpu_table}\n"
    )

    return report


def main():
    parser = argparse.ArgumentParser(description="Step 5.7 FPS / Real-Time Feasibility Analysis CLI")
    parser.add_argument("--results-file", type=str, required=True, help="Path to Step 5.5 results JSON file")
    parser.add_argument("--manifests-dir", type=str, default="data/benchmarks/sr/manifests", help="Path to manifest JSON files directory")
    parser.add_argument("--output-json", type=str, default="data/benchmarks/sr/results/fps_feasibility.json", help="Path to write analyzed results JSON")
    parser.add_argument("--output-report", type=str, default="Markdowns/Results/Step5.7 results and conclusion.md", help="Path to write Markdown report")
    args = parser.parse_args()

    # Load manifest maps
    fps_map, frame_count_map = load_manifests_maps(args.manifests_dir)

    # Load Step 5.5 results
    if not os.path.exists(args.results_file):
        logger.error(f"Results file does not exist: {args.results_file}")
        sys.exit(1)

    with open(args.results_file, "r", encoding="utf-8") as f:
        results_data = json.load(f)

    # Check if input is a list of results or single file
    analyzed_records = []
    if isinstance(results_data, list):
        for rec in results_data:
            analyzed_records.append(analyze_record(rec, fps_map, frame_count_map))
    else:
        analyzed_records.append(analyze_record(results_data, fps_map, frame_count_map))

    # Ensure output directories exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_report)), exist_ok=True)

    # Save structured JSON
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(analyzed_records, f, indent=4)

    # Save human-readable report
    report = generate_markdown_report(analyzed_records)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Step 5.7 completed. JSON written to {args.output_json}, report written to {args.output_report}")


if __name__ == "__main__":
    main()
