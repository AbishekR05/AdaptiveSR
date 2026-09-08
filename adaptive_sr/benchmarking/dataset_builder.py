"""
adaptive_sr.benchmarking.dataset_builder
=========================================
Step 5.8 — Machine-Readable Benchmark Dataset Builder.
Consolidates empirical outputs from Step 5.5, Step 5.6, and Step 5.7
into a single canonical, machine-readable dataset (JSON and CSV).
"""

import os
import sys
import json
import csv
import logging
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure repository root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def make_record_key(model_id: str, scale: int, device: str, input_id: str) -> str:
    """Generates a canonical join key for benchmark records."""
    dev_str = device.lower().replace(":", "_")
    return f"{model_id}_x{scale}_{dev_str}_{input_id}"


def load_json_data(filepath: str) -> Optional[Any]:
    """Helper to safely load JSON files."""
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read JSON file {filepath}: {e}")
        return None


def extract_step55_records(data: Any) -> Dict[str, Dict[str, Any]]:
    """Extracts Step 5.5 benchmark records mapped by canonical key."""
    records = {}
    if data is None:
        return records

    items = data if isinstance(data, list) else [data]

    for item in items:
        if not isinstance(item, dict) or "config" not in item:
            continue

        cfg = item["config"]
        model_id = cfg.get("model_id")
        scale = cfg.get("scale")
        device = cfg.get("device")
        input_id = cfg.get("input_id")

        if not model_id or scale is None or not device or not input_id:
            continue

        key = make_record_key(model_id, scale, device, input_id)

        # Extract latencies & stats
        lat_stats = item.get("latency_statistics", {})
        latency_ms = lat_stats.get("median_latency", lat_stats.get("median"))
        if latency_ms is not None and latency_ms > 0 and latency_ms < 10.0:
            # Latency recorded in seconds; convert to ms
            latency_ms = latency_ms * 1000.0

        p95_ms = lat_stats.get("p95_latency", lat_stats.get("p95"))
        if p95_ms is not None and p95_ms > 0 and p95_ms < 10.0:
            p95_ms = p95_ms * 1000.0

        throughput = item.get("throughput_fps")
        if throughput is None and latency_ms and latency_ms > 0:
            throughput = 1000.0 / latency_ms

        cpu_cfg = cfg.get("cpu_config", {})
        cpu_ids = cpu_cfg.get("cpu_ids") if isinstance(cpu_cfg, dict) else None
        num_threads = cpu_cfg.get("num_threads") if isinstance(cpu_cfg, dict) else None

        records[key] = {
            "model_id": model_id,
            "scale": scale,
            "device": device,
            "input_id": input_id,
            "latency_ms": latency_ms,
            "p95_latency_ms": p95_ms,
            "throughput_fps": throughput,
            "cpu_ids": cpu_ids,
            "num_threads": num_threads,
            "gpu_device_id": cfg.get("gpu_device_id"),
            "resource_summary": item.get("resource_summary", {}),
            "decision_eligible": item.get("decision_eligible", True),
            "eligibility_reason": item.get("eligibility_reason", "N/A"),
            "session_count": item.get("session_count", len(item.get("sessions", [1])) if "sessions" in item else 1),
            "raw_record": item
        }

    return records


def extract_step56_records(data: Any) -> Dict[str, Dict[str, Any]]:
    """Extracts Step 5.6 quality evaluation records mapped by canonical key."""
    records = {}
    if data is None:
        return records

    # Support dict with 'records' key or list of dicts
    items = []
    if isinstance(data, dict):
        items = data.get("records", [])
    elif isinstance(data, list):
        items = data

    for item in items:
        if not isinstance(item, dict):
            continue

        model_id = item.get("model_id")
        scale = item.get("scale")
        device = item.get("device", "cpu")
        input_id = item.get("clip_id", item.get("benchmark_video_id", item.get("input_id")))

        if not model_id or scale is None or not input_id:
            continue

        key = make_record_key(model_id, scale, device, input_id)

        records[key] = {
            "psnr_y": item.get("psnr_mean", item.get("psnr_y")),
            "ssim_y": item.get("ssim_mean", item.get("ssim_y")),
            "vmaf_mean": item.get("vmaf_mean"),
            "vmaf_unavailable": item.get("vmaf_unavailable", True),
            "chunk_count": item.get("chunk_count")
        }

    return records


def extract_step57_records(data: Any) -> Dict[str, Dict[str, Any]]:
    """Extracts Step 5.7 FPS real-time feasibility records mapped by canonical key."""
    records = {}
    if data is None:
        return records

    items = data if isinstance(data, list) else [data]

    for item in items:
        if not isinstance(item, dict):
            continue

        model_id = item.get("model_id")
        scale = item.get("scale")
        device = item.get("device")
        input_id = item.get("benchmark_video_id", item.get("input_id"))

        if not model_id or scale is None or not device or not input_id:
            continue

        key = make_record_key(model_id, scale, device, input_id)

        records[key] = {
            "source_fps": item.get("source_fps"),
            "frame_budget_ms": item.get("frame_budget_ms"),
            "estimated_processing_fps": item.get("estimated_processing_fps"),
            "real_time_ratio": item.get("real_time_ratio"),
            "real_time_feasible": item.get("real_time_feasible"),
            "real_time_feasible_p95_exploratory": item.get("real_time_feasible_p95_exploratory"),
            "p95_confidence": item.get("p95_confidence", "exploratory"),
            "caveats": item.get("caveats", []),
            "source_fps_gap": item.get("source_fps_gap", False)
        }

    return records


def build_unified_dataset(
    step55_file: Optional[str] = None,
    step56_file: Optional[str] = None,
    step57_file: Optional[str] = None,
    step55_raw_data: Optional[Any] = None,
    step56_raw_data: Optional[Any] = None,
    step57_raw_data: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Builds the canonical Step 5.8 unified benchmark dataset by joining records across Steps 5.5, 5.6, and 5.7."""

    # Load data if files provided
    d55 = step55_raw_data if step55_raw_data is not None else (load_json_data(step55_file) if step55_file else None)
    d56 = step56_raw_data if step56_raw_data is not None else (load_json_data(step56_file) if step56_file else None)
    d57 = step57_raw_data if step57_raw_data is not None else (load_json_data(step57_file) if step57_file else None)

    r55 = extract_step55_records(d55)
    r56 = extract_step56_records(d56)
    r57 = extract_step57_records(d57)

    # Union of all canonical keys across sources
    all_keys = sorted(list(set(r55.keys()) | set(r56.keys()) | set(r57.keys())))

    now_iso = datetime.now(timezone.utc).isoformat()
    unified_records = []

    for key in all_keys:
        rec55 = r55.get(key, {})
        rec56 = r56.get(key, {})
        rec57 = r57.get(key, {})

        # Identify core attributes from whichever source contains them
        model_id = rec55.get("model_id") or rec57.get("model_id") or key.split("_x")[0]
        
        # Parse scale and device safely
        scale = rec55.get("scale") or rec57.get("scale")
        if scale is None:
            try:
                scale = int(key.split("_x")[1].split("_")[0])
            except Exception:
                scale = 2

        device = rec55.get("device") or rec57.get("device") or ("cuda" if "cuda" in key else "cpu")
        input_id = rec55.get("input_id") or rec57.get("input_id") or key.split("_")[-1]

        # Consolidated unified record
        record = {
            "record_id": key,
            "model_id": model_id,
            "scale": scale,
            "device": device,
            "input_id": input_id,
            "benchmark_inference": {
                "latency_ms": rec55.get("latency_ms"),
                "p95_latency_ms": rec55.get("p95_latency_ms"),
                "throughput_fps": rec55.get("throughput_fps"),
                "cpu_ids": rec55.get("cpu_ids"),
                "num_threads": rec55.get("num_threads"),
                "gpu_device_id": rec55.get("gpu_device_id"),
                "resource_summary": rec55.get("resource_summary", {}),
                "decision_eligible": rec55.get("decision_eligible", None),
                "eligibility_reason": rec55.get("eligibility_reason", None),
                "session_count": rec55.get("session_count", None),
                "data_present": bool(rec55)
            },
            "quality_evaluation": {
                "psnr_y": rec56.get("psnr_y"),
                "ssim_y": rec56.get("ssim_y"),
                "vmaf_mean": rec56.get("vmaf_mean"),
                "vmaf_unavailable": rec56.get("vmaf_unavailable", True if rec56 else None),
                "data_present": bool(rec56)
            },
            "realtime_feasibility": {
                "source_fps": rec57.get("source_fps"),
                "frame_budget_ms": rec57.get("frame_budget_ms"),
                "estimated_processing_fps": rec57.get("estimated_processing_fps"),
                "real_time_ratio": rec57.get("real_time_ratio"),
                "real_time_feasible": rec57.get("real_time_feasible"),
                "real_time_feasible_p95_exploratory": rec57.get("real_time_feasible_p95_exploratory"),
                "p95_confidence": rec57.get("p95_confidence"),
                "caveats": rec57.get("caveats", []),
                "data_present": bool(rec57)
            },
            "provenance": {
                "step55_source": step55_file if step55_file and rec55 else None,
                "step56_source": step56_file if step56_file and rec56 else None,
                "step57_source": step57_file if step57_file and rec57 else None,
                "dataset_version": "1.0",
                "generated_timestamp": now_iso
            }
        }

        unified_records.append(record)

    return unified_records


def export_csv_dataset(records: List[Dict[str, Any]], csv_filepath: str) -> None:
    """Exports unified benchmark dataset into a flattened CSV format."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_filepath)), exist_ok=True)

    fieldnames = [
        "record_id", "model_id", "scale", "device", "input_id",
        "latency_ms", "p95_latency_ms", "throughput_fps",
        "decision_eligible", "session_count", "eligibility_reason",
        "psnr_y", "ssim_y", "vmaf_mean", "vmaf_unavailable",
        "source_fps", "frame_budget_ms", "estimated_processing_fps",
        "real_time_ratio", "real_time_feasible", "p95_confidence",
        "step55_present", "step56_present", "step57_present"
    ]

    with open(csv_filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rec in records:
            b_inf = rec.get("benchmark_inference", {})
            q_eval = rec.get("quality_evaluation", {})
            r_feas = rec.get("realtime_feasibility", {})

            row = {
                "record_id": rec.get("record_id"),
                "model_id": rec.get("model_id"),
                "scale": rec.get("scale"),
                "device": rec.get("device"),
                "input_id": rec.get("input_id"),
                "latency_ms": b_inf.get("latency_ms"),
                "p95_latency_ms": b_inf.get("p95_latency_ms"),
                "throughput_fps": b_inf.get("throughput_fps"),
                "decision_eligible": b_inf.get("decision_eligible"),
                "session_count": b_inf.get("session_count"),
                "eligibility_reason": b_inf.get("eligibility_reason"),
                "psnr_y": q_eval.get("psnr_y"),
                "ssim_y": q_eval.get("ssim_y"),
                "vmaf_mean": q_eval.get("vmaf_mean"),
                "vmaf_unavailable": q_eval.get("vmaf_unavailable"),
                "source_fps": r_feas.get("source_fps"),
                "frame_budget_ms": r_feas.get("frame_budget_ms"),
                "estimated_processing_fps": r_feas.get("estimated_processing_fps"),
                "real_time_ratio": r_feas.get("real_time_ratio"),
                "real_time_feasible": r_feas.get("real_time_feasible"),
                "p95_confidence": r_feas.get("p95_confidence"),
                "step55_present": b_inf.get("data_present", False),
                "step56_present": q_eval.get("data_present", False),
                "step57_present": r_feas.get("data_present", False),
            }
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Step 5.8 Machine-Readable Benchmark Dataset Builder CLI")
    parser.add_argument("--step55-file", type=str, default="data/benchmarks/sr/results/step55_results.json", help="Path to Step 5.5 results JSON")
    parser.add_argument("--step56-file", type=str, default="data/benchmarks/sr/results/quality_clips.json", help="Path to Step 5.6 quality JSON")
    parser.add_argument("--step57-file", type=str, default="data/benchmarks/sr/results/fps_feasibility.json", help="Path to Step 5.7 feasibility JSON")
    parser.add_argument("--output-json", type=str, default="data/benchmarks/sr/results/unified_benchmark_dataset.json", help="Path to output JSON")
    parser.add_argument("--output-csv", type=str, default="data/benchmarks/sr/results/unified_benchmark_dataset.csv", help="Path to output CSV")
    args = parser.parse_args()

    records = build_unified_dataset(
        step55_file=args.step55_file if os.path.exists(args.step55_file) else None,
        step56_file=args.step56_file if os.path.exists(args.step56_file) else None,
        step57_file=args.step57_file if os.path.exists(args.step57_file) else None,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4)

    export_csv_dataset(records, args.output_csv)
    logger.info(f"Step 5.8 complete. Unified dataset written to {args.output_json} and {args.output_csv} (Total records: {len(records)})")


if __name__ == "__main__":
    main()
