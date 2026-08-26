import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from adaptive_sr.benchmarking.harness import InferenceBenchmarkHarness, BenchmarkConfig, CPUExecutionConfig

def main():
    print("Running live Step 5.5 benchmark for Step 5.7 analysis...")
    harness = InferenceBenchmarkHarness()
    
    # Configure CPU execution case for tinysr x2
    cpu_conf = CPUExecutionConfig(cpu_ids=[1], num_threads=2, exclude_cpu_ids=[])
    cfg = BenchmarkConfig(
        model_id="tinysr",
        scale=2,
        input_id="synthetic_lowmotion_30fps",
        device="cpu",
        cpu_config=cpu_conf,
        warmup_runs=1,
        measured_runs=5
    )

    # Run multi-session benchmark
    res = harness.run_multi_session(cfg, num_sessions=1)
    
    # Dump MultiSessionResult to JSON list
    output_dir = Path("data/benchmarks/sr/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "benchmark_sessions.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump([res.model_dump()], f, indent=4)
        
    print(f"Benchmark sessions saved to {results_path}")

if __name__ == "__main__":
    main()
