import os
import sys
import time
from pathlib import Path

# Add project root to path so we can import src.main
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.main import run_pipeline

CONFIGS = {
    "baseline_tinysr": {"force_model": "tinysr"},
    "baseline_real_esrgan": {"force_model": "real_esrgan"},
    "adaptive": {"force_model": None},
}

CATEGORIES = ["simple", "complex", "mixed"]

def main():
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    
    print("====================================================")
    print("Starting AdaptiveSR Benchmark Baselines Runner")
    print(f"Configs: {list(CONFIGS.keys())}")
    print(f"Categories: {CATEGORIES}")
    print("====================================================\n")
    
    for category in CATEGORIES:
        input_path = f"benchmark_data/{category}_lr.mp4"
        if not os.path.exists(input_path):
            print(f"Error: Input video not found: {input_path}")
            continue
            
        for config_name, config in CONFIGS.items():
            result_path = output_dir / f"{category}__{config_name}.mp4"
            log_path = output_dir / f"{category}__{config_name}.csv"
            
            # Resumability Check: Critical requirement
            if result_path.exists() and log_path.exists():
                print(f"[SKIP] Already completed: {config_name} on category '{category}'")
                continue
                
            print(f"[RUNNING] Config: {config_name} | Category: {category}...")
            t0 = time.perf_counter()
            
            # Run the pipeline
            run_pipeline(
                input_path=input_path,
                output_path=str(result_path),
                config_path="configs/decision_config.yaml",
                log_path=str(log_path),
                poll_interval=0.5,
                force_model=config["force_model"]
            )
            
            dt = time.perf_counter() - t0
            print(f"[FINISHED] Completed in {dt:.2f} seconds\n")
            
    print("All baseline runs completed successfully!")

if __name__ == "__main__":
    main()
