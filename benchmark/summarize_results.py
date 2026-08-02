import os
import json
import pandas as pd
from pathlib import Path

CATEGORIES = ["simple", "complex", "mixed"]
CONFIGS = ["baseline_tinysr", "baseline_real_esrgan", "adaptive"]

def summarize_system_metrics(csv_path):
    if not os.path.exists(csv_path):
        return None
        
    df = pd.read_csv(csv_path)
    total_time_s = df["inference_time_ms"].sum() / 1000.0
    avg_cpu = df["cpu"].mean() * 100.0  # percentage
    avg_gpu = df["gpu"].mean() * 100.0  # percentage
    
    # Battery delta
    battery_delta = None
    if df["battery"].notna().any() and len(df) > 1:
        battery_delta = df["battery"].iloc[0] - df["battery"].iloc[-1]
        
    # Average Temperature
    avg_temp = None
    if df["temperature"].notna().any():
        avg_temp = df["temperature"].mean() * 100.0 # scale back if normalized
        
    # Model Distribution
    total_frames = len(df)
    model_counts = df["selected_model"].value_counts().to_dict()
    model_dist = {model: (count / total_frames) * 100.0 for model, count in model_counts.items()}
    
    # Decision Stability (Switch Rate)
    switches = 0
    if total_frames > 1:
        # Count consecutive changes
        switches = (df["selected_model"] != df["selected_model"].shift()).sum() - 1
    switch_rate = (switches / total_frames) * 100.0 if total_frames > 0 else 0.0
    
    return {
        "total_time_s": total_time_s,
        "avg_cpu": avg_cpu,
        "avg_gpu": avg_gpu,
        "battery_delta": battery_delta,
        "avg_temp": avg_temp,
        "model_dist": model_dist,
        "switch_rate": switch_rate,
        "total_frames": total_frames
    }

def main():
    results_dir = Path("benchmark_results")
    
    quality_rows = []
    system_rows = []
    
    for category in CATEGORIES:
        for config in CONFIGS:
            csv_path = results_dir / f"{category}__{config}.csv"
            quality_json = results_dir / f"{category}__{config}_quality.json"
            
            # Read quality metrics
            psnr_val, ssim_val, lpips_val = "-", "-", "-"
            if quality_json.exists():
                with open(quality_json, "r", encoding="utf-8") as f:
                    q_data = json.load(f)
                    psnr_val = f"{q_data['psnr']:.2f}"
                    ssim_val = f"{q_data['ssim']:.4f}"
                    lpips_val = f"{q_data['lpips']:.4f}"
            
            # Read system metrics
            sys_data = summarize_system_metrics(csv_path)
            
            if sys_data is not None:
                # Dist string
                dist_str = ", ".join([f"{m}: {pct:.1f}%" for m, pct in sys_data["model_dist"].items()])
                
                # Battery string
                bat_str = f"{sys_data['battery_delta']*100:.1f}%" if sys_data["battery_delta"] is not None else "-"
                
                # Temperature string
                # Temperature string
                temp_str = f"{sys_data['avg_temp']:.1f} C" if sys_data["avg_temp"] is not None else "-"
                
                quality_rows.append({
                    "Category": category,
                    "Config": config,
                    "PSNR": psnr_val,
                    "SSIM": ssim_val,
                    "LPIPS": lpips_val
                })
                
                system_rows.append({
                    "Category": category,
                    "Config": config,
                    "Total Time": f"{sys_data['total_time_s']:.2f}s",
                    "Avg CPU": f"{sys_data['avg_cpu']:.1f}%",
                    "Avg GPU": f"{sys_data['avg_gpu']:.1f}%",
                    "Battery Delta": bat_str,
                    "Avg Temp": temp_str,
                    "Switch Rate": f"{sys_data['switch_rate']:.1f}%",
                    "Model Distribution": dist_str
                })
                
    # Generate Markdown Report
    report = []
    report.append("# Benchmark Evaluation Summary Report\n")
    report.append("This report summarizes the experimental evaluation of the AdaptiveSR pipeline across three categories (simple, complex, mixed) comparing forced static baselines with the dynamic adaptive configuration.\n")
    
    report.append("## 1. Visual Quality Comparison")
    report.append("| Category | Configuration | Average PSNR | Average SSIM | Average LPIPS |")
    report.append("| :--- | :--- | :--- | :--- | :--- |")
    for r in quality_rows:
        report.append(f"| {r['Category']} | {r['Config']} | {r['PSNR']} | {r['SSIM']} | {r['LPIPS']} |")
        
    report.append("\n## 2. System Telemetry Comparison")
    report.append("| Category | Configuration | Total Latency | Avg CPU | Avg GPU | Battery Delta | Avg Temp | Switch Rate | Model Distribution |")
    report.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in system_rows:
        report.append(f"| {r['Category']} | {r['Config']} | {r['Total Time']} | {r['Avg CPU']} | {r['Avg GPU']} | {r['Battery Delta']} | {r['Avg Temp']} | {r['Switch Rate']} | {r['Model Distribution']} |")
        
    report_content = "\n".join(report)
    
    # Save to file
    summary_path = results_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print("====================================================")
    print("Benchmark Summary Statistics")
    print("====================================================\n")
    print(report_content)
    print(f"\nReport written to: {summary_path}")

if __name__ == "__main__":
    main()
