import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

output_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Plot Per-Frame Complexity vs Inference Latency for Mixed Category (Adaptive)
mixed_csv = os.path.join(output_dir, "..", "benchmark_results", "mixed__adaptive.csv")
if os.path.exists(mixed_csv):
    df_mixed = pd.read_csv(mixed_csv)
    fig, ax1 = plt.subplots(figsize=(7, 3.2))
    
    frames = df_mixed['frame_no']
    complexity = df_mixed['complexity_score']
    latency = df_mixed['inference_time_ms']
    models = df_mixed['selected_model']

    color = '#1f77b4'
    ax1.set_xlabel('Frame Index', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Complexity Score', color=color, fontsize=11, fontweight='bold')
    line1 = ax1.plot(frames, complexity, color=color, linewidth=2, label='Scene Complexity')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0.2, 0.35)

    ax2 = ax1.twinx()
    color2 = '#d62728'
    ax2.set_ylabel('Latency (ms)', color=color2, fontsize=11, fontweight='bold')
    line2 = ax2.plot(frames, latency, color=color2, linewidth=1.5, linestyle='--', marker='o', markersize=3, label='Inference Latency')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Highlight TinySR bypass/fast frame
    fast_frames = df_mixed[df_mixed['selected_model'] == 'tinysr']
    if not fast_frames.empty:
        for idx, row in fast_frames.iterrows():
            ax2.annotate('TinySR Switch\n(42 ms)', xy=(row['frame_no'], row['inference_time_ms']),
                         xytext=(row['frame_no'] + 4, row['inference_time_ms'] + 300),
                         arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=5),
                         fontsize=8, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6))

    plt.title('AdaptiveSR: Dynamic Scene Complexity vs. Processing Latency (Mixed Benchmark)', fontsize=11, fontweight='bold', pad=10)
    fig.tight_layout()
    plot1_path = os.path.join(output_dir, "frame_latency_complexity.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Generated {plot1_path}")

# 2. Overall Benchmark Comparison Bar Chart (Latency & Quality)
categories = ['Simple', 'Mixed', 'Test Data']
tinysr_latency = [3.40, 2.85, 3.05]
realesrgan_latency = [70.41, 67.44, 72.56]
adaptive_latency = [2.82, 66.37, 65.23]

x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots(figsize=(6.5, 3.2))
rects1 = ax.bar(x - width, tinysr_latency, width, label='Static TinySR', color='#7f7f7f', alpha=0.85)
rects2 = ax.bar(x, realesrgan_latency, width, label='Static Real-ESRGAN', color='#d62728', alpha=0.85)
rects3 = ax.bar(x + width, adaptive_latency, width, label='AdaptiveSR (Ours)', color='#2ca02c', alpha=0.9)

ax.set_ylabel('Total Execution Time (Seconds)', fontsize=10, fontweight='bold')
ax.set_title('End-to-End Execution Latency by Video Category Across GPU Platforms', fontsize=10, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9, fontweight='bold')
ax.legend(frameon=True, fontsize=9)
ax.grid(axis='y', linestyle=':', alpha=0.6)

# Annotate speedup
ax.annotate('96.0% Speedup', xy=(0 + width, 2.82), xytext=(0 + 0.05, 22),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4),
            fontsize=8, fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", fc="#d4edda", ec="#28a745"))

fig.tight_layout()
plot2_path = os.path.join(output_dir, "benchmark_latency_comparison.png")
plt.savefig(plot2_path, dpi=300)
plt.close()
print(f"Generated {plot2_path}")
