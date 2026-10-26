import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

output_dir = os.path.dirname(os.path.abspath(__file__))

# Create a figure with high DPI
fig, ax = plt.subplots(figsize=(7, 7.5), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 105)
ax.axis('off')

# Styling constants
font_title = {'fontsize': 9.5, 'weight': 'bold', 'ha': 'center', 'va': 'center', 'color': '#111'}
font_sub = {'fontsize': 7.8, 'ha': 'center', 'va': 'center', 'color': '#333'}
arrow_props = dict(arrowstyle="-|>", lw=1.3, color="#222", mutation_scale=11)

def draw_box(x, y, w, h, title, subtitle="", bg="#f8f9fa", border="#495057", radius=0.08):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.2,rounding_size={radius}",
                                fc=bg, ec=border, lw=1.2, zorder=2)
    ax.add_patch(box)
    if subtitle:
        ax.text(x + w/2, y + h*0.62, title, fontdict=font_title, zorder=3)
        ax.text(x + w/2, y + h*0.28, subtitle, fontdict=font_sub, zorder=3)
    else:
        ax.text(x + w/2, y + h/2, title, fontdict=font_title, zorder=3)

# 1. Input Stream
draw_box(25, 96, 50, 6, "Input Video Stream", "Decoded YCbCr Sequences", bg="#e3f2fd", border="#1976d2")

# 2. Decoupled Frame Buffer
draw_box(25, 84, 50, 6.5, "Decoupled Video Frame Buffer", "Non-blocking Frame Exchange Pipeline", bg="#ede7f6", border="#5e35b1")
ax.annotate("", xy=(50, 90.5), xytext=(50, 96), arrowprops=arrow_props)

# 3. Parallel Dual-Stream: Scene Extractor & Telemetry Monitor
# Left: Scene Extractor
draw_box(5, 68, 42, 10, "Scene Complexity Extractor", "Laplacian ($V_{tex}$), Canny ($E_{edge}$)\nMacroblock Motion ($M_{mot}$), Blur ($B_{blur}$)", bg="#e8f5e9", border="#2e7d32")
# Right: Telemetry Monitor
draw_box(53, 68, 42, 10, "Device Telemetry Monitor", "Daemon Polling: CPU, GPU, VRAM\nCore Temp ($T$), Battery ($B$), AC State", bg="#fff3e0", border="#ef6c00")

# Split arrows from Frame Buffer
ax.annotate("", xy=(26, 78), xytext=(40, 84), arrowprops=arrow_props)
ax.annotate("", xy=(74, 78), xytext=(60, 84), arrowprops=arrow_props)

# 4. Hard Feasibility Gate
draw_box(15, 52, 70, 9, "Hard Feasibility Gate & Quality Constraints", "Device Capability, CUDA/GPU Safety, $r > 0$\nConservative Non-Contradictory Quality Tier", bg="#fce4ec", border="#c2185b")
ax.annotate("", xy=(35, 61), xytext=(26, 68), arrowprops=arrow_props)
ax.annotate("", xy=(65, 61), xytext=(74, 68), arrowprops=arrow_props)

# 5. Mamdani Fuzzy Inference Engine
draw_box(15, 36, 70, 10, "5-Variable Mamdani Fuzzy Decision Engine", "Inputs: $r$ (Real-time), $b$ (Bandwidth), $q$ (Quality), $c$ (Headroom), $n$ (Network)\nFuzzification $\\to$ 10-Rule Inference $\\to$ Centroid Defuzzification ($S^*$)", bg="#e0f7fa", border="#00838f")
ax.annotate("", xy=(50, 46), xytext=(50, 52), arrowprops=arrow_props)

# 6. Model Dispatch Tier (3 Branches)
# Branch 1: Zero-Power Bypass
draw_box(4, 18, 28, 11, "Zero-Power Bypass", "Critical Battery ($B < 10\\%$)\nLow Complexity ($C_t < 0.15$)\n0.0 ms Execution", bg="#f5f5f5", border="#757575")
# Branch 2: TinySR (FSRCNN)
draw_box(36, 18, 28, 11, "TinySR (FSRCNN)", "Compact CNN (~100 KB)\nSub-3 ms Inference\nSimple/Smooth Frames", bg="#e8eaf6", border="#3949ab")
# Branch 3: Real-ESRGAN / Recurrent
draw_box(68, 18, 28, 11, "Real-ESRGAN / Recurrent", "Perceptual GAN / BasicVSR++\nDynamic Tiling (tile=400/0)\nHigh Complexity Frames", bg="#fbe9e7", border="#d84315")

# Arrows from Fuzzy Engine to Models
ax.annotate("", xy=(18, 29), xytext=(35, 36), arrowprops=arrow_props)
ax.annotate("", xy=(50, 29), xytext=(50, 36), arrowprops=arrow_props)
ax.annotate("", xy=(82, 29), xytext=(65, 36), arrowprops=arrow_props)

# 7. Output Assembler
draw_box(20, 2, 60, 8, "Synchronous Runtime Video Assembler", "Frame Ordering, Resolution Alignment, and Muxing", bg="#e8f5e9", border="#1b5e20")

# Arrows to Assembler
ax.annotate("", xy=(38, 10), xytext=(18, 18), arrowprops=arrow_props)
ax.annotate("", xy=(50, 10), xytext=(50, 18), arrowprops=arrow_props)
ax.annotate("", xy=(62, 10), xytext=(82, 18), arrowprops=arrow_props)

plt.tight_layout()
out_path = os.path.join(output_dir, "system_architecture.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Generated {out_path}")
