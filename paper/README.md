# AdaptiveSR IEEE Conference Research Paper

This directory contains the complete IEEE-format conference paper for the **AdaptiveSR** project.

## Directory Contents

- `AdaptiveSR_IEEE_Conference_Paper.tex`: Main research paper formatted according to IEEE conference standards (`IEEEtran.cls`), including a native TikZ system architecture flowchart.
- `AdaptiveSR_Paper_Preview.html`: Complete 6-page interactive browser preview with live MathJax rendering and one-click "Print / Save as PDF (6 Pages)" button.
- `references.bib`: Verified BibTeX bibliographic database containing 20 authentic peer-reviewed papers with real DOIs and IEEE Xplore / publisher URLs.
- `system_architecture.png`: Vector-style high-resolution system architecture flowchart.
- `generate_architecture_diagram.py`: Script to generate the architecture flowchart.
- `fuzzy_membership_plots.png`: Publication plot of Mamdani membership curves ($r, b, c$).
- `generate_fuzzy_plots.py`: Script to plot the fuzzy membership functions.
- `benchmark_latency_comparison.png`: Generated chart comparing execution latency across video categories evaluated across GPU platforms.
- `frame_latency_complexity.png`: Generated chart displaying frame-by-frame complexity tracking and inference latency transitions.
- `generate_paper_plots.py`: Python script parsing raw benchmark CSV data (`benchmark_results/*.csv`) and generating publication charts.

## Compilation Instructions

### 1. Instant Interactive Preview & PDF Export (Browser)
Open `AdaptiveSR_Paper_Preview.html` directly in any web browser (Chrome, Edge, Firefox). Click **Print / Save as PDF** in the top navigation bar to generate the 6-page publication document immediately.

### 2. Overleaf / Cloud LaTeX (Recommended)
1. Zip the contents of this `paper/` directory.
2. In Overleaf, click **New Project** $\rightarrow$ **Upload Project** and select the `.zip`.
3. Ensure the compiler is set to `pdfLaTeX` and click **Recompile**.

### 3. Local LaTeX Environment (TeX Live / MiKTeX)
Run the following standard sequence in your terminal:
```bash
pdflatex AdaptiveSR_IEEE_Conference_Paper.tex
bibtex AdaptiveSR_IEEE_Conference_Paper
pdflatex AdaptiveSR_IEEE_Conference_Paper.tex
pdflatex AdaptiveSR_IEEE_Conference_Paper.tex
```
