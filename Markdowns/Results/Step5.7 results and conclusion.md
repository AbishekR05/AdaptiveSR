# Step 5.7 — FPS / Real-Time Feasibility Analysis: Results & Conclusions

**Freeze Status:** PIPELINE VALIDATED. Real-world quality evidence deferred pending genuine Layer B natural-video corpus (tracked as future work).

> [!IMPORTANT]
> **SR Inference-Only Feasibility Disclosure:**
> These results measure SR model inference latency only. Decoding, preprocessing, encoding, and network transfer are NOT included. End-to-end streaming real-time feasibility cannot be established from Step 5.5/5.7 data alone.

## 1. Executive Summary

**Fastest Real-Time Eligible Configuration:** None found.

## 2. Quantitative Benchmark Comparisons

| Model | Scale | Device | Latency (median, ms) | Est. FPS | Source FPS | Budget (ms) | Ratio | Real-Time (measured) | Decision-Eligible |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| tinysr | x2 | cpu | 644.47 | 1.55 | 30.0 | 33.33 | 0.05 | NO | NO |

## 3. Real-Time Feasibility Classifications

### 3.1 Measured + Decision-Eligible Configurations (Safe for Production)
- None.

### 3.2 Measured Feasible Only (Not Decision-Eligible due to high variance)
- None.

### 3.3 Failing Configurations (Unfeasible for Real-Time)
- `tinysr` x2 on `cpu` (Latency: **644.47 ms**)

## 4. Scale-Degradation Performance Trend

### tinysr on cpu
- **Scale x2**: Latency **644.47 ms** (1.55 FPS), UNFEASIBLE.

## 5. CPU vs. GPU Performance Comparison

| Model | Scale | CPU Threads | CPU Latency (ms) | GPU Latency (ms) | Ratio (CPU/GPU) | Feasibility Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
