# Benchmark Evaluation Summary Report

This report summarizes the experimental evaluation of the AdaptiveSR pipeline across three categories (simple, complex, mixed) comparing forced static baselines with the dynamic adaptive configuration.

## 1. Visual Quality Comparison
| Category | Configuration | Average PSNR | Average SSIM | Average LPIPS |
| :--- | :--- | :--- | :--- | :--- |
| simple | baseline_tinysr | 20.55 | 0.2194 | 0.1783 |
| simple | baseline_real_esrgan | 51.59 | 0.9991 | 0.0053 |
| simple | adaptive | 20.55 | 0.2194 | 0.1783 |
| complex | baseline_tinysr | 21.59 | 0.6061 | 0.1743 |
| complex | baseline_real_esrgan | 24.19 | 0.9023 | 0.1588 |
| complex | adaptive | 24.19 | 0.9023 | 0.1588 |
| mixed | baseline_tinysr | 24.98 | 0.2846 | 0.0921 |
| mixed | baseline_real_esrgan | 33.53 | 0.8459 | 0.0806 |
| mixed | adaptive | 33.39 | 0.8367 | 0.0846 |

## 2. System Telemetry Comparison
| Category | Configuration | Total Latency | Avg CPU | Avg GPU | Battery Delta | Avg Temp | Switch Rate | Model Distribution |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| simple | baseline_tinysr | 3.40s | 69.4% | 26.3% | 0.0% | 52.9 C | 0.0% | tinysr: 100.0% |
| simple | baseline_real_esrgan | 70.41s | 25.5% | nan% | -1.0% | - | 0.0% | real_esrgan: 100.0% |
| simple | adaptive | 2.82s | 75.2% | nan% | 0.0% | - | 0.0% | tinysr: 100.0% |
| complex | baseline_tinysr | 2.69s | 71.8% | nan% | 0.0% | - | 0.0% | tinysr: 100.0% |
| complex | baseline_real_esrgan | 67.67s | 40.7% | nan% | 0.0% | - | 0.0% | real_esrgan: 100.0% |
| complex | adaptive | 67.85s | 36.5% | nan% | -2.0% | - | 0.0% | real_esrgan: 100.0% |
| mixed | baseline_tinysr | 2.85s | 73.4% | nan% | 0.0% | - | 0.0% | tinysr: 100.0% |
| mixed | baseline_real_esrgan | 67.44s | 31.1% | nan% | -1.0% | - | 0.0% | real_esrgan: 100.0% |
| mixed | adaptive | 66.37s | 35.5% | nan% | 0.0% | - | 3.3% | real_esrgan: 98.3%, tinysr: 1.7% |