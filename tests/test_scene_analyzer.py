import cv2
import numpy as np
import pytest
from src.modules.scene_analyzer import analyze_frame
from src.modules.complexity_estimator import estimate_complexity

def test_determinism():
    frame = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    prev = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    
    res1 = analyze_frame(frame, prev)
    res2 = analyze_frame(frame, prev)
    
    assert res1 == res2
    assert estimate_complexity(res1) == estimate_complexity(res2)

def test_first_frame():
    frame = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    res = analyze_frame(frame, None)
    
    assert res["motion"] == 0.0
    assert not np.isnan(res["motion"])
    for key, val in res.items():
        assert 0.0 <= val <= 1.0

def test_frame_stability():
    frame1 = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(frame1, (50, 50), (100, 100), (255, 255, 255), -1)
    
    frame2 = frame1.copy()
    cv2.rectangle(frame2, (52, 50), (102, 100), (255, 255, 255), -1)
    
    res1 = analyze_frame(frame1, None)
    res2 = analyze_frame(frame2, frame1)
    
    comp1 = estimate_complexity(res1)
    comp2 = estimate_complexity(res2)
    
    # Delta should be small between adjacent frames with minimal motion
    assert abs(comp2 - comp1) < 0.2

def test_rank_sanity():
    # 1. Flat gray frame
    flat = np.ones((240, 320, 3), dtype=np.uint8) * 128
    
    # 2. Geometric details (grid pattern)
    geom = np.zeros((240, 320, 3), dtype=np.uint8)
    for i in range(0, 240, 20):
        geom[i:i+2, :] = 255
    for j in range(0, 320, 20):
        geom[:, j:j+2] = 255
        
    # 3. High-frequency random noise
    noise = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    
    res_flat = analyze_frame(flat, None)
    res_geom = analyze_frame(geom, None)
    res_noise = analyze_frame(noise, None)
    
    comp_flat = estimate_complexity(res_flat)
    comp_geom = estimate_complexity(res_geom)
    comp_noise = estimate_complexity(res_noise)
    
    # Flat complexity should be low, geom moderate, noise very high
    assert comp_flat < comp_geom
    assert comp_geom < comp_noise
