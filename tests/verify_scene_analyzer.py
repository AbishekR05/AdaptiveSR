import cv2
import numpy as np
from src.modules.scene_analyzer import analyze_frame
from src.modules.complexity_estimator import estimate_complexity

def create_synthetic_frames():
    """Generates 5 synthetic frames representing different complexity categories."""
    frames = {}
    width, height = 640, 480

    # 1. Flat sky / blank wall (Solid gray)
    flat = np.ones((height, width, 3), dtype=np.uint8) * 128
    frames["Flat sky / blank wall"] = flat

    # 2. Landscape (moderate detail)
    landscape = np.zeros((height, width, 3), dtype=np.uint8)
    # Background gradient
    for y in range(height):
        landscape[y, :, 0] = int(100 + y * 100 / height) # Blue-ish gradient
        landscape[y, :, 1] = int(150 + y * 50 / height)
        landscape[y, :, 2] = int(200 + y * 20 / height)
    # Green mountain
    pts = np.array([[100, height], [320, 180], [540, height]], np.int32)
    cv2.fillPoly(landscape, [pts], (50, 180, 50))
    # Sun
    cv2.circle(landscape, (500, 120), 40, (0, 220, 220), -1)
    frames["Landscape (moderate detail)"] = landscape

    # 3. Close-up face (geometric mock face)
    face = np.ones((height, width, 3), dtype=np.uint8) * 200
    # Head oval
    cv2.ellipse(face, (320, 240), (140, 180), 0, 0, 360, (180, 180, 220), -1)
    # Eyes
    cv2.circle(face, (260, 200), 20, (255, 255, 255), -1)
    cv2.circle(face, (260, 200), 8, (40, 40, 40), -1)
    cv2.circle(face, (380, 200), 20, (255, 255, 255), -1)
    cv2.circle(face, (380, 200), 8, (40, 40, 40), -1)
    # Smile
    cv2.ellipse(face, (320, 320), (50, 20), 0, 0, 180, (40, 40, 180), 3)
    frames["Close-up face"] = face

    # 4. Moderately busy scene (checkerboard, shapes and text)
    busy = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(0, height, 80):
        for j in range(0, width, 80):
            if (i + j) % 160 == 0:
                cv2.rectangle(busy, (j, i), (j+80, i+80), (100, 100, 100), -1)
    cv2.putText(busy, "Adaptive Edge SR Project", (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.circle(busy, (150, 100), 60, (0, 0, 255), 3)
    cv2.rectangle(busy, (400, 80), (550, 220), (0, 255, 0), 4)
    frames["Moderately busy scene"] = busy

    # 5. Crowded street / high details (dense grid + heavy random noise)
    noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    for i in range(0, height, 12):
        cv2.line(noise, (0, i), (width, i), (255, 255, 255), 1)
    for j in range(0, width, 12):
        cv2.line(noise, (j, 0), (j, height), (255, 255, 255), 1)
    frames["Crowded street / high details"] = noise

    return frames

def main():
    print("=== Phase 3 Validation: Scene Analyzer & Complexity Estimator ===")
    
    # Generate test frames
    frames = create_synthetic_frames()
    
    # 1. Ranking sanity test
    results = {}
    print("\n--- Test 1: Ranking Sanity Test ---")
    print(f"{'Frame Description':<30} | {'Motion':<6} | {'Texture':<7} | {'Edges':<5} | {'Clarity':<7} | {'Complexity':<10}")
    print("-" * 80)
    
    for desc, img in frames.items():
        metrics = analyze_frame(img, prev_frame=None)
        comp = estimate_complexity(metrics)
        results[desc] = {**metrics, "complexity": comp}
        
        print(f"{desc:<30} | "
              f"{metrics['motion']:<6.2f} | "
              f"{metrics['texture']:<7.4f} | "
              f"{metrics['edges']:<5.4f} | "
              f"{metrics['blur_clarity']:<7.4f} | "
              f"{comp:<10.4f}")
              
    # Verify monotonic ranking order: Flat < Landscape < Face < Busy < Noise
    order = [
        "Flat sky / blank wall",
        "Landscape (moderate detail)",
        "Close-up face",
        "Moderately busy scene",
        "Crowded street / high details"
    ]
    
    monotonic = True
    for i in range(len(order) - 1):
        c1 = results[order[i]]["complexity"]
        c2 = results[order[i+1]]["complexity"]
        if c1 >= c2:
            monotonic = False
            
    print(f"\nMonotonicity Check: {'PASSED' if monotonic else 'FAILED'}")
    
    # 2. Determinism test
    print("\n--- Test 2: Determinism Test ---")
    test_img = frames["Close-up face"]
    metrics_run1 = analyze_frame(test_img, None)
    metrics_run2 = analyze_frame(test_img, None)
    
    comp_run1 = estimate_complexity(metrics_run1)
    comp_run2 = estimate_complexity(metrics_run2)
    
    print(f"Run 1 Complexity: {comp_run1:.6f}")
    print(f"Run 2 Complexity: {comp_run2:.6f}")
    determinism_passed = (metrics_run1 == metrics_run2) and (comp_run1 == comp_run2)
    print(f"Determinism Check: {'PASSED' if determinism_passed else 'FAILED'}")

    # 3. Stability test
    print("\n--- Test 3: Frame-to-Frame Stability Test ---")
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame1, (320, 240), 100, (255, 0, 0), -1)
    
    # Frame 2 has the circle shifted slightly (1 pixel) to simulate camera drift
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame2, (321, 240), 100, (255, 0, 0), -1)
    
    m1 = analyze_frame(frame1, prev_frame=None)
    m2 = analyze_frame(frame2, prev_frame=frame1)
    
    c1 = estimate_complexity(m1)
    c2 = estimate_complexity(m2)
    
    delta = abs(c2 - c1)
    print(f"Frame 1 Complexity: {c1:.4f}")
    print(f"Frame 2 (shifted) Complexity: {c2:.4f}")
    print(f"Frame-to-frame delta (|c2 - c1|): {delta:.4f}")
    stability_passed = delta < 0.05
    print(f"Stability Check: {'PASSED' if stability_passed else 'FAILED'}")
    
    # 4. First-frame edge case test
    print("\n--- Test 4: First-Frame Edge Case Test ---")
    first_frame_metrics = analyze_frame(frames["Landscape (moderate detail)"], prev_frame=None)
    first_frame_passed = (first_frame_metrics["motion"] == 0.0)
    print(f"First Frame Motion: {first_frame_metrics['motion']:.2f}")
    print(f"First Frame Edge Case Check: {'PASSED' if first_frame_passed else 'FAILED'}")
    
    # 5. Motion Sensitivity test
    print("\n--- Test 5: Motion Sensitivity Test ---")
    base_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(base_frame, (100, 100), (300, 300), (0, 0, 255), -1)
    
    small_move = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(small_move, (105, 100), (305, 300), (0, 0, 255), -1)
    
    medium_move = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(medium_move, (130, 100), (330, 300), (0, 0, 255), -1)
    
    large_move = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(large_move, (220, 100), (420, 300), (0, 0, 255), -1)
    
    m_small = analyze_frame(small_move, base_frame)["motion"]
    m_medium = analyze_frame(medium_move, base_frame)["motion"]
    m_large = analyze_frame(large_move, base_frame)["motion"]
    
    print(f"Small Shift (5px) Motion: {m_small:.4f}")
    print(f"Medium Shift (30px) Motion: {m_medium:.4f}")
    print(f"Large Shift (120px) Motion: {m_large:.4f}")
    
    motion_sensitivity_passed = (0.0 < m_small < m_medium < m_large <= 1.0)
    print(f"Motion Sensitivity Check: {'PASSED' if motion_sensitivity_passed else 'FAILED'}")
    
    overall_success = (monotonic and determinism_passed and stability_passed and 
                       first_frame_passed and motion_sensitivity_passed)
    print(f"\nOverall Phase 3 Success: {'[SUCCESS]' if overall_success else '[FAILED]'}")

if __name__ == "__main__":
    main()
