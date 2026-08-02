import cv2
import numpy as np

# Scale factor constants to normalize metrics into roughly [0, 1] before clamping
# Note: These are starting values and can be calibrated against real footage.
MOTION_SCALE_FACTOR = 4.0
TEXTURE_SCALE_FACTOR = 500.0
BLUR_SCALE_FACTOR = 300.0

def analyze_frame(frame: np.ndarray, prev_frame: np.ndarray | None) -> dict:
    """
    Analyzes visual characteristics of a frame.
    
    frame: current BGR frame (as read by OpenCV)
    prev_frame: previous BGR frame, or None for the first frame in a video
    
    Returns dict with keys: motion, texture, edges, blur_clarity (all floats, 0-1)
    This function is pure and has no persistent state.
    """
    # 1. Compute motion (0.0 if prev_frame is None)
    motion = 0.0
    if prev_frame is not None:
        # Check shapes match
        if frame.shape == prev_frame.shape:
            gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray_curr, gray_prev)
            raw_motion = float(diff.mean() / 255.0)
            motion = min(raw_motion * MOTION_SCALE_FACTOR, 1.0)
        else:
            # Fallback if shapes differ (e.g. video resize change)
            motion = 0.0

    # Convert current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Compute edge density using Canny
    edges_img = cv2.Canny(gray, threshold1=100, threshold2=200)
    edges = float((edges_img > 0).sum() / edges_img.size)

    # 3. Compute texture & blur clarity (reusing Laplacian variance)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    texture = min(laplacian_var / TEXTURE_SCALE_FACTOR, 1.0)
    blur_clarity = min(laplacian_var / BLUR_SCALE_FACTOR, 1.0)

    return {
        "motion": motion,
        "texture": texture,
        "edges": edges,
        "blur_clarity": blur_clarity
    }
