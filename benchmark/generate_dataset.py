import cv2
import numpy as np
import os

def generate_video_pair(name, gt_w=1280, gt_h=960, lr_w=640, lr_h=480, fps=30, duration_seconds=2, category="mixed"):
    total_frames = int(fps * duration_seconds)
    os.makedirs("benchmark_data", exist_ok=True)
    
    gt_path = f"benchmark_data/{name}_gt.mp4"
    lr_path = f"benchmark_data/{name}_lr.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_gt = cv2.VideoWriter(gt_path, fourcc, fps, (gt_w, gt_h))
    out_lr = cv2.VideoWriter(lr_path, fourcc, fps, (lr_w, lr_h))
    
    for i in range(total_frames):
        # Create High-Res Frame
        frame_gt = np.zeros((gt_h, gt_w, 3), dtype=np.uint8)
        
        if category == "simple":
            # Very low complexity: flat grey background with a static gray square
            frame_gt[:, :] = (30, 30, 30)
            cv2.rectangle(frame_gt, (gt_w//2 - 50, gt_h//2 - 50), (gt_w//2 + 50, gt_h//2 + 50), (100, 100, 100), -1)
            
        elif category == "complex":
            # High complexity: checkerboard grid + high frequency concentric circles
            frame_gt[:, :] = (20, 20, 20)
            # Checkerboard grid
            grid_size = 40
            for y_grid in range(0, gt_h, grid_size):
                for x_grid in range(0, gt_w, grid_size):
                    if ((x_grid // grid_size) + (y_grid // grid_size)) % 2 == 0:
                        frame_gt[y_grid:y_grid+grid_size, x_grid:x_grid+grid_size] = (60, 60, 60)
            # Concentric circles
            for r in range(50, 400, 15):
                cv2.circle(frame_gt, (gt_w//2, gt_h//2), r, (180, 180, 180), 2)
            # Text lines
            cv2.putText(frame_gt, "COMPLEX STATIC GRID TEST", (100, 100), cv2.FONT_HERSHEY_TRIPLEX, 2.0, (255, 255, 255), 4)
            
        else:  # "mixed"
            # Bouncing ball over a checkerboard section (dynamic complexity)
            frame_gt[:, :] = (15, 15, 15)
            # Draw a checkerboard section in the middle
            cv2.rectangle(frame_gt, (300, 200), (980, 760), (40, 40, 40), -1)
            for y_grid in range(200, 760, 40):
                for x_grid in range(300, 980, 40):
                    if ((x_grid // 40) + (y_grid // 40)) % 2 == 0:
                        frame_gt[y_grid:y_grid+40, x_grid:x_grid+40] = (80, 80, 80)
            # Bouncing ball
            x = int(gt_w / 2 + (gt_w / 3) * np.sin(2 * np.pi * i / total_frames))
            y = int(gt_h / 2 + (gt_h / 3) * np.cos(2 * np.pi * i / total_frames))
            cv2.circle(frame_gt, (x, y), 60, (0, 0, 255), -1)
            # Text
            cv2.putText(frame_gt, f"Frame: {i}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
            
        # Write GT Frame
        out_gt.write(frame_gt)
        
        # Create LR Frame by downscaling GT using Area interpolation
        frame_lr = cv2.resize(frame_gt, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
        out_lr.write(frame_lr)
        
    out_gt.release()
    out_lr.release()
    print(f"Generated category '{category}': {gt_path} and {lr_path}")

if __name__ == "__main__":
    generate_video_pair("simple", category="simple")
    generate_video_pair("complex", category="complex")
    generate_video_pair("mixed", category="mixed")
