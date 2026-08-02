import os
import argparse
import time
import logging
from src.utils.logging_setup import setup_logging, MetricsLogger
from src.modules.video_loader import VideoLoader
from src.modules.frame_extractor import FrameExtractor
from src.modules.encoder import VideoEncoder
from src.modules.device_monitor import DeviceMonitor

def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive Resource- and Content-Aware Edge Video Super-Resolution Framework - Phase 1 Passthrough Validation"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input video file"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to save the output video file"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Device monitor polling interval in seconds"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Setup standard logging
    logger = setup_logging()
    logger.info("Initializing AdaptiveSR Passthrough Pipeline...")
    
    # 2. Setup metrics CSV logging
    metrics_csv_path = os.path.join("logs", "metrics.csv")
    metrics_logger = MetricsLogger(filepath=metrics_csv_path)
    
    # 3. Load Video Metadata
    try:
        video_loader = VideoLoader(args.input)
    except Exception as e:
        logger.error(f"Failed to load input video: {e}")
        return
        
    meta = video_loader.get_metadata()
    
    # 4. Start Device Monitor
    monitor = DeviceMonitor(poll_interval=args.poll_interval)
    monitor.start()
    
    # 5. Initialize Encoder
    encoder = VideoEncoder(
        original_input_path=args.input,
        output_path=args.output,
        width=meta["width"],
        height=meta["height"],
        fps=meta["fps"],
        has_audio=meta["has_audio"]
    )
    
    # 6. Process Frames
    extractor = FrameExtractor(args.input)
    
    total_frames = meta["frame_count"]
    logger.info("Starting frame processing...")
    
    start_time = time.time()
    last_print_time = start_time
    
    try:
        for idx, ts_ms, frame in extractor.extract():
            frame_start = time.time()
            
            # For Phase 1 (Passthrough), write frame directly to encoder
            encoder.write_frame(frame)
            
            frame_end = time.time()
            frame_duration_ms = (frame_end - frame_start) * 1000.0
            
            # Fetch current device state
            dev_state = monitor.get_state()
            
            # Update measured FPS in device monitor
            elapsed = time.time() - start_time
            current_fps = (idx + 1) / elapsed if elapsed > 0 else 0.0
            monitor.update_fps(current_fps)
            
            # Log telemetry for this frame to CSV
            metrics_logger.log_frame(
                frame_no=idx,
                selected_model="passthrough",
                complexity=0.0,
                cpu=dev_state.cpu,
                gpu=dev_state.gpu,
                ram=dev_state.ram,
                system_ram=dev_state.system_ram,
                battery=dev_state.battery,
                temp=dev_state.temperature,
                inference_ms=frame_duration_ms,
                decision_reason="Passthrough pipeline validation"
            )
            
            # Periodically print progress & device stats (every 2 seconds)
            now = time.time()
            if now - last_print_time >= 2.0:
                pct = (idx + 1) / total_frames * 100.0 if total_frames > 0 else 0.0
                gpu_str = f"{dev_state.gpu*100:.1f}%" if dev_state.gpu is not None else "N/A"
                batt_str = f"{dev_state.battery*100:.1f}%" if dev_state.battery is not None else "N/A"
                logger.info(
                    f"Progress: {idx+1}/{total_frames} frames ({pct:.1f}%) | "
                    f"Processing FPS: {current_fps:.2f} | "
                    f"CPU: {dev_state.cpu*100:.1f}% | "
                    f"GPU: {gpu_str} | "
                    f"Proc RAM: {dev_state.ram:.1f} MB (Sys: {dev_state.system_ram*100:.1f}%) | "
                    f"Battery: {batt_str}"
                )
                last_print_time = now
                
    except Exception as e:
        logger.error(f"Error encountered during frame processing: {e}")
        
    finally:
        # 7. Cleanup & finalize
        logger.info("Closing video encoder...")
        encoder.close()
        
        logger.info("Stopping device monitor...")
        monitor.stop()
        
        metrics_logger.close()
        
        total_time = time.time() - start_time
        logger.info(f"Pipeline finished in {total_time:.2f} seconds.")
        logger.info(f"Telemetry metrics saved to: {metrics_csv_path}")

if __name__ == "__main__":
    main()
