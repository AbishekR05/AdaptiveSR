import os
import argparse
import time
import logging
from src.utils.logging_setup import setup_logging
from src.modules.video_loader import VideoLoader
from src.modules.frame_extractor import FrameExtractor
from src.modules.encoder import VideoEncoder
from src.modules.device_monitor import DeviceMonitor
from src.modules.decision_engine import DecisionEngine
from src.modules.enhancement_engine import EnhancementEngine, get_inference_device
from src.modules.frame_buffer import FrameBuffer
from src.modules.pipeline_logger import PipelineLogger
from src.modules.scene_analyzer import analyze_frame
from src.modules.complexity_estimator import estimate_complexity
from src.utils.state_types import SceneDescriptor

logger = logging.getLogger("AdaptiveSR.main")

def run_pipeline(input_path, output_path, config_path="configs/decision_config.yaml", log_path=None, poll_interval=0.5, force_model=None):
    # 1. Setup metrics logger
    if log_path is None:
        os.makedirs("logs", exist_ok=True)
        timestamp = int(time.time())
        log_path = os.path.join("logs", f"run_{timestamp}.csv")
        
    pipeline_logger = PipelineLogger(log_path)
    
    # 2. Load original video
    logger.info(f"Loading input video from: {input_path}")
    loader = VideoLoader(input_path)
    meta = loader.get_metadata()
    logger.info(f"Video metadata: {meta['width']}x{meta['height']} @ {meta['fps']} FPS | Frames: {meta['frame_count']}")

    # 3. Initialize components
    device_monitor = DeviceMonitor(poll_interval=poll_interval)
    device_monitor.start()

    decision_engine = DecisionEngine(config_path)
    enhancement_engine = EnhancementEngine(device=get_inference_device())
    frame_buffer = FrameBuffer()

    prev_frame = None
    extractor = FrameExtractor(input_path)
    
    logger.info("Starting adaptive enhancement processing loop...")
    start_time = time.time()
    last_print_time = start_time
    total_frames = meta["frame_count"]

    encoder = None

    try:
        for idx, ts_ms, frame in extractor.extract():
            t_frame_start = time.time()

            # Analyze frame
            scene_metrics = analyze_frame(frame, prev_frame)
            complexity = estimate_complexity(scene_metrics)
            scene_descriptor = SceneDescriptor(
                motion=scene_metrics["motion"],
                texture=scene_metrics["texture"],
                edges=scene_metrics["edges"],
                blur_clarity=scene_metrics["blur_clarity"],
                complexity=complexity
            )

            # Get telemetry state
            device_state = device_monitor.get_state()

            # Update measured FPS in device monitor
            elapsed = time.time() - start_time
            current_fps = (idx + 1) / elapsed if elapsed > 0 else 0.0
            device_monitor.update_fps(current_fps)

            # Decide model
            if force_model is not None:
                from src.utils.state_types import Decision
                decision = Decision(model=force_model, scale=2, reason=f"forced baseline model: {force_model}")
            else:
                decision = decision_engine.decide(device_state, scene_descriptor)

            # Enhance frame
            t_infer_start = time.time()
            enhanced_frame = enhancement_engine.enhance(frame, decision, frame_window=None, device_state=device_state)
            inference_ms = (time.time() - t_infer_start) * 1000.0

            # Store in buffer
            frame_buffer.put(idx, enhanced_frame)

            # Log frame statistics
            pipeline_logger.log_row(
                frame_no=idx,
                timestamp=t_frame_start,
                decision=decision,
                scene=scene_descriptor,
                device=device_state,
                inference_ms=inference_ms
            )

            # Dynamically initialize the VideoEncoder using output shape of first frame
            if encoder is None:
                eh, ew, _ = enhanced_frame.shape
                logger.info(f"Initializing VideoEncoder with upscaled dimensions: {ew}x{eh}")
                encoder = VideoEncoder(
                    original_input_path=input_path,
                    output_path=output_path,
                    width=ew,
                    height=eh,
                    fps=meta["fps"],
                    has_audio=meta["has_audio"]
                )

            # Write frame
            encoder.write_frame(enhanced_frame)

            prev_frame = frame

            # Periodic console reporting (every 2.0s)
            now = time.time()
            if now - last_print_time >= 2.0:
                pct = (idx + 1) / total_frames * 100.0 if total_frames > 0 else 0.0
                gpu_str = f"{device_state.gpu*100:.1f}%" if device_state.gpu is not None else "N/A"
                logger.info(
                    f"Frame {idx+1}/{total_frames} ({pct:.1f}%) | "
                    f"Selected: {decision.model} | "
                    f"Inf: {inference_ms:.1f} ms | "
                    f"CPU: {device_state.cpu*100:.1f}% | "
                    f"GPU: {gpu_str} | "
                    f"RAM: {device_state.ram:.1f} MB"
                )
                last_print_time = now

    except Exception as e:
        logger.error(f"Error occurred during pipeline execution: {e}", exc_info=True)
        raise

    finally:
        logger.info("Cleaning up pipeline threads and writers...")
        device_monitor.stop()
        pipeline_logger.close()
        
        if encoder is not None:
            logger.info("Finalizing encoded output video...")
            encoder.close()
            
        total_time = time.time() - start_time
        logger.info(f"Pipeline processing complete in {total_time:.2f} seconds.")
        logger.info(f"Per-frame logs saved to: {log_path}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive Resource- and Content-Aware Edge Video Super-Resolution Framework - Complete Pipeline Integration"
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
        "--config", "-c",
        default="configs/decision_config.yaml",
        help="Path to decision engine yaml config file"
    )
    parser.add_argument(
        "--log", "-l",
        default=None,
        help="Path to save the per-frame CSV logs"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Device monitor polling interval in seconds"
    )
    parser.add_argument(
        "--force-model", "-f",
        default=None,
        choices=["tinysr", "real_esrgan", "skip"],
        help="Forcibly run a static model for benchmarking"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging()
    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        config_path=args.config,
        log_path=args.log,
        poll_interval=args.poll_interval,
        force_model=args.force_model
    )

if __name__ == "__main__":
    main()
