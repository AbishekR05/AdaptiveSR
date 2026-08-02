import os
import time
import pytest
import numpy as np
import cv2
from src.modules.device_monitor import DeviceMonitor
from src.modules.video_loader import VideoLoader
from src.modules.frame_extractor import FrameExtractor
from src.modules.encoder import VideoEncoder

@pytest.fixture
def dummy_video_path(tmp_path):
    video_file = tmp_path / "dummy.mp4"
    width, height, fps, num_frames = 320, 240, 30, 60
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_file), fourcc, fps, (width, height))
    for i in range(num_frames):
        # Draw a moving rectangle
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (i * 3, 50), (i * 3 + 40, 90), (0, 255, 0), -1)
        out.write(frame)
    out.release()
    
    return str(video_file)

def test_device_monitor():
    monitor = DeviceMonitor(poll_interval=0.1)
    monitor.start()
    time.sleep(0.3)
    
    state = monitor.get_state()
    assert state.cpu >= 0.0
    assert state.ram >= 0.0
    assert state.system_ram >= 0.0
    
    # Update FPS and ensure it registers
    monitor.update_fps(24.5)
    state = monitor.get_state()
    assert state.fps == 24.5
    
    monitor.stop()

def test_video_loader(dummy_video_path):
    loader = VideoLoader(dummy_video_path)
    meta = loader.get_metadata()
    
    assert meta["width"] == 320
    assert meta["height"] == 240
    assert meta["fps"] == 30.0
    assert meta["frame_count"] == 60
    assert meta["duration"] == 2.0

def test_frame_extractor(dummy_video_path):
    extractor = FrameExtractor(dummy_video_path)
    frames = list(extractor.extract())
    
    assert len(frames) == 60
    
    first_idx, first_ts, first_frame = frames[0]
    assert first_idx == 0
    assert first_ts == 0.0
    assert first_frame.shape == (240, 320, 3)

def test_video_encoder(dummy_video_path, tmp_path):
    output_path = str(tmp_path / "output.mp4")
    
    loader = VideoLoader(dummy_video_path)
    meta = loader.get_metadata()
    
    encoder = VideoEncoder(
        original_input_path=dummy_video_path,
        output_path=output_path,
        width=meta["width"],
        height=meta["height"],
        fps=meta["fps"],
        has_audio=meta["has_audio"]
    )
    
    extractor = FrameExtractor(dummy_video_path)
    for idx, ts, frame in extractor.extract():
        encoder.write_frame(frame)
    encoder.close()
    
    # Verify the output file exists and can be parsed
    assert os.path.exists(output_path)
    out_loader = VideoLoader(output_path)
    out_meta = out_loader.get_metadata()
    assert out_meta["width"] == meta["width"]
    assert out_meta["height"] == meta["height"]
    assert out_meta["frame_count"] == meta["frame_count"]

@pytest.fixture
def short_dummy_video_path(tmp_path):
    video_file = tmp_path / "short_dummy.mp4"
    width, height, fps, num_frames = 320, 240, 30, 5
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_file), fourcc, fps, (width, height))
    for i in range(num_frames):
        # Draw a moving rectangle
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (i * 3, 50), (i * 3 + 40, 90), (0, 255, 0), -1)
        out.write(frame)
    out.release()
    
    return str(video_file)

def test_complete_pipeline_run(short_dummy_video_path, tmp_path):
    output_path = str(tmp_path / "enhanced_output.mp4")
    log_path = str(tmp_path / "pipeline_run.csv")
    
    from src.main import run_pipeline
    import csv
    
    # Run the complete pipeline
    run_pipeline(
        input_path=short_dummy_video_path,
        output_path=output_path,
        config_path="configs/decision_config.yaml",
        log_path=log_path,
        poll_interval=0.1
    )
    
    # Verify the output video exists and is upscaled 2x
    assert os.path.exists(output_path)
    out_loader = VideoLoader(output_path)
    out_meta = out_loader.get_metadata()
    assert out_meta["width"] == 640   # 320 * 2
    assert out_meta["height"] == 480  # 240 * 2
    assert out_meta["frame_count"] == 5
    
    # Verify the CSV log file
    assert os.path.exists(log_path)
    with open(log_path, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        
    # Check headers
    assert reader[0] == [
        "frame_no", "timestamp", "selected_model",
        "complexity_score", "cpu", "gpu", "ram", "battery",
        "temperature", "inference_time_ms", "decision_reason"
    ]
    # Check row count matches 5 frames + 1 header row
    assert len(reader) == 6
    
    # Check that model name is logged in every row
    for row in reader[1:]:
        assert row[2] in ["tinysr", "real_esrgan"]
        assert float(row[9]) >= 0.0  # inference_time_ms is populated

def test_pipeline_force_model(short_dummy_video_path, tmp_path):
    output_path = str(tmp_path / "forced_output.mp4")
    log_path = str(tmp_path / "forced_run.csv")
    
    from src.main import run_pipeline
    import csv
    
    # Run pipeline with force_model="tinysr"
    run_pipeline(
        input_path=short_dummy_video_path,
        output_path=output_path,
        config_path="configs/decision_config.yaml",
        log_path=log_path,
        poll_interval=0.1,
        force_model="tinysr"
    )
    
    assert os.path.exists(output_path)
    assert os.path.exists(log_path)
    
    with open(log_path, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        
    # Check that model is forced to tinysr for all 5 frames
    for row in reader[1:]:
        assert row[2] == "tinysr"
        assert "forced baseline model" in row[10]

def test_pipeline_ablation():
    from src.modules.decision_engine import DecisionEngine
    from src.utils.state_types import DeviceState, SceneDescriptor
    
    # Under low battery (0.05) and high temp (0.90) and high complexity (0.95), normal decision should be tinysr
    dev = DeviceState(cpu=0.1, gpu=0.1, ram=0.5, system_ram=0.5, battery=0.05, charging=False, temperature=0.90, fps=30.0)
    scene = SceneDescriptor(motion=0.0, texture=0.95, edges=0.95, blur_clarity=0.95, complexity=0.95)
    
    # 1. Normal behavior (forced tinysr due to device state constraint)
    engine_normal = DecisionEngine(ignore_device=False)
    dec_normal = engine_normal.decide(dev, scene)
    assert dec_normal.model == "tinysr"
    
    # 2. Ablation: ignore_device is True (should ignore the low battery/temp state and decide real_esrgan based on high complexity)
    engine_ablation = DecisionEngine(ignore_device=True)
    dec_ablation = engine_ablation.decide(dev, scene)
    assert dec_ablation.model == "real_esrgan"

def test_pipeline_int8_backend():
    from src.modules.backends.fsrcnn_backend_int8 import infer, load_model
    import numpy as np
    
    # Load session
    session = load_model(scale=2)
    assert session is not None
    
    # Run dynamic INT8 inference
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[25:75, 25:75] = (255, 255, 255)  # draw white square
    
    out = infer(frame, scale=2)
    assert out.shape == (200, 200, 3)

def test_decision_scale_reduction():
    from src.modules.decision_engine import DecisionEngine
    from src.utils.state_types import DeviceState, SceneDescriptor
    
    # Under low battery (0.25) but moderate complexity (0.80), selected model is real_esrgan, but scale should be restricted to 2
    dev = DeviceState(cpu=0.1, gpu=0.1, ram=0.5, system_ram=0.5, battery=0.25, charging=False, temperature=0.30, fps=30.0)
    scene = SceneDescriptor(motion=0.0, texture=0.80, edges=0.80, blur_clarity=0.80, complexity=0.80)
    
    engine = DecisionEngine()
    dec = engine.decide(dev, scene)
    assert dec.model == "real_esrgan"
    assert dec.scale == 2

def test_realesrgan_adaptive_tiling():
    from src.modules.backends.realesrgan_backend import infer, load_model
    from src.utils.state_types import DeviceState
    import numpy as np
    
    # Initialize and mock high GPU load
    dev_busy = DeviceState(cpu=0.1, gpu=0.85, ram=0.5, system_ram=0.5, battery=0.90, charging=True, temperature=0.30, fps=30.0)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Check that high GPU workload dynamically selects tile = 400
    infer(frame, device="cuda", scale=2, device_state=dev_busy)
    upsampler = load_model("cuda", scale=2)
    assert upsampler.tile == 400
    
    # Check that low GPU workload dynamically selects tile = 0 (fastest)
    dev_idle = DeviceState(cpu=0.1, gpu=0.10, ram=0.5, system_ram=0.5, battery=0.90, charging=True, temperature=0.30, fps=30.0)
    infer(frame, device="cuda", scale=2, device_state=dev_idle)
    assert upsampler.tile == 0




