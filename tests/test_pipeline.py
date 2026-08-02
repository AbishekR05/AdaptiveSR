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
