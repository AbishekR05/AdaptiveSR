import os
import subprocess
import json
import logging
import cv2

logger = logging.getLogger("AdaptiveSR.video_loader")

class VideoLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found at: {file_path}")
        self.metadata = self._extract_metadata()

    def _extract_metadata(self):
        # Default empty metadata dict
        meta = {
            "width": 0,
            "height": 0,
            "fps": 0.0,
            "frame_count": 0,
            "duration": 0.0,
            "has_audio": False,
            "codec": "",
        }

        # 1. Try ffprobe for complete metadata including audio
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                self.file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            # Check format info
            fmt = info.get("format", {})
            meta["duration"] = float(fmt.get("duration", 0.0))
            
            # Check stream info
            streams = info.get("streams", [])
            for stream in streams:
                codec_type = stream.get("codec_type")
                if codec_type == "video":
                    meta["width"] = int(stream.get("width", 0))
                    meta["height"] = int(stream.get("height", 0))
                    meta["codec"] = stream.get("codec_name", "")
                    
                    # Parse FPS
                    avg_frame_rate = stream.get("avg_frame_rate", "0/0")
                    if "/" in avg_frame_rate:
                        num, den = map(int, avg_frame_rate.split("/"))
                        meta["fps"] = float(num) / float(den) if den > 0 else 0.0
                    else:
                        meta["fps"] = float(avg_frame_rate)
                        
                    # Frame count
                    nb_frames = stream.get("nb_frames")
                    if nb_frames:
                        meta["frame_count"] = int(nb_frames)
                elif codec_type == "audio":
                    meta["has_audio"] = True
        except Exception as e:
            logger.warning(f"ffprobe failed or was not found. Falling back to OpenCV. Error: {e}")

        # 2. Use OpenCV to fill in/verify basic properties
        try:
            cap = cv2.VideoCapture(self.file_path)
            if not cap.isOpened():
                raise IOError(f"OpenCV could not open video: {self.file_path}")
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # OpenCV values take precedence for frame extraction compatibility
            if width > 0: meta["width"] = width
            if height > 0: meta["height"] = height
            if fps > 0: meta["fps"] = fps
            if frame_count > 0: meta["frame_count"] = frame_count
            
            if meta["duration"] <= 0.0 and meta["fps"] > 0:
                meta["duration"] = meta["frame_count"] / meta["fps"]
                
            cap.release()
        except Exception as e:
            logger.error(f"OpenCV metadata extraction failed: {e}")
            if meta["width"] == 0 or meta["height"] == 0:
                raise e # Propagate if we couldn't get size

        logger.info(f"Loaded video {self.file_path}: {meta['width']}x{meta['height']} @ {meta['fps']:.2f} fps, "
                    f"{meta['frame_count']} frames, audio={meta['has_audio']}")
        return meta

    def get_metadata(self):
        return self.metadata
