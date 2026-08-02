import os
import subprocess
import logging
import cv2

logger = logging.getLogger("AdaptiveSR.encoder")

class VideoEncoder:
    def __init__(self, original_input_path, output_path, width, height, fps, has_audio=False):
        """
        Initializes the VideoEncoder.
        width and height should be the dimensions of the final output video (e.g. after upscaling).
        """
        self.original_input_path = original_input_path
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.has_audio = has_audio
        
        # Temp path for video-only encoding
        self.temp_video_path = output_path + ".temp.mp4"
        self._process = None
        self._start_ffmpeg_process()

    def _start_ffmpeg_process(self):
        # Build ffmpeg command to receive raw BGR24 frames from stdin
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", f"{self.fps}",
            "-i", "-",  # Read from stdin
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-loglevel", "warning",
            self.temp_video_path
        ]
        
        logger.info(f"Starting FFmpeg process for target size {self.width}x{self.height}: {' '.join(cmd)}")
        self._process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write_frame(self, frame):
        """
        Write a frame (numpy array) to the FFmpeg process.
        If frame dimensions don't match target size, it will be resized.
        """
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("FFmpeg process is not running.")
            
        fh, fw, _ = frame.shape
        if fh != self.height or fw != self.width:
            frame = cv2.resize(frame, (self.width, self.height))
            
        self._process.stdin.write(frame.tobytes())

    def close(self):
        """
        Closes the raw video writer process and merges audio if available.
        """
        if self._process:
            if self._process.stdin:
                self._process.stdin.close()
            self._process.wait()
            logger.info("Raw video encoding process completed.")
            
        # Merge audio from the original video
        if self.has_audio and os.path.exists(self.original_input_path):
            logger.info("Merging audio track from original video...")
            cmd = [
                "ffmpeg",
                "-y",
                "-i", self.temp_video_path,
                "-i", self.original_input_path,
                "-map", "0:v",
                "-map", "1:a",
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                "-loglevel", "warning",
                self.output_path
            ]
            try:
                subprocess.run(cmd, check=True)
                logger.info(f"Final video with audio created at: {self.output_path}")
                # Remove temporary video
                if os.path.exists(self.temp_video_path):
                    os.remove(self.temp_video_path)
            except Exception as e:
                logger.error(f"Failed to merge audio: {e}. Keeping video-only file at {self.temp_video_path}")
                # Rename the temp video to output_path if merging failed
                if os.path.exists(self.temp_video_path):
                    if os.path.exists(self.output_path):
                        os.remove(self.output_path)
                    os.rename(self.temp_video_path, self.output_path)
        else:
            logger.info("No audio stream or original input to merge. Finalizing video-only output.")
            if os.path.exists(self.temp_video_path):
                if os.path.exists(self.output_path):
                    os.remove(self.output_path)
                os.rename(self.temp_video_path, self.output_path)
            logger.info(f"Final video created at: {self.output_path}")
