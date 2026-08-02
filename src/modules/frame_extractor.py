import logging
import cv2

logger = logging.getLogger("AdaptiveSR.frame_extractor")

class FrameExtractor:
    def __init__(self, file_path):
        self.file_path = file_path

    def extract(self):
        """
        Yields frames sequentially as a generator to keep memory footprints low.
        Yields: (frame_idx, timestamp_ms, frame)
        """
        cap = cv2.VideoCapture(self.file_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video source for extraction: {self.file_path}")
            return
        
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get current timestamp in milliseconds
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                
                yield frame_idx, timestamp_ms, frame
                frame_idx += 1
        finally:
            cap.release()
            logger.info(f"Frame extraction completed. Total frames processed: {frame_idx}")
