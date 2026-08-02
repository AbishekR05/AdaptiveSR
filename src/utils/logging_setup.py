import os
import logging
import csv
from datetime import datetime

def setup_logging(log_dir="logs", log_level=logging.INFO):
    """
    Sets up the standard application logger.
    Logs standard events to console and a text file.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("AdaptiveSR")
    logger.setLevel(log_level)
    
    # Clear existing handlers to prevent duplicate logs in some environments
    if logger.handlers:
        logger.handlers.clear()
        
    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, "app.log"), encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    logger.info("Logging initialized successfully.")
    return logger

class MetricsLogger:
    """
    Specialized logger for logging per-frame adaptive behaviour metrics.
    Writes structured fields into a CSV file.
    """
    def __init__(self, filepath="logs/metrics.csv"):
        self.filepath = filepath
        self.headers = [
            "frame_no", "timestamp", "selected_model", "complexity", 
            "cpu", "gpu", "ram", "battery", "temp", "inference_ms", "decision_reason"
        ]
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Initialize header if file is empty
        file_exists = os.path.exists(filepath)
        self.file = open(filepath, mode="a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        
        if not file_exists or os.path.getsize(filepath) == 0:
            self.writer.writerow(self.headers)
            self.file.flush()

    def log_frame(self, frame_no: int, selected_model: str, complexity: float,
                  cpu: float, gpu: float | None, ram: float, battery: float | None,
                  temp: float | None, inference_ms: float, decision_reason: str):
        """Logs a single row of execution telemetry to the metrics CSV."""
        timestamp = datetime.now().isoformat()
        gpu_str = f"{gpu:.2f}" if gpu is not None else ""
        batt_str = f"{battery:.2f}" if battery is not None else ""
        temp_str = f"{temp:.2f}" if temp is not None else ""
        
        row = [
            frame_no,
            timestamp,
            selected_model,
            f"{complexity:.4f}",
            f"{cpu:.4f}",
            gpu_str,
            f"{ram:.4f}",
            batt_str,
            temp_str,
            f"{inference_ms:.2f}",
            decision_reason
        ]
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.close()
