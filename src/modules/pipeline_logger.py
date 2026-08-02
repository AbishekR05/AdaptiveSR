import csv

class PipelineLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self._file = open(log_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "frame_no", "timestamp", "selected_model",
            "complexity_score", "cpu", "gpu", "ram", "battery",
            "temperature", "inference_time_ms", "decision_reason"
        ])
        self._file.flush()

    def log_row(self, frame_no, timestamp, decision, scene, device, inference_ms):
        self._writer.writerow([
            frame_no,
            timestamp,
            decision.model,
            scene.complexity,
            device.cpu,
            device.gpu,
            device.ram,
            device.battery,
            device.temperature,
            inference_ms,
            decision.reason
        ])
        self._file.flush()  # flush per row to keep logs if long run aborts

    def close(self):
        if not self._file.closed:
            self._file.close()
