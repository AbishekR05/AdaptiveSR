import os
import threading
import time
import logging
import psutil
from src.utils.state_types import DeviceState

logger = logging.getLogger("AdaptiveSR.device_monitor")

# Attempt NVML initialization for NVIDIA GPUs
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
    logger.info("NVIDIA NVML initialized successfully. GPU monitoring enabled.")
except (ImportError, Exception) as e:
    HAS_NVML = False
    logger.warning(f"NVIDIA NVML initialization failed: {e}. GPU monitoring will be disabled.")

class DeviceMonitor:
    def __init__(self, poll_interval=0.5):
        self.poll_interval = poll_interval
        self._process = psutil.Process(os.getpid())
        self._state = DeviceState(
            cpu=0.0, 
            gpu=None, 
            ram=0.0, 
            system_ram=0.0,
            battery=None,
            charging=None, 
            temperature=None, 
            fps=0.0
        )
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        """Starts the device monitor background polling thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Device monitor thread started.")

    def stop(self):
        """Stops the device monitor background polling thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Device monitor thread stopped.")

    def update_fps(self, fps: float):
        """Updates the current measured frame processing speed (FPS) thread-safely."""
        with self._lock:
            self._state.fps = fps

    def _loop(self):
        while self._running:
            try:
                # 1. CPU utilization (non-blocking call, first call might be 0 but subsequent calls are accurate)
                cpu = psutil.cpu_percent() / 100.0
                
                # 2. RAM utilization (process RSS in MB, and system-wide %)
                try:
                    ram = self._process.memory_info().rss / (1024.0 * 1024.0)
                except Exception:
                    ram = 0.0
                system_ram = psutil.virtual_memory().percent / 100.0
                
                # 3. Battery status
                batt = psutil.sensors_battery()
                battery = batt.percent / 100.0 if batt else None
                charging = batt.power_plugged if batt else None
                
                # 4. GPU + GPU Temp monitoring (NVIDIA)
                gpu = None
                gpu_temp = None
                if HAS_NVML:
                    try:
                        # Assuming index 0 (primary GPU)
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu = rates.gpu / 100.0
                        
                        # Get GPU temperature in C and normalize (0-1, assuming max 100C)
                        temp_c = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        gpu_temp = min(max(temp_c / 100.0, 0.0), 1.0)
                    except Exception as ex:
                        logger.debug(f"Failed to query NVIDIA GPU stats: {ex}")
                
                # 5. System Temperature fallback
                sys_temp = None
                # psutil.sensors_temperatures() typically works on Linux but not Windows
                try:
                    temps = psutil.sensors_temperatures()
                    if temps:
                        # Find CPU core/package temp
                        for key, entries in temps.items():
                            if entries:
                                # Standardize to 0-1 range assuming 100C max
                                sys_temp = min(max(entries[0].current / 100.0, 0.0), 1.0)
                                break
                except Exception:
                    pass
                
                # If we have GPU temp, use it; otherwise fallback to system temperature if available
                final_temp = gpu_temp if gpu_temp is not None else sys_temp
                
                with self._lock:
                    self._state = DeviceState(
                        cpu=cpu,
                        gpu=gpu,
                        ram=ram,
                        system_ram=system_ram,
                        battery=battery,
                        charging=charging,
                        temperature=final_temp,
                        fps=self._state.fps
                    )
            except Exception as e:
                logger.error(f"Error in device monitoring loop: {e}")
                
            time.sleep(self.poll_interval)

    def get_state(self) -> DeviceState:
        """Returns a snapshot of the current device state thread-safely."""
        with self._lock:
            # Return a copy/snapshot
            s = self._state
            return DeviceState(
                cpu=s.cpu,
                gpu=s.gpu,
                ram=s.ram,
                system_ram=s.system_ram,
                battery=s.battery,
                charging=s.charging,
                temperature=s.temperature,
                fps=s.fps
            )

    def __del__(self):
        # Ensure NVML shutdown
        if HAS_NVML:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
