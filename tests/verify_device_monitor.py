import time
import csv
import os
import gc
import threading
import numpy as np
from src.modules.device_monitor import DeviceMonitor

# Flag to control CPU load generation
cpu_load_active = False

def cpu_worker():
    global cpu_load_active
    while cpu_load_active:
        # Perform matrix multiplication to stress the CPU
        a = np.random.rand(600, 600)
        b = np.random.rand(600, 600)
        np.dot(a, b)

def cpu_load_generator():
    global cpu_load_active
    print("[Load Gen] Spawning CPU stress threads...")
    threads = []
    for _ in range(6):  # stress 6 cores/threads
        t = threading.Thread(target=cpu_worker, daemon=True)
        t.start()
        threads.append(t)
    
    while cpu_load_active:
        time.sleep(0.1)
        
    for t in threads:
        t.join()
    print("[Load Gen] CPU load threads stopped.")

def main():
    global cpu_load_active
    print("=== Phase 2 Validation: DeviceMonitor Verification ===")
    
    # 1. Initialize DeviceMonitor
    poll_interval = 0.5
    monitor = DeviceMonitor(poll_interval=poll_interval)
    monitor.start()
    
    # Prepare CSV logging
    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/device_monitor_validation.csv"
    
    headers = ["timestamp", "cpu", "gpu", "proc_ram_mb", "system_ram", "battery", "charging"]
    samples = []
    
    # We will log for 12 seconds total
    start_time = time.time()
    duration = 12.0
    
    # Memory allocation placeholder
    memory_bloat_holder = None
    
    print(f"Running monitoring loop for {duration}s. Sampling every {poll_interval}s...")
    
    # Helper to append current sample
    def record_sample(phase_name):
        state = monitor.get_state()
        ts = time.time() - start_time
        samples.append({
            "timestamp": f"{ts:.2f}",
            "cpu": f"{state.cpu * 100:.1f}%",
            "gpu": f"{state.gpu * 100:.1f}%" if state.gpu is not None else "N/A",
            "proc_ram_mb": f"{state.ram:.2f} MB",
            "system_ram": f"{state.system_ram * 100:.1f}%",
            "battery": f"{state.battery * 100:.1f}%" if state.battery is not None else "N/A",
            "charging": str(state.charging) if state.charging is not None else "N/A",
            "phase": phase_name
        })
        print(
            f"[{phase_name}] Time: {ts:.2f}s | CPU: {state.cpu*100:.1f}% | "
            f"Proc RAM: {state.ram:.1f} MB | Sys RAM: {state.system_ram*100:.1f}%"
        )
    
    next_sample_time = start_time
    load_thread = None
    
    try:
        while True:
            now = time.time()
            elapsed = now - start_time
            if elapsed >= duration:
                break
                
            # Perform actions based on phase
            phase = "Baseline"
            if 3.0 <= elapsed < 8.0:
                phase = "High Load"
                # Trigger CPU load thread once
                if not cpu_load_active:
                    cpu_load_active = True
                    load_thread = threading.Thread(target=cpu_load_generator, daemon=True)
                    load_thread.start()
                
                # Trigger RAM allocation once
                if memory_bloat_holder is None:
                    print("[Load Gen] Allocating ~240MB of memory...")
                    # Allocate and populate 30 million float64 numbers (approx 240 MB) to force physical page mapping (RSS)
                    memory_bloat_holder = np.random.rand(30000, 1000)
                    
            elif elapsed >= 8.0:
                phase = "Cooldown"
                # Stop CPU load thread
                if cpu_load_active:
                    cpu_load_active = False
                    if load_thread:
                        load_thread.join()
                
                # Free memory
                if memory_bloat_holder is not None:
                    print("[Load Gen] Freeing memory...")
                    memory_bloat_holder = None
                    gc.collect()
            
            # Sample state precisely at poll_interval boundaries
            if now >= next_sample_time:
                record_sample(phase)
                next_sample_time += poll_interval
                
            # Sleep slightly to avoid busy-waiting the control loop
            time.sleep(0.05)
            
    finally:
        # Stop monitor
        monitor.stop()
        
        # Write samples to CSV
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers + ["phase"])
            for s in samples:
                writer.writerow([
                    s["timestamp"], s["cpu"], s["gpu"], s["proc_ram_mb"], 
                    s["system_ram"], s["battery"], s["charging"], s["phase"]
                ])
                
        print(f"\nSaved {len(samples)} samples to: {csv_path}")
        
        # Verify result logic
        # Clean baseline: average of samples in "Baseline" phase after 1.0s to let start-up stabilize
        baseline_cpu_vals = [float(s["cpu"].replace("%", "")) for s in samples if s["phase"] == "Baseline" and float(s["timestamp"]) >= 1.0]
        baseline_cpu = sum(baseline_cpu_vals) / len(baseline_cpu_vals) if baseline_cpu_vals else 30.0
        
        high_load_cpu = max(float(s["cpu"].replace("%", "")) for s in samples if s["phase"] == "High Load")
        cooldown_cpu = float(samples[-1]["cpu"].replace("%", ""))
        
        baseline_ram_vals = [float(s["proc_ram_mb"].replace(" MB", "")) for s in samples if s["phase"] == "Baseline" and float(s["timestamp"]) >= 1.0]
        baseline_ram = sum(baseline_ram_vals) / len(baseline_ram_vals) if baseline_ram_vals else 30.0
        
        high_load_ram = max(float(s["proc_ram_mb"].replace(" MB", "")) for s in samples if s["phase"] == "High Load")
        cooldown_ram = float(samples[-1]["proc_ram_mb"].replace(" MB", ""))
        
        print("\n--- Summary Verification ---")
        print(f"CPU: Baseline ({baseline_cpu:.1f}%) -> High Load Peak ({high_load_cpu:.1f}%) -> Cooldown ({cooldown_cpu:.1f}%)")
        print(f"Proc RAM: Baseline ({baseline_ram:.1f} MB) -> High Load Peak ({high_load_ram:.1f} MB) -> Cooldown ({cooldown_ram:.1f} MB)")
        
        # Simple checks to verify detection works
        assert high_load_cpu > baseline_cpu + 10.0, f"Verification Failed: CPU load spike was not detected (Baseline: {baseline_cpu:.1f}%, Peak: {high_load_cpu:.1f}%)!"
        assert high_load_ram > baseline_ram + 100.0, f"Verification Failed: RAM allocation spike was not detected (Baseline: {baseline_ram:.1f} MB, Peak: {high_load_ram:.1f} MB)!"
        assert cooldown_ram < high_load_ram - 100.0, "Verification Failed: RAM deallocation was not detected!"
        print("\n[SUCCESS] DeviceMonitor validation passed. Metrics show correct reactive values.")

if __name__ == "__main__":
    main()
