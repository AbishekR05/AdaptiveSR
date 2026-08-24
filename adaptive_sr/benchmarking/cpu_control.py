"""
adaptive_sr.benchmarking.cpu_control
====================================
Step 5.3 — CPU Affinity + ProcessMonitor Integration.

Provides utility functions, context managers, and configuration models
to execute benchmark workloads under restricted logical CPU bounds,
reusing the frozen Step 4 ProcessMonitor for observability.
"""

import os
import time
import threading
import psutil
from dataclasses import dataclass
from contextlib import contextmanager
from typing import List, Optional

from adaptive_sr.shared.schemas import ProcessResourceSnapshot
from adaptive_sr.monitoring.resource_monitor import ProcessMonitor


# ---------------------------------------------------------------------------
# CPU DISCOVERY AND SELECTION UTILITIES
# ---------------------------------------------------------------------------

def get_available_cpus() -> List[int]:
    """Discovers all logical CPU core IDs available on the host machine.

    Returns
    -------
    List[int]
        List of logical core indices (0-indexed).
    """
    count = psutil.cpu_count(logical=True) or 1
    return list(range(count))


def get_current_affinity() -> List[int]:
    """Obtains the logical CPU affinity mask of the current process.

    Returns
    -------
    List[int]
        Logical CPU IDs that this process is currently allowed to run on.
    """
    return psutil.Process().cpu_affinity()


def set_affinity(cpu_ids: List[int]) -> None:
    """Sets the logical CPU affinity of the current process.

    Parameters
    ----------
    cpu_ids : List[int]
        List of CPU IDs to bind to.
    """
    psutil.Process().cpu_affinity(cpu_ids)


def restore_affinity(previous_affinity: List[int]) -> None:
    """Restores the logical CPU affinity of the current process to a previous mask.

    Parameters
    ----------
    previous_affinity : List[int]
        Target CPU ID list to restore.
    """
    psutil.Process().cpu_affinity(previous_affinity)


def select_cpu_ids(count: int, exclude_cpu_ids: Optional[List[int]] = None) -> List[int]:
    """Deterministically selects a subset of available logical CPU IDs, optionally excluding some.

    For example, if host has [0, 1, 2, 3, 4], calling with count=2 and exclude_cpu_ids=[0]
    returns [1, 2].

    Parameters
    ----------
    count : int
        The number of logical CPU cores to request.
    exclude_cpu_ids : Optional[List[int]]
        CPU ID list to exclude from selection.

    Returns
    -------
    List[int]
        List containing exactly `count` logical CPU core IDs.
    """
    available = get_available_cpus()
    if exclude_cpu_ids:
        available = [c for c in available if c not in exclude_cpu_ids]

    if count <= 0:
        raise ValueError("Requested CPU count must be greater than 0.")
    if count > len(available):
        raise ValueError(
            f"Requested CPU count {count} exceeds available CPUs on this host ({len(available)}) after exclusions."
        )
    return available[:count]


# ---------------------------------------------------------------------------
# CPU EXECUTION CONFIGURATION OBJECT
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CPUExecutionConfig:
    """Immutable execution configuration mapping logical CPU cores and model thread limits.

    Ensures that thread count and hardware core constraints are represented separately.
    """
    cpu_ids: List[int]
    num_threads: Optional[int] = None
    exclude_cpu_ids: Optional[List[int]] = None

    def __post_init__(self) -> None:
        # Validation rules (Step 5.3 §19)
        if not self.cpu_ids:
            raise ValueError("CPU IDs set cannot be empty.")

        available = get_available_cpus()
        excluded = set(self.exclude_cpu_ids) if self.exclude_cpu_ids else set()
        seen = set()
        for cid in self.cpu_ids:
            if cid < 0:
                raise ValueError(f"CPU ID must be non-negative, got: {cid}")
            if cid in seen:
                raise ValueError(f"Duplicate CPU ID detected in request: {cid}")
            seen.add(cid)
            if cid in excluded:
                raise ValueError(f"CPU ID {cid} is in exclude_cpu_ids set, conflict detected.")
            if cid not in available:
                raise ValueError(
                    f"Requested CPU ID {cid} is not available on this host. "
                    f"Host exposes: {available}"
                )

        if self.num_threads is not None:
            if self.num_threads <= 0:
                raise ValueError(f"num_threads must be greater than 0, got: {self.num_threads}")

    @property
    def requested_cpu_count(self) -> int:
        """Returns the number of logical cores requested in this configuration."""
        return len(self.cpu_ids)


# ---------------------------------------------------------------------------
# CPU AFFINITY CONTEXT MANAGER
# ---------------------------------------------------------------------------

@contextmanager
def cpu_affinity_context(cpu_ids: List[int]):
    """Context manager restricting execution to cpu_ids, restoring affinity on exit or error.

    Parameters
    ----------
    cpu_ids : List[int]
        Target CPU ID list to pin the current process to.
    """
    p = psutil.Process()
    previous_affinity = p.cpu_affinity()
    try:
        p.cpu_affinity(cpu_ids)
        yield
    finally:
        # Guaranteed restoration (Step 5.3 §20)
        try:
            p.cpu_affinity(previous_affinity)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# BENCHMARK PROCESS OBSERVER WRAPPER
# ---------------------------------------------------------------------------

class BenchmarkProcessMonitor:
    """Wrapper around Step 4 ProcessMonitor to enable start/stop lifecycle and background sampling."""

    def __init__(self, pid: Optional[int] = None, sample_interval: float = 0.05) -> None:
        """Initializes the background process monitor wrapper.

        Parameters
        ----------
        pid : int or None
            Target PID. Defaults to current process ID.
        sample_interval : float
            Monitoring polling rate in seconds (defaults to 0.05s / 50ms).
        """
        self.process_monitor = ProcessMonitor(pid=pid)
        self.sample_interval = sample_interval
        self.samples: List[ProcessResourceSnapshot] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def pid(self) -> int:
        """The PID of the process being monitored."""
        return self.process_monitor.pid

    def start(self) -> None:
        """Starts the background process-level monitoring thread."""
        if self._thread is not None:
            raise RuntimeError("ProcessMonitor is already running.")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="BenchmarkProcessMonitorThread",
            daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        # Polling loop accumulating snapshots
        while not self._stop_event.is_set():
            try:
                # Reuse Step 4 ProcessMonitor's snapshot method
                snap = self.process_monitor.snapshot(interval=self.sample_interval)
                self.samples.append(snap)
            except Exception:
                # Gracefully exit if process dies or access is restricted
                break
            time.sleep(0.01)

    def stop(self) -> None:
        """Stops the background process monitor and cleans up threads."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def get_samples(self) -> List[ProcessResourceSnapshot]:
        """Returns all ProcessResourceSnapshot telemetry samples collected so far."""
        return self.samples


# ---------------------------------------------------------------------------
# INTEGRATED BENCHMARK WORKLOAD EXECUTION CONTEXT
# ---------------------------------------------------------------------------

@contextmanager
def benchmark_execution_context(
    config: CPUExecutionConfig,
    monitor: Optional[BenchmarkProcessMonitor] = None
):
    """Context manager implementing the complete start-stop resource monitoring and affinity lifecycle.

    Guarantees that both ProcessMonitor termination and CPU affinity restoration occur
    even if exceptions or benchmark failures occur inside the context.

    Parameters
    ----------
    config : CPUExecutionConfig
        The CPU configuration structure (target cpu set and threads count).
    monitor : BenchmarkProcessMonitor or None
        An optional process monitor to run during execution.
    """
    # 1. Capture original CPU affinity
    p = psutil.Process()
    previous_affinity = p.cpu_affinity()
    monitor_started = False
    
    try:
        # 2. Apply requested CPU affinity
        p.cpu_affinity(config.cpu_ids)
        
        # 3. Verify that requested affinity is active
        active_affinity = p.cpu_affinity()
        if active_affinity != config.cpu_ids:
            raise RuntimeError(
                f"Failed to apply CPU affinity core limits. "
                f"Requested: {config.cpu_ids}, active: {active_affinity}"
            )
            
        # 4. Start ProcessMonitor (only after affinity is active)
        if monitor is not None:
            monitor.start()
            monitor_started = True
            
        # 5. Yield to the benchmark workload
        yield
        
    finally:
        # 6. Stop ProcessMonitor (guaranteed cleanup)
        if monitor is not None and monitor_started:
            try:
                monitor.stop()
            except Exception:
                pass
                
        # 7. Restore original CPU affinity (guaranteed cleanup)
        try:
            p.cpu_affinity(previous_affinity)
        except Exception:
            pass
