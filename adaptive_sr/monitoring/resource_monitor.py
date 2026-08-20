"""
adaptive_sr.monitoring.resource_monitor
========================================
Step 4 / Step 4.1 — Edge Resource Monitoring.

PURPOSE
-------
Provides two distinct levels of resource telemetry:

  1. HOST-LEVEL (ResourceMonitor.snapshot)
     Describes the Edge compute environment as a whole.
     "How busy is this Edge host?"
     → Returns EdgeResourceTelemetry.

  2. PROCESS-LEVEL (ProcessMonitor.snapshot)
     Describes a specific OS process.
     "How much CPU and RAM does this particular process consume?"
     → Returns ProcessResourceSnapshot.

HOST vs PROCESS DISTINCTION
-----------------------------
These are fundamentally different quantities that MUST NOT be confused:

  Host cpu_utilization = 60%
    ↳ does NOT mean SR is using 60% CPU
    ↳ does NOT mean 4.8 cores are available for SR

  Process cpu_percent = 320%
    ↳ means this process is using 3.2 logical cores worth of CPU time

This distinction is critical because the current development environment
runs Cloud, Edge, Client, network emulation, and testing on the same
physical machine.  Host-wide CPU is a poor proxy for any specific workload.

STEP 5 CONTRACT
---------------
Step 5 SR benchmarking MUST use ProcessMonitor (process-level measurements)
to characterize SR workload CPU consumption.
Host-wide cpu_utilization MUST NOT be used as a proxy for SR CPU usage.

METRIC DEFINITIONS (summary — full definitions in STEP4_IMPLEMENTATION.md)
---------------------------------------------------------------------------
HOST-LEVEL (EdgeResourceTelemetry):
  cpu_cores_total      : Logical CPU count as reported by the OS.
  cpu_utilization      : System-wide CPU % (ALL processes, ALL cores averaged).
                         Real psutil call — NOT synthesised.
  cpu_cores_available  : OBSERVATIONAL ESTIMATE ONLY.
                         Formula: cpu_cores_total × (1 - cpu_utilization / 100).
                         Must NOT be consumed as a scheduler allocation quantity.
                         Not an OS-level reservation.
  memory_total_bytes   : Total physical RAM (bytes).
  memory_used_bytes    : Used physical RAM (bytes).
  memory_utilization   : memory_used_bytes / memory_total_bytes × 100.
  active_requests      : Caller-supplied in-flight request count.
  queue_depth          : Caller-supplied pending request count.
                         The synchronous FastAPI Edge has NO application-level
                         work queue; callers pass 0.

PROCESS-LEVEL (ProcessResourceSnapshot):
  cpu_percent          : % of ONE logical core used by this process.
                         Values >100 indicate multi-threaded CPU use.
  memory_used_bytes    : Resident Set Size (RSS) of this process.
  memory_percent       : RSS / total RAM × 100.

WINDOWS COMPATIBILITY
---------------------
All psutil calls used here work natively on Windows without elevated
privileges.  psutil.Process is cross-platform (Windows, Linux, macOS).

FUTURE PORTABILITY
------------------
The same abstraction runs unchanged on Linux/Azure VMs.
"""

import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional, List

import psutil

from adaptive_sr.shared.schemas import EdgeResourceTelemetry, ProcessResourceSnapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HOST-LEVEL MONITOR
# ---------------------------------------------------------------------------

class ResourceMonitor:
    """Produces timestamped EdgeResourceTelemetry (host-level) snapshots.

    All measurements describe the Edge HOST environment, not any specific
    process.  See ProcessMonitor for process-level measurements.

    Parameters
    ----------
    cluster_id : str
        Identity of the edge cluster.  Sourced from config, NOT hardcoded.
    edge_id : str
        Identity of this specific Edge node.
    sampling_interval_seconds : float
        Minimum interval between consecutive non-blocking CPU delta reads.
        Default: 1.0 s.

    Usage
    -----
    monitor = ResourceMonitor(cluster_id="cluster_01", edge_id="edge_01")
    telemetry = monitor.snapshot(active_requests=1, queue_depth=0)
    """

    def __init__(
        self,
        cluster_id: str,
        edge_id: str,
        sampling_interval_seconds: float = 1.0,
    ) -> None:
        self.cluster_id = cluster_id
        self.edge_id = edge_id
        self.sampling_interval_seconds = sampling_interval_seconds

        # Prime the psutil CPU counter.  The first call to cpu_percent() with
        # interval=None returns 0.0 on some platforms because there is no prior
        # measurement to delta from.  A short blocking prime call ensures
        # subsequent non-blocking calls return real values.
        psutil.cpu_percent(interval=0.1)
        self._last_cpu_sample_time: float = time.monotonic()

    # ------------------------------------------------------------------
    def snapshot(
        self,
        active_requests: int = 0,
        queue_depth: int = 0,
    ) -> EdgeResourceTelemetry:
        """Collect a real-time HOST-LEVEL resource snapshot.

        Parameters
        ----------
        active_requests : int
            In-flight request count; supplied by the Edge request handler.
        queue_depth : int
            Pending request count.  Pass 0 for the synchronous Edge (no queue).

        Returns
        -------
        EdgeResourceTelemetry
        """
        # ── CPU measurement (HOST-WIDE) ────────────────────────────────
        now = time.monotonic()
        elapsed_since_last = now - self._last_cpu_sample_time

        if elapsed_since_last >= self.sampling_interval_seconds:
            cpu_util = psutil.cpu_percent(interval=None)
            self._last_cpu_sample_time = now
        else:
            # Sampling interval not elapsed — force a short blocking read.
            cpu_util = psutil.cpu_percent(interval=0.05)
            self._last_cpu_sample_time = time.monotonic()

        cpu_total = psutil.cpu_count(logical=True) or 1

        # cpu_cores_available: OBSERVATIONAL ESTIMATE.
        # This is derived from host-wide utilization and must NOT be treated
        # as a physically allocatable or scheduler-reserved core count.
        # See STEP4_IMPLEMENTATION.md §3 and §6 for full semantics.
        cpu_available = max(0.0, cpu_total * (1.0 - cpu_util / 100.0))

        # ── Memory measurement ─────────────────────────────────────────
        vm = psutil.virtual_memory()
        mem_total = vm.total
        mem_used = vm.used
        mem_util = (mem_used / mem_total * 100.0) if mem_total > 0 else 0.0

        # ── Timestamp ──────────────────────────────────────────────────
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"

        return EdgeResourceTelemetry(
            timestamp=ts,
            cluster_id=self.cluster_id,
            edge_id=self.edge_id,
            cpu_cores_total=cpu_total,
            cpu_utilization=round(cpu_util, 2),
            cpu_cores_available=round(cpu_available, 4),
            memory_total_bytes=mem_total,
            memory_used_bytes=mem_used,
            memory_utilization=round(mem_util, 2),
            active_requests=active_requests,
            queue_depth=queue_depth,
        )

    # ------------------------------------------------------------------
    def per_core_utilization(self) -> List[float]:
        """Return per-logical-core CPU utilization percentages.

        Uses psutil.cpu_percent(percpu=True) to expose uneven CPU loading
        across cores.  This is an observational capability preserved for
        future reasoning about per-core load distribution.

        Returns
        -------
        List[float]
            One utilization value per logical CPU core, in core-index order.
            Values are in [0.0, 100.0].

        Notes
        -----
        This method is NOT a scheduler feature.  It does not allocate cores
        or make placement decisions.  It is provided so that Step 5+ can
        detect whether SR threads are concentrating on particular cores.
        """
        # A short blocking interval ensures a real measurement rather than
        # returning 0.0 on the first call.
        return psutil.cpu_percent(percpu=True, interval=0.1)


# ---------------------------------------------------------------------------
# PROCESS-LEVEL MONITOR
# ---------------------------------------------------------------------------

class ProcessMonitor:
    """Produces ProcessResourceSnapshot measurements for a specific OS process.

    This is DISTINCT from ResourceMonitor (host-level).  Use this class when
    you need to answer: "How much CPU does THIS process use?" rather than
    "How busy is the entire host?"

    STEP 5 CONTRACT
    ---------------
    Step 5 SR benchmarking MUST use ProcessMonitor to characterize SR
    workload resource consumption.  Host-wide cpu_utilization MUST NOT
    be used as a proxy for SR CPU usage.

    Parameters
    ----------
    pid : int or None
        OS process ID to monitor.  If None, monitors the current process
        (os.getpid()).  Pass an explicit PID to monitor an external process.

    Usage
    -----
    # Monitor current process
    pm = ProcessMonitor()
    snap = pm.snapshot(interval=0.5)

    # Monitor an external process by PID
    pm = ProcessMonitor(pid=some_pid)
    snap = pm.snapshot(interval=0.5)
    """

    def __init__(self, pid: Optional[int] = None) -> None:
        target_pid = pid if pid is not None else os.getpid()
        self._process = psutil.Process(target_pid)
        # Prime the per-process CPU counter (same reason as host-level priming).
        self._process.cpu_percent(interval=None)

    # ------------------------------------------------------------------
    def snapshot(self, interval: float = 0.1) -> ProcessResourceSnapshot:
        """Collect a real-time process-level resource snapshot.

        Parameters
        ----------
        interval : float
            Blocking measurement interval in seconds.  The psutil
            Process.cpu_percent(interval=...) call blocks for this duration
            and returns the CPU % consumed over that window.
            Default: 0.1 s (100 ms) — short enough for benchmarking loops.

        Returns
        -------
        ProcessResourceSnapshot
            A measurement of this process's CPU and memory usage.

        Notes
        -----
        cpu_percent may exceed 100 on multi-core systems when the process
        uses multiple threads (e.g. 350% ≈ 3.5 logical cores worth of CPU).
        This is psutil's documented behaviour for Process.cpu_percent().
        """
        # Blocking call: measures real CPU consumed during `interval` seconds.
        cpu_pct = self._process.cpu_percent(interval=interval)

        mem_info = self._process.memory_info()
        mem_pct = self._process.memory_percent()
        proc_name = self._process.name()

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"

        return ProcessResourceSnapshot(
            timestamp=ts,
            process_id=self._process.pid,
            process_name=proc_name,
            cpu_percent=round(cpu_pct, 2),
            memory_used_bytes=mem_info.rss,
            memory_percent=round(mem_pct, 4),
        )

    @property
    def pid(self) -> int:
        """The PID of the monitored process."""
        return self._process.pid
