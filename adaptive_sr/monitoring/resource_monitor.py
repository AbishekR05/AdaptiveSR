"""
adaptive_sr.monitoring.resource_monitor
========================================
Step 4 — Edge Resource Monitoring.

PURPOSE
-------
Provides real, system-level resource telemetry for an Edge node.

This module is concerned solely with OBSERVABILITY — measuring the current
state of the compute environment.  It does NOT:
  - allocate CPU cores
  - make scheduling decisions
  - control SR workload placement
  - interact with GPU or CUDA

Those capabilities belong to later steps.

LIBRARY
-------
All system measurements use psutil, which is cross-platform (Windows,
Linux, macOS) and already present in the project requirements.
No Linux-only or Windows-only tool is used.

METRIC DEFINITIONS (summary — full definitions in STEP4_IMPLEMENTATION.md)
---------------------------------------------------------------------------
cpu_cores_total      : Logical CPU count as reported by the OS.
cpu_utilization      : System-wide CPU % averaged across all logical cores.
                       Measured from a real psutil call — NOT synthesised.
cpu_cores_available  : Estimated cores not currently consumed by active work.
                       Formula: cpu_cores_total × (1 - cpu_utilization / 100)
                       This is an estimation; Step 4 has no OS-level core
                       reservation mechanism.
memory_total_bytes   : Total physical RAM (bytes).
memory_used_bytes    : Used physical RAM (bytes).
memory_utilization   : memory_used_bytes / memory_total_bytes × 100.
active_requests      : Caller-supplied count of requests currently being
                       processed by the Edge service.
queue_depth          : Caller-supplied count of requests pending after
                       admission but before execution.
                       The synchronous FastAPI Edge has NO application-level
                       work queue; callers pass 0.  This is documented, not
                       fabricated.

WINDOWS COMPATIBILITY
---------------------
psutil.cpu_percent() and psutil.virtual_memory() work natively on Windows.
No elevated privileges are required.  The first call to cpu_percent() with
interval=None may return 0.0 on some platforms if no blocking measurement
has been made; the monitor primes the counter on initialisation to mitigate
this.

FUTURE PORTABILITY
------------------
The same abstraction runs unchanged on Linux/Azure VMs.  Deploying the
Edge service on Azure Linux requires only that psutil is in requirements.txt
(already satisfied).
"""

import time
import logging
from datetime import datetime, timezone

import psutil

from adaptive_sr.shared.schemas import EdgeResourceTelemetry

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Produces timestamped EdgeResourceTelemetry snapshots.

    Parameters
    ----------
    cluster_id : str
        Identity of the edge cluster (e.g. "cluster_01").  Sourced from
        the Edge service configuration, NOT hardcoded here.
    edge_id : str
        Identity of this specific Edge node (e.g. "edge_01").
    sampling_interval_seconds : float
        Minimum wall-clock interval between consecutive psutil CPU samples.
        psutil.cpu_percent(interval=None) returns the delta since the last
        call; this parameter controls how stale that delta may be before a
        fresh blocking measurement is taken.  Default: 1.0 s.

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
        # measurement to delta from.  Priming with a short blocking call ensures
        # subsequent non-blocking calls return real values.
        psutil.cpu_percent(interval=0.1)
        self._last_cpu_sample_time: float = time.monotonic()

    # ------------------------------------------------------------------
    def snapshot(
        self,
        active_requests: int = 0,
        queue_depth: int = 0,
    ) -> EdgeResourceTelemetry:
        """Collect a real-time resource snapshot.

        Parameters
        ----------
        active_requests : int
            Number of requests currently being processed by the Edge service.
            The caller (e.g. Edge request handler) supplies this value.
        queue_depth : int
            Number of requests pending after admission but before execution.
            The synchronous Edge implementation has no application-level work
            queue; callers should pass 0 and document it as such.

        Returns
        -------
        EdgeResourceTelemetry
            A fully populated, timestamped resource record.
        """
        # ── CPU measurement ────────────────────────────────────────────
        now = time.monotonic()
        elapsed_since_last = now - self._last_cpu_sample_time

        if elapsed_since_last >= self.sampling_interval_seconds:
            # Non-blocking: returns delta since the last call
            cpu_util = psutil.cpu_percent(interval=None)
            self._last_cpu_sample_time = now
        else:
            # Sampling interval not elapsed — force a short blocking read
            # to avoid returning a stale 0.0 on first-call platforms.
            cpu_util = psutil.cpu_percent(interval=0.05)
            self._last_cpu_sample_time = time.monotonic()

        cpu_total = psutil.cpu_count(logical=True) or 1

        # cpu_cores_available: estimation based on observed utilization.
        # This is NOT an OS-level reservation.  See STEP4_IMPLEMENTATION.md
        # §3 for the full semantics and limitations of this formula.
        cpu_available = max(0.0, cpu_total * (1.0 - cpu_util / 100.0))

        # ── Memory measurement ─────────────────────────────────────────
        vm = psutil.virtual_memory()
        mem_total = vm.total
        mem_used = vm.used
        mem_util = (mem_used / mem_total * 100.0) if mem_total > 0 else 0.0

        # ── Timestamp (timezone-aware UTC) ─────────────────────────────
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
