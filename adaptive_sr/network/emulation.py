"""
adaptive_sr.network.emulation
==============================
Step 3.2 — Application-level network emulation adapter.

DESIGN RATIONALE:
  Windows does not provide tc/netem kernel-level packet shaping.
  This module implements emulation at the service boundary by:
    1. Injecting configurable artificial delay (time.sleep) before
       returning the response, simulating network propagation delay.
    2. Throttling response body reads at a rate limited to the
       configured bandwidth, so that transfer_duration_seconds grows
       realistically as bandwidth decreases.

  This is explicitly APPLICATION-LEVEL emulation, not kernel/network-
  stack-level packet shaping. The distinction must be preserved in
  documentation and telemetry. A future Linux/Azure deployment can
  replace EmulatedHttpAdapter with a tc/netem wrapper without any
  other code changes.

CONNECTION REUSE BEHAVIOR (documented per Step 3.2 §5):
  The existing codebase uses bare requests.get() calls with no
  persistent HTTP session. Every call — including health pings and
  chunk fetches — opens a new TCP connection. Therefore:
    - RTT measurements (from health probes) include TCP
      connection-establishment overhead.
    - RTT is connection-inclusive latency, NOT pure propagation delay.
    - This will remain true until Step 3.2+ explicitly introduces
      a persistent session/connection pool.

PACKET LOSS:
  Application-level packet loss is modeled as a probability that a
  request raises a requests.RequestException (simulating a dropped
  connection). This is explicitly documented as application-layer
  fault injection, not true packet-layer loss. If packet_loss_rate
  is set, the adapter will raise an IOError with probability
  packet_loss_rate before the response body is read. Callers must
  handle this exception.
"""

import io
import time
import random
import logging
from typing import Optional, Literal

import requests
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-path configuration model
# ---------------------------------------------------------------------------

class NetworkPathEmulationConfig(BaseModel):
    """Configuration for emulated conditions on a single network path.

    Fields
    ------
    bandwidth_mbps : float or None
        Maximum data rate in Megabits per second.
        None (default) means unlimited — no throttling applied.
    delay_ms : float
        One-way artificial delay injected BEFORE the response body is
        streamed back to the caller, in milliseconds. Default 0.
    packet_loss_rate : float
        Probability [0.0, 1.0] that a request is aborted before the
        response body is read, simulating a dropped connection.
        Default 0.0 (no loss).

        NOTE: This is application-layer fault injection, not true
        packet-layer loss. Callers must handle the resulting IOError.
    """
    bandwidth_mbps: Optional[float] = Field(None, gt=0,
        description="Bandwidth cap in Mbps; None = unlimited")
    delay_ms: float = Field(0.0, ge=0.0,
        description="Artificial one-way delay in ms")
    packet_loss_rate: float = Field(0.0, ge=0.0, le=1.0,
        description="Probability [0, 1] of simulated connection drop")


class NetworkEmulationConfig(BaseModel):
    """Top-level emulation config owning independent per-path configs.

    Fields
    ------
    enabled : bool
        Master switch. When False, the adapter passes requests through
        with zero added delay or throttling.
    client_edge : NetworkPathEmulationConfig
        Emulation parameters applied to Client ↔ Edge communications.
    edge_cloud : NetworkPathEmulationConfig
        Emulation parameters applied to Edge ↔ Cloud communications.
    scenario_name : str or None
        Optional human-readable label for the active scenario; logged
        with telemetry for experiment reproducibility.
    """
    enabled: bool = True
    client_edge: NetworkPathEmulationConfig = Field(
        default_factory=NetworkPathEmulationConfig)
    edge_cloud: NetworkPathEmulationConfig = Field(
        default_factory=NetworkPathEmulationConfig)
    scenario_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Named scenario presets
# ---------------------------------------------------------------------------

# Values are stored as explicit config (not hardcoded throughout the code).
# Adjust bandwidth/delay here as research parameters are finalised.
SCENARIOS: dict[str, NetworkEmulationConfig] = {
    "good": NetworkEmulationConfig(
        scenario_name="good",
        client_edge=NetworkPathEmulationConfig(bandwidth_mbps=20.0, delay_ms=5.0),
        edge_cloud=NetworkPathEmulationConfig(bandwidth_mbps=100.0, delay_ms=2.0),
    ),
    "moderate": NetworkEmulationConfig(
        scenario_name="moderate",
        client_edge=NetworkPathEmulationConfig(bandwidth_mbps=5.0, delay_ms=30.0),
        edge_cloud=NetworkPathEmulationConfig(bandwidth_mbps=20.0, delay_ms=10.0),
    ),
    "poor": NetworkEmulationConfig(
        scenario_name="poor",
        client_edge=NetworkPathEmulationConfig(bandwidth_mbps=1.0, delay_ms=100.0),
        edge_cloud=NetworkPathEmulationConfig(bandwidth_mbps=5.0, delay_ms=50.0),
    ),
    "disabled": NetworkEmulationConfig(
        enabled=False,
        scenario_name="disabled",
        client_edge=NetworkPathEmulationConfig(),
        edge_cloud=NetworkPathEmulationConfig(),
    ),
}


# ---------------------------------------------------------------------------
# Emulated HTTP adapter
# ---------------------------------------------------------------------------

class EmulatedHttpAdapter:
    """Application-level network emulation wrapper around requests.get().

    Usage
    -----
    adapter = EmulatedHttpAdapter(config, network_path="client_edge")
    content, duration = adapter.get(url, **kwargs)

    Parameters
    ----------
    config : NetworkEmulationConfig
        The active emulation configuration.
    network_path : Literal["client_edge", "edge_cloud"]
        Which path-specific config to apply.
    """

    def __init__(
        self,
        config: NetworkEmulationConfig,
        network_path: Literal["client_edge", "edge_cloud"],
    ):
        self.config = config
        self.network_path = network_path

        if network_path == "client_edge":
            self.path_config = config.client_edge
        elif network_path == "edge_cloud":
            self.path_config = config.edge_cloud
        else:
            raise ValueError(f"Unknown network_path: {network_path!r}")

    # ------------------------------------------------------------------
    def get(self, url: str, **kwargs) -> tuple[bytes, float]:
        """Issue a GET request with optional emulated delay and throttle.

        Returns
        -------
        (content_bytes, transfer_duration_seconds)
            content_bytes        : Raw response body.
            transfer_duration_seconds : Wall-clock time spent reading
                                   the response body (after the delay).
                                   This is what callers should use for
                                   throughput calculation.

        Raises
        ------
        IOError
            If packet_loss_rate > 0 and the random draw triggers a loss.
        requests.RequestException
            On any underlying HTTP error.
        """
        pc = self.path_config

        # ── 1. Artificial delay (one-way emulation) ────────────────────
        if self.config.enabled and pc.delay_ms > 0.0:
            time.sleep(pc.delay_ms / 1000.0)

        # ── 2. Issue the real HTTP request ─────────────────────────────
        response = requests.get(url, **kwargs)
        response.raise_for_status()

        # ── 3. Packet-loss simulation (application layer) ──────────────
        if self.config.enabled and pc.packet_loss_rate > 0.0:
            if random.random() < pc.packet_loss_rate:
                raise IOError(
                    f"[Emulated packet loss on {self.network_path}] "
                    f"Request to {url!r} dropped (loss_rate={pc.packet_loss_rate})"
                )

        # ── 4. Bandwidth-throttled body read ───────────────────────────
        transfer_start = time.monotonic()

        if self.config.enabled and pc.bandwidth_mbps is not None:
            # Chunk size: 4 KB — fine-grained enough for sub-second throttling
            chunk_size_bytes = 4096
            bytes_per_second = (pc.bandwidth_mbps * 1_000_000) / 8.0

            buf = io.BytesIO()
            for chunk in response.iter_content(chunk_size=chunk_size_bytes):
                if chunk:
                    chunk_start = time.monotonic()
                    buf.write(chunk)
                    # How long should this chunk take at the configured rate?
                    expected_duration = len(chunk) / bytes_per_second
                    elapsed = time.monotonic() - chunk_start
                    sleep_needed = expected_duration - elapsed
                    if sleep_needed > 0:
                        time.sleep(sleep_needed)
            content = buf.getvalue()
        else:
            content = response.content

        transfer_duration = time.monotonic() - transfer_start
        return content, transfer_duration


def compute_throughput_mbps(bytes_transferred: int, transfer_duration_seconds: float) -> float:
    """Calculate throughput using the Step 3.1 contract formula.

    throughput_mbps = bytes_transferred * 8 / transfer_duration_seconds / 1_000_000

    Never divide by zero — returns 0.0 if duration is zero.
    """
    if transfer_duration_seconds <= 0.0:
        return 0.0
    return (bytes_transferred * 8) / (transfer_duration_seconds * 1_000_000.0)
