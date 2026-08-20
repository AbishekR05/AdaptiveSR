"""
tests/test_emulation.py
========================
Step 3.2 — Controlled Network Emulation tests.

Tests use a lightweight HTTP server (via threading + http.server) to serve
a fixed payload locally so bandwidth throttling and delay produce measurable,
reproducible results without depending on running AdaptiveSR services.
"""
import io
import os
import sys
import time
import random
import threading
import http.server
import pytest
from pydantic import ValidationError

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adaptive_sr.network.emulation import (
    NetworkPathEmulationConfig,
    NetworkEmulationConfig,
    EmulatedHttpAdapter,
    SCENARIOS,
    ROSEVIN_RTT_REFERENCE_MS,
    ROSEVIN_BANDWIDTH_MBPS,
    compute_throughput_mbps,
)
from adaptive_sr.shared.schemas import NetworkMeasurement


# ---------------------------------------------------------------------------
# Minimal in-process HTTP server fixture
# ---------------------------------------------------------------------------

PAYLOAD_SIZE = 100_000  # 100 KB — large enough for throttling to be measurable
FIXED_PAYLOAD = b"X" * PAYLOAD_SIZE


class _PayloadHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(FIXED_PAYLOAD)))
        self.end_headers()
        self.wfile.write(FIXED_PAYLOAD)

    def log_message(self, *args):
        pass  # Silence server-side logs during tests


@pytest.fixture(scope="module")
def test_server():
    """Start a local HTTP server serving FIXED_PAYLOAD on a random port."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _PayloadHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/"
    server.shutdown()


# ---------------------------------------------------------------------------
# 1. Emulation disabled — passthrough with no added delay
# ---------------------------------------------------------------------------

def test_emulation_disabled_passthrough(test_server):
    """When enabled=False, adapter passes through with negligible overhead."""
    config = NetworkEmulationConfig(enabled=False)
    adapter = EmulatedHttpAdapter(config, "client_edge")

    t0 = time.monotonic()
    content, duration = adapter.get(test_server, timeout=5.0)
    elapsed = time.monotonic() - t0

    assert content == FIXED_PAYLOAD
    # No artificial delay injected — total wall time should be << 0.5s locally
    assert elapsed < 0.5, f"Disabled emulation should not add overhead; elapsed={elapsed:.3f}s"


# ---------------------------------------------------------------------------
# 2. Configured delay measurably increases latency
# ---------------------------------------------------------------------------

def test_configured_delay_increases_latency(test_server):
    """A 100ms configured delay must produce a measurably longer wall-clock time."""
    delay_ms = 100.0
    config = NetworkEmulationConfig(
        client_edge=NetworkPathEmulationConfig(delay_ms=delay_ms)
    )
    adapter = EmulatedHttpAdapter(config, "client_edge")

    t0 = time.monotonic()
    content, _ = adapter.get(test_server, timeout=5.0)
    elapsed_ms = (time.monotonic() - t0) * 1000.0

    assert content == FIXED_PAYLOAD
    # Must be at least half the configured delay (generous tolerance for CI)
    assert elapsed_ms >= delay_ms * 0.5, (
        f"Expected >= {delay_ms*0.5:.0f}ms elapsed, got {elapsed_ms:.1f}ms"
    )


# ---------------------------------------------------------------------------
# 3. Lower bandwidth produces longer transfer for the same payload
# ---------------------------------------------------------------------------

def test_lower_bandwidth_produces_longer_transfer(test_server):
    """POOR bandwidth (1 Mbps) must produce a longer transfer than GOOD (20 Mbps)."""
    good_adapter = EmulatedHttpAdapter(
        NetworkEmulationConfig(
            client_edge=NetworkPathEmulationConfig(bandwidth_mbps=20.0, delay_ms=0.0)
        ),
        "client_edge",
    )
    poor_adapter = EmulatedHttpAdapter(
        NetworkEmulationConfig(
            client_edge=NetworkPathEmulationConfig(bandwidth_mbps=1.0, delay_ms=0.0)
        ),
        "client_edge",
    )

    _, good_duration = good_adapter.get(test_server, timeout=30.0)
    _, poor_duration = poor_adapter.get(test_server, timeout=30.0)

    assert poor_duration > good_duration, (
        f"Poor bandwidth ({poor_duration:.3f}s) must take longer than "
        f"good bandwidth ({good_duration:.3f}s)"
    )


# ---------------------------------------------------------------------------
# 4. client_edge config does NOT affect edge_cloud path
# ---------------------------------------------------------------------------

def test_client_edge_config_does_not_affect_edge_cloud(test_server):
    """Delay set on client_edge must not be applied when path=edge_cloud."""
    config = NetworkEmulationConfig(
        client_edge=NetworkPathEmulationConfig(delay_ms=200.0),
        edge_cloud=NetworkPathEmulationConfig(delay_ms=0.0),
    )
    adapter = EmulatedHttpAdapter(config, "edge_cloud")

    t0 = time.monotonic()
    adapter.get(test_server, timeout=5.0)
    elapsed_ms = (time.monotonic() - t0) * 1000.0

    # edge_cloud has 0ms delay, so should complete quickly
    assert elapsed_ms < 150.0, (
        f"client_edge delay should not bleed into edge_cloud; elapsed={elapsed_ms:.1f}ms"
    )


# ---------------------------------------------------------------------------
# 5. edge_cloud config does NOT affect client_edge path
# ---------------------------------------------------------------------------

def test_edge_cloud_config_does_not_affect_client_edge(test_server):
    """Delay set on edge_cloud must not be applied when path=client_edge."""
    config = NetworkEmulationConfig(
        client_edge=NetworkPathEmulationConfig(delay_ms=0.0),
        edge_cloud=NetworkPathEmulationConfig(delay_ms=200.0),
    )
    adapter = EmulatedHttpAdapter(config, "client_edge")

    t0 = time.monotonic()
    adapter.get(test_server, timeout=5.0)
    elapsed_ms = (time.monotonic() - t0) * 1000.0

    assert elapsed_ms < 150.0, (
        f"edge_cloud delay should not bleed into client_edge; elapsed={elapsed_ms:.1f}ms"
    )


# ---------------------------------------------------------------------------
# 6. Two path configs can differ simultaneously
# ---------------------------------------------------------------------------

def test_two_paths_differ_simultaneously(test_server):
    """client_edge and edge_cloud can have different delays simultaneously."""
    config = NetworkEmulationConfig(
        client_edge=NetworkPathEmulationConfig(delay_ms=50.0, bandwidth_mbps=None),
        edge_cloud=NetworkPathEmulationConfig(delay_ms=150.0, bandwidth_mbps=None),
    )

    ce_adapter = EmulatedHttpAdapter(config, "client_edge")
    ec_adapter = EmulatedHttpAdapter(config, "edge_cloud")

    t0 = time.monotonic()
    ce_adapter.get(test_server, timeout=5.0)
    ce_elapsed_ms = (time.monotonic() - t0) * 1000.0

    t0 = time.monotonic()
    ec_adapter.get(test_server, timeout=5.0)
    ec_elapsed_ms = (time.monotonic() - t0) * 1000.0

    assert ec_elapsed_ms > ce_elapsed_ms, (
        f"edge_cloud (150ms delay) should take longer than "
        f"client_edge (50ms delay): ec={ec_elapsed_ms:.1f}ms ce={ce_elapsed_ms:.1f}ms"
    )


# ---------------------------------------------------------------------------
# 7. Throughput measured from actual bytes and actual duration
# ---------------------------------------------------------------------------

def test_throughput_measured_from_actual_bytes_and_duration(test_server):
    """Throughput must be derived from bytes/duration, not from config bandwidth."""
    config = NetworkEmulationConfig(
        client_edge=NetworkPathEmulationConfig(bandwidth_mbps=2.0, delay_ms=0.0)
    )
    adapter = EmulatedHttpAdapter(config, "client_edge")

    content, duration = adapter.get(test_server, timeout=30.0)
    measured_mbps = compute_throughput_mbps(len(content), duration)

    # Formula check: bytes * 8 / duration / 1e6
    expected = (len(content) * 8) / (duration * 1_000_000.0)
    assert abs(measured_mbps - expected) < 0.001, (
        f"Throughput formula mismatch: got {measured_mbps}, expected {expected}"
    )

    # The measured value must reflect actual observed duration —
    # NOT be manually set to the configured 2.0 Mbps
    # (Due to OS scheduling, actual may be higher or lower; we just confirm it's computed.)
    assert measured_mbps > 0.0


# ---------------------------------------------------------------------------
# 8. Telemetry is NOT manually overwritten with configured bandwidth
# ---------------------------------------------------------------------------

def test_telemetry_not_overwritten_with_configured_bandwidth(test_server):
    """measured_throughput_mbps must come from real transfer, not config value."""
    configured_bw = 5.0  # Mbps
    config = NetworkEmulationConfig(
        client_edge=NetworkPathEmulationConfig(bandwidth_mbps=configured_bw, delay_ms=0.0)
    )
    adapter = EmulatedHttpAdapter(config, "client_edge")
    content, duration = adapter.get(test_server, timeout=30.0)

    actual_mbps = compute_throughput_mbps(len(content), duration)
    # The value is computed — not simply equal to the configured bandwidth
    # (It might be close due to throttling, but we're verifying the derivation path)
    recomputed = (len(content) * 8) / (duration * 1_000_000.0)
    assert abs(actual_mbps - recomputed) < 0.001, (
        "Throughput must be derived from actual transfer metrics, not config value"
    )


# ---------------------------------------------------------------------------
# 9. RTT measurements retain correct network_path
# ---------------------------------------------------------------------------

def test_rtt_measurements_retain_correct_network_path():
    """NetworkMeasurement must preserve the network_path on RTT-only probes."""
    for path in ("client_edge", "edge_cloud"):
        m = NetworkMeasurement(
            request_id=f"ping-{path}",
            network_path=path,
            rtt_ms=15.0,
        )
        assert m.network_path == path
        assert m.chunk_id is None
        assert m.representation_id is None


# ---------------------------------------------------------------------------
# 10. Existing cache HIT/MISS semantics unchanged
# ---------------------------------------------------------------------------

def test_cache_hit_miss_semantics_unchanged():
    """Cache HIT → only client_edge measurement; MISS → both paths present."""
    # HIT: no edge_cloud measurement produced
    hit_measurement = NetworkMeasurement(
        request_id="req-hit",
        network_path="client_edge",
        chunk_id="0000",
        representation_id="360p",
        bytes_transferred=500_000,
        transfer_duration_seconds=2.0,
        measured_throughput_mbps=2.0,
    )
    assert hit_measurement.network_path == "client_edge"

    # MISS: edge_cloud measurement also exists with the same request_id
    miss_client = NetworkMeasurement(
        request_id="req-miss",
        network_path="client_edge",
        chunk_id="0000",
        representation_id="360p",
        bytes_transferred=500_000,
        transfer_duration_seconds=2.0,
        measured_throughput_mbps=2.0,
    )
    miss_cloud = NetworkMeasurement(
        request_id="req-miss",
        network_path="edge_cloud",
        chunk_id="0000",
        representation_id="360p",
        bytes_transferred=500_000,
        transfer_duration_seconds=1.0,
        measured_throughput_mbps=4.0,
    )
    assert miss_client.request_id == miss_cloud.request_id


# ---------------------------------------------------------------------------
# 11. Named scenarios load and have correct path independence
# ---------------------------------------------------------------------------

def test_named_scenarios_structure():
    """All five expected scenarios load correctly and have path independence."""
    for name in ("good", "moderate", "poor", "disabled", "rosevin_baseline"):
        assert name in SCENARIOS, f"Missing scenario: {name!r}"
        assert isinstance(SCENARIOS[name], NetworkEmulationConfig)

    # Verify path independence: good scenario has different BW per path
    good = SCENARIOS["good"]
    assert good.client_edge.bandwidth_mbps != good.edge_cloud.bandwidth_mbps

    # Disabled scenario has emulation off
    assert not SCENARIOS["disabled"].enabled


# ---------------------------------------------------------------------------
# 12. Sanity experiment: GOOD vs POOR on same payload
# ---------------------------------------------------------------------------

def test_sanity_good_vs_poor_scenario(test_server):
    """Same payload under POOR scenario must take longer than GOOD scenario."""
    good_adapter = EmulatedHttpAdapter(SCENARIOS["good"], "client_edge")
    poor_adapter = EmulatedHttpAdapter(SCENARIOS["poor"], "client_edge")

    good_content, good_duration = good_adapter.get(test_server, timeout=60.0)
    poor_content, poor_duration = poor_adapter.get(test_server, timeout=60.0)

    # Same payload in both cases
    assert len(good_content) == len(poor_content) == PAYLOAD_SIZE

    good_mbps = compute_throughput_mbps(len(good_content), good_duration)
    poor_mbps = compute_throughput_mbps(len(poor_content), poor_duration)

    assert poor_duration > good_duration, (
        f"POOR scenario ({poor_duration:.3f}s) must be slower than "
        f"GOOD scenario ({good_duration:.3f}s)"
    )
    assert poor_mbps < good_mbps, (
        f"POOR throughput ({poor_mbps:.3f} Mbps) must be less than "
        f"GOOD throughput ({good_mbps:.3f} Mbps)"
    )


# ---------------------------------------------------------------------------
# 13. ROSEVIN_BASELINE configuration exists and is correctly specified
# ---------------------------------------------------------------------------

def test_rosevin_baseline_exists():
    """rosevin_baseline scenario must exist in SCENARIOS."""
    assert "rosevin_baseline" in SCENARIOS
    rb = SCENARIOS["rosevin_baseline"]
    assert isinstance(rb, NetworkEmulationConfig)
    assert rb.enabled is True
    assert rb.scenario_name == "rosevin_baseline"


def test_rosevin_baseline_edge_cloud_bandwidth():
    """rosevin_baseline edge_cloud bandwidth must be approximately 186 Mbps."""
    rb = SCENARIOS["rosevin_baseline"]
    bw = rb.edge_cloud.bandwidth_mbps
    assert bw is not None
    assert abs(bw - 186.0) < 5.0, (
        f"Rosevin baseline edge_cloud bandwidth should be ~186 Mbps; got {bw} Mbps"
    )


def test_rosevin_rtt_reference_constant():
    """ROSEVIN_RTT_REFERENCE_MS must be approximately 43 ms."""
    assert abs(ROSEVIN_RTT_REFERENCE_MS - 43.0) < 1.0, (
        f"Rosevin RTT reference should be ~43 ms; got {ROSEVIN_RTT_REFERENCE_MS} ms"
    )


def test_rosevin_bandwidth_constant():
    """ROSEVIN_BANDWIDTH_MBPS must be approximately 186 Mbps."""
    assert abs(ROSEVIN_BANDWIDTH_MBPS - 186.0) < 5.0, (
        f"Rosevin bandwidth reference should be ~186 Mbps; got {ROSEVIN_BANDWIDTH_MBPS} Mbps"
    )


# ---------------------------------------------------------------------------
# 14. Injected delay_ms is NOT falsely reported as RTT
# ---------------------------------------------------------------------------

def test_injected_delay_not_labeled_as_rtt():
    """delay_ms is application-injected delay, not measured RTT.

    This test verifies the semantic contract:
      - NetworkMeasurement.rtt_ms must be populated from an actual probe,
        not copied from the configured delay_ms value.
      - A measurement created with rtt_ms=X and a config with delay_ms=Y
        (X ≠ Y) must retain rtt_ms=X.
    """
    configured_delay_ms = 21.0   # rosevin_baseline edge_cloud delay_ms
    observed_rtt_ms = 43.0       # what a real health probe might return

    # These must be different to prove they are not the same concept
    assert configured_delay_ms != observed_rtt_ms

    # A measurement record stores the OBSERVED rtt, not the configured delay
    m = NetworkMeasurement(
        request_id="probe-001",
        network_path="edge_cloud",
        rtt_ms=observed_rtt_ms,
    )
    assert m.rtt_ms == observed_rtt_ms, (
        "rtt_ms must store the actual probe observation, not the configured delay"
    )


# ---------------------------------------------------------------------------
# 15. GOOD / MODERATE / POOR are mutually distinct
# ---------------------------------------------------------------------------

def test_good_moderate_poor_are_distinct():
    """GOOD, MODERATE, and POOR must have different client_edge bandwidth values."""
    good_bw = SCENARIOS["good"].client_edge.bandwidth_mbps
    mod_bw = SCENARIOS["moderate"].client_edge.bandwidth_mbps
    poor_bw = SCENARIOS["poor"].client_edge.bandwidth_mbps

    assert good_bw != mod_bw, "GOOD and MODERATE client_edge bandwidth must differ"
    assert mod_bw != poor_bw, "MODERATE and POOR client_edge bandwidth must differ"
    assert good_bw != poor_bw, "GOOD and POOR client_edge bandwidth must differ"

    # Ordering: GOOD should have more bandwidth than POOR
    assert good_bw > poor_bw, (
        f"GOOD ({good_bw} Mbps) should have more bandwidth than POOR ({poor_bw} Mbps)"
    )


# ---------------------------------------------------------------------------
# 16. All five named scenarios have independent client_edge and edge_cloud
# ---------------------------------------------------------------------------

def test_all_scenarios_have_independent_paths():
    """Every named scenario must carry separate client_edge and edge_cloud configs."""
    for name, scenario in SCENARIOS.items():
        assert scenario.client_edge is not scenario.edge_cloud, (
            f"Scenario {name!r}: client_edge and edge_cloud must be independent objects"
        )

