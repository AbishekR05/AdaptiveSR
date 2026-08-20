import os
import sys
import pytest
from pydantic import ValidationError
from datetime import datetime

# Ensure root of repository is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adaptive_sr.shared.schemas import (
    NetworkMeasurement,
    ClientTelemetry,
    EdgeTelemetry
)

def test_distinct_network_paths():
    """Verifies that client_edge and edge_cloud are distinct network paths."""
    m1 = NetworkMeasurement(
        request_id="req-1",
        network_path="client_edge",
        rtt_ms=10.0
    )
    m2 = NetworkMeasurement(
        request_id="req-2",
        network_path="edge_cloud",
        rtt_ms=25.0
    )
    assert m1.network_path == "client_edge"
    assert m2.network_path == "edge_cloud"

def test_rtt_and_transfer_duration_are_separate():
    """Verifies that RTT and transfer duration are separate quantities in the model."""
    m = NetworkMeasurement(
        request_id="req-123",
        network_path="client_edge",
        rtt_ms=15.0,  # Separate RTT (milliseconds)
        bytes_transferred=250000,
        transfer_duration_seconds=2.0,  # Separate transfer duration (seconds)
        measured_throughput_mbps=1.0  # 250000 * 8 / 2.0 / 1e6 = 1.0 Mbps
    )
    assert m.rtt_ms == 15.0
    assert m.transfer_duration_seconds == 2.0

def test_throughput_calculation_validation():
    """Verifies throughput calculation constraints: bytes * 8 / seconds / 1,000,000."""
    # Correct calculation: 250,000 bytes * 8 = 2,000,000 bits. 
    # Transferred in 2.0s -> 1,000,000 bps = 1.0 Mbps.
    m = NetworkMeasurement(
        request_id="req-4",
        network_path="client_edge",
        bytes_transferred=250000,
        transfer_duration_seconds=2.0,
        measured_throughput_mbps=1.0
    )
    assert m.measured_throughput_mbps == 1.0

    # Incorrect calculation must trigger ValidationError
    with pytest.raises(ValidationError) as exc_info:
        NetworkMeasurement(
            request_id="req-4",
            network_path="client_edge",
            bytes_transferred=250000,
            transfer_duration_seconds=2.0,
            measured_throughput_mbps=2.5  # Incorrect!
        )
    assert "Throughput mismatch" in str(exc_info.value)

def test_zero_byte_rtt_probes_no_throughput():
    """Verifies that zero-byte RTT probes do not fabricate or carry non-zero throughput."""
    # Valid: RTT probe with bytes=None or bytes=0, and no throughput
    m1 = NetworkMeasurement(
        request_id="req-ping",
        network_path="client_edge",
        rtt_ms=5.0,
        bytes_transferred=0
    )
    assert m1.measured_throughput_mbps is None

    # Invalid: RTT probe with bytes=0 but claiming non-zero throughput
    with pytest.raises(ValidationError) as exc_info:
        NetworkMeasurement(
            request_id="req-ping",
            network_path="client_edge",
            rtt_ms=5.0,
            bytes_transferred=0,
            measured_throughput_mbps=1.5  # Invalid!
        )
    assert "Zero-byte RTT probes must not fabricate a non-zero throughput" in str(exc_info.value)

def test_cache_hit_miss_delivery_paths():
    """
    Verifies cache HIT/MISS semantics.
    For a cache HIT: client_edge transfer exists, edge_cloud transfer is absent.
    For a cache MISS: both client_edge and edge_cloud transfers exist.
    """
    # 1. Cache HIT Simulation
    # Client-to-Edge transfer exists
    client_edge_hit = NetworkMeasurement(
        request_id="req-hit",
        network_path="client_edge",
        bytes_transferred=500000,
        transfer_duration_seconds=2.0,
        measured_throughput_mbps=2.0
    )
    # Edge-to-Cloud payload transfer is absent (since chunk is cached).
    # We do NOT represent it as 0-speed/duration traffic. We represent it by the absence of measurement,
    # or by explicit cache hit flags in telemetry schemas.
    assert client_edge_hit.network_path == "client_edge"
    
    # 2. Cache MISS Simulation
    # Both transfers exist
    client_edge_miss = NetworkMeasurement(
        request_id="req-miss",
        network_path="client_edge",
        bytes_transferred=500000,
        transfer_duration_seconds=2.0,
        measured_throughput_mbps=2.0
    )
    edge_cloud_miss = NetworkMeasurement(
        request_id="req-miss",
        network_path="edge_cloud",
        bytes_transferred=500000,
        transfer_duration_seconds=1.0,
        measured_throughput_mbps=4.0
    )
    assert client_edge_miss.request_id == edge_cloud_miss.request_id

def test_retains_identities():
    """Verifies that measurements retain request_id, network_path, and timestamp identities."""
    m = NetworkMeasurement(
        request_id="test-req-id-99",
        network_path="client_edge",
        rtt_ms=12.0
    )
    assert m.request_id == "test-req-id-99"
    assert m.network_path == "client_edge"
    assert m.timestamp.endswith("Z")
    
    # Verify ISO-8601 format
    try:
        datetime.fromisoformat(m.timestamp.replace("Z", "+00:00"))
    except ValueError:
        pytest.fail("Timestamp is not in valid ISO-8601 format.")

def test_step0_telemetry_compatibility():
    """Verifies that existing Step 0/0.1 telemetry schemas remain compatible and intact."""
    client_t = ClientTelemetry(
        request_start_time=1000.0,
        response_time=0.5,
        download_duration=0.4,
        bytes_received=50000,
        measured_throughput_mbps=1.0,
        RTT=0.005,
        buffer_before=1.5,
        buffer_after=3.1,
        cache_hit=True,
        stall_duration_seconds=0.0
    )
    edge_t = EdgeTelemetry(
        request_id="req-edge-test",
        video_id="sample",
        chunk_id="0001",
        representation_id="360p",
        cache_hit=False,
        cloud_fetch_time=0.2,
        edge_processing_time=0.01,
        response_time=0.22,
        bytes_sent=50000,
        rtt=0.010,
        sr_processing_time=0.0
    )
    assert client_t.measured_throughput_mbps == 1.0
    assert edge_t.rtt == 0.010

def test_rtt_only_measurement_accepts_null_chunk_and_repr():
    """Verifies that NetworkMeasurement accepts an RTT-only measurement with chunk_id=None and representation_id=None."""
    m = NetworkMeasurement(
        request_id="req-ping-2",
        network_path="client_edge",
        rtt_ms=15.0,
        chunk_id=None,
        representation_id=None
    )
    assert m.chunk_id is None
    assert m.representation_id is None

def test_chunk_transfer_accepts_chunk_and_repr_id():
    """Verifies that NetworkMeasurement accepts a chunk transfer with chunk_id and representation_id present."""
    m = NetworkMeasurement(
        request_id="req-transfer-2",
        network_path="client_edge",
        chunk_id="0004",
        representation_id="720p",
        bytes_transferred=250000,
        transfer_duration_seconds=2.0,
        measured_throughput_mbps=1.0
    )
    assert m.chunk_id == "0004"
    assert m.representation_id == "720p"

def test_request_id_remains_required():
    """Verifies that request_id remains required and validation fails if it's missing."""
    with pytest.raises(ValidationError):
        NetworkMeasurement(
            network_path="client_edge",
            rtt_ms=5.0
        )

def test_network_path_restricted_values():
    """Verifies that network_path remains restricted to 'client_edge' and 'edge_cloud'."""
    # Valid
    m1 = NetworkMeasurement(request_id="r1", network_path="client_edge")
    m2 = NetworkMeasurement(request_id="r2", network_path="edge_cloud")
    assert m1.network_path == "client_edge"
    assert m2.network_path == "edge_cloud"

    # Invalid values must raise ValidationError
    with pytest.raises(ValidationError):
        NetworkMeasurement(
            request_id="r3",
            network_path="invalid_path_name"
        )
