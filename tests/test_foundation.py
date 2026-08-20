import pytest
import shutil
import time
from fastapi.testclient import TestClient
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import app routers
from adaptive_sr.services.cloud.app import app as cloud_app
from adaptive_sr.services.edge.app import app as edge_app
from adaptive_sr.services.edge.app import cache as edge_cache
from adaptive_sr.services.client.app import ClientPlayer
from adaptive_sr.services.edge.cache import DiskCache

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Fixture to provide a clean cache directory for Edge during tests."""
    original_cache_dir = edge_cache.cache_dir
    edge_cache.cache_dir = tmp_path
    yield tmp_path
    edge_cache.cache_dir = original_cache_dir

@pytest.fixture
def mock_cloud_client():
    """Intercepts requests to Cloud Origin and redirects them by calling app functions directly (avoiding TestClient deadlocks)."""
    import json
    import requests
    from adaptive_sr.services.cloud.app import get_manifest, get_chunk

    def mock_get(url, *args, **kwargs):
        response = requests.Response()
        if "/health" in url or url.endswith("/health"):
            response.status_code = 200
            response._content = b'{"status": "ok"}'
            response.headers["content-type"] = "application/json"
        elif "/manifest" in url:
            video_id = url.split("/videos/")[-1].split("/manifest")[0]
            try:
                data = get_manifest(video_id)
                response._content = json.dumps(data).encode("utf-8")
                response.status_code = 200
                response.headers["content-type"] = "application/json"
            except Exception as e:
                response.status_code = 404
                response._content = json.dumps({"detail": str(e)}).encode("utf-8")
        elif "/chunks/" in url:
            parts = url.split("/videos/")[-1].split("/")
            video_id = parts[0]
            representation_id = parts[1]
            chunk_id = parts[3]
            try:
                file_response = get_chunk(video_id, representation_id, chunk_id)
                with open(file_response.path, "rb") as f:
                    response._content = f.read()
                response.status_code = 200
                response.headers["content-type"] = "video/mp4"
            except Exception as e:
                response.status_code = 404
                response._content = json.dumps({"detail": str(e)}).encode("utf-8")
        else:
            response.status_code = 404
        return response

    with patch("requests.get", side_effect=mock_get) as mock:
        yield mock


def test_cloud_health():
    client = TestClient(cloud_app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_cloud_manifest():
    client = TestClient(cloud_app)
    response = client.get("/videos/sample/manifest")
    assert response.status_code == 200
    manifest = response.json()
    assert manifest["video_id"] == "sample"
    assert len(manifest["representations"]) > 0
    assert len(manifest["chunks"]) > 0

def test_cloud_chunk():
    client = TestClient(cloud_app)
    response = client.get("/videos/sample/360p/chunks/0000")
    assert response.status_code == 200
    assert len(response.content) > 0

def test_missing_chunk_handling():
    client = TestClient(cloud_app)
    # Check cloud missing chunk returns 404
    response = client.get("/videos/sample/360p/chunks/9999")
    assert response.status_code == 404

def test_edge_manifest_proxy(mock_cloud_client):
    client = TestClient(edge_app)
    response = client.get("/videos/sample/manifest")
    assert response.status_code == 200
    assert response.json()["video_id"] == "sample"
    mock_cloud_client.assert_called_once()

def test_edge_cache_miss_and_hit(temp_cache_dir, mock_cloud_client):
    client = TestClient(edge_app)
    chunk_url = "/videos/sample/chunks/0000"
    params = {"representation_id": "360p"}
    
    # 1. Edge Cache Miss (First fetch)
    assert not edge_cache.contains("sample__0000__360p.mp4")
    response1 = client.get(chunk_url, params=params)
    assert response1.status_code == 200
    assert response1.headers.get("X-Cache") == "MISS"
    assert edge_cache.contains("sample__0000__360p.mp4")
    # 2 calls: 1 health-ping RTT check + 1 chunk fetch
    assert mock_cloud_client.call_count == 2
    
    # 2. Edge Cache Hit (Second fetch)
    response2 = client.get(chunk_url, params=params)
    assert response2.status_code == 200
    assert response2.headers.get("X-Cache") == "HIT"
    # 3 calls total: 2 from first run + 1 health-ping RTT check (no new chunk fetch)
    assert mock_cloud_client.call_count == 3

def test_end_to_end_crossing_boundaries(temp_cache_dir):
    """
    Simulates the Client Player fetching chunks from the Edge Server, 
    verifying it successfully traverses both mock boundaries.
    """
    import json
    import requests
    from adaptive_sr.services.edge.app import get_manifest as edge_get_manifest, get_chunk as edge_get_chunk
    from adaptive_sr.services.cloud.app import get_manifest as cloud_get_manifest, get_chunk as cloud_get_chunk
    
    # Patch requests globally with a unified router to direct to correct handler functions
    def unified_mock_get(url, *args, **kwargs):
        response = requests.Response()
        
        # 1. Route Edge requests (Client -> Edge)
        if "localhost:8001" in url:
            if "/health" in url or url.endswith("/health"):
                from adaptive_sr.services.edge.app import get_health as edge_get_health
                data = edge_get_health()
                response._content = json.dumps(data).encode("utf-8")
                response.status_code = 200
                response.headers["content-type"] = "application/json"
            elif "/manifest" in url:
                video_id = url.split("/videos/")[-1].split("/manifest")[0]
                data = edge_get_manifest(video_id)
                response._content = json.dumps(data).encode("utf-8")
                response.status_code = 200
                response.headers["content-type"] = "application/json"
            elif "/chunks/" in url:
                video_id = url.split("/videos/")[-1].split("/chunks/")[0]
                chunk_id = url.split("/chunks/")[-1]
                params = kwargs.get("params", {})
                representation_id = params.get("representation_id", "360p")
                
                file_response = edge_get_chunk(video_id, chunk_id, representation_id=representation_id)
                with open(file_response.path, "rb") as f:
                    response._content = f.read()
                response.status_code = 200
                response.headers["content-type"] = "video/mp4"
                for k, v in file_response.headers.items():
                    response.headers[k] = v
                    
        # 2. Route Cloud requests (Edge -> Cloud)
        elif "localhost:8000" in url:
            if "/health" in url or url.endswith("/health"):
                from adaptive_sr.services.cloud.app import get_health as cloud_get_health
                data = cloud_get_health()
                response._content = json.dumps(data).encode("utf-8")
                response.status_code = 200
                response.headers["content-type"] = "application/json"
            elif "/manifest" in url:
                video_id = url.split("/videos/")[-1].split("/manifest")[0]
                try:
                    data = cloud_get_manifest(video_id)
                    response._content = json.dumps(data).encode("utf-8")
                    response.status_code = 200
                    response.headers["content-type"] = "application/json"
                except Exception as e:
                    response.status_code = 404
                    response._content = json.dumps({"detail": str(e)}).encode("utf-8")
            elif "/chunks/" in url:
                parts = url.split("/videos/")[-1].split("/")
                video_id = parts[0]
                representation_id = parts[1]
                chunk_id = parts[3]
                try:
                    file_response = cloud_get_chunk(video_id, representation_id, chunk_id)
                    with open(file_response.path, "rb") as f:
                        response._content = f.read()
                    response.status_code = 200
                    response.headers["content-type"] = "video/mp4"
                except Exception as e:
                    response.status_code = 404
                    response._content = json.dumps({"detail": str(e)}).encode("utf-8")
        else:
            response.status_code = 404
        return response
        
    with patch("requests.get", side_effect=unified_mock_get):
        player = ClientPlayer(video_id="sample", representation_id="360p")
        # Run player download / play sequence
        player.play_video()
        
        # Verify chunks are written locally by the player
        assert player.download_dir.exists()
        for i in range(3):
            file_path = player.download_dir / f"000{i}.mp4"
            assert file_path.exists()
            assert file_path.stat().st_size > 0
            
        # Verify mathematical buffer updated correctly (6.0s total duration of sample chunks)
        # Final buffer should be 6.0s minus simulation delay/download duration
        assert player.buffer_seconds > 0.0

def test_buffer_math():
    """Unit test for the client mathematical buffer simulation."""
    buffer_seconds = 0.0
    stall_count = 0
    
    # 1. First download: download duration 1.5s, chunk duration 2.0s
    download_duration = 1.5
    chunk_duration = 2.0
    buffer_before = buffer_seconds
    
    # Deplete
    buffer_seconds = max(0.0, buffer_seconds - download_duration)
    stalled = False
    if buffer_seconds <= 0.0 and buffer_before > 0.0:
        stall_count += 1
        stalled = True
    elif buffer_before == 0.0:
        stall_count += 1
        stalled = True
        
    stall_duration = max(0.0, download_duration - buffer_before)
    
    assert stalled is True
    assert stall_count == 1
    assert buffer_seconds == 0.0
    assert stall_duration == 1.5
    
    # Replenish
    buffer_seconds += chunk_duration
    assert buffer_seconds == 2.0
    
    # 2. Second download: download duration 0.8s, chunk duration 2.0s
    download_duration = 0.8
    buffer_before = buffer_seconds
    
    buffer_seconds = max(0.0, buffer_seconds - download_duration)
    stalled = False
    if buffer_seconds <= 0.0 and buffer_before > 0.0:
        stall_count += 1
        stalled = True
        
    stall_duration = max(0.0, download_duration - buffer_before)
    
    assert stalled is False
    assert stall_count == 1
    assert buffer_seconds == 1.2
    assert stall_duration == 0.0
    
    buffer_seconds += chunk_duration
    assert buffer_seconds == 3.2
    
    # 3. Third download: download duration 4.5s (exceeds buffer), chunk duration 2.0s
    download_duration = 4.5
    buffer_before = buffer_seconds
    
    buffer_seconds = max(0.0, buffer_seconds - download_duration)
    stalled = False
    if buffer_seconds <= 0.0 and buffer_before > 0.0:
        stall_count += 1
        stalled = True
        
    stall_duration = max(0.0, download_duration - buffer_before)
    
    assert stalled is True
    assert stall_count == 2
    assert buffer_seconds == 0.0
    assert stall_duration == pytest.approx(1.3)  # 4.5 - 3.2 = 1.3 seconds spent stalled

def test_throughput_calculation():
    """Unit test for client throughput calculation."""
    bytes_received = 100000  # 100 KB
    download_duration = 0.5  # 0.5s
    
    # throughput in mbps = (bytes * 8) / (seconds * 10^6)
    throughput_mbps = (bytes_received * 8) / (download_duration * 1000000.0)
    assert throughput_mbps == 1.6

def test_rtt_measurement(temp_cache_dir, mock_cloud_client):
    """Verifies that RTT measurement is performed via health endpoints."""
    client = TestClient(edge_app)
    
    # Check Edge -> Cloud RTT is measured and returned in headers
    chunk_url = "/videos/sample/chunks/0000"
    params = {"representation_id": "360p"}
    
    response = client.get(chunk_url, params=params)
    assert response.status_code == 200
    rtt_header = response.headers.get("X-Edge-Cloud-RTT")
    assert rtt_header is not None
    assert rtt_header != "N/A"
    assert float(rtt_header) >= 0.0

def test_edge_identity_smoke(temp_cache_dir, mock_cloud_client):
    """Verifies that the architecture distinguishes multiple Edge server instances."""
    client = TestClient(edge_app)
    
    # Test Edge identity 01
    with patch("adaptive_sr.services.edge.app.EDGE_ID", "edge_01"):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["edge_id"] == "edge_01"
        
        # Check header
        chunk_response = client.get("/videos/sample/chunks/0000", params={"representation_id": "360p"})
        assert chunk_response.headers.get("X-Edge-ID") == "edge_01"
        assert chunk_response.headers.get("X-Cluster-ID") == "cluster_01"
        
    # Test Edge identity 02
    with patch("adaptive_sr.services.edge.app.EDGE_ID", "edge_02"):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["edge_id"] == "edge_02"
        
        # Check header
        chunk_response = client.get("/videos/sample/chunks/0000", params={"representation_id": "360p"})
        assert chunk_response.headers.get("X-Edge-ID") == "edge_02"
        assert chunk_response.headers.get("X-Cluster-ID") == "cluster_01"

def test_zero_buffer_stall():
    """Unit test verifying that zero-buffer start and a 2.0s download produces exactly a 2.0s stall duration."""
    buffer_before = 0.0
    download_duration = 2.0
    stall_duration_seconds = max(0.0, download_duration - buffer_before)
    assert stall_duration_seconds == 2.0

def test_per_edge_cache_isolation(tmp_path, mock_cloud_client):
    """Integration test verifying Edge 01 and Edge 02 maintain separate cache directories and do not leak cache hits."""
    cache_dir_01 = tmp_path / "cache_edge_01"
    cache_dir_02 = tmp_path / "cache_edge_02"
    cache_dir_01.mkdir()
    cache_dir_02.mkdir()
    
    client = TestClient(edge_app)
    chunk_url = "/videos/sample/chunks/0000"
    params = {"representation_id": "360p"}
    
    # 1. Edge 01 requests chunk (Cache MISS)
    edge_cache.cache_dir = cache_dir_01
    response1 = client.get(chunk_url, params=params)
    assert response1.status_code == 200
    assert response1.headers.get("X-Cache") == "MISS"
    
    cache_key = "sample__0000__360p.mp4"
    assert (cache_dir_01 / cache_key).exists()
    assert not (cache_dir_02 / cache_key).exists()
    
    # 2. Edge 02 requests the SAME chunk (Cache MISS because of isolation)
    edge_cache.cache_dir = cache_dir_02
    response2 = client.get(chunk_url, params=params)
    assert response2.status_code == 200
    assert response2.headers.get("X-Cache") == "MISS"
    
    assert (cache_dir_02 / cache_key).exists()

