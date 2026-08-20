import uvicorn
import requests
import time
import uuid
import json
import logging
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pathlib import Path

from adaptive_sr.shared.config import CLOUD_URL, EDGE_PORT, CLUSTER_ID, EDGE_ID
from adaptive_sr.services.edge.cache import DiskCache

app = FastAPI(title="AdaptiveSR - Edge Server Service")
cache = DiskCache()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("EdgeServer")

@app.get("/health")
def get_health():
    return {
        "status": "ok",
        "cluster_id": CLUSTER_ID,
        "edge_id": EDGE_ID
    }

@app.get("/videos/{video_id}/manifest")
def get_manifest(video_id: str):
    url = f"{CLOUD_URL}/videos/{video_id}/manifest"
    try:
        response = requests.get(url, timeout=5.0)
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Manifest not found on Cloud: {video_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch manifest from Cloud: {e}")

@app.get("/videos/{video_id}/chunks/{chunk_id}")
def get_chunk(video_id: str, chunk_id: str, representation_id: str = Query(..., description="Target video quality representation")):
    request_id = str(uuid.uuid4())
    t_start = time.monotonic()
    
    # Measure Edge -> Cloud RTT independently
    cloud_rtt = None
    try:
        t_rtt_start = time.monotonic()
        ping_resp = requests.get(f"{CLOUD_URL}/health", timeout=2.0)
        ping_resp.raise_for_status()
        cloud_rtt = time.monotonic() - t_rtt_start
    except Exception as e:
        logger.warning(f"Failed to measure Edge-to-Cloud RTT: {e}")
        
    cache_key = f"{video_id}__{chunk_id}__{representation_id}.mp4"
    cloud_fetch_time = 0.0
    cache_hit = False
    
    # Check cache
    cached_path = cache.get(cache_key)
    if cached_path is not None:
        cache_hit = True
    else:
        # Cache miss: fetch from cloud
        url = f"{CLOUD_URL}/videos/{video_id}/{representation_id}/chunks/{chunk_id}"
        t_cloud_start = time.monotonic()
        try:
            response = requests.get(url, timeout=10.0)
            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Segment not found in Cloud.")
            response.raise_for_status()
            
            chunk_data = response.content
            cloud_fetch_time = time.monotonic() - t_cloud_start
            
            # Put in cache
            cached_path = cache.put(cache_key, chunk_data)
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=502, detail=f"Error fetching chunk from Cloud: {e}")
            
    t_end = time.monotonic()
    edge_processing_time = t_end - t_start
    bytes_sent = cached_path.stat().st_size
    
    # Edge Telemetry Record
    telemetry = {
        "request_id": request_id,
        "video_id": video_id,
        "chunk_id": chunk_id,
        "representation_id": representation_id,
        "target_representation_id": representation_id,  # Placeholder for future ABR/SR distinction
        "base_representation_id": representation_id,    # Placeholder for future ABR/SR distinction
        "cache_hit": cache_hit,
        "cloud_fetch_time": cloud_fetch_time,
        "edge_processing_time": edge_processing_time,
        "response_time": edge_processing_time,
        "bytes_sent": bytes_sent,
        "rtt": cloud_rtt,
        "sr_processing_time": 0.0  # Placeholder for future VSR
    }
    
    # Structured log output
    logger.info(json.dumps({
        "event": "edge_telemetry",
        "cluster_id": CLUSTER_ID,
        "edge_id": EDGE_ID,
        "telemetry": telemetry
    }))
    
    headers = {
        "X-Request-ID": request_id,
        "X-Cache": "HIT" if cache_hit else "MISS",
        "X-Cloud-Fetch-Time": f"{cloud_fetch_time:.6f}",
        "X-Edge-Processing-Time": f"{edge_processing_time:.6f}",
        "X-Cluster-ID": CLUSTER_ID,
        "X-Edge-ID": EDGE_ID,
        "X-Edge-Cloud-RTT": f"{cloud_rtt:.6f}" if cloud_rtt is not None else "N/A"
    }
    
    return FileResponse(
        cached_path,
        media_type="video/mp4",
        filename=f"{chunk_id}.mp4",
        headers=headers
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=EDGE_PORT)
