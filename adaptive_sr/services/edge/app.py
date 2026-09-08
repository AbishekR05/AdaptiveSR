import uvicorn
import requests
import time
import uuid
import json
import logging
from typing import Optional
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
def get_chunk(
    video_id: str,
    chunk_id: str,
    representation_id: str = Query(..., description="Target video quality representation"),
    sr_requested: bool = Query(False, description="Whether SR enhancement is requested"),
    model_id: Optional[str] = Query(None, description="SR model identifier (e.g. 'tinysr', 'real_esrgan')"),
    scale: int = Query(2, description="SR scale factor"),
    device: Optional[str] = Query(None, description="Execution device ('cpu' or 'cuda')"),
):
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
    
    # Check base chunk cache
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

    # Safely unwrap parameters if invoked directly as a Python function in tests
    sr_requested_val = sr_requested.default if hasattr(sr_requested, "default") else bool(sr_requested)
    model_id_val = model_id.default if hasattr(model_id, "default") else model_id
    scale_val = scale.default if hasattr(scale, "default") else scale
    device_val = device.default if hasattr(device, "default") else device

    # Determine if SR processing is active
    is_sr_active = bool(sr_requested_val) or (model_id_val is not None)
    sr_status = "not_requested"
    sr_processing_time = 0.0
    effective_model_id = model_id_val
    effective_device = device_val
    effective_scale = scale_val if scale_val is not None else 2
    response_file_path = cached_path
    media_type = "video/mp4"
    input_resolution = None
    output_resolution = None

    if is_sr_active:
        if not effective_model_id:
            effective_model_id = "tinysr"  # Default registered SR model if unspecified

        import torch
        if not effective_device:
            effective_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Requirement 6: Do NOT silently fall back from GPU to CPU
        if effective_device.lower().startswith("cuda") and not torch.cuda.is_available():
            logger.error("CUDA requested for Remote SR but CUDA is unavailable on Edge.")
            telemetry_failed = {
                "request_id": request_id,
                "video_id": video_id,
                "chunk_id": chunk_id,
                "representation_id": representation_id,
                "sr_requested": True,
                "sr_status": "failed",
                "error": "CUDA unavailable on Edge host"
            }
            logger.info(json.dumps({"event": "edge_telemetry", "cluster_id": CLUSTER_ID, "edge_id": EDGE_ID, "telemetry": telemetry_failed}))
            raise HTTPException(status_code=500, detail="CUDA requested for Remote SR but CUDA is unavailable on Edge.")

        # Check SR adapter availability
        try:
            from adaptive_sr.benchmarking.adapters.registry import get_adapter
            adapter = get_adapter(effective_model_id)
        except Exception as e:
            logger.error(f"Failed to resolve SR model adapter '{effective_model_id}': {e}")
            raise HTTPException(status_code=400, detail=f"Invalid or unavailable SR model '{effective_model_id}'")

        # Check SR cache key
        sr_cache_key = f"{video_id}__{chunk_id}__{representation_id}__sr_{effective_model_id}_x{effective_scale}_{effective_device}.mp4"
        sr_cached_path = cache.get(sr_cache_key)

        if sr_cached_path is not None:
            response_file_path = sr_cached_path
            sr_status = "executed"
        else:
            t_sr_start = time.monotonic()
            try:
                import cv2
                import numpy as np

                # Read source video frames
                cap = cv2.VideoCapture(str(cached_path))
                frames = []
                fps_val = cap.get(cv2.CAP_PROP_FPS)
                if not fps_val or fps_val <= 0:
                    fps_val = 30.0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()

                if not frames:
                    # Single image or empty video check: try cv2.imread
                    img = cv2.imread(str(cached_path))
                    if img is not None:
                        frames = [img]
                        media_type = "image/png"

                if not frames:
                    raise RuntimeError(f"Could not decode frames from chunk file: {cached_path}")

                in_h, in_w, _ = frames[0].shape
                input_resolution = [in_h, in_w]

                # Initialize and process through SR adapter
                adapter.initialize(device=effective_device, scale=effective_scale)
                enhanced_frames = adapter.process(frames, scale=effective_scale)
                adapter.close()

                if not isinstance(enhanced_frames, list):
                    enhanced_frames = [enhanced_frames]

                out_h, out_w, _ = enhanced_frames[0].shape
                output_resolution = [out_h, out_w]

                # Encode enhanced frames
                if media_type == "image/png" or len(enhanced_frames) == 1 and not str(cached_path).endswith(".mp4"):
                    _, img_encoded = cv2.imencode(".png", enhanced_frames[0])
                    sr_cached_path = cache.put(sr_cache_key, img_encoded.tobytes())
                    media_type = "image/png"
                else:
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_out:
                        tmp_out_path = tmp_out.name

                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out_writer = cv2.VideoWriter(tmp_out_path, fourcc, fps_val, (out_w, out_h))
                    for f in enhanced_frames:
                        out_writer.write(f)
                    out_writer.release()

                    with open(tmp_out_path, "rb") as f_in:
                        sr_cached_path = cache.put(sr_cache_key, f_in.read())

                    try:
                        os.remove(tmp_out_path)
                    except Exception:
                        pass
                    media_type = "video/mp4"

                sr_processing_time = time.monotonic() - t_sr_start
                response_file_path = sr_cached_path
                sr_status = "executed"

            except Exception as e:
                logger.error(f"SR processing failed at Edge for model '{effective_model_id}': {e}")
                telemetry_err = {
                    "request_id": request_id,
                    "video_id": video_id,
                    "chunk_id": chunk_id,
                    "representation_id": representation_id,
                    "sr_requested": True,
                    "sr_status": "failed",
                    "error": str(e)
                }
                logger.info(json.dumps({"event": "edge_telemetry", "cluster_id": CLUSTER_ID, "edge_id": EDGE_ID, "telemetry": telemetry_err}))
                raise HTTPException(status_code=500, detail=f"SR inference failed at Edge: {e}")

    t_end = time.monotonic()
    edge_processing_time = t_end - t_start
    bytes_sent = response_file_path.stat().st_size
    
    # Edge Telemetry Record
    telemetry = {
        "request_id": request_id,
        "video_id": video_id,
        "chunk_id": chunk_id,
        "representation_id": representation_id,
        "target_representation_id": representation_id,
        "base_representation_id": representation_id,
        "cache_hit": cache_hit,
        "cloud_fetch_time": cloud_fetch_time,
        "edge_processing_time": edge_processing_time,
        "response_time": edge_processing_time,
        "bytes_sent": bytes_sent,
        "rtt": cloud_rtt,
        "sr_requested": is_sr_active,
        "sr_status": sr_status,
        "sr_model_id": effective_model_id if is_sr_active else None,
        "sr_scale": scale if is_sr_active else None,
        "sr_device": effective_device if is_sr_active else None,
        "sr_processing_time": sr_processing_time,
        "input_resolution": input_resolution,
        "output_resolution": output_resolution,
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
        "X-Edge-Cloud-RTT": f"{cloud_rtt:.6f}" if cloud_rtt is not None else "N/A",
        "X-SR-Requested": str(is_sr_active),
        "X-SR-Status": sr_status,
        "X-SR-Model": str(effective_model_id) if is_sr_active else "N/A",
        "X-SR-Scale": str(scale) if is_sr_active else "N/A",
        "X-SR-Device": str(effective_device) if is_sr_active else "N/A",
        "X-SR-Processing-Time": f"{sr_processing_time:.6f}" if is_sr_active else "0.000000"
    }
    
    return FileResponse(
        response_file_path,
        media_type=media_type,
        filename=f"{chunk_id}.mp4",
        headers=headers
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=EDGE_PORT)
