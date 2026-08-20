import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import json

from adaptive_sr.shared.config import CLOUD_STORAGE_DIR, CLOUD_PORT

app = FastAPI(title="AdaptiveSR - Cloud Video Origin Service")

@app.get("/health")
def get_health():
    return {"status": "ok"}

@app.get("/videos")
def list_videos():
    videos_dir = Path(CLOUD_STORAGE_DIR) / "videos"
    if not videos_dir.exists():
        return []
    # List subdirectories under videos
    return [d.name for d in videos_dir.iterdir() if d.is_dir()]

@app.get("/videos/{video_id}/manifest")
def get_manifest(video_id: str):
    manifest_path = Path(CLOUD_STORAGE_DIR) / "videos" / video_id / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"Manifest not found for video: {video_id}")
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/videos/{video_id}/{representation_id}/chunks/{chunk_id}")
def get_chunk(video_id: str, representation_id: str, chunk_id: str):
    chunk_path = Path(CLOUD_STORAGE_DIR) / "videos" / video_id / representation_id / f"{chunk_id}.mp4"
    if not chunk_path.exists():
        raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} for representation {representation_id} not found.")
    
    return FileResponse(chunk_path, media_type="video/mp4", filename=f"{chunk_id}.mp4")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=CLOUD_PORT)
