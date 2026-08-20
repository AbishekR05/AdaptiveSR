import time
import requests
import json
import logging
import argparse
from pathlib import Path

from adaptive_sr.shared.config import EDGE_URL

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("ClientPlayer")

class ClientPlayer:
    def __init__(self, edge_url: str = EDGE_URL, video_id: str = "sample", representation_id: str = "360p"):
        self.edge_url = edge_url
        self.video_id = video_id
        self.representation_id = representation_id
        
        self.buffer_seconds = 0.0
        self.stall_count = 0
        self.download_dir = Path(__file__).resolve().parent / "player" / "downloads" / video_id / representation_id
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def fetch_manifest(self) -> dict:
        url = f"{self.edge_url}/videos/{self.video_id}/manifest"
        logger.info(f"Fetching manifest from Edge: {url}")
        response = requests.get(url, timeout=5.0)
        response.raise_for_status()
        return response.json()

    def play_video(self):
        manifest = self.fetch_manifest()
        logger.info(f"Successfully retrieved manifest for video '{self.video_id}'. Total duration: {manifest['duration']}s")
        
        # Verify selected representation exists
        available_reprs = [r["representation_id"] for r in manifest["representations"]]
        if self.representation_id not in available_reprs:
            logger.warning(f"Representation '{self.representation_id}' not found in manifest. Available options: {available_reprs}")
            self.representation_id = available_reprs[0]
            logger.info(f"Auto-selected representation: {self.representation_id}")
            
        logger.info(f"Starting playback simulation for representation: {self.representation_id}...")
        
        for chunk in manifest["chunks"]:
            chunk_id = chunk["chunk_id"]
            chunk_duration = chunk["duration"]
            
            # Measure Client -> Edge RTT independently
            client_rtt = None
            try:
                t_rtt_start = time.monotonic()
                ping_resp = requests.get(f"{self.edge_url}/health", timeout=2.0)
                ping_resp.raise_for_status()
                client_rtt = time.monotonic() - t_rtt_start
            except Exception as e:
                logger.warning(f"Failed to measure Client-to-Edge RTT: {e}")
                
            # Start timer for chunk request
            t_start = time.monotonic()
            url = f"{self.edge_url}/videos/{self.video_id}/chunks/{chunk_id}"
            params = {"representation_id": self.representation_id}
            
            # HTTP chunk request
            response = requests.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            
            t_end = time.monotonic()
            download_duration = t_end - t_start
            
            # Read telemetry from headers
            request_id = response.headers.get("X-Request-ID", "N/A")
            cache_hit_str = response.headers.get("X-Cache", "MISS")
            cache_hit = (cache_hit_str == "HIT")
            
            # Write chunk locally to demonstrate file transport integrity
            chunk_data = response.content
            bytes_received = len(chunk_data)
            
            output_file = self.download_dir / f"{chunk_id}.mp4"
            with open(output_file, "wb") as f:
                f.write(chunk_data)
                
            # Verify file integrity
            assert output_file.exists() and output_file.stat().st_size == bytes_received, "Integrity verification failed!"
            
            # Throughput calculation (megabits per second)
            measured_throughput_mbps = (bytes_received * 8) / (download_duration * 1000000.0) if download_duration > 0 else 0.0
            
            # Update mathematical buffer
            buffer_before = self.buffer_seconds
            
            # Deplete buffer during download time
            self.buffer_seconds = max(0.0, self.buffer_seconds - download_duration)
            
            stalled = False
            if self.buffer_seconds <= 0.0 and buffer_before > 0.0:
                self.stall_count += 1
                stalled = True
            elif buffer_before == 0.0:
                # Playback was already stalled at start of download
                self.stall_count += 1
                stalled = True
                
            # Calculate stall duration for this chunk request
            stall_duration_seconds = max(0.0, download_duration - buffer_before)
                
            # Replenish buffer with new chunk duration
            self.buffer_seconds += chunk_duration
            buffer_after = self.buffer_seconds
            
            # Client-side Telemetry Log
            telemetry = {
                "request_id": request_id,
                "chunk_id": chunk_id,
                "download_duration_s": download_duration,
                "bytes_received": bytes_received,
                "measured_throughput_mbps": measured_throughput_mbps,
                "buffer_before_s": buffer_before,
                "buffer_after_s": buffer_after,
                "cache_hit": cache_hit,
                "stalled": stalled,
                "total_stalls": self.stall_count,
                "stall_duration_seconds": stall_duration_seconds,
                "RTT": client_rtt
            }
            
            logger.info(json.dumps({
                "event": "client_telemetry",
                "telemetry": telemetry
            }))
            
            # Sleep slightly to simulate real-time playback consumption between requests
            # Under DASH, the player plays the current chunk, so we sleep for chunk_duration
            # to let the buffer deplete naturally in simulated time.
            time.sleep(0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdaptiveSR Player Client Emulator")
    parser.add_argument("--video", default="sample", help="Video ID to play")
    parser.add_argument("--repr", default="360p", help="Target representation ID (360p/720p)")
    parser.add_argument("--edge", default=EDGE_URL, help="Edge Server API base URL")
    args = parser.parse_args()
    
    player = ClientPlayer(edge_url=args.edge, video_id=args.video, representation_id=args.repr)
    try:
        player.play_video()
        logger.info("Playback simulation completed successfully.")
    except Exception as e:
        logger.error(f"Playback error: {e}")
