import os
from pathlib import Path

# Workspace Root D:\Full Stack\AdaptiveSR
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent

CLOUD_URL = os.environ.get("CLOUD_URL", "http://localhost:8000")
EDGE_URL = os.environ.get("EDGE_URL", "http://localhost:8001")

CLUSTER_ID = os.environ.get("CLUSTER_ID", "cluster_01")
EDGE_ID = os.environ.get("EDGE_ID", "edge_01")

CLOUD_PORT = int(os.environ.get("CLOUD_PORT", "8000"))
EDGE_PORT = int(os.environ.get("EDGE_PORT", "8001"))

CLOUD_STORAGE_DIR = Path(os.environ.get("CLOUD_STORAGE_DIR", WORKSPACE_ROOT / "adaptive_sr" / "services" / "cloud" / "storage"))
EDGE_CACHE_DIR = Path(os.environ.get("EDGE_CACHE_DIR", WORKSPACE_ROOT / "adaptive_sr" / "services" / "edge" / "cache"))
