from pathlib import Path
from adaptive_sr.shared.config import EDGE_CACHE_DIR

class DiskCache:
    def __init__(self, cache_dir: Path = EDGE_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str) -> Path:
        return self.cache_dir / key

    def get(self, key: str) -> Path | None:
        """Returns the file path of the cached segment if it exists, otherwise None."""
        path = self._get_path(key)
        if path.exists() and path.is_file():
            return path
        return None

    def put(self, key: str, data: bytes) -> Path:
        """Saves segment bytes to disk and returns the file path."""
        path = self._get_path(key)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def contains(self, key: str) -> bool:
        """Checks if a cache key exists on disk."""
        path = self._get_path(key)
        return path.exists() and path.is_file()

    def clear(self):
        """Clears all cached segments on disk."""
        for item in self.cache_dir.iterdir():
            if item.is_file():
                item.unlink()
