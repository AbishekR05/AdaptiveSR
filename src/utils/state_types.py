from dataclasses import dataclass

@dataclass
class DeviceState:
    cpu: float          # 0-1
    gpu: float | None    # 0-1, None if no GPU
    ram: float           # 0-1 used
    battery: float | None  # 0-1, None if desktop/no battery
    charging: bool | None
    temperature: float | None  # 0-1 normalized, None if unavailable
    fps: float            # current measured processing fps

@dataclass
class SceneDescriptor:
    motion: float
    texture: float
    edges: float
    complexity: float
    degradation: float = 0.0

@dataclass
class Decision:
    model: str
    scale: int
    reason: str
    priority: str = "medium"
