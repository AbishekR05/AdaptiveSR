from dataclasses import dataclass

@dataclass
class DeviceState:
    cpu: float          # 0-1
    gpu: float | None    # 0-1, None if no GPU
    ram: float           # process RSS memory in MB
    system_ram: float    # 0-1 system RAM utilization
    battery: float | None  # 0-1, None if desktop/no battery
    charging: bool | None
    temperature: float | None  # 0-1 normalized, None if unavailable
    fps: float            # current measured processing fps

@dataclass
class SceneDescriptor:
    motion: float       # 0-1
    texture: float      # 0-1
    edges: float         # 0-1
    blur_clarity: float  # 0-1, higher = sharper
    complexity: float    # 0-1, final combined score

@dataclass
class Decision:
    model: str
    scale: int
    reason: str
    priority: str = "medium"
