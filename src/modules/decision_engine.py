import os
import yaml
from src.utils.state_types import DeviceState, SceneDescriptor, Decision

class DecisionEngine:
    def __init__(self, config_path="configs/decision_config.yaml", ignore_device=False, ignore_scene=False):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
            
        self.thresholds = self.cfg["thresholds"]
        self.ignore_device = ignore_device
        self.ignore_scene = ignore_scene

    def decide(self, device: DeviceState, scene: SceneDescriptor) -> Decision:
        if self.ignore_device:
            device = DeviceState(cpu=0.10, gpu=0.10, ram=0.50, system_ram=0.50, battery=0.90, charging=True, temperature=0.30, fps=30.0)
        if self.ignore_scene:
            scene = SceneDescriptor(motion=0.0, texture=0.50, edges=0.50, blur_clarity=0.50, complexity=0.50)

        t = self.thresholds

        # Rule 0: Critical battery + simple frame -> skip enhancement completely to save power (passthrough)
        t_skip = t.get("skip_enhancement", {"battery": 0.10, "temperature": 0.85, "complexity": 0.15})
        if (device.battery is not None and device.battery < t_skip["battery"]) and \
           (scene.complexity < t_skip["complexity"]):
            return Decision(
                model="skip",
                scale=1,
                reason=f"critical battery ({device.battery:.2f} < {t_skip['battery']}) + trivial frame ({scene.complexity:.2f} < {t_skip['complexity']}), passthrough",
                priority="high"
            )

        # Rule 1: Critical device state -> always lightweight, regardless of scene
        if (device.battery is not None and device.battery < t["low_battery"]) and \
           (device.temperature is not None and device.temperature > t["high_temp"]):
            return Decision(
                model="tinysr",
                scale=2,
                reason=f"low battery ({device.battery:.2f} < {t['low_battery']}) + high temp ({device.temperature:.2f} > {t['high_temp']})",
                priority="high"
            )

        # Rule 2: Low complexity -> lightweight regardless of device state
        if scene.complexity < t["low_complexity"]:
            return Decision(
                model="tinysr",
                scale=2,
                reason=f"low complexity ({scene.complexity:.2f} < {t['low_complexity']})"
            )

        # Rule 3: Very high complexity + GPU headroom -> heaviest model (if available)
        from src.modules.model_registry import MODEL_REGISTRY
        if (scene.complexity > t["very_high_complexity"]) and \
           (device.gpu is not None and device.gpu < t["gpu_headroom"]) and \
           MODEL_REGISTRY.get("basicvsr++", {}).get("available", True):
            return Decision(
                model="basicvsr++",
                scale=2,
                reason=f"very high complexity ({scene.complexity:.2f} > {t['very_high_complexity']}) + GPU headroom available ({device.gpu:.2f} < {t['gpu_headroom']})",
                priority="high"
            )

        # Rule 4: High complexity + CPU headroom -> mid-tier model
        if (scene.complexity > t["high_complexity"]) and \
           (device.cpu < t["cpu_headroom"]):
            return Decision(
                model="real_esrgan",
                scale=2,
                reason=f"high complexity ({scene.complexity:.2f} > {t['high_complexity']}) + CPU headroom available ({device.cpu:.2f} < {t['cpu_headroom']})"
            )

        # Rule 5: Fallback — moderate complexity, no strong constraint triggered
        return Decision(
            model="real_esrgan",
            scale=2,
            reason="default: moderate complexity, no constraint triggered"
        )
