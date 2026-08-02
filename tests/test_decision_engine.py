import pytest
from src.utils.state_types import DeviceState, SceneDescriptor
from src.modules.decision_engine import DecisionEngine

@pytest.fixture
def engine():
    return DecisionEngine(config_path="configs/decision_config.yaml")

def test_case_1_low_battery_and_hot(engine):
    # Scenario: Low battery + hot (Rule 1)
    dev = DeviceState(cpu=0.30, gpu=None, ram=0.50, system_ram=0.50, battery=0.15, charging=False, temperature=0.80, fps=30.0)
    scene = SceneDescriptor(motion=0.0, texture=0.5, edges=0.5, blur_clarity=0.5, complexity=0.50)
    decision = engine.decide(dev, scene)
    assert decision.model == "tinysr"
    assert decision.priority == "high"
    assert "0.15" in decision.reason
    assert "0.80" in decision.reason

def test_case_2_low_battery_but_cool(engine):
    # Scenario: Low battery, but cool (falls to Rule 5 fallback)
    dev = DeviceState(cpu=0.30, gpu=None, ram=0.50, system_ram=0.50, battery=0.15, charging=False, temperature=0.40, fps=30.0)
    scene = SceneDescriptor(motion=0.0, texture=0.5, edges=0.5, blur_clarity=0.5, complexity=0.50)
    decision = engine.decide(dev, scene)
    assert decision.model == "real_esrgan"
    assert "default" in decision.reason

def test_case_3_battery_unknown_hot(engine):
    # Scenario: Battery unknown (desktop), hot (falls to Rule 5 fallback)
    dev = DeviceState(cpu=0.30, gpu=None, ram=0.50, system_ram=0.50, battery=None, charging=None, temperature=0.80, fps=30.0)
    scene = SceneDescriptor(motion=0.0, texture=0.5, edges=0.5, blur_clarity=0.5, complexity=0.50)
    decision = engine.decide(dev, scene)
    assert decision.model == "real_esrgan"
    assert "default" in decision.reason

def test_case_4_flat_scene_powerful_device(engine):
    # Scenario: Flat scene, powerful device (Rule 2 complexity short-circuit)
    dev = DeviceState(cpu=0.10, gpu=0.10, ram=0.50, system_ram=0.50, battery=0.90, charging=True, temperature=0.30, fps=30.0)
    scene = SceneDescriptor(motion=0.0, texture=0.1, edges=0.1, blur_clarity=0.1, complexity=0.10)
    decision = engine.decide(dev, scene)
    assert decision.model == "tinysr"
    assert "0.10" in decision.reason

def test_case_5_extreme_complexity_gpu_free(engine):
    # Scenario: Extreme complexity, GPU free (Rule 3)
    from src.modules.model_registry import MODEL_REGISTRY
    dev = DeviceState(cpu=0.10, gpu=0.10, ram=0.50, system_ram=0.50, battery=0.90, charging=True, temperature=0.30, fps=30.0)
    scene = SceneDescriptor(motion=0.0, texture=0.95, edges=0.95, blur_clarity=0.95, complexity=0.95)
    
    # 1. When basicvsr++ is available
    MODEL_REGISTRY["basicvsr++"]["available"] = True
    decision = engine.decide(dev, scene)
    assert decision.model == "basicvsr++"
    assert decision.priority == "high"
    assert "0.95" in decision.reason
    assert "0.10" in decision.reason

    # 2. When basicvsr++ is unavailable (Option A fallback verification)
    MODEL_REGISTRY["basicvsr++"]["available"] = False
    decision_fallback = engine.decide(dev, scene)
    assert decision_fallback.model == "real_esrgan"
    assert "high complexity" in decision_fallback.reason

def test_case_6_extreme_complexity_no_gpu(engine):
    # Scenario: Extreme complexity, no GPU present (falls to Rule 4)
    dev = DeviceState(cpu=0.10, gpu=None, ram=0.50, system_ram=0.50, battery=0.90, charging=True, temperature=0.30, fps=30.0)
    scene = SceneDescriptor(motion=0.0, texture=0.95, edges=0.95, blur_clarity=0.95, complexity=0.95)
    decision = engine.decide(dev, scene)
    assert decision.model == "real_esrgan"
    assert "0.95" in decision.reason
    assert "0.10" in decision.reason

def test_case_7_extreme_complexity_gpu_busy(engine):
    # Scenario: Extreme complexity, GPU present but busy (falls to Rule 4)
    dev = DeviceState(cpu=0.10, gpu=0.90, ram=0.50, system_ram=0.50, battery=0.90, charging=True, temperature=0.30, fps=30.0)
    scene = SceneDescriptor(motion=0.0, texture=0.95, edges=0.95, blur_clarity=0.95, complexity=0.95)
    decision = engine.decide(dev, scene)
    assert decision.model == "real_esrgan"
    assert "0.95" in decision.reason
    assert "0.10" in decision.reason

def test_case_8_high_complexity_cpu_busy(engine):
    # Scenario: High complexity, CPU busy (falls to Rule 5 fallback)
    dev = DeviceState(cpu=0.90, gpu=None, ram=0.50, system_ram=0.50, battery=0.90, charging=True, temperature=0.30, fps=30.0)
    scene = SceneDescriptor(motion=0.0, texture=0.80, edges=0.80, blur_clarity=0.80, complexity=0.80)
    decision = engine.decide(dev, scene)
    assert decision.model == "real_esrgan"
    assert "default" in decision.reason

def test_case_9_all_fields_none(engine):
    # Scenario: Mid complexity, all fields None except cpu (Rule 5 fallback)
    dev = DeviceState(cpu=0.40, gpu=None, ram=0.50, system_ram=0.50, battery=None, charging=None, temperature=None, fps=30.0)
    scene = SceneDescriptor(motion=0.0, texture=0.50, edges=0.50, blur_clarity=0.50, complexity=0.50)
    
    # Verify it doesn't raise and processes safely
    decision = engine.decide(dev, scene)
    assert decision.model == "real_esrgan"
    assert "default" in decision.reason

def test_case_skip_enhancement(engine):
    # Scenario: Critical battery + low complexity (Rule 0)
    dev = DeviceState(cpu=0.20, gpu=None, ram=0.50, system_ram=0.50, battery=0.08, charging=False, temperature=0.30, fps=30.0)
    scene = SceneDescriptor(motion=0.0, texture=0.10, edges=0.10, blur_clarity=0.10, complexity=0.12)
    
    decision = engine.decide(dev, scene)
    assert decision.model == "skip"
    assert decision.scale == 1
    assert "critical battery" in decision.reason
    assert "trivial frame" in decision.reason

