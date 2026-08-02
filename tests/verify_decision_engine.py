from src.utils.state_types import DeviceState, SceneDescriptor
from src.modules.decision_engine import DecisionEngine

def main():
    print("=== Phase 4 Validation: Decision Engine Verification ===")
    
    engine = DecisionEngine(config_path="configs/decision_config.yaml")
    
    # Define test cases: Scenario name, DeviceState, SceneDescriptor, Expected Model, Expected Rule
    test_cases = [
        (
            "Low battery + hot",
            DeviceState(cpu=0.30, gpu=None, ram=0.50, system_ram=0.50, battery=0.15, charging=False, temperature=0.80, fps=30.0),
            SceneDescriptor(motion=0.0, texture=0.5, edges=0.5, blur_clarity=0.5, complexity=0.50),
            "tinysr",
            "Rule 1"
        ),
        (
            "Low battery, but cool",
            DeviceState(cpu=0.30, gpu=None, ram=0.50, system_ram=0.50, battery=0.15, charging=False, temperature=0.40, fps=30.0),
            SceneDescriptor(motion=0.0, texture=0.5, edges=0.5, blur_clarity=0.5, complexity=0.50),
            "real_esrgan",
            "Rule 5"
        ),
        (
            "Battery unknown (desktop), hot",
            DeviceState(cpu=0.30, gpu=None, ram=0.50, system_ram=0.50, battery=None, charging=None, temperature=0.80, fps=30.0),
            SceneDescriptor(motion=0.0, texture=0.5, edges=0.5, blur_clarity=0.5, complexity=0.50),
            "real_esrgan",
            "Rule 5"
        ),
        (
            "Flat scene, powerful device",
            DeviceState(cpu=0.10, gpu=0.10, ram=0.50, system_ram=0.50, battery=0.90, charging=True, temperature=0.30, fps=30.0),
            SceneDescriptor(motion=0.0, texture=0.1, edges=0.1, blur_clarity=0.1, complexity=0.10),
            "tinysr",
            "Rule 2"
        ),
        (
            "Extreme complexity, GPU free",
            DeviceState(cpu=0.10, gpu=0.10, ram=0.50, system_ram=0.50, battery=0.90, charging=True, temperature=0.30, fps=30.0),
            SceneDescriptor(motion=0.0, texture=0.95, edges=0.95, blur_clarity=0.95, complexity=0.95),
            "basicvsr++",
            "Rule 3"
        ),
        (
            "Extreme complexity, no GPU",
            DeviceState(cpu=0.10, gpu=None, ram=0.50, system_ram=0.50, battery=0.90, charging=True, temperature=0.30, fps=30.0),
            SceneDescriptor(motion=0.0, texture=0.95, edges=0.95, blur_clarity=0.95, complexity=0.95),
            "real_esrgan",
            "Rule 4"
        ),
        (
            "Extreme complexity, GPU present but busy",
            DeviceState(cpu=0.10, gpu=0.90, ram=0.50, system_ram=0.50, battery=0.90, charging=True, temperature=0.30, fps=30.0),
            SceneDescriptor(motion=0.0, texture=0.95, edges=0.95, blur_clarity=0.95, complexity=0.95),
            "real_esrgan",
            "Rule 4"
        ),
        (
            "High complexity, CPU busy",
            DeviceState(cpu=0.90, gpu=None, ram=0.50, system_ram=0.50, battery=0.90, charging=True, temperature=0.30, fps=30.0),
            SceneDescriptor(motion=0.0, texture=0.80, edges=0.80, blur_clarity=0.80, complexity=0.80),
            "real_esrgan",
            "Rule 5"
        ),
        (
            "Mid complexity, all fields None except cpu",
            DeviceState(cpu=0.40, gpu=None, ram=0.50, system_ram=0.50, battery=None, charging=None, temperature=None, fps=30.0),
            SceneDescriptor(motion=0.0, texture=0.50, edges=0.50, blur_clarity=0.50, complexity=0.50),
            "real_esrgan",
            "Rule 5"
        )
    ]
    
    print(f"\n{'#':<2} | {'Scenario':<42} | {'Complexity':<10} | {'Expected Model':<14} | {'Actual Model':<14} | {'Status':<6}")
    print("-" * 105)
    
    all_passed = True
    for idx, (scenario, dev, scene, expected_model, expected_rule) in enumerate(test_cases, 1):
        decision = engine.decide(dev, scene)
        status = "PASSED" if decision.model == expected_model else "FAILED"
        if status == "FAILED":
            all_passed = False
        print(f"{idx:<2} | {scenario:<42} | {scene.complexity:<10.2f} | {expected_model:<14} | {decision.model:<14} | {status:<6}")
        print(f"   -> Reason: {decision.reason}")
        print()
        
    print(f"Overall Decision Engine Validation: {'[SUCCESS]' if all_passed else '[FAILED]'}")

if __name__ == "__main__":
    main()
