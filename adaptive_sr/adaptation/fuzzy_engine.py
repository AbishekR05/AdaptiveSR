"""
adaptive_sr.adaptation.fuzzy_engine
===================================
Step 10 — Fuzzy Adaptive Decision Engine.

Builds a Mamdani-style fuzzy decision engine that consumes validated signals from
Step 7 (FPSAdaptationSignal), Step 8 (BitrateAdaptationSignal), and Step 9 (EdgeResourceSignal)
to select the most suitable feasible Super-Resolution (SR) candidate configuration.

Steps 0–9 remain 100% frozen. No arbitrary weighted utility coefficients or fabricated metrics are introduced.
"""

import math
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict

from adaptive_sr.adaptation.fps_adapter import FPSAdaptationSignal
from adaptive_sr.adaptation.bitrate_adapter import BitrateAdaptationSignal
from adaptive_sr.adaptation.edge_evaluator import EdgeResourceSignal


# ============================================================================
# Membership Function Helpers (Triangular & Trapezoidal)
# ============================================================================

def triangle_mf(x: float, a: float, b: float, c: float) -> float:
    """
    Triangular membership function:
    0 for x <= a or x >= c
    (x - a) / (b - a) for a < x < b
    (c - x) / (c - b) for b <= x < c
    """
    if x is None or math.isnan(x):
        return 0.0
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b > a else 1.0
    else:
        return (c - x) / (c - b) if c > b else 1.0


def trapezoid_mf(x: float, a: float, b: float, c: float, d: float) -> float:
    """
    Trapezoidal membership function:
    0 for x <= a or x >= d
    (x - a) / (b - a) for a < x < b
    1.0 for b <= x <= c
    (d - x) / (d - c) for c < x < d
    """
    if x is None or math.isnan(x):
        return 0.0
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b > a else 1.0
    else:
        return (d - x) / (d - c) if d > c else 1.0


# ============================================================================
# Dataclasses & Signal Schemas
# ============================================================================

@dataclass
class CandidateEvaluation:
    candidate_id: str
    edge_id: str
    base_representation_id: str
    target_resolution: str
    model_id: str
    scale: int
    device: str
    hard_feasible: bool
    rejection_reasons: List[str]
    fuzzified_inputs: Dict[str, Dict[str, float]]
    quality_tier: str  # "good", "acceptable", "poor", "unevaluable"
    defuzzified_suitability: Optional[float]
    suitability_label: str  # "very_low", "low", "medium", "high", "very_high", "infeasible", "unevaluable"
    rule_activations: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FuzzyDecisionSignal:
    decision: str  # "selected", "no_suitable_candidate", "no_feasible_candidates"
    selected_candidate: Optional[Dict[str, Any]]
    selected_edge_id: Optional[str]
    selected_representation_id: Optional[str]
    target_resolution: Optional[str]
    model_id: Optional[str]
    scale: Optional[int]
    device: Optional[str]
    fuzzy_suitability: Optional[float]
    suitability_tier: Optional[str]
    min_suitability_threshold: float
    candidate_evaluations: List[Dict[str, Any]]
    rejected_candidates: List[Dict[str, Any]]
    input_signal_provenance: Dict[str, str]
    rule_inference_metadata: Dict[str, Any]
    warnings: List[str]
    baseline_comparison_ready: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Fuzzy Adaptive Decision Engine Class
# ============================================================================

class FuzzyAdaptiveDecisionEngine:
    """
    Step 10 — Mamdani Fuzzy Adaptive Decision Engine.

    Evaluates candidate SR configurations constructed from Step 7 (FPS), Step 8 (Bitrate),
    and Step 9 (Edge Resource) signals. Employs a hard feasibility gate followed by a
    5-variable Mamdani fuzzy inference engine and centroid defuzzification.
    """

    def __init__(self, min_suitability_threshold: float = 35.0):
        """
        :param min_suitability_threshold: Configurable engineering threshold [0..100]
                                          for candidate selection. Default: 35.0.
                                          Note: Threshold is an operational engineering parameter,
                                          subject to ablation/sensitivity testing in Step 12.
        """
        self.min_suitability_threshold = min_suitability_threshold

    # ------------------------------------------------------------------------
    # 10.1 & 10.2 Candidate Construction & Hard Feasibility Gate
    # ------------------------------------------------------------------------

    @staticmethod
    def construct_candidates(
        fps_signals: List[FPSAdaptationSignal],
        bitrate_signals: List[BitrateAdaptationSignal],
        resource_signals: List[EdgeResourceSignal],
    ) -> List[Dict[str, Any]]:
        """
        Constructs candidate configurations by matching tuple keys:
        (base_representation_id, target_resolution, model_id, scale, device, edge_id).
        """
        candidates = []
        for resource in resource_signals:
            fps_match = next(
                (
                    f
                    for f in fps_signals
                    if f.base_representation_id == resource.base_representation_id
                    and f.model_id == resource.model_id
                    and f.scale == resource.scale
                    and f.device == resource.device
                ),
                None,
            )
            bitrate_match = next(
                (
                    b
                    for b in bitrate_signals
                    if b.candidate_representation_id == resource.base_representation_id
                    and b.target_resolution == resource.target_resolution
                    and b.model_id == resource.model_id
                    and b.scale == resource.scale
                    and b.device == resource.device
                ),
                None,
            )

            candidate_id = (
                f"{resource.edge_id}:{resource.base_representation_id}:"
                f"{resource.target_resolution}:{resource.model_id}:x{resource.scale}:{resource.device}"
            )

            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "edge_id": resource.edge_id,
                    "base_representation_id": resource.base_representation_id,
                    "target_resolution": resource.target_resolution,
                    "model_id": resource.model_id,
                    "scale": resource.scale,
                    "device": resource.device,
                    "resource_signal": resource,
                    "fps_signal": fps_match,
                    "bitrate_signal": bitrate_match,
                }
            )
        return candidates

    @staticmethod
    def evaluate_hard_feasibility(candidate: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Evaluates hard feasibility before fuzzy inference. Rejects candidates with:
        - Resource infeasibility (resource_feasible == False)
        - Infeasible status in Step 9
        - Missing FPS or Bitrate signals
        - Missing hardware support or CUDA without GPU
        - Non-positive real_time_ratio (r <= 0.0)

        Note: real_time_ratio < 1.0 (e.g. 0.85 near_realtime) remains hard-feasible;
        soft real-time feasibility tiers are evaluated by the fuzzy engine.
        """
        rejection_reasons = []
        resource: Optional[EdgeResourceSignal] = candidate.get("resource_signal")
        fps: Optional[FPSAdaptationSignal] = candidate.get("fps_signal")
        bitrate: Optional[BitrateAdaptationSignal] = candidate.get("bitrate_signal")

        if resource is None:
            rejection_reasons.append("missing_edge_resource_signal")
            return False, rejection_reasons

        if not resource.resource_feasible or resource.feasibility_status == "infeasible":
            rejection_reasons.append(
                f"step9_resource_infeasible: {resource.feasibility_status}"
            )

        hw = resource.hardware_capability or {}
        if candidate["device"] == "cuda" and not hw.get("gpu_available", False):
            rejection_reasons.append("cuda_requested_but_gpu_unavailable")

        supported_models = hw.get("supported_models", [])
        if supported_models and candidate["model_id"] not in supported_models:
            rejection_reasons.append(
                f"model_{candidate['model_id']}_not_supported_on_edge_{candidate['edge_id']}"
            )

        if fps is None:
            rejection_reasons.append("missing_fps_adaptation_signal")
        else:
            if fps.real_time_ratio is None or fps.real_time_ratio <= 0.0:
                rejection_reasons.append("invalid_or_non_positive_real_time_ratio")

        if bitrate is None:
            rejection_reasons.append("missing_bitrate_adaptation_signal")

        is_feasible = len(rejection_reasons) == 0
        return is_feasible, rejection_reasons

    # ------------------------------------------------------------------------
    # 10.4 Conservative Quality Evidence Policy (Non-Contradictory Evidence)
    # ------------------------------------------------------------------------

    @staticmethod
    def classify_quality_suitability(
        bitrate_signal: Optional[BitrateAdaptationSignal],
    ) -> Tuple[str, List[str]]:
        """
        Classifies visual quality evidence using a conservative non-contradictory evidence policy:
        - "good": At least one available metric meets high threshold (VMAF >= 80, PSNR >= 35, SSIM >= 0.92)
                  AND NO available metric is below poor threshold (VMAF < 60, PSNR < 30, SSIM < 0.85).
        - "poor": ANY available metric is below poor threshold (VMAF < 60, PSNR < 30, SSIM < 0.85).
        - "acceptable": At least one metric available, no metric poor, but high threshold not met.
        - "unevaluable": quality_evaluable == False or no numerical metrics present.

        This avoids allowing a single metric to override contradictory poor metrics (e.g. VMAF=40 with PSNR=36).
        Missing or unevaluable evidence returns "unevaluable" without fabricating fake 0.5 scores.
        """
        warnings = []
        if bitrate_signal is None or not bitrate_signal.quality_evaluable:
            warnings.append("quality_metrics_unavailable_quality_evaluable_false")
            return "unevaluable", warnings

        vmaf = bitrate_signal.vmaf
        psnr = bitrate_signal.psnr_db
        ssim = bitrate_signal.ssim

        has_vmaf = vmaf is not None and not math.isnan(vmaf)
        has_psnr = psnr is not None and not math.isnan(psnr)
        has_ssim = ssim is not None and not math.isnan(ssim)

        if not (has_vmaf or has_psnr or has_ssim):
            warnings.append("no_valid_numerical_quality_metrics_present")
            return "unevaluable", warnings

        # Check for ANY contradictory poor metric
        is_any_poor = (
            (has_vmaf and vmaf < 60.0)
            or (has_psnr and psnr < 30.0)
            or (has_ssim and ssim < 0.85)
        )
        if is_any_poor:
            warnings.append("contradictory_or_poor_quality_metric_detected_classified_poor")
            return "poor", warnings

        # Check for high quality without poor contradiction
        has_high_metric = (
            (has_vmaf and vmaf >= 80.0)
            or (has_psnr and psnr >= 35.0)
            or (has_ssim and ssim >= 0.92)
        )
        if has_high_metric:
            return "good", warnings

        return "acceptable", warnings

    # ------------------------------------------------------------------------
    # 10.3 Fuzzy Inputs & Fuzzification
    # ------------------------------------------------------------------------

    @staticmethod
    def fuzzify_inputs(
        candidate: Dict[str, Any],
        quality_tier: str,
    ) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
        """
        Fuzzifies inputs using explicit continuous membership functions.
        All membership function boundaries are initial engineering operational parameters,
        subject to sensitivity testing and ablation in Step 12.

        A. real_time_ratio [0..2.0+]
           - poor: trapezoid [0, 0, 0.5, 0.8]
           - moderate: triangle [0.6, 0.85, 1.1]
           - good: trapezoid [0.95, 1.2, 3.0, 3.0]

        B. bandwidth_saving_percent [-50..100%]
           - low: trapezoid [-50, -50, 0, 25]
           - medium: triangle [15, 35, 55]
           - high: trapezoid [45, 65, 100, 100]

        C. quality_suitability (Qualitative classification tier)
           - good: 1.0 if tier=="good" else 0.0
           - acceptable: 1.0 if tier=="acceptable" else 0.0
           - poor: 1.0 if tier=="poor" else 0.0
           (unevaluable quality yields 0.0 across all tiers — NO fake 0.5 scores)

        D. edge_resource_condition [0..1.0] (Device-Aware Resource Headroom)
           - For CPU: 1.0 - (cpu_util / 100.0)
           - For CUDA: 1.0 - (max(gpu_util, gpu_mem_used_pct) / 100.0)
           - constrained: trapezoid [0, 0, 0.15, 0.35]
           - moderate: triangle [0.25, 0.50, 0.75]
           - available: trapezoid [0.65, 0.85, 1.0, 1.0]

        E. network_condition [0..1.0] (Measured Network Bandwidth)
           - poor: trapezoid [0, 0, 5.0, 15.0] (<= 5 Mbps)
           - moderate: triangle [10.0, 25.0, 45.0]
           - good: trapezoid [35.0, 50.0, 200.0, 200.0] (>= 50 Mbps)
        """
        warnings = []
        fps: FPSAdaptationSignal = candidate["fps_signal"]
        bitrate: BitrateAdaptationSignal = candidate["bitrate_signal"]
        resource: EdgeResourceSignal = candidate["resource_signal"]

        # A. real_time_ratio
        rt_val = fps.real_time_ratio if fps else 0.0
        f_rt = {
            "poor": trapezoid_mf(rt_val, 0.0, 0.0, 0.5, 0.8),
            "moderate": triangle_mf(rt_val, 0.6, 0.85, 1.1),
            "good": trapezoid_mf(rt_val, 0.95, 1.2, 3.0, 3.0),
        }

        # B. bandwidth_saving_percent
        bw_val = (
            bitrate.bitrate_saving_percent
            if (bitrate and bitrate.bitrate_saving_percent is not None)
            else 0.0
        )
        f_bw = {
            "low": trapezoid_mf(bw_val, -50.0, -50.0, 0.0, 25.0),
            "medium": triangle_mf(bw_val, 15.0, 35.0, 55.0),
            "high": trapezoid_mf(bw_val, 45.0, 65.0, 100.0, 100.0),
        }

        # C. quality_suitability (discrete qualitative tier fuzzification)
        if quality_tier == "good":
            f_q = {"good": 1.0, "acceptable": 0.0, "poor": 0.0}
        elif quality_tier == "acceptable":
            f_q = {"good": 0.0, "acceptable": 1.0, "poor": 0.0}
        elif quality_tier == "poor":
            f_q = {"good": 0.0, "acceptable": 0.0, "poor": 1.0}
        else:
            # "unevaluable" -> all 0.0. No fake 0.5 membership fabricated.
            f_q = {"good": 0.0, "acceptable": 0.0, "poor": 0.0}
            warnings.append("unevaluable_quality_metrics_does_not_activate_positive_quality_rules")

        # D. Device-Aware edge_resource_condition
        res_avail = resource.resource_availability or {}
        hw_cap = resource.hardware_capability or {}
        req_device = candidate["device"]

        if req_device == "cuda" and hw_cap.get("gpu_available", False):
            gpu_util = res_avail.get("gpu_utilization_percent")
            vram_free = res_avail.get("gpu_memory_free_bytes")
            vram_total = hw_cap.get("gpu_memory_total_bytes")

            vram_used_pct = None
            if vram_free is not None and vram_total is not None and vram_total > 0:
                vram_used_pct = (1.0 - (vram_free / float(vram_total))) * 100.0

            if gpu_util is not None and vram_used_pct is not None:
                resource_load = max(gpu_util, vram_used_pct)
            elif gpu_util is not None:
                resource_load = gpu_util
            elif vram_used_pct is not None:
                resource_load = vram_used_pct
            else:
                resource_load = res_avail.get("cpu_utilization_percent", 50.0)
                warnings.append("cuda_device_requested_but_gpu_telemetry_missing_using_cpu_telemetry")
        else:
            cpu_util = res_avail.get("cpu_utilization_percent")
            if cpu_util is None:
                resource_load = 50.0
                warnings.append("missing_cpu_telemetry_using_neutral_resource_load_50pct")
            else:
                resource_load = cpu_util

        res_headroom = max(0.0, min(1.0, (100.0 - resource_load) / 100.0))
        f_res = {
            "constrained": trapezoid_mf(res_headroom, 0.0, 0.0, 0.15, 0.35),
            "moderate": triangle_mf(res_headroom, 0.25, 0.50, 0.75),
            "available": trapezoid_mf(res_headroom, 0.65, 0.85, 1.0, 1.0),
        }

        # E. network_condition (Explicit Measured Network Bandwidth)
        net_telemetry = resource.network_telemetry or {}
        bw_mbps = net_telemetry.get("measured_bandwidth_mbps")

        if bw_mbps is None or math.isnan(bw_mbps):
            f_net = {"poor": 0.0, "moderate": 0.0, "good": 0.0}
            warnings.append("missing_network_bandwidth_telemetry_network_condition_unevaluable")
        else:
            f_net = {
                "poor": trapezoid_mf(bw_mbps, 0.0, 0.0, 5.0, 15.0),
                "moderate": triangle_mf(bw_mbps, 10.0, 25.0, 45.0),
                "good": trapezoid_mf(bw_mbps, 35.0, 50.0, 200.0, 200.0),
            }

        fuzzified = {
            "real_time_ratio": f_rt,
            "bandwidth_saving": f_bw,
            "quality": f_q,
            "resource_condition": f_res,
            "network_condition": f_net,
        }
        return fuzzified, warnings

    # ------------------------------------------------------------------------
    # 10.5 & 10.6 Rule Base & Mamdani Inference
    # ------------------------------------------------------------------------

    @staticmethod
    def evaluate_rules(fuzzified: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Evaluates the Mamdani rule base using AND = min() and OR = max() logic.
        Returns a dictionary mapping linguistic output levels
        ("very_low", "low", "medium", "high", "very_high") to aggregated firing strength (max).

        Rule Base Definition:
        ---------------------
        R1 (Optimal): IF real_time_ratio IS good AND bandwidth_saving IS high AND quality IS good
            AND resource_condition IS available AND network_condition IS good
            THEN suitability IS very_high

        R2 (High Performance - Good/Acceptable Quality & Compute Headroom):
            IF real_time_ratio IS good AND (bandwidth_saving IS high OR medium)
            AND (quality IS good OR acceptable) AND (resource_condition IS available OR moderate)
            THEN suitability IS high

        R3 (High Performance - Bandwidth Saving & Real-Time):
            IF real_time_ratio IS good AND (bandwidth_saving IS high OR medium)
            AND (resource_condition IS available OR moderate)
            THEN suitability IS high

        R4 (Moderate Tradeoff - Balanced Non-Poor Condition):
            IF (real_time_ratio IS good OR moderate) AND (bandwidth_saving IS medium OR low)
            AND (quality IS good OR acceptable) AND (resource_condition IS available OR moderate)
            AND (network_condition IS good OR moderate)
            THEN suitability IS medium

        R5 (Resource / Network Moderate):
            IF real_time_ratio IS moderate AND resource_condition IS moderate
            THEN suitability IS medium

        R6 (Low Gain / Poor Quality):
            IF bandwidth_saving IS low AND quality IS poor
            THEN suitability IS low

        R7 (Network / Resource Constrained):
            IF network_condition IS poor OR resource_condition IS constrained
            THEN suitability IS low

        R8 (Poor Real-Time / Hard Bottleneck):
            IF real_time_ratio IS poor
            THEN suitability IS very_low
        """
        rt = fuzzified["real_time_ratio"]
        bw = fuzzified["bandwidth_saving"]
        q = fuzzified["quality"]
        res = fuzzified["resource_condition"]
        net = fuzzified["network_condition"]

        activations = {
            "very_high": 0.0,
            "high": 0.0,
            "medium": 0.0,
            "low": 0.0,
            "very_low": 0.0,
        }

        # R1 (Optimal)
        r1 = min(rt["good"], bw["high"], q["good"], res["available"], net["good"])
        activations["very_high"] = max(activations["very_high"], r1)

        # R2 (High Performance - Quality & Compute Headroom)
        r2 = min(
            rt["good"],
            max(bw["high"], bw["medium"]),
            max(q["good"], q["acceptable"]),
            max(res["available"], res["moderate"]),
        )
        activations["high"] = max(activations["high"], r2)

        # R3 (High Performance - Bandwidth Saving & Real-Time)
        r3 = min(
            rt["good"],
            max(bw["high"], bw["medium"]),
            max(res["available"], res["moderate"]),
        )
        activations["high"] = max(activations["high"], r3)

        # R4 (Moderate Tradeoff - Non-Poor Balanced State)
        r4 = min(
            max(rt["good"], rt["moderate"]),
            max(bw["medium"], bw["low"]),
            max(q["good"], q["acceptable"]),
            max(res["available"], res["moderate"]),
            max(net["good"], net["moderate"]),
        )
        activations["medium"] = max(activations["medium"], r4)

        # R5 (Resource / Network Moderate)
        r5 = min(rt["moderate"], res["moderate"])
        activations["medium"] = max(activations["medium"], r5)

        # R6 (Low Gain / Poor Quality)
        r6 = min(bw["low"], q["poor"])
        activations["low"] = max(activations["low"], r6)

        # R7 (Network / Resource Constrained)
        r7 = max(net["poor"], res["constrained"])
        activations["low"] = max(activations["low"], r7)

        # R8 (Poor Real-Time / Hard Bottleneck)
        r8 = rt["poor"]
        activations["very_low"] = max(activations["very_low"], r8)

        return activations

    # ------------------------------------------------------------------------
    # 10.5 Centroid Defuzzification & Zero-Activation Fallback
    # ------------------------------------------------------------------------

    @staticmethod
    def defuzzify_centroid(activations: Dict[str, float]) -> Optional[float]:
        """
        Centroid (Center-of-Area) defuzzification over domain [0, 100] with step 0.5.

        Output Membership Shapes:
        - very_low: trapezoid [0, 0, 10, 25]
        - low: triangle [15, 30, 45]
        - medium: triangle [35, 50, 65]
        - high: triangle [55, 70, 85]
        - very_high: trapezoid [75, 90, 100, 100]

        Zero-Activation Fallback Policy:
        If total aggregated area denominator == 0.0 (no rules activated),
        returns None. Candidate is marked "unevaluable" without fabricating an arbitrary score.
        """
        out_mfs = {
            "very_low": lambda y: trapezoid_mf(y, 0.0, 0.0, 10.0, 25.0),
            "low": lambda y: triangle_mf(y, 15.0, 30.0, 45.0),
            "medium": lambda y: triangle_mf(y, 35.0, 50.0, 65.0),
            "high": lambda y: triangle_mf(y, 55.0, 70.0, 85.0),
            "very_high": lambda y: trapezoid_mf(y, 75.0, 90.0, 100.0, 100.0),
        }

        step = 0.5
        numerator = 0.0
        denominator = 0.0
        y = 0.0

        while y <= 100.0:
            mu_y = 0.0
            for term, act in activations.items():
                if act > 0.0:
                    val = min(act, out_mfs[term](y))
                    if val > mu_y:
                        mu_y = val

            numerator += y * mu_y * step
            denominator += mu_y * step
            y += step

        if denominator == 0.0:
            return None

        return numerator / denominator

    @staticmethod
    def classify_suitability_label(score: Optional[float]) -> str:
        """Classifies defuzzified score into linguistic label."""
        if score is None:
            return "unevaluable"
        if score >= 80.0:
            return "very_high"
        elif score >= 60.0:
            return "high"
        elif score >= 40.0:
            return "medium"
        elif score >= 20.0:
            return "low"
        else:
            return "very_low"

    # ------------------------------------------------------------------------
    # 10.7 Decision Evaluation & Selection Engine
    # ------------------------------------------------------------------------

    def evaluate_candidates(
        self,
        fps_signals: List[FPSAdaptationSignal],
        bitrate_signals: List[BitrateAdaptationSignal],
        resource_signals: List[EdgeResourceSignal],
    ) -> FuzzyDecisionSignal:
        """
        Main decision workflow:
        1. Construct candidate configurations.
        2. Evaluate hard feasibility gate.
        3. Classify quality suitability tier (conservative non-contradictory policy).
        4. Fuzzify 5 continuous input variables.
        5. Evaluate Mamdani rules and defuzzify suitability scores via centroid integration.
        6. Apply zero-activation fallback policy (returns None for unevaluable candidates).
        7. Filter candidates meeting min_suitability_threshold.
        8. Apply Unevaluable Quality Selection Policy:
           A candidate with quality_tier == "unevaluable" remains evaluable for resource/FPS analysis,
           BUT CANNOT be selected as the final SR candidate.
        9. Select optimal candidate (with explicit deterministic tie-breaking policy).
        10. Produce machine-readable FuzzyDecisionSignal.
        """
        candidates = self.construct_candidates(fps_signals, bitrate_signals, resource_signals)
        all_evaluations: List[CandidateEvaluation] = []
        feasible_evaluations: List[CandidateEvaluation] = []
        all_warnings: List[str] = []

        if not candidates:
            return FuzzyDecisionSignal(
                decision="no_feasible_candidates",
                selected_candidate=None,
                selected_edge_id=None,
                selected_representation_id=None,
                target_resolution=None,
                model_id=None,
                scale=None,
                device=None,
                fuzzy_suitability=None,
                suitability_tier=None,
                min_suitability_threshold=self.min_suitability_threshold,
                candidate_evaluations=[],
                rejected_candidates=[],
                input_signal_provenance={
                    "fps": "step7_fps_adaptation",
                    "bitrate": "step8_bitrate_adaptation",
                    "edge": "step9_edge_resource",
                },
                rule_inference_metadata={"rule_count": 8, "inference_engine": "Mamdani_Centroid"},
                warnings=["no_candidate_configurations_supplied"],
            )

        for cand in candidates:
            cand_id = cand["candidate_id"]
            is_feasible, rej_reasons = self.evaluate_hard_feasibility(cand)

            if not is_feasible:
                eval_item = CandidateEvaluation(
                    candidate_id=cand_id,
                    edge_id=cand["edge_id"],
                    base_representation_id=cand["base_representation_id"],
                    target_resolution=cand["target_resolution"],
                    model_id=cand["model_id"],
                    scale=cand["scale"],
                    device=cand["device"],
                    hard_feasible=False,
                    rejection_reasons=rej_reasons,
                    fuzzified_inputs={},
                    quality_tier="unevaluable",
                    defuzzified_suitability=None,
                    suitability_label="infeasible",
                    rule_activations={},
                )
                all_evaluations.append(eval_item)
                continue

            # Feasible -> Quality classification & Fuzzy inference
            q_tier, q_warns = self.classify_quality_suitability(cand["bitrate_signal"])
            all_warnings.extend(q_warns)

            fuzzified, f_warns = self.fuzzify_inputs(cand, q_tier)
            all_warnings.extend(f_warns)

            rule_acts = self.evaluate_rules(fuzzified)
            score = self.defuzzify_centroid(rule_acts)

            if score is None:
                eval_item = CandidateEvaluation(
                    candidate_id=cand_id,
                    edge_id=cand["edge_id"],
                    base_representation_id=cand["base_representation_id"],
                    target_resolution=cand["target_resolution"],
                    model_id=cand["model_id"],
                    scale=cand["scale"],
                    device=cand["device"],
                    hard_feasible=False,
                    rejection_reasons=["zero_rule_activation_unevaluable"],
                    fuzzified_inputs=fuzzified,
                    quality_tier=q_tier,
                    defuzzified_suitability=None,
                    suitability_label="unevaluable",
                    rule_activations=rule_acts,
                )
                all_evaluations.append(eval_item)
                all_warnings.append(f"candidate_{cand_id}_zero_rule_activation_marked_unevaluable")
                continue

            label = self.classify_suitability_label(score)

            eval_item = CandidateEvaluation(
                candidate_id=cand_id,
                edge_id=cand["edge_id"],
                base_representation_id=cand["base_representation_id"],
                target_resolution=cand["target_resolution"],
                model_id=cand["model_id"],
                scale=cand["scale"],
                device=cand["device"],
                hard_feasible=True,
                rejection_reasons=[],
                fuzzified_inputs=fuzzified,
                quality_tier=q_tier,
                defuzzified_suitability=score,
                suitability_label=label,
                rule_activations=rule_acts,
            )
            all_evaluations.append(eval_item)
            feasible_evaluations.append(eval_item)

        rejected_items = [e.to_dict() for e in all_evaluations if not e.hard_feasible]

        if not feasible_evaluations:
            return FuzzyDecisionSignal(
                decision="no_feasible_candidates",
                selected_candidate=None,
                selected_edge_id=None,
                selected_representation_id=None,
                target_resolution=None,
                model_id=None,
                scale=None,
                device=None,
                fuzzy_suitability=None,
                suitability_tier=None,
                min_suitability_threshold=self.min_suitability_threshold,
                candidate_evaluations=[e.to_dict() for e in all_evaluations],
                rejected_candidates=rejected_items,
                input_signal_provenance={
                    "fps": "step7_fps_adaptation",
                    "bitrate": "step8_bitrate_adaptation",
                    "edge": "step9_edge_resource",
                },
                rule_inference_metadata={"rule_count": 8, "inference_engine": "Mamdani_Centroid"},
                warnings=list(set(all_warnings)) + ["all_candidates_failed_hard_feasibility_gate_or_zero_activation"],
            )

        # Filter by minimum suitability threshold AND valid quality evidence selection requirement
        # A candidate with quality_tier == "unevaluable" cannot be selected as final SR candidate!
        eligible_evaluations = []
        for e in feasible_evaluations:
            if e.defuzzified_suitability is None or e.defuzzified_suitability < self.min_suitability_threshold:
                rej_dict = e.to_dict()
                rej_dict["rejection_reasons"] = [
                    f"below_min_suitability_threshold_{self.min_suitability_threshold}"
                ]
                rejected_items.append(rej_dict)
            elif e.quality_tier == "unevaluable":
                rej_dict = e.to_dict()
                rej_dict["rejection_reasons"] = [
                    "candidate_lacks_valid_quality_evidence_cannot_be_selected"
                ]
                rejected_items.append(rej_dict)
                all_warnings.append(
                    f"candidate_{e.candidate_id}_has_unevaluable_quality_cannot_be_selected_as_final_sr_candidate"
                )
            else:
                eligible_evaluations.append(e)

        if not eligible_evaluations:
            return FuzzyDecisionSignal(
                decision="no_suitable_candidate",
                selected_candidate=None,
                selected_edge_id=None,
                selected_representation_id=None,
                target_resolution=None,
                model_id=None,
                scale=None,
                device=None,
                fuzzy_suitability=None,
                suitability_tier=None,
                min_suitability_threshold=self.min_suitability_threshold,
                candidate_evaluations=[e.to_dict() for e in all_evaluations],
                rejected_candidates=rejected_items,
                input_signal_provenance={
                    "fps": "step7_fps_adaptation",
                    "bitrate": "step8_bitrate_adaptation",
                    "edge": "step9_edge_resource",
                },
                rule_inference_metadata={"rule_count": 8, "inference_engine": "Mamdani_Centroid"},
                warnings=list(set(all_warnings))
                + ["no_candidate_met_suitability_threshold_and_quality_evidence_requirement"],
            )

        # Explicit Deterministic Tie-Breaking Policy:
        # 1. Primary: defuzzified_suitability (descending)
        # 2. Secondary: real_time_ratio (descending)
        # 3. Tertiary: bitrate_saving_percent (descending)
        # 4. Quaternary: candidate_id (alphabetical ascending)
        sorted_eligible = sorted(
            eligible_evaluations,
            key=lambda item: (
                -(round(item.defuzzified_suitability or 0.0, 4)),
                -(
                    round(
                        next(c for c in candidates if c["candidate_id"] == item.candidate_id)[
                            "fps_signal"
                        ].real_time_ratio,
                        4,
                    )
                ),
                -(
                    round(
                        next(c for c in candidates if c["candidate_id"] == item.candidate_id)[
                            "bitrate_signal"
                        ].bitrate_saving_percent
                        or -999.0,
                        4,
                    )
                ),
                item.candidate_id,
            ),
        )

        winning_eval = sorted_eligible[0]

        selected_dict = {
            "candidate_id": winning_eval.candidate_id,
            "edge_id": winning_eval.edge_id,
            "base_representation_id": winning_eval.base_representation_id,
            "target_resolution": winning_eval.target_resolution,
            "model_id": winning_eval.model_id,
            "scale": winning_eval.scale,
            "device": winning_eval.device,
            "defuzzified_suitability": winning_eval.defuzzified_suitability,
            "suitability_label": winning_eval.suitability_label,
            "quality_tier": winning_eval.quality_tier,
        }

        return FuzzyDecisionSignal(
            decision="selected",
            selected_candidate=selected_dict,
            selected_edge_id=winning_eval.edge_id,
            selected_representation_id=winning_eval.base_representation_id,
            target_resolution=winning_eval.target_resolution,
            model_id=winning_eval.model_id,
            scale=winning_eval.scale,
            device=winning_eval.device,
            fuzzy_suitability=winning_eval.defuzzified_suitability,
            suitability_tier=winning_eval.suitability_label,
            min_suitability_threshold=self.min_suitability_threshold,
            candidate_evaluations=[e.to_dict() for e in all_evaluations],
            rejected_candidates=rejected_items,
            input_signal_provenance={
                "fps": "step7_fps_adaptation",
                "bitrate": "step8_bitrate_adaptation",
                "edge": "step9_edge_resource",
            },
            rule_inference_metadata={
                "rule_count": 8,
                "inference_engine": "Mamdani_Centroid",
                "defuzzification_domain": "[0, 100]",
                "zero_activation_policy": "return_none_unevaluable",
                "quality_evidence_selection_requirement": "enforced_non_unevaluable",
            },
            warnings=list(set(all_warnings)),
        )
