"""
adaptive_sr.adaptation.fuzzy_engine
===================================
Step 10 — Fuzzy Adaptive Decision Engine.

Builds a Mamdani-style fuzzy decision engine that consumes validated signals from
Step 7 (FPSAdaptationSignal), Step 8 (BitrateAdaptationSignal), and Step 9 (EdgeResourceSignal)
to select the most suitable feasible Super-Resolution (SR) candidate configuration.

Steps 0–9 remain 100% frozen. No arbitrary utility coefficients or fabricated metrics are introduced.
"""

import math
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field

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
    quality_suitability_score: Optional[float]
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
        :param min_suitability_threshold: Minimum defuzzified suitability score [0..100]
                                          required for candidate selection. Default: 35.0.
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
            # Find matching FPS signal
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
            # Find matching Bitrate signal
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
        - Non-positive real_time_ratio
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

        # Check hardware presence for CUDA requests
        hw = resource.hardware_capability or {}
        if candidate["device"] == "cuda" and not hw.get("gpu_available", False):
            rejection_reasons.append("cuda_requested_but_gpu_unavailable")

        # Check model support
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
    # 10.4 Quality Suitability Calculation
    # ------------------------------------------------------------------------

    @staticmethod
    def calculate_quality_suitability(
        bitrate_signal: Optional[BitrateAdaptationSignal],
    ) -> Tuple[Optional[float], List[str]]:
        """
        Calculates a transparent quality suitability score [0.0, 1.0] from valid visual quality metrics:
        - VMAF: normalized as vmaf / 100.0 (range 0..1)
        - PSNR: normalized as min(1.0, max(0.0, (psnr - 25.0) / 15.0)) (PSNR 25dB=0, 40dB=1)
        - SSIM: ssim (range 0..1)

        Averages available valid normalized scores. Does NOT fabricate scores when missing.
        Returns (score, warnings). If quality_evaluable is False, score is None.
        """
        warnings = []
        if bitrate_signal is None or not bitrate_signal.quality_evaluable:
            warnings.append("quality_metrics_unavailable_quality_evaluable_false")
            return None, warnings

        scores = []
        if bitrate_signal.vmaf is not None and not math.isnan(bitrate_signal.vmaf):
            vmaf_norm = max(0.0, min(1.0, bitrate_signal.vmaf / 100.0))
            scores.append(vmaf_norm)

        if bitrate_signal.psnr_db is not None and not math.isnan(bitrate_signal.psnr_db):
            psnr_norm = max(0.0, min(1.0, (bitrate_signal.psnr_db - 25.0) / 15.0))
            scores.append(psnr_norm)

        if bitrate_signal.ssim is not None and not math.isnan(bitrate_signal.ssim):
            ssim_norm = max(0.0, min(1.0, bitrate_signal.ssim))
            scores.append(ssim_norm)

        if not scores:
            warnings.append("no_valid_numerical_quality_metrics_present")
            return None, warnings

        avg_score = sum(scores) / len(scores)
        return avg_score, warnings

    # ------------------------------------------------------------------------
    # 10.3 Fuzzy Inputs & Fuzzification
    # ------------------------------------------------------------------------

    @staticmethod
    def fuzzify_inputs(
        candidate: Dict[str, Any],
        quality_score: Optional[float],
    ) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
        """
        Fuzzifies the 5 fuzzy input variables using explicit membership functions:

        A. real_time_ratio [0..2.0+]
           - poor: trapezoid [0, 0, 0.5, 0.8]
           - moderate: triangle [0.6, 0.85, 1.1]
           - good: trapezoid [0.95, 1.2, 3.0, 3.0]

        B. bandwidth_saving_percent [-50..100%]
           - low: trapezoid [-50, -50, 0, 25]
           - medium: triangle [15, 35, 55]
           - high: trapezoid [45, 65, 100, 100]

        C. quality_suitability [0..1.0] (if missing, defaults to neutral 0.5 with warning)
           - poor: trapezoid [0, 0, 0.3, 0.5]
           - acceptable: triangle [0.4, 0.6, 0.8]
           - good: trapezoid [0.7, 0.85, 1.0, 1.0]

        D. edge_resource_condition [0..1.0] (derived from 1.0 - cpu_utilization / 100)
           - constrained: trapezoid [0, 0, 0.15, 0.35]
           - moderate: triangle [0.25, 0.50, 0.75]
           - available: trapezoid [0.65, 0.85, 1.0, 1.0]

        E. network_condition [0..1.0] (derived from network telemetry or neutral 0.5)
           - poor: trapezoid [0, 0, 0.25, 0.50]
           - moderate: triangle [0.35, 0.60, 0.80]
           - good: trapezoid [0.70, 0.85, 1.0, 1.0]
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

        # C. quality_suitability
        if quality_score is None:
            q_val = 0.5  # Neutral fallback when unmeasured
            warnings.append("missing_quality_metrics_using_neutral_fuzzy_membership")
        else:
            q_val = quality_score
        f_q = {
            "poor": trapezoid_mf(q_val, 0.0, 0.0, 0.3, 0.5),
            "acceptable": triangle_mf(q_val, 0.4, 0.6, 0.8),
            "good": trapezoid_mf(q_val, 0.7, 0.85, 1.0, 1.0),
        }

        # D. edge_resource_condition (1.0 - CPU_load / 100.0)
        res_avail = resource.resource_availability or {}
        cpu_load = res_avail.get("cpu_utilization_percent")
        if cpu_load is None:
            res_val = 0.5
            warnings.append("missing_cpu_telemetry_using_neutral_resource_condition")
        else:
            res_val = max(0.0, min(1.0, (100.0 - cpu_load) / 100.0))
        f_res = {
            "constrained": trapezoid_mf(res_val, 0.0, 0.0, 0.15, 0.35),
            "moderate": triangle_mf(res_val, 0.25, 0.50, 0.75),
            "available": trapezoid_mf(res_val, 0.65, 0.85, 1.0, 1.0),
        }

        # E. network_condition
        net_telemetry = resource.network_telemetry or {}
        rtt = net_telemetry.get("cloud_edge_rtt_ms")
        bw_mbps = net_telemetry.get("measured_bandwidth_mbps")
        if rtt is not None:
            # RTT: < 20ms = 1.0, > 100ms = 0.0
            net_val = max(0.0, min(1.0, 1.0 - (rtt - 20.0) / 80.0))
        elif bw_mbps is not None:
            # Bandwidth: > 50Mbps = 1.0, < 5Mbps = 0.0
            net_val = max(0.0, min(1.0, (bw_mbps - 5.0) / 45.0))
        else:
            net_val = 0.5
            warnings.append("missing_network_telemetry_using_neutral_network_condition")

        f_net = {
            "poor": trapezoid_mf(net_val, 0.0, 0.0, 0.25, 0.50),
            "moderate": triangle_mf(net_val, 0.35, 0.60, 0.80),
            "good": trapezoid_mf(net_val, 0.70, 0.85, 1.0, 1.0),
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
        Evaluates the Mamdani rule base using AND = min() logic.
        Returns a dictionary mapping linguistic output levels
        ("very_low", "low", "medium", "high", "very_high") to aggregated firing strength (max).

        Rule Base Definition:
        ---------------------
        R1: IF real_time_ratio IS good AND bandwidth_saving IS high AND quality IS good
            AND resource_condition IS available AND network_condition IS good
            THEN suitability IS very_high (Rationale: Ideal optimal conditions)

        R2: IF real_time_ratio IS good AND bandwidth_saving IS high AND quality IS acceptable
            AND resource_condition IS moderate THEN suitability IS high
            (Rationale: Strong performance with minor resource load)

        R3: IF real_time_ratio IS good AND quality IS good AND bandwidth_saving IS medium
            AND resource_condition IS available THEN suitability IS high
            (Rationale: Excellent visual enhancement and compute headroom)

        R4: IF real_time_ratio IS moderate AND bandwidth_saving IS medium AND quality IS acceptable
            THEN suitability IS medium (Rationale: Balanced candidate)

        R5: IF bandwidth_saving IS low AND quality IS poor THEN suitability IS low
            (Rationale: Low quality return for minor bandwidth saving)

        R6: IF real_time_ratio IS poor OR resource_condition IS constrained THEN suitability IS very_low
            (Rationale: Severe bottleneck risk)

        R7: IF real_time_ratio IS poor THEN suitability IS very_low
            (Rationale: Cannot sustain real-time playback budget)

        R8: IF network_condition IS poor THEN suitability IS low
            (Rationale: Adverse network path degrades delivery reliability)
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

        # R1 (Optimal): Ideal real-time, high bandwidth saving, good quality, available compute, good network
        r1 = min(rt["good"], bw["high"], q["good"], res["available"], net["good"])
        activations["very_high"] = max(activations["very_high"], r1)

        # R2 (High Performance - Good/Acceptable Quality & Compute Headroom)
        r2 = min(rt["good"], max(bw["high"], bw["medium"]), max(q["good"], q["acceptable"]), max(res["available"], res["moderate"]))
        activations["high"] = max(activations["high"], r2)

        # R3 (High Performance - Bandwidth Saving & Real-Time)
        r3 = min(rt["good"], max(bw["high"], bw["medium"]), max(res["available"], res["moderate"]))
        activations["high"] = max(activations["high"], r3)

        # R4 (Moderate Tradeoff - Moderate RT or Medium BW saving)
        r4 = min(max(rt["good"], rt["moderate"]), max(bw["medium"], bw["low"]), max(q["acceptable"], q["poor"]))
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
    # 10.5 Centroid Defuzzification
    # ------------------------------------------------------------------------

    @staticmethod
    def defuzzify_centroid(activations: Dict[str, float]) -> float:
        """
        Centroid (Center-of-Area) defuzzification over domain [0, 100] with step 0.5.

        Output Membership Shapes:
        - very_low: trapezoid [0, 0, 10, 25]
        - low: triangle [15, 30, 45]
        - medium: triangle [35, 50, 65]
        - high: triangle [55, 70, 85]
        - very_high: trapezoid [75, 90, 100, 100]
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
            # Aggregated membership mu(y) = max_t min(activation_t, mf_t(y))
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
            return 0.0

        return numerator / denominator

    @staticmethod
    def classify_suitability_label(score: float) -> str:
        """Classifies defuzzified score into linguistic label."""
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
        3. Fuzzify & evaluate Mamdani rules for hard-feasible candidates.
        4. Defuzzify suitability scores via centroid integration.
        5. Filter candidates meeting min_suitability_threshold.
        6. Select optimal candidate (with deterministic tie-breaking).
        7. Produce machine-readable FuzzyDecisionSignal.
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
                    quality_suitability_score=None,
                    defuzzified_suitability=None,
                    suitability_label="infeasible",
                    rule_activations={},
                )
                all_evaluations.append(eval_item)
                continue

            # Feasible -> Fuzzy inference
            q_score, q_warns = self.calculate_quality_suitability(cand["bitrate_signal"])
            all_warnings.extend(q_warns)

            fuzzified, f_warns = self.fuzzify_inputs(cand, q_score)
            all_warnings.extend(f_warns)

            rule_acts = self.evaluate_rules(fuzzified)
            score = self.defuzzify_centroid(rule_acts)
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
                quality_suitability_score=q_score,
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
                warnings=list(set(all_warnings)) + ["all_candidates_failed_hard_feasibility_gate"],
            )

        # Filter by minimum suitability threshold
        eligible_evaluations = [
            e
            for e in feasible_evaluations
            if e.defuzzified_suitability is not None
            and e.defuzzified_suitability >= self.min_suitability_threshold
        ]

        if not eligible_evaluations:
            for f_item in feasible_evaluations:
                rej_dict = f_item.to_dict()
                rej_dict["rejection_reasons"] = [
                    f"below_min_suitability_threshold_{self.min_suitability_threshold}"
                ]
                rejected_items.append(rej_dict)

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
                + [f"no_candidate_met_min_suitability_threshold_{self.min_suitability_threshold}"],
            )

        # Sort with deterministic tie-breaking:
        # 1. Primary: defuzzified_suitability (descending)
        # 2. Secondary: real_time_ratio (descending)
        # 3. Tertiary: bitrate_saving_percent (descending)
        # 4. Quaternary: candidate_id (alphabetical ascending)
        def sort_key(item: CandidateEvaluation):
            cand_ref = next(c for c in candidates if c["candidate_id"] == item.candidate_id)
            fps_sig: FPSAdaptationSignal = cand_ref["fps_signal"]
            bitrate_sig: BitrateAdaptationSignal = cand_ref["bitrate_signal"]
            rt_ratio = fps_sig.real_time_ratio if fps_sig else 0.0
            bw_saving = (
                bitrate_sig.bitrate_saving_percent
                if (bitrate_sig and bitrate_sig.bitrate_saving_percent is not None)
                else -999.0
            )
            # Use rounded suitability (to 4 decimal places) so floating point epsilon ties are broken deterministically
            return (
                round(item.defuzzified_suitability or 0.0, 4),
                round(rt_ratio, 4),
                round(bw_saving, 4),
                # invert string key for reverse sort compatibility if needed, or use custom tuple comparator
            )

        # Custom sorting logic to handle string ascending correctly
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
        winning_cand = next(c for c in candidates if c["candidate_id"] == winning_eval.candidate_id)

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
            },
            warnings=list(set(all_warnings)),
        )
