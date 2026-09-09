"""
adaptive_sr.adaptation.bitrate_adapter
======================================
Step 8 — Bitrate / Quality Adaptation Layer.

Measures the bandwidth-vs-quality tradeoff of available video representations
and SR-enhanced representations, producing a machine-readable BitrateAdaptationSignal.

This module evaluates bandwidth savings and quality availability for decision-making
in later steps (Steps 9/10). It does NOT make final representation or model selections.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class BitrateAdaptationSignal:
    reference_representation_id: str
    candidate_representation_id: str
    reference_bitrate_bps: float
    candidate_bitrate_bps: float
    bitrate_saving_percent: Optional[float]
    base_resolution: str
    target_resolution: str
    model_id: str
    scale: int
    device: str
    psnr_db: Optional[float]
    ssim: Optional[float]
    vmaf: Optional[float]
    quality_evaluable: bool
    quality_provenance: str
    measurement_provenance: str
    decision_eligible: bool
    quality_equivalent_to_native: bool
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BitrateAdapter:
    """
    Bitrate & Quality Adaptation Layer.
    Calculates bandwidth savings and ingests visual quality metrics (PSNR, SSIM, VMAF)
    to produce machine-readable adaptation signals.
    """

    @staticmethod
    def calculate_bitrate_saving(
        reference_bitrate_bps: Optional[float],
        candidate_bitrate_bps: Optional[float]
    ) -> Optional[float]:
        """
        Calculates percentage bandwidth saving:
        ((reference_bitrate - candidate_bitrate) / reference_bitrate) * 100
        Returns None if reference_bitrate <= 0 or missing.
        """
        if reference_bitrate_bps is None or reference_bitrate_bps <= 0.0:
            return None
        if candidate_bitrate_bps is None or candidate_bitrate_bps < 0.0:
            return None

        ref = float(reference_bitrate_bps)
        cand = float(candidate_bitrate_bps)
        saving = ((ref - cand) / ref) * 100.0
        return float(round(saving, 4))

    @classmethod
    def evaluate(
        cls,
        reference_representation_id: str,
        candidate_representation_id: str,
        reference_bitrate_bps: float,
        candidate_bitrate_bps: float,
        base_resolution: str = "640x360",
        target_resolution: str = "1280x720",
        model_id: str = "tinysr",
        scale: int = 2,
        device: str = "cpu",
        psnr_db: Optional[float] = None,
        ssim: Optional[float] = None,
        vmaf: Optional[float] = None,
        quality_provenance: str = "unmeasured",
        measurement_provenance: str = "direct_measurement",
        decision_eligible: bool = True,
        min_psnr_db: Optional[float] = None,
        min_ssim: Optional[float] = None,
    ) -> BitrateAdaptationSignal:
        """
        Evaluates the bandwidth-vs-quality tradeoff signal for a candidate SR configuration
        relative to a reference representation.
        """
        warnings: List[str] = []

        # Explicit requirement 124 / 8.4: Disclaimer for quality equivalence
        quality_equivalent_to_native = False
        warnings.append("LOWER BITRATE + SR DOES NOT AUTOMATICALLY MEAN EQUIVALENT QUALITY.")

        # Calculate bitrate saving
        saving_pct = cls.calculate_bitrate_saving(reference_bitrate_bps, candidate_bitrate_bps)
        if saving_pct is None:
            if reference_bitrate_bps is None or reference_bitrate_bps <= 0.0:
                warnings.append("Reference bitrate is zero or missing; bandwidth saving cannot be calculated.")
            if candidate_bitrate_bps is None or candidate_bitrate_bps < 0.0:
                warnings.append("Candidate bitrate is invalid or missing.")
        elif saving_pct < 0.0:
            warnings.append(f"Candidate bitrate ({candidate_bitrate_bps:.0f} bps) is higher than reference bitrate ({reference_bitrate_bps:.0f} bps).")
        elif saving_pct == 0.0:
            warnings.append("Candidate bitrate is equal to reference bitrate (0% bandwidth saving).")

        # Quality evaluability check
        has_psnr = psnr_db is not None and psnr_db > 0.0
        has_ssim = ssim is not None and ssim > 0.0
        has_vmaf = vmaf is not None and vmaf > 0.0
        quality_evaluable = has_psnr or has_ssim or has_vmaf

        if vmaf is None:
            warnings.append("VMAF metric unavailable or not measured.")

        if not quality_evaluable:
            warnings.append("No genuine quality metrics (PSNR/SSIM/VMAF) available for candidate configuration.")

        # Configurable quality threshold check (if min_psnr_db or min_ssim requested)
        if min_psnr_db is not None:
            if psnr_db is None or psnr_db < min_psnr_db:
                warnings.append(f"Configurable policy threshold unmet: PSNR ({psnr_db}) below target ({min_psnr_db} dB).")
        if min_ssim is not None:
            if ssim is None or ssim < min_ssim:
                warnings.append(f"Configurable policy threshold unmet: SSIM ({ssim}) below target ({min_ssim}).")

        if not decision_eligible:
            warnings.append("Configuration is not decision-eligible under Step 5 eligibility rules.")

        return BitrateAdaptationSignal(
            reference_representation_id=str(reference_representation_id),
            candidate_representation_id=str(candidate_representation_id),
            reference_bitrate_bps=float(reference_bitrate_bps) if reference_bitrate_bps is not None else 0.0,
            candidate_bitrate_bps=float(candidate_bitrate_bps) if candidate_bitrate_bps is not None else 0.0,
            bitrate_saving_percent=saving_pct,
            base_resolution=str(base_resolution),
            target_resolution=str(target_resolution),
            model_id=str(model_id),
            scale=int(scale),
            device=str(device),
            psnr_db=float(round(psnr_db, 4)) if psnr_db is not None else None,
            ssim=float(round(ssim, 4)) if ssim is not None else None,
            vmaf=float(round(vmaf, 4)) if vmaf is not None else None,
            quality_evaluable=bool(quality_evaluable),
            quality_provenance=str(quality_provenance),
            measurement_provenance=str(measurement_provenance),
            decision_eligible=bool(decision_eligible),
            quality_equivalent_to_native=quality_equivalent_to_native,
            warnings=warnings
        )

    @classmethod
    def evaluate_from_manifest_and_quality(
        cls,
        reference_rep: Dict[str, Any],
        candidate_rep: Dict[str, Any],
        quality_record: Optional[Dict[str, Any]] = None,
        model_id: str = "tinysr",
        scale: int = 2,
        device: str = "cpu"
    ) -> BitrateAdaptationSignal:
        """
        Evaluates bitrate/quality signal from manifest representation dicts and optional Step 5.6 quality record.
        """
        ref_id = reference_rep.get("representation_id") or "reference"
        cand_id = candidate_rep.get("representation_id") or "candidate"
        ref_bitrate = float(reference_rep.get("bitrate") or reference_rep.get("bitrate_bps") or 0.0)
        cand_bitrate = float(candidate_rep.get("bitrate") or candidate_rep.get("bitrate_bps") or 0.0)
        base_res = candidate_rep.get("resolution") or "640x360"
        target_res = reference_rep.get("resolution") or "1280x720"

        psnr_db = None
        ssim_val = None
        vmaf_val = None
        quality_prov = "unmeasured"
        decision_eligible = True

        if quality_record:
            psnr_db = quality_record.get("psnr_db") or quality_record.get("psnr")
            ssim_val = quality_record.get("ssim")
            vmaf_val = quality_record.get("vmaf")
            quality_prov = quality_record.get("evaluation_mode") or quality_record.get("quality_provenance") or "model_inference"
            if "decision_eligible" in quality_record:
                decision_eligible = bool(quality_record["decision_eligible"])

        return cls.evaluate(
            reference_representation_id=ref_id,
            candidate_representation_id=cand_id,
            reference_bitrate_bps=ref_bitrate,
            candidate_bitrate_bps=cand_bitrate,
            base_resolution=base_res,
            target_resolution=target_res,
            model_id=model_id,
            scale=scale,
            device=device,
            psnr_db=psnr_db,
            ssim=ssim_val,
            vmaf=vmaf_val,
            quality_provenance=quality_prov,
            measurement_provenance="step5.6_quality_eval" if quality_record else "manifest",
            decision_eligible=decision_eligible
        )
