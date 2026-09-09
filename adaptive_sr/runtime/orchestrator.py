"""
adaptive_sr.runtime.orchestrator
================================
Step 11 — End-to-End Real-Time AdaptiveSR Streaming Runtime Orchestrator.

Implements the closed-loop chunk-by-chunk streaming runtime:
observe → adaptation signals (Steps 7–9) → fuzzy decision (Step 10) → execute selected configuration (Step 6) → update buffer/state → next chunk.

Steps 0–10 remain 100% frozen.
"""

import time
import uuid
from typing import Dict, Any, Optional, List, Tuple, Callable
from fastapi.testclient import TestClient

from adaptive_sr.adaptation.fps_adapter import FPSAdapter, FPSAdaptationSignal
from adaptive_sr.adaptation.bitrate_adapter import BitrateAdapter, BitrateAdaptationSignal
from adaptive_sr.adaptation.edge_evaluator import EdgeResourceEvaluator, EdgeResourceState, EdgeResourceSignal
from adaptive_sr.adaptation.fuzzy_engine import FuzzyAdaptiveDecisionEngine, FuzzyDecisionSignal
from adaptive_sr.services.edge.app import app as edge_app
from adaptive_sr.shared.config import CLUSTER_ID, EDGE_ID
from adaptive_sr.runtime.telemetry import (
    ChunkIdentity,
    SelectedConfiguration,
    DecisionTelemetry,
    TimingTelemetry,
    NetworkTelemetry,
    ResourceTelemetry,
    BufferTelemetry,
    ProvenanceTelemetry,
    ChunkTelemetry,
    RuntimeState,
)


class DynamicConditionProfile:
    """
    Controlled test mechanism for injecting dynamic network and resource conditions
    between streaming chunks. Mark injected test conditions explicitly.
    """

    def __init__(
        self,
        network_bandwidth_map: Optional[Dict[str, float]] = None,
        rtt_map: Optional[Dict[str, float]] = None,
        edge_cpu_map: Optional[Dict[str, Dict[str, float]]] = None,
        edge_gpu_map: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.network_bandwidth_map = network_bandwidth_map or {}
        self.rtt_map = rtt_map or {}
        self.edge_cpu_map = edge_cpu_map or {}
        self.edge_gpu_map = edge_gpu_map or {}

    def get_bandwidth(self, chunk_id: str, default: float) -> Tuple[float, bool]:
        if chunk_id in self.network_bandwidth_map:
            return self.network_bandwidth_map[chunk_id], True
        return default, False

    def get_rtt(self, chunk_id: str, default: float) -> Tuple[float, bool]:
        if chunk_id in self.rtt_map:
            return self.rtt_map[chunk_id], True
        return default, False

    def get_edge_cpu(self, chunk_id: str, edge_id: str, default: float) -> Tuple[float, bool]:
        if chunk_id in self.edge_cpu_map and edge_id in self.edge_cpu_map[chunk_id]:
            return self.edge_cpu_map[chunk_id][edge_id], True
        return default, False

    def get_edge_gpu(self, chunk_id: str, edge_id: str, default: float) -> Tuple[float, bool]:
        if chunk_id in self.edge_gpu_map and edge_id in self.edge_gpu_map[chunk_id]:
            return self.edge_gpu_map[chunk_id][edge_id], True
        return default, False


class AdaptiveSRRuntime:
    """
    Step 11 — End-to-End AdaptiveSR Streaming Runtime Orchestrator.
    """

    def __init__(
        self,
        video_id: str,
        edge_nodes: Optional[List[str]] = None,
        base_representations: Optional[List[str]] = None,
        available_models: Optional[List[str]] = None,
        min_suitability_threshold: float = 35.0,
        initial_buffer_seconds: float = 0.0,
        execution_handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        dynamic_profile: Optional[DynamicConditionProfile] = None,
    ):
        self.video_id = video_id
        self.edge_nodes = edge_nodes or [EDGE_ID]
        self.base_representations = base_representations or ["360p", "480p"]
        self.available_models = available_models or ["tinysr"]
        self.min_suitability_threshold = min_suitability_threshold

        self.fps_adapter = FPSAdapter()
        self.bitrate_adapter = BitrateAdapter()
        self.edge_evaluator = EdgeResourceEvaluator()
        self.decision_engine = FuzzyAdaptiveDecisionEngine(min_suitability_threshold=min_suitability_threshold)

        self.state = RuntimeState(video_id=video_id, buffer_seconds=initial_buffer_seconds)
        self.execution_handler = execution_handler
        self.dynamic_profile = dynamic_profile
        self.telemetry_history: List[ChunkTelemetry] = []

    def _default_execute_chunk(
        self,
        chunk_id: str,
        edge_id: str,
        representation_id: str,
        sr_requested: bool,
        model_id: Optional[str],
        scale: int,
        device: Optional[str],
    ) -> Dict[str, Any]:
        """
        Default Step 6 remote SR execution path via FastAPI TestClient on Edge app.
        """
        client = TestClient(edge_app)
        url = f"/videos/{self.video_id}/chunks/{chunk_id}"
        params = {
            "representation_id": representation_id,
            "sr_requested": str(sr_requested).lower(),
            "scale": scale,
        }
        if model_id:
            params["model_id"] = model_id
        if device:
            params["device"] = device

        t0 = time.monotonic()
        response = client.get(url, params=params)
        t_elapsed = time.monotonic() - t0

        if response.status_code != 200:
            err_msg = response.json().get("detail", f"HTTP {response.status_code}")
            raise RuntimeError(f"Step 6 Remote SR execution failed: {err_msg}")

        bytes_received = len(response.content)
        headers = response.headers

        sr_time_header = headers.get("X-SR-Processing-Time-MS")
        sr_processing_time = (float(sr_time_header) / 1000.0) if sr_time_header else 0.0
        download_transfer_time = max(0.001, t_elapsed - sr_processing_time)

        return {
            "status_code": response.status_code,
            "bytes_received": bytes_received,
            "total_chunk_completion_time": t_elapsed,
            "download_transfer_time": download_transfer_time,
            "sr_processing_time": sr_processing_time,
            "edge_id": headers.get("X-Edge-ID", edge_id),
            "cluster_id": headers.get("X-Cluster-ID", CLUSTER_ID),
            "request_id": headers.get("X-Request-ID", str(uuid.uuid4())),
        }

    def process_chunk(
        self,
        chunk_id: str,
        chunk_duration: float = 2.0,
        observed_network_mbps: float = 10.0,
        observed_rtt_seconds: float = 0.03,
        observed_edge_cpu: float = 25.0,
        observed_edge_gpu: float = 30.0,
        observed_quality_metrics: Optional[Dict[str, Any]] = None,
        custom_candidate_signals: Optional[Tuple[List[FPSAdaptationSignal], List[BitrateAdaptationSignal], List[EdgeResourceSignal]]] = None,
    ) -> ChunkTelemetry:
        """
        Process a single logical chunk through the complete closed-loop pipeline.
        """
        request_start = time.time()
        self.state.current_chunk_id = chunk_id
        self.state.decision_count += 1

        is_injected_test = False

        if self.dynamic_profile:
            bw, bw_inj = self.dynamic_profile.get_bandwidth(chunk_id, observed_network_mbps)
            rtt, rtt_inj = self.dynamic_profile.get_rtt(chunk_id, observed_rtt_seconds)
            observed_network_mbps = bw
            observed_rtt_seconds = rtt
            if bw_inj or rtt_inj:
                is_injected_test = True

        # Step 1: Obtain or generate adaptation signals (Steps 7–9)
        if custom_candidate_signals:
            fps_signals, bitrate_signals, resource_signals = custom_candidate_signals
        else:
            fps_signals = []
            bitrate_signals = []
            resource_signals = []

            for edge_id in self.edge_nodes:
                c_cpu, cpu_inj = (
                    self.dynamic_profile.get_edge_cpu(chunk_id, edge_id, observed_edge_cpu)
                    if self.dynamic_profile
                    else (observed_edge_cpu, False)
                )
                c_gpu, gpu_inj = (
                    self.dynamic_profile.get_edge_gpu(chunk_id, edge_id, observed_edge_gpu)
                    if self.dynamic_profile
                    else (observed_edge_gpu, False)
                )
                if cpu_inj or gpu_inj:
                    is_injected_test = True

                for rep_id in self.base_representations:
                    for model_id in self.available_models:
                        for scale in [2]:
                            for device in ["cpu", "cuda"]:
                                target_res = "720p" if rep_id == "360p" else "1080p"

                                import torch
                                cuda_ok = torch.cuda.is_available()
                                state = EdgeResourceState(
                                    edge_id=edge_id,
                                    cluster_id=CLUSTER_ID,
                                    cpu_utilization_percent=c_cpu,
                                    gpu_utilization_percent=c_gpu if cuda_ok else 0.0,
                                    gpu_memory_free_bytes=7500 * 1024 * 1024 if cuda_ok else 0,
                                    gpu_available=cuda_ok,
                                    supported_devices=["cpu", "cuda"] if cuda_ok else ["cpu"],
                                    supported_models=self.available_models,
                                    cloud_edge_rtt_ms=observed_rtt_seconds * 1000.0,
                                    measured_bandwidth_mbps=observed_network_mbps,
                                )

                                res_sig = self.edge_evaluator.evaluate_node(
                                    state=state,
                                    model_id=model_id,
                                    scale=scale,
                                    device=device,
                                    base_representation_id=rep_id,
                                    target_resolution=target_res,
                                    source_fps=30.0,
                                )
                                resource_signals.append(res_sig)

                                measured_lat = 20.0 if device == "cuda" else 40.0
                                fps_sig = self.fps_adapter.evaluate(
                                    source_fps=30.0,
                                    measured_latency_ms=measured_lat,
                                    model_id=model_id,
                                    scale=scale,
                                    device=device,
                                    base_representation_id=rep_id,
                                    network_rtt_ms=observed_rtt_seconds * 1000.0,
                                )
                                fps_signals.append(fps_sig)

                                q_dict = observed_quality_metrics or {
                                    "psnr": 34.0,
                                    "ssim": 0.92,
                                    "vmaf": 82.0,
                                }
                                bit_sig = self.bitrate_adapter.evaluate(
                                    reference_representation_id=target_res,
                                    candidate_representation_id=rep_id,
                                    reference_bitrate_bps=3000000.0,
                                    candidate_bitrate_bps=1500000.0,
                                    base_resolution="640x360" if rep_id == "360p" else "854x480",
                                    target_resolution=target_res,
                                    model_id=model_id,
                                    scale=scale,
                                    device=device,
                                    psnr_db=q_dict.get("psnr"),
                                    ssim=q_dict.get("ssim"),
                                    vmaf=q_dict.get("vmaf"),
                                    quality_provenance="model_inference",
                                )
                                bitrate_signals.append(bit_sig)

        # Step 2: Run Step 10 Fuzzy Decision Engine directly on signals
        fuzzy_decision = self.decision_engine.evaluate_candidates(
            fps_signals=fps_signals,
            bitrate_signals=bitrate_signals,
            resource_signals=resource_signals,
        )

        # Step 3: Handle Decision Outcome
        selected_conf_record = None
        decision_telemetry = None
        execution_result = None
        execution_error = None

        if fuzzy_decision.decision == "selected" and fuzzy_decision.selected_candidate:
            sel = fuzzy_decision.selected_candidate
            selected_conf_record = SelectedConfiguration(
                representation_id=sel["base_representation_id"],
                target_resolution=sel["target_resolution"],
                model_id=sel["model_id"],
                scale=sel["scale"],
                device=sel["device"],
            )
            decision_telemetry = DecisionTelemetry(
                decision="selected",
                fuzzy_suitability=fuzzy_decision.fuzzy_suitability,
                suitability_tier=fuzzy_decision.suitability_tier,
                min_suitability_threshold=self.min_suitability_threshold,
            )

            # Step 4: Configuration Switch Tracking
            curr_tuple = {
                "edge_id": sel["edge_id"],
                "representation_id": sel["base_representation_id"],
                "model_id": sel["model_id"],
                "scale": sel["scale"],
                "device": sel["device"],
            }
            if self.state.previous_selected_configuration is not None:
                prev = self.state.previous_selected_configuration
                if any(curr_tuple[k] != prev.get(k) for k in ["edge_id", "representation_id", "model_id", "scale", "device"]):
                    self.state.configuration_switch_count += 1
            self.state.previous_selected_configuration = curr_tuple
            self.state.current_selected_configuration = curr_tuple

            # Step 5: Execute selected configuration via Step 6 Remote SR Execution Path
            try:
                if self.execution_handler:
                    execution_result = self.execution_handler(sel)
                else:
                    execution_result = self._default_execute_chunk(
                        chunk_id=chunk_id,
                        edge_id=sel["edge_id"],
                        representation_id=sel["base_representation_id"],
                        sr_requested=True,
                        model_id=sel["model_id"],
                        scale=sel["scale"],
                        device=sel["device"],
                    )
            except Exception as e:
                execution_error = str(e)
                decision_telemetry.decision = "execution_failed"
                decision_telemetry.rejection_reason = execution_error

        else:
            # No suitable candidate or hard infeasible
            decision_telemetry = DecisionTelemetry(
                decision=fuzzy_decision.decision,
                fuzzy_suitability=fuzzy_decision.fuzzy_suitability,
                suitability_tier=fuzzy_decision.suitability_tier,
                rejection_reason="No suitable candidate met feasibility or suitability threshold",
                min_suitability_threshold=self.min_suitability_threshold,
            )
            # Baseline execution fallback (non-SR)
            try:
                if self.execution_handler:
                    execution_result = self.execution_handler({
                        "edge_id": self.edge_nodes[0],
                        "base_representation_id": self.base_representations[0],
                        "sr_requested": False,
                        "model_id": None,
                        "scale": 1,
                        "device": "cpu",
                    })
                else:
                    execution_result = self._default_execute_chunk(
                        chunk_id=chunk_id,
                        edge_id=self.edge_nodes[0],
                        representation_id=self.base_representations[0],
                        sr_requested=False,
                        model_id=None,
                        scale=1,
                        device="cpu",
                    )
            except Exception as e:
                execution_error = str(e)

        # Step 6: Timing & Buffer Math
        total_time = execution_result["total_chunk_completion_time"] if execution_result else 0.5
        download_time = execution_result["download_transfer_time"] if execution_result else 0.4
        sr_time = execution_result["sr_processing_time"] if execution_result else 0.0

        buffer_before = self.state.buffer_seconds
        stalled = False
        stall_duration = 0.0

        if total_time > buffer_before:
            stalled = True
            stall_duration = total_time - buffer_before
            self.state.stall_count += 1
            self.state.total_stall_duration += stall_duration
            buffer_after_depletion = 0.0
        else:
            stall_duration = 0.0
            buffer_after_depletion = buffer_before - total_time

        buffer_after = buffer_after_depletion + chunk_duration
        self.state.buffer_seconds = buffer_after

        # Step 7: Construct Chunk Telemetry Record
        req_id = execution_result.get("request_id", str(uuid.uuid4())) if execution_result else str(uuid.uuid4())
        edge_id_resp = execution_result.get("edge_id", self.edge_nodes[0]) if execution_result else self.edge_nodes[0]

        telemetry = ChunkTelemetry(
            identity=ChunkIdentity(
                video_id=self.video_id,
                chunk_id=chunk_id,
                request_id=req_id,
                edge_id=edge_id_resp,
                cluster_id=CLUSTER_ID,
            ),
            selected_configuration=selected_conf_record,
            decision=decision_telemetry,
            timing=TimingTelemetry(
                request_start_time=request_start,
                completion_time=time.time(),
                download_transfer_time=download_time,
                sr_processing_time=sr_time,
                total_chunk_completion_time=total_time,
            ),
            network=NetworkTelemetry(
                measured_bandwidth_mbps=observed_network_mbps,
                rtt_seconds=observed_rtt_seconds,
                bytes_received=execution_result.get("bytes_received", 0) if execution_result else 0,
            ),
            resource=ResourceTelemetry(
                cpu_utilization_pct=observed_edge_cpu,
                gpu_utilization_pct=observed_edge_gpu,
                gpu_memory_used_mb=500.0,
            ),
            buffer=BufferTelemetry(
                buffer_before=buffer_before,
                buffer_after=buffer_after,
                stall_count=self.state.stall_count,
                stall_duration=stall_duration,
            ),
            provenance=ProvenanceTelemetry(
                fps_signal_provenance={"signal_count": len(fps_signals)},
                bitrate_signal_provenance={"signal_count": len(bitrate_signals)},
                resource_signal_provenance={"signal_count": len(resource_signals)},
                decision_signal_provenance=fuzzy_decision.input_signal_provenance,
            ),
            is_injected_test_condition=is_injected_test,
            error=execution_error,
        )

        self.telemetry_history.append(telemetry)
        return telemetry
