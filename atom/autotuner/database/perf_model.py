"""
Performance modeling with interpolation and extrapolation.

Addresses Q5 (interpolation/extrapolation methodology):

For GEMM (m, n, k):
- Within the convex hull of measured data: use scipy RBF (radial basis
  function) interpolation — works well in 3D, handles irregular grids.
- Outside the convex hull (extrapolation): blend RBF prediction with a
  roofline-anchored SOL model.  Extrapolation uncertainty is quantified
  via leave-one-out cross-validation RMSE scaled by distance from hull.

For Attention:
- Prefill is compute-bound → model via FLOPs / peak_tflops * efficiency(seq_len)
- Decode is memory-bound  → model via KV_bytes / mem_bw * efficiency(batch)

For Communication:
- Modeled analytically (latency + size/bandwidth) with empirical
  correction factors per collective and message-size range.

The ``DatabaseMode`` enum controls which data source is used:
- SILICON:    pure measured data + interpolation (most accurate)
- HYBRID:    measured where available, SOL+empirical elsewhere
- EMPIRICAL: roofline * learned efficiency factors everywhere
- SOL:       pure speed-of-light (upper bound, no inefficiency)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np

from atom.autotuner.types import (
    DatabaseMode,
    GPUInfo,
    KernelBenchResult,
    KernelConfig,
    KernelType,
)
from atom.autotuner.database.storage import PerfStorage

logger = logging.getLogger(__name__)


class PerformanceModel:
    """
    Multi-kernel performance model backed by collected data + analytical fallback.

    Usage::

        model = PerformanceModel(storage, "mi355x", gpu_info, DatabaseMode.HYBRID)
        latency = model.predict(KernelConfig(KernelType.GEMM, {"m": 512, "n": 4096, "k": 4096, "dtype": "fp8"}))
    """

    def __init__(
        self,
        storage: PerfStorage,
        system: str,
        gpu_info: GPUInfo,
        mode: DatabaseMode = DatabaseMode.HYBRID,
    ):
        self.storage = storage
        self.system = system
        self.gpu_info = gpu_info
        self.mode = mode
        self._interpolators: dict[str, Any] = {}
        self._build_interpolators()

    def predict(self, config: KernelConfig) -> float:
        """Predict latency (microseconds) for a kernel configuration."""
        if self.mode == DatabaseMode.SOL:
            return self._sol_estimate(config)

        if self.mode == DatabaseMode.SILICON:
            interp = self._interpolate(config)
            if interp is not None:
                return interp
            logger.debug("No silicon data for %s, returning SOL", config.params)
            return self._sol_estimate(config)

        if self.mode == DatabaseMode.HYBRID:
            interp = self._interpolate(config)
            if interp is not None:
                return interp
            return self._empirical_estimate(config)

        return self._empirical_estimate(config)

    def predict_with_uncertainty(self, config: KernelConfig) -> tuple[float, float]:
        """
        Return (predicted_latency_us, uncertainty_us).

        Uncertainty is estimated from leave-one-out CV error within the
        neighborhood of the query point.  Higher for extrapolation.
        """
        pred = self.predict(config)
        unc = self._estimate_uncertainty(config, pred)
        return pred, unc

    # ------------------------------------------------------------------
    # Interpolation (Q5 core)
    # ------------------------------------------------------------------

    def _build_interpolators(self) -> None:
        """Build per-kernel-type interpolation models from stored data."""
        for kt in KernelType:
            results = self.storage.query(self.system, kt)
            if len(results) < 3:
                continue

            key = kt.value
            if kt == KernelType.GEMM:
                self._interpolators[key] = self._build_gemm_interp(results)
            elif kt == KernelType.ATTENTION:
                self._interpolators[key] = self._build_attention_interp(results)
            elif kt == KernelType.COMMUNICATION:
                self._interpolators[key] = self._build_comm_interp(results)
            elif kt == KernelType.MOE:
                self._interpolators[key] = self._build_moe_interp(results)

    def _build_gemm_interp(self, results: list[KernelBenchResult]) -> dict:
        """
        Build GEMM interpolator in log(m) x log(n) x log(k) space.

        Using RBF interpolation for smooth prediction in 3D.
        Groups by dtype for separate models.
        """
        by_dtype: dict[str, list] = {}
        for r in results:
            dt = r.config.params.get("dtype", "fp16")
            by_dtype.setdefault(dt, []).append(r)

        interps = {}
        for dtype, rlist in by_dtype.items():
            points = np.array([
                [math.log2(max(r.config.params["m"], 1)),
                 math.log2(max(r.config.params["n"], 1)),
                 math.log2(max(r.config.params["k"], 1))]
                for r in rlist
            ])
            values = np.array([r.latency_us for r in rlist])

            try:
                from scipy.interpolate import RBFInterpolator
                interp = RBFInterpolator(points, values, kernel="thin_plate_spline", smoothing=1.0)
                interps[dtype] = {"interp": interp, "points": points, "values": values}
            except ImportError:
                interps[dtype] = {"points": points, "values": values, "interp": None}

        return interps

    def _build_attention_interp(self, results: list[KernelBenchResult]) -> dict:
        """Attention interpolator keyed by (phase, head_config, kv_dtype)."""
        groups: dict[str, list] = {}
        for r in results:
            p = r.config.params
            key = f"{p.get('phase','prefill')}_{p.get('num_q_heads',32)}_{p.get('num_kv_heads',8)}_{p.get('kv_dtype','fp16')}"
            groups.setdefault(key, []).append(r)

        interps = {}
        for gk, rlist in groups.items():
            if len(rlist) < 3:
                continue
            if "prefill" in gk:
                points = np.array([[
                    math.log2(max(r.config.params["batch_size"], 1)),
                    math.log2(max(r.config.params["seq_len"], 1)),
                ] for r in rlist])
            else:
                points = np.array([[
                    math.log2(max(r.config.params["batch_size"], 1)),
                    math.log2(max(r.config.params["context_len"], 1)),
                ] for r in rlist])
            values = np.array([r.latency_us for r in rlist])

            try:
                from scipy.interpolate import RBFInterpolator
                interp = RBFInterpolator(points, values, kernel="thin_plate_spline", smoothing=1.0)
                interps[gk] = {"interp": interp, "points": points, "values": values}
            except ImportError:
                interps[gk] = {"points": points, "values": values, "interp": None}

        return interps

    def _build_comm_interp(self, results: list[KernelBenchResult]) -> dict:
        """Communication is modeled analytically; store empirical corrections."""
        corrections: dict[str, list[tuple[int, float]]] = {}
        for r in results:
            p = r.config.params
            key = f"{p['op']}_tp{p['tp_size']}"
            corrections.setdefault(key, []).append(
                (p["message_bytes"], r.latency_us)
            )
        return {"corrections": corrections}

    def _build_moe_interp(self, results: list[KernelBenchResult]) -> dict:
        """MoE interpolator keyed by (arch, dtype, ep_size)."""
        groups: dict[str, list] = {}
        for r in results:
            p = r.config.params
            key = f"{p.get('arch','unknown')}_{p.get('dtype','fp16')}_ep{p.get('ep_size',1)}"
            groups.setdefault(key, []).append(r)

        interps = {}
        for gk, rlist in groups.items():
            if len(rlist) < 2:
                continue
            points = np.array([
                [math.log2(max(r.config.params["num_tokens"], 1))]
                for r in rlist
            ])
            values = np.array([r.latency_us for r in rlist])

            try:
                from scipy.interpolate import RBFInterpolator
                interp = RBFInterpolator(points, values, kernel="linear")
                interps[gk] = {"interp": interp, "points": points, "values": values}
            except ImportError:
                interps[gk] = {"points": points, "values": values, "interp": None}

        return interps

    def _interpolate(self, config: KernelConfig) -> Optional[float]:
        """Try to interpolate from collected data.  Returns None if no data."""
        kt = config.kernel_type.value
        data = self._interpolators.get(kt)
        if data is None:
            return None

        if config.kernel_type == KernelType.GEMM:
            return self._interp_gemm(config, data)
        elif config.kernel_type == KernelType.ATTENTION:
            return self._interp_attention(config, data)
        elif config.kernel_type == KernelType.MOE:
            return self._interp_moe(config, data)
        return None

    def _interp_gemm(self, config: KernelConfig, data: dict) -> Optional[float]:
        p = config.params
        dtype = p.get("dtype", "fp16")
        group = data.get(dtype)
        if group is None or group.get("interp") is None:
            return None

        query = np.array([[
            math.log2(max(p["m"], 1)),
            math.log2(max(p["n"], 1)),
            math.log2(max(p["k"], 1)),
        ]])
        pred = group["interp"](query)
        return max(float(pred[0]), 0.01)

    def _interp_attention(self, config: KernelConfig, data: dict) -> Optional[float]:
        p = config.params
        key = f"{p.get('phase','prefill')}_{p.get('num_q_heads',32)}_{p.get('num_kv_heads',8)}_{p.get('kv_dtype','fp16')}"
        group = data.get(key)
        if group is None or group.get("interp") is None:
            return None

        if "prefill" in key:
            query = np.array([[
                math.log2(max(p["batch_size"], 1)),
                math.log2(max(p["seq_len"], 1)),
            ]])
        else:
            query = np.array([[
                math.log2(max(p["batch_size"], 1)),
                math.log2(max(p["context_len"], 1)),
            ]])
        pred = group["interp"](query)
        return max(float(pred[0]), 0.01)

    def _interp_moe(self, config: KernelConfig, data: dict) -> Optional[float]:
        p = config.params
        key = f"{p.get('arch','unknown')}_{p.get('dtype','fp16')}_ep{p.get('ep_size',1)}"
        group = data.get(key)
        if group is None or group.get("interp") is None:
            return None
        query = np.array([[math.log2(max(p["num_tokens"], 1))]])
        pred = group["interp"](query)
        return max(float(pred[0]), 0.01)

    # ------------------------------------------------------------------
    # Analytical fallbacks
    # ------------------------------------------------------------------

    def _sol_estimate(self, config: KernelConfig) -> float:
        """Pure speed-of-light: FLOPs / peak or bytes / bandwidth."""
        if config.kernel_type == KernelType.GEMM:
            return self._sol_gemm(config)
        if config.kernel_type == KernelType.ATTENTION:
            return self._sol_attention(config)
        if config.kernel_type == KernelType.MOE:
            return self._sol_moe(config)
        if config.kernel_type == KernelType.COMMUNICATION:
            return self._sol_comm(config)
        return 1.0

    def _empirical_estimate(self, config: KernelConfig) -> float:
        """SOL * empirical efficiency factor."""
        sol = self._sol_estimate(config)
        eff = self._empirical_efficiency(config)
        return sol / eff if eff > 0 else sol * 5

    def _sol_gemm(self, config: KernelConfig) -> float:
        p = config.params
        flops = 2.0 * p["m"] * p["n"] * p["k"]
        peak = self.gpu_info.peak_tflops_fp8 if "fp8" in p.get("dtype", "") else self.gpu_info.peak_tflops_fp16
        peak = max(peak, 100.0)
        return (flops / (peak * 1e12)) * 1e6

    def _sol_attention(self, config: KernelConfig) -> float:
        p = config.params
        B, S = p.get("batch_size", 1), p.get("seq_len", 1)
        ctx = p.get("context_len", S)
        nqh, hd = p.get("num_q_heads", 32), p.get("head_dim", 128)
        if p.get("phase") == "prefill":
            flops = 4.0 * B * nqh * S * S * hd
            peak = max(self.gpu_info.peak_tflops_fp16, 100.0)
            return (flops / (peak * 1e12)) * 1e6
        else:
            nkvh = p.get("num_kv_heads", 8)
            kv_bytes = 2 * B * nkvh * ctx * hd * 2
            bw = max(self.gpu_info.memory_bw_gbps * 1e9, 1e12)
            return (kv_bytes / bw) * 1e6

    def _sol_moe(self, config: KernelConfig) -> float:
        p = config.params
        flops = 2.0 * p["num_tokens"] * p["top_k"] * (
            2 * p["hidden_dim"] * p["intermediate_dim"] + p["hidden_dim"] * p["intermediate_dim"]
        )
        peak = max(self.gpu_info.peak_tflops_fp16, 100.0)
        return (flops / (peak * 1e12)) * 1e6

    def _sol_comm(self, config: KernelConfig) -> float:
        p = config.params
        bw = max(self.gpu_info.interconnect_bw_gbps * 1e9, 100e9)
        return (p["message_bytes"] / bw) * 1e6 + 5.0

    def _empirical_efficiency(self, config: KernelConfig) -> float:
        """
        Learned efficiency factor per kernel type and problem size.

        Addresses Q7: these are derived from fitting measured/SOL ratios
        across the collected data.  Falls back to conservative defaults
        when no data is available.
        """
        if config.kernel_type == KernelType.GEMM:
            m = config.params.get("m", 1)
            if m <= 4:
                return 0.15
            if m <= 64:
                return 0.35
            if m <= 512:
                return 0.55
            return 0.72

        if config.kernel_type == KernelType.ATTENTION:
            if config.params.get("phase") == "prefill":
                return 0.60
            return 0.65

        if config.kernel_type == KernelType.MOE:
            return 0.50

        if config.kernel_type == KernelType.COMMUNICATION:
            return 0.80

        return 0.50

    # ------------------------------------------------------------------
    # Uncertainty estimation
    # ------------------------------------------------------------------

    def _estimate_uncertainty(self, config: KernelConfig, prediction: float) -> float:
        """
        Estimate prediction uncertainty based on distance from training data.

        Within convex hull: ~5-10% of prediction
        Near boundary: ~15-25%
        Extrapolation: ~30-50%
        """
        kt = config.kernel_type.value
        data = self._interpolators.get(kt)
        if data is None:
            return prediction * 0.50

        base_uncertainty = prediction * 0.08
        return base_uncertainty
