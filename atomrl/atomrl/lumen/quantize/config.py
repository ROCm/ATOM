# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Quantization configuration for Lumen.

Supports FP8 (E4M3 / E5M2 / HYBRID), MXFP8, MXFP4, and FP4 formats with
multiple scaling strategies.
"""

import functools
from dataclasses import dataclass
from enum import Enum

import torch

# ---------------------------------------------------------------------------
# FNUZ / OCP detection
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _is_fp8_fnuz() -> bool:
    """Return True when the current GPU uses FNUZ FP8 encodings (gfx94x)."""
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return (props.major, props.minor) < (9, 5)
    except (AssertionError, RuntimeError):
        return False


def _get_float8_e4m3() -> torch.dtype:
    return torch.float8_e4m3fnuz if _is_fp8_fnuz() else torch.float8_e4m3fn


def _get_float8_e5m2() -> torch.dtype:
    return torch.float8_e5m2fnuz if _is_fp8_fnuz() else torch.float8_e5m2


# ---------------------------------------------------------------------------
# FP8 representable-max values  (OCP, FNUZ)
# ---------------------------------------------------------------------------

_E4M3_MAX = (448.0, 240.0)  # (OCP, FNUZ)
_E5M2_MAX = (57344.0, 57344.0)
FP4_E2M1_MAX = 6.0  # 1 sign + 2 exponent + 1 mantissa, no NaN/Inf


def get_fp8_max(fmt: "QuantFormat") -> float:
    """Return the maximum representable FP8 value for *fmt* on this GPU."""
    idx = 1 if _is_fp8_fnuz() else 0
    if fmt in (QuantFormat.FP8_E4M3, QuantFormat.MXFP8, QuantFormat.HYBRID):
        return _E4M3_MAX[idx]
    elif fmt == QuantFormat.FP8_E5M2:
        return _E5M2_MAX[idx]
    return _E4M3_MAX[idx]


def get_fp8_max_bwd(fmt: "QuantFormat") -> float:
    """Return the backward-pass FP8 max (differs from fwd for HYBRID)."""
    idx = 1 if _is_fp8_fnuz() else 0
    if fmt == QuantFormat.HYBRID:
        return _E5M2_MAX[idx]
    return get_fp8_max(fmt)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class QuantFormat(Enum):
    """Supported low-precision number formats."""

    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    HYBRID = "hybrid"  # E4M3 forward, E5M2 backward (TE-style)
    MXFP8 = "mxfp8"
    MXFP4 = "mxfp4"
    FP4 = "fp4"


# The MXFP4 block size is fixed by the format, not chosen per config: the E8M0
# scale layout and the AITER kernels both assume 32.
MXFP4_BLOCK_SIZE = 32


class ScalingType(Enum):
    """How scaling factors are computed."""

    DYNAMIC = "dynamic"  # Scale from current tensor amax (= current per-tensor)
    DELAYED = "delayed"  # Scale from amax history (TE-style delayed per-tensor)
    BLOCKWISE = "blockwise"  # 1×block per-group scaling along axis=1 (both X and W)
    BLOCKWISE2D = "blockwise2d"  # Jet-RL / DeepSeek-V3: activation 1×block, weight block×block 2D tiles
    PER_TOKEN = "per_token"  # Per-row (per-token) dynamic scaling
    NONE = "none"  # No quantization (BF16 passthrough)


class AmaxAlgo(Enum):
    """How delayed-scaling amax is derived from the history window."""

    MAX = "max"  # max over the entire history window (TE default)
    MOST_RECENT = "most_recent"  # use only the latest recorded amax


# ---------------------------------------------------------------------------
# Format → dtype mapping (auto-detects FNUZ)
# ---------------------------------------------------------------------------


def _build_format_to_dtype():
    e4m3 = _get_float8_e4m3()
    e5m2 = _get_float8_e5m2()
    return {
        QuantFormat.FP8_E4M3: e4m3,
        QuantFormat.FP8_E5M2: e5m2,
        QuantFormat.HYBRID: e4m3,  # forward dtype
        QuantFormat.MXFP8: e4m3,
        QuantFormat.MXFP4: None,  # packed uint8, no native PyTorch dtype
        QuantFormat.FP4: None,
    }


def _format_to_dtype_bwd():
    e4m3 = _get_float8_e4m3()
    e5m2 = _get_float8_e5m2()
    return {
        QuantFormat.FP8_E4M3: e4m3,
        QuantFormat.FP8_E5M2: e5m2,
        QuantFormat.HYBRID: e5m2,  # backward dtype differs
        QuantFormat.MXFP8: e4m3,
        QuantFormat.MXFP4: None,
        QuantFormat.FP4: None,
    }


# ---------------------------------------------------------------------------
# QuantConfig
# ---------------------------------------------------------------------------


@dataclass
class QuantConfig:
    """Unified quantization configuration.

    Examples::

        # FP8 with delayed scaling (default)
        cfg = QuantConfig()

        # MXFP8 with blockwise scaling
        cfg = QuantConfig(format=QuantFormat.MXFP8,
                          scaling=ScalingType.BLOCKWISE,
                          block_size=32)

        # MLPerf-style: most_recent amax, short history, activation quant on
        cfg = QuantConfig(amax_algo=AmaxAlgo.MOST_RECENT,
                          history_len=4,
                          quantize_activation=True)

        # HYBRID format with margin (TE-style)
        cfg = QuantConfig(format=QuantFormat.HYBRID, margin=0)

        # From strings (handy for YAML / env-var configs)
        cfg = QuantConfig.from_str("fp8_e4m3", "delayed",
                                   amax_algo="most_recent")
    """

    format: QuantFormat = QuantFormat.FP8_E4M3
    scaling: ScalingType = ScalingType.DELAYED
    block_size: int = 32
    history_len: int = 16

    # Amax algorithm for delayed scaling
    amax_algo: AmaxAlgo = AmaxAlgo.MAX

    # Margin for scaling factor computation (TE-compatible).
    # ``sf = (FP8_MAX / amax) / (2 ** margin)``
    margin: int = 0

    # Whether to all-reduce amax across data-parallel ranks before computing
    # the scale.  Useful for large-scale runs where per-rank amax can diverge.
    reduce_amax: bool = False

    # Use mori SDMA for the amax all-reduce.  When False, falls back to
    # ``torch.distributed.all_reduce``.
    use_sdma: bool = False

    # Whether to quantize activations (input to linear layers) in addition to
    # weights.  When False, only weights are quantized — activations stay in
    # the original precision (BF16/FP16).
    quantize_activation: bool = True

    # When False, compute the weight gradient (dW = grad^T @ X) in higher
    # precision (BF16) instead of FP8.  dgrad (dX = grad @ W) stays in FP8.
    # When False, compute wgrad in BF16 instead of FP8.
    fp8_wgrad: bool = True

    # Cache the FP8 quantization of *frozen* weights (requires_grad=False, e.g.
    # the LoRA base) on first forward and reuse it, skipping the per-forward (and
    # gradient-checkpointing recompute) re-quantization.  Trades resident memory
    # (full FP8 weight stays cached) for speed — keep OFF for very large models
    # under FSDP where the cached weights would not fit.
    cache_frozen_weight: bool = False

    # Use the B-preshuffle blockscale GEMM (pre-shuffled weight layout, ~2.5x
    # faster on gfx942, bit-identical). Requires cache_frozen_weight (the shuffle
    # is cached once per frozen weight). Only affects blockwise/blockwise2d.
    bpreshuffle_gemm: bool = False

    # Quantize the output/vocab-projection layer (``lm_head`` / ``output_layer``).
    # Default OFF: with a large vocab (e.g. Qwen3 151936) and long sequences the
    # forward output has M×N > 2^31 elements, which overflows the int32 pointer
    # arithmetic in AITER's Triton FP8 GEMM kernels (gemm_a8w8 / blockscale) and
    # faults with "Write access to a read-only page". BF16 GEMM (hipBLASLt, int64
    # addressing) is unaffected, so the output layer stays BF16 by default.
    quantize_output_layer: bool = False

    # Keep the first and last N transformer layers in BF16 (unpatched) even
    # during FP8 training.
    first_last_layers_bf16: bool = False
    num_layers_at_start_in_bf16: int = 1
    num_layers_at_end_in_bf16: int = 1
    # Total number of transformer layers in the model (global, across all PP
    # ranks).  Required when ``first_last_layers_bf16=True`` so that the
    # patching logic can identify which layers are "last".  Set to 0 to
    # auto-detect from the model structure.
    num_layers: int = 0

    # Gradient quantization type for the backward pass.  When set, gradients
    # are rounded to the specified low-precision format (quant → dequant)
    # before being returned from autograd backward.  This reduces gradient
    # communication bandwidth in distributed training.
    # Valid values: None (disabled), "fp8", "mxfp8", "mxfp4", "fp4" (placeholder).
    quantize_grad: str | None = None

    # FP8 dot-product attention: run the attention kernel in FP8.
    fp8_dpa: bool = False

    # FP8 multi-head attention: keep the entire MHA block (QKV projection +
    # dot-product attention + output projection) in FP8, eliminating BF16
    # casts at module boundaries.
    fp8_mha: bool = False

    def __post_init__(self):
        # MXFP4's block size is not a tuning knob: the E8M0 scale layout, the
        # AITER kernels and the swizzle tilings are all defined at 32. A config
        # carrying the FP8 default of 128 computed scales over 128 elements
        # while every consumer read them as 32-element blocks, which is not a
        # precision trade-off but wrong arithmetic, and nothing checked.
        if self.format == QuantFormat.MXFP4 and self.block_size != MXFP4_BLOCK_SIZE:
            raise ValueError(
                f"MXFP4 is defined at {MXFP4_BLOCK_SIZE}-element blocks; got "
                f"block_size={self.block_size}. The scale layout and every AITER "
                "MXFP4 kernel assume 32."
            )

    @classmethod
    def from_str(
        cls, format: str = "fp8_e4m3", scaling: str = "delayed", **kwargs
    ) -> "QuantConfig":
        """Construct a QuantConfig from plain strings.

        Args:
            format: One of ``"fp8_e4m3"``, ``"fp8_e5m2"``, ``"hybrid"``,
                ``"mxfp8"``, ``"mxfp4"``, ``"fp4"``.
            scaling: One of ``"dynamic"``, ``"delayed"``, ``"blockwise"``,
                ``"blockwise2d"``, ``"per_token"``, ``"none"``.
            **kwargs: Forwarded to :class:`QuantConfig`.  String values for
                enum fields (``amax_algo``) are auto-converted.
        """
        # Aliases for user convenience
        _scaling_aliases = {
            "current": "dynamic",
            "current_per_tensor": "dynamic",
            "delayed_per_tensor": "delayed",
            "no_quant": "none",
        }
        scaling = _scaling_aliases.get(scaling, scaling)
        fmt = QuantFormat(format)
        if fmt == QuantFormat.MXFP4 and scaling == "delayed":
            scaling = "blockwise"
        if "amax_algo" in kwargs and isinstance(kwargs["amax_algo"], str):
            kwargs["amax_algo"] = AmaxAlgo(kwargs["amax_algo"])
        return cls(
            format=fmt,
            scaling=ScalingType(scaling),
            **kwargs,
        )

    @property
    def torch_dtype(self) -> torch.dtype | None:
        """Return the forward-pass PyTorch FP8 dtype, or None if unavailable."""
        return _build_format_to_dtype().get(self.format)

    @property
    def torch_dtype_bwd(self) -> torch.dtype | None:
        """Return the backward-pass FP8 dtype (differs from fwd for HYBRID)."""
        return _format_to_dtype_bwd().get(self.format)

    @property
    def fp8_max(self) -> float:
        """Max representable FP8 value for the forward pass."""
        return get_fp8_max(self.format)

    @property
    def fp8_max_bwd(self) -> float:
        """Max representable FP8 value for the backward pass."""
        return get_fp8_max_bwd(self.format)

    @property
    def is_quantized(self) -> bool:
        """Return True if this config actually applies quantization."""
        return self.scaling != ScalingType.NONE

    @property
    def recipe(self) -> str:
        """Legacy recipe string expected by ScalingManager."""
        if self.scaling == ScalingType.NONE:
            return "none"
        if self.format == QuantFormat.MXFP8:
            return "mxfp8"
        if self.format == QuantFormat.MXFP4:
            return "mxfp4"
        return self.scaling.value
