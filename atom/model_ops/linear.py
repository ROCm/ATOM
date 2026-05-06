# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import os
from functools import partial as functools_partial
from typing import Callable, Optional

import torch
from aiter import (
    QuantType,
    dtypes,
    gemm_a4w4,
    gemm_a8w8,
    gemm_a8w8_blockscale_bpreshuffle,
    gemm_a8w8_bpreshuffle,
    get_hip_quant,
)

# import torch.distributed as dist
from aiter.dist.parallel_state import get_tp_group
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.tuned_gemm import tgemm
from aiter.utility import fp4_utils
from atom.config import QuantizationConfig, get_current_atom_config
from atom.quant_spec import LayerQuantConfig
from atom.model_ops.utils import (
    atom_parameter,
    normalize_e4m3fn_to_e4m3fnuz,
    requantize_with_max_scale,
    shuffle_weights,
)
from atom.utils import envs
from atom.utils.decorators import mark_trace
from torch import nn

logger = logging.getLogger("atom")


def use_triton_gemm() -> bool:
    return envs.ATOM_USE_TRITON_GEMM


# Blockscale FP8 GEMM SplitK + zero-init fusion mode. See
# atom/utils/envs.py for the semantics of each mode. Read once at module
# import; flipping at runtime is intentionally not supported (mode is set
# pre-launch via env).
_BLOCKSCALE_SPLITK_MODES = ("none", "splitk", "splitk_fused")


def _resolve_blockscale_splitk_mode() -> str:
    mode = envs.ATOM_BLOCKSCALE_SPLITK_MODE
    if mode not in _BLOCKSCALE_SPLITK_MODES:
        logger.warning(
            "Unknown ATOM_BLOCKSCALE_SPLITK_MODE=%s; falling back to 'none'. "
            "Valid modes: %s",
            mode,
            _BLOCKSCALE_SPLITK_MODES,
        )
        return "none"
    return mode


BLOCKSCALE_SPLITK_MODE = _resolve_blockscale_splitk_mode()


# Optional per-mode tuned-CSV overrides. When set, these supersede the
# default AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE for the
# corresponding mode. Use these to ship a no-SplitK tune (config1) and a
# with-SplitK tune (config2/3) side-by-side and flip via the mode flag.
_BLOCKSCALE_NOSPLITK_TUNED_CSV = os.getenv(
    "ATOM_BLOCKSCALE_BPRESHUFFLE_TUNED_NOSPLITK_CSV", ""
).strip() or None
_BLOCKSCALE_SPLITK_TUNED_CSV = os.getenv(
    "ATOM_BLOCKSCALE_BPRESHUFFLE_TUNED_SPLITK_CSV", ""
).strip() or None


def _blockscale_tuned_csv_for_mode() -> Optional[str]:
    if BLOCKSCALE_SPLITK_MODE == "none":
        return _BLOCKSCALE_NOSPLITK_TUNED_CSV
    return _BLOCKSCALE_SPLITK_TUNED_CSV


_BLOCKSCALE_TUNED_CSV = _blockscale_tuned_csv_for_mode()


# CSV lookup so we only pay the producer-side zero-init when the tuned
# config actually selected splitK > 0 for the running (M, N, K). For
# shapes the tuner picked splitK = 0 the kernel performs no atomic-add
# pass and there is nothing to skip, so pre-zeroing in the producer is
# wasted memory traffic.
try:
    from aiter.ops.gemm_op_a8w8 import get_CKGEMM_config as _get_blockscale_config
except ImportError:
    _get_blockscale_config = None


def _blockscale_splitk_for_shape(m: int, n: int, k: int) -> Optional[int]:
    """Return the splitK column from the active tuned CSV, or None.

    Returns None when the lookup helper is unavailable, when no CSV is
    configured for the active mode, or when the tuner has no entry that
    matches (m, n, k) (including its padded-M fallbacks).

    NOT currently called from ``LinearBase.forward`` because every viable
    way to invoke it from inside the traced backbone (direct call, behind
    ``@torch.compiler.disable``, behind a Python ``if``) trips a Dynamo
    graph break, and ATOM's ``VllmBackend`` asserts a single compile call
    per backend instance. Kept as a utility for offline / non-traced
    callers (e.g., ``op_tests/bench_zero_init_splitk_demo.py``) and as a
    starting point for the shape-aware staging follow-up tracked by
    ``TODO(zero-init-splitk)`` in ``LinearBase.forward``.
    """
    if _get_blockscale_config is None or _BLOCKSCALE_TUNED_CSV is None:
        return None
    try:
        cfg = _get_blockscale_config(m, n, k, tuned_file=_BLOCKSCALE_TUNED_CSV)
    except Exception:  # noqa: BLE001
        return None
    if cfg is None:
        return None
    try:
        return int(cfg.get("splitK", 0))
    except (TypeError, ValueError):
        return None


if use_triton_gemm():
    try:
        # from aiter.ops.triton.gemm_a8w8_blockscale import gemm_a8w8_blockscale_preshuffle as gemm_a8w8_blockscale_bpreshuffle_triton
        from aiter.ops.triton.gemm_afp4wfp4 import (
            gemm_afp4wfp4_preshuffle,
        )  # noqa: E402
    except ImportError as e:
        logger.warning(f"Triton FP4 GEMM not available: {e}")
        gemm_afp4wfp4_preshuffle = None

    # For Triton FP8 Blockscale GEMM is mostly slower then AITER GEMM, we turn off Triton FP8 GEMM
    try:
        from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
            gemm_a8w8_blockscale_preshuffle as gemm_a8w8_blockscale_bpreshuffle_triton,
        )  # noqa: E402
    except ImportError as e:
        logger.warning(f"Triton w8a8 GEMM not available: {e}")
        gemm_a8w8_blockscale_bpreshuffle_triton = None
else:
    gemm_afp4wfp4_preshuffle = None
    gemm_a8w8_blockscale_bpreshuffle_triton = None
from atom.model_ops.utils import MXFP4_QUANT_BLOCK_SIZE  # noqa


def divide(numerator, denominator):
    assert (
        numerator % denominator == 0
    ), f"numerator {numerator} denominator {denominator}"
    return numerator // denominator


def gemm_a4w4_quant_fake(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    otype: torch.dtype,
    weight_scale: torch.Tensor,
    params_dtype: torch.dtype,
    input_scale: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], weight.shape[0]), dtype=otype, device=x.device)


# It's important to use mutates_args=[] to avoid functionized_v2 op generation
@torch_compile_guard(gen_fake=gemm_a4w4_quant_fake, mutates_args=[])
def gemm_a4w4_quant(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    otype: torch.dtype,
    weight_scale: torch.Tensor,
    params_dtype: torch.dtype,
    input_scale: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    if gemm_afp4wfp4_preshuffle is None:
        if x_scale is None:
            quant_func = get_hip_quant(QuantType.per_1x32)
            x, x_scale = quant_func(
                x,
                quant_dtype=params_dtype,
                scale=input_scale,
                shuffle=True,
            )
        else:
            x_scale = x_scale.view(torch.float8_e8m0fnu)
            x = x.view(torch.float4_e2m1fn_x2)

        m = x.view(-1, x.size(-1)).shape[0]
        y = torch.empty(
            (
                (m + MXFP4_QUANT_BLOCK_SIZE - 1)
                // MXFP4_QUANT_BLOCK_SIZE
                * MXFP4_QUANT_BLOCK_SIZE,
                output_size,
            ),
            dtype=otype,
            device=x.device,
        )
        y = gemm_a4w4(
            x,
            weight,
            x_scale,
            weight_scale,
            y,
        )
    else:
        m, k = x.view(-1, x.size(-1)).shape

        y = torch.empty(
            (
                (m + MXFP4_QUANT_BLOCK_SIZE - 1)
                // MXFP4_QUANT_BLOCK_SIZE
                * MXFP4_QUANT_BLOCK_SIZE,
                output_size,
            ),
            dtype=otype,
            device=x.device,
        )
        if x_scale is None:
            quant_func = get_hip_quant(QuantType.per_1x32)
            x, x_scale = quant_func(
                x,
                quant_dtype=params_dtype,
                shuffle=(m >= MXFP4_QUANT_BLOCK_SIZE),
            )
        else:
            x_scale = x_scale.view(torch.float8_e8m0fnu)
            x = x.view(torch.float4_e2m1fn_x2)

        if m >= MXFP4_QUANT_BLOCK_SIZE:
            x_scale = x_scale.view(torch.uint8).view(
                x_scale.shape[0] // MXFP4_QUANT_BLOCK_SIZE, -1
            )
        else:
            x_scale = x_scale[:m, ...].view(torch.uint8)

        y = gemm_afp4wfp4_preshuffle(
            x.view(torch.uint8),
            weight.view(torch.uint8).view(weight.shape[0] // 16, -1),
            x_scale,
            weight_scale.view(torch.uint8).view(
                weight_scale.shape[0] // MXFP4_QUANT_BLOCK_SIZE, -1
            ),
            y=y,
        )

    return y[:m, ...]


def gemm_a8w8_blockscale_preshuffle_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    prefix: str = "",
    out: Optional[torch.Tensor] = None,
    y_is_zeroed: bool = False,
) -> torch.Tensor:
    if out is not None:
        return out
    return torch.empty((*x.shape[:-1], weight.shape[0]), dtype=dtype, device=x.device)


@mark_trace(torch_compile=False)
@torch_compile_guard(gen_fake=gemm_a8w8_blockscale_preshuffle_fake, mutates_args=[])
def gemm_a8w8_blockscale_preshuffle_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    prefix: str = "",
    out: Optional[torch.Tensor] = None,
    y_is_zeroed: bool = False,
) -> torch.Tensor:
    """ATOM blockscale FP8 a8w8 (per_1x128) preshuffle GEMM dispatch.

    Args:
        x, weight, x_scale, w_scale: standard FP8 a8w8 blockscale operands.
        dtype: output dtype (bf16/fp16).
        prefix: layer prefix used for tracing.
        out: optional pre-allocated output tensor of shape (M, N), dtype
            ``dtype``. When provided the GEMM writes into this tensor instead
            of allocating its own. Required for the SplitK-zero-init fused
            mode where the producer kernel has already zeroed the buffer.
        y_is_zeroed: when True, the caller has pre-zeroed ``out`` and the
            GEMM may skip its own ``Y.zero_()`` before the SplitK
            ``atomic_add`` accumulation. Only meaningful with splitK > 0.
    """
    # Triton fallback path: no SplitK / no zero-init contract today, so
    # fall through to the AITER bpreshuffle path whenever the caller asks
    # for any non-default behavior.
    if (
        gemm_a8w8_blockscale_bpreshuffle_triton is not None
        and out is None
        and not y_is_zeroed
        and BLOCKSCALE_SPLITK_MODE == "none"
    ):
        weight_shuffled = weight.reshape(weight.shape[0] // 16, weight.shape[1] * 16)
        y = gemm_a8w8_blockscale_bpreshuffle_triton(
            x, weight_shuffled, x_scale, w_scale, dtype
        )
    else:
        y = gemm_a8w8_blockscale_bpreshuffle(
            x,
            weight,
            x_scale,
            w_scale,
            dtype,
            out=out,
            y_is_zeroed=y_is_zeroed,
            tuned_file=_BLOCKSCALE_TUNED_CSV,
        )
    return y


class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int | list[int],
        tp_dim: int | None = None,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = False,
        source_quant_dtype: torch.dtype | None = None,
        prefix: str = "",
    ):
        self.prefix = prefix
        layer_quant_config = (
            quant_config.get_layer_quant_config(prefix)
            if quant_config is not None
            else LayerQuantConfig()
        )
        quant_type = layer_quant_config.quant_type
        params_dtype = layer_quant_config.quant_dtype
        self.source_quant_dtype = source_quant_dtype
        self.layer_quant_config = layer_quant_config
        super().__init__()
        self.reduce_results = reduce_results
        self.input_size = input_size
        self.output_size = (
            output_size if isinstance(output_size, int) else sum(output_size)
        )
        self.tp_dim = tp_dim
        self.tp_rank = get_tp_group().rank_in_group
        self.tp_size = get_tp_group().world_size
        self.output_partition_sizes = (
            output_size if isinstance(output_size, list) else [output_size]
        )
        if tp_dim == 1:
            self.input_size = divide(input_size, self.tp_size)
        elif tp_dim == 0:
            self.output_size = divide(self.output_size, self.tp_size)
            self.output_partition_sizes = [
                divide(s, self.tp_size) for s in self.output_partition_sizes
            ]

        if self.source_quant_dtype is not None:
            weight_size = (self.output_size, self.input_size)
            self.weight = atom_parameter(
                torch.empty(weight_size, dtype=self.source_quant_dtype)
            )
        else:
            weight_size = (
                (self.output_size, self.input_size)
                if params_dtype not in [dtypes.fp4x2, dtypes.i4x2]
                else (self.output_size, self.input_size // 2)
            )
            self.weight = atom_parameter(torch.empty(weight_size, dtype=params_dtype))
        if bias:
            output_type = get_current_atom_config().torch_dtype
            self.bias = atom_parameter(torch.empty(self.output_size, dtype=output_type))
            self.bias.weight_loader_process = self.weight_loader_process
        else:
            self.register_parameter("bias", None)
        self.quant_type = quant_type
        self.params_dtype = params_dtype

        if quant_type != QuantType.No and self.source_quant_dtype is None:
            if quant_type == QuantType.per_Tensor:
                self.weight_scale = atom_parameter(
                    torch.empty(len(self.output_partition_sizes), 1, dtype=dtypes.fp32)
                )
                if not layer_quant_config.is_dynamic:
                    self.input_scale = atom_parameter(
                        torch.empty(
                            len(self.output_partition_sizes), 1, dtype=dtypes.fp32
                        )
                    )
                    self.input_scale.weight_loader_process = self.weight_loader_process
                    self.input_scale.weight_loader = self.weight_loader
            elif quant_type == QuantType.per_Token:
                self.weight_scale = atom_parameter(
                    torch.empty(self.output_size, 1, dtype=dtypes.fp32)
                )
            elif quant_type == QuantType.per_1x128:
                self.weight_scale = atom_parameter(
                    torch.empty(
                        (self.output_size + 127) // 128,
                        (self.input_size + 127) // 128,
                        dtype=dtypes.fp32,
                    )
                )
            elif quant_type == QuantType.per_1x32:
                self.weight_scale = atom_parameter(
                    torch.empty(
                        self.output_size,
                        (self.input_size + 31) // 32,
                        dtype=dtypes.fp8_e8m0,
                    )
                )
            self.weight.weight_loader_process = self.weight_loader_process
            self.weight_scale.weight_loader_process = self.weight_loader_process
        else:
            self.weight.weight_loader_process = self.weight_loader_process
            self.register_parameter("weight_scale", None)
        self.weight.weight_loader = self.weight_loader
        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader
        if self.weight_scale is not None:
            self.weight_scale.weight_loader = self.weight_loader
        self.need_normalize_e4m3fn_to_e4m3fnuz = params_dtype == torch.float8_e4m3fnuz
        self.quant_func = get_hip_quant(self.quant_type)

    @staticmethod
    def weight_loader_process(
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        post_process_func: Callable = lambda a: a,
    ):
        if (
            param.data.dtype != loaded_weight.dtype
            and param.data.element_size() == loaded_weight.element_size()
        ):
            param.data = param.data.view(loaded_weight.dtype)
        loaded_weight = post_process_func(loaded_weight)
        if (
            loaded_weight.shape != param.data.shape
            and loaded_weight.numel() == param.data.numel()
        ):
            loaded_weight = loaded_weight.reshape(param.data.shape)
        if param.data.dtype != dtypes.fp4x2:
            param.data.copy_(loaded_weight)
        else:
            param.data.view(torch.uint8).copy_(loaded_weight.view(torch.uint8))

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        param.weight_loader_process(param_data, loaded_weight)

    def process_weights_after_loading(self):
        if (
            self.quant_type == QuantType.per_Tensor
            and len(self.output_partition_sizes) > 1
        ):
            weight_scale, weight = requantize_with_max_scale(
                weight=self.weight.data,
                weight_scale=self.weight_scale.data,
                logical_widths=self.output_partition_sizes,
                normalize_e4m3fn_to_e4m3fnuz=self.need_normalize_e4m3fn_to_e4m3fnuz,
            )
            self.weight.data = weight
            self.weight_scale.data = weight_scale.view(-1)
            if hasattr(self, "input_scale"):
                self.input_scale.data = (
                    self.input_scale.data.max() * 2.0
                    if self.need_normalize_e4m3fn_to_e4m3fnuz
                    else self.input_scale.data.max()
                )
        elif self.need_normalize_e4m3fn_to_e4m3fnuz:
            self.weight.data, self.weight_scale.data, _ = normalize_e4m3fn_to_e4m3fnuz(
                self.weight.data, self.weight_scale.data
            )
        if (
            self.source_quant_dtype == torch.bfloat16
            and self.quant_type == QuantType.per_1x32
            and self.params_dtype == torch.float4_e2m1fn_x2
        ):
            w_q, w_s = self.quant_func(
                self.weight.data,
                quant_dtype=self.params_dtype,
                shuffle=False,
            )
            self.weight.data = w_q
            self.weight_scale = atom_parameter(w_s)
            # Only quantized 2D GEMM weights use aiter's preshuffle layout.
            # Qwen3-Next/Qwen3.5 GDN conv1d expands its weight to 3D, so FP8/blocked
            # quantized models must keep that tensor unshuffled here.
            if self.weight.dim() == 2:
                shuffle_weights(self.weight)
            # self.weight_scale.data = fp4_utils.e8m0_shuffle(self.weight_scale.data)
        else:
            if (
                self.quant_type == QuantType.per_Token
                and self.params_dtype == dtypes.fp8
            ) or (self.quant_type in [QuantType.per_1x32, QuantType.per_1x128]):
                if self.weight.dim() == 2:
                    shuffle_weights(self.weight)
                # self.weight_scale.data = fp4_utils.e8m0_shuffle(self.weight_scale.data)
        # shuffle weight scale once so no reshuffling for every gemm
        if self.quant_type == QuantType.per_1x32:
            self.weight_scale.data = fp4_utils.e8m0_shuffle(self.weight_scale.data)

    @mark_trace
    def forward(
        self, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None, otype=dtypes.bf16
    ) -> torch.Tensor:
        if self.quant_type.value == QuantType.No.value:
            y = tgemm.mm(
                x,
                self.weight,
                self.bias,
                otype=otype,
            )
        else:
            # For per_1x128 + SplitK + fused-zero-init mode: pre-allocate the
            # GEMM output Y here so we can hand it to the producer quant
            # kernel as a zero-fill target. For "splitk" (non-fused) we still
            # pre-allocate so the GEMM-side Y.zero_() writes into a buffer
            # we own (otherwise gemm_a8w8_blockscale_bpreshuffle allocates
            # its own and we lose the ability to compare configs trivially).
            #
            # KNOWN LIMITATION (intentional, see TODO below):
            # In ``splitk_fused`` mode we currently stage the producer-side
            # zero-init unconditionally, regardless of whether the tuned CSV
            # picked splitK>0 for this (M,N,K). The ideal behavior is to
            # only zero when splitK>0, but a CSV lookup at this point would
            # have to call into pandas-backed ``get_CKGEMM_config``. Wrapping
            # that in ``@torch.compiler.disable`` (or any other shape-aware
            # branching here) introduces a Dynamo graph break, which is
            # fatal under ATOM's one-shot ``VllmBackend`` (asserts a single
            # compile call per backend instance). Until we can hoist the
            # decision out of the traced region (e.g., precomputing per
            # LinearBase at module-init for the warmup M), we accept the
            # extra zero-write traffic for splitK=0 layers in exchange for
            # a Dynamo-stable call signature.
            #
            # TODO(zero-init-splitk): make staging shape-aware without
            # tripping the one-shot VllmBackend. Options to evaluate:
            #   1. Resolve the splitK bool at LinearBase.__init__ for each
            #      cudagraph_capture_size and store on `self`.
            #   2. Pre-compute splitK in a preprocessing pass over the
            #      module tree before warmup.
            #   3. Wrap the lookup in a torch custom op that returns a
            #      0-d bool tensor so dynamo treats it as a graph node
            #      instead of a Python branch.
            preallocated_y: Optional[torch.Tensor] = None
            zero_init_target: Optional[torch.Tensor] = None
            # producer_zeroed flips True only when we actually pass
            # gemm_out_zero_init=Y to a producer call below. y_is_zeroed
            # must be driven off this -- *not* off our intent to fuse --
            # because Qwen3-Next-style fused upstream RMSNorm-quant ops
            # feed LinearBase with x_scale already set, in which case we
            # skip the in-LinearBase quant call and never hand Y to any
            # producer. Claiming y_is_zeroed=True without an actual
            # zero-fill would make the SplitK atomic_add accumulate onto
            # uninitialized memory.
            producer_zeroed = False
            if (
                self.quant_type.value == QuantType.per_1x128.value
                and BLOCKSCALE_SPLITK_MODE != "none"
            ):
                m = x.shape[0] if x.dim() == 2 else int(x.numel() // x.shape[-1])
                preallocated_y = torch.empty(
                    m, self.output_size, dtype=otype, device=x.device
                )
                if BLOCKSCALE_SPLITK_MODE == "splitk_fused":
                    # Stage zero-init unconditionally; see KNOWN LIMITATION
                    # block above for why this is not currently shape-aware.
                    zero_init_target = preallocated_y

            if x_scale is None:
                quant_func = self.quant_func
                if self.quant_type.value == QuantType.per_1x128.value:
                    quant_func = functools_partial(
                        self.quant_func, transpose_scale=True
                    )
                if self.quant_type.value != QuantType.per_1x32.value:
                    if self.quant_type.value == QuantType.per_1x128.value:
                        # Always pass gemm_out_zero_init= as a kwarg here so
                        # Dynamo sees a single, stable call signature for
                        # per_group_quant_hip across all per_1x128 layers in
                        # the traced backbone (mixing "kwarg present" vs
                        # "kwarg absent" between sibling LinearBase calls
                        # produces two specialized graphs and trips ATOM's
                        # one-shot VllmBackend assertion). The custom op
                        # schema is `Tensor(a7!)? gemm_out_zero_init=None`,
                        # so passing None when we are not staging a
                        # producer-fused zero-init is a no-op.
                        x, x_scale = quant_func(
                            x,
                            quant_dtype=self.params_dtype,
                            scale=getattr(self, "input_scale", None),
                            gemm_out_zero_init=zero_init_target,
                        )
                        producer_zeroed = zero_init_target is not None
                    else:
                        x, x_scale = quant_func(
                            x,
                            quant_dtype=self.params_dtype,
                            scale=getattr(self, "input_scale", None),
                        )
            if self.quant_type.value == QuantType.per_Tensor.value:
                y = tgemm.mm(
                    x,
                    self.weight,
                    self.bias,
                    otype=otype,
                    scale_a=x_scale,
                    scale_b=self.weight_scale,
                )
            elif self.quant_type.value == QuantType.per_Token.value:
                if self.params_dtype == dtypes.i8:
                    y = gemm_a8w8(
                        x,
                        self.weight,
                        x_scale,
                        self.weight_scale,
                        self.bias,
                        dtype=otype,
                    )
                else:
                    y = gemm_a8w8_bpreshuffle(
                        x,
                        self.weight,
                        x_scale,
                        self.weight_scale,
                        dtype=otype,
                    )
                    if self.bias is not None:
                        y += self.bias
            elif self.quant_type.value == QuantType.per_1x128.value:
                y = gemm_a8w8_blockscale_preshuffle_impl(
                    x,
                    self.weight,
                    x_scale,
                    self.weight_scale,
                    dtype=otype,
                    prefix=self.prefix,
                    out=preallocated_y,
                    # y_is_zeroed must agree with what a producer
                    # actually wrote: True only when we successfully
                    # passed gemm_out_zero_init= to a producer call
                    # this iteration. False here when splitK=0 is
                    # harmless (kernel skips its own zero because
                    # k_batch=1); False when splitK>0 falls back to
                    # the kernel-side Y.zero_().
                    y_is_zeroed=producer_zeroed,
                )
                if self.bias is not None:
                    y += self.bias
            elif self.quant_type.value == QuantType.per_1x32.value:
                y = gemm_a4w4_quant(
                    x,
                    x_scale,
                    self.weight,
                    otype,
                    self.weight_scale.data,
                    self.params_dtype,
                    getattr(self, "input_scale", None),
                    self.output_size,
                )
                if self.bias is not None:
                    y += self.bias
        if self.tp_dim == 1 and self.tp_size > 1 and self.reduce_results:
            y = get_tp_group().all_reduce(y, ca_fp8_quant=False)
        return y


class ReplicatedLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        **kwargs,
    ):
        super().__init__(
            input_size,
            output_size,
            tp_dim=None,
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        param.weight_loader_process(param_data, loaded_weight)


class ColumnParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        **kwargs,
    ):
        self.tp_dim = 0
        super().__init__(
            input_size,
            output_size,
            self.tp_dim,
            bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param.weight_loader_process(param_data, loaded_weight)


class MergedColumnParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        **kwargs,
    ):
        self.output_sizes = output_sizes
        super().__init__(
            input_size,
            output_sizes,
            tp_dim=0,
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | tuple[int, ...] | None = None,
    ):
        # Support loading multiple consecutive shards in a single tensor.
        # This mirrors vLLM's behavior for packed modules like QKV.
        if isinstance(loaded_shard_id, tuple):
            if len(loaded_shard_id) == 0:
                raise ValueError("loaded_shard_id tuple cannot be empty")
            if any(idx < 0 or idx >= len(self.output_sizes) for idx in loaded_shard_id):
                raise ValueError(
                    f"Invalid shard id in {loaded_shard_id}; "
                    f"valid range is [0, {len(self.output_sizes) - 1}]"
                )
            if len(loaded_shard_id) > 1 and any(
                b - a != 1 for a, b in zip(loaded_shard_id[:-1], loaded_shard_id[1:])
            ):
                raise ValueError(
                    "Shard id with multiple indices should be consecutive. "
                    f"Got shard id {loaded_shard_id}."
                )

            # Split loaded_weight by the requested shard sizes (pre-TP),
            # then load each shard individually.
            shard_sizes = [self.output_sizes[i] for i in loaded_shard_id]
            current_offset = 0
            for shard_id, shard_size in zip(loaded_shard_id, shard_sizes):
                if param is getattr(self, "weight_scale", None) or param is getattr(
                    self, "input_scale", None
                ):
                    shard_size //= 128
                shard = loaded_weight.narrow(self.tp_dim, current_offset, shard_size)
                self.weight_loader(param, shard, shard_id)
                current_offset += shard_size
            return

        if loaded_shard_id is None:
            # Loaded weight is already fused on disk
            # Split it and load each shard individually.
            param_data = param.data
            # Check if this is weight or weight_scale
            is_scale_param = param is getattr(
                self, "weight_scale", None
            ) or param is getattr(self, "input_scale", None)

            # For fused weight, need to match param shape
            if param_data.shape == loaded_weight.shape:
                # Shapes match - direct copy
                param.weight_loader_process(param_data, loaded_weight)
                return

            # Otherwise, split the fused weight and load each output shard
            current_offset = 0
            for shard_id, output_size in enumerate(self.output_sizes):
                shard_size = output_size
                if is_scale_param and self.quant_type == QuantType.per_1x128:
                    shard_size //= 128

                shard = loaded_weight.narrow(self.tp_dim, current_offset, shard_size)
                self.weight_loader(param, shard, shard_id)
                current_offset += shard_size
            return

        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        if param is getattr(self, "weight_scale", None) or param is getattr(
            self, "input_scale", None
        ):
            if self.quant_type == QuantType.per_1x128:
                shard_offset = (shard_offset + 127) // 128
                shard_size = (shard_size + 127) // 128
            elif self.quant_type == QuantType.per_Tensor:
                loaded_weight = loaded_weight.view(1, 1).repeat(self.tp_size, 1)
                shard_offset = loaded_shard_id
                shard_size = 1

        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param.weight_loader_process(param_data, loaded_weight)


class QKVZBAParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        head_k_dim: int,
        head_v_dim: int,
        num_k_heads: int,
        num_v_heads: int,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        **kwargs,
    ):
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        tp_size = get_tp_group().world_size
        self.num_k_heads = divide(self.num_k_heads, tp_size)
        self.num_v_heads = divide(self.num_v_heads, tp_size)
        output_sizes = [
            (2 * head_k_dim * self.num_k_heads + 2 * head_v_dim * self.num_v_heads)
            * tp_size,
            2 * self.num_v_heads * tp_size,
        ]
        super().__init__(
            input_size,
            output_sizes,
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str
    ):
        param_data = param.data
        assert loaded_shard_id in ["qkvz", "ba", "qkv", "z", "b", "a"]
        if loaded_shard_id == "qkvz":
            shard_size = (
                2 * self.num_k_heads * self.head_k_dim
                + 2 * self.num_v_heads * self.head_v_dim
            )
            shard_offset = 0
            shard_rank = self.tp_rank
        elif loaded_shard_id == "qkv":
            shard_size = (
                2 * self.num_k_heads * self.head_k_dim
                + self.num_v_heads * self.head_v_dim
            )
            shard_offset = 0
            shard_rank = self.tp_rank
        elif loaded_shard_id == "z":
            shard_size = self.num_v_heads * self.head_v_dim
            shard_offset = (
                2 * self.num_k_heads * self.head_k_dim
                + self.num_v_heads * self.head_v_dim
            )
            shard_rank = self.tp_rank
        elif loaded_shard_id == "ba":
            shard_size = 2 * self.num_v_heads
            shard_offset = (
                2 * self.num_k_heads * self.head_k_dim
                + 2 * self.num_v_heads * self.head_v_dim
            )
            shard_rank = self.tp_rank
        elif loaded_shard_id == "b":
            shard_size = self.num_v_heads
            shard_offset = (
                2 * self.num_k_heads * self.head_k_dim
                + 2 * self.num_v_heads * self.head_v_dim
            )
            shard_rank = self.tp_rank
        elif loaded_shard_id == "a":
            shard_size = self.num_v_heads
            shard_offset = (
                2 * self.num_k_heads * self.head_k_dim
                + 2 * self.num_v_heads * self.head_v_dim
                + self.num_v_heads
            )
            shard_rank = self.tp_rank

        if param is getattr(self, "weight_scale", None) or param is getattr(
            self, "input_scale", None
        ):
            if self.quant_type == QuantType.per_1x128:
                shard_offset = (shard_offset + 127) // 128
                shard_size = (shard_size + 127) // 128
            elif self.quant_type == QuantType.per_Tensor:
                loaded_weight = loaded_weight.view(1, 1).repeat(self.tp_size, 1)
                shard_offset = ["qkvz", "ba"].index(loaded_shard_id)
                shard_size = 1
        start_idx = shard_rank * shard_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param.weight_loader_process(param_data, loaded_weight)


class QKVGParallelLinear(ColumnParallelLinear):
    """QKV + output-Gate parallel linear.

    Rearranges interleaved Q+Gate weights from HF checkpoint into grouped
    layout [Gate, Q, K, V] during loading, so inference uses a single split().
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        source_quant_dtype: torch.dtype | None = None,
        prefix: str = "",
        **kwargs,
    ):
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        tp_size = get_tp_group().world_size
        self.num_heads = divide(self.total_num_heads, tp_size)
        if self.total_num_kv_heads >= tp_size:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        else:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)

        input_size = hidden_size
        output_sizes = [
            self.num_heads * self.head_size * tp_size,  # Gate
            self.num_heads * self.head_size * tp_size,  # Q
            self.num_kv_heads * self.head_size * tp_size,  # K
            self.num_kv_heads * self.head_size * tp_size,  # V
        ]

        super().__init__(
            input_size,
            output_sizes,
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    def _deinterleave(
        self, weight: torch.Tensor, head_stride: int | None = None
    ) -> torch.Tensor:
        """Rearrange Q+Gate from interleaved [q0,g0,q1,g1,...] to grouped [Q_all, Gate_all].

        Args:
            head_stride: number of elements per head along dim 0.
                         Defaults to self.head_size (weights); use head_size//128
                         for per-1x128 scales.
        """
        hs = head_stride if head_stride is not None else self.head_size
        return (
            weight.view(self.num_heads, 2, hs, -1)
            .transpose(0, 1)
            .reshape(self.num_heads * 2 * hs, -1)
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str
    ):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        q_size = self.num_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        # Layout: [Gate, Q, K, V]
        if loaded_shard_id == "q":
            # HF q_proj contains interleaved Q+Gate; deinterleave then
            # write Gate to offset 0 and Q to offset q_size
            shard_size = q_size * 2
            shard_offset = 0  # placeholder, handled below
            shard_rank = self.tp_rank
        elif loaded_shard_id == "k":
            shard_size = kv_size
            shard_offset = q_size * 2
            shard_rank = self.tp_rank // self.num_kv_head_replicas
        else:
            shard_size = kv_size
            shard_offset = q_size * 2 + kv_size
            shard_rank = self.tp_rank // self.num_kv_head_replicas

        is_scale = param is getattr(self, "weight_scale", None) or param is getattr(
            self, "input_scale", None
        )

        if loaded_shard_id == "q":
            # Q+Gate: deinterleave [q0,g0,...] -> [Q_all, Gate_all], then
            # write Gate to offset 0 and Q to offset q_size → layout [Gate, Q, ...]
            if is_scale and self.quant_type == QuantType.per_Tensor:
                loaded_weight = loaded_weight.view(1, 1).repeat(self.tp_size, 1)
                # Gate scale -> slot 0, Q scale -> slot 1
                q_scale = loaded_weight.narrow(self.tp_dim, shard_rank, 1)
                param.weight_loader_process(
                    param_data.narrow(self.tp_dim, 0, 1), q_scale.clone()
                )
                param.weight_loader_process(
                    param_data.narrow(self.tp_dim, 1, 1), q_scale
                )
                return

            scale_factor = (
                128 if (is_scale and self.quant_type == QuantType.per_1x128) else 1
            )
            half = q_size // scale_factor
            start_idx = shard_rank * shard_size // scale_factor
            loaded_weight = loaded_weight.narrow(
                self.tp_dim, start_idx, shard_size // scale_factor
            )
            stride = self.head_size // scale_factor
            loaded_weight = self._deinterleave(
                loaded_weight, head_stride=stride if scale_factor > 1 else None
            )
            q_part = loaded_weight.narrow(self.tp_dim, 0, half)
            gate_part = loaded_weight.narrow(self.tp_dim, half, half)
            q_offset = q_size // scale_factor
            # Gate at offset 0, Q at offset q_size
            param.weight_loader_process(
                param_data.narrow(self.tp_dim, 0, half), gate_part
            )
            param.weight_loader_process(
                param_data.narrow(self.tp_dim, q_offset, half), q_part
            )
        else:
            # K or V: straightforward load
            if is_scale:
                if self.quant_type == QuantType.per_1x128:
                    shard_offset //= 128
                    shard_size //= 128
                elif self.quant_type == QuantType.per_Tensor:
                    loaded_weight = loaded_weight.view(1, 1).repeat(self.tp_size, 1)
                    # [Gate, Q, K, V] -> K=2, V=3
                    shard_offset = {"k": 2, "v": 3}[loaded_shard_id]
                    shard_size = 1

            start_idx = shard_rank * shard_size
            param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
            loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
            param.weight_loader_process(param_data, loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        v_head_size: int | None = None,
        **kwargs,
    ):
        self.head_size = head_size
        self.v_head_size = v_head_size if v_head_size is not None else head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        tp_size = get_tp_group().world_size
        self.num_heads = divide(self.total_num_heads, tp_size)
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)

        input_size = hidden_size
        output_sizes = [
            self.num_heads * self.head_size * tp_size,
            self.num_kv_heads * self.head_size * tp_size,
            self.num_kv_heads * self.v_head_size * tp_size,
        ]

        super().__init__(
            input_size,
            output_sizes,
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str
    ):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
            shard_rank = self.tp_rank
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
            shard_rank = self.tp_rank // self.num_kv_head_replicas
        else:
            shard_size = self.num_kv_heads * self.v_head_size
            shard_offset = (
                self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            )
            shard_rank = self.tp_rank // self.num_kv_head_replicas
        if param is getattr(self, "weight_scale", None) or param is getattr(
            self, "input_scale", None
        ):
            if self.quant_type == QuantType.per_1x128:
                shard_offset = (shard_offset + 127) // 128
                shard_size = (shard_size + 127) // 128
            elif self.quant_type == QuantType.per_Tensor:
                loaded_weight = loaded_weight.view(1, 1).repeat(self.tp_size, 1)
                shard_offset = ["q", "k", "v"].index(loaded_shard_id)
                shard_size = 1

        start_idx = shard_rank * shard_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param.weight_loader_process(param_data, loaded_weight)


class RowParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        **kwargs,
    ):
        self.tp_rank = get_tp_group().rank_in_group
        super().__init__(
            input_size,
            output_size,
            tp_dim=1,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param is not getattr(self, "bias", None):
            if len(loaded_weight.shape) == 0:
                loaded_weight = loaded_weight.view(1, 1)
            if loaded_weight.ndim <= self.tp_dim:
                # dims < tp_dim (1D per-channel scale with
                # tp_dim=1)
                param.weight_loader_process(param_data, loaded_weight)
                return
            shard_size = param_data.size(self.tp_dim)
            if loaded_weight.size(self.tp_dim) == 1 and self.tp_size > 1:
                loaded_weight = loaded_weight.repeat(1, self.tp_size)
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        else:
            if self.tp_size > 0 and self.tp_rank != 0:
                loaded_weight.zero_()
        param.weight_loader_process(param_data, loaded_weight)


class MergedReplicatedLinear(ReplicatedLinear):
    def __init__(
        self,
        input_size: int,
        output_size: list[int],
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        **kwargs,
    ):
        self.output_sizes = output_size
        super().__init__(
            input_size,
            sum(output_size),  # ？
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[int] = None,
    ):  # ？
        param_data = param.data
        assert loaded_shard_id is not None
        assert loaded_shard_id < len(self.output_sizes)
        if param is getattr(self, "weight_scale", None) or param is getattr(
            self, "input_scale", None
        ):
            if self.quant_type == QuantType.per_1x128:
                shard_offset = (
                    sum(self.output_sizes[:loaded_shard_id]) + 128 - 1
                ) // 128
                shard_size = (self.output_sizes[loaded_shard_id] + 128 - 1) // 128
            elif self.quant_type == QuantType.per_Tensor:
                shard_offset = loaded_shard_id
                shard_size = 1
            else:
                # Per-channel same layout as weights
                shard_offset = sum(self.output_sizes[:loaded_shard_id])
                shard_size = self.output_sizes[loaded_shard_id]
        else:
            shard_offset = sum(self.output_sizes[:loaded_shard_id])
            shard_size = self.output_sizes[loaded_shard_id]
        param_data = param_data.narrow(0, shard_offset, shard_size)
        param.weight_loader_process(param_data, loaded_weight)
