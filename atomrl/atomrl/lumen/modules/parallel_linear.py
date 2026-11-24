# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Tensor-parallel linear modules using Lumen FP8 GEMM kernels.

These use Megatron's own TP communication primitives (autograd-aware
all-gather, reduce-scatter, etc.) and route the GEMM through Lumen's
:func:`~atomrl.lumen.ops.quantize.linear.quantized_linear`.

When ``scaling_type="none"`` (the default) and ``delay_wgrad=False``,
plain BF16 ``F.linear`` is used unless ``use_gemm_bf16=True``.  Tuned
AITER GEMM has no autograd, so ``use_gemm_bf16`` routes through
:func:`~atomrl.lumen.ops.quantize.linear.quantized_linear` with
``scaling_type="none"``.  ``delay_wgrad=True`` uses the same path so
backward can defer wgrad.

To enable FP8, call :func:`enable_fp8` on the module or set
``scaling_type`` to one of the supported quantization modes.
"""

import threading
import warnings
from collections.abc import Callable

import torch
import torch.nn.functional as F
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
    set_tensor_model_parallel_attributes,
)
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from torch import nn
from torch.nn.parameter import Parameter

from atomrl.lumen.modules._megatron_compat import (
    condition_init_method,
    ensure_metadata_has_dp_cp_group,
)

__all__ = ["LumenColumnParallelLinear", "LumenRowParallelLinear", "_DeferredWgrad"]


class _DeferredWgrad:
    """Stores and executes deferred weight-gradient computations.

    Used with ``delay_wgrad=True`` in parallel linear modules to split the
    backward pass into dgrad (immediate) and wgrad (deferred).  The deferred
    wgrad GEMM can then overlap with the next layer's communication.

    Supports queuing multiple closures so that modules with multiple GEMMs
    per forward (e.g. ``LumenGroupedLinear`` with per-expert loops) can
    defer all wgrads safely.  ``execute()`` runs all queued closures in
    FIFO order.
    """

    def __init__(self):
        self._queue = []

    def defer(self, fn_or_weight, compute_fn=None):
        """Enqueue a deferred wgrad closure.

        Two calling conventions are supported:

        1. ``defer(fn)`` — ``fn()`` is a self-contained closure that
           computes the weight gradient **and** accumulates it into the
           correct buffer.  Used by the autograd backward.

        2. ``defer(weight, compute_fn)`` — legacy / benchmark API.
           ``compute_fn()`` returns the gradient tensor.  ``execute()``
           will accumulate it into ``weight.main_grad`` or ``weight.grad``.
        """
        if compute_fn is not None:
            weight = fn_or_weight

            # Legacy wrapper: unconditionally targets main_grad when present.
            # Autograd closures gate on gradient_accumulation_fusion internally.
            def _wrapped():
                gw = compute_fn()
                if hasattr(weight, "main_grad"):
                    weight.main_grad.add_(gw)
                elif weight.grad is not None:
                    weight.grad.add_(gw)
                else:
                    weight.grad = gw

            self._queue.append(_wrapped)
        else:
            self._queue.append(fn_or_weight)

    def execute(self):
        """Run all queued wgrad computations in FIFO order."""
        for fn in self._queue:
            fn()
        self._queue.clear()

    @property
    def has_pending(self):
        return len(self._queue) > 0


def _get_megatron_arg(name, default):
    """Read a Megatron argument before or after global args initialization."""
    try:
        from megatron.training import get_args

        args = get_args()
    except (ImportError, RuntimeError, AssertionError):
        return default
    return getattr(args, name, default)


def _use_sdma_from_args() -> bool:
    """Read ``--use-sdma`` from Megatron args (safe fallback to False)."""
    return _get_megatron_arg("use_sdma", False)


def _tp_comm_overlap_from_args() -> bool:
    """Read ``--lumen-tp-comm-overlap`` from args (safe fallback to False)."""
    return _get_megatron_arg("lumen_tp_comm_overlap", False)


def _overlap_mode_from_args():
    return _get_megatron_arg("lumen_tp_comm_overlap_mode", "none")


def _overlap_chunks_from_args():
    return _get_megatron_arg("lumen_tp_comm_overlap_chunks", 4)


def _overlap_method_from_args():
    return _get_megatron_arg("lumen_tp_comm_overlap_method", "nccl")


def _get_tp_group(tp_group, is_expert, pg_collection=None):
    if tp_group is not None:
        return tp_group
    if pg_collection is not None and is_expert:
        return getattr(pg_collection, "expt_tp", None)
    from megatron.core.tensor_parallel.layers import (
        get_tensor_model_parallel_group_if_none,
    )

    return get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)


def _pg_size(group):
    if group is None or not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(group=group)


def _pg_rank(group):
    if group is None or not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank(group=group)


_swiglu_amax_tls = threading.local()


def _pop_swiglu_amax():
    """Pop the pre-computed amax from the last SwiGLU FP8 quantization."""
    amax = getattr(_swiglu_amax_tls, "amax", None)
    _swiglu_amax_tls.amax = None
    return amax


def _resolve_pre_quantized_input_with_swiglu_cache(
    pre_quantized_input, consume_fp8_activation: bool
):
    """Attach fused SwiGLU FP8 cache when this GEMM quantizes activations; else clear it."""
    try:
        from atomrl.lumen.models._swiglu_fp8_fuse import (
            discard_swiglu_fp8_cache,
            pop_swiglu_fp8_cache,
        )
    except ImportError:
        return pre_quantized_input

    if pre_quantized_input is not None:
        discard_swiglu_fp8_cache()
        return pre_quantized_input

    if consume_fp8_activation:
        cached = pop_swiglu_fp8_cache()
        if cached is not None:
            fp8_data, scale = cached[0], cached[1]
            amax = cached[2] if len(cached) > 2 else None
            _swiglu_amax_tls.amax = amax
            return (fp8_data, scale)
        return None

    discard_swiglu_fp8_cache()
    return None


def _discard_swiglu_fp8_cache_safe() -> None:
    try:
        from atomrl.lumen.models._swiglu_fp8_fuse import discard_swiglu_fp8_cache

        discard_swiglu_fp8_cache()
    except ImportError:
        pass


def _do_gemm(
    input_,
    weight,
    bias,
    scaling_manager,
    scaling_type,
    fp8_dtype,
    block_size,
    gradient_accumulation_fusion=False,
    delay_wgrad=False,
    deferred_wgrad=None,
    activation_tensor_id=None,
    pre_quantized_input=None,
    use_gemm_bf16=False,
):
    """Route to Lumen FP8 GEMM or standard F.linear.

    ``delay_wgrad=True`` or ``use_gemm_bf16=True`` always routes through
    :func:`~atomrl.lumen.ops.quantize.linear.quantized_linear` (even for BF16).
    Tuned AITER GEMM has no autograd; ``_QuantizedLinearFn`` supplies it.

    When the weight carries ``_fp8_desc`` (FP8 param storage), the weight
    is already in FP8 format.  For FP8 GEMMs we pass the pre-quantized
    weight and its scale directly via ``pre_quantized_weight``, avoiding
    any BF16 materialization.  For BF16 GEMMs we dequantize into a
    short-lived temporary.
    """
    from atomrl.lumen.ops.quantize.linear import quantized_linear

    _fp8_stored = hasattr(weight, "_fp8_desc") and weight.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
    )
    if (
        not _fp8_stored
        and hasattr(weight, "_fp8_scale")
        and weight.dtype
        in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2,
        )
    ):
        from atomrl.lumen.quantize.descriptor import FP8Descriptor

        weight._fp8_desc = FP8Descriptor(
            data=weight.data,
            scale=weight._fp8_scale,
            fp8_dtype=weight.dtype,
        )
        _fp8_stored = True

    if _fp8_stored and weight._fp8_desc.data.data_ptr() != weight.data.data_ptr():
        weight._fp8_desc.invalidate_transpose()
        del weight._fp8_desc
        _fp8_stored = False

    if _fp8_stored and scaling_type != "none":
        _stored_scale = weight._fp8_desc.scale
        if _stored_scale.numel() > 1:
            # blockwise/blockwise2d: GEMM consumes the block scale directly
            # (dequant factor), NOT the reciprocal used for per-tensor scaling.
            gemm_scale = _stored_scale
        else:
            gemm_scale = getattr(weight, "_fp8_scale_reciprocal", None)
            if gemm_scale is None:
                gemm_scale = 1.0 / _stored_scale
        _pqi = _resolve_pre_quantized_input_with_swiglu_cache(
            pre_quantized_input,
            consume_fp8_activation=True,
        )
        return quantized_linear(
            input_,
            weight,
            bias,
            scaling_manager=scaling_manager,
            scaling_type=scaling_type,
            fp8_dtype=fp8_dtype,
            block_size=block_size,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
            delay_wgrad=delay_wgrad,
            deferred_wgrad=deferred_wgrad,
            pre_quantized_weight=(weight.data, gemm_scale),
            activation_tensor_id=activation_tensor_id,
            pre_quantized_input=_pqi,
        )

    if _fp8_stored:
        orig_dtype = getattr(weight, "_fp8_original_dtype", torch.bfloat16)
        weight_bf16 = (weight.data.to(torch.float32) / weight._fp8_desc.scale).to(
            orig_dtype
        )
        if delay_wgrad or use_gemm_bf16:
            _discard_swiglu_fp8_cache_safe()
            return quantized_linear(
                input_,
                weight_bf16,
                bias,
                scaling_manager=scaling_manager,
                scaling_type="none",
                fp8_dtype=fp8_dtype,
                block_size=block_size,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                delay_wgrad=delay_wgrad,
                deferred_wgrad=deferred_wgrad,
            )
        _discard_swiglu_fp8_cache_safe()
        return F.linear(input_, weight_bf16, bias)

    if scaling_type != "none" or delay_wgrad or use_gemm_bf16:
        fp8_weight_cache = None
        fp8_weight_scale = None
        if scaling_type == "mxfp4":
            # Native Lumen linears bypass quantize._replace_forward, where the
            # per-optimizer-step MXFP4 cache is normally populated. Reuse the
            # same cache on the Parameter itself so eight gradient-accumulation
            # micro-batches do not re-quantize and transpose an unchanged weight.
            # register_mxfp4_weight_optimizer_hooks clears it after step().
            from atomrl.lumen.quantize import _mxfp4_cached_weight

            gemm_rows = input_.numel() // input_.shape[-1]
            fp8_weight_cache, fp8_weight_scale = _mxfp4_cached_weight(
                weight,
                weight,
                None,
                None,
                scaling_type,
                fp8_dtype,
                block_size,
                gemm_rows=gemm_rows,
            )
        _pqi = _resolve_pre_quantized_input_with_swiglu_cache(
            pre_quantized_input,
            consume_fp8_activation=(scaling_type != "none"),
        )
        return quantized_linear(
            input_,
            weight,
            bias,
            scaling_manager=scaling_manager,
            scaling_type=scaling_type,
            fp8_dtype=fp8_dtype,
            block_size=block_size,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
            delay_wgrad=delay_wgrad,
            deferred_wgrad=deferred_wgrad,
            activation_tensor_id=activation_tensor_id,
            pre_quantized_input=_pqi,
            fp8_weight_cache=fp8_weight_cache,
            fp8_weight_scale=fp8_weight_scale,
        )
    _discard_swiglu_fp8_cache_safe()
    return F.linear(input_, weight, bias)


# ---------------------------------------------------------------------------
# ColumnParallelLinear
# ---------------------------------------------------------------------------


class LumenColumnParallelLinear(nn.Module):
    """Column-parallel linear using Lumen GEMM.

    Weight is ``[output_size // tp_size, input_size]``.
    Output is ``[*, output_size // tp_size]`` (each TP rank holds a shard).

    Can be used directly in Megatron layer specs.
    """

    _lora_tp_mode = "column"

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config,
        init_method: Callable,
        gather_output: bool = False,
        bias: bool = True,
        skip_bias_add: bool = False,
        is_expert: bool = False,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: str | None = None,
        tp_group: torch.distributed.ProcessGroup | None = None,
        stride: int = 1,
        pg_collection=None,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert
        self.stride = stride

        self.tp_group = _get_tp_group(tp_group, is_expert, pg_collection)
        self.tp_size = _pg_size(self.tp_group)
        tp_rank = _pg_rank(self.tp_group)

        self.expert_parallel = getattr(config, "expert_model_parallel_size", 1) > 1
        self.explicit_expert_comm = is_expert and (
            self.tp_size > 1 or self.expert_parallel
        )
        self.output_size_per_partition = divide(output_size, self.tp_size)

        self.sequence_parallel = getattr(config, "sequence_parallel", False)
        if self.sequence_parallel and self.tp_size <= 1:
            warnings.warn("sequence_parallel disabled: tp_size <= 1")
            self.sequence_parallel = False

        self.allreduce_dgrad = (
            self.tp_size > 1
            and not self.sequence_parallel
            and not self.explicit_expert_comm
        )

        self.use_sdma = _use_sdma_from_args()
        self.tp_comm_overlap = (
            getattr(config, "lumen_tp_comm_overlap", False)
            or _tp_comm_overlap_from_args()
        )
        self._sdma_comm = None

        # FP8 config (disabled by default; enabled via enable_fp8())
        self.scaling_type = "none"
        self.scaling_manager = None
        from atomrl.lumen.quantize.config import _get_float8_e4m3

        self.fp8_dtype = _get_float8_e4m3()
        self.block_size = 128
        self.gradient_accumulation_fusion = False
        self.delay_wgrad = False
        self.fp8_activation_store = False
        self._deferred_wgrad = _DeferredWgrad()
        self.use_gemm_bf16 = False

        # Pipeline overlap config (fused comm-GEMM)
        _cfg_mode = getattr(config, "lumen_tp_comm_overlap_mode", None)
        self._overlap_mode = (
            _cfg_mode if _cfg_mode is not None else _overlap_mode_from_args()
        )
        _cfg_chunks = getattr(config, "lumen_tp_comm_overlap_chunks", None)
        self._num_chunks = (
            _cfg_chunks if _cfg_chunks is not None else _overlap_chunks_from_args()
        )
        _cfg_method = getattr(config, "lumen_tp_comm_overlap_method", None)
        self._overlap_method = (
            _cfg_method if _cfg_method is not None else _overlap_method_from_args()
        )
        self._pipeline_ag = None
        self._pipeline_rs_for_bwd = None

        # Weight allocation
        if not skip_weight_param_allocation:
            if getattr(config, "use_cpu_initialization", False):
                self.weight = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        input_size,
                        dtype=config.params_dtype,
                    )
                )
                if getattr(config, "perform_initialization", True):
                    _initialize_affine_weight_cpu(
                        self.weight,
                        output_size,
                        input_size,
                        self.output_size_per_partition,
                        0,
                        condition_init_method(config, init_method),
                        stride=stride,
                        return_master_weight=False,
                        rank=tp_rank,
                        world_size=self.tp_size,
                        skip_set_tensor_parallel_attributes=True,
                    )
                set_tensor_model_parallel_attributes(
                    tensor=self.weight, is_parallel=True, dim=0, stride=stride
                )
            else:
                self.weight = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        input_size,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
                if getattr(config, "perform_initialization", True):
                    _initialize_affine_weight_gpu(
                        self.weight,
                        init_method,
                        partition_dim=0,
                        stride=stride,
                        is_expert=is_expert,
                    )
                else:
                    set_tensor_model_parallel_attributes(
                        tensor=self.weight, is_parallel=True, dim=0, stride=stride
                    )
            self.weight.allreduce = not (is_expert and self.expert_parallel)
        else:
            self.weight = None

        if bias:
            if getattr(config, "use_cpu_initialization", False):
                self.bias = Parameter(
                    torch.empty(
                        self.output_size_per_partition, dtype=config.params_dtype
                    )
                )
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            if getattr(config, "perform_initialization", True):
                with torch.no_grad():
                    self.bias.zero_()
            self.bias.allreduce = not (is_expert and self.expert_parallel)
        else:
            self.register_parameter("bias", None)

    def _get_sdma_comm(self):
        if self._sdma_comm is None:
            from atomrl.lumen.modules.sdma_comm import SdmaTpComm

            self._sdma_comm = SdmaTpComm.get(self.tp_group)
        return self._sdma_comm

    def _make_comm_backend(self):
        if self._overlap_method == "sdma":
            from atomrl.lumen.modules.comm_overlap import SdmaCommBackend
            from atomrl.lumen.modules.sdma_comm import SdmaTpComm

            if self._sdma_comm is None:
                self._sdma_comm = SdmaTpComm.get(self.tp_group)
            return SdmaCommBackend(self._sdma_comm)
        from atomrl.lumen.modules.comm_overlap import NcclCommBackend

        return NcclCommBackend(self.tp_group)

    def forward(self, input_: torch.Tensor, weight=None, runtime_gather_output=None):
        if weight is None:
            weight = self.weight

        if (
            self.tp_size > 1
            and self.tp_comm_overlap
            and self._overlap_mode == "pipeline"
            and self.sequence_parallel
            and not self.explicit_expert_comm
        ):
            output_parallel, gemm_bias = self._forward_fused_pipelined_column(
                input_, weight
            )
        elif (
            self.use_sdma
            and self.tp_size > 1
            and self.tp_comm_overlap
            and self.sequence_parallel
            and not self.explicit_expert_comm
        ):
            output_parallel, gemm_bias = self._forward_sdma_overlap_column(
                input_, weight
            )
        else:
            if self.use_sdma and self.tp_size > 1:
                input_parallel = self._forward_sdma_pre_gemm(input_)
            else:
                if self.sequence_parallel or self.explicit_expert_comm:
                    input_parallel = input_
                else:
                    # The copy region is the only source of the dgrad all-reduce here,
                    # since _do_gemm implements no allreduce_dgrad path of its own.
                    input_parallel = copy_to_tensor_model_parallel_region(
                        input_, group=self.tp_group
                    )

                if self.sequence_parallel and not self.explicit_expert_comm:
                    input_parallel = gather_from_sequence_parallel_region(
                        input_parallel,
                        tensor_parallel_output_grad=True,
                        group=self.tp_group,
                    )

            gemm_bias = self.bias if not self.skip_bias_add else None
            output_parallel = _do_gemm(
                input_parallel,
                weight,
                gemm_bias,
                self.scaling_manager,
                self.scaling_type,
                self.fp8_dtype,
                self.block_size,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                delay_wgrad=self.delay_wgrad,
                deferred_wgrad=self._deferred_wgrad if self.delay_wgrad else None,
                activation_tensor_id=getattr(self, "_activation_tensor_id", None),
                use_gemm_bf16=self.use_gemm_bf16,
            )

        gather = self.gather_output
        if runtime_gather_output is not None:
            gather = runtime_gather_output

        if self.use_sdma and self.tp_size > 1 and gather:
            from atomrl.lumen.modules.sdma_comm import (
                sdma_gather_from_tensor_model_parallel_region,
            )

            output = sdma_gather_from_tensor_model_parallel_region(
                output_parallel,
                self._get_sdma_comm(),
            )
        elif gather:
            output = gather_from_tensor_model_parallel_region(
                output_parallel, group=self.tp_group
            )
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def _forward_sdma_pre_gemm(self, input_: torch.Tensor) -> torch.Tensor:
        """SDMA-based input preparation for column-parallel GEMM."""
        from atomrl.lumen.modules.sdma_comm import (
            sdma_copy_to_tensor_model_parallel_region,
            sdma_gather_from_sequence_parallel_region,
        )

        comm = self._get_sdma_comm()
        if self.sequence_parallel or self.explicit_expert_comm:
            input_parallel = input_
        else:
            input_parallel = sdma_copy_to_tensor_model_parallel_region(input_, comm)

        if self.sequence_parallel and not self.explicit_expert_comm:
            input_parallel = sdma_gather_from_sequence_parallel_region(
                input_parallel, comm
            )

        return input_parallel

    def _forward_sdma_overlap_column(
        self, input_: torch.Tensor, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Overlap allgather with GEMM: compute local-shard GEMM while allgather runs."""
        comm = self._get_sdma_comm()
        if getattr(self, "_sdma_stream", None) is None:
            self._sdma_stream = torch.cuda.Stream(device=input_.device)
        sdma_stream = self._sdma_stream
        compute_stream = torch.cuda.current_stream(input_.device)

        input_parallel = input_ if input_.is_contiguous() else input_.contiguous()
        local_dim0 = input_parallel.shape[0]
        comm.allgather_dim0_async(input_parallel, stream=sdma_stream)

        gemm_bias = self.bias if not self.skip_bias_add else None
        _act_tid = getattr(self, "_activation_tensor_id", None)
        out_local = _do_gemm(
            input_parallel,
            weight,
            gemm_bias,
            self.scaling_manager,
            self.scaling_type,
            self.fp8_dtype,
            self.block_size,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            delay_wgrad=self.delay_wgrad,
            deferred_wgrad=self._deferred_wgrad if self.delay_wgrad else None,
            activation_tensor_id=_act_tid,
            use_gemm_bf16=self.use_gemm_bf16,
        )

        input_gathered = comm.wait_allgather_dim0(stream=sdma_stream)
        compute_stream.wait_stream(sdma_stream)

        out_remaining = _do_gemm(
            input_gathered[local_dim0:, ...],
            weight,
            None,
            self.scaling_manager,
            self.scaling_type,
            self.fp8_dtype,
            self.block_size,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            delay_wgrad=self.delay_wgrad,
            deferred_wgrad=self._deferred_wgrad if self.delay_wgrad else None,
            activation_tensor_id=_act_tid,
            use_gemm_bf16=self.use_gemm_bf16,
        )
        output_parallel = torch.cat([out_local, out_remaining], dim=0)
        return output_parallel, self.bias if self.skip_bias_add else None

    def _forward_fused_pipelined_column(
        self, input_: torch.Tensor, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Fused pipelined AG+GEMM forward with real comm-GEMM overlap."""
        from atomrl.lumen.modules.comm_overlap import (
            PipelinedAllgatherGemm,
            PipelinedGemmReduceScatter,
            fused_column_parallel_forward,
        )

        if self._pipeline_ag is None:
            backend = self._make_comm_backend()
            self._pipeline_ag = PipelinedAllgatherGemm(self._num_chunks, backend)
            self._pipeline_rs_for_bwd = PipelinedGemmReduceScatter(
                self._num_chunks, backend
            )

        bias = self.bias if not self.skip_bias_add else None
        output = fused_column_parallel_forward(
            input_ if input_.is_contiguous() else input_.contiguous(),
            weight,
            bias,
            self._pipeline_ag,
            self._pipeline_rs_for_bwd,
            self.weight,
            self.gradient_accumulation_fusion,
            self.delay_wgrad,
            self._deferred_wgrad if self.delay_wgrad else None,
            bias is not None,
            self.scaling_manager,
            self.scaling_type,
            self.fp8_dtype,
            self.block_size,
        )
        return output, self.bias if self.skip_bias_add else None

    def enable_fp8(
        self,
        scaling_manager=None,
        scaling_type="dynamic",
        fp8_dtype=None,
        block_size=None,
    ):
        from atomrl.lumen.quantize import QuantConfig, ScalingManager

        self.scaling_type = scaling_type
        self.scaling_manager = scaling_manager or ScalingManager(QuantConfig())
        if fp8_dtype is not None:
            self.fp8_dtype = fp8_dtype
        if block_size is not None:
            self.block_size = block_size

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            {"weight": 0, "bias": 0},
            sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )

    def set_extra_state(self, state):
        pass

    def get_extra_state(self):
        return torch.empty(0, dtype=torch.uint8)

    def execute_deferred_wgrad(self):
        """Execute any deferred weight gradient computation."""
        self._deferred_wgrad.execute()

    def backward_dw(self):
        """Megatron-compatible API: execute deferred weight gradient.

        Megatron's fine-grained 1F1B scheduler calls ``module.backward_dw()``
        to run the previously deferred wgrad GEMM.  This is a thin wrapper
        around :meth:`execute_deferred_wgrad`.
        """
        self._deferred_wgrad.execute()

    def __repr__(self):
        return (
            f"{type(self).__name__}(in={self.input_size}, out={self.output_size}, "
            f"bias={self.bias is not None}, TP={self.tp_size}, "
            f"fp8={self.scaling_type})"
        )


# ---------------------------------------------------------------------------
# RowParallelLinear
# ---------------------------------------------------------------------------


class LumenRowParallelLinear(nn.Module):
    """Row-parallel linear using Lumen GEMM.

    Weight is ``[output_size, input_size // tp_size]``.
    Input is already split across TP ranks; output is all-reduced.

    Row-parallel linear using Lumen GEMM.
    """

    _lora_tp_mode = "row"

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config,
        init_method: Callable,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: str | None = None,
        tp_group: torch.distributed.ProcessGroup | None = None,
        pg_collection=None,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert

        self.tp_group = _get_tp_group(tp_group, is_expert, pg_collection)
        self.tp_size = _pg_size(self.tp_group)
        tp_rank = _pg_rank(self.tp_group)

        self.expert_parallel = getattr(config, "expert_model_parallel_size", 1) > 1
        self.explicit_expert_comm = is_expert and (
            self.tp_size > 1 or self.expert_parallel
        )
        self.input_size_per_partition = divide(input_size, self.tp_size)

        self.sequence_parallel = getattr(config, "sequence_parallel", False)
        if self.sequence_parallel and not input_is_parallel:
            raise RuntimeError("sequence_parallel requires input_is_parallel=True")

        self.use_sdma = _use_sdma_from_args()
        self.tp_comm_overlap = (
            getattr(config, "lumen_tp_comm_overlap", False)
            or _tp_comm_overlap_from_args()
        )
        self._sdma_comm = None

        # FP8 config
        self.scaling_type = "none"
        self.scaling_manager = None
        from atomrl.lumen.quantize.config import _get_float8_e4m3

        self.fp8_dtype = _get_float8_e4m3()
        self.block_size = 128
        self.gradient_accumulation_fusion = False
        self.delay_wgrad = False
        self.fp8_activation_store = False
        self._deferred_wgrad = _DeferredWgrad()
        self.use_gemm_bf16 = False

        _cfg_mode = getattr(config, "lumen_tp_comm_overlap_mode", None)
        self._overlap_mode = (
            _cfg_mode if _cfg_mode is not None else _overlap_mode_from_args()
        )
        _cfg_chunks = getattr(config, "lumen_tp_comm_overlap_chunks", None)
        self._num_chunks = (
            _cfg_chunks if _cfg_chunks is not None else _overlap_chunks_from_args()
        )
        _cfg_method = getattr(config, "lumen_tp_comm_overlap_method", None)
        self._overlap_method = (
            _cfg_method if _cfg_method is not None else _overlap_method_from_args()
        )
        self._pipeline_rs = None
        self._pipeline_ag_for_bwd = None

        # Weight
        if getattr(config, "use_cpu_initialization", False):
            self.weight = Parameter(
                torch.empty(
                    output_size,
                    self.input_size_per_partition,
                    dtype=config.params_dtype,
                )
            )
            if getattr(config, "perform_initialization", True):
                _initialize_affine_weight_cpu(
                    self.weight,
                    output_size,
                    input_size,
                    self.input_size_per_partition,
                    1,
                    condition_init_method(config, init_method),
                    stride=1,
                    return_master_weight=False,
                    params_dtype=config.params_dtype,
                    rank=tp_rank,
                    world_size=self.tp_size,
                    skip_set_tensor_parallel_attributes=True,
                )
        else:
            self.weight = Parameter(
                torch.empty(
                    output_size,
                    self.input_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if getattr(config, "perform_initialization", True):
                _initialize_affine_weight_gpu(
                    self.weight,
                    init_method,
                    partition_dim=1,
                    stride=1,
                    is_expert=is_expert,
                )
        self.weight.allreduce = not (is_expert and self.expert_parallel)

        if bias:
            if getattr(config, "use_cpu_initialization", False):
                self.bias = Parameter(
                    torch.empty(output_size, dtype=config.params_dtype)
                )
            else:
                self.bias = Parameter(
                    torch.empty(
                        output_size,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
            if getattr(config, "perform_initialization", True):
                with torch.no_grad():
                    self.bias.zero_()
            self.bias.allreduce = not (is_expert and self.expert_parallel)
            self.bias.sequence_parallel = self.sequence_parallel
        else:
            self.register_parameter("bias", None)

    def _get_sdma_comm(self):
        if self._sdma_comm is None:
            from atomrl.lumen.modules.sdma_comm import SdmaTpComm

            self._sdma_comm = SdmaTpComm.get(self.tp_group)
        return self._sdma_comm

    def _make_comm_backend(self):
        if self._overlap_method == "sdma":
            from atomrl.lumen.modules.comm_overlap import SdmaCommBackend
            from atomrl.lumen.modules.sdma_comm import SdmaTpComm

            if self._sdma_comm is None:
                self._sdma_comm = SdmaTpComm.get(self.tp_group)
            return SdmaCommBackend(self._sdma_comm)
        from atomrl.lumen.modules.comm_overlap import NcclCommBackend

        return NcclCommBackend(self.tp_group)

    def forward(self, input_: torch.Tensor):
        if self.input_is_parallel:
            input_parallel = input_
        else:
            from megatron.core.tensor_parallel.mappings import (
                scatter_to_tensor_model_parallel_region,
            )

            input_parallel = scatter_to_tensor_model_parallel_region(
                input_, group=self.tp_group
            )

        if (
            self.tp_size > 1
            and self.tp_comm_overlap
            and self._overlap_mode == "pipeline"
            and self.sequence_parallel
            and not self.explicit_expert_comm
        ):
            output_ = self._forward_fused_pipelined_row(input_parallel)
        else:
            output_parallel = _do_gemm(
                input_parallel,
                self.weight,
                None,
                self.scaling_manager,
                self.scaling_type,
                self.fp8_dtype,
                self.block_size,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                delay_wgrad=self.delay_wgrad,
                deferred_wgrad=self._deferred_wgrad if self.delay_wgrad else None,
                activation_tensor_id=getattr(self, "_activation_tensor_id", None),
                use_gemm_bf16=self.use_gemm_bf16,
            )

            if self.explicit_expert_comm:
                output_ = output_parallel
            elif self.use_sdma and self.tp_size > 1 and self.tp_comm_overlap:
                output_ = self._forward_sdma_overlap_row(output_parallel)
            elif self.use_sdma and self.tp_size > 1:
                output_ = self._forward_sdma_post_gemm(output_parallel)
            elif self.sequence_parallel:
                output_ = reduce_scatter_to_sequence_parallel_region(
                    output_parallel, group=self.tp_group
                )
            else:
                output_ = reduce_from_tensor_model_parallel_region(
                    output_parallel, group=self.tp_group
                )

        if not self.skip_bias_add:
            output = (output_ + self.bias) if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias

    def _forward_sdma_post_gemm(self, output_parallel: torch.Tensor) -> torch.Tensor:
        """SDMA-based output reduction for row-parallel GEMM."""
        from atomrl.lumen.modules.sdma_comm import (
            sdma_reduce_from_tensor_model_parallel_region,
            sdma_reduce_scatter_to_sequence_parallel_region,
        )

        comm = self._get_sdma_comm()
        if self.sequence_parallel:
            return sdma_reduce_scatter_to_sequence_parallel_region(
                output_parallel, comm
            )
        else:
            return sdma_reduce_from_tensor_model_parallel_region(output_parallel, comm)

    def _forward_sdma_overlap_row(self, output_parallel: torch.Tensor) -> torch.Tensor:
        """Overlap reduce-scatter / allreduce with GEMM via async SDMA."""
        comm = self._get_sdma_comm()
        if getattr(self, "_sdma_stream", None) is None:
            self._sdma_stream = torch.cuda.Stream(device=output_parallel.device)
        sdma_stream = self._sdma_stream
        compute_stream = torch.cuda.current_stream(output_parallel.device)
        sdma_stream.wait_stream(compute_stream)

        if self.sequence_parallel:
            comm.reduce_scatter_dim0_async(output_parallel, stream=sdma_stream)
            return comm.wait_reduce_scatter_dim0(stream=sdma_stream)
        else:
            comm.allreduce_sum_async(output_parallel, stream=sdma_stream)
            comm.wait_allreduce_sum(stream=sdma_stream)
            return output_parallel

    def _forward_fused_pipelined_row(
        self, input_parallel: torch.Tensor
    ) -> torch.Tensor:
        """Fused pipelined GEMM+RS forward with real comm-GEMM overlap."""
        from atomrl.lumen.modules.comm_overlap import (
            PipelinedAllgatherGemm,
            PipelinedGemmReduceScatter,
            fused_row_parallel_forward,
        )

        if self._pipeline_rs is None:
            backend = self._make_comm_backend()
            self._pipeline_rs = PipelinedGemmReduceScatter(self._num_chunks, backend)
            self._pipeline_ag_for_bwd = PipelinedAllgatherGemm(
                self._num_chunks, backend
            )

        return fused_row_parallel_forward(
            input_parallel,
            self.weight,
            self._pipeline_rs,
            self._pipeline_ag_for_bwd,
            self.weight,
            self.gradient_accumulation_fusion,
            self.delay_wgrad,
            self._deferred_wgrad if self.delay_wgrad else None,
            self.scaling_manager,
            self.scaling_type,
            self.fp8_dtype,
            self.block_size,
        )

    def enable_fp8(
        self,
        scaling_manager=None,
        scaling_type="dynamic",
        fp8_dtype=None,
        block_size=None,
    ):
        from atomrl.lumen.quantize import QuantConfig, ScalingManager

        self.scaling_type = scaling_type
        self.scaling_manager = scaling_manager or ScalingManager(QuantConfig())
        if fp8_dtype is not None:
            self.fp8_dtype = fp8_dtype
        if block_size is not None:
            self.block_size = block_size

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        state_dict = self.state_dict(prefix="", keep_vars=True)
        shard_map = {"weight": 1}
        if self.bias is not None:
            shard_map["bias"] = 0
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            shard_map,
            sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )

    def set_extra_state(self, state):
        pass

    def get_extra_state(self):
        return torch.empty(0, dtype=torch.uint8)

    def execute_deferred_wgrad(self):
        """Execute any deferred weight gradient computation."""
        self._deferred_wgrad.execute()

    def backward_dw(self):
        """Megatron-compatible API: execute deferred weight gradient.

        Megatron's fine-grained 1F1B scheduler calls ``module.backward_dw()``
        to run the previously deferred wgrad GEMM.  This is a thin wrapper
        around :meth:`execute_deferred_wgrad`.
        """
        self._deferred_wgrad.execute()

    def __repr__(self):
        return (
            f"{type(self).__name__}(in={self.input_size}, out={self.output_size}, "
            f"bias={self.bias is not None}, TP={self.tp_size}, "
            f"fp8={self.scaling_type})"
        )
