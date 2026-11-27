# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""FP8 parameter storage, all-gather communication, and dequant hooks.

**Layer 1** — local FP8 cache + lazy re-quant (wired via
:class:`~atomrl.lumen.quantize.scaling_manager.ScalingManager`).

**Layer 2** — FP8 all-gather communication: each rank quantizes its
local weight shard to FP8 (1 byte/elem), all-gathers the FP8 shards
(half bandwidth vs BF16), then dequantizes per-shard back to BF16.
Supports NCCL and SDMA backends.
"""

import logging

import torch
import torch.distributed as dist
from torch import nn

from atomrl.lumen.quantize.descriptor import FP8Descriptor

logger = logging.getLogger(__name__)


class _FP8LinearFunc(torch.autograd.Function):
    """F.linear that saves only FP8 weight + scale for backward.

    Standard F.linear saves the full BF16 weight for every layer's backward,
    causing N_layers of BF16 copies to accumulate.  This version saves only
    the 1-byte FP8 weight + a scalar scale and reconstructs BF16 on-the-fly
    during backward.  Since FP8PM weights are frozen, we skip grad_weight
    entirely and don't save the input tensor either.
    """

    @staticmethod
    def forward(ctx, input, fp8_weight, scale, bias, original_dtype):
        bf16_weight = dequantize_param_from_fp8(fp8_weight, scale, original_dtype)
        output = torch.nn.functional.linear(input, bf16_weight, bias)
        # Clone the FP8 weight so we don't pin the FSDP2 allgathered buffer.
        # Without clone, the autograd reference prevents FSDP2 param_offload
        # from freeing allgathered copies, causing all layers' weights to
        # accumulate on GPU (69 GB vs 48 GB for Qwen 0.5B).
        # The clone is FP8 (1 byte/elem), so total overhead is ~N_params bytes.
        ctx.save_for_backward(fp8_weight.clone(), scale)
        ctx.has_bias = bias is not None
        ctx.original_dtype = original_dtype
        return output

    @staticmethod
    def backward(ctx, grad_output):
        fp8_weight, scale = ctx.saved_tensors
        bf16_weight = dequantize_param_from_fp8(fp8_weight, scale, ctx.original_dtype)

        grad_input = grad_output @ bf16_weight
        grad_bias = (
            grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))
            if ctx.has_bias
            else None
        )
        return grad_input, None, None, grad_bias, None


class _FP8MegatronLinearFunc(torch.autograd.Function):
    """Megatron-compatible linear that quantizes BF16→FP8 on-the-fly.

    Unlike the ``nn.Linear`` variant, this does NOT require pre-quantized
    weights.  Parameters stay in BF16 (preserving Megatron's distributed
    optimizer and DDP), while the autograd graph stores compact FP8 weight
    + scale instead of the full BF16 copy—halving the weight portion of
    saved-tensor memory.  Handles ``allreduce_dgrad`` for tensor-parallel
    gradient communication.
    """

    @staticmethod
    def forward(ctx, input, weight, bias, allreduce_dgrad, tp_group, fp8_dtype):
        output = torch.matmul(input, weight.t())
        if bias is not None:
            output = output + bias
        fp8_weight, scale = quantize_param_to_fp8(weight.detach(), fp8_dtype)
        ctx.save_for_backward(input.detach(), fp8_weight, scale)
        ctx.original_dtype = weight.dtype
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.tp_group = tp_group
        ctx.has_bias = bias is not None
        ctx.compute_weight_grad = weight.requires_grad
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_saved, fp8_weight, scale = ctx.saved_tensors
        bf16_weight = dequantize_param_from_fp8(fp8_weight, scale, ctx.original_dtype)

        grad_input = grad_output.matmul(bf16_weight)
        if ctx.allreduce_dgrad:
            dist.all_reduce(grad_input, group=ctx.tp_group)

        grad_weight = None
        if ctx.compute_weight_grad:
            go = grad_output.reshape(-1, grad_output.shape[-1])
            inp = input_saved.reshape(-1, input_saved.shape[-1])
            grad_weight = go.t().matmul(inp)

        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(dim=0)

        return grad_input, grad_weight, grad_bias, None, None, None


def quantize_param_to_fp8(
    param: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple:
    """Quantize a parameter tensor to FP8 with per-tensor scale.

    Returns:
        Tuple of (fp8_tensor, scale) where scale is a scalar float32.
    """
    amax = param.abs().amax().clamp(min=1e-12)
    if fp8_dtype == torch.float8_e4m3fn:
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
    elif fp8_dtype == torch.float8_e4m3fnuz:
        fp8_max = torch.finfo(torch.float8_e4m3fnuz).max
    else:
        fp8_max = 448.0
    scale = fp8_max / amax
    fp8_param = (param.float() * scale).to(fp8_dtype)
    return fp8_param, scale


def dequantize_param_from_fp8(
    fp8_param: torch.Tensor,
    scale: torch.Tensor,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an FP8 parameter back to target dtype."""
    return (fp8_param.to(torch.float32) / scale).to(target_dtype)


def _split_dim0_chunks(
    tensor: torch.Tensor, num_chunks: int
) -> tuple[torch.Tensor, ...]:
    """Split a dim-0 shard into evenly sized chunks."""
    if num_chunks < 1:
        raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")

    rows = tensor.shape[0]
    if rows % num_chunks != 0:
        raise ValueError(f"num_chunks={num_chunks} must evenly divide dim0 rows={rows}")

    return tensor.chunk(num_chunks, dim=0)


def _dequantize_gathered_fp8_chunk(
    fp8_chunk: torch.Tensor,
    per_rank_scales: torch.Tensor,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize one gathered FP8 chunk using its per-rank scales."""
    world_size = per_rank_scales.numel()
    # Pre-allocate full output to avoid N intermediate BF16 tensors + torch.cat.
    out = torch.empty(fp8_chunk.shape, dtype=target_dtype, device=fp8_chunk.device)
    rows_per_rank = fp8_chunk.shape[0] // world_size
    for rank in range(world_size):
        row_s = rank * rows_per_rank
        row_e = row_s + rows_per_rank
        out[row_s:row_e].copy_(
            fp8_chunk[row_s:row_e].to(torch.float32).div_(per_rank_scales[rank])
        )
    return out


def fp8_allgather_weight(
    weight_shard: torch.Tensor,
    group: dist.ProcessGroup | None = None,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    target_dtype: torch.dtype = torch.bfloat16,
    use_sdma: bool = False,
    num_chunks: int = 1,
) -> torch.Tensor:
    """All-gather a weight shard via FP8, dequantize per-shard to *target_dtype*.

    This is the "Layer 2" FP8 param all-gather: each rank quantizes its
    local shard to FP8, communicates only the FP8 bytes (half the volume
    of BF16), then each rank independently dequantizes back to full
    precision.  Per-rank scales are also all-gathered so that each
    chunk is dequantized with its originating rank's scale.

    Args:
        weight_shard: Local weight shard ``[rows_local, cols]`` in BF16/FP32.
        group: Process group for the all-gather. Defaults to WORLD.
        fp8_dtype: Target FP8 dtype for quantization.
        target_dtype: Dtype to dequantize into after gathering.
        use_sdma: Use MORI SDMA allgather instead of NCCL.
        num_chunks: Number of dim-0 chunks to communicate sequentially.

    Returns:
        Full weight tensor ``[rows_full, cols]`` in *target_dtype*.
    """
    if group is None:
        group = dist.group.WORLD
    world_size = dist.get_world_size(group=group)

    if world_size <= 1:
        return weight_shard.to(target_dtype)

    fp8_shard, local_scale = quantize_param_to_fp8(weight_shard, fp8_dtype)

    # All-gather per-rank scales (tiny: one float32 scalar per rank)
    scale_scalar = local_scale.float().reshape(1).to(fp8_shard.device)
    all_scales = torch.empty(world_size, dtype=torch.float32, device=fp8_shard.device)
    dist.all_gather_into_tensor(all_scales, scale_scalar, group=group)

    fp8_shard_chunks = _split_dim0_chunks(fp8_shard, num_chunks)
    dequantized_full_chunks = []

    for fp8_shard_chunk in fp8_shard_chunks:
        full_shape = list(fp8_shard_chunk.shape)
        full_shape[0] *= world_size
        fp8_full_chunk = torch.empty(
            full_shape, dtype=fp8_shard_chunk.dtype, device=fp8_shard_chunk.device
        )

        if use_sdma:
            from atomrl.lumen.modules.sdma_comm import SdmaTpComm

            comm = SdmaTpComm.get(group)
            fp8_full_chunk = comm.allgather_dim0(fp8_shard_chunk)
        else:
            dist.all_gather_into_tensor(
                fp8_full_chunk, fp8_shard_chunk.contiguous(), group=group
            )

        dequantized_full_chunks.append(
            _dequantize_gathered_fp8_chunk(fp8_full_chunk, all_scales, target_dtype)
        )

    return (
        dequantized_full_chunks[0]
        if len(dequantized_full_chunks) == 1
        else torch.cat(dequantized_full_chunks, dim=0)
    )


def fp8_allgather_weight_pipelined(
    weight_shards: list,
    group: dist.ProcessGroup | None = None,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    target_dtype: torch.dtype = torch.bfloat16,
    use_sdma: bool = False,
    num_chunks: int = 1,
) -> list:
    """Pipelined FP8 all-gather for multiple weight shards.

    Overlaps allgather(i+1) on a comm stream with dequant(i) on the
    compute stream, hiding dequant latency behind communication.

    Args:
        weight_shards: List of local weight shards to all-gather.
        group: Process group. Defaults to WORLD.
        fp8_dtype: Target FP8 dtype.
        target_dtype: Dtype to dequantize into after gathering.
        use_sdma: Use MORI SDMA allgather instead of NCCL.
        num_chunks: Number of dim-0 chunks to pipeline per weight shard.

    Returns:
        List of full (dequantized) weight tensors.
    """
    if group is None:
        group = dist.group.WORLD
    world_size = dist.get_world_size(group=group)

    if world_size <= 1:
        return [s.to(target_dtype) for s in weight_shards]

    N = len(weight_shards)
    device = weight_shards[0].device
    fp8_shards = []
    local_scales = []
    fp8_shard_chunks = []
    fp8_out_chunks = []
    all_scales_list = []

    for shard in weight_shards:
        fp8_s, sc = quantize_param_to_fp8(shard, fp8_dtype)
        fp8_shards.append(fp8_s)
        local_scales.append(sc)
        shard_chunks = _split_dim0_chunks(fp8_s, num_chunks)
        fp8_shard_chunks.append(shard_chunks)
        chunk_outs = []
        for shard_chunk in shard_chunks:
            full_shape = list(shard_chunk.shape)
            full_shape[0] *= world_size
            chunk_outs.append(
                torch.empty(full_shape, dtype=shard_chunk.dtype, device=device)
            )
        fp8_out_chunks.append(chunk_outs)

    # All-gather per-rank scales for each weight (batched, tiny payload)
    for sc in local_scales:
        sc_tensor = sc.float().reshape(1).to(device)
        gathered = torch.empty(world_size, dtype=torch.float32, device=device)
        dist.all_gather_into_tensor(gathered, sc_tensor, group=group)
        all_scales_list.append(gathered)

    comm_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.current_stream(device)
    # Keep NCCL subgroup collectives on `comm_stream` ordered after the scale
    # all-gathers already enqueued on the current stream.
    comm_stream.wait_stream(compute_stream)

    results = [[None] * num_chunks for _ in range(N)]
    chunk_tasks = [
        (weight_idx, chunk_idx)
        for weight_idx in range(N)
        for chunk_idx in range(num_chunks)
    ]

    def _do_gather(weight_idx: int, chunk_idx: int):
        if use_sdma:
            from atomrl.lumen.modules.sdma_comm import SdmaTpComm

            comm = SdmaTpComm.get(group)
            fp8_out_chunks[weight_idx][chunk_idx] = comm.allgather_dim0(
                fp8_shard_chunks[weight_idx][chunk_idx]
            )
        else:
            dist.all_gather_into_tensor(
                fp8_out_chunks[weight_idx][chunk_idx],
                fp8_shard_chunks[weight_idx][chunk_idx].contiguous(),
                group=group,
            )

    def _do_dequant(weight_idx: int, chunk_idx: int):
        results[weight_idx][chunk_idx] = _dequantize_gathered_fp8_chunk(
            fp8_out_chunks[weight_idx][chunk_idx],
            all_scales_list[weight_idx],
            target_dtype,
        )

    with torch.cuda.stream(comm_stream):
        _do_gather(*chunk_tasks[0])

    for task_idx, (weight_idx, chunk_idx) in enumerate(chunk_tasks):
        compute_stream.wait_stream(comm_stream)
        if task_idx + 1 < len(chunk_tasks):
            with torch.cuda.stream(comm_stream):
                _do_gather(*chunk_tasks[task_idx + 1])
        _do_dequant(weight_idx, chunk_idx)

    compute_stream.wait_stream(comm_stream)
    return [
        weight_chunks[0] if len(weight_chunks) == 1 else torch.cat(weight_chunks, dim=0)
        for weight_chunks in results
    ]


class FP8ParamManager:
    """Manages FP8 parameter quantization and all-gather hooks.

    When enabled, linear layer weights are stored in FP8 and
    dequantized on-the-fly before computation. All-gather operations
    communicate FP8 data (half the bandwidth of BF16).

    Supports both ``nn.Linear`` and Megatron-Core's
    ``ColumnParallelLinear`` / ``RowParallelLinear``.

    Args:
        fp8_dtype: Target FP8 dtype for parameters.
    """

    def __init__(self, fp8_dtype: torch.dtype = torch.float8_e4m3fn):
        self.fp8_dtype = fp8_dtype
        self._param_scales: dict = {}
        self._original_dtypes: dict = {}
        self._hooks: list = []
        self._wrapped_modules: list = []
        self._state_dict_handles: list = []
        self._megatron_modules: set = set()

    _QUANTIZABLE_TYPES = (nn.Linear,)
    _MEGATRON_TYPES: tuple = ()
    _megatron_probed: bool = False

    @classmethod
    def _get_quantizable_types(cls) -> tuple:
        """Return all quantizable module types, including Megatron if available."""
        if not cls._megatron_probed:
            cls._megatron_probed = True
            try:
                from megatron.core.tensor_parallel.layers import (
                    ColumnParallelLinear,
                    RowParallelLinear,
                )

                cls._MEGATRON_TYPES = (ColumnParallelLinear, RowParallelLinear)
            except ImportError:
                cls._MEGATRON_TYPES = ()
        return cls._QUANTIZABLE_TYPES + cls._MEGATRON_TYPES

    def _is_megatron_module(self, module: nn.Module) -> bool:
        self._get_quantizable_types()  # ensures probing
        return self._MEGATRON_TYPES and isinstance(module, self._MEGATRON_TYPES)

    def quantize_params(self, model: nn.Module) -> int:
        """Quantize linear weights in the model to FP8.

        For ``nn.Linear``, quantizes in-place (param.data becomes FP8).
        For Megatron ``ColumnParallelLinear`` / ``RowParallelLinear``,
        records the module for on-the-fly forward-hook wrapping only—
        parameters stay in BF16 so Megatron's distributed optimizer and
        DDP remain functional.

        Returns:
            Number of parameters targeted.
        """
        quantizable = self._get_quantizable_types()
        count = 0
        megatron_count = 0
        for name, module in model.named_modules():
            if not isinstance(module, quantizable):
                continue
            weight = getattr(module, "weight", None)
            if weight is None or not isinstance(weight, nn.Parameter):
                continue
            if weight.dtype in (
                torch.float8_e4m3fn,
                torch.float8_e4m3fnuz,
                torch.float8_e5m2,
            ):
                continue

            if self._is_megatron_module(module):
                self._megatron_modules.add(name)
                megatron_count += 1
                count += 1
                continue

            self._original_dtypes[name] = weight.dtype
            fp8_weight, scale = quantize_param_to_fp8(weight.data, self.fp8_dtype)
            self._param_scales[name] = scale

            weight.data = fp8_weight.to(weight.device)
            weight._fp8_desc = FP8Descriptor(
                data=weight.data, scale=scale, fp8_dtype=self.fp8_dtype
            )
            weight._fp8_original_dtype = self._original_dtypes[name]
            weight.requires_grad_(False)
            count += 1

        linear_count = count - megatron_count
        logger.info(
            "Quantized %d parameters to FP8 (%s): %d nn.Linear (in-place), "
            "%d Megatron parallel (on-the-fly)",
            count,
            self.fp8_dtype,
            linear_count,
            megatron_count,
        )
        return count

    def register_dequant_hooks(self, model: nn.Module) -> int:
        """Patch forward to dequant FP8→BF16 with minimal memory.

        For ``nn.Linear`` (in ``_param_scales``), replaces ``forward``
        with ``_FP8LinearFunc`` that dequantizes pre-quantized FP8 data.
        For Megatron parallel linears (in ``_megatron_modules``), wraps
        ``_forward_impl`` with ``_FP8MegatronLinearFunc`` that quantizes
        BF16→FP8 on-the-fly and saves compact FP8 for backward.

        Returns:
            Number of modules patched.
        """
        self.register_state_dict_hooks(model)
        count = 0
        for name, module in model.named_modules():
            if name in self._param_scales:
                self._wrap_forward_to_use_dequant(module)
                self._wrapped_modules.append(module)
                count += 1
            elif name in self._megatron_modules:
                self._wrap_megatron_forward_to_use_dequant(module)
                self._wrapped_modules.append(module)
                count += 1
        return count

    def register_state_dict_hooks(self, model: nn.Module) -> int:
        """Register hooks so ``state_dict`` serializes dequantized weights.

        FP8 parameters are temporarily cast to their original dtype (default
        bfloat16) before tensors are collected, then restored so runtime
        storage stays in FP8.
        """

        def _pre_save(module, prefix, keep_vars):
            for param in module._parameters.values():
                if param is None or not hasattr(param, "_fp8_desc"):
                    continue
                orig_dtype = getattr(param, "_fp8_original_dtype", torch.bfloat16)
                param._state_dict_data_backup = param.data
                scale = param._fp8_desc.scale.to(param.data.device)
                param.data = (param.data.to(torch.float32) / scale).to(orig_dtype)

        def _post_save(module, state_dict, prefix, local_metadata):
            for param in module._parameters.values():
                if param is None or not hasattr(param, "_state_dict_data_backup"):
                    continue
                param.data = param._state_dict_data_backup
                del param._state_dict_data_backup

        count = 0
        for mod in model.modules():
            if not any(
                p is not None and hasattr(p, "_fp8_desc")
                for p in mod._parameters.values()
            ):
                continue
            self._state_dict_handles.append(mod.register_state_dict_pre_hook(_pre_save))
            self._state_dict_handles.append(
                mod.register_state_dict_post_hook(_post_save)
            )
            count += 1
        return count

    def _wrap_forward_to_use_dequant(self, module: nn.Module) -> None:
        """Replace forward to use FP8→BF16 dequant with minimal memory overhead.

        Uses ``_FP8LinearFunc`` which saves only the compact FP8 weight + scale
        for backward instead of a full BF16 copy, keeping peak memory close to
        the FP8 storage footprint.
        """
        if hasattr(module, "_fp8_original_forward"):
            return
        original_forward = module.forward
        original_dtype = getattr(module.weight, "_fp8_original_dtype", torch.bfloat16)

        def fp8_aware_forward(*args, **kwargs):
            weight = module.weight
            if hasattr(weight, "_fp8_desc"):
                fp8_data = weight.data
                scale = weight._fp8_desc.scale.to(fp8_data.device)
                return _FP8LinearFunc.apply(
                    args[0], fp8_data, scale, module.bias, original_dtype
                )
            return torch.nn.functional.linear(args[0], weight, module.bias)

        module._fp8_original_forward = original_forward
        module.forward = fp8_aware_forward

    def _wrap_megatron_forward_to_use_dequant(self, module: nn.Module) -> None:
        """Wrap Megatron ``_forward_impl`` with on-the-fly FP8 quantization.

        Uses ``_FP8MegatronLinearFunc`` which quantizes BF16 weights to
        FP8 during forward and saves the compact FP8 data for backward,
        halving weight storage in the autograd graph.  Parameters stay in
        BF16 so Megatron's distributed optimizer and DDP are unaffected.
        """
        if hasattr(module, "_fp8_original_forward_impl"):
            return
        original_forward_impl = module._forward_impl
        fp8_dtype = self.fp8_dtype

        def fp8_forward_impl(
            input,
            weight,
            bias=None,
            gradient_accumulation_fusion=False,
            allreduce_dgrad=False,
            sequence_parallel=False,
            tp_group=None,
            grad_output_buffer=None,
            wgrad_deferral_limit=None,
        ):
            if sequence_parallel:
                try:
                    from megatron.core.tensor_parallel.mappings import (
                        gather_from_sequence_parallel_region,
                    )

                    input = gather_from_sequence_parallel_region(
                        input,
                        tensor_parallel_output_grad=True,
                        group=tp_group,
                    )
                except ImportError:
                    pass
            return _FP8MegatronLinearFunc.apply(
                input,
                weight,
                bias,
                allreduce_dgrad,
                tp_group,
                fp8_dtype,
            )

        module._fp8_original_forward_impl = original_forward_impl
        module._forward_impl = fp8_forward_impl

    def remove_hooks(self):
        """Remove all registered hooks and restore original forwards."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        for handle in self._state_dict_handles:
            handle.remove()
        self._state_dict_handles.clear()
        for module in self._wrapped_modules:
            if hasattr(module, "_fp8_original_forward"):
                module.forward = module._fp8_original_forward
                del module._fp8_original_forward
            if hasattr(module, "_fp8_original_forward_impl"):
                module._forward_impl = module._fp8_original_forward_impl
                del module._fp8_original_forward_impl
        self._wrapped_modules.clear()

    def memory_savings_bytes(self, model: nn.Module) -> int:
        """Estimate memory savings from FP8 params (bytes)."""
        saved = 0
        for name in self._param_scales:
            parts = name.split(".")
            mod = model
            for p in parts:
                mod = getattr(mod, p, None)
                if mod is None:
                    break
            if mod is not None and hasattr(mod, "weight"):
                numel = mod.weight.numel()
                saved += numel  # saving 1 byte per element (2 bytes BF16 -> 1 byte FP8)
        return saved
