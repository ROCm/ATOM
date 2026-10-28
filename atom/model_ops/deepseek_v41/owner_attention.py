# SPDX-License-Identifier: MIT
"""Full-head sparse attention on row owners with immutable loaded weights.

Head/group weights are replicated once after loading. Queries and projected
outputs remain on owners; only the shared KV latent is gathered. The native
sparse attention, RoPE, MXFP8 GEMMs and cache writers retain their arithmetic.
"""

import copy

import torch
import torch.distributed as dist
from aiter.dist.parallel_state import get_tp_group
from aiter.ops.batched_gemm_op_a8w8 import batched_gemm_a8w8_mxscale_bpreshuffle
from aiter.ops.inverse_rope_group_quant import inverse_rope_group_quant
from aiter.ops.shuffle import shuffle_weight
from torch import nn

from atom.model_ops.deepseek_v41.row_sharded_mhc import gather_norm
from atom.model_ops.v4_kernels import sparse_attn_v4_paged_2src


def unshuffle_fp8(weight):
    """Inverse of the gfx950 16x16 byte-weight permutation, for K concatenation."""
    n, k = weight.shape
    return (
        weight.view(torch.uint8)
        .view(1, n // 16, k // 32, 2, 16, 16)
        .permute(0, 1, 4, 2, 3, 5)
        .contiguous()
        .view(n, k)
        .view(weight.dtype)
    )


def _gather(value, dim):
    group = get_tp_group()
    # RCCL does not accept E8M0: transport scale/weight bytes losslessly.
    dtype = value.dtype
    transport = value.view(torch.uint8) if value.element_size() == 1 else value
    parts = [torch.empty_like(transport) for _ in range(group.world_size)]
    dist.all_gather(parts, transport.contiguous(), group=group.device_group)
    return torch.cat(parts, dim=dim).view(dtype)


def bind_linear(linear, dim):
    """Copy the loaded parameter contract, gathering source bytes without requant."""
    assert linear.native_a8_group_rows == 32
    assert linear.weight.dtype == torch.float8_e4m3fn
    assert linear.bias is None
    shuffled = getattr(linear.weight, "is_shuffled", False)
    if dim == 1 and shuffled:
        weight = shuffle_weight(_gather(unshuffle_fp8(linear.weight), dim))
    else:
        weight = _gather(linear.weight, dim)
        if shuffled:
            weight.is_shuffled = True
    scale = _gather(linear.weight_scale, dim)
    bound = copy.copy(linear)
    bound._parameters = dict(linear._parameters)
    bound.weight = nn.Parameter(weight, requires_grad=False)
    bound.weight.is_shuffled = shuffled
    bound.weight_scale = nn.Parameter(scale, requires_grad=False)
    bound.input_size = weight.shape[1]
    bound.output_size = weight.shape[0]
    bound.output_partition_sizes = [bound.output_size]
    bound.reduce_results = False
    return bound


def bind_model(model):
    """Called exactly once after source weights have reached their final layouts."""
    if not model.use_row_sharded_mhc:
        raise ValueError("owner attention requires the owner-state deployment")
    group = get_tp_group()
    assert group.world_size == 4
    # Wait for every rank's checkpoint load before the first GPU collective
    # initializes RCCL. Uneven shard I/O must not consume its setup timeout.
    dist.barrier(group=group.cpu_group)
    extra_bytes = 0
    for block in model.layers:
        if not block.row_sharded_mhc:
            continue
        attention = block.attn
        if getattr(attention, "owner_attention_bound", False):
            raise RuntimeError("owner attention weights already bound")
        for name, dim in (("wq_b", 0), ("wo_a", 0), ("wo_b", 1)):
            bound = bind_linear(getattr(attention, name), dim)
            attention.__dict__[name + "_owned"] = bound
            extra_bytes += sum(p.numel() * p.element_size() for p in bound.parameters())
        attention.__dict__["sink_owned"] = _gather(attention.attn_sink, 0)
        extra_bytes += (
            attention.sink_owned.numel() * attention.sink_owned.element_size()
        )
        attention.owner_attention_bound = True
    torch.cuda.synchronize()
    print(
        f"OWNER_ATTENTION_BOUND rank={group.rank_in_group} extra_bytes={extra_bytes}",
        flush=True,
    )


def project_out(attention, output, rope, positions):
    """All output groups on owner rows, returning full hidden without a collective."""
    rows = output.shape[0]
    groups = attention.groups * get_tp_group().world_size
    fp8, scales = inverse_rope_group_quant(
        output,
        positions.flatten(),
        rope.cos_cache,
        rope.sin_cache,
        num_groups=groups,
        quant_group_size=32,
    )
    linear = attention.wo_a_owned
    projected = batched_gemm_a8w8_mxscale_bpreshuffle(
        fp8,
        linear.weight.view(groups, attention.o_rank, -1),
        scales,
        linear.weight_scale.view(groups, attention.o_rank // 32, -1),
        dtype=output.dtype,
    )
    return attention.wo_b_owned(projected.view(1, rows, -1))


def forward(
    attention, hidden, cache, step, rope, *, hidden_fp8=None, hidden_scale=None
):
    """One complete owner attention boundary, including every required cache write."""
    assert attention.owner_attention_bound and not step.decode
    group = get_tp_group()
    capacity = hidden.shape[-2]
    start = group.rank_in_group * capacity
    valid = min(capacity, step.width - start)
    positions = cache.rope_positions(step)[start : start + valid]
    compressed = attention._fork_compress(hidden, cache, step, rope, row_sharded=True)
    q_lora, kv_pre = attention.project_qkv(
        hidden if hidden_fp8 is None else hidden_fp8, hidden_scale
    )
    qr, qr_scale, kv_normed = attention.qk_norm(q_lora, kv_pre)
    # The indexer reads the same local qr that the attention consumes.
    if attention.indexer is not None:
        index_query = attention.indexer.project(
            hidden,
            qr,
            qr_scale,
            cache,
            step,
            rope,
            row_sharded=True,
        )
        assert not compressed  # this deployment excludes side streams
        attention.indexer.score(*index_query, cache, step, row_sharded=True)
    heads = attention.heads * group.world_size
    query = attention.wq_b_owned(qr, x_scale=qr_scale).view(
        capacity, heads, attention.head_dim
    )[:valid]
    # All ranks retain their native replicated KV cache. Q never leaves its owner.
    kv_normed = gather_norm(kv_normed, step.width)
    kv, window_kv = cache.rope_quant_window(
        attention.spec.layer_id, None, kv_normed, rope, step
    )
    cache.rotate_query(query, rope, positions)
    prefix, pptr, extend, eptr = cache.attention_indices(attention.spec, step)
    pptr = pptr[start : start + valid + 1]
    eptr = eptr[start : start + valid + 1]
    if cache.packed:
        # packed_prefill assumes a local, capacity-declared flat CSR index list.
        raise ValueError("owner attention prototype currently requires BF16 KV")
    output = sparse_attn_v4_paged_2src(
        query,
        cache.pool,
        prefix,
        pptr,
        kv.flatten(0, 1),
        extend,
        eptr,
        attention.sink_owned,
        attention.softmax_scale,
        out=query,
    )
    cache.write_window(attention.spec.layer_id, window_kv, step)
    result = project_out(attention, output, rope, positions)
    if valid < capacity:
        result = torch.cat(
            (result, result.new_zeros((1, capacity - valid, result.shape[-1]))), dim=1
        )
    return result
