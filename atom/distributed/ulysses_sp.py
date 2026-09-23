# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Ulysses sequence parallelism for the native ATOM LLM path.

Ranks own contiguous token chunks outside attention. Two all-to-alls trade
``[T/W, H, D]`` for ``[T, H/W, D]`` around attention, which retains global
sequence metadata. Dense weights are replicated; MoE weights remain sharded.
SP uses AITER's PCP rank dimension and runs with ``-tp 1 -sp W``.
"""

from functools import partial

import torch

from atom.utils import envs

# Keep GPU imports lazy for CPU-only metadata consumers. Set by ModelRunner.
_SP_WORLD_SIZE: int = 1


def set_sp_world_size(world_size: int) -> None:
    """Record the configured SP width (called once per worker at startup)."""
    global _SP_WORLD_SIZE
    _SP_WORLD_SIZE = max(1, int(world_size))


def get_sp_world_size() -> int:
    return _SP_WORLD_SIZE


def sp_is_enabled() -> bool:
    return _SP_WORLD_SIZE > 1


def get_sp_rank() -> int:
    if _SP_WORLD_SIZE <= 1:
        return 0
    from aiter.dist.parallel_state import get_prefill_context_model_parallel_rank

    return get_prefill_context_model_parallel_rank()


def get_sp_group():
    from aiter.dist.parallel_state import get_pcp_group

    return get_pcp_group()


def attn_head_shard_size() -> int:
    """Head-sharding factor used by attention, metadata, and KV-cache geometry."""
    from aiter.dist.parallel_state import get_tp_group

    return get_tp_group().world_size * _SP_WORLD_SIZE


def sp_pad_len(total_tokens: int, sp_size: int | None = None) -> int:
    """Token count rounded up so the sequence splits evenly across the group."""
    if sp_size is None:
        sp_size = _SP_WORLD_SIZE
    if sp_size <= 1:
        return total_tokens
    rem = total_tokens % sp_size
    return total_tokens if rem == 0 else total_tokens + (sp_size - rem)


def sp_tokens_across_ranks(num_tokens: int | None) -> tuple[int, ...] | None:
    """Per-rank token counts used to size routed MoE receive buffers."""
    if _SP_WORLD_SIZE <= 1 or num_tokens is None:
        return None
    return (sp_pad_len(num_tokens) // _SP_WORLD_SIZE,) * _SP_WORLD_SIZE


def sp_local_slice(total_tokens: int) -> slice:
    """This rank's contiguous token range within a padded sequence."""
    local = sp_pad_len(total_tokens) // _SP_WORLD_SIZE
    start = get_sp_rank() * local
    return slice(start, start + local)


def sp_split_tokens(x: torch.Tensor) -> torch.Tensor:
    """Take this rank's contiguous chunk, padding with zeros along dim 0.

    Attention and the final gather discard padding before returning results.
    """
    if _SP_WORLD_SIZE <= 1:
        return x
    total = x.shape[0]
    padded = sp_pad_len(total)
    if padded != total:
        pad = x.new_zeros(padded - total, *x.shape[1:])
        x = torch.cat([x, pad], dim=0)
    return x[sp_local_slice(total)].contiguous()


# Small payloads can benefit from the registered-buffer all-gather. Large
# prefills use all-to-all to avoid receiving unused heads.
_ALLGATHER_MAX_PAYLOAD_BYTES = 2 * 1024 * 1024


def _custom_gather_ok(payload: torch.Tensor, group) -> bool:
    """Whether AITER's registered-buffer all-gather supports this payload.

    All predicates must agree across ranks: choosing different collectives
    deadlocks. Padded SP shards have equal shapes on every rank.
    """
    if not envs.ATOM_USE_CUSTOM_ALL_GATHER:
        return False
    ca_comm = getattr(getattr(group, "device_communicator", None), "ca_comm", None)
    if ca_comm is None or ca_comm.disabled:
        return False
    return bool(ca_comm.should_custom_ag(payload))


def _custom_reduce_scatter_ok(payload: torch.Tensor, group) -> bool:
    """Rank-invariant gate for AITER's registered-buffer reduce-scatter."""
    ca_comm = getattr(getattr(group, "device_communicator", None), "ca_comm", None)
    if ca_comm is None or ca_comm.disabled:
        return False
    return bool(ca_comm.should_custom_rs(payload, 0))


def _prefer_all_gather(payload: torch.Tensor, group) -> bool:
    """Use registered-buffer all-gather only for small supported payloads."""
    if payload.numel() * payload.element_size() > _ALLGATHER_MAX_PAYLOAD_BYTES:
        return False
    return _custom_gather_ok(payload, group)


def _all_gather_tokens(x: torch.Tensor) -> torch.Tensor:
    """Gather token shards, preferring graph-compatible AITER transports."""
    x = x.contiguous()
    group = get_sp_group()
    if _custom_gather_ok(x, group):
        return group.all_gather(x, use_custom=True, dim=0)
    pynccl = getattr(getattr(group, "device_communicator", None), "pynccl_comm", None)
    if pynccl is not None and not pynccl.disabled:
        out = x.new_empty((x.shape[0] * _SP_WORLD_SIZE, *x.shape[1:]))
        pynccl.all_gather(out, x)
        return out
    return group.all_gather(x, dim=0)


def sp_gather_tokens(x: torch.Tensor, total_tokens: int) -> torch.Tensor:
    """Inverse of `sp_split_tokens`: all-gather the chunks and drop the pad."""
    if _SP_WORLD_SIZE <= 1:
        return x
    return _all_gather_tokens(x)[:total_tokens]


def ulysses_gather_heads(x: torch.Tensor) -> torch.Tensor:
    """``[S, H/W*D] -> [S/W, H*D]``: give up the sequence, regain all heads."""
    w = _SP_WORLD_SIZE
    if w <= 1:
        return x
    x = x.reshape(x.shape[0], -1)
    s_local, width = x.shape[0] // w, x.shape[1]
    group = get_sp_group()

    # AITER's last-dimension gather requires 16-byte alignment.
    if width * x.element_size() % 16 == 0 and _prefer_all_gather(x, group):
        start = group.rank_in_group * s_local
        return group.all_gather(x, use_custom=True, dim=1)[start : start + s_local]

    return _swap_heads(x, s_local, width)


def _swap_heads(x, s_local, width):
    """``[w*s_local, width] -> [s_local, w*width]`` by all-to-all."""
    from atom.distributed.sp_kernels import all_to_all_into

    send = x.reshape(_SP_WORLD_SIZE, s_local, width).contiguous()
    recv = torch.empty_like(send)
    all_to_all_into(recv, send, get_sp_group())
    # recv[i] is head-group i for our token chunk -> concatenate on heads.
    return recv.permute(1, 0, 2).reshape(s_local, _SP_WORLD_SIZE * width)


def sp_moe_gather(x: torch.Tensor) -> torch.Tensor:
    """Gather token shards so each rank can evaluate its local experts."""
    if _SP_WORLD_SIZE <= 1:
        return x
    return _all_gather_tokens(x)


def sp_moe_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    """Sum expert contributions in the input dtype and keep this token shard."""
    if _SP_WORLD_SIZE <= 1:
        return x
    x = x.contiguous()
    group = get_sp_group()
    # The generic communicator allocates a second output and copies it even
    # for dim=0. PyNccl can write the final token shard directly.
    if not _custom_reduce_scatter_ok(x, group):
        pynccl = getattr(
            getattr(group, "device_communicator", None), "pynccl_comm", None
        )
        if pynccl is not None and not pynccl.disabled:
            out = x.new_empty((x.shape[0] // _SP_WORLD_SIZE, *x.shape[1:]))
            pynccl.reduce_scatter(out, x)
            return out
    return group.reduce_scatter(x, dim=0)


_SCATTER = 0  # zero means one head shard per rank
_REPLICATE = 1  # one shard, replicated on every rank
# Other positive values specify how many head shards a field has. For GQA
# with fewer KV heads than ranks, consecutive query-head owners share a shard.


def _exchange(send):
    """All-to-all a ``[W, S/W, width]`` send buffer into global token order."""
    from atom.distributed.sp_kernels import all_to_all_into

    recv = torch.empty_like(send)
    all_to_all_into(recv, send, get_sp_group())
    # Rank-major receive order is global token order for contiguous shards.
    return recv.reshape(recv.shape[0] * recv.shape[1], recv.shape[2])


def _exchange_tensors(tensors, shards=None):
    """Exchange separate fields into one packed tensor with all sequence rows."""
    w = _SP_WORLD_SIZE
    shards = shards or [w] * len(tensors)
    widths = [t.shape[-1] // n for t, n in zip(tensors, shards)]
    proto = tensors[0]
    # Copy directly into the send buffer, replicating shared KV head shards.
    send = proto.new_empty((w, proto.shape[0], sum(widths)))
    col = 0
    for tensor, width, n in zip(tensors, widths, shards):
        field = tensor.view(tensor.shape[0], n, width).permute(1, 0, 2)
        send[:, :, col : col + width].view(n, w // n, tensor.shape[0], width).copy_(
            field[:, None]
        )
        col += width
    return _exchange(send)


def _exchange_columns(source, spec, owner):
    """Exchange fused fields described by (column offset, width, head shards)."""
    w = _SP_WORLD_SIZE
    group = get_sp_group()
    if not _prefer_all_gather(source, group):
        from atom.distributed.sp_kernels import pack_fields

        return _exchange(pack_fields(source, tuple(spec), w))
    columns = getattr(owner, "_sp_send_columns", None)
    if columns is None or columns.device != source.device:
        # Device aranges also support first use during graph capture.
        arange = partial(torch.arange, device=source.device)
        ranges = []
        for offset, width, mode in spec:
            shards = mode or w
            local = width // shards
            start = offset + (group.rank_in_group // (w // shards)) * local
            ranges.append(arange(local) + start)
        columns = torch.cat(ranges)
        owner._sp_send_columns = columns

    return group.custom_all_gather(source).index_select(1, columns)


def ulysses_attention(attn, query, key, value, positions, q_scale, qkv):
    """Run this rank's head slice over the full sequence, then restore tokens.

    Called inside the attention custom op, keeping collectives and padding
    slices opaque to Dynamo.
    """
    w = _SP_WORLD_SIZE
    kv_shards = min(w, key.shape[-1] // attn.head_dim)
    if qkv is None:
        packed = _exchange_tensors([query, key, value], [w, kv_shards, kv_shards])
    else:
        # MiniMax-M3's fused kernel also reads indexer fields by column offset.
        q_size, kv_size = query.shape[-1], key.shape[-1]
        spec = [
            (0, q_size, _SCATTER),
            (q_size, kv_size, kv_shards),
            (q_size + kv_size, kv_size, kv_shards),
        ]
        if any(
            t.untyped_storage().data_ptr() != qkv.untyped_storage().data_ptr()
            or t.storage_offset() != qkv.storage_offset() + offset
            for t, (offset, _, _) in zip((query, key, value), spec)
        ):
            raise ValueError(
                "Ulysses SP: q/k/v must be the leading column ranges of the "
                "fused qkv tensor when one is supplied."
            )
        tail = qkv.shape[-1] - (q_size + 2 * kv_size)
        if tail > 0:
            idx_dim = getattr(attn.impl, "index_head_dim", 0)
            if not idx_dim:
                raise ValueError(
                    f"Ulysses SP: fused qkv has {tail} trailing columns but "
                    f"{type(attn.impl).__name__} declares no indexer layout."
                )
            index_q_width = tail - idx_dim
            index_q_start = qkv.shape[-1] - (index_q_width + idx_dim)
            index_shards = min(w, index_q_width // idx_dim)
            spec.append((index_q_start, index_q_width, index_shards))
            spec.append((index_q_start + index_q_width, idx_dim, _REPLICATE))
        packed = _exchange_columns(qkv, spec, attn)

    # Global positions and attention metadata exclude the divisibility pad.
    real_tokens = positions.shape[-1]
    padded_tokens = packed.shape[0]
    if real_tokens < padded_tokens:
        packed = packed[:real_tokens]

    # Attention accepts column views, so avoid another Q/K/V copy.
    q_width = query.shape[-1] // w
    kv_width = key.shape[-1] // kv_shards
    out = attn.impl.forward(
        query=packed[:, :q_width],
        key=packed[:, q_width : q_width + kv_width],
        value=packed[:, q_width + kv_width : q_width + 2 * kv_width],
        position=positions,
        q_scale=q_scale,
        # Preserve the caller's choice of fused or separate Q/K/V kernels.
        qkv=packed if qkv is not None else None,
    )

    if real_tokens < padded_tokens:
        out = torch.cat(
            [out, out.new_zeros(padded_tokens - real_tokens, *out.shape[1:])], dim=0
        )
    return ulysses_gather_heads(out)
