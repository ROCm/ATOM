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

from atom.distributed.token_transport import (
    ALLGATHER_MAX_PAYLOAD_BYTES,
    custom_gather_ok,
    gather_heads,
    gather_tokens,
)

# AITER is imported where used: `forward_context` imports this module on CPU.
# Set by ModelRunner.
_SP_WORLD_SIZE: int = 1


def set_sp_world_size(world_size: int) -> None:
    """Record the configured SP width (called once per worker at startup)."""
    global _SP_WORLD_SIZE
    _SP_WORLD_SIZE = max(1, int(world_size))


def get_sp_world_size() -> int:
    return _SP_WORLD_SIZE


def sp_is_enabled() -> bool:
    return _SP_WORLD_SIZE > 1


def attn_head_shard_size() -> int:
    """Head-sharding factor used by attention, metadata, and KV-cache geometry."""
    from aiter.dist.parallel_state import get_tp_group

    return get_tp_group().world_size * _SP_WORLD_SIZE


def check_sp_attention_layer(sp, name, sinks, alibi_slopes, rotary_emb, index_heads):
    """Refuse what the config cannot see: per-layer features and indexer heads."""
    if sinks is not None or alibi_slopes is not None:
        raise ValueError("Ulysses SP does not support attention sinks or ALiBi slopes.")
    # The model body sees only this rank's token shard, so RoPE must run in the
    # impl on the exchanged sequence with the global positions.
    if rotary_emb is None:
        raise ValueError(
            f"Ulysses SP requires RoPE inside the attention impl; {name} has none."
        )
    # Indexer query heads shard or replicate like KV heads.
    if index_heads % sp and sp % index_heads:
        raise ValueError(
            f"--sequence-parallel-size {sp} cannot shard the {index_heads} "
            f"indexer query heads of {name}."
        )


def sp_local_len(total_tokens: int) -> int:
    """Rows of a ``total_tokens`` sequence each rank owns."""
    return -(-total_tokens // _SP_WORLD_SIZE)


def sp_local_slice(total_tokens: int) -> slice:
    """This rank's contiguous token range. The last rank's may run past
    ``total_tokens``; attention and the final gather drop those rows."""
    from aiter.dist.parallel_state import get_pcp_group

    local = sp_local_len(total_tokens)
    start = get_pcp_group().rank_in_group * local
    return slice(start, start + local)


def sp_gather_tokens(x: torch.Tensor, total_tokens: int) -> torch.Tensor:
    """All-gather every rank's token range and drop the rows past the sequence."""
    from aiter.dist.parallel_state import get_pcp_group

    if _SP_WORLD_SIZE <= 1:
        return x
    return gather_tokens(x, get_pcp_group())[:total_tokens]


def sp_gather_rows(x: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """``sp_gather_tokens(x, T)[rows]``, moving only the selected rows.

    Every rank sends one row per index, and the copy from its owner is kept.
    """
    from aiter.dist.parallel_state import get_pcp_group

    if _SP_WORLD_SIZE <= 1:
        return x[rows].contiguous()
    group = get_pcp_group()
    local, n = x.shape[0], rows.numel()
    picked = x[(rows - group.rank_in_group * local).clamp(0, local - 1)]
    owner = rows.long() // local
    gathered = gather_tokens(picked, group)
    return gathered[owner * n + torch.arange(n, device=x.device)]


_SCATTER = 0  # zero means one head shard per rank
_REPLICATE = 1  # one shard, replicated on every rank
# Other positive values specify how many head shards a field has. For GQA
# with fewer KV heads than ranks, consecutive query-head owners share a shard.


def _exchange(send, group):
    """All-to-all a ``[W, S/W, width]`` send buffer into global token order."""
    from atom.distributed.sp_kernels import all_to_all_into

    recv = torch.empty_like(send)
    all_to_all_into(recv, send, group)
    # Rank-major receive order is global token order for contiguous shards.
    return recv.reshape(recv.shape[0] * recv.shape[1], recv.shape[2])


def _exchange_tensors(tensors, shards, group):
    """Exchange separate fields into one packed tensor with all sequence rows."""
    w = _SP_WORLD_SIZE
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
    return _exchange(send, group)


def _exchange_columns(source, spec, owner, group):
    """Exchange fused fields described by (column offset, width, head shards)."""
    w = _SP_WORLD_SIZE
    if not custom_gather_ok(source, group, ALLGATHER_MAX_PAYLOAD_BYTES):
        from atom.distributed.sp_kernels import pack_fields

        return _exchange(pack_fields(source, tuple(spec), w), group)
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


def ulysses_attention(
    attn, query, key, value, positions, q_scale, qkv, output_transform=None
):
    """Run this rank's head slice over the full sequence, then restore tokens.

    Called inside the attention custom op, keeping collectives and padding
    slices opaque to Dynamo.
    """
    from aiter.dist.parallel_state import get_pcp_group

    w = _SP_WORLD_SIZE
    group = get_pcp_group()
    kv_shards = min(w, key.shape[-1] // attn.head_dim)
    if qkv is None:
        packed = _exchange_tensors(
            [query, key, value], [w, kv_shards, kv_shards], group
        )
    else:
        # Fused kernels may also read indexer fields by column offset.
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
        packed = _exchange_columns(qkv, spec, attn, group)

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
    if output_transform is not None:
        return output_transform(out)
    return gather_heads(out, group)
