# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Ulysses sequence parallelism for the native ATOM LLM path.

Ranks own contiguous token chunks outside attention. Two all-to-alls trade
``[T/W, H, D]`` for ``[T, H/W, D]`` around attention
(`atom.model_ops.sp_attention`), which retains global sequence metadata.
Dense weights are replicated; MoE weights remain sharded. SP uses AITER's PCP
rank dimension and runs with ``-tp 1 -sp W``.
"""

from functools import partial

import torch

from atom.distributed.token_transport import (
    ALLGATHER_MAX_PAYLOAD_BYTES,
    custom_gather_ok,
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


def local_num_kv_heads(num_kv_heads: int) -> int:
    """KV heads one rank holds: its share of ``num_kv_heads``, or a single
    replicated head when the heads are fewer than the attention head shards."""
    shards = attn_head_shard_size()
    if num_kv_heads % shards and shards % num_kv_heads:
        raise ValueError(
            f"{num_kv_heads} KV heads neither shard nor replicate evenly over "
            f"{shards} attention head shards."
        )
    return max(1, num_kv_heads // shards)


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
    communicator = get_pcp_group().device_communicator
    return communicator.all_gatherv(x.contiguous())[:total_tokens]


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
    gathered = group.device_communicator.all_gatherv(picked)
    return gathered[owner * n + torch.arange(n, device=x.device)]


SCATTER = 0  # zero means one head shard per rank
REPLICATE = 1  # one shard, replicated on every rank
# Other positive values specify how many head shards a field has. For GQA
# with fewer KV heads than ranks, consecutive query-head owners share a shard.


def _exchange(send, group):
    """All-to-all a ``[W, S/W, width]`` send buffer into global token order."""
    recv = torch.empty_like(send)
    group.device_communicator.all_to_all(recv, send)
    # Rank-major receive order is global token order for contiguous shards.
    return recv.reshape(recv.shape[0] * recv.shape[1], recv.shape[2])


def exchange_tensors(tensors, shards, group):
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


def send_columns(spec, group, device) -> torch.Tensor:
    """This rank's source columns of the fused fields in ``spec``, packed."""
    w = _SP_WORLD_SIZE
    # Device aranges also support first use during graph capture.
    arange = partial(torch.arange, device=device)
    ranges = []
    for offset, width, mode in spec:
        shards = mode or w
        local = width // shards
        start = offset + (group.rank_in_group // (w // shards)) * local
        ranges.append(arange(local) + start)
    return torch.cat(ranges)


def exchange_columns(source, spec, group, columns):
    """Exchange fused fields described by (column offset, width, head shards).

    ``columns`` is ``send_columns(spec, ...)``, read by the all-gather path.
    """
    if not custom_gather_ok(source, group, ALLGATHER_MAX_PAYLOAD_BYTES):
        from atom.distributed.sp_kernels import pack_fields

        return _exchange(pack_fields(source, tuple(spec), _SP_WORLD_SIZE), group)
    return group.custom_all_gather(source).index_select(1, columns)
