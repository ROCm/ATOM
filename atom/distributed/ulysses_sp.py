# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Ulysses sequence parallelism for the native ATOM LLM path.

Ranks own contiguous token chunks outside attention. Two all-to-alls trade
``[T/W, H, D]`` for ``[T, H/W, D]`` around attention
(`atom.model_ops.sp_attention`), which retains global sequence metadata.
Dense weights are replicated; MoE weights remain sharded. SP uses AITER's PCP
rank dimension and runs with ``-tp 1 -sp W``.
"""

import torch

from atom.config import get_current_atom_config

# AITER is imported where used: `Config` imports this module on CPU.

# Per-rank payloads up to this are all-gathered rather than sent all-to-all.
# Measured on SP4 the two cross at ~3.7 MB; 2 MiB keeps a margin.
ALLGATHER_MAX_PAYLOAD_BYTES = 2 * 1024 * 1024


def get_sp_world_size() -> int:
    return get_current_atom_config().sequence_parallel_size


def sp_is_enabled() -> bool:
    return get_sp_world_size() > 1


def check_aiter_sp_api() -> None:
    """Refuse an AITER without ROCm/aiter#7346 at startup, not at first use."""
    from aiter.dist.device_communicators import (
        communicator_cuda,
        custom_all_reduce,
    )
    from aiter.ops import communication

    needs = {
        communicator_cuda.CudaCommunicator: ["all_to_all"],
        custom_all_reduce.CustomAllreduce: ["registered_input_view", "head_exchange"],
        communication: ["MegaMoeSetupError"],
    }
    missing = [
        name for api, names in needs.items() for name in names if not hasattr(api, name)
    ]
    if missing:
        raise RuntimeError(
            "Ulysses SP needs AITER with ROCm/aiter#7346; this AITER lacks "
            + ", ".join(missing)
        )


def attn_head_shard_size() -> int:
    """Head-sharding factor used by attention, metadata, and KV-cache geometry."""
    from aiter.dist.parallel_state import get_tp_group

    return get_tp_group().world_size * get_sp_world_size()


def sp_head_shards(heads: int, shards: int) -> tuple[int, int]:
    """(heads one rank holds, distinct head shards) of ``heads`` split over
    ``shards`` ranks; when fewer, each head replicates to consecutive ranks."""
    if heads % shards and shards % heads:
        raise ValueError(
            f"{heads} heads neither shard nor replicate evenly over {shards} ranks."
        )
    return max(1, heads // shards), min(heads, shards)


def local_num_kv_heads(num_kv_heads: int) -> int:
    """KV heads one rank holds over the attention head shards."""
    return sp_head_shards(num_kv_heads, attn_head_shard_size())[0]


def check_sp_attention_layer(name, sinks, alibi_slopes, rotary_emb):
    """Refuse the per-layer features the config cannot see."""
    if sinks is not None or alibi_slopes is not None:
        raise ValueError("Ulysses SP does not support attention sinks or ALiBi slopes.")
    # The model body sees only this rank's token shard, so RoPE must run in the
    # impl on the exchanged sequence with the global positions.
    if rotary_emb is None:
        raise ValueError(
            f"Ulysses SP requires RoPE inside the attention impl; {name} has none."
        )


def sp_local_len(total_tokens: int) -> int:
    """Rows of a ``total_tokens`` sequence each rank owns."""
    return -(-total_tokens // get_sp_world_size())


def sp_local_slice(total_tokens: int) -> slice:
    """This rank's contiguous token range. Trailing ranks' may run past
    ``total_tokens``, some entirely; attention and the final gather drop those
    rows."""
    from aiter.dist.parallel_state import get_pcp_group

    local = sp_local_len(total_tokens)
    start = get_pcp_group().rank_in_group * local
    return slice(start, start + local)


def sp_gather_tokens(x: torch.Tensor, total_tokens: int) -> torch.Tensor:
    """All-gather every rank's token range and drop the rows past the sequence."""
    from aiter.dist.parallel_state import get_pcp_group

    if not sp_is_enabled():
        return x
    communicator = get_pcp_group().device_communicator
    return communicator.all_gatherv(x.contiguous())[:total_tokens]


def sp_gather_rows(x: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """``sp_gather_tokens(x, T)[rows]``, moving only the selected rows.

    Every rank sends one row per index, and the copy from its owner is kept.
    """
    from aiter.dist.parallel_state import get_pcp_group

    if not sp_is_enabled():
        return x[rows].contiguous()
    group = get_pcp_group()
    local, n = x.shape[0], rows.numel()
    picked = x[(rows - group.rank_in_group * local).clamp(0, local - 1)]
    owner = rows.long() // local
    gathered = group.device_communicator.all_gatherv(picked)
    return gathered[owner * n + torch.arange(n, device=x.device)]


def all_to_all_blocks(send, group):
    """All-to-all a ``[W, rows, width]`` buffer: block i goes to rank i, and
    block i of the result came from rank i."""
    recv = torch.empty_like(send)
    group.device_communicator.all_to_all(recv, send)
    return recv


def send_columns(fields, device) -> torch.Tensor:
    """``[W, columns]``: the fused-qkv columns each rank receives, in order.

    ``fields`` are the (heads, head dim) column ranges of the fused qkv; each
    shards or replicates its heads as `sp_head_shards` says.
    """
    w = get_sp_world_size()
    columns, offset = [], 0
    for heads, dim in fields:
        local, shards = sp_head_shards(heads, w)
        first = torch.arange(w) // (w // shards) * local * dim + offset
        columns.append(first[:, None] + torch.arange(local * dim))
        offset += heads * dim
    return torch.cat(columns, 1).to(device=device, dtype=torch.int32)


def exchange_columns(source, group, columns):
    """Give each rank its ``columns`` row of ``source`` for every token."""
    if source.numel() * source.element_size() > ALLGATHER_MAX_PAYLOAD_BYTES:
        from atom.distributed.sp_kernels import pack_columns

        # Rank-major receive order is global token order for contiguous shards.
        return all_to_all_blocks(pack_columns(source, columns), group).flatten(0, 1)
    gathered = group.device_communicator.all_gather(source.contiguous(), 0)
    return gathered.index_select(1, columns[group.rank_in_group])
