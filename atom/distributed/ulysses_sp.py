# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Ulysses sequence parallelism (SP) for the native ATOM LLM path.

Ulysses shards the token sequence across the SP group for every layer except
attention, then trades sequence for heads around attention itself::

    linear/MoE:  [T/W, H,   D]      weights replicated (dense) or EP-sharded
                    | all-to-all
    attention:   [T,   H/W, D]      full sequence, this rank's head slice
                    | all-to-all
    linear/MoE:  [T/W, H,   D]

Against tensor parallelism at the same GPU count this replaces two per-layer
all-reduces of the full hidden state with two all-to-alls that move
``(W-1)/W * T * H * D / W`` elements, at the cost of replicating the attention
and dense weights on every rank. Routed experts stay sharded via EP, so the
memory delta is bounded by the (small) dense side of the model.

The split is contiguous, not round-robin as in PCP: attention is head-sharded
and still sees every token, so contiguous chunks carry no causal-mask
imbalance, and they make the all-to-all a plain equal-split exchange with no
re-ranging on either side.

The SP group rides on the PCP dimension of aiter's rank grid (world =
dp x pp x pcp x tp), which already spans the right ranks and is already folded
into the EP group. Pure Ulysses therefore runs at ``-tp 1 -sp W``.
"""

from functools import partial

import torch
from aiter.dist.parallel_state import (
    get_pcp_group,
    get_prefill_context_model_parallel_rank,
    get_tp_group,
)

from atom.utils import envs

# Set by Config.__post_init__ so head-count math is available during model
# construction, before the process groups exist.
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
    return get_prefill_context_model_parallel_rank()


def get_sp_group():
    return get_pcp_group()


def attn_head_shard_size() -> int:
    """Ways the attention heads are split: TP shards them, SP shards them again.

    Every per-rank head count (q/kv heads on the impl, the KV pool geometry,
    the metadata builder's head count) must divide by this rather than by the
    TP size alone, or the KV cache will be sized for heads this rank never
    computes.
    """
    return get_tp_group().world_size * _SP_WORLD_SIZE


def sp_pad_len(total_tokens: int, sp_size: int | None = None) -> int:
    """Token count rounded up so the sequence splits evenly across the group."""
    if sp_size is None:
        sp_size = _SP_WORLD_SIZE
    if sp_size <= 1:
        return total_tokens
    rem = total_tokens % sp_size
    return total_tokens if rem == 0 else total_tokens + (sp_size - rem)


def sp_local_slice(total_tokens: int) -> slice:
    """This rank's contiguous token range within a padded sequence."""
    local = sp_pad_len(total_tokens) // _SP_WORLD_SIZE
    start = get_sp_rank() * local
    return slice(start, start + local)


def sp_split_tokens(x: torch.Tensor) -> torch.Tensor:
    """Take this rank's contiguous 1/W chunk along dim 0, padding the tail.

    Padding rows are copies of nothing in particular -- they are dropped again
    by `sp_gather_tokens` and never reach attention (the attention op slices
    back to the real token count before running the kernel), so their contents
    are irrelevant as long as they are finite.
    """
    if _SP_WORLD_SIZE <= 1:
        return x
    total = x.shape[0]
    padded = sp_pad_len(total)
    if padded != total:
        pad = x.new_zeros(padded - total, *x.shape[1:])
        x = torch.cat([x, pad], dim=0)
    return x[sp_local_slice(total)].contiguous()


# Above this the all-to-all's smaller payload wins; below it, sending W times
# the bytes through aiter's registered buffers still beats RCCL's generic
# kernel. Measured at SP4 on this exchange's shape (`bench_sp_exchange.py`,
# per-rank input bytes -> us, all-to-all vs all-gather):
#
#     77K   131 -> 92     3080K  117 -> 88
#   1232K   129 -> 89     3696K   80 -> 82   <- crossover
#   2464K   119 -> 91     4928K   81 -> 105
#
# There is no second crossover further up: the gather sends W times the bytes,
# so it only falls further behind (at 78M, 536 -> 1534), and past the 64 MiB
# registered-buffer ceiling it is unavailable anyway. A 30k prefill split four
# ways lands at 148M, well inside all-to-all's half of the range.
#
# `indexer_cp` puts its own crossover at 512KB, but that is a different payload
# shape and a different pair of kernels; this one runs six times further.
_ALLGATHER_MAX_PAYLOAD_BYTES = 2 * 1024 * 1024


def _custom_gather_ok(payload: torch.Tensor, group) -> bool:
    """Whether aiter's registered-buffer all-gather can carry this payload.

    EVERY TERM MUST BE RANK-INVARIANT. This picks between two collectives, so a
    split decision does not degrade -- it deadlocks, with part of the group in
    aiter's gather and the rest in RCCL's, and no error from either. SP shards
    tokens evenly and pads to the group, so `payload` has the same shape on
    every rank; the rest is fixed when the group is built.
    """
    if not envs.ATOM_USE_CUSTOM_ALL_GATHER:
        return False
    ca_comm = getattr(getattr(group, "device_communicator", None), "ca_comm", None)
    if ca_comm is None or ca_comm.disabled:
        return False
    return bool(ca_comm.should_custom_ag(payload))


def _custom_reduce_scatter_ok(payload: torch.Tensor, group) -> bool:
    """Whether aiter's registered-buffer reduce-scatter can carry this payload.

    Rank-invariant for the reason in `_custom_gather_ok`.
    """
    ca_comm = getattr(getattr(group, "device_communicator", None), "ca_comm", None)
    if ca_comm is None or ca_comm.disabled:
        return False
    return bool(ca_comm.should_custom_rs(payload, 0))


def _quick_all_reduce_ok(payload: torch.Tensor, group) -> bool:
    """Whether aiter's quantized all-reduce can carry this payload.

    Rank-invariant for the reason in `_custom_gather_ok`. Note this kernel
    quantizes (`AITER_QUICK_REDUCE_QUANTIZATION`, INT4 as the servers set it),
    so it is only interchangeable with an exact collective where the model
    already tolerates it -- which is why the only caller is the MoE reduce,
    the same sum TP sends through this kernel.
    """
    qr_comm = getattr(getattr(group, "device_communicator", None), "qr_comm", None)
    if qr_comm is None or qr_comm.disabled:
        return False
    return bool(qr_comm.should_quick_allreduce(payload))


def _prefer_all_gather(payload: torch.Tensor, group) -> bool:
    """Whether to trade this exchange's all-to-all for an all-gather.

    The size test comes first because it is the performance decision; the rest
    is availability, and when availability says no the all-to-all is correct at
    every size.
    """
    if payload.numel() * payload.element_size() > _ALLGATHER_MAX_PAYLOAD_BYTES:
        return False
    return _custom_gather_ok(payload, group)


def _all_gather_tokens(x: torch.Tensor) -> torch.Tensor:
    """All-gather token shards, over aiter's buffers where they can take it.

    `all_gather` defaults to `use_custom=False`, which is RCCL -- slower, and
    its WorkNCCL end event is recorded inside capture but queried later by the
    watchdog thread (see `all_gather_with_padding` for that crash).
    """
    x = x.contiguous()
    group = get_sp_group()
    return group.all_gather(x, use_custom=_custom_gather_ok(x, group), dim=0)


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

    # Concatenating on the head axis puts the heads in the same rank order the
    # all-to-all rebuilds, and leaves our token chunk as a plain row slice --
    # so this transport also skips the transpose the other one cannot avoid.
    # aiter's gather takes the last dim only in 16-byte multiples.
    if width * x.element_size() % 16 == 0 and _prefer_all_gather(x, group):
        start = group.rank_in_group * s_local
        return group.all_gather(x, use_custom=True, dim=1)[start : start + s_local]

    send = x.reshape(w, s_local, width).contiguous()
    recv = torch.empty_like(send)
    torch.distributed.all_to_all_single(recv, send, group=group.device_group)
    # recv[i] is head-group i for our token chunk -> concatenate on heads.
    return recv.permute(1, 0, 2).reshape(s_local, w * width)


def sp_moe_gather(x: torch.Tensor) -> torch.Tensor:
    """Collect the group's token shards ahead of an expert-parallel MoE.

    Expert parallelism shards the experts, so a token can only be served by the
    rank that owns its expert -- but under SP each rank holds a different token
    shard, so without this a token routed off-rank would contribute nothing at
    all (silently: the kernels just skip non-local expert ids).

    Gathering to the full sequence and reducing back is the same traffic a TP
    MoE's all-reduce already costs, and at these EP widths it is also what a
    dispatch/combine all-to-all would move: with top_k of the experts spread
    over sp ranks, nearly every token has work on nearly every rank anyway.
    """
    if _SP_WORLD_SIZE <= 1:
        return x
    return _all_gather_tokens(x)


def sp_moe_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    """Sum every rank's expert contributions and keep this rank's token shard.

    Reduce-scatter is the shape that fits, but aiter only accelerates it under
    the 64 MiB registered-buffer ceiling and a prefill MoE output is several
    hundred MB, so it lands on RCCL's generic kernel. All-reduce over the same
    tensor has a quantized kernel with a 2 GB ceiling -- the one TP's own MoE
    reduce already runs on -- and the scatter half is then a free row slice.
    Paying for the rows we discard still wins: at 30k tokens x 6144, 1921us on
    RCCL's reduce-scatter against 1263us this way.
    """
    if _SP_WORLD_SIZE <= 1:
        return x
    x = x.contiguous()
    group = get_sp_group()
    if not _custom_reduce_scatter_ok(x, group) and _quick_all_reduce_ok(x, group):
        return group.all_reduce(x)[sp_local_slice(x.shape[0])]
    return group.reduce_scatter(x, dim=0)


_SCATTER = 0  # split this field's heads across the group
_REPLICATE = 1  # no head axis to split: every rank needs the whole thing


def _exchange(send):
    """All-to-all a ``[W, S/W, width]`` send buffer into global token order."""
    recv = torch.empty_like(send)
    torch.distributed.all_to_all_single(recv, send, group=get_sp_group().device_group)
    # recv[i] holds rank i's tokens for our slice; ranks own contiguous token
    # chunks, so concatenating on rank rebuilds global token order.
    return recv.reshape(recv.shape[0] * recv.shape[1], recv.shape[2])


def _exchange_tensors(tensors):
    """Trade tokens for heads on several tensors in a single all-to-all.

    Each arrives as ``[S/W, width]`` on this rank's token chunk; the result is
    one packed ``[S, sum(width/W)]`` tensor on this rank's head slice, in the
    order given. Batching them into one collective rather than one each
    matters more than it looks: this runs once per attention layer, and decode
    is latency-bound, not bandwidth-bound.

    This is the path for models that hand q/k/v over as separate tensors; when
    they are ranges of one fused projection, `_exchange_columns` builds the
    same buffer without a copy per tensor.
    """
    w = _SP_WORLD_SIZE
    widths = [t.shape[-1] // w for t in tensors]
    proto = tensors[0]
    # Packed straight into the send buffer rather than cat-then-stack: the
    # obvious spelling of this copies every element twice.
    send = proto.new_empty((w, proto.shape[0], sum(widths)))
    col = 0
    for tensor, width in zip(tensors, widths):
        # Heads are the major axis of the flat width, so destination rank i
        # owns one contiguous column range: [S, w*width] -> [w, S, width].
        send[:, :, col : col + width].copy_(
            tensor.view(tensor.shape[0], w, width).permute(1, 0, 2)
        )
        col += width
    return _exchange(send)


def _exchange_columns(source, spec, owner):
    """Same exchange, for fields that are column ranges of one fused tensor.

    `spec` lists them as ``(column offset, full width, mode)`` into `source`.
    Because they share a tensor, the whole send buffer is one gather along the
    column axis instead of a copy per field -- worth the indirection at 60
    attention layers, where decode pays each launch in latency no matter how
    small the payload. The permutation only depends on the layer's widths, so
    it is built once and kept on the layer.

    A ``_REPLICATE`` field is taken whole for every destination, so its slot
    comes back carrying every rank's tokens -- an all-gather, folded into the
    same exchange for free.
    """
    w = _SP_WORLD_SIZE
    columns = getattr(owner, "_sp_send_columns", None)
    if columns is None or columns.device != source.device:
        # Built from arange rather than a host list so there is no host-to-
        # device copy to capture, in case the first call lands inside a graph.
        arange = partial(torch.arange, device=source.device)
        dest = arange(w).unsqueeze(1)
        ranges = []
        for offset, width, mode in spec:
            if mode is _REPLICATE:
                ranges.append((arange(width) + offset).expand(w, width))
            else:
                local = width // w
                ranges.append(arange(local) + (offset + dest * local))
        columns = torch.cat(ranges, dim=1)
        owner._sp_send_columns = columns

    group = get_sp_group()
    if _prefer_all_gather(source, group):
        # Same result by the other transport: take every rank's tokens whole,
        # then keep only our own columns. Four times the bytes, but through
        # aiter's registered buffers instead of RCCL's generic kernel, and the
        # column pass is the one this path would run anyway.
        return group.custom_all_gather(source).index_select(
            1, columns[group.rank_in_group]
        )

    tokens, local_width = source.shape[0], columns.shape[1]
    send = torch.gather(
        source.unsqueeze(0).expand(w, -1, -1),
        2,
        columns.unsqueeze(1).expand(w, tokens, local_width),
    )
    return _exchange(send)


def ulysses_attention(attn, query, key, value, positions, q_scale, qkv):
    """Run this rank's head slice of `attn` over the whole sequence.

    Q/K/V arrive as this rank's token chunk carrying every head; the
    all-to-all swaps that for every token carrying this rank's head slice,
    which is exactly the shape the unchanged attention kernels and the
    (global, unsharded) attention metadata already expect. The reverse
    exchange puts the output back on this rank's token chunk.

    Called from inside the attention custom op, so the collectives are already
    opaque to Dynamo and the padding slice below never reaches a traced graph.
    """
    w = _SP_WORLD_SIZE
    if qkv is None:
        packed = _exchange_tensors([query, key, value])
    else:
        # Models that fuse the projection hand q/k/v over as column ranges of
        # `qkv`, so describe the exchange against that tensor and gather it in
        # one pass. MiniMax-M3 also packs indexer fields after q/k/v, and its
        # fused qk-norm/RoPE/KV-insert kernel finds them by offset -- so they
        # must ride the same exchange and land in the same order, at the
        # sharded widths the impl re-splits with. index_k is a single head
        # shared by every indexer head, so it has no head axis to trade and
        # replicates instead.
        q_size, kv_size = query.shape[-1], key.shape[-1]
        spec = [
            (0, q_size, _SCATTER),
            (q_size, kv_size, _SCATTER),
            (q_size + kv_size, kv_size, _SCATTER),
        ]
        # The impl re-splits `qkv` at these same offsets, so a caller whose
        # q/k/v are not those column ranges is already broken -- but say so
        # here rather than let the gather quietly send the wrong columns.
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
            index_q_width = attn.impl.index_q_size * w
            index_q_start = qkv.shape[-1] - (index_q_width + idx_dim)
            spec.append((index_q_start, index_q_width, _SCATTER))
            spec.append((index_q_start + index_q_width, idx_dim, _REPLICATE))
        packed = _exchange_columns(qkv, spec, attn)

    # `positions` is passed through unsharded, so it still counts the real
    # tokens; anything past that is the divisibility pad, which the metadata
    # does not describe and the kernels must not see.
    real_tokens = positions.shape[-1]
    padded_tokens = packed.shape[0]
    if real_tokens < padded_tokens:
        packed = packed[:real_tokens]

    # Column views, left non-contiguous on purpose: this is the same thing the
    # models hand the impl unsharded (`qkv.split(...)`), so the kernels already
    # take it, and materializing them would copy q/k/v a second time.
    q_width = query.shape[-1] // w
    kv_width = key.shape[-1] // w
    out = attn.impl.forward(
        query=packed[:, :q_width],
        key=packed[:, q_width : q_width + kv_width],
        value=packed[:, q_width + kv_width : q_width + 2 * kv_width],
        position=positions,
        q_scale=q_scale,
        # Only hand a packed tensor to callers that supplied one: elsewhere its
        # presence would switch the impl onto a fused path the unsharded run
        # never takes.
        qkv=packed if qkv is not None else None,
    )

    if real_tokens < padded_tokens:
        out = torch.cat(
            [out, out.new_zeros(padded_tokens - real_tokens, *out.shape[1:])], dim=0
        )
    return ulysses_gather_heads(out)
