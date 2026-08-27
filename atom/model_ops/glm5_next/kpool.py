# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""kpool: pooled indexer-key compression for GLM-5.3-Flash.

The sparse indexer does not score every token. Each group of ``index_kpool``
consecutive tokens is compressed into ONE cached entry, top-k runs at that pool
granularity (``index_topk // index_kpool`` pools), and each selected pool then
expands back to the ``index_kpool`` token positions it covers. The trailing
incomplete pool ("the tail") is always selected, so the newest tokens are never
dropped.

The compression is, per pool ``p`` and per dimension ``d``:

    w[slot, d] = softmax over slot of (gate[p, slot, d] + ape[slot, d])
    pooled[p, d] = sum_slot w[slot, d] * k[p, slot, d]

Note the softmax runs **over the pool's slots, independently per dimension** --
it is not a scalar per-slot gate. Getting that wrong produces plausible-looking
values and a quiet accuracy loss, which is why `pool_compress_ref` below exists
as the oracle the Triton kernel is tested against.

``pooled`` is then Hadamard-128 rotated and quantized to FP8 with a ue8m0
(power-of-two) scale, matching the basis the cached keys are scored in.

Ported from vLLM PR #53906 (`vllm/models/glm5next/nvidia/ops/kpool_compress.py`).
"""

from __future__ import annotations

import torch

# --------------------------------------------------------------------------
# Reference implementations (the correctness oracle; not the fast path)
# --------------------------------------------------------------------------


def pool_compress_ref(
    k: torch.Tensor,
    gate: torch.Tensor,
    ape: torch.Tensor,
) -> torch.Tensor:
    """Softmax-pool whole pools of indexer keys.

    Args:
        k:    ``[num_pools, pool, head_dim]`` layer-normed indexer keys.
        gate: ``[num_pools, pool, head_dim]`` per-token gate scores.
        ape:  ``[pool, head_dim]`` learned per-slot bias.

    Returns:
        ``[num_pools, head_dim]`` pooled keys, in fp32.
    """
    scores = gate.float() + ape.float().unsqueeze(0)
    # dim=1 is the slot axis: one softmax per (pool, dim) over the pool's slots.
    weights = scores.softmax(dim=1)
    return (weights * k.float()).sum(dim=1)


def hadamard128_ref(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized Walsh-Hadamard transform over a 128-wide last dim."""
    assert x.shape[-1] == 128, f"expected head_dim 128, got {x.shape[-1]}"
    out = x.float().clone()
    step = 1
    while step < 128:
        view = out.view(*out.shape[:-1], 128 // (2 * step), 2, step)
        a = view[..., 0, :].clone()
        b = view[..., 1, :].clone()
        view[..., 0, :] = a + b
        view[..., 1, :] = a - b
        step *= 2
    return out


def quant_fp8_ue8m0_ref(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-vector absmax FP8-e4m3 quant with a power-of-two (ue8m0) scale."""
    fp8_max = 448.0
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(absmax / fp8_max)))
    q = (x / scale).clamp(-fp8_max, fp8_max)
    return q, scale.squeeze(-1)


def compress_pools_ref(
    k: torch.Tensor, gate: torch.Tensor, ape: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full compression: softmax-pool -> Hadamard-128 -> FP8/ue8m0."""
    pooled = pool_compress_ref(k, gate, ape)
    # bf16 round-trips mirror the fused kernel's intermediate precision.
    pooled = pooled.to(torch.bfloat16).float()
    rotated = hadamard128_ref(pooled).to(torch.bfloat16).float()
    return quant_fp8_ue8m0_ref(rotated)


# --------------------------------------------------------------------------
# Pool -> token expansion
# --------------------------------------------------------------------------


def history_group_budget_for_topk(topk: int, pool_size: int) -> int:
    """How many pools to select so expanding them yields ``topk`` tokens."""
    assert topk % pool_size == 0, (topk, pool_size)
    return topk // pool_size


def expand_pools_to_tokens(
    pool_ids: torch.Tensor,
    pool_valid: torch.Tensor,
    topk: int,
    pool_size: int,
) -> torch.Tensor:
    """Expand selected pool ids to token ids.

    Args:
        pool_ids:   ``[rows, topk // pool_size]`` selected pool indices.
        pool_valid: same shape, False where the slot is padding.

    Returns:
        ``[rows, topk]`` token indices, ``-1`` where invalid.
    """
    assert pool_ids.shape[1] == history_group_budget_for_topk(topk, pool_size)
    offsets = torch.arange(pool_size, device=pool_ids.device, dtype=torch.int64)
    token_ids = pool_ids.to(torch.int64).unsqueeze(-1) * pool_size + offsets
    token_ids = token_ids.reshape(pool_ids.shape[0], topk)
    valid = (
        pool_valid.unsqueeze(-1)
        .expand(-1, -1, pool_size)
        .reshape(pool_ids.shape[0], topk)
    )
    return torch.where(
        valid,
        token_ids.to(torch.int32),
        torch.full_like(token_ids, -1, dtype=torch.int32),
    )


def append_tail_to_topk(
    topk_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Append the trailing incomplete pool's tokens.

    ``index_kpool_always_select_tail``: the in-progress pool is never compressed
    yet, so its raw tokens are appended after the expanded history rather than
    being scored.
    """
    tail = pool_size - 1
    if tail == 0:
        return topk_tokens
    rows = topk_tokens.shape[0]
    device = topk_tokens.device
    pooled_end = (seq_lens // pool_size) * pool_size  # first tail token
    offs = torch.arange(tail, device=device, dtype=torch.int64)
    tail_ids = pooled_end.to(torch.int64).unsqueeze(1) + offs.unsqueeze(0)
    tail_valid = tail_ids < seq_lens.to(torch.int64).unsqueeze(1)
    tail_ids = torch.where(
        tail_valid,
        tail_ids.to(torch.int32),
        torch.full_like(tail_ids, -1, dtype=torch.int32),
    )
    return torch.cat([topk_tokens, tail_ids.view(rows, tail)], dim=1)
