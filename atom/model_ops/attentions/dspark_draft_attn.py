# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Page-table read for the GQA DFlash/DSpark draft block.

The block pass writes its own T rows into the draft pool first
(`KimiK3DFlashAttention._write_block_kv`), so request ``b``'s pool rows are
keys ``[0, kv_lens[b])``, ``kv_lens = context_lens = anchor + 1 + T``, and every
block row attends all of them. Key ``t`` sits at slot
``block_tables[b, t // E] * E + t % E``, where ``E`` is the pool's entry span
(``block_size * dcp_size`` for this draft, which is not in the dcp group). The
read takes exactly that, so the proposer builds no ``kv_indptr``/``kv_indices``
stream for it.

`DSparkDraftUnifiedAttention` is aiter's ``unified_attention`` -- the kernel
`PagedAttentionImpl.paged_attention_unified` already decodes through -- on the
FOLDED query ``[bs, HKV * T * G, D]`` with ``max_seqlen_q = 1``. That kernel
asserts ``causal``, and with one query row per sequence the causal mask admits
every key ``< seqused_k``, which is the block's bidirectional mask exactly. The
fold puts a request's T block rows and its G query heads per KV head on one KV
head, so one KV tile serves all ``T * G`` of them, and the all-decode launch is
the split-KV grid, so a long context is spread over segments instead of walked
by one program. The block rows come back out of the pool with the prefix, so
over an fp8 pool they are read quantized.

Over an fp8 pool the query is cast to the pool's dtype too. That is the
precision SGLang's verify kernel runs this checkpoint at (it casts q to the
pool dtype for the prefix dot), and it is what lets gfx950 take aiter's Gluon
kernel, which needs q and KV in one dtype: about 1.6x faster than the Triton
fallback at bs 32 / 128k context.
"""

from __future__ import annotations

import torch


def fold_block_queries(q: torch.Tensor, bs: int, num_kv_heads: int) -> torch.Tensor:
    """``[bs * T, HQ, D]`` -> ``[bs, HKV * T * G, D]``, KV-head-major.

    `unified_attention` maps query head ``h`` to KV head
    ``h // num_queries_per_kv``, so every row sharing a KV head has to be
    contiguous in the head axis: ``(kv_head, t, g)``. Query head ``h`` of a
    GQA layer is ``(h // G, h % G)``.
    """
    rows, hq, d = q.shape
    t = rows // bs
    g = hq // num_kv_heads
    return (
        q.view(bs, t, num_kv_heads, g, d)
        .permute(0, 2, 1, 3, 4)
        .reshape(bs, num_kv_heads * t * g, d)
    )


def unfold_block_output(
    of: torch.Tensor, o: torch.Tensor, num_kv_heads: int
) -> torch.Tensor:
    """Inverse of `fold_block_queries`, written into ``o`` ``[bs * T, HQ, D]``."""
    bs = of.shape[0]
    rows, hq, d = o.shape
    t = rows // bs
    g = hq // num_kv_heads
    o.view(bs, t, num_kv_heads, g, d).copy_(
        of.view(bs, num_kv_heads, t, g, d).permute(0, 2, 1, 3, 4)
    )
    return o


class DSparkDraftUnifiedAttention:
    """aiter ``unified_attention`` (3D split-KV) on the folded draft block."""

    def __init__(self, device, max_bs: int = 0):
        from aiter.ops.triton.attention.unified_attention import unified_attention

        self.device = device
        self._ua = unified_attention
        # q's descale when it is cast to an fp8 pool's dtype: the cast is a
        # plain one, so 1.0. Built here so no replay allocates it.
        self._q_descale = torch.ones(1, dtype=torch.float32, device=device)
        # One query row per sequence: `cu_seqlens_q` is a constant ramp. Built
        # here, ahead of any capture, for the engine's widest batch, so a
        # replay slices storage the recording saw rather than allocating.
        self._cu_seqlens_q = (
            torch.arange(max_bs + 1, dtype=torch.int32, device=device)
            if max_bs > 0
            else None
        )

    def _cu_seqlens(self, bs: int) -> torch.Tensor:
        if self._cu_seqlens_q is None or self._cu_seqlens_q.shape[0] < bs + 1:
            # No ceiling was given, so only an eager caller lands here.
            self._cu_seqlens_q = torch.arange(
                bs + 1, dtype=torch.int32, device=self.device
            )
        return self._cu_seqlens_q[: bs + 1]

    def __call__(
        self,
        q,  # [bs * T, HQ, D]
        k_cache,  # [entries, E, HKV, D] FLASH layout
        v_cache,  # [entries, E, HKV, D]
        block_tables,  # [bs, W] entry ids
        kv_lens,  # [bs] keys each request attends, block rows included
        o,  # [bs * T, HQ, D] dense output
        sm_scale,
        k_descale,  # [1] fp32, the pool's per-tensor scale
        v_descale,  # [1] fp32
    ):
        bs = kv_lens.shape[0]
        hkv = k_cache.shape[2]
        qf = fold_block_queries(q, bs, hkv)
        of = torch.empty_like(qf)
        quantized = k_cache.dtype.itemsize == 1
        if quantized:
            qf = qf.to(k_cache.dtype)
        self._ua(
            q=qf,
            k=k_cache,
            v=v_cache,
            out=of,
            cu_seqlens_q=self._cu_seqlens(bs),
            max_seqlen_q=1,
            seqused_k=kv_lens,
            # Static: the table's token span. A device-side max would sync.
            max_seqlen_k=block_tables.shape[1] * k_cache.shape[1],
            softmax_scale=sm_scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_tables,
            softcap=0,
            q_descale=self._q_descale if quantized else None,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        return unfold_block_output(of, o, hkv)
