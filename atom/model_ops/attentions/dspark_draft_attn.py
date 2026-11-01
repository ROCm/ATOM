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

Where aiter's Gluon kernel serves this read (gfx950), the query is cast to an
fp8 pool's dtype too, because Gluon needs q and KV in one dtype: about 1.6x
faster than the Triton path at bs 32 / 128k context. The cast costs precision
only down to what this checkpoint was validated at (SGLang's verify kernel
casts q to the pool dtype for the prefix dot), and it is clamped to the fp8
range first, since a plain cast turns an out-of-range value into NaN. On
gfx1250 the 3D Gluon path needs ``shuffled_kv_cache``, which this FLASH-layout
pool is not, so q stays bf16 there, as everywhere else.
"""

from __future__ import annotations

import torch


def fold_block_queries(
    q: torch.Tensor, bs: int, num_kv_heads: int, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``[bs * T, HQ, D]`` -> ``[bs, HKV * T * G, D]``, KV-head-major.

    `unified_attention` maps query head ``h`` to KV head
    ``h // num_queries_per_kv``, so every row sharing a KV head has to be
    contiguous in the head axis: ``(kv_head, t, g)``. Query head ``h`` of a
    GQA layer is ``(h // G, h % G)``. With ``out``, the fold is one ``copy_``
    into it, casting to ``out``'s dtype on the way.
    """
    rows, hq, d = q.shape
    t = rows // bs
    g = hq // num_kv_heads
    src = q.view(bs, t, num_kv_heads, g, d).permute(0, 2, 1, 3, 4)
    if out is None:
        return src.reshape(bs, num_kv_heads * t * g, d)
    out.view(bs, num_kv_heads, t, g, d).copy_(src)
    return out


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

    def __init__(self, device, max_bs: int, max_model_len: int):
        from aiter.ops.triton.attention import unified_attention as ua

        self.device = device
        self._ua = ua.unified_attention
        # What aiter picks its kernel and split config from. A heuristic, not a
        # bound: tiles come from each request's own `seqused_k`, which may run
        # T past it for a block anchored at the last position.
        self.max_model_len = max_model_len
        # Cast q to an fp8 pool's dtype only where that selects the Gluon
        # kernel: aiter's own gfx950 test plus its kernel having loaded.
        self.q_to_pool_dtype = (
            getattr(ua, "DEVICE_ARCH", None) == "gfx950"
            and getattr(ua, "_unified_attention_kernel_gfx950", None) is not None
        )
        # q's descale when it is cast: the cast is a plain one, so 1.0.
        self._q_descale = torch.ones(1, dtype=torch.float32, device=device)
        # One query row per sequence, so `cu_seqlens_q` is a constant ramp,
        # sized for the engine's widest batch so a replay only slices it.
        self._cu_seqlens_q = torch.arange(max_bs + 1, dtype=torch.int32, device=device)

    def __call__(
        self,
        q,  # [bs * T, HQ, D]; clamped in place when cast to fp8
        k_cache,  # [entries, E, HKV, D] FLASH layout
        v_cache,  # [entries, E, HKV, D]
        block_tables,  # [bs, W] entry ids
        kv_lens,  # [bs] keys each request attends, block rows included
        sm_scale,
        k_descale,  # [1] fp32, the pool's per-tensor scale
        v_descale,  # [1] fp32
    ) -> torch.Tensor:
        """Returns the block's attention output, ``[bs * T, HQ, D]`` in q's dtype."""
        bs = kv_lens.shape[0]
        hkv = k_cache.shape[2]
        quantized = self.q_to_pool_dtype and k_cache.dtype.itemsize == 1
        if quantized:
            fp8_max = torch.finfo(k_cache.dtype).max
            q.clamp_(-fp8_max, fp8_max)
        rows, hq, d = q.shape
        qf = q.new_empty(
            (bs, hq * rows // bs, d), dtype=k_cache.dtype if quantized else q.dtype
        )
        fold_block_queries(q, bs, hkv, out=qf)
        of = q.new_empty(qf.shape)
        self._ua(
            q=qf,
            k=k_cache,
            v=v_cache,
            out=of,
            cu_seqlens_q=self._cu_seqlens_q[: bs + 1],
            max_seqlen_q=1,
            seqused_k=kv_lens,
            max_seqlen_k=self.max_model_len,
            softmax_scale=sm_scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_tables,
            softcap=0,
            q_descale=self._q_descale if quantized else None,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        return unfold_block_output(of, q.new_empty(q.shape), hkv)
