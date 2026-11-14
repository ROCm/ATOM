# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Page-table read of a DSpark block over a FLASH-layout `MhaKvPool`.

The caller writes the block's own T rows into the pool before the read, so
request ``b``'s pool rows are keys ``[0, kv_lens[b])`` and every block row
attends all of them. Key ``t`` sits at slot
``block_tables[b, t // E] * E + t % E``, where ``E`` is the pool's entry span
(`MhaKvPool.fa_views`). The read takes exactly that, so the caller builds no
``kv_indptr``/``kv_indices`` stream for it.

`DSparkGQABlockAttention` is aiter's ``unified_attention`` -- the kernel
`PagedAttentionImpl.paged_attention_unified` already decodes through -- on the
FOLDED query ``[bs, HKV * T * G, D]`` with ``max_seqlen_q = 1``. That kernel
asserts ``causal``, and with one query row per sequence the causal mask admits
every key ``< seqused_k``, which is the block's bidirectional mask exactly. The
fold puts a request's T block rows and its G query heads per KV head on one KV
head, so one KV tile serves all ``T * G`` of them, and the all-decode launch is
the split-KV grid, so a long context is spread over segments instead of walked
by one program. The block rows come back out of the pool with the prefix, so
over an fp8 pool they are read quantized.

Over an fp8 pool the query is cast to the pool's dtype only where aiter's own
gfx950 Gluon gate accepts this read with q in that dtype, because Gluon needs q
and KV in one dtype; anywhere else the cast would quantize q for nothing. The
read asks the gate once, when it is built. The cast is clamped to the fp8
range first, since a plain cast turns an out-of-range value into NaN. On
gfx1250 the 3D Gluon path needs ``shuffled_kv_cache``, which this FLASH-layout
pool is not, so q stays bf16 there, as everywhere else.
"""

from __future__ import annotations

import torch
from aiter.ops.triton.attention import unified_attention as ua
from aiter.ops.triton.utils.device_info import get_num_sms

# The `_UAParams` fields a deployment does not fix: a call's tensors and the
# counts read off them, and what aiter derives inside the call from its own
# constants (`k_width`, `scale_k_width`, `block_scales_size`, `target_num_prgms`).
_UNFIXED_FIELDS = (
    "q",
    "k",
    "v",
    "out",
    "cu_seqlens_q",
    "seqused_k",
    "block_table",
    "num_tokens",
    "num_seqs",
    "total_num_q_blocks",
    "num_2d_prgms",
    "num_blocks",
    "k_width",
    "scale_k_width",
    "block_scales_size",
    "target_num_prgms",
)


def gluon_takes_pool_dtype_q(
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    kv_dtype: torch.dtype,
    softmax_scale: float,
    max_seqlen_k: int,
) -> bool:
    """Whether aiter's gfx950 Gluon gate accepts this read with q in ``kv_dtype``.

    aiter exposes no public form of the gate, so this calls its private
    ``_gfx950_gluon_supported`` on a ``_UAParams`` carrying every field the
    read fixes for the deployment, as `DSparkGQABlockAttention.__call__`
    passes it: ``num_query_heads`` is the folded query's. Only
    ``_UNFIXED_FIELDS`` are ``None``; the optional fields keep aiter's own
    defaults. A renamed symbol, or a new required field, raises; a new
    defaulted field keeps its default.
    """
    params = ua._UAParams(
        **dict.fromkeys(_UNFIXED_FIELDS),
        softmax_scale=softmax_scale,
        softcap=0,
        causal=True,
        sliding_window=0,
        max_seqlen_q=1,
        max_seqlen_k=max_seqlen_k,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        num_queries_per_kv=num_query_heads // num_kv_heads,
        head_size=head_size,
        head_size_v=head_size,
        block_size=block_size,
        q_dtype=kv_dtype,
        kv_cache_dtype=kv_dtype,
        v_cache_dtype=kv_dtype,  # K and V share the one pool's dtype
        all_decode=True,
        spec_decode=False,  # aiter: equal_seqlen_q and not all_decode
        shuffled_kv_cache=False,
        use_alibi_slopes=False,
        use_qq_bias=False,
        num_sms=get_num_sms(),
    )
    return bool(ua._gfx950_gluon_supported(params))


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


class DSparkGQABlockAttention:
    """aiter ``unified_attention`` (3D split-KV) on the folded draft block.

    Built once per deployment, for the block it folds, ``block_rows`` rows of
    ``num_heads`` query heads over ``num_kv_heads`` KV heads per rank, and the
    pool it reads: K and V heads of ``head_size``, entries of ``block_size``
    tokens, stored in ``kv_dtype``.
    """

    def __init__(
        self,
        device,
        max_bs: int,
        max_model_len: int,
        num_heads: int,
        num_kv_heads: int,
        block_rows: int,
        head_size: int,
        block_size: int,
        kv_dtype: torch.dtype,
        softmax_scale: float,
    ):
        self._ua = ua.unified_attention
        # What aiter picks its kernel and split config from. A heuristic, not a
        # bound: tiles come from each request's own `seqused_k`, which may run
        # T past it for a block anchored at the last position.
        self.max_model_len = max_model_len
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.block_size = block_size
        self.kv_dtype = kv_dtype
        self.softmax_scale = softmax_scale
        # Whether a read casts q to the pool's dtype.
        self.casts_q = kv_dtype.itemsize == 1 and gluon_takes_pool_dtype_q(
            num_heads * block_rows,
            num_kv_heads,
            head_size,
            block_size,
            kv_dtype,
            softmax_scale,
            max_model_len,
        )
        # q's descale when it is cast: the cast is a plain one, so 1.0.
        self._q_descale = torch.ones(1, dtype=torch.float32, device=device)
        # One query row per sequence, so `cu_seqlens_q` is a constant ramp,
        # sized for the engine's widest batch so a replay only slices it.
        self._cu_seqlens_q = torch.arange(max_bs + 1, dtype=torch.int32, device=device)

    def check_cache(self, k_cache, v_cache) -> None:
        """Refuse a pool other than the one this read was built for: the q cast
        was decided for its geometry and dtype, and the read takes both off the
        cache it is handed."""
        built = (self.block_size, self.num_kv_heads, self.head_size)
        for name, cache in (("K", k_cache), ("V", v_cache)):
            if (tuple(cache.shape[1:]), cache.dtype) != (built, self.kv_dtype):
                raise ValueError(
                    f"the DSpark block read was built for [entries, *{built}] in "
                    f"{self.kv_dtype}, but is bound to a {name} cache of "
                    f"{tuple(cache.shape)} in {cache.dtype}"
                )

    def __call__(
        self,
        q,  # [bs * T, HQ, D]; clamped in place when cast to fp8
        k_cache,  # [entries, E, HKV, D] FLASH layout
        v_cache,  # [entries, E, HKV, D]
        block_tables,  # [bs, W] entry ids
        kv_lens,  # [bs] keys each request attends, block rows included
        k_descale,  # [1] fp32, the pool's per-tensor scale
        v_descale,  # [1] fp32
    ) -> torch.Tensor:
        """Returns the block's attention output, ``[bs * T, HQ, D]`` in q's dtype."""
        bs = kv_lens.shape[0]
        hkv = k_cache.shape[2]
        quantized = self.casts_q
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
            softmax_scale=self.softmax_scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_tables,
            softcap=0,
            q_descale=self._q_descale if quantized else None,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        return unfold_block_output(of, q.new_empty(q.shape), hkv)
