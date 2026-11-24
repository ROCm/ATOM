# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The W4A4 expert host on 128-row blocks (513-4096 tokens; 449-3583 with the shared expert): one routing-sort launch
writes the tile map and, to ``SORT_SCALES_MAX_TOKENS``, the sorted scale tiles (``common/sort_scales_all.py``), then
GEMM1 + GEMM2 run in one launch (``gemm12_a4w4``; with the shared expert ``gemm12_a4w4_shared_a8w8``).
"""

import functools

import torch

from .prefill_h6144_i768 import GEMM2_N_SPLIT, Mxfp4PrefillScales, gemm2_grid

H, I, BM = 6144, 768, 128  # noqa: E741


# the sort writes the scale tiles too (one launch): -1.1..-1.6 us at 513-2048, -0.7 at 4096; +1.4 at 8192 (BM256 host)
SORT_SCALES_MAX_TOKENS = 4096


def sort_writes_scales(n_tokens: int) -> bool:
    return n_tokens <= SORT_SCALES_MAX_TOKENS


# one launch at every row count of this host (README); not below 513: gemm12_a4w4 is not deterministic at 1-17 tokens
GEMM12_TOKENS = (513, 4096)
# its GEMM2 with adaptive chunks at 513-704 and 3584-4096, fixed chunks between
GEMM12_ADAPTIVE_MAX, GEMM12_ADAPTIVE_FROM = 704, 3584


def gemm12_for(n_tokens: int) -> bool:
    return GEMM12_TOKENS[0] <= n_tokens <= GEMM12_TOKENS[1]


def gemm12_adaptive(n_tokens: int) -> bool:
    return n_tokens <= GEMM12_ADAPTIVE_MAX or n_tokens >= GEMM12_ADAPTIVE_FROM


@functools.cache
def _gemm12(num_experts, topk, n_split, n_tokens=2049, *, adaptive=False):
    from .gemm12_a4w4 import compile_moe_gemm12_mxfp4_h6144_i768
    return compile_moe_gemm12_mxfp4_h6144_i768(
        E=num_experts, topk=topk, n_tokens=n_tokens, n_split=n_split, adaptive=adaptive,
    )


@functools.cache
def _gemm12_shared(n_tokens,adaptive,nt):
    from .gemm12_a4w4_shared_a8w8 import compile_moe_gemm12_mxfp4_shared_mxfp8_h6144_i768
    return compile_moe_gemm12_mxfp4_shared_mxfp8_h6144_i768(n_tokens=n_tokens,adaptive=adaptive,shared_nt=nt)


# m-block counters of gemm12_a4w4 (ready + consumer count per block, 256 B apart), zeroed once per device; every call
# leaves them at zero
_GEMM12_MAX_BLOCKS = 2048  # 1494 BM256 blocks at 87381 tokens
_ready: dict = {}


def gemm12_ready_counters(device):
    from .a16w4_utils import READY_STRIDE
    key = torch.device(device).index
    if key not in _ready:
        _ready[key] = torch.zeros(
            (_GEMM12_MAX_BLOCKS * READY_STRIDE // 4,), dtype=torch.int32, device=device
        )
    return _ready[key]


def a4w4_experts(
    x_quant: tuple[torch.Tensor, torch.Tensor],
    w13: torch.Tensor,
    w2: torch.Tensor,
    scales: Mxfp4PrefillScales,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    shared=None,
) -> torch.Tensor:
    """The slots ``topk_weights[:, k] * expert_{topk_ids[:, k]}(x)``, bf16 [n, topk, H], for the MXFP4 input ``x_quant``
    = (e2m1 codes [n, H/2], e8m0 [n, H/32], row-major); deterministic (no atomics into outputs).

    With ``shared=(FP8 codes, tiled scales, SharedWeights)`` (449-3583 rows) it returns ``(partials, shared)`` from the
    combined persistent launch, which borrows the otherwise unused last ready-arena slot: the routed blocks stay below
    that slot and every call restores both shared metadata words to zero.
    """
    from .common.launch import _run_compiled
    from .common.sort import SortBuffers
    from .common.sort_count_all import moe_sort_count_all
    from .common.sort_scales_all import moe_sort_scales_all
    from .common.scale_sort_tile_map import scale_sort_tile_map
    from .common.tile_map import (
        tile_map_grid,
    )

    xq, xs = x_quant
    n = xq.shape[0]
    num_experts = w13.shape[0]
    topk = topk_ids.shape[1]
    dev = xq.device
    stream = torch.cuda.current_stream()
    n_split = GEMM2_N_SPLIT
    # canonical rows dedupe the probe and padding-epilogue variants
    probe_rows = 513 if n <= 1536 else (1537 if n <= 2048 else 2049)
    b = SortBuffers.allocate(n, num_experts, topk, BM, dev)
    nmb = b.max_sorted // BM
    rows = nmb * BM
    grid1 = tile_map_grid(nmb, I)
    tile_map = torch.empty((grid1 + 1,), dtype=torch.int32, device=dev)
    ids32 = topk_ids.to(torch.int32).contiguous()
    w32 = topk_weights.to(torch.float32).contiguous()
    a_s = torch.empty(
        ((b.sorted_ids.shape[0] + 31) // 32 * 32, H // 32),
        dtype=torch.uint8,
        device=dev,
    )
    if sort_writes_scales(n):
        moe_sort_scales_all(b, ids32, w32, xs, a_s, tile_map, grid1, n, E=num_experts, I=I)
    else:
        moe_sort_count_all(b, ids32, w32, tile_map, grid1, n, E=num_experts, I=I)
        scale_sort_tile_map(
            xs, b.sorted_ids, b.sorted_expert_ids, b.num_valid_ids,
            a_s, tile_map, grid1, n, BM, with_tile_map=False,
        )  # fmt: skip
    hq = torch.empty((rows, I // 2), dtype=torch.uint8, device=dev)
    hs = torch.empty((rows * (I // 32),), dtype=torch.uint8, device=dev)
    partials = torch.empty((n * topk, H), dtype=torch.bfloat16, device=dev)
    shared_args=()
    if shared is not None:
        assert 385<=n<=4096 and nmb<_GEMM12_MAX_BLOCKS  # bit-equal from 385 rows
        q8,s8,sw=shared
        hq8=torch.empty((n,768),device=dev,dtype=torch.uint8)
        hs8=torch.empty(((n+31)//32*32,24),device=dev,dtype=torch.uint8)
        shared_out=torch.empty((n,6144),device=dev,dtype=torch.bfloat16)
        p1=((n+255)//256)*6
        if n<=1536:p1=((p1+7)//8*8)*2
        shared_args=(q8.view(torch.int8).view(-1),s8.view(-1),sw.w13.view(torch.int8).view(-1),sw.s13.view(-1),
                     hq8.view(-1),hs8.view(-1),sw.w2.view(torch.int8).view(-1),sw.s2.view(-1),shared_out.view(-1),
                     gemm12_ready_counters(dev).data_ptr()+(_GEMM12_MAX_BLOCKS-1)*256,
                     (p1+7)//8*8,min(((n+255)//256)*24,256))
    assert (gemm12_for(n) or shared is not None) and nmb <= _GEMM12_MAX_BLOCKS
    _run_compiled(
        (_gemm12(num_experts, topk, n_split, probe_rows, adaptive=gemm12_adaptive(n))
         if shared is None else _gemm12_shared(probe_rows,gemm12_adaptive(n),n>=2048)),
        xq.view(-1),
        w13.view(torch.uint8).view(torch.int64).view(-1),
        hq.view(-1),
        a_s.view(-1),
        scales.w13.view(-1),
        hs,
        b.sorted_ids,
        b.sorted_expert_ids,
        n,
        nmb,
        int(a_s.numel()),
        tile_map,
        grid1,
        w2.view(torch.uint8).view(torch.int64).view(-1),
        partials.view(-1),
        scales.w2.view(-1),
        b.sorted_weights,
        b.num_valid_ids,
        gemm12_ready_counters(dev).data_ptr(),
        *shared_args,
        gemm2_grid(nmb, n_split),
        stream,
    )
    partials = partials.view(n, topk, H)
    if shared is not None:return partials,shared_out
    return partials
