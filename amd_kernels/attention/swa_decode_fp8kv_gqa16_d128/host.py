# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache + launch wrapper of the FlyDSL DSpark draft decode attention."""

import functools

import numpy as np
import torch

from . import kernel as _kernel
from . import max_window
from .device import initialize_xcd_topology, on_gfx950

MAX_SPLITS = 6
# The split partials' scratch holds this many (request, KV head) x split workgroups;
# a launch with more does not split.
MAX_SPLIT_WGS = 1024


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the
    cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.cache
def _get_kernel(
    ql: int,
    splits: int,
    window: int,
    causal: bool,
    block_size: int,
    xcds: int,
    waves: int,
    quant: bool = False,
    batch_rows: bool = False,
    group_rows: int = 0,
):
    if group_rows == 1:
        from .bf16mfma_gqa16_d128_r7_tq1 import compile_swa_fp8kv_bf16mfma_gqa16_d128_r7_tq1

        return compile_swa_fp8kv_bf16mfma_gqa16_d128_r7_tq1(
            ql, splits, window, causal, block_size, xcds, waves, quant=quant
        )
    if group_rows == 2:
        from .bf16mfma_gqa16_d128_r7_tq2 import compile_swa_fp8kv_bf16mfma_gqa16_d128_r7_tq2

        return compile_swa_fp8kv_bf16mfma_gqa16_d128_r7_tq2(
            ql, splits, window, causal, block_size, xcds, waves, quant=quant
        )
    if group_rows == 3:
        from .bf16mfma_gqa16_d128_r7_tq3 import compile_swa_fp8kv_bf16mfma_gqa16_d128_r7_tq3

        return compile_swa_fp8kv_bf16mfma_gqa16_d128_r7_tq3(
            ql, splits, window, causal, block_size, xcds, waves, quant=quant
        )
    if batch_rows:
        from .bf16mfma_gqa16_d128_r7_batch2 import compile_swa_fp8kv_bf16mfma_gqa16_d128_r7_batch2

        return compile_swa_fp8kv_bf16mfma_gqa16_d128_r7_batch2(
            ql, splits, window, causal, block_size, xcds, waves, quant=quant
        )
    return _kernel.compile_swa_decode_fp8kv(
        ql, splits, window, causal, block_size, xcds, waves, quant=quant
    )


# The query tiles also serve the non-causal seven-query block (the upstream DSpark
# draft): each group's rows keep their positions and read keys up to the request's end.
NONCAUSAL_Q7_TILES = True


def _tp4_q7_rows(reqs: int) -> int:
    """Rows/workgroup for the 4096-window TP4 draft (causal or not), the request-wide
    kernel's key splits.

    Preserve split counts: changing them moves BF16 P rounding and fails the
    L2 equivalence gate. The query tiles only change summation order.
    """
    if reqs <= 5:
        return 1
    # TQ3's 1-row groups run a 1-row body and take the last workgroups: bf16 output,
    # TQ3 vs TQ2 +19.9% at 10 requests, -15.3% at 11; vs the request-wide kernel
    # -8.0% at 48, -0.3% at 56, +4.0% at 64.
    if reqs <= 10:
        return 2
    if reqs <= 48:
        return 3
    return 0


@functools.cache
def _split_buffers(device: torch.device):
    """The splits' partials and tickets (zero; the last split of a request resets
    its own). Allocated on first use (the warm-up), so graph capture records no
    allocation. Sized for MAX_QL rows (8: 72 MiB of partials, once per device)."""
    part = torch.empty(
        MAX_SPLIT_WGS * _kernel.MAX_QL * 64 * _kernel.PART_F,
        dtype=torch.float32,
        device=device,
    )
    return part, torch.zeros(MAX_SPLIT_WGS, dtype=torch.int32, device=device)


@functools.cache
def _cu_count(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def decode_splits(num_rows: int, cus: int) -> int:
    """Workgroups per (request, KV head): one round of workgroups (one per CU), at
    most 6. Graph-timed on MI355X (DSpark 3 rows, 30K-210K contexts, window 4096),
    the best split count at 32 / 45 / 64 / 96 requests is 5-6 / 5 / 4 / 2; past one
    round a split costs more than it saves (the last split's combine)."""
    return max(1, min(MAX_SPLITS, cus // max(num_rows, 1)))


def plan_splits(
    num_reqs: int,
    num_kv_heads: int,
    xcds: int,
    cus: int,
    splits: int | None = None,
) -> int:
    """The split count ``swa_decode_fp8kv`` launches with (``splits`` None = its
    automatic choice); ``xcds`` 0 = no XCD grouping."""
    group = xcds if xcds else 1
    grid_reqs = -(-num_reqs // group) * group
    if splits is None:
        # Size against the padded grid (every row count): the launch spans
        # grid_reqs * num_kv_heads * splits workgroups, which must fit in one
        # round of CUs (num_reqs undercounts once the grid is padded to the XCDs:
        # 41 / 42 requests on 256 CUs would take 48 x 6 = 288 workgroups).
        splits = decode_splits(grid_reqs * num_kv_heads, cus)
    if grid_reqs * num_kv_heads * splits > MAX_SPLIT_WGS:
        splits = 1
    return splits


def launch_splits(q, kv_cache, num_reqs, splits=None, xcd_count=None) -> int:
    """``plan_splits`` with the device's topology, as ``swa_decode_fp8kv`` uses."""
    if xcd_count is None:
        xcd_count = initialize_xcd_topology()
    xcds = xcd_count if xcd_count and xcd_count > 1 else 0
    return plan_splits(num_reqs, kv_cache.shape[1], xcds, _cu_count(q.device), splits)


def supports_swa_decode_fp8kv(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    max_query_len: int,
    window: int,
) -> bool:
    """The kernel's case: gfx950, 128-dim heads, 16 query heads per KV head, an
    e4m3 cache, a window whose page lookups fit 64 lanes, at most 8 query rows."""
    return (
        on_gfx950()
        and q.dtype == torch.bfloat16
        and q.dim() == 3
        and q.shape[2] == _kernel.D
        and kv_cache.dtype in (torch.uint8, torch.float8_e4m3fn)
        and kv_cache.dim() == 4
        and q.shape[1] == kv_cache.shape[1] * _kernel.HG
        and kv_cache.shape[3] == _kernel.TOK_B
        and kv_cache.stride(2) == _kernel.TOK_B
        and kv_cache.stride(3) == 1
        and kv_cache.shape[2] % _kernel.CH == 0
        and 1 <= max_query_len <= _kernel.MAX_QL
        and 0 < window <= max_window(max_query_len)
    )


def swa_decode_fp8kv(
    q: torch.Tensor,  # [rows, H, 128] bf16
    kv_cache: torch.Tensor,  # [pages, Hkv, block_size, 256] e4m3 (K then V)
    block_table: torch.Tensor,  # [reqs, max_blocks] int32
    seq_lens: torch.Tensor,  # [reqs] int32
    query_start_loc: torch.Tensor,  # [reqs + 1] int32
    max_query_len: int,
    sm_scale: float,
    window: int,
    causal: bool,
    k_scale: torch.Tensor,  # fp32 per-tensor dequantization scales
    v_scale: torch.Tensor,
    output: torch.Tensor | None,  # [rows, H, 128] bf16, written for every request row
    splits: int | None = None,
    xcd_count: int | None = None,
    waves: int = 4,
    quant_out: tuple[torch.Tensor, torch.Tensor, int] | None = None,
) -> None:
    """Request r's rows query_start_loc[r] .. query_start_loc[r + 1] are the last
    positions of its sequence; row p sees keys k with p - window < k < seq_len
    (causal: k <= p).

    ``quant_out`` = (codes, scales, tile): instead of ``output``
    (pass None), write its MXFP8 quant as the o_proj's input, byte for byte what
    ``third_party/aiter/mxfp8_gemm``'s quantizers make of the bf16 output (as
    ``nvfp4_sparse_decode``'s ``quant_out``): e4m3 codes ``[>= ceil(rows / 16) *
    16, H * 128]`` uint8, row-major (tile 0) or in the decode GEMM's (16, 16) tile
    layout (tile > 0, a multiple of 16), and e8m0 scales ``[>= ceil(rows / 32) *
    32, H * 4]`` uint8 in AITER's shuffled layout. Rows of no request and the
    padding rows are not written."""
    rows, num_heads, head_dim = q.shape
    num_reqs = seq_lens.shape[0]
    if rows == 0 or num_reqs == 0:
        return
    num_kv_heads = kv_cache.shape[1]
    block_size = kv_cache.shape[2]
    assert q.is_contiguous()
    if quant_out is None:
        assert output is not None and output.is_contiguous()
        assert output.dtype == torch.bfloat16 and output.shape == q.shape
        qc_ptr = qs_ptr = tile = 0
    else:
        codes, scales, tile = quant_out
        assert tile % 16 == 0, tile  # 0: row-major codes; else 16-row tiles
        assert codes.dtype == torch.uint8 and codes.is_contiguous()
        assert scales.dtype == torch.uint8 and scales.is_contiguous()
        assert codes.shape[0] >= (rows + 15) // 16 * 16
        assert codes.shape[1] == num_heads * head_dim
        assert scales.shape[0] >= (rows + 31) // 32 * 32
        assert scales.shape[1] == num_heads * head_dim // 32
        qc_ptr, qs_ptr = codes.data_ptr(), scales.data_ptr()
    for t in (block_table, seq_lens, query_start_loc):
        assert t.dtype == torch.int32 and t.stride(-1) == 1
    assert kv_cache.stride(0) < 2**31 and kv_cache.stride(1) < 2**31
    if xcd_count is None:
        xcd_count = initialize_xcd_topology()
    xcds = xcd_count if xcd_count and xcd_count > 1 else 0
    group = xcds if xcds else 1
    grid_reqs = -(-num_reqs // group) * group
    splits = plan_splits(num_reqs, num_kv_heads, xcds, _cu_count(q.device), splits)
    # the query tiles and the 2-row batch kernel run split windows only (the TP4 draft
    # at <= 128 requests); one split takes the request-wide kernel
    group_rows = 0
    if (num_kv_heads == 1 and max_query_len == 7 and window == 4096
            and block_size == 128 and (causal or NONCAUSAL_Q7_TILES) and xcds == 8
            and waves == 4 and splits > 1):
        group_rows = _tp4_q7_rows(num_reqs)
    part, cnt = _split_buffers(q.device)
    c = float(np.float32(sm_scale) * np.float32(_kernel.LOG2E))
    _run_compiled(
        _get_kernel(
            max_query_len,
            splits,
            window,
            causal,
            block_size,
            xcds,
            waves,
            quant_out is not None,
            num_kv_heads == 1 and max_query_len == 7 and splits > 1 and waves == 4,
            group_rows,
        ),
        q.data_ptr(),
        kv_cache.data_ptr(),
        block_table.data_ptr(),
        seq_lens.data_ptr(),
        query_start_loc.data_ptr(),
        output.data_ptr() if output is not None else 0,
        k_scale.data_ptr(),
        v_scale.data_ptr(),
        part.data_ptr(),
        cnt.data_ptr(),
        part.numel() * 4,
        num_reqs,
        rows,
        num_heads,
        num_kv_heads,
        block_table.stride(0),
        kv_cache.stride(0),
        kv_cache.stride(1),
        c,
        qc_ptr,
        qs_ptr,
        tile,
        grid_reqs * splits,
        torch.cuda.current_stream(q.device),
    )
