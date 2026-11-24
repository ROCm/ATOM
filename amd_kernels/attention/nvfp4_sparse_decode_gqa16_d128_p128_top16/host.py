# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The host of the NVFP4 sparse decode: the schedule (variant, splits, grid), the compile cache, the warm-up."""

import functools

import numpy as np
import torch
from amd_kernels.warm_up import entry

from . import kernel as _kernel
from .device import initialize_xcd_topology, nvfp4_e4m3_lut

MAX_SPLITS = 8
# the split partials' scratch holds this many (row, KV head) workgroup rows; a launch with more rows does not split
MAX_SPLIT_ROWS = 1024
# the tail grid splits the leftover pairs at most this many ways: ql 8, 1 request 4 splits 10.9 us, 8 12.3, 2 12.0
MAX_TAIL_SPLITS = 4


def _run_compiled(exe, *args):
    """The first call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.cache
def _get_fp8_quad(quant):
    from .fp8.row_quad import compile_nvfp4_sparse_gqa16_d128_p128_top16_q8_tq4

    return compile_nvfp4_sparse_gqa16_d128_p128_top16_q8_tq4(quant=quant)


@functools.cache
def _get_kernel(
    splits: int,
    topk: int,
    xcds: int,
    quant: bool,
    exact_fp16: bool,
    tail: bool = False,
    fp8: bool = False,
    row_xcd: bool = False,
    prefetch: bool = False,
    fixed: bool = False,
):
    if fixed and not fp8:
        from .bf16mfma_gqa16_d128_p128_top16_q8_fixed import compile_nvfp4_bf16mfma_gqa16_d128_p128_top16_q8_fixed
        return compile_nvfp4_bf16mfma_gqa16_d128_p128_top16_q8_fixed(splits, quant)
    if prefetch and not fp8:
        from .bf16mfma_gqa16_d128_p128_top16_prefetch import (
            compile_nvfp4_bf16mfma_gqa16_d128_p128_top16_prefetch,
        )

        return compile_nvfp4_bf16mfma_gqa16_d128_p128_top16_prefetch(splits, quant)
    if row_xcd and fp8:
        from .fp8mfma_gqa16_d128_p128_top16_q8_rowxcd import (
            compile_nvfp4_fp8mfma_gqa16_d128_p128_top16_q8_rowxcd,
        )

        return compile_nvfp4_fp8mfma_gqa16_d128_p128_top16_q8_rowxcd(
            splits, topk, xcds, quant, tail
        )
    if row_xcd and exact_fp16:
        from .f16mfma_gqa16_d128_p128_top16_q8_rowxcd import compile_nvfp4_f16mfma_gqa16_d128_p128_top16_q8_rowxcd

        return compile_nvfp4_f16mfma_gqa16_d128_p128_top16_q8_rowxcd(splits, quant)
    if row_xcd:
        from .bf16mfma_gqa16_d128_p128_top16_q8_rowxcd import (
            compile_nvfp4_bf16mfma_gqa16_d128_p128_top16_q8_rowxcd,
        )

        return compile_nvfp4_bf16mfma_gqa16_d128_p128_top16_q8_rowxcd(splits, quant)
    if fp8:
        from .fp8 import kernel as _kernel_fp8

        return _kernel_fp8.compile_nvfp4_fp8mfma_sparse_decode(
            splits, topk, xcds, quant, tail, prefetch, fixed
        )
    return _kernel.compile_nvfp4_sparse_decode(
        splits, topk, xcds, quant, exact_fp16, tail
    )


@functools.cache
def _get_kernel_select(splits: int, quant: bool, exact_fp16: bool):
    if exact_fp16:
        from .f16mfma_gqa16_d128_p128_top16_q8_rowxcd_select import (
            compile_nvfp4_f16mfma_gqa16_d128_p128_top16_q8_rowxcd_select,
        )

        return compile_nvfp4_f16mfma_gqa16_d128_p128_top16_q8_rowxcd_select(splits, quant)
    from .bf16mfma_gqa16_d128_p128_top16_q8_rowxcd_select import (
        compile_nvfp4_bf16mfma_gqa16_d128_p128_top16_q8_rowxcd_select,
    )

    return compile_nvfp4_bf16mfma_gqa16_d128_p128_top16_q8_rowxcd_select(splits, quant)


@functools.cache
def _get_kernel_select_generic(
    splits: int, topk: int, xcds: int, quant: bool, exact_fp16: bool, tail: bool
):
    from .kernel_select import compile_nvfp4_sparse_decode_select

    return compile_nvfp4_sparse_decode_select(splits, topk, xcds, quant, exact_fp16, tail)


@functools.cache
def _get_kernel_select_prefetch(splits: int, quant: bool):
    from .bf16mfma_gqa16_d128_p128_top16_prefetch_select import (
        compile_nvfp4_bf16mfma_gqa16_d128_p128_top16_prefetch_select,
    )

    return compile_nvfp4_bf16mfma_gqa16_d128_p128_top16_prefetch_select(splits, quant)


@functools.cache
def _get_kernel_select_fixed(splits: int, quant: bool):
    from .bf16mfma_gqa16_d128_p128_top16_q8_fixed_select import (
        compile_nvfp4_bf16mfma_gqa16_d128_p128_top16_q8_fixed_select,
    )

    return compile_nvfp4_bf16mfma_gqa16_d128_p128_top16_q8_fixed_select(splits, quant)


def _select_ids(index_score, seq_lens, topk_idx, rows, query_len, max_seq_len):
    """The indexer's selector launch: ``topk_idx`` from ``index_score``."""
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_decode import (
        nvfp4_index_top16_from_scores,
    )

    nvfp4_index_top16_from_scores(
        index_score, seq_lens, topk_idx, rows=rows, query_len=query_len,
        max_seq_len=max_seq_len,
    )


@functools.cache
def _split_buffers(device: torch.device):
    """The splits' partials and tickets (zero; a row's last split resets its own), allocated on first use (the
    warm-up) so that graph capture records no allocation."""
    part = torch.empty(
        MAX_SPLIT_ROWS * MAX_SPLITS * 64 * _kernel.PART_F,
        dtype=torch.float32,
        device=device,
    )
    return part, torch.zeros(MAX_SPLIT_ROWS, dtype=torch.int32, device=device)


@functools.cache
def _unit_scale(device: torch.device) -> torch.Tensor:
    return torch.ones(1, dtype=torch.float32, device=device)


# (ql, requests) where the padded one-split grid beats the tail grid beyond A/A noise (up to 10%)
_PADDED_S1 = {6: (8, 29, 40), 8: (8, 14, 22, 30, 38, 46)}
# (ql, requests) where the tail grid's split count adds a CU round: ql 6 x 50 requests 4 splits 55.17 us, 1 split 51.36
_TAIL_ONE_SPLIT = {6: (50,)}


# fp8: the four-row union from here (the same chunk P bits): per-row / union 768 rows 38.5 / 42.1 us, 832 47.2 / 42.2
_FP8_UNION_MIN_ROWS = 769


def _tp4_q8_splits(rows: int) -> int:
    """Splits at 1 KV head, 8 rows per request, top-16, 8 XCDs: even block counts of >= 4 per split (an odd
    count masks a block of the 2-block ping-pong, 2-block splits lose in the combine), one split from 200 rows. fp8
    partial CU rounds take the tail grid before this table."""
    if rows <= 64:
        return 4
    if rows <= 128:
        return 2
    if rows <= 192:
        return 4
    return 1


def _tuned_splits(
    exact_fp16: bool,
    quant: bool,
    decode_query_len: int,
    num_kv_heads: int,
    topk: int,
    reqs: int,
    xcds: int,
) -> int | None:
    """The measured split counts; None where no measurement covers the shape (the caller takes ``decode_splits``)."""
    if (
        num_kv_heads == 1
        and decode_query_len == 8
        and topk == 16
        and xcds == 8
    ):
        return _tp4_q8_splits(reqs * decode_query_len)
    if not (exact_fp16 and quant and num_kv_heads == 4 and topk == 16 and xcds == 8):
        return None
    # (q = 6 and 8 take the tail grid, ``_use_tail``, except these requests.)
    if reqs in _PADDED_S1.get(decode_query_len, ()):
        return 1
    return None


def _use_tail(
    exact_fp16: bool,
    quant: bool,
    decode_query_len: int,
    num_kv_heads: int,
    topk: int,
    xcd_count: int,
    reqs: int,
) -> bool:
    """Whether an automatic launch takes the tail grid (``tail_schedule``): 4 KV heads, exact fp16, MXFP8 output,
    ql 6 or 8, top-16, 8 XCDs, except the ``_PADDED_S1`` batches."""
    return (
        reqs not in _PADDED_S1.get(decode_query_len, ())
        and exact_fp16
        and quant
        and decode_query_len in (6, 8)
        and num_kv_heads == 4
        and topk == 16
        and xcd_count == 8
    )


def tail_schedule(
    pairs: int, topk: int, cus: int, xcds: int, tail_splits: int | None = None
) -> tuple[int, int, int, int]:
    """(heavy, tail pairs per XCD, tail splits, grid) of the tail grid.

    A (row, KV head) pair is one workgroup's work (``topk`` blocks, VALU-bound), so the time follows the most loaded
    CU. The whole rounds, ``heavy`` = the largest multiple of ``cus`` pairs, run unsplit (no combine); only the
    ``pairs - heavy`` leftover pairs split, ``decode_splits`` ways (at most ``MAX_TAIL_SPLITS``), so they spread over
    all CUs instead of adding a round. ``grid`` = heavy + xcds x ceil(tail / xcds) x splits workgroups."""
    assert cus % xcds == 0, (cus, xcds)
    heavy = pairs // cus * cus
    tail = pairs - heavy
    if tail == 0:
        return heavy, 0, 1, heavy
    s = tail_splits or min(decode_splits(tail, topk, cus), MAX_TAIL_SPLITS)
    tpx = -(-tail // xcds)
    assert 1 <= s <= MAX_SPLITS and xcds * tpx <= MAX_SPLIT_ROWS
    return heavy, tpx, s, heavy + xcds * tpx * s


def decode_splits(num_rows: int, num_blocks: int, cus: int) -> int:
    """Splits of a row's blocks. A workgroup walks its blocks serially and the kernel is VALU-bound, so the time
    follows the work per CU: ceil(rows x splits / CUs) rounds of ceil(blocks / splits) blocks plus about one block of
    fixed cost per workgroup (prologue, combine); the fewest splits among the minima, every split at least two blocks.
    It picked the fastest count at 16-1024 rows (320 rows: 4 splits 28.0 us, 1 split 36.6)."""
    best_cost, best = None, 1
    for s in range(1, MAX_SPLITS + 1):
        per = -(-num_blocks // s)
        if s > 1 and per < 2:
            break
        cost = -(-(num_rows * s) // cus) * (per + 1)
        if best_cost is None or cost < best_cost:
            best_cost, best = cost, s
    return best


@functools.cache
def _cu_count(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


# ``quant_out`` from this many rows: attention + o_proj -1.0..-11.4% at 8-512 rows, +1.5% at 4 (bench_attn_oproj_mxfp8)
MXFP8_OUT_MIN_ROWS = 8


def nvfp4_sparse_decode(
    q: torch.Tensor,  # [rows, H, 128] bf16
    kv_cache: torch.Tensor,  # [pages, 2 * Hkv, 128, 72] uint8
    topk_idx: torch.Tensor,  # [Hkv, rows, topk] int32 logical blocks, -1 = none
    block_table: torch.Tensor,  # [reqs, max_blocks] int32
    seq_lens: torch.Tensor,  # [reqs] int32
    sm_scale: float,
    output: torch.Tensor,  # [rows, H, 128] bf16, written
    k_scale: torch.Tensor | None = None,  # fp32 global scales (None = 1)
    v_scale: torch.Tensor | None = None,
    decode_query_len: int = 1,
    splits: int | None = None,
    xcd_count: int | None = None,
    quant_out: tuple[torch.Tensor, torch.Tensor, int] | None = None,
    compute_type: torch.dtype = torch.bfloat16,
    tail_splits: int | None = None,
    index_score: torch.Tensor | None = None,  # [Hkv, >= rows, >= width] fp32
    index_max_seq_len: int | None = None,
) -> None:
    """NVFP4 sparse decode attention into ``output`` (contract: ``reference.py``): row i is row i % decode_query_len
    of request i // decode_query_len.

    Args:
        compute_type: bf16 (K / V rounded to e4m3, bf16 Q and P), fp16 (exact K / V, fp16 Q and P; Q must stay
            finite in fp16) or float8_e4m3fn (opt-in fp8 MFMAs: NV's Q8KV4, P per 32-key chunk); fp32 accumulation.
        quant_out: (codes, scales, tile): write the bf16 output's MXFP8 quant (vLLM's quantizer bytes, the o_proj
            input) instead of ``output``: e4m3 codes ``[>= ceil(rows / 16) * 16, H * 128]`` uint8, row-major (tile 0)
            or in the decode GEMM's (16, 16) tiles (tile 16), e8m0 scales ``[>= ceil(rows / 32) * 32, H * 4]`` in
            AITER's shuffled layout. Padding rows are not written.
        splits: None: the host's schedule (the tail grid for the ``_use_tail`` shapes); a count keeps the padded grid.
        tail_splits: the tail grid with that many tail splits (sweeps).
        index_score: the indexer's scores (``nvfp4_index_score_rows``; ``index_max_seq_len`` its host max): the call
            writes ``topk_idx`` (``nvfp4_index_score_top16``'s ids, bit for bit) before reading it. At 1 KV head,
            top-16, bf16 and <= 256 rows the attention workgroups select themselves (the ``*_select`` kernels);
            elsewhere the selector launches first.
    """
    rows, num_heads, head_dim = q.shape
    if rows == 0:
        return
    if compute_type not in (torch.bfloat16, torch.float16, torch.float8_e4m3fn):
        raise ValueError("NVFP4 sparse decode compute must be BF16, FP16 or FP8")
    exact_fp16 = compute_type == torch.float16
    # fp8 MFMAs (``fp8/kernel.py``) on the bf16 path's schedules (splits, XCD grid)
    fp8 = compute_type == torch.float8_e4m3fn
    quant = quant_out is not None
    if quant_out is not None:
        codes, scales, tile = quant_out
        assert tile % 16 == 0, tile  # 0: row-major codes; else 16-row tiles
        assert codes.dtype == torch.uint8 and codes.is_contiguous()
        assert scales.dtype == torch.uint8 and scales.is_contiguous()
        assert codes.shape[0] >= (rows + max(tile, 16) - 1) // max(tile, 16) * max(
            tile, 16
        )
        assert codes.shape[1] == num_heads * head_dim
        assert scales.shape[0] >= (rows + 31) // 32 * 32
        assert scales.shape[1] == num_heads * head_dim // 32
    num_kv_heads = kv_cache.shape[1] // 2
    topk = topk_idx.shape[-1]
    assert rows == seq_lens.shape[0] * decode_query_len
    assert head_dim == _kernel.D and num_heads == num_kv_heads * _kernel.HG
    assert q.dtype == torch.bfloat16 and output.dtype == torch.bfloat16
    assert q.is_contiguous() and output.is_contiguous()
    assert kv_cache.dtype == torch.uint8 and kv_cache.stride(1) == _kernel.BLK * 72
    assert kv_cache.stride(3) == 1 and kv_cache.stride(0) < 2**31
    assert topk_idx.dtype == torch.int32 and topk_idx.shape[0] == num_kv_heads
    assert topk_idx.shape[1] >= rows
    assert topk_idx.stride(2) == 1 and topk_idx.stride(1) == topk
    for t in (block_table, seq_lens):
        assert t.dtype == torch.int32 and t.stride(-1) == 1
    width = 0
    if index_score is not None:
        from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import score_width

        assert index_max_seq_len is not None and index_max_seq_len > 0
        width = score_width(index_max_seq_len)
        assert index_score.dtype == torch.float32 and index_score.stride(2) == 1
        assert index_score.shape[0] == num_kv_heads and index_score.shape[1] >= rows
        assert index_score.shape[2] >= width
    # fp8 from _FP8_UNION_MIN_ROWS rows: four rows per workgroup over their page union; explicit split sweeps stay
    # per row
    if (fp8 and num_kv_heads == 1 and decode_query_len == 8 and topk == 16
            and rows >= _FP8_UNION_MIN_ROWS and splits is None and tail_splits is None):
        if index_score is not None:
            _select_ids(index_score, seq_lens, topk_idx, rows, decode_query_len, index_max_seq_len)
        unit = _unit_scale(q.device)
        ks = unit if k_scale is None else k_scale
        vs = unit if v_scale is None else v_scale
        c = float(np.float32(sm_scale) * np.float32(_kernel.LOG2E))
        _run_compiled(
            _get_fp8_quad(quant), q.data_ptr(), kv_cache.data_ptr(), topk_idx.data_ptr(),
            seq_lens.data_ptr(), block_table.data_ptr(),
            codes.data_ptr() if quant else output.data_ptr(), ks.data_ptr(), vs.data_ptr(),
            nvfp4_e4m3_lut(q.device).data_ptr(), scales.data_ptr() if quant else 0,
            decode_query_len, tile if quant else 0, rows, num_heads, num_kv_heads,
            topk_idx.stride(0), block_table.stride(0), kv_cache.stride(0), c,
            torch.cuda.current_stream(q.device),
        )
        return
    if xcd_count is None:
        xcd_count = initialize_xcd_topology()
    xcds = xcd_count if xcd_count and xcd_count > 1 else 0
    # fp8 at partial CU rounds: the tail grid (whole rounds unsplit, the leftover rows split; chunk P ignores splits)
    if (fp8 and num_kv_heads == 1 and decode_query_len == 8 and topk == 16
            and xcds == 8 and splits is None and tail_splits is None):
        if 256 < rows <= 320 or 512 < rows <= 576:
            tail_splits = 4
        elif 320 < rows <= 384 or 576 < rows <= 640:
            tail_splits = 2
    tail = tail_splits is not None or (
        splits is None
        and _use_tail(
            exact_fp16,
            quant,
            decode_query_len,
            num_kv_heads,
            topk,
            xcds,
            seq_lens.numel(),
        )
    )
    # <= 4 requests of 8 rows: rows spread over the XCDs, a row's 4 splits on one of them (grid.x a multiple of 8,
    # so the partials stay XCD-local)
    row_xcd = (
        not tail and num_kv_heads == 1
        and decode_query_len == 8 and topk == 16 and xcds == 8 and rows <= 32
        and (splits is None or splits == 4)
    )
    heavy = tpx = 0
    # no padded XCD grid for <= 4 multi-row requests in fp16 at 4 KV heads (measured at ql 4 and 8; at 1 KV head
    # it wins: ql 4 x 1 request 10.34 -> 8.04 us)
    if (
        not tail
        and exact_fp16
        and num_kv_heads > 1
        and decode_query_len > 1
        and rows <= 4 * decode_query_len
    ):
        xcds = 0
    if tail:
        assert xcds > 0 and splits is None, "the tail grid needs the XCD grid"
        # (Automatic tail grid => the serving gate of ``_use_tail`` holds.)
        if tail_splits is None and seq_lens.numel() in _TAIL_ONE_SPLIT.get(
            decode_query_len, ()
        ):
            tail_splits = 1
        heavy, tpx, splits, grid_x = tail_schedule(
            rows * num_kv_heads, topk, _cu_count(q.device), xcds, tail_splits
        )
    elif xcds:
        group = xcds if row_xcd else xcds * decode_query_len
        grid_x = -(-rows // group) * group
    else:
        grid_x = rows
    if splits is None:  # (the tail grid has chosen its splits)
        wg_rows = grid_x * num_kv_heads
        if wg_rows > MAX_SPLIT_ROWS:
            splits = 1
        else:
            tuned = _tuned_splits(
                exact_fp16,
                quant,
                decode_query_len,
                num_kv_heads,
                topk,
                seq_lens.numel(),
                xcds,
            )
            if tuned is not None:
                splits = tuned
            else:
                splits = decode_splits(rows * num_kv_heads, topk, _cu_count(q.device))
    assert tail or splits == 1 or grid_x * num_kv_heads <= MAX_SPLIT_ROWS
    # (fp16 keeps kernel.py here: its folded prefetch variant measured 64 rows 9.87 vs 9.76 us, 256 rows -0.4%)
    prefetch = (
        not tail and not row_xcd and not exact_fp16 and not fp8
        and num_kv_heads == 1 and decode_query_len == 8 and topk == 16 and xcds == 8
        and splits in (1, 2, 4) and grid_x * splits <= _cu_count(q.device)
    )
    # fp8 per row, the same first-block overlap and grid bound: -3.3..-3.9% at 40-128 rows, -0.8% at 256
    fp8_prefetch = (
        fp8 and not tail and not row_xcd
        and num_kv_heads == 1 and decode_query_len == 8 and topk == 16 and xcds == 8
        and splits in (1, 2, 4) and grid_x * splits <= _cu_count(q.device)
    )
    fp8_q8 = (
        fp8 and not row_xcd
        and num_kv_heads == 1 and decode_query_len == 8 and topk == 16 and xcds == 8
        and splits in (1, 2, 4)
    )
    fixed = (
        not tail and not row_xcd and not prefetch and not exact_fp16 and not fp8
        and num_kv_heads == 1 and decode_query_len == 8 and topk == 16 and xcds == 8
        and splits in (2, 4) and grid_x <= 192
    )
    # (measured through 256 rows; larger batches keep the selector launch)
    generic = not row_xcd and not prefetch and not fixed and not fp8
    select = (
        index_score is not None and num_kv_heads == 1 and topk == 16 and rows <= 256
        and ((row_xcd and not fp8) or prefetch or fixed or generic)
    )
    if index_score is not None and not select:
        _select_ids(index_score, seq_lens, topk_idx, rows, decode_query_len, index_max_seq_len)
    part, cnt = _split_buffers(q.device)
    unit = _unit_scale(q.device)
    ks = unit if k_scale is None else k_scale
    vs = unit if v_scale is None else v_scale
    c = float(np.float32(sm_scale) * np.float32(_kernel.LOG2E))
    extra = ()
    if select:
        extra = (index_score.data_ptr(), index_score.stride(1), width)
    _run_compiled(
        (
            _get_kernel_select(splits, quant, exact_fp16)
            if select and row_xcd
            else _get_kernel_select_prefetch(splits, quant)
            if select and prefetch
            else _get_kernel_select_fixed(splits, quant)
            if select and fixed
            else _get_kernel_select_generic(splits, topk, xcds, quant, exact_fp16, tail)
            if select
            else _get_kernel(splits, topk, xcds, quant, exact_fp16, tail, fp8, True)
            if row_xcd
            else _get_kernel(splits, topk, xcds, quant, exact_fp16, tail, False, False, True)
            if prefetch
            else _get_kernel(splits, topk, xcds, quant, exact_fp16, tail, False, False, False, True)
            if fixed
            else _get_kernel(splits, topk, xcds, quant, exact_fp16, tail, fp8=True,
                             prefetch=fp8_prefetch, fixed=fp8_q8)
            if fp8
            else _get_kernel(splits, topk, xcds, quant, exact_fp16, tail)
        ),
        q.data_ptr(),
        kv_cache.data_ptr(),
        topk_idx.data_ptr(),
        block_table.data_ptr(),
        seq_lens.data_ptr(),
        output.data_ptr(),
        ks.data_ptr(),
        vs.data_ptr(),
        0 if exact_fp16 else nvfp4_e4m3_lut(q.device).data_ptr(),
        part.data_ptr(),
        cnt.data_ptr(),
        codes.data_ptr() if quant else 0,
        scales.data_ptr() if quant else 0,
        part.numel() * 4,
        rows,
        tile if quant else 0,
        decode_query_len,
        num_heads,
        num_kv_heads,
        topk_idx.stride(0),
        block_table.stride(0),
        kv_cache.stride(0),
        c,
        grid_x,
        heavy,
        tpx,
        *extra,
        torch.cuda.current_stream(q.device),
    )


# ---- warm-up (root README rule 4)

# the request counts the first-call warm-up walks (``amd_kernels.warm_up``): the variant follows from the row and
# request counts alone, so every count up to the TP4 batch limit; ``warm_up_decode`` walks a caller's own limit
WARM_MAX_REQS = 256


def _decode_mode(q, kv_cache, topk_idx, block_table, seq_lens, sm_scale, output, k_scale=None, v_scale=None,
                 decode_query_len=1, splits=None, xcd_count=None, quant_out=None, compute_type=torch.bfloat16,
                 tail_splits=None, index_score=None, index_max_seq_len=None):
    """What fixes the variants besides the request count (``amd_kernels.warm_up``)."""
    return (q.shape[1], kv_cache.shape[1] // 2, topk_idx.shape[-1], decode_query_len, quant_out is not None,
            compute_type, splits, xcd_count, tail_splits, index_score is not None, q.device.index)


def _decode_sizes(mode):
    return range(1, WARM_MAX_REQS + 1)


def _decode_inputs(mode, n, q, *args, **kwargs):
    """One call at ``n`` requests: zero pages, one live block per row."""
    heads, hkv, topk, ql, quant, compute, splits, xcds, tail, select, _ = mode
    dev = q.device
    rows = n * ql
    q0 = torch.zeros((rows, heads, _kernel.D), dtype=torch.bfloat16, device=dev)
    kv = torch.zeros((2, 2 * hkv, _kernel.BLK, 72), dtype=torch.uint8, device=dev)
    idx = torch.full((hkv, rows, topk), -1, dtype=torch.int32, device=dev)
    idx[..., 0] = 0
    one = torch.ones((), dtype=torch.float32, device=dev)
    quant_out = _quant_buffers(rows, heads * _kernel.D, dev) if quant else None
    return (
        (q0, kv, idx, torch.zeros((n, 1), dtype=torch.int32, device=dev),
         torch.full((n,), ql, dtype=torch.int32, device=dev), _kernel.D**-0.5, torch.empty_like(q0)),
        dict(k_scale=one, v_scale=one, decode_query_len=ql, splits=splits, xcd_count=xcds,
             quant_out=quant_out, compute_type=compute, tail_splits=tail,
             **(_index_score_args(hkv, rows, dev) if select else {})),
    )


def _index_score_args(hkv: int, rows: int, device) -> dict:
    """``index_score`` / ``index_max_seq_len`` of a warm-up call: one block per row (no variant depends on the
    context)."""
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import score_width

    return dict(index_score=torch.zeros((hkv, rows, score_width(1)), dtype=torch.float32, device=device),
                index_max_seq_len=1)


def _quant_buffers(rows: int, width: int, device) -> tuple[torch.Tensor, torch.Tensor, int]:
    """MXFP8 output buffers in the (16, 16) tile layout (the tile is a launch argument, not part of the variant)."""
    padded = -(-rows // 16) * 16
    codes = torch.empty((padded, width), dtype=torch.uint8, device=device)
    scales = torch.empty((-(-padded // 32) * 32, width // 32), dtype=torch.uint8, device=device)
    return codes, scales, 16


nvfp4_sparse_decode = entry(
    "attention/nvfp4_sparse_decode_gqa16_d128_p128_top16",
    mode=_decode_mode, sizes=_decode_sizes, inputs=_decode_inputs,
)(nvfp4_sparse_decode)


def warm_up_decode(
    device: torch.device,
    *,
    num_heads: int,
    num_kv_heads: int,
    query_len: int,
    max_reqs: int,
    quant_min_rows: int | None = None,
    index_select: bool = False,
    compute_type: torch.dtype = torch.bfloat16,
) -> None:
    """Compile every variant a server of ``max_reqs`` requests x ``query_len`` rows can launch (eager decode rows of
    any count, FULL decode graphs): one call per request count with the bf16 output, and with the MXFP8 output from
    ``quant_min_rows`` rows (None: never); ``index_select``: the ``index_score`` variants too (``index_decode``'s
    one-GPU route); ``compute_type``: ``nvfp4_sparse_decode``'s."""
    dev = torch.device(device)
    with torch.inference_mode():
        kv = torch.zeros((2, 2 * num_kv_heads, _kernel.BLK, 72), dtype=torch.uint8, device=dev)
        one = torch.ones((), dtype=torch.float32, device=dev)
        for reqs in range(1, max_reqs + 1):
            rows = reqs * query_len
            q = torch.zeros((rows, num_heads, _kernel.D), dtype=torch.bfloat16, device=dev)
            idx = torch.full((num_kv_heads, rows, 16), -1, dtype=torch.int32, device=dev)
            idx[..., 0] = 0
            bt = torch.zeros((reqs, 1), dtype=torch.int32, device=dev)
            seq = torch.full((reqs,), query_len, dtype=torch.int32, device=dev)
            out = torch.empty_like(q)
            quant = (None,)
            if quant_min_rows is not None and rows >= quant_min_rows:
                quant += (_quant_buffers(rows, num_heads * _kernel.D, dev),)
            selects = ({}, _index_score_args(num_kv_heads, rows, dev)) if index_select else ({},)
            for quant_out in quant:
                for sel in selects:
                    nvfp4_sparse_decode(q, kv, idx, bt, seq, _kernel.D**-0.5, out, k_scale=one, v_scale=one,
                                        decode_query_len=query_len, quant_out=quant_out, compute_type=compute_type,
                                        **sel)
