# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile caches and launch wrappers of the NVFP4 decode index scorers."""

import functools
from amd_kernels.warm_up import entry, launcher

import torch

from ..score_prefill.host import nvfp4_e4m3_lut

from . import kernel as _kernel
from ..host_stream import current_stream as _current_stream

# waves to aim for (3 per SIMD at 158 VGPRs): swept 1024-8192, cold L2, 16-100 requests: best or within 1 us everywhere
TARGET_WAVES = 3072


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _end_bytes(t: torch.Tensor) -> int:
    """Bytes from the data pointer to the end of the last element."""
    return (sum((n - 1) * s for n, s in zip(t.shape, t.stride())) + 1) * t.element_size()


# Scores go through LDS units and leave as whole rows from this footprint (scattered dword stores into 32 MB cost 17%
# at 16 rows x 512 requests x 128K, into 8 MB nothing), if the units still balance over the wave slots.
BUFFER_MIN_BYTES = 8 << 20
UNITS_PER_SLOT = 4


def _unit(max_blocks: int, slots: int, umax: int, footprint: int) -> int:
    """Consecutive blocks per unit (a power of two, 4..umax) such that every wave slot gets >= UNITS_PER_SLOT units,
    or 0: single strided positions and direct stores."""
    if footprint < BUFFER_MIN_BYTES:
        return 0
    u = umax
    while u > 4 and _cdiv(max_blocks, u) < UNITS_PER_SLOT * slots:
        u //= 2
    return u if _cdiv(max_blocks, u) >= UNITS_PER_SLOT * slots else 0


def _run_compiled(exe, *args, waves_per_eu: int | None = None):
    """The first call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction.
    ``waves_per_eu``: the occupancy the register allocator must leave room for (``--amdgpu-waves-per-eu``)."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        compile = flyc.compile[{"waves_per_eu": waves_per_eu}] if waves_per_eu else flyc.compile
        exe._cf = compile(exe, *args)
    else:
        cf(*args)


@functools.cache
@launcher
def _get_kernel(rows: int, heads: int, buffered: bool):
    return _kernel.compile_score_decode(rows, heads, buffered)


@functools.cache
@launcher
def _get_kernel_devpos(rows: int, heads: int):
    return _kernel.compile_score_decode(rows, heads, device_positions=True)


def nvfp4_index_score_decode(
    q_idx: torch.Tensor,  # [num_reqs * query_len, H, 128] e4m3
    index_cache: torch.Tensor,  # NVFP4 index pages [num_blocks, 128, 72] uint8
    score: torch.Tensor,  # [H, num_reqs * query_len, >= cdiv(max_seq_len, 128)] fp32
    block_table: torch.Tensor,  # [num_reqs, >= max_blocks] int32
    seq_lens: torch.Tensor,  # [num_reqs] int32
    init_blocks: int,
    local_blocks: int,
    query_len: int,
    max_seq_len: int,
) -> None:
    """The decode index scores (``reference.decode_scores_reference``) for blocks below ``cdiv(seq_len, 128)``, fp8
    keys."""
    total_q, num_heads, _ = q_idx.shape
    num_reqs = total_q // query_len
    if num_reqs == 0:
        return
    rows = num_heads * query_len
    max_blocks = max(1, _cdiv(max_seq_len, _kernel.BLK))
    # Chunk size from the work: num_reqs * max_blocks pages over ~TARGET_WAVES waves. A wave of a shared position gets
    # chunk / NW of its pages, so a chunk of NW * (pages per wave) requests; with at most one page per wave, chunks of
    # one request, each wave on its own positions.
    pages_per_wave = _cdiv(num_reqs * max_blocks, TARGET_WAVES)
    s_bytes = _end_bytes(score)
    assert s_bytes < 2**31 * 4, "score offsets must fit in int32 elements"
    footprint = num_reqs * rows * max_blocks * 4
    chunk = 1 if pages_per_wave <= 1 else _kernel.NW * pages_per_wave
    chunk_cap = _kernel.chunk_max(_kernel.padded_rows(rows))
    chunk = max(1, min(chunk, chunk_cap, num_reqs))
    nch = _cdiv(num_reqs, chunk)
    chunk = _cdiv(num_reqs, nch)
    # A workgroup's waves share its positions, a position's pages going round robin over them. With fewer requests per
    # chunk than waves, a wave takes its own positions: one shared position would hold fewer pages than waves, and
    # walking it costs a dependent table load per position.
    per_wave = chunk < _kernel.NW
    wanted = max(1, TARGET_WAVES // (_kernel.NW * nch))
    # past 16 rows: 2 waves per SIMD either way, and the buffered flush cost 10% (4 heads x 5 queries, 32K-128K)
    unit = (
        _unit(
            max_blocks,
            wanted * (_kernel.NW if per_wave else 1),
            _kernel.UMAX,
            footprint,
        )
        if _kernel.padded_rows(rows) <= 16
        else 0
    )
    units = _cdiv(max_blocks, max(1, unit))
    positions = _cdiv(units, _kernel.NW) if per_wave else units
    splits = max(1, min(positions, wanted))
    # per-wave chunks of 2-3 requests: shared positions where the device lengths fit one position per workgroup
    dev_pos = per_wave and chunk >= 2 and unit == 0
    _run_compiled(
        _get_kernel_devpos(rows, num_heads)
        if dev_pos
        else _get_kernel(rows, num_heads, unit > 0),
        q_idx.data_ptr(),
        index_cache.data_ptr(),
        nvfp4_e4m3_lut(q_idx.device).data_ptr(),
        score.data_ptr(),
        block_table.data_ptr(),
        seq_lens.data_ptr(),
        int(index_cache.stride(0)),
        int(q_idx.stride(0)),
        int(q_idx.stride(1)),
        _end_bytes(q_idx),
        int(score.stride(0)),
        int(score.stride(1)),
        s_bytes,
        int(block_table.stride(0)),
        _end_bytes(block_table),
        int(num_reqs),
        int(chunk),
        int(nch),
        int(splits),
        int(init_blocks),
        int(local_blocks),
        int(per_wave),
        int(unit),
        _current_stream(),
        # the buffered variant at the plain one's occupancy (154-169 VGPRs, its flush 169-171); past 16 rows a hint
        # spilled
        waves_per_eu=3 if unit > 0 and _kernel.padded_rows(rows) <= 16 else None,
    )


def _score_rows(q_idx, index_cache, score, block_table, seq_lens, init_blocks,
                local_blocks, query_len, max_seq_len):
    """The scores of ``nvfp4_index_score_top16``: one index head x 4 / 8 rows through 16 requests on the one-head
    scorers (two waves per page at 1 / 3 requests, a scalar first table entry at 2, sectors at 4 x 8 rows, independent
    scans otherwise), else ``nvfp4_index_score_decode``."""
    rows, heads, _ = q_idx.shape
    plain_head = heads == 1 and query_len in (4, 8) and rows > 0
    if plain_head and rows // query_len <= 16:
        from .score_d128_p128_h1 import compile_score
        max_blocks = max(1, _cdiv(max_seq_len, 128))
        reqs = rows // query_len
        page_coop = reqs in (1, 3)
        waves = 2 if page_coop else 1
        splits = min(max(1, TARGET_WAVES // (waves * reqs)), max_blocks)
        # A lane holds one strided page-table entry: a 1M context needs at most eleven per wave.
        assert _cdiv(max_blocks, splits) <= 64
        from .score_page_d128_p128_h1 import compile_score_page
        if reqs == 2:
            # XCD-local 128-B score lines at 8 rows only
            from .score_firstpage_d128_p128_h1 import compile_score_first_page
            exe = compile_score_first_page(query_len, query_len == 8)
        elif reqs == 4 and query_len == 8:
            from .score_sector_d128_p128_h1 import compile_score_sector
            exe = compile_score_sector(query_len)
        else:
            exe = compile_score_page(query_len) if page_coop else compile_score(query_len, waves)
        _run_compiled(
            exe, q_idx.data_ptr(), index_cache.data_ptr(),
            nvfp4_e4m3_lut(q_idx.device).data_ptr(), score.data_ptr(),
            block_table.data_ptr(), seq_lens.data_ptr(), int(index_cache.stride(0)),
            int(q_idx.stride(0)), int(score.stride(1)), _end_bytes(score),
            _end_bytes(block_table), max_blocks, splits, reqs, int(block_table.stride(0)), int(init_blocks),
            int(local_blocks), _current_stream(),
        )
    else:
        nvfp4_index_score_decode(
            q_idx, index_cache, score, block_table, seq_lens,
            init_blocks, local_blocks, query_len=query_len, max_seq_len=max_seq_len,
        )


def nvfp4_index_score_top16(
    q_idx: torch.Tensor,  # [num_reqs * query_len, H, 128] e4m3
    index_cache: torch.Tensor,  # NVFP4 index pages [num_blocks, 128, 72] uint8
    block_table: torch.Tensor,  # [num_reqs, >= max_blocks] int32
    seq_lens: torch.Tensor,  # [num_reqs] int32
    init_blocks: int,
    local_blocks: int,
    topk_idx: torch.Tensor,  # [H, num_reqs * query_len, 16] int32
    *,
    query_len: int,
    max_seq_len: int,
    score: torch.Tensor | None = None,  # [H, rows, >= score_width] fp32 workspace
) -> None:
    """The decode top-16 block ids of every (head, row) in the plain block order, in two launches: the scores
    (``nvfp4_index_score_rows``) then the selection (``nvfp4_index_top16_from_scores``). ``score``: a workspace
    (contents unspecified after); None allocates one. Graph-safe (lengths are read on the device)."""
    from ..reference import score_width

    if score is None:
        score = torch.empty(
            q_idx.shape[1],
            q_idx.shape[0],
            score_width(max_seq_len),
            dtype=torch.float32,
            device=q_idx.device,
        )
    _score_rows(
        q_idx, index_cache, score, block_table, seq_lens, init_blocks, local_blocks,
        query_len, max_seq_len,
    )
    nvfp4_index_top16_from_scores(
        score, seq_lens, topk_idx, rows=q_idx.shape[0], query_len=query_len,
        max_seq_len=max_seq_len,
    )


def nvfp4_index_top16_from_scores(
    score: torch.Tensor,  # [H, rows, >= score_width] fp32 of nvfp4_index_score_rows
    seq_lens: torch.Tensor,  # [num_reqs] int32
    topk_idx: torch.Tensor,  # [H, rows, 16] int32, written
    *,
    rows: int,  # query rows (num_reqs * query_len)
    query_len: int,
    max_seq_len: int,
) -> None:
    """The selection half of ``nvfp4_index_score_top16``, one launch: the plain-order four-wave selector through 128
    rows at one head x 4 / 8 rows, else ``topk.index_topk_decode_indices``. On the scores of
    ``nvfp4_index_score_rows`` the ids equal the entry's bit for bit (for a consumer that cannot run
    ``row_top16_prologue_ids``)."""
    from ..reference import score_width
    from ..topk import index_topk_decode_indices

    heads, rows = int(topk_idx.shape[0]), int(rows)
    plain_head = heads == 1 and query_len in (4, 8) and rows > 0
    if plain_head and rows <= 128:
        _run_compiled(
            _get_top16_plain_wave4(), score.data_ptr(), int(score.stride(0)),
            int(score.stride(1)), score_width(max_seq_len), seq_lens.data_ptr(),
            int(query_len), topk_idx.data_ptr(), int(topk_idx.stride(0)),
            int(topk_idx.stride(1)), int(rows), int(heads), _current_stream(),
        )
    else:
        index_topk_decode_indices(
            score, topk_idx, seq_lens, max_seq_len=max_seq_len, query_len=query_len
        )


@functools.cache
@launcher
def _get_top16_plain_wave4():
    from ..topk.top16_indices_p128_plain_wave4 import compile_top16_indices_wave4
    return compile_top16_indices_wave4()


def _top16_mode(q_idx, index_cache, block_table, seq_lens, init_blocks, local_blocks,
                topk_idx, *, query_len, max_seq_len, score=None):
    return q_idx.shape[1], query_len, q_idx.device.index


def _top16_sizes(mode):
    # requests: 1 / 3 two waves per page, 2 first page, 4 sectors (8 rows), 8 independent scans (5-16 requests); with
    # the larger sizes they also cover the other modes' chunked variants (device positions, direct and buffered
    # stores) and both selectors
    return (1, 2, 3, 4, 8, 32, 64, 128, 256)


def _top16_inputs(mode, n, q_idx, index_cache, block_table, seq_lens, init_blocks,
                  local_blocks, topk_idx, *, query_len, max_seq_len, score=None):
    heads, ql, _ = mode
    rows = n * ql
    dev = q_idx.device
    hmax = 1 << 20
    # Compile-only: two zero pages suffice. No served page data is copied.
    args = (
        torch.zeros(rows, heads, 128, dtype=torch.float8_e4m3fn, device=dev),
        torch.zeros(2, 128, 72, dtype=torch.uint8, device=dev),
        torch.zeros(n, hmax // 128, dtype=torch.int32, device=dev),
        torch.full((n,), ql, dtype=torch.int32, device=dev),
        init_blocks, local_blocks,
        torch.empty(heads, rows, 16, dtype=torch.int32, device=dev),
    )
    return args, dict(query_len=ql, max_seq_len=hmax)


nvfp4_index_score_top16 = entry(
    "indexer/nvfp4_index_score_top16_d128_p128", mode=_top16_mode,
    sizes=_top16_sizes, inputs=_top16_inputs,
)(nvfp4_index_score_top16)


def nvfp4_index_score_rows(
    q_idx: torch.Tensor,  # [num_reqs * query_len, H, 128] e4m3
    index_cache: torch.Tensor,  # NVFP4 index pages [num_blocks, 128, 72] uint8
    score: torch.Tensor,  # [H, rows, >= score_width] fp32
    block_table: torch.Tensor,  # [num_reqs, >= max_blocks] int32
    seq_lens: torch.Tensor,  # [num_reqs] int32
    init_blocks: int,
    local_blocks: int,
    *,
    query_len: int,
    max_seq_len: int,
) -> None:
    """The scores ``nvfp4_index_score_top16`` selects from, in one launch, for a consumer kernel that runs the top-16
    in its own prologue (``topk.top16_indices_p128_plain_wave4.row_top16_prologue_ids``). Bitwise
    ``nvfp4_index_score_decode``'s scores below the row's block count (all a top-16 reads); entries past it
    unspecified. Graph-safe (lengths are read on the device)."""
    _score_rows(
        q_idx, index_cache, score, block_table, seq_lens, init_blocks, local_blocks,
        query_len, max_seq_len,
    )


def _rows_mode(q_idx, index_cache, score, block_table, seq_lens, init_blocks,
               local_blocks, *, query_len, max_seq_len):
    return q_idx.shape[1], query_len, q_idx.device.index


def _rows_inputs(mode, n, q_idx, index_cache, score, block_table, seq_lens,
                 init_blocks, local_blocks, *, query_len, max_seq_len):
    from ..reference import score_width

    (q, cache, table, seq, init, local, _), kw = _top16_inputs(
        mode, n, q_idx, index_cache, block_table, seq_lens, init_blocks,
        local_blocks, None, query_len=query_len, max_seq_len=max_seq_len,
    )
    heads, rows = q.shape[1], q.shape[0]
    work = torch.empty(heads, rows, score_width(kw["max_seq_len"]), dtype=torch.float32,
                       device=q.device)
    return (q, cache, work, table, seq, init, local), kw


nvfp4_index_score_rows = entry(
    "indexer/nvfp4_index_score_rows_d128_p128", mode=_rows_mode,
    sizes=_top16_sizes, inputs=_rows_inputs,
)(nvfp4_index_score_rows)
