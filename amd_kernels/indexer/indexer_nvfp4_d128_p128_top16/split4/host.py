# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Buffers and launches of the decode indexer split over four GPUs (TP4, gfx950).

Each object is made by a collective (every rank at the same point, outside graph capture) and owns its buffers in
uncached memory mapped by the peers, plus device call counters (graph-safe).

- ``IndexSplit4`` (the decode buffers): the score receive buffer [max_rows, max_width] fp32 (this rank's head), the
  index-q receive buffer [4, max_rows, 128] e4m3 and the flag words.
- ``IndexSplit4RankOwned`` (rank-owned cache: rank r holds the blocks b % 4 == r, table entry p at slot p // 4):
  those buffers; rank r scores its blocks for all four heads (``decode_score.py``).
"""

import functools
import logging

import torch
import torch.distributed as dist
from amd_kernels.warm_up import entry, launcher

from .. import hip_ipc
from . import kernel as _kernel
from ..host_stream import current_stream as _current_stream

logger = logging.getLogger(__name__)

WORLD = _kernel.WORLD


def _run_compiled(exe, *args):
    import flydsl.compiler as flyc

    # a held launcher is kept: a warm-up (amd_kernels.warm_up) compiles in compile-only mode and stops held
    # handles itself
    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def _end_bytes(t: torch.Tensor) -> int:
    """Bytes from the data pointer to the end of the last element"""
    return (
        sum((n - 1) * s for n, s in zip(t.shape, t.stride())) + 1
    ) * t.element_size()


class _DevicePointer:
    """A raw device allocation as a tensor (``__cuda_array_interface__``)."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


def _exchange(obj, rank: int, group) -> list:
    out = [[None] for _ in range(WORLD)]
    out[rank][0] = obj
    ranks = sorted(dist.get_process_group_ranks(group=group))
    for i, r in enumerate(ranks):
        dist.broadcast_object_list(out[i], src=r, group=group, device="cpu")
    return [o[0] for o in out]


class IndexSplit4:
    def __init__(
        self, rank: int, group, device: torch.device, max_rows: int, max_width: int
    ):
        self.rank, self.device = rank, device
        self.max_rows, self.max_width = max_rows, max_width
        score_bytes = max_rows * max_width * 4
        iq_bytes = WORLD * max_rows * _kernel.D
        mine = [
            hip_ipc.allocate_shared_buffer_and_handle(n)
            for n in (score_bytes, iq_bytes, _kernel.FLAG_BYTES)
        ]
        peers = _exchange([h for _, h in mine], rank, group)
        ptrs = [
            [p if r == rank else hip_ipc.open_mem_handle(peers[r][k]) for r in range(WORLD)]
            for k, (p, _) in enumerate(mine)
        ]
        self.score_ptrs, self.iq_ptrs, self.flag_ptrs = ptrs
        self.score_rcv = torch.as_tensor(
            _DevicePointer(self.score_ptrs[rank], score_bytes), device=device
        ).view(torch.float32)
        flags = torch.as_tensor(
            _DevicePointer(self.flag_ptrs[rank], _kernel.FLAG_BYTES), device=device
        )
        flags.zero_()
        # per-workgroup call counts (the flag values), equal on every rank
        self.counts = torch.zeros(_kernel.CNT_WORDS, dtype=torch.int32, device=device)
        torch.accelerator.synchronize(device)
        dist.barrier(group=group)


@functools.cache
@launcher
def _score_push_rank_owned_kernel(query_len: int, topk: bool, small: bool):
    from . import decode_score

    return decode_score.compile_score_push(query_len, topk, small)


def nvfp4_index_score_decode_rank_owned_push(
    iq: torch.Tensor,  # this rank's index head [num_reqs * query_len, 128] e4m3
    index_cache: torch.Tensor,  # NVFP4 index pages [num_blocks, 128, 72] uint8
    dests: list[int],  # head h's scores [rows, row_stride] fp32 (rank h's buffer)
    row_stride: int,
    dest_bytes: int,
    block_table: torch.Tensor,  # [num_reqs, >= max_blocks] int32
    seq_lens: torch.Tensor,  # [num_reqs] int32
    init_blocks: int,
    local_blocks: int,
    rank: int,
    counts: torch.Tensor,  # the device counters
    flags: list[int],  # rank p's flag words
    iq_bufs: list[int],  # rank p's index-q receive buffer [4, iq_rows, 128] e4m3
    iq_rows: int,
    grid: int = 0,
    query_len: int = 4,
    sharded: bool = True,
    topk_idx: torch.Tensor | None = None,
    max_seq_len: int = 0,
) -> None:
    """Rank ``rank``'s blocks (b % 4 == rank) of all four heads' decode scores, bit for bit the single-head scorer's,
    stored straight into the heads' buffers ``dests`` in the rank-owned order (block 64 c + 4 j + r at 64 c + 16 r +
    j), the four heads' index-q rows exchanged on the way; returns once every rank's scores have arrived
    (``decode_score.py``). ``sharded``: ``index_cache`` holds only this rank's pages, table entry p at slot p // 4.
    ``topk_idx`` [1, rows, 16] int32: also this rank's head's top 16 per row in the same launch, bit for bit
    ``index_topk_decode_indices(..., rank_owned_order=True)`` over ``max_seq_len``'s score width."""
    from ..score_prefill.host import nvfp4_e4m3_lut
    from . import decode_score as ds

    rows = iq.shape[0]
    assert 1 <= query_len <= ds.MAX_QL
    assert rows % query_len == 0 and rows <= iq_rows
    assert iq.dtype == torch.float8_e4m3fn and iq.shape[1] == ds.D
    assert iq.stride(1) == 1 and iq.stride(0) == ds.D
    num_reqs = rows // query_len
    assert num_reqs <= ds.MAX_REQS
    if not grid:
        grid = ds.grid_for(num_reqs, query_len)
    assert 0 < grid <= ds.SC_MAX_WG
    if num_reqs == 0:
        return
    assert dest_bytes < 2**31 * 4, "score offsets must fit in int32 elements"
    tk_ptr = tk_row = tk_width = 0
    if topk_idx is not None:
        assert topk_idx.dtype == torch.int32 and topk_idx.stride(2) == 1
        assert topk_idx.shape == (1, rows, 16)
        slots = 1 << ((-(-max_seq_len // (ds.BLK * 64))) - 1).bit_length()
        tk_width = slots * 64
        assert 0 < max_seq_len and tk_width <= row_stride
        # the selecting workgroups (one per row, the last ones; one more collects) spin until their rows are
        # complete: few enough to stay resident beside the rest (and a DBO set on the other stream)
        assert grid >= 2 and min(grid - 1, rows) <= 128, rows
        tk_ptr, tk_row = topk_idx.data_ptr(), int(topk_idx.stride(1))
    _run_compiled(
        _score_push_rank_owned_kernel(query_len, topk_idx is not None, num_reqs <= 32),
        iq.data_ptr(),
        index_cache.data_ptr(),
        nvfp4_e4m3_lut(iq.device).data_ptr(),
        *dests,
        block_table.data_ptr(),
        seq_lens.data_ptr(),
        counts.data_ptr(),
        *flags,
        *iq_bufs,
        int(iq_rows),
        int(index_cache.stride(0)),
        int(row_stride),
        int(dest_bytes),
        int(block_table.stride(0)),
        _end_bytes(block_table),
        int(num_reqs),
        int(init_blocks),
        int(local_blocks),
        int(rank),
        2 if sharded else 0,
        int(grid),
        tk_ptr,
        tk_row,
        tk_width,
        _current_stream(),
    )


class IndexSplit4RankOwned(IndexSplit4):
    """The rank-owned index cache's decode indexer (``decode_score.py``), up to ``decode_score.MAX_QL`` rows per
    request; buffers as ``IndexSplit4``, plus the one-launch ``score_topk``'s receive buffer
    [min(max_rows, TK_MAX_ROWS), max_width] fp32, all ``decode_score.MARKER`` between calls."""

    def __init__(
        self,
        rank: int,
        group,
        device: torch.device,
        max_rows: int,
        max_width: int,
        query_len: int,
    ):
        from . import decode_score

        super().__init__(rank, group, device, max_rows, max_width)
        self.query_len = query_len  # 1 + the draft length
        nbytes = min(max_rows, decode_score.TK_MAX_ROWS) * max_width * 4
        ptr, handle = hip_ipc.allocate_shared_buffer_and_handle(nbytes)
        peers = _exchange(handle, rank, group)
        self.mark_ptrs = [ptr if r == rank else hip_ipc.open_mem_handle(peers[r]) for r in range(WORLD)]
        mark = torch.as_tensor(_DevicePointer(ptr, nbytes), device=device).view(torch.int32)
        mark.fill_(decode_score.MARKER)
        torch.accelerator.synchronize(device)
        dist.barrier(group=group)

    def score_push(
        self,
        iq: torch.Tensor,
        index_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        init_blocks: int,
        local_blocks: int,
        width: int,
        grid: int = 0,
        sharded: bool = True,
    ) -> torch.Tensor:
        """``nvfp4_index_score_decode_rank_owned_push`` into this object's buffers; returns this rank's head over
        every block in the rank-owned order, [1, rows, width] (a view of the receive buffer, valid until the next
        call). ``sharded`` False: the same on a replicated cache (tests)."""
        rows = iq.shape[0]
        assert rows <= self.max_rows and width <= self.max_width
        assert width % 64 == 0 and rows == seq_lens.numel() * self.query_len
        nvfp4_index_score_decode_rank_owned_push(
            iq,
            index_cache,
            self.score_ptrs,
            width,
            rows * width * 4,
            block_table,
            seq_lens,
            init_blocks,
            local_blocks,
            self.rank,
            self.counts,
            self.flag_ptrs,
            self.iq_ptrs,
            self.max_rows,
            grid=grid,
            query_len=self.query_len,
            sharded=sharded,
        )
        return self.score_rcv[: rows * width].view(1, rows, width)

    def score_topk(
        self,
        iq: torch.Tensor,
        index_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        init_blocks: int,
        local_blocks: int,
        width: int,
        topk_idx: torch.Tensor,
        max_seq_len: int,
        grid: int = 0,
        sharded: bool = True,
    ) -> torch.Tensor | None:
        """``score_push``, then this rank's head's top 16 per row into ``topk_idx`` [1, rows, 16] int32, bit for bit
        ``topk.index_topk_decode_indices(..., rank_owned_order=True)``: one launch up to ``decode_score.TK_MAX_ROWS``
        rows (the scorer selects in its tail, on a receive buffer of its own that it leaves all ``MARKER``; returns
        None), two above (returns the scores as ``score_push`` does)."""
        from . import decode_score as ds

        rows = iq.shape[0]
        assert rows <= self.max_rows and width <= self.max_width
        assert width % 64 == 0 and rows == seq_lens.numel() * self.query_len
        if rows > ds.TK_MAX_ROWS:
            from ..topk import index_topk_decode_indices

            score = self.score_push(
                iq,
                index_cache,
                block_table,
                seq_lens,
                init_blocks,
                local_blocks,
                width,
                grid=grid,
                sharded=sharded,
            )
            index_topk_decode_indices(
                score,
                topk_idx,
                seq_lens,
                max_seq_len=max_seq_len,
                block_size=ds.BLK,
                query_len=self.query_len,
                rank_owned_order=True,
            )
            return score
        nvfp4_index_score_decode_rank_owned_push(
            iq,
            index_cache,
            self.mark_ptrs,
            width,
            rows * width * 4,
            block_table,
            seq_lens,
            init_blocks,
            local_blocks,
            self.rank,
            self.counts,
            self.flag_ptrs,
            self.iq_ptrs,
            self.max_rows,
            grid=grid,
            query_len=self.query_len,
            sharded=sharded,
            topk_idx=topk_idx,
            max_seq_len=max_seq_len,
        )
        return None


def _rank_owned_top16_mode(cp, iq, *args, **kwargs):
    return cp.query_len, cp.max_rows // cp.query_len, cp.max_width, iq.device.index


def _rank_owned_top16_sizes(mode):
    ql, max_reqs, _, _ = mode
    # the prefix-sum capacity, the fused / standalone selection and the > 512-row selector edges
    edges = (1, 32, 33, 128 // ql, 128 // ql + 1, 512 // ql, 512 // ql + 1, 256, max_reqs)
    return sorted({min(n, max_reqs) for n in edges} - {0})


def _rank_owned_top16_inputs(mode, n, cp, iq, index_cache, block_table, seq_lens,
                            init_blocks, local_blocks, width, topk_idx, max_seq_len,
                            grid=0, sharded=True):
    ql, _, width, _ = mode
    rows, dev = n * ql, iq.device
    hmax = width * 128
    args = (
        cp,
        torch.zeros(rows, 128, dtype=torch.float8_e4m3fn, device=dev),
        index_cache,
        torch.zeros(n, width, dtype=torch.int32, device=dev),
        torch.full((n,), ql, dtype=torch.int32, device=dev),
        init_blocks, local_blocks, width,
        torch.empty(1, rows, 16, dtype=torch.int32, device=dev), hmax,
    )
    return args, dict(grid=grid, sharded=sharded)


IndexSplit4RankOwned.score_topk = entry(
    "indexer/nvfp4_index_score_top16_rank_owned_d128_p128",
    mode=_rank_owned_top16_mode, sizes=_rank_owned_top16_sizes,
    inputs=_rank_owned_top16_inputs,
)(IndexSplit4RankOwned.score_topk)


# one object per DBO microbatch (each microbatch's calls keep their own order on every rank); empty: not initialized
_RANK_OWNED: list[IndexSplit4RankOwned] = []


def init_rank_owned(
    rank: int,
    world_size: int,
    group,
    device: torch.device,
    max_width: int,
    max_reqs: int,
    query_len: int,
    sets: int,
) -> None:
    """Create ``sets`` of this rank's ``IndexSplit4RankOwned``s (one per microbatch) for ``max_reqs`` requests of
    ``query_len`` decode rows; a collective: every rank of ``group`` (a CPU (gloo) group of the four ranks) calls it at
    the same point, once, outside graph capture."""
    if _RANK_OWNED:
        return
    from . import decode_score

    assert world_size == WORLD
    assert 1 <= query_len <= decode_score.MAX_QL, query_len
    assert max_reqs <= decode_score.MAX_REQS, max_reqs
    rows = max_reqs * query_len
    for _ in range(sets):
        _RANK_OWNED.append(
            IndexSplit4RankOwned(rank, group, device, rows, max_width, query_len)
        )
    logger.info(
        "rank-owned index cache: %d x split decode buffers for %d rows (%d x %d) x %d blocks (%.1f MB per rank)",
        sets,
        rows,
        max_reqs,
        query_len,
        max_width,
        sets * rows * (max_width * 4 + WORLD * _kernel.D) / 2**20,
    )


def get_rank_owned(ubatch: int = 0) -> IndexSplit4RankOwned | None:
    """This rank's ``IndexSplit4RankOwned`` of DBO microbatch ``ubatch``, or None."""
    if not _RANK_OWNED:
        return None
    return _RANK_OWNED[ubatch]
