# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The communicator of the staged-partial TP4 all-reduce + add + Gemma RMSNorm kernels (one bf16 partial per rank, the
attention o_proj's): eager and unregistered graph calls stage the partials into uncached buffers; the optional
capture context registers peer inputs, which the faster graph tiers read in place."""

import contextlib
import functools

import torch
import torch.distributed as dist
from amd_kernels.warm_up import launcher

from amd_kernels.common import hip_runtime as ops
from ..add_rmsnorm.common import xcd_grid
from . import ar_add_rmsnorm_mxfp8_oneshot as _oneshot8
from . import ar_add_rmsnorm_mxfp8_stream as _stream8
from . import ar_add_rmsnorm_mxfp8_stream_pull as _stream8_pull
from . import ar_add_rmsnorm_oneshot_pull_ack as _oneshot_ack
from . import ar_add_rmsnorm_mxfp8_oneshot_pull_ack as _oneshot8_ack
from . import ar_add_rmsnorm_mxfp8_oneshot_pull_ack_cached_local as _oneshot8_ack_cached_local
from . import ar_add_rmsnorm_oneshot as _oneshot
from . import ar_add_rmsnorm_oneshot_owner_order as _oneshot_order
from . import ar_add_rmsnorm_oneshot_flags128_owner_order as _oneshot_flags128_order
from . import ar_add_rmsnorm_mxfp8_oneshot_owner_order as _oneshot8_order
from . import ar_add_rmsnorm_stream_lossless_scatter as _stream_lossless_scatter
from . import ar_add_rmsnorm_stream as _stream
from . import ar_add_rmsnorm_stream_pull as _stream_pull
from . import ar_add_rmsnorm_batch_pull as _batch_pull
from . import ar_add_rmsnorm_mxfp8_batch_pull as _batch8_pull
from . import ar_add_rmsnorm_mxfp8_stream_hoist_weight as _stream8_hoist
from . import ar_add_rmsnorm_register as _register
from . import ar_add_rmsnorm_mxfp8_register as _register8
from . import ar_add_rmsnorm_pull as _pull
from . import ar_add_rmsnorm_pull_lossless as _pull_lossless
from . import ar_add_rmsnorm_mxfp8_pull as _pull8
from .host import ArNormComm, H, WORLD, _bf16_kernel, _f32_bits, _out_or_empty, _run_compiled, exchange
from ..add_rmsnorm.host import code_tile_rows_for

# Every tier and range below is measured (benchmark_ar_add_rmsnorm_rows.py: graphs or eager as served, max over ranks,
# both capture orders, same output bytes).
ONESHOT_MAX_ROWS = 13  # the one-shot exchange beats the scoped-FTZ register kernel to 13 rows
assert ONESHOT_MAX_ROWS <= _oneshot.ONESHOT_ROWS
# the register two-stage kernel retains a row to here; above, rows stream under batched handshakes (registered
# graphs keep the ``ArNormComm`` kernel at 260-284 rows, rows % 4 == 0)
REGISTER_MAX_ROWS = 320
STREAM_MIN_ROWS = REGISTER_MAX_ROWS + 1
# larger prefills take the sequence-parallel hand-offs (the stream kernels' buffers hold up to _stream.MAX_ROWS - 1)
STREAM_MAX_ROWS = 4095
assert STREAM_MAX_ROWS < _stream.MAX_ROWS
# registered inputs from 288 rows: the streamed kernel reads them in place, no staging copy (loses at 257-272)
STREAM_PULL_MIN_ROWS = 288
STREAM_PULL_MAX_ROWS = STREAM_MAX_ROWS
STREAM_PULL_HOIST_BF16_RANGES = ((288, 944),)  # the in-place kernel loads the weight once (2048 rows lose)
# MXFP8 out: one-shot, register two-stage, streamed two-stage; row-major codes always take the streamed kernel
ONESHOT_MAX_ROWS_MXFP8 = 15
REGISTER_MAX_ROWS_MXFP8 = 256
STREAM_MIN_ROWS_MXFP8 = REGISTER_MAX_ROWS_MXFP8 + 1
STREAM_PULL_MIN_ROWS_MXFP8 = STREAM_MIN_ROWS_MXFP8
STREAM_PULL_MAX_ROWS_MXFP8 = STREAM_MAX_ROWS
STREAM_PULL_HOIST_MXFP8_TILED_RANGES = ((257, 288), (481, 536), (641, 768), (1281, 1536), (2049, 2816))
STREAM_PULL_HOIST_MXFP8_ROW_RANGES = ((1281, 1536),)
STREAM_PULL_ROWMAJOR_RANGES = ((1024, 2048), (3073, 4095))
assert ONESHOT_MAX_ROWS_MXFP8 <= _oneshot8.ONESHOT_ROWS

# the staged bf16 stream codes both exchanges (same bytes) at its own rows per workgroup, the smallest whose grid stays
# resident
LOSSLESS_SCATTER_MAX_GRID = 256

# registered inputs are complete at launch: one arrival and one publication per workgroup batch, where it wins
BATCH_PULL_BF16_RANGES = ((641, 1024), (1281, 2048))
BATCH_PULL_MXFP8_TILED_RANGES = ((641, 1024), (1281, 4095))
BATCH_PULL_MXFP8_ROW_RANGES = ((1281, 1536), (3073, 4095))

# the staged row-major MXFP8 stream loads the read-only weight once per batch here (sum and quant unchanged)
HOIST_MXFP8_ROW_RANGES = ((32, 1024), (2049, 3072))


def _hoist_weight(rows, ranges):
    return any(first <= rows <= last for first,last in ranges)


# registered graphs: the pull's all-gather as the lossless 12-bit code (0.781x the bf16 bytes), wins from 128 rows
LOSSLESS_BF16_MIN_ROWS = 128
LOSSLESS_BF16_MAX_ROWS = 256
assert LOSSLESS_BF16_MAX_ROWS <= _pull_lossless.MAX_LOSSLESS_ROWS


# registered graphs: the one-shot reads the peers' inputs in place and acks the reads (below the two-stage crossover)
ONESHOT_PULL_ACK_MAX_ROWS = 11
ONESHOT_PULL_ACK_MAX_ROWS_MXFP8 = 13


def _disjoint_input_outputs(x, *outputs):
    """No output overlaps ``x``: the direct readers ack only after the norm stores start, so an overlap takes the
    staged path."""
    start, end = x.data_ptr(), x.data_ptr() + x.numel() * x.element_size()
    return all(t.data_ptr() >= end or t.data_ptr() + t.numel() * t.element_size() <= start
               for t in outputs)


@functools.cache
def _oneshot_ack_kernel():
    return _oneshot_ack.compile_ar_add_rmsnorm_oneshot_pull_ack()


@functools.cache
def _oneshot8_ack_cached_local_kernel():
    return _oneshot8_ack_cached_local.compile_ar_add_rmsnorm_mxfp8_oneshot_pull_ack_cached_local()


@functools.cache
def _oneshot8_ack_kernel():
    return _oneshot8_ack.compile_ar_add_rmsnorm_mxfp8_oneshot_pull_ack()


@functools.cache
def _batch_pull_kernel(rpw, hoist=False):
    return _batch_pull.compile_ar_add_rmsnorm_batch_pull(rpw, hoist)


@functools.cache
def _batch8_pull_kernel(rpw, tiled_q, hoist=False):
    return _batch8_pull.compile_ar_add_rmsnorm_mxfp8_batch_pull(rpw, tiled_q, hoist)


@functools.cache
def _stream_pull_kernel(rpw, hoist=False):
    return _stream_pull.compile_ar_add_rmsnorm_stream_pull(rpw, hoist)


@functools.cache
def _stream8_pull_kernel(rpw, tiled_q, hoist=False):
    return _stream8_pull.compile_ar_add_rmsnorm_mxfp8_stream_pull(rpw, tiled_q, hoist)


@functools.cache
def _stream8_hoist_kernel(rpw):
    return _stream8_hoist.compile_ar_add_rmsnorm_mxfp8_stream_hoist_weight(rpw)


@functools.cache
def _oneshot_kernel(wide_flags=False, owner_order=False):
    if wide_flags:
        return _oneshot_flags128_order.compile_ar_add_rmsnorm_oneshot_flags128_owner_order()
    if owner_order:
        return _oneshot_order.compile_ar_add_rmsnorm_oneshot_owner_order()
    return _oneshot.compile_ar_add_rmsnorm_oneshot()


@functools.cache
@launcher
def _stream_lossless_scatter_kernel(rpw):
    return _stream_lossless_scatter.compile_ar_add_rmsnorm_stream_lossless_scatter(rpw)


@functools.cache
def _oneshot8_kernel(owner_order=False):
    if owner_order:
        return _oneshot8_order.compile_ar_add_rmsnorm_mxfp8_oneshot_owner_order()
    return _oneshot8.compile_ar_add_rmsnorm_mxfp8_oneshot()


@functools.cache
def _stream8_kernel(rpw: int, tiled_q: bool):
    return _stream8.compile_ar_add_rmsnorm_mxfp8_stream(rpw, tiled_q)


@functools.cache
def _register_kernel(wave: bool, scoped: bool):
    return _register.compile_ar_add_rmsnorm_register(wave, scoped)


@functools.cache
def _register8_kernel(wave: bool, scoped: bool):
    return _register8.compile_ar_add_rmsnorm_mxfp8_register(wave, scoped)


@functools.cache
def _pull_kernel(wave: bool):
    return _pull.compile_ar_add_rmsnorm_pull(wave)


@functools.cache
def _pull_lossless_kernel():
    return _pull_lossless.compile_ar_add_rmsnorm_pull_lossless()


@functools.cache
def _pull8_kernel(wave: bool):
    return _pull8.compile_ar_add_rmsnorm_mxfp8_pull(wave)


class StagedArNormComm:
    """A collective on the TP CPU group, once per rank: uncached buffers per rank (staged partials and flags) mapped by
    the peers, and per-row call counts per flag domain (the same on every rank: the ranks make the same calls in the
    same order). One communicator per stream."""

    def __init__(self, rank: int, group, device: torch.device):
        self.rank, self.group, self.device = rank, group, device
        self._inplace = None
        ptr, handle = ops.allocate_shared_buffer_and_handle(max(_oneshot.ONESHOT_BUF_BYTES,_oneshot_flags128_order.ONESHOT_BUF_BYTES))
        sptr, shandle = ops.allocate_shared_buffer_and_handle(_stream.STREAM_BUF_BYTES)
        optr, ohandle = ops.allocate_shared_buffer_and_handle(_oneshot8.ONESHOT_BUF_BYTES)
        rptr, rhandle = ops.allocate_shared_buffer_and_handle(_register.REGISTER_BUF_BYTES)
        rawptr, rawhandle = ops.allocate_shared_buffer_and_handle((_stream.MAX_ROWS // WORLD) * H * 2)
        inrawptr, inrawhandle = ops.allocate_shared_buffer_and_handle(_stream.MAX_ROWS * H * 2)
        peers = exchange((handle, shandle, ohandle, rhandle, rawhandle, inrawhandle), rank, group)
        self.bufs = [ptr if r == rank else ops.open_mem_handle(peers[r][0]) for r in range(WORLD)]
        self.stream_bufs = [sptr if r == rank else ops.open_mem_handle(peers[r][1]) for r in range(WORLD)]
        self.bufs8 = [optr if r == rank else ops.open_mem_handle(peers[r][2]) for r in range(WORLD)]
        self.register_bufs = [rptr if r == rank else ops.open_mem_handle(peers[r][3]) for r in range(WORLD)]
        self.stream_raw_bufs = [rawptr if r == rank else ops.open_mem_handle(peers[r][4]) for r in range(WORLD)]
        self.stream_raw_table = torch.tensor(self.stream_raw_bufs, dtype=torch.int64, device=device)
        self.stream_input_raw_bufs = [inrawptr if r == rank else ops.open_mem_handle(peers[r][5]) for r in range(WORLD)]
        self.stream_input_raw_table = torch.tensor(self.stream_input_raw_bufs, dtype=torch.int64, device=device)
        self.register_counts = torch.zeros(_register.MAX_ROWS, dtype=torch.int32, device=device)
        self.counts8 = torch.zeros(_oneshot8.MAX_ROWS, dtype=torch.int32, device=device)
        # one count array per flag domain: a long run of one protocol must not push another's flags past the signed
        # comparison window
        self.counts = torch.zeros(_oneshot.MAX_ROWS, dtype=torch.int32, device=device)
        self.ack_counts = torch.zeros(_oneshot.MAX_ROWS, dtype=torch.int32, device=device)
        self.ack_counts8 = torch.zeros(_oneshot8.MAX_ROWS, dtype=torch.int32, device=device)
        self.stream_counts = torch.zeros(_stream.MAX_ROWS, dtype=torch.int32, device=device)
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    oneshot_max_rows = ONESHOT_MAX_ROWS  # tests / benchmark may move the tiers
    stream_min_rows = STREAM_MIN_ROWS
    oneshot_max_rows_mxfp8 = ONESHOT_MAX_ROWS_MXFP8
    stream_min_rows_mxfp8 = STREAM_MIN_ROWS_MXFP8
    oneshot_pull_ack_max_rows = ONESHOT_PULL_ACK_MAX_ROWS
    oneshot_pull_ack_max_rows_mxfp8 = ONESHOT_PULL_ACK_MAX_ROWS_MXFP8
    oneshot_ack_cached_local_max_rows_mxfp8 = 4  # the cached own-input read wins at 1-4 rows
    graph_min_rows = 11
    graph_min_rows_mxfp8 = 14
    register_max_rows = REGISTER_MAX_ROWS
    register_max_rows_mxfp8 = REGISTER_MAX_ROWS_MXFP8
    lossless_min_rows = LOSSLESS_BF16_MIN_ROWS  # tests / benchmark may move the lossless tier
    lossless_max_rows = LOSSLESS_BF16_MAX_ROWS
    stream_pull_min_rows = STREAM_PULL_MIN_ROWS  # tests / benchmark may move the registered stream tier
    stream_pull_max_rows = STREAM_PULL_MAX_ROWS
    batch_pull_ranges = BATCH_PULL_BF16_RANGES
    batch_pull_mxfp8_tiled_ranges = BATCH_PULL_MXFP8_TILED_RANGES
    batch_pull_mxfp8_row_ranges = BATCH_PULL_MXFP8_ROW_RANGES
    stream_pull_hoist_ranges = STREAM_PULL_HOIST_BF16_RANGES
    stream_pull_min_rows_mxfp8 = STREAM_PULL_MIN_ROWS_MXFP8
    stream_pull_max_rows_mxfp8 = STREAM_PULL_MAX_ROWS_MXFP8
    stream_pull_rowmajor_ranges = STREAM_PULL_ROWMAJOR_RANGES
    stream_pull_hoist_mxfp8_tiled_ranges = STREAM_PULL_HOIST_MXFP8_TILED_RANGES
    stream_pull_hoist_mxfp8_row_ranges = STREAM_PULL_HOIST_MXFP8_ROW_RANGES

    @contextlib.contextmanager
    def capture(self):
        """Optional, around CUDA graph capture on every rank (enter before ``torch.cuda.graph``, same captures):
        registers the captured calls' inputs for the registered tiers; without it every call stages."""
        if self._inplace is None:
            self._inplace = ArNormComm(self.rank, self.group, self.device)
        with self._inplace.capture():
            yield

    def _registered_oneshot_ack(self, x, mxfp8=False, code_tile_rows=16):
        c = self._inplace
        last = self.oneshot_pull_ack_max_rows_mxfp8 if mxfp8 else self.oneshot_pull_ack_max_rows
        return (c is not None and c.capturing and torch.cuda.is_current_stream_capturing()
                and x.is_contiguous() and 0 < x.shape[0] <= last
                and (not mxfp8 or code_tile_rows > 0))

    def _registered(self, x, mxfp8=False, code_tile_rows=16):
        c = self._inplace
        limit = self.register_max_rows_mxfp8 if mxfp8 else self.register_max_rows
        lo = self.graph_min_rows_mxfp8 if mxfp8 else self.graph_min_rows
        return (
            c is not None and c.capturing and torch.cuda.is_current_stream_capturing()
            and x.is_contiguous() and lo <= x.shape[0] <= limit
            and (not mxfp8 or code_tile_rows > 0)
        )

    def _registered_stream(self, x, mxfp8=False, code_tile_rows=16):
        """A registered graph call above the register tier: the streamed kernel reads the inputs in place."""
        c = self._inplace
        rows = x.shape[0]
        lo, hi = ((self.stream_pull_min_rows_mxfp8, self.stream_pull_max_rows_mxfp8) if mxfp8
                  else (self.stream_pull_min_rows, self.stream_pull_max_rows))
        in_range = lo <= rows <= hi and (
            not mxfp8 or code_tile_rows > 0 or _hoist_weight(rows, self.stream_pull_rowmajor_ranges))
        return (
            c is not None and c.capturing and torch.cuda.is_current_stream_capturing()
            and x.is_contiguous() and in_range
        )

    def pull_wave_handoffs(self, rows, mxfp8=False):
        """The registered pull's hand-offs: per wave at 37-96 rows (MXFP8 49-96) and above 256, per workgroup elsewhere
        (no staging stores to drain); its rank sum always runs under the scoped FP32 FTZ."""
        return (49 if mxfp8 else 37) <= rows <= 96 or rows > 256

    def supports(self, rows: int, mxfp8: bool = False) -> bool:
        """All-reduce row counts supported for either output, eager and graphs."""
        return 0 < rows <= STREAM_MAX_ROWS

    def accepts(self, x: torch.Tensor, residual: torch.Tensor) -> bool:
        """Whether ``ar_add_rmsnorm`` / ``ar_add_rmsnorm_mxfp8`` take these tensors (their
        asserts, without raising): bf16 [rows, H] with unit column stride and 16-byte rows."""
        return (
            x.dim() == 2
            and x.shape[1] == H
            and self.supports(x.shape[0])
            and residual.shape == x.shape
            and x.dtype == residual.dtype == torch.bfloat16
            and x.stride(1) == 1
            and residual.stride(1) == 1
            and x.stride(0) % 8 == 0
            and residual.stride(0) % 8 == 0
        )

    def register_config(self, rows):
        """(wave hand-offs, scoped FTZ) of the register kernels."""
        return rows >= 29, rows <= 96

    def rows_per_wg(self, rows: int) -> int:
        """The streamed kernel's rows per workgroup (measured): the grid stays near 128 workgroups, R 2 (<= 240
        workgroups) to 480 rows."""
        if rows <= 480:
            return 2
        if rows <= 640:
            return 4
        if rows <= 1280:
            return 8
        if rows <= 2048:
            return 16
        return 32

    def lossless_scatter_rows_per_wg(self, rows: int) -> int:
        """The two-phase coded stream's rows per workgroup: the smallest R >= 2 whose grid 4 xcd_grid(ceil(q / R))
        stays resident (<= 256 workgroups; 12 waves, 61 VGPRs and 2.3 KB LDS each, two per CU); fewer rows per
        workgroup won at every measured size."""
        q = (rows + WORLD - 1) // WORLD
        r = 2
        while WORLD * xcd_grid((q + r - 1) // r) > LOSSLESS_SCATTER_MAX_GRID:
            r *= 2
        return r

    def warm_up(self, weight: torch.Tensor, eps: float, code_tile_rows: int = 16) -> None:
        """Compile every variant before serving, both code layouts (a collective: every rank calls it;
        ``code_tile_rows``: the dense layers' tile)."""
        bf16_rows = {1, 7, 8, self.oneshot_max_rows, 14, 19, 29, 97, 321, 481, 641, 1281, 2049, STREAM_MAX_ROWS}
        # every rows-per-workgroup key of the coded stream
        bf16_rows.update(r for r in range(STREAM_MIN_ROWS + 1, STREAM_MAX_ROWS + 1)
                         if self.lossless_scatter_rows_per_wg(r) != self.lossless_scatter_rows_per_wg(r - 1))
        for rows in sorted(bf16_rows):
            x = torch.zeros((rows, H), dtype=torch.bfloat16, device=self.device)
            self.ar_add_rmsnorm(x, x, weight, eps)
        # the in-place kernels: each key compiles outside capture on the mapped staging buffers (valid peer memory,
        # outputs discarded)
        staged = torch.tensor([b + _stream.STG_OFF for b in self.stream_bufs], dtype=torch.int64, device=self.device)
        pull_rows = {}
        for rows in range(STREAM_PULL_MIN_ROWS, STREAM_PULL_MAX_ROWS + 1):
            pull_rows.setdefault((self.rows_per_wg(rows), _hoist_weight(rows, self.stream_pull_hoist_ranges),
                                  _hoist_weight(rows, self.batch_pull_ranges)), rows)
        for key, rows in sorted(pull_rows.items()):
            x = torch.zeros((rows, H), dtype=torch.bfloat16, device=self.device)
            y = torch.empty_like(x)
            _run_compiled(
                (_batch_pull_kernel if key[2] else _stream_pull_kernel)(*key[:2]),
                rows, (rows + WORLD - 1) // WORLD, self.rank, x.stride(0), x.stride(0), staged.data_ptr(),
                *self.stream_bufs, x.data_ptr(), weight.data_ptr(), y.data_ptr(), y.data_ptr(),
                self.stream_counts.data_ptr(), _f32_bits(eps), torch.cuda.current_stream(),
            )
        for tile in (0, code_tile_rows or 16):
            quant_rows = {1, 8, self.oneshot_max_rows_mxfp8, 16, 19, 29, 97, 257, 481, 641, 1281, 2049}
            if not tile:  # each row-major hoist range: its first row and the first row after it
                quant_rows.update(r for first,last in HOIST_MXFP8_ROW_RANGES
                                  for r in (first,last+1) if r <= STREAM_MAX_ROWS)
            for rows in sorted(quant_rows):
                x = torch.zeros((rows, H), dtype=torch.bfloat16, device=self.device)
                self.ar_add_rmsnorm_mxfp8(x, x, weight, eps, tile)
        for tile in (0, code_tile_rows or 16):
            keys = {}
            for rows in range(STREAM_PULL_MIN_ROWS_MXFP8, STREAM_PULL_MAX_ROWS_MXFP8 + 1):
                if tile or _hoist_weight(rows, self.stream_pull_rowmajor_ranges):
                    rpw = self.rows_per_wg(rows) if tile else max(8, self.rows_per_wg(rows))
                    ranges = self.stream_pull_hoist_mxfp8_tiled_ranges if tile else self.stream_pull_hoist_mxfp8_row_ranges
                    batch_ranges = self.batch_pull_mxfp8_tiled_ranges if tile else self.batch_pull_mxfp8_row_ranges
                    keys.setdefault((rpw, _hoist_weight(rows, ranges), _hoist_weight(rows, batch_ranges)), rows)
            for (rpw, hoist, batch), rows in sorted(keys.items()):
                x = torch.zeros((rows, H), dtype=torch.bfloat16, device=self.device)
                tiled = (rows + tile - 1) // tile * tile if tile else rows
                q = torch.empty((tiled, H), dtype=torch.float8_e4m3fn, device=self.device)
                sc = torch.empty(((tiled + 31) // 32 * 32, H // 32), dtype=torch.uint8, device=self.device)
                y = torch.empty_like(x)
                _run_compiled(
                    (_batch8_pull_kernel if batch else _stream8_pull_kernel)(rpw, tile > 0, hoist), rows, (rows + WORLD - 1) // WORLD, self.rank, x.stride(0),
                    x.stride(0), staged.data_ptr(), *self.stream_bufs, x.data_ptr(), weight.data_ptr(), y.data_ptr(),
                    q.data_ptr(), sc.data_ptr(), q.numel(), sc.numel(), self.stream_counts.data_ptr(), _f32_bits(eps),
                    torch.cuda.current_stream(),
                )
        # the registered one-shots: outside capture on the mapped buffers, no registration needed
        mapped = torch.tensor(self.bufs, dtype=torch.int64, device=self.device)
        x = torch.zeros((16, H), dtype=torch.bfloat16, device=self.device)
        y = torch.empty_like(x)
        q = torch.empty((16,H), dtype=torch.float8_e4m3fn, device=self.device)
        sc = torch.empty((32,H//32), dtype=torch.uint8, device=self.device)
        _run_compiled(_oneshot_ack_kernel(), 8, 2, self.rank, H, H, mapped.data_ptr(),
                      *self.bufs, x.data_ptr(), weight.data_ptr(), y.data_ptr(), y.data_ptr(),
                      self.ack_counts.data_ptr(), _f32_bits(eps), torch.cuda.current_stream())
        _run_compiled(_oneshot8_ack_kernel(), 8, 2, self.rank, H, H, mapped.data_ptr(),
                      *self.bufs, x.data_ptr(), weight.data_ptr(), y.data_ptr(), q.data_ptr(),
                      sc.data_ptr(), q.numel(), sc.numel(), self.ack_counts8.data_ptr(),
                      _f32_bits(eps), torch.cuda.current_stream())
        _run_compiled(_oneshot8_ack_cached_local_kernel(), 4, 1, self.rank, H, H, mapped.data_ptr(),
                      *self.bufs, x.data_ptr(), weight.data_ptr(), y.data_ptr(), q.data_ptr(),
                      sc.data_ptr(), q.numel(), sc.numel(), self.ack_counts8.data_ptr(),
                      _f32_bits(eps), torch.cuda.current_stream())
        with self.capture():
            graph = torch.cuda.CUDAGraph()
            x = torch.zeros((320, H), dtype=torch.bfloat16, device=self.device)
            with torch.cuda.graph(graph):
                for rows in (8, 16, 64):
                    self.ar_add_rmsnorm(x[:rows], x[:rows], weight, eps)
                    self.ar_add_rmsnorm_mxfp8(x[:rows], x[:rows], weight, eps, code_tile_rows or 16)
                self.ar_add_rmsnorm(x[:LOSSLESS_BF16_MIN_ROWS], x[:LOSSLESS_BF16_MIN_ROWS], weight, eps)
                self.ar_add_rmsnorm(x, x, weight, eps)
                # the ``ArNormComm`` kernel serves registered 260-284 rows (rows % 4 == 0)
                self.ar_add_rmsnorm(x[:284], x[:284], weight, eps)
        graph.replay()
        torch.cuda.synchronize(self.device)

    def ar_add_rmsnorm(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        out: torch.Tensor | None = None,
        res_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``add_rmsnorm.gemma_add_rmsnorm`` of all-reduce(x) + residual: (bf16 out, residual out); every rank passes
        its own bf16 [rows, H] partial (``reference.allreduce_rows_reference``)."""
        rows = x.shape[0]
        assert self.supports(rows), rows
        assert x.shape == (rows, H) and x.dtype == torch.bfloat16
        assert x.stride(1) == 1 and x.stride(0) % 8 == 0
        assert residual.shape == (rows, H) and residual.dtype == torch.bfloat16
        assert residual.stride(1) == 1 and residual.stride(0) % 8 == 0
        assert weight.dtype == torch.bfloat16 and weight.is_contiguous()
        out = _out_or_empty(out, (rows, H), torch.bfloat16, self.device)
        res_out = _out_or_empty(res_out, (rows, H), torch.bfloat16, self.device)
        if self._registered_oneshot_ack(x) and _disjoint_input_outputs(x, out, res_out):
            _run_compiled(_oneshot_ack_kernel(), rows, (rows + WORLD - 1) // WORLD,
                          self.rank, x.stride(0), residual.stride(0), self._inplace._row(x),
                          *self.bufs, residual.data_ptr(), weight.data_ptr(), out.data_ptr(),
                          res_out.data_ptr(), self.ack_counts.data_ptr(), _f32_bits(eps), torch.cuda.current_stream())
            return out, res_out
        if self._registered_stream(x):
            _run_compiled(
                (_batch_pull_kernel if _hoist_weight(rows, self.batch_pull_ranges) else _stream_pull_kernel)(
                    self.rows_per_wg(rows), _hoist_weight(rows, self.stream_pull_hoist_ranges)),
                rows, (rows + WORLD - 1) // WORLD, self.rank, x.stride(0), residual.stride(0),
                self._inplace._row(x), *self.stream_bufs, residual.data_ptr(), weight.data_ptr(),
                out.data_ptr(), res_out.data_ptr(), self.stream_counts.data_ptr(), _f32_bits(eps),
                torch.cuda.current_stream(),
            )
            return out, res_out
        if self._registered(x) and rows > 256 and rows % WORLD == 0:
            _run_compiled(
                _bf16_kernel(), rows // WORLD, self.rank, residual.stride(0), self._inplace._row(x),
                *self._inplace.bufs, residual.data_ptr(), weight.data_ptr(), out.data_ptr(), res_out.data_ptr(),
                self._inplace.counts.data_ptr(), _f32_bits(eps), torch.cuda.current_stream(),
            )
            return out, res_out
        if self._registered(x):
            lossless = self.lossless_min_rows <= rows <= self.lossless_max_rows
            _run_compiled(
                _pull_lossless_kernel() if lossless else _pull_kernel(self.pull_wave_handoffs(rows)),
                rows, (rows + WORLD - 1) // WORLD, self.rank, x.stride(0), residual.stride(0),
                self._inplace._row(x), *self.register_bufs, residual.data_ptr(), weight.data_ptr(),
                out.data_ptr(), res_out.data_ptr(), self.register_counts.data_ptr(), _f32_bits(eps),
                torch.cuda.current_stream(),
            )
            return out, res_out
        if self.oneshot_max_rows < rows <= self.register_max_rows:
            _run_compiled(
                _register_kernel(*self.register_config(rows)),
                rows, (rows + WORLD - 1) // WORLD, self.rank, x.stride(0), residual.stride(0),
                x.data_ptr(), *self.register_bufs, residual.data_ptr(), weight.data_ptr(),
                out.data_ptr(), res_out.data_ptr(), self.register_counts.data_ptr(), _f32_bits(eps),
                torch.cuda.current_stream(),
            )
            return out, res_out
        if rows > self.oneshot_max_rows:
            _run_compiled(
                _stream_lossless_scatter_kernel(self.lossless_scatter_rows_per_wg(rows)),
                rows,
                (rows + WORLD - 1) // WORLD,
                self.rank,
                x.stride(0),
                residual.stride(0),
                x.data_ptr(),
                *self.stream_bufs,
                residual.data_ptr(),
                weight.data_ptr(),
                out.data_ptr(),
                res_out.data_ptr(),
                self.stream_counts.data_ptr(),
                self.stream_raw_table.data_ptr(),
                self.stream_input_raw_table.data_ptr(),
                _f32_bits(eps),
                torch.cuda.current_stream(),
            )
            return out, res_out
        _run_compiled(
            _oneshot_kernel(3 <= rows <= 6 or rows == 8, 3 <= rows <= 13),
            rows,
            (rows + WORLD - 1) // WORLD,
            self.rank,
            x.stride(0),
            residual.stride(0),
            x.data_ptr(),
            *self.bufs,
            residual.data_ptr(),
            weight.data_ptr(),
            out.data_ptr(),
            res_out.data_ptr(),
            self.counts.data_ptr(),
            _f32_bits(eps),
            torch.cuda.current_stream(),
        )
        return out, res_out

    def ar_add_rmsnorm_mxfp8(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        code_tile_rows: int | None = None,
        q_out: torch.Tensor | None = None,
        scale_out: torch.Tensor | None = None,
        res_out: torch.Tensor | None = None,
        *,
        gemm_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``add_rmsnorm.gemma_add_rmsnorm_mxfp8`` of all-reduce(x) + residual; every rank passes its own bf16 [rows, H]
        partial.

        Returns (codes in the (16, 16) layout padded to ``code_tile_rows`` when > 0, row-major at 0, the tiled norm's
        arithmetic in both; AITER-layout scales; residual out). ``gemm_weight`` instead of ``code_tile_rows``: the
        layout that GEMM reads at this row count."""
        rows = x.shape[0]
        if gemm_weight is not None:
            code_tile_rows = code_tile_rows_for(rows, gemm_weight)
        assert code_tile_rows is not None
        assert self.supports(rows, True), rows
        assert x.shape == (rows, H) and x.dtype == torch.bfloat16
        assert x.stride(1) == 1 and x.stride(0) % 8 == 0
        assert residual.shape == (rows, H) and residual.dtype == torch.bfloat16
        assert residual.stride(1) == 1 and residual.stride(0) % 8 == 0
        assert weight.dtype == torch.bfloat16 and weight.is_contiguous()
        assert code_tile_rows >= 0 and code_tile_rows % 16 == 0
        tiled = (rows + code_tile_rows - 1) // code_tile_rows * code_tile_rows if code_tile_rows else rows
        q = _out_or_empty(q_out, (tiled, H), torch.float8_e4m3fn, self.device)
        scale = _out_or_empty(scale_out, ((tiled + 31) // 32 * 32, H // 32), torch.uint8, self.device)
        res_out = _out_or_empty(res_out, (rows, H), torch.bfloat16, self.device)
        common = (rows, (rows + WORLD - 1) // WORLD, self.rank, residual.stride(0), x.stride(0), x.data_ptr())
        tail = (
            residual.data_ptr(),
            weight.data_ptr(),
            res_out.data_ptr(),
            q.data_ptr(),
            scale.data_ptr(),
            q.numel(),
            scale.numel(),
        )
        if self._registered_oneshot_ack(x, True, code_tile_rows) and _disjoint_input_outputs(x, q, scale, res_out):
            ack_kernel = (_oneshot8_ack_cached_local_kernel if rows <= self.oneshot_ack_cached_local_max_rows_mxfp8
                          else _oneshot8_ack_kernel)
            _run_compiled(ack_kernel(), rows, (rows + WORLD - 1) // WORLD,
                          self.rank, residual.stride(0), x.stride(0), self._inplace._row(x),
                          *self.bufs, *tail, self.ack_counts8.data_ptr(), _f32_bits(eps), torch.cuda.current_stream())
            return q, scale, res_out
        if self._registered(x, True, code_tile_rows):
            registered_common = (rows, (rows + WORLD - 1) // WORLD, self.rank, residual.stride(0),
                                 x.stride(0), self._inplace._row(x))
            _run_compiled(
                _pull8_kernel(self.pull_wave_handoffs(rows, True)), *registered_common, *self.register_bufs, *tail,
                self.register_counts.data_ptr(), _f32_bits(eps), torch.cuda.current_stream(),
            )
        elif code_tile_rows and rows <= self.oneshot_max_rows_mxfp8:
            _run_compiled(
                _oneshot8_kernel(7 <= rows <= 15), *common, *self.bufs8, *tail, self.counts8.data_ptr(), _f32_bits(eps),
                torch.cuda.current_stream(),
            )
        elif code_tile_rows and rows <= self.register_max_rows_mxfp8:
            _run_compiled(
                _register8_kernel(*self.register_config(rows)), *common, *self.register_bufs, *tail,
                self.register_counts.data_ptr(), _f32_bits(eps), torch.cuda.current_stream(),
            )
        elif self._registered_stream(x, True, code_tile_rows):
            stream_row = self._inplace._row(x)
            rpw = self.rows_per_wg(rows)
            if not code_tile_rows:
                rpw = max(8, rpw)  # the row-major codes: 8+ rows per workgroup
            hoist_ranges = self.stream_pull_hoist_mxfp8_tiled_ranges if code_tile_rows else self.stream_pull_hoist_mxfp8_row_ranges
            batch_ranges = self.batch_pull_mxfp8_tiled_ranges if code_tile_rows else self.batch_pull_mxfp8_row_ranges
            _run_compiled(
                (_batch8_pull_kernel if _hoist_weight(rows, batch_ranges) else _stream8_pull_kernel)(
                    rpw, code_tile_rows > 0, _hoist_weight(rows, hoist_ranges)),
                *common[:-1], stream_row,
                *self.stream_bufs, *tail, self.stream_counts.data_ptr(), _f32_bits(eps), torch.cuda.current_stream(),
            )
        else:
            rpw = self.rows_per_wg(rows)
            if not code_tile_rows:
                rpw = max(8, rpw)  # the row-major codes: 8+ rows per workgroup
            hoist = not code_tile_rows and _hoist_weight(rows, HOIST_MXFP8_ROW_RANGES)
            _run_compiled(
                _stream8_hoist_kernel(rpw) if hoist else _stream8_kernel(rpw, code_tile_rows > 0), *common, *self.stream_bufs, *tail,
                self.stream_counts.data_ptr(), _f32_bits(eps), torch.cuda.current_stream(),
            )
        return q, scale, res_out
