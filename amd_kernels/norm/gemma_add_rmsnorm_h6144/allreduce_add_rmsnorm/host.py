# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Communicators of the fused TP4 all-reduce + add + Gemma RMSNorm kernels: ``ArNormComm`` (registered graph inputs,
the staged communicator's ``capture()``) and ``SlotsArNormComm`` (the MoE combine tail).

``ArNormComm`` (a collective on the TP CPU group, once per rank) maps one uncached buffer per rank (flags + reduced
rows, ``common.BUF_BYTES``) into the peers and keeps the per-row call counts and a table of input addresses: the kernels
read the peers' inputs through an int64[4] row of it. As in AITER's all-reduce, each call captured inside ``capture()``
takes a row, and when it exits the ranks exchange IPC handles of the inputs' allocations and fill the rows. The
registered tiers of ``staged_host.StagedArNormComm`` launch on it (``_bf16_kernel`` among them)."""

import contextlib
import ctypes
import functools

import torch
import torch.distributed as dist

from amd_kernels.common import hip_runtime as ops
from ..add_rmsnorm.host import _f32_bits, _out_or_empty, _run_compiled, code_tile_rows_for
from amd_kernels.common.hip_runtime import load_hip_runtime

from . import ar_add_rmsnorm as _bf16
from . import common as _c
from . import slots4_ar_add_rmsnorm_mxfp8_lossless as _slots4_lossless
from . import slots4_ar_add_rmsnorm_mxfp8 as _slots4
from . import slots4_ar_add_rmsnorm_mxfp8_oneshot as _slots4_1x
from . import slots4_ar_add_rmsnorm_mxfp8_stream as _slots4_stream
from . import slots4_ar_add_rmsnorm_mxfp8_stream_lossless as _slots4_stream_lossless
from . import slots4_ar_add_rmsnorm_mxfp8_stream_lossless_scatter as _slots4_stream_lossless_scatter
from . import slots4_ar_add_rmsnorm_mxfp8_reducer as _slots4_reducer
from . import slots4_ar_add_rmsnorm_mxfp8_stream_hoist_weight as _slots4_stream_hoist

WORLD = _c.WORLD
H = _c.H
MAX_ENTRIES = 1 << 15  # captured calls (rows of the address table)


@functools.cache
def _bf16_kernel():
    return _bf16.compile_ar_add_rmsnorm()


@functools.cache
def _slots4_kernel(
    rpw: int,
    tiled_q: bool = True,
    streamed: bool = False,
    wave_handoffs: bool = False,
    scoped_ftz: bool = False,
    reducer_only: bool = False,
    hoist_weight: bool = False,
):
    if rpw == 0:
        return _slots4_1x.compile_slots4_ar_add_rmsnorm_mxfp8_oneshot()
    if streamed:
        if hoist_weight:
            return _slots4_stream_hoist.compile_slots4_ar_add_rmsnorm_mxfp8_stream_hoist_weight(rpw, tiled_q)
        assert tiled_q, "row-major codes stream hoisted or coded"
        return _slots4_stream.compile_slots4_ar_add_rmsnorm_mxfp8_stream(rpw)
    if reducer_only:
        return _slots4_reducer.compile_slots4_ar_add_rmsnorm_mxfp8_reducer()
    return _slots4.compile_slots4_ar_add_rmsnorm_mxfp8(rpw, wave_handoffs, scoped_ftz)


@functools.cache
def _slots4_lossless_kernel():
    return _slots4_lossless.compile_slots4_ar_add_rmsnorm_mxfp8_lossless()


@functools.cache
def _slots4_stream_lossless_kernel(rpw: int, tiled_q: bool):
    return _slots4_stream_lossless.compile_slots4_ar_add_rmsnorm_mxfp8_stream_lossless(rpw, tiled_q)


@functools.cache
def _slots4_stream_lossless_scatter_kernel(rpw: int, tiled_q: bool):
    return _slots4_stream_lossless_scatter.compile_slots4_ar_add_rmsnorm_mxfp8_stream_lossless_scatter(rpw, tiled_q)


# The tiers below were measured on the real layer-39 MoE entry + this tail (benchmark_slots4_ar_add_rmsnorm.py, graphs,
# max over ranks, both capture orders, same output bytes).
# the register kernel with the lossless 12-bit all-gather (tiled codes); larger rows need a whole-entry gate
LOSSLESS_SLOTS_MIN_ROWS = 160
LOSSLESS_SLOTS_MAX_ROWS = 192
# the streamed kernel with the lossless all-gather serves every streamed row here, at its own rows per workgroup (the
# served R lost where a rank group fills 17-22 of its 24 workgroup slots); R4 and the hoisted R8 tiers stay raw
LOSSLESS_STREAM_RANGES = ((769, _slots4_stream.MAX_ROWS - 1),)
LOSSLESS_RPW_MAX_ROWS = ((8, 1536), (16, 3072))  # else the streamed R (32)
# the reduce-scatter coded too, at the smallest R >= 4 with <= 256 workgroups: wins only in these ranges (at the
# gather tier's own R it lost)
LOSSLESS_SCATTER_STREAM_RANGES = ((769, 1024), (1537, 2048), (3073, 4095))
LOSSLESS_SCATTER_MAX_GRID = 256


def lossless_scatter_rows_per_wg(rows: int) -> int:
    q = (rows + WORLD - 1) // WORLD
    r = 4
    while WORLD * (((q + r - 1) // r + 7) // 8 * 8) > LOSSLESS_SCATTER_MAX_GRID:
        r *= 2
    return r


def lossless_rows_per_wg(rows: int, rpw: int) -> int:
    for r, hi in LOSSLESS_RPW_MAX_ROWS:
        if rows <= hi:
            return min(rpw, r)
    return rpw
assert all(hi < _slots4_stream.MAX_ROWS for _, hi in LOSSLESS_STREAM_RANGES)
assert all(lo <= hi < _slots4_stream.MAX_ROWS for lo, hi in LOSSLESS_SCATTER_STREAM_RANGES)
assert (_slots4_stream.MAX_ROWS - 1 + WORLD - 1) // WORLD <= _slots4_stream_lossless.RAW_ROWS


# measured with benchmark_slots4_ar_add_rmsnorm.py (graphs of 100 calls, max over ranks):
WAVE_HANDOFF_MIN_ROWS = 29  # register kernels: per-wave hand-offs win from 29 rows (8x the flag atomics lose below)
SCOPED_FTZ_MAX_ROWS = 96  # register kernels: the scoped hardware FTZ beats software masks to 96 rows, loses from 112
REDUCER_ONLY_MAX_ROWS = 64  # pullers skip peer loads and rank sums: -0.2 us at 29-64 rows, noise above
WEIGHT_HOIST_R8_MAX_ROWS = 768  # the R8 streamed kernel loads the weight once per batch (above: the coded tiers)
ONESHOT_MAX_ROWS = 14  # the scoped-FTZ register kernel wins from 15 rows
assert ONESHOT_MAX_ROWS <= _slots4_1x.ONESHOT_ROWS


class _DevicePointer:
    """A raw device allocation as a tensor (``__cuda_array_interface__``)."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


@functools.cache
def _hip():
    lib = load_hip_runtime()
    lib.hipMemGetAddressRange.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
    ]
    lib.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_IpcHandle), ctypes.c_void_p]
    lib.hipIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        _IpcHandle,
        ctypes.c_uint,
    ]
    return lib


def _alloc_handle(ptr: int) -> tuple[int, bytes]:
    """(base of the allocation holding ptr, its IPC handle's 64 raw bytes)."""
    hip = _hip()
    base, size = ctypes.c_void_p(), ctypes.c_size_t()
    rc = hip.hipMemGetAddressRange(ctypes.byref(base), ctypes.byref(size), ptr)
    assert rc == 0, f"hipMemGetAddressRange failed: {rc}"
    h = _IpcHandle()
    rc = hip.hipIpcGetMemHandle(ctypes.byref(h), base)
    assert rc == 0, f"hipIpcGetMemHandle failed: {rc}"
    # the raw bytes: a c_char field reads back only up to its first NUL
    return base.value, ctypes.string_at(ctypes.addressof(h), ctypes.sizeof(h))


def _open_handle(raw: bytes) -> int:
    hip = _hip()
    h = _IpcHandle()
    ctypes.memmove(ctypes.addressof(h), raw, len(raw))
    ptr = ctypes.c_void_p()
    rc = hip.hipIpcOpenMemHandle(ctypes.byref(ptr), h, 1)  # lazy peer access
    assert rc == 0, f"hipIpcOpenMemHandle failed: {rc}"
    return ptr.value


def exchange(obj, rank: int, group) -> list:
    """Every rank's ``obj`` (broadcasts on the CPU group: all_gather_object fails
    on gloo under inference mode)."""
    out = [[None] for _ in range(WORLD)]
    out[rank][0] = obj
    ranks = sorted(dist.get_process_group_ranks(group=group))
    for i, r in enumerate(ranks):
        dist.broadcast_object_list(out[i], src=r, group=group, device="cpu")
    return [o[0] for o in out]


class ArNormComm:
    def __init__(self, rank: int, group, device: torch.device):
        self.rank, self.group, self.device = rank, group, device
        ptr, handle = ops.allocate_shared_buffer_and_handle(_c.BUF_BYTES)
        peers = exchange(handle, rank, group)
        self.bufs = [
            ptr if r == rank else ops.open_mem_handle(peers[r]) for r in range(WORLD)
        ]
        flags = torch.as_tensor(_DevicePointer(ptr, _c.FLAG_BYTES), device=device)
        flags.zero_()
        self.counts = torch.zeros(_c.MAX_ROWS, dtype=torch.int32, device=device)
        self.table = torch.zeros((MAX_ENTRIES, WORLD), dtype=torch.int64, device=device)
        self._inputs: list[int] = []  # this rank's input address per table row
        self._filled = 0
        self._opened: dict[bytes, int] = {}
        self.capturing = False
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    @contextlib.contextmanager
    def capture(self):
        """Around graph capture (every rank, same captures): the fused calls
        recorded inside take table rows, filled when it exits."""
        self.capturing = True
        try:
            yield
        finally:
            self.capturing = False
            self.register_graph_buffers()

    def _row(self, x: torch.Tensor) -> int:
        n = len(self._inputs)
        assert n < MAX_ENTRIES, "allreduce_add_rmsnorm: address table full"
        self._inputs.append(x.data_ptr())
        return self.table.data_ptr() + n * WORLD * 8

    def register_graph_buffers(self) -> None:
        """Fill the rows taken since the last call with the 4 ranks' input
        addresses (a collective: the ranks captured the same calls in order)."""
        new = self._inputs[self._filled :]
        counts = exchange(len(new), self.rank, self.group)
        assert len(set(counts)) == 1, f"allreduce_add_rmsnorm: captured calls differ {counts}"
        if not new:
            return
        handles: list[bytes] = []
        index: dict[int, int] = {}
        metas = []
        for p in new:
            base, raw = _alloc_handle(p)
            if base not in index:
                index[base] = len(handles)
                handles.append(raw)
            metas.append((index[base], p - base))
        every = exchange((handles, metas), self.rank, self.group)
        rows = torch.empty((len(new), WORLD), dtype=torch.int64)
        for r, (hs, ms) in enumerate(every):
            if r == self.rank:
                rows[:, r] = torch.tensor(new, dtype=torch.int64)
                continue
            bases = []
            for raw in hs:
                if raw not in self._opened:
                    self._opened[raw] = _open_handle(raw)
                bases.append(self._opened[raw])
            rows[:, r] = torch.tensor([bases[i] + off for i, off in ms])
        self.table[self._filled : self._filled + len(new)].copy_(rows)
        torch.cuda.synchronize(self.device)
        self._filled += len(new)


class SlotsArNormComm:
    """The communicator of ``slots4_ar_add_rmsnorm_mxfp8`` (a collective on the TP CPU group, once per rank).

    One uncached buffer per rank (flags, reduced rows, staged partials; the one-shot and streaming domains disjoint)
    mapped by the peers, and the per-row call counts. The kernels read no peer input, so calls need
    no registration: eager or captured alike. One communicator per stream; the call order must match on all ranks."""

    def __init__(self, rank: int, group, device: torch.device):
        self.rank, self.group, self.device = rank, group, device
        ptr, handle = ops.allocate_shared_buffer_and_handle(
            _slots4_1x.ONESHOT_BUF_BYTES + _slots4_stream.STREAM_BUF_BYTES
        )
        # the lossless tiers' raw waves: own arenas, disjoint from the buffer above (one row per call row)
        rawptr, rawhandle = ops.allocate_shared_buffer_and_handle(_slots4_stream_lossless.RAW_BYTES)
        inrawptr, inrawhandle = ops.allocate_shared_buffer_and_handle(
            _slots4_stream_lossless_scatter.INPUT_RAW_BYTES)
        peers = exchange((handle, rawhandle, inrawhandle), rank, group)
        self.bufs = [
            ptr if r == rank else ops.open_mem_handle(peers[r][0]) for r in range(WORLD)
        ]
        self.raw_bufs = [
            rawptr if r == rank else ops.open_mem_handle(peers[r][1]) for r in range(WORLD)
        ]
        self.raw_table = torch.tensor(self.raw_bufs, dtype=torch.int64, device=device)
        self.input_raw_bufs = [
            inrawptr if r == rank else ops.open_mem_handle(peers[r][2]) for r in range(WORLD)
        ]
        self.input_raw_table = torch.tensor(self.input_raw_bufs, dtype=torch.int64, device=device)
        self.counts = torch.zeros(_slots4_stream.MAX_ROWS, dtype=torch.int32, device=device)
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    lossless_min_rows = LOSSLESS_SLOTS_MIN_ROWS  # tests may disable the tier (the raw register kernel there)
    lossless_max_rows = LOSSLESS_SLOTS_MAX_ROWS

    def scoped_ftz(self, rows: int) -> bool:
        """Hardware FP32 FTZ around the register kernels' rank sum up to ``SCOPED_FTZ_MAX_ROWS``."""
        return rows <= SCOPED_FTZ_MAX_ROWS

    def wave_handoffs(self, rows: int) -> bool:
        """Per-wave hand-offs in the register kernels from ``WAVE_HANDOFF_MIN_ROWS``."""
        return rows >= WAVE_HANDOFF_MIN_ROWS

    def rows_per_wg(self, rows: int) -> int:
        """Rows per workgroup (measured): 0 = the one-shot, 1 / 2 the register kernels, 4-32 the streaming kernel (its
        operands die at each row, so registers do not grow with R)."""
        if rows <= ONESHOT_MAX_ROWS:
            return 0
        if rows <= 256:
            return 1
        if rows <= 384:
            return 2
        if rows <= 640:
            return 4
        if rows <= 1280:
            return 8
        if rows <= 2048:
            return 16
        return 32

    def hoist_weight(self, rows: int, rpw: int) -> bool:
        return rpw == 8 and rows <= WEIGHT_HOIST_R8_MAX_ROWS

    def supports(self, rows: int) -> bool:
        """Row counts ``slots4_ar_add_rmsnorm_mxfp8`` takes, eager and captured (1-4095: larger prefills take the
        sequence-parallel hand-offs)."""
        return 0 < rows < _slots4_stream.MAX_ROWS

    def warm_up(self, weight: torch.Tensor, eps: float, code_tile_rows: int = 16) -> None:
        """Compile every variant before serving, both code layouts (a collective: every rank calls it with its own
        weight)."""
        row_tiers = {
            WORLD, ONESHOT_MAX_ROWS + 1, WAVE_HANDOFF_MIN_ROWS, REDUCER_ONLY_MAX_ROWS + 1, SCOPED_FTZ_MAX_ROWS + 1,
            LOSSLESS_SLOTS_MIN_ROWS,
            *(r for lo, hi in LOSSLESS_STREAM_RANGES for r in (lo - 1, lo, hi, hi + 1)
              if r < _slots4_stream.MAX_ROWS),
            *(r for _, hi in LOSSLESS_RPW_MAX_ROWS for r in (hi, hi + 1)),
            *(r for lo, hi in LOSSLESS_SCATTER_STREAM_RANGES for r in (lo - 1, lo, hi, hi + 1)
              if r < _slots4_stream.MAX_ROWS),
            # every rows-per-workgroup key of the two-phase tier
            *(r for lo, hi in LOSSLESS_SCATTER_STREAM_RANGES for r in range(lo + 1, hi + 1)
              if lossless_scatter_rows_per_wg(r) != lossless_scatter_rows_per_wg(r - 1)),
            384, 512, 1024, 2048, _slots4_stream.MAX_ROWS - 1, WEIGHT_HOIST_R8_MAX_ROWS, WEIGHT_HOIST_R8_MAX_ROWS + 1,
        }
        for tile in (0, code_tile_rows or 16):
            for rows in sorted(row_tiers):
                slots = torch.zeros((rows, 4, H), dtype=torch.bfloat16, device=self.device)
                row = torch.zeros((rows, H), dtype=torch.bfloat16, device=self.device)
                self.slots4_ar_add_rmsnorm_mxfp8(
                    slots, row, row, weight, eps, tile
                )
        torch.cuda.synchronize(self.device)

    def slots4_ar_add_rmsnorm_mxfp8(
        self,
        slots: torch.Tensor,
        shared: torch.Tensor,
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
        """``add_rmsnorm.gemma_add_rmsnorm_mxfp8`` of all-reduce(slot 0 + .. + slot 3 + shared) + residual, 1-4095 rows.

        Every rank passes its own contiguous bf16 [rows, 4, H] slots and [rows, H] shared output. Returns (codes in the
        (16, 16) layout when code_tile_rows > 0, row-major at 0; AITER-layout scales; residual out); ``gemm_weight``
        instead of ``code_tile_rows``: the layout that GEMM reads at this row count."""
        rows = slots.shape[0]
        if gemm_weight is not None:  # the layout that GEMM reads at this row count
            code_tile_rows = code_tile_rows_for(rows, gemm_weight)
        assert code_tile_rows is not None
        assert self.supports(rows), rows
        assert slots.shape == (rows, 4, H) and slots.dtype == torch.bfloat16
        assert shared.shape == (rows, H) and shared.dtype == torch.bfloat16
        assert slots.is_contiguous() and shared.is_contiguous()
        assert residual.shape == (rows, H) and residual.dtype == torch.bfloat16
        assert residual.stride(1) == 1 and residual.stride(0) % 8 == 0
        assert weight.dtype == torch.bfloat16 and weight.is_contiguous()
        assert code_tile_rows >= 0 and code_tile_rows % 16 == 0
        tiled = (
            (rows + code_tile_rows - 1) // code_tile_rows * code_tile_rows
            if code_tile_rows else rows
        )
        q = _out_or_empty(q_out, (tiled, H), torch.float8_e4m3fn, self.device)
        scale = _out_or_empty(
            scale_out, ((tiled + 31) // 32 * 32, H // 32), torch.uint8, self.device
        )
        res_out = _out_or_empty(res_out, (rows, H), torch.bfloat16, self.device)
        rpw = self.rows_per_wg(rows)
        assert rpw or rows <= _slots4_1x.ONESHOT_ROWS, "the one-shot's buffer holds ONESHOT_ROWS rows"
        if not code_tile_rows:
            rpw = max(8, rpw)  # the streaming kernel also writes row-major codes
        streamed = rpw >= 4  # R 4 from 385 rows, and every row-major call
        bufs = [b + _slots4_1x.ONESHOT_BUF_BYTES for b in self.bufs] if streamed else self.bufs
        lossless = code_tile_rows > 0 and self.lossless_min_rows <= rows <= self.lossless_max_rows
        hoist = streamed and self.hoist_weight(rows, rpw)
        stream_lossless = streamed and any(lo <= rows <= hi for lo, hi in LOSSLESS_STREAM_RANGES)
        scatter = streamed and any(lo <= rows <= hi for lo, hi in LOSSLESS_SCATTER_STREAM_RANGES)
        if scatter:
            rpw = lossless_scatter_rows_per_wg(rows)
        elif stream_lossless:
            rpw = lossless_rows_per_wg(rows, rpw)
        _run_compiled(
            _slots4_lossless_kernel() if lossless
            else _slots4_stream_lossless_scatter_kernel(rpw, code_tile_rows > 0) if scatter
            else _slots4_stream_lossless_kernel(rpw, code_tile_rows > 0) if stream_lossless
            else _slots4_kernel(
                rpw, code_tile_rows > 0, streamed,
                not streamed and rpw > 0 and self.wave_handoffs(rows),
                not streamed and rpw > 0 and self.scoped_ftz(rows),
                not streamed and rpw == 1 and self.wave_handoffs(rows)
                and self.scoped_ftz(rows) and rows <= REDUCER_ONLY_MAX_ROWS,
                hoist,
            ),
            rows,
            (rows + WORLD - 1) // WORLD,
            self.rank,
            residual.stride(0),
            slots.data_ptr(),
            shared.data_ptr(),
            *bufs,
            residual.data_ptr(),
            weight.data_ptr(),
            res_out.data_ptr(),
            q.data_ptr(),
            scale.data_ptr(),
            q.numel(),
            scale.numel(),
            self.counts.data_ptr(),
            *((self.raw_table.data_ptr(),) if stream_lossless or scatter else ()),
            *((self.input_raw_table.data_ptr(),) if scatter else ()),
            _f32_bits(eps),
            torch.cuda.current_stream(),
        )
        return q, scale, res_out
