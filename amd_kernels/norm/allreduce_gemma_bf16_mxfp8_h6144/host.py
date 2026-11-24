# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""TP4 graph registration and eager staging for the bf16-row norm rule."""
import ctypes
import functools

import torch
import torch.distributed as dist

from amd_kernels.common import hip_runtime as ops
from .ipc import GraphInputs, _DevicePointer, _f32_bits, _run_compiled, exchange
from . import common as C
from .kernel import compile_allreduce_gemma_bf16_mxfp8


def _release(ptr, peer):
    hip = ops._hip()
    fn = hip.hipIpcCloseMemHandle if peer else hip.hipFree
    fn.argtypes = [ctypes.c_void_p]
    rc = fn(ctypes.c_void_p(ptr))
    assert rc == 0, f"HIP buffer release failed: {rc}"


@functools.cache
def _kernel(rpw, tiled, quantized, owner):
    return compile_allreduce_gemma_bf16_mxfp8(rpw, tiled, quantized, owner)


# Below this row count the owner form is slower (7 rows: 8.13 vs 7.63 us): the
# owned-residual calls run the all-row form, which reads and writes every residual row.
OWNER_MIN_ROWS = 32


def owned_rows(rows, rank):
    """[start, stop) of the rows whose sum, residual and norm ``rank`` owns."""
    q = (rows + 3) // 4
    return min(rank * q, rows), min((rank + 1) * q, rows)


def rows_per_group(rows):
    # Keep <=256 workgroups: the peers must all reach the handshake.
    if rows <= 256:
        return 1
    if rows <= 512:
        return 2
    if rows <= 1024:
        return 8
    return 16


class Bf16ArNormComm(GraphInputs):
    """One communicator per stream, created collectively on four ranks.

    Inside ``capture()``, peer graph inputs are read in place. Eager calls stage
    inputs into alternating shared slots, then launch the same kernel. A completed
    call has observed every peer start that call; therefore all peers finished
    the call two slots ago. This prevents a fast rank overwriting a slow peer's
    previous input. Graph calls require the ``capture()`` context. All ranks call the
    same operations in the same order; close only after graph work completes.
    """
    def __init__(self, rank, group, device):
        if dist.get_world_size(group) != 4:
            raise ValueError('four ranks required')
        device = torch.device(device)
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        if torch.cuda.current_device() != device.index or dist.get_rank(group) != rank:
            raise ValueError('select this rank device before creating the communicator')
        self.rank, self.group, self.device = rank, group, device
        ptr, handle = ops.allocate_shared_buffer_and_handle(C.BUF_BYTES)
        peers = exchange(handle, rank, group)
        self.bufs = [ptr if r == rank else ops.open_mem_handle(peers[r]) for r in range(4)]
        torch.as_tensor(_DevicePointer(ptr, C.FLAG_BYTES), device=device).zero_()
        self.counts = torch.zeros(C.MAX_ROWS, dtype=torch.int32, device=device)
        self.table = torch.zeros((1 << 15, 4), dtype=torch.int64, device=device)
        self._inputs, self._filled, self._opened, self.capturing = [], 0, {}, False
        sptr, shandle = ops.allocate_shared_buffer_and_handle(2 * C.MAX_ROWS * C.H * 2)
        every = exchange(shandle, rank, group)
        self.stage_ptrs = [sptr if r == rank else ops.open_mem_handle(every[r]) for r in range(4)]
        self.stage = torch.as_tensor(_DevicePointer(sptr, 2 * C.MAX_ROWS * C.H * 2), device=device).view(torch.bfloat16).view(2, C.MAX_ROWS, C.H)
        self.stage_table = torch.tensor([[p + slot * C.MAX_ROWS * C.H * 2 for p in self.stage_ptrs] for slot in range(2)], dtype=torch.int64, device=device)
        self._eager_call = 0
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    def _launch(self, x, residual, weight, eps, tile, quantized, owner=False):
        if not self.bufs:
            raise RuntimeError("communicator is closed")
        m = x.shape[0]
        if not (0 < m <= C.MAX_ROWS and tuple(x.shape) == (m, C.H)):
            raise ValueError('expected 1..2048 rows of hidden 6144')
        if x.device != self.device or residual.device != self.device or weight.device != self.device:
            raise ValueError("all tensors must be on the communicator device")
        if x.dtype != torch.bfloat16 or not x.is_contiguous():
            raise ValueError('partial must be contiguous bf16')
        if residual.shape != x.shape or residual.dtype != x.dtype or residual.stride(1) != 1 or residual.stride(0) % 8:
            raise ValueError('residual must be bf16 with a row stride divisible by eight')
        if tuple(weight.shape) != (C.H,) or weight.dtype != x.dtype or not weight.is_contiguous():
            raise ValueError('weight must be contiguous bf16 [6144]')
        if tile < 0 or tile % 16:
            raise ValueError('code tile rows must be zero or a multiple of 16')
        if self.capturing and torch.cuda.is_current_stream_capturing():
            xs = self._row(x)
        else:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('graph calls require the communicator capture() context')
            slot = self._eager_call % 2
            self._eager_call += 1
            self.stage[slot, :m].copy_(x)
            xs = self.stage_table[slot].data_ptr()
        padded = (m + max(tile, 1) - 1) // max(tile, 1) * max(tile, 1)
        out = torch.empty((padded if quantized else m, C.H), dtype=torch.float8_e4m3fn if quantized else torch.bfloat16, device=self.device)
        scales = torch.empty(((padded + 31) // 32 * 32, C.H // 32), dtype=torch.uint8, device=self.device) if quantized else None
        res = torch.empty_like(x)
        owner = owner and m >= OWNER_MIN_ROWS
        _run_compiled(_kernel(rows_per_group(m), tile > 0, quantized, owner),
                      m, self.rank, xs, *self.bufs, self.counts.data_ptr(), residual.stride(0),
                      residual.data_ptr(), weight.data_ptr(), out.data_ptr(), scales.data_ptr() if quantized else 0, res.data_ptr(),
                      out.numel() * out.element_size(), scales.numel() if quantized else 0, _f32_bits(eps), torch.cuda.current_stream())
        return (out, scales, res) if quantized else (out, res)

    def ar_add_rmsnorm(self, x, residual, weight, eps=1e-6, owned_residual=False):
        """``owned_residual``: from ``OWNER_MIN_ROWS`` rows on, only this rank's
        ``owned_rows`` of ``residual`` are read and of the residual output written; the
        owner norms its rows and the peers copy the bf16 result. Fewer rows run the
        all-row form. So pass a full residual or the residual output of an
        ``owned_residual`` call of the same row count. Outputs equal the default form's
        bit for bit."""
        return self._launch(x, residual, weight, eps, 0, False, owned_residual)

    def ar_add_rmsnorm_mxfp8(self, x, residual, weight, eps=1e-6, code_tile_rows=16, owned_residual=False):
        """``owned_residual``: as in ``ar_add_rmsnorm``; the peers copy the owner's
        codes and scale bytes (about half the bytes of a bf16 row)."""
        return self._launch(x, residual, weight, eps, code_tile_rows, True, owned_residual)

    def owned_rows(self, rows):
        return owned_rows(rows, self.rank)

    def warm_up(self, weight, eps=1e-6):
        """Compile all row tiers, both code layouts, bf16 output and both residual
        forms collectively (the 1-row tier's owner form from 224 rows)."""
        for rows, forms in ((7, (False,)), (224, (True,)), (448, (False, True)), (896, (False, True)),
                            (1792, (False, True))):
            x = torch.zeros((rows, C.H), dtype=torch.bfloat16, device=self.device)
            for owner in forms:
                self.ar_add_rmsnorm(x, x, weight, eps, owner)
                for tile in (0, 16):
                    self.ar_add_rmsnorm_mxfp8(x, x, weight, eps, tile, owner)

    def close(self):
        if not self.bufs:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        for r, ptr in enumerate(self.bufs):
            if r == self.rank:
                _release(ptr, False)
            else:
                _release(ptr, True)
        for r, ptr in enumerate(self.stage_ptrs):
            if r == self.rank:
                _release(ptr, False)
            else:
                _release(ptr, True)
        for ptr in self._opened.values():
            _release(ptr, True)
        self._opened.clear()
        self.bufs = []
        self.stage_ptrs = []
