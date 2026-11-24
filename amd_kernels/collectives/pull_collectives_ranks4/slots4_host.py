# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-safe MoE combine + reduce-scatter on four MI355X GPUs.

One communicator per stream/microbatch. Construct collectively and warm up
before graph capture. Buffers and counters live for the process lifetime, as
in Pools. Neither inputs nor outputs may alias the communicator's pool.
"""
import functools
import struct
import torch

from amd_kernels.common import hip_runtime as ops
from .host import Pools, WORLD, _exchange, _run_compiled, _kernels
from .device_sequence import compile_device_fence_publish, compile_device_pull
from .slots4_stage_pull_unroll4 import FLAG_BYTES, compile_slots4_stage_pull_unroll4
from .slots4_stage_rows import compile_slots4_stage_rows
from .bf16_stage import compile_bf16_stage
from .sum4_add_rmsnorm_mxfp8 import compile_sum4_add_rmsnorm_mxfp8

H = 6144
MAX_ROWS = 32768


@functools.cache
def _slots4_kernels():
    return compile_slots4_stage_pull_unroll4(), compile_device_fence_publish(), compile_device_pull()


class SlotsReduceScatterComm:
    """Weighted slots + shared -> this rank's BF16 rows, fixed FP32 sum order.

    The slot sum + shared stage overlaps the pull into staging (four reads per
    chunk at every size). Supports total rows divisible by four, 4..32768. The
    collective moves BF16; no quantized communication and no atomic output adds.
    Calls on different ranks must use the same shape and order. Concurrent
    streams need distinct communicators. Graph replays and eager calls may be
    interleaved.
    """
    def __init__(self, rank, cpu_group, device, max_rows=MAX_ROWS):
        assert 4 <= max_rows <= MAX_ROWS and max_rows % WORLD == 0
        self.device = torch.device(device)
        self.max_rows = max_rows
        self.pools = Pools(rank, WORLD, cpu_group, self.device, max_rows * H * 2, sets=1)
        self.counter = torch.zeros(1, dtype=torch.int32, device=self.device)
        # the stage + pull launch's per-chunk flags (uncached, zeroed, mapped by the peers)
        ptr, handle = ops.allocate_shared_buffer_and_handle(FLAG_BYTES)
        handles = _exchange(handle, rank, WORLD, cpu_group)
        self.chunk_flags = [ptr if r == rank else ops.open_mem_handle(h)
                            for r, h in enumerate(handles)]
        self.stage_pull, self.publish, self.pull = _slots4_kernels()
        self.sum4 = _kernels()[-1]
        self.pull32 = compile_device_pull(32)

    def _check_inputs(self, slots, shared):
        """Validate the slot layout and collective row range."""
        rows = slots.shape[0]
        assert 4 <= rows <= self.max_rows and rows % WORLD == 0
        assert slots.shape == (rows, 4, H) and shared.shape == (rows, H)
        for x in (slots, shared):
            assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.device == self.device
        return rows

    def _stage_slots(self, slots, shared):
        """Slot sum + shared into the pool, overlapped with the pull into staging (row-independent: warm-up uses tiny
        operands)."""
        rows = self._check_inputs(slots, shared)
        p = self.pools
        staging = torch.empty(rows * H * 2, dtype=torch.uint8, device=self.device)
        _run_compiled(self.stage_pull, slots.data_ptr(), shared.data_ptr(), staging.data_ptr(),
                      *p.pool_ptrs, *self.chunk_flags, self.counter, p.slot_bytes, rows, p.rank,
                      torch.cuda.current_stream(self.device))
        return staging

    def _pull_staged(self, rows):
        stream = torch.cuda.current_stream(self.device)
        p = self.pools
        _run_compiled(self.publish, p.flag_ptrs[p.rank], self.counter, stream)
        n = rows * H * 2
        block = n // WORLD
        staging = torch.empty(n, dtype=torch.uint8, device=self.device)
        # Five-round sweep: 32 WGs wins by 3% at 4K/8K. At >=16K,
        # 16 WGs already fills the links and retains more compute capacity.
        pull = self.pull32 if 4096 <= rows <= 8192 else self.pull
        _run_compiled(pull, staging.data_ptr(), *p.pool_ptrs, *p.flag_ptrs,
                      block, p.rank * block, self.counter, p.slot_bytes, p.rank, stream)
        return staging

    def _sum(self, staging, rows, out):
        if out is None:
            out = torch.empty((rows // WORLD, H), dtype=torch.bfloat16, device=self.device)
        assert out.shape == (rows // WORLD, H) and out.dtype == torch.bfloat16
        assert out.is_contiguous() and out.device == self.device
        block = rows * H * 2 // WORLD
        srcs = [staging.data_ptr() + r * block for r in range(WORLD)]
        _run_compiled(self.sum4, out.data_ptr(), *srcs, block, torch.cuda.current_stream(self.device))
        return out

    def slots4_reduce_scatter(self, slots, shared, out=None):
        """Slot sum and shared add fused into BF16 reduce-scatter publication."""
        staging = self._stage_slots(slots, shared)
        return self._sum(staging, slots.shape[0], out)

    def reduce_scatter(self, x, out=None):
        """Dense BF16 RS, same transport, graph-safe device sequence counters."""
        rows = x.shape[0]
        assert 4 <= rows <= self.max_rows and rows % WORLD == 0
        assert x.shape == (rows, H) and x.dtype == torch.bfloat16
        assert x.is_contiguous() and x.device == self.device
        p = self.pools
        _run_compiled(_dense_stage(), x.data_ptr(), p.pool_ptrs[p.rank], self.counter,
                      p.slot_bytes, rows, torch.cuda.current_stream(self.device))
        return self._sum(self._pull_staged(rows), rows, out)

    def slots4_reduce_scatter_add_rmsnorm_mxfp8(self, slots, shared, residual, weight, eps):
        """Slot sum + shared + BF16 RS + shard norm/quant in two launches.

        Returns (row-major codes, shuffled scales, residual_out).
        """
        rows = self._check_inputs(slots, shared) // WORLD
        assert residual.shape == (rows, H) and residual.dtype == torch.bfloat16
        assert residual.is_contiguous() and residual.device == self.device
        assert weight.shape == (H,) and weight.dtype == torch.bfloat16
        assert weight.is_contiguous() and weight.device == self.device
        q = torch.empty((rows, H), dtype=torch.float8_e4m3fn, device=self.device)
        scale = torch.empty(((rows + 31)//32*32, H//32), dtype=torch.uint8, device=self.device)
        res_out = torch.empty_like(residual)
        staging = self._stage_slots(slots, shared)
        _run_compiled(_sum_norm(), staging.data_ptr(), residual.data_ptr(),
                      weight.data_ptr(), q.data_ptr(), scale.data_ptr(), res_out.data_ptr(),
                      q.numel(), scale.numel(), rows, H, H,
                      struct.unpack('<i', struct.pack('<f', eps))[0],
                      torch.cuda.current_stream(self.device))
        return q, scale, res_out

    def warm_up(self):
        """Collective: compile all row-independent variants before graph capture."""
        s = torch.zeros((4, 4, H), dtype=torch.bfloat16, device=self.device)
        self.slots4_reduce_scatter(s, s[:, 0].contiguous())
        self.reduce_scatter(s[:, 0].contiguous())
        r = torch.zeros((1, H), dtype=torch.bfloat16, device=self.device)
        self.slots4_reduce_scatter_add_rmsnorm_mxfp8(s, s[:, 0].contiguous(), r, r[0], 1e-6)
        # The 32-workgroup dense pull is reachable only when the pool supports 4096 rows.
        if self.max_rows >= 4096:
            self.reduce_scatter(torch.zeros((4096, H), dtype=torch.bfloat16, device=self.device))
        torch.cuda.synchronize(self.device)


@functools.cache
def _dense_stage():
    return compile_bf16_stage()


@functools.cache
def _sum_norm():
    return compile_sum4_add_rmsnorm_mxfp8()


@functools.cache
def _stage_rows():
    """The slot sum + shared stage alone (the SDMA hand-offs' reduce-scatter source)."""
    return compile_slots4_stage_rows()
