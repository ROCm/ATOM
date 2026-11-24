# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-safe, lossless gather of row-major tensor fields on four ranks."""
import functools

import torch

from .host import Pools, WORLD, _run_compiled
from .byte_stage import compile_byte_stage
from .device_sequence import compile_device_fence_publish
from .pull_bounded import compile_device_pull_bounded
from .stage6_publish import WGS, compile_stage6_publish

# Joined TP4 gate: one staging/publication launch through 4096 compact shard rows.
SIX_MAX_BYTES = 4096 * 7328


@functools.cache
def _gather_kernels():
    return compile_byte_stage(), compile_device_fence_publish(), compile_device_pull_bounded()


class AllGatherComm:
    """Gather a list of contiguous tensor shards, preserving every byte.

    Construct collectively with the same slot capacity on all four ranks.
    Each field has a positive byte length divisible by 16 and 16-byte-aligned
    input/output addresses. Fields occupy
    256-byte-aligned regions. Outputs concatenate ranks along their first axis.
    One communicator per ordered stream/microbatch. Eager calls and graph replay
    can interleave; every rank must issue the same shapes and operation order.
    Inputs and outputs must not alias this communicator's IPC pool.
    """

    def __init__(self, rank, cpu_group, device, slot_bytes):
        assert 0 < slot_bytes < 2**31 and slot_bytes % 256 == 0
        self.device = torch.device(device)
        self.pools = Pools(rank, WORLD, cpu_group, self.device, slot_bytes, sets=1)
        self.counter = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.stage, self.publish, self.pull = _gather_kernels()
        self.stage_six = _stage_six()

    def all_gather(self, outs, ins):
        """Stage inputs, publish once, and pull each field in rank order."""
        assert len(outs) == len(ins) and len(ins) > 0
        p = self.pools
        offsets, offset = [], 0
        for out, x in zip(outs, ins):
            assert x.ndim > 0 and x.is_contiguous() and out.is_contiguous()
            assert x.data_ptr() % 16 == 0 and out.data_ptr() % 16 == 0
            assert x.device == self.device and out.device == self.device and out.dtype == x.dtype
            assert tuple(out.shape) == (WORLD * x.shape[0], *x.shape[1:])
            n = x.numel() * x.element_size()
            assert n > 0 and n % 16 == 0
            offset = (offset + 255) // 256 * 256
            assert offset + n <= p.slot_bytes
            for t in (x, out):
                assert not (p.pool.data_ptr() <= t.data_ptr() < p.pool.data_ptr() + p.pool.numel())
            offsets.append((offset, n))
            offset += n
        stream = torch.cuda.current_stream(self.device)
        if len(ins) == 6 and sum(n for _,n in offsets) <= SIX_MAX_BYTES:
            sizes = [n for _,n in offsets]
            total = sum(sizes)
            counts = [1+n*(WGS-6)//total for n in sizes[:5]]
            bounds = []
            for count in counts:
                bounds.append(count+(bounds[-1] if bounds else 0))
            _run_compiled(self.stage_six,*(x.data_ptr() for x in ins),*(off for off,_ in offsets),
                          *sizes,*bounds,p.pool_ptrs[p.rank],p.slot_bytes,self.counter,p.flag_ptrs[p.rank],stream)
        else:
            for x, (off, n) in zip(ins, offsets):
                _run_compiled(self.stage, x.data_ptr(), p.pool_ptrs[p.rank] + off,
                              self.counter, p.slot_bytes, n, stream)
            _run_compiled(self.publish, p.flag_ptrs[p.rank], self.counter, stream)
        for out, (off, n) in zip(outs, offsets):
            _run_compiled(self.pull, out.data_ptr(), *(ptr + off for ptr in p.pool_ptrs),
                          *p.flag_ptrs, n, 0, self.counter, p.slot_bytes, p.rank, stream)

    def warm_up(self):
        """Collective: compile the byte-count-independent transport before capture."""
        x = torch.zeros((1, 16), device=self.device, dtype=torch.uint8)
        out = torch.empty((WORLD, 16), device=self.device, dtype=torch.uint8)
        self.all_gather([out], [x])
        if self.pools.slot_bytes >= 1536:
            self.all_gather([torch.empty_like(out) for _ in range(6)],[x]*6)
        torch.cuda.synchronize(self.device)



@functools.cache
def _stage_six():
    return compile_stage6_publish()
