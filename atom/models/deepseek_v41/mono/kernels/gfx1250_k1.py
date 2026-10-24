# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Cooperative gfx1250 K1 projection: one shared producer per row.

This is the local hidden -> Q/KV projection compute, without mHC, index,
ring insertion or peer communication. Explicit cooperative HIP dispatch and
an occupancy check guarantee residency. All waits are bounded and fail closed.
Scratch belongs to one stream and cannot be reused concurrently; a nonzero
diagnostic word requires stopping dispatch and inspecting the failed step.
"""

import ctypes
from functools import lru_cache
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels.tensor_shim import _preload_compiled, check_e8m0, ptr_arg
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T

from atom.models.deepseek_v41.mono.kernels.gfx1250_gemv import (
    THREADS,
    gemv_device,
    gemv_lds_bytes,
    grouped_load,
    grouped_ready,
    grouped_lds_bytes,
    projection_load,
    projection_ready,
)
from atom.models.deepseek_v41.mono.kernels.gfx1250_producers import (
    hidden_norm_device,
    qkv_norm_device,
)
from atom.mono.device.ops import traced
from atom.mono.runtime.compile import _code_object, launcher_resources

WAIT_TICKS = 10_000_000  # 100 ms at this platform's 100 MHz steady counter.


def _clock():
    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.readsteadycounter", [], [], []))


def _i32_pointer(p):
    return fx.recast_iter(fx.PointerType.get(fx.Int32.ir_type, p.address_space, 4), p)


@traced
def _publish(flags, index, epoch):
    # gpu.barrier is a workgroup release/acquire: it carries every writer's
    # payload to the leader. The leader's device release publishes that
    # transitive history once per CTA, rather than once per wave.
    fx.gpu.barrier()
    if fx.thread_idx.x == 0:
        fx.generic_store(
            flags + index,
            epoch,
            memory_order=fx.AtomicOrdering.Release,
            syncscope="agent",
        )


@traced
def _await(flags, offset, count, epoch, diag, lds, tid, workers, phase):
    """Cooperative gather with one polling thread per source slot.

    All threads return the same CTA-local result. No caller uses a payload
    after its wait fails, even if other CTAs have already completed their work.
    """
    for group in range_constexpr((count + THREADS - 1) // THREADS):
        index = tid + group * THREADS
        if index < count:
            value = fx.generic_load(
                flags + offset + index,
                memory_order=fx.AtomicOrdering.Monotonic,
                syncscope="agent",
            )
            aborted = fx.generic_load(
                diag, memory_order=fx.AtomicOrdering.Monotonic, syncscope="agent"
            )
            if (value != epoch) & (aborted == 0):
                t0 = _clock()
                while (value != epoch) & (aborted == 0) & (_clock() - t0 < WAIT_TICKS):
                    fx.rocdl.s_sleep(1)
                    value = fx.generic_load(
                        flags + offset + index,
                        memory_order=fx.AtomicOrdering.Monotonic,
                        syncscope="agent",
                    )
                    aborted = fx.generic_load(
                        diag,
                        memory_order=fx.AtomicOrdering.Monotonic,
                        syncscope="agent",
                    )
            if value != epoch:
                # Preserve the first failed wait: phase, waiting CTA and slot.
                record = (
                    fx.Int32(phase) * 1_000_000
                    + fx.Int32(fx.block_idx.x) * 1000
                    + index
                )
                fx.atomic_cas(
                    diag,
                    fx.Int32(0),
                    record,
                    success_ordering=fx.AtomicOrdering.Release,
                    failure_ordering=fx.AtomicOrdering.Monotonic,
                    syncscope="agent",
                )
            else:
                # The acquire fence must follow the flag read in this same
                # thread. The following workgroup barrier then shares its
                # acquired producer history with the rest of the CTA.
                fx.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Acquire)
    fx.gpu.barrier()
    if tid == 0:
        ok = (
            fx.generic_load(
                diag, memory_order=fx.AtomicOrdering.Monotonic, syncscope="agent"
            )
            == 0
        )
        fx.ptr_store(ok.select(fx.Int32(1), fx.Int32(0)), lds)
    fx.gpu.barrier()
    result = fx.ptr_load(lds) != 0
    fx.gpu.barrier()
    return result


def _kernel_name(rows, heads, workers):
    return f"atom_gfx1250_k1_projection_r{rows}_h{heads}_b{workers}"


@lru_cache(maxsize=24)
def _launcher(rows, heads, workers):
    # This launcher materializes the ELF only. Actual dispatch must go through
    # the residency-checked cooperative graph below, never this ordinary JIT
    # launch directly.
    @flyc.jit
    def launch(
        x: fx.Pointer,
        nw: fx.Pointer,
        wa: fx.Pointer,
        sa: fx.Pointer,
        qnw: fx.Pointer,
        knw: fx.Pointer,
        wb: fx.Pointer,
        sb: fx.Pointer,
        cos: fx.Pointer,
        sin: fx.Pointer,
        query: fx.Pointer,
        kv: fx.Pointer,
        hn: fx.Pointer,
        hx: fx.Pointer,
        hs: fx.Pointer,
        qkv: fx.Pointer,
        q: fx.Pointer,
        qs: fx.Pointer,
        state: fx.Pointer,
        stream: fx.Stream,
    ):
        @flyc.kernel(
            name=_kernel_name(rows, heads, workers), known_block_size=[THREADS, 1, 1]
        )
        def kernel(
            x: fx.Pointer,
            nw: fx.Pointer,
            wa: fx.Pointer,
            sa: fx.Pointer,
            qnw: fx.Pointer,
            knw: fx.Pointer,
            wb: fx.Pointer,
            sb: fx.Pointer,
            cos: fx.Pointer,
            sin: fx.Pointer,
            query: fx.Pointer,
            kv: fx.Pointer,
            hn: fx.Pointer,
            hx: fx.Pointer,
            hs: fx.Pointer,
            qkv: fx.Pointer,
            q: fx.Pointer,
            qs: fx.Pointer,
            state: fx.Pointer,
        ):
            tid, bid = fx.Int32(fx.thread_idx.x), fx.Int32(fx.block_idx.x)
            arena = fx.SharedAllocator(static=False)
            arena.allocate(max(gemv_lds_bytes(rows, 5120), grouped_lds_bytes(rows, 4)))
            lds_addr = fx.Int64(fx.ptrtoint(arena.base_ptr))
            local = _i32_pointer(arena.base_ptr)
            flags = _i32_pointer(state)
            diag = flags + 3 * workers
            # Per-worker completion epochs avoid a reset launch and are fresh
            # under graph replay. Same-stream reuse is a required contract.
            if tid == 0:
                previous = fx.generic_load(
                    flags + 2 * workers + bid,
                    memory_order=fx.AtomicOrdering.Acquire,
                    syncscope="agent",
                )
                fx.ptr_store(previous + 1, local)
            fx.gpu.barrier()
            epoch = fx.ptr_load(local)
            fx.gpu.barrier()
            if bid < 112:
                aops = projection_load(wa, sa, tid, bid, 5120, 1792)
                if bid < rows:
                    hidden_norm_device(x, nw, hn, hx, hs, lds_addr, tid, bid, rows)
                    _publish(flags, bid, epoch)
                ready = _await(flags, 0, rows, epoch, diag, local, tid, workers, 1)
                if ready:
                    projection_ready(
                        hx,
                        hs,
                        wa,
                        sa,
                        cos,
                        sin,
                        qkv,
                        qkv,
                        aops,
                        lds_addr,
                        tid,
                        bid,
                        rows,
                        5120,
                        1792,
                    )
                    for task in range(bid + workers, 112, workers):
                        gemv_device(
                            hx,
                            hs,
                            wa,
                            sa,
                            cos,
                            sin,
                            qkv,
                            qkv,
                            lds_addr,
                            tid,
                            task,
                            rows,
                            5120,
                            1792,
                        )
                    _publish(flags, workers + bid, epoch)
            if bid < heads * 8:
                bops = grouped_load(wb, sb, tid, bid, heads * 512, 4)
                if bid < rows:
                    completed = _await(
                        flags,
                        workers,
                        min(workers, 112),
                        epoch,
                        diag,
                        local,
                        tid,
                        workers,
                        2,
                    )
                    if completed:
                        qkv_norm_device(
                            qkv, qnw, knw, cos, sin, q, qs, kv, lds_addr, tid, bid, rows
                        )
                        _publish(flags, 2 * workers + bid, epoch)
                normalized = _await(
                    flags, 2 * workers, rows, epoch, diag, local, tid, workers, 3
                )
                if normalized:
                    grouped_ready(
                        q,
                        qs,
                        wb,
                        sb,
                        cos,
                        sin,
                        query,
                        bops,
                        lds_addr,
                        tid,
                        bid,
                        rows,
                        heads * 512,
                        4,
                    )
                    for task in range(bid + workers, heads * 8, workers):
                        bops = grouped_load(wb, sb, tid, task, heads * 512, 4)
                        grouped_ready(
                            q,
                            qs,
                            wb,
                            sb,
                            cos,
                            sin,
                            query,
                            bops,
                            lds_addr,
                            tid,
                            task,
                            rows,
                            heads * 512,
                            4,
                        )
            if tid == 0:
                fx.generic_store(
                    flags + 2 * workers + bid,
                    epoch,
                    memory_order=fx.AtomicOrdering.Monotonic,
                    syncscope="agent",
                )

        kernel(
            x,
            nw,
            wa,
            sa,
            qnw,
            knw,
            wb,
            sb,
            cos,
            sin,
            query,
            kv,
            hn,
            hx,
            hs,
            qkv,
            q,
            qs,
            state,
        ).launch(grid=(workers, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch


class _NodeAttribute(ctypes.Union):
    _fields_ = [
        ("pad", ctypes.c_char * 64),
        ("cooperative", ctypes.c_int),
        ("_alignment", ctypes.c_uint64),
    ]


class _CooperativeKernel:
    """Own a HIP module and launch its pointer-only ABI cooperatively.

    FlyDSL currently lowers launch_func without cooperative dispatch. Loading
    its compiled ELF explicitly permits setting and verifying cooperative
    graph-node attributes before any dispatch. The cached object retains its
    module for graph lifetime. Eager graphs retain their buffers until process
    exit; this prototype expects a fixed, bounded set of scratch allocations.
    """

    def __init__(self, launcher, args, name, rows, workers, device):
        _preload_compiled(launcher, *args)
        resources = launcher_resources(launcher)
        if len(resources) != 1 or any(
            resources[0][k] for k in ("scratch", "vgpr_spill", "sgpr_spill")
        ):
            raise RuntimeError(f"[gfx1250 K1] unverified resources: {resources}")
        paths = {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "libamdhip64.so" in line
        }
        if len(paths) != 1:
            raise RuntimeError("[gfx1250 K1] cannot identify the active HIP library")
        self.hip = ctypes.CDLL(paths.pop())
        P, U = ctypes.c_void_p, ctypes.c_uint
        self.hip.hipModuleLoadData.argtypes = [ctypes.POINTER(P), P]
        self.hip.hipModuleGetFunction.argtypes = [ctypes.POINTER(P), P, ctypes.c_char_p]
        self.hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            P,
            ctypes.c_int,
            ctypes.c_size_t,
        ]
        self.hip.hipDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.hip.hipModuleLaunchKernel.argtypes = [
            P,
            U,
            U,
            U,
            U,
            U,
            U,
            U,
            P,
            ctypes.POINTER(P),
            P,
        ]
        self.hip.hipStreamGetCaptureInfo_v2.argtypes = [
            P,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(P),
            ctypes.POINTER(ctypes.POINTER(P)),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.hip.hipGraphKernelNodeSetAttribute.argtypes = [
            P,
            ctypes.c_int,
            ctypes.POINTER(_NodeAttribute),
        ]
        self.hip.hipGraphKernelNodeGetAttribute.argtypes = [
            P,
            ctypes.c_int,
            ctypes.POINTER(_NodeAttribute),
        ]
        support = ctypes.c_int()
        self._check(self.hip.hipDeviceGetAttribute(ctypes.byref(support), 10, device))
        if support.value != 1:
            raise RuntimeError("[gfx1250 K1] cooperative launch unavailable")
        artifacts = list(launcher._mem_cache.values())
        if len(artifacts) != 1:
            raise RuntimeError("[gfx1250 K1] ambiguous compiled artifact")
        self.binary = ctypes.create_string_buffer(_code_object(artifacts[0]._ir_text))
        self.module, self.function = P(), P()
        self._check(self.hip.hipModuleLoadData(ctypes.byref(self.module), self.binary))
        self._check(
            self.hip.hipModuleGetFunction(
                ctypes.byref(self.function), self.module, name.encode()
            )
        )
        self.smem = max(gemv_lds_bytes(rows, 5120), grouped_lds_bytes(rows, 4))
        blocks = ctypes.c_int()
        self._check(
            self.hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
                ctypes.byref(blocks), self.function, THREADS, self.smem
            )
        )
        self.capacity = (
            blocks.value
            * torch.cuda.get_device_properties(device).multi_processor_count
        )
        if blocks.value < 1 or workers > self.capacity:
            raise RuntimeError(
                f"[gfx1250 K1] grid {workers} exceeds cooperative capacity {self.capacity}"
            )
        self.resources, self.workers = resources, workers
        self.graphs = {}

    @staticmethod
    def _check(code):
        if code:
            raise RuntimeError(f"[gfx1250 K1] HIP error {code}; stop dispatch")

    def __call__(self, tensors, stream):
        cells = [ctypes.c_void_p(t.data_ptr()) for t in tensors]
        params = (ctypes.c_void_p * len(cells))(*[ctypes.addressof(c) for c in cells])
        if torch.cuda.is_current_stream_capturing():
            # Capture without dispatch, then mark this exact module node
            # cooperative before graph instantiation or replay. The direct
            # module cooperative API in this HIP build crashes after capture.
            self._check(
                self.hip.hipModuleLaunchKernel(
                    self.function,
                    self.workers,
                    1,
                    1,
                    THREADS,
                    1,
                    1,
                    self.smem,
                    stream,
                    params,
                    None,
                )
            )
            status, token, graph = (
                ctypes.c_int(),
                ctypes.c_ulonglong(),
                ctypes.c_void_p(),
            )
            dependencies, count = ctypes.POINTER(ctypes.c_void_p)(), ctypes.c_size_t()
            self._check(
                self.hip.hipStreamGetCaptureInfo_v2(
                    stream,
                    ctypes.byref(status),
                    ctypes.byref(token),
                    ctypes.byref(graph),
                    ctypes.byref(dependencies),
                    ctypes.byref(count),
                )
            )
            if status.value != 1 or count.value != 1:
                raise RuntimeError("[gfx1250 K1] ambiguous captured module node")
            attr = _NodeAttribute()
            attr.cooperative = 1
            self._check(
                self.hip.hipGraphKernelNodeSetAttribute(
                    dependencies[0], 2, ctypes.byref(attr)
                )
            )
            checked = _NodeAttribute()
            self._check(
                self.hip.hipGraphKernelNodeGetAttribute(
                    dependencies[0], 2, ctypes.byref(checked)
                )
            )
            if checked.cooperative != 1:
                raise RuntimeError("[gfx1250 K1] graph node is not cooperative")
        else:
            key = tuple(t.data_ptr() for t in tensors)
            if key not in self.graphs:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self(tensors, torch.cuda.current_stream().cuda_stream)
                # Keep all buffers and the module alive while a cached graph
                # refers to their raw pointers. Only one kernel is captured.
                self.graphs[key] = (graph, tuple(tensors))
            self.graphs[key][0].replay()


_COOPERATIVE = {}


def k1_projection_gfx1250(
    x, nw, wa, sa, qnw, knw, wb, sb, cos, sin, query, kv, scratch, *, workers=256
):
    """Single local K1 projection launch, rows1/2, TP2/TP4 head geometry.

    ``scratch`` contains caller-owned hn/hx/hs/qkv/q/qs tensors and zero-
    initialized int32 state[3*workers+1]. Do not share scratch across streams.
    Check state[-1] after validation and when diagnosing a failed step. A
    nonzero value invalidates outputs and forbids another dispatch with it;
    the first failed wait is encoded as phase*1_000_000 + CTA*1000 + slot.
    Inputs must be finite; weights use native gfx1250 preshuffle and raw E8M0.
    """
    rows = x.shape[0] if x.ndim == 2 else 0
    heads = query.shape[1] if query.ndim == 3 else 0
    if (
        rows not in (1, 2)
        or heads not in (16, 32)
        or workers not in (64, 112, 128, 256, 512)
    ):
        raise ValueError("[gfx1250 K1] invalid rows, heads or worker count")
    specs = [
        (x, (rows, 5120), torch.bfloat16),
        (nw, (5120,), torch.bfloat16),
        (wa, (1792, 5120), torch.float8_e4m3fn),
        (qnw, (1280,), torch.bfloat16),
        (knw, (512,), torch.bfloat16),
        (wb, (heads * 512, 1280), torch.float8_e4m3fn),
        (cos, (rows, 32), torch.bfloat16),
        (sin, (rows, 32), torch.bfloat16),
        (query, (rows, heads, 512), torch.bfloat16),
        (kv, (rows, 512), torch.bfloat16),
        (scratch["hn"], (rows, 5120), torch.bfloat16),
        (scratch["hx"], (rows, 5120), torch.float8_e4m3fn),
        (scratch["qkv"], (rows, 1792), torch.bfloat16),
        (scratch["q"], (rows, 1280), torch.float8_e4m3fn),
        (scratch["state"], (3 * workers + 1,), torch.int32),
    ]
    for t, shape, dtype in specs:
        if tuple(t.shape) != shape or t.dtype != dtype or not t.is_contiguous():
            raise ValueError(f"[gfx1250 K1] requires contiguous {dtype} {shape}")
    for t, shape, name in [
        (sa, (56, 160), "sa"),
        (sb, (heads * 16, 40), "sb"),
        (scratch["hs"], (rows, 160), "hs"),
        (scratch["qs"], (rows, 40), "qs"),
    ]:
        check_e8m0(t, name, "gfx1250 K1", shape)
    tensors = [x, nw, wa, sa, qnw, knw, wb, sb, cos, sin, query, kv] + [
        scratch[k] for k in ("hn", "hx", "hs", "qkv", "q", "qs", "state")
    ]
    if not x.is_cuda or any(t.device != x.device for t in tensors):
        raise ValueError("[gfx1250 K1] buffers must share one GPU")
    device = x.device.index
    if torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] != "gfx1250":
        raise RuntimeError("[gfx1250 K1] requires gfx1250")
    if torch.cuda.current_device() != device:
        raise RuntimeError("[gfx1250 K1] select the buffer device before dispatch")
    for i in range(10, len(tensors)):
        a = tensors[i]
        lo, hi = a.data_ptr(), a.data_ptr() + a.numel() * a.element_size()
        if any(
            lo < b.data_ptr() + b.numel() * b.element_size() and b.data_ptr() < hi
            for b in tensors[:i]
        ):
            raise ValueError("[gfx1250 K1] written buffers must not overlap")
    key = (device, rows, heads, workers)
    stream = torch.cuda.current_stream(device).cuda_stream
    if key not in _COOPERATIVE:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("[gfx1250 K1] warm up before graph capture")
        args = tuple(ptr_arg(t) for t in tensors) + (stream,)
        _COOPERATIVE[key] = _CooperativeKernel(
            _launcher(rows, heads, workers),
            args,
            _kernel_name(rows, heads, workers),
            rows,
            workers,
            device,
        )
    _COOPERATIVE[key](tensors, stream)
    return query, kv
