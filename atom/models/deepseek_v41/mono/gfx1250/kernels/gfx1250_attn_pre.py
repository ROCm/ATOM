# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Complete local K1: delayed mHC, attention/index projections and ring write.

Rows1/2 support TP2/TP4 head geometry and the FP8 index path. Index rows1
uses 288 cooperative CTAs (256 N64 wq_b tasks and 32 N128 IQ heads); index
rows2 uses 256 CTAs with N32 wq_b tasks. Q and KV publish independently;
IQ reduces, rotates and quantizes in registers. This block excludes K2,
index scoring, compression and peer communication. All buffers belong to
one stream; state starts at zero and any diagnostic invalidates the block.
"""

from functools import lru_cache
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
import torch
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg, check_e8m0
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_k1 import (
    _CooperativeKernel,
    _publish,
    _await,
    _i32_pointer,
)
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_mhc import slice_device, gate_device
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_index import (
    iw_device,
    iq_projection_quant_device,
    ring_device,
)
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_producers import (
    hidden_norm_device,
    qkv_norm_device,
)
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_gemv import (
    gemv_device,
    grouped_load,
    grouped_ready,
)

# Six publication/exit banks for the maximum 288-CTA grid plus diagnostic.
# Shorter grids retain the same final diagnostic location outside all banks.
STATE_SIZE = 6 * 288 + 1

ARG_COUNT = 43


class PreparedK1:
    """A validated, residency-checked binding, held for the graph lifetime."""

    def __init__(self, runtime, tensors, outputs):
        self.runtime, self.tensors, self.outputs = runtime, tuple(tensors), outputs

    def __call__(self):
        self.runtime(self.tensors, torch.cuda.current_stream().cuda_stream)
        return self.outputs


def prepare_attn_pre_kernel(key):
    """Load all variants before the first peer launch; do not dispatch."""
    device = torch.cuda.current_device()
    full_key = (device, *key)
    if full_key not in _RUNTIMES:
        import flydsl.compiler as flyc

        pointer = flyc.from_c_void_p(fx.Uint8, 0)
        rows, _, _, feeds_index, *_ = key
        _RUNTIMES[full_key] = _CooperativeKernel(
            _launcher(*key),
            (pointer,) * ARG_COUNT + (torch.cuda.current_stream().cuda_stream,),
            _name(*key), rows, _grid(rows, feeds_index), device,
            smem=(8192 if feeds_index and rows == 2 else 4096) + 32,
        )
    return _RUNTIMES[full_key]


def _grid(rows, feeds_index):
    return 288 if feeds_index and rows == 1 else 256


def _name(rows, heads, fold, feeds_index, pool_rows, ring_off, index_first, shared_generation=False):
    return f"atom_gfx1250_attn_pre_r{rows}_h{heads}_f{int(fold)}_ix{int(feeds_index)}_p{pool_rows}_o{ring_off}_s{int(index_first)}_iq_quant_handoff_gen{int(shared_generation)}"


@lru_cache(maxsize=32)
def _launcher(rows, heads, fold, feeds_index, pool_rows, ring_off, index_first, shared_generation=False):
    workers = _grid(rows, feeds_index)
    # Q/IW consumers occupy separate resident CTAs; no grid-wide barrier.
    iq_first = 256 if rows == 1 else 224
    consumer_workers = iq_first if feeds_index else workers
    groups = 2 if feeds_index and rows == 2 else 4

    @flyc.jit
    def launch(
        res: fx.Pointer,
        y: fx.Pointer,
        pre: fx.Pointer,
        post: fx.Pointer,
        comb: fx.Pointer,
        fn: fx.Pointer,
        scale: fx.Pointer,
        bias: fx.Pointer,
        nw: fx.Pointer,
        wa: fx.Pointer,
        sa: fx.Pointer,
        qnw: fx.Pointer,
        knw: fx.Pointer,
        wb: fx.Pointer,
        sb: fx.Pointer,
        wiq: fx.Pointer,
        siq: fx.Pointer,
        wiw: fx.Pointer,
        cos: fx.Pointer,
        sin: fx.Pointer,
        ring: fx.Pointer,
        rr: fx.Pointer,
        rout: fx.Pointer,
        collapse: fx.Pointer,
        pmix: fx.Pointer,
        preout: fx.Pointer,
        postout: fx.Pointer,
        combout: fx.Pointer,
        hn: fx.Pointer,
        hx: fx.Pointer,
        hs: fx.Pointer,
        qkv: fx.Pointer,
        q: fx.Pointer,
        qs: fx.Pointer,
        query: fx.Pointer,
        kv: fx.Pointer,
        iq: fx.Pointer,
        iw: fx.Pointer,
        iq8: fx.Pointer,
        iqs: fx.Pointer,
        swiw: fx.Pointer,
        state: fx.Pointer,
        generation: fx.Pointer,
        stream: fx.Stream,
    ):

        @flyc.kernel(
            name=_name(
                rows, heads, fold, feeds_index, pool_rows, ring_off, index_first, shared_generation
            ),
            known_block_size=[256, 1, 1],
        )
        def kernel(
            res: fx.Pointer,
            y: fx.Pointer,
            pre: fx.Pointer,
            post: fx.Pointer,
            comb: fx.Pointer,
            fn: fx.Pointer,
            scale: fx.Pointer,
            bias: fx.Pointer,
            nw: fx.Pointer,
            wa: fx.Pointer,
            sa: fx.Pointer,
            qnw: fx.Pointer,
            knw: fx.Pointer,
            wb: fx.Pointer,
            sb: fx.Pointer,
            wiq: fx.Pointer,
            siq: fx.Pointer,
            wiw: fx.Pointer,
            cos: fx.Pointer,
            sin: fx.Pointer,
            ring: fx.Pointer,
            rr: fx.Pointer,
            rout: fx.Pointer,
            collapse: fx.Pointer,
            pmix: fx.Pointer,
            preout: fx.Pointer,
            postout: fx.Pointer,
            combout: fx.Pointer,
            hn: fx.Pointer,
            hx: fx.Pointer,
            hs: fx.Pointer,
            qkv: fx.Pointer,
            q: fx.Pointer,
            qs: fx.Pointer,
            query: fx.Pointer,
            kv: fx.Pointer,
            iq: fx.Pointer,
            iw: fx.Pointer,
            iq8: fx.Pointer,
            iqs: fx.Pointer,
            swiw: fx.Pointer,
            state: fx.Pointer,
            generation: fx.Pointer,
        ):
            tid, bid = (fx.Int32(fx.thread_idx.x), fx.Int32(fx.block_idx.x))
            arena = fx.SharedAllocator(static=False)
            workspace = 8192 if feeds_index and rows == 2 else 4096
            arena.allocate(workspace + 32)
            addr = fx.Int64(fx.ptrtoint(arena.base_ptr))
            local = _i32_pointer(arena.base_ptr)
            # Status must never alias producer or projection payload LDS.
            status = local + workspace // 4
            flags = _i32_pointer(state)
            diag = flags + STATE_SIZE - 1
            if tid == 0:
                current = fx.Int32(0)
                if const_expr(shared_generation):
                    # A shared scratch crosses 256/288-CTA variants. Their
                    # completion banks have different offsets and some CTAs
                    # did not run the previous layer; never derive its epoch
                    # from a CTA's old completion. The layer metadata is fixed.
                    current = fx.ptr_load(_i32_pointer(generation)) + 1
                else:
                    current = fx.generic_load(
                        flags + 5 * workers + bid,
                        memory_order=fx.AtomicOrdering.Acquire,
                        syncscope="agent",
                    ) + 1
                fx.ptr_store(current, local)
            fx.gpu.barrier()
            epoch = fx.ptr_load(local)
            fx.gpu.barrier()
            if bid < 160:
                slice_device(
                    res,
                    y,
                    pre,
                    post,
                    comb,
                    fn,
                    rout,
                    collapse,
                    pmix,
                    arena.base_ptr,
                    tid,
                    bid,
                    rows,
                    fold,
                )
                _publish(flags, bid, epoch, synchronized=not feeds_index)
            if (bid >= 160) & (bid < 160 + rows):
                ready = _await(
                    flags,
                    0,
                    160,
                    epoch,
                    diag,
                    local,
                    tid,
                    workers,
                    1,
                    status=status if feeds_index and rows == 2 else None,
                )
                if ready:
                    hidden_norm_device(
                        collapse, nw, hn, hx, hs, addr, tid, bid - 160, rows, True, pmix
                    )
                    _publish(
                        flags, workers + bid - 160, epoch, synchronized=not feeds_index
                    )
            if (bid >= 162) & (bid < 162 + rows):
                ready = _await(
                    flags,
                    0,
                    160,
                    epoch,
                    diag,
                    local,
                    tid,
                    workers,
                    2,
                    status=status if feeds_index and rows == 2 else None,
                )
                if ready:
                    gate_device(
                        pmix,
                        scale,
                        bias,
                        preout,
                        postout,
                        combout,
                        arena.base_ptr,
                        tid,
                        bid - 162,
                        rows,
                    )
            if const_expr(feeds_index):
                if (bid >= 112) & (bid < 144):
                    ready = _await(
                        flags,
                        workers,
                        rows,
                        epoch,
                        diag,
                        local,
                        tid,
                        workers,
                        3,
                        status=status if feeds_index and rows == 2 else None,
                    )
                    if ready:
                        iw_device(hn, wiw, iw, arena.base_ptr, tid, bid - 112, rows)
                        _publish(
                            flags,
                            3 * workers + bid - 112,
                            epoch,
                            synchronized=not feeds_index,
                        )
            if bid < 112:
                ready = _await(
                    flags,
                    workers,
                    rows,
                    epoch,
                    diag,
                    local,
                    tid,
                    workers,
                    4,
                    status=status if feeds_index and rows == 2 else None,
                )
                if ready:
                    gemv_device(
                        hx,
                        hs,
                        wa,
                        sa,
                        cos,
                        sin,
                        qkv,
                        qkv,
                        addr,
                        tid,
                        bid,
                        rows,
                        5120,
                        1792,
                    )
                    _publish(
                        flags, 2 * workers + bid, epoch, synchronized=not feeds_index
                    )
            # Only the 80 Q tiles delay Q consumers; KV/ring are independent.
            if (bid >= 164) & (bid < 164 + rows):
                ready = _await(
                    flags,
                    2 * workers,
                    80,
                    epoch,
                    diag,
                    local,
                    tid,
                    workers,
                    5,
                    status=status if feeds_index and rows == 2 else None,
                )
                if ready:
                    row = bid - 164
                    qkv_norm_device(
                        qkv,
                        qnw,
                        knw,
                        cos,
                        sin,
                        q,
                        qs,
                        kv,
                        addr,
                        tid,
                        row,
                        rows,
                        q_only=True,
                    )
                    _publish(
                        flags, 4 * workers + row, epoch, synchronized=not feeds_index
                    )
            if (bid >= 166) & (bid < 166 + rows):
                ready = _await(
                    flags,
                    2 * workers + 80,
                    32,
                    epoch,
                    diag,
                    local,
                    tid,
                    workers,
                    9,
                    status=status if feeds_index and rows == 2 else None,
                )
                if ready:
                    row = bid - 166
                    qkv_norm_device(
                        qkv,
                        qnw,
                        knw,
                        cos,
                        sin,
                        q,
                        qs,
                        kv,
                        addr,
                        tid,
                        row,
                        rows,
                        kv_only=True,
                    )
                    ring_device(kv, ring, rr, diag, tid, row, rows, pool_rows, ring_off)
            # Acquire Q and the matching IW together, before the fused IQ body.
            ready = fx.Int32(1) == 1
            if const_expr(feeds_index):
                if (bid >= iq_first) & (bid < iq_first + 32):
                    head = bid - iq_first
                    ready = _await(
                        flags,
                        4 * workers,
                        rows,
                        epoch,
                        diag,
                        local,
                        tid,
                        workers,
                        7,
                        status=status,
                        extra_offset=3 * workers + head,
                        poll_sleep=(
                            8 if feeds_index and rows == 1 else 2 if rows == 1 else 1
                        ),
                    )
                else:
                    ready = _await(
                        flags,
                        4 * workers,
                        rows,
                        epoch,
                        diag,
                        local,
                        tid,
                        workers,
                        7,
                        status=status,
                        poll_sleep=(
                            8 if feeds_index and rows == 1 else 2 if rows == 1 else 1
                        ),
                    )
            else:
                ready = _await(
                    flags,
                    4 * workers,
                    rows,
                    epoch,
                    diag,
                    local,
                    tid,
                    workers,
                    7,
                    status=status,
                    poll_sleep=(
                        8 if feeds_index and rows == 1 else 2 if rows == 1 else 1
                    ),
                )
            if ready:
                if const_expr(feeds_index):
                    if (bid >= iq_first) & (bid < iq_first + 32):
                        head = bid - iq_first
                        ops = grouped_load(wiq, siq, tid, head, 4096, 8)
                        iq_projection_quant_device(
                            q,
                            qs,
                            cos,
                            sin,
                            iq,
                            iq8,
                            iqs,
                            iw,
                            swiw,
                            ops,
                            addr,
                            tid,
                            head,
                            rows,
                        )
                    if bid < consumer_workers:
                        for task in range(
                            bid, heads * (32 // groups), consumer_workers
                        ):
                            ops = grouped_load(wb, sb, tid, task, heads * 512, groups)
                            grouped_ready(
                                q,
                                qs,
                                wb,
                                sb,
                                cos,
                                sin,
                                query,
                                ops,
                                addr,
                                tid,
                                task,
                                rows,
                                heads * 512,
                                groups,
                            )
                else:
                    for task in range(bid, heads * (32 // groups), consumer_workers):
                        ops = grouped_load(wb, sb, tid, task, heads * 512, groups)
                        grouped_ready(
                            q,
                            qs,
                            wb,
                            sb,
                            cos,
                            sin,
                            query,
                            ops,
                            addr,
                            tid,
                            task,
                            rows,
                            heads * 512,
                            groups,
                        )
            # Same-stream replay advances every CTA, including idle workers.
            if tid == 0:
                fx.generic_store(
                    flags + 5 * workers + bid,
                    epoch,
                    memory_order=fx.AtomicOrdering.Monotonic,
                    syncscope="agent",
                )

        kernel(
            res,
            y,
            pre,
            post,
            comb,
            fn,
            scale,
            bias,
            nw,
            wa,
            sa,
            qnw,
            knw,
            wb,
            sb,
            wiq,
            siq,
            wiw,
            cos,
            sin,
            ring,
            rr,
            rout,
            collapse,
            pmix,
            preout,
            postout,
            combout,
            hn,
            hx,
            hs,
            qkv,
            q,
            qs,
            query,
            kv,
            iq,
            iw,
            iq8,
            iqs,
            swiw,
            state,
            generation,
        ).launch(grid=(workers, 1, 1), block=(256, 1, 1), stream=stream)

    return launch


_RUNTIMES = {}


def attn_pre_gfx1250(
    residual,
    y,
    pre,
    post,
    comb,
    weights,
    cos,
    sin,
    buffers,
    ring,
    ring_rows,
    *,
    fold,
    feeds_index,
    ring_off=0,
    index_first=True,
    _prepare_only=False,
    generation=None,
):
    """Run the complete local K1 seam, without changing the model dispatcher.

    ``weights`` uses checkpoint suffixes for fn/scale/base/norm and q/kv norm;
    wa/sa and wb/sb are native preshuffled MXFP8 projections. wiq/siq and wiw
    are replicated index projections. ``buffers`` owns rout/collapse/partial,
    pre/post/comb, hn/hx/hs/qkv/q/qs/query/kv, iq/iw/iq8/iqs/swiw and state.
    No-fold returns residual by alias and does not write rout. Unused index
    buffers are not accessed. state[int32,1729] must begin at zero and belong
    to this stream. Finite inputs are required; ring metadata is checked on
    device on each replay, including -1 pads, bounds and duplicate rows.
    IQ uses independent CTAs; index_first is retained for API compatibility
    and does not alter this schedule. A nonzero diagnostic invalidates all
    outputs and forbids scratch reuse.
    """
    rows = residual.shape[0] if residual.ndim == 3 else 0
    heads = buffers["query"].shape[1] if buffers["query"].ndim == 3 else 0
    if rows not in (1, 2) or heads not in (16, 32):
        raise ValueError("[gfx1250 complete K1] requires rows1/2 and heads16/32")
    if (
        not isinstance(ring_off, int)
        or ring_off < 0
        or ring.ndim != 2
        or (ring.shape[1] != 512)
        or (ring.shape[0] < 1)
    ):
        raise ValueError("[gfx1250 complete K1] invalid ring geometry")
    w, b = (weights, buffers)
    values = dict(
        res=residual,
        y=y,
        pre=pre,
        post=post,
        comb=comb,
        fn=w["hc_attn_fn"],
        scale=w["hc_attn_scale"],
        bias=w["hc_attn_base"],
        nw=w["attn_norm.weight"],
        wa=w["wa"],
        sa=w["sa"],
        qnw=w["attn.q_norm.weight"],
        knw=w["attn.kv_norm.weight"],
        wb=w["wb"],
        sb=w["sb"],
        wiq=w.get("wiq", w["wb"]),
        siq=w.get("siq", w["sb"]),
        wiw=w.get("wiw", w["attn_norm.weight"]),
        cos=cos,
        sin=sin,
        ring=ring,
        rr=ring_rows,
        generation=b["state"][:1] if generation is None else generation,
        rout=b["rout"] if fold else residual,
        collapse=b["collapse"],
        pmix=b["partial"],
        preout=b["pre"],
        postout=b["post"],
        combout=b["comb"],
        **{
            k: b[k]
            for k in ("hn", "hx", "hs", "qkv", "q", "qs", "query", "kv", "state")
        },
        **{
            k: b[k] if feeds_index else b["kv"]
            for k in ("iq", "iw", "iq8", "iqs", "swiw")
        },
    )
    bf = torch.bfloat16
    f32 = torch.float32
    fp8 = torch.float8_e4m3fn
    specs = dict(
        res=((rows, 4, 5120), bf),
        y=((rows, 5120), bf),
        pre=((rows, 4), f32),
        post=((rows, 4), f32),
        comb=((rows, 4, 4), f32),
        fn=((24, 20480), f32),
        scale=((3,), f32),
        bias=((24,), f32),
        nw=((5120,), bf),
        wa=((1792, 5120), fp8),
        qnw=((1280,), bf),
        knw=((512,), bf),
        wb=((heads * 512, 1280), fp8),
        cos=((rows, 32), bf),
        sin=((rows, 32), bf),
        ring=((ring.shape[0], 512), bf),
        rr=((rows,), torch.int32),
        rout=((rows, 4, 5120), bf),
        collapse=((rows, 5120), bf),
        pmix=((rows, 160, 26), f32),
        preout=((rows, 4), f32),
        postout=((rows, 4), f32),
        combout=((rows, 4, 4), f32),
        hn=((rows, 5120), bf),
        hx=((rows, 5120), fp8),
        qkv=((rows, 1792), bf),
        q=((rows, 1280), fp8),
        query=((rows, heads, 512), bf),
        kv=((rows, 512), bf),
        state=((STATE_SIZE,), torch.int32),
    )
    if feeds_index:
        specs.update(
            wiq=((4096, 1280), fp8),
            wiw=((32, 5120), bf),
            iq=((rows, 32, 128), bf),
            iw=((rows, 32), bf),
            iq8=((rows, 32, 128), fp8),
            iqs=((rows, 32, 1), f32),
            swiw=((rows, 32), f32),
        )
    if generation is not None:
        specs["generation"] = ((1,), torch.int32)
    for k, (shape, dtype) in specs.items():
        t = values[k]
        if tuple(t.shape) != shape or t.dtype != dtype or (not t.is_contiguous()):
            raise ValueError(
                f"[gfx1250 complete K1] {k} must be contiguous {dtype} {shape}"
            )
    for k, shape in [
        ("sa", (56, 160)),
        ("sb", (heads * 16, 40)),
        ("hs", (rows, 160)),
        ("qs", (rows, 40)),
    ] + ([("siq", (128, 40))] if feeds_index else []):
        check_e8m0(values[k], k, "gfx1250 complete K1", shape)
    tensors = [
        values[k]
        for k in [
            "res",
            "y",
            "pre",
            "post",
            "comb",
            "fn",
            "scale",
            "bias",
            "nw",
            "wa",
            "sa",
            "qnw",
            "knw",
            "wb",
            "sb",
            "wiq",
            "siq",
            "wiw",
            "cos",
            "sin",
            "ring",
            "rr",
            "rout",
            "collapse",
            "pmix",
            "preout",
            "postout",
            "combout",
            "hn",
            "hx",
            "hs",
            "qkv",
            "q",
            "qs",
            "query",
            "kv",
            "iq",
            "iw",
            "iq8",
            "iqs",
            "swiw",
            "state",
            "generation",
        ]
    ]
    device = residual.device.index
    if not residual.is_cuda or any((t.device != residual.device for t in tensors)):
        raise ValueError("[gfx1250 complete K1] all tensors must share a GPU")
    if (
        torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] != "gfx1250"
        or torch.cuda.current_device() != device
    ):
        raise RuntimeError("[gfx1250 complete K1] select the gfx1250 buffer device")
    written = [
        "ring",
        "collapse",
        "pmix",
        "preout",
        "postout",
        "combout",
        "hn",
        "hx",
        "hs",
        "qkv",
        "q",
        "qs",
        "query",
        "kv",
        "state",
    ]
    if fold:
        written.append("rout")
    if feeds_index:
        written.extend(["iq", "iw", "iq8", "iqs", "swiw"])
    active = [
        k
        for k in values
        if (
            k not in ("wiq", "siq", "wiw", "iq", "iw", "iq8", "iqs", "swiw")
            or feeds_index
        )
        and (k != "rout" or fold)
        and (k != "generation" or generation is not None)
    ]
    for k in written:
        a = values[k]
        lo, hi = (a.data_ptr(), a.data_ptr() + a.numel() * a.element_size())
        for j in active:
            if j == k:
                continue
            c = values[j]
            if lo < c.data_ptr() + c.numel() * c.element_size() and c.data_ptr() < hi:
                raise ValueError(f"[gfx1250 complete K1] written {k} overlaps {j}")
    key = (
        device,
        rows,
        heads,
        bool(fold),
        bool(feeds_index),
        ring.shape[0],
        ring_off,
        bool(index_first),
        generation is not None,
    )
    workers = _grid(rows, feeds_index)
    stream = torch.cuda.current_stream(device).cuda_stream
    if key not in _RUNTIMES:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("[gfx1250 complete K1] warm up before graph capture")
        params = key[1:]
        _RUNTIMES[key] = _CooperativeKernel(
            _launcher(*params),
            tuple((ptr_arg(t) for t in tensors)) + (stream,),
            _name(*params),
            rows,
            workers,
            device,
            smem=(8192 if feeds_index and rows == 2 else 4096) + 32,
        )
    outputs = (values["rout"], b["pre"], b["post"], b["comb"], b["query"], b["kv"])
    prepared = PreparedK1(_RUNTIMES[key], tensors, outputs)
    if _prepare_only:
        return prepared
    return prepared()
