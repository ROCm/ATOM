# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A DSpark draft layer as persistent launches (gfx1250, TP4, decode blocks of <= 8 rows): two launches around
the production paged attention.

* pre (``kernels/dspark_draft_pre_gfx1250.py``): the fused qkv MXFP8 GEMV, the q / k Gemma RMSNorm, the neox RoPE
  of the first 64 lanes and the per-token fp8 K / V cache writes -- production's qkv GEMM and
  ``triton_fused_norm_rope_cache`` in one launch.
* post (``kernels/dspark_draft_post_gfx1250.py``): o_proj, its all-reduce + residual + post-attention Gemma
  RMSNorm (+ MXFP8), gate_up, SwiGLU-OAI (+ MXFP8), down, its all-reduce + residual + the next layer's input
  norm (+ MXFP8) or the final norm -- production's quantize_fp8, o_proj, k5, gate_up, swiglu_oai_mxfp8, down and
  k5 in one launch. The all-reduces push into and poll the custom all-reduce's k5 inbox with its counters (no
  allocation or mapping of their own), so they interleave with k5 calls.

Numerics: checked on one GPU against the production launches on the draft checkpoint's weights (emulated
peers for the all-reduces): q, the K / V cache rows and scales, x after the first all-reduce + norm, h, both
all-reduced outputs and the k5 inbox / counter state are byte-identical. The GEMV chains are production's split
chains (qkv, gate_up: its K quarters, summed in split order); o_proj and down sum four K quarters in wave order
(production: one chain), which can move a partial by one bf16 ulp (1 of 43,008 values on one rank of the check).

Served when (``usable``): gfx1250, TP4 with the k5 one-shot all-reduce context, rows 1..8, preshuffled MXFP8
1x32 linears at the draft's shapes, the fp8 KV cache. Anything else keeps the production path. Bounded waits
count in an error word; ``poll_errors`` reports it.
"""

import ctypes
import logging

import torch

logger = logging.getLogger("atom")

HIDDEN = 6144
INTER = 3072
QSIZE = 2048
MAX_ROWS = 8
# Measured choices (single GPU, emulated peers; TP4 gate): whether o_proj + its all-reduce run inside the post
# launch (else production o_proj + k5, then the MLP launch), and the post launch's gate_up weight-load delay.
# 11:50: the merged post launch does not beat production (29.00 vs 28.71 us a layer, one GPU, best setting: its
# 1160 workgroups at 408 VGPRs leave ~512 resident, so its later roles start late); the MLP launch does.
POST_FUSES_OPROJ = False
POST_GU_DELAY = 0
# The block attention launch: correct (fp64 error below production's) but 12.27 vs 10.99 us a layer (one GPU,
# plan once a pass) as of 11:40: production PA until it is faster.
PA_LAUNCH = False
_state: dict = {}  # device index -> (post scratch, counters, pre partials, PA scratch, PA counters)
_info = None
_calls = 0
_err_reported = 0


def _k5_info():
    """(inbox addresses, counter address, slot, rank, max_rows) of the TP group's k5 context, or None."""
    global _info
    if _info is not None:
        return _info or None
    _info = False
    try:
        from aiter.dist.parallel_state import get_tp_group

        group = get_tp_group()
        ca = getattr(group.device_communicator, "ca_comm", None)
        ctx = int(getattr(ca, "_k5_ctx", 0) or 0) if ca is not None else 0
        if not ctx:
            return None
        # aiter's k5ar::Ctx: int rank; int64 max_rows; int64 slot; char* inbox[4]; uint32* ctr
        raw = ctypes.string_at(ctx, 64)
        rank = int.from_bytes(raw[0:4], "little", signed=True)
        max_rows = int.from_bytes(raw[8:16], "little", signed=True)
        slot = int.from_bytes(raw[16:24], "little", signed=True)
        inbox = [int.from_bytes(raw[24 + 8 * i : 32 + 8 * i], "little") for i in range(4)]
        ctr = int.from_bytes(raw[56:64], "little")
        if not (
            rank == group.rank_in_group
            and 0 < max_rows <= 4096
            and slot == max_rows * (HIDDEN // 8)
            and ctr == inbox[rank] + 3 * 4 * slot * 16
        ):
            logger.warning("DSpark draft launches: unexpected k5 context layout, not used")
            return None
        _info = (inbox, ctr, slot, rank, max_rows)
    except Exception as e:  # noqa: BLE001
        logger.warning("DSpark draft launches: no k5 context (%s)", e)
        return None
    return _info


def _mxfp8(lin, n, k) -> bool:
    return (
        getattr(lin, "native_a8_group_rows", None) == 1
        and getattr(lin.weight, "is_shuffled", False)
        and lin.weight.dtype == torch.float8_e4m3fn
        and tuple(lin.weight.shape) == (n, k)
        and tuple(lin.weight_scale.shape) == (n, k // 32)
        and getattr(lin, "bias", None) is None
    )


def usable(layer, rows: int, cache) -> bool:
    """The same answer on every rank: shapes, arch, the k5 context, the cache format."""
    from aiter.dist.parallel_state import get_tensor_model_parallel_world_size
    from aiter.jit.utils.chip_info import get_gfx_runtime

    attn, mlp = layer.self_attn, layer.mlp
    if cache is None or not 0 < rows <= MAX_ROWS:
        return False
    k16, v16, ks16, vs16 = cache
    return (
        get_gfx_runtime() == "gfx1250"
        and get_tensor_model_parallel_world_size() == 4
        and attn.num_heads == 16
        and attn.num_kv_heads == 1
        and attn.head_dim == 128
        and attn.rotary_emb.rotary_dim == 64
        and k16.dtype == torch.float8_e4m3fn
        and tuple(k16.shape[1:]) == (1, 8, 16, 16)
        and tuple(v16.shape[1:]) == (1, 1, 128, 16)
        and ks16 is not None
        and ks16.dtype == torch.float32
        and all(t.is_contiguous() for t in cache)
        and _mxfp8(attn.qkv_proj, QSIZE + 256, HIDDEN)
        and _mxfp8(attn.o_proj, HIDDEN, QSIZE)
        and _mxfp8(mlp.gate_up_proj, 2 * INTER, HIDDEN)
        and _mxfp8(mlp.down_proj, HIDDEN, INTER)
        and mlp.swiglu_limit is not None
        and _k5_info() is not None
    )


def _scratch(device):
    from atom.model_ops.kernels import dspark_draft_post_gfx1250 as post
    from atom.model_ops.kernels import dspark_draft_pre_gfx1250 as pre

    st = _state.get(device.index)
    if st is None:
        if torch.cuda.is_current_stream_capturing():
            return None  # first seen inside a capture: production this time
        # the counters are never reset (epoch-tagged): zeroed once, in normal device memory
        from atom.model_ops.kernels import dspark_draft_pa_gfx1250 as pa

        st = (
            torch.zeros(post.SCR_BYTES, dtype=torch.uint8, device=device),
            torch.zeros(post.CTL_BYTES, dtype=torch.uint8, device=device),
            torch.zeros(pre.PART_BYTES, dtype=torch.uint8, device=device),
            torch.zeros(pa.SCR_BYTES, dtype=torch.uint8, device=device),
            torch.zeros(pa.CTL_BYTES, dtype=torch.uint8, device=device),
        )
        _state[device.index] = st
    return st


def pre_launch(attn, x, x_scale, positions, slots, cache):
    """q bf16 [rows, 16, 128] (normed, roped); the block's K / V rows written into ``cache``. None: production."""
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels.tensor_shim import (
        _run_compiled,
        current_stream_arg,
        ptr_arg,
    )

    from atom.model_ops.kernels import dspark_draft_pre_gfx1250 as k

    st = _scratch(x.device)
    if st is None:
        return None
    part = st[2]
    k16, v16, ks16, vs16 = cache
    cos = attn.rotary_emb.cos_cache.reshape(-1, 32)
    sin = attn.rotary_emb.sin_cache.reshape(-1, 32)
    if cos.dtype not in (torch.float32, torch.bfloat16) or positions.dtype not in (torch.int32, torch.int64):
        return None
    m = x.shape[0]
    q = torch.empty((m, 16, 128), dtype=torch.bfloat16, device=x.device)
    w = attn.qkv_proj
    u8 = lambda t: t.view(torch.uint8)  # noqa: E731
    spec = (
        int(cos.dtype == torch.bfloat16),
        0,
        int(positions.dtype == torch.int64),
        int(slots.dtype == torch.int64),
    )
    _run_compiled(
        k.launch_draft_pre,
        ptr_arg(x), ptr_arg(u8(x_scale)), ptr_arg(w.weight), ptr_arg(u8(w.weight_scale)),
        ptr_arg(attn.q_norm.weight), ptr_arg(attn.k_norm.weight), ptr_arg(cos), ptr_arg(sin),
        ptr_arg(positions), ptr_arg(slots), ptr_arg(u8(k16)), ptr_arg(u8(v16)), ptr_arg(ks16), ptr_arg(vs16),
        ptr_arg(q), ptr_arg(part), ptr_arg(part, fx.Int32),
        int(m), int(x.stride(0)), int(cos.stride(0)), float(attn.q_norm.variance_epsilon),
        current_stream_arg(),
        *spec,
        specialization_key=("dspark_draft_pre", x.device.index) + spec,
    )  # fmt: skip
    return q


def post_launch(attn_out, o_proj, mlp, residual, post_norm, next_norm, fp8_out: bool):
    """o_proj -> all-reduce + residual + post_norm -> MLP -> all-reduce + residual + next_norm. Returns
    (out, out_scale | None, residual_out), or None (the caller runs production)."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels.tensor_shim import (
        _run_compiled,
        current_stream_arg,
        ptr_arg,
    )

    from atom.model_ops.kernels import dspark_draft_post_gfx1250 as k

    st = _scratch(attn_out.device)
    if st is None:
        return None
    scr, ctl = st[0], st[1]
    inbox, ctr, slot, rank, max_rows = _k5_info()
    m = attn_out.shape[0]
    dev = attn_out.device
    out = torch.empty(
        (m, HIDDEN), dtype=torch.float8_e4m3fn if fp8_out else torch.bfloat16, device=dev
    )
    out_scale = (
        torch.empty((m, HIDDEN // 32), dtype=torch.float8_e8m0fnu, device=dev) if fp8_out else None
    )
    residual = residual.contiguous()
    res_out = torch.empty_like(residual)
    w1, w2 = mlp.gate_up_proj, mlp.down_proj
    u8 = lambda t: t.view(torch.uint8)  # noqa: E731
    spec = (
        1,
        int(fp8_out),
        int(max_rows),
        0,
        float(mlp.swiglu_alpha),
        float(mlp.swiglu_beta),
        float(mlp.swiglu_limit),
        POST_GU_DELAY,
        16,
    )
    _run_compiled(
        k.launch_draft_post,
        ptr_arg(attn_out), ptr_arg(attn_out), ptr_arg(o_proj.weight), ptr_arg(u8(o_proj.weight_scale)),
        ptr_arg(w1.weight), ptr_arg(u8(w1.weight_scale)), ptr_arg(w2.weight), ptr_arg(u8(w2.weight_scale)),
        ptr_arg(residual), ptr_arg(post_norm.weight), ptr_arg(next_norm.weight),
        ptr_arg(out), ptr_arg(u8(out_scale) if fp8_out else residual), ptr_arg(res_out),
        flyc.from_c_void_p(fx.Int32, int(ctr)), ptr_arg(scr), ptr_arg(ctl), ptr_arg(scr, fx.Int32),
        int(inbox[0]), int(inbox[1]), int(inbox[2]), int(inbox[3]),
        int(m), int(attn_out.stride(0) * attn_out.element_size()), int(slot), int(rank), int(k.MAXIT),
        float(next_norm.variance_epsilon),
        current_stream_arg(),
        *spec,
        specialization_key=("dspark_draft_post", dev.index) + spec,
    )  # fmt: skip
    return out, out_scale, res_out


def poll_errors(every: int = 512) -> None:
    """Report a bounded wait that gave up (read every ``every`` calls: a sync)."""
    global _calls
    _calls += 1
    if _calls % every or not _state:
        return
    from atom.model_ops.kernels import dspark_draft_pa_gfx1250 as pa
    from atom.model_ops.kernels import dspark_draft_post_gfx1250 as k

    for st in _state.values():
        for ctl, word in ((st[1], k.ERR), (st[4], pa.ERR)):
            _report(ctl[4 * word : 4 * word + 4].view(torch.int32))


def _report(err) -> None:
    global _err_reported
    v = int(err.item())
    if v:
        _err_reported += 1
        err.zero_()
        logger.error(
            "DSpark draft launches: %d bounded waits gave up (report %d); that block's drafts were wrong, "
            "verification rejected them",
            v,
            _err_reported,
        )


def pa_usable(window: int, rows: int, bt16_rows, ctx_rows) -> bool:
    """The block attention launch: ONE request's rows (the 1-request block), a sliding window that bounds a row's
    keys by the launch's 72 chunks of 64 (window + page offset + the block <= 4608), int32 tables."""
    from atom.model_ops.kernels import dspark_draft_pa_gfx1250 as pa

    return (
        PA_LAUNCH
        and 0 < window <= pa.N_CH * pa.KC - 16 - pa.MAXROWS
        and 0 < rows <= pa.MAXROWS
        and bt16_rows is not None
        and bt16_rows.dtype == torch.int32
        and bt16_rows.stride(1) == 1
        and ctx_rows.dtype == torch.int32
        and ctx_rows.is_contiguous()
    )


def pa_launch(q, cache, bt16_rows, ctx_rows, sm_scale):
    """[context | block] attention of one request's block rows: bf16 [rows, 16, 128], or None (production)."""
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels.tensor_shim import (
        _run_compiled,
        current_stream_arg,
        ptr_arg,
    )

    from atom.model_ops.kernels import dspark_draft_pa_gfx1250 as k

    st = _scratch(q.device)
    if st is None:
        return None
    scr, ctl = st[3], st[4]
    k16, v16, ks16, vs16 = cache
    m = q.shape[0]
    out = torch.empty((m, 16, 128), dtype=torch.bfloat16, device=q.device)
    u8 = lambda t: t.view(torch.uint8)  # noqa: E731
    _run_compiled(
        k.launch_draft_pa,
        ptr_arg(q), ptr_arg(u8(k16)), ptr_arg(u8(v16)), ptr_arg(ks16), ptr_arg(vs16), ptr_arg(bt16_rows),
        ptr_arg(ctx_rows), ptr_arg(out), ptr_arg(scr), ptr_arg(ctl), ptr_arg(scr, fx.Int32),
        int(m), int(bt16_rows.stride(0)), float(sm_scale), 20000000,
        current_stream_arg(),
        0,
        specialization_key=("dspark_draft_pa", q.device.index),
    )  # fmt: skip
    return out


def mlp_launch(x, x_scale, mlp, residual, next_norm, fp8_out: bool):
    """The MLP half alone (the post launch without o_proj and its all-reduce): x = the post-attention norm's MXFP8
    output, gate_up -> SwiGLU-OAI -> down -> all-reduce + residual + next_norm. Returns (out, out_scale | None,
    residual_out), or None (production)."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels.tensor_shim import (
        _run_compiled,
        current_stream_arg,
        ptr_arg,
    )

    from atom.model_ops.kernels import dspark_draft_post_gfx1250 as k

    if x_scale is None or x.dtype != torch.float8_e4m3fn or x.stride(1) != 1 or x.stride(0) % 16:
        return None
    st = _scratch(x.device)
    if st is None:
        return None
    scr, ctl = st[0], st[1]
    inbox, ctr, slot, rank, max_rows = _k5_info()
    m = x.shape[0]
    dev = x.device
    out = torch.empty(
        (m, HIDDEN), dtype=torch.float8_e4m3fn if fp8_out else torch.bfloat16, device=dev
    )
    out_scale = (
        torch.empty((m, HIDDEN // 32), dtype=torch.float8_e8m0fnu, device=dev) if fp8_out else None
    )
    residual = residual.contiguous()
    res_out = torch.empty_like(residual)
    w1, w2 = mlp.gate_up_proj, mlp.down_proj
    u8 = lambda t: t.view(torch.uint8)  # noqa: E731
    spec = (
        0,
        int(fp8_out),
        int(max_rows),
        0,
        float(mlp.swiglu_alpha),
        float(mlp.swiglu_beta),
        float(mlp.swiglu_limit),
        0,
        16,
    )
    _run_compiled(
        k.launch_draft_post,
        ptr_arg(x), ptr_arg(u8(x_scale)), ptr_arg(w1.weight), ptr_arg(u8(w1.weight_scale)),
        ptr_arg(w1.weight), ptr_arg(u8(w1.weight_scale)), ptr_arg(w2.weight), ptr_arg(u8(w2.weight_scale)),
        ptr_arg(residual), ptr_arg(next_norm.weight), ptr_arg(next_norm.weight),
        ptr_arg(out), ptr_arg(u8(out_scale) if fp8_out else residual), ptr_arg(res_out),
        flyc.from_c_void_p(fx.Int32, int(ctr)), ptr_arg(scr), ptr_arg(ctl), ptr_arg(scr, fx.Int32),
        int(inbox[0]), int(inbox[1]), int(inbox[2]), int(inbox[3]),
        int(m), int(x.stride(0)), int(slot), int(rank), int(k.MAXIT), float(next_norm.variance_epsilon),
        current_stream_arg(),
        *spec,
        specialization_key=("dspark_draft_mlp", dev.index) + spec,
    )  # fmt: skip
    return out, out_scale, res_out
