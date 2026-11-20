# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_FOLD_ATTN_GATE (default off): the attention output gate and o_proj's 1x32 MXFP8 quant in
the epilogue of the gfx1250 split-KV reduce (reduce_segments_split_oneshot_gluon), instead of the
separate `_gated_mx32_kernel` launch (sigmoid_gate.output_gate_mul_mx32) after the attention.

Bitwise the unfused pair: the reduce's own gate epilogue (sigmoid rounded to the gate dtype, the
product to bf16, as `_gated_mx32_kernel`), then `mxfp8_1x32.mx32_quant_groups` of those bf16 values
(the same amax floor / cap, RoundUp e8m0 byte, clamp, RNE e4m3). A reduce program owns one
(token, head) over all D = 256 head-dim elements = 8 whole 32-groups of the o_proj K axis, so the
quant needs no cross-program data.

Scope: decode / verify (any row count: row-major x_scale under 32 rows, else n32k4 with the pad
rows written by the last row's programs), no row selection, the one-shot split reduce. Anywhere else the attention gates its bf16 output as before
(or forward_impl gates it) and `finish` runs the standalone quant: same bytes, one more launch.

Mechanism: Qwen3NextAttention.forward opens `request(gate, m, k)` and hands the attention the gate
(the ATOM_UA_FUSED_OUTPUT_GATE plumbing, forward_impl -> unified_attention(output_gate=...)).
aiter's `_reduce_segments_split_gfx1250` is wrapped (once, at the first request) so that the
reduce of a request's own gate tensor launches the kernel below, writing (fp8, e8m0) into the
request's buffers instead of the bf16 output.
"""

from __future__ import annotations

import os
import threading

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_FOLD_ENV = None


def fold_enabled() -> bool:
    global _FOLD_ENV
    if _FOLD_ENV is None:
        _FOLD_ENV = os.environ.get("ATOM_FOLD_ATTN_GATE", "0") == "1"
    return _FOLD_ENV


@gluon.jit
def reduce_split_oneshot_gate_mx32_gluon(
    q_out_ptr,  # [num_rows, H * D] fp8 e4m3 (o_proj's x)
    s_out_ptr,  # flat uint8 e8m0 x_scale of o_proj (mxfp8_1x32 layout: row-major M < 32, else n32k4)
    segm_output_ptr,  # [num_tokens, H, S, D] fp32
    segm_max_ptr,  # [num_tokens, H, S] fp32 (log2 domain)
    segm_expsum_ptr,  # [num_tokens, H, S] fp32
    seq_lens_ptr,  # [num_seqs]
    num_seqs,
    segm_tiles_ptr,  # [1] int32 tiles a KV segment covers (token split), or None
    gate_ptr,  # [num_tokens, H * D] bf16 output gate
    gate_stride_0,
    num_rows,
    H: gl.constexpr,
    S: gl.constexpr,
    D: gl.constexpr,
    TILE_SIZE: gl.constexpr,
    SEQLEN_Q: gl.constexpr,
    SB: gl.constexpr,
    CHUNK: gl.constexpr,
    SPEC: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    N32K4: gl.constexpr,  # num_rows >= 32: the n32k4 scale layout, pad rows to 32 hold 0x7F
):
    """aiter reduce_segments_split_oneshot_gluon (gate epilogue, no rows / out scale / fp8 out),
    line for line up to the gated bf16 result, then the 1x32 MXFP8 quant of it."""
    token = gl.program_id(0)
    head = gl.program_id(1)

    VEC: gl.constexpr = D // (32 * NUM_WARPS)
    BLK: gl.constexpr = gl.BlockedLayout([SB, VEC], [1, 32], [1, NUM_WARPS], [1, 0])
    L_S: gl.constexpr = gl.SliceLayout(1, BLK)
    L_D: gl.constexpr = gl.SliceLayout(0, BLK)
    STAT: gl.constexpr = gl.BlockedLayout([max(1, S // 32)], [32], [NUM_WARPS], [0])
    SMEM2: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    SMEM1: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    NG: gl.constexpr = CHUNK // SB
    HALF: gl.constexpr = CHUNK // 2
    NSPEC: gl.constexpr = SPEC // HALF
    # the [D] row as [D // 32 groups, 32]: VEC consecutive elements a lane, 32 // VEC lanes a
    # group, VEC groups a warp (the same element -> lane / register map as L_D)
    G: gl.constexpr = D // 32
    L1: gl.constexpr = gl.BlockedLayout([VEC], [32], [NUM_WARPS], [0])
    L2: gl.constexpr = gl.BlockedLayout([1, VEC], [VEC, 32 // VEC], [NUM_WARPS, 1], [1, 0])

    seq_idx = gl.minimum(token // SEQLEN_Q, num_seqs - 1)
    seq_len = gl.load(seq_lens_ptr + seq_idx)
    if segm_tiles_ptr is not None:
        tiles_per_segment = gl.load(segm_tiles_ptr)
    else:
        tiles_per_segment = (seq_len + S * TILE_SIZE - 1) // (S * TILE_SIZE)

    offs_d = gl.arange(0, D, layout=L_D)
    offs_s = gl.arange(0, SB, layout=L_S)
    offs_all = gl.arange(0, S, layout=STAT)
    stat_off = token * (H * S) + head * S
    o_base = segm_output_ptr + token.to(gl.int64) * (H * S * D) + head * (S * D)
    m_base = segm_max_ptr + stat_off
    e_base = segm_expsum_ptr + stat_off

    gate = gl.load(gate_ptr + token.to(gl.int64) * gate_stride_0 + head * D + offs_d)
    o_smem = gl.allocate_shared_memory(
        segm_output_ptr.dtype.element_ty, [2, HALF, D], SMEM2
    )
    m_smem = gl.allocate_shared_memory(segm_max_ptr.dtype.element_ty, [2, HALF], SMEM1)
    if NSPEC > 0:
        o_spec = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            o_base, [S, D], [D, 1], [HALF, D], SMEM2
        )
        m_spec = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            m_base, [S], [1], [HALF], SMEM1
        )
        for p in gl.static_range(NSPEC):
            gl.amd.gfx1250.tdm.async_load(o_spec, [p * HALF, 0], o_smem.index(p))
            gl.amd.gfx1250.tdm.async_load(m_spec, [p * HALF], m_smem.index(p))
    segm_max = gl.amd.gfx1250.buffer_load(m_base, offs_all)
    segm_expsum = gl.amd.gfx1250.buffer_load(e_base, offs_all)
    gl.amd.hint.sched_barrier()

    span = gl.maximum(tiles_per_segment, 1) * TILE_SIZE
    num_segments = gl.minimum((seq_len + span - 1) // span, S)
    num_steps = (num_segments + SB - 1) // SB
    o_live = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        o_base, [num_segments, D], [D, 1], [HALF, D], SMEM2
    )
    m_live = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        m_base, [num_segments], [1], [HALF], SMEM1
    )
    if num_segments > SPEC:
        for p in gl.static_range(NSPEC, 2):
            gl.amd.gfx1250.tdm.async_load(o_live, [p * HALF, 0], o_smem.index(p))
            gl.amd.gfx1250.tdm.async_load(m_live, [p * HALF], m_smem.index(p))

    gate = (1 / (1 + gl.exp(-gate.to(gl.float32)))).to(gate.dtype).to(gl.float32)
    gate = gl.inline_asm_elementwise(
        "v_mov_b32 $0, $1", "=v,v", [gate], dtype=gl.float32, is_pure=False, pack=1
    )

    live = offs_all < num_segments
    segm_max = gl.where(live, segm_max, float("-inf"))
    segm_expsum = gl.where(live, segm_expsum, 0.0)
    overall_max = gl.max(segm_max, axis=0)
    overall_expsum = gl.sum(segm_expsum * gl.exp2(segm_max - overall_max), axis=0)

    o_steps = o_smem.reinterpret(shape=[NG, SB, D], layout=SMEM2)
    m_steps = m_smem.reinterpret(shape=[NG, SB], layout=SMEM1)
    acc = gl.zeros([D], gl.float32, layout=L_D)
    for c in range((num_segments + CHUNK - 1) // CHUNK):
        if c > 0:
            if NUM_WARPS > 1:
                gl.barrier()
            for p in gl.static_range(2):
                gl.amd.gfx1250.tdm.async_load(
                    o_live, [c * CHUNK + p * HALF, 0], o_smem.index(p)
                )
                gl.amd.gfx1250.tdm.async_load(
                    m_live, [c * CHUNK + p * HALF], m_smem.index(p)
                )
        gl.amd.gfx1250.tdm.async_wait(0)
        if NUM_WARPS > 1:
            gl.barrier()
        for g in range(gl.minimum(NG, num_steps - c * NG)):
            segs = (c * NG + g) * SB + offs_s
            live_s = segs < num_segments
            o = gl.where(live_s[:, None], o_steps.index(g).load(BLK), 0.0)
            m = gl.where(live_s, m_steps.index(g).load(L_S), float("-inf"))
            w = gl.exp2(m - overall_max)
            acc += gl.sum(o * w[:, None], axis=0)
    gl.amd.gfx1250.tdm.async_wait(0)

    acc = gl.where(overall_expsum == 0.0, gl.full_like(acc, 0.0), acc / overall_expsum)
    result = acc.to(gl.bfloat16)
    result = (result.to(gl.float32) * gate).to(gl.bfloat16)

    # ---- the 1x32 MXFP8 quant of the gated bf16 row (mxfp8_1x32.mx32_quant_groups)
    y = gl.convert_layout(result.to(gl.float32), L1)
    y = gl.convert_layout(gl.reshape(y, [G, 32]), L2)
    amax = gl.max(gl.abs(y), axis=1)
    amax = gl.minimum(gl.maximum(amax, 1e-10), 448.0 * 6.646139978924579e35)  # 2^119
    u = (amax * (1.0 / 448.0)).to(gl.uint32, bitcast=True)
    e = ((u >> 23) & 0xFF).to(gl.int32)
    e = e + (((u & 0x7FFFFF) != 0) & (e < 0xFF)).to(gl.int32)
    inv = ((254 - gl.minimum(e, 253)).to(gl.uint32) << 23).to(gl.float32, bitcast=True)
    v = gl.minimum(gl.maximum(y * inv[:, None], -448.0), 448.0)
    qv = v.to(gl.float8e4nv)

    KG: gl.constexpr = H * G
    offs_g = gl.arange(0, G, layout=gl.SliceLayout(1, L2))
    offs_c = gl.arange(0, 32, layout=gl.SliceLayout(0, L2))
    row = token.to(gl.int64)
    gl.store(
        q_out_ptr + row * (H * D) + head * D + offs_g[:, None] * 32 + offs_c[None, :], qv
    )
    gi = head * G + offs_g
    if N32K4:
        # (m // 32) * K + (g // 4) * 128 + (m % 32) * 4 + g % 4, K = KG * 32 (mxfp8_1x32)
        s_col = (gi // 4) * 128 + gi % 4
        gl.store(s_out_ptr + (row // 32) * (KG * 32) + (row % 32) * 4 + s_col, e.to(gl.uint8))
        if token == num_rows - 1:
            # the last row's programs fill the pad rows of their head's groups with e8m0 1.0
            ones = gl.full([G], 0x7F, gl.uint8, layout=gl.SliceLayout(1, L2))
            for r in range(num_rows, (num_rows + 31) // 32 * 32):
                r64 = r.to(gl.int64)
                gl.store(s_out_ptr + (r64 // 32) * (KG * 32) + (r64 % 32) * 4 + s_col, ones)
    else:
        gl.store(s_out_ptr + row * KG + gi, e.to(gl.uint8))


# ------------------------------------------------------------------------------------------------
# request plumbing
# ------------------------------------------------------------------------------------------------
class _Request:
    __slots__ = ("gate", "q", "s", "applied")

    def __init__(self, gate, q, s):
        self.gate, self.q, self.s, self.applied = gate, q, s, False


_TLS = threading.local()
_HOOKED = False
_LAST_APPLIED = None  # whether the last finished request was taken by the reduce (diagnostics)


def _current():
    return getattr(_TLS, "req", None)


def _install_hook():
    """Wrap `_reduce_segments_split_gfx1250` in the globals of the unified_attention that ATOM's
    attention calls. That is the function under aiter's fast-launch wrapper (whose replays never
    reach a request: a launch with the request's buffers is not replayable, and under CUDA graph
    capture the wrapper always runs the function); its module object can differ from the one
    `import aiter.ops.triton.attention.unified_attention` returns, so take it from attention_mha."""
    global _HOOKED
    if _HOOKED:
        return
    import atom.model_ops.attention_mha as _am

    seen = set()
    for fn in (_am.unified_attention,):
        G = getattr(fn, "__wrapped__", fn).__globals__
        if id(G) not in seen and "_reduce_segments_split_gfx1250" in G:
            seen.add(id(G))
            _hook_globals(G)
    _HOOKED = True


def _hook_globals(G):
    import types

    UA = types.SimpleNamespace(**{k: G[k] for k in (
        "_reduce_segments_split_oneshot_kernel_gfx1250", "_GFX12_SPLIT_REDUCE_CONFIG",
        "_GFX12_SPLIT_REDUCE_ONESHOT_CONFIG")})
    # aiter's (CHUNK, SPEC, warps) for a launch shape (AITER_GFX12_SPLIT_REDUCE_ONESHOT_AUTO),
    # where aiter has it; else its fixed _GFX12_SPLIT_REDUCE_ONESHOT_CONFIG
    oneshot_config = G.get("gfx12_split_reduce_oneshot_config")
    orig = G["_reduce_segments_split_gfx1250"]

    def _reduce_segments_split_gfx1250(
        params,
        segm_output,
        segm_max,
        segm_expsum,
        NUM_SEGMENTS,
        TILE_SIZE,
        segm_tiles=None,
        output_gate=None,
        output_rows=None,
        oneshot=False,
        output_mx32=None,
    ):
        if output_mx32 is not None:
            # ATOM_UA_REDUCE_GATE_MX32's request (aiter's own output_mx32 reduce): pass it through
            return orig(
                params,
                segm_output,
                segm_max,
                segm_expsum,
                NUM_SEGMENTS,
                TILE_SIZE,
                segm_tiles=segm_tiles,
                output_gate=output_gate,
                output_rows=output_rows,
                oneshot=oneshot,
                output_mx32=output_mx32,
            )
        req = _current()
        if (
            req is not None
            and output_gate is not None
            and output_gate.data_ptr() == req.gate.data_ptr()
            and output_gate.stride() == req.gate.stride()
            and output_rows is None
            and oneshot
            and UA._reduce_segments_split_oneshot_kernel_gfx1250 is not None
            and params.output_scale is None
            and params.out.dtype == torch.bfloat16
            and output_gate.dtype == torch.bfloat16
            and params.head_size_v % 32 == 0
            and req.q.shape == (params.num_tokens, params.num_query_heads * params.head_size_v)
        ):
            sb, _num_buf, _nw = UA._GFX12_SPLIT_REDUCE_CONFIG
            sb = min(sb, NUM_SEGMENTS)
            if oneshot_config is not None:
                chunk, spec, warps = oneshot_config(
                    params.num_tokens,
                    params.num_query_heads,
                    NUM_SEGMENTS,
                    params.num_seqs,
                    params.num_kv_heads,
                    params.head_size_v,
                )
            else:
                chunk, spec, warps = UA._GFX12_SPLIT_REDUCE_ONESHOT_CONFIG
            chunk = max(sb, min(chunk, NUM_SEGMENTS) // sb * sb)
            D = params.head_size_v
            if D % (32 * warps) == 0 and (D // (32 * warps)) * warps == D // 32:
                if params.num_tokens:
                    reduce_split_oneshot_gate_mx32_gluon[
                        (params.num_tokens, params.num_query_heads)
                    ](
                        req.q,
                        req.s,
                        segm_output,
                        segm_max,
                        segm_expsum,
                        params.seqused_k,
                        params.num_seqs,
                        segm_tiles,
                        output_gate,
                        output_gate.stride(0),
                        params.num_tokens,
                        H=params.num_query_heads,
                        S=NUM_SEGMENTS,
                        D=D,
                        TILE_SIZE=TILE_SIZE,
                        SEQLEN_Q=1 if params.all_decode else int(params.max_seqlen_q),
                        SB=sb,
                        CHUNK=chunk,
                        SPEC=min(spec, chunk) // sb * sb,
                        NUM_WARPS=warps,
                        N32K4=params.num_tokens >= 32,
                        num_warps=warps,
                    )
                req.applied = True
                return
        return orig(
            params,
            segm_output,
            segm_max,
            segm_expsum,
            NUM_SEGMENTS,
            TILE_SIZE,
            segm_tiles=segm_tiles,
            output_gate=output_gate,
            output_rows=output_rows,
            oneshot=oneshot,
            output_mx32=output_mx32,
        )

    G["_reduce_segments_split_gfx1250"] = _reduce_segments_split_gfx1250


def can_fold(attn_module, gate, select_rows) -> bool:
    """Whether this forward of Qwen3NextAttention takes the fold (else the unchanged path)."""
    return (
        fold_enabled()
        and attn_module.mx32_output_gate
        and not attn_module.skip_out_proj
        and not attn_module.gate_in_attention
        and select_rows is None
        and gate is not None
        and gate.dim() == 2
        and gate.dtype == torch.bfloat16
        and gate.stride(1) == 1
        and gate.shape[0] > 0
        and gate.shape[1] % 128 == 0
        and not torch.compiler.is_compiling()
    )


def open_request(gate):
    """Open a fold request for this attention's gate, with o_proj's (fp8, flat e8m0) input
    buffers; `finish` closes it."""
    from atom.model_ops.mxfp8_1x32 import alloc

    _install_hook()
    q, s = alloc(gate.shape[0], gate.shape[1], gate.device)
    req = _Request(gate, q, s)
    _TLS.req = req
    return req


def finish(req, attn_output):
    """Close the request; o_proj's (x_fp8, x_scale): the reduce's when it took the request, else
    the standalone quant of the attention's (already gated) bf16 output."""
    global _LAST_APPLIED
    _TLS.req = None
    _LAST_APPLIED = req.applied
    if req.applied:
        return req.q, req.s
    from atom.model_ops.mxfp8_1x32 import note, quant_into

    note("FOLD_ATTN_GATE fallback: gated attention output, standalone quant")
    quant_into(attn_output, req.q, req.s)
    return req.q, req.s
