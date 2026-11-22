# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_FOLD_QKNORM_QKV_BF16: FOLD B (fold_qkv_qknorm) for the MTP drafter's bf16 qkv_proj.

The drafter's attention runs its qkv projection as aiter's bf16 gluon GEMM (`gemm_a16w16`, 16 x 32 tiles at decode
sizes), then `fused_qk_norm_mrope_cache` (Q/K Gemma RMSNorm, interleaved MRoPE, Q -> fp8, K -> NVFP4 and V -> fp8
paged-cache writes; the draft layer's cache is the same fp8_k4v8 A8K4V8 layout as the target's). Here the GEMM runs
with a CLUSTER tile of 16 x 256 (one head) over num_ctas CTAs along N, each CTA computing its 16 x (256 / num_ctas)
piece with the bf16 kernel's own K pipeline (BLOCK_K tiles of 16x16x32 WMMAs, so every output element accumulates in
the same order: bitwise the same y). After the C tile is TDM-stored, the cluster barrier makes the head visible to
all of its CTAs and fold B's epilogue (`fold_qkv_qknorm._fold_epilogue`, the native row kernel's arithmetic) runs on
it. Gate tiles do nothing more. No cross-cluster communication.

Also (env-gated, default off) the A8K4V4PT cache (ATOM_FOLD_QKNORM_QKV_BF16_FP4PT=1: fp4pt V, e2m1 + one fp32 scale per
token and KV head, fold B's `_fold_v_fp4pt`; the native pair's row kernel is fused_qk_norm_mrope_cache_k4v4pt) and
decode batches up to 48 rows (ATOM_FOLD_QKNORM_QKV_BF16_MAX_M, default 16: the drafter's pass 0 is 4 rows a sequence,
20-48 rows at bs 5-12) with fold B's 48-row tiling: one M tile of 32 x 256 (8 CTAs) / 64 x 256 (4 CTAs, 8 warps)."""

import functools
import os

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_mxfp8 import cga_bases
from atom.model_ops.fold_qkv_qknorm import (
    _epi_layouts,
    _fold_epilogue_rows,
    _fold_pre,
)


@gluon.jit
def _gemm_bf16_qkv_qknorm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_bn, stride_cm,
    q_weight_ptr, k_weight_ptr, positions_ptr, stride_positions, cos_ptr, sin_ptr, stride_cos,
    q_out_ptr, k_out_ptr, slot_mapping_ptr, num_slots, k_cache_ptr, v_cache_ptr, kv_inv_scale,
    BLOCK_SIZE_M: gl.constexpr, BLOCK_SIZE_N: gl.constexpr, BLOCK_SIZE_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr, num_ctas: gl.constexpr,
    wmma_layout: gl.constexpr, dot_a_layout: gl.constexpr, dot_b_layout: gl.constexpr,
    shared_A: gl.constexpr, shared_B: gl.constexpr, shared_C: gl.constexpr,
    EPS: gl.constexpr, Q_HEADS: gl.constexpr, K_HEADS: gl.constexpr,
    SECTION_H: gl.constexpr, SECTION_W: gl.constexpr, BLOCK_SIZE: gl.constexpr,
    FP8_MAX: gl.constexpr, SCALE_PRESHUFFLE: gl.constexpr,
    row_layout: gl.constexpr, rope_layout: gl.constexpr, kl: gl.constexpr, kd: gl.constexpr,
    ks: gl.constexpr, kl64: gl.constexpr, kd64: gl.constexpr, ks64: gl.constexpr,
    FOLD: gl.constexpr = True,
    V_FP4PT: gl.constexpr = False,
    KLDS: gl.constexpr = False,
    KWORD: gl.constexpr = False,
):
    gl.static_assert(BLOCK_SIZE_N == 256)
    pid = gl.program_id(axis=0)
    tiles_n = gl.cdiv(N, BLOCK_SIZE_N)
    tile_m = pid // tiles_n
    tile_n = pid - tile_m * tiles_n
    GATE_TILES: gl.constexpr = Q_HEADS

    w_e, wlo_e, whi_e, cos_e, sin_e, slot_e = _fold_pre(
        M, tile_m, tile_n - GATE_TILES, q_weight_ptr, k_weight_ptr, positions_ptr, stride_positions, cos_ptr,
        sin_ptr, stride_cos, slot_mapping_ptr, num_slots, BLOCK_SIZE_M, Q_HEADS, SECTION_H, SECTION_W,
        row_layout, rope_layout, kd, ks, kd64, ks64)

    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr + tile_m * BLOCK_SIZE_M * stride_am,
        shape=(M - tile_m * BLOCK_SIZE_M, K), strides=(stride_am, 1),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), layout=shared_A)
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_ptr + tile_n * BLOCK_SIZE_N * stride_bn,
        shape=(N - tile_n * BLOCK_SIZE_N, K), strides=(stride_bn, 1),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K), layout=shared_B)
    smem_A = gl.allocate_shared_memory(a_ptr.type.element_ty, [NUM_BUFFERS, BLOCK_SIZE_M, BLOCK_SIZE_K],
                                       layout=shared_A)
    smem_B = gl.allocate_shared_memory(b_ptr.type.element_ty, [NUM_BUFFERS, BLOCK_SIZE_N, BLOCK_SIZE_K],
                                       layout=shared_B)
    k_tiles = K // BLOCK_SIZE_K
    load_idx = 0
    compute_idx = 0
    acc = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=wmma_layout)
    for _ in gl.static_range(NUM_BUFFERS):
        slot = load_idx % NUM_BUFFERS
        gl.amd.gfx1250.tdm.async_load(a_desc, [0, load_idx * BLOCK_SIZE_K], smem_A.index(slot))
        gl.amd.gfx1250.tdm.async_load(b_desc, [0, load_idx * BLOCK_SIZE_K], smem_B.index(slot))
        load_idx += 1

    gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
    cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_A.index(0), dot_a_layout)
    cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_B.index(0).permute([1, 0]), dot_b_layout)
    if num_ctas > 1:
        gl.amd.gfx1250.cluster.arrive()
    for _ in range(k_tiles - NUM_BUFFERS):
        acc = gl.amd.gfx1250.wmma(cur_a, cur_b, acc)
        if num_ctas > 1:
            gl.amd.gfx1250.cluster.wait()
        slot = load_idx % NUM_BUFFERS
        gl.amd.gfx1250.tdm.async_load(a_desc, [0, load_idx * BLOCK_SIZE_K], smem_A.index(slot))
        gl.amd.gfx1250.tdm.async_load(b_desc, [0, load_idx * BLOCK_SIZE_K], smem_B.index(slot))
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
        load_idx += 1
        nxt = (compute_idx + 1) % NUM_BUFFERS
        cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_A.index(nxt), dot_a_layout)
        cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_B.index(nxt).permute([1, 0]), dot_b_layout)
        compute_idx += 1
        if num_ctas > 1:
            gl.amd.gfx1250.cluster.arrive()
    if num_ctas > 1 and NUM_BUFFERS > 1:
        gl.amd.gfx1250.cluster.wait()
    for i in gl.static_range(NUM_BUFFERS - 1):
        acc = gl.amd.gfx1250.wmma(cur_a, cur_b, acc)
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)
        nxt = (compute_idx + 1) % NUM_BUFFERS
        cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_A.index(nxt), dot_a_layout)
        cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_B.index(nxt).permute([1, 0]), dot_b_layout)
        compute_idx += 1
    if num_ctas > 1 and NUM_BUFFERS > 1:
        gl.amd.gfx1250.cluster.arrive()
    acc = gl.amd.gfx1250.wmma(cur_a, cur_b, acc)
    if num_ctas > 1:
        gl.amd.gfx1250.cluster.wait()

    c_buffer = gl.allocate_shared_memory(c_ptr.type.element_ty, shape=[BLOCK_SIZE_M, BLOCK_SIZE_N], layout=shared_C)
    c_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=c_ptr, shape=(M, N), strides=(stride_cm, 1),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), layout=shared_C)
    c_buffer.store(acc.to(c_ptr.type.element_ty))
    gl.amd.gfx1250.tdm.async_store(c_desc, [tile_m * BLOCK_SIZE_M, tile_n * BLOCK_SIZE_N], c_buffer)
    gl.amd.gfx1250.tdm.async_wait(0)

    if FOLD:
        if tile_n >= GATE_TILES:
            if num_ctas > 1:
                gl.amd.gfx1250.cluster.arrive()
            # fold B's epilogue on the tile's rows (several rows a wave from 33 rows: register repeats); V_FP4PT:
            # the A8K4V4PT V page (e2m1 + one fp32 scale per token and KV head) instead of the fp8 one
            _fold_epilogue_rows(c_ptr, stride_cm, M, tile_m * BLOCK_SIZE_M, tile_n - GATE_TILES,
                                w_e, wlo_e, whi_e, cos_e, sin_e, slot_e,
                                q_weight_ptr, k_weight_ptr, positions_ptr, stride_positions, cos_ptr, sin_ptr,
                                stride_cos, q_out_ptr, k_out_ptr, slot_mapping_ptr, num_slots, k_cache_ptr,
                                v_cache_ptr, kv_inv_scale, BLOCK_SIZE_M, num_ctas, EPS, Q_HEADS, K_HEADS,
                                SECTION_H, SECTION_W, BLOCK_SIZE, FP8_MAX, SCALE_PRESHUFFLE,
                                row_layout, rope_layout, kl, kd, ks, kl64, kd64, ks64, num_ctas > 1, V_FP4PT,
                                KLDS, KWORD)


# Cluster tile 16 x 256 (one head); the per-CTA tile is 16 x (256 / num_ctas), BLOCK_K and the WMMA shape those of
# aiter's tuned bf16 decode configs (16 x 32 / 16 x 64, BLOCK_K 256, 16x16x32 WMMA): the same K order per element.
# M 17-48 (the drafter's pass 0 at bs 5-12: 4 rows a sequence; graph buckets 5-8, 12 -> 20-32, 48 rows) as fold B's
# 48-row extension: one M tile (the weights read once; two 32-row tiles: +3 us), 32 x 256 over 8 CTAs (32 x 32 each,
# 4 warps: one row a wave in the epilogue) up to 32 rows, 64 x 256 over 8 CTAs (64 x 32 each, 4 warps: 2 rows a wave
# held in registers, K's RoPE groups after one LDS conversion) up to 48, with native's BLOCK_K 128 / 6 buffers there
# (task2 sweeps, gfx1250, 10 layers a graph, fp4pt: 64 x 256 over 4 CTAs 8 warps (fold B's) +1.5 us of GEMM,
# BLOCK_K 256 / 3-4 buffers, 7-8 buffers (LDS: 1 CTA a WGP), 8 warps, KLDS off: all slower). KWORD: dword K stores.
FOLD_CONFIGS = {
# (measured, gfx1250, 10 layers a graph: 4 buffers best at M 1-16; 3 / 5 / 8 buffers, 4 / 16 CTAs and 4 warps slower)
    16: dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=256, BLOCK_SIZE_K=256, num_warps=2, NUM_BUFFERS=4, num_ctas=8),
    32: dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=256, BLOCK_SIZE_K=256, num_warps=4, NUM_BUFFERS=4, num_ctas=8, KWORD=True),
    48: dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=256, BLOCK_SIZE_K=128, num_warps=4, NUM_BUFFERS=6, num_ctas=8, KLDS=True,
             KWORD=True),
}
if os.environ.get("ATOM_FOLD_QKNORM_QKV_BF16_CFG"):
    # tuning: {"<bucket>": {config overrides}}
    for _b, _c in __import__("json").loads(os.environ["ATOM_FOLD_QKNORM_QKV_BF16_CFG"]).items():
        FOLD_CONFIGS.setdefault(int(_b), dict(FOLD_CONFIGS[48])).update(_c)
KERNEL_MAX_M = 48
# ATOM_FOLD_QKNORM_QKV_BF16_MAX_M: the largest decode batch (rows) folded (default 16, up to KERNEL_MAX_M = 48)
FOLD_MAX_M = max(1, min(KERNEL_MAX_M, int(os.environ.get("ATOM_FOLD_QKNORM_QKV_BF16_MAX_M", "16"))))
# ATOM_FOLD_QKNORM_QKV_BF16_FP4PT=1: also fold for an A8K4V4PT cache (--kv_cache_dtype fp8_k4v4pt: fp4pt V)
FOLD_FP4PT = os.environ.get("ATOM_FOLD_QKNORM_QKV_BF16_FP4PT", "0") == "1"


def fold_config(M):
    for b in sorted(FOLD_CONFIGS):
        if M <= b:
            return dict(FOLD_CONFIGS[b])
    return None


def _wmma_warp_bases(num_warps):
    # aiter gemm_a16w16.create_wmma_layouts
    bases = []
    for i in range(num_warps.bit_length() - 1):
        bases.append([0, 1] if i == 0 else [1 << (i - 1), 0])
    return bases


@functools.cache
def _layouts(num_warps, BLOCK_M, BLOCK_N, BLOCK_K, num_ctas):
    cga = cga_bases(1, num_ctas)
    wmma = gl.amd.AMDWMMALayout(version=3, transposed=True, warp_bases=_wmma_warp_bases(num_warps), reg_bases=[],
                                instr_shape=[16, 16, 32], cga_layout=cga)
    dot_a = gl.DotOperandLayout(operand_index=0, parent=wmma, k_width=8)
    dot_b = gl.DotOperandLayout(operand_index=1, parent=wmma, k_width=8)
    cga_a = dot_a.cga_layout
    cga_b = [[n, k] for k, n in dot_b.cga_layout]
    pad_k = min(BLOCK_K, 256 * 32 // 16)
    shared_A = gl.PaddedSharedLayout.with_identity_for([[pad_k, 8]], [BLOCK_M, BLOCK_K], [1, 0], cga_a)
    shared_B = gl.PaddedSharedLayout.with_identity_for([[pad_k, 8]], [BLOCK_N, BLOCK_K], [1, 0], cga_b)
    shared_C = gl.PaddedSharedLayout.with_identity_for([[BLOCK_N // num_ctas, 8]], [BLOCK_M, BLOCK_N], [1, 0], cga)
    d = dict(wmma_layout=wmma, dot_a_layout=dot_a, dot_b_layout=dot_b, shared_A=shared_A, shared_B=shared_B,
             shared_C=shared_C)
    d.update(_epi_layouts(num_warps, num_ctas))
    return d


def fused_gemm_bf16_qkv_qknorm_mrope_cache(
    x, w, q_weight, k_weight, eps, positions, cos, sin, mrope_section,
    k_cache, v_cache, slot_mapping, kv_scale, num_q_heads, num_kv_heads, y=None, config=None, fold=True,
):
    """y = x @ w^T (bf16 [M, N], N = (2 Hq + 2 Hk) * 256: gate | q | k | v), bitwise aiter's gemm_a16w16 at decode
    sizes, plus what fused_qk_norm_mrope_cache(q, k, v, ..., kv_scale) does for an A8K4V8 cache (v_cache fp8 5-D) or
    fused_qk_norm_mrope_cache_k4v4pt for an A8K4V4PT one (v_cache [blocks, Hk, 128, 132] fp4pt, 4-D): returns
    (y, q_out fp8 [M, Hq*256], k_out bf16 [M, Hk*256]) and writes K (NVFP4) / V (fp8 | fp4pt) into the cache."""
    M, K = x.shape
    N = w.shape[0]
    assert M <= KERNEL_MAX_M and N == (2 * num_q_heads + 2 * num_kv_heads) * 256
    assert x.stride(1) == 1 and w.stride(1) == 1
    cfg = fold_config(M)
    if config:
        cfg.update(config)
    assert K % cfg["BLOCK_SIZE_K"] == 0
    if y is None:
        y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    v_fp4pt = v_cache.dim() == 4
    if v_fp4pt:
        from atom.model_ops.fold_qkv_qknorm import _fp8_e4m3

        q_dtype = _fp8_e4m3()
        vc = v_cache.view(torch.uint8)
        assert vc.shape == (k_cache.shape[0], num_kv_heads, 128, 256 // 2 + 4) and vc.is_contiguous()
    else:
        q_dtype = v_cache.dtype
        vc = v_cache
    q_out = torch.empty((M, num_q_heads * 256), dtype=q_dtype, device=x.device)
    k_out = torch.empty((M, num_kv_heads * 256), dtype=torch.bfloat16, device=x.device)
    cfg["NUM_BUFFERS"] = min(cfg["NUM_BUFFERS"], K // cfg["BLOCK_SIZE_K"])
    layouts = _layouts(cfg["num_warps"], cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"],
                       cfg["num_ctas"])
    if slot_mapping.shape[0] == 0:
        slot_mapping = torch.full((1,), -1, dtype=slot_mapping.dtype, device=slot_mapping.device)
    page_block = k_cache.shape[2]
    half = cos.shape[1]
    assert half == 32
    grid = (triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * (N // 256),)
    _gemm_bf16_qkv_qknorm_kernel[grid](
        x, w, y, M, N, K, x.stride(0), w.stride(0), y.stride(0),
        q_weight, k_weight, positions, positions.stride(0), cos, sin, cos.stride(0),
        q_out, k_out, slot_mapping, slot_mapping.shape[0], k_cache, vc, 1.0 / kv_scale,
        EPS=eps, Q_HEADS=num_q_heads, K_HEADS=num_kv_heads, SECTION_H=mrope_section[1], SECTION_W=mrope_section[2],
        BLOCK_SIZE=page_block, FP8_MAX=torch.finfo(q_dtype).max, SCALE_PRESHUFFLE=min(128, page_block),
        FOLD=fold, V_FP4PT=v_fp4pt, **cfg, **layouts,
    )
    return y, q_out, k_out


# ------------------------------------------------------------------------------------------------------- ATOM glue
ENABLED = os.environ.get("ATOM_FOLD_QKNORM_QKV_BF16", "0") == "1"


def try_fold_qkv_bf16(attn, positions, hidden_states):
    """The bf16 qkv_proj (unquantized: the MTP drafter's) with the Q/K norm, MRoPE and the A8K4V8 (or, with
    ATOM_FOLD_QKNORM_QKV_BF16_FP4PT, A8K4V4PT) cache write in its epilogue, for a decode batch of at most FOLD_MAX_M
    (ATOM_FOLD_QKNORM_QKV_BF16_MAX_M) rows; leaves the fp8 Q / bf16 K for the attention's rope_cache
    (impl._fold_pending, as fold_qkv_qknorm.try_fold_qkv). None where it does not apply."""
    if not ENABLED or torch.compiler.is_compiling():
        return None
    from atom.utils.forward_context import get_forward_context

    layer = attn.qkv_proj
    impl = getattr(attn.attn, "impl", None)
    w = getattr(layer, "weight", None)
    if (
        w is None
        or w.dtype != torch.bfloat16
        or w.dim() != 2
        or w.stride(1) != 1
        or getattr(layer, "bias", None) is not None
        or getattr(w, "is_shuffled", False)
        or getattr(layer, "mx1x32", False)
        or getattr(layer, "tp_size", 1) != 1
        or impl is None
        or not getattr(impl, "k4v8", False)
        or (getattr(impl, "k4v4pt", False) and not FOLD_FP4PT)
        or not attn.fuse_qk_norm_rope_cache
        or not attn.attn_output_gate
        or attn.rotary_emb is None
        or getattr(attn.rotary_emb, "rotary_dim", 0) != 64
        or attn.head_dim != 256
    ):
        return None
    fc = get_forward_context()
    ctx = fc.context
    if ctx is None or ctx.is_prefill or ctx.is_dummy_run or fc.attn_metadata is None:
        return None
    n, k = w.shape
    x2 = hidden_states.reshape(-1, k)
    m = x2.shape[0]
    if m == 0 or m > FOLD_MAX_M or x2.dtype != torch.bfloat16 or x2.stride(1) != 1:
        return None
    if positions is None or positions.dim() != 2 or positions.shape != (3, m) or positions.stride(1) != 1:
        return None
    kvd = fc.kv_cache_data[f"layer_{impl.layer_num}"]
    k_cache, v_cache = kvd.k_cache, kvd.v_cache
    if not (k_cache.dtype == torch.uint8 and k_cache.dim() == 4 and k_cache.shape[2] == 128):
        return None
    if getattr(impl, "k4v4pt", False):
        if not (v_cache.dim() == 4 and v_cache.dtype == torch.float4_e2m1fn_x2 and v_cache.is_contiguous()
                and v_cache.shape[2] == 128 and v_cache.view(torch.uint8).shape[3] == 256 // 2 + 4):
            return None
    elif v_cache.dim() != 5:
        return None
    rot = attn.rotary_emb
    half = rot.rotary_dim // 2
    y, q_out, k_out = fused_gemm_bf16_qkv_qknorm_mrope_cache(
        x2, w, attn.q_norm.weight, attn.k_norm.weight, attn.q_norm.variance_epsilon, positions,
        rot.cos_cache.view(-1, half), rot.sin_cache.view(-1, half), rot.mrope_section, k_cache, v_cache,
        fc.attn_metadata.slot_mapping, impl.kv_scale_float, attn.num_heads, attn.num_kv_heads,
    )
    impl._fold_pending = (q_out, k_out)
    return y.view(*hidden_states.shape[:-1], n)
