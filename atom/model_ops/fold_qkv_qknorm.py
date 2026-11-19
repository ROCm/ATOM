# FOLD B: the attention qkv MXFP8 GEMM (gfx1250 Gluon, preshuffled 1x32) with the Q/K Gemma RMSNorm, the
# interleaved MRoPE, Q -> fp8, K -> NVFP4 and V -> fp8 (A8K4V8) or fp4pt (A8K4V4PT: e2m1 + one fp32 scale per token
# and KV head) paged-cache writes (block 128) in its epilogue: the work of fused_qk_norm_mrope_cache's row kernel
# (K_NVFP4 mode) / fused_qk_norm_mrope_cache_k4v4pt's (V_FP4PT), without its launch.
#
# Columns of the GEMM output are gate | q | k | v (Hq*256 | Hq*256 | Hk*256 | Hk*256). The kernel runs the
# preshuffle GEMM with a CLUSTER tile of BLOCK_M x 256 (BLOCK_N = one head), split over num_ctas CTAs along N
# (each CTA keeps the per-CTA tile of the tuned config). After the C tile is TDM-stored, the cluster barrier makes
# the whole head visible to all of its CTAs, and each CTA re-reads the head (bf16, from L2) for its share of the
# rows and runs the native row kernel's arithmetic on them (same arithmetic per element, so the same bits).
# Gate tiles do nothing more. No cross-cluster communication. Decode batches up to 48 rows (ext48b): x scales
# n32k4-shuffled from 32 rows, an epilogue holding several rows a wave (register repeats) or a runtime row loop.
import functools

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_mxfp8 import (
    _issue_tdm_stage,
    _load_a_scales,
    _load_b_scales,
    cluster_shape,
    depreshuffle_b_raw_to_kn,
    get_gemm_afp8wfp8_preshuffle_layouts,
)

HEAD_DIM = gl.constexpr(256)
ROPE_HALF = gl.constexpr(32)
NVFP4_GROUP = gl.constexpr(16)
# K-path parts (debug knob): 1 = also write the bf16 normalized/RoPE'd K rows (k_out; unified-attention decode reads only
# the cache, so off), 2 = NVFP4 groups 4..15, 4 = NVFP4 groups 0..3 (the RoPE halves)
DBGK = gl.constexpr(6)


@gluon.jit
def _nvfp4_quant(x, N: gl.constexpr, BLOCK_M: gl.constexpr, L2: gl.constexpr):
    # aiter fused_qk_norm_mrope._nvfp4_quant, the [1, 16] layout passed in (it carries the cluster's CGA)
    G: gl.constexpr = N // NVFP4_GROUP
    x = gl.convert_layout(x, L2)
    x = gl.reshape(x, [BLOCK_M, G, NVFP4_GROUP])
    amax = gl.max(gl.abs(x), axis=2)
    scale = amax / 6.0
    qx = x * (1.0 / scale)[:, :, None]
    qx = qx.to(gl.uint32, bitcast=True)
    s = qx & 0x80000000
    qx = qx ^ s
    qf = qx.to(gl.float32, bitcast=True)
    denormal = (qf < 1.0) & (qf < 6.0)
    normal = (qf >= 1.0) & (qf < 6.0)
    den = (qf + 4194304.0).to(gl.uint32, bitcast=True) - 1249902592
    den = den.to(gl.uint8)
    mant_odd = (qx >> 22) & 1
    nrm = ((qx - 1054867457 + mant_odd) >> 22).to(gl.uint8)
    e2m1 = gl.where(normal, nrm, gl.where(denormal, den, 7).to(gl.uint8))
    e2m1 = e2m1 | (s >> 28).to(gl.uint8)
    e2m1 = gl.reshape(e2m1, [BLOCK_M, G, NVFP4_GROUP // 2, 2])
    evens, odds = gl.split(e2m1)
    packed = evens | (odds << 4)
    return gl.reshape(packed, [BLOCK_M, N // 2]), scale


@gluon.jit
def _store_k_nvfp4(k_cache_ptr, x, page, offset, write, COL0: gl.constexpr, N: gl.constexpr,
                   BLOCK_M: gl.constexpr, BLOCK_SIZE: gl.constexpr, SCALE_PRESHUFFLE: gl.constexpr,
                   store_layout: gl.constexpr, L2: gl.constexpr, TAIL: gl.constexpr):
    # aiter _store_k_nvfp4 (TAIL False: columns [COL0, COL0 + N)) and _store_k_nvfp4_tail (TAIL True: the full row
    # quantized with its first 64 values zeroed, bytes [32, 128) / scales [4, 16) stored)
    D_BYTES: gl.constexpr = HEAD_DIM // 2
    D_SCALES: gl.constexpr = HEAD_DIM // NVFP4_GROUP
    P: gl.constexpr = SCALE_PRESHUFFLE
    P_QUARTER: gl.constexpr = SCALE_PRESHUFFLE // 4
    data, scales = _nvfp4_quant(x, N, BLOCK_M, L2)
    data = gl.convert_layout(data, store_layout)
    scales = gl.convert_layout(scales, store_layout)
    j = COL0 // 2 + gl.arange(0, N // 2, layout=gl.SliceLayout(0, store_layout))
    data_dst = (page + (offset // 16) * (D_BYTES * 16) + (offset % 16) * 16)[:, None] + ((j // 16) * 256 + j % 16)[None, :]
    g = COL0 // NVFP4_GROUP + gl.arange(0, N // NVFP4_GROUP, layout=gl.SliceLayout(0, store_layout))
    sub = offset % P
    scale_dst = (page + BLOCK_SIZE * D_BYTES + (offset // P) * (D_SCALES * P) + (sub % P_QUARTER) * (4 * D_SCALES)
                 + (sub // P_QUARTER) * D_SCALES)[:, None] + g[None, :]
    if TAIL:
        gl.store(k_cache_ptr + data_dst, data, mask=write[:, None] & (j >= ROPE_HALF)[None, :])
        gl.store(k_cache_ptr + scale_dst, scales.to(gl.float8e4nv).to(gl.uint8, bitcast=True),
                 mask=write[:, None] & (g >= 2 * ROPE_HALF // NVFP4_GROUP)[None, :])
    else:
        gl.store(k_cache_ptr + data_dst, data, mask=write[:, None])
        gl.store(k_cache_ptr + scale_dst, scales.to(gl.float8e4nv).to(gl.uint8, bitcast=True), mask=write[:, None])


@gluon.jit
def _load_slot(slot_mapping_ptr, num_slots, M, tile_m, BM: gl.constexpr, layout: gl.constexpr):
    token = tile_m * BM + gl.arange(0, BM, layout=layout)
    raw = gl.load(slot_mapping_ptr + gl.minimum(token, num_slots - 1))
    return gl.where((token < M) & (token < num_slots), raw.to(gl.int64), -1)


@gluon.jit
def _load_slot_rows(slot_mapping_ptr, num_slots, M, row0, BM: gl.constexpr, layout: gl.constexpr):
    token = row0 + gl.arange(0, BM, layout=layout)
    raw = gl.load(slot_mapping_ptr + gl.minimum(token, num_slots - 1))
    return gl.where((token < M) & (token < num_slots), raw.to(gl.int64), -1)


@gluon.jit
def _fold_pre(M, tile_m, hidx, q_weight_ptr, k_weight_ptr, positions_ptr, stride_positions, cos_ptr, sin_ptr,
              stride_cos, slot_mapping_ptr, num_slots, BM: gl.constexpr, Q_HEADS: gl.constexpr,
              SECTION_H: gl.constexpr, SECTION_W: gl.constexpr,
              row_layout: gl.constexpr, rope_layout: gl.constexpr, kd: gl.constexpr, ks: gl.constexpr,
              kd64: gl.constexpr, ks64: gl.constexpr):
    # the epilogue's loads that do not need the C tile, issued at kernel start (they land during the GEMM)
    return _fold_pre_rows(M, tile_m * BM, hidx, q_weight_ptr, k_weight_ptr, positions_ptr, stride_positions, cos_ptr,
                          sin_ptr, stride_cos, slot_mapping_ptr, num_slots, BM, Q_HEADS, SECTION_H, SECTION_W,
                          row_layout, rope_layout, kd, ks, kd64, ks64)


@gluon.jit
def _fold_pre_rows(M, row0, hidx, q_weight_ptr, k_weight_ptr, positions_ptr, stride_positions, cos_ptr, sin_ptr,
                   stride_cos, slot_mapping_ptr, num_slots, BM: gl.constexpr, Q_HEADS: gl.constexpr,
                   SECTION_H: gl.constexpr, SECTION_W: gl.constexpr,
                   row_layout: gl.constexpr, rope_layout: gl.constexpr, kd: gl.constexpr, ks: gl.constexpr,
                   kd64: gl.constexpr, ks64: gl.constexpr):
    # _fold_pre for the BM rows from row0 (a runtime row loop's chunk)
    is_q = hidx < Q_HEADS
    offs_n = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, row_layout))
    offs_r = gl.arange(0, ROPE_HALF, layout=gl.SliceLayout(0, rope_layout))
    token_r = row0 + gl.arange(0, BM, layout=gl.SliceLayout(1, rope_layout))
    mask_r = token_r < M
    pos_t = gl.load(positions_ptr + token_r, mask=mask_r, other=0)
    pos_h = gl.load(positions_ptr + stride_positions + token_r, mask=mask_r, other=0)
    pos_w = gl.load(positions_ptr + 2 * stride_positions + token_r, mask=mask_r, other=0)
    w_ptr = gl.where(is_q, q_weight_ptr, k_weight_ptr)
    w = gl.load(w_ptr + offs_n)[None, :]
    w_lo = gl.load(w_ptr + offs_r)[None, :]
    w_hi = gl.load(w_ptr + ROPE_HALF + offs_r)[None, :]
    use_h = (offs_r % 3 == 1) & (offs_r < 3 * SECTION_H)
    use_w = (offs_r % 3 == 2) & (offs_r < 3 * SECTION_W)
    pos = gl.where(use_h[None, :], pos_h[:, None], gl.where(use_w[None, :], pos_w[:, None], pos_t[:, None]))
    cos = gl.load(cos_ptr + pos * stride_cos + offs_r[None, :])
    sin = gl.load(sin_ptr + pos * stride_cos + offs_r[None, :])
    slot_raw = gl.load(slot_mapping_ptr + gl.minimum(token_r, num_slots - 1))
    slot = gl.where(mask_r & (token_r < num_slots), slot_raw.to(gl.int64), -1)
    slots = (slot, _load_slot_rows(slot_mapping_ptr, num_slots, M, row0, BM, gl.SliceLayout(1, kd)),
             _load_slot_rows(slot_mapping_ptr, num_slots, M, row0, BM, gl.SliceLayout(1, ks)),
             _load_slot_rows(slot_mapping_ptr, num_slots, M, row0, BM, gl.SliceLayout(1, kd64)),
             _load_slot_rows(slot_mapping_ptr, num_slots, M, row0, BM, gl.SliceLayout(1, ks64)))
    return w, w_lo, w_hi, cos, sin, slots


@gluon.jit
def _fold_epilogue(c_ptr, stride_c_m, M, tile_m, hidx, w, w_lo, w_hi, cos, sin, slots,
                   q_weight_ptr, k_weight_ptr, positions_ptr, stride_positions, cos_ptr, sin_ptr, stride_cos,
                   q_out_ptr, k_out_ptr, slot_mapping_ptr, num_slots, k_cache_ptr, v_cache_ptr, kv_inv_scale,
                   BLOCK_SIZE_M: gl.constexpr, num_ctas: gl.constexpr, EPS: gl.constexpr,
                   Q_HEADS: gl.constexpr, K_HEADS: gl.constexpr,
                   SECTION_H: gl.constexpr, SECTION_W: gl.constexpr, BLOCK_SIZE: gl.constexpr,
                   FP8_MAX: gl.constexpr, SCALE_PRESHUFFLE: gl.constexpr,
                   row_layout: gl.constexpr, rope_layout: gl.constexpr, kl: gl.constexpr, kd: gl.constexpr,
                   ks: gl.constexpr, kl64: gl.constexpr, kd64: gl.constexpr, ks64: gl.constexpr):
    # hidx: the tile's head among the Q, K, V heads after the gate (uniform over the cluster). The C tile of the
    # whole cluster is in memory once the barrier passes; this CTA does its share of the rows (the CGA of the
    # layouts splits rows over the cluster's CTAs). A8K4V8 (fp8 V), the BLOCK_SIZE_M rows of the tile in one go.
    _fold_epilogue_rows(c_ptr, stride_c_m, M, tile_m * BLOCK_SIZE_M, hidx, w, w_lo, w_hi, cos, sin, slots,
                        q_weight_ptr, k_weight_ptr, positions_ptr, stride_positions, cos_ptr, sin_ptr, stride_cos,
                        q_out_ptr, k_out_ptr, slot_mapping_ptr, num_slots, k_cache_ptr, v_cache_ptr, kv_inv_scale,
                        BLOCK_SIZE_M, num_ctas, EPS, Q_HEADS, K_HEADS, SECTION_H, SECTION_W, BLOCK_SIZE,
                        FP8_MAX, SCALE_PRESHUFFLE, row_layout, rope_layout, kl, kd, ks, kl64, kd64, ks64,
                        num_ctas > 1, False)


@gluon.jit
def _fold_epilogue_rows(c_ptr, stride_c_m, M, row0, hidx, w, w_lo, w_hi, cos, sin, slots,
                        q_weight_ptr, k_weight_ptr, positions_ptr, stride_positions, cos_ptr, sin_ptr, stride_cos,
                        q_out_ptr, k_out_ptr, slot_mapping_ptr, num_slots, k_cache_ptr, v_cache_ptr, kv_inv_scale,
                        BM: gl.constexpr, num_ctas: gl.constexpr, EPS: gl.constexpr,
                        Q_HEADS: gl.constexpr, K_HEADS: gl.constexpr,
                        SECTION_H: gl.constexpr, SECTION_W: gl.constexpr, BLOCK_SIZE: gl.constexpr,
                        FP8_MAX: gl.constexpr, SCALE_PRESHUFFLE: gl.constexpr,
                        row_layout: gl.constexpr, rope_layout: gl.constexpr, kl: gl.constexpr, kd: gl.constexpr,
                        ks: gl.constexpr, kl64: gl.constexpr, kd64: gl.constexpr, ks64: gl.constexpr,
                        WAIT: gl.constexpr, V_FP4PT: gl.constexpr, KLDS: gl.constexpr = False,
                        KWORD: gl.constexpr = False):
    # the BM rows [row0, row0 + BM) of the tile (WAIT: this call waits on the cluster barrier first). V_FP4PT: the
    # A8K4V4PT V page (e2m1 + one fp32 scale per token and head) instead of the fp8 one.
    QK: gl.constexpr = Q_HEADS + K_HEADS
    GATE_COLS: gl.constexpr = Q_HEADS * HEAD_DIM
    col0 = GATE_COLS + hidx * HEAD_DIM
    if hidx < QK:
        _fold_qk(c_ptr, stride_c_m, M, row0, hidx, col0, w, w_lo, w_hi, cos, sin, slots,
                 q_weight_ptr, k_weight_ptr, positions_ptr, stride_positions, cos_ptr, sin_ptr, stride_cos,
                 q_out_ptr, k_out_ptr, slot_mapping_ptr, num_slots, k_cache_ptr,
                 BM, WAIT, EPS, Q_HEADS, K_HEADS, SECTION_H, SECTION_W, BLOCK_SIZE,
                 FP8_MAX, SCALE_PRESHUFFLE, row_layout, rope_layout, kl, kd, ks, kl64, kd64, ks64, KLDS, KWORD)
    else:
        if V_FP4PT:
            _fold_v_fp4pt(c_ptr, stride_c_m, M, row0, hidx - QK, col0, slots[0], v_cache_ptr,
                          BM, WAIT, K_HEADS, BLOCK_SIZE, row_layout)
        else:
            _fold_v(c_ptr, stride_c_m, M, row0, hidx - QK, col0, slots[0], slot_mapping_ptr, num_slots, v_cache_ptr,
                    kv_inv_scale, BM, WAIT, K_HEADS, BLOCK_SIZE, FP8_MAX, rope_layout)


@gluon.jit
def _nvfp4_groups_store(k_cache_ptr, x, slot_d, slot_s, kh, G0: gl.constexpr, N: gl.constexpr, SKIP_G: gl.constexpr,
                        BM: gl.constexpr, K_HEADS: gl.constexpr, BLOCK_SIZE: gl.constexpr,
                        SCALE_PRESHUFFLE: gl.constexpr, kd: gl.constexpr, ks: gl.constexpr, WORDS: gl.constexpr = False):
    # x [BM, N] fp32, a 16-value group in one thread: groups G0 .. G0 + N / 16 of the K rows, NVFP4-quantized with
    # the arithmetic of aiter's _nvfp4_quant, stored into the shuffled NVFP4 K page (groups < SKIP_G not stored).
    D_BYTES: gl.constexpr = HEAD_DIM // 2
    D_SCALES: gl.constexpr = HEAD_DIM // NVFP4_GROUP
    K_ROW: gl.constexpr = D_BYTES + D_SCALES
    P: gl.constexpr = SCALE_PRESHUFFLE
    P_QUARTER: gl.constexpr = SCALE_PRESHUFFLE // 4
    G: gl.constexpr = N // NVFP4_GROUP
    x = gl.reshape(x, [BM, G, NVFP4_GROUP])
    amax = gl.max(gl.abs(x), axis=2)
    scale = amax / 6.0
    qx = x * (1.0 / scale)[:, :, None]
    qx = qx.to(gl.uint32, bitcast=True)
    sg = qx & 0x80000000
    qx = qx ^ sg
    qf = qx.to(gl.float32, bitcast=True)
    denormal = (qf < 1.0) & (qf < 6.0)
    normal = (qf >= 1.0) & (qf < 6.0)
    den = (qf + 4194304.0).to(gl.uint32, bitcast=True) - 1249902592
    den = den.to(gl.uint8)
    mant_odd = (qx >> 22) & 1
    nrm = ((qx - 1054867457 + mant_odd) >> 22).to(gl.uint8)
    e2m1 = gl.where(normal, nrm, gl.where(denormal, den, 7).to(gl.uint8))
    e2m1 = e2m1 | (sg >> 28).to(gl.uint8)
    e2m1 = gl.reshape(e2m1, [BM, G, NVFP4_GROUP // 2, 2])
    evens, odds = gl.split(e2m1)
    packed = gl.convert_layout(gl.reshape(evens | (odds << 4), [BM, N // 2]), kd)
    if WORDS:
        # 4 consecutive bytes of a lane (4 | the bytes a lane holds) as one dword store (16-byte-aligned runs)
        kdw: gl.constexpr = gl.BlockedLayout([1, kd.size_per_thread[1] // 4], kd.threads_per_warp, kd.warps_per_cta,
                                             kd.order, kd.cga_layout)
        p4 = gl.reshape(packed.to(gl.uint32), [BM, N // 8, 2, 2])
        pe, po = gl.split(p4)
        b0, b2 = gl.split(pe)
        b1, b3 = gl.split(po)
        word = gl.convert_layout(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24), kdw)
        sw = gl.convert_layout(slot_d, gl.SliceLayout(1, kdw))
        block = sw // BLOCK_SIZE
        offset = sw % BLOCK_SIZE
        page = (block * K_HEADS + kh) * (BLOCK_SIZE * K_ROW)
        j = G0 * (NVFP4_GROUP // 2) + 4 * gl.arange(0, N // 8, layout=gl.SliceLayout(0, kdw))
        data_dst = (page + (offset // 16) * (D_BYTES * 16) + (offset % 16) * 16)[:, None] + ((j // 16) * 256 + j % 16)[None, :]
        gl.store((k_cache_ptr + data_dst).to(gl.pointer_type(gl.uint32)), word,
                 mask=(sw >= 0)[:, None] & (j >= SKIP_G * (NVFP4_GROUP // 2))[None, :])
    else:
        block = slot_d // BLOCK_SIZE
        offset = slot_d % BLOCK_SIZE
        page = (block * K_HEADS + kh) * (BLOCK_SIZE * K_ROW)
        j = G0 * (NVFP4_GROUP // 2) + gl.arange(0, N // 2, layout=gl.SliceLayout(0, kd))
        data_dst = (page + (offset // 16) * (D_BYTES * 16) + (offset % 16) * 16)[:, None] + ((j // 16) * 256 + j % 16)[None, :]
        gl.store(k_cache_ptr + data_dst, packed,
                 mask=(slot_d >= 0)[:, None] & (j >= SKIP_G * (NVFP4_GROUP // 2))[None, :])
    sc = gl.convert_layout(scale, ks)
    block = slot_s // BLOCK_SIZE
    offset = slot_s % BLOCK_SIZE
    page = (block * K_HEADS + kh) * (BLOCK_SIZE * K_ROW)
    g = G0 + gl.arange(0, G, layout=gl.SliceLayout(0, ks))
    sub = offset % P
    scale_dst = (page + BLOCK_SIZE * D_BYTES + (offset // P) * (D_SCALES * P) + (sub % P_QUARTER) * (4 * D_SCALES)
                 + (sub // P_QUARTER) * D_SCALES)[:, None] + g[None, :]
    gl.store(k_cache_ptr + scale_dst, sc.to(gl.float8e4nv).to(gl.uint8, bitcast=True),
             mask=(slot_s >= 0)[:, None] & (g >= SKIP_G)[None, :])


@gluon.jit
def _nvfp4_rope_store(k_cache_ptr, r, slot_r, kh, G0: gl.constexpr, BM: gl.constexpr, K_HEADS: gl.constexpr,
                      BLOCK_SIZE: gl.constexpr, SCALE_PRESHUFFLE: gl.constexpr, rope_layout: gl.constexpr):
    # One RoPE half (r [BM, 32] fp32, one value a lane): NVFP4 groups G0, G0 + 1 of the K rows, quantized in the
    # RoPE layout (the 16-value amax and the e2m1 pairs over lanes), arithmetic of aiter's _nvfp4_quant.
    D_BYTES: gl.constexpr = HEAD_DIM // 2
    D_SCALES: gl.constexpr = HEAD_DIM // NVFP4_GROUP
    K_ROW: gl.constexpr = D_BYTES + D_SCALES
    P: gl.constexpr = SCALE_PRESHUFFLE
    P_QUARTER: gl.constexpr = SCALE_PRESHUFFLE // 4
    x = gl.reshape(r, [BM, 2, NVFP4_GROUP])
    amax = gl.max(gl.abs(x), axis=2)
    scale = amax / 6.0
    qx = x * (1.0 / scale)[:, :, None]
    qx = qx.to(gl.uint32, bitcast=True)
    sg = qx & 0x80000000
    qx = qx ^ sg
    qf = qx.to(gl.float32, bitcast=True)
    denormal = (qf < 1.0) & (qf < 6.0)
    normal = (qf >= 1.0) & (qf < 6.0)
    den = (qf + 4194304.0).to(gl.uint32, bitcast=True) - 1249902592
    den = den.to(gl.uint8)
    mant_odd = (qx >> 22) & 1
    nrm = ((qx - 1054867457 + mant_odd) >> 22).to(gl.uint8)
    e2m1 = gl.where(normal, nrm, gl.where(denormal, den, 7).to(gl.uint8))
    e2m1 = (e2m1 | (sg >> 28).to(gl.uint8)).to(gl.int32)
    c = G0 * NVFP4_GROUP + gl.arange(0, 2 * NVFP4_GROUP, layout=gl.SliceLayout(0, rope_layout))
    block = slot_r // BLOCK_SIZE
    offset = slot_r % BLOCK_SIZE
    page = (block * K_HEADS + kh) * (BLOCK_SIZE * K_ROW)
    sub = offset % P
    dbase = page + (offset // 16) * (D_BYTES * 16) + (offset % 16) * 16
    sbase = (page + BLOCK_SIZE * D_BYTES + (offset // P) * (D_SCALES * P) + (sub % P_QUARTER) * (4 * D_SCALES)
             + (sub // P_QUARTER) * D_SCALES)
    zero = gl.zeros([BM, 2 * NVFP4_GROUP], gl.int32, rope_layout)
    # the byte of a pair: low nibble the even value
    codes = gl.reshape(e2m1, [BM, 2 * NVFP4_GROUP]) << (4 * (c % 2))[None, :]
    byte = gl.sum(gl.reshape(codes, [BM, NVFP4_GROUP, 2]), axis=2)
    jb = c // 2
    jcol = gl.max(gl.reshape(((jb // 16) * 256 + jb % 16)[None, :] + zero, [BM, NVFP4_GROUP, 2]), axis=2)
    bl: gl.constexpr = jcol.type.layout
    dbase_b = gl.convert_layout(dbase, gl.SliceLayout(1, bl))
    w_b = gl.convert_layout(slot_r >= 0, gl.SliceLayout(1, bl))
    gl.store(k_cache_ptr + (dbase_b[:, None] + jcol), byte.to(gl.uint8), mask=w_b[:, None] & (jcol >= 0))
    gcol = gl.max(gl.reshape((c // NVFP4_GROUP)[None, :] + zero, [BM, 2, NVFP4_GROUP]), axis=2)
    gl_: gl.constexpr = gcol.type.layout
    sbase_g = gl.convert_layout(sbase, gl.SliceLayout(1, gl_))
    w_g = gl.convert_layout(slot_r >= 0, gl.SliceLayout(1, gl_))
    sc = gl.convert_layout(scale, gl_)
    gl.store(k_cache_ptr + (sbase_g[:, None] + gcol), sc.to(gl.float8e4nv).to(gl.uint8, bitcast=True),
             mask=w_g[:, None] & (gcol >= 0))


@gluon.jit
def _fold_qk(c_ptr, stride_c_m, M, row0, hidx, col0, w, w_lo, w_hi, cos, sin, slots,
             q_weight_ptr, k_weight_ptr, positions_ptr, stride_positions, cos_ptr, sin_ptr, stride_cos,
             q_out_ptr, k_out_ptr, slot_mapping_ptr, num_slots, k_cache_ptr,
             BM: gl.constexpr, WAIT: gl.constexpr, EPS: gl.constexpr,
             Q_HEADS: gl.constexpr, K_HEADS: gl.constexpr,
             SECTION_H: gl.constexpr, SECTION_W: gl.constexpr, BLOCK_SIZE: gl.constexpr,
             FP8_MAX: gl.constexpr, SCALE_PRESHUFFLE: gl.constexpr,
             row_layout: gl.constexpr, rope_layout: gl.constexpr, kl: gl.constexpr, kd: gl.constexpr,
             ks: gl.constexpr, kl64: gl.constexpr, kd64: gl.constexpr, ks64: gl.constexpr, KLDS: gl.constexpr = False,
             KWORD: gl.constexpr = False):
    # KLDS: the RoPE groups 0..3 of K quantized after one layout conversion (LDS) of the two RoPE halves into a
    # 16-values-a-lane layout (fewer dependent cross-lane steps when a wave holds several rows) instead of in the
    # RoPE layout
    is_q = hidx < Q_HEADS
    if True:
        offs_m = gl.arange(0, BM, layout=gl.SliceLayout(1, row_layout))
        offs_n = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, row_layout))
        token = row0 + offs_m
        mask_m = token < M
        offs_r = gl.arange(0, ROPE_HALF, layout=gl.SliceLayout(0, rope_layout))
        token_r = gl.convert_layout(token, gl.SliceLayout(1, rope_layout))
        mask_r = gl.convert_layout(mask_m, gl.SliceLayout(1, rope_layout))
        if WAIT:
            gl.amd.gfx1250.cluster.wait()
        # rows past M read row M - 1 (unmasked loads: a masked load of these wave-uniform rows is a branch)
        x_row = c_ptr + gl.minimum(token, M - 1).to(gl.int64) * stride_c_m + col0
        x_row_r = gl.convert_layout(x_row, gl.SliceLayout(1, rope_layout))
        x = gl.load(x_row[:, None] + offs_n[None, :])
        x_lo = gl.load(x_row_r[:, None] + offs_r[None, :])
        x_hi = gl.load(x_row_r[:, None] + ROPE_HALF + offs_r[None, :])

        # the native row kernel's arithmetic
        x = x.to(gl.float32)
        rstd = gl.rsqrt(gl.sum(x * x, axis=1) / HEAD_DIM + EPS)
        y = (x * rstd[:, None] * (w.to(gl.float32) + 1.0)).to(k_out_ptr.dtype.element_ty)
        tail = (offs_n >= 2 * ROPE_HALF)[None, :]
        rstd_r = gl.convert_layout(rstd, gl.SliceLayout(1, rope_layout))[:, None]
        y_lo = x_lo.to(gl.float32) * rstd_r * (w_lo.to(gl.float32) + 1.0)
        y_hi = x_hi.to(gl.float32) * rstd_r * (w_hi.to(gl.float32) + 1.0)
        rope_lo = (y_lo * cos - y_hi * sin).to(k_out_ptr.dtype.element_ty)
        rope_hi = (y_hi * cos + y_lo * sin).to(k_out_ptr.dtype.element_ty)

        if is_q:
            fp8_ty = q_out_ptr.dtype.element_ty
            offs_out = token * (Q_HEADS * HEAD_DIM) + hidx * HEAD_DIM
            out_n = offs_out[:, None] + offs_n[None, :]
            q_n = gl.minimum(gl.maximum(y.to(gl.float32), -FP8_MAX), FP8_MAX)
            gl.store(q_out_ptr + out_n, q_n.to(fp8_ty), mask=mask_m[:, None] & tail)
            offs_out_r = gl.convert_layout(offs_out, gl.SliceLayout(1, rope_layout))
            out_r = offs_out_r[:, None] + offs_r[None, :]
            q_lo = gl.minimum(gl.maximum(rope_lo.to(gl.float32), -FP8_MAX), FP8_MAX)
            q_hi = gl.minimum(gl.maximum(rope_hi.to(gl.float32), -FP8_MAX), FP8_MAX)
            gl.store(q_out_ptr + out_r, q_lo.to(fp8_ty), mask=mask_r[:, None])
            gl.store(q_out_ptr + ROPE_HALF + out_r, q_hi.to(fp8_ty), mask=mask_r[:, None])
        else:
            kh = hidx - Q_HEADS
            offs_out = token * (K_HEADS * HEAD_DIM) + kh * HEAD_DIM
            out_n = offs_out[:, None] + offs_n[None, :]
            if DBGK & 1:
                gl.store(k_out_ptr + out_n, y, mask=mask_m[:, None] & tail)
                offs_out_r = gl.convert_layout(offs_out, gl.SliceLayout(1, rope_layout))
                out_r = offs_out_r[:, None] + offs_r[None, :]
                gl.store(k_out_ptr + out_r, rope_lo, mask=mask_r[:, None])
                gl.store(k_out_ptr + ROPE_HALF + out_r, rope_hi, mask=mask_r[:, None])
            # NVFP4 K: groups 4..15 from the normalized row, groups 0..3 from the RoPE halves
            if DBGK & 2:
                _nvfp4_groups_store(k_cache_ptr, y.to(gl.float32), slots[1], slots[2], kh, 0, HEAD_DIM, 4,
                                    BM, K_HEADS, BLOCK_SIZE, SCALE_PRESHUFFLE, kd, ks, KWORD)
            if DBGK & 4:
                if KLDS:
                    r64 = gl.reshape(gl.permute(gl.join(rope_lo, rope_hi), (0, 2, 1)), [BM, 2 * ROPE_HALF])
                    _nvfp4_groups_store(k_cache_ptr, gl.convert_layout(r64, kl64).to(gl.float32), slots[3], slots[4],
                                        kh, 0, 2 * ROPE_HALF, 0, BM, K_HEADS, BLOCK_SIZE, SCALE_PRESHUFFLE, kd64, ks64,
                                        KWORD)
                else:
                    _nvfp4_rope_store(k_cache_ptr, rope_lo.to(gl.float32), slots[0], kh, 0, BM, K_HEADS, BLOCK_SIZE,
                                      SCALE_PRESHUFFLE, rope_layout)
                    _nvfp4_rope_store(k_cache_ptr, rope_hi.to(gl.float32), slots[0], kh, 2, BM, K_HEADS, BLOCK_SIZE,
                                      SCALE_PRESHUFFLE, rope_layout)


@gluon.jit
def _fold_v(c_ptr, stride_c_m, M, row0, vh, col0, slot_r, slot_mapping_ptr, num_slots, v_cache_ptr, kv_inv_scale,
            BM: gl.constexpr, WAIT: gl.constexpr, K_HEADS: gl.constexpr, BLOCK_SIZE: gl.constexpr,
            FP8_MAX: gl.constexpr, row_layout: gl.constexpr):
    # V head: fp8 with the per-tensor scale into the 5-D V page [block, head, offset // 16, d, offset % 16]
    # (8 consecutive values a lane: one 16-byte load; the stores are bytes 16 apart either way)
    X: gl.constexpr = 16
    offs_m = gl.arange(0, BM, layout=gl.SliceLayout(1, row_layout))
    token_r = row0 + offs_m
    mask_r = token_r < M
    slot = gl.convert_layout(slot_r, gl.SliceLayout(1, row_layout))
    offs_v = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, row_layout))
    if WAIT:
        gl.amd.gfx1250.cluster.wait()
    v_row = c_ptr + gl.minimum(token_r, M - 1).to(gl.int64) * stride_c_m + col0
    v = gl.load(v_row[:, None] + offs_v[None, :])
    write = (slot >= 0)[:, None]
    block = slot // BLOCK_SIZE
    offset = slot % BLOCK_SIZE
    cache_row = (block * K_HEADS + vh) * (HEAD_DIM * BLOCK_SIZE)
    v_dst = cache_row + (offset // X) * (HEAD_DIM * X) + offset % X
    v_c = gl.minimum(gl.maximum(v.to(gl.float32) * kv_inv_scale, -FP8_MAX), FP8_MAX)
    v_c = v_c.to(v_cache_ptr.dtype.element_ty)
    gl.store(v_cache_ptr + v_dst[:, None] + offs_v[None, :] * X, v_c, mask=write)


@gluon.jit
def _fold_v_fp4pt(c_ptr, stride_c_m, M, row0, vh, col0, slot_r, v_cache_ptr,
                  BM: gl.constexpr, WAIT: gl.constexpr, K_HEADS: gl.constexpr, BLOCK_SIZE: gl.constexpr,
                  vl: gl.constexpr):
    # V head, A8K4V4PT: aiter fused_qk_norm_mrope's _fp4pt_quant + _store_v_fp4pt_rows on the BM rows from row0
    # (vl: 8 consecutive values a lane; amax / div_rn / the e2m1 midpoint compares are exact, so any layout gives
    # the native bytes). V page [block, head, 128 tokens x 128 data bytes (NVFP4 data shuffle) | 128 fp32 scales].
    D_BYTES: gl.constexpr = HEAD_DIM // 2
    V_ROW: gl.constexpr = HEAD_DIM // 2 + 4
    offs_m = gl.arange(0, BM, layout=gl.SliceLayout(1, vl))
    token = row0 + offs_m
    slot = gl.convert_layout(slot_r, gl.SliceLayout(1, vl))
    offs_v = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, vl))
    if WAIT:
        gl.amd.gfx1250.cluster.wait()
    v_row = c_ptr + gl.minimum(token, M - 1).to(gl.int64) * stride_c_m + col0
    x = gl.load(v_row[:, None] + offs_v[None, :]).to(gl.float32)
    amax = gl.max(gl.abs(x), axis=1)
    s = amax * 0.1666666716337204
    s_div = gl.where(s > 0, s, 1.0)
    qv = gl.div_rn(x, gl.zeros_like(x) + s_div[:, None])
    mag = gl.abs(qv)
    code = ((mag > 0.25).to(gl.int32) + (mag > 0.75).to(gl.int32) + (mag > 1.25).to(gl.int32)
            + (mag > 1.75).to(gl.int32) + (mag > 2.5).to(gl.int32) + (mag > 3.5).to(gl.int32)
            + (mag > 5.0).to(gl.int32))
    code = code + ((mag == 0.75) | (mag == 1.75) | (mag == 3.5)).to(gl.int32)
    code = code | ((qv < 0).to(gl.int32) << 3)
    code = gl.reshape(code, [BM, HEAD_DIM // 2, 2])
    lo, hi = gl.split(code)
    SL: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [vl.warps_per_cta[0], 1], [1, 0], vl.cga_layout)
    packed = gl.convert_layout((lo | (hi << 4)).to(gl.uint8), SL)
    rows: gl.constexpr = gl.SliceLayout(1, SL)
    slot_p = gl.convert_layout(slot, rows)
    block = slot_p // BLOCK_SIZE
    offset = slot_p % BLOCK_SIZE
    page = (block * K_HEADS + vh) * (BLOCK_SIZE * V_ROW)
    j = gl.arange(0, D_BYTES, layout=gl.SliceLayout(0, SL))
    dst = (page + (offset // 16) * (D_BYTES * 16) + (offset % 16) * 16)[:, None] + ((j // 16) * 256 + j % 16)[None, :]
    gl.store(v_cache_ptr + dst, packed, mask=(slot_p >= 0)[:, None])
    s_ptr = (v_cache_ptr + page + BLOCK_SIZE * D_BYTES + 4 * offset).to(gl.pointer_type(gl.float32))
    gl.store(s_ptr, gl.convert_layout(s, rows), mask=slot_p >= 0)


@gluon.jit
def _gemm_mxfp8_qkv_qknorm_kernel(
    a_fp8_ptr, b_preshuf_ptr, c_ptr, a_scale_ptr, b_scale_ptr,
    M, N, K_elems,
    stride_a_m, stride_a_kbytes, stride_b_n16, stride_b_kshuf, stride_c_m, stride_c_n,
    stride_as_m, stride_as_k, stride_bs_n, stride_bs_k,
    q_weight_ptr, k_weight_ptr, positions_ptr, stride_positions, cos_ptr, sin_ptr, stride_cos,
    q_out_ptr, k_out_ptr, slot_mapping_ptr, num_slots, k_cache_ptr, v_cache_ptr, kv_inv_scale,
    BLOCK_SIZE_M: gl.constexpr, BLOCK_SIZE_N: gl.constexpr, BLOCK_SIZE_K: gl.constexpr,
    num_warps: gl.constexpr, NUM_BUFFERS: gl.constexpr,
    wmma_layout: gl.constexpr, wmma_acc_layout: gl.constexpr,
    shared_A: gl.constexpr, shared_B: gl.constexpr, shared_AS: gl.constexpr, shared_BS: gl.constexpr,
    shared_C: gl.constexpr, dot_a_layout: gl.constexpr, dot_b_layout: gl.constexpr,
    a_scale_layout: gl.constexpr, b_scale_layout: gl.constexpr, num_ctas: gl.constexpr,
    EPS: gl.constexpr, Q_HEADS: gl.constexpr, K_HEADS: gl.constexpr,
    SECTION_H: gl.constexpr, SECTION_W: gl.constexpr, BLOCK_SIZE: gl.constexpr,
    FP8_MAX: gl.constexpr, SCALE_PRESHUFFLE: gl.constexpr,
    row_layout: gl.constexpr, rope_layout: gl.constexpr, kl: gl.constexpr, kd: gl.constexpr,
    ks: gl.constexpr, kl64: gl.constexpr, kd64: gl.constexpr, ks64: gl.constexpr,
    SCALE_MODE: gl.constexpr = 2, B2B_WMMA: gl.constexpr = 0,
    A_FMT: gl.constexpr = "e4m3", B_FMT: gl.constexpr = "e4m3",
    FOLD: gl.constexpr = True,
    DBG: gl.constexpr = 0,
    EPI_MASK: gl.constexpr = 7,
    A_PF: gl.constexpr = 1,
    BM_SUB: gl.constexpr = 0,
    V_FP4PT: gl.constexpr = False,
    KLDS: gl.constexpr = False,
    KWORD: gl.constexpr = False,
):
    # _gemm_mxfp8_preshuffle_kernel restricted to NUM_KSUB 1, EPI_SPLIT 1, 1x32 x / w scales (n32k4), unshuffled A;
    # A_PF 1: un-shuffled (M, K // 32) x scales (M < 32), 32: n32k4-shuffled ones (M >= 32, 32+ rows a CTA). Then the
    # fold epilogue: with BM_SUB (rows of the epilogue layouts, one a wave) < BLOCK_SIZE_M, a runtime loop over the
    # tile's valid rows in BM_SUB chunks.
    if num_warps == 2:
        TDM_OPS_PER_STAGE: gl.constexpr = 2
    else:
        TDM_OPS_PER_STAGE: gl.constexpr = 1
    SCALE_GROUP_ELEMS: gl.constexpr = 32
    BLOCK_K_BYTES: gl.constexpr = BLOCK_SIZE_K
    K_GROUPS: gl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_ELEMS
    gl.static_assert(SCALE_MODE == 2)
    PRESHUFFLE_FACTOR: gl.constexpr = 32
    SCALE_KWIDTH: gl.constexpr = 4
    CTAS_M: gl.constexpr = cluster_shape(num_ctas, BLOCK_SIZE_M, BLOCK_SIZE_N)[0]
    gl.static_assert(A_PF == 1 or (BLOCK_SIZE_M // CTAS_M) % 32 == 0)
    gl.static_assert(BLOCK_SIZE_N == HEAD_DIM)
    A_PRESHUFFLE_FACTOR: gl.constexpr = A_PF
    gl.static_assert(A_PF == 1 or BLOCK_SIZE_M % 32 == 0)
    AS_STEP: gl.constexpr = K_GROUPS * A_PF
    if BM_SUB == 0:
        EPI_M: gl.constexpr = BLOCK_SIZE_M
    else:
        EPI_M: gl.constexpr = BM_SUB
    BS_STEP: gl.constexpr = K_GROUPS * PRESHUFFLE_FACTOR

    if B2B_WMMA:
        gl.inline_asm_elementwise(
            "s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1; v_mov_b32 $0, 0;",
            constraints="=v", args=[], dtype=gl.int32, is_pure=False, pack=1)

    pid = gl.program_id(axis=0)
    tiles_n = gl.cdiv(N, BLOCK_SIZE_N)
    tile_m = pid // tiles_n
    tile_n = pid - tile_m * tiles_n

    GATE_TILES: gl.constexpr = Q_HEADS
    if True:
        w_e, wlo_e, whi_e, cos_e, sin_e, slot_e = _fold_pre_rows(
            M, tile_m * BLOCK_SIZE_M, tile_n - GATE_TILES, q_weight_ptr, k_weight_ptr, positions_ptr, stride_positions, cos_ptr,
            sin_ptr, stride_cos, slot_mapping_ptr, num_slots, EPI_M, Q_HEADS, SECTION_H, SECTION_W,
            row_layout, rope_layout, kd, ks, kd64, ks64)

    K_bytes = K_elems
    k_tiles = gl.cdiv(K_bytes, BLOCK_K_BYTES)
    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_fp8_ptr + tile_m * BLOCK_SIZE_M * stride_a_m,
        shape=(M - tile_m * BLOCK_SIZE_M, K_bytes), strides=(stride_a_m, stride_a_kbytes),
        block_shape=(BLOCK_SIZE_M, BLOCK_K_BYTES), layout=shared_A)
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_preshuf_ptr + tile_n * (BLOCK_SIZE_N // 16) * stride_b_n16,
        shape=(gl.cdiv(N, 16) - tile_n * (BLOCK_SIZE_N // 16), K_bytes * 16), strides=(stride_b_n16, stride_b_kshuf),
        block_shape=(BLOCK_SIZE_N // 16, BLOCK_K_BYTES * 16), layout=shared_B)
    k_scale_cols = K_elems // SCALE_GROUP_ELEMS
    as_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_scale_ptr + tile_m * (BLOCK_SIZE_M // A_PF) * stride_as_m,
        shape=(gl.cdiv(M, A_PF) - tile_m * (BLOCK_SIZE_M // A_PF), k_scale_cols * A_PF),
        strides=(stride_as_m, stride_as_k),
        block_shape=(BLOCK_SIZE_M // A_PF, K_GROUPS * A_PF), layout=shared_AS)
    bs_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_scale_ptr + tile_n * (BLOCK_SIZE_N // PRESHUFFLE_FACTOR) * stride_bs_n,
        shape=(gl.cdiv(N, PRESHUFFLE_FACTOR) - tile_n * (BLOCK_SIZE_N // PRESHUFFLE_FACTOR),
               k_scale_cols * PRESHUFFLE_FACTOR),
        strides=(stride_bs_n, stride_bs_k),
        block_shape=(BLOCK_SIZE_N // PRESHUFFLE_FACTOR, K_GROUPS * PRESHUFFLE_FACTOR), layout=shared_BS)

    smem_A = gl.allocate_shared_memory(a_fp8_ptr.type.element_ty, [NUM_BUFFERS, BLOCK_SIZE_M, BLOCK_K_BYTES],
                                       layout=shared_A)
    smem_B = gl.allocate_shared_memory(b_preshuf_ptr.type.element_ty,
                                       [NUM_BUFFERS, BLOCK_SIZE_N // 16, BLOCK_K_BYTES * 16], layout=shared_B)
    smem_AS = gl.allocate_shared_memory(a_scale_ptr.type.element_ty,
                                        [NUM_BUFFERS, BLOCK_SIZE_M // A_PF, K_GROUPS * A_PF], layout=shared_AS)
    smem_BS = gl.allocate_shared_memory(b_scale_ptr.type.element_ty,
                                        [NUM_BUFFERS, BLOCK_SIZE_N // PRESHUFFLE_FACTOR, K_GROUPS * PRESHUFFLE_FACTOR],
                                        layout=shared_BS)
    load_idx = 0
    compute_idx = 0
    acc = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=wmma_acc_layout)
    for _ in gl.static_range(NUM_BUFFERS):
        slot = load_idx % NUM_BUFFERS
        a_slot = smem_A.index(slot)
        b_slot = smem_B.index(slot)
        as_slot = smem_AS.index(slot)
        bs_slot = smem_BS.index(slot)
        off_a = load_idx * BLOCK_K_BYTES
        off_b = load_idx * BLOCK_K_BYTES * 16
        off_as = load_idx * AS_STEP
        off_bs = load_idx * BS_STEP
        _issue_tdm_stage(a_desc, b_desc, as_desc, bs_desc, a_slot, b_slot, as_slot, bs_slot, off_a, off_b, off_as, off_bs, num_warps, False)  # fmt: skip
        load_idx += 1

    gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * TDM_OPS_PER_STAGE)
    slot_c = compute_idx % NUM_BUFFERS
    cur_A = smem_A.index(slot_c).load(layout=dot_a_layout)
    cur_B = depreshuffle_b_raw_to_kn(smem_B.index(slot_c), BLOCK_N=BLOCK_SIZE_N, BLOCK_K_BYTES=BLOCK_K_BYTES).load(
        layout=dot_b_layout)
    cur_AS = _load_a_scales(smem_AS.index(slot_c), BLOCK_SIZE_M, K_GROUPS, A_PRESHUFFLE_FACTOR, SCALE_KWIDTH, 32, a_scale_layout)  # fmt: skip
    cur_BS = _load_b_scales(smem_BS.index(slot_c), BLOCK_SIZE_N, K_GROUPS, PRESHUFFLE_FACTOR, SCALE_KWIDTH, 1, 1, b_scale_layout, None, None)  # fmt: skip
    if num_ctas > 1:
        gl.amd.gfx1250.cluster.arrive()
    main_iters = k_tiles - (NUM_BUFFERS)
    for _ in range(main_iters):
        acc = gl.amd.gfx1250.wmma_scaled(cur_A, cur_AS, A_FMT, cur_B, cur_BS, B_FMT, acc)
        if num_ctas > 1:
            gl.amd.gfx1250.cluster.wait()
        slot = load_idx % NUM_BUFFERS
        a_slot = smem_A.index(slot)
        b_slot = smem_B.index(slot)
        as_slot = smem_AS.index(slot)
        bs_slot = smem_BS.index(slot)
        off_a = load_idx * BLOCK_K_BYTES
        off_b = load_idx * BLOCK_K_BYTES * 16
        off_as = load_idx * AS_STEP
        off_bs = load_idx * BS_STEP
        _issue_tdm_stage(a_desc, b_desc, as_desc, bs_desc, a_slot, b_slot, as_slot, bs_slot, off_a, off_b, off_as, off_bs, num_warps, False)  # fmt: skip
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * TDM_OPS_PER_STAGE)
        load_idx += 1
        next_slot = (compute_idx + 1) % NUM_BUFFERS
        cur_A = smem_A.index(next_slot).load(layout=dot_a_layout)
        cur_B = depreshuffle_b_raw_to_kn(smem_B.index(next_slot), BLOCK_N=BLOCK_SIZE_N, BLOCK_K_BYTES=BLOCK_K_BYTES).load(layout=dot_b_layout)  # fmt: skip
        cur_AS = _load_a_scales(smem_AS.index(next_slot), BLOCK_SIZE_M, K_GROUPS, A_PRESHUFFLE_FACTOR, SCALE_KWIDTH, 32, a_scale_layout)  # fmt: skip
        cur_BS = _load_b_scales(smem_BS.index(next_slot), BLOCK_SIZE_N, K_GROUPS, PRESHUFFLE_FACTOR, SCALE_KWIDTH, 1, 1, b_scale_layout, None, None)  # fmt: skip
        compute_idx += 1
        if num_ctas > 1:
            gl.amd.gfx1250.cluster.arrive()
    if num_ctas > 1 and NUM_BUFFERS > 1:
        gl.amd.gfx1250.cluster.wait()
    for i in gl.static_range(NUM_BUFFERS - 1):
        acc = gl.amd.gfx1250.wmma_scaled(cur_A, cur_AS, A_FMT, cur_B, cur_BS, B_FMT, acc)
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * TDM_OPS_PER_STAGE)
        next_slot = (compute_idx + 1) % NUM_BUFFERS
        cur_A = smem_A.index(next_slot).load(layout=dot_a_layout)
        cur_B = depreshuffle_b_raw_to_kn(smem_B.index(next_slot), BLOCK_N=BLOCK_SIZE_N, BLOCK_K_BYTES=BLOCK_K_BYTES).load(layout=dot_b_layout)  # fmt: skip
        cur_AS = _load_a_scales(smem_AS.index(next_slot), BLOCK_SIZE_M, K_GROUPS, A_PRESHUFFLE_FACTOR, SCALE_KWIDTH, 32, a_scale_layout)  # fmt: skip
        cur_BS = _load_b_scales(smem_BS.index(next_slot), BLOCK_SIZE_N, K_GROUPS, PRESHUFFLE_FACTOR, SCALE_KWIDTH, 1, 1, b_scale_layout, None, None)  # fmt: skip
        compute_idx += 1
    if num_ctas > 1 and NUM_BUFFERS > 1:
        gl.amd.gfx1250.cluster.arrive()
    acc = gl.amd.gfx1250.wmma_scaled(cur_A, cur_AS, A_FMT, cur_B, cur_BS, B_FMT, acc)
    if num_ctas > 1:
        gl.amd.gfx1250.cluster.wait()

    c_buffer = gl.allocate_shared_memory(c_ptr.type.element_ty, shape=[BLOCK_SIZE_M, BLOCK_SIZE_N], layout=shared_C)
    c_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=c_ptr, shape=(M, N), strides=(stride_c_m, stride_c_n),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), layout=shared_C)
    c_buffer.store(acc.to(c_ptr.type.element_ty))
    gl.amd.gfx1250.tdm.async_store(c_desc, [tile_m * BLOCK_SIZE_M, tile_n * BLOCK_SIZE_N], c_buffer)
    gl.amd.gfx1250.tdm.async_wait(0)

    if DBG == 1:
        if tile_n >= GATE_TILES:
            gl.amd.gfx1250.cluster.arrive()
            gl.amd.gfx1250.cluster.wait()
    if DBG == 2:
        if tile_n >= GATE_TILES:
            gl.amd.gfx1250.cluster.arrive()
            gl.amd.gfx1250.cluster.wait()
            offs_m = gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, row_layout))
            offs_n = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, row_layout))
            xx = gl.load(c_ptr + gl.minimum(offs_m, M - 1)[:, None] * stride_c_m + tile_n * HEAD_DIM + offs_n[None, :])
            sm = gl.sum(xx.to(gl.float32), axis=1)
            gl.store(k_out_ptr + offs_m + tile_n * 16, sm.to(gl.bfloat16), mask=offs_m < M)
    if DBG == 3:
        if tile_n >= GATE_TILES:
            offs_m = gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, row_layout))
            offs_n = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, row_layout))
            xx = gl.load(c_ptr + gl.minimum(offs_m, M - 1)[:, None] * stride_c_m + tile_n * HEAD_DIM + offs_n[None, :])
            sm = gl.sum(xx.to(gl.float32), axis=1)
            gl.store(k_out_ptr + offs_m + tile_n * 16, sm.to(gl.bfloat16), mask=offs_m < M)
    hk = tile_n - GATE_TILES
    kind_ok = ((hk >= 0) & (hk < Q_HEADS) & ((EPI_MASK & 1) != 0)) | ((hk >= Q_HEADS) & (hk < Q_HEADS + K_HEADS) & ((EPI_MASK & 2) != 0)) | ((hk >= Q_HEADS + K_HEADS) & ((EPI_MASK & 4) != 0))
    if FOLD:
        if kind_ok:
            if num_ctas > 1:
                gl.amd.gfx1250.cluster.arrive()
            if EPI_M == BLOCK_SIZE_M:
                _fold_epilogue_rows(c_ptr, stride_c_m, M, tile_m * BLOCK_SIZE_M, tile_n - GATE_TILES,
                                    w_e, wlo_e, whi_e, cos_e, sin_e, slot_e,
                                    q_weight_ptr, k_weight_ptr, positions_ptr, stride_positions, cos_ptr, sin_ptr,
                                    stride_cos, q_out_ptr, k_out_ptr, slot_mapping_ptr, num_slots, k_cache_ptr,
                                    v_cache_ptr, kv_inv_scale, BLOCK_SIZE_M, num_ctas, EPS, Q_HEADS, K_HEADS,
                                    SECTION_H, SECTION_W, BLOCK_SIZE, FP8_MAX, SCALE_PRESHUFFLE,
                                    row_layout, rope_layout, kl, kd, ks, kl64, kd64, ks64, num_ctas > 1, V_FP4PT, KLDS, KWORD)
            else:
                # runtime row loop: the valid rows of the tile in EPI_M chunks, one row a wave each; chunk 0's
                # row-dependent loads were issued at kernel start, chunk i + 1's are issued before chunk i's work
                if num_ctas > 1:
                    gl.amd.gfx1250.cluster.wait()
                row_base = tile_m * BLOCK_SIZE_M
                n_chunks = gl.cdiv(gl.minimum(M - row_base, BLOCK_SIZE_M), EPI_M)
                cos_c = cos_e
                sin_c = sin_e
                slots_c = slot_e
                for it in range(n_chunks):
                    row0 = row_base + it * EPI_M
                    _w, _wlo, _whi, cos_n, sin_n, slot_n = _fold_pre_rows(
                        M, row0 + EPI_M, tile_n - GATE_TILES, q_weight_ptr, k_weight_ptr, positions_ptr,
                        stride_positions, cos_ptr, sin_ptr, stride_cos, slot_mapping_ptr, num_slots, EPI_M, Q_HEADS,
                        SECTION_H, SECTION_W, row_layout, rope_layout, kd, ks, kd64, ks64)
                    _fold_epilogue_rows(c_ptr, stride_c_m, M, row0, tile_n - GATE_TILES,
                                        w_e, wlo_e, whi_e, cos_c, sin_c, slots_c,
                                        q_weight_ptr, k_weight_ptr, positions_ptr, stride_positions, cos_ptr, sin_ptr,
                                        stride_cos, q_out_ptr, k_out_ptr, slot_mapping_ptr, num_slots, k_cache_ptr,
                                        v_cache_ptr, kv_inv_scale, EPI_M, num_ctas, EPS, Q_HEADS, K_HEADS,
                                        SECTION_H, SECTION_W, BLOCK_SIZE, FP8_MAX, SCALE_PRESHUFFLE,
                                        row_layout, rope_layout, kl, kd, ks, kl64, kd64, ks64, False, V_FP4PT, KLDS, KWORD)
                    cos_c = cos_n
                    sin_c = sin_n
                    slots_c = slot_n


# ------------------------------------------------------------------------------------------------------- host
# Head-aligned cluster tiles (BLOCK_N 256 = one head) keeping the tuned per-CTA tile of each decode bucket:
#   M <= 4 : tuned 16x32, 2 warps, 1 CTA       -> cluster 16x256 over 8 CTAs (16x32 each)
#   M <= 8 : tuned 16x128 over 4 CTAs (16x32)  -> 16x256 over 8 CTAs (16x32 each)
#   M <= 16: tuned 16x256 over 4 CTAs (16x64)  -> the same
#   M <= 32: tuned 16x128 / 4 CTAs (M <= 31), 32x32 1 CTA (M 32) -> 32x256 over 8 CTAs (32x32 each, 4 warps: one
#            row a wave in the epilogue); one M tile (two 16-row tiles read the weights twice: +1 us)
#   M <= 48: tuned 64x256 over 4 CTAs (64x64, 4 warps) -> the same tile with 8 warps: the epilogue holds 2 rows a
#            wave (register repeats: their latencies overlap; a runtime row loop is ~1.3 us slower; 4 warps / 4 rows a
#            wave is faster in a back-to-back bench but ~0.4 us slower in the engine, where the GEMM is HBM-bound
#            anyway), K's RoPE groups after one LDS layout conversion (KLDS). x scales n32k4-shuffled from M 32
#            (A_PF 32), as the GEMM's caller gives them.
# (ext48b/kcheck sweeps, gfx1250 MI455X, instruction prefetch on: 64x256 over 8 CTAs (2x4 or 1x8 split), 8 warps,
#  two 32-row M tiles (the weights read twice), the RoPE in the norm layout, touching the K / V cache lines early,
#  K/V-first tile order: all slower or no gain.)
FOLD_CONFIGS = {
    4: dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=256, BLOCK_SIZE_K=512, num_warps=2, NUM_BUFFERS=4, num_ctas=8, B2B_WMMA=0),
    8: dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=256, BLOCK_SIZE_K=512, num_warps=4, NUM_BUFFERS=4, num_ctas=8, B2B_WMMA=1),
    16: dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=256, BLOCK_SIZE_K=512, num_warps=4, NUM_BUFFERS=4, num_ctas=4, B2B_WMMA=1),
    32: dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=256, BLOCK_SIZE_K=512, num_warps=4, NUM_BUFFERS=4, num_ctas=8, B2B_WMMA=1),
    48: dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=256, BLOCK_SIZE_K=512, num_warps=8, NUM_BUFFERS=4, num_ctas=4, B2B_WMMA=0,
             BM_SUB=0, KLDS=True),
}
if __import__("os").environ.get("ATOM_FOLD_QKNORM_QKV_CFG"):
    # tuning: {"<bucket>": {config overrides}}, e.g. {"48": {"BM_SUB": 16}}
    for _b, _c in __import__("json").loads(__import__("os").environ["ATOM_FOLD_QKNORM_QKV_CFG"]).items():
        FOLD_CONFIGS.setdefault(int(_b), dict(FOLD_CONFIGS[48])).update(_c)
FOLD_MAX_M = 16          # the ATOM glue's default (today's); ATOM_FOLD_QKNORM_QKV_MAX_M raises it up to KERNEL_MAX_M
KERNEL_MAX_M = 48


def fold_config(M):
    for b in sorted(FOLD_CONFIGS):
        if M <= b:
            return dict(FOLD_CONFIGS[b])
    return None


@functools.cache
def _epi_layouts(num_warps, num_ctas):
    cga = [[1 << i, 0] for i in range(num_ctas.bit_length() - 1)]
    return dict(
        row_layout=gl.BlockedLayout([1, HEAD_DIM // 32], [1, 32], [num_warps, 1], [1, 0], cga),
        rope_layout=gl.BlockedLayout([1, 1], [1, 32], [num_warps, 1], [1, 0], cga),
        kl=gl.BlockedLayout([1, 16], [2, 16], [num_warps, 1], [1, 0], cga),
        kd=gl.BlockedLayout([1, 4], [1, 32], [num_warps, 1], [1, 0], cga),
        ks=gl.SliceLayout(2, gl.BlockedLayout([1, 1, 8], [1, 16, 2], [num_warps, 1, 1], [2, 1, 0],
                                              [b + [0] for b in cga])),
        kl64=gl.BlockedLayout([1, 16], [8, 4], [num_warps, 1], [1, 0], cga),
        kd64=gl.BlockedLayout([1, 8], [8, 4], [num_warps, 1], [1, 0], cga),
        ks64=gl.SliceLayout(2, gl.BlockedLayout([1, 1, 16], [8, 4, 1], [num_warps, 1, 1], [2, 1, 0],
                                                [b + [0] for b in cga])),
    )


@functools.cache
def _gemm_layouts(num_warps, BM, BN, BK, num_ctas):
    layouts = get_gemm_afp8wfp8_preshuffle_layouts(num_warps, BM, BN, BK, num_ctas, 2, 1, 0, 1, 32, (1, 32))
    for k_ in ("b_scale_layout_epi", "shared_C_epi"):
        layouts.pop(k_, None)
    layouts.update(_epi_layouts(num_warps, num_ctas))
    return layouts


def _fp8_e4m3():
    from aiter.ops.triton.utils.types import e4m3_dtype

    return e4m3_dtype


def fused_gemm_mxfp8_qkv_qknorm_mrope_cache(
    x, w, x_scales, w_scales, q_weight, k_weight, eps, positions, cos, sin, mrope_section,
    k_cache, v_cache, slot_mapping, kv_scale, num_q_heads, num_kv_heads, y=None, config=None,
):
    """y = x @ w^T (bf16 [M, N], N = (2 Hq + 2 Hk) * 256: gate | q | k | v) like gemm_afp8wfp8_preshuffle
    (n32k4, 1x32; x_scales un-shuffled (M, K // 32) for M < 32, n32k4 (pad32(M) // 32, K) from 32), plus what
    fused_qk_norm_mrope_cache(q, k, v, ..., kv_scale) does for an A8K4V8 cache (v_cache fp8 5-D) or
    fused_qk_norm_mrope_cache_k4v4pt for an A8K4V4PT one (v_cache [blocks, Hk, 128, 132] fp4pt, 4-D):
    returns (y, q_out fp8 [M, Hq*256], k_out bf16 [M, Hk*256]) and writes K (NVFP4) / V (fp8 | fp4pt) into the cache."""
    M, K = x.shape
    N = w.shape[0] * 16
    assert M <= KERNEL_MAX_M and N == (2 * num_q_heads + 2 * num_kv_heads) * 256
    v_fp4pt = v_cache.dim() == 4
    cfg = fold_config(M)
    if config:
        cfg.update(config)
    a_pf = 32 if M >= 32 else 1
    assert a_pf == 1 or cfg["BLOCK_SIZE_M"] >= 32
    sub = cfg["num_warps"] * cfg["num_ctas"]
    cfg.setdefault("BM_SUB", sub if sub < cfg["BLOCK_SIZE_M"] else 0)
    if y is None:
        y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    if v_fp4pt:
        q_dtype = _fp8_e4m3()
        vc = v_cache.view(torch.uint8)
        assert vc.shape == (k_cache.shape[0], num_kv_heads, 128, 256 // 2 + 4) and vc.is_contiguous()
    else:
        q_dtype = v_cache.dtype
        vc = v_cache
    q_out = torch.empty((M, num_q_heads * 256), dtype=q_dtype, device=x.device)
    k_out = torch.empty((M, num_kv_heads * 256), dtype=torch.bfloat16, device=x.device)
    k_tiles = triton.cdiv(K, cfg["BLOCK_SIZE_K"])
    cfg["NUM_BUFFERS"] = min(cfg["NUM_BUFFERS"], k_tiles)
    layouts = _gemm_layouts(cfg["num_warps"], cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"],
                            cfg["num_ctas"])
    x_k = x.view(torch.uint8)
    w_k = w.view(torch.uint8)
    xs, ws = x_scales.view(torch.uint8), w_scales.view(torch.uint8)
    if slot_mapping.shape[0] == 0:
        slot_mapping = torch.full((1,), -1, dtype=slot_mapping.dtype, device=slot_mapping.device)
    page_block = k_cache.shape[2]
    half = cos.shape[1]
    assert half == 32
    grid = (triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * (N // 256),)
    _gemm_mxfp8_qkv_qknorm_kernel[grid](
        x_k, w_k, y, xs, ws, M, N, K,
        x_k.stride(0), x_k.stride(1), w_k.stride(0), w_k.stride(1), y.stride(0), y.stride(1),
        xs.stride(0), xs.stride(1), ws.stride(0), ws.stride(1),
        q_weight, k_weight, positions, positions.stride(0), cos, sin, cos.stride(0),
        q_out, k_out, slot_mapping, slot_mapping.shape[0], k_cache, vc, 1.0 / kv_scale,
        EPS=eps, Q_HEADS=num_q_heads, K_HEADS=num_kv_heads, SECTION_H=mrope_section[1], SECTION_W=mrope_section[2],
        BLOCK_SIZE=page_block, FP8_MAX=torch.finfo(q_dtype).max, SCALE_PRESHUFFLE=min(128, page_block),
        SCALE_MODE=2, A_PF=a_pf, V_FP4PT=v_fp4pt, **cfg, **layouts,
    )
    return y, q_out, k_out


# ------------------------------------------------------------------------------------------------------- ATOM glue
_FOLD_ENV = None
# ATOM_FOLD_QKNORM_QKV_FP4PT=1: also fold for an A8K4V4PT cache (--kv_cache_dtype fp8_k4v4pt: fp4pt V);
# ATOM_FOLD_QKNORM_QKV_MAX_M: the largest decode batch (rows) folded (default 16, up to KERNEL_MAX_M = 48).
_FP4PT_ENV = None
_MAX_M_ENV = None


def fold_enabled() -> bool:
    global _FOLD_ENV
    if _FOLD_ENV is None:
        import os

        _FOLD_ENV = os.environ.get("ATOM_FOLD_QKNORM_QKV", "0") == "1"
    return _FOLD_ENV


def _fold_fp4pt() -> bool:
    global _FP4PT_ENV
    if _FP4PT_ENV is None:
        import os

        _FP4PT_ENV = os.environ.get("ATOM_FOLD_QKNORM_QKV_FP4PT", "0") == "1"
    return _FP4PT_ENV


def _fold_max_m() -> int:
    global _MAX_M_ENV
    if _MAX_M_ENV is None:
        import os

        _MAX_M_ENV = max(1, min(KERNEL_MAX_M, int(os.environ.get("ATOM_FOLD_QKNORM_QKV_MAX_M", str(FOLD_MAX_M)))))
    return _MAX_M_ENV


def try_fold_qkv(attn, positions, hidden_states, x_scale):
    """ATOM_FOLD_QKNORM_QKV: Qwen3NextAttention's qkv_proj with the Q/K norm, MRoPE and the A8K4V8 (or, with
    ATOM_FOLD_QKNORM_QKV_FP4PT, A8K4V4PT) cache write in its epilogue, for a decode batch of at most
    ATOM_FOLD_QKNORM_QKV_MAX_M (16) rows. Returns the qkv_proj output (gate | q | k | v, bf16) and leaves the fp8 Q /
    bf16 K for the attention's rope_cache (impl._fold_pending), or None (not applicable: the caller runs qkv_proj and
    the attention runs fused_qk_norm_mrope_cache(_k4v4pt) as before)."""
    if not fold_enabled() or torch.compiler.is_compiling():
        return None
    from atom.utils.forward_context import get_forward_context

    layer = attn.qkv_proj
    impl = getattr(attn.attn, "impl", None)
    if (
        not getattr(layer, "mx1x32", False)
        or impl is None
        or not getattr(impl, "k4v8", False)
        or (getattr(impl, "k4v4pt", False) and not _fold_fp4pt())
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
    n, k = layer.mx1x32_nk
    x2 = hidden_states.reshape(-1, k)
    m = x2.shape[0]
    if m == 0 or m > _fold_max_m():
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
    from atom.model_ops import mxfp8_1x32 as mx

    if x_scale is None:
        xq, xs = mx.quant_standalone(x2)
    else:
        xq, xs = x2, mx.gemm_scale_view(x_scale.reshape(-1), m, k)
    rot = attn.rotary_emb
    half = rot.rotary_dim // 2
    y, q_out, k_out = fused_gemm_mxfp8_qkv_qknorm_mrope_cache(
        xq, layer.weight, xs, layer.mx1x32_w_scale, attn.q_norm.weight, attn.k_norm.weight,
        attn.q_norm.variance_epsilon, positions, rot.cos_cache.view(-1, half), rot.sin_cache.view(-1, half),
        rot.mrope_section, k_cache, v_cache, fc.attn_metadata.slot_mapping, impl.kv_scale_float,
        attn.num_heads, attn.num_kv_heads,
    )
    impl._fold_pending = (q_out, k_out)
    return y.view(*hidden_states.shape[:-1], n)
