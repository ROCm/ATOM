# Derived from SGLang (https://github.com/sgl-project/sglang), Apache-2.0; modified for ATOM.
"""Sparse decode over the packed NVFP4 (MSA-v1) K/V cache for gfx950, in Gluon.

One program per (row, top-k chunk, kv head) with 4 warps; warp w owns tokens 32w..32w+31
of every selected block and keeps its own online softmax, merged across warps at the end,
so the block loop has no LDS traffic and no barrier. K and V go from global memory
straight into MFMA operands, which fixes how the MFMA indices map to head dims:
  * S = Q K^T: K-index 32s + 8a + e is dim 32a + 8s + e. A lane loads 16 payload bytes
    (dims 32a..32a+31) of a token and each dword dequantizes in place (cvt + scale mul).
  * O = P V: N-index 16e + g is dim 8g + e, so a lane needs payload dword g of 8 tokens;
    the conversion pairs two tokens per register (cvt + pack + scale mul).
  * P: S column 16n + 4b + i is PV K-index 8b + 4n + i, so the S result registers are
    the PV A operand as is.
e2m1 times E4M3 is exact in fp16, so the dequantized values are exact; outputs agree
with attention over the dequantized cache up to fp32 rounding.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import buffer_load, mfma

from .nvfp4 import Nvfp4Cache
from .nvfp4_gfx950 import fits_buffer_offsets
from .utils import robust_allocator

_LOG2E = 1.4426950408889634
_LN2 = gl.constexpr(0.6931471805599453)
_WARPS = 4
# measured on MI350X (M3.1 verify, 1-128 requests x 32k-150k); re-tune with the kernel
_TARGET_GRID = 256


@gluon.constexpr_function
def _dim_layout(layout, rank, d):
    for k in reversed(range(rank)):
        if k != d:
            layout = gl.SliceLayout(k, layout)
    return layout


@gluon.jit
def _idx(n: gl.constexpr, layout: gl.constexpr, rank: gl.constexpr, d: gl.constexpr):
    """arange(n) along dim d of a rank-``rank`` tensor in ``layout``, size 1 elsewhere."""
    x = gl.arange(0, n, layout=_dim_layout(layout, rank, d))
    for k in gl.static_range(rank):
        if k != d:
            x = gl.expand_dims(x, k)
    return x


@gluon.jit
def _k_dword_to_f16(dw, scale):
    # pack=8: eight copies of a payload dword (codes 8w..8w+7 of a row) and four of its
    # (scale, scale) pair -> those 8 values in order.
    return gl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_f16_fp4 $0, $4, 1.0\n\t"
        "v_cvt_scalef32_pk_f16_fp4 $1, $4, 1.0 op_sel:[1,0,0]\n\t"
        "v_cvt_scalef32_pk_f16_fp4 $2, $4, 1.0 op_sel:[0,1,0]\n\t"
        "v_cvt_scalef32_pk_f16_fp4 $3, $4, 1.0 op_sel:[1,1,0]\n\t"
        "v_pk_mul_f16 $0, $0, $12\n\t"
        "v_pk_mul_f16 $1, $1, $12\n\t"
        "v_pk_mul_f16 $2, $2, $12\n\t"
        "v_pk_mul_f16 $3, $3, $12",
        "=&v,=&v,=&v,=&v,v,v,v,v,v,v,v,v,v,v,v,v",
        [dw, scale],
        dtype=gl.float16,
        is_pure=True,
        pack=8,
    )


@gluon.jit
def _v_dwords_to_f16(dw, scale):
    # pack=2: payload dwords of two rows and their scale pair -> output e holds code e of
    # both rows (first row in the low half); the last two outputs are scratch.
    return gl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_f16_fp4 $8, $10, 1.0\n\t"
        "v_cvt_scalef32_pk_f16_fp4 $9, $11, 1.0\n\t"
        "v_pack_b32_f16 $0, $8, $9\n\t"
        "v_pack_b32_f16 $1, $8, $9 op_sel:[1,1,0]\n\t"
        "v_cvt_scalef32_pk_f16_fp4 $8, $10, 1.0 op_sel:[1,0,0]\n\t"
        "v_cvt_scalef32_pk_f16_fp4 $9, $11, 1.0 op_sel:[1,0,0]\n\t"
        "v_pack_b32_f16 $2, $8, $9\n\t"
        "v_pack_b32_f16 $3, $8, $9 op_sel:[1,1,0]\n\t"
        "v_cvt_scalef32_pk_f16_fp4 $8, $10, 1.0 op_sel:[0,1,0]\n\t"
        "v_cvt_scalef32_pk_f16_fp4 $9, $11, 1.0 op_sel:[0,1,0]\n\t"
        "v_pack_b32_f16 $4, $8, $9\n\t"
        "v_pack_b32_f16 $5, $8, $9 op_sel:[1,1,0]\n\t"
        "v_cvt_scalef32_pk_f16_fp4 $8, $10, 1.0 op_sel:[1,1,0]\n\t"
        "v_cvt_scalef32_pk_f16_fp4 $9, $11, 1.0 op_sel:[1,1,0]\n\t"
        "v_pack_b32_f16 $6, $8, $9\n\t"
        "v_pack_b32_f16 $7, $8, $9 op_sel:[1,1,0]\n\t"
        "v_pk_mul_f16 $0, $0, $12\n\t"
        "v_pk_mul_f16 $1, $1, $12\n\t"
        "v_pk_mul_f16 $2, $2, $12\n\t"
        "v_pk_mul_f16 $3, $3, $12\n\t"
        "v_pk_mul_f16 $4, $4, $12\n\t"
        "v_pk_mul_f16 $5, $5, $12\n\t"
        "v_pk_mul_f16 $6, $6, $12\n\t"
        "v_pk_mul_f16 $7, $7, $12",
        "=&v,=&v,=&v,=&v,=&v,=&v,=&v,=&v,=&v,=&v,v,v,v",
        [dw, scale],
        dtype=(gl.float16,) * 10,
        is_pure=True,
        pack=2,
    )


@gluon.jit
def _field(rec, ar, j: gl.constexpr):
    return gl.sum(gl.where(ar == j, rec, 0), axis=0)


@gluon.jit
def _block_table(
    block_table_ptr,
    idx_ptr,
    row_seq_len,
    row_req,
    pid_b,
    pid_kh,
    c_start,
    chunk,
    max_slots,
    max_topk,
    max_kv_len,
    stride_bt_b,
    stride_ti_h,
    stride_ti_b,
    pay_page,
    pay_head,
    sc_page,
    sc_head,
    BN: gl.constexpr,
    TOPK_P: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    PAY_ROW: gl.constexpr,
    SC_ROW: gl.constexpr,
    W: gl.constexpr,
):
    """LDS table of the row's top-k blocks, [TOPK_P] x (payload offset in 64-byte units,
    scale offset in 8-byte units, rows before seq_len), and the chunk's end entry."""
    t_l: gl.constexpr = gl.BlockedLayout([1], [64], [W], [0])
    seq_len = gl.minimum(gl.load(row_seq_len + pid_b), max_kv_len).to(gl.int32)
    sid = gl.load(row_req + pid_b).to(gl.int64)
    off_t = gl.arange(0, TOPK_P, layout=t_l)
    idx_base = idx_ptr + pid_kh * stride_ti_h + pid_b * stride_ti_b
    topk = gl.load(idx_base + off_t, mask=off_t < max_topk, other=-1)
    real_topk = gl.sum((topk >= 0).to(gl.int32), axis=0)
    # A zero-length row (padding) reads no block: its top-k may be uninitialized.
    real_topk = gl.where(seq_len > 0, real_topk, 0)
    c_end = gl.minimum(c_start + chunk, real_topk)
    in_chunk = (off_t >= c_start) & (off_t < c_end)
    c_tok = topk * BN
    # [reqs, max_blocks] table of 128-token blocks (BN == PAGE_SIZE): a selected
    # block's first slot is its id * 128.
    blk = gl.load(
        block_table_ptr + sid * stride_bt_b + topk,
        mask=in_chunk & (c_tok < seq_len),
        other=0,
    )
    base = (blk.to(gl.int64) * PAGE_SIZE % max_slots).to(gl.int32)
    page = base // PAGE_SIZE
    row0 = base % PAGE_SIZE
    # Coarse units so the loads see their alignment (16-byte K payload, 2-byte K scales).
    pay_off = (page * pay_page + pid_kh * pay_head) // 16 + row0 * (PAY_ROW // 64)
    sc_off = (page * sc_page + pid_kh * sc_head) // 8 + row0 * (SC_ROW // 8)
    nvalid = seq_len - c_tok
    # join(join(a, c), join(b, d)) flattens to [a, b, c, d].
    rec = gl.reshape(
        gl.join(gl.join(pay_off, nvalid), gl.join(sc_off, nvalid)), [TOPK_P, 4]
    )
    tab = gl.allocate_shared_memory(
        gl.int32, [TOPK_P, 4], gl.SwizzledSharedLayout(1, 1, 1, [1, 0]), rec
    )
    # Rank-1 layout so `.index(i)` picks entry i.
    tab = tab.reinterpret(gl.int32, [TOPK_P, 4], gl.SwizzledSharedLayout(1, 1, 1, [0]))
    return tab, c_end


@gluon.jit
def _load_tiles(
    ptrs,
    tab,
    i,
    c_end,
    offsets,
    W: gl.constexpr,
    PAY_ROW: gl.constexpr,
    SC_ROW: gl.constexpr,
):
    """Raw K/V tiles of table entry i; entries from c_end on reload the chunk's last
    block with no valid rows. The per-register offset parts (k_s, ks_s, v_row) are added
    last, so the loads take them as instruction offsets."""
    k_pay_ptr, k_sc_ptr, v_pay_ptr, v_sc_ptr = ptrs
    k_lane, k_s, ks_lane, ks_s, v_lane, vs_lane, v_tok_lane, v_row = offsets
    rec_l: gl.constexpr = gl.BlockedLayout([4], [64], [W], [0])
    rec = tab.index(gl.minimum(i, gl.maximum(c_end - 1, 0))).load(rec_l)
    ar = gl.arange(0, 4, layout=rec_l)
    pay_off = _field(rec, ar, 0) * 16
    sc_off = _field(rec, ar, 1) * 8
    nvalid = gl.where(i < c_end, _field(rec, ar, 2), 0)
    ky = buffer_load(ptr=k_pay_ptr, offsets=(k_lane + pay_off) + k_s)
    ks = buffer_load(ptr=k_sc_ptr, offsets=(ks_lane + sc_off) + ks_s)
    vz = buffer_load(ptr=v_pay_ptr, offsets=(v_lane + pay_off) + v_row * (PAY_ROW // 4))
    # Rows from seq_len on read scale 0, so whatever their codes hold adds 0 to P V.
    vs = buffer_load(
        ptr=v_sc_ptr,
        offsets=(vs_lane + sc_off) + v_row * SC_ROW,
        mask=v_row < nvalid - v_tok_lane,
    )
    return ky, ks, vz, vs, nvalid


@gluon.jit
def _k_operand(ky, ks, k5_l: gl.constexpr, k6_l: gl.constexpr, dot_b: gl.constexpr):
    """ky [W, s, a, tok] u32 and ks [W, s // 2, a, tok] u8 -> K^T [W, 128, tok] fp16."""
    W: gl.constexpr = ky.shape[0]
    TW: gl.constexpr = ky.shape[3]
    y = gl.broadcast(
        gl.expand_dims(ky, 3), gl.zeros([W, 4, 4, 8, TW], gl.uint32, k5_l)
    )[0]
    y = gl.convert_layout(gl.reshape(y, [W, 128, TW]), dot_b, assert_trivial=True)
    s = ks.to(gl.float8e4nv, bitcast=True).to(gl.float16)
    s5_l: gl.constexpr = gl.SliceLayout(4, k6_l)
    s = gl.broadcast(
        gl.expand_dims(s, 2), gl.zeros([W, 2, 2, 4, TW], gl.float16, s5_l)
    )[0]
    s = gl.broadcast(
        gl.expand_dims(s, 4), gl.zeros([W, 2, 2, 4, 8, TW], gl.float16, k6_l)
    )[0]
    s = gl.convert_layout(gl.reshape(s, [W, 128, TW]), dot_b, assert_trivial=True)
    return _k_dword_to_f16(y, s)


@gluon.jit
def _v_operand(vz, vs, dot_b: gl.constexpr):
    """vz [W, k // 8, k % 8, g] u32 (dword g of the row PV K-index k reads) and its
    scales -> V [W, k, 16e + g] fp16."""
    W: gl.constexpr = vz.shape[0]
    TW: gl.constexpr = vz.shape[1] * vz.shape[2]
    vz = gl.convert_layout(gl.reshape(vz, [W, TW, 16]), dot_b, assert_trivial=True)
    vs = gl.convert_layout(gl.reshape(vs, [W, TW, 16]), dot_b, assert_trivial=True)
    v0, v1, v2, v3, v4, v5, v6, v7, _, _ = _v_dwords_to_f16(
        vz, vs.to(gl.float8e4nv, bitcast=True).to(gl.float16)
    )
    v = gl.join(
        gl.join(gl.join(v0, v1), gl.join(v2, v3)),
        gl.join(gl.join(v4, v5), gl.join(v6, v7)),
    )
    v = gl.reshape(gl.permute(v, [0, 1, 5, 4, 3, 2]), [W, TW, 128])
    return gl.convert_layout(v, dot_b, assert_trivial=True)


@gluon.jit
def _attend(
    qa,
    tiles,
    state,
    s_tok,
    qk_scale,
    mfma_l: gl.constexpr,
    k5_l: gl.constexpr,
    k6_l: gl.constexpr,
):
    """Online-softmax update of each warp's (m_i, l_i, acc) with one block."""
    ky, ks, vz, vs, nvalid = tiles
    m_i, l_i, acc = state
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_l, k_width=8)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_l, k_width=8)
    W: gl.constexpr = qa.shape[0]
    H: gl.constexpr = qa.shape[1]
    TW: gl.constexpr = ky.shape[3]
    kb = _k_operand(ky, ks, k5_l, k6_l, dot_b)
    s = mfma(qa, kb, gl.zeros([W, H, TW], gl.float32, mfma_l))
    if nvalid < W * TW:
        s = gl.where(s_tok < nvalid, s, float("-inf"))
    # m_i is in log2 units; max(s) * qk_scale == max(s * qk_scale) as qk_scale > 0.
    m_new = gl.maximum(m_i, gl.max(s, axis=2) * qk_scale)
    p = gl.exp2(s * qk_scale - m_new[:, :, None])
    alpha = gl.exp2(m_i - m_new)
    l_i = l_i * alpha + gl.sum(p, axis=2)
    # S column 16n + 4b + i as PV K-index 8b + 4n + i: the same registers.
    pa = gl.reshape(p.to(gl.float16), [W, H, 2, 4, 4])
    pa = gl.reshape(gl.permute(pa, [0, 1, 3, 2, 4]), [W, H, TW])
    pa = gl.convert_layout(pa, dot_a, assert_trivial=True)
    acc = mfma(pa, _v_operand(vz, vs, dot_b), acc * alpha[:, :, None])
    return m_new, l_i, acc


@gluon.jit
def _store_merged(
    o_ptr,
    lse_ptr,
    state,
    o_base,
    lse_base,
    stride_o_h,
    stride_l_h,
    v_scale,
):
    """Merge the warps' softmax states and store O [H, D] (dims in order) and its lse."""
    m_i, l_i, acc = state
    W: gl.constexpr = acc.shape[0]
    H: gl.constexpr = acc.shape[1]
    D: gl.constexpr = acc.shape[2]
    m = gl.max(m_i, axis=0)
    f = gl.exp2(m_i - m[None, :])
    l_sum = gl.sum(l_i * f, axis=0)
    has = l_sum > 0
    f = f * (v_scale / gl.where(has, l_sum, 1.0))[None, :]
    o = gl.sum(acc * f[:, :, None], axis=0)
    # Column 16e + g holds dim 8g + e.
    o = gl.reshape(gl.permute(gl.reshape(o, [H, 8, 16]), [0, 2, 1]), [H, D])
    out_l: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [W, 1], [1, 0])
    o = gl.convert_layout(o.to(o_ptr.dtype.element_ty), out_l)
    oh = gl.arange(0, H, layout=gl.SliceLayout(1, out_l))
    od = gl.arange(0, D, layout=gl.SliceLayout(0, out_l))
    gl.store(o_ptr + o_base + oh[:, None] * stride_o_h + od[None, :], o)
    lse = gl.where(has, (m + gl.log2(gl.where(has, l_sum, 1.0))) * _LN2, float("-inf"))
    lh = gl.arange(0, H, layout=m.type.layout)
    gl.store(lse_ptr + lse_base + lh * stride_l_h, lse)


@gluon.jit
def _nvfp4_sparse_decode_kernel(
    q_ptr,  # [B, qh, D]
    k_pay_ptr,  # u32 view of the payload
    k_sc_ptr,
    v_pay_ptr,
    v_sc_ptr,
    block_table_ptr,  # [reqs, max_blocks]
    idx_ptr,  # [kh, B, topk]
    o_ptr,  # [C, B, qh, D]
    lse_ptr,  # [C, B, qh]
    row_seq_len,
    row_req,
    max_slots,
    batch_size,
    max_topk,
    max_kv_len,
    qk_scale,  # sm_scale * log2(e)
    v_scale,
    stride_q_b,
    stride_q_h,
    pay_page,  # u32 units
    pay_head,
    sc_page,  # bytes
    sc_head,
    stride_bt_b,
    stride_ti_h,
    stride_ti_b,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    H: gl.constexpr,
    D: gl.constexpr,
    BN: gl.constexpr,
    TOPK_P: gl.constexpr,
    NUM_TOPK_CHUNKS: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    PAY_ROW: gl.constexpr,  # bytes
    SC_ROW: gl.constexpr,
    num_warps: gl.constexpr,
):
    W: gl.constexpr = num_warps
    TW: gl.constexpr = BN // num_warps
    gl.static_assert(TW == 32 and D == 128 and H == 16)
    pid_bc = gl.program_id(0)
    pid_kh = gl.program_id(1)
    pid_b = pid_bc % batch_size
    pid_c = pid_bc // batch_size
    chunk = (max_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    c_start = pid_c * chunk

    mfma_l: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[W, 1, 1]
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_l, k_width=8)
    # K payload as [W, s, a, e, tok] and K scales as [W, s // 2, s % 2, a, e, tok]; merging
    # the middle dims gives the [W, 128, tok] QK B operand exactly.
    k5_l: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 0, 0, 1, 0], [0, 0, 0, 2, 0], [0, 0, 0, 4, 0], [0, 1, 0, 0, 0],
                   [0, 2, 0, 0, 0], [0, 0, 0, 0, 16]],
        lane_bases=[[0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 0, 0, 0, 4], [0, 0, 0, 0, 8],
                    [0, 0, 1, 0, 0], [0, 0, 2, 0, 0]],
        warp_bases=[[1, 0, 0, 0, 0], [2, 0, 0, 0, 0]],
        block_bases=[],
        shape=[W, 4, 4, 8, TW],
    )  # fmt: skip
    k6_l: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 4, 0],
                   [0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 16]],
        lane_bases=[[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 4],
                    [0, 0, 0, 0, 0, 8], [0, 0, 0, 1, 0, 0], [0, 0, 0, 2, 0, 0]],
        warp_bases=[[1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0]],
        block_bases=[],
        shape=[W, 2, 2, 4, 8, TW],
    )  # fmt: skip
    # V payload as [W, k // 8, k % 8, g]; merging dims 1-2 gives a [W, 32, 16] PV B operand.
    v4_l: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 4, 0]],
        lane_bases=[[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 4], [0, 0, 0, 8], [0, 1, 0, 0],
                    [0, 2, 0, 0]],
        warp_bases=[[1, 0, 0, 0], [2, 0, 0, 0]],
        block_bases=[],
        shape=[W, 4, 8, 16],
    )  # fmt: skip
    ky_l: gl.constexpr = gl.SliceLayout(3, k5_l)
    ks_l: gl.constexpr = gl.SliceLayout(2, gl.SliceLayout(4, k6_l))

    tab, c_end = _block_table(
        block_table_ptr, idx_ptr, row_seq_len, row_req, pid_b, pid_kh, c_start, chunk,
        max_slots, max_topk, max_kv_len, stride_bt_b, stride_ti_h, stride_ti_b,
        pay_page, pay_head, sc_page, sc_head, BN, TOPK_P, PAGE_SIZE, PAY_ROW, SC_ROW, W,
    )  # fmt: skip

    # Q in the K-index order 32s + 8a + e <-> dim 32a + 8s + e, the same for every warp.
    qk = _idx(D, dot_a, 3, 2)
    q_dim = 32 * ((qk // 8) % 4) + 8 * (qk // 32) + qk % 8
    q_off = (
        pid_b * stride_q_b + (pid_kh * H + _idx(H, dot_a, 3, 1)) * stride_q_h + q_dim
    )
    qa = buffer_load(ptr=q_ptr, offsets=q_off + 0 * _idx(W, dot_a, 3, 0)).to(gl.float16)

    # In-block offsets: K reads dword 4a + s of token 32w + t; V reads dword g of the
    # token at S column c = 4kb + j % 4 + 16 (j // 4) for PV K-index 8kb + j.
    k_tok = _idx(W, ky_l, 4, 0) * TW + _idx(TW, ky_l, 4, 3)
    k_lane = k_tok * (PAY_ROW // 4) + 4 * _idx(4, ky_l, 4, 2)
    k_s = _idx(4, ky_l, 4, 1)
    ks_tok = _idx(W, ks_l, 4, 0) * TW + _idx(TW, ks_l, 4, 3)
    ks_lane = ks_tok * SC_ROW + 2 * _idx(4, ks_l, 4, 2)
    ks_s = _idx(2, ks_l, 4, 1)
    v_tok_lane = _idx(W, v4_l, 4, 0) * TW + 4 * _idx(4, v4_l, 4, 1)
    vj = _idx(8, v4_l, 4, 2)
    v_row = vj % 4 + 16 * (vj // 4)
    vg = _idx(16, v4_l, 4, 3)
    v_lane = v_tok_lane * (PAY_ROW // 4) + vg
    vs_lane = v_tok_lane * SC_ROW + vg // 2
    s_tok = _idx(W, mfma_l, 3, 0) * TW + _idx(TW, mfma_l, 3, 2)

    ptrs = (k_pay_ptr, k_sc_ptr, v_pay_ptr, v_sc_ptr)
    offsets = (k_lane, k_s, ks_lane, ks_s, v_lane, vs_lane, v_tok_lane, v_row)
    m_i = gl.full([W, H], -1e30, gl.float32, gl.SliceLayout(2, mfma_l))
    l_i = gl.zeros([W, H], gl.float32, gl.SliceLayout(2, mfma_l))
    state = (m_i, l_i, gl.zeros([W, H, D], gl.float32, mfma_l))
    # Unrolled by two: block i + 1 loads at the top and block i + 2 after block i, so each
    # load has a block of compute to land, and no prefetched tile is register-copied (the
    # copy would wait on the load).
    tiles_a = _load_tiles(ptrs, tab, c_start, c_end, offsets, W, PAY_ROW, SC_ROW)
    for i in range(c_start, c_end, 2):
        tiles_b = _load_tiles(ptrs, tab, i + 1, c_end, offsets, W, PAY_ROW, SC_ROW)
        state = _attend(qa, tiles_a, state, s_tok, qk_scale, mfma_l, k5_l, k6_l)
        tiles_a = _load_tiles(ptrs, tab, i + 2, c_end, offsets, W, PAY_ROW, SC_ROW)
        state = _attend(qa, tiles_b, state, s_tok, qk_scale, mfma_l, k5_l, k6_l)

    o_base = pid_c * stride_o_c + pid_b * stride_o_b + pid_kh * H * stride_o_h
    lse_base = pid_c * stride_l_c + pid_b * stride_l_b + pid_kh * H * stride_l_h
    _store_merged(
        o_ptr, lse_ptr, state, o_base, lse_base, stride_o_h, stride_l_h, v_scale
    )


def can_use_nvfp4_decode_gfx950(
    q: torch.Tensor,
    k_cache: Nvfp4Cache,
    v_cache: Nvfp4Cache,
    topk_idx: torch.Tensor,
    block_size: int,
) -> bool:
    """M3.1 TP4 shapes over pool-layout caches that 32-bit buffer offsets can address."""
    caches_ok = all(
        cache.page_size % block_size == 0
        and cache.payload_strides()[2] == 64
        and cache.scale_strides()[2] == 8
        and cache.payload.stride(-1) == 1
        and cache.scales.stride(-1) == 1
        and all(s % 64 == 0 for s in cache.payload_strides()[:2])
        and all(s % 8 == 0 for s in cache.scale_strides()[:2])
        and cache.payload.data_ptr() % 16 == 0
        and cache.scales.data_ptr() % 2 == 0
        and fits_buffer_offsets(cache)
        for cache in (k_cache, v_cache)
    )
    return (
        caches_ok
        and k_cache.payload.stride() == v_cache.payload.stride()
        and k_cache.scales.stride() == v_cache.scales.stride()
        and q.dtype in (torch.bfloat16, torch.float16)
        and q.shape[-1] == 128
        and q.stride(-1) == 1
        and q.shape[1] == 16 * k_cache.num_heads
        and block_size == 128
        and topk_idx.stride(-1) == 1
    )


@torch.no_grad()
def flash_decode_nvfp4_gfx950(
    q: torch.Tensor,  # [rows, num_q_heads, 128]
    k_cache: Nvfp4Cache,
    v_cache: Nvfp4Cache,
    block_table: torch.Tensor,  # [reqs, max_blocks] 128-token block ids
    row_seq_len: torch.Tensor,  # [rows] causal length each row attends to
    row_req: torch.Tensor,  # [rows] request (block-table row) of each row
    block_size: int,
    topk_idx: torch.Tensor,  # [num_kv_heads, rows, topk] block ids, -1 padded
    sm_scale: float,
) -> torch.Tensor:
    """Sparse attention of every row over its top-k blocks; callers check
    `can_use_nvfp4_decode_gfx950` first. Returns O [rows, num_q_heads, 128]."""
    from .merge import _merge_topk_attn_out_kernel

    triton.set_allocator(robust_allocator)
    batch_size, num_q_heads, head_dim = q.shape
    page_size = k_cache.page_size
    num_kv_heads = k_cache.num_heads
    max_topk = topk_idx.shape[2]
    target = max(1, min(max_topk, _TARGET_GRID // max(1, batch_size * num_kv_heads)))
    num_topk_chunks = 1 << (target.bit_length() - 1)
    o_partial = torch.empty(
        num_topk_chunks,
        batch_size,
        num_q_heads,
        head_dim,
        dtype=q.dtype,
        device=q.device,
    )
    lse_partial = torch.empty(
        num_topk_chunks, batch_size, num_q_heads, dtype=torch.float32, device=q.device
    )
    pay_page, pay_head, pay_row = k_cache.payload_strides()
    sc_page, sc_head, sc_row = k_cache.scale_strides()
    _nvfp4_sparse_decode_kernel[(batch_size * num_topk_chunks, num_kv_heads)](
        q,
        k_cache.payload.view(torch.uint32),
        k_cache.scales,
        v_cache.payload.view(torch.uint32),
        v_cache.scales,
        block_table,
        topk_idx,
        o_partial,
        lse_partial,
        row_seq_len,
        row_req,
        k_cache.payload.shape[0] * page_size,
        batch_size,
        max_topk,
        block_table.shape[1] * page_size,
        sm_scale * _LOG2E,
        1.0,
        q.stride(0),
        q.stride(1),
        pay_page // 4,
        pay_head // 4,
        sc_page,
        sc_head,
        block_table.stride(0),
        topk_idx.stride(0),
        topk_idx.stride(1),
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        H=num_q_heads // num_kv_heads,
        D=head_dim,
        BN=block_size,
        TOPK_P=triton.next_power_of_2(max_topk),
        NUM_TOPK_CHUNKS=num_topk_chunks,
        PAGE_SIZE=page_size,
        PAY_ROW=pay_row,
        SC_ROW=sc_row,
        num_warps=_WARPS,
    )
    if num_topk_chunks == 1:
        return o_partial[0]
    _merge_topk_attn_out_kernel[(batch_size, num_q_heads)](
        o_partial,
        lse_partial,
        head_dim,
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        NUM_TOPK_CHUNKS=num_topk_chunks,
    )
    return o_partial[0]
