// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// NVFP4 MLA decode prep for gfx1250, byte for byte what `_decode_prep_kernel`
// in nvfp4_q_triton.py writes with NOPE_DONE, N16, KSPLIT 1 and RING 16:
//
// Q blocks (2 per batch row, 64 heads each): the RoPE part of q_aux. A head's
//   64 RoPE channels are read by 8 lanes, 16 contiguous bytes each; a lane
//   does two heads 32 apart. q_aux row: 0..31 QR codes, 32..63 QE codes,
//   64..95 the NoPE scales (left as written), 96..99 QR scales, 100..103 QE
//   scales, 104.. 0x7F.
// K blocks (1 per token): the new token into its tail row, its 16-token tile
//   requantized into the page (NoPE: one scale per 16 x 16 block, codes over
//   amax / 6), and its RoPE part (per token, plus a residual) into the AUX
//   tile. A thread owns one tile row's 32 channels (two blocks); the 16 rows
//   of a block are 16 lanes of a wave.
// With TDM (K blocks first or last, one per token, q_rope rows 128 head
//   strides apart): a K wave's 64 channels of the tail rows and a Q wave's 8
//   heads of q_rope come into LDS as one tensor load each (opus::tdm), so a
//   wave's rows are one request instead of a load per lane.
//
// RoPE is the RoPE kernel's v_pk_mul_bf16 / v_pk_fma_bf16. Quantization is
// `_fp4`: x / div correctly rounded (Markstein, exact for these operands),
// then the hardware FP32 -> E2M1 conversion, which matches `_e2m1` for every
// input; dequantized values come back through the hardware E2M1 -> FP32
// conversion. Build with -ffp-contract=off.

#include <cstdint>
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "opus/opus.hpp"

namespace {

typedef float v8f __attribute__((ext_vector_type(8)));

struct QGain {
    static constexpr float g = 6.72f, r = 1.0f / 6.72f;
};
struct KvGain {
    static constexpr float g = 89.6f, r = 1.0f / 89.6f;
};
constexpr int kR = 2;  // Q heads per lane
constexpr int kQBlocksPerRow = 2;  // 128 heads / (32 * kR)

__device__ __forceinline__ float as_f(uint32_t b) { return __uint_as_float(b); }
__device__ __forceinline__ uint32_t as_u(float f) { return __float_as_uint(f); }
__device__ __forceinline__ float bf(uint32_t h) { return as_f(h << 16); }

// `_e4m3`: v in [2**-9, 448], nearest even. The hardware conversion equals it
// on exactly that range (no saturation or underflow cases).
__device__ __forceinline__ float e4m3(float v, uint32_t& code) {
    const int pk = __builtin_amdgcn_cvt_pk_fp8_f32(v, v, 0, false);
    code = uint32_t(pk) & 0xFF;
    return __builtin_amdgcn_cvt_f32_fp8(pk, 0);
}

// RN(a / gain) for E4M3 scales and E2M1 x E4M3 products.
template <class Gn>
__device__ __forceinline__ float div_gain(float a) {
    const float q = a * Gn::r;
    return __fmaf_rn(__fmaf_rn(-Gn::g, q, a), Gn::r, q);
}

__device__ __forceinline__ uint32_t pk_mul_bf16(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("v_pk_mul_bf16 %0, %1, %2" : "=v"(d) : "v"(a), "v"(b));
    return d;
}

__device__ __forceinline__ uint32_t pk_fma_bf16(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("v_pk_fma_bf16 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
    return d;
}

// RN(amax / 6): a multiply and an FMA correction (Markstein), exact for every
// FP32 significand; the division itself where the correction could underflow.
// EXACT_TINY = false: below 1e-30 only some value below 1e-30, enough for a
// scale that clamps to 2**-9 there.
template <bool EXACT_TINY = true>
__device__ __forceinline__ float div6(float a) {
    constexpr float r6 = 1.0f / 6.0f;
    if (EXACT_TINY && a < 1e-30f) return __fdiv_rn(a, 6.0f);
    const float q = a * r6;
    return __fmaf_rn(__fmaf_rn(-6.0f, q, a), r6, q);
}

// E4M3 code -> value (finite, positive codes).
__device__ __forceinline__ float e4m3_value(uint32_t c) {
    const uint32_t e = c >> 3, m = c & 7;
    return e == 0 ? float(m) * (1.0f / 512) : float(8 + m) * as_f(uint32_t(e - 10 + 127) << 23);
}

// One 16-value block's quantizer: `_fp4(ideal=IDEAL)` at the gain. With
// ``rtab`` (non-ideal only), RN(1 / div) is looked up by the scale code; with
// RCP it is v_rcp_f32 and one Newton step, equal to RN(1 / div) for every
// divisor these blocks can produce (checked exhaustively) except the ideal
// ones below 1e-30, whose reciprocal overflows and keeps the division.
template <class Gn, bool IDEAL, bool RCP = false>
struct Block {
    float scale, div, rdiv;
    uint32_t scode;
    __device__ __forceinline__ explicit Block(float amax, const float* rtab = nullptr) {
        // Ideal blocks divide by d0 itself; the others only round it into a scale.
        const float d0 = amax == 0.0f ? 1.0f : div6<IDEAL>(amax);
        scale = e4m3(fminf(fmaxf(d0 * Gn::g, 1.0f / 512), 448.0f), scode);
        div = IDEAL ? d0 : div_gain<Gn>(scale);
        if (RCP && (!IDEAL || div >= 1e-30f)) {
            const float r0 = __builtin_amdgcn_rcpf(div);
            rdiv = __fmaf_rn(__fmaf_rn(div, -r0, 1.0f), r0, r0);
        } else {
            rdiv = (!IDEAL && rtab) ? rtab[scode] : __fdiv_rn(1.0f, div);
        }
    }
    // 8 codes, low nibble first: x / div correctly rounded (signed, so -0
    // stays -0), then FP32 -> E2M1 at scale 2**0.
    __device__ __forceinline__ uint32_t codes(const float* x) const {
        v8f q;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float q0 = x[i] * rdiv;
            q[i] = __fmaf_rn(-__fmaf_rn(div, q0, -x[i]), rdiv, q0);
        }
        return __builtin_amdgcn_cvt_scalef32_pk8_fp4_f32(q, 1.0f);
    }
    // The signed dequantized values of 8 codes.
    __device__ __forceinline__ void deq(uint32_t c, float* out) const {
        // E8M0 scale 2**0 in every byte: the lane halves read different bytes.
        const v8f qv = __builtin_amdgcn_cvt_scale_pk8_f32_fp4(c, 0x7F7F7F7Fu, 0);
#pragma unroll
        for (int i = 0; i < 8; ++i) out[i] = copysignf(div_gain<Gn>(fabsf(qv[i]) * scale), qv[i]);
    }
};

// 8 RoPE channels (4 GPT-J pairs, bf16 words) at cos/sin index 4 j + p:
// bf16(x * cos + bf16(rot * sin)), as the RoPE kernel computes them.
__device__ __forceinline__ void rope8(uint32_t (&w)[4], uint2 cs, uint2 sn) {
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint32_t cw = p < 2 ? cs.x : cs.y, sw = p < 2 ? sn.x : sn.y;
        const uint32_t c = ((p & 1) ? cw >> 16 : cw & 0xFFFFu) * 0x10001u;
        const uint32_t s = ((p & 1) ? sw >> 16 : sw & 0xFFFFu) * 0x10001u;
        const uint32_t rot = ((w[p] >> 16) | (w[p] << 16)) ^ 0x8000u;  // (-odd, even)
        w[p] = pk_fma_bf16(w[p], c, pk_mul_bf16(rot, s));
    }
}

__device__ __forceinline__ void unpack8(const uint32_t (&w)[4], float* x) {
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        x[2 * p] = bf(w[p] & 0xFFFFu);
        x[2 * p + 1] = bf(w[p] >> 16);
    }
}

__device__ __forceinline__ float pair_max(float v) { return fmaxf(v, __shfl_xor(v, 1)); }

// v from lane i ^ MASK of the same 16-lane row (DPP row_xmask).
template <int MASK>
__device__ __forceinline__ float row_xor(float v) {
    return as_f(uint32_t(__builtin_amdgcn_update_dpp(0, int(as_u(v)), 0x160 + MASK, 0xF, 0xF, false)));
}

// Lanes j < 8 hold 8 channels of a 64-channel RoPE vector (blocks are lane
// pairs): base and residual codes (4 bytes each per lane), and the packed
// scale words for lanes 0 / 1 of the group starting at lane g: four base
// scales, four residual scales.
template <class Gn, bool RCP = false>
__device__ __forceinline__ void rope_quant(const float (&x)[8], int g, int j, uint32_t& bc, uint32_t& ec,
                                           uint32_t& word, const float* rtab = nullptr) {
    float m = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) m = fmaxf(m, fabsf(x[i]));
    const Block<Gn, false, RCP> base(pair_max(m), rtab);
    bc = base.codes(x);
    float res[8];
    base.deq(bc, res);
    float me = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        res[i] = x[i] - res[i];
        me = fmaxf(me, fabsf(res[i]));
    }
    const Block<Gn, false, RCP> resq(pair_max(me), rtab);
    ec = resq.codes(res);
    const uint32_t sc = base.scode | (resq.scode << 8);
    const uint32_t k0 = __shfl(sc, g), k1 = __shfl(sc, g + 2);
    const uint32_t k2 = __shfl(sc, g + 4), k3 = __shfl(sc, g + 6);
    const int sh = j == 1 ? 8 : 0;
    word = ((k0 >> sh) & 0xFF) | (((k1 >> sh) & 0xFF) << 8) | (((k2 >> sh) & 0xFF) << 16) |
           (((k3 >> sh) & 0xFF) << 24);
}

struct Args {
    const uint16_t* q_rope;
    uint8_t* qa;
    int stride_qrb, stride_qrh, q_blocks, k_blocks;
    uint16_t* ring;
    const uint16_t* k_nope;
    const uint16_t* k_rope;
    const int32_t* req;
    uint8_t* kv_nope;
    uint8_t* kv_aux;
    int stride_req, stride_slot, stride_knope, stride_krope;
    const int64_t* pos;
    const int64_t* slot;
    const uint16_t* cos;
    const uint16_t* sin;
    int stride_cos_t;
    int order;  // block order: 0 Q then K, 1 K then Q, 2 Q, Q, K interleaved
    int k_split;  // K blocks per token, 1 or 2
    int q_rows;  // batch rows (the TDM q_rope window: rows x 128 heads)
};

template <bool FUSED>
__device__ __forceinline__ void cos_sin(const Args& a, int64_t p, int j, uint2& cs, uint2& sn) {
    if (FUSED) {
        const int64_t t = p * a.stride_cos_t + 4 * j;
        cs = *reinterpret_cast<const uint2*>(a.cos + t);
        sn = *reinterpret_cast<const uint2*>(a.sin + t);
    }
}

template <bool FUSED, bool RCP>
__device__ __forceinline__ void q_block(const Args& a, int qb, const float* rtab) {
    const int b = qb / kQBlocksPerRow;
    const int lane = threadIdx.x;
    const int j = lane & 7;
    const int h0 = (qb % kQBlocksPerRow) * (32 * kR) + (lane >> 3);
    uint8_t* out = a.qa + (int64_t(b) * 128 + h0) * 128;
    constexpr int kHeadStep = 32 * 128;  // bytes between this lane's heads

    if (a.slot[b] < 0) {
        // A padding row: the shader never reads its Q; zero codes and scales.
        const uint32_t pad = j < 2 ? 0u : 0x7F7F7F7Fu;
#pragma unroll
        for (int r = 0; r < kR; ++r) {
            *reinterpret_cast<uint2*>(out + r * kHeadStep + 8 * j) = make_uint2(0, 0);
            *reinterpret_cast<uint32_t*>(out + r * kHeadStep + 96 + 4 * j) = pad;
        }
        return;
    }
    uint4 raw[kR];
    const uint16_t* src = a.q_rope + int64_t(b) * a.stride_qrb + int64_t(h0) * a.stride_qrh + 8 * j;
#pragma unroll
    for (int r = 0; r < kR; ++r) raw[r] = *reinterpret_cast<const uint4*>(src + int64_t(r) * 32 * a.stride_qrh);
    uint2 cs = make_uint2(0, 0), sn = make_uint2(0, 0);
    cos_sin<FUSED>(a, a.pos[b], j, cs, sn);
#pragma unroll
    for (int r = 0; r < kR; ++r) {
        uint32_t w[4] = {raw[r].x, raw[r].y, raw[r].z, raw[r].w};
        if (FUSED) rope8(w, cs, sn);
        float x[8];
        unpack8(w, x);
        const int g = lane & ~7;
        uint32_t bc, ec, word;
        rope_quant<QGain, RCP>(x, g, j, bc, ec, word, RCP ? nullptr : rtab);
        // Bytes 0..63: lane j writes 8 at 8 j, the codes of lanes 2j, 2j + 1
        // (QR for j < 4, QE after).
        const int s0 = g + 2 * (j & 3);
        const uint32_t b_lo = __shfl(bc, s0), b_hi = __shfl(bc, s0 + 1);
        const uint32_t e_lo = __shfl(ec, s0), e_hi = __shfl(ec, s0 + 1);
        *reinterpret_cast<uint2*>(out + r * kHeadStep + 8 * j) =
            j < 4 ? make_uint2(b_lo, b_hi) : make_uint2(e_lo, e_hi);
        *reinterpret_cast<uint32_t*>(out + r * kHeadStep + 96 + 4 * j) = j < 2 ? word : 0x7F7F7F7Fu;
    }
}

constexpr int kQtHeads = 8;  // a TDM Q wave's heads
constexpr int kTdmLds = 8 * 16 * 64;  // K: 8 waves x [16 rows, 64]; Q: 8 x [8 heads, 64]
#if defined(__gfx1250__)
using QtWin = opus::tdm<uint16_t, opus::seq<64, kQtHeads>>;
using KtWin = opus::tdm<uint16_t, opus::seq<64, 16>>;

// q_block with q_rope by TDM: the same grid, wave w's 8 heads as one [8, 64]
// tile of the [rows x 128, 64] q_rope window, issued before the padding test.
// The block leaves only after the load landed (the next block reuses the LDS).
template <bool FUSED, bool RCP>
__device__ __forceinline__ void q_block_tdm(const Args& a, int qb, uint16_t* lds) {
    const int lane = threadIdx.x & 31;
    const int wave = __builtin_amdgcn_readfirstlane(int(threadIdx.x >> 5));
    const int j = lane & 7, hl = lane >> 3;  // 8 lanes a head; heads hl and hl + 4 of the wave's 8
    const int b = qb / kQBlocksPerRow;
    const int h8 = (qb % kQBlocksPerRow) * 64 + wave * kQtHeads;
    uint16_t* tile = lds + wave * kQtHeads * 64;
    QtWin win = opus::make_tdm<QtWin>(uint32_t(reinterpret_cast<uintptr_t>(tile)), a.q_rope, 64u,
                                      uint32_t(a.q_rows) * 128u, uint64_t(a.stride_qrh), 0u,
                                      uint32_t(b) * 128u + uint32_t(h8));
    win.async_load(0u);
    uint8_t* out = a.qa + (int64_t(b) * 128 + h8 + hl) * 128;
    constexpr int kHeadStep = 4 * 128;  // heads hl and hl + 4
    if (a.slot[b] < 0) {
        const uint32_t pad = j < 2 ? 0u : 0x7F7F7F7Fu;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            *reinterpret_cast<uint2*>(out + r * kHeadStep + 8 * j) = make_uint2(0, 0);
            *reinterpret_cast<uint32_t*>(out + r * kHeadStep + 96 + 4 * j) = pad;
        }
        opus::s_wait_tensorcnt<0>();
        return;
    }
    uint2 cs = make_uint2(0, 0), sn = make_uint2(0, 0);
    cos_sin<FUSED>(a, a.pos[b], j, cs, sn);
    opus::s_wait_tensorcnt<0>();
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const uint4 raw = *reinterpret_cast<const uint4*>(tile + (hl + 4 * r) * 64 + 8 * j);
        uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
        if (FUSED) rope8(w, cs, sn);
        float x[8];
        unpack8(w, x);
        const int g = lane & ~7;
        uint32_t bc, ec, word;
        rope_quant<QGain, RCP>(x, g, j, bc, ec, word);
        const int s0 = g + 2 * (j & 3);
        const uint32_t b_lo = __shfl(bc, s0), b_hi = __shfl(bc, s0 + 1);
        const uint32_t e_lo = __shfl(ec, s0), e_hi = __shfl(ec, s0 + 1);
        *reinterpret_cast<uint2*>(out + r * kHeadStep + 8 * j) =
            j < 4 ? make_uint2(b_lo, b_hi) : make_uint2(e_lo, e_hi);
        *reinterpret_cast<uint32_t*>(out + r * kHeadStep + 96 + 4 * j) = j < 2 ? word : 0x7F7F7F7Fu;
    }
}
#endif

// SPLIT blocks per token: block `half` of them does its 512 / SPLIT NoPE
// channels; the last also the RoPE part and a new tile's padding.
// TDM (SPLIT 1): wave w's 64 channels of rows 0 .. rn - 1 of the ring as one
// [16, 64] tile (rows from rn on clamped away), read back from LDS.
template <bool FUSED, int SPLIT, bool RCP, bool TDM = false>
__device__ __forceinline__ void k_block(const Args& a, int n, int half, uint16_t* lds = nullptr) {
    static_assert(!TDM || SPLIT == 1, "TDM K blocks take a whole token");
    // NoPE: thread t has row r = t % 16 and NB 16-channel blocks from c0.
    constexpr int NB = 2 / SPLIT;
    const int t = threadIdx.x;
    const int r = t & 15;
    const int c0 = (half * 16 + (t >> 4)) * 16 * NB;
    // Read with the slot, before the padding test (every row of k_nope /
    // k_rope exists, padding rows included): the new row's NoPE slice, by
    // every lane of its channel group, and its RoPE part.
    const int64_t slot = a.slot[n];
    const int64_t p = a.pos[n];
    const int64_t req = a.req[n];
    uint4 kn[2 * NB];
    const uint4* ks = reinterpret_cast<const uint4*>(a.k_nope + int64_t(n) * a.stride_knope + c0);
#pragma unroll
    for (int i = 0; i < 2 * NB; ++i) kn[i] = ks[i];
    uint4 kr = make_uint4(0, 0, 0, 0);
    if (half == SPLIT - 1 && t < 8)
        kr = *reinterpret_cast<const uint4*>(a.k_rope + int64_t(n) * a.stride_krope + 8 * t);
    // A padding row has nothing to keep: no tail write, no tile.
    if (slot < 0) return;
    const int rn = int(p & 15);  // the new token's tile row; valid = rn + 1
    const int tile = int((p & 63) >> 4);
    uint16_t* ring_row0 = a.ring + req * a.stride_req;
    uint8_t* nope_tile = a.kv_nope + (slot >> 6) * 16384 + tile * 4096;
    uint8_t* aux_tile = a.kv_aux + (slot >> 6) * 8192 + tile * 2048;

    {
        uint4 v[2 * NB] = {};
#if defined(__gfx1250__)
        const uint16_t* ktile = nullptr;
        if constexpr (TDM) {
            const int wave = __builtin_amdgcn_readfirstlane(int(t >> 5));
            uint16_t* tile = lds + wave * 16 * 64;
            KtWin win = opus::make_tdm<KtWin>(uint32_t(reinterpret_cast<uintptr_t>(tile)), ring_row0 + wave * 64, 64u,
                                              uint32_t(rn), uint64_t(a.stride_slot), 0u, 0u);
            win.async_load(0u);
            opus::s_wait_tensorcnt<0>();
            ktile = tile + r * 64 + ((t >> 4) & 1) * 32;
        }
#endif
        if (r < rn) {
#if defined(__gfx1250__)
            if constexpr (TDM) {
#pragma unroll
                for (int i = 0; i < 2 * NB; ++i) v[i] = *reinterpret_cast<const uint4*>(ktile + 8 * i);
            } else
#endif
            {
                const uint4* s = reinterpret_cast<const uint4*>(ring_row0 + int64_t(r) * a.stride_slot + c0);
#pragma unroll
                for (int i = 0; i < 2 * NB; ++i) v[i] = s[i];
            }
        } else if (r == rn) {
            uint4* d = reinterpret_cast<uint4*>(ring_row0 + int64_t(rn) * a.stride_slot + c0);
#pragma unroll
            for (int i = 0; i < 2 * NB; ++i) {
                v[i] = kn[i];
                d[i] = v[i];
            }
        }
        float x[16 * NB];
#pragma unroll
        for (int i = 0; i < 2 * NB; ++i) {
            const uint32_t w[4] = {v[i].x, v[i].y, v[i].z, v[i].w};
            unpack8(w, x + 8 * i);
        }
        uint32_t c[2 * NB], sc[NB];
#pragma unroll
        for (int k = 0; k < NB; ++k) {
            float m = 0.0f;
#pragma unroll
            for (int i = 0; i < 16; ++i) m = fmaxf(m, fabsf(x[16 * k + i]));
            // Over the 16 rows: the 16 lanes of a DPP row, all active.
            m = fmaxf(m, row_xor<1>(m));
            m = fmaxf(m, row_xor<2>(m));
            m = fmaxf(m, row_xor<4>(m));
            m = fmaxf(m, row_xor<8>(m));
            const Block<KvGain, true, RCP> blk(m);
            c[2 * k] = blk.codes(x + 16 * k);
            c[2 * k + 1] = blk.codes(x + 16 * k + 8);
            sc[k] = r <= rn ? blk.scode : 127u;
        }
        // 16-token x 32-channel bricks: channel group, token, byte. Two
        // 16 x 16-byte scale bricks: k * 256 + row * 16 + s for index k * 16 + s.
        const int b0 = c0 / 2, s0 = c0 / 16;
        uint8_t* codes = nope_tile + (b0 >> 4) * 256 + r * 16 + (b0 & 15);
        uint8_t* scales = aux_tile + 1024 + (s0 >> 4) * 256 + r * 16 + (s0 & 15);
        if (NB == 2) {
            *reinterpret_cast<uint4*>(codes) = make_uint4(c[0], c[1], c[2 % (2 * NB)], c[3 % (2 * NB)]);
            *reinterpret_cast<uint16_t*>(scales) = uint16_t(sc[0] | (sc[1 % NB] << 8));
        } else {
            *reinterpret_cast<uint2*>(codes) = make_uint2(c[0], c[1]);
            *scales = uint8_t(sc[0]);
        }
    }
    if (half != SPLIT - 1) return;

    // RoPE: lanes 0..7, the new row only (per-token scales).
    if (t < 8) {
        const int j = t;
        uint32_t w[4] = {kr.x, kr.y, kr.z, kr.w};
        if (FUSED) {
            uint2 cs, sn;
            cos_sin<FUSED>(a, p, j, cs, sn);
            rope8(w, cs, sn);
        }
        *reinterpret_cast<uint4*>(ring_row0 + int64_t(rn) * a.stride_slot + 512 + 8 * j) =
            make_uint4(w[0], w[1], w[2], w[3]);
        float x[8];
        unpack8(w, x);
        uint32_t bc, ec, word;
        rope_quant<KvGain, RCP>(x, 0, j, bc, ec, word);
        // Codes 0 / 512: two 16 x 16-byte bricks each, row rn; scales at 1536 + 16 rn.
        const int off = (j >> 2) * 256 + rn * 16 + (j & 3) * 4;
        *reinterpret_cast<uint32_t*>(aux_tile + off) = bc;
        *reinterpret_cast<uint32_t*>(aux_tile + 512 + off) = ec;
        if (j < 2) *reinterpret_cast<uint32_t*>(aux_tile + 1536 + rn * 16 + 4 * j) = word;
    }
    // A tile started here: its later rows and the padding as the producer
    // leaves them (zero codes, scale 0x7F).
    if (rn == 0) {
        const uint4 z = make_uint4(0, 0, 0, 0), f = make_uint4(~0u / 255 * 127, ~0u / 255 * 127,
                                                              ~0u / 255 * 127, ~0u / 255 * 127);
        if (t < 60) {  // base / residual code bricks (0, 256, 512, 768), rows 1..15
            *reinterpret_cast<uint4*>(aux_tile + (t / 15) * 256 + (1 + t % 15) * 16) = z;
        } else if (t < 75) {  // rows 1..15 of the scale rows: scales and padding
            *reinterpret_cast<uint4*>(aux_tile + 1536 + (t - 59) * 16) = f;
        } else if (t == 75) {  // row 0's padding
            *reinterpret_cast<uint2*>(aux_tile + 1536 + 8) = make_uint2(f.x, f.y);
        } else if (t < 92) {  // 1792 .. 2047
            *reinterpret_cast<uint4*>(aux_tile + 1792 + (t - 76) * 16) = f;
        }
    }
}

template <bool FUSED, bool RCP, bool TDM = false>
__global__ __launch_bounds__(256) void decode_prep_kernel(Args a) {
    const int bid = blockIdx.x;
#if defined(__gfx1250__)
    if constexpr (TDM) {
        __shared__ uint16_t tdm_lds[kTdmLds];
        const int kb = a.order == 1 ? bid : bid - a.q_blocks;
        if (kb >= 0 && kb < a.k_blocks) {
            k_block<FUSED, 1, RCP, true>(a, kb, 0, tdm_lds);
            return;
        }
        q_block_tdm<FUSED, RCP>(a, a.order == 1 ? bid - a.k_blocks : bid, tdm_lds);
        return;
    }
#endif
    if (a.order == 2 && a.k_blocks) {
        // Each batch row's two Q blocks, then its K block.
        const int row = bid / 3, part = bid - 3 * row;
        if (part == kQBlocksPerRow) {
            k_block<FUSED, 1, RCP>(a, row, 0);
            return;
        }
        __shared__ float rtab2[128];
        if (!RCP) {
            if (threadIdx.x < 128)
                rtab2[threadIdx.x] =
                    threadIdx.x == 0 ? 0.0f : __fdiv_rn(1.0f, div_gain<QGain>(e4m3_value(threadIdx.x)));
            __syncthreads();
        }
        q_block<FUSED, RCP>(a, row * kQBlocksPerRow + part, rtab2);
        return;
    }
    const int kb = a.order == 1 ? bid : bid - a.q_blocks;
    if (kb >= 0 && kb < a.k_blocks) {
        if (a.k_split == 2)
            k_block<FUSED, 2, RCP>(a, kb >> 1, kb & 1);
        else
            k_block<FUSED, 1, RCP>(a, kb, 0);
        return;
    }
    // RN(1 / RN(scale / 6.72)) for each E4M3 scale code a Q block can produce.
    __shared__ float rtab[128];
    if (!RCP) {
        if (threadIdx.x < 128)
            rtab[threadIdx.x] = threadIdx.x == 0 ? 0.0f : __fdiv_rn(1.0f, div_gain<QGain>(e4m3_value(threadIdx.x)));
        __syncthreads();
    }
    q_block<FUSED, RCP>(a, a.order == 1 ? bid - a.k_blocks : bid, rtab);
}

void check_rope_cache(const torch::Tensor& cos, const torch::Tensor& sin) {
    TORCH_CHECK(cos.scalar_type() == torch::kBFloat16 && sin.scalar_type() == torch::kBFloat16 &&
                    cos.stride(-1) == 1 && cos.strides() == sin.strides() && (cos.stride(0) % 4) == 0 &&
                    (reinterpret_cast<uintptr_t>(cos.data_ptr()) % 8) == 0 &&
                    (reinterpret_cast<uintptr_t>(sin.data_ptr()) % 8) == 0,
                "fused RoPE needs BF16 cos/sin caches with 8-byte aligned rows");
}

bool aligned16(const torch::Tensor& t) { return (reinterpret_cast<uintptr_t>(t.data_ptr()) % 16) == 0; }

template <bool TDM>
void launch_t(const Args& a, int blocks, bool fused, bool rcp, hipStream_t stream) {
    if (fused && rcp)
        decode_prep_kernel<true, true, TDM><<<blocks, 256, 0, stream>>>(a);
    else if (fused)
        decode_prep_kernel<true, false, TDM><<<blocks, 256, 0, stream>>>(a);
    else if (rcp)
        decode_prep_kernel<false, true, TDM><<<blocks, 256, 0, stream>>>(a);
    else
        decode_prep_kernel<false, false, TDM><<<blocks, 256, 0, stream>>>(a);
}

void launch(const Args& a, int blocks, bool fused, bool rcp = false, bool tdm = false) {
    if (!blocks) return;
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (tdm)
        launch_t<true>(a, blocks, fused, rcp, stream);
    else
        launch_t<false>(a, blocks, fused, rcp, stream);
}

Args q_args(const torch::Tensor& q_rope, const torch::Tensor& qa, const torch::Tensor& pos,
            const torch::Tensor& slot, const torch::Tensor& cos, const torch::Tensor& sin, bool fused) {
    const int batch = q_rope.size(0);
    TORCH_CHECK(q_rope.scalar_type() == torch::kBFloat16 && q_rope.size(1) == 128 && q_rope.size(2) == 64 &&
                    q_rope.stride(2) == 1 && q_rope.stride(0) % 8 == 0 && q_rope.stride(1) % 8 == 0 &&
                    aligned16(q_rope), "q_rope must be BF16 [B, 128, 64], 16-byte aligned rows");
    TORCH_CHECK(qa.scalar_type() == torch::kUInt8 && qa.is_contiguous() && qa.size(0) == batch &&
                    qa.size(1) == 128 && qa.size(2) == 128, "q_aux must be uint8 [B, 128, 128], contiguous");
    TORCH_CHECK(pos.scalar_type() == torch::kInt64 && slot.scalar_type() == torch::kInt64 && pos.is_contiguous() &&
                    slot.is_contiguous() && pos.numel() >= batch && slot.numel() >= batch,
                "positions and slots must be contiguous int64 [B]");
    if (fused) check_rope_cache(cos, sin);
    Args a{};
    a.q_rope = reinterpret_cast<const uint16_t*>(q_rope.data_ptr());
    a.qa = qa.data_ptr<uint8_t>();
    a.stride_qrb = q_rope.stride(0);
    a.stride_qrh = q_rope.stride(1);
    a.q_blocks = batch * kQBlocksPerRow;
    a.pos = pos.data_ptr<int64_t>();
    a.slot = slot.data_ptr<int64_t>();
    a.cos = reinterpret_cast<const uint16_t*>(cos.data_ptr());
    a.sin = reinterpret_cast<const uint16_t*>(sin.data_ptr());
    a.stride_cos_t = fused ? cos.stride(0) : 0;
    return a;
}

}  // namespace

void q_rope_quant(torch::Tensor q_rope, torch::Tensor qa, torch::Tensor pos, torch::Tensor slot,
                  torch::Tensor cos, torch::Tensor sin, bool fused) {
    Args a = q_args(q_rope, qa, pos, slot, cos, sin, fused);
    a.k_blocks = 0;
    a.order = 0;
    launch(a, a.q_blocks, fused);
}

void decode_prep(torch::Tensor q_rope, torch::Tensor qa, torch::Tensor ring, torch::Tensor req,
                 torch::Tensor k_nope, torch::Tensor k_rope, torch::Tensor kv_nope, torch::Tensor kv_aux,
                 torch::Tensor pos, torch::Tensor slot, torch::Tensor cos, torch::Tensor sin, bool fused,
                 int64_t order, int64_t k_split, bool rcp, bool tdm) {
    Args a = q_args(q_rope, qa, pos, slot, cos, sin, fused);
    // order 3 / 4: the K blocks / the Q blocks alone (timing probes).
    TORCH_CHECK(order >= 0 && order <= 4, "order must be 0 .. 4");
    const int part = order == 3 ? 1 : order == 4 ? 2 : 0;
    if (part) order = 1;
    TORCH_CHECK(k_split == 1 || k_split == 2, "k_split must be 1 or 2");
    a.order = int(order);
    const int batch = q_rope.size(0);
    TORCH_CHECK(ring.scalar_type() == torch::kBFloat16 && ring.dim() == 3 && ring.size(1) == 16 &&
                    ring.size(2) == 576 && ring.stride(2) == 1 && ring.stride(0) % 8 == 0 &&
                    ring.stride(1) % 8 == 0 && aligned16(ring), "ring must be BF16 [reqs, 16, 576]");
    TORCH_CHECK(req.scalar_type() == torch::kInt32 && req.is_contiguous() && req.numel() >= batch,
                "req_ids must be contiguous int32 [B]");
    TORCH_CHECK(k_nope.scalar_type() == torch::kBFloat16 && k_nope.size(1) == 512 && k_nope.stride(1) == 1 &&
                    k_nope.stride(0) % 8 == 0 && aligned16(k_nope), "k_nope must be BF16 [B, 512]");
    TORCH_CHECK(k_rope.scalar_type() == torch::kBFloat16 && k_rope.size(1) == 64 && k_rope.stride(1) == 1 &&
                    k_rope.stride(0) % 8 == 0 && aligned16(k_rope), "k_rope must be BF16 [B, 64]");
    TORCH_CHECK(kv_nope.is_contiguous() && kv_aux.is_contiguous() && kv_nope.size(-1) == 16384 &&
                    kv_aux.size(-1) == 8192, "cache planes must be contiguous [pages, 16384] / [pages, 8192]");
    a.ring = reinterpret_cast<uint16_t*>(ring.data_ptr());
    a.k_nope = reinterpret_cast<const uint16_t*>(k_nope.data_ptr());
    a.k_rope = reinterpret_cast<const uint16_t*>(k_rope.data_ptr());
    a.req = req.data_ptr<int32_t>();
    a.kv_nope = kv_nope.data_ptr<uint8_t>();
    a.kv_aux = kv_aux.data_ptr<uint8_t>();
    a.stride_req = ring.stride(0);
    a.stride_slot = ring.stride(1);
    a.stride_knope = k_nope.stride(0);
    a.stride_krope = k_rope.stride(0);
    a.k_split = order == 2 ? 1 : int(k_split);
    a.k_blocks = part == 2 ? 0 : batch * a.k_split;
    a.q_rows = batch;
    // TDM: K blocks all before or after the Q blocks, one per token, and q_rope
    // rows 128 head strides apart (rows x heads one axis of the TDM window).
    tdm = tdm && order != 2 && a.k_split == 1 && q_rope.stride(0) == 128 * q_rope.stride(1);
    launch(a, (part == 1 ? 0 : a.q_blocks) + a.k_blocks, fused, rcp, tdm);
}
