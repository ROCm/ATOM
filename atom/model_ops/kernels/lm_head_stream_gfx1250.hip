// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 ATOM Project Contributors
//
// The LM head at decode rows, out[M, N] = x[M, K] @ w[N, K]^T (bf16, M <= 8), as a TDM-streamed WMMA chain that
// keeps opus_gemm's arithmetic: per 16-row group of w, one f32 accumulator a 16 x 16 C slice, chained over K in
// 32-wide chunks from 0 upwards (v_wmma_f32_16x16x32_bf16, w in src0, x in src1, as opus's swap_ab), then one RNE
// cast to bf16. No split-K, so every column is the same reduction as opus's.
//
// Layout: one CTA a CU, NW waves. CTA c owns 16-row groups [c * G / ncta, (c + 1) * G / ncta) of the G = N / 16, its
// waves split them (contiguous, at most GW_MAX a wave) and each wave streams its groups' rows through its own TDM
// ring, KB columns a slot, SLOTS slots: no CTA barrier in the loop, and a wave's bytes in flight are proportional to
// its rows, so the waves of a CU (and the CUs) finish together. x is staged once in LDS (the M real rows; the
// padding columns' lanes feed zeros).
#include <hip/hip_runtime.h>
#include <stdint.h>
#include "opus/opus.hpp"

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

#define NW 16
#define KB 128  // K elements a TDM slot
#define SLOTS 3
#define GW_MAX 1  // 16-row groups a wave (a CTA's groups <= NW * GW_MAX)
#define TH_VAL 1  // non-temporal TDM loads of w (opus tdm_traits::load_temporal_hint)
#define SCOPE_VAL 2  // cu 0, se 1, dev 2, sys 3
#define NTHREADS (NW * 32)
#define MT 8
#define KMAX 6144
#define W_PITCH (KB + 8)  // elements: a 16-B pad every row (TDM padding): 16 rows on distinct banks
#define SLOT_ELEMS (16 * GW_MAX * W_PITCH)
#define X_PITCH (KMAX + 8)
#define X_LDS_BYTES (MT * X_PITCH * 2)
#define W_LDS_BYTES (NW * SLOTS * SLOT_ELEMS * 2)
#define LDS_BYTES (X_LDS_BYTES + W_LDS_BYTES)
static_assert(KB % 32 == 0, "KB is whole WMMA K chunks");
static_assert(GW_MAX == 1, "the kernel runs one 16-row group a wave");
static_assert(LDS_BYTES <= 327680, "LDS");

struct U32x8 {
  u32x4 lo, hi;
};
__device__ __forceinline__ bf16x16 make_op(u32x4 lo, u32x4 hi) {
  U32x8 t{lo, hi};
  return __builtin_bit_cast(bf16x16, t);
}

#define W_POL                                                                             \
  opus::tdm_traits::make_cache_policy((opus::tdm_traits::load_temporal_hint)(TH_VAL), \
                                      (opus::tdm_traits::scope)(SCOPE_VAL))
using WinW = opus::tdm<unsigned short, opus::seq<KB, 16 * GW_MAX>, opus::tdm_traits::padding<unsigned short, KB, 8>,
                       opus::tdm_traits::cache<W_POL>>;

// RG groups over all of K: per slot, every ds_load first (x and RG w fragments a chunk), then the chunk-ordered
// WMMAs; then the slot is refilled with load q + SLOTS. Stores the groups' logits (16 B a lane and group).
template <int RG, typename Issue>
__device__ __forceinline__ void l_groups(const unsigned short* ring, const unsigned short* xrow, bool xreal, int g0,
                                         int nkb, Issue& issue, __bf16* out, long ldo) {
  const int lane = threadIdx.x & 31, col = lane & 15, h = lane >> 4;
  const u32x4 z4 = {0, 0, 0, 0};
  f32x8 acc[RG];
#pragma unroll
  for (int g = 0; g < RG; ++g) acc[g] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
  for (int q = 0; q < nkb; ++q) {
    // load q landed; the loads issued after it may stay in flight
    if (q + SLOTS <= nkb) {
      __builtin_amdgcn_s_wait_tensorcnt(SLOTS - 1);
    } else {
      __builtin_amdgcn_s_wait_tensorcnt(0);
    }
    asm volatile("" ::: "memory");
    const unsigned short* slot = ring + (q % SLOTS) * SLOT_ELEMS + col * W_PITCH + 8 * h;
    constexpr int NJ = KB / 32;
    u32x4 xv[NJ][2], wv[NJ][RG][2];
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const int kk = q * KB + 32 * j;
      xv[j][0] = *(const u32x4*)(xrow + kk);
      xv[j][1] = *(const u32x4*)(xrow + kk + 16);
#pragma unroll
      for (int g = 0; g < RG; ++g) {
        const unsigned short* wr = slot + g * 16 * W_PITCH + 32 * j;
        wv[j][g][0] = *(const u32x4*)wr;
        wv[j][g][1] = *(const u32x4*)(wr + 16);
      }
    }
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const bf16x16 xo = make_op(xreal ? xv[j][0] : z4, xreal ? xv[j][1] : z4);
#pragma unroll
      for (int g = 0; g < RG; ++g)
        acc[g] = __builtin_amdgcn_wmma_f32_16x16x32_bf16(false, make_op(wv[j][g][0], wv[j][g][1]), false, xo,
                                                         (short)0, acc[g], false, false);
    }
    // WAR: this slot's reads retired, then it takes load q + SLOTS
    asm volatile("s_wait_dscnt 0x0" ::: "memory");
    if (q + SLOTS < nkb) issue(q + SLOTS);
  }
  // lane (col, h) holds rows 8h..8h+7 of each group for token col
  if (xreal) {
#pragma unroll
    for (int g = 0; g < RG; ++g)
      *(u32x4*)(out + col * ldo + (long)(g0 + g) * 16 + 8 * h) =
          __builtin_bit_cast(u32x4, __builtin_convertvector(acc[g], bf16x8));
  }
}

__global__ __launch_bounds__(NTHREADS, 1) void lm_head_stream_kernel(const __bf16* __restrict__ x, int m, long ldx,
                                                                     const __bf16* __restrict__ w, int n, int k,
                                                                     __bf16* __restrict__ out, long ldo) {
  extern __shared__ __attribute__((aligned(16))) char lds[];
  unsigned short* xl = (unsigned short*)lds;
  unsigned short* wl = (unsigned short*)(lds + X_LDS_BYTES);
  const int tid = threadIdx.x, lane = tid & 31;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int G = n >> 4;
  const int cg0 = (int)(((long)blockIdx.x * G) / gridDim.x);
  const int cg1 = (int)(((long)(blockIdx.x + 1) * G) / gridDim.x);
  const int cnt = cg1 - cg0;
  const int wg0 = cg0 + (wave * cnt) / NW, wg1 = cg0 + ((wave + 1) * cnt) / NW;
  const int ng = wg1 - wg0;
  const int nkb = ng > 0 ? k / KB : 0;  // this wave's TDM loads, in consumption order
  unsigned short* ring = wl + wave * SLOTS * SLOT_ELEMS;
  const unsigned ring_lds = (unsigned)reinterpret_cast<__UINTPTR_TYPE__>(ring);
  auto issue = [&](int q) {
    WinW win = opus::make_tdm<WinW>(ring_lds, w + (size_t)wg0 * 16 * k, (opus::u32_t)k, (opus::u32_t)(ng * 16),
                                    (opus::u64_t)k, (opus::u32_t)(q * KB), 0u);
    win.set_tile_dim1((opus::u32_t)(ng * 16));
    win.async_load((opus::u32_t)((q % SLOTS) * SLOT_ELEMS));
  };
  // prologue: the ring's first SLOTS loads, then x into LDS (one CTA barrier, the only one)
  for (int q = 0; q < SLOTS && q < nkb; ++q) issue(q);
  {
    const int per_row = k >> 3;  // 16-B chunks a row
    for (int i = tid; i < m * per_row; i += NTHREADS) {
      const int t = i / per_row, c = i - t * per_row;
      *(u32x4*)(xl + t * X_PITCH + c * 8) = *(const u32x4*)(x + t * ldx + c * 8);
    }
  }
  __syncthreads();

  const int col = lane & 15, h = lane >> 4;
  const bool xreal = col < m;
  const unsigned short* xrow = xl + (xreal ? col : 0) * X_PITCH + 8 * h;
  if (ng > 0) l_groups<1>(ring, xrow, xreal, wg0, nkb, issue, out, ldo);  // ng <= GW_MAX: the launcher checks it
}

static int set_attr(const void* f) {
  return hipFuncSetAttribute(f, hipFuncAttributeMaxDynamicSharedMemorySize, LDS_BYTES) == hipSuccess ? 0 : -4;
}

// x [m, k] (row stride ldx), w [n, k] contiguous, out [m, n] (row stride ldo); n % 16 == 0, k % KB == 0, k <= KMAX,
// 1 <= m <= 8, and a wave's groups <= GW_MAX. Returns -1 / -2 when the shape is refused (nothing launched).
extern "C" int lm_head_stream_launch(const void* x, int m, long ldx, const void* w, int n, int k, void* out, long ldo,
                                     int ncta, void* stream) {
  if (m < 1 || m > MT || n % 16 || k % KB || k > KMAX || ncta < 1) return -1;
  if (((n / 16 + ncta - 1) / ncta + NW - 1) / NW > GW_MAX) return -2;
  static int attr = set_attr((const void*)lm_head_stream_kernel);
  if (attr) return attr;
  hipLaunchKernelGGL(lm_head_stream_kernel, dim3(ncta), dim3(NTHREADS), LDS_BYTES, (hipStream_t)stream,
                     (const __bf16*)x, m, ldx, (const __bf16*)w, n, k, (__bf16*)out, ldo);
  return hipGetLastError() == hipSuccess ? 0 : -5;
}
