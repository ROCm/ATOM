// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// DRAFT TAIL (11-27), phase L first: the target LM head, out[M, N] = x[M, K] @ w[N, K]^T (bf16, M <= 8), as a
// TDM-streamed WMMA chain that keeps the served opus_gemm's arithmetic: per 16-row group of w, one f32 accumulator
// a 16 x 16 C slice, chained over K in 32-wide chunks from 0 upwards (v_wmma_f32_16x16x32_bf16, w in src0, x in
// src1, as opus's swap_ab), then one RNE cast to bf16. No split-K, so every column is the same reduction as opus's.
//
// Layout: one CTA a CU, NW waves. CTA c owns 16-row groups [c * G / ncta, (c + 1) * G / ncta) of the G = N / 16, its
// waves split them (contiguous, ng <= GW_MAX a wave) and each wave streams ALL its groups' rows through its own
// TDM ring, KB columns a slot, SLOTS slots: no CTA barrier in the loop, and a wave's bytes in flight are
// proportional to its rows, so the waves of a CU (and the CUs, 12 or 13 groups) finish together. x is staged once
// in LDS (the M real rows; the padding columns' lanes feed zeros). ROUNDS > 1 runs a wave's groups in that many
// row rounds (each over all of K), so a round's logits are final before the launch ends (the all-gather can start).
#include <hip/hip_runtime.h>
#include <stdint.h>
#include "opus/opus.hpp"

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

#ifndef NW
#define NW 16
#endif
#ifndef KB
#define KB 128  // K elements a TDM slot
#endif
#ifndef SLOTS
#define SLOTS 3
#endif
#ifndef GW_MAX
#define GW_MAX 1  // 16-row groups a wave (a CTA's groups <= NW * GW_MAX)
#endif
#ifndef ROUNDS
#define ROUNDS 1
#endif
#ifndef TH_NT
#define TH_NT 1  // 1: non-temporal TDM loads of w (TH_VAL 1)
#endif
#ifndef TH_VAL
#define TH_VAL (TH_NT ? 1 : 0)  // the TDM loads' temporal hint (opus tdm_traits::load_temporal_hint)
#endif
#ifndef SCOPE_VAL
#define SCOPE_VAL 2  // cu 0, se 1, dev 2, sys 3
#endif
#define NTHREADS (NW * 32)
#define MT 8
#ifndef KMAX
#define KMAX 6144
#endif
#define W_PITCH (KB + 8)  // elements: a 16-B pad every row (TDM padding): 16 rows on distinct banks
#define SLOT_ELEMS (16 * GW_MAX * W_PITCH)
#define X_PITCH (KMAX + 8)
#define X_LDS_BYTES (MT * X_PITCH * 2)
#define W_LDS_BYTES (NW * SLOTS * SLOT_ELEMS * 2)
#define LDS_BYTES (X_LDS_BYTES + W_LDS_BYTES)
static_assert(KB % 32 == 0, "KB is whole WMMA K chunks");
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

// One round of RG groups over all of K: per slot, every ds_load first (x and RG w fragments a chunk), then the
// chunk-ordered WMMAs; then the slot is refilled with load q + SLOTS. Stores the round's logits (16 B a group).
template <int RG, typename Issue, typename Epi>
__device__ __forceinline__ void l_round(const unsigned short* ring, const unsigned short* xrow, bool xreal, int r,
                                        int r0, int nkb, int nq, Issue& issue, Epi& epi) {
  const int lane = threadIdx.x & 31, col = lane & 15, h = lane >> 4;
  const u32x4 z4 = {0, 0, 0, 0};
  f32x8 acc[RG];
#pragma unroll
  for (int g = 0; g < RG; ++g) acc[g] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
  for (int kb = 0; kb < nkb; ++kb) {
    const int q = r * nkb + kb;
    // load q landed; the loads issued after it may stay in flight
    if (q + SLOTS <= nq) {
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
      const int kk = kb * KB + 32 * j;
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
    if (q + SLOTS < nq) issue(q + SLOTS);
  }
  // lane (col, h) holds rows 8h..8h+7 of each group for token col
  if (xreal) {
#pragma unroll
    for (int g = 0; g < RG; ++g) epi(r0 + g, __builtin_bit_cast(u32x4, __builtin_convertvector(acc[g], bf16x8)));
  }
}

// TP (the logits all-gather folded in, lamport_all_gather's protocol and codec): every rank's inbox[3 bufs][4 src]
// [MT rows][n] bf16 is mapped on every rank (IPC). Call c pushes this rank's 16-B packs (codec applied) into slot
// [c % 3][rank] of every peer as each round ends, writes them into its own gathered output, re-arms its own slots
// of buffer (c + 2) % 3 at entry (peers are at most one call ahead), and at the end polls its groups' packs of
// every peer in buffer c % 3 (system-scope 16-B loads until no half-word is the 0xffff sentinel) into the output.
struct TailTp {
  unsigned short* inbox[4];
  int rank, ranks;
  unsigned long long* cnt;  // a call counter a CTA
  int* err;
  unsigned long long* tl;  // null, or stamps [cta][wave][4]: entry, loop start, rounds + pushes done, receive done
};
#ifndef POLL_BOUND
#define POLL_BOUND 4000000
#endif
#ifndef TP_PUSH
#define TP_PUSH 1  // 0 (timing only): no pushes and no receive
#endif
#ifndef TP_RECV
#define TP_RECV 1  // 0 (timing only): pushes but no receive wait
#endif
typedef __attribute__((address_space(1))) u32x4* g128_t;
__device__ __forceinline__ unsigned codec_sentinel(unsigned w) {
  const unsigned x = ~w;
  const unsigned t = ((x & 0x7fff7fffu) + 0x7fff7fffu) | x;
  return w ^ ((t & 0x80008000u) ^ 0x80008000u);  // 0xffff half-words -> 0x7fff, others unchanged
}
__device__ __forceinline__ u32x4 codec4(u32x4 v) {
#pragma unroll
  for (int c = 0; c < 4; ++c) v[c] = codec_sentinel(v[c]);
  return v;
}
__device__ __forceinline__ unsigned short* tp_slot(const TailTp& t, int dst, int buf, int src, int row, long col,
                                                   int n) {
  return t.inbox[dst] + (((long)buf * 4 + src) * MT + row) * n + col;
}
__device__ __forceinline__ unsigned long long ld_dev(const unsigned long long* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// x: the [m, k] operand as columns [k0, k0 + k) of up to 5 row-major blocks of width bw laid side by side (the
// drafter's tap buffers for the context fc's K-slice; a plain x is one block, k0 = 0), row stride ldx each.
struct XTaps {
  const __bf16* p[5];
  int bw, k0;
};

template <bool TP>
__global__ __launch_bounds__(NTHREADS, 1) void tail_l_kernel(XTaps xt, int m, long ldx,
                                                             const __bf16* __restrict__ w, int n, int k,
                                                             __bf16* __restrict__ out, long ldo, TailTp tp) {
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
  const int nkb = k / KB;
  unsigned short* ring = wl + wave * SLOTS * SLOT_ELEMS;
  const unsigned ring_lds = (unsigned)reinterpret_cast<__UINTPTR_TYPE__>(ring);

  // the wave's rounds: round r = groups [wg0 + r * ng / nr, wg0 + (r + 1) * ng / nr), each over all of K
  const int ng = wg1 - wg0;
  const int nr = ng <= 0 ? 0 : (ng < ROUNDS ? ng : ROUNDS);
  const int nq = nr * nkb;  // this wave's TDM loads, in consumption order
  auto issue = [&](int q) {
    const int r = q / nkb, kb = q - r * nkb;
    const int r0 = wg0 + (r * ng) / nr, r1 = wg0 + ((r + 1) * ng) / nr;
    WinW win = opus::make_tdm<WinW>(ring_lds, w + (size_t)r0 * 16 * k, (opus::u32_t)k,
                                    (opus::u32_t)((r1 - r0) * 16), (opus::u64_t)k, (opus::u32_t)(kb * KB), 0u);
    win.set_tile_dim1((opus::u32_t)((r1 - r0) * 16));
    win.async_load((opus::u32_t)((q % SLOTS) * SLOT_ELEMS));
  };
  const unsigned long long t_entry = wall_clock64();
  // prologue: the ring's first SLOTS loads, then x into LDS (one CTA barrier, the only one)
  for (int q = 0; q < SLOTS && q < nq; ++q) issue(q);
  {
    const int per_row = k >> 3;  // 16-B chunks a row
    for (int i = tid; i < m * per_row; i += NTHREADS) {
      const int t = i / per_row, c = i - t * per_row;
      const int gc = xt.k0 + c * 8, b = gc / xt.bw;
      *(u32x4*)(xl + t * X_PITCH + c * 8) = *(const u32x4*)(xt.p[b] + t * ldx + (gc - b * xt.bw));
    }
  }
  __syncthreads();

  const int col = lane & 15, h = lane >> 4;
  const bool xreal = col < m;
  const unsigned short* xrow = xl + (xreal ? col : 0) * X_PITCH + 8 * h;
  unsigned long long calls = 0;
  int gbuf = 0;
  if constexpr (TP) {
    calls = ld_dev(tp.cnt + blockIdx.x);
    gbuf = (int)(calls % 3ull);
    // re-arm this wave's packs of every peer in buffer (calls + 2) % 3
    const u32x4 sent = {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
    if (xreal) {
      for (int g = wg0; g < wg1; ++g)
        for (int s2 = 0; s2 < tp.ranks; ++s2)
          if (s2 != tp.rank)
            __builtin_amdgcn_global_store_b128(
                (g128_t)tp_slot(tp, tp.rank, (gbuf + 2) % 3, s2, col, (long)g * 16 + 8 * h, n), sent, "");
    }
  }
  auto epi = [&](int g, u32x4 v) {
    const long c = (long)g * 16 + 8 * h;
    if constexpr (TP) {
      v = codec4(v);
      for (int d = 0; d < tp.ranks; ++d)
        if (TP_PUSH && d != tp.rank)
          __builtin_amdgcn_global_store_b128((g128_t)tp_slot(tp, d, gbuf, tp.rank, col, c, n), v, "");
      *(u32x4*)(out + col * ldo + (long)tp.rank * n + c) = v;
    } else {
      *(u32x4*)(out + col * ldo + c) = v;
    }
  };
  const unsigned long long t_loop = wall_clock64();
  for (int r = 0; r < nr; ++r) {
    const int r0 = wg0 + (r * ng) / nr, r1 = wg0 + ((r + 1) * ng) / nr;
    switch (r1 - r0) {  // straight-line bodies a group count: every slot's ds_loads issue as one batch
#define TAIL_RG(n) \
  case n:          \
    l_round<n>(ring, xrow, xreal, r, r0, nkb, nq, issue, epi); \
    break;
      TAIL_RG(1)
#if GW_MAX >= 2
      TAIL_RG(2)
#endif
#if GW_MAX >= 3
      TAIL_RG(3)
#endif
#if GW_MAX >= 4
      TAIL_RG(4)
#endif
#if GW_MAX >= 5
      TAIL_RG(5)
#endif
#if GW_MAX >= 6
      TAIL_RG(6)
#endif
#if GW_MAX >= 7
      TAIL_RG(7)
#endif
#undef TAIL_RG
      default:
        break;
    }
  }
  const unsigned long long t_rounds = wall_clock64();
  if constexpr (TP) {
#if TP_PUSH && TP_RECV
    // this wave's groups of every peer, polled out of buffer calls % 3 into the gathered output: every pending
    // pack's load in flight at once (system scope), then the checks; until no lane of the wave waits
    {
      const int ng = wg1 - wg0;
      bool need[GW_MAX][3];
      const unsigned short* src[GW_MAX][3];
      __bf16* dst[GW_MAX][3];
#pragma unroll
      for (int gi = 0; gi < GW_MAX; ++gi)
#pragma unroll
        for (int pi = 0; pi < 3; ++pi) {
          const int s2 = (tp.rank + 1 + pi) % tp.ranks;
          const long c = (long)(wg0 + gi) * 16 + 8 * h;
          need[gi][pi] = xreal && gi < ng && pi < tp.ranks - 1;
          src[gi][pi] = tp_slot(tp, tp.rank, gbuf, s2, col, c, n);
          dst[gi][pi] = out + col * ldo + (long)s2 * n + c;
        }
      int it = 0;
      for (;;) {
        asm volatile("" ::: "memory");
        u32x4 q[GW_MAX][3];
#pragma unroll
        for (int gi = 0; gi < GW_MAX; ++gi)
#pragma unroll
          for (int pi = 0; pi < 3; ++pi)
            if (need[gi][pi]) q[gi][pi] = __builtin_amdgcn_global_load_b128((g128_t)src[gi][pi], "");
        bool any = false;
#pragma unroll
        for (int gi = 0; gi < GW_MAX; ++gi)
#pragma unroll
          for (int pi = 0; pi < 3; ++pi)
            if (need[gi][pi]) {
              const u32x4 v = q[gi][pi];
              const unsigned mm = (codec_sentinel(v[0]) ^ v[0]) | (codec_sentinel(v[1]) ^ v[1]) |
                                  (codec_sentinel(v[2]) ^ v[2]) | (codec_sentinel(v[3]) ^ v[3]);
              if (mm) {
                any = true;
              } else {
                *(u32x4*)dst[gi][pi] = v;
                need[gi][pi] = false;
              }
            }
        if (!__builtin_amdgcn_ballot_w32(any)) break;
        if (++it > POLL_BOUND) {
          if (any) atomicOr(tp.err, 1);
          break;
        }
        __builtin_amdgcn_s_sleep(1);
      }
    }
#endif
    if (tp.tl != nullptr && lane == 0) {
      unsigned long long* st = tp.tl + ((size_t)blockIdx.x * NW + wave) * 4;
      st[0] = t_entry;
      st[1] = t_loop;
      st[2] = t_rounds;
      st[3] = wall_clock64();
    }
    __syncthreads();
    if (threadIdx.x == 0) __hip_atomic_store(tp.cnt + blockIdx.x, calls + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
}

extern "C" int tail_l_lds_bytes() { return LDS_BYTES; }

// x [m, k] (row stride ldx), w [n, k] contiguous, out [m, n] (row stride ldo); n % 16 == 0, k % KB == 0, k <= KMAX,
// m <= 8, every CTA's groups <= NW * GW_MAX.
static int set_attr(const void* f) {
  return hipFuncSetAttribute(f, hipFuncAttributeMaxDynamicSharedMemorySize, LDS_BYTES) == hipSuccess ? 0 : -4;
}
static int check(int m, int n, int k, int ncta) {
  if (m < 1 || m > MT || n % 16 || k % KB || k > KMAX || ncta < 1) return -1;
  const int G = n / 16;
  if (((G + ncta - 1) / ncta + NW - 1) / NW > GW_MAX) return -2;
  return 0;
}

// x [m, k] (row stride ldx), w [n, k] contiguous, out [m, n] (row stride ldo); n % 16 == 0, k % KB == 0, k <= KMAX,
// m <= 8, a wave's groups <= GW_MAX.
extern "C" int tail_l_launch(const void* x, int m, long ldx, const void* w, int n, int k, void* out, long ldo,
                             int ncta, void* stream) {
  if (int e = check(m, n, k, ncta)) return e;
  static int attr = set_attr((const void*)tail_l_kernel<false>);
  if (attr) return attr;
  TailTp tp{};
  XTaps xt{{(const __bf16*)x}, k, 0};
  hipLaunchKernelGGL(tail_l_kernel<false>, dim3(ncta), dim3(NTHREADS), LDS_BYTES, (hipStream_t)stream, xt, m, ldx, (const __bf16*)w, n, k, (__bf16*)out, ldo, tp);
  return hipGetLastError() == hipSuccess ? 0 : -5;
}

// TP: w is this rank's vocab shard [n, k]; out the gathered logits [m, ranks * n] (row stride ldo); inbox0..3 every
// rank's inbox (3 * 4 * MT * n bf16, all 0xffff before the first call), cnt ncta call counters (0 before the first
// call, the same on every rank), err an int.
extern "C" int tail_la_launch_tl(const void* x, int m, long ldx, const void* w, int n, int k, void* out, long ldo,
                                 int rank, int ranks, void* inbox0, void* inbox1, void* inbox2, void* inbox3,
                                 void* cnt, void* err, int ncta, void* stream, void* tl);
extern "C" int tail_la_launch(const void* x, int m, long ldx, const void* w, int n, int k, void* out, long ldo,
                              int rank, int ranks, void* inbox0, void* inbox1, void* inbox2, void* inbox3, void* cnt,
                              void* err, int ncta, void* stream) {
  return tail_la_launch_tl(x, m, ldx, w, n, k, out, ldo, rank, ranks, inbox0, inbox1, inbox2, inbox3, cnt, err, ncta,
                           stream, nullptr);
}
extern "C" int tail_la_launch_tl(const void* x, int m, long ldx, const void* w, int n, int k, void* out, long ldo,
                                 int rank, int ranks, void* inbox0, void* inbox1, void* inbox2, void* inbox3,
                                 void* cnt, void* err, int ncta, void* stream, void* tl) {
  if (int e = check(m, n, k, ncta)) return e;
  if (ranks < 2 || ranks > 4 || rank < 0 || rank >= ranks) return -3;
  static int attr = set_attr((const void*)tail_l_kernel<true>);
  if (attr) return attr;
  TailTp tp{{(unsigned short*)inbox0, (unsigned short*)inbox1, (unsigned short*)inbox2, (unsigned short*)inbox3},
            rank, ranks, (unsigned long long*)cnt, (int*)err, (unsigned long long*)tl};
  XTaps xt{{(const __bf16*)x}, k, 0};
  hipLaunchKernelGGL(tail_l_kernel<true>, dim3(ncta), dim3(NTHREADS), LDS_BYTES, (hipStream_t)stream, xt, m, ldx, (const __bf16*)w, n, k, (__bf16*)out, ldo, tp);
  return hipGetLastError() == hipSuccess ? 0 : -5;
}

// F (the draft context fc on this rank's K-slice): x = columns [k0, k0 + k) of the nt tap blocks [m, bw] (row
// stride ldx), w [n, k] the fc weight's K-slice, out [m, n]. Same arithmetic as tail_l_launch.
extern "C" int tail_f_launch(const void* t0, const void* t1, const void* t2, const void* t3, const void* t4, int bw,
                             int k0, int m, long ldx, const void* w, int n, int k, void* out, long ldo, int ncta,
                             void* stream) {
  if (int e = check(m, n, k, ncta)) return e;
  if (bw % 8 || k0 % 8 || bw < 8) return -3;
  static int attr = set_attr((const void*)tail_l_kernel<false>);
  if (attr) return attr;
  TailTp tp{};
  XTaps xt{{(const __bf16*)t0, (const __bf16*)t1, (const __bf16*)t2, (const __bf16*)t3, (const __bf16*)t4}, bw, k0};
  hipLaunchKernelGGL(tail_l_kernel<false>, dim3(ncta), dim3(NTHREADS), LDS_BYTES, (hipStream_t)stream, xt, m, ldx,
                     (const __bf16*)w, n, k, (__bf16*)out, ldo, tp);
  return hipGetLastError() == hipSuccess ? 0 : -5;
}

