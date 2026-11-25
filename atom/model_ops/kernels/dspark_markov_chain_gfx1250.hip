// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Built by atom/model_ops/dspark_markov_chain.py (hipcc, gfx1250, -DFILL_TDM=1, aiter's opus include).
// DRF C5 v2 / v3 (FILL_TDM=1: step-0 fill through TDM into padded LDS rows of 528 B): the DSpark Markov chain (7 sequential argmax steps) in one persistent launch.
// Same arithmetic as v1 (markov_chain.hip, bit-exact vs production): per 16-vocab tile the 8-WMMA chain Triton's
// stage1 emits (W2 first operand, embed second, K split lanes 0-15 = K 0-7|16-23, lanes 16-31 = K 8-15|24-31),
// v = fp32(base) + acc, lowest id on ties.
// v2 changes (from the v1 timeline: fill 34 us, compute 6.9 us a step, exchange 3.0 us a step):
//  - step-0 fill: 32 x 16 B loads in flight a thread (v1: 8);
//  - the tiles past LDS (~10 a CTA) live in VGPRs for the whole launch (operand layout, <= 2 a wave);
//  - base logits for step k+1 are loaded into registers before step k's exchange wait;
//  - two LDS tiles a loop iteration (two independent WMMA chains);
//  - exchange slots may live in uncached memory (allocated by markov2_alloc_uncached).
// Exchange, tags and bounded polls as in v1 (MODE 0: every CTA reads all slots; MODE 1: CTA 0 reduces).

#include <hip/hip_runtime.h>
#include <stdint.h>
#ifndef FILL_TDM
#define FILL_TDM 0
#endif
#ifndef SWAP_OPS
#define SWAP_OPS 1  // 1: embed in the first WMMA slot, W2 tile in the second: output id l in lane l (l < 16)
#endif
#if FILL_TDM
#include "opus/opus.hpp"
#endif

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

#define RANK 256
#define TILE_BYTES (16 * RANK * 2)  // 8192
#ifndef NTHREADS
#define NTHREADS 256
#endif
#ifndef NWAVES
#define NWAVES 8
#endif
#define MAX_STEPS 8
#define POLL_BOUND 400000
#ifndef REM_MAX
#define REM_MAX 2    // remainder tiles a wave holds in VGPRs
#endif
#ifndef TPW_MAX
#define TPW_MAX 8    // tiles a wave handles in total (LDS + remainder), upper bound for base prefetch
#endif
#define FILL_BATCH 32
#define NSTAMP (2 + 3 * MAX_STEPS)
#if FILL_TDM
#define ROW_PITCH 528  // 512 B row + 16 B pad (TDM padding): rows r = 0..15 hit distinct banks, no swizzle
#else
#define ROW_PITCH 512  // XOR-swizzled 16 B chunks
#endif
#define TILE_LDS (16 * ROW_PITCH)

struct U32x8 {
  u32x4 lo, hi;
};
__device__ __forceinline__ bf16x16 make_op(u32x4 lo, u32x4 hi) {
  U32x8 t{lo, hi};
  return __builtin_bit_cast(bf16x16, t);
}
__device__ __forceinline__ int koff(int j, int h, int part) { return 32 * j + 8 * h + 16 * part; }
__device__ __forceinline__ f32x8 wmma_step(bf16x16 w2op, bf16x16 eop, f32x8 acc) {
#if SWAP_OPS
  return __builtin_amdgcn_wmma_f32_16x16x32_bf16(false, eop, false, w2op, (short)0, acc, false, false);
#else
  return __builtin_amdgcn_wmma_f32_16x16x32_bf16(false, w2op, false, eop, (short)0, acc, false, false);
#endif
}
__device__ __forceinline__ bool better(float v, int i, float bv, int bi) {
  return v > bv || (v == bv && i < bi);
}
__device__ __forceinline__ float bf16_lo(unsigned int w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16_hi(unsigned int w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ void wave_argmax(float& v, int& i) {
#pragma unroll
  for (int m = 16; m >= 1; m >>= 1) {
    float ov = __shfl_xor(v, m, 32);
    int oi = __shfl_xor(i, m, 32);
    if (better(ov, oi, v, i)) {
      v = ov;
      i = oi;
    }
  }
}
__device__ __forceinline__ unsigned long long ld_dev(const unsigned long long* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void st_dev(unsigned long long* p, unsigned long long v) {
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ unsigned long long pack_slot(float v, int id, unsigned tag) {
  return ((unsigned long long)__float_as_uint(v) << 32) | ((unsigned long long)(id & 0x3ffff) << 14) | (tag & 0x3fff);
}

// one tile's 8-WMMA chain from LDS (row r of the tile at tb, chunks XOR-swizzled by r)
__device__ __forceinline__ f32x8 tile_lds(const unsigned char* tb, int r, int h, const bf16x16* eb) {
  f32x8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  // all 16 LDS reads first, then the 8-WMMA chain (v2 waited on each pair before its WMMA: 8 LDS round trips a tile)
  u32x4 lo[8], hi[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    int c0 = koff(j, h, 0) >> 3, c1 = koff(j, h, 1) >> 3;
#if FILL_TDM
    lo[j] = *(const u32x4*)(tb + (c0 << 4));
    hi[j] = *(const u32x4*)(tb + (c1 << 4));
#else
    lo[j] = *(const u32x4*)(tb + ((c0 ^ r) << 4));
    hi[j] = *(const u32x4*)(tb + ((c1 ^ r) << 4));
#endif
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) acc = wmma_step(make_op(lo[j], hi[j]), eb[j], acc);
  return acc;
}

template <int MODE, typename IdT, bool DBG>
__global__ __launch_bounds__(NTHREADS, 1) void markov_chain2_kernel(
    const __bf16* __restrict__ base, long base_stride, IdT* __restrict__ ids, long ids_stride,
    const __bf16* __restrict__ w1, const __bf16* __restrict__ w2, __bf16* __restrict__ embed_out, long embed_stride,
    unsigned long long* __restrict__ slots, int* __restrict__ err, float* __restrict__ dbg_vals, int V, int nsteps,
    int lds_tiles, unsigned long long* __restrict__ tl) {
  extern __shared__ __attribute__((aligned(16))) unsigned char lds[];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int ncta = gridDim.x, cta = blockIdx.x;
  const int h = lane >> 4, r = lane & 15;
  const bool live = (r == 0);

  const int ntiles = V >> 4;
  const int q = ntiles / ncta, rem = ntiles % ncta;
  const int t0 = cta * q + min(cta, rem);
  const int nt = q + (cta < rem ? 1 : 0);
  const int nl = min(nt, lds_tiles);

  unsigned char* scratch = lds + (size_t)lds_tiles * TILE_LDS;
  float* red_v = (float*)scratch;
  int* red_i = (int*)(red_v + NWAVES);
  int* bcast = red_i + NWAVES;
  unsigned long long* tls = (unsigned long long*)(scratch + 192);  // after red_v/red_i (<= 16 waves) and bcast
  const bool do_tl = (tl != nullptr) && (tid == 0);
  if (do_tl) tls[0] = wall_clock64();

  unsigned long long* win = slots + (size_t)MAX_STEPS * ncta;
  const unsigned tag = ((unsigned)(ld_dev(&slots[(size_t)(nsteps - 1) * ncta + cta]) & 0x3fff) + 1) & 0x3fff;

  // ---- prologue 1: LDS fill ----
#if FILL_TDM
  {
    // this CTA's nl tiles are nl * 16 consecutive W2 rows of 256 bf16; wave w loads rows [w * rpw, ...) in ops of
    // 32 rows (at most 3 in flight a wave), LDS rows padded to ROW_PITCH.
    using Win = opus::tdm<unsigned short, opus::seq<RANK, 32>, opus::tdm_traits::padding<unsigned short, RANK, 8>>;
    const int nrows = nl * 16;
    const int rpw = (nrows + NWAVES - 1) / NWAVES;
    const int r0 = min(nrows, wave * rpw), r1 = min(nrows, r0 + rpw);
    if (r1 > r0) {
      const unsigned lds_base = (unsigned)reinterpret_cast<__UINTPTR_TYPE__>(lds);
      Win win = opus::make_tdm<Win>(lds_base, w2 + (size_t)t0 * 16 * RANK, (opus::u32_t)RANK, (opus::u32_t)r1,
                                    (opus::u64_t)RANK, 0u, (opus::u32_t)r0);
      int issued = 0;
      for (int rr = r0; rr < r1; rr += 32) {
        if (issued == 3) {
          __builtin_amdgcn_s_wait_tensorcnt(2);
          --issued;
        }
        // exactly this wave's rows: a tile past the tensor extent is zero-filled into LDS, i.e. into the next
        // wave's rows (seen in the 1-CTA smoke: 20 rows a wave clobbered)
        win.set_tile_dim1((opus::u32_t)min(32, r1 - rr));
        win.async_load((opus::u32_t)(rr * (ROW_PITCH / 2)));
        win.move(opus::number<0>{}, 32);
        ++issued;
      }
      __builtin_amdgcn_s_wait_tensorcnt(0);
    }
  }
#else
  {
    const int nchunk = nl * 512;
    for (int b = 0; b < nchunk; b += NTHREADS * FILL_BATCH) {
      u32x4 v[FILL_BATCH];
#pragma unroll
      for (int u = 0; u < FILL_BATCH; ++u) {
        int i = min(b + u * NTHREADS + tid, nchunk - 1);  // branch-free: no exec change between loads
        int t = i >> 9, rc = i & 511, rr = rc >> 5, c = rc & 31;
        v[u] = __builtin_nontemporal_load((const u32x4*)(w2 + ((size_t)(t0 + t) * 16 + rr) * RANK) + c);
      }
#pragma unroll
      for (int u = 0; u < FILL_BATCH; ++u) {
        int i = min(b + u * NTHREADS + tid, nchunk - 1);
        int t = i >> 9, rc = i & 511, rr = rc >> 5, c = rc & 31;
        *(u32x4*)(lds + (size_t)t * TILE_BYTES + rr * 512 + ((c ^ rr) << 4)) = v[u];
      }
    }
  }
#endif
  // ---- prologue 2: this wave's remainder tiles (nl + wave + 8 i) into VGPRs, operand layout ----
  u32x4 rlo[REM_MAX][8], rhi[REM_MAX][8];
#pragma unroll
  for (int i = 0; i < REM_MAX; ++i) {
    const int t = min(nl + wave + NWAVES * i, nt - 1);  // branch-free; a clamped duplicate is never consumed
    {
      const u32x4* g = (const u32x4*)(w2 + ((size_t)(t0 + t) * 16 + r) * RANK);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        rlo[i][j] = __builtin_nontemporal_load(g + (koff(j, h, 0) >> 3));
        rhi[i][j] = __builtin_nontemporal_load(g + (koff(j, h, 1) >> 3));
      }
    }
  }
  // tiles this wave handles: LDS tiles wave, wave+8, ... (< nl), then remainder tiles; base values (live lanes)
  // for step 0 into registers
  u32x4 bpre[TPW_MAX];
  auto load_base = [&](int k) {
    const __bf16* bk = base + (size_t)k * base_stride;
#pragma unroll
    for (int s = 0; s < TPW_MAX; ++s) {
      int t = (s < TPW_MAX - REM_MAX) ? (wave + NWAVES * s) : (nl + wave + NWAVES * (s - (TPW_MAX - REM_MAX)));
      t = min(t, nt - 1);  // branch-free; values of a clamped slot are never consumed
#if SWAP_OPS
      bpre[s].x = *(const unsigned short*)(bk + (t0 + t) * 16 + r);
#else
      bpre[s] = *(const u32x4*)(bk + (t0 + t) * 16 + 8 * h);
#endif
    }
  };
  load_base(0);
  __syncthreads();
  if (do_tl) tls[1] = wall_clock64();

  int prev = (int)ids[0];
  for (int k = 0; k < nsteps; ++k) {
    const __bf16* e = w1 + (size_t)prev * RANK;
    bf16x16 eb[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      // every lane loads (same 512 B row: one transaction a half), non-column-0 lanes zeroed by select
      u32x4 lo = *(const u32x4*)(e + koff(j, h, 0));
      u32x4 hi = *(const u32x4*)(e + koff(j, h, 1));
      const u32x4 z = {0, 0, 0, 0};
      eb[j] = make_op(live ? lo : z, live ? hi : z);
    }
    if (cta == 0 && tid < 32)
      *(u32x4*)(embed_out + (size_t)k * embed_stride + tid * 8) = *(const u32x4*)(e + tid * 8);
    if (do_tl) {
      // order the stamp after the embed loads (wave 0, lane 0 is live)
      asm volatile("s_wait_loadcnt 0x0" ::: "memory");
      tls[2 + 3 * k] = wall_clock64();
    }

    float bv = -__builtin_inff();
    int bi = 0x7fffffff;
    auto consume = [&](const f32x8& acc, int t, const u32x4& b4) {
#if SWAP_OPS
      // D[0][n]: lane n (< 16) holds vocab id (t0 + t) * 16 + n in element 0
      if (h == 0) {
        const int vid = (t0 + t) * 16 + r;
        float v = bf16_lo(b4.x) + acc[0];
        if (DBG) dbg_vals[(size_t)k * V + vid] = v;
        if (better(v, vid, bv, bi)) {
          bv = v;
          bi = vid;
        }
      }
      return;
#endif
      if (live) {
        const int vb = (t0 + t) * 16 + 8 * h;
        float bb[8] = {bf16_lo(b4.x), bf16_hi(b4.x), bf16_lo(b4.y), bf16_hi(b4.y),
                       bf16_lo(b4.z), bf16_hi(b4.z), bf16_lo(b4.w), bf16_hi(b4.w)};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          float v = bb[i] + acc[i];
          if (DBG) dbg_vals[(size_t)k * V + vb + i] = v;
          if (better(v, vb + i, bv, bi)) {
            bv = v;
            bi = vb + i;
          }
        }
      }
    };
    // LDS tiles, two a step of the loop (s and s+1 of this wave's list)
#pragma unroll
    for (int s = 0; s < TPW_MAX - REM_MAX; s += 2) {
      const int ta = wave + NWAVES * s, tb = wave + NWAVES * (s + 1);
      if (ta < nl) {
        f32x8 aa = tile_lds(lds + (size_t)ta * TILE_LDS + r * ROW_PITCH, r, h, eb);
        if (tb < nl) {
          f32x8 ab = tile_lds(lds + (size_t)tb * TILE_LDS + r * ROW_PITCH, r, h, eb);
          consume(aa, ta, bpre[s]);
          consume(ab, tb, bpre[s + 1]);
        } else {
          consume(aa, ta, bpre[s]);
        }
      }
    }
    // remainder tiles from VGPRs
#pragma unroll
    for (int i = 0; i < REM_MAX; ++i) {
      const int t = nl + wave + NWAVES * i;
      if (t < nt) {
        f32x8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int j = 0; j < 8; ++j) acc = wmma_step(make_op(rlo[i][j], rhi[i][j]), eb[j], acc);
        consume(acc, t, bpre[TPW_MAX - REM_MAX + i]);
      }
    }
    // next step's base values: issued now, consumed after the exchange
    if (k + 1 < nsteps) load_base(k + 1);

    wave_argmax(bv, bi);
    if (lane == 0) {
      red_v[wave] = bv;
      red_i[wave] = bi;
    }
    __syncthreads();
    if (tid == 0) {
      float cv = red_v[0];
      int ci = red_i[0];
      for (int w = 1; w < NWAVES; ++w)
        if (better(red_v[w], red_i[w], cv, ci)) {
          cv = red_v[w];
          ci = red_i[w];
        }
      if (ci == 0x7fffffff) ci = 0x3ffff;
      if (do_tl) tls[3 + 3 * k] = wall_clock64();
      st_dev(&slots[(size_t)k * ncta + cta], pack_slot(cv + 0.0f, ci, tag));
    }

    int winner;
    if (MODE == 0 || cta == 0) {
      float pv = -__builtin_inff();
      int pi = 0x7fffffff;
      if (tid < ncta) {
        unsigned long long sv;
        int it = 0;
        for (;;) {
          sv = ld_dev(&slots[(size_t)k * ncta + tid]);
          if ((unsigned)(sv & 0x3fff) == tag) break;
          if (++it > POLL_BOUND) {
            atomicOr(err, 1 << k);
            break;
          }
          __builtin_amdgcn_s_sleep(16);
        }
        pv = __uint_as_float((unsigned)(sv >> 32));
        pi = (int)((sv >> 14) & 0x3ffff);
      }
      wave_argmax(pv, pi);
      __syncthreads();
      if (lane == 0) {
        red_v[wave] = pv;
        red_i[wave] = pi;
      }
      __syncthreads();
      if (tid == 0) {
        float cv = red_v[0];
        int ci = red_i[0];
        for (int w = 1; w < NWAVES; ++w)
          if (better(red_v[w], red_i[w], cv, ci)) {
            cv = red_v[w];
            ci = red_i[w];
          }
        if (ci >= V) ci = 0;
        bcast[0] = ci;
        if (MODE == 1) st_dev(&win[k], ((unsigned long long)tag << 32) | (unsigned)ci);
      }
      __syncthreads();
      winner = bcast[0];
    } else {
      if (tid == 0) {
        unsigned long long sv;
        int it = 0;
        for (;;) {
          sv = ld_dev(&win[k]);
          if ((unsigned)(sv >> 32) == tag) break;
          if (++it > POLL_BOUND) {
            atomicOr(err, 1 << k);
            break;
          }
          __builtin_amdgcn_s_sleep(16);
        }
        bcast[0] = (int)(sv & 0xffffffffu);
      }
      __syncthreads();
      winner = bcast[0];
    }
    if (cta == 0 && tid == 0) ids[(size_t)(k + 1) * ids_stride] = (IdT)winner;
    if (do_tl) tls[4 + 3 * k] = wall_clock64();
    prev = winner;
    __syncthreads();
  }
  if (do_tl)
    for (int i = 0; i < 2 + 3 * nsteps; ++i) tl[(size_t)cta * NSTAMP + i] = tls[i];
}

template <int M, typename IdT, bool D>
static int launch_t(const void* base, long base_stride, void* ids, long ids_stride, const void* w1, const void* w2,
                    void* embed_out, long embed_stride, void* slots, void* err, void* dbg, int V, int nsteps,
                    int lds_tiles, int ncta, void* tl, hipStream_t stream) {
  auto kern = markov_chain2_kernel<M, IdT, D>;
  size_t lds = (size_t)lds_tiles * TILE_LDS + 512;
  if (lds > 327680) return -4;
  hipError_t e = hipFuncSetAttribute((const void*)kern, hipFuncAttributeMaxDynamicSharedMemorySize, (int)lds);
  if (e != hipSuccess) return 1000 + (int)e;
  hipLaunchKernelGGL(kern, dim3(ncta), dim3(NTHREADS), lds, stream, (const __bf16*)base, base_stride, (IdT*)ids,
                     ids_stride, (const __bf16*)w1, (const __bf16*)w2, (__bf16*)embed_out, embed_stride,
                     (unsigned long long*)slots, (int*)err, (float*)dbg, V, nsteps, lds_tiles,
                     (unsigned long long*)tl);
  return (int)hipGetLastError();
}

// Same C ABI as v1's markov_chain_launch (layout must be 0).
extern "C" int markov_chain_launch(int layout, int mode, int id64, int dbg, const void* base, long base_stride,
                                   void* ids, long ids_stride, const void* w1, const void* w2, void* embed_out,
                                   long embed_stride, void* slots, void* err, void* dbg_vals, int V, int nsteps,
                                   int lds_tiles, int ncta, void* stream, void* tl) {
  if (layout != 0 || nsteps < 1 || nsteps > MAX_STEPS || (V & 15) || ncta < 1 || ncta > NTHREADS) return -1;
  // every wave's tile list must fit the unrolled bounds
  int ntiles = V >> 4, q = ntiles / ncta + 1, nl = q < lds_tiles ? q : lds_tiles;
  if ((nl + NWAVES - 1) / NWAVES > TPW_MAX - REM_MAX || (q - nl + NWAVES - 1) / NWAVES > REM_MAX) return -3;
  hipStream_t s = (hipStream_t)stream;
#define ARGS base, base_stride, ids, ids_stride, w1, w2, embed_out, embed_stride, slots, err, dbg_vals, V, nsteps, \
             lds_tiles, ncta, tl, s
  if (mode == 0) {
    if (id64) return dbg ? launch_t<0, long long, true>(ARGS) : launch_t<0, long long, false>(ARGS);
    return dbg ? launch_t<0, int, true>(ARGS) : launch_t<0, int, false>(ARGS);
  }
  if (id64) return dbg ? launch_t<1, long long, true>(ARGS) : launch_t<1, long long, false>(ARGS);
  return dbg ? launch_t<1, int, true>(ARGS) : launch_t<1, int, false>(ARGS);
}
