// SPDX-License-Identifier: MIT
// gfx942 delayed mHC seam for prefill-sized row counts, the HIP half of mhc_rows_gfx942.py. Adapted from SGLang's
// hc_post_rows_gfx942.hip (Apache-2.0): one wave per row reads 1 KB contiguous per instruction, where the split-K
// Triton seam reads 64-128 B per row per instruction.
//
// mhc_rows_gfx942_*: R'_j = post_j * y + sum_h comb[h][j] * R_h (bf16), x1 = sum_j pre_j * R'_j (bf16),
// sq = (sum R'^2, sum x1^2), and the sublayer's RMSNorm of x1 written in place: out = bf16(x1 * rstd * w).
// mhc_gates_rows_gfx942: one thread per row, the gates from the mixing dots (hi + lo products of the GEMM).
#include <hip/hip_runtime.h>
#include <stdint.h>

constexpr int kHc = 4;

struct RowArgs {
  const uint16_t* x;
  const uint16_t* res;
  const float* post_in;
  const float* comb_in;
  const float* pre_prev;
  const void* norm_w;
  uint16_t* res_out;
  uint16_t* y;
  float* sq;
  int64_t x_stride;
  int64_t res_stride;
  int32_t m;
  int32_t h;
  int32_t w_f32;
  float norm_eps;
};

__device__ __forceinline__ float bf16_to_f32(uint32_t b) { return __uint_as_float(b << 16); }

__device__ __forceinline__ uint32_t f32_to_bf16(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (u >> 16) | 0x40u;
  return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}

__device__ __forceinline__ uint32_t word_of(const uint4& v, int e) {
  return e < 2 ? (e == 0 ? v.x : v.y) : (e == 2 ? v.z : v.w);
}

__device__ __forceinline__ uint32_t lane_bf16(const uint4& v, int e) {
  return (word_of(v, e >> 1) >> ((e & 1) * 16)) & 0xffffu;
}

__device__ __forceinline__ uint4 pack8(const uint32_t* b) {
  return make_uint4(b[0] | (b[1] << 16), b[2] | (b[3] << 16), b[4] | (b[5] << 16), b[6] | (b[7] << 16));
}

template <bool kPost>
__device__ __forceinline__ void mhc_rows(const RowArgs& a) {
  const int row = blockIdx.x * 4 + (threadIdx.x >> 6);
  const int lane = threadIdx.x & 63;
  if (row >= a.m) return;
  float pk[kHc], cm[kHc][kHc], pp[kHc];
  if constexpr (kPost) {
#pragma unroll
    for (int k = 0; k < kHc; ++k) pk[k] = a.post_in[row * kHc + k];
#pragma unroll
    for (int j = 0; j < kHc; ++j)
#pragma unroll
      for (int k = 0; k < kHc; ++k) cm[j][k] = a.comb_in[row * kHc * kHc + j * kHc + k];
  }
#pragma unroll
  for (int k = 0; k < kHc; ++k) pp[k] = a.pre_prev[row * kHc + k];
  const uint16_t* xrow = kPost ? a.x + static_cast<int64_t>(row) * a.x_stride : nullptr;
  const uint16_t* rrow = a.res + static_cast<int64_t>(row) * a.res_stride;
  uint16_t* orow = kPost ? a.res_out + static_cast<int64_t>(row) * (kHc * a.h) : nullptr;
  uint16_t* yrow = a.y + static_cast<int64_t>(row) * a.h;
  float sq_r = 0.f, sq_x = 0.f;
  for (int col = 8 * lane; col < a.h; col += 512) {
    uint4 xv, rv[kHc];
    if constexpr (kPost) xv = *reinterpret_cast<const uint4*>(xrow + col);
#pragma unroll
    for (int k = 0; k < kHc; ++k) rv[k] = *reinterpret_cast<const uint4*>(rrow + static_cast<int64_t>(k) * a.h + col);
    uint32_t bits[kHc][8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      if constexpr (kPost) {
        float r[kHc];
#pragma unroll
        for (int k = 0; k < kHc; ++k) r[k] = bf16_to_f32(lane_bf16(rv[k], e));
        const float xa = bf16_to_f32(lane_bf16(xv, e));
#pragma unroll
        for (int k = 0; k < kHc; ++k) {
          float v = xa * pk[k];
          v = v + r[0] * cm[0][k];
          v = v + r[1] * cm[1][k];
          v = v + r[2] * cm[2][k];
          v = v + r[3] * cm[3][k];
          bits[k][e] = f32_to_bf16(v);
        }
      } else {
#pragma unroll
        for (int k = 0; k < kHc; ++k) bits[k][e] = lane_bf16(rv[k], e);
      }
    }
    if constexpr (kPost) {
#pragma unroll
      for (int k = 0; k < kHc; ++k)
        *reinterpret_cast<uint4*>(orow + static_cast<int64_t>(k) * a.h + col) = pack8(bits[k]);
    }
    uint32_t yb[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      float yv = 0.f;
#pragma unroll
      for (int k = 0; k < kHc; ++k) {
        const float g = bf16_to_f32(bits[k][e]);
        yv = yv + pp[k] * g;
        sq_r = __builtin_fmaf(g, g, sq_r);
      }
      yb[e] = f32_to_bf16(yv);
      const float yr = bf16_to_f32(yb[e]);
      sq_x = __builtin_fmaf(yr, yr, sq_x);
    }
    *reinterpret_cast<uint4*>(yrow + col) = pack8(yb);
  }
#pragma unroll
  for (int off = 32; off > 0; off >>= 1) {
    sq_r += __shfl_xor(sq_r, off);
    sq_x += __shfl_xor(sq_x, off);
  }
  if (lane == 0) {
    a.sq[2 * row] = sq_r;
    a.sq[2 * row + 1] = sq_x;
  }
  const float rstd = 1.f / sqrtf(__builtin_fmaf(sq_x, 1.f / a.h, a.norm_eps));
  // each lane rereads only the columns it wrote above
  for (int col = 8 * lane; col < a.h; col += 512) {
    const uint4 yv = *reinterpret_cast<const uint4*>(yrow + col);
    float w[8];
    if (a.w_f32) {
      const float4* wp = reinterpret_cast<const float4*>(static_cast<const float*>(a.norm_w) + col);
      const float4 w0 = wp[0], w1 = wp[1];
      w[0] = w0.x; w[1] = w0.y; w[2] = w0.z; w[3] = w0.w;
      w[4] = w1.x; w[5] = w1.y; w[6] = w1.z; w[7] = w1.w;
    } else {
      const uint4 wv = *reinterpret_cast<const uint4*>(static_cast<const uint16_t*>(a.norm_w) + col);
#pragma unroll
      for (int e = 0; e < 8; ++e) w[e] = bf16_to_f32(lane_bf16(wv, e));
    }
    uint32_t ob[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) ob[e] = f32_to_bf16((bf16_to_f32(lane_bf16(yv, e)) * rstd) * w[e]);
    *reinterpret_cast<uint4*>(yrow + col) = pack8(ob);
  }
}

extern "C" __global__ void __launch_bounds__(256) mhc_rows_gfx942_post(const RowArgs a) { mhc_rows<true>(a); }
extern "C" __global__ void __launch_bounds__(256) mhc_rows_gfx942_nopost(const RowArgs a) { mhc_rows<false>(a); }

// dots(i, t) = dots[t * dot_t + i * dot_i]: rows [0, 24) hold the hi-weight products, [24, 48) the lo ones.
struct GateArgs {
  const float* dots;
  const float* sq;
  const float* scale;
  const float* base;
  float* pre;
  float* post;
  float* comb;
  int64_t dot_t;
  int64_t dot_i;
  int32_t m;
  int32_t iters;
  float inv_k;
  float rms_eps;
  float pre_eps;
  float sk_eps;
  float post_mult;
};

__device__ __forceinline__ float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

extern "C" __global__ void __launch_bounds__(256) mhc_gates_rows_gfx942(const GateArgs a) {
  const int row = blockIdx.x * 256 + threadIdx.x;
  if (row >= a.m) return;
  const float* d = a.dots + static_cast<int64_t>(row) * a.dot_t;
  float v[6 * kHc];
#pragma unroll
  for (int i = 0; i < 6 * kHc; ++i) v[i] = d[i * a.dot_i] + d[(i + 6 * kHc) * a.dot_i];
  const float rs = 1.f / sqrtf(__builtin_fmaf(a.sq[2 * row], a.inv_k, a.rms_eps));
  const float s0 = a.scale[0], s1 = a.scale[1], s2 = a.scale[2];
  float pre[kHc], post[kHc], c[kHc * kHc];
#pragma unroll
  for (int j = 0; j < kHc; ++j) {
    pre[j] = sigmoidf(v[j] * rs * s0 + a.base[j]) + a.pre_eps;
    post[j] = sigmoidf(v[kHc + j] * rs * s1 + a.base[kHc + j]) * a.post_mult;
  }
#pragma unroll
  for (int i = 0; i < kHc * kHc; ++i) c[i] = v[2 * kHc + i] * rs * s2 + a.base[2 * kHc + i];
#pragma unroll
  for (int j = 0; j < kHc; ++j) {
    float* r = c + j * kHc;
    const float mx = fmaxf(fmaxf(r[0], r[1]), fmaxf(r[2], r[3]));
#pragma unroll
    for (int k = 0; k < kHc; ++k) r[k] = expf(r[k] - mx);
    const float s = (r[0] + r[1]) + (r[2] + r[3]);
#pragma unroll
    for (int k = 0; k < kHc; ++k) r[k] = r[k] / s + a.sk_eps;
  }
  for (int it = 0; it < a.iters; ++it) {
    if (it > 0) {
#pragma unroll
      for (int j = 0; j < kHc; ++j) {
        float* r = c + j * kHc;
        const float s = (r[0] + r[1]) + (r[2] + r[3]) + a.sk_eps;
#pragma unroll
        for (int k = 0; k < kHc; ++k) r[k] = r[k] / s;
      }
    }
#pragma unroll
    for (int k = 0; k < kHc; ++k) {
      const float s = (c[k] + c[kHc + k]) + (c[2 * kHc + k] + c[3 * kHc + k]) + a.sk_eps;
#pragma unroll
      for (int j = 0; j < kHc; ++j) c[j * kHc + k] = c[j * kHc + k] / s;
    }
  }
  reinterpret_cast<float4*>(a.pre)[row] = make_float4(pre[0], pre[1], pre[2], pre[3]);
  reinterpret_cast<float4*>(a.post)[row] = make_float4(post[0], post[1], post[2], post[3]);
  float4* co = reinterpret_cast<float4*>(a.comb + static_cast<int64_t>(row) * (kHc * kHc));
#pragma unroll
  for (int j = 0; j < kHc; ++j) co[j] = make_float4(c[j * kHc], c[j * kHc + 1], c[j * kHc + 2], c[j * kHc + 3]);
}
