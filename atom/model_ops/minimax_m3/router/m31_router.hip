// SPDX-License-Identifier: Apache-2.0
// Adapted from SGLang (Apache-2.0), python/sglang/kernels/ops/moe/smallm_router_gfx950/m31_router.hip
// (rev fc9e1c8d), keeping its router GEMV, gate and sort; this version drops the fused ptpc shared-expert
// GEMM, reads the token count from the device and writes aiter's aux sort (plain-row padding, m_indices,
// reverse_sorted).
//
// MiniMax-M3.1 decode router for 1..16 tokens in one launch on gfx950: router GEMV (bf16 x [M, 6144] times
// bf16 W [128, 6144], fp32 split-K partials) -> sigmoid, correction bias, top-4, renormalize, routed scaling
// -> aiter's block_m 16 MoE sort. The last CTA to finish its GEMV item does the reduction, gate and sort.
// Built by router.py.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));

struct __attribute__((packed)) Args {
    const uint16_t* x; const uint16_t* w; const float* bias;
    float* part; int32_t* cnt; float* topk_w; int32_t* topk_ids;
    int32_t* sorted_ids; float* sorted_w; int32_t* sorted_eids; int32_t* num_valid;
    int32_t* m_indices; int32_t* reverse_sorted;
    // The sorted ids again in moe_sorting's layout (padding rows (TOPK << 24) | tok); the aux layout above pads
    // with the plain row count. Which one the MoE reads depends on the pipeline aiter selects.
    int32_t* sorted_ids_ms;
    // Rows [n_local[0], tok) are padding: no routes, their reverse_sorted entries -1. Null: every row is real.
    const int32_t* n_local;
    // The MoE output [tok, K] bf16, zeroed here for an accumulating GEMM2 (aiter's sort zeroes it otherwise).
    // Null: nothing to zero.
    uint4* moe_buf;
    int32_t tok, x_stride; float scale;
};
static_assert(sizeof(Args) == 140, "kernarg ABI drift: keep in sync with Args in router.py");

static constexpr int K = 6144, E = 128, TOPK = 4, BM = 16, NT = 256;

// lane value from the DPP partner: quad_perm xor 1 / xor 2, then half-row and row mirrors (16-lane butterfly)
template <int CTRL>
__device__ __forceinline__ int dpp(int v) { return __builtin_amdgcn_mov_dpp(v, CTRL, 0xF, 0xF, false); }
template <int CTRL>
__device__ __forceinline__ void topk_step(float& bv, int& bi, float& ba) {
    const float ov = __int_as_float(dpp<CTRL>(__float_as_int(bv))), oa = __int_as_float(dpp<CTRL>(__float_as_int(ba)));
    const int oi = dpp<CTRL>(bi);
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; ba = oa; }
}

// workgroup barrier ordering LDS only: __syncthreads would also wait for this wave's outstanding global stores
__device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

__device__ __forceinline__ f32x4 mfma(u32x4 a, u32x4 b, f32x4 c) {
    return __builtin_amdgcn_mfma_f32_16x16x32_bf16(__builtin_bit_cast(bf16x8, a), __builtin_bit_cast(bf16x8, b), c, 0, 0, 0);
}

// bytes per LDS-staged weight / x row, and the staged rows' total
template <int BK, int S>
constexpr int router_rowb() { return (K + BK * S - 1) / (BK * S) * BK * 2 + 16; }
template <int BN, int BK, int S>
constexpr int router_lds() { return (BN + BM) * router_rowb<BK, S>(); }

// BN experts x 16 rows per CTA, BK-wide K steps strided by BK * S, S split partials.
template <int BN, int BK, int S>
__device__ __forceinline__ void router_body(const Args& a, const int item, uint8_t* const lds) {
    constexpr int NB = E / BN, NSUB = BN / 16, STEPS = BK / 32, NIT = (K + BK * S - 1) / (BK * S);
    constexpr int NITEMS = NB * S;
    __shared__ int s_last;
    __shared__ float s_w[BM * TOPK];
    __shared__ __attribute__((aligned(16))) int s_e[BM * TOPK];
    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6, r = lane & 15, g = lane >> 4;
    const int nb = item % NB, s = item / NB, MP = a.tok;

    // every CTA zeroes its share of the MoE output first; the GEMM2 that accumulates into it runs later on the
    // same stream
    if (a.moe_buf) {
        const int chunks = MP * K * 2 / 16;
        for (int i = item * NT + tid; i < chunks; i += NITEMS * NT) a.moe_buf[i] = uint4{0u, 0u, 0u, 0u};
    }

    constexpr int V = E * BM / NT;            // tail: logits per thread, token m = tid / G, experts c .. c + V
    constexpr int G = E / V;                  // lanes per token
    const int m = tid / G, c = (tid % G) * V;
    float bias[V];   // loaded up front: the tail would otherwise wait on it
#pragma unroll
    for (int v4 = 0; v4 < V / 4; ++v4) {
        const f32x4 b4 = *(const f32x4*)(a.bias + c + 4 * v4);
#pragma unroll
        for (int q = 0; q < 4; ++q) bias[4 * v4 + q] = b4[q];
    }

    // all four waves stage the item's weight rows and x rows in LDS; wave j < NSUB then runs subtile j's chain
    constexpr int KL = NIT * BK, ROWB = router_rowb<BK, S>();   // bytes per staged row (+16: bank skew)
    uint8_t* const s_w2 = lds;
    uint8_t* const s_x2 = lds + BN * ROWB;
    {
        constexpr int WCH = BN * KL / 8, XCH = BM * KL / 8;   // 16-byte chunks
        u32x4 tw[(WCH + NT - 1) / NT], tx[(XCH + NT - 1) / NT];
#pragma unroll
        for (int u = 0; u < (WCH + NT - 1) / NT; ++u) {
            const int cidx = tid + NT * u, row = cidx / (KL / 8), kc = (cidx % (KL / 8)) * 8;
            const int k = s * BK + (kc / BK) * BK * S + kc % BK;
            tw[u] = u32x4{0u, 0u, 0u, 0u};
            if (cidx < WCH && k < K) tw[u] = *(const u32x4*)(a.w + (size_t)(nb * BN + row) * K + k);
        }
#pragma unroll
        for (int u = 0; u < (XCH + NT - 1) / NT; ++u) {
            const int cidx = tid + NT * u, row = cidx / (KL / 8), kc = (cidx % (KL / 8)) * 8;
            const int k = s * BK + (kc / BK) * BK * S + kc % BK;
            tx[u] = u32x4{0u, 0u, 0u, 0u};
            if (cidx < XCH && row < MP && k < K) tx[u] = *(const u32x4*)(a.x + (size_t)row * a.x_stride + k);
        }
#pragma unroll
        for (int u = 0; u < (WCH + NT - 1) / NT; ++u) {
            const int cidx = tid + NT * u, row = cidx / (KL / 8), kc = (cidx % (KL / 8)) * 8;
            if (cidx < WCH) *(u32x4*)(s_w2 + row * ROWB + kc * 2) = tw[u];
        }
#pragma unroll
        for (int u = 0; u < (XCH + NT - 1) / NT; ++u) {
            const int cidx = tid + NT * u, row = cidx / (KL / 8), kc = (cidx % (KL / 8)) * 8;
            if (cidx < XCH) *(u32x4*)(s_x2 + row * ROWB + kc * 2) = tx[u];
        }
    }
    __syncthreads();
    if (wave < NSUB) {
        f32x4 acc = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int it = 0; it < NIT; ++it) {
            if (s * BK + it * BK * S < K) {
#pragma unroll
                for (int t = 0; t < STEPS; ++t) {
                    const int kb = (it * BK + 32 * t + 8 * g) * 2;
                    const u32x4 wf = *(const u32x4*)(s_w2 + (16 * wave + r) * ROWB + kb);
                    const u32x4 xf = *(const u32x4*)(s_x2 + r * ROWB + kb);
                    acc = mfma(wf, xf, acc);
                }
            }
        }
        // D[expert 4g + i][token r]
        *(f32x4*)(a.part + (size_t)(s * BM + r) * E + nb * BN + 16 * wave + 4 * g) = acc;
    }
    __syncthreads();   // also completes every wave's partial stores before the release below
    if (tid == 0) {
        const int old = __hip_atomic_fetch_add(a.cnt, 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        s_last = old == NITEMS - 1;
    }
    __syncthreads();
    if (!s_last) return;

    // real rows: the step's token count when the caller passes one, else every row
    const int M = a.n_local ? min(max(*a.n_local, 0), MP) : MP;

    // ---- reduction: logits[m][n] = ((0 + p0) + p1) + ...
    float lg[V];
#pragma unroll
    for (int v = 0; v < V; ++v) lg[v] = 0.f;
    if (m < M) {   // rows of padding tokens are never read
#pragma unroll
        for (int sp = 0; sp < S; ++sp) {
            const float* src = a.part + (size_t)(sp * BM + m) * E + c;
#pragma unroll
            for (int v4 = 0; v4 < V / 4; ++v4) {
                const f32x4 p = *(const f32x4*)(src + 4 * v4);
#pragma unroll
                for (int q = 0; q < 4; ++q) lg[4 * v4 + q] += p[q];
            }
        }
    }
    if (tid == 0) __hip_atomic_store(a.cnt, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    // ---- gate (sigmoid): act = 1 / (exp2(-x log2 e) + 1), rank by act + bias, NaN -> -1e30
    float act[V], cur[V];
#pragma unroll
    for (int v = 0; v < V; ++v) {
        act[v] = 1.0f / (__builtin_exp2f(lg[v] * -1.44269502f) + 1.0f);
        const float bv = act[v] + bias[v];
        cur[v] = bv == bv ? bv : -1e30f;
    }
    float sel_v[TOPK];
    int sel_i[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        float bv = -__builtin_inff(), ba = 0.f;
        int bi = E + 1;
#pragma unroll
        for (int v = 0; v < V; ++v)
            if (cur[v] > bv) { bv = cur[v]; bi = c + v; ba = act[v]; }
        static_assert(G == 16, "the DPP butterfly covers one 16-lane row");
        // max value, lowest expert id among equals, over the token's 16 lanes
        topk_step<0xB1>(bv, bi, ba);    // quad_perm(1, 0, 3, 2)
        topk_step<0x4E>(bv, bi, ba);    // quad_perm(2, 3, 0, 1)
        topk_step<0x141>(bv, bi, ba);   // row_half_mirror
        topk_step<0x140>(bv, bi, ba);   // row_mirror
        sel_v[k] = ba;
        sel_i[k] = bi;
#pragma unroll
        for (int v = 0; v < V; ++v)
            if (c + v == bi) cur[v] = -__builtin_inff();
    }
    float wk[TOPK];
    {
        const float sum = (sel_v[0] + sel_v[2]) + (sel_v[1] + sel_v[3]);
        const float norm = sum > 0.f ? sum : 1.0f;
#pragma unroll
        for (int k = 0; k < TOPK; ++k) wk[k] = (sel_v[k] / norm) * a.scale;
    }
    if (tid % G == 0) {   // rows of tokens >= M get a sentinel no pair matches
#pragma unroll
        for (int k = 0; k < TOPK; ++k) { s_w[m * TOPK + k] = wk[k]; s_e[m * TOPK + k] = m < M ? sel_i[k] : -1; }
        if (m < MP) {   // padding rows still get finite ids/weights: callers may read every row
            const bool real = m < M;
            *(f32x4*)(a.topk_w + m * TOPK) =
                real ? f32x4{wk[0], wk[1], wk[2], wk[3]} : f32x4{0.f, 0.f, 0.f, 0.f};
            *(int4*)(a.topk_ids + m * TOPK) =
                real ? int4{sel_i[0], sel_i[1], sel_i[2], sel_i[3]} : int4{0, 1, 2, 3};
        }
    }
    lds_barrier();

    // ---- sort (aiter layout, block_m 16): experts ascending, pairs in (token, slot) order within an expert.
    // With <= 16 tokens an expert holds <= 16 pairs, i.e. one block, so its block index is the number of distinct
    // experts below it. One wave; the pair list is read from LDS. The layout is aiter's aux sort (output_aux):
    // a route's id is (slot << 24) | token, a padding row's the plain row count MP (no slot bits), as aiter's
    // own sort writes them without a local token count.
    const int P = M * TOPK;
    if (tid < 64) {
        const bool live = tid < P;
        const int e = live ? s_e[tid] : 0, tok = tid / TOPK;
        // distinct-expert mask, 4 x 32 bits, OR-reduced over the wave
        uint32_t u[4];
#pragma unroll
        for (int wd = 0; wd < 4; ++wd) {
            uint32_t bit = (live && (e >> 5) == wd) ? 1u << (e & 31) : 0u;
            bit |= (uint32_t)dpp<0xB1>((int)bit);
            bit |= (uint32_t)dpp<0x4E>((int)bit);
            bit |= (uint32_t)dpp<0x141>((int)bit);
            bit |= (uint32_t)dpp<0x140>((int)bit);
            u[wd] = (uint32_t)__builtin_amdgcn_readlane((int)bit, 0) | (uint32_t)__builtin_amdgcn_readlane((int)bit, 16) |
                    (uint32_t)__builtin_amdgcn_readlane((int)bit, 32) | (uint32_t)__builtin_amdgcn_readlane((int)bit, 48);
        }
        int rank = 0, n = 0;   // pairs of earlier tokens / of all tokens on this expert
#pragma unroll
        for (int mq = 0; mq < BM; ++mq) {
            const int4 eq = *(const int4*)(s_e + mq * TOPK);
            const int hit = (eq.x == e) + (eq.y == e) + (eq.z == e) + (eq.w == e);
            if (mq < M) n += hit;
            if (mq < tok) rank += hit;
        }
        int before = 0, total = 0;
#pragma unroll
        for (int wd = 0; wd < 4; ++wd) {
            total += __builtin_popcount(u[wd]);
            const int lo = e - 32 * wd;
            before += lo >= 32 ? __builtin_popcount(u[wd]) : (lo > 0 ? __builtin_popcount(u[wd] & ((1u << lo) - 1u)) : 0);
        }
        const int pos = before * BM + rank;
        if (live) {
            a.sorted_ids[pos] = ((tid % TOPK) << 24) | tok;
            a.sorted_ids_ms[pos] = ((tid % TOPK) << 24) | tok;
            a.sorted_w[pos] = s_w[tid];
            a.m_indices[pos] = tok;
            if (rank == 0) {   // the expert's first pair pads its block and names it
                for (int j = n; j < BM; ++j) {
                    a.sorted_ids[before * BM + j] = MP;
                    a.sorted_ids_ms[before * BM + j] = (TOPK << 24) | MP;
                    a.sorted_w[before * BM + j] = 0.f;
                    a.m_indices[before * BM + j] = MP;
                }
                a.sorted_eids[before] = e;
            }
        }
        if (tid < MP * TOPK) a.reverse_sorted[tid] = live ? pos : -1;
        if (tid == 0) { a.num_valid[0] = total * BM; a.num_valid[1] = M; }
    }
}

template <int BN, int BK, int S>
__device__ __forceinline__ void router_kernel(const Args& a) {
    __shared__ __attribute__((aligned(16))) uint8_t lds[router_lds<BN, BK, S>()];
    router_body<BN, BK, S>(a, blockIdx.x, lds);
}
extern "C" __global__ void __launch_bounds__(256) m31_router_t8(Args a) { router_kernel<32, 256, 12>(a); }
extern "C" __global__ void __launch_bounds__(256) m31_router_t16(Args a) { router_kernel<16, 512, 8>(a); }
