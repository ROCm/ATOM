// Fused FP8 quantization + transpose + amax kernel for Lumen (HIP/ROCm).
//
// Reads BF16/FP16 input (M, N) once and writes:
//   - out_row (M, N) in FP8 row-major
//   - out_col (N, M) in FP8 (transposed)
//   - amax    (1,)   float32 max(abs(input))

#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

// Tile size (square). 64 or 128. 128 reduces block count and makes the
// transposed (scatter) writes more clustered; overridable at build time.
#ifndef LUMEN_CT_TILE
#define LUMEN_CT_TILE 128
#endif

namespace lumen {

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value)
{
    float old;
    old = (value >= 0)
              ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
              : __uint_as_float(
                    atomicMin((unsigned int*)addr, __float_as_uint(value)));
    return old;
}

// FP8 output type is templated (DTYPE_O) so the same kernel serves both
// E4M3 (forward / weights) and E5M2 (hybrid backward gradients). MI300X
// (gfx942) uses the *_fnuz variants; CUDA builds use the OCP variants. The
// saturation bound is passed in (fp8_max) since it differs per format:
//   E4M3-fnuz=240, E4M3-ocp=448, E5M2(both)=57344.

__device__ __forceinline__ float to_float(__hip_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }

// float -> FP8 via HIP conversion intrinsic. Templated on output type so both
// E4M3 and E5M2 (fnuz) work; the static_cast<OType>(float) path fails to
// resolve for E5M2 in device template instantiation, so use the intrinsic with
// an explicit interpret enum instead.
// float -> FP8 storage byte. Templated on an interpret tag so both E4M3 and
// E5M2 (fnuz) work. We keep everything at the raw uint8 storage level: the
// __hip_fp8_* struct types have no device default constructor, which breaks
// both __shared__ arrays and locals under template instantiation — so we never
// materialize those structs, only their 1-byte storage.
enum class Fp8Fmt { E4M3_FNUZ, E5M2_FNUZ };

template <Fp8Fmt FMT>
__device__ __forceinline__ unsigned char float_to_fp8_bits(float x)
{
    if (FMT == Fp8Fmt::E4M3_FNUZ)
        return __hip_cvt_float_to_fp8(x, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
    else
        return __hip_cvt_float_to_fp8(x, __HIP_SATFINITE, __HIP_E5M2_FNUZ);
}

// 64x64 tile, 256 threads. Vectorized: each thread processes 8-element runs.
//
// Memory access is the bottleneck (a naive scalar version reaches only ~31% of
// HBM bandwidth because it issues 2B loads / 1B stores per element). v3 loads
// 8 BF16 at once (int4 = 16B) and stores 8 FP8 at once (uint64 = 8B) for BOTH
// the row-major output and the transposed output — turning 64 strided 1-byte
// column writes into 8 vectorized writes.
//
// A full-tile fast path handles the common case (tile fully inside the matrix
// and N % 8 == 0 for alignment); a scalar fallback covers ragged edges so the
// kernel stays correct for arbitrary shapes.
// Shared memory is padded (+1 column) to avoid bank conflicts on transpose read.
template <typename DTYPE_I, Fp8Fmt FMT>
__global__ void fused_quant_transpose_amax_kernel_v2(
    unsigned char* __restrict__ out_row,
    unsigned char* __restrict__ out_col,
    float* __restrict__ amax_out,
    DTYPE_I const* __restrict__ input,
    float const* __restrict__ scale,
    float const FP8_MAX_VAL,
    int64_t const M,
    int32_t const N)
{
    constexpr int TILE = LUMEN_CT_TILE;
    constexpr int BLOCK = 256;
    constexpr int VEC = 8;                    // 8 BF16 = int4 load; 8 FP8 = uint64 store
    constexpr int COLS_V = TILE / VEC;        // vectors per row
    constexpr int VPT = (TILE * TILE / VEC) / BLOCK;  // vectors per thread

    int const tile_row = blockIdx.x;
    int const tile_col = blockIdx.y;
    int const tid = threadIdx.x;

    int const row_start = tile_row * TILE;
    int const col_start = tile_col * TILE;
    float const inv_scale = 1.0f / (*scale);

    // +1 padding to avoid shared memory bank conflicts during column reads
    __shared__ unsigned char smem[TILE][TILE + 1];

    float thread_amax = 0.0f;

    bool const full =
        (row_start + TILE <= M) && (col_start + TILE <= N) && ((N % VEC) == 0);

    if (full)
    {
        // Phase 1 (vectorized): int4 load -> quantize 8 -> uint64 row store + smem.
        #pragma unroll
        for (int i = 0; i < VPT; ++i)
        {
            int v = tid + i * BLOCK;          // 0..511
            int lr = v / COLS_V;              // tile row 0..63
            int lc = (v % COLS_V) * VEC;      // tile col base 0,8,...,56
            int64_t gr = row_start + lr;
            int gc = col_start + lc;

            int4 raw = *reinterpret_cast<int4 const*>(&input[gr * N + gc]);
            DTYPE_I const* hv = reinterpret_cast<DTYPE_I const*>(&raw);

            uint64_t packed = 0;
            #pragma unroll
            for (int k = 0; k < VEC; ++k)
            {
                float val = to_float(hv[k]);
                thread_amax = fmaxf(thread_amax, fabsf(val));
                float s = fminf(fmaxf(val * inv_scale, -FP8_MAX_VAL), FP8_MAX_VAL);
                unsigned char q = float_to_fp8_bits<FMT>(s);
                smem[lr][lc + k] = q;
                packed |= static_cast<uint64_t>(q) << (8 * k);
            }
            *reinterpret_cast<uint64_t*>(&out_row[gr * N + gc]) = packed;
        }

        __syncthreads();

        // Phase 2 (vectorized): read a smem column, pack 8, uint64 transposed store.
        #pragma unroll
        for (int i = 0; i < VPT; ++i)
        {
            int v = tid + i * BLOCK;
            int lc = v / COLS_V;              // tile col 0..63 (row of out_col)
            int lr = (v % COLS_V) * VEC;      // tile row base 0,8,...
            int64_t gc = col_start + lc;
            int gr = row_start + lr;

            uint64_t packed = 0;
            #pragma unroll
            for (int k = 0; k < VEC; ++k)
                packed |= static_cast<uint64_t>(smem[lr + k][lc]) << (8 * k);

            *reinterpret_cast<uint64_t*>(&out_col[gc * M + gr]) = packed;
        }
    }
    else
    {
        // Scalar fallback for ragged edge tiles.
        constexpr int ITERS = TILE * TILE / BLOCK;  // 16
        #pragma unroll
        for (int i = 0; i < ITERS; ++i)
        {
            int idx = tid + i * BLOCK;
            int lr = idx / TILE;
            int lc = idx % TILE;
            int64_t gr = row_start + lr;
            int gc = col_start + lc;

            float val = 0.0f;
            if (gr < M && gc < N)
                val = to_float(input[gr * N + gc]);
            thread_amax = fmaxf(thread_amax, fabsf(val));
            float s = fminf(fmaxf(val * inv_scale, -FP8_MAX_VAL), FP8_MAX_VAL);
            unsigned char q = float_to_fp8_bits<FMT>(s);
            if (gr < M && gc < N)
                out_row[gr * N + gc] = q;
            smem[lr][lc] = q;
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < ITERS; ++i)
        {
            int idx = tid + i * BLOCK;
            int lr = idx / TILE;
            int lc = idx % TILE;
            int gr = row_start + lc;
            int64_t gc = col_start + lr;
            if (gr < M && gc < N)
                out_col[gc * M + gr] = smem[lc][lr];
        }
    }

    // Warp-level reduction for amax
    for (int offset = 32; offset > 0; offset >>= 1)
        thread_amax = fmaxf(thread_amax, __shfl_xor(thread_amax, offset, 64));

    // Per-warp amax into shared
    __shared__ float warp_max[4];
    int lane = tid % 64;
    int wid = tid / 64;
    if (lane == 0) warp_max[wid] = thread_amax;
    __syncthreads();

    if (tid < 4)
    {
        float val = warp_max[tid];
        for (int offset = 2; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_xor(val, offset, 64));
        if (tid == 0)
            atomicMaxFloat(amax_out, val);
    }
}

void static_quant_transpose_amax(
    torch::Tensor& out_row,
    torch::Tensor& out_col,
    torch::Tensor& amax_out,
    torch::Tensor const& input,
    torch::Tensor const& scale)
{
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(out_row.scalar_type() == out_col.scalar_type(),
                "out_row and out_col must share dtype");

    int64_t const M = input.size(0);
    int32_t const N = input.size(1);

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    constexpr int TILE = LUMEN_CT_TILE;
    dim3 grid((M + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    dim3 block(256);

#define LUMEN_LAUNCH_CT(ITYPE_C, FMT_C, MAXV)                                     \
    fused_quant_transpose_amax_kernel_v2<ITYPE_C, FMT_C>                          \
        <<<grid, block, 0, stream>>>(                                             \
            reinterpret_cast<unsigned char*>(out_row.data_ptr()),                 \
            reinterpret_cast<unsigned char*>(out_col.data_ptr()),                 \
            amax_out.data_ptr<float>(),                                           \
            reinterpret_cast<ITYPE_C const*>(input.data_ptr()),                   \
            scale.data_ptr<float>(),                                              \
            static_cast<float>(MAXV), M, N)

#define LUMEN_DISPATCH_OUT(ITYPE_C)                                               \
    do {                                                                         \
        auto ot = out_row.scalar_type();                                         \
        if (ot == at::ScalarType::Float8_e4m3fnuz)                               \
            LUMEN_LAUNCH_CT(ITYPE_C, Fp8Fmt::E4M3_FNUZ, 240.0f);                \
        else if (ot == at::ScalarType::Float8_e5m2fnuz)                          \
            LUMEN_LAUNCH_CT(ITYPE_C, Fp8Fmt::E5M2_FNUZ, 57344.0f);              \
        else                                                                     \
            TORCH_CHECK(false, "Unsupported output fp8 dtype (gfx942 fnuz only): ", ot); \
    } while (0)

    if (input.scalar_type() == at::ScalarType::BFloat16)
        LUMEN_DISPATCH_OUT(__hip_bfloat16);
    else if (input.scalar_type() == at::ScalarType::Half)
        LUMEN_DISPATCH_OUT(__half);
    else
        TORCH_CHECK(false, "Unsupported input dtype: ", input.scalar_type());

#undef LUMEN_DISPATCH_OUT
#undef LUMEN_LAUNCH_CT
}

} // namespace lumen

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("static_quant_transpose_amax",
          &lumen::static_quant_transpose_amax,
          py::arg("out_row"),
          py::arg("out_col"),
          py::arg("amax_out"),
          py::arg("input"),
          py::arg("scale"));
}
