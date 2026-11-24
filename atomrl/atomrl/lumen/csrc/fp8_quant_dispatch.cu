// C++ FP8 quantization dispatch — fused scale+quant+amax in a single call.
//
// Replaces the Python hot path:
//   get_scale() → _compute_scale_kernel → static_quant_with_amax → update_amax
// with a single C++ function that owns amax history, computes scale inline,
// launches one HIP kernel, and updates history.
//
// Only targets: delayed scaling + MOST_RECENT amax algo + CUDA tensors.
// Gated by LUMEN_CPP_QUANT_DISPATCH=1.

#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>

#include <deque>
#include <string>
#include <tuple>
#include <unordered_map>

namespace lumen {

// --- Device helpers ---

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value)
{
    float old;
    old = (value >= 0)
              ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
              : __uint_as_float(
                    atomicMin((unsigned int*)addr, __float_as_uint(value)));
    return old;
}

__device__ __forceinline__ float to_float(__hip_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }

// FP8 conversion via clamp+reinterpret: store as uint8_t
// E4M3FNUZ: clamp to [-240, 240], cast via __hip_bfloat16 roundtrip
// We use the simple approach: clamp → round to nearest representable FP8 value
// via the __builtin intrinsics available in ROCm.

// For gfx942/gfx950, use packed FP8 conversion intrinsic.
// Fallback: manual clamp + truncation via __hip_cvt_float_to_fp8

// E4M3FNUZ saturation flag
#if defined(__gfx942__) || defined(__gfx950__)
#define HAS_FP8_INTRINSIC 1
#endif

// Device function: convert float to FP8 E4M3FNUZ as uint8_t
__device__ __forceinline__ uint8_t float_to_fp8_e4m3fnuz(float val)
{
#ifdef HAS_FP8_INTRINSIC
    // Use packed conversion: convert 1 float via the byte-extract approach
    // __builtin_amdgcn_cvt_pk_fp8_f32: converts 2 floats to packed FP8
    union { unsigned int u32; uint8_t u8[4]; } cvt;
    cvt.u32 = __builtin_amdgcn_cvt_pk_fp8_f32(val, 0.0f, 0, false); // word=0, negate=false
    return cvt.u8[0];
#else
    // Fallback: clamp and use __hip_cvt_float_to_fp8 if available
    // For non-gfx942, this path shouldn't be reached in production
    return 0;
#endif
}

// Device function: convert float to FP8 E5M2FNUZ as uint8_t
__device__ __forceinline__ uint8_t float_to_fp8_e5m2fnuz(float val)
{
#ifdef HAS_FP8_INTRINSIC
    union { unsigned int u32; uint8_t u8[4]; } cvt;
    cvt.u32 = __builtin_amdgcn_cvt_pk_bf8_f32(val, 0.0f, 0, false);
    return cvt.u8[0];
#else
    return 0;
#endif
}


// --- Fused scale+quant+amax kernel (row-based, no transpose) ---
//
// Each block handles one row. Computes:
//   scale = (prev_amax > 0) ? prev_amax * eff_max_recip : 1.0
//   out[row] = clamp(input[row] / scale, -FP8_MAX, FP8_MAX)
//   amax_out = max(abs(input)) across all rows
//   scale_out = scale
//
// Template parameter IS_E4M3: true for forward (E4M3FNUZ), false for backward (E5M2FNUZ)

template <typename DTYPE_I, bool IS_E4M3, int BLOCK_SIZE>
__global__ void fused_scale_quant_amax_kernel(
    uint8_t*       __restrict__ out,
    float*         __restrict__ amax_out,
    float*         __restrict__ scale_out,
    DTYPE_I const* __restrict__ input,
    float const*   __restrict__ prev_amax_ptr,
    float          eff_max_recip,
    float          fp8_max_val,
    int64_t        M,
    int32_t        N)
{
    int const row = blockIdx.x;
    if (row >= M) return;

    // Compute scale from previous step's amax (redundant across threads, cheap)
    float prev_amax = *prev_amax_ptr;
    float scale = (prev_amax > 0.0f) ? prev_amax * eff_max_recip : 1.0f;
    float inv_scale = 1.0f / scale;

    // Write scale once
    if (row == 0 && threadIdx.x == 0)
        *scale_out = scale;

    // Process columns: quant + track amax
    float thread_amax = 0.0f;
    DTYPE_I const* row_ptr = input + (int64_t)row * N;
    uint8_t*       out_ptr = out + (int64_t)row * N;

    for (int c = threadIdx.x; c < N; c += BLOCK_SIZE) {
        float val = to_float(row_ptr[c]);
        float abs_val = fabsf(val);
        thread_amax = fmaxf(thread_amax, abs_val);

        // Clamp then convert to FP8
        float scaled = fminf(fmaxf(val * inv_scale, -fp8_max_val), fp8_max_val);
        if constexpr (IS_E4M3) {
            out_ptr[c] = float_to_fp8_e4m3fnuz(scaled);
        } else {
            out_ptr[c] = float_to_fp8_e5m2fnuz(scaled);
        }
    }

    // Warp-level reduction (wavefront=64 on MI300X)
    for (int offset = 32; offset > 0; offset >>= 1)
        thread_amax = fmaxf(thread_amax, __shfl_xor(thread_amax, offset, 64));

    // Block-level reduction via shared memory
    constexpr int N_WARPS = BLOCK_SIZE / 64;
    __shared__ float warp_max[N_WARPS];
    int lane = threadIdx.x % 64;
    int wid = threadIdx.x / 64;
    if (lane == 0)
        warp_max[wid] = thread_amax;
    __syncthreads();

    if (threadIdx.x < N_WARPS) {
        float val = warp_max[threadIdx.x];
        // Final reduction among warps
        for (int offset = N_WARPS / 2; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_xor(val, offset, 64));
        if (threadIdx.x == 0)
            atomicMaxFloat(amax_out, val);
    }
}

// --- Kernel launcher (handles input dtype x FP8 format dispatch) ---

template <bool IS_E4M3>
void launch_fused_scale_quant_amax(
    at::Tensor& out,
    at::Tensor& amax_out,
    at::Tensor& scale_out,
    at::Tensor const& input,
    at::Tensor const& prev_amax,
    float eff_max_recip,
    float fp8_max_val)
{
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    int64_t const M = input.size(0);
    int32_t const N = input.size(1);

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    constexpr int BLOCK = 256;
    dim3 grid(M);
    dim3 block(BLOCK);

    if (input.scalar_type() == at::ScalarType::BFloat16) {
        fused_scale_quant_amax_kernel<__hip_bfloat16, IS_E4M3, BLOCK>
            <<<grid, block, 0, stream>>>(
                reinterpret_cast<uint8_t*>(out.data_ptr()),
                amax_out.data_ptr<float>(),
                scale_out.data_ptr<float>(),
                reinterpret_cast<__hip_bfloat16 const*>(input.data_ptr()),
                prev_amax.data_ptr<float>(),
                eff_max_recip,
                fp8_max_val,
                M, N);
    } else if (input.scalar_type() == at::ScalarType::Half) {
        fused_scale_quant_amax_kernel<__half, IS_E4M3, BLOCK>
            <<<grid, block, 0, stream>>>(
                reinterpret_cast<uint8_t*>(out.data_ptr()),
                amax_out.data_ptr<float>(),
                scale_out.data_ptr<float>(),
                reinterpret_cast<__half const*>(input.data_ptr()),
                prev_amax.data_ptr<float>(),
                eff_max_recip,
                fp8_max_val,
                M, N);
    } else {
        TORCH_CHECK(false, "Unsupported input dtype: ", input.scalar_type());
    }
}

// --- FP8QuantDispatcher class ---

struct FP8QuantDispatcher {
    float fp8_max_fwd_;
    float fp8_max_bwd_;
    float eff_max_recip_fwd_;
    float eff_max_recip_bwd_;
    int   history_len_;

    // Per-tensor amax history
    std::unordered_map<std::string, std::deque<at::Tensor>> history_;

    // Pre-allocated scratch (lazily initialized per device)
    at::Tensor amax_scratch_;
    at::Tensor scale_scratch_;
    at::Tensor fp8_max_fwd_tensor_;
    at::Tensor fp8_max_bwd_tensor_;
    bool scratch_initialized_ = false;

    FP8QuantDispatcher(
        double fp8_max_fwd,
        double fp8_max_bwd,
        int margin,
        int history_len)
        : fp8_max_fwd_(static_cast<float>(fp8_max_fwd)),
          fp8_max_bwd_(static_cast<float>(fp8_max_bwd)),
          history_len_(history_len)
    {
        float divisor = static_cast<float>(1 << margin);
        eff_max_recip_fwd_ = 1.0f / (fp8_max_fwd_ / divisor);
        eff_max_recip_bwd_ = 1.0f / (fp8_max_bwd_ / divisor);
    }

    void ensure_scratch(at::Device device) {
        if (scratch_initialized_ && amax_scratch_.device() == device)
            return;
        auto opts = at::TensorOptions().dtype(at::kFloat).device(device);
        amax_scratch_ = at::zeros({1}, opts);
        scale_scratch_ = at::zeros({1}, opts);
        fp8_max_fwd_tensor_ = at::full({1}, fp8_max_fwd_, opts);
        fp8_max_bwd_tensor_ = at::full({1}, fp8_max_bwd_, opts);
        scratch_initialized_ = true;
    }

    // Core hot path: replaces get_scale + quant + update_amax
    std::tuple<at::Tensor, at::Tensor> quantize(
        const std::string& tensor_id,
        at::Tensor input,
        bool backward)
    {
        ensure_scratch(input.device());

        float fp8_max = backward ? fp8_max_bwd_ : fp8_max_fwd_;
        float eff_max_recip = backward ? eff_max_recip_bwd_ : eff_max_recip_fwd_;

        // 1. Lookup amax history
        auto it = history_.find(tensor_id);
        at::Tensor prev_amax;
        if (it == history_.end() || it->second.empty()) {
            // First call: use fp8_max as amax → scale = 1.0
            prev_amax = backward ? fp8_max_bwd_tensor_ : fp8_max_fwd_tensor_;
        } else {
            prev_amax = it->second.back();
            // Ensure on correct device
            if (prev_amax.device() != input.device())
                prev_amax = prev_amax.to(input.device());
        }

        // 2. Prepare input as contiguous 2D
        auto orig_sizes = input.sizes().vec();
        at::Tensor input_2d;
        if (input.dim() == 2) {
            input_2d = input.contiguous();
        } else {
            input_2d = input.reshape({-1, input.size(-1)}).contiguous();
        }
        int64_t M = input_2d.size(0);
        int32_t N = input_2d.size(1);

        // 3. Allocate output (FP8 = uint8 storage)
        auto fp8_dtype = backward ? at::kFloat8_e5m2fnuz : at::kFloat8_e4m3fnuz;
        auto out = at::empty({M, N}, input_2d.options().dtype(fp8_dtype));

        // 4. Zero amax scratch (hipMemsetAsync avoids aten::fill_ dispatch overhead)
        hipMemsetAsync(amax_scratch_.data_ptr<float>(), 0, sizeof(float),
                       at::hip::getCurrentHIPStream());

        // 5. Launch fused kernel
        if (backward) {
            launch_fused_scale_quant_amax<false>(
                out, amax_scratch_, scale_scratch_, input_2d,
                prev_amax, eff_max_recip, fp8_max);
        } else {
            launch_fused_scale_quant_amax<true>(
                out, amax_scratch_, scale_scratch_, input_2d,
                prev_amax, eff_max_recip, fp8_max);
        }

        // 6. Update history
        at::Tensor new_amax = amax_scratch_.clone();
        auto& deq = history_[tensor_id];
        deq.push_back(new_amax);
        while (static_cast<int>(deq.size()) > history_len_)
            deq.pop_front();

        // 7. Return (fp8_data, scale)
        at::Tensor fp8_data = (input.dim() == 2) ? out : out.view(orig_sizes);
        at::Tensor scale = scale_scratch_.clone();

        return std::make_tuple(fp8_data, scale);
    }

    void reset() {
        history_.clear();
    }

    // Export history to Python dict for debugging / checkpoint
    py::dict export_history() const {
        py::dict d;
        for (auto const& [tid, deq] : history_) {
            py::list l;
            for (auto const& t : deq)
                l.append(t);
            d[py::cast(tid)] = l;
        }
        return d;
    }

    // Import history from Python dict
    void import_history(py::dict d) {
        history_.clear();
        for (auto& item : d) {
            std::string tid = py::cast<std::string>(item.first);
            py::list l = py::cast<py::list>(item.second);
            auto& deq = history_[tid];
            for (auto& t : l)
                deq.push_back(py::cast<at::Tensor>(t));
        }
    }
};

} // namespace lumen

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Lumen FP8 C++ quantization dispatch — fused scale+quant+amax";

    py::class_<lumen::FP8QuantDispatcher>(m, "FP8QuantDispatcher")
        .def(py::init<double, double, int, int>(),
             py::arg("fp8_max_fwd"),
             py::arg("fp8_max_bwd"),
             py::arg("margin"),
             py::arg("history_len"))
        .def("quantize", &lumen::FP8QuantDispatcher::quantize,
             py::arg("tensor_id"),
             py::arg("input"),
             py::arg("backward"))
        .def("reset", &lumen::FP8QuantDispatcher::reset)
        .def("export_history", &lumen::FP8QuantDispatcher::export_history)
        .def("import_history", &lumen::FP8QuantDispatcher::import_history);
}
