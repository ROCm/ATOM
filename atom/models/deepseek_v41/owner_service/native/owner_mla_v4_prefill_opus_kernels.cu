// SPDX-License-Identifier: MIT
// Reuse the source-aligned native validation, device templates and reduction.
// This translation unit only adds a four-head-tile launch policy. The include
// path and compiler flags are inherited from the unchanged native AITER op.
#include "mla_v4_prefill_opus_kernels.cu"
#include "owner_grouped_attention.h"

void owner_grouped_attention_fwd(
    aiter_tensor_t& q, aiter_tensor_t& unified_kv,
    aiter_tensor_t& kv_indices_prefix, aiter_tensor_t& kv_indptr_prefix,
    aiter_tensor_t& kv, aiter_tensor_t& kv_indices_extend,
    aiter_tensor_t& kv_indptr_extend, aiter_tensor_t& attn_sink,
    aiter_tensor_t& partial_o, aiter_tensor_t& partial_max,
    aiter_tensor_t& partial_sum, aiter_tensor_t& out,
    float softmax_scale, int num_splits)
{
    const opus_mla_v4_prefill_kargs kargs = opus_mla_v4_prefill_a16w16_kargs(
        q, unified_kv, kv_indices_prefix, kv_indptr_prefix, kv, kv_indices_extend,
        kv_indptr_extend, attn_sink, out, softmax_scale);
    const int N = kargs.N;
    const int H = kargs.H;
    const int D = kargs.D;
    AITER_CHECK(H == 64 && q.dtype() == AITER_DTYPE_bf16,
                "Grouped owner attention is BF16 with four original H16 shards");
    AITER_CHECK(q.is_contiguous() && out.is_contiguous(),
                "Grouped owner Q/output must have the same dense layout");
    AITER_CHECK(num_splits == 1 || num_splits == 4,
                "Preserve the native H16 automatic split policy");
    if (N == 0) return;
    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    if (num_splits == 1) {
        using Traits = opus_mla_v4_prefill_a16w16_16mx1_16nx4_traits<16, 64, 512, 4, bf16_t>;
        dim3 grid(N, ceil_div(H, Traits::Q_TILE_SIZE * Traits::T_M), 1);
        opus_mla_v4_prefill_a16w16_16mx1_16nx4_kernel<Traits>
            <<<grid, Traits::BLOCK_SIZE, 0, stream>>>(kargs);
        HIP_CALL_LAUNCH(hipGetLastError());
        return;
    }

    for (aiter_tensor_t* t : {&partial_o, &partial_max, &partial_sum})
        AITER_CHECK(t->dtype() == AITER_DTYPE_fp32 && t->is_contiguous(),
                    "Grouped split scratch must be contiguous FP32");
    AITER_CHECK(partial_o.numel() >= static_cast<int64_t>(N) * num_splits * H * D &&
                    partial_max.numel() >= static_cast<int64_t>(N) * num_splits * H &&
                    partial_sum.numel() >= static_cast<int64_t>(N) * num_splits * H,
                "Grouped split scratch must retain all heads and native splits");
    const opus_mla_v4_prefill_split_kargs split_kargs{
        kargs, reinterpret_cast<float*>(partial_o.data_ptr()),
        reinterpret_cast<float*>(partial_max.data_ptr()),
        reinterpret_cast<float*>(partial_sum.data_ptr()), num_splits};
    using Traits = opus_mla_v4_prefill_a16w16_16mx1_16nx4_traits<16, 64, 512, 4, bf16_t, true>;
    dim3 grid(N, ceil_div(H, Traits::Q_TILE_SIZE * Traits::T_M), num_splits);
    opus_mla_v4_prefill_a16w16_16mx1_16nx4_kernel<Traits>
        <<<grid, Traits::BLOCK_SIZE, 0, stream>>>(split_kargs);
    HIP_CALL_LAUNCH(hipGetLastError());
    // The native reducer treats heads independently; only H's storage stride
    // expands from 16 to 64. No extra summation crosses a head boundary.
    aiter::pa_decode_ps_reduce_hip_kernel<__hip_bfloat16, float, float, false, 512, 1, 4>
        <<<dim3(N, H, 1), 512, 0, stream>>>(
            reinterpret_cast<__hip_bfloat16*>(out.data_ptr()),
            split_kargs.partial_sum, split_kargs.partial_max, split_kargs.partial_o,
            nullptr, kargs.stride_qo_n, 0, kargs.stride_qo_h, 0,
            num_splits * H, 1, H, num_splits * H * D, D, H * D, 0);
    HIP_CALL_LAUNCH(hipGetLastError());
}
