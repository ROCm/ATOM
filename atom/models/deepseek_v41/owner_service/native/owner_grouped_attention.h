// SPDX-License-Identifier: MIT
#pragma once
#include "aiter_tensor.h"

void owner_grouped_attention_fwd(
    aiter_tensor_t& q, aiter_tensor_t& unified_kv,
    aiter_tensor_t& kv_indices_prefix, aiter_tensor_t& kv_indptr_prefix,
    aiter_tensor_t& kv, aiter_tensor_t& kv_indices_extend,
    aiter_tensor_t& kv_indptr_extend, aiter_tensor_t& attn_sink,
    aiter_tensor_t& partial_o, aiter_tensor_t& partial_max,
    aiter_tensor_t& partial_sum, aiter_tensor_t& out,
    float softmax_scale, int num_splits);
