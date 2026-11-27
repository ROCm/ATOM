# SPDX-License-Identifier: MIT
"""Check explicit FP8 drafting before loading weights or capturing graphs."""

import torch
from aiter.ops.mla_sparse_prefill import get_mla_sparse_prefill_v41_backend


def validate_fp8_draft_support(config):
    model = config.hf_config
    tp = 1 if config.enable_dp_attention else config.tensor_parallel_size
    shape = (1, model.num_attention_heads // tp, model.head_dim)
    backend = get_mla_sparse_prefill_v41_backend(
        torch.device("cuda"),
        shape,
        unified_kv_dtype=torch.float8_e4m3fn,
        softmax_scale=model.head_dim**-0.5,
        q_dtype=torch.float8_e4m3fn,
        q_scale_dtype=torch.float8_e8m0fnu,
        unified_kv_scale_dtype=torch.float8_e8m0fnu,
        kv_dtype=torch.float8_e4m3fn,
        kv_scale_dtype=torch.float8_e8m0fnu,
        layout="contiguous",
    )
    if backend is None:
        raise ValueError(
            "draft_kv_dtype=fp8 requires supported AITER V4.1 A8W8 attention. "
            "Use draft_kv_dtype=auto for BF16 drafting."
        )
