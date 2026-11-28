# SPDX-License-Identifier: MIT
"""Check explicit FP8 drafting before loading weights or capturing graphs."""

import torch

from atom.model_ops.attentions.deepseek_v41.fp8_attention import (
    native_fp8_attention_on,
)


def validate_fp8_draft_support(config):
    model = config.hf_config
    tp = 1 if config.enable_dp_attention else config.tensor_parallel_size
    if not native_fp8_attention_on(
        torch.device("cuda"), model.num_attention_heads // tp, model.head_dim
    ):
        raise ValueError(
            "draft_kv_dtype=fp8 requires supported AITER V4.1 A8W8 attention. "
            "Use draft_kv_dtype=auto for BF16 drafting."
        )
