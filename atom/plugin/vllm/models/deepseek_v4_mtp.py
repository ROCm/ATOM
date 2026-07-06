"""vLLM-specific DeepSeek-V4 MTP wrapper."""

from atom.models import deepseek_v4 as deepseek_v4_base
from atom.models.deepseek_v4_mtp import DeepseekV4MTP as DeepseekV4MTPBase
from atom.plugin.vllm.models.deepseek_v4 import DeepseekV4AttentionVllm


class DeepseekV4MTP(DeepseekV4MTPBase):
    """Build native DeepSeek-V4 MTP blocks with the vLLM V4 attention variant."""

    def __init__(self, *args, **kwargs):
        original_attn_cls = deepseek_v4_base.DeepseekV4Attention
        deepseek_v4_base.DeepseekV4Attention = DeepseekV4AttentionVllm
        try:
            super().__init__(*args, **kwargs)
        finally:
            deepseek_v4_base.DeepseekV4Attention = original_attn_cls
