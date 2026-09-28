"""vLLM-specific DeepSeek-V4 MTP wrapper."""

from atom.models import deepseek_v4 as deepseek_v4_base
from atom.models.deepseek_v4_mtp import DeepseekV4MTP as DeepseekV4MTPBase
from atom.plugin.vllm.models.deepseek_v4 import DeepseekV4AttentionVllm, IndexerVllm


class DeepseekV4MTP(DeepseekV4MTPBase):
    """Build native DeepSeek-V4 MTP blocks with the vLLM V4 attention variant.

    Also rebind the ``Indexer`` global to the mixed-batch-split ``IndexerVllm``
    so the draft's sparse indexer matches the target model's behaviour under
    vLLM continuous batching / spec-verify.
    """

    def __init__(self, *args, **kwargs):
        original_attn_cls = deepseek_v4_base.DeepseekV4Attention
        original_indexer_cls = deepseek_v4_base.Indexer
        deepseek_v4_base.DeepseekV4Attention = DeepseekV4AttentionVllm
        deepseek_v4_base.Indexer = IndexerVllm
        try:
            super().__init__(*args, **kwargs)
        finally:
            deepseek_v4_base.DeepseekV4Attention = original_attn_cls
            deepseek_v4_base.Indexer = original_indexer_cls
