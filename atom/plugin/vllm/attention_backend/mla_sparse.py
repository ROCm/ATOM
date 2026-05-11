# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Type

import torch
from atom.model_ops.attention_mla import _MLA_MIN_HEADS
from atom.plugin.vllm.attention.metadata import (
    get_max_prefill_buffer_size,
    vllmMLASparseAttentionMetadataBuilderMethods,
    vllmMLASparseIndexerAttentionMetadataBuilderMethods,
)
from atom.plugin.vllm.attention.backend import AiterMlaBackendForVllm
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
)


class AiterMLASparseBackend(AiterMlaBackendForVllm):
    """
    Sparse MLA attention backend for main attention layers to provide sparse
    metadata builder for top-k index conversion and ragged kernel call.
    """

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_builder_cls() -> Type["AiterMLASparseMetadataBuilder"]:
        return AiterMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls():
        from atom.plugin.vllm.attention.layer import AttentionForVllmMLA

        return AttentionForVllmMLA

    @classmethod
    def is_sparse(cls) -> bool:
        return True


class AiterMLASparseMetadataBuilder(
    vllmMLASparseAttentionMetadataBuilderMethods, AttentionMetadataBuilder
):
    """vLLM-only metadata builder for sparse MLA main attention."""

    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    reorder_batch_threshold = 1

    def __init__(
        self,
        kv_cache_spec=None,
        layer_names=None,
        config=None,
        device=None,
        model_runner=None,
    ):
        super().__init__(kv_cache_spec, layer_names, config, device)
        from vllm.config import VllmConfig
        from vllm.model_executor.layers.attention.mla_attention import (
            get_mla_dims,
        )

        assert isinstance(config, VllmConfig)

        self.vllm_config = config
        self.model_config = config.model_config
        self.kv_cache_spec = kv_cache_spec
        self.device = device
        max_num_batched_tokens = config.scheduler_config.max_num_batched_tokens

        parallel_config = config.parallel_config
        self.num_heads = self.model_config.get_num_attention_heads(parallel_config)
        self.padded_num_heads = max(self.num_heads, _MLA_MIN_HEADS)
        self.mla_dims = get_mla_dims(self.model_config)
        self.topk_tokens = config.model_config.hf_config.index_topk
        self.topk_tokens_tensor = torch.tensor(
            [self.topk_tokens], device=device, dtype=torch.int32
        )
        self.max_model_len_tensor = torch.tensor(
            [self.model_config.max_model_len], device=device, dtype=torch.int32
        )
        self.dummy_block_table = torch.empty(
            (1, 1), dtype=torch.int32, device=self.device
        )

        self.req_id_per_token_buffer = torch.empty(
            (max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )
        self.qo_indptr = torch.arange(
            0, max_num_batched_tokens + 1, dtype=torch.int32, device=device
        )
        self.paged_kv_last_page_len = torch.ones(
            max_num_batched_tokens, dtype=torch.int32, device=device
        )
        self.paged_kv_indices = torch.zeros(
            [max_num_batched_tokens * self.topk_tokens],
            dtype=torch.int32,
            device=device,
        )
        self.paged_kv_indptr = torch.zeros(
            [max_num_batched_tokens + 1], dtype=torch.int32, device=device
        )


class AiterMLASparseIndexerBackend(AiterMlaBackendForVllm):

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_builder_cls() -> Type["AiterMLASparseIndexerMetadataBuilder"]:
        return AiterMLASparseIndexerMetadataBuilder

    @staticmethod
    def get_impl_cls():
        from atom.plugin.vllm.attention.layer import AttentionForVllmMLA

        return AttentionForVllmMLA

    @classmethod
    def is_sparse(cls) -> bool:
        return True


class AiterMLASparseIndexerMetadataBuilder(
    vllmMLASparseIndexerAttentionMetadataBuilderMethods, AttentionMetadataBuilder
):
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    reorder_batch_threshold = 1

    def __init__(
        self,
        kv_cache_spec=None,
        layer_names=None,
        config=None,
        device=None,
        model_runner=None,
    ):
        super().__init__(kv_cache_spec, layer_names, config, device)
        from vllm.config import VllmConfig

        try:
            from vllm.utils.platform_utils import num_compute_units
        except ImportError:
            from vllm.utils.platform_utils import get_cu_count as num_compute_units
        from vllm.v1.worker.cp_utils import get_total_cp_world_size
        from vllm.utils.math_utils import cdiv

        assert isinstance(config, VllmConfig)

        self.vllm_config = config
        self.model_config = config.model_config
        self.kv_cache_spec = kv_cache_spec
        self.device = device
        max_num_batched_tokens = config.scheduler_config.max_num_batched_tokens

        self.max_prefill_buffer_size = get_max_prefill_buffer_size(
            self.model_config.max_model_len
        )
        self.num_speculative_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config
            else 0
        )
        self.reorder_batch_threshold += self.num_speculative_tokens

        sm_count = num_compute_units(self.device.index)
        self.num_sms = sm_count

        self.decode_lens_buffer = torch.empty(
            (max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        self.arange_buffer = torch.arange(
            config.scheduler_config.max_num_seqs * (1 + self.num_speculative_tokens),
            dtype=torch.int32,
            device=self.device,
        )
        self.expanded_seq_lens_buffer = torch.zeros(
            (max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        max_num_blocks_per_req = cdiv(
            self.vllm_config.model_config.max_model_len,
            self.kv_cache_spec.block_size * get_total_cp_world_size(),
        )
        self.expanded_block_table_buffer = torch.zeros(
            (
                max_num_batched_tokens,
                max_num_blocks_per_req,
            ),
            dtype=torch.int32,
            device=self.device,
        )
        self.scheduler_metadata_buffer = torch.empty(
            (self.num_sms + 1, 2), dtype=torch.int32, device=self.device
        )
