# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from typing import Type

import torch
from aiter.ops.triton.attention.mla_decode import csr_to_dense_block_table
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attention_mla import MLAAttention
from atom.utils.forward_context import AttentionMetaData

from .aiter_mla import AiterMLAMetadataBuilder
from .backends import AttentionBackend

logger = logging.getLogger("atom")


class TritonMLABackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_TRITON_MLA"

    @staticmethod
    def get_builder_cls() -> Type["TritonMLAMetadataBuilder"]:
        return TritonMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> Type["MLAAttention"]:
        return MLAAttention


class TritonMLAMetadataBuilder(AiterMLAMetadataBuilder):

    def __init__(self, model_runner):
        super().__init__(model_runner)

        hf = model_runner.config.hf_config
        kv_lora_rank = hf.kv_lora_rank
        num_kv_splits = 4
        triton_mla_buffers = {
            "triton_block_table": torch.zeros(
                self.max_bs,
                self.max_num_blocks_per_seq,
                dtype=torch.int32,
                device=self.device,
            ),
            "triton_attn_logits": torch.empty(
                self.max_bs,
                self.padded_num_attention_heads,
                num_kv_splits,
                kv_lora_rank + 1,
                dtype=torch.float32,
                device=self.device,
            ),
            "triton_lse": torch.empty(
                self.max_bs,
                self.padded_num_attention_heads,
                dtype=torch.float32,
                device=self.device,
            ),
        }
        self.model_runner.forward_vars.update(triton_mla_buffers)

    def set_mla_persistent_worker_buffers(
        self, bs, max_q_len, only_update=False, num_reject_tokens=None
    ):
        # Triton MLA does not use aiter persistent worker buffers
        return {}

    def prepare_decode(self, batch: ScheduledBatch, bs: int):
        attn_metadata, positions = super().prepare_decode(batch, bs)

        scheduled_bs = batch.total_seqs_num_decode
        max_seqlen_k = attn_metadata.max_seqlen_k
        var = self.model_runner.forward_vars

        triton_bt = var["triton_block_table"][:scheduled_bs, :max_seqlen_k]
        triton_bt.zero_()
        csr_to_dense_block_table(
            attn_metadata.kv_indices,
            attn_metadata.kv_indptr,
            triton_bt,
            max_seqlen_k,
            scheduled_bs,
        )
        attn_metadata.triton_block_table = triton_bt
        attn_metadata.triton_attn_logits = var["triton_attn_logits"][:scheduled_bs]
        attn_metadata.triton_lse = var["triton_lse"][:scheduled_bs]

        return attn_metadata, positions

    def build_for_cudagraph_capture(self, bs: int) -> AttentionMetaData:
        attn_metadata, context = super().build_for_cudagraph_capture(bs)

        var = self.model_runner.forward_vars
        attn_metadata.triton_block_table = var["triton_block_table"][:bs]
        attn_metadata.triton_attn_logits = var["triton_attn_logits"][:bs]
        attn_metadata.triton_lse = var["triton_lse"][:bs]

        return attn_metadata, context
