# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from typing import Type

import torch
from aiter.ops.triton.attention.mla_decode import csr_to_dense_block_table
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attention_mla import MLAAttention
from atom.utils import envs
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

    def prepare_prefill(self, batch: ScheduledBatch):
        attn_metadata, positions = super().prepare_prefill(batch)

        if envs.ATOM_USE_TRITON_MLA_SHUFFLE_KV and attn_metadata.has_cached:
            # The shuffled cached-prefix gather (gather_kv_b_proj with
            # shuffled_kv_cache=True) reads block_size-token blocks, so it needs
            # block-granular CSR indices (logical block ids) instead of the
            # token-granular kv_indices used by the plain layout. Build them from
            # the full per-seq context (cached + just-written new tokens).
            bs = batch.total_seqs_num_prefill
            block_size = self.model_runner.block_size
            # All GPU: derive block counts from the (already on-device) full
            # context lengths and pack the dense logical block table — populated
            # by super().prepare_prefill for has_cached — into CSR via a masked
            # select (row-major == per-seq CSR order).
            var = self.model_runner.forward_vars
            ctx = attn_metadata.context_lens[:bs]  # int32 [bs], full context
            block_counts = (ctx + (block_size - 1)) // block_size  # [bs]

            indptr = torch.zeros(bs + 1, dtype=torch.int32, device=self.device)
            indptr[1:] = torch.cumsum(block_counts, dim=0).to(torch.int32)

            block_tables = var["block_tables"].gpu[:bs]  # [bs, max_blocks] logical
            col = torch.arange(block_tables.shape[1], device=self.device)
            mask = col[None, :] < block_counts[:, None]
            indices = block_tables[mask].to(torch.int32)

            attn_metadata.shuffle_kv_block_indptr = indptr
            attn_metadata.shuffle_kv_block_indices = indices
            # Attention-level chunked gather is not implemented for the shuffled
            # layout; force the single-pass gather. (This does NOT disable
            # scheduler-level chunked prefill — has_cached forwards still work.)
            attn_metadata.mla_chunk_meta = None

        return attn_metadata, positions

    def build_for_cudagraph_capture(self, bs: int) -> AttentionMetaData:
        attn_metadata, context = super().build_for_cudagraph_capture(bs)

        var = self.model_runner.forward_vars
        attn_metadata.triton_block_table = var["triton_block_table"][:bs]
        attn_metadata.triton_attn_logits = var["triton_attn_logits"][:bs]
        attn_metadata.triton_lse = var["triton_lse"][:bs]

        return attn_metadata, context
