# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging

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
    def get_builder_cls() -> type["TritonMLAMetadataBuilder"]:
        return TritonMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["MLAAttention"]:
        return MLAAttention


class TritonMLAMetadataBuilder(AiterMLAMetadataBuilder):

    def __init__(self, model_runner):
        super().__init__(model_runner)

        hf = model_runner.config.hf_config
        kv_lora_rank = hf.kv_lora_rank
        num_kv_splits = 4
        triton_mla_buffers = {
            # int64, not int32: aiter's Triton decode multiplies each KV row id by
            # the row stride (576) in the id's own dtype. An int32 offset wraps past
            # row 2**31 / 576 = 3,728,270 (block 233,016 of 16 tokens), and the
            # decode then reads outside the KV cache.
            "triton_block_table": torch.zeros(
                self.max_bs,
                self.max_num_blocks_per_seq,
                dtype=torch.int64,
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
        self, bs, max_q_len, only_update=False, num_reject_tokens=None, **kwargs
    ):
        # Triton MLA does not use aiter persistent worker buffers.
        # `**kwargs` absorbs arguments the aiter caller grows over time (e.g.
        # `is_cp_round_robin`); this override returns nothing, so ignoring them
        # is correct, and pinning the exact signature only breaks the Triton
        # path whenever the aiter side adds a keyword.
        return {}

    def _cut_decode_buffers(
        self,
        rows: int,
        max_seqlen_k: int,
        kv_indices: torch.Tensor,
        kv_indptr: torch.Tensor,
    ) -> dict:
        """The three buffers `attention_mla`'s Triton branch reads, cut to `rows`.

        One implementation for both entry points, because the defect was that
        only `prepare_decode` cut them: a mid-step then ran its own wider batch
        against the verify forward's cut. Costs a zero plus a densify per
        mid-step, which that path did not pay before.
        """
        var = self.model_runner.forward_vars
        block_table = var["triton_block_table"][:rows, :max_seqlen_k]
        block_table.zero_()
        csr_to_dense_block_table(kv_indices, kv_indptr, block_table, max_seqlen_k, rows)
        return {
            "triton_block_table": block_table,
            "triton_attn_logits": var["triton_attn_logits"][:rows],
            "triton_lse": var["triton_lse"][:rows],
        }

    def prepare_decode(
        self,
        batch: ScheduledBatch,
        running_bs: int,
        running_tokens: int,
        max_seqlen_q: int,
    ):
        attn_metadata, positions = super().prepare_decode(
            batch, running_bs, running_tokens, max_seqlen_q
        )
        for name, buf in self._cut_decode_buffers(
            batch.total_seqs_num_decode,
            attn_metadata.max_seqlen_k,
            attn_metadata.kv_indices,
            attn_metadata.kv_indptr,
        ).items():
            setattr(attn_metadata, name, buf)
        return attn_metadata, positions

    def prepare_mtp_decode(
        self,
        bs: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        positions: torch.Tensor,
        only_update: bool = False,
        num_reject_tokens: torch.Tensor | None = None,
        **fused,
    ):
        """Re-cut the Triton decode buffers off the `kv_indptr` the base rebuilt.

        `**fused` rather than the base's three keyword-only arguments spelled
        out again: this class inherits `fuse_mtp_decode_position_update`, so
        `EagleProposer` passes them, and an override that restated them could
        fall behind the base without anything noticing until the call raised.
        """
        result = super().prepare_mtp_decode(
            bs,
            max_seqlen_q,
            max_seqlen_k,
            positions,
            only_update,
            num_reject_tokens,
            **fused,
        )
        running_bs = int(positions.shape[-1])  # rows; see the base contract
        var = self.model_runner.forward_vars
        return result | self._cut_decode_buffers(
            running_bs,
            max_seqlen_k,
            var["kv_indices"].gpu,
            var["kv_indptr"].gpu[: running_bs + 1],
        )

    def build_for_cudagraph_capture(self, bs: int) -> AttentionMetaData:
        attn_metadata, context = super().build_for_cudagraph_capture(bs)

        var = self.model_runner.forward_vars
        attn_metadata.triton_block_table = var["triton_block_table"][:bs]
        attn_metadata.triton_attn_logits = var["triton_attn_logits"][:bs]
        attn_metadata.triton_lse = var["triton_lse"][:bs]

        return attn_metadata, context
