# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass
from typing import Optional, Type

import numpy as np
import torch
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attention_mha import Attention
from atom.utils.block_convert import block_table_convert_triton
from atom.utils.forward_context import AttentionMetaData, Context

from .backends import AttentionBackend, CommonAttentionBuilder


class AiterBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_ATTENTION"

    @staticmethod
    def get_builder_cls() -> Type["AiterAttentionMetadataBuilder"]:
        return AiterAttentionMetadataBuilder

    @staticmethod
    def get_impl_cls() -> Type["Attention"]:
        return Attention


class AiterAttentionMetadataBuilder(CommonAttentionBuilder):
    BLOCK_TABLE_EXTENDER: list[list[int]] = [[]]

    def __init__(self, model_runner):
        super().__init__(model_runner)

    def prepare_decode(self, batch: ScheduledBatch, bs: int, is_dummy_run: bool=False):
        scheduled_bs = batch.total_seqs_num_decode
        self.total_blocks = 0
        dropout_p = 0.0
        max_q_len = 1
        min_seqlen_q = 0

        context_lens = batch.context_lens
        max_seqlen_k = max(context_lens)
        positions = [i - 1 for i in context_lens]
        slot_mapping = [
            block_table[-1] * self.model_runner.block_size + last_block_num - 1
            for block_table, last_block_num in zip(
                batch.block_tables, batch.last_block_num_tokens
            )
        ]
        slot_mapping.extend([-1] * (bs - scheduled_bs))

        self.prepare_block_tables(batch)

        var = self.model_runner.forward_vars
        sum_scheduled_tokens = batch.total_tokens_num_decode
        if is_dummy_run:
            var["slot_mapping"].np[:bs] = -1
        else:
            var["slot_mapping"].np[:bs] = slot_mapping
            
        var["positions"].np[:sum_scheduled_tokens] = positions
        var["context_lens"].np[:scheduled_bs] = context_lens
        var["context_lens"].np[scheduled_bs:bs] = 0
        vars_used = [
            ("slot_mapping", bs),  # TODO: MTP support
            ("context_lens", bs),
            ("block_tables", bs),
        ]
        if self.has_sliding_window:
            vars_used.append(("cu_seqlens_q", bs + 1))
        ctx = {el: var[el].copy_to_gpu(num) for el, num in vars_used}
        if self.block_ratio > 1 and "block_tables" in ctx:
            block_table_convert_triton(
                var["block_tables"].gpu[:bs],
                var["block_tables_converted"].gpu[:bs],
                var["context_lens"].gpu[:bs],
                self.block_ratio,
            )
            ctx["block_tables_converted"] = var["block_tables_converted"].gpu[:bs]
        attn_metadata = AttentionMetaData(
            dropout_p=dropout_p,
            max_q_len=max_q_len,
            max_seqlen_q=max_q_len,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_q=min_seqlen_q,
            **ctx,
        )
        positions = var["positions"].copy_to_gpu(sum_scheduled_tokens)
        return attn_metadata, positions

    def build_for_cudagraph_capture(self, bs: int) -> AttentionMetaData:
        var = self.model_runner.forward_vars
        attn_metadata = AttentionMetaData(
            slot_mapping=var["slot_mapping"].gpu[:bs],
            context_lens=var["context_lens"].gpu[:bs],
            block_tables=var["block_tables"].gpu[:bs],
            max_q_len=1,
            cu_seqlens_q=var["cu_seqlens_q"].gpu[: bs + 1],
            max_seqlen_k=self.model_runner.config.max_model_len,
            block_tables_converted=(
                var["block_tables_converted"].gpu[:bs]
                if "block_tables_converted" in var
                else None
            ),
        )

        positions = var["positions"].copy_to_gpu(bs)
        context = Context(
            positions=positions, is_prefill=False, batch_size=bs, graph_bs=bs
        )
        return attn_metadata, context
