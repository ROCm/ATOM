# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass
from typing import Optional, Type

import numpy as np
import torch
from atom.model_engine.scheduler import ScheduledBatch
import atom.model_ops as ops
from atom.model_ops.paged_attention import PagedAttention
from atom.model_ops.attention_mha import PagedAttentionImpl
from atom.model_ops.radix_attention import RadixAttention
from atom.utils.block_convert import block_table_convert_triton
from atom.utils.forward_context import AttentionMetaData, Context

from .backends import AttentionBackend, CommonAttentionBuilder
from atom.plugin.prepare import is_plugin_mode
from atom.plugin.attention import AiterAttentionMetadataBuilderDecoratorForPluginMode
from atom.plugin.attention import AiterBackendDecoratorForPluginMode


@AiterBackendDecoratorForPluginMode
class AiterBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_ATTENTION" if not is_plugin_mode() else "CUSTOM"

    @staticmethod
    def get_builder_cls() -> Type["AiterAttentionMetadataBuilder"]:
        return AiterAttentionMetadataBuilder

    @staticmethod
    def get_impl_cls():
        if ops.ATTN_CLS == PagedAttention:
            return PagedAttentionImpl
        elif ops.ATTN_CLS == RadixAttention:
            raise NotImplementedError("RadixAttention is not supported for now")


@AiterAttentionMetadataBuilderDecoratorForPluginMode(default_base_class=CommonAttentionBuilder)
class AiterAttentionMetadataBuilder:
    BLOCK_TABLE_EXTENDER: list[list[int]] = [[]]

    def __init__(
        self,
        # for plugin mode
        kv_cache_spec = None,
        layer_names = None,
        config = None,
        device = None,
        # for server mode
        model_runner = None,
    ):
        # Note: Cannot use super() here because the class is dynamically created by decorator
        # Use explicit parent class call instead
        CommonAttentionBuilder.__init__(self, model_runner)

    def prepare_decode(self, batch: ScheduledBatch, bs: int):
        # print('--------------------------------', flush=True)
        # print('[zejun] ATOM(server), prepare_decode', flush=True)
        # print('[zejun] ATOM(server), batch.context_lens = ', batch.context_lens, flush=True)
        # print('[zejun] ATOM(server), batch.num_cached_tokens = ', batch.num_cached_tokens, flush=True)
        # print('[zejun] ATOM(server), batch.num_scheduled_tokens = ', batch.num_scheduled_tokens, flush=True)
        # print('[zejun] ATOM(server), batch.block_tables = ', batch.block_tables, flush=True)
        # for block_table in batch.block_tables:
        #     print('[zejun] ATOM(server), block_table = ', block_table, flush=True)
        # print('[zejun] ATOM(server), batch.last_block_num_tokens = ', batch.last_block_num_tokens, flush=True)
        # # for block_num_tokens in batch.last_block_num_tokens:
        # #     print('[zejun] ATOM(server), block_num_tokens shape = ', block_num_tokens.shape, flush=True)
        # print('[zejun] ATOM(server), batch.total_tokens_num = ', batch.total_tokens_num, flush=True)
        # print('[zejun] ATOM(server), batch.total_tokens_num_prefill = ', batch.total_tokens_num_prefill, flush=True)
        # print('[zejun] ATOM(server), batch.total_tokens_num_decode = ', batch.total_tokens_num_decode, flush=True)
        # print('[zejun] ATOM(server), batch.total_seqs_num = ', batch.total_seqs_num, flush=True)
        # print('[zejun] ATOM(server), batch.total_seqs_num_prefill = ', batch.total_seqs_num_prefill, flush=True)
        # print('[zejun] ATOM(server), batch.total_seqs_num_decode = ', batch.total_seqs_num_decode, flush=True)
        # print('--------------------------------', flush=True)
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
        # print('[zejun] ATOM(server), bs = ', bs, flush=True)
        # print('[zejun] ATOM(server), scheduled_bs = ', scheduled_bs, flush=True)

        self.prepare_block_tables(batch)

        var = self.model_runner.forward_vars
        sum_scheduled_tokens = batch.total_tokens_num_decode
        if batch.is_dummy_run:
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
            ("cu_seqlens_q", bs + 1),
        ]
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
        # print(f"[zejun] ATOM prepare_decode - scheduled_bs: {scheduled_bs}", flush=True)
        # print(f"[zejun] ATOM prepare_decode - bs: {bs}", flush=True)
        # print(f"[zejun] ATOM prepare_decode - AttentionMetaData members:", flush=True)
        # print(f"  cu_seqlens_q: {attn_metadata.cu_seqlens_q}", flush=True)
        # print(f"  cu_seqlens_k: {attn_metadata.cu_seqlens_k}", flush=True)
        # print(f"  max_seqlen_q: {attn_metadata.max_seqlen_q}", flush=True)
        # print(f"  max_seqlen_k: {attn_metadata.max_seqlen_k}", flush=True)
        # print(f"  min_seqlen_q: {attn_metadata.min_seqlen_q}", flush=True)
        # print(f"  slot_mapping: {attn_metadata.slot_mapping}", flush=True)
        # print(f"  context_lens: {attn_metadata.context_lens}", flush=True)
        # if attn_metadata.block_tables is not None:
        #     print(f"  block_tables shape : {attn_metadata.block_tables.shape}", flush=True)
        # print(f"  dropout_p: {attn_metadata.dropout_p}", flush=True)
        # print(f"  max_q_len: {attn_metadata.max_q_len}", flush=True)
        # print(f"  kv_indptr: {attn_metadata.kv_indptr}", flush=True)
        # print(f"  kv_indices: {attn_metadata.kv_indices}", flush=True)
        # print(f"  kv_last_page_lens: {attn_metadata.kv_last_page_lens}", flush=True)
        # print(f"  cu_seqlen_ks: {attn_metadata.cu_seqlen_ks}", flush=True)
        # print(f"  cu_seqlen_ke: {attn_metadata.cu_seqlen_ke}", flush=True)
        # print(f"  sparse_kv_indptr: {attn_metadata.sparse_kv_indptr}", flush=True)
        # print(f"  work_meta_data: {attn_metadata.work_meta_data}", flush=True)
        # print(f"  work_indptr: {attn_metadata.work_indptr}", flush=True)
        # print(f"  work_info_set: {attn_metadata.work_info_set}", flush=True)
        # print(f"  reduce_indptr: {attn_metadata.reduce_indptr}", flush=True)
        # print(f"  reduce_final_map: {attn_metadata.reduce_final_map}", flush=True)
        # print(f"  reduce_partial_map: {attn_metadata.reduce_partial_map}", flush=True)
        # print(f"  context: {attn_metadata.context}", flush=True)

        # print(f"  block_tables_converted: {attn_metadata.block_tables_converted}", flush=True)
        # print(f"  kv_indices_converted: {attn_metadata.kv_indices_converted}", flush=True)

        positions = var["positions"].copy_to_gpu(sum_scheduled_tokens)
        # print(f"  positions: {positions}", flush=True)
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
