# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import itertools
from typing import Type

import aiter
import numpy as np
import torch
import triton
from aiter.dist.parallel_state import get_tp_group
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attention_mha import Attention
from atom.utils import CpuGpuBuffer
from atom.utils.block_convert import block_table_convert_triton
from atom.utils.forward_context import AttentionMetaData, Context

from .backends import AttentionBackend, CommonAttentionBuilder

def cdiv(a, b):
    return (a + b - 1) // b

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
        self.block_size = 1024 if model_runner.block_size == 1024 else 16
        super().__init__(model_runner)
        config = model_runner.config
        hf_config = config.hf_config
        self.num_attention_heads = (
            hf_config.num_attention_heads // get_tp_group().world_size
        )
        # For speculative decode (MTP), max_qlen = num_speculative_tokens + 1
        if (
            config.speculative_config is not None
            and config.speculative_config.num_speculative_tokens is not None
        ):
            max_qlen = config.speculative_config.num_speculative_tokens + 1
        else:
            max_qlen = 1

        num_head_k = max(1, hf_config.num_key_value_heads // get_tp_group().world_size)
        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_set_size, work_info_set_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = aiter.get_pa_metadata_info_v1(
            self.max_bs,
            num_head_k,
        )

        i32_kwargs = {"dtype": torch.int32, "device": self.device}

        pa_persistent_metadata = {
            "max_qlen": max_qlen,
            "work_meta_data": torch.empty(
                work_meta_data_size, dtype=work_meta_data_type, device=self.device
            ),
            "work_indptr": torch.empty(
                work_indptr_size, dtype=work_indptr_type, device=self.device
            ),
            "work_info_set": torch.empty(
                work_info_set_size, dtype=work_info_set_type, device=self.device
            ),
            "reduce_indptr": torch.empty(
                reduce_indptr_size, dtype=reduce_indptr_type, device=self.device
            ),
            "reduce_final_map": torch.empty(
                reduce_final_map_size, dtype=reduce_final_map_type, device=self.device
            ),
            "reduce_partial_map": torch.empty(
                reduce_partial_map_size,
                dtype=reduce_partial_map_type,
                device=self.device,
            ),
            "kv_indptr": CpuGpuBuffer(self.max_bs + 1, **i32_kwargs),
            "kv_indices": CpuGpuBuffer(
                self.max_bs * self.max_num_blocks_per_seq // self.block_ratio,
                **i32_kwargs,
            ),
        }
        self.model_runner.forward_vars.update(pa_persistent_metadata)

    def set_aiter_persistent_worker_buffers(self, bs: int):
        config = self.model_runner.config
        hf_config = config.hf_config
        num_query_heads = self.num_attention_heads
        num_kv_heads = max(
            1, hf_config.num_key_value_heads // get_tp_group().world_size
        )
        block_size = self.block_size

        var = self.model_runner.forward_vars
        max_qlen = var["max_qlen"]

        qo_indptr = var["cu_seqlens_q"].gpu[: bs + 1]
        kv_indptr = var["kv_indptr"].gpu[: bs + 1]
        seq_lens_kv = var["context_lens"].gpu[:bs]

        work_meta_data = var["work_meta_data"]
        work_indptr = var["work_indptr"]
        work_info_set = var["work_info_set"]
        reduce_indptr = var["reduce_indptr"]
        reduce_final_map = var["reduce_final_map"]
        reduce_partial_map = var["reduce_partial_map"]

        aiter.get_pa_metadata_v1(
            qo_indptr,
            kv_indptr,
            seq_lens_kv,
            num_query_heads // num_kv_heads,
            num_kv_heads,
            True,
            work_meta_data,
            work_indptr,
            work_info_set,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            kv_granularity=max(block_size, 16),
            block_size=block_size,
            max_seqlen_qo=int(max_qlen),
            uni_seqlen_qo=max_qlen,
            fast_mode=True,
            max_split_per_batch=-1,
        )

        return {
            "work_meta_data": work_meta_data,
            "work_indptr": work_indptr,
            "work_info_set": work_info_set,
            "reduce_indptr": reduce_indptr,
            "reduce_final_map": reduce_final_map,
            "reduce_partial_map": reduce_partial_map,
        }

    def prepare_decode(self, batch: ScheduledBatch, bs: int):
        scheduled_bs = batch.total_seqs_num_decode
        self.total_blocks = 0
        dropout_p = 0.0
        max_seqlen_q = batch.num_spec_step + 1
        min_seqlen_q = 0

        context_lens = np.asarray(batch.context_lens, dtype=np.int32)
        block_tables = batch.block_tables

        if max_seqlen_q > 1:
            num_rejected = self.model_runner.tokenID_processor.num_rejected
            if num_rejected is not None:
                context_lens -= num_rejected
                num_blocks = cdiv(context_lens, self.model_runner.block_size)
                block_tables = [bt[:n] for bt, n in zip(block_tables, num_blocks)]

            slot_mapping = [
                block_table[pos // self.model_runner.block_size]
                * self.model_runner.block_size
                + (pos % self.model_runner.block_size)
                for block_table, seq_len in zip(block_tables, context_lens)
                for pos in range(seq_len - max_seqlen_q, seq_len)
            ]
        else:
            slot_mapping = [
                block_table[-1] * self.model_runner.block_size + last_block_num - 1
                for block_table, last_block_num in zip(
                    block_tables, batch.last_block_num_tokens
                )
            ]
        positions = np.tile(
            np.arange(max_seqlen_q, dtype=np.int32), scheduled_bs
        ) + np.repeat(context_lens - max_seqlen_q, max_seqlen_q)
        max_seqlen_k = np.max(context_lens)

        self.prepare_block_tables(batch)

        var = self.model_runner.forward_vars
        sum_scheduled_tokens = batch.total_tokens_num_decode
        var["slot_mapping"].np[: bs * max_seqlen_q] = -1
        if not batch.is_dummy_run:
            var["slot_mapping"].np[:sum_scheduled_tokens] = slot_mapping

        var["positions"].np[:sum_scheduled_tokens] = positions
        var["context_lens"].np[:scheduled_bs] = context_lens
        var["context_lens"].np[scheduled_bs:bs] = 0

        # Prepare kv_indptr and kv_indices for persistent attention
        num_blocks_per_seq = cdiv(context_lens, self.block_size)
        kv_indptr = np.cumsum(num_blocks_per_seq)
        sum_blocks = kv_indptr[-1] if len(kv_indptr) > 0 else 0
        sum_blocks_before_converted = cdiv(num_blocks_per_seq, self.block_ratio).sum()

        var["kv_indices"].np[:sum_blocks_before_converted] = np.fromiter(
            itertools.chain.from_iterable(block_tables),
            dtype=np.int32,
            count=sum_blocks_before_converted,
        )
        var["kv_indptr"].np[0] = 0
        var["kv_indptr"].np[1 : scheduled_bs + 1] = kv_indptr
        var["kv_indptr"].np[scheduled_bs + 1 : bs + 1] = sum_blocks

        vars_used = [
            ("slot_mapping", bs * max_seqlen_q),
            ("context_lens", bs),
            ("cu_seqlens_q", bs + 1),
            ("block_tables", bs),
            ("kv_indptr", bs + 1),
            ("kv_indices", sum_blocks_before_converted),
        ]

        ctx = {el: var[el].copy_to_gpu(num) for el, num in vars_used}
        if self.block_size == 1024:
            ctx_pa_ps = self.set_aiter_persistent_worker_buffers(bs)
            ctx.update(ctx_pa_ps)
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
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_q=min_seqlen_q,
            **ctx,
        )
        positions = var["positions"].copy_to_gpu(sum_scheduled_tokens)
        return attn_metadata, positions

    def build_for_cudagraph_capture(self, bs: int) -> AttentionMetaData:
        var = self.model_runner.forward_vars
        if self.block_size == 1024:
            ctx_pa_ps = self.set_aiter_persistent_worker_buffers(bs)
        else:
            ctx_pa_ps = {}
        max_q_len = var["mtp_k"] + 1 if "mtp_k" in var else 1
        attn_metadata = AttentionMetaData(
            slot_mapping=var["slot_mapping"].gpu[:bs * max_q_len],
            context_lens=var["context_lens"].gpu[:bs],
            block_tables=var["block_tables"].gpu[:bs],
            max_seqlen_q=max_q_len,
            cu_seqlens_q=var["cu_seqlens_q"].gpu[: bs + 1],
            kv_indptr=var["kv_indptr"].gpu[: bs + 1],
            kv_indices=var["kv_indices"].gpu[:],
            max_seqlen_k=self.model_runner.config.max_model_len,
            block_tables_converted=(
                var["block_tables_converted"].gpu[:bs]
                if "block_tables_converted" in var
                else None
            ),
            **ctx_pa_ps,
        )

        positions = var["positions"].copy_to_gpu(bs * max_q_len)
        context = Context(
            positions=positions, is_prefill=False, batch_size=bs, graph_bs=bs
        )
        return attn_metadata, context
