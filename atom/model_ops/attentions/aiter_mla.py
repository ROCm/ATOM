# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import itertools
from typing import Type

import numpy as np
import torch
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attention_mla import MLAAttention
from atom.utils import CpuGpuBuffer
from atom.utils.block_convert import (
    block_table_convert_triton,
    kv_indices_convert_triton,
)
from atom.utils.forward_context import AttentionMetaData, Context

from aiter import dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.dist.parallel_state import get_tp_group

from .backends import AttentionBackend, CommonAttentionBuilder


class AiterMLABackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_MLA"

    @staticmethod
    def get_builder_cls() -> Type["AiterMLAMetadataBuilder"]:
        return AiterMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> Type["MLAAttention"]:
        return MLAAttention


class AiterMLAMetadataBuilder(CommonAttentionBuilder):

    def __init__(self, model_runner):
        self.block_size = 1
        super().__init__(model_runner)
        config = model_runner.config
        hf_config = config.hf_config
        self.num_attention_heads = (
            hf_config.num_attention_heads // get_tp_group().world_size
        )
        self.is_sparse = model_runner.is_deepseek_v32
        self.index_topk = hf_config.index_topk if self.is_sparse else -1

        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_set_size, work_info_set_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_mla_metadata_info_v1(
            self.max_bs,
            1,
            self.num_attention_heads,
            torch.bfloat16,
            dtypes.d_dtypes[config.kv_cache_dtype],
            is_sparse=self.is_sparse,
            fast_mode=True,
        )
        i32_kwargs = {"dtype": torch.int32, "device": self.device}

        mla_metadata = {
            # AITER MLA specific persistent buffers
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
            "kv_last_page_lens": CpuGpuBuffer(self.max_bs, **i32_kwargs),
        }
        if self.block_ratio > 1:
            mla_metadata["kv_indices_converted"] = CpuGpuBuffer(
                self.max_bs * self.max_num_blocks_per_seq, **i32_kwargs
            )
        mla_metadata["kv_last_page_lens"].cpu.fill_(1)
        mla_metadata["kv_last_page_lens"].copy_to_gpu()
        if self.is_sparse:
            mla_metadata["cu_seqlen_ke"] = CpuGpuBuffer(
                self.max_num_batched_tokens, **i32_kwargs
            )
            mla_metadata["cu_seqlen_ks"] = CpuGpuBuffer(
                self.max_num_batched_tokens, **i32_kwargs
            )
            mla_metadata["sparse_kv_indptr"] = CpuGpuBuffer(
                self.max_num_batched_tokens + 1, **i32_kwargs
            )
            mla_metadata["sparse_cu_seqlens_q"] = CpuGpuBuffer(
                self.max_num_batched_tokens + 1, **i32_kwargs
            )
            mla_metadata["sparse_cu_seqlens_q"].np[:] = np.arange(
                self.max_num_batched_tokens + 1, dtype=np.int32
            )
            mla_metadata["sparse_cu_seqlens_q"].copy_to_gpu()
            mla_metadata["sparse_kv_last_page_lens"] = CpuGpuBuffer(
                self.max_num_batched_tokens, **i32_kwargs
            )
            mla_metadata["sparse_kv_last_page_lens"].np[:] = 1
            mla_metadata["sparse_kv_last_page_lens"].copy_to_gpu()

        self.model_runner.forward_vars.update(mla_metadata)

    def set_mla_persistent_worker_buffers(self, bs: int):
        split_params = {
            "kv_granularity": max(self.block_size, 16),
            "max_seqlen_qo": 1,
            "uni_seqlen_qo": 1,
            "fast_mode": 1,
            "max_split_per_batch": 16,
        }
        var = self.model_runner.forward_vars
        work_meta_data = var["work_meta_data"]
        work_info_set = var["work_info_set"]
        work_indptr = var["work_indptr"]
        reduce_indptr = var["reduce_indptr"]
        reduce_final_map = var["reduce_final_map"]
        reduce_partial_map = var["reduce_partial_map"]
        get_mla_metadata_v1(
            var["cu_seqlens_q"].gpu[: bs + 1],
            (
                var["sparse_kv_indptr"].gpu[: bs + 1]
                if self.is_sparse
                else var["kv_indptr"].gpu[: bs + 1]
            ),
            self.num_attention_heads,
            1,  # nhead_kv,
            True,
            work_meta_data,
            work_info_set,
            work_indptr,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            **split_params,
        )
        return {
            "work_meta_data": work_meta_data,
            "work_info_set": work_info_set,
            "work_indptr": work_indptr,
            "reduce_indptr": reduce_indptr,
            "reduce_final_map": reduce_final_map,
            "reduce_partial_map": reduce_partial_map,
        }

    def prepare_prefill(self, batch: ScheduledBatch):
        attn_metadata, positions = super().prepare_prefill(batch)
        bs = batch.total_seqs_num_prefill
        sum_scheduled_tokens = batch.total_tokens_num_prefill
        var = self.model_runner.forward_vars
        if self.is_sparse and attn_metadata.max_seqlen_k > self.index_topk:
            if attn_metadata.block_tables is None:
                self.prepare_block_tables(batch)
                attn_metadata.block_tables = var["block_tables"].copy_to_gpu(bs)
                if self.block_ratio > 1:
                    block_table_convert_triton(
                        var["block_tables"].gpu[:bs],
                        var["block_tables_converted"].gpu[:bs],
                        var["context_lens"].gpu[:bs],
                        self.block_ratio,
                    )
                    attn_metadata.block_tables = var["block_tables_converted"].gpu[:bs]
            var["cu_seqlen_ke"].np[:sum_scheduled_tokens] = (
                np.arange(sum_scheduled_tokens, dtype=np.int32) + 1
            )
            counts = var["cu_seqlens_q"].np[1 : bs + 1] - var["cu_seqlens_q"].np[:bs]
            var["cu_seqlen_ks"].np[:sum_scheduled_tokens] = np.repeat(
                var["cu_seqlens_q"].np[:bs], counts
            )
            attn_metadata.cu_seqlen_ks = var["cu_seqlen_ks"].copy_to_gpu(
                sum_scheduled_tokens
            )
            attn_metadata.cu_seqlen_ke = var["cu_seqlen_ke"].copy_to_gpu(
                sum_scheduled_tokens
            )
            attn_metadata.sparse_cu_seqlens_q = var["sparse_cu_seqlens_q"].gpu[
                : sum_scheduled_tokens + 1
            ]
            attn_metadata.kv_last_page_lens = var["sparse_kv_last_page_lens"].gpu[
                :sum_scheduled_tokens
            ]

            attn_metadata.token_to_seq_idxs = torch.repeat_interleave(
                torch.arange(bs, dtype=torch.int32, device=self.device),
                attn_metadata.context_lens,
            )
            var["sparse_kv_indptr"].np[0] = 0
            var["sparse_kv_indptr"].np[1 : sum_scheduled_tokens + 1] = np.cumsum(
                np.minimum(
                    np.concatenate([np.arange(1, s + 1) for s in counts]),
                    self.index_topk,
                ),
                dtype=np.int32,
            )
            attn_metadata.sparse_kv_indptr = var["sparse_kv_indptr"].copy_to_gpu(
                sum_scheduled_tokens + 1
            )
        return attn_metadata, positions

    def prepare_decode(self, batch: ScheduledBatch, bs: int):
        scheduled_bs = batch.total_seqs_num_decode
        dropout_p = 0.0
        max_q_len = 1

        context_lens = batch.context_lens
        positions = [i - 1 for i in context_lens]
        slot_mapping = [
            block_table[-1] * self.model_runner.block_size + last_block_num - 1
            for block_table, last_block_num in zip(
                batch.block_tables, batch.last_block_num_tokens
            )
        ]

        var = self.model_runner.forward_vars
        sum_scheduled_tokens = batch.total_tokens_num_decode
        if batch.is_dummy_run:
            var["slot_mapping"].np[:bs] = -1
        else:
            var["slot_mapping"].np[:scheduled_bs] = slot_mapping
            var["slot_mapping"].np[scheduled_bs:bs] = -1
        var["positions"].np[:sum_scheduled_tokens] = positions
        var["context_lens"].np[:scheduled_bs] = context_lens
        # var["context_lens"].np[scheduled_bs:bs] = 0

        num_blocks_per_seq = [
            (ctx + self.block_size - 1) // self.block_size for ctx in batch.context_lens
        ]
        kv_indptr = np.cumsum(num_blocks_per_seq)
        sum_blocks = kv_indptr[-1]
        sum_blocks_before_converted = sum(
            [(i + self.block_ratio - 1) // self.block_ratio for i in num_blocks_per_seq]
        )

        def prepare_kv_indices():
            var["kv_indices"].np[:sum_blocks_before_converted] = np.fromiter(
                itertools.chain.from_iterable(batch.block_tables),
                dtype=np.int32,
                count=sum_blocks_before_converted,
            )

        prepare_kv_indices()
        var["kv_indptr"].np[1 : scheduled_bs + 1] = kv_indptr
        var["kv_indptr"].np[scheduled_bs + 1 : bs + 1] = sum_blocks
        var["kv_last_page_lens"].np[:scheduled_bs] = (
            batch.last_block_num_tokens if self.block_size != 1 else 1
        )
        var["kv_last_page_lens"].np[scheduled_bs:bs] = 0
        vars_used = [
            ("slot_mapping", bs),  # TODO: MTP support
            ("context_lens", bs),
            ("cu_seqlens_q", bs + 1),
            ("kv_indptr", bs + 1),
            ("kv_indices", sum_blocks),
            ("kv_last_page_lens", bs),
        ]
        if self.is_sparse:
            self.prepare_block_tables(batch)
            vars_used.append(("block_tables", bs))
            index_topk = self.index_topk
            sparse_context_lens = np.clip(var["context_lens"].np[:bs], None, index_topk)
            var["sparse_kv_indptr"].np[1 : bs + 1] = np.cumsum(
                sparse_context_lens, dtype=np.int32
            )
            var["sparse_kv_indptr"].np[scheduled_bs : bs + 1] = var[
                "sparse_kv_indptr"
            ].np[scheduled_bs]
            vars_used.append(("sparse_kv_indptr", bs + 1))

        ctx = {el: var[el].copy_to_gpu(num) for el, num in vars_used}
        ctx_mla_ps = self.set_mla_persistent_worker_buffers(bs)
        ctx.update(ctx_mla_ps)
        if self.block_ratio > 1:
            kv_indices_convert_triton(
                var["kv_indices"].gpu[:sum_blocks_before_converted],
                var["kv_indices_converted"].gpu[:sum_blocks],
                var["kv_indptr"].gpu[: bs + 1],
                self.block_ratio,
                self.block_size,
            )
            ctx["kv_indices_converted"] = var["kv_indices_converted"].gpu[:sum_blocks]

            if "block_tables" in ctx:
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
            **ctx,
        )
        positions = var["positions"].copy_to_gpu(sum_scheduled_tokens)
        # if str(positions.device) == "cuda:0":
        #     for el, var in ctx.items():
        #         print(f"{el}: {var}")
        return attn_metadata, positions

    def build_for_cudagraph_capture(self, bs: int) -> AttentionMetaData:
        var = self.model_runner.forward_vars
        sparse_kv_indptr = var["sparse_kv_indptr"].gpu if self.is_sparse else None
        ctx_mla_ps = self.set_mla_persistent_worker_buffers(bs)
        attn_matadata = AttentionMetaData(
            slot_mapping=var["slot_mapping"].gpu[:bs],
            context_lens=var["context_lens"].gpu[:bs],
            block_tables=var["block_tables"].gpu[:bs],
            max_q_len=1,
            cu_seqlens_q=var["cu_seqlens_q"].gpu[: bs + 1],
            kv_indptr=var["kv_indptr"].gpu[: bs + 1],
            kv_indices=var["kv_indices"].gpu[:],
            kv_last_page_lens=var["kv_last_page_lens"].gpu[:bs],
            sparse_kv_indptr=sparse_kv_indptr,
            block_tables_converted=(
                var["block_tables_converted"].gpu[:bs]
                if "block_tables_converted" in var
                else None
            ),
            kv_indices_converted=(
                var["kv_indices_converted"].gpu[:]
                if "kv_indices_converted" in var
                else None
            ),
            **ctx_mla_ps,
        )
        positions = var["positions"].copy_to_gpu(bs)
        context = Context(
            positions=positions, is_prefill=False, batch_size=bs, graph_bs=bs
        )
        return attn_matadata, context
