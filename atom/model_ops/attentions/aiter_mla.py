# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from typing import Type

import numpy as np
import torch
from aiter import (
    decode_update_mla_metadata_v1,
    dtypes,
    get_mla_metadata_info_v1,
    get_mla_metadata_v1,
)
from aiter.dist.parallel_state import get_tp_group
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attention_mla import _MLA_MIN_HEADS, MLAAttention
from atom.plugin.attention import (
    AiterBackendDecoratorForPluginMode,
    AiterMLAAttentionMetadataBuilderDecoratorForPluginMode,
)
from atom.plugin.prepare import is_plugin_mode
from atom.utils import CpuGpuBuffer
from atom.utils.block_convert import (
    block_table_convert_triton,
    kv_indices_generate_triton,
)
from atom.utils.forward_context import AttentionMetaData, Context

from .backends import AttentionBackend, CommonAttentionBuilder

logger = logging.getLogger("atom")


def cdiv(a, b):
    return (a + b - 1) // b


@AiterBackendDecoratorForPluginMode
class AiterMLABackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_MLA" if not is_plugin_mode() else "CUSTOM"

    @staticmethod
    def get_builder_cls() -> Type["AiterMLAMetadataBuilder"]:
        return AiterMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> Type["MLAAttention"]:
        return MLAAttention


@AiterMLAAttentionMetadataBuilderDecoratorForPluginMode(
    default_base_class=CommonAttentionBuilder
)
class AiterMLAMetadataBuilder(CommonAttentionBuilder):

    def __init__(self, model_runner):
        self.block_size = 1
        CommonAttentionBuilder.__init__(self, model_runner)
        config = model_runner.config
        hf_config = config.hf_config
        self.num_attention_heads = (
            hf_config.num_attention_heads // get_tp_group().world_size
        )
        self.padded_num_attention_heads = max(self.num_attention_heads, _MLA_MIN_HEADS)
        self.is_sparse = model_runner.is_deepseek_v32
        self.index_topk = hf_config.index_topk if self.is_sparse else -1
        self.dtype_kv = dtypes.d_dtypes[config.kv_cache_dtype]
        self.dtype_q = self.dtype_kv

        max_seqlen_qo = getattr(model_runner, "num_spec_tokens", 0) + 1
        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_set_size, work_info_set_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_mla_metadata_info_v1(
            self.max_bs,
            max_seqlen_qo,
            self.padded_num_attention_heads,
            self.dtype_q,
            self.dtype_kv,
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
                self.max_bs * self.max_num_blocks_per_seq,
                **i32_kwargs,
            ),
            "kv_last_page_lens": CpuGpuBuffer(self.max_bs, **i32_kwargs),
        }
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
        # Per-ubatch buffers for CUDAGraph TBO
        if config.enable_tbo:
            self._allocate_ubatch_buffers(
                max_seqlen_qo,
                work_meta_data_size,
                work_meta_data_type,
                work_indptr_size,
                work_indptr_type,
                work_info_set_size,
                work_info_set_type,
                reduce_indptr_size,
                reduce_indptr_type,
                reduce_final_map_size,
                reduce_final_map_type,
                reduce_partial_map_size,
                reduce_partial_map_type,
            )

    _NUM_TBO_UBATCHES = 2

    def _allocate_ubatch_buffers(
        self,
        max_seqlen_qo,
        work_meta_data_size,
        work_meta_data_type,
        work_indptr_size,
        work_indptr_type,
        work_info_set_size,
        work_info_set_type,
        reduce_indptr_size,
        reduce_indptr_type,
        reduce_final_map_size,
        reduce_final_map_type,
        reduce_partial_map_size,
        reduce_partial_map_type,
    ):
        """Allocate per-ubatch CpuGpuBuffers for CUDAGraph TBO."""
        i32_kwargs = {"dtype": torch.int32, "device": self.device}
        i64_kwargs = {"dtype": torch.int64, "device": self.device}
        var = self.model_runner.forward_vars
        ub_max_bs = self.max_bs  # allocate full size for safety

        for ub_idx in range(self._NUM_TBO_UBATCHES):
            p = f"ub{ub_idx}_"
            var[f"{p}kv_indptr"] = CpuGpuBuffer(ub_max_bs + 1, **i32_kwargs)
            var[f"{p}kv_indices"] = CpuGpuBuffer(
                self.max_bs * self.max_num_blocks_per_seq,
                **i32_kwargs,
            )
            var[f"{p}context_lens"] = CpuGpuBuffer(ub_max_bs, **i32_kwargs)
            var[f"{p}kv_last_page_lens"] = CpuGpuBuffer(ub_max_bs, **i32_kwargs)
            var[f"{p}kv_last_page_lens"].cpu.fill_(0)
            var[f"{p}kv_last_page_lens"].copy_to_gpu()
            var[f"{p}slot_mapping"] = CpuGpuBuffer(
                ub_max_bs * max_seqlen_qo,
                **i64_kwargs,
            )
            var[f"{p}block_tables"] = CpuGpuBuffer(
                ub_max_bs,
                self.max_num_blocks_per_seq // self.block_ratio,
                **i32_kwargs,
            )
            if self.block_ratio > 1:
                var[f"{p}block_tables_converted"] = CpuGpuBuffer(
                    ub_max_bs,
                    self.max_num_blocks_per_seq,
                    **i32_kwargs,
                )
            var[f"{p}cu_seqlens_q"] = CpuGpuBuffer(ub_max_bs + 1, **i32_kwargs)
            # cu_seqlens_q for decode is constant [0, q, 2q, ..., bs*q]
            var[f"{p}cu_seqlens_q"].cpu.copy_(
                torch.arange(
                    0,
                    (ub_max_bs + 1) * max_seqlen_qo,
                    step=max_seqlen_qo,
                    dtype=torch.int32,
                )
            )
            var[f"{p}cu_seqlens_q"].copy_to_gpu()

            if self.is_sparse:
                var[f"{p}sparse_kv_indptr"] = CpuGpuBuffer(
                    ub_max_bs + 1,
                    **i32_kwargs,
                )

            # MLA work buffers per ubatch (GPU only)
            var[f"{p}work_meta_data"] = torch.empty(
                work_meta_data_size,
                dtype=work_meta_data_type,
                device=self.device,
            )
            var[f"{p}work_indptr"] = torch.empty(
                work_indptr_size,
                dtype=work_indptr_type,
                device=self.device,
            )
            var[f"{p}work_info_set"] = torch.empty(
                work_info_set_size,
                dtype=work_info_set_type,
                device=self.device,
            )
            var[f"{p}reduce_indptr"] = torch.empty(
                reduce_indptr_size,
                dtype=reduce_indptr_type,
                device=self.device,
            )
            var[f"{p}reduce_final_map"] = torch.empty(
                reduce_final_map_size,
                dtype=reduce_final_map_type,
                device=self.device,
            )
            var[f"{p}reduce_partial_map"] = torch.empty(
                reduce_partial_map_size,
                dtype=reduce_partial_map_type,
                device=self.device,
            )

    @property
    def prep_stream(self):
        # return self.model_runner.tokenID_processor.async_copy_stream
        return self.model_runner.async_execute_stream

    def set_mla_persistent_worker_buffers(
        self,
        bs: int,
        max_q_len: int,
        only_update: bool = False,
        num_reject_tokens: torch.Tensor = None,
    ):
        split_params = {
            "kv_granularity": max(self.block_size, 16),
            "max_seqlen_qo": max_q_len,
            "uni_seqlen_qo": max_q_len,
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
        if only_update:
            decode_update_mla_metadata_v1(
                var["cu_seqlens_q"].gpu[: bs + 1],
                (
                    var["sparse_kv_indptr"].gpu[: bs + 1]
                    if self.is_sparse
                    else var["kv_indptr"].gpu[: bs + 1]
                ),
                var["kv_last_page_lens"].gpu[:bs],
                self.padded_num_attention_heads,
                1,  # nhead_kv,
                True,
                work_meta_data,
                work_info_set,
                work_indptr,
                reduce_indptr,
                reduce_final_map,
                reduce_partial_map,
                page_size=self.block_size,
                kv_granularity=max(self.block_size, 16),
                max_seqlen_qo=max_q_len,
                dtype_q=self.dtype_q,
                dtype_kv=self.dtype_kv,
                num_reject_tokens=num_reject_tokens,
            )
        else:
            get_mla_metadata_v1(
                var["cu_seqlens_q"].gpu[: bs + 1],
                (
                    var["sparse_kv_indptr"].gpu[: bs + 1]
                    if self.is_sparse
                    else var["kv_indptr"].gpu[: bs + 1]
                ),
                var["kv_last_page_lens"].gpu[:bs],
                self.padded_num_attention_heads,
                1,  # nhead_kv,
                True,
                work_meta_data,
                work_info_set,
                work_indptr,
                reduce_indptr,
                reduce_final_map,
                reduce_partial_map,
                page_size=self.block_size,
                dtype_q=self.dtype_q,
                dtype_kv=self.dtype_kv,
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

    def prepare_mtp_decode(
        self,
        bs: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        only_update: bool = False,
        num_reject_tokens: torch.Tensor = None,
    ):
        var = self.model_runner.forward_vars
        kv_indptr = var["kv_indptr"].gpu[: bs + 1]
        if self.is_sparse:
            assert False, "TODO: MTP decode is not supported for sparse attention yet"
        else:
            assert self.block_size == 1
            kv_indptr += var["cu_seqlens_q"].gpu[: bs + 1]

        kv_indices_generate_triton(
            var["block_tables"].gpu[:bs],
            var["kv_indices"].gpu,
            kv_indptr,
            self.block_ratio,
            max_seqlen_k,
        )
        return self.set_mla_persistent_worker_buffers(
            bs, max_seqlen_q, only_update, num_reject_tokens
        )

    def prepare_prefill(self, batch: ScheduledBatch):
        attn_metadata, positions = CommonAttentionBuilder.prepare_prefill(self, batch)
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
            counts = var["cu_seqlens_q"].np[1 : bs + 1] - var["cu_seqlens_q"].np[:bs]
            if attn_metadata.has_cached:
                # Full context (cached + new): use cu_seqlens_k for indexer
                var["cu_seqlen_ks"].np[:sum_scheduled_tokens] = np.repeat(
                    var["cu_seqlens_k"].np[:-1], counts
                )
                var["cu_seqlen_ke"].np[:sum_scheduled_tokens] = np.repeat(
                    var["cu_seqlens_k"].np[1:], counts
                )
            else:
                var["cu_seqlen_ke"].np[:sum_scheduled_tokens] = (
                    np.arange(sum_scheduled_tokens, dtype=np.int32) + 1
                )
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

            # Per-query req_id: token_id 0..sum_scheduled_tokens-1 maps to batch id.
            # Use counts (new tokens per batch), not context_lens (full seq len).
            attn_metadata.token_to_seq_idxs = torch.repeat_interleave(
                torch.arange(bs, dtype=torch.int32, device=self.device),
                torch.tensor(counts, dtype=torch.int64, device=self.device),
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

        if hasattr(self.model_runner, "drafter") or attn_metadata.has_cached:
            # Populate kv_last_page_lens for full sequence (needed for MLA prefill with
            # prefix cache; decode does the same)
            if self.model_runner.block_size != 1:
                var["kv_last_page_lens"].np[:bs] = np.asarray(
                    batch.last_block_num_tokens[:bs], dtype=np.int32
                )
            else:
                var["kv_last_page_lens"].np[:bs] = 1
            var["kv_last_page_lens"].copy_to_gpu()

            attn_metadata.kv_indices = var["kv_indices"].gpu
            attn_metadata.kv_indptr = var["kv_indptr"].gpu[: bs + 1]
            attn_metadata.kv_indptr[0] = 0
            attn_metadata.kv_indptr[1 : bs + 1] = torch.cumsum(
                attn_metadata.context_lens, 0
            )
            attn_metadata.kv_last_page_lens = var["kv_last_page_lens"].gpu[:bs]

            # kv_indices_generate_triton expects RAW block_tables (physical block ids,
            # one per block_ratio tokens). When is_sparse, attn_metadata.block_tables
            # may have been overwritten with block_tables_converted (slot per token).
            # Always use raw block_tables for kv_indices.
            self.prepare_block_tables(batch)
            block_tables_for_kv = var["block_tables"].copy_to_gpu(bs)
            kv_indices_generate_triton(
                block_tables_for_kv,
                attn_metadata.kv_indices,
                attn_metadata.kv_indptr,
                self.block_ratio,
                attn_metadata.max_seqlen_k,
            )

        attn_metadata.dtype_q = self.dtype_q
        return attn_metadata, positions

    def prepare_decode(self, batch: ScheduledBatch, bs: int):
        scheduled_bs = batch.total_seqs_num_decode
        dropout_p = 0.0
        max_seqlen_q = 1
        if hasattr(self.model_runner, "drafter"):
            max_seqlen_q = self.model_runner.drafter.mtp_k + 1

        var = self.model_runner.forward_vars
        context_lens = np.asarray(batch.context_lens, dtype=np.int32)
        block_tables = batch.block_tables
        if not batch.is_dummy_run:
            if max_seqlen_q > 1:
                # Get num_rejected (already mapped to current batch order in prepare_input_ids)
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

        # Use scheduled_bs since in dummy run, total_seqs_num_decode is 1.
        sum_scheduled_tokens = scheduled_bs * max_seqlen_q
        var["slot_mapping"].np[: bs * max_seqlen_q] = -1
        if not batch.is_dummy_run:
            var["slot_mapping"].np[:sum_scheduled_tokens] = slot_mapping
        var["positions"].np[:sum_scheduled_tokens] = positions
        var["context_lens"].np[:scheduled_bs] = context_lens

        num_blocks_per_seq = cdiv(context_lens, self.block_size)
        kv_indptr = np.cumsum(num_blocks_per_seq)
        sum_blocks = kv_indptr[-1]

        self.prepare_block_tables(batch)
        var["kv_indptr"].np[1 : scheduled_bs + 1] = kv_indptr
        var["kv_indptr"].np[scheduled_bs + 1 : bs + 1] = sum_blocks
        var["kv_last_page_lens"].np[:scheduled_bs] = (
            batch.last_block_num_tokens if self.block_size != 1 else 1
        )
        var["kv_last_page_lens"].np[scheduled_bs:bs] = 0
        vars_used = [
            ("slot_mapping", bs * max_seqlen_q),
            ("context_lens", bs),
            ("cu_seqlens_q", bs + 1),
            # ("kv_indptr", bs + 1),
            ("kv_last_page_lens", bs),
            ("block_tables", bs),
        ]
        metadata_deps = {
            "cu_seqlens_q",
            "kv_last_page_lens",
        }

        if self.is_sparse:
            index_topk = self.index_topk
            sparse_context_lens = np.clip(var["context_lens"].np[:bs], None, index_topk)
            var["sparse_kv_indptr"].np[1 : bs + 1] = np.cumsum(
                sparse_context_lens, dtype=np.int32
            )
            var["sparse_kv_indptr"].np[scheduled_bs : bs + 1] = var[
                "sparse_kv_indptr"
            ].np[scheduled_bs]
            vars_used.append(("sparse_kv_indptr", bs + 1))
            metadata_deps.add("sparse_kv_indptr")

        prep_stream = self.prep_stream
        vars_for_metadata = [(el, num) for el, num in vars_used if el in metadata_deps]
        vars_remaining = [(el, num) for el, num in vars_used if el not in metadata_deps]
        max_seqlen_k = context_lens.max()

        ctx = {}
        ctx["kv_indptr"] = var["kv_indptr"].copy_to_gpu(bs + 1)
        # prep_stream does remaining copies + kv_indices
        current_stream = torch.cuda.current_stream()
        prep_stream.wait_stream(current_stream)
        with torch.cuda.stream(prep_stream):
            ctx_rest = {el: var[el].copy_to_gpu(num) for el, num in vars_remaining}
            ctx.update(ctx_rest)
            ctx["kv_indices"] = var["kv_indices"].gpu
            kv_indices_generate_triton(
                ctx["block_tables"],
                ctx["kv_indices"],
                ctx["kv_indptr"],
                self.block_ratio,
                max_seqlen_k,
            )

        # metadata copies on main_stream
        positions = var["positions"].copy_to_gpu(sum_scheduled_tokens)
        ctx.update({el: var[el].copy_to_gpu(num) for el, num in vars_for_metadata})
        ctx_mla_ps = self.set_mla_persistent_worker_buffers(bs, max_seqlen_q)
        ctx.update(ctx_mla_ps)
        current_stream.wait_stream(prep_stream)
        # if self.block_ratio > 1:
        #     if "block_tables" in ctx:
        #         block_table_convert_triton(
        #             var["block_tables"].gpu[:bs],
        #             var["block_tables_converted"].gpu[:bs],
        #             var["context_lens"].gpu[:bs],
        #             self.block_ratio,
        #         )
        #         ctx["block_tables_converted"] = var["block_tables_converted"].gpu[:bs]
        attn_metadata = AttentionMetaData(
            dropout_p=dropout_p,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            **ctx,
        )
        attn_metadata.dtype_q = self.dtype_q

        # Use bs (graph_bs) >= 2 instead of scheduled_bs >= 2 to avoid accuracy issue:
        if self.model_runner.config.enable_tbo_decode and bs >= 2:
            self._prepare_ubatch_decode(
                scheduled_bs,
                bs,
                max_seqlen_q,
                context_lens,
            )

        return attn_metadata, positions

    def _prepare_ubatch_decode(
        self,
        scheduled_bs: int,
        bs: int,
        max_seqlen_q: int,
        context_lens: np.ndarray,
    ):
        """
        Splits the full-batch data into per-ubatch .
        """
        var = self.model_runner.forward_vars
        N = self._NUM_TBO_UBATCHES
        half = bs // N

        ub_ranges = [
            (0, half),
            (half, bs),
        ]
        padded_bs_list = [half, bs - half]

        for ub_idx, ((req_start, req_end), padded_bs) in enumerate(
            zip(ub_ranges, padded_bs_list)
        ):
            p = f"ub{ub_idx}_"
            # How many real requests fall in this ubatch's range
            ub_real_reqs = max(0, min(scheduled_bs, req_end) - req_start)

            var[f"{p}context_lens"].np[:ub_real_reqs] = var["context_lens"].np[
                req_start : req_start + ub_real_reqs
            ]
            var[f"{p}context_lens"].np[ub_real_reqs:padded_bs] = 0

            var[f"{p}kv_last_page_lens"].np[:ub_real_reqs] = var[
                "kv_last_page_lens"
            ].np[req_start : req_start + ub_real_reqs]
            var[f"{p}kv_last_page_lens"].np[ub_real_reqs:padded_bs] = 0

            tok_start = req_start * max_seqlen_q
            ub_real_tokens = ub_real_reqs * max_seqlen_q
            padded_tok_count = padded_bs * max_seqlen_q
            var[f"{p}slot_mapping"].np[:ub_real_tokens] = var["slot_mapping"].np[
                tok_start : tok_start + ub_real_tokens
            ]
            var[f"{p}slot_mapping"].np[ub_real_tokens:padded_tok_count] = -1

            var[f"{p}block_tables"].np[:ub_real_reqs] = var["block_tables"].np[
                req_start : req_start + ub_real_reqs
            ]
            var[f"{p}block_tables"].np[ub_real_reqs:padded_bs] = 0

            full_kv_indptr = var["kv_indptr"].np
            base = full_kv_indptr[req_start]
            var[f"{p}kv_indptr"].np[0] = 0
            if ub_real_reqs > 0:
                var[f"{p}kv_indptr"].np[1 : ub_real_reqs + 1] = (
                    full_kv_indptr[req_start + 1 : req_start + ub_real_reqs + 1] - base
                )
            last_val = var[f"{p}kv_indptr"].np[ub_real_reqs] if ub_real_reqs > 0 else 0
            var[f"{p}kv_indptr"].np[ub_real_reqs + 1 : padded_bs + 1] = last_val

            if self.is_sparse:
                full_sparse = var["sparse_kv_indptr"].np
                sparse_base = full_sparse[req_start]
                var[f"{p}sparse_kv_indptr"].np[0] = 0
                if ub_real_reqs > 0:
                    var[f"{p}sparse_kv_indptr"].np[1 : ub_real_reqs + 1] = (
                        full_sparse[req_start + 1 : req_start + ub_real_reqs + 1]
                        - sparse_base
                    )
                sparse_last = (
                    var[f"{p}sparse_kv_indptr"].np[ub_real_reqs]
                    if ub_real_reqs > 0
                    else 0
                )
                var[f"{p}sparse_kv_indptr"].np[
                    ub_real_reqs + 1 : padded_bs + 1
                ] = sparse_last

            last_cu = ub_real_reqs * max_seqlen_q
            var[f"{p}cu_seqlens_q"].np[: ub_real_reqs + 1] = np.arange(
                0,
                (ub_real_reqs + 1) * max_seqlen_q,
                max_seqlen_q,
                dtype=np.int32,
            )
            var[f"{p}cu_seqlens_q"].np[ub_real_reqs + 1 : padded_bs + 1] = last_cu

            vars_used = [
                (f"{p}context_lens", padded_bs),
                (f"{p}kv_last_page_lens", padded_bs),
                (f"{p}slot_mapping", padded_tok_count),
                (f"{p}block_tables", padded_bs),
                (f"{p}kv_indptr", padded_bs + 1),
                (f"{p}cu_seqlens_q", padded_bs + 1),
            ]
            if self.is_sparse:
                vars_used.append((f"{p}sparse_kv_indptr", padded_bs + 1))

            for el, num in vars_used:
                var[el].copy_to_gpu(num)

            ub_max_seqlen_k = (
                int(context_lens[req_start : req_start + ub_real_reqs].max())
                if ub_real_reqs > 0
                else 0
            )
            kv_indices_generate_triton(
                var[f"{p}block_tables"].gpu[:padded_bs],
                var[f"{p}kv_indices"].gpu,
                var[f"{p}kv_indptr"].gpu[: padded_bs + 1],
                self.block_ratio,
                ub_max_seqlen_k,
            )

            self._set_ubatch_mla_buffers(padded_bs, max_seqlen_q, ub_idx)

    def _set_ubatch_mla_buffers(self, padded_bs, max_q_len, ubatch_idx):
        """Compute MLA work buffers for a per-ubatch forward_vars set."""
        p = f"ub{ubatch_idx}_"
        var = self.model_runner.forward_vars

        kv_indptr_for_mla = var[f"{p}kv_indptr"].gpu[: padded_bs + 1]
        if self.is_sparse:
            kv_indptr_for_mla = var[f"{p}sparse_kv_indptr"].gpu[: padded_bs + 1]

        get_mla_metadata_v1(
            var[f"{p}cu_seqlens_q"].gpu[: padded_bs + 1],
            kv_indptr_for_mla,
            var[f"{p}kv_last_page_lens"].gpu[:padded_bs],
            self.padded_num_attention_heads,
            1,  # nhead_kv
            True,
            var[f"{p}work_meta_data"],
            var[f"{p}work_info_set"],
            var[f"{p}work_indptr"],
            var[f"{p}reduce_indptr"],
            var[f"{p}reduce_final_map"],
            var[f"{p}reduce_partial_map"],
            page_size=self.block_size,
            dtype_q=self.dtype_q,
            dtype_kv=self.dtype_kv,
            kv_granularity=max(self.block_size, 16),
            max_seqlen_qo=max_q_len,
            uni_seqlen_qo=max_q_len,
            fast_mode=1,
            max_split_per_batch=16,
        )

    def build_for_cudagraph_capture(self, bs: int) -> AttentionMetaData:
        var = self.model_runner.forward_vars
        sparse_kv_indptr = var["sparse_kv_indptr"].gpu if self.is_sparse else None
        max_q_len = var["mtp_k"] + 1 if "mtp_k" in var else 1
        ctx_mla_ps = self.set_mla_persistent_worker_buffers(bs, max_q_len)
        attn_matadata = AttentionMetaData(
            slot_mapping=var["slot_mapping"].gpu[: bs * max_q_len],
            context_lens=var["context_lens"].gpu[:bs],
            block_tables=var["block_tables"].gpu[:bs],
            max_seqlen_q=max_q_len,
            cu_seqlens_q=var["cu_seqlens_q"].gpu[: bs + 1],
            kv_indptr=var["kv_indptr"].gpu[: bs + 1],
            kv_indices=var["kv_indices"].gpu,
            kv_last_page_lens=var["kv_last_page_lens"].gpu[:bs],
            sparse_kv_indptr=sparse_kv_indptr,
            block_tables_converted=(
                var["block_tables_converted"].gpu[:bs]
                if "block_tables_converted" in var
                else None
            ),
            **ctx_mla_ps,
        )
        attn_matadata.dtype_q = self.dtype_q
        positions = var["positions"].copy_to_gpu(bs * max_q_len)
        context = Context(
            positions=positions, is_prefill=False, batch_size=bs, graph_bs=bs
        )
        return attn_matadata, context

    def build_ubatch_metadata(
        self,
        ubatch_idx: int,
        padded_bs: int,
    ) -> AttentionMetaData:
        """Create per-ubatch AttentionMetaData from pre-allocated forward_vars."""
        var = self.model_runner.forward_vars
        p = f"ub{ubatch_idx}_"
        max_q_len = var["mtp_k"] + 1 if "mtp_k" in var else 1

        # Compute MLA work buffers for this ubatch
        self._set_ubatch_mla_buffers(padded_bs, max_q_len, ubatch_idx)

        attn = AttentionMetaData(
            slot_mapping=var[f"{p}slot_mapping"].gpu[: padded_bs * max_q_len],
            context_lens=var[f"{p}context_lens"].gpu[:padded_bs],
            block_tables=var[f"{p}block_tables"].gpu[:padded_bs],
            max_seqlen_q=max_q_len,
            cu_seqlens_q=var[f"{p}cu_seqlens_q"].gpu[: padded_bs + 1],
            kv_indptr=var[f"{p}kv_indptr"].gpu[: padded_bs + 1],
            kv_indices=var[f"{p}kv_indices"].gpu,
            kv_last_page_lens=var[f"{p}kv_last_page_lens"].gpu[:padded_bs],
            sparse_kv_indptr=(
                var[f"{p}sparse_kv_indptr"].gpu[: padded_bs + 1]
                if self.is_sparse
                else None
            ),
            block_tables_converted=(
                var[f"{p}block_tables_converted"].gpu[:padded_bs]
                if f"{p}block_tables_converted" in var
                else None
            ),
            work_meta_data=var[f"{p}work_meta_data"],
            work_info_set=var[f"{p}work_info_set"],
            work_indptr=var[f"{p}work_indptr"],
            reduce_indptr=var[f"{p}reduce_indptr"],
            reduce_final_map=var[f"{p}reduce_final_map"],
            reduce_partial_map=var[f"{p}reduce_partial_map"],
        )
        attn.dtype_q = self.dtype_q
        return attn

    def build_ubatch_prefill_metadata(
        self,
        attn_metadata: AttentionMetaData,
        ub_slice,
        padded_bs: int,
    ) -> AttentionMetaData:
        """
        Split prefill AttentionMetaData for MLA.
        """
        from atom.utils.tbo.ubatch_splitting import split_attn_metadata

        ub_attn = split_attn_metadata(attn_metadata, ub_slice, padded_bs)

        ts = ub_slice.token_slice
        rs = ub_slice.request_slice
        req_start = rs.start

        if (
            hasattr(attn_metadata, "cu_seqlen_ks")
            and attn_metadata.cu_seqlen_ks is not None
        ):
            ub_attn.cu_seqlen_ks = attn_metadata.cu_seqlen_ks[ts]

        if (
            hasattr(attn_metadata, "cu_seqlen_ke")
            and attn_metadata.cu_seqlen_ke is not None
        ):
            ub_attn.cu_seqlen_ke = attn_metadata.cu_seqlen_ke[ts]

        if (
            hasattr(attn_metadata, "sparse_cu_seqlens_q")
            and attn_metadata.sparse_cu_seqlens_q is not None
        ):
            base = attn_metadata.sparse_cu_seqlens_q[ts.start]
            ub_attn.sparse_cu_seqlens_q = (
                attn_metadata.sparse_cu_seqlens_q[ts.start : ts.stop + 1] - base
            )

        if (
            hasattr(attn_metadata, "token_to_seq_idxs")
            and attn_metadata.token_to_seq_idxs is not None
        ):
            ub_attn.token_to_seq_idxs = attn_metadata.token_to_seq_idxs[ts] - req_start

        total_tokens = (
            attn_metadata.slot_mapping.shape[0]
            if attn_metadata.slot_mapping is not None
            else 0
        )
        if (
            attn_metadata.kv_last_page_lens is not None
            and attn_metadata.kv_last_page_lens.shape[0] == total_tokens
        ):
            ub_attn.kv_last_page_lens = attn_metadata.kv_last_page_lens[ts]

        if (
            attn_metadata.sparse_kv_indptr is not None
            and attn_metadata.sparse_kv_indptr.shape[0] == total_tokens + 1
        ):
            base = attn_metadata.sparse_kv_indptr[ts.start]
            ub_attn.sparse_kv_indptr = (
                attn_metadata.sparse_kv_indptr[ts.start : ts.stop + 1] - base
            )

        return ub_attn
