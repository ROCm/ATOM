# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Type

import aiter
import numpy as np
import torch
from aiter.dist.parallel_state import get_tp_group
from atom.model_engine.scheduler import ScheduledBatch
from atom.utils import CpuGpuBuffer
from atom.utils.block_convert import (
    block_table_convert_triton,
    kv_indices_generate_triton,
)
import atom.model_ops as ops
from atom.model_ops.paged_attention import PagedAttention
from atom.model_ops.attention_mha import PagedAttentionImpl
from atom.model_ops.radix_attention import RadixAttention
from atom.utils.forward_context import AttentionMetaData, Context

from .backends import AttentionBackend, CommonAttentionBuilder
from atom.plugin.prepare import is_plugin_mode
from atom.plugin.attention import AiterAttentionMetadataBuilderDecoratorForPluginMode
from atom.plugin.attention import AiterBackendDecoratorForPluginMode


def cdiv(a, b):
    return (a + b - 1) // b


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
        attn_cls = ops.Attention
        if attn_cls == PagedAttention:
            return PagedAttentionImpl
        elif attn_cls == RadixAttention:
            raise NotImplementedError("RadixAttention is not supported for now")
        raise NotImplementedError(
            f"Unsupported attention class {attn_cls!r} configured in ops.Attention"
        )


@AiterAttentionMetadataBuilderDecoratorForPluginMode(
    default_base_class=CommonAttentionBuilder
)
class AiterAttentionMetadataBuilder:
    BLOCK_TABLE_EXTENDER: list[list[int]] = [[]]

    def __init__(
        self,
        kv_cache_spec=None,
        layer_names=None,
        config=None,
        device=None,
        model_runner=None,
    ):
        self.block_size = 1024 if model_runner.block_size == 1024 else 16
        # Note: Cannot use super() here because the class is dynamically created by decorator
        # Use explicit parent class call instead
        CommonAttentionBuilder.__init__(self, model_runner)
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
                self.max_bs * self.max_num_blocks_per_seq,
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

    def prepare_mixed(self, batch: ScheduledBatch, bs: int):
        """Build mixed prefill+decode metadata with AITER persistent buffers for decode."""
        var = self.model_runner.forward_vars

        n_prefill_seqs = batch.total_seqs_num_prefill
        n_prefill_tokens = batch.total_tokens_num_prefill
        n_decode_seqs = batch.total_seqs_num_decode
        n_decode_tokens = batch.total_tokens_num_decode
        total_tokens = n_prefill_tokens + n_decode_tokens

        # ---- Prefill sub-metadata (reuse CommonAttentionBuilder logic) ----
        positions_prefill = []
        cu_seqlens_q_p = [0]
        cu_seqlens_k_p = [0]
        max_seqlen_q_p = 0
        max_seqlen_k_p = 0
        slot_mapping_prefill = []
        has_cached = False

        for i in range(n_prefill_seqs):
            seqlen = batch.context_lens[i]
            cached_seqlen = batch.num_kv_computed[i]
            if cached_seqlen > 0:
                has_cached = True
            positions_prefill.extend(list(range(cached_seqlen, seqlen)))
            seqlen_q = seqlen - cached_seqlen
            seqlen_k = seqlen
            cu_seqlens_q_p.append(cu_seqlens_q_p[-1] + seqlen_q)
            cu_seqlens_k_p.append(cu_seqlens_k_p[-1] + seqlen_k)
            max_seqlen_q_p = max(seqlen_q, max_seqlen_q_p)
            max_seqlen_k_p = max(seqlen_k, max_seqlen_k_p)
            if not batch.block_tables:
                continue
            block_table = batch.block_tables[i]
            block_size = self.model_runner.block_size
            first_blk = cached_seqlen // block_size
            last_blk = (seqlen - 1) // block_size
            for blk_idx in range(first_blk, last_blk + 1):
                blk_start = block_table[blk_idx] * block_size
                off_start = cached_seqlen % block_size if blk_idx == first_blk else 0
                off_end = (
                    ((seqlen - 1) % block_size) + 1
                    if blk_idx == last_blk
                    else block_size
                )
                slot_mapping_prefill.extend(
                    range(blk_start + off_start, blk_start + off_end)
                )

        if has_cached:
            self.prepare_block_tables(batch)

        cu_seqlens_k_p_tensor = torch.tensor(
            cu_seqlens_k_p, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_q_p_tensor = torch.tensor(
            cu_seqlens_q_p, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping_p_tensor = torch.tensor(
            slot_mapping_prefill if slot_mapping_prefill else [-1] * n_prefill_tokens,
            dtype=torch.int64,
            pin_memory=True,
        ).cuda(non_blocking=True)

        num_cached_tokens = None
        total_kv = n_prefill_tokens
        if has_cached:
            num_cached_tokens = torch.tensor(
                batch.num_kv_computed[:n_prefill_seqs],
                dtype=torch.int32,
                pin_memory=True,
            ).cuda(non_blocking=True)
            total_kv = sum(batch.context_lens[:n_prefill_seqs])

        prefill_attn_metadata = AttentionMetaData(
            cu_seqlens_q=cu_seqlens_q_p_tensor,
            cu_seqlens_k=cu_seqlens_k_p_tensor,
            max_seqlen_q=max_seqlen_q_p,
            max_seqlen_k=max_seqlen_k_p,
            slot_mapping=slot_mapping_p_tensor,
            context_lens=torch.tensor(
                batch.context_lens[:n_prefill_seqs].tolist(),
                dtype=torch.int32,
                pin_memory=True,
            ).cuda(non_blocking=True),
            block_tables=(
                var["block_tables"].gpu[:n_prefill_seqs] if has_cached else None
            ),
            seq_starts=(var["seq_starts"].gpu[:n_prefill_seqs] if has_cached else None),
            has_cached=has_cached,
            total_kv=total_kv,
            num_cached_tokens=num_cached_tokens,
        )

        # ---- Decode sub-metadata with AITER persistent buffers ----
        decode_context_lens = np.asarray(
            batch.context_lens[n_prefill_seqs:], dtype=np.int32
        )
        decode_block_tables = batch.block_tables[n_prefill_seqs:]

        max_seqlen_q_d = batch.num_spec_step + 1
        slot_mapping_decode = [
            bt[-1] * self.model_runner.block_size + lbt - 1
            for bt, lbt in zip(
                decode_block_tables, batch.last_block_num_tokens[n_prefill_seqs:]
            )
        ]
        positions_decode = np.asarray(decode_context_lens - 1, dtype=np.int32)
        max_seqlen_k_d = int(decode_context_lens.max()) if n_decode_seqs > 0 else 0

        # Write decode block tables into var buffer (after prefill seqs)
        block_tables_np = var["block_tables"].np
        for i, bt in enumerate(decode_block_tables):
            row = n_prefill_seqs + i
            block_tables_np[row] = 0
            block_tables_np[row, : len(bt)] = bt

        # Write decode context_lens and cu_seqlens_q into var buffers
        var["context_lens"].np[:n_decode_seqs] = decode_context_lens
        var["context_lens"].np[n_decode_seqs:bs] = 0
        # cu_seqlens_q for decode: 1 token per seq
        # (already initialized as 0,1,2,...,max_bs in __init__)

        # Prepare kv_indptr and kv_indices for persistent attention
        num_blocks_per_seq = cdiv(decode_context_lens, self.block_size)
        kv_indptr = np.cumsum(num_blocks_per_seq)
        sum_blocks = kv_indptr[-1] if len(kv_indptr) > 0 else 0

        var["kv_indptr"].np[0] = 0
        var["kv_indptr"].np[1 : n_decode_seqs + 1] = kv_indptr
        var["kv_indptr"].np[n_decode_seqs + 1 : bs + 1] = sum_blocks

        # Copy decode-specific buffers to GPU
        # Block tables: copy full range then slice for decode
        var["block_tables"].copy_to_gpu(n_prefill_seqs + n_decode_seqs)
        decode_block_tables_gpu = var["block_tables"].gpu[
            n_prefill_seqs : n_prefill_seqs + n_decode_seqs
        ]

        vars_used = [
            ("context_lens", bs),
            ("cu_seqlens_q", bs + 1),
            ("kv_indptr", bs + 1),
        ]
        ctx = {el: var[el].copy_to_gpu(num) for el, num in vars_used}

        if self.block_size == 1024:
            ctx_pa_ps = self.set_aiter_persistent_worker_buffers(bs)
            ctx.update(ctx_pa_ps)

        ctx["kv_indices"] = var["kv_indices"].gpu
        kv_indices_generate_triton(
            decode_block_tables_gpu,
            ctx["kv_indices"],
            ctx["kv_indptr"],
            self.block_ratio,
            max_seqlen_k_d,
        )

        if self.block_ratio > 1:
            block_table_convert_triton(
                decode_block_tables_gpu,
                var["block_tables_converted"].gpu[
                    n_prefill_seqs : n_prefill_seqs + n_decode_seqs
                ],
                ctx["context_lens"][:n_decode_seqs],
                self.block_ratio,
            )
            ctx["block_tables_converted"] = var["block_tables_converted"].gpu[
                n_prefill_seqs : n_prefill_seqs + n_decode_seqs
            ]

        decode_slot_mapping_tensor = torch.tensor(
            slot_mapping_decode, dtype=torch.int64, pin_memory=True
        ).cuda(non_blocking=True)

        decode_attn_metadata = AttentionMetaData(
            cu_seqlens_q=ctx["cu_seqlens_q"],
            max_seqlen_q=max_seqlen_q_d,
            max_seqlen_k=max_seqlen_k_d,
            slot_mapping=decode_slot_mapping_tensor,
            context_lens=ctx["context_lens"],
            block_tables=decode_block_tables_gpu,
            kv_indptr=ctx["kv_indptr"],
            kv_indices=ctx["kv_indices"],
            **{
                k: v
                for k, v in ctx.items()
                if k not in ("context_lens", "cu_seqlens_q", "kv_indptr", "kv_indices")
            },
        )

        # ---- Merge positions and slot_mapping ----
        var["positions"].np[:n_prefill_tokens] = positions_prefill
        var["positions"].np[n_prefill_tokens:total_tokens] = positions_decode
        positions = var["positions"].copy_to_gpu(total_tokens)

        merged_slot_np = np.empty(total_tokens, dtype=np.int64)
        merged_slot_np[:n_prefill_tokens] = (
            slot_mapping_prefill if slot_mapping_prefill else [-1] * n_prefill_tokens
        )
        merged_slot_np[n_prefill_tokens:total_tokens] = slot_mapping_decode
        merged_slot_mapping = torch.tensor(
            merged_slot_np, dtype=torch.int64, pin_memory=True
        ).cuda(non_blocking=True)

        attn_metadata = AttentionMetaData(
            slot_mapping=merged_slot_mapping,
            prefill_attn_metadata=prefill_attn_metadata,
            decode_attn_metadata=decode_attn_metadata,
        )

        return attn_metadata, positions

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
            var["slot_mapping"].np[:sum_scheduled_tokens] = slot_mapping[
                :sum_scheduled_tokens
            ]

        var["positions"].np[:sum_scheduled_tokens] = positions
        var["context_lens"].np[:scheduled_bs] = context_lens
        var["context_lens"].np[scheduled_bs:bs] = 0

        # Prepare kv_indptr and kv_indices for persistent attention
        num_blocks_per_seq = cdiv(context_lens, self.block_size)
        kv_indptr = np.cumsum(num_blocks_per_seq)
        sum_blocks = kv_indptr[-1] if len(kv_indptr) > 0 else 0

        var["kv_indptr"].np[0] = 0
        var["kv_indptr"].np[1 : scheduled_bs + 1] = kv_indptr
        var["kv_indptr"].np[scheduled_bs + 1 : bs + 1] = sum_blocks

        vars_used = [
            ("slot_mapping", bs * max_seqlen_q),
            ("context_lens", bs),
            ("cu_seqlens_q", bs + 1),
            ("block_tables", bs),
            ("kv_indptr", bs + 1),
        ]

        ctx = {el: var[el].copy_to_gpu(num) for el, num in vars_used}
        if self.block_size == 1024:
            ctx_pa_ps = self.set_aiter_persistent_worker_buffers(bs)
            ctx.update(ctx_pa_ps)

        ctx["kv_indices"] = var["kv_indices"].gpu
        max_seqlen_k = context_lens.max()
        kv_indices_generate_triton(
            ctx["block_tables"],
            ctx["kv_indices"],
            ctx["kv_indptr"],
            self.block_ratio,
            max_seqlen_k,
        )
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
        attn_metadata = AttentionMetaData(
            slot_mapping=var["slot_mapping"].gpu[:bs],
            context_lens=var["context_lens"].gpu[:bs],
            block_tables=var["block_tables"].gpu[:bs],
            max_seqlen_q=var["max_qlen"],
            cu_seqlens_q=var["cu_seqlens_q"].gpu[: bs + 1],
            kv_indptr=var["kv_indptr"].gpu[: bs + 1],
            kv_indices=var["kv_indices"].gpu,
            max_seqlen_k=self.model_runner.config.max_model_len,
            block_tables_converted=(
                var["block_tables_converted"].gpu[:bs]
                if "block_tables_converted" in var
                else None
            ),
            **ctx_pa_ps,
        )

        positions = var["positions"].copy_to_gpu(bs)
        context = Context(
            positions=positions, is_prefill=False, batch_size=bs, graph_bs=bs
        )
        return attn_metadata, context
