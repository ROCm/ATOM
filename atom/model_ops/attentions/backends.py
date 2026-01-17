# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar

import numpy as np
import torch
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.sequence import Sequence
from atom.model_ops.attention_mla import MLAModules
from atom.utils import CpuGpuBuffer
from atom.utils.block_convert import block_table_convert_triton
from atom.utils.forward_context import AttentionMetaData
from torch import nn

T = TypeVar("T", bound="BroadcastableModelInput")


class BroadcastableModelInput(ABC):

    @abstractmethod
    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        """
        Extract broadcastable fields. Override for fields that require some
        custom deserialization.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_broadcasted_tensor_dict(
        cls: Type[T],
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> T:
        """
        Pop fields from the given tensor_dict and populate a new instance of
        BroadcastableModelInput.
        """
        raise NotImplementedError


class AttentionBackend(ABC):
    """Abstract class for attention backends."""

    # For some attention backends, we allocate an output tensor before
    # calling the custom op. When piecewise cudagraph is enabled, this
    # makes sure the output tensor is allocated inside the cudagraph.
    accept_output_buffer: bool = False

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_builder_cls() -> Type["AttentionMetadataBuilder"]:
        raise NotImplementedError

    @staticmethod
    def get_impl_cls() -> Type["AttentionImpl"]:
        return AttentionImpl


class AttentionMetadataBuilder(ABC, Generic[T]):
    """Abstract class for attention metadata builders."""

    @abstractmethod
    def __init__(self, block_size: int) -> None:
        """Create the builder, remember some configuration and parameters."""
        raise NotImplementedError

    @abstractmethod
    def prepare_decode(self, batch: ScheduledBatch, bs: int):
        raise NotImplementedError

    @abstractmethod
    def prepare_prefill(self, batch: ScheduledBatch):
        raise NotImplementedError

    @abstractmethod
    def build(self, batch: ScheduledBatch, bs: int):
        raise NotImplementedError

    @abstractmethod
    def build_for_cudagraph_capture(self, bs: int) -> AttentionMetaData:
        raise NotImplementedError


class CommonAttentionBuilder(AttentionMetadataBuilder[T], Generic[T]):
    def __init__(self, model_runner):
        self.model_runner = model_runner
        self.block_size = 16 if not model_runner.use_mla else 1
        assert model_runner.block_size % self.block_size == 0
        self.block_ratio = model_runner.block_size // self.block_size
        self.device = model_runner.device
        config = model_runner.config
        hf_config = config.hf_config
        self.max_num_batched_tokens = model_runner.max_num_batched_tokens
        self.max_bs = model_runner.max_bs
        self.max_num_blocks_per_seq = (
            config.max_model_len + self.block_size - 1
        ) // self.block_size

        i64_kwargs = {"dtype": torch.int64, "device": self.device}
        i32_kwargs = {"dtype": torch.int32, "device": self.device}

        attn_metadata = {
            "slot_mapping": CpuGpuBuffer(self.max_num_batched_tokens, **i64_kwargs),
            "context_lens": CpuGpuBuffer(self.max_bs, **i32_kwargs),
            "block_tables": CpuGpuBuffer(
                self.max_bs,
                self.max_num_blocks_per_seq // self.block_ratio,
                **i32_kwargs,
            ),
            "cu_seqlens_q": CpuGpuBuffer(self.max_bs + 1, **i32_kwargs),
            "cu_seqlens_k": CpuGpuBuffer(self.max_bs + 1, **i32_kwargs),
        }
        if self.block_ratio > 1:
            attn_metadata["block_tables_converted"] = CpuGpuBuffer(
                self.max_bs,
                self.max_num_blocks_per_seq,
                **i32_kwargs,
            )

        attn_metadata["cu_seqlens_q"].cpu.copy_(
            torch.arange(0, self.max_bs + 1, step=1, dtype=torch.int32)
        )
        attn_metadata["cu_seqlens_q"].copy_to_gpu()
        self.model_runner.forward_vars.update(attn_metadata)
        self.has_sliding_window = hasattr(hf_config, "sliding_window")

    def prepare_block_tables(self, batch: ScheduledBatch):
        var = self.model_runner.forward_vars
        block_tables = var["block_tables"].np
        for i, block_table in enumerate(batch.block_tables):
            block_tables[i] = 0
            block_tables[i, : len(block_table)] = block_table

    def prepare_prefill(self, batch: ScheduledBatch):
        # print('--------------------------------', flush=True)
        # print('[zejun] ATOM, prepare_prefill', flush=True)
        # print('[zejun] ATOM, batch.context_lens = ', batch.context_lens, flush=True)
        # print('[zejun] ATOM, batch.num_cached_tokens = ', batch.num_cached_tokens, flush=True)
        # print('[zejun] ATOM, batch.num_scheduled_tokens = ', batch.num_scheduled_tokens, flush=True)
        # print('[zejun] ATOM, batch.block_tables = ', batch.block_tables, flush=True)
        # for block_table in batch.block_tables:
        #     print('[zejun] ATOM, block_table = ', block_table, flush=True)
        # print('[zejun] ATOM, batch.last_block_num_tokens = ', batch.last_block_num_tokens, flush=True)
        # # for block_num_tokens in batch.last_block_num_tokens:
        # #     print('[zejun] ATOM, block_num_tokens shape = ', block_num_tokens.shape, flush=True)
        # print('[zejun] ATOM, batch.total_tokens_num = ', batch.total_tokens_num, flush=True)
        # print('[zejun] ATOM, batch.total_tokens_num_prefill = ', batch.total_tokens_num_prefill, flush=True)
        # print('[zejun] ATOM, batch.total_tokens_num_decode = ', batch.total_tokens_num_decode, flush=True)
        # print('[zejun] ATOM, batch.total_seqs_num = ', batch.total_seqs_num, flush=True)
        # print('[zejun] ATOM, batch.total_seqs_num_prefill = ', batch.total_seqs_num_prefill, flush=True)
        # print('[zejun] ATOM, batch.total_seqs_num_decode = ', batch.total_seqs_num_decode, flush=True)
        # print('--------------------------------', flush=True)

        bs = batch.total_seqs_num_prefill
        sum_scheduled_tokens = batch.total_tokens_num_prefill
        var = self.model_runner.forward_vars
        positions = []
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        # seqs = list(batch.seqs.values())
        # seqs = seqs[:bs]
        for i in range(bs):
            seqlen = batch.context_lens[i]
            cached_seqlen = batch.num_cached_tokens[i]
            positions.extend(list(range(cached_seqlen, seqlen)))
            seqlen_q = seqlen - cached_seqlen
            seqlen_k = seqlen
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not batch.block_tables:
                continue
            num_blocks = (
                seqlen + self.model_runner.block_size - 1
            ) // self.model_runner.block_size
            # print('[zejun] ATOM(server) num_blocks = ', num_blocks, flush=True)
            num_cached_blocks = (
                cached_seqlen + self.model_runner.block_size - 1
            ) // self.model_runner.block_size
            # print('[zejun] ATOM(server) num_cached_blocks = ', num_cached_blocks, flush=True)
            last_block_tokens = batch.last_block_num_tokens[i]
            # print('[zejun] ATOM(server) last_block_tokens = ', last_block_tokens, flush=True)
            block_table = batch.block_tables[i]
            # print('[zejun] ATOM(server) block_table = ', block_table, flush=True)
            for i in range(num_cached_blocks, num_blocks):
                start = block_table[i] * self.model_runner.block_size
                # print('[zejun] ATOM(server)[', i, '] start = ', start, flush=True)
                if i != num_blocks - 1:
                    end = start + self.model_runner.block_size
                else:
                    end = start + last_block_tokens
                # print('[zejun] ATOM(server)[', i, '] end = ', end, flush=True)
                slot_mapping.extend(list(range(start, end)))
                # print('[zejun] ATOM(server)[', i, '] slot_mapping = ', slot_mapping, flush=True)
        if cu_seqlens_k[-1] > batch.total_tokens_num:  # prefix cache
            self.prepare_block_tables(batch)
        var["positions"].np[:sum_scheduled_tokens] = positions
        var["slot_mapping"].np[: len(slot_mapping)] = slot_mapping
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True)
        var["context_lens"].np[:bs] = batch.context_lens[:bs]
        min_seqlen_q = 0
        dropout_p = 0.0
        vars_used = [
            ("cu_seqlens_q", bs + 1),
            ("slot_mapping", len(slot_mapping)),
            ("context_lens", bs),
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
            cu_seqlens_k=cu_seqlens_k.cuda(non_blocking=True),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_q=min_seqlen_q,
            dropout_p=dropout_p,
            **ctx,
        )
        # print(f"[zejun] ATOM prepare_prefill - AttentionMetaData members:", flush=True)
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
        # return var["positions"].copy_to_gpu(sum_scheduled_tokens)

    def build(self, batch: ScheduledBatch, bs: int):
        is_prefill = batch.total_tokens_num_prefill > 0
        if is_prefill:
            return self.prepare_prefill(batch)
        else:
            return self.prepare_decode(batch, bs)


class AttentionImpl(nn.Module):
    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        layer_num: int = 0,
        mla_modules: MLAModules = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        position: torch.Tensor = None,
    ) -> torch.Tensor:
        raise NotImplementedError
