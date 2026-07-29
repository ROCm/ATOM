# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import math

import torch
from aiter.dist.parallel_state import get_tp_group

from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attentions.gdn_attn import (
    PAD_SLOT_ID,
    GDNAttentionMetadata,
    compute_causal_conv1d_metadata,
)
from atom.utils import CpuGpuBuffer
from atom.utils.forward_context import AttentionMetaData


class KimiGDNStateMixin:
    """Kimi-K3 state-cache and metadata support for its KDA layers."""

    def __init__(self, model_runner, **kwargs):
        super().__init__(model_runner=model_runner, **kwargs)
        self._init_kimi_gdn_state(model_runner)

    def _init_kimi_gdn_state(self, model_runner):
        hf = model_runner.config.hf_config
        lin = getattr(hf, "linear_attn_config", {}) or {}
        model_runner.full_attention_layers = [
            int(i) - 1 for i in lin.get("full_attn_layers", [])
        ]
        model_runner.kda_attention_layers = [
            int(i) - 1 for i in lin.get("kda_layers", [])
        ]
        model_runner.num_full_attn = len(model_runner.full_attention_layers)
        model_runner.num_gdn_attn_state = len(model_runner.kda_attention_layers)
        hf.linear_num_key_heads = getattr(
            hf, "linear_num_key_heads", lin.get("num_heads", hf.num_attention_heads)
        )
        hf.linear_num_value_heads = getattr(
            hf,
            "linear_num_value_heads",
            lin.get("num_heads", hf.num_attention_heads),
        )
        hf.linear_key_head_dim = getattr(
            hf, "linear_key_head_dim", lin.get("head_dim", hf.qk_nope_head_dim)
        )
        hf.linear_value_head_dim = getattr(
            hf, "linear_value_head_dim", lin.get("head_dim", hf.v_head_dim)
        )
        hf.linear_conv_kernel_dim = getattr(
            hf,
            "linear_conv_kernel_dim",
            lin.get("short_conv_kernel_size", 4),
        )

        self.non_spec_state_indices_tensor = CpuGpuBuffer(
            (self.max_bs,),
            dtype=torch.int32,
            device=self.device,
        )
        self.non_spec_query_start_loc = torch.arange(
            start=0,
            end=self.max_bs + 1,
            dtype=torch.int32,
            device=self.device,
        )

        gdn_metadata = {
            "non_spec_state_indices": self.non_spec_state_indices_tensor,
            "non_spec_query_start_loc": self.non_spec_query_start_loc,
        }
        self.model_runner.forward_vars.update(gdn_metadata)

    @staticmethod
    def _state_shape(
        tp_world_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads
        conv_state_shape = (
            conv_kernel_size - 1,
            conv_dim // tp_world_size,
        )
        temporal_state_shape = (
            num_v_heads // tp_world_size,
            head_v_dim,
            head_k_dim,
        )
        return conv_state_shape, temporal_state_shape

    def _state_dtypes(self) -> tuple[torch.dtype, torch.dtype]:
        return self.model_runner.config.torch_dtype, torch.float32

    def _state_shape_for_runner(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        hf = self.model_runner.config.hf_config
        return self._state_shape(
            get_tp_group().world_size,
            hf.linear_num_key_heads,
            hf.linear_num_value_heads,
            hf.linear_key_head_dim,
            hf.linear_value_head_dim,
            hf.linear_conv_kernel_dim,
        )

    def compute_per_req_cache_bytes(self) -> int:
        shape_k, shape_v = self._state_shape_for_runner()
        dt_k, dt_v = self._state_dtypes()
        per_layer = (
            math.prod(shape_k) * dt_k.itemsize + math.prod(shape_v) * dt_v.itemsize
        )
        return self.model_runner.num_gdn_attn_state * per_layer

    def slots_per_req(self) -> int:
        return 1

    def allocate_per_req_cache(self, num_slots: int) -> dict[str, torch.Tensor]:
        shape_k, shape_v = self._state_shape_for_runner()
        dt_k, dt_v = self._state_dtypes()
        n = self.model_runner.num_gdn_attn_state
        return {
            "mamba_k_cache": torch.zeros(
                (n, num_slots) + shape_k, dtype=dt_k, device="cuda"
            ),
            "mamba_v_cache": torch.zeros(
                (n, num_slots) + shape_v, dtype=dt_v, device="cuda"
            ),
        }

    def prepare_state_indices(self, batch: ScheduledBatch):
        non_spec_state_indices = self.non_spec_state_indices_tensor.np
        for idx, slot_group in enumerate(batch.per_req_cache_groups):
            non_spec_state_indices[idx] = slot_group

    def prepare_gdn_metadata(
        self,
        batch: ScheduledBatch,
        attn_metadata: AttentionMetaData,
        is_prefill: bool = False,
        *,
        prepare_block_tables: bool = True,
    ) -> GDNAttentionMetadata:
        num_decodes = batch.total_seqs_num_decode
        num_prefills = batch.total_seqs_num_prefill
        num_decode_tokens = batch.total_tokens_num_decode
        num_prefill_tokens = batch.total_tokens_num_prefill
        num_reqs = batch.total_seqs_num
        if prepare_block_tables:
            self.prepare_block_tables(batch)

        query_start_loc = attn_metadata.cu_seqlens_q
        context_lens_tensor = torch.zeros(batch.total_seqs_num_prefill).cuda()
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        self.prepare_state_indices(batch)
        non_spec_state_indices_tensor = self.non_spec_state_indices_tensor.copy_to_gpu(
            num_reqs
        )

        if num_prefills > 0:
            has_initial_state = context_lens_tensor > 0
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(query_start_loc)
            )
        else:
            has_initial_state = None

        return GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=batch.total_tokens_num,
            has_initial_state=has_initial_state,
            spec_query_start_loc=None,
            non_spec_query_start_loc=query_start_loc,
            spec_state_indices_tensor=None,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=None,
            spec_token_indx=None,
            non_spec_token_indx=None,
            num_accepted_tokens=None,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )

    def _attach_gdn_decode_metadata(
        self,
        batch,
        attn_metadata,
        *,
        prepare_block_tables: bool = True,
    ) -> None:
        num_decodes = batch.total_seqs_num_decode
        gdn_metadata = self.prepare_gdn_metadata(
            batch,
            attn_metadata,
            prepare_block_tables=prepare_block_tables,
        )

        self.non_spec_state_indices_tensor.gpu[num_decodes:].fill_(PAD_SLOT_ID)

        self.non_spec_query_start_loc[: num_decodes + 1].copy_(
            gdn_metadata.non_spec_query_start_loc[: num_decodes + 1],
            non_blocking=True,
        )
        self.non_spec_query_start_loc[num_decodes + 1 :].fill_(
            gdn_metadata.non_spec_query_start_loc[num_decodes]
        )
        gdn_metadata.non_spec_query_start_loc = self.non_spec_query_start_loc[
            : num_decodes + 1
        ]

        attn_metadata.gdn_metadata = gdn_metadata

    def _build_gdn_capture_metadata(self, bs: int):
        return GDNAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=bs,
            num_decode_tokens=bs,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=bs,
            has_initial_state=None,
            spec_query_start_loc=None,
            non_spec_query_start_loc=self.non_spec_query_start_loc[: bs + 1],
            spec_state_indices_tensor=None,
            non_spec_state_indices_tensor=self.non_spec_state_indices_tensor.gpu[:bs],
            spec_sequence_masks=None,
            spec_token_indx=None,
            non_spec_token_indx=None,
            num_accepted_tokens=None,
            nums_dict=None,
            batch_ptr=None,
            token_chunk_offset_ptr=None,
        )
