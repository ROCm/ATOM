# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import math
from dataclasses import dataclass
from typing import Type

import numpy as np
import torch
from aiter.dist.parallel_state import get_tp_group
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attention_gdn import GatedDeltaNet
from atom.utils import CpuGpuBuffer
from atom.utils.forward_context import AttentionMetaData, Context

from .aiter_attention import (
    AiterBackend,
    AiterAttentionMetadataBuilder,
    kv_indices_generate_triton,
)


class GDNAttentionBackend(AiterBackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_GDN_ATTENTION"

    @staticmethod
    def get_builder_cls() -> Type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder

    @staticmethod
    def get_impl_cls() -> Type["GatedDeltaNet"]:
        return GatedDeltaNet


@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


class GDNAttentionMetadataBuilder(AiterAttentionMetadataBuilder):

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        model_runner,
    ):
        super().__init__(model_runner=model_runner)
        # Hybrid model layer-counting state (formerly set as a side effect
        # inside ModelRunner._compute_block_bytes' qwen_next branch).
        # Promoted to runner attributes here so all consumers
        # (build_kv_cache_tensor, allocate_kv_cache_tensors, the per-req
        # cache hooks) can read them as `self.model_runner.<name>` without
        # a hidden ordering dependency on _compute_block_bytes being
        # called first.
        hf = model_runner.config.hf_config
        model_runner.full_attention_interval = hf.full_attention_interval
        model_runner.num_full_attn = (
            hf.num_hidden_layers // model_runner.full_attention_interval
        )
        model_runner.num_gdn_attn_state = (
            hf.num_hidden_layers - model_runner.num_full_attn
        )

        self.num_spec = 0
        if hasattr(model_runner, "drafter"):
            self.num_spec = model_runner.drafter.mtp_k
        self.use_spec_decode = self.num_spec > 0

        self.spec_state_indices_tensor = CpuGpuBuffer(
            (self.max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=self.device,
        )
        self.non_spec_state_indices_tensor = CpuGpuBuffer(
            (self.max_bs,),
            dtype=torch.int32,
            device=self.device,
        )
        self.spec_sequence_masks = torch.ones(
            (self.max_bs,),
            dtype=torch.bool,
            device=self.device,
        )
        self.spec_token_indx = torch.arange(
            (self.max_bs * (self.num_spec + 1)),
            dtype=torch.int32,
            device=self.device,
        )
        self.non_spec_token_indx = torch.empty(
            (self.max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=self.device,
        )
        self.spec_query_start_loc = torch.arange(
            start=0,
            end=(self.max_bs + 1) * (self.num_spec + 1),
            step=(self.num_spec + 1),
            dtype=torch.int32,
            device=self.device,
        )
        self.non_spec_query_start_loc = torch.arange(
            start=0,
            end=self.max_bs + 1,
            dtype=torch.int32,
            device=self.device,
        )
        self.num_accepted_tokens = torch.ones(
            (self.max_bs,),
            dtype=torch.int32,
            device=self.device,
        )

        gdn_metadata = {
            "spec_state_indices": self.spec_state_indices_tensor,
            "non_spec_state_indices": self.non_spec_state_indices_tensor,
            "spec_sequence_masks": self.spec_sequence_masks,
            "spec_token_indx": self.spec_token_indx,
            "non_spec_token_indx": self.non_spec_token_indx,
            "spec_query_start_loc": self.spec_query_start_loc,
            "non_spec_query_start_loc": self.non_spec_query_start_loc,
            "num_accepted_tokens": self.num_accepted_tokens,
        }
        self.model_runner.forward_vars.update(gdn_metadata)

    # ------------------------------------------------------------------ #
    # Per-request cache hooks (called from ModelRunner via base class).  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _state_shape(
        tp_world_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int,
        num_spec: int = 0,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """GDN per-layer state shape (conv_state, temporal_state).

        Moved from ModelRunner.gated_delta_net_state_shape() so that the
        GDN-specific tensor layout lives next to the GDN-specific code that
        consumes it. Identical math.
        """
        conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads
        conv_state_shape = (
            conv_kernel_size - 1 + num_spec,
            conv_dim // tp_world_size,
        )
        temporal_state_shape = (
            num_v_heads // tp_world_size,
            head_v_dim,
            head_k_dim,
        )
        return conv_state_shape, temporal_state_shape

    def _state_dtypes(self) -> tuple[torch.dtype, torch.dtype]:
        return (
            self.model_runner.config.torch_dtype,
            self.model_runner.config.torch_dtype,
        )

    def _state_shape_for_runner(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        hf = self.model_runner.config.hf_config
        return self._state_shape(
            get_tp_group().world_size,
            hf.linear_num_key_heads,
            hf.linear_num_value_heads,
            hf.linear_key_head_dim,
            hf.linear_value_head_dim,
            hf.linear_conv_kernel_dim,
            self.model_runner.num_spec_tokens,
        )

    def compute_per_req_cache_bytes(self) -> int:
        """GDN: conv_state + temporal_state, summed over all GDN layers."""
        shape_k, shape_v = self._state_shape_for_runner()
        dt_k, dt_v = self._state_dtypes()
        per_layer = (
            math.prod(shape_k) * dt_k.itemsize + math.prod(shape_v) * dt_v.itemsize
        )
        return self.model_runner.num_gdn_attn_state * per_layer

    def slots_per_req(self) -> int:
        """GDN reserves one extra slot per speculative token for rollback."""
        return 1 + self.num_spec

    def allocate_per_req_cache(self, num_slots: int) -> dict[str, torch.Tensor]:
        """Allocate mamba_k_cache / mamba_v_cache.

        Names preserved for backward compat with `attention_gdn.py` which
        accesses them as `model_runner.mamba_{k,v}_cache`.
        """
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

    def compute_block_bytes(self) -> int:
        """GDN hybrid: only full-attention layer slots contribute paged KV
        bytes (linear-attention layers' state lives in the per-request
        cache pool, accounted separately via compute_per_req_cache_bytes).
        """
        from aiter import dtypes

        runner = self.model_runner
        config = runner.config
        hf_config = config.hf_config
        num_kv_heads = runner._get_num_kv_heads()
        total = runner._get_total_num_layers()
        num_draft = total - hf_config.num_hidden_layers
        n_full = runner.num_full_attn + num_draft
        kv_dtype_size = dtypes.d_dtypes[config.kv_cache_dtype].itemsize

        # kv_cache: [2, n_full, blocks, block_size, num_kv_heads, head_dim]
        block_bytes = (
            2
            * n_full
            * runner.physical_block_size
            * num_kv_heads
            * hf_config.head_dim
            * kv_dtype_size
        )
        # kv_scale: [2, n_full, blocks, num_kv_heads, block_size] fp32
        block_bytes += 2 * n_full * num_kv_heads * runner.physical_block_size * 4
        return block_bytes

    def allocate_kv_cache_tensors(
        self, num_kv_heads: int, num_draft_layers: int
    ) -> dict:
        """GDN hybrid: KV cache only covers full-attention layer slots
        (linear-attention layers don't store paged KV; they use the
        per-request mamba_k/v_cache pool allocated separately).

        Layout: `[2, num_full_attn + num_draft_layers, ...]` — note this
        differs from AiterAttentionMetadataBuilder's `num_hidden_layers`
        first dim. The slot index math is in build_kv_cache_tensor's
        attn_idx computation (skips linear-attn slots).
        """
        from aiter import dtypes

        runner = self.model_runner
        config = runner.config
        hf_config = config.hf_config
        n_full = runner.num_full_attn + num_draft_layers
        return {
            "kv_cache": torch.zeros(
                2,
                n_full,
                runner.num_physical_kvcache_blocks,
                runner.physical_block_size,
                num_kv_heads,
                hf_config.head_dim,
                dtype=dtypes.d_dtypes[config.kv_cache_dtype],
                device="cuda",
            ),
            "kv_scale": torch.zeros(
                2,
                n_full,
                runner.num_physical_kvcache_blocks,
                num_kv_heads,
                runner.physical_block_size,
                dtype=dtypes.fp32,
                device="cuda",
            ),
        }

    def build_kv_cache_tensor(self, layer_id: int, module):
        """Dispatch by module type:

        - `base_linear_attention` (GDN linear attention) → wrap the slot
          slice of mamba_k_cache / mamba_v_cache
        - everything else (full-attention MHA layers in the hybrid model,
          plus modules of types this builder doesn't recognize) → defer
          to AiterAttentionMetadataBuilder.build_kv_cache_tensor
        """
        if hasattr(module, "base_linear_attention"):
            from atom.config import KVCacheTensor

            runner = self.model_runner
            interval = runner.full_attention_interval
            gdn_idx = (layer_id // interval) * (interval - 1) + (layer_id % interval)
            return KVCacheTensor(
                layer_num=layer_id,
                k_cache=runner.mamba_k_cache[gdn_idx],
                v_cache=runner.mamba_v_cache[gdn_idx],
                k_scale=None,
                v_scale=None,
            )
        return super().build_kv_cache_tensor(layer_id, module)

    def prepare_state_indices(self, batch: ScheduledBatch, with_spec: bool = False):
        non_spec_state_indices = self.non_spec_state_indices_tensor.np
        spec_state_indices = self.spec_state_indices_tensor.np
        slots_per_group = 1 + self.num_spec
        for idx, slot_group in enumerate(batch.per_req_cache_groups):
            non_spec_state_indices[idx] = 0
            spec_state_indices[idx] = 0
            base = slot_group * slots_per_group

            if not with_spec:
                non_spec_state_indices[idx] = base
            else:
                spec_state_indices[idx, : 1 + self.num_spec] = np.arange(
                    base, base + 1 + self.num_spec
                )

    def prepare_num_accepted_tokens(self, batch: ScheduledBatch):
        self.num_accepted_tokens.fill_(1)

        if self.model_runner.tokenID_processor.num_bonus is None:
            return
        for idx, num_bonus in enumerate(self.model_runner.tokenID_processor.num_bonus):
            self.num_accepted_tokens[idx] = num_bonus + 1

    def prepare_gdn_metadata(
        self,
        batch: ScheduledBatch,
        attn_metadata: AttentionMetaData,
        is_prefill: bool = False,
    ) -> GDNAttentionMetadata:

        num_decodes = batch.total_seqs_num_decode
        num_prefills = batch.total_seqs_num_prefill
        num_decode_tokens = batch.total_tokens_num_decode
        num_prefill_tokens = batch.total_tokens_num_prefill
        num_reqs = batch.total_seqs_num
        self.prepare_block_tables(batch)

        context_lens_tensor = attn_metadata.context_lens
        query_start_loc = attn_metadata.cu_seqlens_q
        context_lens_tensor = torch.zeros((batch.total_seqs_num_prefill)).cuda()
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        if not self.use_spec_decode or is_prefill:
            self.prepare_state_indices(batch, with_spec=False)
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = (
                self.non_spec_state_indices_tensor.copy_to_gpu(num_reqs)
            )
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            num_accepted_tokens = None
            spec_sequence_masks = None
            num_spec_decodes = 0
            num_spec_decode_tokens = 0
        else:
            self.prepare_state_indices(batch, with_spec=True)
            self.prepare_num_accepted_tokens(batch)
            spec_token_size = min(
                num_decodes * (self.num_spec + 1), query_start_loc[-1].item()
            )
            spec_token_indx = torch.arange(
                spec_token_size, dtype=torch.int32, device=self.device
            )
            non_spec_token_indx = torch.empty(
                0, dtype=torch.int32, device=query_start_loc.device
            )
            spec_sequence_masks = torch.ones(
                num_reqs, dtype=torch.bool, device=self.device
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor.copy_to_gpu(
                num_reqs
            )
            non_spec_state_indices_tensor = None
            spec_query_start_loc = query_start_loc
            non_spec_query_start_loc = None
            num_accepted_tokens = self.num_accepted_tokens[:num_reqs]
            num_spec_decodes = num_decodes
            num_prefills = 0
            num_decodes = 0
            num_spec_decode_tokens = num_decode_tokens
            num_decode_tokens = 0
            num_prefill_tokens = 0

        if num_prefills > 0:
            has_initial_state = context_lens_tensor > 0
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(non_spec_query_start_loc)
            )
        else:
            has_initial_state = None

        gdn_attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=batch.total_tokens_num,
            has_initial_state=has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        return gdn_attn_metadata

    def prepare_prefill(  # type: ignore[override]
        self,
        batch: ScheduledBatch,
    ) -> GDNAttentionMetadata:
        attn_metadata, positions = super().prepare_prefill(batch)
        if batch.block_tables == []:
            attn_metadata.gdn_metadata = None
            return attn_metadata, positions
        gdn_metadata = self.prepare_gdn_metadata(batch, attn_metadata, is_prefill=True)

        attn_metadata.gdn_metadata = gdn_metadata
        return attn_metadata, positions

    def prepare_decode(  # type: ignore[override]
        self,
        batch: ScheduledBatch,
        bs: int,
    ) -> GDNAttentionMetadata:
        num_decodes = batch.total_seqs_num_decode
        attn_metadata, positions = super().prepare_decode(batch, bs)
        self.model_runner.forward_vars["cu_seqlens_q"].cpu[
            bs:
        ] = batch.total_tokens_num_decode
        # we fill the attn_metadata cu_seqlens_q here since aiter attn won't calc it for decode
        attn_metadata.cu_seqlens_q = self.model_runner.forward_vars[
            "cu_seqlens_q"
        ].copy_to_gpu(bs + 1)

        gdn_metadata = self.prepare_gdn_metadata(batch, attn_metadata)

        # transfer data to ps buffer
        if self.use_spec_decode:
            self.spec_state_indices_tensor.gpu[num_decodes:, :].fill_(PAD_SLOT_ID)

            self.spec_sequence_masks[:num_decodes].copy_(
                gdn_metadata.spec_sequence_masks, non_blocking=True
            )
            self.spec_sequence_masks[num_decodes:].fill_(False)
            gdn_metadata.spec_sequence_masks = self.spec_sequence_masks[:num_decodes]

            self.spec_token_indx[: gdn_metadata.spec_token_indx.size(0)].copy_(
                gdn_metadata.spec_token_indx, non_blocking=True
            )
            gdn_metadata.spec_token_indx = self.spec_token_indx[
                : gdn_metadata.spec_token_indx.size(0)
            ]

            self.spec_query_start_loc[: num_decodes + 1].copy_(
                gdn_metadata.spec_query_start_loc[: num_decodes + 1], non_blocking=True
            )
            spec_num_query_tokens = self.spec_query_start_loc[num_decodes]
            self.spec_query_start_loc[num_decodes + 1 :].fill_(spec_num_query_tokens)
            gdn_metadata.spec_query_start_loc = self.spec_query_start_loc[
                : num_decodes + 1
            ]

            self.num_accepted_tokens[:num_decodes].copy_(
                gdn_metadata.num_accepted_tokens[:num_decodes], non_blocking=True
            )
            self.num_accepted_tokens[num_decodes:].fill_(1)
            gdn_metadata.num_accepted_tokens = self.num_accepted_tokens[:num_decodes]
        else:
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
        return attn_metadata, positions

    def prepare_mtp_decode(
        self,
        bs: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        only_update: bool = False,
        num_reject_tokens=None,
    ):
        var = self.model_runner.forward_vars

        # GDN hybrid models use paged KV cache for full-attention layers.
        # Regenerate kv_indices for the new max_seqlen_k after adding a
        # draft token; kv_indptr stays unchanged (block count is stable).
        # Note: only_update and num_reject_tokens are unused here — GDN's
        # paged attention does not use persistent worker buffers that need
        # incremental updates (unlike MLA). The full kv_indices regeneration
        # is always correct regardless of the update mode.
        kv_indptr = var["kv_indptr"].gpu[: bs + 1]
        kv_indices_generate_triton(
            var["block_tables"].gpu[:bs],
            var["kv_indices"].gpu,
            kv_indptr,
            self.block_ratio,
            max_seqlen_k,
        )

        result = {}
        if self.block_size == 1024:
            result = self.set_aiter_persistent_worker_buffers(bs)
        return result

    def build_for_cudagraph_capture(self, bs: int):
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
            kv_indices=var["kv_indices"].gpu[:],
            max_seqlen_k=self.model_runner.config.max_model_len,
            block_tables_converted=(
                var["block_tables_converted"].gpu[:bs]
                if "block_tables_converted" in var
                else None
            ),
            **ctx_pa_ps,
        )

        if self.use_spec_decode:
            gdn_metadata = GDNAttentionMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decodes=0,
                num_decode_tokens=0,
                num_spec_decodes=bs,
                num_spec_decode_tokens=bs * (self.num_spec + 1),
                num_actual_tokens=bs * (self.num_spec + 1),
                has_initial_state=None,
                spec_query_start_loc=self.spec_query_start_loc[: bs + 1],
                non_spec_query_start_loc=None,
                spec_state_indices_tensor=self.spec_state_indices_tensor.gpu[:bs],
                non_spec_state_indices_tensor=None,
                spec_sequence_masks=self.spec_sequence_masks[:bs],
                spec_token_indx=self.spec_token_indx[: bs * (self.num_spec + 1)],
                non_spec_token_indx=self.non_spec_token_indx[:0],
                num_accepted_tokens=self.num_accepted_tokens[:bs],
                nums_dict=None,
                batch_ptr=None,
                token_chunk_offset_ptr=None,
            )
        else:
            gdn_metadata = GDNAttentionMetadata(
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
                non_spec_state_indices_tensor=self.non_spec_state_indices_tensor.gpu[
                    :bs
                ],
                spec_sequence_masks=None,
                spec_token_indx=None,
                non_spec_token_indx=None,
                num_accepted_tokens=None,
                nums_dict=None,
                batch_ptr=None,
                token_chunk_offset_ptr=None,
            )
        attn_metadata.gdn_metadata = gdn_metadata

        positions = var["positions"].copy_to_gpu(bs)
        context = Context(
            positions=positions, is_prefill=False, batch_size=bs, graph_bs=bs
        )
        return attn_metadata, context


PAD_SLOT_ID = -1


def compute_causal_conv1d_metadata(query_start_loc_p: torch.Tensor):
    # Needed for causal_conv1d
    seqlens = query_start_loc_p.diff().to("cpu")
    nums_dict = {}  # type: ignore
    batch_ptr = None
    token_chunk_offset_ptr = None
    device = query_start_loc_p.device
    for BLOCK_M in [8]:  # cover all BLOCK_M values
        nums = -(-seqlens // BLOCK_M)
        nums_dict[BLOCK_M] = {}
        nums_dict[BLOCK_M]["nums"] = nums
        nums_dict[BLOCK_M]["tot"] = nums.sum().item()
        mlist = torch.from_numpy(np.repeat(np.arange(len(nums)), nums))
        nums_dict[BLOCK_M]["mlist"] = mlist
        mlist_len = len(nums_dict[BLOCK_M]["mlist"])
        nums_dict[BLOCK_M]["mlist_len"] = mlist_len
        MAX_NUM_PROGRAMS = max(1024, mlist_len) * 2
        offsetlist = []  # type: ignore
        for idx, num in enumerate(nums):
            offsetlist.extend(range(num))
        offsetlist = torch.tensor(offsetlist, dtype=torch.int32)
        nums_dict[BLOCK_M]["offsetlist"] = offsetlist

        if batch_ptr is None:
            # Update default value after class definition
            batch_ptr = torch.full(
                (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=device
            )
            token_chunk_offset_ptr = torch.full(
                (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=device
            )
        else:
            if batch_ptr.nelement() < MAX_NUM_PROGRAMS:
                batch_ptr.resize_(MAX_NUM_PROGRAMS).fill_(PAD_SLOT_ID)
                token_chunk_offset_ptr.resize_(MAX_NUM_PROGRAMS).fill_(  # type: ignore
                    PAD_SLOT_ID
                )

        batch_ptr[0:mlist_len].copy_(mlist)
        token_chunk_offset_ptr[0:mlist_len].copy_(offsetlist)  # type: ignore
        nums_dict[BLOCK_M]["batch_ptr"] = batch_ptr
        nums_dict[BLOCK_M]["token_chunk_offset_ptr"] = token_chunk_offset_ptr  # type: ignore

    return nums_dict, batch_ptr, token_chunk_offset_ptr
