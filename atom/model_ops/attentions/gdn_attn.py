# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from aiter.dist.parallel_state import get_tp_group

from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.state_runtime import StateTransfer
from atom.model_ops.attention_gdn import GatedDeltaNet
from atom.utils import CpuGpuBuffer
from atom.utils.forward_context import AttentionMetaData, Context

from .aiter_attention import (
    AiterAttentionMetadataBuilder,
    AiterBackend,
    kv_indices_generate_triton,
)
from .sub_pool_spec import SubPoolSpec, page_pool, state_pool


class GDNAttentionBackend(AiterBackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_GDN_ATTENTION"

    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["GatedDeltaNet"]:
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
    # Slots the incoming state is READ from, when a state fork makes that differ
    # from `non_spec_state_indices_tensor`. Same tensor otherwise; None on the
    # spec path, which never carries a fork.
    non_spec_state_indices_in_tensor: torch.Tensor | None = None
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


class GDNStateMixin:
    def __init__(self, model_runner, **kwargs):
        super().__init__(model_runner=model_runner, **kwargs)
        self._init_gdn_state(model_runner)

    def _init_gdn_state(
        self,
        model_runner,
    ):
        # Hybrid model layer-counting state (formerly set as a side effect
        # inside the qwen_next branch of the KV sizing path).
        # Promoted to runner attributes here so all consumers
        # (build_kv_cache_tensor, allocate_kv_cache_tensors, the per-req
        # cache hooks) can read them as `self.model_runner.<name>` without
        # a hidden ordering dependency on the KV sizing path being
        # called first.
        hf = model_runner.config.hf_config
        if getattr(hf, "model_type", None) == "kimi_linear":
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
        else:
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
        # Read side of a state fork. Only the prefill path can carry one (a
        # fork is always followed by at least `min_fork_tokens` prompt tokens),
        # so the spec/decode index buffers have no counterpart.
        self.non_spec_state_indices_in_tensor = CpuGpuBuffer(
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
        if (
            getattr(self.model_runner.config.hf_config, "model_type", None)
            == "kimi_linear"
        ):
            return (
                self.model_runner.config.torch_dtype,
                torch.float32,
            )
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

    def state_transfer(self) -> StateTransfer:
        """Declare one-token fork checkpoint support for recurrent state."""
        return StateTransfer.fork(1)

    def state_spec(self) -> SubPoolSpec:
        """The GDN state pool: conv_state + temporal_state over all GDN
        layers, with one extra slot per speculative token for rollback.

        Concrete builders splice this into their `sub_pool_specs()` alongside
        whatever paged KV pool they own.
        """
        shape_k, shape_v = self._state_shape_for_runner()
        dt_k, dt_v = self._state_dtypes()
        per_layer = (
            math.prod(shape_k) * dt_k.itemsize + math.prod(shape_v) * dt_v.itemsize
        )
        return state_pool(
            STATE_SLOT_CLASS,
            self.model_runner.num_gdn_attn_state * per_layer,
            entries_per_req=1 + self.num_spec,
        )

    def allocate_per_req_cache(
        self, entries: dict[str, int]
    ) -> dict[str, torch.Tensor]:
        """Allocate mamba_k_cache / mamba_v_cache.

        Names preserved for backward compat with `attention_gdn.py` which
        accesses them as `model_runner.mamba_{k,v}_cache`.
        """
        num_slots = entries.get(STATE_SLOT_CLASS, 0)
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

    def relocate_state_slots(self, pairs: Sequence[tuple[int, int]]) -> None:
        """Relocate a live GDN group between logical Active Slot spans.

        A group is `1 + num_spec` consecutive slots — the extra ones hold the
        per-draft states a rejected speculation rolls back to — so a group moves
        as that whole span or the rollback slots go with the wrong owner.

        GDN checkpoints by forking, not by copying, so this is not on the
        checkpoint path: it exists because moving the pool's boundary has to be
        able to relocate a group that is in the way.

        Both caches are layer-major with the slot as the second axis, so a
        group's rows are strided rather than contiguous and there is no single
        range to copy. `_foreach_copy_` keeps it to one launch for the batch.
        """
        span = 1 + self.num_spec
        caches = (self.model_runner.mamba_k_cache, self.model_runner.mamba_v_cache)
        destinations, sources = [], []
        for src_group, dst_group in pairs:
            src_slot, dst_slot = src_group * span, dst_group * span
            for cache in caches:
                destinations.append(cache[:, dst_slot : dst_slot + span])
                sources.append(cache[:, src_slot : src_slot + span])
        if destinations:
            torch._foreach_copy_(destinations, sources)

    def prepare_state_indices(self, batch: ScheduledBatch, with_spec: bool = False):
        non_spec_state_indices = self.non_spec_state_indices_tensor.np
        non_spec_state_indices_in = self.non_spec_state_indices_in_tensor.np
        spec_state_indices = self.spec_state_indices_tensor.np
        slots_per_group = 1 + self.num_spec
        fork_srcs = getattr(batch, "state_fork_srcs", None) or ()
        assert not (with_spec and any(s >= 0 for s in fork_srcs)), (
            "state fork on the spec-decode path: spec_state_indices_tensor has "
            "no read-side counterpart (BlockManager only forks onto prefill)"
        )
        for idx, slot_group in enumerate(batch.per_req_cache_groups):
            non_spec_state_indices[idx] = 0
            non_spec_state_indices_in[idx] = 0
            spec_state_indices[idx] = 0
            base = slot_group * slots_per_group

            if not with_spec:
                non_spec_state_indices[idx] = base
                # A forked seq reads the group it published (or resumed from)
                # and writes the fresh one for this forward only.
                src = fork_srcs[idx] if idx < len(fork_srcs) else -1
                non_spec_state_indices_in[idx] = (
                    src * slots_per_group if src >= 0 else base
                )
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
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        if not self.use_spec_decode or is_prefill:
            self.prepare_state_indices(batch, with_spec=False)
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = (
                self.non_spec_state_indices_tensor.copy_to_gpu(num_reqs)
            )
            # Always its own buffer, never aliased to the write tensor: this
            # branch also serves non-spec decode, which runs from a captured
            # CUDAGraph where the argument address is baked in at capture.
            non_spec_state_indices_in_tensor = (
                self.non_spec_state_indices_in_tensor.copy_to_gpu(num_reqs)
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
            non_spec_state_indices_in_tensor = None
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
            # Tokens already folded into each request's state before this
            # forward: earlier prefill chunks, or a resumed state checkpoint.
            # It has to be the chunk's START offset — `attn_metadata`'s
            # `context_lens` is the END (cached + scheduled) and would claim an
            # incoming state on a cold first chunk, making the recurrence start
            # from whatever the recycled state group still held. The backend
            # leaves `num_cached_tokens` None when no row has any, which is the
            # same all-False answer.
            cached = attn_metadata.num_cached_tokens
            has_initial_state = (
                cached[:num_prefills] > 0
                if cached is not None
                else torch.zeros(num_prefills, dtype=torch.bool, device=self.device)
            )
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
            non_spec_state_indices_in_tensor=non_spec_state_indices_in_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        return gdn_attn_metadata

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
            self.non_spec_state_indices_in_tensor.gpu[num_decodes:].fill_(PAD_SLOT_ID)

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
                non_spec_state_indices_in_tensor=(
                    self.non_spec_state_indices_in_tensor.gpu[:bs]
                ),
                spec_sequence_masks=None,
                spec_token_indx=None,
                non_spec_token_indx=None,
                num_accepted_tokens=None,
                nums_dict=None,
                batch_ptr=None,
                token_chunk_offset_ptr=None,
            )
        return gdn_metadata


class GDNAttentionMetadataBuilder(GDNStateMixin, AiterAttentionMetadataBuilder):

    reorder_batch_threshold: int = 1
    # `prepare_mtp_decode` below regenerates kv_indices and nothing else, so it
    # cannot absorb the position bump the fused path hands off to the backend.
    # Inherited as True from the MHA builder, which made `EagleProposer` pass
    # `update_context_lens` / `positions_out` into a signature that has neither.
    fuse_mtp_decode_position_update = False

    def sub_pool_specs(self) -> list[SubPoolSpec]:
        """GDN hybrid: a paged KV pool holding ONLY the full-attention layer
        slots, plus the per-request state pool for the linear-attention
        layers (`GDNStateMixin.state_spec`).
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
        return [page_pool(block_bytes), self.state_spec()]

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
        - everything else → defer to
          AiterAttentionMetadataBuilder.build_kv_cache_tensor
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
        attn_metadata, positions = super().prepare_decode(batch, bs)
        self.model_runner.forward_vars["cu_seqlens_q"].cpu[
            bs:
        ] = batch.total_tokens_num_decode
        # we fill the attn_metadata cu_seqlens_q here since aiter attn won't calc it for decode
        attn_metadata.cu_seqlens_q = self.model_runner.forward_vars[
            "cu_seqlens_q"
        ].copy_to_gpu(bs + 1)

        self._attach_gdn_decode_metadata(batch, attn_metadata)
        return attn_metadata, positions

    def prepare_mtp_decode(
        self,
        bs: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        positions: torch.Tensor,  # [total_tokens] int32
        only_update: bool = False,
        num_reject_tokens=None,
    ):
        running_bs = int(positions.shape[-1])  # rows; see the base contract
        var = self.model_runner.forward_vars

        # GDN hybrid models use paged KV cache for full-attention layers.
        # Regenerate kv_indices for the new max_seqlen_k after adding a
        # draft token; kv_indptr stays unchanged (block count is stable).
        # Note: only_update and num_reject_tokens are unused here — GDN's
        # paged attention does not use persistent worker buffers that need
        # incremental updates (unlike MLA). The full kv_indices regeneration
        # is always correct regardless of the update mode.
        kv_indptr = var["kv_indptr"].gpu[: running_bs + 1]
        kv_indices_generate_triton(
            var["block_tables"].gpu[:running_bs],
            var["kv_indices"].gpu,
            kv_indptr,
            self.block_ratio,
            max_seqlen_k,
        )

        result = {}
        if self.block_size == 1024:
            result = self.set_aiter_persistent_worker_buffers(running_bs)
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
            **ctx_pa_ps,
        )

        attn_metadata.gdn_metadata = self._build_gdn_capture_metadata(bs)

        positions = var["positions"].copy_to_gpu(bs)
        # A capture runs a full synthetic batch, so nothing is padded and the
        # scheduled shape is the running one.
        capture_tokens = bs * int(var["max_qlen"])
        context = Context(
            positions=positions,
            is_prefill=False,
            scheduled_bs=bs,
            running_bs=bs,
            scheduled_tokens=capture_tokens,
            running_tokens=capture_tokens,
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
