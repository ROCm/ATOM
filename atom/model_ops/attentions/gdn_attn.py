# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from aiter.dist.parallel_state import get_tp_group

from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.state_runtime import StateTransfer
from atom.model_ops.attention_gdn import GatedDeltaNet
from atom.model_ops.fla_ops.replayssm import (
    replayssm_buffer_shapes,
    replayssm_commit,
)
from atom.utils import CpuGpuBuffer, envs
from atom.utils.forward_context import AttentionMetaData, Context

from .aiter_attention import (
    AiterAttentionMetadataBuilder,
    AiterBackend,
    kv_indices_generate_triton,
)
from .paged_state_copy import (
    SegmentedCopyPlan,
    launch_copy_descriptor,
    plan_segmented_copy,
)
from .sub_pool_spec import SubPoolSpec, page_pool, state_pool

if TYPE_CHECKING:
    from atom.kv_transfer.disaggregation.types import KVTransferRegion

logger = logging.getLogger("atom")


def slot_indexed_caches(runner, replayssm: bool) -> list[torch.Tensor]:
    """The `(layer, slot, ...)` caches that together hold one slot's state.

    `relocate_state_slots` moves exactly these, and `state_slot_regions`
    describes exactly these; anything added to the pool belongs here so that
    neither can be extended without the other.

    Takes the runner rather than the builder so that both callers reach it the
    same way whether or not they are bound: the state tests drive those two
    methods unbound, against a stub that has the caches but no methods.
    """
    caches = [runner.mamba_k_cache, runner.mamba_v_cache]
    if replayssm:
        caches += [
            runner.replayssm_buf_k,
            runner.replayssm_buf_u,
            runner.replayssm_buf_g,
        ]
    return caches


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

    # Recurrent-state checkpoints this step must write, as the device index
    # tensors `_checkpoint_targets` builds, or None when it reaches none. Built
    # once per step and read by every layer.
    ssm_checkpoints: dict | None = None
    # First chunk index of each sequence within this step's `h`, the same
    # mapping the chunk kernel builds internally. Only computed when there are
    # checkpoints to place against it.
    ssm_chunk_offsets: torch.Tensor | None = None
    # --- ReplaySSM ---------------------------------------------------------
    # When enabled the recurrent state is NOT snapshotted per speculative
    # token, so `spec_state_indices_tensor` collapses to a single slot per
    # request and `slot_idx` (1-D) addresses both the checkpoint pool and the
    # record buffers.  `write_pos` is the per-slot committed-record cursor,
    # advanced once per forward by `replayssm_commit` (never by a layer).
    replayssm: bool = False
    slot_idx: torch.Tensor | None = None  # shape: [batch,]
    write_pos: torch.Tensor | None = None  # shape: [num_slots,]
    replayssm_cache_len: int = 0
    replayssm_route: str = "auto"
    replayssm_max_query_len: int = 1

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
        model_type = getattr(hf, "model_type", None)
        if model_type in ("kimi_linear", "glm5_next_text"):
            lin = getattr(hf, "linear_attn_config", {}) or {}
            # Kimi-Linear numbers these 1-based; GLM-5.3-Flash numbers them
            # 0-based (its layer_types[3] is the first full-attention layer, and
            # full_attn_layers correspondingly starts at 3).
            offset = 0 if model_type == "glm5_next_text" else 1
            if model_type == "glm5_next_text" and hasattr(hf, "glm5_full_attn_layers"):
                # The model normalizer derives these from layer_types when a
                # config revision omits the redundant linear-attn lists.
                model_runner.full_attention_layers = list(hf.glm5_full_attn_layers)
                model_runner.kda_attention_layers = list(hf.glm5_kda_layers)
            else:
                model_runner.full_attention_layers = [
                    int(i) - offset for i in lin.get("full_attn_layers", [])
                ]
                model_runner.kda_attention_layers = [
                    int(i) - offset for i in lin.get("kda_layers", [])
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

        # --- ReplaySSM ------------------------------------------------------
        # The verify window is mtp_k+1 tokens (anchor + drafts); the record
        # buffer has to hold two of them for the early-flush invariant.
        # Default to whatever speculative decoding is doing: ReplaySSM pays for
        # itself when there are drafts to roll back, and does not when there
        # are none.  `ATOM_ENABLE_REPLAYSSM` overrides in either direction.
        replayssm_override = envs.ATOM_ENABLE_REPLAYSSM
        self.replayssm = (
            self.use_spec_decode if replayssm_override is None else replayssm_override
        )
        self.replayssm_max_query_len = self.num_spec + 1
        self.replayssm_route = envs.ATOM_REPLAYSSM_ROUTE
        self.replayssm_cache_len = 0
        if self.replayssm:
            requested_cache_len = envs.ATOM_REPLAYSSM_CACHE_LEN
            min_cache_len = 2 * self.replayssm_max_query_len
            self.replayssm_cache_len = max(requested_cache_len, min_cache_len)
            if self.replayssm_cache_len != requested_cache_len:
                logger.warning(
                    "ATOM_REPLAYSSM_CACHE_LEN=%d is below the required "
                    "2*(mtp_k+1)=%d; raising it to %d.",
                    requested_cache_len,
                    min_cache_len,
                    self.replayssm_cache_len,
                )
            logger.info(
                "ReplaySSM enabled for linear attention (%s): cache_len=%d, "
                "route=%s, verify window=%d (1 state slot per request instead "
                "of %d).",
                (
                    "ATOM_ENABLE_REPLAYSSM=1"
                    if replayssm_override
                    else "default, speculative decoding is on"
                ),
                self.replayssm_cache_len,
                self.replayssm_route,
                self.replayssm_max_query_len,
                self.num_spec + 1,
            )
        else:
            logger.info(
                "ReplaySSM disabled for linear attention (%s).",
                (
                    "ATOM_ENABLE_REPLAYSSM=0"
                    if replayssm_override is not None
                    else "default, no speculative decoding"
                ),
            )

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
        # KDA recurrence accumulates in fp32 and aiter's chunk_kimi_delta_attn
        # reads the state back verbatim, so the temporal state must be fp32 for
        # every KDA model (Kimi-Linear and GLM-5.3-Flash).
        if getattr(self.model_runner.config.hf_config, "model_type", None) in (
            "kimi_linear",
            "glm5_next_text",
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

    def _replayssm_buffer_shapes(self):
        """Per-slot (k, u, g) record buffer shapes, or None when disabled."""
        if not self.replayssm:
            return None
        hf = self.model_runner.config.hf_config
        tp = get_tp_group().world_size
        return replayssm_buffer_shapes(
            self.replayssm_cache_len,
            hf.linear_num_value_heads // tp,
            hf.linear_key_head_dim,
            hf.linear_value_head_dim,
            self._is_kda(),
        )

    def _is_kda(self) -> bool:
        # Both KDA models, matching the two sibling checks above. GLM-5.3-Flash
        # was added to those and missed here, which mattered because this one
        # feeds `_replayssm_buffer_shapes()`: answering "not KDA" for a KDA
        # model sizes the ReplaySSM record buffers for the wrong head geometry
        # and the recurrence writes past them.
        return getattr(self.model_runner.config.hf_config, "model_type", None) in (
            "kimi_linear",
            "glm5_next_text",
        )

    def _replayssm_bytes_per_slot(self) -> int:
        """Record-buffer bytes per slot, summed over all linear-attn layers."""
        shapes = self._replayssm_buffer_shapes()
        if shapes is None:
            return 0
        rec_dtype = self.model_runner.config.torch_dtype
        sk, su, sg = shapes
        per_layer = (
            math.prod(sk) * rec_dtype.itemsize
            + math.prod(su) * rec_dtype.itemsize
            + math.prod(sg) * 4  # g stays fp32: it is exponentiated on rebuild
        )
        return self.model_runner.num_gdn_attn_state * per_layer

    def state_transfer(self) -> StateTransfer:
        """A fork whose successor forward need only carry one token.

        Both halves of the GDN state come out of a forward self-contained at any
        length. The recurrent state is rewritten whole, and every write path in
        `causal_conv1d` stores the full `state_len` window to the output slot —
        the short-chunk paths get there by loading the previous window from the
        *input* slot, shifting left and appending x — so the new slot stops
        depending on the old one the moment the forward returns. The layout
        alone would suggest `conv_kernel_dim - 1`; the kernel closes that gap.

        A fork rather than a copy because the state is two per-family tensors
        rather than one contiguous entry, so there is no single range to
        duplicate — and at one token the fork binds almost nothing anyway.

        NOT midstep-readable, though the machinery for it is present and its
        numerical claim holds. The chunk kernel materializes the recurrent
        state at every 64-token boundary, `write_state_checkpoints` copies
        those out, and `tests/test_gdn_midstep_state_gpu.py` shows a slice of
        `h` is bit-exact against a forward stopped there. What is missing is
        everything between: the write path declines on six conditions that
        `commit_midstep` cannot see, its row index spans three differently
        scoped sequence lists, and its SSM read floors to a 64 grid that
        `midstep_positions` does not enforce (`hash_block_size` defaults to
        16). Each of those stores a findable image holding the wrong state,
        which is worse than storing nothing.

        None of it has ever run under a server: Kimi-K3 takes the PAGE path and
        cannot reach this one, so every measurement in this area is of the
        other mechanism. Declaring `False` costs a shortened prefill chunk per
        placement — the cost every backend paid before — and is what the
        evidence supports. Flip it back with the fixes and an end-to-end run,
        not before.

        Exact, not approximate, when it is turned back on: `h` is `k.new_empty`
        and `_state_dtypes` returns `config.torch_dtype`, so slicing `h` rounds
        exactly where a shortened forward would. That rests on the two dtypes
        agreeing; kimi_linear's fp32 v side is the one pool that breaks it, and
        it overrides (`_KimiMLAGDNCommon.state_transfer`).
        """
        return StateTransfer.fork(1, readable_midstep=False)

    def state_spec(self) -> SubPoolSpec:
        """The GDN state pool: conv_state + temporal_state over all GDN
        layers, with one extra slot per speculative token for rollback.

        Under ReplaySSM there is no per-draft fan-out: `slots_per_req()` drops
        to 1 and the (k, u, g) record buffers ride along in the same per-slot
        budget.

        Concrete builders splice this into their `sub_pool_specs()` alongside
        whatever paged KV pool they own.

        Sized for in-flight requests and nothing more. A retained checkpoint
        sits in a slot `max_num_seqs` left spare, so how many can be kept is
        set by concurrency rather than by how much reuse the traffic has —
        which is why a *lower* max_num_seqs measures a *worse* hit rate on
        prefix-reusing traffic. Decoupling the two is what the PAGE path does,
        by keeping the image in KV blocks instead of a slot; a flat cushion
        here would buy the same thing for `fork` at the cost of a knob nobody
        can size without measuring, so it is not offered.
        """
        shape_k, shape_v = self._state_shape_for_runner()
        dt_k, dt_v = self._state_dtypes()
        per_layer = (
            math.prod(shape_k) * dt_k.itemsize + math.prod(shape_v) * dt_v.itemsize
        )
        return state_pool(
            STATE_SLOT_CLASS,
            self.model_runner.num_gdn_attn_state * per_layer
            + self._replayssm_bytes_per_slot(),
            entries_per_req=self.slots_per_req(),
        )

    def slots_per_req(self) -> int:
        """Baseline GDN reserves one extra state slot per speculative token so
        a rejected draft can be rolled back by resuming from a different slot.

        ReplaySSM reconstructs the state from cached inputs instead, so one
        slot per request is enough regardless of the MTP window.  (This also
        drops the conv-state over-allocation that came along for the ride:
        `causal_conv1d_update` only ever addresses column 0, because it rolls
        the conv window back in place via `num_accepted_tokens`.)
        """
        return 1 if self.replayssm else 1 + self.num_spec

    def allocate_per_req_cache(
        self, entries: dict[str, int]
    ) -> dict[str, torch.Tensor]:
        """Allocate mamba_k_cache / mamba_v_cache (+ ReplaySSM buffers).

        Names preserved for backward compat with `attention_gdn.py` which
        accesses them as `model_runner.mamba_{k,v}_cache`.
        """
        num_slots = entries.get(STATE_SLOT_CLASS, 0)
        shape_k, shape_v = self._state_shape_for_runner()
        dt_k, dt_v = self._state_dtypes()
        n = self.model_runner.num_gdn_attn_state
        tensors = {
            "mamba_k_cache": torch.zeros(
                (n, num_slots) + shape_k, dtype=dt_k, device="cuda"
            ),
            "mamba_v_cache": torch.zeros(
                (n, num_slots) + shape_v, dtype=dt_v, device="cuda"
            ),
        }
        shapes = self._replayssm_buffer_shapes()
        if shapes is not None:
            sk, su, sg = shapes
            rec_dtype = self.model_runner.config.torch_dtype
            tensors["replayssm_buf_k"] = torch.zeros(
                (n, num_slots) + sk, dtype=rec_dtype, device="cuda"
            )
            tensors["replayssm_buf_u"] = torch.zeros(
                (n, num_slots) + su, dtype=rec_dtype, device="cuda"
            )
            tensors["replayssm_buf_g"] = torch.zeros(
                (n, num_slots) + sg, dtype=torch.float32, device="cuda"
            )
            # One cursor per slot, shared by every linear-attention layer:
            # the record index depends on the sequence's decode history, not
            # on which layer is running.
            tensors["replayssm_write_pos"] = torch.zeros(
                num_slots, dtype=torch.int32, device="cuda"
            )
        self._assert_checkpoint_geometry_still_holds()
        return tensors

    def _assert_checkpoint_geometry_still_holds(self) -> None:
        """The spec was built during sizing; check the pool agrees with it.

        `checkpoint_image_bytes` is asked once before these tensors exist and
        the answer travels to the scheduler as `PagedStateCheckpointSpec`. If
        the shapes moved in between, the scheduler would reserve PAGE units for
        one image while the worker scattered a different one — raw pointers, so
        the first sign would be corrupt state rather than an error.

        A no-op unless this run copies; a fork keeps no spec.
        """
        runtime = getattr(self.model_runner, "state_runtime", None)
        spec = None if runtime is None else runtime.checkpoint_spec
        if spec is None:
            return
        image = self.checkpoint_image_bytes()
        if image != spec.image_bytes:
            raise RuntimeError(
                f"the state pool holds a {image} B checkpoint image but the "
                f"scheduler reserved units for {spec.image_bytes} B"
            )
        if runtime.transfer.paged_layout_id != spec.layout_id:
            raise RuntimeError(
                f"state layout {runtime.transfer.paged_layout_id!r} does not "
                f"match the spec's {spec.layout_id!r}"
            )

    # ------------------------ PAGE-copy checkpoints ------------------------ #
    #
    # A checkpoint image is a byte copy of one Active Slot into
    # `ceil(image_bytes / page_unit_bytes)` ordinary KV blocks, run by
    # `PagedStateCheckpointCoordinator`. Everything below is the source side of
    # that copy: this class owns `mamba_{k,v}_cache`, so it can say where a
    # slot's bytes are, but not where a PAGE unit is -- that belongs to
    # whichever builder owns the paged pool (`_page_unit_regions`).
    #
    # Dormant under `StateTransfer.fork`: nothing produces a store op unless a
    # subclass declares `copy()`. Only `_KimiMLAGDNCommon` does today.

    def _checkpoint_layer_ranges(self) -> list[list[tuple[int, int]]]:
        """Per plane, one `(offset, nbytes)` per layer for slot 0.

        Both caches are `(num_layers, num_slots, *state)` and contiguous, so
        layer L of slot S is the single range at `(L * num_slots + S) *
        per_layer_bytes`. This returns the S=0 column; a real slot adds
        `S * per_layer_bytes` to every offset, which `_checkpoint_slot_bases`
        does in one vectorised step.

        Plane order is conv (`mamba_k_cache`) then ssm (`mamba_v_cache`), all
        layers of one before any of the other -- stated in the layout id as
        `order=conv-all-layers,ssm-all-layers`, because shapes alone cannot say
        it and a reader assembling the image interleaved would get every layer
        but the first wrong.

        The whole slot is carried: unlike V4's compressor ring, no part of a
        KDA state is provably dead at a boundary.

        Sole owner of the segment order. `_checkpoint_segment_sizes` and
        `_checkpoint_slot_bases` both read it rather than each walking the
        planes themselves, which is how a plan's segment index and an address
        row stay talking about the same segment.
        """
        num_slots = self._checkpoint_num_slots()
        return [
            [(layer * num_slots * nbytes, nbytes) for layer in range(n_layers)]
            for nbytes, n_layers in self._checkpoint_plane_shapes()
        ]

    def _checkpoint_plane_shapes(self) -> list[tuple[int, int]]:
        """`(per_layer_bytes, num_layers)` per plane, conv first then ssm.

        Pure geometry from the state shapes, so it answers before the tensors
        exist -- which `checkpoint_image_bytes` needs, being called during
        sizing.
        """
        shape_k, shape_v = self._state_shape_for_runner()
        dt_k, dt_v = self._state_dtypes()
        n = self.model_runner.num_gdn_attn_state
        return [
            (math.prod(shape_k) * dt_k.itemsize, n),
            (math.prod(shape_v) * dt_v.itemsize, n),
        ]

    def _checkpoint_plane_tensors(self) -> list[torch.Tensor]:
        """Allocated source planes in the order `_checkpoint_plane_shapes` names.

        Kept separate from the pure geometry method because checkpoint sizing
        runs before these tensors exist. Hybrid backends may extend both lists
        with state that shares the same slot lifetime.
        """
        return [
            self.model_runner.mamba_k_cache,
            self.model_runner.mamba_v_cache,
        ]

    def _checkpoint_num_slots(self) -> int:
        """The slot axis of the state tensors, or 1 before they exist.

        Only the *offsets* scale with it; the image's size does not. So a
        pre-allocation caller (sizing) gets a consistent answer from the same
        code the post-allocation callers use.
        """
        cache = getattr(self.model_runner, "mamba_k_cache", None)
        return 1 if cache is None else int(cache.shape[1])

    def _checkpoint_segment_sizes(self) -> list[int]:
        """The image as the copy planner reads it: one size per source segment.

        The one place the per-plane ranges are flattened. Sizing wants their
        total and the planner wants the list, and the two answering from
        different comprehensions is how an image gets priced at one shape and
        cut at another.
        """
        return [
            nbytes for ranges in self._checkpoint_layer_ranges() for _, nbytes in ranges
        ]

    def checkpoint_image_bytes(self) -> int:
        """Bytes one checkpoint image holds. Priced before the pool exists.

        Independent of `num_slots`: a checkpoint is one slot, and the slot
        count only moves where slots sit, not how big one is.
        """
        return sum(nbytes * n for nbytes, n in self._checkpoint_plane_shapes())

    def _checkpoint_slot_bases(self) -> np.ndarray:
        """`[slot, segment]` start address of every source segment of a copy.

        Segments in the order `_checkpoint_layer_ranges` walks the planes,
        which is the order `_checkpoint_copy_plan` builds the source stream
        in -- so a plan's segment index addresses a row of this directly.

        Built as one vectorised expression rather than V4's per-slot tensor
        views: a K3 slot's offsets are affine in `(layer, slot)`, so `2 * S`
        materialised views would buy addresses that are one multiplication
        each.

        Keyed on the addresses and the slot count it was built from rather
        than cleared by a hook, so a pool that moves underneath invalidates
        this by disagreeing with its own key.
        """
        plane_tensors = getattr(self, "_checkpoint_plane_tensors", None)
        planes = (
            GDNStateMixin._checkpoint_plane_tensors(self)
            if plane_tensors is None
            else plane_tensors()
        )
        num_slots = int(planes[0].shape[1])
        key = tuple(cache.data_ptr() for cache in planes) + (num_slots,)
        cached = getattr(self, "_checkpoint_slot_base_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]

        slots = np.arange(num_slots, dtype=np.int64)[:, None]
        columns = []
        for cache, (nbytes, n_layers) in zip(
            planes, self._checkpoint_plane_shapes(), strict=True
        ):
            if not cache.is_contiguous():
                raise RuntimeError(
                    "a PAGE-copy state plane must be contiguous; "
                    f"{tuple(cache.shape)} stride {cache.stride()} is not"
                )
            if int(cache.shape[0]) != n_layers or int(cache.shape[1]) != num_slots:
                raise RuntimeError(
                    "a PAGE-copy state plane disagrees with its checkpoint "
                    f"geometry: tensor={tuple(cache.shape[:2])}, "
                    f"geometry=({n_layers}, {num_slots})"
                )
            layers = np.arange(n_layers, dtype=np.int64)[None, :]
            columns.append(cache.data_ptr() + (layers * num_slots + slots) * nbytes)
        bases = np.hstack(columns)
        self._checkpoint_slot_base_cache = (key, bases)
        return bases

    def _page_unit_regions(self) -> tuple[np.ndarray, np.ndarray]:
        """Base address and per-unit stride of every region a PAGE id owns.

        The destination side of the copy, which lives in whatever paged pool
        the concrete builder owns -- this class only knows about the state
        planes. `_KimiMLAGDNCommon` answers it for K3's MLA pool.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares PAGE-copy checkpoints but does "
            "not say where a PAGE unit is"
        )

    def _checkpoint_copy_plan(self) -> SegmentedCopyPlan:
        """Where a slot's segments meet a whole image's PAGE regions.

        Both streams are geometry. The state ranges come from the layout, and
        every image is `units_per_checkpoint` units of identical region sizes --
        `_validate_paged_state_op` refuses anything else. So the cut points are
        the same for every store and every restore this worker will ever do,
        and the walk that finds them runs once instead of once an op.
        """
        if getattr(self, "_checkpoint_plan_cache", None) is None:
            spec = self.model_runner.state_runtime.checkpoint_spec
            self._checkpoint_plan_cache = plan_segmented_copy(
                self._checkpoint_segment_sizes(),
                # Sizes from the same array `_page_unit_bases` takes addresses
                # from, tiled the way it ravels. Spelling the destination
                # stream out a second time here would let the two orders
                # diverge, and a plan cut against one order and addressed
                # through the other lands whole regions in the wrong unit.
                self._page_unit_stream_sizes(spec.units_per_checkpoint),
                spec.image_bytes,
            )
        return self._checkpoint_plan_cache

    def _checkpoint_descriptor_buffer(self) -> CpuGpuBuffer:
        """Pinned staging for a step's whole descriptor, sized for the worst step.

        Pinned because the alternative synchronizes: a pageable H2D from
        `build()` makes the host wait out the forward already enqueued. Reused
        because allocating pinned memory is itself a synchronizing call.

        A step can carry at most one store and one restore per sequence, so two
        per sequence bounds it. The caller checks that bound rather than growing
        on demand: a descriptor that did not fit would otherwise be silently
        truncated into a copy of the wrong shape.

        The store half of that bound is not a property of the batch -- it is
        held by `PagedStateCheckpointCoordinator._supersede`, which keeps one
        pending boundary per sequence. A change that let two of a sequence's
        boundaries drain together would raise from `build()` here, and would be
        storing one of them from the wrong slot besides; the two constraints
        have the same owner and move together.
        """
        if getattr(self, "_checkpoint_descriptor", None) is None:
            plan = self._checkpoint_copy_plan()
            max_ops = 2 * int(self.model_runner.config.max_num_seqs)
            self._checkpoint_descriptor = CpuGpuBuffer(
                max_ops * plan.num_spans,
                3,
                dtype=torch.int64,
                device=self.model_runner.mamba_k_cache.device,
            )
        return self._checkpoint_descriptor

    def _validate_paged_state_op(self, op) -> None:
        """Refuse an op this worker cannot honour, before it addresses memory.

        `layout_id` is the cross-worker check: the scheduler priced the image
        against one geometry and this worker reassembles it against its own, so
        a mismatch would put every byte at the wrong offset rather than fail.

        Unit ids are checked against the **logical** block count, which is what
        `BlockPool` hands out and what `sub_pool_specs` priced. The tensor is
        shaped in physical blocks and K3's `block_ratio` is 128, so a check
        against the physical count would admit ids 128x out of range.
        """
        spec = self.model_runner.state_runtime.checkpoint_spec
        if spec is None:
            raise RuntimeError("a paged state copy arrived with no checkpoint spec")
        if op.layout_id != spec.layout_id:
            raise RuntimeError(
                f"state checkpoint layout mismatch: op {op.layout_id!r} "
                f"against this worker's {spec.layout_id!r}"
            )
        if op.total_bytes != spec.image_bytes:
            raise RuntimeError(
                f"a checkpoint image is {spec.image_bytes} B but the op names "
                f"{op.total_bytes}"
            )
        if len(op.unit_ids) != spec.units_per_checkpoint:
            raise RuntimeError(
                f"a checkpoint takes {spec.units_per_checkpoint} PAGE units but "
                f"the op names {len(op.unit_ids)}"
            )
        num_blocks = int(self.model_runner.config.num_kvcache_blocks)
        if any(unit < 0 or unit >= num_blocks for unit in op.unit_ids):
            raise RuntimeError("state checkpoint PAGE unit is out of range")

    def execute_paged_state_copies(self, store_ops, restore_ops) -> None:
        """Copy raw checkpoint bytes between slots and non-contiguous PAGEs.

        Every op of either direction goes into one descriptor and one launch.
        A store and a restore are the same intersection read opposite ways, so
        they share the plan too -- and each direction is described in a single
        vectorised pass, which is why they are batched apart rather than
        interleaved.
        """
        if not store_ops and not restore_ops:
            return
        for op in (*store_ops, *restore_ops):
            self._validate_paged_state_op(op)

        plan = self._checkpoint_copy_plan()
        slot_bases = self._checkpoint_slot_bases()
        per_op = plan.num_spans
        total = (len(store_ops) + len(restore_ops)) * per_op
        staging = self._checkpoint_descriptor_buffer()
        if total > staging.np.shape[0]:
            raise RuntimeError(
                f"a step asked to copy {total // per_op} checkpoints, more "
                f"than the {staging.np.shape[0] // per_op} its descriptor was "
                "sized for"
            )
        descriptor = staging.np[:total]
        at = 0
        for ops, storing in ((store_ops, True), (restore_ops, False)):
            if not ops:
                continue
            end = at + len(ops) * per_op
            slots = [op.src_slot if storing else op.dst_slot for op in ops]
            plan.write_descriptor(
                descriptor[at:end],
                slot_bases[slots],
                self._page_unit_bases([op.unit_ids for op in ops]),
                forward=storing,
            )
            at = end
        launch_copy_descriptor(staging.copy_to_gpu(total), plan)

    def warmup_per_req_cache(self) -> None:
        """Run one checkpoint copy now, so the first real one is only a copy.

        `execute_paged_state_copies` is reachable only from `build()`, so
        everything it builds lazily -- the copy plan, the slot base table, the
        tiling's upload, the pinned descriptor, and the Triton JIT of
        `_copy_tiles_kernel` -- otherwise lands inside the batch of whichever
        request first crosses a rung.

        Slot 0 into the pool's first units. Both are real addresses, which is
        the point: a warmup on scratch would compile a kernel and fill nothing.
        The bytes it writes are read by nobody -- a KV block is written before
        it is read, and this runs before any block has been handed out.
        """
        runtime = getattr(self.model_runner, "state_runtime", None)
        spec = None if runtime is None else runtime.checkpoint_spec
        if spec is None:
            return
        plan = self._checkpoint_copy_plan()
        if not plan.num_spans:
            return
        staging = self._checkpoint_descriptor_buffer()
        plan.write_descriptor(
            staging.np[: plan.num_spans],
            self._checkpoint_slot_bases()[:1],
            self._page_unit_bases([list(range(spec.units_per_checkpoint))]),
        )
        launch_copy_descriptor(staging.copy_to_gpu(plan.num_spans), plan)

    def state_entry_views(self, slot: int) -> list[torch.Tensor]:
        """One contiguous slice per (cache, layer) — the slot's whole state.

        Both caches are layer-major with the slot on axis 1, so one slot's rows
        are strided and there is no single range covering them. Slicing per
        layer makes each piece contiguous, which is what the staging packer
        requires; `relocate_state_slots` keeps its own strided views because
        `_foreach_copy_` has no such constraint and one launch beats `LAYERS`.

        One slot, not a request's whole set: a checkpoint is exactly the
        committed state (#2045), and the speculation scratch beside it is this
        request's own and resumable by nobody.
        """
        # Reject an out-of-range slot in BOTH directions, for parity with the
        # DSV4 twin (`deepseek_v4_attn.py`), whose `self._slot_views()[slot]` is
        # a list index and raises `IndexError` on a stray -1 *or* an
        # out-of-range positive. Here `cache[layer, slot : slot + 1]` clamps
        # silently instead: a -1 (the "no slot" sentinel used elsewhere on this
        # path) gathers/scatters the last slot's state under this request's
        # identity, and a too-large slot yields a zero-element view
        # (`numel() == 0`, no error) that the staging packer moves as 0 bytes --
        # `StateByteCodec.get` then returns True on state never written and the
        # request resumes on whatever the slot already held. Either way: silent
        # cross-request corruption. The upstream invariant is a real, in-range
        # slot (`StateSlotPool.pop` never returns negative, and
        # `_attach_state_slots` pops before requesting the load), so this guards
        # the shape rather than a demonstrated path; fail loudly if it is ever
        # violated instead of corrupting silently.
        num_slots = self.model_runner.mamba_k_cache.shape[1]
        if not 0 <= slot < num_slots:
            raise IndexError(
                f"state_entry_views: slot {slot} out of range [0, {num_slots})"
            )
        views = []
        for cache in (
            self.model_runner.mamba_k_cache,
            self.model_runner.mamba_v_cache,
        ):
            for layer in range(cache.shape[0]):
                views.append(cache[layer, slot : slot + 1])
        return views

    def state_slot_regions(self) -> tuple[list["KVTransferRegion"], int]:
        """RDMA regions for the state pool, addressed by state slot.

        The third view of the bytes `relocate_state_slots` moves and
        `state_entry_views` names, for the one consumer that needs neither a
        copy nor a view but an address: a peer's NIC.

        One region per (cache, layer), not per cache. A region resolves an
        index as `base + slot * unit`, so a slot's bytes must be contiguous
        and evenly strided -- true within a layer, false across them, since
        every cache is `(layer, slot) + state_shape` and a slot's rows are a
        layer-stride apart.

        Returns the regions and the pool's slot count. The count is the
        allocated width, not the live request count: it is what makes a
        region's extent, and both ends compute addresses from it.
        """
        from atom.kv_transfer.disaggregation.types import KVTransferRegion

        caches = slot_indexed_caches(self.model_runner, self.replayssm)
        num_slots = caches[0].shape[1]
        regions: list[KVTransferRegion] = []
        for cache in caches:
            assert cache.is_contiguous(), (
                "slot-indexed state cache must be contiguous: a transfer "
                "region addresses a slot as base + slot * unit_bytes"
            )
            assert cache.shape[1] == num_slots, (
                f"state caches disagree on slot count: {cache.shape[1]} != "
                f"{num_slots}; one slot index has to address all of them"
            )
            slot_bytes = cache[0, 0].numel() * cache.element_size()
            layer_bytes = num_slots * slot_bytes
            for layer in range(cache.shape[0]):
                regions.append(
                    KVTransferRegion(
                        base_addr=cache.data_ptr() + layer * layer_bytes,
                        total_bytes=layer_bytes,
                        unit_bytes=slot_bytes,
                    )
                )

        if self.replayssm:
            # The cursor has no layer axis -- one int32 per slot, shared by
            # every layer -- so the whole tensor is a single region whose unit
            # is one element. Without it the records above are unreadable: the
            # reader cannot tell which of them the sequence has committed.
            pos = self.model_runner.replayssm_write_pos
            regions.append(
                KVTransferRegion(
                    base_addr=pos.data_ptr(),
                    total_bytes=pos.numel() * pos.element_size(),
                    unit_bytes=pos.element_size(),
                )
            )

        return regions, num_slots

    def relocate_state_slots(self, pairs: Sequence[tuple[int, int]]) -> None:
        """Relocate a live GDN state slot between Active Slot positions.

        A slot is one complete recurrent state and moves on its own. A request
        holding several — a committed state plus `num_spec` rollback slots —
        is several such moves, and the caller names each one, because nothing
        about the set is contiguous.

        GDN checkpoints by forking, not by copying, so this is not on the
        checkpoint path: it exists because moving the pool's boundary has to be
        able to relocate a slot that is in the way, and relocation is a byte
        move whatever mechanism the class uses to checkpoint. A backend
        declaring `StateTransfer.fork` therefore still owes this method.

        Under ReplaySSM the records and the cursor travel with the slot. They
        are not an accelerator for the state, they are part of it: the
        checkpoint alone only describes the sequence up to the last flush, and
        a slot relocated without them resumes against whatever the destination
        slot's previous tenant left behind.

        Both caches are layer-major with the slot as the second axis, so one
        slot's rows are strided rather than contiguous and there is no single
        range to copy. `_foreach_copy_` keeps it to one launch for the batch.
        The cursor is the exception — one entry per slot, no layer axis.
        """
        runner = self.model_runner
        caches = slot_indexed_caches(runner, self.replayssm)
        cursor = None
        if self.replayssm:
            # One entry per slot, no layer axis -- moved alongside, not with
            # the layer-major caches above.
            cursor = runner.replayssm_write_pos
        destinations, sources = [], []
        for src, dst in pairs:
            for cache in caches:
                destinations.append(cache[:, dst])
                sources.append(cache[:, src])
            if cursor is not None:
                destinations.append(cursor[dst : dst + 1])
                sources.append(cursor[src : src + 1])
        if destinations:
            torch._foreach_copy_(destinations, sources)

    def _checkpoint_targets(self, batch: ScheduledBatch) -> dict | None:
        """Checkpoints this step reaches, as device index tensors.

        Every reserved position this step covers, `cached < p <= cached +
        scheduled`, is a target — INCLUDING one at the step's end. That end
        case needs its own copy like any other: the chunk kernel leaves the
        final state in the sequence's RUNTIME slot, and a checkpoint slot is
        never the runtime slot. Assuming otherwise leaves the checkpoint
        unwritten while `commit_midstep` publishes it anyway, so a later
        request resumes from whatever the slot's previous tenant left behind.

        `is_end` marks those targets, because their source differs: the state
        at the end of a sequence's tokens is not in `h`, which holds chunk
        boundaries strictly before the end — `chunk_offsets[row] + T // 64` is
        already the NEXT sequence's first chunk. It exists only in the runtime
        slot, so the kernel reads `runtime_slots[i]` instead.

        Built once per step, not per layer: every GDN layer copies the same
        targets, so the H2D transfer is hoisted here and each layer just
        launches one kernel over it.

        Offsets are relative to the start of the sequence's slice OF THIS
        STEP. `h` and the conv input only ever hold this step's tokens, and
        `cu_seqlens` / `chunk_offsets` locate each sequence within them — so
        the kernel reconstructs an absolute index as `cu_seqlens[row] + off`
        (conv) or `chunk_offsets[row] + off // 64` (SSM). Both bases are
        per-sequence; a shared base silently captures one sequence's state into
        another's checkpoint whenever a batch holds two prefills.

        A target is dropped when this step holds too few tokens before it to
        fill the conv window. Both halves of a checkpoint must land together —
        an SSM state at P paired with a conv window from elsewhere is silently
        wrong, and worse than no checkpoint at all, because it is findable.

        Slots, not a separate checkpoint region: a checkpoint here IS an
        ordinary pool slot, indexed exactly as every other slot on this path
        is. One slot is the whole checkpoint — a resumed prefix has no
        speculation to roll back, so it needs no scratch beside it.
        """
        all_saves = getattr(batch, "state_save_all", None)
        if not all_saves:
            return None
        # Tokens of conv history a checkpoint needs behind it: the conv state
        # width. From the config, so it tracks the model rather than assuming.
        state_len = self.model_runner.config.hf_config.linear_conv_kernel_dim - 1
        cached = batch.num_cached_tokens
        sched = batch.num_scheduled_tokens
        runtime_slots = batch.state_slots_committed
        limit = self.model_runner.mamba_k_cache.shape[1]

        found = []
        # A seq may hold several reservations (a grid rung, a demand, the
        # prompt-end anchor); take every one this step reaches.
        for i, reservations in enumerate(all_saves):
            if i >= len(runtime_slots):
                continue
            start = int(cached[i])
            end = start + int(sched[i])
            for dst_slot, p in reservations:
                dst = int(dst_slot)
                p = int(p)
                # `dst >= limit` would mean the scheduler's pool outgrew this
                # rank's tensor; skipping degrades to "no checkpoint", which
                # is always safe, where writing would corrupt another slot.
                if not 0 <= dst < limit:
                    continue
                if not (start + state_len <= p <= end):
                    continue
                found.append((i, dst, p - start, int(p == end), runtime_slots[i]))
        if not found:
            return None

        def mk(col):
            return torch.tensor(col, dtype=torch.int32, device=self.device)

        rows, slots, offs, is_end, runtime = zip(*found)
        return {
            "rows": mk(rows),
            "slots": mk(slots),
            "offs": mk(offs),
            "is_end": mk(is_end),
            "runtime": mk(runtime),
        }

    def prepare_state_indices(self, batch: ScheduledBatch, with_spec: bool = False):
        """Fill the index tensors the GDN kernels gather their state through.

        The seq's own slot list is written straight in — no base, no stride.
        The pool hands out slots one at a time and a request's set is not
        adjacent; the kernels never assumed it was (the ssm kernel loads each
        index out of this tensor, and the conv path is handed column 0 alone),
        so this is where a contiguity assumption would have been *invented*
        rather than a place one has to be honoured.
        """
        non_spec_state_indices = self.non_spec_state_indices_tensor.np
        non_spec_state_indices_in = self.non_spec_state_indices_in_tensor.np
        spec_state_indices = self.spec_state_indices_tensor.np
        fork_srcs = getattr(batch, "state_fork_srcs", None) or ()
        assert not (with_spec and any(s >= 0 for s in fork_srcs)), (
            "state fork on the spec-decode path: spec_state_indices_tensor has "
            "no read-side counterpart (BlockManager only forks onto prefill)"
        )
        for idx, slots in enumerate(batch.state_slots):
            non_spec_state_indices[idx] = 0
            non_spec_state_indices_in[idx] = 0
            spec_state_indices[idx] = 0
            committed = slots[0]

            if not with_spec:
                non_spec_state_indices[idx] = committed
                # A forked seq reads the slot it published (or resumed from)
                # and writes the fresh one for this forward only. The source is
                # a checkpoint, which is one slot, so it needs no translation.
                src = fork_srcs[idx] if idx < len(fork_srcs) else -1
                non_spec_state_indices_in[idx] = src if src >= 0 else committed
            else:
                spec_state_indices[idx, : len(slots)] = slots
                if self.replayssm:
                    # ReplaySSM holds one slot per request, and that same index
                    # addresses the checkpoint pool and the record buffers
                    # alike. Mirror it into the 1-D tensor so `slot_idx` has
                    # somewhere to read it from -- the spec path otherwise
                    # fills only the 2-D one.
                    non_spec_state_indices[idx] = committed

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

        # Prefill only: `prepare_prefill` pads this past the requests it
        # scheduled, for a draft pass that follows. Decode's padding is older
        # than that and everything below is written for it.
        query_start_loc = attn_metadata.cu_seqlens_q
        if is_prefill:
            query_start_loc = query_start_loc[: num_prefills + 1]
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
            if self.replayssm:
                # `prepare_state_indices` mirrored the single slot into the
                # 1-D tensor too; ship that to the device for `slot_idx`.
                self.non_spec_state_indices_tensor.copy_to_gpu(num_reqs)
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
        if self.replayssm:
            self._attach_replayssm(gdn_attn_metadata, num_reqs, is_prefill)
        return gdn_attn_metadata

    def _attach_replayssm(
        self, md: GDNAttentionMetadata, num_reqs: int, is_prefill: bool
    ) -> None:
        """Fill the ReplaySSM fields and move the record cursor exactly once.

        The cursor advance is a *forward-level* action, not a layer-level one:
        every linear-attention layer in the step must see the same `write_pos`,
        so it happens here (metadata prep, outside any captured graph) rather
        than inside the layer kernel.
        """
        slot_idx = self.non_spec_state_indices_tensor.gpu[:num_reqs]
        write_pos = self.model_runner.replayssm_write_pos
        md.replayssm = True
        md.slot_idx = slot_idx
        md.write_pos = write_pos
        md.replayssm_cache_len = self.replayssm_cache_len
        md.replayssm_route = self.replayssm_route
        md.replayssm_max_query_len = self.replayssm_max_query_len

        if is_prefill:
            # A prefill (re)initialises the checkpoint wholesale via
            # `chunk_gated_delta_rule` and writes NO records, so the next
            # forward must fold none.  Parking the cursor at -1 rather than 0
            # is what says that: `num_bonus` is 0 on the first decode after a
            # prefill, so the commit below would otherwise advance 0 -> 1 and
            # fold record 0 -- which this generation never wrote, and which on
            # a reused slot still holds the previous tenant's.  The sentinel is
            # also what makes slot reuse safe without a block-manager hook.
            # Drop any PAD before indexing. `per_req_cache_groups` is filtered
            # to non-negative groups, so it can be shorter than `num_reqs` and
            # leave stale tail entries in the slice -- and a stale tail from a
            # decode step is PAD_SLOT_ID. index_fill_ would read that as "from
            # the end" and park a live, unrelated slot's cursor.
            live = slot_idx[slot_idx >= 0].to(torch.int64)
            write_pos.index_fill_(0, live, -1)
            return

        # Decode: apply the PREVIOUS step's accepted counts.  For non-spec
        # decode `num_accepted_tokens` stays at its initialised value of 1,
        # which is exactly one committed token per step.
        replayssm_commit(
            write_pos,
            slot_idx,
            self.num_accepted_tokens[:num_reqs],
            self.replayssm_max_query_len,
            self.replayssm_cache_len,
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

        # transfer data to ps buffer
        if self.replayssm:
            # Idle graph-padding entries must resolve to PAD so the kernel
            # skips them instead of touching slot 0's checkpoint.
            self.non_spec_state_indices_tensor.gpu[num_decodes:].fill_(PAD_SLOT_ID)
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
        if self.replayssm:
            # Capture-time only wires up the (address-stable) buffers; the
            # cursor is deliberately NOT advanced here.  Warmup and capture
            # replay dummy batches, and letting them commit would leave real
            # sequences resuming from records that were never written.
            gdn_metadata.replayssm = True
            gdn_metadata.slot_idx = self.non_spec_state_indices_tensor.gpu[:bs]
            gdn_metadata.write_pos = self.model_runner.replayssm_write_pos
            gdn_metadata.replayssm_cache_len = self.replayssm_cache_len
            gdn_metadata.replayssm_route = self.replayssm_route
            gdn_metadata.replayssm_max_query_len = self.replayssm_max_query_len
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
                replay_buf_k=(
                    runner.replayssm_buf_k[gdn_idx] if self.replayssm else None
                ),
                replay_buf_u=(
                    runner.replayssm_buf_u[gdn_idx] if self.replayssm else None
                ),
                replay_buf_g=(
                    runner.replayssm_buf_g[gdn_idx] if self.replayssm else None
                ),
                # Slot-addressed recurrent state, not paged KV. It has to be
                # registered (the linear-attention forward reads its state out
                # of `kv_cache_data`), but no block-addressed mover may touch it.
                per_request_state=True,
            )
        return super().build_kv_cache_tensor(layer_id, module)

    def prepare_prefill(  # type: ignore[override]
        self,
        batch: ScheduledBatch,
        running_bs: int,
    ) -> GDNAttentionMetadata:
        attn_metadata, positions = super().prepare_prefill(batch, running_bs)
        if batch.block_tables == []:
            attn_metadata.gdn_metadata = None
            return attn_metadata, positions
        gdn_metadata = self.prepare_gdn_metadata(batch, attn_metadata, is_prefill=True)

        gdn_metadata.ssm_checkpoints = self._checkpoint_targets(batch)
        if gdn_metadata.ssm_checkpoints is not None:
            # Same mapping the chunk kernel builds internally, computed once
            # per step rather than per layer.
            from atom.model_ops.fla_ops.chunk import CHUNK_SIZE
            from atom.model_ops.fla_ops.index import prepare_chunk_offsets

            gdn_metadata.ssm_chunk_offsets = prepare_chunk_offsets(
                gdn_metadata.non_spec_query_start_loc, CHUNK_SIZE
            )

        attn_metadata.gdn_metadata = gdn_metadata
        return attn_metadata, positions

    def prepare_decode(  # type: ignore[override]
        self,
        batch: ScheduledBatch,
        running_bs: int,
        running_tokens: int,
        max_seqlen_q: int,
    ) -> GDNAttentionMetadata:
        attn_metadata, positions = super().prepare_decode(
            batch, running_bs, running_tokens, max_seqlen_q
        )
        self.model_runner.forward_vars["cu_seqlens_q"].cpu[
            running_bs:
        ] = batch.total_tokens_num_decode
        # we fill the attn_metadata cu_seqlens_q here since aiter attn won't calc it for decode
        attn_metadata.cu_seqlens_q = self.model_runner.forward_vars[
            "cu_seqlens_q"
        ].copy_to_gpu(running_bs + 1)

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
