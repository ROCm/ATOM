# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
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
from .sub_pool_spec import SubPoolSpec, page_pool, state_pool

logger = logging.getLogger("atom")


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

        # --- ReplaySSM ------------------------------------------------------
        # The verify window is mtp_k+1 tokens (anchor + drafts); the record
        # buffer has to hold two of them for the early-flush invariant.
        self.replayssm = envs.ATOM_ENABLE_REPLAYSSM
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
                "ReplaySSM enabled for linear attention: cache_len=%d, "
                "route=%s, verify window=%d (1 state slot per request instead "
                "of %d).",
                self.replayssm_cache_len,
                self.replayssm_route,
                self.replayssm_max_query_len,
                self.num_spec + 1,
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
        return (
            getattr(self.model_runner.config.hf_config, "model_type", None)
            == "kimi_linear"
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
        *input* slot, shifting left and appending x — so the new group stops
        depending on the old one the moment the forward returns.

        Reading the state layout alone suggests `conv_kernel_dim - 1` instead,
        on the theory that a shorter forward leaves the new group holding a
        window the old group still owns part of. The kernel closes that gap.

        A fork rather than a copy because the state is two per-family tensors
        rather than one contiguous entry, so there is no single range to
        duplicate — and at one token the fork binds almost nothing anyway.

        Midstep-readable, which is a separate claim and rests on separate
        machinery: the chunk kernel materializes the recurrent state at every
        64-token boundary and `write_state_checkpoints` copies those out, so a
        checkpoint inside a prompt is a copy rather than a shortened forward.
        The engine stops cutting prefill chunks onto the checkpoint ladder for
        this backend — see `BlockManager.checkpoint_cut`. True here only
        because `_checkpoint_targets` and the copy-out kernel exist; a backend
        that declares it without them keeps zero checkpoints and says nothing.

        Costs no accuracy, which is worth stating because the obvious reading
        says it must: `h` is bf16 while the recurrence carries fp32, so a state
        sliced out of it is pre-rounded. The shortened forward it replaces
        rounds the same fp32 value on the way into the pool, though, because
        the two dtypes are the same one — `h` is allocated as `k.new_empty`
        and `_state_dtypes` returns `config.torch_dtype` — so the rounding is
        common to both paths and the difference is nil. Measured on MI355: the
        checkpoint matches a cut prefill's stored state exactly across 56
        (seed, length, boundary) combinations, and resuming from one reproduces
        the remaining tokens' outputs bit for bit. See
        `tests/test_gdn_state_checkpoint_gpu.py`.

        That argument is about the two dtypes agreeing, not about the copy, so
        it does not survive a pool allocated at higher precision than `h`.
        `_state_dtypes` builds exactly one such pool — kimi_linear's fp32 v
        side — and that model is already off this path for a different reason
        (`_KimiMLAGDNCommon.state_transfer`).
        """
        return StateTransfer.fork(1, readable_midstep=True)

    def state_spec(self) -> SubPoolSpec:
        """The GDN state pool: conv_state + temporal_state over all GDN
        layers, `slots_per_req()` of them per in-flight request, plus the
        ReplaySSM (k, u, g) record buffers when that is on — those ride in the
        per-slot entry rather than in a pool of their own, which is what lets
        one `entries_per_req` number size both.

        Concrete builders splice this into their `sub_pool_specs()` alongside
        whatever paged KV pool they own.

        `--state-checkpoint-slots` buys entries beyond the in-flight floor.
        Without it a retained checkpoint can only sit in a slot `max_num_seqs`
        left spare, so the room to keep one is set by concurrency rather than
        by how much reuse the traffic has — the reason a *lower* max_num_seqs
        measures a *worse* hit rate on prefix-reusing traffic.

        The extra entries are counted one per checkpoint, NOT `slots_per_req()`
        each: a checkpoint only ever holds a committed state, and the rollback
        slots beside it are speculation scratch a resumed prefix has no use
        for. At `--num-speculative-tokens 2` that is the difference between a
        checkpoint costing three slots and costing one, and the slots it does
        not take stay available to the KV cache, which is sized out of what is
        left after this. Under ReplaySSM the two counts coincide, because
        rollback is a cursor move and a live request holds one slot as well.
        """
        shape_k, shape_v = self._state_shape_for_runner()
        dt_k, dt_v = self._state_dtypes()
        per_layer = (
            math.prod(shape_k) * dt_k.itemsize + math.prod(shape_v) * dt_v.itemsize
        )
        # The offload tier's spill ring is added on top of this by
        # `state_pool()`, in slots: a checkpoint here is one slot, so a staging
        # entry is one row.
        extra = max(0, getattr(self.model_runner.config, "state_checkpoint_slots", 0))
        return state_pool(
            STATE_SLOT_CLASS,
            self.model_runner.num_gdn_attn_state * per_layer
            + self._replayssm_bytes_per_slot(),
            entries_per_req=self.slots_per_req(),
            extra_entries=extra,
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
        out = {
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
            out["replayssm_buf_k"] = torch.zeros(
                (n, num_slots) + sk, dtype=rec_dtype, device="cuda"
            )
            out["replayssm_buf_u"] = torch.zeros(
                (n, num_slots) + su, dtype=rec_dtype, device="cuda"
            )
            out["replayssm_buf_g"] = torch.zeros(
                (n, num_slots) + sg, dtype=torch.float32, device="cuda"
            )
            # One cursor per slot, shared by every linear-attention layer:
            # the record index depends on the sequence's decode history, not
            # on which layer is running.
            out["replayssm_write_pos"] = torch.zeros(
                num_slots, dtype=torch.int32, device="cuda"
            )
        return out

    def state_entry_views(self, slot: int) -> list[torch.Tensor]:
        """Contiguous views covering the whole of one slot's GDN state.

        The byte-level counterpart of `relocate_state_slots`: that method moves
        a slot between two pool indices, this one names the same bytes so
        something outside the pool -- the LMCache offload tier -- can read or
        write them.

        One view per (cache, layer) rather than one for the slot: both caches
        are layer-major with the slot on axis 1, so a slot's rows are strided
        and no single range covers them. The Triton staging packer builds its
        segment table from `seg.is_contiguous()` and refuses a strided view.
        """
        views = []
        for cache in (
            self.model_runner.mamba_k_cache,
            self.model_runner.mamba_v_cache,
        ):
            for layer in range(cache.shape[0]):
                views.append(cache[layer, slot : slot + 1])
        return views

    def relocate_state_slots(self, pairs: Sequence[tuple[int, int]]) -> None:
        """Relocate one slot's whole GDN state, both families, all layers.

        A slot is one complete recurrent state and moves on its own. A request
        holding several — a committed state plus the rollback slots a rejected
        speculation resumes from — is several such moves, and the caller names
        each one, because nothing about the set is contiguous. That is also why
        this method needs no notion of `slots_per_req()`: under ReplaySSM a
        request holds one slot and under the baseline `1 + num_spec`, and
        either way the caller has already resolved the set into single slots.

        GDN checkpoints by forking, not by copying, so this is not on the
        checkpoint path: it exists because moving the pool's boundary has to be
        able to relocate a slot that is in the way, and relocation is a byte
        move whatever mechanism the class uses to checkpoint. A backend
        declaring `StateTransfer.fork` therefore still owes this method.

        Both caches are layer-major with the slot as the second axis, so one
        slot's rows are strided rather than contiguous and there is no single
        range to copy. `_foreach_copy_` keeps it to one launch for the batch.

        Under ReplaySSM the slot's state is not only the two caches: the
        (k, u, g) records and the write cursor are as much a part of it, since
        the checkpoint alone does not determine the sequence's current state
        without the records written after it. They are laid out on the same
        axes and so move the same way — except the cursor, which is indexed by
        slot directly.
        """
        runner = self.model_runner
        caches = [runner.mamba_k_cache, runner.mamba_v_cache]
        if self.replayssm:
            caches += [
                runner.replayssm_buf_k,
                runner.replayssm_buf_u,
                runner.replayssm_buf_g,
            ]
        destinations, sources = [], []
        for src, dst in pairs:
            for cache in caches:
                destinations.append(cache[:, dst])
                sources.append(cache[:, src])
        if destinations:
            torch._foreach_copy_(destinations, sources)
        if self.replayssm and pairs:
            write_pos = runner.replayssm_write_pos
            src_idx = torch.tensor(
                [src for src, _ in pairs], dtype=torch.int64, device=write_pos.device
            )
            dst_idx = torch.tensor(
                [dst for _, dst in pairs], dtype=torch.int64, device=write_pos.device
            )
            # `write_pos[src_idx]` already materialises a new tensor, so the
            # scatter cannot read a value this same call just wrote. That is
            # only about this one line: whether a batch may chain (a->b, b->c)
            # at all is the caller's to answer, and the cache copies above
            # would need the same answer.
            write_pos.index_copy_(0, dst_idx, write_pos[src_idx])

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
        per-sequence: omitting them is what made an earlier Python-loop
        version silently capture one sequence's state into another's
        checkpoint whenever a batch held two prefills.

        A target is dropped when this step holds too few tokens before it to
        fill the conv window. Both halves of a checkpoint must land together —
        an SSM state at P paired with a conv window from elsewhere is silently
        wrong, and worse than no checkpoint at all, because it is findable.

        Slots, not a separate checkpoint region: a checkpoint here IS an
        ordinary pool slot, indexed exactly as every other slot on this path
        is. (The upstream branch appends checkpoints after the runtime slots
        and offsets them by a `state_cache_base`; that region does not exist
        in this pool.) One slot is the whole checkpoint — a resumed prefix has
        no speculation to roll back, so it needs no scratch beside it.
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
            elif self.replayssm:
                # No per-draft fan-out: the one slot the request holds
                # addresses conv state, checkpoint and record buffers alike.
                # It goes into BOTH tensors because `_attach_replayssm` reads
                # `slot_idx` out of the 1-D one even on the spec path, where
                # the rest of this method leaves it untouched.
                non_spec_state_indices[idx] = committed
                spec_state_indices[idx, 0] = committed
            else:
                spec_state_indices[idx, : len(slots)] = slots

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
            # `chunk_gated_delta_rule`, so any records left over from a prior
            # tenant of this slot are stale.  Zeroing the cursor here is also
            # what makes slot reuse safe without a block-manager hook.
            write_pos.index_fill_(0, slot_idx.to(torch.int64), 0)
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
                # registered (attention_gdn reads its state out of
                # `kv_cache_data`), but no block-addressed mover may touch it.
                per_request_state=True,
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

        # Interior checkpoints: the impl slices them out of the chunk kernel's
        # per-chunk states, so a checkpoint mid-prompt costs no extra forward.
        # Positions at the step's end are sourced from the runtime slot; both
        # kinds are tagged and written by one kernel.
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
            **ctx_pa_ps,
        )

        attn_metadata.gdn_metadata = self._build_gdn_capture_metadata(bs)

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
