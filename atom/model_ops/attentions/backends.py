# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

if TYPE_CHECKING:
    from atom.distributed.pcp_utils import PcpBalGroup
    from atom.kv_transfer.disaggregation.types import KVTransferTensors

import numpy as np
import torch
from torch import nn

from atom.config import DCPConfig
from atom.distributed.dcp_utils import get_dcp_rank, get_dcp_world_size
from atom.distributed.ulysses_sp import attn_head_shard_size
from atom.model_engine.page_unit_checkpoint import (
    CheckpointRestoreOp,
    CheckpointStoreOp,
    checkpoint_copy_ops_per_batch,
)
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.state_runtime import StateTransfer
from atom.model_ops.attention_mla import MLAModules
from atom.model_ops.attentions.pool_layout.paged_state_copy import DescriptorStaging
from atom.model_ops.attentions.pool_layout.pool_rows import PoolRowsMixin
from atom.model_ops.attentions.pool_layout.sub_pool_spec import SubPoolSpec
from atom.model_ops.attentions.sparse_mla_head_groups import PERSISTENT_WORK_BUFFERS
from atom.model_ops.attentions.token_layout.batch_ids import (
    build_batch_ids_device,
    build_cu_seqlens,
    build_query_layout,
)
from atom.model_ops.attentions.token_layout.prefill import (
    PrefillBatch,
    prefill_positions,
)
from atom.model_ops.attentions.token_layout.slots import slot_mapping
from atom.model_ops.dcp_ops import dcp_prefill_slot_mapping
from atom.utils import CpuGpuBuffer
from atom.utils.block_convert import kv_indices_generate_triton
from atom.utils.block_tables import block_table_state
from atom.utils.forward_context import AttentionMetaData, AttnState
from atom.utils.h2d import h2d_producer
from atom.utils.tbo.ubatch_splitting import (
    NUM_TBO_UBATCHES,
    UBatchSlice,
    decode_ubatch_token_order,
)

logger = logging.getLogger("atom")
T = TypeVar("T")

# "Written nowhere" -- see `_write_prefill_slots`. Same value the cache kernels
# and `fla_ops.replayssm` skip on.
PAD_SLOT_ID = -1


class AttentionBackend(ABC):
    """Abstract class for attention backends."""

    #: Whether a *draft* of this flavor caches into a pool of its own. False
    #: means its rows are the target's -- an MLA draft's latent is the target's
    #: latent -- and `make_kv_pool` below is never asked. A property of the
    #: flavor, so a class answer and not a call.
    DRAFT_OWNS_KV_POOL: ClassVar[bool] = False

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return AttentionImpl

    @staticmethod
    def make_kv_pool(
        hf_config,
        *,
        world_size: int,
        target_block_size: int,
        layers: int,
        kv_dtype,
    ):
        """A pool of `layers` rows at this config's geometry.

        Asked of a *draft* model's backend, and only where `DRAFT_OWNS_KV_POOL`
        — the target's layers are its builder's own business. Both counts are
        told rather than read: `layers` only the walk knows, and the block only
        the target builder, whose it is and not this backend's for the reason
        in `DraftKvBuilder.kv_pool`.
        """
        raise NotImplementedError


class AttentionMetadataBuilder(ABC, Generic[T]):
    """Abstract class for attention metadata builders."""

    @abstractmethod
    def __init__(self, block_size: int) -> None:
        """Create the builder, remember some configuration and parameters."""
        raise NotImplementedError

    @abstractmethod
    def prepare_decode(
        self,
        batch: ScheduledBatch,
        running_bs: int,
        running_tokens: int,
        max_seqlen_q: int,
        decode_ubatch_slices: tuple[UBatchSlice, ...] | None = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def prepare_prefill(
        self,
        batch: ScheduledBatch,
        running_bs: int,
        running_tokens: int,
        ubatch_slices: tuple[UBatchSlice, ...] | None = None,
        pcp_groups: "tuple[PcpBalGroup, ...] | None" = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def build(
        self,
        batch: ScheduledBatch,
        running_bs: int,
        running_tokens: int,
        max_seqlen_q: int,
        *,
        is_prefill: bool,
        ubatch_slices: tuple[UBatchSlice, ...] | None = None,
        pcp_groups: "tuple[PcpBalGroup, ...] | None" = None,
    ):
        raise NotImplementedError

    def h2d_group_members(self) -> dict[str, tuple[str, ...]]:
        """The buffers each phase group publishes beside its producers' groups'
        (`ModelRunner._init_h2d_publication` composes them): a fresh dict, a
        builder adding to its base's."""
        return {}

    def query_lengths(
        self, batch: ScheduledBatch, is_prefill: bool, running_tokens: int
    ) -> np.ndarray:
        """The query lengths a step's query layout lays out
        (`prepare_query_metadata`): its scheduled requests'."""
        return np.asarray(batch.num_scheduled_tokens[: batch.total_seqs_num], np.int32)

    def commit_speculative_state(self, metadata, last_token_indices):
        """Commit accepted target rows before draft reads and state checkpoints.

        Backends with tentative recurrent state select the accepted prefix here.
        Paged KV backends whose rejected rows are hidden by sequence length need
        no additional action.
        """
        return

    def ubatch_forward(self, metadata):
        """Keep parent resources alive until all microbatch work is enqueued."""
        return nullcontext()

    def prepare_model_inputs(self, input_ids, metadata):
        """Prepare model inputs after state maintenance and final token staging."""

    def step_needs_eager(self, batch: ScheduledBatch) -> bool:
        """True when this backend cannot run `batch`'s decode from a captured
        graph, whatever its width. Read before the step's metadata is built,
        from the batch alone: every TP rank holds the same batch, so every
        rank answers alike, and `ForwardMode.decide` agrees it across DP."""
        return False

    def close(self):
        """Release backend-owned host workers and mapped resources."""
        self.release_kv_pools()

    def prepare_mtp_decode(
        self,
        bs: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        positions: torch.Tensor,
        only_update: bool = False,
        num_reject_tokens: torch.Tensor | None = None,
    ):
        """Rebuild this backend's metadata for one serial-draft mid-step.

        The draft runs one row per sequence, and `positions` is that row buffer
        -- so `positions.shape[-1]` is this step's `running_bs`, and it is what
        every shape here follows. The `bs` argument is the `scheduled_bs`: the
        count of those rows that carry a real sequence. They are equal whenever
        nothing is padded, and an override must read the buffer rather than the
        argument, because a drafter may run the wider batch the target just ran.

        The LAST axis, not the first: MRoPE positions are `[3, N]` (the token
        axis is last), and every other layout is `[N]`, where the two agree.

        Backends that distinguish the two mark the padded tail so downstream
        kernels skip it; those that do not simply never read `bs`, the same way
        most ignore `only_update` / `num_reject_tokens`.

        Returns per-forward metadata the caller installs on `attn_metadata`;
        `{}` when the backend mutates it in place.
        """
        raise NotImplementedError

    @abstractmethod
    def build_for_cudagraph_capture(
        self,
        bs: int,
        *,
        decode_ubatch_slices: tuple[UBatchSlice, ...] | None = None,
    ) -> AttentionMetaData:
        """``decode_ubatch_slices``: the micro-batches a bucket records as a TBO
        graph, whose metadata the capture hangs on `AttentionMetaData.ubatches`
        (`capture_ubatch_decode`)."""
        raise NotImplementedError

    def cache_write_targets(self) -> list[CpuGpuBuffer]:
        """Index buffers a CUDA graph capture would write cache through.

        Declared here rather than matched on `forward_vars` key names, because a
        backend can aim cache writes with a buffer that is named differently or
        not registered there at all. The default is the paged slot mapping, by
        suffix so TBO's per-ubatch mirrors come along; a backend with other write
        targets overrides this, and must accept `PAD_SLOT_ID` in each of them as
        "skip this row".
        """
        var = self.model_runner.forward_vars
        return [buf for name, buf in var.items() if name.endswith("slot_mapping")]

    def blank_cache_write_targets(self) -> None:
        """Point the capture's cache writes at nothing.

        A capture runs the model for real, so it writes through these; replay
        overwrites them with real rows.
        """
        targets = self.cache_write_targets()
        for buf in targets:
            buf.acquire_write()
        for buf in targets:
            buf.np[:] = PAD_SLOT_ID
            buf.copy_to_gpu()

    # ------------------------------------------------------------------ #
    # Cache sizing — one byte currency for every cache class.             #
    # ------------------------------------------------------------------ #

    def sub_pool_specs(self) -> list[SubPoolSpec]:
        """Every cache class this attention type needs, expressed in bytes.

        One `SubPoolSpec` per class: paged token KV, a window-freed SWA pool,
        a per-request recurrent/compressor state pool. ModelRunner feeds the
        list to `plan_pools` to turn a byte budget into entry counts, so the
        runner never needs to know which architecture it is sizing.

        Specs sharing a `name` are one sub-pool — their `entry_bytes` sum and
        they share an entry index space. That is how a heterogeneous Eagle3
        draft KV pool rides the target model's block ids.

        Default is empty: a builder that owns no cache (e.g. a draft builder
        with no KV of its own) contributes nothing to the budget.
        """
        return []

    def allocate_per_req_cache(self, entries: dict[str, int]) -> dict[str, object]:
        """Allocate this backend's per-request state.

        Called by ModelRunner.allocate_kv_cache() with the entry count sizing
        assigned to every cache class. The builder indexes the classes it
        declared in `sub_pool_specs` — the runner does not know their names.
        Returns a dict mapping attribute name → value; ModelRunner does
        `setattr(self, name, value)` so model layers can reach them as
        `model_runner.<name>` (preserving existing names like `mamba_k_cache`).
        Values are usually tensors, but a backend may also publish the object
        that owns them — DeepSeek-V4 publishes its `EntryMajorArena` alongside the
        per-layer views so the PD path can address a whole entry.
        """
        return {}

    def state_transfer(self) -> StateTransfer:
        """How this backend hands one request's state to another slot.

        A checkpoint is the state as of some boundary, kept where a later
        request can resume from it, so every backend with per-request state has
        to say how one gets there. There are three answers:

        `StateTransfer.fork(n)` — the state rolls and is not one range to
        duplicate, so the old slot goes to the index and the request takes a
        fresh one, reading the old and writing the new for exactly one forward.
        That forward has to leave the new slot self-contained (a single read
        index cannot span both), which takes `n` *committed* tokens.
        `BlockManager` walks a checkpoint/hit point back to the previous block
        boundary until it fits. Run by `StateSlotPool`.

        `StateTransfer.copy(layout_id)` — one request's state is a contiguous
        byte range, so the checkpoint is a duplicate of it and the owner is left
        alone. No forward is bound and no boundary is disqualified for lack of
        room, which is what makes a decode boundary checkpointable at all: a
        decode step commits `1 + accepted_drafts` tokens and acceptance is not
        knowable when the checkpoint has to be decided. Run by
        `PagedStateCheckpointCoordinator`, which stores the image in PAGE units
        rather than in a state slot; `layout_id` is what keeps a stored image
        and the running geometry in agreement.

        `StateTransfer.none()` (default) — no per-request state, or none that can
        be handed over; the checkpoint index stays empty and prefix hits shrink
        to 0 for its models.

        `fork` and `copy` each take a second and independent argument,
        `readable_midstep`: can this backend snapshot a boundary *inside* a
        forward, or only at the forward's last token? False, the default, makes
        `BlockManager` shorten a prefill chunk onto every checkpoint position —
        one forward per rung. True says those positions can be read out of
        intermediates the kernel already materializes, so the chunk runs full
        length and the backend owes `write_state_checkpoints` instead.
        """
        return StateTransfer.none()

    def checkpoint_image_bytes(self) -> int | None:
        """Bytes of an Active Slot a checkpoint image has to hold.

        `None` means all of them: the safe answer, and the one a backend that
        has not worked out which of its bytes a resumer skips should keep
        giving. A backend returns less only when it can name bytes no resumer
        reads — for a ring whose next reader starts exactly at the checkpoint
        boundary, that is the whole ring.
        """
        return None

    def state_entry_views(self, slot: int) -> list["torch.Tensor"]:
        """Contiguous views covering the whole of `slot`'s per-request state.

        The byte-level counterpart of `relocate_state_slots`, so the offload
        tier can read or write the same bytes. Every view must be contiguous --
        the Triton packer refuses a strided one -- so a class whose slot is
        strided (GDN, slot on axis 1) returns one view per layer.
        """
        raise NotImplementedError(
            f"{type(self).__name__} owns per-request state but does not "
            "implement state_entry_views"
        )

    def relocate_state_slots(self, pairs: Sequence[tuple[int, int]]) -> None:
        """Move live state between contiguous Active Slots."""
        raise NotImplementedError(
            f"{type(self).__name__} owns per-request state but does not "
            "implement relocate_state_slots"
        )

    def execute_paged_state_copies(
        self,
        store_ops: Sequence[CheckpointStoreOp],
        restore_ops: Sequence[CheckpointRestoreOp],
        descriptor_slot: int = 0,
    ) -> None:
        """Copy checkpoints between Active Slots and arbitrary PAGEs.

        Stateful backends use ``descriptor_slot`` to isolate concurrent copy
        descriptors. The default implementation only handles the empty no-op
        case, so it intentionally leaves that interface argument unused.
        """
        if store_ops or restore_ops:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement PAGE-backed state copy"
            )

    def reserve_checkpoint_descriptors(self, descriptor_slots) -> None:
        """Allocate staging for out-of-band copy slots before serving.

        Allocating pinned memory synchronizes, so a slot first touched by a
        mid-serving restore would stall its thread. A connector that will pass
        these slots to `execute_paged_state_copies` reserves them up front.
        """
        self._checkpoint_staging().reserve(descriptor_slots)

    def _checkpoint_descriptor_device(self) -> torch.device:
        """Device of the planes `execute_paged_state_copies` copies between."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement PAGE-backed state copy"
        )

    def _checkpoint_staging(self) -> DescriptorStaging:
        """Pinned staging for a step's whole descriptor, sized for the worst step.

        Pinned because the alternative synchronizes: a pageable H2D from
        `build()` makes the host wait out the forward already enqueued, which
        measured 2.9 ms behind 4 ms of work against 0.1 ms staged. Reused
        because allocating pinned memory is itself a synchronizing call.

        A step can carry at most one store and one restore per sequence, so two
        per sequence bounds it (`checkpoint_copy_ops_per_batch`, the same budget
        the coordinator begins queued midstep stores within). The caller checks that bound rather than growing
        on demand: a descriptor that did not fit would otherwise be silently
        truncated into a copy of the wrong shape.

        The store half of that bound is not a property of the batch -- it is
        held by `PagedStateCheckpointCoordinator._supersede`, which keeps one
        pending boundary per sequence. A change that let two of a sequence's
        boundaries drain together would raise from `build()` here, and would be
        storing one of them from the wrong slot besides; the two constraints
        have the same owner and move together.

        Slot 0 serves `build()`; each other descriptor slot is a separate,
        separately fenced buffer (see `DescriptorStaging`).
        """
        staging = getattr(self, "_checkpoint_staging_cache", None)
        if staging is None:
            plan = self._checkpoint_copy_plan()
            # The same budget the scheduler's coordinator begins stores within.
            max_ops = checkpoint_copy_ops_per_batch(
                self.model_runner.config.max_num_seqs
            )
            staging = self._checkpoint_staging_cache = DescriptorStaging(
                max_ops * plan.num_spans, self._checkpoint_descriptor_device()
            )
        return staging

    def warmup_per_req_cache(self) -> None:
        """Pay whatever the first checkpoint copy would pay, before serving.

        Called once by ModelRunner after `allocate_per_req_cache`'s pools are
        installed, which is the earliest a backend can reach its own addresses.
        Nothing else warms this path: `execute_paged_state_copies` runs only
        from `build()`, so a backend that compiles a kernel or fills a cache
        there does it inside a live request's batch. A no-op by default.
        """

    def get_kv_transfer_tensors(self) -> "KVTransferTensors | None":
        """Return RDMA transfer regions for PD disaggregation.

        Each attention backend overrides this to describe its block-indexed
        and slot-indexed tensor regions.  The KV connector uses the result
        to register RDMA memory and compute transfer offsets without knowing
        the backend's internal layout.

        Returns ``None`` when KV transfer is not configured or tensors have
        not been allocated yet.
        """
        return None

    def paged_pool_bytes(self, blocks: int) -> int:
        """Bytes of the runner's one paged allocation this builder's pool takes.

        Zero, the default, means "I allocate my own" -- DeepSeek-V4 does,
        because half of what its PAGE entry is priced for lives in the plane
        pool `allocate_per_req_cache` makes later.

        An answer must come from the declaration `sub_pool_specs` prices, not
        from a second spelling of it: the runner hands back exactly this many
        bytes and refuses a builder that was charged for another number.
        """
        return 0

    def allocate_kv_cache_tensors(self, *, blocks: int, buf) -> dict[str, Any]:
        """Allocate the model's primary paged KV cache tensors.

        `blocks` is the scheduler block count every rank was told to build at.
        A parameter and not a runner attribute because sizing runs per
        subprocess and the answers differ: read the local estimate and one pool
        is built at one count while something sized off another is built at a
        second -- a class of bug no unit test reaches. Overrides record it as
        `self.num_blocks`, converted to their own page.

        `buf` is this builder's region of the runner's paged allocation,
        `paged_pool_bytes` long -- empty for a builder that answered zero and
        allocates its own.

        The decode side of a P/D pair calls this too, with `buf` a region of an
        imported pool. Nothing here may write into `buf`: that one arrives
        already holding the peer's KV.

        Builders own the per-attention-type tensor layout (single 576-dim MLA
        tensor vs split-K/V MHA tensor; full-rank vs hybrid-only-full-attn-rows
        for Qwen3-Next; per-module deferred for MiMo-V2). The runner only
        setattr's the returned dict onto itself, so model layers can access
        tensors as `model_runner.<name>` (preserving existing names: kv_cache,
        kv_scale, index_cache, etc.).

        Values may be Tensors, None (deferred allocation), or scalar metadata
        (e.g. aligned_index_dim) needed downstream by build_kv_cache_tensor.
        Returns empty dict for builders that do not own the main KV pool.
        """
        return {}

    def release_kv_pools(self) -> None:
        """Drop the backing of every pool this builder holds, keeping the
        declarations. A rollout sleep frees the runner's paged buffer, and a
        pool's views would hold it alive; `allocate_kv_cache_tensors` puts the
        backing back. A builder that holds no pool has nothing to drop.
        """

    def build_kv_cache_tensor(self, module):
        """Build the vLLM-style `KVCacheTensor` registration entry for one
        attention module, OR return None if this builder does not recognize
        the module type.

        Called from ModelRunner.allocate_kv_cache()'s binding loop for every
        module of the model. The builder owns:
          - module-type detection (e.g. `hasattr(module, "use_mla")`)
          - which of its pools the module's layer belongs to, and its row in
            that pool from `pool_rows`
          - per-module tensor slicing out of the pool that holds the row
          - any `setattr(module, "k_cache", ...)` side effects per the
            existing module convention
          - returning a `KVCacheTensor` ModelRunner appends to its registry

        Builders override this for the module types they handle; subclasses
        chain via `super().build_kv_cache_tensor(...)` to inherit shared
        paths (e.g. `GDNAttentionMetadataBuilder` handles
        `base_linear_attention` and delegates `base_attention` MHA modules
        to its `AiterAttentionMetadataBuilder` parent).

        Default: unknown module types get no tensor.
        """
        return


class CommonAttentionBuilder(PoolRowsMixin, AttentionMetadataBuilder[T], Generic[T]):
    def __init__(self, model_runner):
        self.model_runner = model_runner
        assert model_runner.block_size % self.block_size == 0
        self.block_ratio = model_runner.block_size // self.block_size
        # Blocks the pools were last built at, in this backend's own page --
        # what its kernels index; 0 until something backs them. Held, not
        # re-read off the runner: a rollout resume can re-allocate at a
        # *smaller* count, and a view addresses what it was built at.
        self.num_blocks = 0
        self.device = model_runner.device
        config = model_runner.config
        hf_config = config.hf_config
        self.dcp_world_size = get_dcp_world_size()
        self.dcp_rank = get_dcp_rank()
        # DCP KV-cache interleave granularity S (1 = token-level round-robin).
        self.cp_kv_cache_interleave_size = getattr(
            config, "dcp_config", DCPConfig()
        ).interleave_size
        self.max_num_batched_tokens = model_runner.max_num_batched_tokens
        self.max_bs = model_runner.max_bs
        # Host scratch for the token axis, resident for the same reason every
        # mirror is: a step runs one slot mapping between a lot of unrelated
        # allocation, so a fresh temporary of this size costs what a cold
        # allocation costs, not what a warm free-list hit costs.
        self.token_axis_scratch = np.empty(self.max_num_batched_tokens, dtype=np.int64)
        # Every row's own index, resident so no step rebuilds it. One buffer for
        # three readers that each want the same numbers: a cu_seqlens ramp at
        # one token per sequence (hence `+ 1`), DSpark's token -> request map,
        # and the MTP draft's, which needs a source already on the device (see
        # `prepare_mtp_decode`).
        self.row_ids = torch.arange(
            self.max_bs + 1, device=self.device, dtype=torch.int32
        )
        self.max_num_blocks_per_seq = (
            config.max_model_len + self.block_size - 1
        ) // self.block_size
        # Width of every `block_tables` buffer, and of anything gathered from one.
        self.block_table_cols = self.max_num_blocks_per_seq // self.block_ratio
        # Per-rank attention head count. eagle.propose's mid-step path reads
        # this to gate the `do_attn_metadata_update` branch. Subclasses that
        # need a kernel-minimum-padded count set `self.padded_num_attention_heads`
        # separately (it does NOT replace this attribute).
        self.num_attention_heads = (
            hf_config.num_attention_heads // attn_head_shard_size()
        )

        i64_kwargs = {"dtype": torch.int64, "device": self.device}
        i32_kwargs = {"dtype": torch.int32, "device": self.device}

        # A request's attention metadata, written by either phase.
        seq_i64 = dict(i64_kwargs, publication_group="seq_metadata")
        seq_i32 = dict(i32_kwargs, publication_group="seq_metadata")

        attn_metadata = {
            "slot_mapping": CpuGpuBuffer(self.max_num_batched_tokens, **seq_i64),
            "context_lens": CpuGpuBuffer(self.max_bs, **seq_i32),
            "block_tables": CpuGpuBuffer(self.max_bs, self.block_table_cols, **seq_i32),
            "cu_seqlens_q": CpuGpuBuffer(
                self.max_bs + 1, **i32_kwargs, publication_group="early"
            ),
            "cu_seqlens_k": CpuGpuBuffer(self.max_bs + 1, **seq_i32),
            # Uploaded only on a prefix-cache hit, so consumers read
            # `AttentionMetaData.num_cached_tokens is None` as "no row has any".
            "num_cached_tokens": CpuGpuBuffer(self.max_bs, **seq_i32),
            # seq_starts for cp_mha_gather_cache: always zeros (prefix at position 0)
            "seq_starts": CpuGpuBuffer(self.max_bs, **seq_i32),
            # token -> seq over this fwd's QUERY tokens; `-1` on the CUDAGraph
            # pad tail. Every backend needs it (see token_layout/batch_ids.py);
            # `prepare_query_metadata` builds it with the query prefix.
            "batch_id_per_q_token": CpuGpuBuffer(
                self.max_num_batched_tokens, **i32_kwargs, publication_group="early"
            ),
        }
        if config.enable_tbo_decode:
            # A split decode's packed tokens in its micro-batches' order, and
            # each micro-batch's scheduled tokens (`decode_ubatch_token_order`).
            attn_metadata["ubatch_token_order"] = CpuGpuBuffer(
                self.max_num_batched_tokens, **i64_kwargs, publication_group="early"
            )
            attn_metadata["ubatch_scheduled_tokens"] = CpuGpuBuffer(
                NUM_TBO_UBATCHES, **i32_kwargs, publication_group="early"
            )

        attn_metadata["cu_seqlens_q"].cpu.copy_(
            torch.arange(0, self.max_bs + 1, step=1, dtype=torch.int32)
        )
        attn_metadata["cu_seqlens_q"].copy_to_gpu()
        attn_metadata["seq_starts"].cpu.zero_()
        attn_metadata["seq_starts"].copy_to_gpu()
        self.model_runner.forward_vars.update(attn_metadata)

    def h2d_group_members(self) -> dict[str, tuple[str, ...]]:
        # A micro-batch's buffers, tagged `ub{i}_metadata`, go up as one packed
        # upload with whichever phase it runs.
        groups = super().h2d_group_members()
        var = self.model_runner.forward_vars
        for ub_idx in range(NUM_TBO_UBATCHES):
            name = f"ub{ub_idx}_metadata"
            members = tuple(
                key
                for key, buffer in var.items()
                if isinstance(buffer, CpuGpuBuffer) and buffer.publication_group == name
            )
            if members:
                groups[name] = members
        return groups

    def _build_token_block_tables(
        self, attn_metadata, running_tokens: int, max_seqlen_q: int
    ) -> None:
        """Nothing to build: this backend has no sparse indexer.

        `EagleProposer` builds on whatever builder the target uses, so every
        backend a draft can run against has to answer. Only the MLA one
        overrides.
        """

    def prepare_block_tables(self, batch: ScheduledBatch, running_bs=None):
        """Prepare the shared CPU snapshot, reusing unchanged page mappings."""
        return block_table_state(
            self.model_runner.forward_vars["block_tables"]
        ).prepare(batch.block_tables, pad_to=running_bs)

    def prepare_decode(
        self,
        batch: ScheduledBatch,
        running_bs: int,
        running_tokens: int,
        max_seqlen_q: int,
        decode_ubatch_slices: tuple[UBatchSlice, ...] | None = None,
    ):
        """`stage_decode` (and a split decode's `stage_ubatch_decode`), the
        phase's uploads, then `finish_decode` (and `finish_ubatch_decode`)."""
        group = self.model_runner.h2d_groups["decode_metadata"]
        attn_metadata, positions = self.stage_decode(
            batch, running_bs, running_tokens, max_seqlen_q, group
        )
        ubatch_groups = (
            self.stage_ubatch_decode(
                attn_metadata,
                *self.decode_spans(batch),
                decode_ubatch_slices,
                max_seqlen_q,
            )
            if decode_ubatch_slices
            else ()
        )
        for staged in (group, *ubatch_groups):
            staged.publish()
        self.finish_decode(
            batch, attn_metadata, running_bs, running_tokens, max_seqlen_q
        )
        if decode_ubatch_slices:
            self.finish_ubatch_decode(
                attn_metadata,
                batch.total_seqs_num_decode,
                decode_ubatch_slices,
                max_seqlen_q,
            )
        return attn_metadata, positions

    def stage_decode(
        self,
        batch: ScheduledBatch,
        running_bs: int,
        running_tokens: int,
        max_seqlen_q: int,
        group,
    ):
        """Write the step's decode metadata on the host and count it into
        ``group``; return the metadata over GPU views the publish fills. A
        builder adds its own sources after `super()`."""
        raise NotImplementedError

    def finish_decode(
        self,
        batch: ScheduledBatch,
        attn_metadata: AttentionMetaData,
        running_bs: int,
        running_tokens: int,
        max_seqlen_q: int,
    ) -> None:
        """The GPU work reading the step's uploaded decode metadata. A builder
        adds its own after `super()`."""

    def _ubatch_share(self, rows: int) -> int:
        """A TBO micro-batch's most of the step's ``rows`` tokens: half, cut at
        the token midpoint -- all of them under PCP, whose micro-batches are
        whole request groups."""
        if self.model_runner.config.prefill_context_parallel_size > 1:
            return rows
        return -(-rows // NUM_TBO_UBATCHES)

    def _ubatch_tokens(self) -> int:
        """The most tokens a TBO micro-batch runs."""
        return self._ubatch_share(self.max_num_batched_tokens)

    def _ubatch_buffer_shapes(self) -> dict[str, tuple[int, ...]]:
        """The step's buffers each TBO micro-batch stages its own copy of
        (`_allocate_ubatch_buffers`), by shape: a prefill micro-batch runs up
        to every request of the step, either phase's up to `_ubatch_tokens`.
        A builder adds its own to `super()`'s."""
        var = self.model_runner.forward_vars
        bs, tokens = self.max_bs, self._ubatch_tokens()
        shapes = {
            "cu_seqlens_q": (bs + 1,),
            "batch_id_per_q_token": (tokens,),
            "slot_mapping": (tokens,),
            "context_lens": (bs,),
            "block_tables": (bs, self.block_table_cols),
            "cu_seqlens_k": (bs + 1,),
            "num_cached_tokens": (bs,),
            "seq_starts": (bs,),
        }
        # What `_ubatch_decode_metadata` and `finish_ubatch` read, where the
        # builder has it.
        if "kv_indptr" in var:
            shapes["kv_indptr"] = (bs + 1,)
        if "kv_indices" in var:
            shapes["kv_indices"] = tuple(var["kv_indices"].cpu.shape)
        for name in PERSISTENT_WORK_BUFFERS:
            if name in var:
                shapes[name] = tuple(var[name].shape)
        return shapes

    def _allocate_ubatch_buffers(self, max_seqlen_q: int) -> None:
        """Each TBO micro-batch's ``ub{i}_`` copies of `_ubatch_buffer_shapes`,
        of the step's dtypes; a published one in the micro-batch's
        ``ub{i}_metadata`` group, which both phases' groups publish
        (`h2d_group_members`). A builder calls it once its own step buffers
        exist."""
        var = self.model_runner.forward_vars
        for ub_idx in range(NUM_TBO_UBATCHES):
            prefix = f"ub{ub_idx}_"
            for name, shape in self._ubatch_buffer_shapes().items():
                src = var[name]
                if isinstance(src, torch.Tensor):
                    var[prefix + name] = torch.empty(
                        shape, dtype=src.dtype, device=self.device
                    )
                    continue
                var[prefix + name] = CpuGpuBuffer(
                    *shape,
                    dtype=src.cpu.dtype,
                    device=self.device,
                    publication_group=(
                        f"{prefix}metadata" if src.publication_group else None
                    ),
                )
            # A recorded decode reads it before any step stages it.
            cu_seqlens_q = var[f"{prefix}cu_seqlens_q"]
            cu_seqlens_q.cpu.copy_(
                torch.arange(cu_seqlens_q.cpu.shape[0], dtype=torch.int32)
                * max_seqlen_q
            )
            cu_seqlens_q.copy_to_gpu()

    def _ubatch_decode_rows(self) -> tuple[str, ...]:
        """A TBO decode micro-batch's per-request host rows, sliced off the
        step's. A builder adds its own to `super()`'s."""
        return ("context_lens",)

    def _ubatch_decode_indptrs(self) -> tuple[str, ...]:
        """The step's prefix sums a micro-batch rebases to its first request."""
        return ("kv_indptr",)

    def _ubatch_decode_token_rows(self) -> tuple[str, ...]:
        """A TBO decode micro-batch's per-token host rows past `slot_mapping`,
        its own tokens' sliced off the step's, zero past them. A builder adds
        its own to `super()`'s."""
        return ()

    def stage_ubatch_decode(
        self,
        attn_metadata: AttentionMetaData,
        scheduled_bs: int,
        lens: np.ndarray,
        cu: np.ndarray,
        decode_ubatch_slices: tuple[UBatchSlice, ...],
        max_seqlen_q: int,
    ) -> list:
        """Split the step's staged decode host mirrors -- ``scheduled_bs``
        requests of query ``lens``, prefix ``cu`` -- into one ``ub{i}_`` set per
        micro-batch of ``decode_ubatch_slices``, counted into its group, and
        hang each micro-batch's metadata on ``attn_metadata.ubatches``; return
        the groups. A builder writes its own per micro-batch in
        `_stage_decode_ubatch`."""
        names = tuple(f"ub{i}_metadata" for i in range(len(decode_ubatch_slices)))
        self._check_metadata_writable(*names)
        var = self.model_runner.forward_vars
        rows = self._ubatch_decode_rows()
        groups, ubatches = [], []
        for ub_idx, (ub, name) in enumerate(zip(decode_ubatch_slices, names)):
            prefix = f"ub{ub_idx}_"
            start = ub.request_slice.start
            ub_scheduled_bs = ub.scheduled_bs(scheduled_bs)
            group = self.model_runner.h2d_groups[name]
            self._slice_ubatch_rows(prefix, rows, start, ub_scheduled_bs, ub.running_bs)
            group.set_counts({prefix + row: ub.running_bs for row in rows})
            block_table_state(var[f"{prefix}block_tables"]).stage(ub.running_bs, group)
            ubatches.append(
                self._stage_decode_ubatch(
                    ub_idx,
                    ub,
                    lens[start : start + ub_scheduled_bs],
                    cu,
                    max_seqlen_q,
                    group,
                )
            )
            groups.append(group)
        attn_metadata.ubatches = ubatches
        self._attach_decode_ubatch_order(attn_metadata, decode_ubatch_slices)
        return groups

    def _attach_decode_ubatch_order(
        self,
        attn_metadata: AttentionMetaData,
        decode_ubatch_slices: tuple[UBatchSlice, ...],
    ) -> None:
        """Hand a split decode's metadata the token order its micro-batches
        run in, and each micro-batch its own scheduled tokens, both staged
        with the query prefix (`prepare_query_metadata`)."""
        var = self.model_runner.forward_vars
        attn_metadata.ubatch_token_order = var["ubatch_token_order"].gpu[
            : decode_ubatch_slices[-1].token_slice.stop
        ]
        scheduled = var["ubatch_scheduled_tokens"].gpu
        for ub_idx, ub_attn in enumerate(attn_metadata.ubatches):
            ub_attn.scheduled_tokens_gpu = scheduled[ub_idx : ub_idx + 1]

    def _stage_decode_ubatch(
        self,
        ubatch_idx: int,
        ub: UBatchSlice,
        lens: np.ndarray,
        cu: np.ndarray,
        max_seqlen_q: int,
        group,
    ) -> AttentionMetaData:
        """Micro-batch ``ub``'s own host rows past its per-request ones -- its
        requests' query ``lens``, the step's prefix ``cu`` -- counted into
        ``group``; return its metadata (`_ubatch_decode_metadata`)."""
        var = self.model_runner.forward_vars
        prefix = f"ub{ubatch_idx}_"
        start = ub.request_slice.start
        ub_scheduled_bs = len(lens)
        scheduled = ub.scheduled_token_slice
        ub_scheduled_tokens = ub.scheduled_tokens
        counts = {}
        for name in self._ubatch_decode_indptrs():
            full, indptr = var[name].np, var[prefix + name].np
            indptr[: ub_scheduled_bs + 1] = (
                full[start : start + ub_scheduled_bs + 1] - full[start]
            )
            indptr[ub_scheduled_bs + 1 : ub.running_bs + 1] = indptr[ub_scheduled_bs]
            counts[prefix + name] = ub.running_bs + 1
        slots = var[f"{prefix}slot_mapping"].np
        slots[:ub_scheduled_tokens] = var["slot_mapping"].np[scheduled]
        slots[ub_scheduled_tokens : ub.running_tokens] = -1
        for name in self._ubatch_decode_token_rows():
            rows = var[prefix + name].np
            rows[:ub_scheduled_tokens] = var[name].np[scheduled]
            rows[ub_scheduled_tokens : ub.running_tokens] = 0
            counts[prefix + name] = ub.running_tokens
        build_cu_seqlens(lens, ub.running_bs, out=var[f"{prefix}cu_seqlens_q"].np)
        counts[f"{prefix}slot_mapping"] = ub.running_tokens
        counts[f"{prefix}cu_seqlens_q"] = ub.running_bs + 1
        group.set_counts(counts)
        return self._ubatch_decode_metadata(ubatch_idx, ub, max_seqlen_q)

    def _ubatch_decode_metadata(
        self, ubatch_idx: int, ub: UBatchSlice, max_seqlen_q: int
    ) -> AttentionMetaData:
        """Micro-batch ``ub``'s decode metadata over its ``ub{i}_`` GPU views,
        filled by its upload and `finish_ubatch`."""
        var = self.model_runner.forward_vars
        prefix = f"ub{ubatch_idx}_"
        running_bs = ub.running_bs
        attn = AttentionMetaData(
            slot_mapping=var[f"{prefix}slot_mapping"].gpu[: ub.running_tokens],
            context_lens=var[f"{prefix}context_lens"].gpu[:running_bs],
            block_tables=var[f"{prefix}block_tables"].gpu[:running_bs],
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=self.model_runner.config.max_model_len,
            cu_seqlens_q=var[f"{prefix}cu_seqlens_q"].gpu[: running_bs + 1],
            kv_indptr=var[f"{prefix}kv_indptr"].gpu[: running_bs + 1],
            kv_indices=var[f"{prefix}kv_indices"].gpu,
        )
        for name in PERSISTENT_WORK_BUFFERS:
            if prefix + name in var:
                setattr(attn, name, var[prefix + name])
        return attn

    def _slice_ubatch_rows(
        self, prefix: str, names, start: int, scheduled_bs: int, running_bs: int
    ) -> None:
        """Copy the step's per-request host rows ``names`` and block table,
        ``scheduled_bs`` from ``start``, into a micro-batch's ``prefix`` set,
        zero out to its ``running_bs``."""
        var = self.model_runner.forward_vars
        for name in names:
            rows = var[prefix + name].np
            rows[:scheduled_bs] = var[name].np[start : start + scheduled_bs]
            rows[scheduled_bs:running_bs] = 0
        block_table_state(var["block_tables"]).slice_to(
            var[f"{prefix}block_tables"], start, scheduled_bs, pad_to=running_bs
        )

    def finish_ubatch_decode(
        self,
        attn_metadata: AttentionMetaData,
        scheduled_bs: int,
        decode_ubatch_slices: tuple[UBatchSlice, ...],
        max_seqlen_q: int,
    ) -> None:
        """Each micro-batch's GPU work over its uploaded ``ub{i}_`` metadata
        (`finish_ubatch`), of a step of ``scheduled_bs`` requests."""
        for ub_idx, (ub, ub_attn) in enumerate(
            zip(decode_ubatch_slices, attn_metadata.ubatches)
        ):
            self.finish_ubatch(ub_idx, ub, ub_attn, max_seqlen_q)

    def finish_ubatch(
        self,
        ubatch_idx: int,
        ub: UBatchSlice,
        ub_attn: AttentionMetaData,
        max_seqlen_q: int,
    ) -> None:
        """One decode micro-batch's GPU work into ``ub_attn``. A builder adds
        its own after `super()`."""
        var = self.model_runner.forward_vars
        prefix = f"ub{ubatch_idx}_"
        running_bs = ub.running_bs
        kv_indices_generate_triton(
            var[f"{prefix}block_tables"].gpu[:running_bs],
            var[f"{prefix}kv_indices"].gpu,
            var[f"{prefix}kv_indptr"].gpu[: running_bs + 1],
            self.block_ratio,
            int(var[f"{prefix}context_lens"].np[:running_bs].max(initial=0)),
        )

    def capture_ubatch_decode(
        self,
        attn_metadata: AttentionMetaData,
        scheduled_bs: int,
        decode_ubatch_slices: tuple[UBatchSlice, ...],
        max_seqlen_q: int,
    ) -> None:
        """A captured TBO bucket's (of ``scheduled_bs`` requests) micro-batch
        metadata, staged as a replay's is (`stage_ubatch_decode`) off the
        step's capture layout (`publish_capture_layout`), published, then
        finished: the ``ub{i}_`` buffers the recording reads hold that layout,
        not what the step before the capture left in them."""
        lens = np.full(scheduled_bs, max_seqlen_q, dtype=np.int32)
        cu = np.arange(scheduled_bs + 1, dtype=np.int32) * max_seqlen_q
        for group in self.stage_ubatch_decode(
            attn_metadata, scheduled_bs, lens, cu, decode_ubatch_slices, max_seqlen_q
        ):
            group.publish()
        self.finish_ubatch_decode(
            attn_metadata, scheduled_bs, decode_ubatch_slices, max_seqlen_q
        )

    def publish_capture_layout(self, bs: int, max_seqlen_q: int) -> None:
        """Publish the decode a capture records over into the step's decode
        rows: ``bs`` requests of ``max_seqlen_q`` tokens, each reading one key
        in block 0's first slot and writing none. The capture's eager warmup
        reads defined values -- not what the step before it left, which a
        prefill's are -- and a TBO bucket's micro-batches slice them as a
        replay's (`capture_ubatch_decode`). A builder adds its own rows to
        `super()`'s counts (`_capture_layout_counts`)."""
        self._check_metadata_writable("decode_metadata")
        group = self.model_runner.h2d_groups["decode_metadata"]
        group.set_counts(self._capture_layout_counts(bs, max_seqlen_q))
        var = self.model_runner.forward_vars
        block_table_state(var["block_tables"]).prepare(
            np.zeros((bs, self.block_table_cols), dtype=np.int32)
        )
        block_table_state(var["block_tables"]).stage(bs, group)
        group.publish()
        var["kv_indices"].gpu[:bs].zero_()

    def _capture_layout_counts(self, bs: int, max_seqlen_q: int) -> dict[str, int]:
        """`publish_capture_layout`'s host rows, written; their counts."""
        var = self.model_runner.forward_vars
        tokens = bs * max_seqlen_q
        var["context_lens"].np[:bs] = 1
        var["kv_indptr"].np[: bs + 1] = np.arange(bs + 1, dtype=np.int32)
        var["slot_mapping"].np[:tokens] = PAD_SLOT_ID
        return {"context_lens": bs, "kv_indptr": bs + 1, "slot_mapping": tokens}

    def _check_metadata_writable(self, *group_names: str) -> None:
        """Preflight persistent host sources before any producer mutates them."""
        groups = getattr(self.model_runner, "h2d_groups", None)
        if groups is not None:
            for name in group_names:
                group = groups.get(name)
                if group is not None:
                    group.check_writable()

    @h2d_producer("mrope", runner="model_runner")
    def _mrope_cpu_view(self, num_tokens: int) -> np.ndarray:
        return (
            self.model_runner.forward_vars["mrope_positions"]
            .np.reshape(-1)[: 3 * num_tokens]
            .reshape(3, num_tokens)
        )

    def _stage_mrope(self, num_tokens: int, group) -> torch.Tensor:
        """Count the MRoPE planes into the phase's ``group``; return their view."""
        buf = self.model_runner.forward_vars["mrope_positions"]
        group.set_count(buf, 3 * num_tokens)
        return self.model_runner._mrope_positions_view(num_tokens)

    def _build_mrope_prefill_positions(
        self, batch: ScheduledBatch, running_tokens: int, group
    ) -> torch.Tensor | None:
        if not getattr(self.model_runner, "use_mrope", False):
            return None

        scheduled_tokens = batch.total_tokens_num_prefill
        # The running tokens are the axis stride, as for decode.
        positions = self._mrope_cpu_view(running_tokens)
        positions[:, scheduled_tokens:] = 0
        offset = 0
        for req_id, seqlen, cached_seqlen in zip(
            batch.req_ids, batch.context_lens, batch.num_cached_tokens
        ):
            num_tokens = int(seqlen) - int(cached_seqlen)
            mrope_positions = batch.mrope_positions_by_req.get(req_id)
            if mrope_positions is None:
                positions[:, offset : offset + num_tokens] = np.arange(
                    cached_seqlen, seqlen, dtype=np.int64
                )[None, :]
            else:
                positions[:, offset : offset + num_tokens] = mrope_positions[
                    :, cached_seqlen:seqlen
                ]
            offset += num_tokens

        return self._stage_mrope(running_tokens, group)

    def _build_mrope_decode_positions(
        self,
        batch: ScheduledBatch,
        context_lens: np.ndarray,
        group,
        *,
        running_tokens: int | None = None,
    ) -> torch.Tensor | None:
        if not getattr(self.model_runner, "use_mrope", False):
            return None

        scheduled_tokens = batch.total_tokens_num_decode
        # Every decode model view uses the padded token count as its axis
        # stride, in both eager execution and graph replay.
        width = scheduled_tokens if running_tokens is None else running_tokens
        positions = self._mrope_cpu_view(width)
        positions[:, scheduled_tokens:] = 0
        lengths = batch.num_scheduled_tokens[: len(context_lens)]
        offset = 0
        for req_id, context_len, length in zip(batch.req_ids, context_lens, lengths):
            start = int(context_len) - int(length)
            stop = int(context_len)
            delta = batch.mrope_position_deltas.get(req_id)
            if delta is None:
                base = np.arange(start, stop, dtype=np.int64)
            else:
                base = np.arange(start + int(delta), stop + int(delta), dtype=np.int64)
            positions[:, offset : offset + int(length)] = base[None, :]
            offset += int(length)

        return self._stage_mrope(width, group)

    @h2d_producer("early", runner="model_runner")
    def prepare_query_metadata(
        self,
        lengths: np.ndarray,
        running_bs: int,
        running_tokens: int,
        decode_ubatch_slices: tuple[UBatchSlice, ...] | None = None,
    ) -> dict[str, int]:
        """Stage the query-side metadata of ``lengths`` (each scheduled
        request's query tokens): what depends on the query layout alone, and
        is the same for every backend. KV-side metadata stays in each builder's
        `prepare_prefill` / `prepare_decode`.

        `cu_seqlens_q`, the token -> request map (`batch_id_per_q_token`, out
        to `running_tokens`, `-1` past the scheduled tokens), and a split
        decode's token order (`decode_ubatch_token_order`) with each
        micro-batch's scheduled tokens. Returns each staged buffer's count:
        `prepare_model` publishes them with the token inputs, everything else
        through `publish_query_metadata`.
        """
        var = self.model_runner.forward_vars
        build_query_layout(
            lengths,
            running_bs,
            running_tokens,
            cu_seqlens_q=var["cu_seqlens_q"].np,
            batch_ids=var["batch_id_per_q_token"].np,
        )
        counts = {
            "cu_seqlens_q": running_bs + 1,
            "batch_id_per_q_token": running_tokens,
        }
        if decode_ubatch_slices:
            counts["ubatch_token_order"] = decode_ubatch_token_order(
                decode_ubatch_slices,
                int(lengths.sum()),
                var["ubatch_token_order"].np,
            )
            var["ubatch_scheduled_tokens"].np[: len(decode_ubatch_slices)] = [
                ub.scheduled_tokens for ub in decode_ubatch_slices
            ]
            counts["ubatch_scheduled_tokens"] = len(decode_ubatch_slices)
        return counts

    def publish_query_metadata(
        self,
        lengths: np.ndarray,
        running_bs: int,
        running_tokens: int,
        decode_ubatch_slices: tuple[UBatchSlice, ...] | None = None,
    ) -> None:
        """`prepare_query_metadata`, published on its own: for a step no token
        inputs carry (a capture, the vLLM plugin)."""
        counts = self.prepare_query_metadata(
            lengths, running_bs, running_tokens, decode_ubatch_slices
        )
        group = self.model_runner.h2d_groups["early"]
        group.set_counts(counts)
        group.publish()

    def decode_spans(self, batch: ScheduledBatch) -> tuple[int, np.ndarray, np.ndarray]:
        """A pure-decode step's `(bs, per-request lengths, exclusive prefix sum)`.

        One place, so no caller can pair a length vector with someone else's
        cumsum: both come off `num_scheduled_tokens` and the buffer
        `prepare_query_metadata` staged from it -- which is also why this is
        only valid after that call, not from the shrink helpers that run before.
        """
        bs = batch.total_seqs_num_decode
        return (
            bs,
            batch.num_scheduled_tokens[:bs],
            self.model_runner.forward_vars["cu_seqlens_q"].np[: bs + 1],
        )

    def _stage_prefill_seq_lens(
        self, context_lens: np.ndarray, running_bs: int, prefix: str = ""
    ) -> int:
        """Write the two per-sequence length mirrors of the ``prefix`` set;
        return the requests' total KV.

        Straight into the mirrors: this step is their only writer, so a
        temporary would just be copied over.

        Both are padded out to `running_bs`, the width a draft pass that follows
        this step runs at and the one `prepare_decode` fills to. A flat cumsum
        gives each fabricated row zero query length and a zero context makes it
        read nothing; left unwritten they hold the previous step's, and the
        drafter attends to a since-freed request's blocks.
        """
        var = self.model_runner.forward_vars
        scheduled_bs = len(context_lens)
        cu_k = var[f"{prefix}cu_seqlens_k"].np
        build_cu_seqlens(context_lens, running_bs, out=cu_k)
        lens = var[f"{prefix}context_lens"].np
        lens[:scheduled_bs] = context_lens
        lens[scheduled_bs:running_bs] = 0
        return int(cu_k[scheduled_bs])

    def _write_prefill_slots(
        self,
        batch: PrefillBatch,
        positions: np.ndarray,
        prefix: str = "",
    ) -> None:
        """Place every scheduled token's KV slot into the ``prefix`` set's
        `slot_mapping` mirror, off its packed `block_tables` mirror.

        `-1` means "written nowhere": either the batch carries no block tables
        at all, or under DCP this token belongs to another rank.
        """
        var = self.model_runner.forward_vars
        slots = var[f"{prefix}slot_mapping"].np[: positions.shape[0]]
        scheduled_bs = batch.scheduled_bs
        if not batch.block_tables:
            slots[:] = -1
            return
        assert len(batch.block_tables) == scheduled_bs, (
            f"block_tables has {len(batch.block_tables)} rows for a prefill "
            f"of {scheduled_bs} requests"
        )
        block_size = self.model_runner.block_size
        if self.dcp_world_size > 1:
            dcp_slots = dcp_prefill_slot_mapping(
                batch.block_tables,
                batch.cached_lens.tolist(),
                batch.context_lens.tolist(),
                block_size,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_kv_cache_interleave_size,
            )
            assert len(dcp_slots) == slots.shape[0], (
                f"dcp slot mapping is {len(dcp_slots)} long, not the "
                f"{slots.shape[0]} tokens this prefill forwards"
            )
            slots[:] = dcp_slots
            return
        slot_mapping(
            positions,
            batch.seqlens_q,
            var[f"{prefix}block_tables"].np,
            block_size,
            out=slots,
            scratch=self.token_axis_scratch,
        )

    def _pad_tokens(self, scheduled_tokens: int, running_tokens: int) -> None:
        """The padding tokens `[scheduled_tokens, running_tokens)`: position 0
        and no KV slot. The builder that writes the scheduled tokens' entries
        calls it and publishes both buffers out to `running_tokens`."""
        var = self.model_runner.forward_vars
        var["positions"].np[scheduled_tokens:running_tokens] = 0
        var["slot_mapping"].np[scheduled_tokens:running_tokens] = -1

    def _prefill_block_table_rows(
        self, batch: PrefillBatch, running_bs: int, has_cached: bool
    ):
        """How many block-table rows a prefill uploads, `running_bs` like a
        decode's; None for none. A drafter's steps after this prefill read the
        table as a decode does, so it goes up whenever there is one."""
        if has_cached or hasattr(self.model_runner, "drafter"):
            return running_bs
        return None

    def prepare_prefill(
        self,
        batch: ScheduledBatch,
        running_bs: int,
        running_tokens: int,
        ubatch_slices: tuple[UBatchSlice, ...] | None = None,
        pcp_groups: "tuple[PcpBalGroup, ...] | None" = None,
    ):
        """`stage_prefill` (and a split prefill's `stage_ubatch_prefill`), the
        phase's uploads, then `finish_prefill` (and `finish_ubatch_prefill`),
        the micro-batches cut once (`_prefill_ubatches`)."""
        group = self.model_runner.h2d_groups["prefill_metadata"]
        attn_metadata, positions = self.stage_prefill(
            batch, running_bs, running_tokens, group
        )
        ubatches = (
            self._prefill_ubatches(batch, ubatch_slices, pcp_groups)
            if ubatch_slices
            else ()
        )
        ubatch_groups = (
            self.stage_ubatch_prefill(batch, attn_metadata, ubatches, pcp_groups)
            if ubatches
            else ()
        )
        for staged in (group, *ubatch_groups):
            staged.publish()
        self.finish_prefill(batch, attn_metadata, running_bs, running_tokens)
        if ubatches:
            self.finish_ubatch_prefill(batch, attn_metadata, ubatches, pcp_groups)
        return attn_metadata, positions

    def stage_ubatch_prefill(
        self,
        batch: ScheduledBatch,
        attn_metadata: AttentionMetaData,
        ubatches: list[tuple[UBatchSlice, PrefillBatch]],
        pcp_groups: "tuple[PcpBalGroup, ...] | None",
    ) -> list:
        """Stage each micro-batch of ``ubatches`` (`_prefill_ubatches`) as a
        prefill of its own, into its ``ub{i}_`` buffers, counted into its
        ``ub{i}_metadata`` group, its metadata hung on
        ``attn_metadata.ubatches``; return the groups."""
        names = tuple(f"ub{i}_metadata" for i in range(len(ubatches)))
        self._check_metadata_writable(*names)
        var = self.model_runner.forward_vars
        groups, ubatch_metadata = [], []
        for ub_idx, ((ub, ub_batch), name) in enumerate(zip(ubatches, names)):
            prefix = f"ub{ub_idx}_"
            group = self.model_runner.h2d_groups[name]
            build_query_layout(
                ub_batch.seqlens_q,
                ub.running_bs,
                ub.running_tokens,
                cu_seqlens_q=var[f"{prefix}cu_seqlens_q"].np,
                batch_ids=var[f"{prefix}batch_id_per_q_token"].np,
            )
            group.set_counts(
                {
                    f"{prefix}cu_seqlens_q": ub.running_bs + 1,
                    f"{prefix}batch_id_per_q_token": ub.running_tokens,
                }
            )
            ubatch_metadata.append(
                self._stage_prefill_batch(
                    ub_batch,
                    var["positions"].np[
                        ub.token_slice.start : ub.token_slice.start
                        + ub_batch.scheduled_tokens
                    ],
                    ub.running_bs,
                    ub.running_tokens,
                    group,
                    prefix,
                )
            )
            groups.append(group)
        attn_metadata.ubatches = ubatch_metadata
        return groups

    def _prefill_ubatches(
        self,
        batch: ScheduledBatch,
        ubatch_slices: tuple[UBatchSlice, ...],
        pcp_groups: "tuple[PcpBalGroup, ...] | None",
    ) -> list[tuple[UBatchSlice, PrefillBatch]]:
        """Each micro-batch of a split prefill, in the step's token axis, and
        its batch (`PrefillBatch.cut`): a slice of ``ubatch_slices``, or under
        PCP its request group's tokens. A prefill with no request has each
        micro-batch's padding a request of its own (`PrefillBatch.padding`),
        neither resuming the other's."""
        if pcp_groups is not None:
            ubatch_slices = tuple(
                UBatchSlice(
                    request_slice=slice(group.req_start, group.req_stop),
                    token_slice=slice(group.tok_start, group.tok_end),
                )
                for group in pcp_groups
            )
        if not batch.total_seqs_num_prefill:
            return [
                (ub, PrefillBatch.padding(ub.running_tokens, batch.is_dummy_run))
                for ub in ubatch_slices
            ]
        prefill = self.prefill_batch(batch, ubatch_slices[-1].token_slice.stop)
        cu_seqlens_q = self.model_runner.forward_vars["cu_seqlens_q"].np[
            : prefill.scheduled_bs + 1
        ]
        return [
            (
                ub,
                prefill.cut(ub.request_slice, ub.scheduled_token_slice, cu_seqlens_q),
            )
            for ub in ubatch_slices
        ]

    def finish_ubatch_prefill(
        self,
        batch: ScheduledBatch,
        attn_metadata: AttentionMetaData,
        ubatches: list[tuple[UBatchSlice, PrefillBatch]],
        pcp_groups: "tuple[PcpBalGroup, ...] | None",
    ) -> None:
        """Each micro-batch's GPU work over its uploaded ``ub{i}_`` metadata
        (`_finish_prefill_batch`)."""
        for ub_idx, ((_, ub_batch), ub_attn) in enumerate(
            zip(ubatches, attn_metadata.ubatches)
        ):
            self._finish_prefill_batch(ub_batch, ub_attn, f"ub{ub_idx}_")

    def finish_prefill(
        self,
        batch: ScheduledBatch,
        attn_metadata: AttentionMetaData,
        running_bs: int,
        running_tokens: int,
    ) -> None:
        """The GPU work reading the step's uploaded prefill metadata
        (`_finish_prefill_batch`). A builder adds its own step-wide work
        after `super()`."""
        self._finish_prefill_batch(
            self.prefill_batch(batch, running_tokens), attn_metadata
        )

    def _finish_prefill_batch(
        self,
        batch: PrefillBatch,
        attn_metadata: AttentionMetaData,
        prefix: str = "",
    ) -> None:
        """The GPU work reading a prefill's uploaded metadata -- the step's or a
        micro-batch's, of the ``prefix`` set. A builder adds its own after
        `super()`."""
        if attn_metadata.has_cached:
            # Layer-invariant, so built once here rather than in every layer's
            # prefix gather. On the device: `total_kv` has no upper bound.
            # `context_lens` is `running_bs` wide and its padded tail is zero,
            # so it still sums to exactly `total_kv` -- which is handed over so
            # the build does not stop the device to measure itself.
            attn_metadata.batch_id_per_k_token = build_batch_ids_device(
                attn_metadata.context_lens, total=attn_metadata.total_kv
            )

    def query_lengths(
        self, batch: ScheduledBatch, is_prefill: bool, running_tokens: int
    ) -> np.ndarray:
        # A prefill with no request lays out its padding request.
        if is_prefill and not batch.total_seqs_num:
            return self.prefill_batch(batch, running_tokens).seqlens_q
        return super().query_lengths(batch, is_prefill, running_tokens)

    def prefill_batch(self, batch: ScheduledBatch, running_tokens: int) -> PrefillBatch:
        """The step's prefill batch: its first `total_seqs_num_prefill`
        requests, or with none its ``running_tokens`` of padding as one
        request (`PrefillBatch.padding`) -- what its query layout lays out
        (`query_lengths`)."""
        bs = batch.total_seqs_num_prefill
        if not bs:
            return PrefillBatch.padding(running_tokens, batch.is_dummy_run)
        return PrefillBatch(
            context_lens=batch.context_lens[:bs],
            seqlens_q=batch.num_scheduled_tokens[:bs],
            block_tables=batch.block_tables[:bs],
            is_dummy_run=batch.is_dummy_run,
        )

    @h2d_producer("seq_metadata", "positions", runner="model_runner")
    def stage_prefill(
        self, batch: ScheduledBatch, running_bs: int, running_tokens: int, group
    ):
        """Write the step's positions and prefill metadata on the host and
        count them into ``group``; return the metadata over GPU views the
        publish fills (`_stage_prefill_batch`), and the positions."""
        prefill = self.prefill_batch(batch, running_tokens)
        scheduled_bs = batch.total_seqs_num_prefill
        scheduled_tokens = batch.total_tokens_num_prefill
        var = self.model_runner.forward_vars
        cu_seqlens_q = var["cu_seqlens_q"].np[: prefill.scheduled_bs + 1]
        # Two asserts because there are two ways this can be wrong and only one
        # of them is about the buffer. The first says `prepare_query_metadata` ran
        # over these rows; the second is the cross-check the derivation of
        # `cached_lens` would otherwise have cost -- `context_lens` is
        # `num_cached_tokens + num_scheduled_tokens` for a PREFILL row and
        # `seq.num_tokens` for a DECODE one (`scheduler.py`), so a decode row
        # among the first `scheduled_bs` shows up here elementwise even in the
        # cases where the totals still happen to agree.
        assert cu_seqlens_q[prefill.scheduled_bs] == prefill.scheduled_tokens, (
            f"published cu_seqlens_q ends at {cu_seqlens_q[prefill.scheduled_bs]}, "
            f"not the {prefill.scheduled_tokens} tokens of the prefill's "
            f"{prefill.scheduled_bs} requests: the batch's first rows are not its "
            "prefill rows"
        )
        # The scheduled requests' alone: a padding request has no position.
        cached_lens = prefill.cached_lens[:scheduled_bs]
        # `.tolist()` against the list, not `np.array_equal` against it: the
        # latter rebuilds an array from the list every step and costs 6x for
        # the same answer (6.1 vs 1.0 us at bs=256).
        assert cached_lens.tolist() == batch.num_cached_tokens[:scheduled_bs], (
            f"derived cached prefix {cached_lens.tolist()} != the scheduler's "
            f"{batch.num_cached_tokens[:scheduled_bs]}"
        )
        prefill_positions(
            self.model_runner.arange_np[:scheduled_tokens],
            cached_lens,
            cu_seqlens_q[: scheduled_bs + 1],
            prefill.seqlens_q[:scheduled_bs],
            out=var["positions"].np[:scheduled_tokens],
        )
        var["positions"].np[scheduled_tokens:running_tokens] = 0
        attn_metadata = self._stage_prefill_batch(
            prefill,
            var["positions"].np[: prefill.scheduled_tokens],
            running_bs,
            running_tokens,
            group,
        )
        mrope_positions = self._build_mrope_prefill_positions(
            batch, running_tokens, group
        )
        if mrope_positions is not None:
            return attn_metadata, mrope_positions
        # Published with the prefill mirrors (`prefill_metadata`).
        group.set_counts({"positions": running_tokens})
        return attn_metadata, var["positions"].gpu[:running_tokens]

    def _stage_prefill_batch(
        self,
        batch: PrefillBatch,
        positions: np.ndarray,
        running_bs: int,
        running_tokens: int,
        group,
        prefix: str = "",
    ) -> AttentionMetaData:
        """Write a prefill ``batch``'s metadata -- the step's, or a
        micro-batch's in the ``prefix`` set -- at its tokens' ``positions``, on
        the host, and count it into ``group``; return it over GPU views the
        publish fills. Its query layout (`cu_seqlens_q`, `batch_id_per_q_token`)
        is staged already. A builder adds its own sources after `super()`.

        Which mirrors go depends on `has_cached`: with no prefix hit nothing
        reads the block tables, the zero `seq_starts`, or the cached prefix, so
        the step does not pay for uploading them. `num_cached_tokens` staying
        absent is also how a consumer learns there is no prefix at all.
        """
        var = self.model_runner.forward_vars
        scheduled_bs = batch.scheduled_bs
        cached_lens = batch.cached_lens
        # `> 0`, not a truth test: a decode row's prefix comes out negative.
        has_cached = bool((cached_lens > 0).any())
        total_kv = self._stage_prefill_seq_lens(batch.context_lens, running_bs, prefix)
        # Packed every step, out to `running_bs`, not only on the steps that
        # upload it (`_prefill_block_table_rows`): the slots are read off it.
        block_table_state(var[f"{prefix}block_tables"]).prepare(
            batch.block_tables, pad_to=running_bs
        )
        self._write_prefill_slots(batch, positions, prefix)
        var[f"{prefix}slot_mapping"].np[len(positions) : running_tokens] = -1

        vars_used = [
            ("cu_seqlens_k", running_bs + 1),
            ("slot_mapping", running_tokens),
            ("context_lens", running_bs),
        ]
        if has_cached:
            var[f"{prefix}num_cached_tokens"].np[:scheduled_bs] = cached_lens
            vars_used += [
                ("seq_starts", scheduled_bs),
                ("num_cached_tokens", scheduled_bs),
            ]
        group.set_counts({prefix + name: count for name, count in vars_used})
        ctx = {name: var[prefix + name].gpu[:count] for name, count in vars_used}
        block_rows = self._prefill_block_table_rows(batch, running_bs, has_cached)
        if block_rows is not None:
            ctx["block_tables"] = block_table_state(var[f"{prefix}block_tables"]).stage(
                block_rows, group
            )
        ctx["cu_seqlens_q"] = var[f"{prefix}cu_seqlens_q"].gpu[: running_bs + 1]
        ctx["batch_id_per_q_token"] = var[f"{prefix}batch_id_per_q_token"].gpu[
            :running_tokens
        ]
        return AttentionMetaData(
            # Python ints: a numpy scalar breaks downstream Triton constexpr
            # binding (`tl.minimum` rejects numpy scalars).
            max_seqlen_q=int(batch.seqlens_q.max(initial=0)),
            max_seqlen_k=int(batch.context_lens.max(initial=0)),
            min_seqlen_q=0,
            dropout_p=0.0,
            has_cached=has_cached,
            total_kv=int(total_kv),
            state=AttnState.PREFILL_PREFIX if has_cached else AttnState.PREFILL_NATIVE,
            **ctx,
        )

    def build(
        self,
        batch: ScheduledBatch,
        running_bs: int,
        running_tokens: int,
        max_seqlen_q: int,
        *,
        is_prefill: bool,
        ubatch_slices: tuple[UBatchSlice, ...] | None = None,
        pcp_groups: "tuple[PcpBalGroup, ...] | None" = None,
    ):
        """Build this step's metadata at the shape `prepare_inputs` settled.

        ``is_prefill`` is the step's phase as `ForwardMode.decide` settled it,
        the group's for a batch with no request.

        ``ubatch_slices``: the TBO micro-batches the step is cut into
        (`ForwardMode.ubatch_slices`), whose metadata the backend builds with
        the step's onto `AttentionMetaData.ubatches`; ``pcp_groups`` a PCP
        prefill's request groups (`ForwardMode.pcp_groups`).

        The step's two units and nothing else: `running_bs` sizes everything
        per-sequence, `running_tokens` everything per-row. Both arrive as
        arguments because neither is derivable from `batch` -- the scheduler
        built it before the DP sync and before any graph was picked -- and
        neither is derivable from the other, since a ragged step replays a flat
        token bucket rather than a `running_bs * q` grid.

        Prefill and decode take both alike: `running_tokens` past the scheduled
        tokens is padding (a token-graph bucket, or an idle DP rank's share of
        a ragged step), and the per-sequence tail pads to `running_bs` for the
        draft pass that follows. A decode's padding is visible past attention:
        the LM head runs every running token, and `postprocess` cuts back to
        the scheduled ones, the one place it does.
        """
        # Run state maintenance on the compute stream before the forward.
        state_ops = batch.state_maintenance_ops
        if state_ops.relocations:
            self.relocate_state_slots(state_ops.relocations)
        if state_ops.checkpoint_stores or state_ops.checkpoint_restores:
            self.execute_paged_state_copies(
                state_ops.checkpoint_stores, state_ops.checkpoint_restores
            )
        if is_prefill:
            built = self.prepare_prefill(
                batch, running_bs, running_tokens, ubatch_slices, pcp_groups
            )
        else:
            built = self.prepare_decode(
                batch, running_bs, running_tokens, max_seqlen_q, ubatch_slices
            )
        attn_metadata = built[0]
        if attn_metadata.batch_id_per_q_token is None:
            # Published with the query prefix (`prepare_query_metadata`).
            attn_metadata.batch_id_per_q_token = self.model_runner.forward_vars[
                "batch_id_per_q_token"
            ].gpu[:running_tokens]
        return built

    # Whether a prefill replays token graphs (`ForwardMode.token_graph`): every
    # launch outside the model's eager regions is sized by the token count
    # alone.
    supports_token_graph = False

    # The `running_bs` a prefill's model records some of its layers at as
    # `ReqGraph`s, each on `build_for_req_graph_capture`'s step
    # (`ModelRunner._record_prefill_req_graphs`). Empty: none.
    prefill_req_graph_sizes: tuple[int, ...] = ()


class AttentionImpl(nn.Module):
    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
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
