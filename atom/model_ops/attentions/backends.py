# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Generic, Optional, TypeVar

if TYPE_CHECKING:
    from atom.kv_transfer.disaggregation.types import KVTransferTensors

import numpy as np
import torch
from aiter.dist.parallel_state import get_tp_group
from torch import nn

from atom.config import DCPConfig
from atom.distributed.dcp_utils import get_dcp_rank, get_dcp_world_size
from atom.model_engine.page_unit_checkpoint import (
    CheckpointRestoreOp,
    CheckpointStoreOp,
)
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.state_runtime import StateTransfer
from atom.model_ops.attention_mla import MLAModules
from atom.model_ops.attentions.pool_layout.sub_pool_spec import SubPoolSpec
from atom.model_ops.attentions.token_layout.prefill import prefill_positions
from atom.model_ops.attentions.token_layout.slots import slot_mapping
from atom.model_ops.dcp_ops import dcp_prefill_slot_mapping
from atom.utils import CpuGpuBuffer, pack_rows
from atom.utils.forward_context import AttentionMetaData, AttnState, ForwardMode
from atom.utils.tbo.ubatch_splitting import (
    UBatchSlice,
    attach_tbo_cpu_lens,
    split_attn_metadata,
)
from atom.utils.tbo.ubatching import tbo_enabled

logger = logging.getLogger("atom")
T = TypeVar("T", bound="BroadcastableModelInput")


class BroadcastableModelInput(ABC):

    @abstractmethod
    def as_broadcastable_tensor_dict(self) -> dict[str, Any]:
        """
        Extract broadcastable fields. Override for fields that require some
        custom deserialization.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_broadcasted_tensor_dict(
        cls: type[T],
        tensor_dict: dict[str, Any],
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
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return AttentionImpl


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
    ):
        raise NotImplementedError

    @abstractmethod
    def prepare_prefill(self, batch: ScheduledBatch, running_bs: int):
        raise NotImplementedError

    @abstractmethod
    def build(
        self,
        batch: ScheduledBatch,
        running_bs: int,
        running_tokens: int,
        max_seqlen_q: int,
    ):
        raise NotImplementedError

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
    def build_for_cudagraph_capture(self, bs: int) -> AttentionMetaData:
        raise NotImplementedError

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
        that owns them — DeepSeek-V4 publishes its `StateArena` alongside the
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
    ) -> None:
        """Copy checkpoints between Active Slots and arbitrary PAGEs."""
        if store_ops or restore_ops:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement PAGE-backed state copy"
            )

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

    def allocate_kv_cache_tensors(
        self, num_kv_heads: int, num_draft_layers: int
    ) -> dict[str, Any]:
        """Allocate the model's primary paged KV cache tensors.

        Called by ModelRunner.allocate_kv_cache() after num_physical_kvcache_blocks
        is known. Builders own the per-attention-type tensor layout (single
        576-dim MLA tensor vs split-K/V MHA tensor; full-rank vs hybrid-only-
        full-attn-rows for Qwen3-Next; per-module deferred for MiMo-V2). The
        runner only setattr's the returned dict onto itself, so model layers
        can access tensors as `model_runner.<name>` (preserving existing
        names: kv_cache, kv_scale, index_cache, etc.).

        Values may be Tensors, None (deferred allocation), or scalar metadata
        (e.g. aligned_index_dim) needed downstream by build_kv_cache_tensor.
        Returns empty dict for builders that do not own the main KV pool.
        """
        return {}

    def build_kv_cache_tensor(self, layer_id: int, module):
        """Build the vLLM-style `KVCacheTensor` registration entry for one
        attention module, OR return None if this builder does not recognize
        the module type.

        Called from ModelRunner.allocate_kv_cache()'s binding loop for every
        module of the model. The builder owns:
          - module-type detection (e.g. `hasattr(module, "use_mla")`)
          - per-attention-type slot index math (attn_idx, gdn_idx, ...)
          - per-module tensor slicing from runner-owned tensors
            (self.model_runner.kv_cache, .mamba_k_cache, ...)
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


class CommonAttentionBuilder(AttentionMetadataBuilder[T], Generic[T]):
    def __init__(self, model_runner):
        self.model_runner = model_runner
        assert model_runner.block_size % self.block_size == 0
        self.block_ratio = model_runner.block_size // self.block_size
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
        # three readers that each want the same numbers: a cu_seqlens ramp at one
        # token per sequence (hence `+ 1`), the real prefix a padded
        # `batch_id_per_token` is restored from, and DSpark's token -> request map.
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
            hf_config.num_attention_heads // get_tp_group().world_size
        )

        i64_kwargs = {"dtype": torch.int64, "device": self.device}
        i32_kwargs = {"dtype": torch.int32, "device": self.device}

        attn_metadata = {
            "slot_mapping": CpuGpuBuffer(self.max_num_batched_tokens, **i64_kwargs),
            "context_lens": CpuGpuBuffer(self.max_bs, **i32_kwargs),
            "block_tables": CpuGpuBuffer(
                self.max_bs, self.block_table_cols, **i32_kwargs
            ),
            "cu_seqlens_q": CpuGpuBuffer(self.max_bs + 1, **i32_kwargs),
            "cu_seqlens_k": CpuGpuBuffer(self.max_bs + 1, **i32_kwargs),
            # Uploaded only on a prefix-cache hit, so consumers read
            # `AttentionMetaData.num_cached_tokens is None` as "no row has any".
            "num_cached_tokens": CpuGpuBuffer(self.max_bs, **i32_kwargs),
            # seq_starts for cp_mha_gather_cache: always zeros (prefix at position 0)
            "seq_starts": CpuGpuBuffer(self.max_bs, **i32_kwargs),
        }

        attn_metadata["cu_seqlens_q"].cpu.copy_(
            torch.arange(0, self.max_bs + 1, step=1, dtype=torch.int32)
        )
        attn_metadata["cu_seqlens_q"].copy_to_gpu()
        attn_metadata["seq_starts"].cpu.zero_()
        attn_metadata["seq_starts"].copy_to_gpu()
        self.model_runner.forward_vars.update(attn_metadata)
        self.has_sliding_window = hasattr(hf_config, "sliding_window")

    def prepare_block_tables(self, batch: ScheduledBatch):
        """Marshal the batch's block tables into `forward_vars["block_tables"]`.

        Runs on every prefill step, not only the ones that upload the buffer:
        `prepare_prefill` reads it back to place each token's KV slot, so a
        caller that wants the table on the device only has to upload it.
        """
        pack_rows(self.model_runner.forward_vars["block_tables"].np, batch.block_tables)

    def _mrope_cpu_view(self, num_tokens: int) -> np.ndarray:
        return (
            self.model_runner.forward_vars["mrope_positions"]
            .np.reshape(-1)[: 3 * num_tokens]
            .reshape(3, num_tokens)
        )

    def _copy_mrope_to_gpu(self, num_tokens: int) -> torch.Tensor:
        buf = self.model_runner.forward_vars["mrope_positions"]
        buf.gpu.reshape(-1)[: 3 * num_tokens].copy_(
            buf.cpu.reshape(-1)[: 3 * num_tokens], non_blocking=True
        )
        return self.model_runner._mrope_positions_view(num_tokens)

    def _build_mrope_prefill_positions(
        self, batch: ScheduledBatch
    ) -> torch.Tensor | None:
        if not getattr(self.model_runner, "use_mrope", False):
            return None

        scheduled_tokens = batch.total_tokens_num_prefill
        positions = self._mrope_cpu_view(scheduled_tokens)
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

        return self._copy_mrope_to_gpu(scheduled_tokens)

    def _build_mrope_decode_positions(
        self,
        batch: ScheduledBatch,
        context_lens: np.ndarray,
        max_seqlen_q: int,
    ) -> torch.Tensor | None:
        if not getattr(self.model_runner, "use_mrope", False):
            return None

        scheduled_tokens = batch.total_tokens_num_decode
        positions = self._mrope_cpu_view(scheduled_tokens)
        offset = 0
        for req_id, context_len in zip(batch.req_ids, context_lens):
            start = int(context_len) - max_seqlen_q
            stop = int(context_len)
            delta = batch.mrope_position_deltas.get(req_id)
            if delta is None:
                base = np.arange(start, stop, dtype=np.int64)
            else:
                base = np.arange(start + int(delta), stop + int(delta), dtype=np.int64)
            positions[:, offset : offset + max_seqlen_q] = base[None, :]
            offset += max_seqlen_q

        return self._copy_mrope_to_gpu(scheduled_tokens)

    def publish_cu_seqlens_q(
        self, batch: ScheduledBatch, forward_mode: ForwardMode
    ) -> None:
        """Publish this step's `cu_seqlens_q`. The only writer.

        Lives here because this class declares the buffer and defines its
        layout, but is CALLED from `prepare_model` before `prepare_input_ids`,
        which addresses each request's span through it and so cannot wait for
        `build()`. `prepare_prefill` cross-checks against this rather than
        deriving its own. Slot 0 is 0 from allocation.

        The tail out to `running_bs` gets the flat cumsum: attention runs at
        that width whether or not a graph is replayed, and a zero-length row
        reads nothing. Left unwritten it holds the previous step's.
        """
        scheduled_bs = batch.total_seqs_num
        assert forward_mode.running_bs >= scheduled_bs, (
            f"running_bs={forward_mode.running_bs} < scheduled_bs={scheduled_bs}; "
            "ForwardMode.decide invariant violated"
        )
        cu = self.model_runner.forward_vars["cu_seqlens_q"]
        cu.np[1 : scheduled_bs + 1] = np.cumsum(batch.num_scheduled_tokens)
        cu.np[scheduled_bs + 1 : forward_mode.running_bs + 1] = batch.total_tokens_num
        # The step's only H2D for this buffer; every consumer slices `.gpu`.
        cu.copy_to_gpu(forward_mode.running_bs + 1)

    def decode_spans(self, batch: ScheduledBatch) -> tuple[int, np.ndarray, np.ndarray]:
        """A pure-decode step's `(bs, per-request lengths, exclusive prefix sum)`.

        One place, so no caller can pair a length vector with someone else's
        cumsum: both come off `num_scheduled_tokens` and the buffer
        `publish_cu_seqlens_q` published from it -- which is also why this is
        only valid after that call, not from the shrink helpers that run before.
        """
        bs = batch.total_seqs_num_decode
        return (
            bs,
            batch.num_scheduled_tokens[:bs],
            self.model_runner.forward_vars["cu_seqlens_q"].np[: bs + 1],
        )

    def _publish_prefill_seq_lens(
        self, context_lens: np.ndarray, scheduled_bs: int, running_bs: int
    ) -> int:
        """Write the two per-sequence length mirrors; return the batch's total KV.

        Straight into the mirrors: this step is their only writer, so a
        temporary would just be copied over.

        Both are padded out to `running_bs`, the width a draft pass that follows
        this step runs at and the one `prepare_decode` fills to. A flat cumsum
        gives each fabricated row zero query length and a zero context makes it
        read nothing; left unwritten they hold the previous step's, and the
        drafter attends to a since-freed request's blocks.
        """
        cu_k = self.model_runner.forward_vars["cu_seqlens_k"].np
        cu_k[0] = 0
        np.cumsum(context_lens, out=cu_k[1 : scheduled_bs + 1])
        cu_k[scheduled_bs + 1 : running_bs + 1] = cu_k[scheduled_bs]
        lens = self.model_runner.forward_vars["context_lens"].np
        lens[:scheduled_bs] = context_lens
        lens[scheduled_bs:running_bs] = 0
        return int(cu_k[scheduled_bs])

    def _write_prefill_slots(
        self,
        batch: ScheduledBatch,
        scheduled_bs: int,
        positions: np.ndarray,
        seqlens_q: np.ndarray,
        cached_lens: np.ndarray,
        context_lens: np.ndarray,
    ) -> None:
        """Place every scheduled token's KV slot into the `slot_mapping` mirror.

        `-1` means "written nowhere": either the batch carries no block tables
        at all, or under DCP this token belongs to another rank.
        """
        block_size = self.model_runner.block_size
        var = self.model_runner.forward_vars
        slots = var["slot_mapping"].np[: positions.shape[0]]
        if not batch.block_tables:
            slots[:] = -1
            return
        assert len(batch.block_tables) >= scheduled_bs, (
            f"block_tables has {len(batch.block_tables)} rows for a batch "
            f"whose first {scheduled_bs} are being prefilled"
        )
        # Marshalled every step, not only on the steps that upload it: the dense
        # path below reads this buffer rather than the batch's ragged rows, so
        # one pack serves both readers. The upload stays gated on `has_cached`.
        self.prepare_block_tables(batch)
        if self.dcp_world_size > 1:
            dcp_slots = dcp_prefill_slot_mapping(
                batch.block_tables[:scheduled_bs],
                cached_lens.tolist(),
                context_lens.tolist(),
                block_size,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_kv_cache_interleave_size,
            )
            assert len(dcp_slots) == slots.shape[0], (
                f"dcp slot mapping is {len(dcp_slots)} long, not the "
                f"{slots.shape[0]} tokens this step forwards"
            )
            slots[:] = dcp_slots
            return
        slot_mapping(
            positions,
            seqlens_q,
            var["block_tables"].np,
            block_size,
            out=slots,
            scratch=self.token_axis_scratch,
        )

    def _upload_prefill_mirrors(
        self,
        scheduled_bs: int,
        running_bs: int,
        scheduled_tokens: int,
        has_cached: bool,
        cached_lens: np.ndarray,
    ) -> dict:
        """Send this step's mirrors to the device; return them keyed for
        `AttentionMetaData`.

        Which ones go depends on `has_cached`: with no prefix hit nothing reads
        the block tables, the zero `seq_starts`, or the cached prefix, so the
        step does not pay for uploading them. `num_cached_tokens` staying
        absent is also how a consumer learns there is no prefix at all.
        """
        var = self.model_runner.forward_vars
        vars_used = [
            ("cu_seqlens_k", running_bs + 1),
            ("slot_mapping", scheduled_tokens),
            ("context_lens", running_bs),
        ]
        if has_cached:
            # `cached_lens`, not `batch.num_cached_tokens`: the same numbers,
            # already an int32 array rather than a Python list.
            var["num_cached_tokens"].np[:scheduled_bs] = cached_lens
            vars_used += [
                ("block_tables", scheduled_bs),
                ("seq_starts", scheduled_bs),
                ("num_cached_tokens", scheduled_bs),
            ]
        ctx = {el: var[el].copy_to_gpu(num) for el, num in vars_used}
        # Already on the device: `publish_cu_seqlens_q` uploads it for every
        # step, so this is a view. One writer AND one upload for this buffer.
        ctx["cu_seqlens_q"] = var["cu_seqlens_q"].gpu[: running_bs + 1]
        return ctx

    def prepare_prefill(self, batch: ScheduledBatch, running_bs: int):
        scheduled_bs = batch.total_seqs_num_prefill
        scheduled_tokens = batch.total_tokens_num_prefill
        var = self.model_runner.forward_vars

        # The cached prefix is subtracted out of the two arrays this step
        # already holds rather than read from `batch.num_cached_tokens`, so it
        # cannot disagree with the query lengths the rest of the step uses.
        context_lens = batch.context_lens[:scheduled_bs]
        seqlens_q = batch.num_scheduled_tokens[:scheduled_bs]
        cached_lens = context_lens - seqlens_q
        cu_seqlens_q = var["cu_seqlens_q"].np[: scheduled_bs + 1]
        # Two asserts because there are two ways this can be wrong and only one
        # of them is about the buffer. The first says `publish_cu_seqlens_q` ran
        # over these rows; the second is the cross-check the derivation above
        # would otherwise have cost -- `context_lens` is
        # `num_cached_tokens + num_scheduled_tokens` for a PREFILL row and
        # `seq.num_tokens` for a DECODE one (`scheduler.py`), so a decode row
        # among the first `scheduled_bs` shows up here elementwise even in the
        # cases where the totals still happen to agree.
        assert cu_seqlens_q[scheduled_bs] == scheduled_tokens, (
            f"published cu_seqlens_q ends at {cu_seqlens_q[scheduled_bs]}, not the "
            f"{scheduled_tokens} tokens scheduled for prefill: the batch's "
            f"first {scheduled_bs} rows are not its prefill rows"
        )
        # `.tolist()` against the list, not `np.array_equal` against it: the
        # latter rebuilds an array from the list every step and costs 6x for
        # the same answer (6.1 vs 1.0 us at bs=256).
        assert cached_lens.tolist() == batch.num_cached_tokens[:scheduled_bs], (
            f"derived cached prefix {cached_lens.tolist()} != the scheduler's "
            f"{batch.num_cached_tokens[:scheduled_bs]}"
        )
        # `> 0`, not a truth test: the asserts above are what stand between this
        # and a decode row, and a decode row's prefix comes out negative.
        has_cached = bool((cached_lens > 0).any())
        max_seqlen_q = int(seqlens_q.max(initial=0))
        max_seqlen_k = int(context_lens.max(initial=0))

        total_kv = self._publish_prefill_seq_lens(
            context_lens, scheduled_bs, running_bs
        )
        positions = prefill_positions(
            self.model_runner.arange_np[:scheduled_tokens],
            cached_lens,
            cu_seqlens_q,
            seqlens_q,
            out=var["positions"].np[:scheduled_tokens],
        )
        self._write_prefill_slots(
            batch, scheduled_bs, positions, seqlens_q, cached_lens, context_lens
        )

        ctx = self._upload_prefill_mirrors(
            scheduled_bs, running_bs, scheduled_tokens, has_cached, cached_lens
        )
        attn_metadata = AttentionMetaData(
            # Cast to python int — numpy.int32 leaks in via batch.context_lens
            # (numpy array) and breaks downstream Triton kernel constexpr
            # binding (`tl.minimum` rejects numpy scalars).
            max_seqlen_q=int(max_seqlen_q),
            max_seqlen_k=int(max_seqlen_k),
            min_seqlen_q=0,
            dropout_p=0.0,
            has_cached=has_cached,
            total_kv=int(total_kv),
            state=AttnState.PREFILL_PREFIX if has_cached else AttnState.PREFILL_NATIVE,
            **ctx,
        )
        mrope_positions = self._build_mrope_prefill_positions(batch)
        if mrope_positions is not None:
            positions = mrope_positions
        else:
            positions = var["positions"].copy_to_gpu(scheduled_tokens)

        return attn_metadata, positions

    def build_ubatch_prefill_metadata(
        self,
        attn_metadata: AttentionMetaData,
        ub_slice: UBatchSlice,
        running_bs: int,
        ubatch_idx: int = 0,
    ) -> AttentionMetaData:
        del ubatch_idx  # only used by builders with per-ubatch plan buffers
        return split_attn_metadata(attn_metadata, ub_slice, running_bs)

    def _attach_tbo_prefill_cpu_lens(
        self, attn_metadata: AttentionMetaData, bs: int
    ) -> None:
        """Publish CPU (numpy) copies of the per-request length arrays so that
        split_attn_metadata can recompute per-ubatch max_seqlen_q/k and total_kv
        on the host with zero device sync.
        """
        if not tbo_enabled():
            return
        var = self.model_runner.forward_vars
        attach_tbo_cpu_lens(
            attn_metadata, "context_lens", var["context_lens"].np[:bs].copy()
        )
        attach_tbo_cpu_lens(
            attn_metadata, "cu_seqlens_q", var["cu_seqlens_q"].np[: bs + 1].copy()
        )
        attach_tbo_cpu_lens(
            attn_metadata, "cu_seqlens_k", var["cu_seqlens_k"].np[: bs + 1].copy()
        )

    def build(
        self,
        batch: ScheduledBatch,
        running_bs: int,
        running_tokens: int,
        max_seqlen_q: int,
    ):
        """Build this step's metadata at the shape `prepare_inputs` settled.

        The step's two units and nothing else: `running_bs` sizes everything
        per-sequence, `running_tokens` everything per-row. Both arrive as
        arguments because neither is derivable from `batch` -- the scheduler
        built it before the DP sync and before any graph was picked -- and
        neither is derivable from the other, since a ragged step replays a flat
        token bucket rather than a `running_bs * q` grid.

        Prefill takes only the first: its rows are the prompt's, so there is no
        `running_tokens` to hand it, but its per-sequence tail still pads to
        `running_bs` for the draft pass that follows. That padding is visible
        past attention -- `cu_seqlens_q` doubles as the selector for the
        sampler's rows, so the LM head emits `running_bs` of them. `postprocess`
        cuts back to the scheduled batch, and that is the one place it does.
        """
        # Run state maintenance on the compute stream before the forward.
        state_ops = batch.state_maintenance_ops
        if state_ops.relocations:
            self.relocate_state_slots(state_ops.relocations)
        if state_ops.checkpoint_stores or state_ops.checkpoint_restores:
            self.execute_paged_state_copies(
                state_ops.checkpoint_stores, state_ops.checkpoint_restores
            )
        is_prefill = batch.total_tokens_num_prefill > 0
        if is_prefill:
            return self.prepare_prefill(batch, running_bs)
        else:
            return self.prepare_decode(batch, running_bs, running_tokens, max_seqlen_q)


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
