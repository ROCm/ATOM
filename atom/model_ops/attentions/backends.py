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
from atom.model_ops.attentions.sub_pool_spec import SubPoolSpec
from atom.model_ops.dcp_ops import dcp_local_index, dcp_owner_rank
from atom.utils import CpuGpuBuffer, pack_rows
from atom.utils.forward_context import AttentionMetaData, AttnState
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
    def prepare_decode(self, batch: ScheduledBatch, bs: int):
        raise NotImplementedError

    @abstractmethod
    def prepare_prefill(self, batch: ScheduledBatch):
        raise NotImplementedError

    @abstractmethod
    def build(self, batch: ScheduledBatch, bs: int):
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
        """Declare this backend's per-request state checkpoint capability."""
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

    def prepare_block_tables(self, batch: ScheduledBatch, limit: int | None = None):
        """Marshal the batch's block tables into `forward_vars["block_tables"]`.

        `limit` caps how many rows are taken, for callers scheduling fewer
        sequences than the batch carries.
        """
        var = self.model_runner.forward_vars
        rows = batch.block_tables if limit is None else batch.block_tables[:limit]
        pack_rows(var["block_tables"].np, rows)

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

        total_tokens = batch.total_tokens_num_prefill
        positions = self._mrope_cpu_view(total_tokens)
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

        return self._copy_mrope_to_gpu(total_tokens)

    def _build_mrope_decode_positions(
        self,
        batch: ScheduledBatch,
        context_lens: np.ndarray,
        max_seqlen_q: int,
    ) -> torch.Tensor | None:
        if not getattr(self.model_runner, "use_mrope", False):
            return None

        total_tokens = batch.total_tokens_num_decode
        positions = self._mrope_cpu_view(total_tokens)
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

        return self._copy_mrope_to_gpu(total_tokens)

    def prepare_prefill(self, batch: ScheduledBatch):
        bs = batch.total_seqs_num_prefill
        sum_scheduled_tokens = batch.total_tokens_num_prefill
        var = self.model_runner.forward_vars
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        has_cached = False
        # seqs = list(batch.seqs.values())
        # seqs = seqs[:bs]
        for i in range(bs):
            seqlen = batch.context_lens[i]
            cached_seqlen = batch.num_cached_tokens[i]
            if cached_seqlen > 0:
                has_cached = True
            positions.extend(list(range(cached_seqlen, seqlen)))
            seqlen_q = seqlen - cached_seqlen
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not batch.block_tables:
                continue
            block_table = batch.block_tables[i]
            block_size = self.model_runner.block_size
            if self.dcp_world_size > 1:
                W = self.dcp_world_size
                S = self.cp_kv_cache_interleave_size
                virtual_block_size = block_size * W
                for pos in range(cached_seqlen, seqlen):
                    # Block-level interleave: token pos is owned by rank
                    # (pos//S)%W at local index (pos//(S*W))*S + pos%S. S=1 is the
                    # original round-robin. blk_idx = pos // (block_size*W) equals
                    # local_index // block_size (needs block_size % S == 0).
                    if dcp_owner_rank(pos, W, S) == self.dcp_rank:
                        blk_idx = pos // virtual_block_size
                        local_offset = dcp_local_index(pos, W, S) % block_size
                        slot_mapping.append(
                            block_table[blk_idx] * block_size + local_offset
                        )
                    else:
                        slot_mapping.append(-1)
            else:
                first_blk = cached_seqlen // block_size
                last_blk = (seqlen - 1) // block_size
                for blk_idx in range(first_blk, last_blk + 1):
                    blk_start = block_table[blk_idx] * block_size
                    # Offset within block: skip already-cached prefix in first block
                    off_start = (
                        cached_seqlen % block_size if blk_idx == first_blk else 0
                    )
                    # End within block: partial last block
                    off_end = (
                        ((seqlen - 1) % block_size) + 1
                        if blk_idx == last_blk
                        else block_size
                    )
                    slot_mapping.extend(
                        range(blk_start + off_start, blk_start + off_end)
                    )
        if has_cached:
            self.prepare_block_tables(batch)
        # Validate metadata consistency
        assert (
            len(positions) == sum_scheduled_tokens
        ), f"positions length {len(positions)} != sum_scheduled_tokens {sum_scheduled_tokens}"
        if batch.block_tables:
            assert (
                len(slot_mapping) == sum_scheduled_tokens
            ), f"slot_mapping length {len(slot_mapping)} != sum_scheduled_tokens {sum_scheduled_tokens}"
        assert (
            cu_seqlens_q[-1] == sum_scheduled_tokens
        ), f"cu_seqlens_q[-1]={cu_seqlens_q[-1]} != sum_scheduled_tokens={sum_scheduled_tokens}"
        var["positions"].np[:sum_scheduled_tokens] = positions
        var["slot_mapping"].np[:sum_scheduled_tokens] = -1
        var["slot_mapping"].np[: len(slot_mapping)] = slot_mapping
        var["cu_seqlens_q"].np[: bs + 1] = cu_seqlens_q
        var["cu_seqlens_k"].np[: bs + 1] = cu_seqlens_k
        var["context_lens"].np[:bs] = batch.context_lens[:bs]
        min_seqlen_q = 0
        dropout_p = 0.0
        vars_used = [
            ("cu_seqlens_q", bs + 1),
            ("cu_seqlens_k", bs + 1),
            ("slot_mapping", sum_scheduled_tokens),
            ("context_lens", bs),
        ]
        if has_cached:
            vars_used.append(("block_tables", bs))
            vars_used.append(("seq_starts", bs))

        ctx = {el: var[el].copy_to_gpu(num) for el, num in vars_used}
        num_cached_tokens = None
        if has_cached:
            num_cached_tokens = torch.tensor(
                batch.num_cached_tokens[:bs], dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)
            total_tokens = sum(batch.context_lens[:bs])
        total_kv = total_tokens if has_cached else sum_scheduled_tokens
        attn_metadata = AttentionMetaData(
            # Cast to python int — numpy.int32 leaks in via batch.context_lens
            # (numpy array) and breaks downstream Triton kernel constexpr
            # binding (`tl.minimum` rejects numpy scalars).
            max_seqlen_q=int(max_seqlen_q),
            max_seqlen_k=int(max_seqlen_k),
            min_seqlen_q=int(min_seqlen_q),
            dropout_p=dropout_p,
            has_cached=has_cached,
            total_kv=int(total_kv),
            num_cached_tokens=num_cached_tokens,
            state=AttnState.PREFILL_PREFIX if has_cached else AttnState.PREFILL_NATIVE,
            **ctx,
        )
        mrope_positions = self._build_mrope_prefill_positions(batch)
        if mrope_positions is not None:
            positions = mrope_positions
        else:
            positions = var["positions"].copy_to_gpu(sum_scheduled_tokens)

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

    def build(self, batch: ScheduledBatch, bs: int):
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
