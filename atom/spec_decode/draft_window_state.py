# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""A sliding-window draft's KV as per-request rings (`draft_window.py`): copy
checkpoints in the target's PAGE units; a P/D transfer moves a ring as a slot."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch

from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.page_unit_checkpoint import checkpoint_copy_ops_per_batch
from atom.model_engine.state_runtime import StateTransfer
from atom.model_ops.attentions.pool_layout.paged_state_copy import (
    DescriptorStaging,
    launch_copy_descriptor,
    plan_segmented_copy,
)
from atom.model_ops.attentions.pool_layout.sub_pool_spec import (
    SubPoolSpec,
    state_pool,
)
from atom.spec_decode.draft_kv import DraftKvBuilder
from atom.spec_decode.draft_window import DraftWindow
from atom.utils import CpuGpuBuffer

logger = logging.getLogger("atom")

# The step buffers this builder publishes, as one H2D group of its own: the
# target's metadata group is already up by the time a step's slots are known.
DRAFT_STATE_GROUP = "draft_state"


class DraftWindowStateBuilder(DraftKvBuilder):
    """A draft's KV as per-request rings; `make_pool` None on a stage without
    the draft's layers."""

    # Copy checkpoints of this state may span PP stages: a stage holding none
    # of it takes part with no image of its own.
    paged_checkpoints_span_pipeline = True

    def __init__(
        self,
        model_runner,
        make_pool,
        window: DraftWindow,
        block_size: int | None = None,
        kv_scale: float | None = None,
    ):
        if model_runner.config.decode_context_parallel_size > 1:
            raise ValueError(
                "a draft window kept as per-request state is one ring per rank "
                "of every position; DCP shards positions across ranks"
            )
        super().__init__(model_runner, make_pool, block_size, kv_scale)
        self.window = window
        self.num_state_slots = 0
        self._slot_regions_cache = None
        self._target_regions_cache = None
        self._copy_plan_cache = None
        self._staging = None

    @property
    def has_pool(self) -> bool:
        return self._make_pool is not None

    @property
    def kv_pool(self):
        return super().kv_pool if self.has_pool else None

    def build_kv_cache_tensor(self, module):
        return super().build_kv_cache_tensor(module) if self.has_pool else None

    def _pooled_models(self) -> list:
        return super()._pooled_models() if self.has_pool else []

    def release_kv_pools(self) -> None:
        if self.has_pool:
            super().release_kv_pools()

    def ring_entry_bytes(self) -> int:
        """Bytes of one request's ring: 0 on a stage without the draft."""
        if not self.has_pool:
            return 0
        return self.kv_pool.entry_bytes * self.window.ring_blocks

    def sub_pool_specs(self) -> list[SubPoolSpec]:
        return [
            state_pool(STATE_SLOT_CLASS, self.ring_entry_bytes(), entries_per_req=1)
        ]

    def paged_pool_bytes(self, blocks: int) -> int:
        return 0

    def allocate_kv_cache_tensors(self, *, blocks: int, buf) -> dict:
        self.num_blocks = blocks
        return {}

    def allocate_per_req_cache(self, entries: dict[str, int]) -> dict:
        """The rings: entry 0 is the null page, then `ring_blocks` a slot."""
        self.num_state_slots = int(entries.get(STATE_SLOT_CLASS, 0))
        if not self.has_pool:
            return {}
        runner, pool = self.model_runner, self.kv_pool
        if pool.block_size != runner.block_size:
            raise ValueError(
                f"the draft ring's entries are {pool.block_size} tokens and the "
                f"scheduler's blocks {runner.block_size}; a ring entry is one block"
            )
        pool.allocate(1 + self.num_state_slots * self.window.ring_blocks, runner.device)
        logger.info(
            "Allocated draft window rings: %d slots x %d entries of %d B "
            "(window %d)",
            self.num_state_slots,
            self.window.ring_blocks,
            pool.entry_bytes,
            self.window.window,
        )
        return {}

    def warmup_per_req_cache(self) -> None:
        spec = self.model_runner.state_runtime.checkpoint_spec
        if spec is None or not self.has_pool:
            return
        self._copy_plan()
        self._checkpoint_staging()

    def state_transfer(self) -> StateTransfer:
        window = self.window
        return StateTransfer.copy(
            f"dspark-window-v1:w{window.window}r{window.ring_blocks}",
            rebuild_tokens=window.rebuild_tokens,
            trailing_slack=window.trailing_slack,
        )

    def state_transfer_slots(self) -> int:
        return 0

    def checkpoint_image_bytes(self) -> int | None:
        """A ring's published bytes, its fields without their alignment: what
        the slot regions cover and a checkpoint copies."""
        if not self.has_pool:
            return None
        from atom.model_ops.attentions.pool_layout.entry_arena import (
            published_bytes_for,
        )

        per_entry = sum(published_bytes_for(g) for g in self.kv_pool.field_groups)
        return self.window.ring_blocks * per_entry

    def _slot_rows(self, slot: int) -> slice:
        first = 1 + slot * self.window.ring_blocks
        return slice(first, first + self.window.ring_blocks)

    def _region_tensors(self) -> list[torch.Tensor]:
        return (
            [] if not self.has_pool else [t for _, t in self.kv_pool.region_tensors()]
        )

    def state_entry_views(self, slot: int) -> list[torch.Tensor]:
        rows = self._slot_rows(slot)
        return [t[rows].reshape(-1).view(torch.uint8) for t in self._region_tensors()]

    def relocate_state_slots(self, pairs: Sequence[tuple[int, int]]) -> None:
        tensors = self._region_tensors()
        if not tensors or not pairs:
            return
        dsts, srcs = [], []
        for src, dst in pairs:
            dsts += [t[self._slot_rows(dst)] for t in tensors]
            srcs += [t[self._slot_rows(src)] for t in tensors]
        torch._foreach_copy_(dsts, srcs)

    def page_unit_segments(self, unit_ids) -> tuple[np.ndarray, np.ndarray]:
        """`(addresses, sizes)` int64 arrays of an image's bytes in the target's
        PAGE units, in copy order: what a state tier puts from."""
        ptrs, sizes, left = [], [], self.checkpoint_image_bytes() or 0
        for unit in list(unit_ids)[: self._own_units()]:
            for region in self._target_regions():
                if left <= 0:
                    break
                size = min(region.unit_bytes, left)
                ptrs.append(region.unit_addr(unit))
                sizes.append(size)
                left -= size
        return np.asarray(ptrs, dtype=np.int64), np.asarray(sizes, dtype=np.int64)

    def page_unit_views(self, unit_ids):
        """A draft window's images are read as `page_unit_segments`."""
        raise NotImplementedError("a draft window's images are read as segments")

    def slot_regions(self):
        """Each pool region's per-slot ring, slot-indexed past the null entry:
        what a checkpoint copies from and a P/D transfer moves."""
        if self._slot_regions_cache is None:
            from atom.kv_transfer.disaggregation.types import KVTransferRegion

            ring = self.window.ring_blocks
            regions = []
            if self.has_pool:
                roles = [role for role, _ in self.kv_pool.region_tensors()]
                for role, (base, entry_bytes) in zip(
                    roles, self.kv_pool.region_layout(), strict=True
                ):
                    regions.append(
                        KVTransferRegion(
                            base_addr=base + entry_bytes,
                            total_bytes=self.num_state_slots * ring * entry_bytes,
                            unit_bytes=ring * entry_bytes,
                            semantic_role=f"draft-ring:{role}",
                        )
                    )
            self._slot_regions_cache = regions
        return self._slot_regions_cache

    def _target_regions(self):
        """The target's PAGE regions, block-indexed: where an image is stored."""
        if self._target_regions_cache is None:
            tensors = self.model_runner.attn_metadata_builder.get_kv_transfer_tensors()
            if tensors is None or not tensors.pages:
                raise RuntimeError(
                    "the draft window's checkpoints are stored in the target's "
                    "PAGE units, but the target publishes no PAGE regions"
                )
            self._target_regions_cache = [page.region for page in tensors.pages]
        return self._target_regions_cache

    def _copy_plan(self):
        if self._copy_plan_cache is None:
            spec = self.model_runner.state_runtime.checkpoint_spec
            unit_sizes = [r.unit_bytes for r in self._target_regions()]
            if sum(unit_sizes) != spec.page_unit_bytes:
                raise RuntimeError(
                    f"the target publishes {sum(unit_sizes)} B a block but its "
                    f"checkpoint unit is {spec.page_unit_bytes} B"
                )
            self._copy_plan_cache = plan_segmented_copy(
                [r.unit_bytes for r in self.slot_regions()],
                unit_sizes * self._own_units(),
                spec.image_bytes,
            )
        return self._copy_plan_cache

    def _own_units(self) -> int:
        """Units of an op this stage's image fills; under PP an op names the
        stages' joint count and a smaller image leaves the tail unwritten."""
        spec = self.model_runner.state_runtime.checkpoint_spec
        return -(-spec.image_bytes // spec.page_unit_bytes)

    def _checkpoint_staging(self) -> DescriptorStaging:
        if self._staging is None:
            max_ops = checkpoint_copy_ops_per_batch(
                self.model_runner.config.max_num_seqs
            )
            self._staging = DescriptorStaging(
                max_ops * self._copy_plan().num_spans, self.model_runner.device
            )
        return self._staging

    def reserve_checkpoint_descriptors(self, descriptor_slots) -> None:
        if self.has_pool:
            self._checkpoint_staging().reserve(descriptor_slots)

    def execute_paged_state_copies(
        self, store_ops, restore_ops, descriptor_slot: int = 0
    ) -> None:
        """Copy images between rings and the target's PAGE units, every op of
        both directions in one descriptor and one launch."""
        if not self.has_pool or not (store_ops or restore_ops):
            return
        spec = self.model_runner.state_runtime.checkpoint_spec
        for op in (*store_ops, *restore_ops):
            if op.layout_id != spec.layout_id or op.total_bytes != spec.op_bytes:
                raise RuntimeError(
                    f"draft window checkpoint {op.layout_id!r}/{op.total_bytes} B "
                    f"does not match this worker's {spec.layout_id!r}/"
                    f"{spec.op_bytes} B"
                )
        units = self._own_units()
        plan = self._copy_plan()
        per_op = plan.num_spans
        total = (len(store_ops) + len(restore_ops)) * per_op
        staging = self._checkpoint_staging()
        rows = staging.rows(descriptor_slot)
        if total > rows.shape[0]:
            raise RuntimeError(
                f"a step asked to copy {total // per_op} draft window "
                f"checkpoints, more than its descriptor holds"
            )
        slot_regions, unit_regions = self.slot_regions(), self._target_regions()
        at = 0
        for ops, storing in ((store_ops, True), (restore_ops, False)):
            if not ops:
                continue
            slots = [op.src_slot if storing else op.dst_slot for op in ops]
            slot_bases = np.array(
                [[r.unit_addr(s) for r in slot_regions] for s in slots], np.int64
            )
            unit_bases = np.array(
                [
                    [r.unit_addr(u) for u in op.unit_ids[:units] for r in unit_regions]
                    for op in ops
                ],
                np.int64,
            )
            end = at + len(ops) * per_op
            plan.write_descriptor(rows[at:end], slot_bases, unit_bases, forward=storing)
            at = end
        launch_copy_descriptor(staging.upload(descriptor_slot, total), plan)

    def get_kv_transfer_tensors(self):
        from atom.kv_transfer.disaggregation.types import KVTransferTensors

        regions = list(self.slot_regions())
        ring = self.window.ring_blocks
        # Cut as the writer pages an entry (`page_tokens`, 16 for M3.1): a decode at
        # k times this TP takes the k-th of every page's heads.
        pages = super().get_kv_transfer_tensors().pages if self.has_pool else []
        segments = [
            tuple(heads * width for width in widths) * ring
            for heads, widths in (page.head_major_runs for page in pages)
        ]
        return KVTransferTensors(
            slot_regions=regions,
            num_slots=self.num_state_slots,
            # The same list on every TP size and PP stage that holds the draft.
            slot_region_consumer_indices=list(range(len(regions))),
            slot_region_tp_segments=segments,
            slot_region_tp_replicated=[False] * len(regions),
            tp_replication_factor=1,
        )

    def declare_step_buffers(self, forward_vars: dict) -> None:
        """The ring table and context slots, under a group of their own; before
        the runner clones its forward variables and builds its H2D groups."""
        tables = forward_vars["block_tables"]
        rows, cols = tables.cpu.shape
        tokens = forward_vars["slot_mapping"].cpu.shape[0]
        device = torch.device(self.model_runner.device)
        kwargs = {
            "device": device,
            "pin_memory": device.type == "cuda",
            "publication_group": DRAFT_STATE_GROUP,
        }
        forward_vars["draft_block_tables"] = CpuGpuBuffer(
            rows, cols, dtype=torch.int32, **kwargs
        )
        forward_vars["draft_ring_slots"] = CpuGpuBuffer(
            tokens, dtype=torch.int64, **kwargs
        )

    def stage_step(self, batch, running_bs: int, running_tokens: int) -> None:
        """Publish this step's ring table and the ring slot of every target row,
        after the target's metadata (`AttentionMetadataBuilder.build`)."""
        runner = self.model_runner
        var = runner.forward_vars
        group = runner.h2d_groups[DRAFT_STATE_GROUP]
        scheduled_bs = batch.total_seqs_num
        slots = np.full(running_bs, -1, dtype=np.int64)
        committed = batch.state_slots_committed[:scheduled_bs]
        slots[: len(committed)] = committed
        tables = var["draft_block_tables"]
        tables.acquire_write()
        table = self.window.ring_table(slots, tables.np.shape[1])
        tables.np[:running_bs] = table
        group.set_count(tables, running_bs)

        ring_slots = var["draft_ring_slots"]
        ring_slots.acquire_write()
        positions = var["positions"].np[:running_tokens]
        row_seq = var["batch_id_per_q_token"].np[:running_tokens].astype(np.int64)
        row_seq[row_seq >= len(committed)] = -1
        seq_ends = np.zeros(running_bs, dtype=np.int64)
        real = row_seq >= 0
        np.maximum.at(seq_ends, row_seq[real], positions[real] + 1)
        ring_slots.np[:running_tokens] = self.window.context_slots(
            positions, row_seq, seq_ends, table
        )
        group.set_count(ring_slots, running_tokens)
        group.publish()


__all__ = ["DRAFT_STATE_GROUP", "DraftWindowStateBuilder"]
