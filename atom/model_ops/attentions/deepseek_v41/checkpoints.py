# SPDX-License-Identifier: MIT
"""Byte-exact state movement; no knowledge of attention or Engram arithmetic."""

import numpy as np
import torch

from atom.model_ops.attentions.pool_layout.paged_state_copy import (
    DescriptorStaging,
    launch_copy_descriptor,
    plan_segmented_copy,
)


class StateCopies:
    def __init__(self, cache, spec, max_batch):
        self.cache, self.spec = cache, spec
        if (spec.page_unit_bytes, spec.slot_bytes, spec.image_bytes) != (
            cache.geometry.paged_bytes,
            cache.geometry.state_bytes,
            cache.geometry.state_bytes,
        ):
            raise ValueError("Checkpoint geometry differs from the allocated cache")
        # A PAGE unit is several regions, not one run: its main page and its
        # rows in each index plane. The destination stream repeats them per
        # unit, which is what `write_descriptor` wants one address per.
        self.regions = cache.unit_regions()
        self.plan = plan_segmented_copy(
            [spec.slot_bytes],
            [
                size
                for _ in range(spec.units_per_checkpoint)
                for _, size in self.regions
            ],
            spec.image_bytes,
        )
        self.staging = DescriptorStaging(
            2 * max_batch * self.plan.num_spans, cache.pool.device
        )
        self.staging.reserve([0])

    def entry(self, slot):
        self.cache.require_committed()
        if not 0 <= slot < self.cache.num_slots:
            raise IndexError(f"STATE slot {slot} is out of range")
        return self.cache.state_bytes[slot]

    def relocate(self, pairs):
        # Snapshot sources first: swaps and overlapping relocation chains must
        # not read a destination that an earlier copy has already replaced.
        sources = [self.entry(src).clone() for src, _ in pairs]
        targets = [self.entry(dst) for _, dst in pairs]
        if targets:
            torch._foreach_copy_(targets, sources)

    def _validate(self, op, storing):
        self.entry(op.src_slot if storing else op.dst_slot)
        if (
            op.layout_id != self.spec.layout_id
            or op.total_bytes != self.spec.image_bytes
        ):
            raise ValueError("Checkpoint layout or size mismatch")
        if len(op.unit_ids) != self.spec.units_per_checkpoint or len(
            set(op.unit_ids)
        ) != len(op.unit_ids):
            raise ValueError(
                "Checkpoint needs distinct PAGE units of the declared size"
            )
        if any(unit < 0 or unit >= self.cache.num_pages for unit in op.unit_ids):
            raise IndexError("Checkpoint PAGE unit is out of range")

    def execute(self, stores, restores, descriptor_slot=0):
        if not stores and not restores:
            return
        for ops, storing in ((stores, True), (restores, False)):
            for op in ops:
                self._validate(op, storing)
        if not self.cache.pool.is_cuda:
            for ops, storing in ((stores, True), (restores, False)):
                for op in ops:
                    slot = self.entry(op.src_slot if storing else op.dst_slot)
                    at = 0
                    for unit in op.unit_ids:
                        for view in self.cache.unit_views(unit):
                            take = min(view.numel(), slot.numel() - at)
                            if take <= 0:
                                break
                            state, page = slot[at : at + take], view[:take]
                            (page if storing else state).copy_(
                                state if storing else page
                            )
                            at += take
            return
        rows = self.staging.rows(descriptor_slot)
        total = (len(stores) + len(restores)) * self.plan.num_spans
        if total > rows.shape[0]:
            raise ValueError("Checkpoint copy batch exceeds descriptor capacity")
        at = 0
        for ops, storing in ((stores, True), (restores, False)):
            if not ops:
                continue
            end = at + len(ops) * self.plan.num_spans
            slot_bases = np.asarray(
                [
                    [self.entry(op.src_slot if storing else op.dst_slot).data_ptr()]
                    for op in ops
                ],
                dtype=np.int64,
            )
            page_bases = np.asarray(
                [
                    [
                        base + unit * size
                        for unit in op.unit_ids
                        for base, size in self.regions
                    ]
                    for op in ops
                ],
                dtype=np.int64,
            )
            self.plan.write_descriptor(
                rows[at:end], slot_bases, page_bases, forward=storing
            )
            at = end
        descriptor = self.staging.upload(descriptor_slot, total)
        cut = len(stores) * self.plan.num_spans
        # Restore may consume an image stored in this same maintenance batch.
        launch_copy_descriptor(descriptor[:cut], self.plan)
        launch_copy_descriptor(descriptor[cut:], self.plan)

    def warmup(self):
        if not self.cache.pool.is_cuda:
            return
        slot = self.entry(0)
        # Into the slot itself, one address per destination segment. Segments
        # past the image get an address the plan never emits a span for.
        offsets, at = [], 0
        for _ in range(self.spec.units_per_checkpoint):
            for _, size in self.regions:
                offsets.append(at)
                at += size
        dst = np.asarray([[slot.data_ptr() + off for off in offsets]], dtype=np.int64)
        self.plan.write_descriptor(
            self.staging.rows(0)[: self.plan.num_spans],
            np.asarray([[slot.data_ptr()]], dtype=np.int64),
            dst,
        )
        launch_copy_descriptor(self.staging.upload(0, self.plan.num_spans), self.plan)
