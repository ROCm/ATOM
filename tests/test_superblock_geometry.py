# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for `SuperblockGeometry` — the arithmetic, and the shapes it must hit.

The K3 numbers here are not illustrative. They were derived independently from
the model's own config in Phase 0 and confirmed against the running server's
tensor sizes; a change that moves them is a change that breaks the pool.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from atom.model_engine.superblock_geometry import (
    LayerField,
    SuperblockGeometry,
    assert_reads_tensor_stride,
    plan_state_fields,
)

# --- K3 TP8, block_size 128, fp8 KV --------------------------------------- #
K3_MLA_LAYERS = 24
K3_KDA_LAYERS = 69
K3_BLOCK_SIZE = 128
K3_ENTRY = 512 + 64  # kv_lora_rank + qk_rope_head_dim
K3_BLOCK_BYTES = K3_MLA_LAYERS * K3_BLOCK_SIZE * K3_ENTRY  # fp8: 1 B/elem

K3_SHAPE_K = (3, 4608)  # conv_kernel-1, conv_dim/TP   bf16
K3_SHAPE_V = (12, 128, 128)  # v_heads/TP, head_v, head_k   fp32
K3_PER_LAYER = math.prod(K3_SHAPE_K) * 2 + math.prod(K3_SHAPE_V) * 4


def k3_fields(align: int = 0) -> list[LayerField]:
    """69 KDA layers, each a bf16 conv side then an fp32 temporal side."""
    return plan_state_fields(
        [(K3_SHAPE_K, 2), (K3_SHAPE_V, 4)] * K3_KDA_LAYERS, align=align
    )


def k3_geometry(num_supers: int = 96, align: int = 0) -> SuperblockGeometry:
    return SuperblockGeometry(
        block_bytes=K3_BLOCK_BYTES,
        state_fields=k3_fields(align),
        num_supers=num_supers,
        align=1,
    )


class TestK3Shape:
    """The numbers Phase 0 measured. These are the contract."""

    def test_block_bytes(self):
        assert K3_BLOCK_BYTES == 1_769_472

    def test_per_layer_bytes(self):
        assert K3_PER_LAYER == 814_080

    def test_state_bytes(self):
        assert k3_geometry().state_bytes == 56_171_520

    def test_blocks_per_super_is_32(self):
        assert k3_geometry().blocks_per_super == 32

    def test_super_bytes(self):
        assert k3_geometry().super_bytes == 56_623_104

    def test_waste_is_under_one_percent(self):
        g = k3_geometry()
        assert g.state_waste_bytes == 451_584
        assert 100.0 * g.state_waste_bytes / g.super_bytes == pytest.approx(
            0.80, abs=0.01
        )


class TestBlocksPerSuperIsNotAConstant:
    """32 is K3-at-this-config, not a property of the scheme.

    Hardcoding it would silently mis-size every other block_size and dtype.
    """

    @pytest.mark.parametrize(
        "block_size,itemsize,expected",
        [
            (128, 1, 32),  # K3 today: fp8
            (128, 2, 16),  # bf16 KV halves the count
            (64, 1, 64),  # a smaller page doubles it
            (256, 1, 16),
        ],
    )
    def test_scales_with_block_size_and_dtype(self, block_size, itemsize, expected):
        block_bytes = K3_MLA_LAYERS * block_size * K3_ENTRY * itemsize
        g = SuperblockGeometry(block_bytes, k3_fields(), num_supers=4, align=1)
        assert g.blocks_per_super == expected

    def test_super_bytes_stays_near_the_state_size(self):
        """Whatever the block size, a superblock is one state rounded up."""
        for block_size in (16, 32, 64, 128, 256):
            block_bytes = K3_MLA_LAYERS * block_size * K3_ENTRY
            g = SuperblockGeometry(block_bytes, k3_fields(), num_supers=4, align=1)
            assert g.state_bytes <= g.super_bytes
            assert g.state_waste_bytes / g.super_bytes < 0.01


class TestSpans:
    def test_slot_span_is_one_whole_superblock(self):
        """A checkpoint copies one range, not one per layer."""
        g = k3_geometry()
        start, stop = g.slot_span(7)
        assert start == 7 * g.super_bytes
        assert stop - start == g.super_bytes

    def test_slot_spans_do_not_overlap(self):
        g = k3_geometry(num_supers=8)
        spans = [g.slot_span(i) for i in range(8)]
        for (_, a_stop), (b_start, _) in pairwise(spans):
            assert a_stop == b_start

    def test_block_span_tiles_its_superblock(self):
        g = k3_geometry()
        first = g.block_span(0)[0]
        last = g.block_span(g.blocks_per_super - 1)[1]
        assert first == 0
        assert last == g.super_bytes

    def test_super_of_block_agrees_with_block_span(self):
        g = k3_geometry(num_supers=4)
        for block_id in (0, 1, 31, 32, 33, 95, 127):
            index = g.super_of_block(block_id)
            start, stop = g.block_span(block_id)
            lo, hi = g.slot_span(index)
            assert lo <= start < stop <= hi

    def test_state_fits_in_the_slot_it_is_given(self):
        g = k3_geometry()
        for layer in range(len(g.state_fields)):
            off, _ = g.state_field_offset(3, layer)
            field = g.state_fields[layer]
            lo, hi = g.slot_span(3)
            assert lo <= off
            assert off + field.nbytes <= hi


class TestStateViewParams:
    def test_slot_stride_is_a_whole_superblock(self):
        """This is what makes the view non-contiguous, and it is deliberate."""
        g = k3_geometry()
        _, stride, _ = g.state_view_params(0)
        assert stride * g.state_fields[0].itemsize == g.super_bytes

    def test_offset_and_shape_match_the_field(self):
        g = k3_geometry()
        for layer in (0, 1, 68, 137):
            off, stride, shape = g.state_view_params(layer)
            field = g.state_fields[layer]
            assert off * field.itemsize == field.offset
            assert shape == field.shape
            assert stride == g.super_bytes // field.itemsize

    def test_rejects_a_misaligned_field(self):
        fields = [LayerField(offset=2, shape=(4,), itemsize=4)]  # 2 % 4 != 0
        g = SuperblockGeometry(
            block_bytes=1024, state_fields=fields, num_supers=2, align=1
        )
        with pytest.raises(ValueError, match="misaligned"):
            g.state_view_params(0)

    def test_rejects_a_superblock_that_does_not_divide(self):
        fields = [LayerField(offset=0, shape=(1,), itemsize=4)]
        g = SuperblockGeometry(
            block_bytes=6, state_fields=fields, num_supers=2, align=1
        )
        assert g.super_bytes % 4  # premise
        with pytest.raises(ValueError, match="does not divide"):
            g.state_view_params(0)


class TestPlanStateFields:
    def test_fields_do_not_overlap(self):
        fields = k3_fields()
        for a, b in pairwise(fields):
            assert a.offset + a.nbytes <= b.offset

    def test_every_field_offset_divides_by_its_own_itemsize(self):
        for field in k3_fields():
            assert field.offset % field.itemsize == 0

    def test_alignment_pads_but_does_not_reorder(self):
        packed = plan_state_fields([((3,), 2), ((5,), 4)], align=0)
        padded = plan_state_fields([((3,), 2), ((5,), 4)], align=256)
        assert [f.shape for f in packed] == [f.shape for f in padded]
        assert padded[1].offset % 256 == 0
        assert padded[1].offset >= packed[1].offset

    def test_padding_counts_toward_state_bytes(self):
        """Otherwise a superblock could be sized under what the fields span."""
        fields = plan_state_fields([((3,), 2), ((5,), 4)], align=256)
        g = SuperblockGeometry(1024, fields, num_supers=2, align=1)
        assert g.state_bytes == fields[-1].offset + fields[-1].nbytes


class TestCapacity:
    def test_num_blocks_counts_every_superblock_as_kv(self):
        g = k3_geometry(num_supers=10)
        assert g.num_blocks == 10 * 32

    def test_total_bytes(self):
        g = k3_geometry(num_supers=10)
        assert g.total_bytes == 10 * g.super_bytes

    def test_zero_supers_is_legal(self):
        """Sizing states a cost before a count is known; V4 does the same."""
        g = k3_geometry(num_supers=0)
        assert g.num_blocks == 0
        assert g.total_bytes == 0
        assert g.blocks_per_super == 32  # the shape still holds


class TestGuards:
    def test_rejects_zero_block_bytes(self):
        with pytest.raises(ValueError, match="block_bytes"):
            SuperblockGeometry(0, k3_fields(), num_supers=1)

    def test_rejects_negative_supers(self):
        with pytest.raises(ValueError, match="num_supers"):
            SuperblockGeometry(K3_BLOCK_BYTES, k3_fields(), num_supers=-1)

    def test_rejects_an_unaligned_superblock(self):
        fields = [LayerField(offset=0, shape=(1,), itemsize=1)]
        with pytest.raises(ValueError, match="aligned"):
            SuperblockGeometry(100, fields, num_supers=1, align=256)

    def test_k3_superblock_is_256b_aligned(self):
        """Not luck: block_bytes is a product of layer count, block size and
        entry width, so it carries the alignment for free."""
        SuperblockGeometry(K3_BLOCK_BYTES, k3_fields(), num_supers=96, align=256)


class TestAiterKernelGuard:
    """Phase 0.1 found the aiter kernel derives its slot stride from the shape.

    On a strided view it reads a neighbouring slot's bytes -- no error, no NaN.
    """

    def test_rejects_an_aiter_kernel(self):
        def kernel():
            pass

        kernel.__module__ = "aiter.ops.triton._triton_kernels.gated_delta_rule"
        with pytest.raises(RuntimeError, match="stride"):
            assert_reads_tensor_stride(kernel)

    def test_accepts_the_atom_wrapper(self):
        def kernel():
            pass

        kernel.__module__ = "atom.model_ops.fla_ops.fused_sigmoid_gating"
        assert_reads_tensor_stride(kernel)

    def test_accepts_the_real_atom_kernel(self):
        pytest.importorskip("torch")
        from atom.model_ops.fla_ops.fused_sigmoid_gating import (
            fused_sigmoid_gating_delta_rule_update,
        )

        assert_reads_tensor_stride(fused_sigmoid_gating_delta_rule_update)
