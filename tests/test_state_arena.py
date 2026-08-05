# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Layout arithmetic for `StateArena`.

The property under test throughout is that the two ways of looking at the
same memory agree: the per-layer views the kernels bind, and the contiguous
per-entry byte range that checkpointing and RDMA use. Everything runs on CPU
— the arena is pure indexing, no kernels.
"""

import math

import pytest
import torch

from atom.model_ops.attentions.state_arena import (
    StateArena,
    StateField,
    entry_bytes_for,
)

# Shaped after DeepSeek-V4's compressor state, scaled down: three families,
# each a (kv, score) pair, two of them on the CSA layer count and one on HCA.
NEG_INF = float("-inf")
V4_LIKE = [
    StateField("csa_main_kv", 3, (8, 32), torch.float32),
    StateField("csa_main_score", 3, (8, 32), torch.float32, fill=NEG_INF),
    StateField("csa_idx_kv", 3, (8, 16), torch.float32),
    StateField("csa_idx_score", 3, (8, 16), torch.float32, fill=NEG_INF),
    StateField("hca_main_kv", 2, (128, 32), torch.float32),
    StateField("hca_main_score", 2, (128, 32), torch.float32, fill=NEG_INF),
]


def build(fields=V4_LIKE, entries=5) -> StateArena:
    return StateArena(fields, entries, device="cpu")


class TestEntryBytes:

    def test_sum_of_fields_when_naturally_aligned(self):
        """Real state shapes are coarse multiples of the alignment, so the
        budget is a plain sum — sizing must not pay for padding it will not
        get."""
        expected = sum(f.bytes_per_entry for f in V4_LIKE)
        assert expected % 256 == 0
        assert entry_bytes_for(V4_LIKE) == expected

    def test_pads_between_misaligned_fields(self):
        odd = [
            StateField("a", 1, (3,), torch.float32),  # 12 B
            StateField("b", 1, (3,), torch.float32),  # 12 B
        ]
        # Each field starts on its own 256 B boundary, and the entry as a
        # whole is rounded so entry i+1 starts aligned too.
        assert entry_bytes_for(odd) == 512

    def test_sizing_and_allocation_use_the_same_expression(self):
        """`entry_bytes_for` is what the byte budget is computed from before
        any GPU exists; the built arena must not disagree with it."""
        arena = build()
        assert arena.entry_bytes == entry_bytes_for(V4_LIKE)
        assert arena.total_bytes == arena.entries * arena.entry_bytes
        assert arena.buf.numel() == arena.total_bytes


class TestViewsAreDropInShapes:

    def test_shape_matches_the_standalone_tensor(self):
        arena = build()
        for field in V4_LIKE:
            view = arena.view(field.name)
            assert view.shape == (field.layers, arena.entries) + field.shape
            assert view.dtype == field.dtype

    def test_slot_stride_is_the_whole_entry(self):
        """The only difference from a standalone allocation. Kernels that
        take the slot stride as an argument are unaffected by this; one that
        assumes contiguity is not."""
        arena = build()
        view = arena.view("csa_main_kv")
        itemsize = torch.float32.itemsize
        assert view.stride(1) == arena.entry_bytes // itemsize
        assert view.stride(0) == math.prod((8, 32))
        assert view.stride(-1) == 1

    def test_trailing_dims_stay_contiguous(self):
        """Kernels index the innermost dim with a bare `+ d`."""
        arena = build()
        for field in V4_LIKE:
            per_layer_slot = arena.view(field.name)[0, 0]
            assert per_layer_slot.is_contiguous()


class TestViewsAndEntriesAgree:

    def test_write_through_view_lands_in_that_entry(self):
        arena = build()
        arena.view("hca_main_kv").zero_()
        arena.view("hca_main_kv")[1, 3].fill_(7.0)

        touched = arena.entry(3).view(torch.float32)
        assert (touched == 7.0).sum() == 128 * 32
        for other in (0, 1, 2, 4):
            assert (arena.entry(other).view(torch.float32) == 7.0).sum() == 0

    def test_entries_are_contiguous_and_disjoint(self):
        arena = build()
        for i in range(arena.entries):
            assert arena.entry(i).is_contiguous()
            assert arena.entry(i).numel() == arena.entry_bytes
        base = arena.buf.data_ptr()
        for i in range(arena.entries):
            assert arena.entry(i).data_ptr() == base + i * arena.entry_bytes

    def test_entry_bytes_equal_the_hand_rolled_gather(self):
        """The layout DeepSeek-V4's PD path builds per transfer today:
        each field's `[:, slot]` flattened, concatenated in field order.
        Making that physical is the whole point of the arena, so the two
        must be byte-identical."""
        arena = build()
        torch.manual_seed(0)
        for field in V4_LIKE:
            arena.view(field.name).copy_(
                torch.randn((field.layers, arena.entries) + field.shape)
            )

        slot = 2
        gathered = torch.cat([arena.view(f.name)[:, slot].reshape(-1) for f in V4_LIKE])
        assert torch.equal(arena.entry(slot).view(torch.float32), gathered)

    def test_field_offsets_are_ascending_and_inside_the_entry(self):
        arena = build()
        offsets = [arena.field_offset(f.name) for f in V4_LIKE]
        assert offsets == sorted(offsets)
        last = V4_LIKE[-1]
        assert arena.field_offset(last.name) + last.bytes_per_entry <= arena.entry_bytes


class TestInitialFill:

    def test_kv_zero_score_neg_inf(self):
        arena = build()
        for field in V4_LIKE:
            view = arena.view(field.name)
            if field.fill == 0.0:
                assert torch.equal(view, torch.zeros_like(view))
            else:
                assert torch.isneginf(view).all()

    def test_alignment_padding_is_initialized_too(self):
        """Padding falls outside every field view, but an entry is copied
        whole by checkpointing and RDMA — so it must not be whatever the
        allocator last left there."""
        arena = build([StateField("a", 1, (3,), torch.float32)], entries=2)
        assert arena.entry_bytes == 256  # 12 B of field, 244 B of padding
        arena.view("a").fill_(1.0)
        for i in range(arena.entries):
            assert (arena.entry(i)[12:] == 0).all()


class TestMixedDtypes:

    def test_fields_may_differ_in_dtype(self):
        """GDN keeps its recurrent k and v in different dtypes."""
        fields = [
            StateField("k", 2, (4, 8), torch.bfloat16),
            StateField("v", 2, (4, 8), torch.float32),
        ]
        arena = StateArena(fields, 3, device="cpu")
        assert arena.view("k").dtype == torch.bfloat16
        assert arena.view("v").dtype == torch.float32
        arena.view("k")[1, 2].fill_(1.5)
        arena.view("v")[1, 2].fill_(2.5)
        assert arena.view("k")[1, 2].eq(1.5).all()
        assert arena.view("v")[1, 2].eq(2.5).all()
        assert arena.view("k")[1, 1].eq(0).all()


class TestRejectsBadFieldLists:

    def test_empty(self):
        with pytest.raises(ValueError, match="at least one field"):
            StateArena([], 4, device="cpu")

    def test_duplicate_names(self):
        dup = [
            StateField("a", 1, (4,), torch.float32),
            StateField("a", 1, (4,), torch.float32),
        ]
        with pytest.raises(ValueError, match="duplicate field names"):
            StateArena(dup, 4, device="cpu")
