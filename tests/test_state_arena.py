# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Layout arithmetic for `StateArena`.

The property under test throughout is that the two ways of looking at the
same memory agree: the per-layer views the kernels bind, and the contiguous
per-entry byte range that checkpointing and RDMA use. Everything runs on CPU
— the arena is pure indexing, no kernels.
"""

import math
from itertools import pairwise

import pytest
import torch

from atom.model_ops.attentions.state_arena import (
    StateArena,
    StateField,
    entry_bytes_for,
    plan_regions,
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


class TestPlanRegions:
    """Packing for the one allocation every per-request pool is carved from.

    Kept here rather than beside the V4 backend so it runs without importing
    AITER. That matters most for the fp8 shape: it carves a RoPE pool per
    layer on top of the unified pools, and cannot be exercised end to end
    while the fused fp8 SWA write is paged-only.
    """

    def test_regions_are_aligned_disjoint_and_in_order(self):
        sizes = [1, 255, 256, 257, 4096, 3]
        offsets, total = plan_regions(sizes)
        assert len(offsets) == len(sizes)
        for off in offsets:
            assert off % 256 == 0
        for (a, n), b in zip(zip(offsets, sizes), offsets[1:]):
            assert a + n <= b, "region overruns the next one"
        assert offsets[-1] + sizes[-1] <= total
        assert total % 256 == 0

    def test_total_is_alignable_so_plans_concatenate(self):
        a, total_a = plan_regions([100, 200])
        b, total_b = plan_regions([300])
        joint, total_joint = plan_regions([100, 200, 300])
        assert joint[:2] == a
        assert joint[2] == total_a + b[0]
        assert total_joint == total_a + total_b

    def test_empty_plan(self):
        assert plan_regions([]) == ([], 0)

    def test_zero_sized_region_still_gets_an_offset(self):
        offsets, _ = plan_regions([256, 0, 256])
        assert len(offsets) == 3
        assert offsets[1] == offsets[2] == 256

    @pytest.mark.parametrize("with_rope", [False, True])
    def test_v4_shaped_layout(self, with_rope):
        """bf16 carves one region per layer plus the arena; fp8 carves two."""
        layers, head_dim, rope_dim = 4, 512, 64
        pages = [1000, 1000, 5000, 3000]
        sizes = [p * head_dim * 2 for p in pages]
        if with_rope:
            sizes += [p * rope_dim * 2 for p in pages]
        arena_bytes = entry_bytes_for(V4_LIKE) * 7
        sizes.append(arena_bytes)

        offsets, total = plan_regions(sizes)
        kv = offsets[:layers]
        rope = offsets[layers : 2 * layers] if with_rope else []
        arena = offsets[-1]

        assert len(rope) == (layers if with_rope else 0)
        # The arena must clear every pool, which is the invariant that broke
        # when `StateArena.view()` addressed from the host allocation's base.
        assert arena >= max(o + s for o, s in zip(offsets[:-1], sizes[:-1]))
        assert arena + arena_bytes <= total
        assert all(a < b for a, b in pairwise(kv))


class TestCarvedBuf:
    """An arena carved out of a larger buffer must stay inside its slice.

    `view()` reaches the storage through `as_strided`, whose storage_offset is
    absolute; an owned buffer sits at offset 0, so forgetting to add the
    slice's own offset is invisible until someone passes `buf`. What it costs
    when it is not caught: every field view starts at the front of the host
    allocation and the arena writes through whatever was carved before it.
    """

    @staticmethod
    def _carve(head_bytes: int, entries: int = 5):
        want = entry_bytes_for(V4_LIKE) * entries
        host = torch.zeros(head_bytes + want, dtype=torch.uint8)
        arena = StateArena(V4_LIKE, entries, device="cpu", buf=host[head_bytes:])
        return host, arena

    def test_views_start_inside_the_slice_not_at_the_host_base(self):
        host, arena = self._carve(4096)
        for field in V4_LIKE:
            offset = arena.view(field.name).data_ptr() - host.data_ptr()
            assert offset >= 4096, (
                f"{field.name} view starts {4096 - offset} bytes before the "
                "arena — it is addressing from the host allocation's base"
            )

    def test_head_of_the_host_allocation_is_untouched(self):
        head_bytes = 4096
        host, arena = self._carve(head_bytes)
        host[:head_bytes] = 0xAB
        for field in V4_LIKE:
            arena.view(field.name).fill_(1.0)
        assert bool((host[:head_bytes] == 0xAB).all()), (
            "writing through the field views modified memory carved before " "the arena"
        )

    def test_rejects_a_misaligned_slice(self):
        want = entry_bytes_for(V4_LIKE) * 2
        host = torch.zeros(8 + want, dtype=torch.uint8)
        with pytest.raises(ValueError, match="boundary"):
            StateArena(V4_LIKE, 2, device="cpu", buf=host[8:])

    def test_carved_and_owned_agree_field_for_field(self):
        _, carved = self._carve(4096)
        owned = build()
        for field in V4_LIKE:
            c, o = carved.view(field.name), owned.view(field.name)
            assert c.shape == o.shape
            assert c.stride() == o.stride()
            assert c.data_ptr() - carved.buf.data_ptr() == (
                o.data_ptr() - owned.buf.data_ptr()
            )


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
