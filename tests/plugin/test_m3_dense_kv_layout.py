# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pin the page-16 addressing of M3's K/V-separated dense KV cache.

These are byte-offset assertions, not kernel tests: they run on CPU and check
that ATOM's page-16 reinterpretation lands inside the half of the block it is
supposed to. That is exactly the class of error a GPU run cannot surface --
a page id off by one side writes real KV into real KV and the model keeps
answering, just against the wrong tokens.
"""

from __future__ import annotations

import pytest
import torch

from atom.plugin.vllm.attention.m3_dense_kv_layout import (
    ASM_PAGE_SIZE,
    page16_views,
    pages_per_side,
    rebase_slots_to_page16,
)

# M3 at TP4: 4 KV heads / 4 ranks, head_dim 128, sparse-mandated page 128.
NUM_BLOCKS = 6
BLOCK_SIZE = 128
NUM_KV_HEADS = 1
HEAD_SIZE = 128


def _lbhnc_cache(dtype=torch.uint8):
    """The per-layer view vLLM 0.28 builds for a num_head_slots=2 spec.

    ``create_kv_cache_views`` gives ``[B, H, N, C]``; LBHNC is layer-compact and
    block-compact, so within one layer the blocks are a dense contiguous run.
    """
    return torch.zeros(NUM_BLOCKS, 2, BLOCK_SIZE, NUM_KV_HEADS * HEAD_SIZE, dtype=dtype)


def _side_bounds(block: int, side: int):
    """[start, end) element offsets of one side's plane inside the buffer."""
    plane = BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE
    start = block * 2 * plane + side * plane
    return start, start + plane


# ── slot rebasing ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("block", [0, 1, 5])
@pytest.mark.parametrize("offset", [0, 1, 15, 16, 127])
def test_rebase_doubles_only_the_block_component(block, offset):
    slot = torch.tensor([block * BLOCK_SIZE + offset], dtype=torch.long)

    rebased = rebase_slots_to_page16(slot, BLOCK_SIZE)

    assert rebased.tolist() == [2 * block * BLOCK_SIZE + offset]


def test_rebase_keeps_padding_slots_negative():
    # vLLM pads a captured graph's slot mapping with -1; the writer skips those.
    slots = torch.tensor([-1, 0, -1, BLOCK_SIZE + 3], dtype=torch.long)

    rebased = rebase_slots_to_page16(slots, BLOCK_SIZE)

    assert rebased[0].item() < 0 and rebased[2].item() < 0
    assert rebased.tolist()[1] == 0
    assert rebased.tolist()[3] == 2 * BLOCK_SIZE + 3


def test_rebase_does_not_mutate_its_input():
    slots = torch.tensor([BLOCK_SIZE + 7], dtype=torch.long)
    before = slots.clone()

    rebase_slots_to_page16(slots, BLOCK_SIZE)

    assert torch.equal(slots, before)


def test_rebase_accepts_a_preallocated_out_buffer():
    # The decode path reuses one buffer so a captured graph allocates nothing.
    slots = torch.tensor([BLOCK_SIZE + 7], dtype=torch.long)
    out = torch.empty_like(slots)

    result = rebase_slots_to_page16(slots, BLOCK_SIZE, out=out)

    assert result.data_ptr() == out.data_ptr()
    assert result.tolist() == [2 * BLOCK_SIZE + 7]


# ── page geometry ─────────────────────────────────────────────────────────


def test_a_block_spans_two_sides_worth_of_pages():
    # The bug this guards: assuming block_size/16 pages (the dense-plane
    # stride) when the packed layout gives 2x that.
    assert pages_per_side(BLOCK_SIZE) == BLOCK_SIZE // ASM_PAGE_SIZE == 8


@pytest.mark.parametrize("dtype", [torch.uint8, torch.bfloat16])
def test_views_alias_the_cache_without_copying(dtype):
    kv = _lbhnc_cache(dtype)

    key_view, value_view = page16_views(kv, NUM_KV_HEADS, HEAD_SIZE)

    assert key_view.data_ptr() == kv.data_ptr()
    assert key_view.numel() == kv.numel()
    # V is pre-shifted by one side so one page id addresses both halves.
    assert value_view.numel() == kv.numel() - kv.numel() // (2 * NUM_BLOCKS)


@pytest.mark.parametrize("block", [0, 1, 5])
@pytest.mark.parametrize("offset", [0, 1, 16, 127])
def test_rebased_page_lands_in_the_right_half_of_the_right_block(block, offset):
    """The whole contract in one assertion.

    Writer: page = rebased_slot // 16. Reader: same page id into both views.
    K must land in the block's first plane, V in its second -- never the
    neighbour's, and never each other's.
    """
    kv = _lbhnc_cache()
    key_view, value_view = page16_views(kv, NUM_KV_HEADS, HEAD_SIZE)
    slot = torch.tensor([block * BLOCK_SIZE + offset], dtype=torch.long)
    page = int(rebase_slots_to_page16(slot, BLOCK_SIZE).item()) // ASM_PAGE_SIZE

    base = kv.view(-1)[0].data_ptr()
    itemsize = kv.element_size()
    k_off = (key_view[page].data_ptr() - base) // itemsize
    v_off = (value_view[page].data_ptr() - base) // itemsize
    page_elems = ASM_PAGE_SIZE * NUM_KV_HEADS * HEAD_SIZE

    k_lo, k_hi = _side_bounds(block, 0)
    v_lo, v_hi = _side_bounds(block, 1)
    assert k_lo <= k_off and k_off + page_elems <= k_hi, "K page escaped its plane"
    assert v_lo <= v_off and v_off + page_elems <= v_hi, "V page escaped its plane"


def test_every_page_id_is_used_exactly_once_across_all_blocks():
    """No two logical tokens share a physical page, and nothing is unreachable."""
    kv = _lbhnc_cache()
    key_view, _ = page16_views(kv, NUM_KV_HEADS, HEAD_SIZE)
    slots = torch.arange(NUM_BLOCKS * BLOCK_SIZE, dtype=torch.long)

    pages = (rebase_slots_to_page16(slots, BLOCK_SIZE) // ASM_PAGE_SIZE).tolist()

    # Each 16-token page is claimed by exactly its 16 tokens.
    counts = {}
    for p in pages:
        counts[p] = counts.get(p, 0) + 1
    assert set(counts.values()) == {ASM_PAGE_SIZE}
    assert len(counts) == NUM_BLOCKS * pages_per_side(BLOCK_SIZE)
    assert max(counts) < key_view.shape[0]


# ── guards: a wrong layout must raise, never mis-address ──────────────────


def test_a_packed_kv_cache_is_refused():
    # This is what a spec WITHOUT num_head_slots=2 produces: K|V in the
    # content dim. Silently page-16-ing it would interleave the sides.
    packed = torch.zeros(
        NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, 2 * HEAD_SIZE, dtype=torch.uint8
    )

    with pytest.raises(ValueError, match="num_head_slots"):
        page16_views(packed, NUM_KV_HEADS, HEAD_SIZE)


def test_a_non_contiguous_cache_is_refused():
    # Right shape, wrong strides -- what a layout whose blocks are not dense
    # pages (or a padded page) would hand us. Slicing a wider buffer is the
    # cheapest way to reproduce that here.
    strided = torch.zeros(
        NUM_BLOCKS, 2, BLOCK_SIZE, 2 * NUM_KV_HEADS * HEAD_SIZE, dtype=torch.uint8
    )[..., : NUM_KV_HEADS * HEAD_SIZE]
    assert strided.shape == (NUM_BLOCKS, 2, BLOCK_SIZE, NUM_KV_HEADS * HEAD_SIZE)
    assert not strided.is_contiguous()

    with pytest.raises(ValueError, match="contiguous"):
        page16_views(strided, NUM_KV_HEADS, HEAD_SIZE)


def test_a_block_size_that_is_not_whole_pages_is_refused():
    kv = torch.zeros(2, 2, 20, NUM_KV_HEADS * HEAD_SIZE, dtype=torch.uint8)

    with pytest.raises(ValueError, match="multiple"):
        page16_views(kv, NUM_KV_HEADS, HEAD_SIZE)


def test_a_mismatched_content_dim_is_refused():
    kv = _lbhnc_cache()

    with pytest.raises(ValueError, match="content dim"):
        page16_views(kv, NUM_KV_HEADS, HEAD_SIZE * 2)


# ── block table expansion ─────────────────────────────────────────────────


def test_block_table_expands_to_the_k_pages_of_each_block():
    from atom.plugin.vllm.attention.m3_dense_kv_layout import (
        expand_block_table_to_page16,
    )

    table = torch.tensor([[0, 3], [5, 1]], dtype=torch.int32)

    pages = expand_block_table_to_page16(table, BLOCK_SIZE)

    per_side = pages_per_side(BLOCK_SIZE)
    assert pages.shape == (2, 2 * per_side)
    # block 0 -> pages 0..7, block 3 -> pages 48..55 (2*3*8)
    assert pages[0].tolist() == list(range(8)) + list(range(48, 56))
    assert pages[1].tolist() == list(range(80, 88)) + list(range(16, 24))


def test_expanded_pages_agree_with_the_rebased_slot_of_every_token():
    """The two rebasings must not drift apart -- the writer uses the slot
    mapping, the reader uses the block table, and they address one cache."""
    from atom.plugin.vllm.attention.m3_dense_kv_layout import (
        expand_block_table_to_page16,
    )

    blocks = [4, 0, 2]
    table = torch.tensor([blocks], dtype=torch.long)
    pages = expand_block_table_to_page16(table, BLOCK_SIZE)[0].tolist()

    for logical_pos in range(len(blocks) * BLOCK_SIZE):
        block = blocks[logical_pos // BLOCK_SIZE]
        offset = logical_pos % BLOCK_SIZE
        slot = torch.tensor([block * BLOCK_SIZE + offset], dtype=torch.long)
        writer_page = (
            int(rebase_slots_to_page16(slot, BLOCK_SIZE).item()) // ASM_PAGE_SIZE
        )
        reader_page = pages[logical_pos // ASM_PAGE_SIZE]
        assert writer_page == reader_page, f"drift at logical position {logical_pos}"


def test_expansion_can_write_into_a_preallocated_buffer():
    from atom.plugin.vllm.attention.m3_dense_kv_layout import (
        expand_block_table_to_page16,
    )

    table = torch.tensor([[0, 3]], dtype=torch.int32)
    out = torch.empty(1, 2 * pages_per_side(BLOCK_SIZE), dtype=torch.int32)

    result = expand_block_table_to_page16(table, BLOCK_SIZE, out=out)

    assert result.data_ptr() == out.data_ptr()
    assert out[0, :8].tolist() == list(range(8))


# ── why LHBNC would need none of the above ────────────────────────────────


def _per_layer_view(layout: str):
    """The [B, H, N, C] view vLLM builds; the layout only changes the strides."""
    content = NUM_KV_HEADS * HEAD_SIZE
    strides = {
        # physical [H][B][N][C]: one dense K plane, then one dense V plane
        "LHBNC": (BLOCK_SIZE * content, NUM_BLOCKS * BLOCK_SIZE * content, content, 1),
        # physical [B][H][N][C]: both planes packed inside each block
        "LBHNC": (2 * BLOCK_SIZE * content, BLOCK_SIZE * content, content, 1),
    }[layout]
    buf = torch.zeros(NUM_BLOCKS * 2 * BLOCK_SIZE * content, dtype=torch.uint8)
    return torch.as_strided(buf, (NUM_BLOCKS, 2, BLOCK_SIZE, content), strides)


def test_lhbnc_is_atoms_native_geometry_and_needs_no_rebasing():
    """Under dense planes the adaptation is two free views, and block b's page
    still starts at b * page_size -- which is what every caller assumes."""
    adapted = (
        _per_layer_view("LHBNC")
        .transpose(0, 1)
        .unflatten(-1, (NUM_KV_HEADS, HEAD_SIZE))
    )

    assert adapted.shape == (2, NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    assert adapted.is_contiguous()
    key, value = adapted.unbind(0)
    assert key.is_contiguous() and value.is_contiguous()


def test_lbhnc_cannot_be_adapted_the_same_way():
    """Packing both planes into a block is what forces the page-16 rebasing:
    neither side is a contiguous run any more."""
    adapted = (
        _per_layer_view("LBHNC")
        .transpose(0, 1)
        .unflatten(-1, (NUM_KV_HEADS, HEAD_SIZE))
    )

    assert adapted.shape == (2, NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    assert not adapted.is_contiguous()
    key, _ = adapted.unbind(0)
    assert not key.is_contiguous()


# ── as_atom_5d: the LHBNC path the dense layer actually takes ─────────────


def test_as_atom_5d_matches_atoms_native_shape():
    from atom.plugin.vllm.attention.m3_dense_kv_layout import as_atom_5d

    adapted = as_atom_5d(_per_layer_view("LHBNC"), NUM_KV_HEADS, HEAD_SIZE)

    # exactly what AttentionForVllmMHA allocates for itself outside the plugin
    assert adapted.shape == (2, NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    assert adapted.is_contiguous()


def test_as_atom_5d_addresses_the_same_bytes_as_the_vllm_view():
    """Every (block, side, token, head, dim) cell must survive the two views.

    This is the assertion that would catch a transposed or mis-unflattened
    adaptation -- which reads real KV from the wrong cell and never faults.
    """
    from atom.plugin.vllm.attention.m3_dense_kv_layout import as_atom_5d

    vllm_view = _per_layer_view("LHBNC")
    adapted = as_atom_5d(vllm_view, NUM_KV_HEADS, HEAD_SIZE)

    for block in (0, 1, NUM_BLOCKS - 1):
        for side in (0, 1):
            for token in (0, 17, BLOCK_SIZE - 1):
                for head in range(NUM_KV_HEADS):
                    for dim in (0, 63, HEAD_SIZE - 1):
                        marker = (block * 7 + side * 3 + token + dim) % 251 + 1
                        vllm_view[block, side, token, head * HEAD_SIZE + dim] = marker
                        assert (
                            adapted[side, block, token, head, dim].item() == marker
                        ), f"cell drift at {(block, side, token, head, dim)}"


def test_as_atom_5d_keeps_block_pages_at_their_natural_offset():
    """The property every caller relies on: block b's page starts at
    b * page_size, so slot // 16 needs no rebasing."""
    from atom.plugin.vllm.attention.m3_dense_kv_layout import as_atom_5d

    adapted = as_atom_5d(_per_layer_view("LHBNC"), NUM_KV_HEADS, HEAD_SIZE)
    key = adapted[0]
    page_elems = BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE
    base = key.data_ptr()

    for block in range(NUM_BLOCKS):
        offset = (key[block].data_ptr() - base) // key.element_size()
        assert offset == block * page_elems


def test_as_atom_5d_refuses_the_packed_layout():
    from atom.plugin.vllm.attention.m3_dense_kv_layout import as_atom_5d

    with pytest.raises(ValueError, match="LHBNC"):
        as_atom_5d(_per_layer_view("LBHNC"), NUM_KV_HEADS, HEAD_SIZE)


def test_as_atom_5d_refuses_a_cache_that_was_never_separated():
    from atom.plugin.vllm.attention.m3_dense_kv_layout import as_atom_5d

    packed = torch.zeros(
        NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, 2 * HEAD_SIZE, dtype=torch.uint8
    )

    with pytest.raises(ValueError, match="num_head_slots"):
        as_atom_5d(packed, NUM_KV_HEADS, HEAD_SIZE)


def test_as_atom_5d_refuses_a_mismatched_content_dim():
    from atom.plugin.vllm.attention.m3_dense_kv_layout import as_atom_5d

    with pytest.raises(ValueError, match="content dim"):
        as_atom_5d(_per_layer_view("LHBNC"), NUM_KV_HEADS, HEAD_SIZE * 2)
