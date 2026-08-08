# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""ref2va packed layout: reference material ahead of the target.

The layout is easy to get *plausibly* wrong -- right row count, right shapes,
wrong temporal placement -- so these tests pin the ordering and the timeline
rather than just the totals.
"""

import pytest
import torch

from atom.diffusion.models.minimax_h3.packed_sequence import (
    PACKED_SEQUENCE_ALIGNMENT,
    TAG_AUDIO,
    TAG_PAD,
    TAG_TEXT,
    TAG_VIDEO,
    build_packed_sequence_ref2va,
    temporal_position_span,
)

TEXT, LT, LH, LW, AT = 12, 4, 8, 12, 5
FRAME_ROWS = (LH // 2) * (LW // 2)  # 24


def build(ref_blocks, **kw):
    return build_packed_sequence_ref2va(
        text_len=TEXT,
        latent_t=LT,
        latent_h=LH,
        latent_w=LW,
        audio_t=AT,
        ref_blocks=ref_blocks,
        **kw,
    )


def test_no_reference_blocks_matches_the_t2va_totals():
    p = build([])
    assert p["cond_rows"] == 0 and p["cond_audio_rows"] == 0
    assert p["used_len"] == TEXT + AT * 2 + LT * FRAME_ROWS
    assert bool(p["update_mask"].all()) and bool(p["audio_update_mask"].all())


def test_image_block_rows_and_single_temporal_slot():
    p = build([{"kind": "image", "latent_h": 4, "latent_w": 6}])
    rows = (4 // 2) * (6 // 2)
    assert p["cond_rows"] == rows
    g = p["img_position_ids"]
    ref = g[TEXT : TEXT + rows]
    # One image occupies exactly one integer slot on the shared timeline.
    assert torch.equal(ref[:, 0], torch.full((rows,), float(TEXT), dtype=g.dtype))
    assert float(g[TEXT + rows, 0]) == pytest.approx(TEXT + 1.0)


def test_audio_block_advances_the_cursor_by_its_own_length():
    ref_t = 3
    p = build([{"kind": "audio", "ref_audio_t": ref_t}])
    assert p["cond_audio_rows"] == ref_t * 2
    g = p["img_position_ids"]
    # Target audio starts after the reference audio's own span.
    target_audio_start = TEXT + ref_t * 2
    assert float(g[target_audio_start, 0]) == pytest.approx(TEXT + ref_t)


def test_video_block_packs_its_audio_immediately_before_its_video():
    block = {
        "kind": "video_audio",
        "ref_audio_t": 3,
        "latent_t": 2,
        "latent_h": 4,
        "latent_w": 4,
    }
    p = build([block])
    frame_rows = 4
    a_rows, v_rows = 3 * 2, 2 * frame_rows
    assert p["cond_audio_rows"] == a_rows
    assert p["cond_rows"] == v_rows
    # Audio rows come first, then video rows, contiguously after the text.
    assert torch.equal(
        p["audio_pos"][:a_rows], torch.arange(TEXT, TEXT + a_rows, dtype=torch.long)
    )
    assert torch.equal(
        p["img_pos"][:v_rows],
        torch.arange(TEXT + a_rows, TEXT + a_rows + v_rows, dtype=torch.long),
    )


def test_video_block_audio_and_video_share_a_temporal_origin():
    block = {
        "kind": "video_audio",
        "ref_audio_t": 3,
        "latent_t": 2,
        "latent_h": 4,
        "latent_w": 4,
    }
    p = build([block])
    g = p["img_position_ids"]
    a_start = TEXT
    v_start = TEXT + 3 * 2
    assert float(g[a_start, 0]) == pytest.approx(float(TEXT))
    assert float(g[v_start, 0]) == pytest.approx(float(TEXT))


def test_video_block_advances_by_the_longer_of_its_two_spans():
    """A short soundtrack must not shorten the video's temporal footprint."""
    short = {
        "kind": "video_audio",
        "ref_audio_t": 1,
        "latent_t": 4,
        "latent_h": 4,
        "latent_w": 4,
    }
    p = build([short])
    span = temporal_position_span(4)
    assert span > 1.0
    g = p["img_position_ids"]
    target_audio_start = p["audio_pos"][2].item()  # first target audio row
    assert float(g[target_audio_start, 0]) == pytest.approx(TEXT + span)


def test_reference_audio_needs_its_own_update_mask():
    """One mask cannot express 'hold these audio rows but step those'."""
    p = build([{"kind": "audio", "ref_audio_t": 3}])
    assert p["audio_update_mask"].tolist() == [False] * 6 + [True] * (AT * 2)
    # Image-side rows are all target here, so the visual mask stays all-True.
    assert bool(p["update_mask"].all())


def test_blocks_are_laid_out_in_request_order():
    a = {"kind": "image", "latent_h": 4, "latent_w": 4}
    b = {"kind": "audio", "ref_audio_t": 2}
    first = build([a, b])
    second = build([b, a])
    assert first["used_len"] == second["used_len"]
    # Same rows, different placement: the image leads in one and trails in the
    # other, so the temporal cursor lands differently.
    assert not torch.equal(first["img_position_ids"], second["img_position_ids"])


def test_reference_video_audio_pins_to_its_own_width_grid():
    """A reference clip's audio rows key off that clip's grid, not the target's."""
    block = {
        "kind": "video_audio",
        "ref_audio_t": 2,
        "latent_t": 1,
        "latent_h": 4,
        "latent_w": 16,
    }
    p = build([block])
    g = p["img_position_ids"]
    ref_w = g[TEXT + 4 : TEXT + 4 + 32, 2]  # this block's video rows
    ref_audio_w = g[TEXT : TEXT + 4, 2]
    assert float(ref_audio_w[0]) == pytest.approx(float(ref_w.min()))
    assert float(ref_audio_w[-1]) == pytest.approx(float(ref_w.max()))


def test_tags_cover_every_row_exactly_once():
    p = build(
        [
            {"kind": "image", "latent_h": 4, "latent_w": 4},
            {"kind": "audio", "ref_audio_t": 2},
        ]
    )
    tags = p["token_tags"]
    assert int((tags == TAG_TEXT).sum()) == TEXT
    assert int((tags == TAG_VIDEO).sum()) == int(p["img_pos"].numel())
    assert int((tags == TAG_AUDIO).sum()) == int(p["audio_pos"].numel())
    assert int((tags == TAG_PAD).sum()) == int(p["seq_len"]) - int(p["used_len"])


def test_text_tags_can_be_overridden_for_multimodal_prompts():
    tags = torch.ones(TEXT, dtype=torch.long)
    tags[2:5] = TAG_VIDEO
    p = build([{"kind": "image", "latent_h": 4, "latent_w": 4}], text_token_tags=tags)
    assert torch.equal(p["token_tags"][:TEXT], tags)


def test_sequence_is_padded_to_the_alignment():
    p = build([{"kind": "image", "latent_h": 4, "latent_w": 4}])
    assert p["seq_len"] % PACKED_SEQUENCE_ALIGNMENT == 0
    assert p["cu_seqlens"].tolist() == [0, p["used_len"], p["seq_len"]]


def test_explicit_seq_len_must_fit():
    with pytest.raises(ValueError, match="smaller than"):
        build([], seq_len=8)


def test_unknown_block_kind_is_rejected():
    with pytest.raises(ValueError, match="kind"):
        build([{"kind": "subtitle"}])


def test_non_patch_aligned_reference_is_rejected():
    with pytest.raises(ValueError, match="patch-aligned"):
        build([{"kind": "image", "latent_h": 5, "latent_w": 4}])
