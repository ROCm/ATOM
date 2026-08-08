# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the MiniMax-H3 t2va packed-sequence builder.

The full value-level comparison against the captured golden inputs lives in
/md0/validate_packed_seq.py (it needs the .pt captures). These cover the
invariants in CI, plus the observed token counts.
"""

import pytest
import torch

from atom.diffusion.models.minimax_h3.packed_sequence import (
    TAG_AUDIO,
    TAG_PAD,
    TAG_TEXT,
    TAG_VIDEO,
    build_local_embedding_layout,
    build_packed_sequence_t2va,
)

# Resolved geometry of the captured 1344x768 / 5.1667 s request.
OBS = {
    "text_len": 2,
    "latent_t": 37,
    "latent_h": 48,
    "latent_w": 84,
    "audio_t": 207,
}
OBS_SEQ = 37760
OBS_USED = 37712
OBS_VIDEO = 37296
OBS_AUDIO = 414
WORLD = 4


@pytest.fixture(scope="module")
def packed():
    return build_packed_sequence_t2va(**OBS)


def test_counts_match_the_live_capture(packed):
    assert packed["seq_len"] == OBS_SEQ
    assert packed["used_len"] == OBS_USED
    assert packed["img_pos"].numel() == OBS_VIDEO
    assert packed["audio_pos"].numel() == OBS_AUDIO
    assert packed["text_pos"].numel() == 2
    assert packed["cu_seqlens"].tolist() == [0, OBS_USED, OBS_SEQ]


def test_blocks_are_contiguous_and_non_overlapping(packed):
    text, audio, img = packed["text_pos"], packed["audio_pos"], packed["img_pos"]
    assert text[-1] + 1 == audio[0]
    assert audio[-1] + 1 == img[0]
    assert img[-1] + 1 == OBS_USED
    all_rows = torch.cat([text, audio, img])
    assert all_rows.unique().numel() == all_rows.numel()


def test_token_tags_partition_the_sequence(packed):
    tags = packed["token_tags"]
    assert tags.numel() == OBS_SEQ
    assert int((tags == TAG_TEXT).sum()) == 2
    assert int((tags == TAG_AUDIO).sum()) == OBS_AUDIO
    assert int((tags == TAG_VIDEO).sum()) == OBS_VIDEO
    # Everything past used_len is padding, and nothing before it is.
    assert int((tags == TAG_PAD).sum()) == OBS_SEQ - OBS_USED
    assert bool((tags[OBS_USED:] == TAG_PAD).all())
    assert not bool((tags[:OBS_USED] == TAG_PAD).any())


def test_position_grid_shape_and_padding(packed):
    g = packed["img_position_ids"]
    assert g.shape == (OBS_SEQ, 3)
    assert g.dtype is torch.float64
    assert bool((g[OBS_USED:] == 0).all()), "pad rows must stay at the origin"


def test_temporal_axis_continues_the_text_counter(packed):
    """Video time starts at text_len, not 0 -- easy to get wrong."""
    g = packed["img_position_ids"]
    first_video = int(packed["img_pos"][0])
    assert g[first_video, 0] == pytest.approx(float(OBS["text_len"]))
    # ...and advances (5/3)*1 for the first token of each 5-group.
    frame_rows = (OBS["latent_h"] // 2) * (OBS["latent_w"] // 2)
    second_frame = first_video + frame_rows
    assert g[second_frame, 0] > g[first_video, 0]


def test_audio_rows_are_channel_major_pinned_to_w_extremes(packed):
    g = packed["img_position_ids"]
    a0 = int(packed["audio_pos"][0])
    at = OBS["audio_t"]
    left = g[a0, 2].item()
    right = g[a0 + at, 2].item()
    assert left != right
    assert bool((g[a0 : a0 + at, 2] == left).all())
    assert bool((g[a0 + at : a0 + 2 * at, 2] == right).all())
    # Channel-major means the temporal counter restarts for the second channel.
    assert g[a0, 0].item() == g[a0 + at, 0].item()


def test_update_mask_is_all_true_for_t2va(packed):
    """t2va has no conditioning rows, so every video row is generated."""
    assert bool(packed["update_mask"].all())
    assert packed["update_mask"].numel() == OBS_VIDEO


def test_builder_rejects_bad_inputs():
    with pytest.raises(ValueError, match="text_len"):
        build_packed_sequence_t2va(**{**OBS, "text_len": 0})
    with pytest.raises(ValueError, match="not divisible by patch"):
        build_packed_sequence_t2va(**{**OBS, "latent_h": 47})


# ── per-rank layout ───────────────────────────────────────────────────────


def test_layout_shards_tile_the_sequence_exactly(packed):
    local = OBS_SEQ // WORLD
    seen_img, seen_audio = [], []
    for rank in range(WORLD):
        layout = build_local_embedding_layout(
            img_pos=packed["img_pos"],
            audio_pos=packed["audio_pos"],
            text_pos=packed["text_pos"],
            row_start=rank * local,
            row_stop=(rank + 1) * local,
        )
        seen_img.append(layout["img_global_ids"])
        seen_audio.append(layout["audio_global_ids"])
        # Row ids must be in range for the shard.
        assert bool((layout["img_row_ids"] >= 0).all())
        assert bool((layout["img_row_ids"] < local).all())

    assert torch.cat(seen_img).numel() == OBS_VIDEO
    assert torch.cat(seen_audio).numel() == OBS_AUDIO


def test_text_range_is_empty_at_text_len_for_shards_without_text(packed):
    """A text-free shard reports (text_len, text_len), matching the reference."""
    local = OBS_SEQ // WORLD
    first = build_local_embedding_layout(
        img_pos=packed["img_pos"],
        audio_pos=packed["audio_pos"],
        text_pos=packed["text_pos"],
        row_start=0,
        row_stop=local,
    )
    assert (first["text_source_start"], first["text_source_stop"]) == (0, 2)

    second = build_local_embedding_layout(
        img_pos=packed["img_pos"],
        audio_pos=packed["audio_pos"],
        text_pos=packed["text_pos"],
        row_start=local,
        row_stop=2 * local,
    )
    assert (second["text_source_start"], second["text_source_stop"]) == (2, 2)


def test_layout_rejects_empty_shard(packed):
    with pytest.raises(ValueError, match="empty row shard"):
        build_local_embedding_layout(
            img_pos=packed["img_pos"],
            audio_pos=packed["audio_pos"],
            text_pos=packed["text_pos"],
            row_start=100,
            row_stop=100,
        )
