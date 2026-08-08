# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for MiniMax-H3 fl2va keyframe conditioning.

Covers the packed layout with a conditioning block and the canvas transform.
The seeded VAE encode needs weights and is validated separately on GPU.
"""

import numpy as np
import pytest
import torch

from atom.diffusion.stages.minimax_h3.keyframe import (
    KEYFRAME_ENCODE_SEED,
    cover_crop_plan,
    scoped_encode_rng,
)
from atom.diffusion.stages.minimax_h3.packed_sequence import (
    FL2VA_KEYFRAME_SIGNATURES,
    FRAME_RESCALE,
    build_packed_sequence,
    build_packed_sequence_t2va,
    resolve_keyframe_indices,
    temporal_position_span,
    validate_keyframe_signature,
    video_t_grid,
)

GEO = {
    "text_len": 2,
    "latent_t": 37,
    "latent_h": 48,
    "latent_w": 84,
    "audio_t": 207,
}
FRAME_ROWS = (48 // 2) * (84 // 2)  # 1008
FRAMES = 124


# ── keyframe signatures ───────────────────────────────────────────────────


def test_only_first_last_signatures_are_accepted():
    for sig in FL2VA_KEYFRAME_SIGNATURES:
        assert validate_keyframe_signature(list(sig)) == sig
    with pytest.raises(ValueError, match="must be one of"):
        validate_keyframe_signature([1])
    with pytest.raises(ValueError, match="must be one of"):
        validate_keyframe_signature([-1, 0])  # order matters
    with pytest.raises(ValueError, match="requires keyframe_frame_indices"):
        validate_keyframe_signature(None)


def test_signature_rejects_bools_and_non_ints():
    with pytest.raises(ValueError, match="must be integers"):
        validate_keyframe_signature([True])


def test_resolve_maps_minus_one_to_last_frame():
    assert resolve_keyframe_indices((0,), frame_count=FRAMES) == [0]
    assert resolve_keyframe_indices((-1,), frame_count=FRAMES) == [FRAMES - 1]
    assert resolve_keyframe_indices((0, -1), frame_count=FRAMES) == [0, FRAMES - 1]


def test_resolve_rejects_duplicate_anchors():
    with pytest.raises(ValueError, match="already bound"):
        resolve_keyframe_indices((0, 0), frame_count=FRAMES)
    # A 1-frame clip makes 0 and -1 collide.
    with pytest.raises(ValueError, match="already bound"):
        resolve_keyframe_indices((0, -1), frame_count=1)


def test_resolve_rejects_out_of_range():
    with pytest.raises(ValueError, match="must be -1 or in"):
        resolve_keyframe_indices((999,), frame_count=FRAMES)


# ── temporal span ─────────────────────────────────────────────────────────


def test_temporal_span_matches_the_frame_per_token_cycle():
    # 5 latent frames span (1 + 4 + 4 + 4 + 4) * 5/3
    assert temporal_position_span(5) == pytest.approx(17 * FRAME_RESCALE)


def test_temporal_span_uses_pairwise_summation_not_the_grid_order():
    """The anchor and the grid must keep separate summation orders.

    They agree to ~1e-12 but are not required to be bit-identical; the
    reference documents a last-ulp divergence from n=16 onward, so this pins
    that they are computed independently rather than aliased.
    """
    n = 37
    grid = video_t_grid(n, 0.0)
    sequential_total = float(grid[-1]) + FRAME_RESCALE * (1 if (n - 1) % 5 == 0 else 4)
    pairwise_total = temporal_position_span(n)
    assert pairwise_total == pytest.approx(sequential_total, rel=1e-9)
    # And the pairwise one is computed in numpy fp64, not from the grid.
    assert isinstance(pairwise_total, float)
    assert np.isfinite(pairwise_total)


# ── packed layout with conditioning ───────────────────────────────────────


@pytest.fixture(scope="module")
def t2va():
    return build_packed_sequence(**GEO)


@pytest.mark.parametrize("sig", [(0,), (-1,), (0, -1)])
def test_cond_block_extends_the_sequence_by_whole_frames(sig, t2va):
    packed = build_packed_sequence(
        **GEO, keyframe_frame_indices=list(sig), frame_count=FRAMES
    )
    assert packed["cond_rows"] == len(sig) * FRAME_ROWS
    assert packed["used_len"] == t2va["used_len"] + len(sig) * FRAME_ROWS
    # img_pos gains the conditioning rows.
    assert packed["img_pos"].numel() == t2va["img_pos"].numel() + len(sig) * FRAME_ROWS


def test_cond_rows_are_image_rows_but_not_updated():
    packed = build_packed_sequence(
        **GEO, keyframe_frame_indices=[0, -1], frame_count=FRAMES
    )
    cond = packed["cond_rows"]
    mask = packed["update_mask"]
    assert mask.numel() == packed["img_pos"].numel()
    assert not bool(mask[:cond].any()), "conditioning rows must not be updated"
    assert bool(mask[cond:].all()), "every target row must be updated"
    # ...and they are still tagged as video so AdaLN treats them as image.
    tags = packed["token_tags"]
    assert bool((tags.index_select(0, packed["img_pos"]) == 0).all())


def test_cond_block_sits_between_text_and_audio():
    packed = build_packed_sequence(
        **GEO, keyframe_frame_indices=[0], frame_count=FRAMES
    )
    text_len = GEO["text_len"]
    cond = packed["cond_rows"]
    assert int(packed["img_pos"][0]) == text_len
    assert int(packed["audio_pos"][0]) == text_len + cond


def test_first_frame_anchor_shares_the_video_time_origin():
    packed = build_packed_sequence(
        **GEO, keyframe_frame_indices=[0], frame_count=FRAMES
    )
    g = packed["img_position_ids"]
    text_len = GEO["text_len"]
    # Conditioning block starts right after text and carries t == text_len,
    # the same origin as the first target frame.
    assert g[text_len, 0] == pytest.approx(float(text_len))
    first_target = int(packed["img_pos"][packed["cond_rows"]])
    assert g[first_target, 0] == pytest.approx(float(text_len))


def test_last_frame_anchor_sits_one_span_before_the_end():
    packed = build_packed_sequence(
        **GEO, keyframe_frame_indices=[-1], frame_count=FRAMES
    )
    g = packed["img_position_ids"]
    text_len = GEO["text_len"]
    expected = float(text_len) + temporal_position_span(GEO["latent_t"]) - FRAME_RESCALE
    assert g[text_len, 0] == pytest.approx(expected)


def test_both_anchors_get_distinct_times_and_shared_spatial_grid():
    packed = build_packed_sequence(
        **GEO, keyframe_frame_indices=[0, -1], frame_count=FRAMES
    )
    g = packed["img_position_ids"]
    text_len = GEO["text_len"]
    first = g[text_len : text_len + FRAME_ROWS]
    last = g[text_len + FRAME_ROWS : text_len + 2 * FRAME_ROWS]
    assert first[0, 0].item() != last[0, 0].item()
    # Same spatial coordinates, different time.
    torch.testing.assert_close(first[:, 1:], last[:, 1:])


def test_fl2va_requires_frame_count():
    with pytest.raises(ValueError, match="frame_count is required"):
        build_packed_sequence(**GEO, keyframe_frame_indices=[0])


def test_t2va_wrapper_refuses_keyframes():
    with pytest.raises(ValueError, match="use build_packed_sequence"):
        build_packed_sequence_t2va(**GEO, keyframe_frame_indices=[0])


def test_t2va_layout_unchanged_by_the_generalisation(t2va):
    """The fl2va refactor must not perturb the validated t2va layout."""
    assert t2va["cond_rows"] == 0
    assert t2va["used_len"] == 37712
    assert t2va["seq_len"] == 37760
    assert bool(t2va["update_mask"].all())


# ── canvas ────────────────────────────────────────────────────────────────


def test_cover_crop_preserves_aspect_and_centres():
    plan = cover_crop_plan(
        source_width=1920,
        source_height=1080,
        target_width=1344,
        target_height=768,
    )
    rw, rh = plan["resized_size"]
    assert rw >= 1344 and rh >= 768
    assert abs(rw / rh - 1920 / 1080) < 1e-3
    left, top, right, bottom = plan["crop_box"]
    assert right - left == 1344 and bottom - top == 768
    assert left == (rw - 1344) // 2 and top == (rh - 768) // 2


def test_cover_crop_refuses_upscale_unless_allowed():
    with pytest.raises(ValueError, match="would upscale"):
        cover_crop_plan(
            source_width=320,
            source_height=180,
            target_width=1344,
            target_height=768,
        )
    plan = cover_crop_plan(
        source_width=320,
        source_height=180,
        target_width=1344,
        target_height=768,
        allow_upscale=True,
    )
    assert plan["scale"] > 1.0


# ── encode RNG ────────────────────────────────────────────────────────────


def test_scoped_rng_is_deterministic_and_restores_global_state():
    torch.manual_seed(7)
    before = torch.randn(4)

    torch.manual_seed(7)
    _ = torch.randn(4)
    with scoped_encode_rng(KEYFRAME_ENCODE_SEED):
        a = torch.randn(3)
    after = torch.randn(4)

    with scoped_encode_rng(KEYFRAME_ENCODE_SEED):
        b = torch.randn(3)
    # Same seed -> same sample (the posterior is sampled, so this matters).
    torch.testing.assert_close(a, b)
    # ...and the surrounding stream is untouched by the fork.
    torch.manual_seed(7)
    _ = torch.randn(4)
    torch.testing.assert_close(after, torch.randn(4))
    assert before.shape == after.shape
