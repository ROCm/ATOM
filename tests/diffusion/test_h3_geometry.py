# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-H3 layout: canvas/latent geometry, the packed sequence it
produces, and the conditioning noise applied to it.
"""

import itertools

import pytest
import torch

from atom.diffusion.models.minimax_h3.conditioning import (
    MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    MINIMAX_H3_IMGVID_COND_TIMESTEP,
    imgvid_cond_noise_aug_rows,
)
from atom.diffusion.models.minimax_h3.denoise import (
    MiniMaxH3EulerAncestralEta0Scheduler,
    minimax_h3_euler_eta0_step,
    minimax_h3_rf_v_to_x0,
)
from atom.diffusion.models.minimax_h3.layout import (
    TAG_AUDIO,
    TAG_PAD,
    TAG_TEXT,
    TAG_VIDEO,
    MiniMaxH3Geometry,
    align_frame_count,
    audio_latent_t,
    build_local_embedding_layout,
    build_packed_sequence,
    patchify_video_latent,
    time_shift_sigmas,
    video_latent_t,
)

# ==========================================================================
# CANVAS AND LATENT GEOMETRY
# ==========================================================================
#
# Geometry and sampler tests for MiniMax-H3.
#
# The geometry expectations are the values observed in a live 4-rank capture at
# 1344x768 / 5.1667 s (see /md0/dit_golden and /md0/shapes*.json), not numbers
# re-derived from the same formulas the code uses.

# Observed in the live capture.
OBS_HEIGHT, OBS_WIDTH = 768, 1344
OBS_FRAMES = 124
OBS_DURATION = 5.166667
OBS_TEXT_LEN = 2
OBS_VIDEO_ROWS = 37296
OBS_AUDIO_ROWS = 414
OBS_USED = 37712
OBS_SEQ = 37760
OBS_LATENT_T = 37


# ── geometry ──────────────────────────────────────────────────────────────


def test_geometry_reproduces_the_live_capture():
    g = MiniMaxH3Geometry.resolve(
        height=OBS_HEIGHT,
        width=OBS_WIDTH,
        frame_count=OBS_FRAMES,
        duration_seconds=OBS_DURATION,
        text_len=OBS_TEXT_LEN,
    )
    assert g.latent_t == OBS_LATENT_T
    assert (g.latent_h, g.latent_w) == (48, 84)
    assert g.audio_t == 207
    assert g.video_rows == OBS_VIDEO_ROWS
    assert g.audio_rows == OBS_AUDIO_ROWS
    assert g.used_len == OBS_USED
    assert g.seq_len == OBS_SEQ


def test_frame_alignment_is_17n_plus_5():
    assert align_frame_count(124) == 124  # already on the boundary
    assert align_frame_count(125) == 141
    assert align_frame_count(1) == 5
    assert align_frame_count(0) == 1
    for n in (5, 22, 39, 124):
        assert align_frame_count(n) == n


def test_video_latent_t_is_five_per_seventeen_frames():
    for frames, latent_t in ((5, 2), (22, 7), (39, 12), (124, 37), (991, 292)):
        assert video_latent_t(align_frame_count(frames)) == latent_t


def test_audio_latent_t_rounds_at_40hz():
    assert audio_latent_t(5.166667) == 207
    assert audio_latent_t(4.0) == 160
    assert audio_latent_t(15.0) == 600


def test_geometry_rejects_unaligned_resolution():
    with pytest.raises(ValueError, match="multiples of"):
        MiniMaxH3Geometry.resolve(
            height=770,
            width=1344,
            frame_count=124,
            duration_seconds=5.0,
            text_len=2,
        )


def test_ulysses_divisibility_gate():
    g = MiniMaxH3Geometry.resolve(
        height=OBS_HEIGHT,
        width=OBS_WIDTH,
        frame_count=OBS_FRAMES,
        duration_seconds=OBS_DURATION,
        text_len=OBS_TEXT_LEN,
    )
    for world in (1, 2, 4, 8):
        g.validate_ulysses(world)  # 37760 divides by all of these
    # 7 divides the head count (56/7=8) but not the sequence -- this is exactly
    # why Ulysses-7 is unusable while GPU 0 is busy.
    with pytest.raises(ValueError, match="does not divide"):
        g.validate_ulysses(7)


# ── sigma schedule ────────────────────────────────────────────────────────


def test_sigma_schedule_length_and_endpoints():
    sigmas = time_shift_sigmas(num_steps=50, shift_scale=12.0)
    assert len(sigmas) == 50
    assert sigmas[0] == pytest.approx(1.0)
    assert sigmas[-1] == pytest.approx(0.0)
    # 50 sigmas -> 49 denoise iterations, matching the reference server.
    assert len(sigmas) - 1 == 49


def test_sigma_schedule_is_monotonically_decreasing():
    for shift in (3.0, 6.0, 12.0):
        sigmas = time_shift_sigmas(num_steps=50, shift_scale=shift)
        assert all(a > b for a, b in itertools.pairwise(sigmas))


def test_larger_shift_holds_sigma_higher_for_longer():
    """Bigger flow shift spends more steps at high noise."""
    low = time_shift_sigmas(num_steps=50, shift_scale=3.0)
    high = time_shift_sigmas(num_steps=50, shift_scale=12.0)
    mid = len(low) // 2
    assert high[mid] > low[mid]


def test_sigma_schedule_rejects_bad_args():
    with pytest.raises(ValueError, match="shift_scale"):
        time_shift_sigmas(num_steps=50, shift_scale=0.0)
    with pytest.raises(ValueError, match="num_steps"):
        time_shift_sigmas(num_steps=0, shift_scale=6.0)


# ── sampler ───────────────────────────────────────────────────────────────


def test_v_to_x0_identity():
    xt = torch.randn(4, 8)
    v = torch.randn(4, 8)
    t = torch.tensor([0.25])
    torch.testing.assert_close(minimax_h3_rf_v_to_x0(xt, v, t), xt + 0.75 * v)


def test_euler_step_interpolates_between_state_and_denoised():
    state = torch.zeros(4)
    denoised = torch.ones(4)
    out = minimax_h3_euler_eta0_step(state, denoised, sigma_curr=1.0, sigma_next=0.25)
    # r = 0.25 -> 0.25*0 + 0.75*1
    torch.testing.assert_close(out, torch.full((4,), 0.75))


def test_euler_step_at_sigma_zero_is_a_no_op():
    state = torch.randn(4)
    out = minimax_h3_euler_eta0_step(
        state, torch.randn(4), sigma_curr=0.0, sigma_next=0.0
    )
    torch.testing.assert_close(out, state)


def test_euler_step_accumulates_in_fp32_for_bf16_state():
    """bf16 in, bf16 out, but the interpolation must not round mid-way."""
    state = torch.zeros(4, dtype=torch.bfloat16)
    denoised = torch.ones(4, dtype=torch.bfloat16)
    out = minimax_h3_euler_eta0_step(
        state, denoised, sigma_curr=1.0, sigma_next=1.0 - 1 / 512
    )
    assert out.dtype is torch.bfloat16
    assert out[0].item() > 0.0  # a bf16-rounded ratio would flush this to 0


def test_scheduler_steps_both_modalities_on_separate_schedules():
    sched = MiniMaxH3EulerAncestralEta0Scheduler()
    sched.set_shift(12.0)  # no-op by contract
    v = torch.randn(4, 8)
    a = torch.randn(4, 4)
    nxt_v, nxt_a = sched.step(
        visual_latent=v,
        audio_latent=a,
        noise_pred_visual=torch.randn(4, 8),
        noise_pred_audio=torch.randn(4, 4),
        video_timestep=torch.tensor([0.5]),
        audio_timestep=torch.tensor([0.25]),
        video_sigma_curr=0.5,
        video_sigma_next=0.4,
        audio_sigma_curr=0.75,
        audio_sigma_next=0.6,
    )
    assert nxt_v.shape == v.shape and nxt_a.shape == a.shape
    assert torch.isfinite(nxt_v).all() and torch.isfinite(nxt_a).all()


def test_scheduler_catches_sigma_timestep_drift():
    """sigma must equal 1 - t; drift silently corrupts output otherwise."""
    sched = MiniMaxH3EulerAncestralEta0Scheduler()
    with pytest.raises(ValueError, match="video_sigma_curr must equal"):
        sched.step(
            visual_latent=torch.randn(2, 2),
            audio_latent=torch.randn(2, 2),
            noise_pred_visual=torch.randn(2, 2),
            noise_pred_audio=torch.randn(2, 2),
            video_timestep=torch.tensor([0.5]),
            audio_timestep=torch.tensor([0.5]),
            video_sigma_curr=0.9,  # should be 0.5
            video_sigma_next=0.4,
            audio_sigma_curr=0.5,
            audio_sigma_next=0.4,
        )


# ==========================================================================
# PACKED SEQUENCE LAYOUT
# ==========================================================================
#
# Tests for the MiniMax-H3 t2va packed-sequence builder.
#
# The full value-level comparison against the captured golden inputs lives in
# /md0/validate_packed_seq.py (it needs the .pt captures). These cover the
# invariants in CI, plus the observed token counts.

# Resolved geometry of the captured 1344x768 / 5.1667 s request.
OBS = {
    "text_len": 2,
    "latent_t": 37,
    "latent_h": 48,
    "latent_w": 84,
    "audio_t": 207,
}
OBS_VIDEO = 37296
OBS_AUDIO = 414
WORLD = 4


@pytest.fixture(scope="module")
def packed():
    return build_packed_sequence(**OBS)


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


def test_warmup_geometry_matches_a_real_request():
    """The warm shape must be the shape requests actually pack.

    Warming a geometry that no request produces would still pay the
    shape-independent costs but would quietly stop pre-paying the rest, and
    nothing downstream would notice.
    """
    from atom.diffusion.models.minimax_h3.layout import (
        MiniMaxH3Geometry,
        build_packed_sequence,
    )
    from atom.diffusion.models.minimax_h3.pipeline import MiniMaxH3Pipeline

    geo = MiniMaxH3Geometry.resolve(**MiniMaxH3Pipeline.WARMUP_GEOMETRY)
    packed = build_packed_sequence(
        text_len=MiniMaxH3Pipeline.WARMUP_GEOMETRY["text_len"],
        latent_t=geo.latent_t,
        latent_h=geo.latent_h,
        latent_w=geo.latent_w,
        audio_t=geo.audio_t,
    )
    # The validated 1344x768 / 5.17 s t2va layout, measured against the
    # reference: 414 audio + 37,296 video rows, padded to a 64 boundary.
    assert int(packed["img_pos"].numel()) == 37296
    assert int(packed["audio_pos"].numel()) == 414
    assert int(packed["seq_len"]) == 37760
    # Every supported Ulysses degree must divide it, or warmup silently
    # no-ops on exactly the topologies it was written for.
    for degree in (1, 2, 4, 8):
        assert int(packed["seq_len"]) % degree == 0


# ==========================================================================
# CONDITIONING NOISE
# ==========================================================================
#
# Conditioning noise augmentation.
#
# Every one of these is a way to be plausibly wrong: the output is always a
# tensor of the right shape holding roughly the anchor, so only the exact RNG
# contract distinguishes correct from silently-different.

LT, LH, LW = 1, 4, 6
ROWS = LT * (LH // 2) * (LW // 2)


def clean_rows(n=ROWS, width=96):
    return torch.arange(n * width, dtype=torch.float32).view(n, width) / (n * width)


def test_released_visual_coefficient_is_the_captured_timestep():
    assert MINIMAX_H3_IMGVID_COND_TIMESTEP == 0.999
    assert MINIMAX_H3_AUDIO_REF_COND_TIMESTEP == 1.0


def test_coefficient_one_is_an_exact_passthrough():
    rows = clean_rows()
    out = imgvid_cond_noise_aug_rows(
        rows,
        condition_shapes=[(LT, LH, LW)],
        target_latent_t=8,
        seed=7,
        noise_aug=1.0,
    )
    assert out is rows


def test_output_is_the_exact_lerp_of_clean_and_the_seeded_draw():
    """Pin the recipe itself: a*clean + (1-a)*noise, noise drawn at
    target_latent_t + num_conditions frames and sliced to the condition."""
    rows = clean_rows()
    a, seed, target_t = 0.75, 1101, 8
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        1, 24, target_t + 1, LH, LW, generator=generator, dtype=torch.float32
    )[:, :, :LT]
    expected = a * rows + (1.0 - a) * patchify_video_latent(noise, patch_size=(1, 2, 2))
    out = imgvid_cond_noise_aug_rows(
        rows,
        condition_shapes=[(LT, LH, LW)],
        target_latent_t=target_t,
        seed=seed,
        noise_aug=a,
    )
    assert torch.allclose(out, expected, atol=0, rtol=0)


def test_draw_length_depends_on_the_target_not_the_condition():
    """Slicing a longer draw is not the same sample as drawing the short one."""
    rows = clean_rows()
    short = imgvid_cond_noise_aug_rows(
        rows,
        condition_shapes=[(LT, LH, LW)],
        target_latent_t=4,
        seed=3,
        noise_aug=0.5,
    )
    long = imgvid_cond_noise_aug_rows(
        rows,
        condition_shapes=[(LT, LH, LW)],
        target_latent_t=9,
        seed=3,
        noise_aug=0.5,
    )
    assert not torch.allclose(short, long)


def test_condition_count_feeds_back_into_every_condition_s_noise():
    """Adding a second anchor changes the first one's noise -- the draw length
    is target_latent_t + len(conditions)."""
    one = imgvid_cond_noise_aug_rows(
        clean_rows(),
        condition_shapes=[(LT, LH, LW)],
        target_latent_t=8,
        seed=3,
        noise_aug=0.5,
    )
    two = imgvid_cond_noise_aug_rows(
        clean_rows(2 * ROWS),
        condition_shapes=[(LT, LH, LW), (LT, LH, LW)],
        target_latent_t=8,
        seed=3,
        noise_aug=0.5,
    )
    assert not torch.allclose(one, two[:ROWS])


def test_each_condition_restarts_the_same_rng_stream():
    """Two identical conditions get identical noise, not a continued stream."""
    rows = torch.zeros(2 * ROWS, 96)
    out = imgvid_cond_noise_aug_rows(
        rows,
        condition_shapes=[(LT, LH, LW), (LT, LH, LW)],
        target_latent_t=8,
        seed=5,
        noise_aug=0.5,
    )
    assert torch.equal(out[:ROWS], out[ROWS:])


def test_seed_changes_the_result():
    kw = {
        "condition_shapes": [(LT, LH, LW)],
        "target_latent_t": 8,
        "noise_aug": 0.5,
    }
    a = imgvid_cond_noise_aug_rows(clean_rows(), seed=1, **kw)
    b = imgvid_cond_noise_aug_rows(clean_rows(), seed=2, **kw)
    assert not torch.allclose(a, b)
