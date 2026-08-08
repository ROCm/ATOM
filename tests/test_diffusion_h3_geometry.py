# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Geometry and sampler tests for MiniMax-H3.

The geometry expectations are the values observed in a live 4-rank capture at
1344x768 / 5.1667 s (see /md0/dit_golden and /md0/shapes*.json), not numbers
re-derived from the same formulas the code uses.
"""

import itertools

import pytest
import torch

from atom.diffusion.models.schedulers.euler_ancestral_h3 import (
    MiniMaxH3EulerAncestralEta0Scheduler,
    minimax_h3_euler_eta0_step,
    minimax_h3_rf_v_to_x0,
)
from atom.diffusion.stages.minimax_h3.geometry import (
    MiniMaxH3Geometry,
    align_frame_count,
    audio_latent_t,
    frame_count_from_video_latent_t,
    time_shift_sigmas,
    video_latent_t,
)

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


def test_video_latent_t_roundtrips():
    for frames in (5, 22, 39, 124, 990):
        aligned = align_frame_count(frames)
        assert frame_count_from_video_latent_t(video_latent_t(aligned)) == aligned


def test_frame_count_from_latent_t_rejects_off_lattice():
    with pytest.raises(ValueError, match="5n\\+2"):
        frame_count_from_video_latent_t(4)


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


def test_v_to_x0_rejects_out_of_range_timestep():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        minimax_h3_rf_v_to_x0(torch.randn(2), torch.randn(2), torch.tensor([1.5]))


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


def test_euler_step_rejects_nonzero_next_at_sigma_zero():
    with pytest.raises(ValueError, match="sigma_next must be 0"):
        minimax_h3_euler_eta0_step(
            torch.randn(4), torch.randn(4), sigma_curr=0.0, sigma_next=0.5
        )


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
