# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Conditioning noise augmentation.

Every one of these is a way to be plausibly wrong: the output is always a
tensor of the right shape holding roughly the anchor, so only the exact RNG
contract distinguishes correct from silently-different.
"""

import torch

from atom.diffusion.models.minimax_h3.conditioning import (
    MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    MINIMAX_H3_IMGVID_COND_TIMESTEP,
    audio_cond_noise_aug_rows,
    imgvid_cond_noise_aug_rows,
)
from atom.diffusion.models.minimax_h3.layout import patchify_video_latent

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


def test_audio_default_is_a_passthrough():
    rows = clean_rows(4, width=32)
    assert audio_cond_noise_aug_rows(rows, condition_audio_t=[2], seed=1) is rows


def test_audio_uses_seed_plus_one():
    rows = torch.zeros(4, 32)
    out = audio_cond_noise_aug_rows(rows, condition_audio_t=[2], seed=10, noise_aug=0.0)
    generator = torch.Generator(device="cpu").manual_seed(11)
    expected = torch.randn((4, 32), generator=generator, dtype=torch.float32)
    assert torch.allclose(out, expected, atol=0, rtol=0)
