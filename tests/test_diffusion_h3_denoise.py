# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for MiniMax-H3 latent prep, token packing and the denoise loop.

CPU only; the DiT is replaced by a stub so the loop's plumbing (row<->packed
scatter, per-step timestep conditioning, sampler wiring) is what's under test.
"""

import pytest
import torch

from atom.diffusion.stages.minimax_h3.denoise import (
    build_timestep_conditioning,
    run_denoise_loop,
)
from atom.diffusion.stages.minimax_h3.latent_prep import (
    build_initial_latents,
    scatter_rows_into_packed,
)
from atom.diffusion.stages.minimax_h3.packed_sequence import (
    TAG_AUDIO,
    TAG_TEXT,
    TAG_VIDEO,
    build_packed_sequence_t2va,
)
from atom.diffusion.stages.minimax_h3.packed_tokens import (
    patchify_video_latent,
    unpack_audio_tokens,
    unpatchify_video_tokens,
)

# Small but structurally faithful: patch (1,2,2), 24 video / 32 audio channels.
SMALL = {"text_len": 2, "latent_t": 3, "latent_h": 4, "latent_w": 6, "audio_t": 5}


@pytest.fixture(scope="module")
def packed():
    return build_packed_sequence_t2va(**SMALL)


# ── token packing ─────────────────────────────────────────────────────────


def test_patchify_unpatchify_roundtrip():
    latent = torch.randn(1, 24, 3, 4, 6)
    rows = patchify_video_latent(latent, patch_size=(1, 2, 2))
    assert rows.shape == (3 * 2 * 3, 24 * 1 * 2 * 2)
    back = unpatchify_video_tokens(
        rows, latent_shape=(3, 2, 3, 24), patch_size=(1, 2, 2)
    )
    torch.testing.assert_close(back, latent)


def test_patchify_rejects_indivisible_grid():
    with pytest.raises(ValueError, match="not divisible by patch"):
        patchify_video_latent(torch.randn(1, 24, 3, 5, 6), patch_size=(1, 2, 2))


def test_unpack_audio_tokens_is_channel_major():
    rows = torch.arange(10 * 32, dtype=torch.float32).reshape(10, 32)
    out = unpack_audio_tokens(rows, audio_t=10, audio_channel=2)
    assert out.shape == (2, 32, 5)
    # First channel is the first half of the rows.
    torch.testing.assert_close(out[0].permute(1, 0), rows[:5])


def test_unpack_audio_rejects_mismatched_count():
    with pytest.raises(ValueError, match="!= audio_t"):
        unpack_audio_tokens(torch.randn(9, 32), audio_t=10, audio_channel=2)


# ── latent prep ───────────────────────────────────────────────────────────


def test_initial_latents_shapes_and_dtype():
    v, a = build_initial_latents(seed=1101, **_latent_kwargs())
    assert v.shape == (3 * 2 * 3, 96)
    assert a.shape == (5 * 2, 32)
    assert v.dtype is torch.float32 and a.dtype is torch.float32
    assert v.device.type == "cpu" and a.device.type == "cpu"


def _latent_kwargs():
    return {
        "latent_t": SMALL["latent_t"],
        "latent_h": SMALL["latent_h"],
        "latent_w": SMALL["latent_w"],
        "audio_t": SMALL["audio_t"],
    }


def test_initial_latents_are_seed_reproducible():
    a1, b1 = build_initial_latents(seed=7, **_latent_kwargs())
    a2, b2 = build_initial_latents(seed=7, **_latent_kwargs())
    torch.testing.assert_close(a1, a2)
    torch.testing.assert_close(b1, b2)
    a3, _ = build_initial_latents(seed=8, **_latent_kwargs())
    assert not torch.allclose(a1, a3)


def test_video_noise_is_drawn_on_the_raw_latent_then_patchified():
    """Drawing in row shape gives a different sample for the same seed."""
    seed = 1101
    v, _ = build_initial_latents(seed=seed, **_latent_kwargs())
    gen = torch.Generator().manual_seed(seed)
    expected = patchify_video_latent(
        torch.randn(1, 24, 3, 4, 6, generator=gen, dtype=torch.float32),
        patch_size=(1, 2, 2),
    )
    torch.testing.assert_close(v, expected)

    gen_wrong = torch.Generator().manual_seed(seed)
    wrong = torch.randn(v.shape[0], 96, generator=gen_wrong, dtype=torch.float32)
    assert not torch.allclose(v, wrong)


def test_audio_uses_an_independent_generator_with_the_same_seed():
    seed = 1101
    _, a = build_initial_latents(seed=seed, **_latent_kwargs())
    gen = torch.Generator().manual_seed(seed)
    expected = torch.randn(10, 32, generator=gen, dtype=torch.float32)
    torch.testing.assert_close(a, expected)


def test_default_seed_is_42():
    a, _ = build_initial_latents(**_latent_kwargs())
    b, _ = build_initial_latents(seed=42, **_latent_kwargs())
    torch.testing.assert_close(a, b)


def test_scatter_places_rows_at_their_global_ids(packed):
    v, a = build_initial_latents(seed=3, **_latent_kwargs())
    x, audio_x = scatter_rows_into_packed(
        video_rows=v,
        audio_rows=a,
        img_pos=packed["img_pos"],
        audio_pos=packed["audio_pos"],
        seq_len=packed["seq_len"],
    )
    assert x.shape == (1, packed["seq_len"], 96)
    assert audio_x.shape == (1, packed["seq_len"], 32)
    torch.testing.assert_close(x[0].index_select(0, packed["img_pos"]), v)
    torch.testing.assert_close(audio_x[0].index_select(0, packed["audio_pos"]), a)
    # Text rows carry no latent.
    assert bool((x[0, : SMALL["text_len"]] == 0).all())


# ── timestep conditioning ─────────────────────────────────────────────────


def test_equal_timesteps_collapse_to_one_unique(packed):
    """Step 0 has both modalities at t=0, matching the captured [1]-shaped
    unique_timesteps."""
    unique, inverse, combined = build_timestep_conditioning(
        token_tags=packed["token_tags"],
        img_pos=packed["img_pos"],
        audio_pos=packed["audio_pos"],
        video_timestep=0.0,
        audio_timestep=0.0,
    )
    assert unique.numel() == 1
    assert int(inverse.max()) == 0
    assert int(combined.max()) < 3


def test_differing_timesteps_give_two_uniques_and_correct_gather(packed):
    unique, inverse, combined = build_timestep_conditioning(
        token_tags=packed["token_tags"],
        img_pos=packed["img_pos"],
        audio_pos=packed["audio_pos"],
        video_timestep=0.4,
        audio_timestep=0.7,
    )
    assert unique.numel() == 2
    # Every token must resolve back to its own modality's timestep.
    per_token = unique[inverse]
    assert bool((per_token.index_select(0, packed["audio_pos"]) == 0.7).all())
    assert bool((per_token.index_select(0, packed["img_pos"]) == 0.4).all())
    # combined = tag + 3*inverse, so it stays inside the AdaLN table.
    assert int(combined.max()) < 3 * unique.numel()

    tags = packed["token_tags"]
    for pos, tag in ((packed["img_pos"], TAG_VIDEO), (packed["audio_pos"], TAG_AUDIO)):
        assert bool((tags.index_select(0, pos) == tag).all())
    assert int(tags[0]) == TAG_TEXT


# ── denoise loop ──────────────────────────────────────────────────────────


def test_denoise_loop_runs_and_converges_to_the_clean_estimate(packed):
    """With a stub DiT predicting a constant velocity, the loop must end at the
    implied x0 and take exactly len(sigmas)-1 steps."""
    v0, a0 = build_initial_latents(seed=5, **_latent_kwargs())
    calls = []

    def stub_dit(**kwargs):
        calls.append(kwargs)
        n_v = kwargs["x"].shape[1]
        del n_v
        return (
            torch.zeros_like(v0),
            torch.zeros_like(a0),
        )

    sigmas = [1.0, 0.5, 0.0]
    seen = []
    v, a = run_denoise_loop(
        dit=stub_dit,
        video_rows=v0,
        audio_rows=a0,
        packed=packed,
        video_sigmas=sigmas,
        audio_sigmas=sigmas,
        rank_slice=(0, packed["seq_len"]),
        prompt_embeds=torch.randn(2, 8, dtype=torch.bfloat16),
        refined_prompt_embeds_length=2,
        rope_cache=torch.zeros(packed["seq_len"], 96, dtype=torch.bfloat16),
        progress=lambda i, n: seen.append((i, n)),
    )

    assert len(calls) == len(sigmas) - 1
    assert seen == [(1, 2), (2, 2)]
    assert v.shape == v0.shape and a.shape == a0.shape
    # v = 0 means denoised == state, so the interpolation is a fixed point.
    torch.testing.assert_close(v, v0)
    torch.testing.assert_close(a, a0)


def test_denoise_loop_feeds_the_dit_full_length_buffers(packed):
    v0, a0 = build_initial_latents(seed=5, **_latent_kwargs())
    captured = {}

    def stub_dit(**kwargs):
        captured.update(kwargs)
        return torch.zeros_like(v0), torch.zeros_like(a0)

    run_denoise_loop(
        dit=stub_dit,
        video_rows=v0,
        audio_rows=a0,
        packed=packed,
        video_sigmas=[1.0, 0.0],
        audio_sigmas=[1.0, 0.0],
        rank_slice=(0, packed["seq_len"]),
        prompt_embeds=torch.randn(2, 8, dtype=torch.bfloat16),
        refined_prompt_embeds_length=2,
        rope_cache=torch.zeros(packed["seq_len"], 96, dtype=torch.bfloat16),
    )
    assert captured["x"].shape == (1, packed["seq_len"], 96)
    assert captured["audio_x"].shape == (1, packed["seq_len"], 32)
    assert (
        captured["packed_seq_params"]["cu_seqlens_q"].tolist()
        == packed["cu_seqlens"].tolist()
    )
    assert captured["skip_mask_out_condition"] is True


def test_denoise_loop_rejects_mismatched_schedules(packed):
    v0, a0 = build_initial_latents(seed=5, **_latent_kwargs())
    with pytest.raises(ValueError, match="same length"):
        run_denoise_loop(
            dit=lambda **_: (v0, a0),
            video_rows=v0,
            audio_rows=a0,
            packed=packed,
            video_sigmas=[1.0, 0.5, 0.0],
            audio_sigmas=[1.0, 0.0],
            rank_slice=(0, packed["seq_len"]),
            prompt_embeds=torch.randn(2, 8, dtype=torch.bfloat16),
            refined_prompt_embeds_length=2,
            rope_cache=torch.zeros(packed["seq_len"], 96, dtype=torch.bfloat16),
        )
