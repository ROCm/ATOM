# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-H3 networks: the DiT's blocks and fused kernels, the denoise
loop over them, VAE decode, and text-encoder parity with the reference.
"""

import pytest
import torch

from atom.diffusion.models.minimax_h3.arch import MiniMaxH3DiTArchConfig
from atom.diffusion.models.minimax_h3.components import (
    crop_to_canvas,
    denormalize_latents,
    denormalize_pixels,
)
from atom.diffusion.models.minimax_h3.conditioning import (
    QWEN_TEMPORAL_PATCH,
    REFERENCE_IMAGE_MULTIPLE,
    REFERENCE_IMAGE_SHORT_EDGE,
    audio_vae_determinism,
    resize_reference_image,
    resolve_reference_image_shape,
    sample_reference_video_frames,
)
from atom.diffusion.models.minimax_h3.denoise import (
    build_timestep_conditioning,
    run_denoise_loop,
)
from atom.diffusion.models.minimax_h3.dit import (
    MiniMaxH3DiTModel,
    reorder_grouped_qkv_to_qkv,
)
from atom.diffusion.models.minimax_h3.layout import (
    TAG_AUDIO,
    TAG_TEXT,
    TAG_VIDEO,
    build_initial_latents,
    build_packed_sequence,
    patchify_video_latent,
    scatter_rows_into_packed,
    unpack_audio_tokens,
    unpatchify_video_tokens,
)
from atom.diffusion.mux import frames_to_uint8, write_video_with_audio

# ==========================================================================
# DIT BLOCKS AND FUSED KERNELS
# ==========================================================================
#
# Structural tests for the ATOM MiniMax-H3 DiT.
#
# Runs a tiny model on CPU so the packed-sequence plumbing, the embedding
# scatter, and the output row selection are exercised without weights or a GPU.
# Numerical parity against the real checkpoint is a separate GPU test that diffs
# against /md0/dit_golden.

HIDDEN = 64
HEADS = 4
HEAD_DIM = 32
S = 16
N_TEXT = 2
N_AUDIO = 3
N_IMG = S - N_TEXT - N_AUDIO  # 11


def tiny_arch() -> MiniMaxH3DiTArchConfig:
    return MiniMaxH3DiTArchConfig(
        num_layers=2,
        token_refiner_num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        attention_head_dim=HEAD_DIM,
        ffn_hidden_size=128,
        latents_dim=4,
        audio_latents_dim=8,
        text_dim=32,
        timestep_input_dim=16,
        time_embed_hidden_size=HIDDEN,
        time_embed_dim=32,
        adaln_out_features=18 * HIDDEN,
        final_adaln_out_features=2 * HIDDEN,
        rope_inv_freq_len=4,  # rope_dim 24 <= head_dim 32
    )


def make_inputs(arch: MiniMaxH3DiTArchConfig, *, refined: bool = True) -> dict:
    """Synthetic inputs matching the measured serving contract."""
    audio_ids = torch.arange(N_TEXT, N_TEXT + N_AUDIO)
    img_ids = torch.arange(N_TEXT + N_AUDIO, S)

    # combined index = token_tag + modality_num * inverse_index; one timestep,
    # so values live in [0, 3).
    combined = torch.zeros(S, dtype=torch.long)
    combined[audio_ids] = 1
    combined[img_ids] = 2

    prompt = (
        torch.randn(N_TEXT, HIDDEN, dtype=torch.bfloat16)
        if refined
        else torch.randn(N_TEXT, arch.text_dim, dtype=torch.bfloat16)
    )

    return {
        "x": torch.randn(1, S, arch.video_patch_dim),
        "audio_x": torch.randn(1, S, arch.audio_latents_dim),
        "img_position_ids": torch.rand(1, S, 3),
        "unique_timesteps": torch.tensor([0.5]),
        "inverse_indices": torch.zeros(S, dtype=torch.long),
        "prompt_embeds": prompt,
        **({"refined_prompt_embeds_length": N_TEXT} if refined else {}),
        "packed_seq_params": {
            "cu_seqlens_q": torch.tensor([0, S], dtype=torch.int32),
            "max_seqlen_q": S,
        },
        "refiner_packed_seq_params": {
            "cu_seqlens_q": torch.tensor([0, N_TEXT], dtype=torch.int32),
            "max_seqlen_q": N_TEXT,
        },
        "local_embedding_layout": {
            "text_source_start": 0,
            "text_source_stop": N_TEXT,
            "img_global_ids": img_ids,
            "img_row_ids": img_ids,
            "audio_global_ids": audio_ids,
            "audio_row_ids": audio_ids,
        },
        "block_combined_indices": combined,
        "img_pos_for_infer_output_info": {"position_ids": img_ids},
        "audio_pos_info": {"position_ids": audio_ids},
        "img_pos_info": {"position_ids": img_ids},
        "text_pos_info": {"position_ids": torch.arange(N_TEXT)},
        "update_mask": torch.ones(N_IMG, dtype=torch.bool),
        "skip_mask_out_condition": True,
    }


# ── config ────────────────────────────────────────────────────────────────


def test_arch_derived_dims():
    a = MiniMaxH3DiTArchConfig()
    assert a.inner_dim == 56 * 128
    assert a.rope_dim == 96
    assert a.video_patch_dim == 24 * 1 * 2 * 2  # 96, matches the captured x width


def test_validate_ulysses_rejects_indivisible_heads():
    a = MiniMaxH3DiTArchConfig()
    a.validate_ulysses(8)  # 56 / 8 = 7
    with pytest.raises(ValueError, match="divisible"):
        a.validate_ulysses(5)


# ── grouped QKV reorder ───────────────────────────────────────────────────


def test_grouped_qkv_reorder_moves_the_right_rows():
    groups, heads_per_group, head_dim = 2, 3, 4
    per_group = (heads_per_group + 2) * head_dim
    w = torch.arange(groups * per_group * 2, dtype=torch.float32).reshape(
        groups * per_group, 2
    )
    out = reorder_grouped_qkv_to_qkv(
        w,
        num_query_groups=groups,
        heads_per_group=heads_per_group,
        head_dim=head_dim,
    )
    assert out.shape == w.shape
    q_rows = groups * heads_per_group * head_dim
    # Group 0's Q block must land at the front of the Q section, and group 1's
    # K head must follow group 0's inside the K section.
    torch.testing.assert_close(
        out[: heads_per_group * head_dim], w[: heads_per_group * head_dim]
    )
    k0 = w[heads_per_group * head_dim : (heads_per_group + 1) * head_dim]
    torch.testing.assert_close(out[q_rows : q_rows + head_dim], k0)


# ── model ─────────────────────────────────────────────────────────────────


def test_forward_shapes_with_prerefined_text():
    arch = tiny_arch()
    model = MiniMaxH3DiTModel(arch).eval()
    with torch.no_grad():
        video, audio = model(**make_inputs(arch))
    assert video.shape == (N_IMG, arch.video_patch_dim)
    assert audio.shape == (N_AUDIO, arch.audio_latents_dim)
    assert video.dtype is torch.float32 and audio.dtype is torch.float32
    assert torch.isfinite(video).all() and torch.isfinite(audio).all()


def test_forward_runs_the_token_refiner_when_text_is_raw():
    arch = tiny_arch()
    model = MiniMaxH3DiTModel(arch).eval()
    with torch.no_grad():
        video, audio = model(**make_inputs(arch, refined=False))
    assert video.shape == (N_IMG, arch.video_patch_dim)
    assert audio.shape == (N_AUDIO, arch.audio_latents_dim)


def test_update_mask_zeroes_condition_rows():
    arch = tiny_arch()
    model = MiniMaxH3DiTModel(arch).eval()
    kwargs = make_inputs(arch)
    kwargs["skip_mask_out_condition"] = False
    mask = torch.ones(N_IMG, dtype=torch.bool)
    mask[:2] = False
    kwargs["update_mask"] = mask
    with torch.no_grad():
        video, _ = model(**kwargs)
    assert torch.count_nonzero(video[:2]) == 0
    assert torch.count_nonzero(video[2:]) > 0


def test_rope_cache_is_built_when_absent_and_matches_supplied():
    """An omitted rope_cache must reproduce what build_rope_cache would give."""
    arch = tiny_arch()
    torch.manual_seed(0)
    model = MiniMaxH3DiTModel(arch).eval()
    with torch.no_grad():
        model.rope.inv_freq.copy_(torch.rand(arch.rope_inv_freq_len))

    kwargs = make_inputs(arch)
    with torch.no_grad():
        built = model.build_rope_cache(kwargs["img_position_ids"], 0, S)
        a_video, a_audio = model(**kwargs)
        kwargs["rope_cache"] = (built, torch.arange(S))
        b_video, b_audio = model(**kwargs)

    assert built.shape == (S, arch.rope_dim)
    torch.testing.assert_close(a_video, b_video)
    torch.testing.assert_close(a_audio, b_audio)


def test_sequence_must_divide_across_ulysses_world():
    arch = tiny_arch()
    model = MiniMaxH3DiTModel(arch).eval()
    model.ulysses._world_size = 5  # 16 % 5 != 0
    with pytest.raises(ValueError, match="must divide across"):
        model(**make_inputs(arch))


def test_rope_inv_freq_is_initialised_not_uninitialised_memory():
    """The checkpoint always supplies this buffer, but a model built without
    weights must still be well-defined: torch.empty here produced NaN
    velocities on roughly one run in fifteen."""
    from atom.diffusion.models.minimax_h3.dit import MiniMaxH3Rope

    rope = MiniMaxH3Rope(16)
    assert bool(torch.isfinite(rope.inv_freq).all())
    assert float(rope.inv_freq[0]) == 1.0
    # Standard 1 / theta^(i/n) at theta = 10000, matching the checkpoint.
    expected = 1.0 / (10000.0 ** (torch.arange(16, dtype=torch.float32) / 16))
    assert torch.allclose(rope.inv_freq, expected)


def test_dit_without_loaded_weights_produces_finite_output():
    arch = tiny_arch()
    model = MiniMaxH3DiTModel(arch).eval()
    with torch.no_grad():
        video, audio = model(**make_inputs(arch))
    assert bool(torch.isfinite(video).all())
    assert bool(torch.isfinite(audio).all())


# ==========================================================================
# DENOISE LOOP
# ==========================================================================
#
# Tests for MiniMax-H3 latent prep, token packing and the denoise loop.
#
# CPU only; the DiT is replaced by a stub so the loop's plumbing (row<->packed
# scatter, per-step timestep conditioning, sampler wiring) is what's under test.

# Small but structurally faithful: patch (1,2,2), 24 video / 32 audio channels.
SMALL = {"text_len": 2, "latent_t": 3, "latent_h": 4, "latent_w": 6, "audio_t": 5}


@pytest.fixture(scope="module")
def packed():
    return build_packed_sequence(**SMALL)


# ── token packing ─────────────────────────────────────────────────────────


def test_patchify_unpatchify_roundtrip():
    latent = torch.randn(1, 24, 3, 4, 6)
    rows = patchify_video_latent(latent, patch_size=(1, 2, 2))
    assert rows.shape == (3 * 2 * 3, 24 * 1 * 2 * 2)
    back = unpatchify_video_tokens(
        rows, latent_shape=(3, 2, 3, 24), patch_size=(1, 2, 2)
    )
    torch.testing.assert_close(back, latent)


def test_unpack_audio_tokens_is_channel_major():
    rows = torch.arange(10 * 32, dtype=torch.float32).reshape(10, 32)
    out = unpack_audio_tokens(rows, audio_t=10, audio_channel=2)
    assert out.shape == (2, 32, 5)
    # First channel is the first half of the rows.
    torch.testing.assert_close(out[0].permute(1, 0), rows[:5])


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


# ==========================================================================
# VAE DECODE
# ==========================================================================
#
# Tests for MiniMax-H3 VAE decode helpers and the MP4 mux.
#
# The VAE modules themselves live in the checkpoint and need the weights, so
# these cover the pure transforms around them: latent de-normalisation, canvas
# cropping, frame quantisation and the container contract.

av = pytest.importorskip("av", reason="PyAV needed for the mux tests")


# ── latent de-normalisation ───────────────────────────────────────────────


def test_denormalize_applies_per_channel():
    latents = torch.ones(1, 3, 2, 2)
    out = denormalize_latents(latents, mean=[1.0, 2.0, 3.0], std=[10.0, 20.0, 30.0])
    assert out[0, 0].unique().tolist() == [11.0]
    assert out[0, 1].unique().tolist() == [22.0]
    assert out[0, 2].unique().tolist() == [33.0]


def test_denormalize_does_not_mutate_input():
    latents = torch.ones(1, 2, 2)
    before = latents.clone()
    denormalize_latents(latents, mean=[0.0, 1.0], std=[2.0, 3.0])
    torch.testing.assert_close(latents, before)


# ── canvas crop ───────────────────────────────────────────────────────────


def test_crop_removes_vae_tile_padding_from_bottom_right():
    frames = torch.arange(1 * 3 * 2 * 8 * 10, dtype=torch.float32).reshape(
        1, 3, 2, 8, 10
    )
    out = crop_to_canvas(frames, height=6, width=7)
    assert out.shape == (1, 3, 2, 6, 7)
    # Cropping is from the origin: the top-left pixel must be preserved.
    torch.testing.assert_close(out[0, 0, 0, 0, 0], frames[0, 0, 0, 0, 0])
    torch.testing.assert_close(out[0, :, :, :6, :7], frames[0, :, :, :6, :7])


def test_crop_is_a_no_op_when_already_exact():
    frames = torch.randn(1, 3, 2, 6, 7)
    assert crop_to_canvas(frames, height=6, width=7) is frames


# ── frame quantisation ────────────────────────────────────────────────────


def test_frames_to_uint8_maps_the_unit_range():
    """Input is [0, 1] -- what denormalize_pixels produces, per the reference's
    transform_rev(x).clamp(0, 1)."""
    frames = torch.tensor([0.0, 0.5, 1.0]).view(1, 1, 3, 1, 1).repeat(1, 3, 1, 1, 1)
    out = frames_to_uint8(frames)
    assert out.shape == (3, 1, 1, 3)
    assert out.dtype.name == "uint8"
    assert [int(out[i, 0, 0, 0]) for i in range(3)] == [0, 128, 255]


def test_frames_to_uint8_clamps_out_of_range():
    assert int(frames_to_uint8(torch.full((1, 3, 1, 2, 2), 5.0)).max()) == 255
    assert int(frames_to_uint8(torch.full((1, 3, 1, 2, 2), -5.0)).min()) == 0


# ── mux ───────────────────────────────────────────────────────────────────


def _probe(path):
    with av.open(path) as c:
        return {
            s.type: {
                "codec": s.codec_context.name,
                "rate": getattr(s.codec_context, "sample_rate", None),
                "channels": getattr(s.codec_context, "channels", None),
            }
            for s in c.streams
        }


def test_mux_writes_h264_plus_aac_stereo(tmp_path):
    """The H3 output contract: H.264 24 fps + one AAC stereo stream."""
    frames = torch.zeros(1, 3, 8, 64, 64)
    audio = torch.zeros(2, 32000 // 3)
    out = write_video_with_audio(str(tmp_path / "clip.mp4"), frames, audio)

    streams = _probe(out)
    assert streams["video"]["codec"] == "h264"
    assert "audio" in streams, "H3 output must carry an audio track"
    assert streams["audio"]["codec"] == "aac"
    assert streams["audio"]["rate"] == 32000
    assert streams["audio"]["channels"] == 2


def test_mux_video_only_is_allowed_but_has_no_audio_stream(tmp_path):
    frames = torch.zeros(1, 3, 4, 64, 64)
    out = write_video_with_audio(str(tmp_path / "silent.mp4"), frames, None)
    streams = _probe(out)
    assert streams["video"]["codec"] == "h264"
    assert "audio" not in streams


def test_mux_preserves_frame_count(tmp_path):
    frames = torch.zeros(1, 3, 12, 64, 64)
    out = write_video_with_audio(str(tmp_path / "count.mp4"), frames, None)
    with av.open(out) as c:
        decoded = sum(1 for _ in c.decode(video=0))
    assert decoded == 12


# ── pixel de-normalisation ────────────────────────────────────────────────


class _FakeVAE:
    """Stands in for the VAE's stored imagenet transform_rev."""

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def transform_rev(self, x):
        mean = torch.tensor(self.MEAN).view(1, 3, 1, 1)
        std = torch.tensor(self.STD).view(1, 3, 1, 1)
        return x * std + mean


def test_denormalize_pixels_inverts_imagenet_normalisation():
    """A normalized mid-grey must come back as the imagenet mean per channel."""
    frames = torch.zeros(1, 3, 2, 4, 5)
    out = denormalize_pixels(frames, _FakeVAE())
    assert out.shape == frames.shape
    for c, m in enumerate(_FakeVAE.MEAN):
        assert out[0, c].unique().tolist() == pytest.approx([m], abs=1e-6)


def test_denormalize_pixels_is_per_channel_not_global():
    """The whole point: the correction differs per channel, so a global affine
    fix cannot reproduce it."""
    frames = torch.ones(1, 3, 1, 2, 2)
    out = denormalize_pixels(frames, _FakeVAE())
    vals = [out[0, c].mean().item() for c in range(3)]
    assert len({round(v, 6) for v in vals}) == 3


def test_denormalize_pixels_clamps_to_unit_range():
    frames = torch.full((1, 3, 1, 2, 2), 100.0)
    out = denormalize_pixels(frames, _FakeVAE())
    assert float(out.max()) <= 1.0
    assert float(out.min()) >= 0.0


# ==========================================================================
# TEXT-ENCODER REFERENCE PARITY
# ==========================================================================
#
# ref2va reference-material geometry and sampling rules.
#
# The encode itself needs the real VAEs, so these cover the deterministic parts:
# shape resolution, the Qwen frame/timestamp sampling, and the determinism
# context's restore behaviour.

np = pytest.importorskip("numpy")


@pytest.mark.parametrize(
    ("width", "height"),
    [(1344, 768), (1333, 777), (320, 240), (1600, 800)],
)
def test_reference_shape_invariants(width, height):
    """A reference always reaches the short edge, on grid, aspect preserved.

    Unlike the target canvas it is upscaled to get there -- 320x240 comes out
    larger than it went in.
    """
    shape = resolve_reference_image_shape(width=width, height=height)
    assert shape["short_edge"] == REFERENCE_IMAGE_SHORT_EDGE
    assert min(shape["width"], shape["height"]) == REFERENCE_IMAGE_SHORT_EDGE
    assert shape["width"] % REFERENCE_IMAGE_MULTIPLE == 0
    assert shape["height"] % REFERENCE_IMAGE_MULTIPLE == 0
    assert shape["width"] / shape["height"] == pytest.approx(width / height, rel=0.02)


def test_extreme_ratios_are_rejected():
    with pytest.raises(ValueError, match="1:4 to 4:1"):
        resolve_reference_image_shape(width=5000, height=1000)


@pytest.mark.parametrize("width,height", [(0, 100), (100, -1), (float("inf"), 100)])
def test_degenerate_dimensions_are_rejected(width, height):
    with pytest.raises(ValueError, match="positive finite"):
        resolve_reference_image_shape(width=width, height=height)


def test_resize_is_a_no_op_at_the_target_size():
    """No resample pass when the source already is the target -- LANCZOS on an
    identity resize is not bit-exact."""
    pil = pytest.importorskip("PIL.Image")
    image = pil.new("RGB", (64, 32), color=(11, 22, 33))
    out = resize_reference_image(image, target_width=64, target_height=32)
    assert out.size == image.size
    assert np.array_equal(np.asarray(out), np.asarray(image))


def test_qwen_sampling_takes_every_twelfth_frame():
    """24 FPS video, 2 FPS Qwen view."""
    frames = np.zeros((25, 4, 4, 3), dtype=np.uint8)
    frames[:, 0, 0, 0] = np.arange(25, dtype=np.uint8)
    out = sample_reference_video_frames(frames)
    assert out["frames"].shape[0] == 3
    assert out["frames"][:, 0, 0, 0].tolist() == [0, 12, 24]


def test_block_timestamps_pair_frames_and_pad_with_the_last():
    """An odd sample count pads with the final frame, so the trailing block's
    timestamp is that frame's own time, not an extrapolation."""
    frames = np.zeros((25, 4, 4, 3), dtype=np.uint8)  # -> 3 sampled at 2 FPS
    out = sample_reference_video_frames(frames)
    assert len(out["block_timestamps"]) == 2
    assert out["block_timestamps"] == pytest.approx([0.25, 1.0])


def test_even_sample_counts_need_no_padding():
    frames = np.zeros((13, 4, 4, 3), dtype=np.uint8)  # -> 2 sampled
    out = sample_reference_video_frames(frames)
    assert len(out["block_timestamps"]) == 1
    assert out["block_timestamps"] == pytest.approx([0.25])


def test_block_count_is_the_padded_sample_count_over_the_patch():
    frames = np.zeros((61, 4, 4, 3), dtype=np.uint8)  # -> 6 sampled
    out = sample_reference_video_frames(frames)
    sampled = int(out["frames"].shape[0])
    expected = (sampled + QWEN_TEMPORAL_PATCH - 1) // QWEN_TEMPORAL_PATCH
    assert len(out["block_timestamps"]) == expected


def test_empty_frames_are_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        sample_reference_video_frames(np.zeros((0, 4, 4, 3), dtype=np.uint8))


def test_determinism_context_restores_the_backend_flags():
    before = (
        torch.backends.cudnn.enabled,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    with audio_vae_determinism():
        assert torch.backends.cudnn.enabled is False
        assert torch.backends.cudnn.deterministic is True
    after = (
        torch.backends.cudnn.enabled,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    assert before == after


def test_determinism_context_is_reentrant():
    """A caller may wrap a whole multi-reference loop; the inner uses must not
    restore the flags early."""
    with audio_vae_determinism():
        with audio_vae_determinism():
            assert torch.backends.cudnn.enabled is False
        assert torch.backends.cudnn.enabled is False
    assert audio_vae_determinism._depth == 0
