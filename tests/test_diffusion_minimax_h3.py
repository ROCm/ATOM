# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Structural tests for the ATOM MiniMax-H3 DiT.

Runs a tiny model on CPU so the packed-sequence plumbing, the embedding
scatter, and the output row selection are exercised without weights or a GPU.
Numerical parity against the real checkpoint is a separate GPU test that diffs
against /md0/dit_golden.
"""

import pytest
import torch

from atom.diffusion.configs.minimax_h3 import MiniMaxH3DiTArchConfig
from atom.diffusion.models.dits.minimax_h3 import (
    MiniMaxH3DiTModel,
    reorder_grouped_qkv_to_qkv,
)

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


def test_arch_rejects_inconsistent_adaln():
    with pytest.raises(ValueError, match="adaln_out_features"):
        MiniMaxH3DiTArchConfig(adaln_out_features=1234)


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


def test_grouped_qkv_reorder_rejects_bad_shape():
    with pytest.raises(ValueError, match="output dim"):
        reorder_grouped_qkv_to_qkv(
            torch.zeros(7, 2), num_query_groups=2, heads_per_group=3, head_dim=4
        )


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


def test_mismatched_update_mask_is_rejected():
    arch = tiny_arch()
    model = MiniMaxH3DiTModel(arch).eval()
    kwargs = make_inputs(arch)
    kwargs["skip_mask_out_condition"] = False
    kwargs["update_mask"] = torch.ones(N_IMG + 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="update_mask length"):
        model(**kwargs)


def test_missing_embedding_layout_is_rejected():
    arch = tiny_arch()
    model = MiniMaxH3DiTModel(arch).eval()
    kwargs = make_inputs(arch)
    del kwargs["local_embedding_layout"]
    with pytest.raises(KeyError, match="local_embedding_layout"):
        model(**kwargs)


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


def test_rope_rejects_wrong_position_rank():
    arch = tiny_arch()
    model = MiniMaxH3DiTModel(arch).eval()
    with pytest.raises(ValueError, match=r"\[1, S, 3\]"):
        model.rope(torch.rand(S, 3))


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
    from atom.diffusion.models.dits.minimax_h3 import MiniMaxH3Rope

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
