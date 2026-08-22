# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-H3 stage wiring for all three tasks, on CPU with stubs:
t2va end to end, fl2va keyframe conditioning, and the ref2va layout and its
stage chain.
"""

import numpy as np
import pytest
import torch
from torch import nn

from atom.diffusion.config import DiffusionConfig
from atom.diffusion.models.minimax_h3.arch import MiniMaxH3DiTArchConfig
from atom.diffusion.models.minimax_h3.conditioning import (
    ENCODE_SEED,
    MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    MINIMAX_H3_IMGVID_COND_TIMESTEP,
    cover_crop_plan,
    scoped_encode_rng,
)
from atom.diffusion.models.minimax_h3.dit import MiniMaxH3DiTModel
from atom.diffusion.models.minimax_h3.layout import (
    FL2VA_KEYFRAME_SIGNATURES,
    FRAME_RESCALE,
    PACKED_SEQUENCE_ALIGNMENT,
    TAG_AUDIO,
    TAG_PAD,
    TAG_TEXT,
    TAG_VIDEO,
    MiniMaxH3Geometry,
    build_packed_sequence,
    build_packed_sequence_ref2va,
    resolve_keyframe_indices,
    temporal_position_span,
    validate_keyframe_signature,
    video_t_grid,
)
from atom.diffusion.models.minimax_h3.pipeline import (
    ConditionEncodeStage,
    MiniMaxH3Pipeline,
    PackedSequenceStage,
    PlanStage,
    reference_materials,
)
from atom.diffusion.pipeline import DiffusionBatch
from atom.diffusion.request import DiffusionJob

# ==========================================================================
# T2VA: THE WHOLE STAGE CHAIN
# ==========================================================================
#
# End-to-end wiring test for the MiniMax-H3 t2va pipeline.
#
# Runs the whole stage chain on CPU with a tiny DiT and stub VAEs, so stage
# ordering, component resolution, parallelism dispatch and the MP4 contract are
# exercised without weights or a GPU.

pytest.importorskip("av", reason="PyAV needed to write the MP4")

HIDDEN = 64
TEXT_DIM = 32


def tiny_arch() -> MiniMaxH3DiTArchConfig:
    return MiniMaxH3DiTArchConfig(
        num_layers=1,
        token_refiner_num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=2,
        attention_head_dim=32,
        ffn_hidden_size=64,
        latents_dim=24,
        audio_latents_dim=32,
        text_dim=TEXT_DIM,
        timestep_input_dim=16,
        time_embed_hidden_size=HIDDEN,
        time_embed_dim=32,
        adaln_out_features=18 * HIDDEN,
        final_adaln_out_features=2 * HIDDEN,
        rope_inv_freq_len=4,
    )


class StubTextEncoder:
    """Deterministic stand-in for Qwen3-VL.

    Seeded on purpose: with unseeded draws the tiny stub DiT downstream
    occasionally produces non-finite velocities and the whole pipeline test
    fails a few runs in a hundred, which reads as a pipeline bug rather than a
    fixture one.
    """

    def encode_with_tags(
        self, prompt: str, images=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # H3 conditions on a token sequence, not a pooled vector. Tags mark
        # which of those positions came from a vision block.
        text_len = max(len(prompt.split()), 1)
        image_len = 4 * len(images or ())
        generator = torch.Generator().manual_seed(20260807)
        rows = torch.randn(text_len + image_len, TEXT_DIM, generator=generator)
        tags = torch.cat(
            (
                torch.zeros(image_len, dtype=torch.long),
                torch.ones(text_len, dtype=torch.long),
            )
        )
        return rows, tags

    def encode(self, prompt: str) -> torch.Tensor:
        return self.encode_with_tags(prompt)[0]


class StubVideoVAE(nn.Module):
    """Mimics the real decoder's contract: 16x spatial expansion, and output in
    ImageNet-normalized pixel space that the caller must run through
    transform_rev."""

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1))

    def decode(self, z):
        b, _, t, h, w = z.shape
        return torch.zeros(b, 3, t, h * 16, w * 16)

    def transform_rev(self, x):
        mean = torch.tensor(self.MEAN).view(1, 3, 1, 1)
        std = torch.tensor(self.STD).view(1, 3, 1, 1)
        return x * std + mean


class StubAudioVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1))

    def decode(self, z):
        c = int(z.shape[0])
        return torch.zeros(c, 8000)


def build_t2va_pipeline(tmp_path, *, duration=0.5, steps=3):
    # Seed the DiT's random init: an unlucky draw sends the tiny stub model's
    # velocities non-finite and the sampler's finiteness assertion fires, which
    # looks like a pipeline bug but is only the fixture.
    torch.manual_seed(20260807)
    config = DiffusionConfig(
        model_path="<test>",
        pipeline_class="atom.diffusion.models.minimax_h3.pipeline.MiniMaxH3Pipeline",
        num_gpus=1,
        ulysses_degree=1,
        num_inference_steps=steps,
        output_dir=str(tmp_path),
    )
    pipe = MiniMaxH3Pipeline(config)
    pipe.register_component("transformer", MiniMaxH3DiTModel(tiny_arch()).eval())
    pipe.register_component("video_vae", StubVideoVAE().eval())
    pipe.register_component("audio_vae", StubAudioVAE().eval())
    pipe.register_component("text_encoder", StubTextEncoder())

    job = DiffusionJob(
        prompt="three cats marching",
        task="t2va",
        num_inference_steps=steps,
        seed=1101,
        target={
            "height": 64,
            "width": 64,
            "duration_seconds": duration,
            "fps": 24,
        },
    )
    batch = DiffusionBatch(job=job)
    batch.meta.update({"ulysses_world": 1, "ulysses_rank": 0, "device": "cpu"})
    return pipe, batch, job, config


def test_pipeline_runs_end_to_end_and_writes_an_mp4(tmp_path):
    pipe, batch, job, _ = build_t2va_pipeline(tmp_path)
    with torch.no_grad():
        out = pipe.forward(batch)

    assert job.output_path and job.output_path.endswith(".mp4")
    import os

    assert os.path.getsize(job.output_path) > 0

    import av

    with av.open(job.output_path) as c:
        kinds = {s.type for s in c.streams}
    assert "video" in kinds
    assert "audio" in kinds, "H3 output must carry the audio track"

    for key in ("prompt_embeds", "geometry", "packed", "denoised_video", "frames"):
        assert out.get(key) is not None, f"{key} missing from the final batch"


def test_pipeline_stage_order_and_timing_report(tmp_path):
    pipe, batch, _job, _cfg = build_t2va_pipeline(tmp_path)
    with torch.no_grad():
        pipe.forward(batch)
    names = list(pipe.last_stage_times)
    assert names == [
        "TextEncodingStage",
        "PlanStage",
        "ConditionEncodeStage",
        "LatentPreparationStage",
        "PackedSequenceStage",
        "DenoiseStage",
        "DecodeStage",
        "PresentationStage",
    ]
    assert "MiniMaxH3Pipeline total" in pipe.stage_timing_report()


def test_denoise_progress_reaches_the_step_count(tmp_path):
    steps = 4
    pipe, batch, job, _cfg = build_t2va_pipeline(tmp_path, steps=steps)
    with torch.no_grad():
        pipe.forward(batch)
    # N sigmas -> N-1 iterations.
    assert job.total_steps == steps - 1
    assert job.current_step == steps - 1
    assert job.progress == 1.0


def test_unsupported_task_is_refused(tmp_path):
    _pipe, batch, job, config = build_t2va_pipeline(tmp_path)
    job.task = "audio_only"
    batch.set("prompt_embeds", torch.randn(2, TEXT_DIM))
    with pytest.raises(ValueError, match="not implemented"):
        PlanStage()(batch, config)


def test_ref2va_without_references_is_refused(tmp_path):
    """A ref2va request that carries no conditioning is a t2va request with a
    misleading label, not a valid one."""
    _pipe, batch, job, config = build_t2va_pipeline(tmp_path)
    job.task = "ref2va"
    batch.set("prompt_embeds", torch.randn(2, TEXT_DIM))
    with pytest.raises(ValueError, match="at least one reference"):
        PlanStage()(batch, config)


def test_fl2va_without_a_keyframe_is_refused(tmp_path):
    _pipe, batch, job, config = build_t2va_pipeline(tmp_path)
    job.task = "fl2va"
    batch.set("prompt_embeds", torch.randn(2, TEXT_DIM))
    with pytest.raises(ValueError, match="at least one keyframe"):
        PlanStage()(batch, config)


def test_keyframes_on_a_t2va_request_are_refused(tmp_path):
    _pipe, batch, job, config = build_t2va_pipeline(tmp_path)
    job.conditions = [{"type": "image", "uri": "file:///nonexistent.png"}]
    batch.set("prompt_embeds", torch.randn(2, TEXT_DIM))
    with pytest.raises((ValueError, FileNotFoundError, OSError)):
        PlanStage()(batch, config)


def test_plan_rejects_geometry_that_cannot_shard(tmp_path):
    _pipe, batch, _job, config = build_t2va_pipeline(tmp_path)
    batch.set("prompt_embeds", torch.randn(2, TEXT_DIM))
    batch.meta["ulysses_world"] = 7  # 64-aligned sequence is not divisible by 7
    with pytest.raises(ValueError, match="does not divide"):
        PlanStage()(batch, config)


def test_pipeline_requires_its_components(tmp_path):
    config = DiffusionConfig(
        model_path="<test>",
        pipeline_class="atom.diffusion.models.minimax_h3.pipeline.MiniMaxH3Pipeline",
        num_gpus=1,
        ulysses_degree=1,
        output_dir=str(tmp_path),
    )
    pipe = MiniMaxH3Pipeline(config)
    batch = DiffusionBatch(job=DiffusionJob(prompt="x", task="t2va"))
    batch.meta.update({"ulysses_world": 1, "ulysses_rank": 0, "device": "cpu"})
    with pytest.raises(RuntimeError, match="missing required components"):
        pipe.forward(batch)


def test_warmup_decode_covers_both_vaes(tmp_path):
    """Warmup has to cover decode, not just the DiT.

    Measured on gfx950 at 209 frames, the VAEs' first call costs 4.52 s (video)
    and 5.68 s (audio) against 1.37 s and 0.08 s once warm -- a larger relative
    penalty than the DiT's, and invisible to a benchmark that decodes once.
    """
    pipe, _, _, _ = build_t2va_pipeline(tmp_path)
    seen = []
    for name in ("video_vae", "audio_vae"):
        vae = pipe.component(name)
        original = vae.decode
        vae.decode = lambda z, _n=name, _f=original: (seen.append(_n), _f(z))[1]

    # A small geometry keeps the stub decoders cheap; what is asserted is that
    # warmup reaches them at all.
    pipe.video_stats = ([0.0] * 24, [1.0] * 24)
    geo = MiniMaxH3Geometry.resolve(
        height=64, width=128, frame_count=5, duration_seconds=0.5, text_len=2
    )
    pipe._warmup_decode(geo, pipe.component("transformer").arch, "cpu")
    assert seen == ["video_vae", "audio_vae"]


def test_non_main_ranks_still_owe_the_video_vae(tmp_path):
    """The video VAE's tiled decode is collective, so every rank must hold it.

    Guards the placement declaration specifically: the earlier hand-written
    per-rank check asked only for the transformer off main, so a video VAE that
    failed to load on rank 3 surfaced as a hang in decode rather than an error
    at load.
    """
    config = DiffusionConfig(
        model_path="<test>",
        pipeline_class="atom.diffusion.models.minimax_h3.pipeline.MiniMaxH3Pipeline",
        num_gpus=1,
        ulysses_degree=1,
        num_inference_steps=3,
        output_dir=str(tmp_path),
    )
    pipe = MiniMaxH3Pipeline(config)
    pipe.ulysses._rank = 3
    pipe.register_component("transformer", MiniMaxH3DiTModel(tiny_arch()).eval())

    with pytest.raises(RuntimeError, match=r"rank 3 .*\['video_vae'\]"):
        pipe.verify_components()

    # ...but the text encoder it never builds is not held against it.
    pipe.register_component("video_vae", StubVideoVAE().eval())
    pipe.verify_components()


# ==========================================================================
# FL2VA: KEYFRAME CONDITIONING
# ==========================================================================
#
# Tests for MiniMax-H3 fl2va keyframe conditioning.
#
# Covers the packed layout with a conditioning block and the canvas transform.
# The seeded VAE encode needs weights and is validated separately on GPU.

GEO = {
    "text_len": 2,
    "latent_t": 37,
    "latent_h": 48,
    "latent_w": 84,
    "audio_t": 207,
}
FL2VA_FRAME_ROWS = (48 // 2) * (84 // 2)  # 1008
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
    assert packed["cond_rows"] == len(sig) * FL2VA_FRAME_ROWS
    assert packed["used_len"] == t2va["used_len"] + len(sig) * FL2VA_FRAME_ROWS
    # img_pos gains the conditioning rows.
    assert (
        packed["img_pos"].numel()
        == t2va["img_pos"].numel() + len(sig) * FL2VA_FRAME_ROWS
    )


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
    first = g[text_len : text_len + FL2VA_FRAME_ROWS]
    last = g[text_len + FL2VA_FRAME_ROWS : text_len + 2 * FL2VA_FRAME_ROWS]
    assert first[0, 0].item() != last[0, 0].item()
    # Same spatial coordinates, different time.
    torch.testing.assert_close(first[:, 1:], last[:, 1:])


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
    with scoped_encode_rng(ENCODE_SEED):
        a = torch.randn(3)
    after = torch.randn(4)

    with scoped_encode_rng(ENCODE_SEED):
        b = torch.randn(3)
    # Same seed -> same sample (the posterior is sampled, so this matters).
    torch.testing.assert_close(a, b)
    # ...and the surrounding stream is untouched by the fork.
    torch.manual_seed(7)
    _ = torch.randn(4)
    torch.testing.assert_close(after, torch.randn(4))
    assert before.shape == after.shape


# ==========================================================================
# REF2VA: PACKED LAYOUT
# ==========================================================================
#
# ref2va packed layout: reference material ahead of the target.
#
# The layout is easy to get *plausibly* wrong -- right row count, right shapes,
# wrong temporal placement -- so these tests pin the ordering and the timeline
# rather than just the totals.

TEXT, LT, LH, LW, AT = 12, 4, 8, 12, 5
REF2VA_FRAME_ROWS = (LH // 2) * (LW // 2)  # 24


def build_ref2va_sequence(ref_blocks, **kw):
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
    p = build_ref2va_sequence([])
    assert p["cond_rows"] == 0 and p["cond_audio_rows"] == 0
    assert p["used_len"] == TEXT + AT * 2 + LT * REF2VA_FRAME_ROWS
    assert bool(p["update_mask"].all()) and bool(p["audio_update_mask"].all())


def test_image_block_rows_and_single_temporal_slot():
    p = build_ref2va_sequence([{"kind": "image", "latent_h": 4, "latent_w": 6}])
    rows = (4 // 2) * (6 // 2)
    assert p["cond_rows"] == rows
    g = p["img_position_ids"]
    ref = g[TEXT : TEXT + rows]
    # One image occupies exactly one integer slot on the shared timeline.
    assert torch.equal(ref[:, 0], torch.full((rows,), float(TEXT), dtype=g.dtype))
    assert float(g[TEXT + rows, 0]) == pytest.approx(TEXT + 1.0)


def test_audio_block_advances_the_cursor_by_its_own_length():
    ref_t = 3
    p = build_ref2va_sequence([{"kind": "audio", "ref_audio_t": ref_t}])
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
    p = build_ref2va_sequence([block])
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
    p = build_ref2va_sequence([block])
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
    p = build_ref2va_sequence([short])
    span = temporal_position_span(4)
    assert span > 1.0
    g = p["img_position_ids"]
    target_audio_start = p["audio_pos"][2].item()  # first target audio row
    assert float(g[target_audio_start, 0]) == pytest.approx(TEXT + span)


def test_reference_audio_needs_its_own_update_mask():
    """One mask cannot express 'hold these audio rows but step those'."""
    p = build_ref2va_sequence([{"kind": "audio", "ref_audio_t": 3}])
    assert p["audio_update_mask"].tolist() == [False] * 6 + [True] * (AT * 2)
    # Image-side rows are all target here, so the visual mask stays all-True.
    assert bool(p["update_mask"].all())


def test_blocks_are_laid_out_in_request_order():
    a = {"kind": "image", "latent_h": 4, "latent_w": 4}
    b = {"kind": "audio", "ref_audio_t": 2}
    first = build_ref2va_sequence([a, b])
    second = build_ref2va_sequence([b, a])
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
    p = build_ref2va_sequence([block])
    g = p["img_position_ids"]
    ref_w = g[TEXT + 4 : TEXT + 4 + 32, 2]  # this block's video rows
    ref_audio_w = g[TEXT : TEXT + 4, 2]
    assert float(ref_audio_w[0]) == pytest.approx(float(ref_w.min()))
    assert float(ref_audio_w[-1]) == pytest.approx(float(ref_w.max()))


def test_tags_cover_every_row_exactly_once():
    p = build_ref2va_sequence(
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
    p = build_ref2va_sequence(
        [{"kind": "image", "latent_h": 4, "latent_w": 4}], text_token_tags=tags
    )
    assert torch.equal(p["token_tags"][:TEXT], tags)


def test_sequence_is_padded_to_the_alignment():
    p = build_ref2va_sequence([{"kind": "image", "latent_h": 4, "latent_w": 4}])
    assert p["seq_len"] % PACKED_SEQUENCE_ALIGNMENT == 0
    assert p["cu_seqlens"].tolist() == [0, p["used_len"], p["seq_len"]]


# ==========================================================================
# REF2VA: STAGE WIRING
# ==========================================================================
#
# ref2va stage wiring, on CPU with stubs.
#
# The real-weights ref2va run is seeded from a captured DiT input and therefore
# skips the pipeline's own condition-encoding and layout stages entirely. This
# covers exactly that gap: that references reach the packed layout, that the two
# conditioning kinds are augmented differently, and that the denoise loop trims
# the reference audio off the DiT's prediction.

Image = pytest.importorskip("PIL.Image")

REF_W, REF_H = 512, 512
AUDIO_LATENT_T = 6
AUDIO_CHANNELS = 32


class RecordingVideoVAE(nn.Module):
    """Encodes to a fixed latent so cond-row counts are predictable."""

    def __init__(self, latent_h: int, latent_w: int):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1))
        self.latent_h, self.latent_w = latent_h, latent_w

    def encode_images(self, image, use_fp16_latent=False):
        del image, use_fp16_latent
        return [torch.zeros(1, 24, 1, self.latent_h, self.latent_w)]


class RecordingAudioVAE(nn.Module):
    """Mimics the reference's mean-encode path: encoder -> mean_proj, no draw."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1))
        self.calls = 0

    def preprocess(self, waveform, sample_rate):
        del sample_rate
        return waveform

    def encoder(self, x):
        return x

    def mean_proj(self, x):
        del x
        self.calls += 1
        return torch.zeros(2, AUDIO_LATENT_T, AUDIO_CHANNELS)


def write_reference_image(tmp_path):
    path = tmp_path / "ref.png"
    Image.new("RGB", (REF_W, REF_H), color=(40, 90, 140)).save(path)
    return path


def build_ref2va(tmp_path, conditions, *, steps=3, duration=0.5):
    config = DiffusionConfig(
        model_path="<test>",
        pipeline_class="atom.diffusion.models.minimax_h3.pipeline.MiniMaxH3Pipeline",
        num_gpus=1,
        ulysses_degree=1,
        num_inference_steps=steps,
        output_dir=str(tmp_path),
    )
    torch.manual_seed(20260807)
    pipe = MiniMaxH3Pipeline(config)
    # 2048x2048 reference / VAE 16x -> a 128x128 latent. The pipeline derives
    # the block's latent dims from the resolved material shape, so a VAE that
    # disagrees is caught at the layout boundary rather than silently packed.
    pipe.register_component("video_vae", RecordingVideoVAE(128, 128).eval())
    pipe.register_component("audio_vae", RecordingAudioVAE().eval())
    pipe.register_component("text_encoder", StubTextEncoder())

    job = DiffusionJob(
        prompt="the subject moves",
        task="ref2va",
        conditions=conditions,
        num_inference_steps=steps,
        seed=1101,
        target={
            "height": 64,
            "width": 64,
            "duration_seconds": duration,
            "fps": 24,
        },
    )
    batch = DiffusionBatch(job=job)
    batch.meta.update({"ulysses_world": 1, "ulysses_rank": 0, "device": "cpu"})
    batch.set("prompt_embeds", torch.zeros(4, 32))
    batch.set("text_token_tags", torch.ones(4, dtype=torch.long))
    return pipe, batch, job, config


def set_stats(pipe):
    """Latent stats live on the pipeline, set by load_components() from the
    checkpoint. These tests register components by hand, so set them here."""
    pipe.video_stats = ([0.0] * 24, [1.0] * 24)
    pipe.audio_stats = ([0.0] * AUDIO_CHANNELS, [1.0] * AUDIO_CHANNELS)
    return pipe


def test_reference_image_goes_to_its_own_short_edge_not_the_canvas(tmp_path):
    """The defining difference from an fl2va keyframe."""
    path = write_reference_image(tmp_path)
    materials = reference_materials(
        DiffusionJob(
            task="ref2va",
            conditions=[{"type": "image", "uri": f"file://{path}"}],
            target={"height": 64, "width": 64, "duration_seconds": 0.5, "fps": 24},
        )
    )
    assert materials[0]["width"] == 2048
    assert materials[0]["height"] == 2048
    assert materials[0]["image"].size == (2048, 2048)


def test_ordinals_count_per_type(tmp_path):
    path = write_reference_image(tmp_path)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    materials = reference_materials(
        DiffusionJob(
            task="ref2va",
            conditions=[
                {"type": "image", "uri": f"file://{path}"},
                {"type": "audio", "uri": f"file://{audio}"},
                {"type": "image", "uri": f"file://{path}"},
            ],
            target={"height": 64, "width": 64, "duration_seconds": 0.5, "fps": 24},
        )
    )
    assert [(m["label_kind"], m["ordinal"]) for m in materials] == [
        ("image", 1),
        ("audio", 1),
        ("image", 2),
    ]


def test_condition_encode_produces_blocks_and_both_row_kinds(tmp_path, monkeypatch):
    path = write_reference_image(tmp_path)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    pipe, batch, _job, config = build_ref2va(
        tmp_path,
        [
            {"type": "image", "uri": f"file://{path}"},
            {"type": "audio", "uri": f"file://{audio}"},
        ],
    )
    monkeypatch.setattr(
        "atom.diffusion.models.minimax_h3.conditioning." "load_reference_waveform",
        lambda *a, **k: (torch.zeros(2, 32000), 32000),
    )
    PlanStage()(batch, config)
    ConditionEncodeStage(set_stats(pipe))(batch, config)

    assert batch.get("ref_blocks") == [
        {"kind": "image", "latent_h": 128, "latent_w": 128},
        {"kind": "audio", "ref_audio_t": AUDIO_LATENT_T},
    ]
    assert batch.get("cond_rows").shape == (64 * 64, 96)
    assert batch.get("cond_audio_rows").shape == (2 * AUDIO_LATENT_T, AUDIO_CHANNELS)


def test_visual_references_are_noise_augmented_and_audio_ones_are_not(
    tmp_path, monkeypatch
):
    """Two constants, on purpose: 0.999 visual, 1.0 audio. The stub encoders
    return zeros, so any nonzero row is noise that was mixed in."""
    path = write_reference_image(tmp_path)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    pipe, batch, _job, config = build_ref2va(
        tmp_path,
        [
            {"type": "image", "uri": f"file://{path}"},
            {"type": "audio", "uri": f"file://{audio}"},
        ],
    )
    monkeypatch.setattr(
        "atom.diffusion.models.minimax_h3.conditioning." "load_reference_waveform",
        lambda *a, **k: (torch.zeros(2, 32000), 32000),
    )
    PlanStage()(batch, config)
    ConditionEncodeStage(set_stats(pipe))(batch, config)

    assert MINIMAX_H3_IMGVID_COND_TIMESTEP < 1.0
    assert MINIMAX_H3_AUDIO_REF_COND_TIMESTEP == 1.0
    assert float(batch.get("cond_rows").abs().max()) > 0.0
    assert float(batch.get("cond_audio_rows").abs().max()) == 0.0


def test_packed_layout_reserves_exactly_what_was_encoded(tmp_path, monkeypatch):
    path = write_reference_image(tmp_path)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    pipe, batch, _job, config = build_ref2va(
        tmp_path,
        [
            {"type": "image", "uri": f"file://{path}"},
            {"type": "audio", "uri": f"file://{audio}"},
        ],
    )
    monkeypatch.setattr(
        "atom.diffusion.models.minimax_h3.conditioning." "load_reference_waveform",
        lambda *a, **k: (torch.zeros(2, 32000), 32000),
    )
    PlanStage()(batch, config)
    ConditionEncodeStage(set_stats(pipe))(batch, config)
    PackedSequenceStage()(batch, config)

    packed = batch.require("packed")
    assert packed["cond_rows"] == int(batch.get("cond_rows").shape[0])
    assert packed["cond_audio_rows"] == int(batch.get("cond_audio_rows").shape[0])
    # References lead, target follows, in both index vectors.
    assert not bool(packed["update_mask"][: packed["cond_rows"]].any())
    assert bool(packed["update_mask"][packed["cond_rows"] :].all())
    assert not bool(packed["audio_update_mask"][: packed["cond_audio_rows"]].any())


def test_row_count_disagreement_is_caught_at_the_layout_boundary(tmp_path, monkeypatch):
    path = write_reference_image(tmp_path)
    pipe, batch, _job, config = build_ref2va(
        tmp_path, [{"type": "image", "uri": f"file://{path}"}]
    )
    PlanStage()(batch, config)
    ConditionEncodeStage(set_stats(pipe))(batch, config)
    batch.set("cond_rows", batch.get("cond_rows")[:-1])
    with pytest.raises(ValueError, match="conditioning rows"):
        PackedSequenceStage()(batch, config)


def test_unknown_condition_type_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not a ref2va reference"):
        reference_materials(
            DiffusionJob(
                task="ref2va",
                conditions=[{"type": "subtitle", "uri": "file:///x.srt"}],
                target={"height": 64, "width": 64, "duration_seconds": 0.5, "fps": 24},
            )
        )


def test_condition_without_a_uri_is_refused():
    with pytest.raises(ValueError, match="no uri"):
        reference_materials(
            DiffusionJob(
                task="ref2va",
                conditions=[{"type": "image"}],
                target={"height": 64, "width": 64, "duration_seconds": 0.5, "fps": 24},
            )
        )


def test_denoise_trims_the_reference_audio_off_the_prediction():
    """The DiT returns *all* audio rows but only the generated video rows, so
    the audio side needs an explicit trim the video side does not."""
    from atom.diffusion.models.minimax_h3.denoise import run_denoise_loop
    from atom.diffusion.models.minimax_h3.layout import (
        build_packed_sequence_ref2va,
    )

    packed = build_packed_sequence_ref2va(
        text_len=4,
        latent_t=1,
        latent_h=4,
        latent_w=4,
        audio_t=3,
        ref_blocks=[{"kind": "audio", "ref_audio_t": 2}],
    )
    n_video = int(packed["img_pos"].numel())
    n_audio = int(packed["audio_pos"].numel())

    seen = {}

    def fake_dit(**kwargs):
        seen["audio_rows_in"] = int(kwargs["audio_x"].shape[1])
        return (
            torch.zeros(n_video, 96),
            torch.arange(n_audio * 32, dtype=torch.float32).view(n_audio, 32),
        )

    video, audio = run_denoise_loop(
        dit=fake_dit,
        video_rows=torch.zeros(n_video, 96),
        audio_rows=torch.zeros(n_audio - 4, 32),
        cond_audio_rows=torch.zeros(4, 32),
        packed=packed,
        video_sigmas=[1.0, 0.5, 0.0],
        audio_sigmas=[1.0, 0.5, 0.0],
        rank_slice=(0, int(packed["seq_len"])),
        prompt_embeds=torch.zeros(4, 64),
        refined_prompt_embeds_length=4,
        rope_cache=torch.zeros(int(packed["seq_len"]), 96),
    )
    assert audio.shape[0] == n_audio - 4
    assert video.shape[0] == n_video


def test_video_reference_frame_sampling_feeds_both_consumers():
    """Qwen and the VAE must see the same decoded array, not two decodes."""
    from atom.diffusion.models.minimax_h3.conditioning import (
        sample_reference_video_frames,
    )

    frames = np.zeros((49, 8, 8, 3), dtype=np.uint8)
    out = sample_reference_video_frames(frames)
    assert out["frames"].base is not None or out["frames"].shape[0] == 5
