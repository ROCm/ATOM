# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end wiring test for the MiniMax-H3 t2va pipeline.

Runs the whole stage chain on CPU with a tiny DiT and stub VAEs, so stage
ordering, component resolution, parallelism dispatch and the MP4 contract are
exercised without weights or a GPU.
"""

import pytest
import torch
from torch import nn

from atom.diffusion.config import ComponentConfig, DiffusionConfig
from atom.diffusion.configs.minimax_h3 import MiniMaxH3DiTArchConfig
from atom.diffusion.models.dits.minimax_h3 import MiniMaxH3DiTModel
from atom.diffusion.pipelines.minimax_h3 import MiniMaxH3Pipeline, PlanStage
from atom.diffusion.request import DiffusionJob
from atom.diffusion.stages.base import DiffusionBatch

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


def build(tmp_path, *, duration=0.5, steps=3):
    # Seed the DiT's random init: an unlucky draw sends the tiny stub model's
    # velocities non-finite and the sampler's finiteness assertion fires, which
    # looks like a pipeline bug but is only the fixture.
    torch.manual_seed(20260807)
    config = DiffusionConfig(
        model_path="<test>",
        pipeline_class="atom.diffusion.pipelines.minimax_h3.MiniMaxH3Pipeline",
        components=[
            ComponentConfig(name="transformer", class_path="torch.nn.Identity")
        ],
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
    pipe, batch, job, _ = build(tmp_path)
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
    pipe, batch, _job, _cfg = build(tmp_path)
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
    pipe, batch, job, _cfg = build(tmp_path, steps=steps)
    with torch.no_grad():
        pipe.forward(batch)
    # N sigmas -> N-1 iterations.
    assert job.total_steps == steps - 1
    assert job.current_step == steps - 1
    assert job.progress == 1.0


def test_unsupported_task_is_refused(tmp_path):
    _pipe, batch, job, config = build(tmp_path)
    job.task = "audio_only"
    batch.set("prompt_embeds", torch.randn(2, TEXT_DIM))
    with pytest.raises(ValueError, match="not implemented"):
        PlanStage()(batch, config)


def test_ref2va_without_references_is_refused(tmp_path):
    """A ref2va request that carries no conditioning is a t2va request with a
    misleading label, not a valid one."""
    _pipe, batch, job, config = build(tmp_path)
    job.task = "ref2va"
    batch.set("prompt_embeds", torch.randn(2, TEXT_DIM))
    with pytest.raises(ValueError, match="at least one reference"):
        PlanStage()(batch, config)


def test_fl2va_without_a_keyframe_is_refused(tmp_path):
    _pipe, batch, job, config = build(tmp_path)
    job.task = "fl2va"
    batch.set("prompt_embeds", torch.randn(2, TEXT_DIM))
    with pytest.raises(ValueError, match="at least one keyframe"):
        PlanStage()(batch, config)


def test_keyframes_on_a_t2va_request_are_refused(tmp_path):
    _pipe, batch, job, config = build(tmp_path)
    job.conditions = [{"type": "image", "uri": "file:///nonexistent.png"}]
    batch.set("prompt_embeds", torch.randn(2, TEXT_DIM))
    with pytest.raises((ValueError, FileNotFoundError, OSError)):
        PlanStage()(batch, config)


def test_plan_rejects_geometry_that_cannot_shard(tmp_path):
    _pipe, batch, _job, config = build(tmp_path)
    batch.set("prompt_embeds", torch.randn(2, TEXT_DIM))
    batch.meta["ulysses_world"] = 7  # 64-aligned sequence is not divisible by 7
    with pytest.raises(ValueError, match="does not divide"):
        PlanStage()(batch, config)


def test_pipeline_requires_its_components(tmp_path):
    config = DiffusionConfig(
        model_path="<test>",
        pipeline_class="atom.diffusion.pipelines.minimax_h3.MiniMaxH3Pipeline",
        num_gpus=1,
        ulysses_degree=1,
        output_dir=str(tmp_path),
    )
    pipe = MiniMaxH3Pipeline(config)
    batch = DiffusionBatch(job=DiffusionJob(prompt="x", task="t2va"))
    batch.meta.update({"ulysses_world": 1, "ulysses_rank": 0, "device": "cpu"})
    with pytest.raises(RuntimeError, match="missing required components"):
        pipe.forward(batch)
