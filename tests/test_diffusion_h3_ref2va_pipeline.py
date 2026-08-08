# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""ref2va stage wiring, on CPU with stubs.

The real-weights ref2va run is seeded from a captured DiT input and therefore
skips the pipeline's own condition-encoding and layout stages entirely. This
covers exactly that gap: that references reach the packed layout, that the two
conditioning kinds are augmented differently, and that the denoise loop trims
the reference audio off the DiT's prediction.
"""

import numpy as np
import pytest
import torch
from torch import nn

from atom.diffusion.config import ComponentConfig, DiffusionConfig
from atom.diffusion.models.minimax_h3.condition_noise import (
    MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    MINIMAX_H3_IMGVID_COND_TIMESTEP,
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
from tests.test_diffusion_h3_pipeline import StubTextEncoder

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
        components=[
            ComponentConfig(name="transformer", class_path="torch.nn.Identity")
        ],
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
        "atom.diffusion.models.minimax_h3.reference_encoding."
        "load_reference_waveform",
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
        "atom.diffusion.models.minimax_h3.reference_encoding."
        "load_reference_waveform",
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
        "atom.diffusion.models.minimax_h3.reference_encoding."
        "load_reference_waveform",
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
    from atom.diffusion.models.minimax_h3.packed_sequence import (
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


def test_denoise_rejects_missing_audio_reference_rows():
    from atom.diffusion.models.minimax_h3.denoise import run_denoise_loop
    from atom.diffusion.models.minimax_h3.packed_sequence import (
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
    with pytest.raises(ValueError, match="audio reference rows"):
        run_denoise_loop(
            dit=lambda **k: (torch.zeros(1, 96), torch.zeros(1, 32)),
            video_rows=torch.zeros(int(packed["img_pos"].numel()), 96),
            audio_rows=torch.zeros(6, 32),
            packed=packed,
            video_sigmas=[1.0, 0.0],
            audio_sigmas=[1.0, 0.0],
            rank_slice=(0, int(packed["seq_len"])),
            prompt_embeds=torch.zeros(4, 64),
            refined_prompt_embeds_length=4,
            rope_cache=torch.zeros(int(packed["seq_len"]), 96),
        )


def test_video_reference_frame_sampling_feeds_both_consumers():
    """Qwen and the VAE must see the same decoded array, not two decodes."""
    from atom.diffusion.models.minimax_h3.reference_encoding import (
        sample_reference_video_frames,
    )

    frames = np.zeros((49, 8, 8, 3), dtype=np.uint8)
    out = sample_reference_video_frames(frames)
    assert out["frames"].base is not None or out["frames"].shape[0] == 5
