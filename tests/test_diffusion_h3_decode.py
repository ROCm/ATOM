# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for MiniMax-H3 VAE decode helpers and the MP4 mux.

The VAE modules themselves live in the checkpoint and need the weights, so
these cover the pure transforms around them: latent de-normalisation, canvas
cropping, frame quantisation and the container contract.
"""

import pytest
import torch

from atom.diffusion.models.vaes.minimax_h3 import (
    crop_to_canvas,
    denormalize_latents,
    denormalize_pixels,
)
from atom.diffusion.postprocess.mux import frames_to_uint8, write_video_with_audio

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


def test_denormalize_rejects_channel_mismatch():
    with pytest.raises(ValueError, match="channel mismatch"):
        denormalize_latents(torch.ones(1, 4, 2), mean=[0.0, 1.0], std=[1.0, 1.0])


def test_denormalize_rejects_mean_std_shape_mismatch():
    with pytest.raises(ValueError, match="mean/std shape mismatch"):
        denormalize_latents(torch.ones(1, 2, 2), mean=[0.0, 1.0], std=[1.0])


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


def test_crop_rejects_upscaling():
    with pytest.raises(ValueError, match="smaller than the target canvas"):
        crop_to_canvas(torch.randn(1, 3, 2, 4, 4), height=8, width=8)


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


def test_frames_to_uint8_rejects_wrong_channel_count():
    with pytest.raises(ValueError, match="3 colour channels"):
        frames_to_uint8(torch.randn(1, 4, 2, 8, 8))


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


def test_denormalize_pixels_requires_transform_rev():
    class _NoTransform:
        pass

    with pytest.raises(AttributeError, match="transform_rev"):
        denormalize_pixels(torch.zeros(1, 3, 1, 2, 2), _NoTransform())


def test_denormalize_pixels_rejects_wrong_rank():
    with pytest.raises(ValueError, match="rank 5"):
        denormalize_pixels(torch.zeros(3, 4, 5), _FakeVAE())
