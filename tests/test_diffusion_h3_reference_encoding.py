# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""ref2va reference-material geometry and sampling rules.

The encode itself needs the real VAEs, so these cover the deterministic parts:
shape resolution, the Qwen frame/timestamp sampling, and the determinism
context's restore behaviour.
"""

import pytest
import torch

from atom.diffusion.models.minimax_h3.conditioning import (
    QWEN_TEMPORAL_PATCH,
    REFERENCE_IMAGE_MULTIPLE,
    REFERENCE_IMAGE_SHORT_EDGE,
    audio_vae_determinism,
    resize_reference_image,
    resolve_reference_image_shape,
    sample_reference_video_frames,
)

np = pytest.importorskip("numpy")


def test_short_edge_lands_on_the_reference_target():
    shape = resolve_reference_image_shape(width=1344, height=768)
    assert shape["short_edge"] == REFERENCE_IMAGE_SHORT_EDGE
    assert shape["height"] == REFERENCE_IMAGE_SHORT_EDGE


def test_both_axes_round_to_the_32_grid():
    shape = resolve_reference_image_shape(width=1333, height=777)
    assert shape["width"] % REFERENCE_IMAGE_MULTIPLE == 0
    assert shape["height"] % REFERENCE_IMAGE_MULTIPLE == 0


def test_small_images_are_upscaled():
    """Unlike the target canvas, a reference always reaches the short edge."""
    shape = resolve_reference_image_shape(width=320, height=240)
    assert shape["short_edge"] == REFERENCE_IMAGE_SHORT_EDGE
    assert shape["width"] > 320


def test_aspect_ratio_is_preserved_within_the_rounding_grid():
    shape = resolve_reference_image_shape(width=1600, height=800)
    assert shape["width"] == pytest.approx(2 * shape["height"], rel=0.02)


def test_extreme_ratios_are_rejected():
    with pytest.raises(ValueError, match="1:4 to 4:1"):
        resolve_reference_image_shape(width=5000, height=1000)


@pytest.mark.parametrize("width,height", [(0, 100), (100, -1), (float("inf"), 100)])
def test_degenerate_dimensions_are_rejected(width, height):
    with pytest.raises(ValueError, match="positive finite"):
        resolve_reference_image_shape(width=width, height=height)


def test_resize_rejects_unaligned_targets():
    pil = pytest.importorskip("PIL.Image")
    image = pil.new("RGB", (64, 64))
    with pytest.raises(ValueError, match="aligned"):
        resize_reference_image(image, target_width=100, target_height=64)


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
