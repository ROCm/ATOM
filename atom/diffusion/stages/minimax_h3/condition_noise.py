# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Noise recipe follows the reference in sgl-project/sglang
# (.../minimax_h3/condition_noise.py, Apache-2.0).

"""MiniMax-H3 conditioning noise augmentation.

Visual conditioning rows -- fl2va keyframes and ref2va image/video references
-- are **not** fed to the DiT clean. They are mixed with seeded noise on the
rectified-flow line at a fixed timestep:

    rows = a * clean + (1 - a) * noise,      a = MINIMAX_H3_IMGVID_COND_TIMESTEP

and the same ``a`` is what the DiT sees as those rows' timestep. So the 0.999
in the captured ``unique_timesteps`` is not a "nearly clean" convention -- it
is literally the mixing coefficient, and value and timestep must agree or the
model is told the anchor is cleaner than it is.

Audio references use ``a = 1.0``, i.e. no augmentation at all. Keeping them as
separate constants rather than one shared knob is deliberate: they differ.

The RNG contract is fiddly and unobservable if wrong -- a wrong draw yields a
plausible anchor and a silently different video:

* a **fresh CPU generator per condition**, so concatenating conditions and
  drawing once is *not* equivalent for multi-reference requests;
* visual conditions seed with ``seed``, audio with ``seed + 1``;
* the visual draw is taken at ``target_latent_t + num_conditions`` frames and
  then **sliced** to the condition's own length -- drawing the condition's
  length directly gives a different sample for the same seed;
* the mix is evaluated in fp32.
"""

from collections.abc import Sequence

import torch

from atom.diffusion.stages.minimax_h3.packed_tokens import patchify_video_latent

# Both are timesteps *and* mixing coefficients; see the module docstring.
MINIMAX_H3_IMGVID_COND_TIMESTEP = 0.999
MINIMAX_H3_AUDIO_REF_COND_TIMESTEP = 1.0

AUDIO_COND_CHANNELS = 2
LATENT_CHANNELS = 24
VIDEO_ROW_WIDTH = 96
AUDIO_ROW_WIDTH = 32
COND_PATCH_SIZE = (1, 2, 2)


def _check_noise_aug(noise_aug: float) -> float:
    noise_aug = float(noise_aug)
    if not 0.0 <= noise_aug <= 1.0:
        raise ValueError(f"noise_aug must be in [0, 1], got {noise_aug}")
    return noise_aug


def imgvid_cond_noise_aug_rows(
    clean_rows: torch.Tensor,
    *,
    condition_shapes: Sequence[Sequence[int]],
    target_latent_t: int,
    seed: int,
    noise_aug: float = MINIMAX_H3_IMGVID_COND_TIMESTEP,
) -> torch.Tensor:
    """Noise-augment packed visual conditioning rows ``[n, 96]``.

    ``condition_shapes`` holds ``(latent_t, latent_h, latent_w)`` per condition
    in packed order. The number of conditions is itself part of the noise draw
    length, so adding a second keyframe changes the first one's noise.
    """
    noise_aug = _check_noise_aug(noise_aug)
    if noise_aug == 1.0:
        return clean_rows
    if clean_rows.ndim != 2 or int(clean_rows.shape[1]) != VIDEO_ROW_WIDTH:
        raise ValueError(
            f"visual condition rows must be [n, {VIDEO_ROW_WIDTH}], got "
            f"{list(clean_rows.shape)}"
        )
    target_latent_t = int(target_latent_t)
    if target_latent_t <= 0:
        raise ValueError(f"target_latent_t must be positive, got {target_latent_t}")

    shapes: list[tuple[int, int, int]] = []
    expected_rows = 0
    for index, raw in enumerate(condition_shapes):
        if len(raw) != 3:
            raise ValueError(
                f"condition_shapes[{index}] must be (latent_t, latent_h, "
                f"latent_w), got {list(raw)}"
            )
        lt, lh, lw = (int(v) for v in raw)
        if lt <= 0 or lh <= 0 or lw <= 0:
            raise ValueError(f"condition_shapes[{index}] must be positive: {list(raw)}")
        if lh % 2 or lw % 2:
            raise ValueError(
                f"condition_shapes[{index}] spatial dims must be even: {(lt, lh, lw)}"
            )
        shapes.append((lt, lh, lw))
        expected_rows += lt * (lh // 2) * (lw // 2)
    if not shapes:
        raise ValueError("condition_shapes must not be empty")
    if int(clean_rows.shape[0]) != expected_rows:
        raise ValueError(
            f"got {int(clean_rows.shape[0])} visual condition rows, shapes imply "
            f"{expected_rows}"
        )

    num_conditions = len(shapes)
    coefficient = torch.tensor(noise_aug, dtype=torch.float32, device=clean_rows.device)
    out: list[torch.Tensor] = []
    offset = 0
    for latent_t, latent_h, latent_w in shapes:
        draw_t = target_latent_t + num_conditions
        if draw_t < latent_t:
            raise ValueError(
                f"condition latent_t {latent_t} exceeds the noise draw length "
                f"{draw_t}"
            )
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(
            1,
            LATENT_CHANNELS,
            draw_t,
            latent_h,
            latent_w,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )[:, :, :latent_t]
        noise_rows = patchify_video_latent(noise, patch_size=COND_PATCH_SIZE).to(
            device=clean_rows.device, dtype=torch.float32
        )
        count = int(noise_rows.shape[0])
        clean = clean_rows[offset : offset + count].to(torch.float32)
        out.append(coefficient * clean + (1.0 - coefficient) * noise_rows)
        offset += count
    return (out[0] if len(out) == 1 else torch.cat(out, dim=0)).contiguous()


def audio_cond_noise_aug_rows(
    clean_rows: torch.Tensor,
    *,
    condition_audio_t: Sequence[int],
    seed: int,
    noise_aug: float = MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
) -> torch.Tensor:
    """Noise-augment packed audio reference rows ``[n, 32]``.

    A no-op at the released default (``noise_aug = 1.0``); kept complete so the
    audio path is not silently different from the visual one if that changes.
    """
    noise_aug = _check_noise_aug(noise_aug)
    if noise_aug == 1.0:
        return clean_rows
    if clean_rows.ndim != 2 or int(clean_rows.shape[1]) != AUDIO_ROW_WIDTH:
        raise ValueError(
            f"audio condition rows must be [n, {AUDIO_ROW_WIDTH}], got "
            f"{list(clean_rows.shape)}"
        )
    lengths = [int(v) for v in condition_audio_t]
    if not lengths:
        raise ValueError("condition_audio_t must not be empty")
    if any(v <= 0 for v in lengths):
        raise ValueError(f"condition audio lengths must be positive, got {lengths}")
    expected_rows = AUDIO_COND_CHANNELS * sum(lengths)
    if int(clean_rows.shape[0]) != expected_rows:
        raise ValueError(
            f"got {int(clean_rows.shape[0])} audio condition rows, lengths imply "
            f"{expected_rows}"
        )

    coefficient = torch.tensor(noise_aug, dtype=torch.float32)
    out: list[torch.Tensor] = []
    offset = 0
    for audio_t in lengths:
        count = AUDIO_COND_CHANNELS * audio_t
        clean = clean_rows[offset : offset + count].detach().cpu().float()
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
        noise = torch.randn(
            clean.shape, generator=generator, dtype=torch.float32, device="cpu"
        )
        out.append(coefficient * clean + (1.0 - coefficient) * noise)
        offset += count
    rows = out[0] if len(out) == 1 else torch.cat(out, dim=0)
    return rows.to(device=clean_rows.device, dtype=torch.float32).contiguous()
