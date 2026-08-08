# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Frame/latent arithmetic and the sigma schedule follow the reference in
# sgl-project/sglang (.../minimax_h3/time_request.py, Apache-2.0).

"""MiniMax-H3 request geometry: frames, latents, token counts, sigmas.

Every formula here is checked against a live 1344x768 / 5.1667 s capture:

    124 frames        -> video latent T 37
    37 x (48/2 x 84/2) -> 37,296 video rows
    5.1667 s          -> audio latent T 207 -> 414 audio rows
    2 text + 414 + 37,296 = 37,712, padded to 37,760

so the numbers in the tests are observed, not derived twice.
"""

from dataclasses import dataclass

# The video VAE compresses 16x spatially; the DiT then applies a (1, 2, 2)
# patch, so each latent frame contributes (H/16/2) * (W/16/2) rows.
VAE_SPATIAL_COMPRESSION = 16

# Audio latents are produced at 40 Hz.
AUDIO_LATENT_HZ = 40.0

# Two interleaved audio channels per latent step.
AUDIO_CHANNELS = 2

# The packed sequence is padded up to this multiple.
PACKED_SEQUENCE_ALIGNMENT = 64


def align_frame_count(frame_count: int) -> int:
    """Snap up to H3's 17n+5 frame boundary."""
    if frame_count <= 0:
        return 1
    current = int(frame_count)
    return current + (5 - current) % 17


def video_latent_t(frame_count: int) -> int:
    """Frames -> video latent T (1 or 5n+2)."""
    if frame_count <= 5:
        return 2
    return ((int(frame_count) - 5) // 17) * 5 + 2


def frame_count_from_video_latent_t(latent_t: int) -> int:
    """Inverse of :func:`video_latent_t`."""
    if latent_t == 1:
        return 1
    if latent_t < 2 or (latent_t - 2) % 5 != 0:
        raise ValueError(f"video latent T must be 1 or match 5n+2, got {latent_t}")
    return 17 * ((latent_t - 2) // 5) + 5


def audio_latent_t(duration_seconds: float) -> int:
    """Duration -> audio latent T, rounded at the 40 Hz boundary."""
    # round() with no ndigits already returns int.
    return round(float(duration_seconds) * AUDIO_LATENT_HZ)


def time_shift_sigmas(*, num_steps: int = 50, shift_scale: float = 6.0) -> list[float]:
    """Rectified-flow sigma schedule with a flow shift.

    ``sigma = s*b / (1 + (s-1)*b)`` over ``b`` linearly spaced on [1, 0].
    Returns ``num_steps`` sigmas ending at 0, so a denoise loop runs
    ``num_steps - 1`` iterations (50 sigmas -> 49 steps, which is what the
    reference server reports).
    """
    if shift_scale <= 0:
        raise ValueError(f"shift_scale must be > 0, got {shift_scale}")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be > 0, got {num_steps}")

    import torch

    base = torch.linspace(1.0, 0.0, int(num_steps), dtype=torch.float32)
    shifted = shift_scale * base / (1 + (shift_scale - 1) * base)
    shifted = torch.unique_consecutive(shifted)
    if num_steps > 1 and shifted[-1].item() > 0.0:
        shifted = torch.cat([shifted, torch.zeros(1, dtype=shifted.dtype)])
    return [float(v) for v in shifted.tolist()]


@dataclass(frozen=True)
class MiniMaxH3Geometry:
    """Resolved token layout for one request."""

    height: int
    width: int
    frame_count: int
    duration_seconds: float
    text_len: int

    latent_t: int
    latent_h: int
    latent_w: int
    audio_t: int

    video_rows: int
    audio_rows: int
    used_len: int
    seq_len: int

    @classmethod
    def resolve(
        cls,
        *,
        height: int,
        width: int,
        frame_count: int,
        duration_seconds: float,
        text_len: int,
        patch_size: tuple[int, int, int] = (1, 2, 2),
    ) -> "MiniMaxH3Geometry":
        if height % VAE_SPATIAL_COMPRESSION or width % VAE_SPATIAL_COMPRESSION:
            raise ValueError(
                f"height and width must be multiples of "
                f"{VAE_SPATIAL_COMPRESSION}, got {height}x{width}"
            )
        aligned_frames = align_frame_count(frame_count)
        lt = video_latent_t(aligned_frames)
        lh = height // VAE_SPATIAL_COMPRESSION
        lw = width // VAE_SPATIAL_COMPRESSION

        _, ph, pw = patch_size
        if lh % ph or lw % pw:
            raise ValueError(
                f"latent grid {lh}x{lw} is not divisible by patch {ph}x{pw}"
            )
        frame_rows = (lh // ph) * (lw // pw)
        video_rows = lt * frame_rows

        at = audio_latent_t(duration_seconds)
        audio_rows = at * AUDIO_CHANNELS

        used = text_len + audio_rows + video_rows
        seq = (
            (used + PACKED_SEQUENCE_ALIGNMENT - 1)
            // PACKED_SEQUENCE_ALIGNMENT
            * PACKED_SEQUENCE_ALIGNMENT
        )
        return cls(
            height=height,
            width=width,
            frame_count=aligned_frames,
            duration_seconds=duration_seconds,
            text_len=text_len,
            latent_t=lt,
            latent_h=lh,
            latent_w=lw,
            audio_t=at,
            video_rows=video_rows,
            audio_rows=audio_rows,
            used_len=used,
            seq_len=seq,
        )

    def validate_ulysses(self, world_size: int) -> None:
        """The padded sequence must split evenly across the Ulysses group."""
        if self.seq_len % world_size:
            raise ValueError(
                f"packed sequence {self.seq_len} does not divide across "
                f"ulysses world size {world_size}; alignment is "
                f"{PACKED_SEQUENCE_ALIGNMENT}, so degrees above that or "
                f"non-divisors of it cannot work"
            )
