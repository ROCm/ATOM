# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Wan2.1 video VAE: 3D causal conv encoder / decoder, 16 latent channels,
8x spatial and 4x temporal compression.

Numerically this is Diffusers' ``AutoencoderKLWan`` (Wan2.1 configuration):
module names match it, so a Diffusers ``vae/`` checkpoint loads strictly, and
every op matches it in order and dtype. What differs is execution, not math:

* Every conv runs through :func:`atom.diffusion.models.wan.conv_dispatch.conv3d`,
  the 2D resample convs included -- as ``(1, kH, kW)`` conv3d with frames
  folded into the batch, the shape AITER's Wan2.1 tile table is keyed on.
* Tensors stay logical NCDHW but are physically NDHWC
  (``channels_last_3d``) end to end by default, the layout AITER's FlyDSL
  conv is tuned in. Every reshaping op below is written so that it keeps the
  layout as a view rather than paying a transpose.
* The causal temporal pad, the cached tail frames and the spatial pad are
  staged into one buffer per conv (:func:`_stage_causal_input`), where
  Diffusers pays a ``cat`` and a ``pad``.
* On GPU, the ``silu(norm(x))`` in front of each causal conv is one Triton
  kernel that writes into that buffer (:func:`_norm_silu_conv`), rounding
  where the unfused ops do.

Video is processed causally in temporal chunks: encode sees the first frame,
then 4 frames at a time; decode sees one latent frame at a time. Each causal
conv carries the last :data:`CACHE_T` frames of its input from one chunk to the
next in a :class:`FeatureCache`, so chunked output equals whole-clip causal
output.
"""

import itertools
import json
import logging
import os
from dataclasses import dataclass, field, fields

import torch
import torch.nn.functional as F
from torch import nn

from atom.diffusion.models.wan.conv_dispatch import conv3d, memory_format_of

logger = logging.getLogger(__name__)

CACHE_T = 2

WAN21_LATENTS_MEAN = (
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
)
WAN21_LATENTS_STD = (
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
)


@dataclass(frozen=True)
class WanVAEConfig:
    """The Wan2.1 subset of Diffusers' ``AutoencoderKLWan`` config."""

    base_dim: int = 96
    z_dim: int = 16
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    temporal_downsample: tuple[bool, ...] = (False, True, True)
    in_channels: int = 3
    out_channels: int = 3
    latents_mean: tuple[float, ...] = field(default=WAN21_LATENTS_MEAN)
    latents_std: tuple[float, ...] = field(default=WAN21_LATENTS_STD)

    def __post_init__(self):
        if len(self.temporal_downsample) != len(self.dim_mult) - 1:
            raise ValueError(
                f"temporal_downsample has {len(self.temporal_downsample)} "
                f"entries, need one per downsample ({len(self.dim_mult) - 1})"
            )
        if not len(self.latents_mean) == len(self.latents_std) == self.z_dim:
            raise ValueError(
                f"latents_mean/std need {self.z_dim} entries, got "
                f"{len(self.latents_mean)}/{len(self.latents_std)}"
            )

    @property
    def spatial_compression(self) -> int:
        return 2 ** (len(self.dim_mult) - 1)

    @property
    def temporal_compression(self) -> int:
        return 2 ** sum(self.temporal_downsample)

    @classmethod
    def from_diffusers(cls, config: dict) -> "WanVAEConfig":
        """From a Diffusers ``vae/config.json``; refuses non-Wan2.1 variants.

        Wan2.2's VAE (``is_residual``, ``patch_size``) and a decoder wider
        than the encoder are different networks, not options of this one.
        """
        if config.get("is_residual"):
            raise ValueError("is_residual=True is the Wan2.2 VAE; not supported")
        if config.get("patch_size") is not None:
            raise ValueError("patch_size is the Wan2.2 VAE; not supported")
        decoder_base_dim = config.get("decoder_base_dim")
        if decoder_base_dim is not None and decoder_base_dim != config.get(
            "base_dim", cls.base_dim
        ):
            raise ValueError("decoder_base_dim != base_dim is not supported")
        if config.get("attn_scales"):
            raise ValueError("attn_scales is not supported (Wan2.1 uses none)")
        renamed = {"temperal_downsample": "temporal_downsample"}
        known = {f.name for f in fields(cls)}
        kwargs = {}
        for key, value in config.items():
            key = renamed.get(key, key)
            if key in known:
                kwargs[key] = tuple(value) if isinstance(value, list) else value
        return cls(**kwargs)


class FeatureCache:
    """The tail frames each causal conv carries from one chunk to the next.

    Slots are positional: every chunk visits the causal convs in the same
    order, so the n-th visit of a chunk reads what the n-th visit of the
    previous chunk stored. :meth:`start_chunk` rewinds to the first slot.
    """

    def __init__(self):
        self._slots: list = []
        self._next = 0

    def start_chunk(self) -> None:
        self._next = 0

    def peek(self):
        """What this slot's previous chunk stored, or None on the first."""
        return self._slots[self._next] if self._next < len(self._slots) else None

    def store(self, value) -> None:
        if self._next < len(self._slots):
            self._slots[self._next] = value
        else:
            self._slots.append(value)
        self._next += 1


# An upsample3d slot after the first chunk: that chunk skipped the temporal
# upsample, so the next one starts its time_conv from zeros.
_SKIPPED_TIME_UPSAMPLE = object()


def _cat_frames(parts: list[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    """``torch.cat(parts, dim=2)`` in ``like``'s memory format.

    ``torch.cat`` falls back to NCDHW when its inputs' formats disagree, and
    to torch a one-frame batch-1 tensor -- every first chunk -- has no
    definite format, so a plain cat drops NDHWC there.
    """
    b, c, _, h, w = like.shape
    frames = sum(int(p.shape[2]) for p in parts)
    out = torch.empty(
        (b, c, frames, h, w),
        dtype=like.dtype,
        device=like.device,
        memory_format=memory_format_of(like),
    )
    start = 0
    for part in parts:
        out[:, :, start : start + part.shape[2]].copy_(part)
        start += part.shape[2]
    return out


def _cache_tail(x: torch.Tensor, prev: torch.Tensor | None) -> torch.Tensor:
    """The last CACHE_T frames, topped up from ``prev`` for 1-frame chunks."""
    tail = x[:, :, -CACHE_T:]
    if tail.shape[2] < CACHE_T and prev is not None:
        return _cat_frames([prev[:, :, -1:], tail], like=x)
    return tail.clone()


def _stage_causal_input(
    x: torch.Tensor,
    cache: torch.Tensor | None,
    *,
    front: int,
    pad_h: int,
    pad_w: int,
) -> torch.Tensor:
    """``F.pad(cat([cache, x], 2), (pw, pw, ph, ph, front - len(cache), 0))``
    in one buffer of ``x``'s memory format.

    Bitwise what Diffusers computes, but the activation is copied once where
    cat then pad copy it twice -- at 480x832 that buffer is ~460 MB per conv.
    """
    cache_t = 0 if cache is None else int(cache.shape[2])
    if not (front - cache_t or cache_t or pad_h or pad_w):
        return x
    out, slot = _staging_buffer(x, cache, front=front, pad_h=pad_h, pad_w=pad_w)
    slot.copy_(x)
    return out


def _staging_buffer(
    x: torch.Tensor,
    cache: torch.Tensor | None,
    *,
    front: int,
    pad_h: int,
    pad_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`_stage_causal_input`'s buffer with the pads and ``cache`` in
    place, and the view of it that ``x`` goes into."""
    cache_t = 0 if cache is None else int(cache.shape[2])
    if cache_t > front:
        raise ValueError(f"cache holds {cache_t} frames, conv pads only {front}")
    b, c, t, h, w = x.shape
    zero_t = front - cache_t
    out = torch.empty(
        (b, c, front + t, h + 2 * pad_h, w + 2 * pad_w),
        dtype=x.dtype,
        device=x.device,
        memory_format=memory_format_of(x),
    )
    if zero_t:
        out[:, :, :zero_t].zero_()
    if pad_h:
        out[:, :, zero_t:, :pad_h].zero_()
        out[:, :, zero_t:, -pad_h:].zero_()
    if pad_w:
        out[:, :, zero_t:, pad_h : pad_h + h, :pad_w].zero_()
        out[:, :, zero_t:, pad_h : pad_h + h, -pad_w:].zero_()
    inner = out[:, :, :, pad_h : pad_h + h, pad_w : pad_w + w]
    if cache_t:
        inner[:, :, zero_t:front].copy_(cache)
    return out, inner[:, :, front:]


class WanCausalConv3d(nn.Conv3d):
    """Conv3d that pads time on the front only (``2 * pad_t`` frames), filling
    that pad from the previous chunk's tail frames when given them."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride)
        pad = (padding,) * 3 if isinstance(padding, int) else tuple(padding)
        self.causal_front = 2 * pad[0]
        self.pad_h, self.pad_w = pad[1], pad[2]

    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
        *,
        staged: bool = False,
    ):
        """``staged``: ``x`` is already the padded buffer, e.g. from
        :func:`_norm_silu_conv`."""
        if not staged:
            x = _stage_causal_input(
                x,
                cache if self.causal_front else None,
                front=self.causal_front,
                pad_h=self.pad_h,
                pad_w=self.pad_w,
            )
        return conv3d(x, self.weight, self.bias, self.stride)


class FrameConv2d(nn.Conv2d):
    """A 2D conv applied to every frame of ``[B, C, T, H, W]``.

    Runs as a ``(1, kH, kW)`` conv3d with frames folded into the batch,
    which is a view for an NDHWC tensor and the shape (``N = B*T, D = 1``)
    AITER's Wan2.1 table is tuned on. Keeps the 4-D Diffusers weight.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        frames = x.transpose(1, 2).reshape(b * t, c, 1, h, w)
        y = conv3d(
            frames,
            self.weight.unsqueeze(2),
            self.bias,
            stride=(1, *self.stride),
            padding=(0, *self.padding),
        )
        _, k, _, ho, wo = y.shape
        y = y.reshape(b, t, k, ho, wo).transpose(1, 2)
        return y.contiguous(memory_format=memory_format_of(x))


class _FrameUpsample2x(nn.Module):
    """Per-frame 2x nearest-exact upsample (Diffusers' ``WanUpsample``).

    Diffusers upsamples in fp32; nearest only copies elements, so running it
    in the input dtype gives the same bits without the two casts.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        frames = x.transpose(1, 2).reshape(b * t, c, h, w)
        up = F.interpolate(frames, scale_factor=(2.0, 2.0), mode="nearest-exact")
        up = up.reshape(b, t, c, 2 * h, 2 * w).transpose(1, 2)
        return up.contiguous(memory_format=memory_format_of(x))


class _FramePadBottomRight(nn.Module):
    """``nn.ZeroPad2d((0, 1, 0, 1))`` on every frame."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (0, 1, 0, 1)).contiguous(memory_format=memory_format_of(x))


class WanRMSNorm(nn.Module):
    """L2-normalize over channels (in fp32), then ``* sqrt(C) * gamma``."""

    def __init__(self, dim: int, images: bool = True):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones((dim, 1, 1) if images else (dim, 1, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(x.float(), dim=1).to(x.dtype)
        return normalized * self.scale * self.gamma


def _cached_conv(conv: WanCausalConv3d, x: torch.Tensor, cache: FeatureCache | None):
    if cache is None:
        return conv(x)
    prev = cache.peek()
    tail = _cache_tail(x, prev)
    out = conv(x, prev)
    cache.store(tail)
    return out


def _norm_silu_conv(
    norm: WanRMSNorm,
    conv: WanCausalConv3d,
    x: torch.Tensor,
    cache: FeatureCache | None,
):
    """``_cached_conv(conv, F.silu(norm(x)), cache)``.

    On GPU the norm and SiLU run as one kernel that writes straight into the
    conv's staging buffer, and the cached tail frames are read back from it.
    """
    if x.device.type != "cuda" or torch.is_grad_enabled():
        return _cached_conv(conv, F.silu(norm(x)), cache)
    from atom.diffusion.models.wan.norm_kernels import rms_norm_silu

    prev = cache.peek() if cache is not None else None
    staged_prev = prev if conv.causal_front else None
    pad_h, pad_w = conv.pad_h, conv.pad_w
    staged, slot = _staging_buffer(
        x, staged_prev, front=conv.causal_front, pad_h=pad_h, pad_w=pad_w
    )
    rms_norm_silu(x, norm.gamma, norm.scale, out=slot)
    if cache is not None:
        _, _, t, h, w = x.shape
        frames = min(CACHE_T, t + (0 if prev is None else int(prev.shape[2])))
        if frames > conv.causal_front + t:
            raise ValueError("conv pads too few frames to hold its cached tail")
        tail = staged[:, :, -frames:, pad_h : pad_h + h, pad_w : pad_w + w]
        cache.store(tail.clone(memory_format=memory_format_of(x)))
    return conv(staged, staged=True)


class WanResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm1 = WanRMSNorm(in_dim, images=False)
        self.conv1 = WanCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = WanRMSNorm(out_dim, images=False)
        self.conv2 = WanCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            WanCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x, cache: FeatureCache | None = None):
        h = self.conv_shortcut(x)
        x = _norm_silu_conv(self.norm1, self.conv1, x, cache)
        x = _norm_silu_conv(self.norm2, self.conv2, x, cache)
        return x + h


class WanAttentionBlock(nn.Module):
    """Single-head self-attention within each frame.

    The 1x1 ``to_qkv`` / ``proj`` convs run as ``F.linear`` over the channel
    axis -- the same contraction -- on ``[B*T, H*W, C]`` tokens, which is a
    view of an NDHWC tensor.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = WanRMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x, cache: FeatureCache | None = None):
        b, c, t, h, w = x.shape
        frames = self.norm(x.transpose(1, 2).reshape(b * t, c, h, w))
        tokens = frames.flatten(2).transpose(1, 2)
        qkv = F.linear(tokens, self.to_qkv.weight.flatten(1), self.to_qkv.bias)
        q, k, v = qkv.unsqueeze(1).chunk(3, dim=-1)
        out = F.scaled_dot_product_attention(q, k, v).squeeze(1)
        out = F.linear(out, self.proj.weight.flatten(1), self.proj.bias)
        out = out.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        return out.contiguous(memory_format=memory_format_of(x)) + x


class WanMidBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.resnets = nn.ModuleList([WanResidualBlock(dim, dim) for _ in range(2)])
        self.attentions = nn.ModuleList([WanAttentionBlock(dim)])

    def forward(self, x, cache: FeatureCache | None = None):
        x = self.resnets[0](x, cache)
        x = self.attentions[0](x)
        return self.resnets[1](x, cache)


class WanResample(nn.Module):
    """Spatial 2x down / up per frame, plus a temporal 2x for the 3d modes."""

    def __init__(self, dim: int, mode: str):
        super().__init__()
        self.mode = mode
        if mode in ("upsample2d", "upsample3d"):
            self.resample = nn.Sequential(
                _FrameUpsample2x(), FrameConv2d(dim, dim // 2, 3, padding=1)
            )
            if mode == "upsample3d":
                self.time_conv = WanCausalConv3d(
                    dim, dim * 2, (3, 1, 1), (1, 1, 1), (1, 0, 0)
                )
        elif mode in ("downsample2d", "downsample3d"):
            self.resample = nn.Sequential(
                _FramePadBottomRight(), FrameConv2d(dim, dim, 3, stride=2)
            )
            if mode == "downsample3d":
                self.time_conv = WanCausalConv3d(dim, dim, (3, 1, 1), (2, 1, 1))
        else:
            raise ValueError(f"unknown resample mode {mode!r}")

    def forward(self, x, cache: FeatureCache | None = None):
        if self.mode == "upsample3d" and cache is not None:
            x = self._upsample_time(x, cache)
        x = self.resample(x)
        if self.mode == "downsample3d" and cache is not None:
            x = self._downsample_time(x, cache)
        return x

    def _upsample_time(self, x, cache: FeatureCache):
        prev = cache.peek()
        if prev is None:
            # The first chunk is the first frame alone, never doubled.
            cache.store(_SKIPPED_TIME_UPSAMPLE)
            return x
        if prev is _SKIPPED_TIME_UPSAMPLE:
            prev = None
            tail = x[:, :, -CACHE_T:]
            if tail.shape[2] < CACHE_T:
                tail = _cat_frames([torch.zeros_like(tail), tail], like=x)
            else:
                tail = tail.clone()
        else:
            tail = _cache_tail(x, prev)
        y = self.time_conv(x, prev)
        cache.store(tail)
        # Channels [j*C:(j+1)*C] are output frame 2t+j.
        b, c, t, h, w = x.shape
        if memory_format_of(y) == torch.channels_last_3d:
            ndhwc = y.permute(0, 2, 3, 4, 1).reshape(b, t, h, w, 2, c)
            ndhwc = ndhwc.permute(0, 1, 4, 2, 3, 5).reshape(b, 2 * t, h, w, c)
            return ndhwc.permute(0, 4, 1, 2, 3)
        y = y.reshape(b, 2, c, t, h, w)
        return torch.stack((y[:, 0], y[:, 1]), 3).reshape(b, c, 2 * t, h, w)

    def _downsample_time(self, x, cache: FeatureCache):
        prev = cache.peek()
        tail = x[:, :, -1:].clone()
        if prev is not None:
            x = self.time_conv(_cat_frames([prev, x], like=x))
        cache.store(tail)
        return x


class WanEncoder3d(nn.Module):
    def __init__(self, cfg: WanVAEConfig):
        super().__init__()
        dims = [cfg.base_dim * u for u in (1, *cfg.dim_mult)]
        self.conv_in = WanCausalConv3d(cfg.in_channels, dims[0], 3, padding=1)
        down_blocks = []
        last = len(cfg.dim_mult) - 1
        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            for _ in range(cfg.num_res_blocks):
                down_blocks.append(WanResidualBlock(in_dim, out_dim))
                in_dim = out_dim
            if i != last:
                mode = "downsample3d" if cfg.temporal_downsample[i] else "downsample2d"
                down_blocks.append(WanResample(out_dim, mode))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.mid_block = WanMidBlock(dims[-1])
        self.norm_out = WanRMSNorm(dims[-1], images=False)
        self.conv_out = WanCausalConv3d(dims[-1], cfg.z_dim * 2, 3, padding=1)

    def forward(self, x, cache: FeatureCache | None = None):
        x = _cached_conv(self.conv_in, x, cache)
        for layer in self.down_blocks:
            x = layer(x, cache)
        x = self.mid_block(x, cache)
        return _norm_silu_conv(self.norm_out, self.conv_out, x, cache)


class WanUpBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_res_blocks: int, mode):
        super().__init__()
        self.resnets = nn.ModuleList(
            WanResidualBlock(in_dim if i == 0 else out_dim, out_dim)
            for i in range(num_res_blocks + 1)
        )
        self.upsamplers = (
            nn.ModuleList([WanResample(out_dim, mode)]) if mode is not None else None
        )

    def forward(self, x, cache: FeatureCache | None = None):
        for resnet in self.resnets:
            x = resnet(x, cache)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x, cache)
        return x


class WanDecoder3d(nn.Module):
    def __init__(self, cfg: WanVAEConfig):
        super().__init__()
        dims = [cfg.base_dim * u for u in (cfg.dim_mult[-1], *cfg.dim_mult[::-1])]
        temporal_upsample = cfg.temporal_downsample[::-1]
        self.conv_in = WanCausalConv3d(cfg.z_dim, dims[0], 3, padding=1)
        self.mid_block = WanMidBlock(dims[0])
        up_blocks = []
        last = len(cfg.dim_mult) - 1
        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            # Each upsampler halves the channels it hands to the next block.
            if i > 0:
                in_dim //= 2
            mode = None
            if i != last:
                mode = "upsample3d" if temporal_upsample[i] else "upsample2d"
            up_blocks.append(WanUpBlock(in_dim, out_dim, cfg.num_res_blocks, mode))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.norm_out = WanRMSNorm(dims[-1], images=False)
        self.conv_out = WanCausalConv3d(dims[-1], cfg.out_channels, 3, padding=1)

    def forward(self, x, cache: FeatureCache | None = None):
        x = _cached_conv(self.conv_in, x, cache)
        x = self.mid_block(x, cache)
        for block in self.up_blocks:
            x = block(x, cache)
        return _norm_silu_conv(self.norm_out, self.conv_out, x, cache)


class WanVAE(nn.Module):
    """Wan2.1 VAE. ``encode`` / ``decode`` take and return contiguous NCDHW.

    ``channels_last`` picks the working layout; outputs are the same either
    way, it only decides which layout the conv kernels see.
    """

    def __init__(self, config: WanVAEConfig | None = None, *, channels_last=True):
        super().__init__()
        self.config = config or WanVAEConfig()
        self.channels_last = channels_last
        z = self.config.z_dim
        self.encoder = WanEncoder3d(self.config)
        self.quant_conv = WanCausalConv3d(z * 2, z * 2, 1)
        self.post_quant_conv = WanCausalConv3d(z, z, 1)
        self.decoder = WanDecoder3d(self.config)

    @property
    def dtype(self) -> torch.dtype:
        return self.quant_conv.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.quant_conv.weight.device

    def _working(self, x: torch.Tensor) -> torch.Tensor:
        fmt = torch.channels_last_3d if self.channels_last else torch.contiguous_format
        return x.to(device=self.device, dtype=self.dtype).contiguous(memory_format=fmt)

    @torch.no_grad()
    def encode_moments(self, video: torch.Tensor) -> torch.Tensor:
        """``[B, 3, 1 + 4k, H, W]`` in [-1, 1] -> ``[B, 2*z, 1 + k, H/8, W/8]``.

        Channels ``[:z]`` are the posterior mean, ``[z:]`` its log-variance:
        Diffusers' ``DiagonalGaussianDistribution`` parameters.
        """
        if video.dim() != 5:
            raise ValueError(f"video must be [B, C, T, H, W], got {list(video.shape)}")
        _, _, frames, height, width = video.shape
        chunk = self.config.temporal_compression
        stride = self.config.spatial_compression
        if (frames - 1) % chunk:
            raise ValueError(f"frame count must be 1 + {chunk}k, got {frames}")
        if height % stride or width % stride:
            raise ValueError(
                f"H and W must be multiples of {stride}, got {height}x{width}"
            )
        x = self._working(video)
        cache = FeatureCache()
        chunks = []
        bounds = [(0, 1)] + [(s, s + chunk) for s in range(1, frames, chunk)]
        for start, stop in bounds:
            cache.start_chunk()
            chunks.append(self.encoder(x[:, :, start:stop], cache))
        moments = self.quant_conv(_cat_frames(chunks, like=chunks[-1]))
        return moments.contiguous()

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """Posterior mean latents (Diffusers ``latent_dist.mode()``), unnormalized."""
        return self.encode_moments(video)[:, : self.config.z_dim].contiguous()

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Unnormalized ``[B, z, T, h, w]`` -> video ``[B, 3, 1 + 4(T-1), 8h, 8w]``
        clamped to [-1, 1]."""
        if latents.dim() != 5 or latents.shape[1] != self.config.z_dim:
            raise ValueError(
                f"latents must be [B, {self.config.z_dim}, T, h, w], "
                f"got {list(latents.shape)}"
            )
        x = self.post_quant_conv(self._working(latents))
        cache = FeatureCache()
        frames = []
        for i in range(x.shape[2]):
            cache.start_chunk()
            frames.append(self.decoder(x[:, :, i : i + 1], cache))
        return torch.cat(frames, dim=2).clamp_(-1.0, 1.0).contiguous()

    def _latent_stats(self, latents: torch.Tensor):
        shape = (1, self.config.z_dim, 1, 1, 1)
        mean = torch.tensor(self.config.latents_mean, device=latents.device)
        std = torch.tensor(self.config.latents_std, device=latents.device)
        return mean.view(shape).to(latents.dtype), std.view(shape).to(latents.dtype)

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """VAE latents -> the DiT's space: ``(z - mean) / std``."""
        mean, std = self._latent_stats(latents)
        return (latents - mean) / std

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """DiT latents -> VAE latents: ``z * std + mean``."""
        mean, std = self._latent_stats(latents)
        return latents * std + mean


DIFFUSERS_WEIGHTS = "diffusion_pytorch_model.safetensors"
DIFFUSERS_WEIGHTS_INDEX = "diffusion_pytorch_model.safetensors.index.json"


def resolve_vae_dir(path: str) -> str:
    """A Diffusers checkpoint root or its ``vae/`` subfolder -> the ``vae/`` dir."""
    for candidate in (path, os.path.join(path, "vae")):
        has_weights = any(
            os.path.exists(os.path.join(candidate, name))
            for name in (DIFFUSERS_WEIGHTS, DIFFUSERS_WEIGHTS_INDEX)
        )
        if has_weights and os.path.isfile(os.path.join(candidate, "config.json")):
            return candidate
    raise FileNotFoundError(
        f"no Diffusers VAE (config.json + {DIFFUSERS_WEIGHTS}) in {path} or {path}/vae"
    )


def _read_diffusers_state_dict(vae_dir: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    index_path = os.path.join(vae_dir, DIFFUSERS_WEIGHTS_INDEX)
    if os.path.exists(index_path):
        with open(index_path) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
    elif os.path.exists(os.path.join(vae_dir, DIFFUSERS_WEIGHTS)):
        shards = [DIFFUSERS_WEIGHTS]
    else:
        raise FileNotFoundError(f"no {DIFFUSERS_WEIGHTS} under {vae_dir}")
    state_dict = {}
    for shard in shards:
        state_dict.update(load_file(os.path.join(vae_dir, shard)))
    return state_dict


_RESIDUAL_RENAMES = (
    (".residual.0.", ".norm1."),
    (".residual.2.", ".conv1."),
    (".residual.3.", ".norm2."),
    (".residual.6.", ".conv2."),
    (".shortcut.", ".conv_shortcut."),
)
_TOP_LEVEL_RENAMES = (
    ("conv1.", "quant_conv."),
    ("conv2.", "post_quant_conv."),
    ("encoder.conv1.", "encoder.conv_in."),
    ("decoder.conv1.", "decoder.conv_in."),
    ("encoder.head.0.", "encoder.norm_out."),
    ("encoder.head.2.", "encoder.conv_out."),
    ("decoder.head.0.", "decoder.norm_out."),
    ("decoder.head.2.", "decoder.conv_out."),
    ("encoder.middle.0.", "encoder.mid_block.resnets.0."),
    ("encoder.middle.1.", "encoder.mid_block.attentions.0."),
    ("encoder.middle.2.", "encoder.mid_block.resnets.1."),
    ("decoder.middle.0.", "decoder.mid_block.resnets.0."),
    ("decoder.middle.1.", "decoder.mid_block.attentions.0."),
    ("decoder.middle.2.", "decoder.mid_block.resnets.1."),
    ("encoder.downsamples.", "encoder.down_blocks."),
)


def convert_original_wan_vae_state_dict(
    state_dict: dict[str, torch.Tensor], config: WanVAEConfig | None = None
) -> dict[str, torch.Tensor]:
    """Wan's own ``Wan2.1_VAE.pth`` names -> Diffusers / this module's names.

    The original decoder keeps its levels in one flat ``upsamples`` list --
    ``num_res_blocks + 1`` residual blocks, then a resample unless last --
    where Diffusers nests them as ``up_blocks.i.{resnets.j, upsamplers.0}``.
    The encoder's ``downsamples`` is flat in both.
    """
    config = config or WanVAEConfig()
    decoder_slots = []
    for level in range(len(config.dim_mult)):
        for j in range(config.num_res_blocks + 1):
            decoder_slots.append(f"decoder.up_blocks.{level}.resnets.{j}.")
        if level != len(config.dim_mult) - 1:
            decoder_slots.append(f"decoder.up_blocks.{level}.upsamplers.0.")

    converted = {}
    for key, value in state_dict.items():
        new = key
        if key.startswith("decoder.upsamples."):
            index, rest = key[len("decoder.upsamples.") :].split(".", 1)
            new = decoder_slots[int(index)] + rest
        else:
            for old, replacement in _TOP_LEVEL_RENAMES:
                if key.startswith(old):
                    new = replacement + key[len(old) :]
                    break
        for old, replacement in _RESIDUAL_RENAMES:
            new = new.replace(old, replacement)
        converted[new] = value
    return converted


def load_wan_vae(
    path: str,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    channels_last: bool = True,
) -> WanVAE:
    """Load a Wan2.1 VAE from a Diffusers checkpoint (root or ``vae/``) or
    from the original ``Wan2.1_VAE.pth``. Strict: any missing or unexpected
    tensor is an error, since a partly loaded VAE decodes plausible noise."""
    if os.path.isfile(path):
        config = WanVAEConfig()
        raw = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = convert_original_wan_vae_state_dict(raw, config)
    else:
        vae_dir = resolve_vae_dir(path)
        with open(os.path.join(vae_dir, "config.json")) as f:
            config = WanVAEConfig.from_diffusers(json.load(f))
        state_dict = _read_diffusers_state_dict(vae_dir)

    with torch.device("meta"):
        model = WanVAE(config, channels_last=channels_last)
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Wan VAE checkpoint/module mismatch at {path}: {len(missing)} missing "
            f"(e.g. {missing[:3]}), {len(unexpected)} unexpected "
            f"(e.g. {unexpected[:3]})"
        )
    model = model.to(device=device, dtype=dtype).eval()
    logger.info(
        "loaded Wan VAE from %s: %d tensors on %s (%s, %s)",
        path,
        len(state_dict),
        device,
        dtype,
        "NDHWC" if channels_last else "NCDHW",
    )
    return model
