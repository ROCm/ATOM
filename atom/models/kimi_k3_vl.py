# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Native MoonViT3d vision encoder for Kimi-K3.

Pure PyTorch port of the reference ``modeling_kimi_k3.py`` vision stack: the
MoonViT3d tower, the ``sd2_tpool`` patch merger and the PatchMergerMLPV2
projector.  Module and parameter names mirror the checkpoint exactly
(``vision_tower.*`` / ``mm_projector.*``), so the loader needs no rename rules.

The tower is replicated on every TP rank rather than sharded: at ~0.45B params
it is negligible next to the language stack, and replication keeps the image
embeddings bit-identical across ranks (they are scattered into the token
embeddings before the first collective).
"""

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from aiter import flash_attn_varlen_func
from torch import nn

# The 2D RoPE grid is precomputed per (height, width); the processor caps both
# at `patch_limit_on_one_side` (512 in preprocessor_config.json).
MAX_GRID_SIDE = 512


def _as_pair(value) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, Sequence) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"Expected an int or a length-2 sequence, got {value!r}")


def _get_1d_sincos_pos_embed(embed_dim: int, t_size: int) -> np.ndarray:
    """Sin/cos temporal position table, matching the reference construction."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = np.arange(t_size, dtype=np.float32).reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class KimiK3InterpPosEmb(nn.Module):
    """Learnable 2D position embedding, bilinearly resampled per image.

    Mirrors ``Learnable2DInterpPosEmbDivided_fixed``: the learned ``height x
    width`` grid is interpolated to each image's patch grid, and for multi-frame
    inputs a fixed sin/cos temporal embedding is added on top.
    """

    def __init__(
        self,
        height: int,
        width: int,
        num_frames: int,
        dim: int,
        interpolation_mode: str = "bicubic",
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.interpolation_mode = interpolation_mode
        # Always overwritten by the checkpoint, but initialized like the
        # reference so an unloaded tower produces finite values instead of
        # whatever `torch.empty` picked up.
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        nn.init.normal_(self.weight)
        self.register_buffer(
            "time_weight",
            torch.from_numpy(_get_1d_sincos_pos_embed(dim, num_frames))
            .float()
            .unsqueeze(1),
            persistent=False,
        )

    def _resample(self, height: int, width: int) -> torch.Tensor:
        if (height, width) == self.weight.shape[:-1]:
            return self.weight.flatten(end_dim=1)
        resampled = F.interpolate(
            self.weight.permute(2, 0, 1).unsqueeze(0),
            size=(height, width),
            mode=self.interpolation_mode,
        )
        return resampled.squeeze(0).permute(1, 2, 0).flatten(end_dim=1)

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for t, h, w in grid_thws.tolist():
            if t > self.num_frames:
                raise ValueError(
                    f"frame count {t} > pos-emb num_frames {self.num_frames}"
                )
            pos_emb_2d = self._resample(h, w)
            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = (
                    pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) + self.time_weight[0:t]
                )
            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))
        return x + torch.cat(pos_embs)


class KimiK3PatchEmbed(nn.Module):
    """Patch projection over pre-patchified pixels, plus position embedding.

    The processor hands over patches of shape ``[L, 3, patch, patch]``, so the
    ``Conv2d`` degenerates into a per-patch linear projection.
    """

    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: int | Sequence[int] = (14, 14),
        pos_emb_height: int = 64,
        pos_emb_width: int = 64,
        pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        patch_embed_proj_bias: bool = False,
        pos_emb_interpolation_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        self.patch_size = _as_pair(patch_size)
        self.proj = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=patch_embed_proj_bias,
        )
        if pos_emb_type != "divided_fixed":
            raise NotImplementedError(f"Unsupported pos_emb_type: {pos_emb_type}")
        self.pos_emb = KimiK3InterpPosEmb(
            height=pos_emb_height,
            width=pos_emb_width,
            num_frames=pos_emb_time,
            dim=out_dim,
            interpolation_mode=pos_emb_interpolation_mode,
        )

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).view(x.size(0), -1)
        return self.pos_emb(x, grid_thws)


def rope_2d_freqs_cis(
    grid_thws: torch.Tensor,
    dim: int,
    device: torch.device,
    theta_base: float = 10000.0,
) -> torch.Tensor:
    """2D rotary frequencies for every patch of every image in the batch.

    Returns a ``complex64`` tensor of shape ``(sum(t*h*w), dim // 2)`` whose
    entries alternate between the width and height axes:
    ``ret[p, 2*i] = cis(x * theta_base ** (-4*i/dim))`` and
    ``ret[p, 2*i+1] = cis(y * theta_base ** (-4*i/dim))``.

    Computed directly per grid instead of slicing a precomputed
    ``MAX_GRID_SIDE x MAX_GRID_SIDE`` table (which would cost 134 MB of VRAM for
    the same values).
    """
    if dim % 4 != 0:
        raise ValueError(f"rope dim must be divisible by 4, got {dim}")
    dim_range = torch.arange(0, dim, 4, device=device, dtype=torch.float32)[: dim // 4]
    freqs = 1.0 / (theta_base ** (dim_range / dim))

    per_grid: dict[tuple[int, int], torch.Tensor] = {}
    out = []
    for t, h, w in grid_thws.tolist():
        if not (1 <= h <= MAX_GRID_SIDE and 1 <= w <= MAX_GRID_SIDE):
            raise ValueError(
                f"patch grid {h}x{w} exceeds the supported {MAX_GRID_SIDE} limit"
            )
        cached = per_grid.get((h, w))
        if cached is None:
            x_freqs = torch.outer(
                torch.arange(w, device=device, dtype=torch.float32), freqs
            )
            y_freqs = torch.outer(
                torch.arange(h, device=device, dtype=torch.float32), freqs
            )
            x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
            y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
            cached = torch.stack(
                [
                    x_cis.unsqueeze(0).expand(h, w, dim // 4),
                    y_cis.unsqueeze(1).expand(h, w, dim // 4),
                ],
                dim=-1,
            ).reshape(h * w, dim // 2)
            per_grid[(h, w)] = cached
        out.append(cached.repeat(t, 1))
    return torch.cat(out, dim=0)


def apply_rope_2d(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate query/key pairs by the 2D frequencies.

    ``xq``/``xk`` are ``(L, num_heads, head_dim)``; ``freqs_cis`` is
    ``(L, head_dim // 2)``.
    """
    freqs_cis = freqs_cis.unsqueeze(-2)
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    """Non-causal packed attention over per-image segments.

    ``cu_seqlens`` bounds each image's patch grid, so a patch only ever attends
    within its own image. Returns ``(L, num_heads * head_dim)``.
    """
    out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        softmax_scale=1.0 / math.sqrt(q.shape[-1]),
        causal=False,
    )
    if isinstance(out, tuple):
        out = out[0]
    return out.flatten(start_dim=-2)


class KimiK3VisionMLP(nn.Module):
    """MLP2 block: ``fc0 -> gelu(tanh) -> fc1``."""

    def __init__(self, dims: list[int], bias: bool = False):
        super().__init__()
        assert len(dims) == 3
        self.fc0 = nn.Linear(dims[0], dims[1], bias=bias)
        self.fc1 = nn.Linear(dims[1], dims[2], bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(F.gelu(self.fc0(x), approximate="tanh"))


class KimiK3VisionBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        qkv_hidden_size: int | None = None,
        norm_type: str = "rmsnorm",
        attn_bias: bool = False,
        linear_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.qkv_hidden_size = qkv_hidden_size or hidden_dim
        if self.qkv_hidden_size % num_heads:
            raise ValueError(
                f"qkv_hidden_size {self.qkv_hidden_size} not divisible by "
                f"num_heads {num_heads}"
            )
        self.head_dim = self.qkv_hidden_size // num_heads

        if norm_type == "layernorm":
            self.norm0 = nn.LayerNorm(hidden_dim)
            self.norm1 = nn.LayerNorm(hidden_dim)
        elif norm_type == "rmsnorm":
            # Default eps (torch.finfo(dtype).eps), matching the reference.
            self.norm0 = nn.RMSNorm(hidden_dim)
            self.norm1 = nn.RMSNorm(hidden_dim)
        else:
            raise NotImplementedError(f"Unsupported norm_type: {norm_type}")

        self.mlp = KimiK3VisionMLP([hidden_dim, mlp_dim, hidden_dim], bias=linear_bias)
        self.wqkv = nn.Linear(hidden_dim, self.qkv_hidden_size * 3, bias=attn_bias)
        self.wo = nn.Linear(self.qkv_hidden_size, hidden_dim, bias=attn_bias)

    def _attention(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rope_freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        xqkv = self.wqkv(x)
        xqkv = xqkv.view(*xqkv.shape[:-1], 3, self.num_heads, self.head_dim)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)
        xq, xk = apply_rope_2d(xq, xk, rope_freqs_cis)
        attn_out = varlen_attention(xq, xk, xv, cu_seqlens, max_seqlen)
        return self.wo(attn_out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rope_freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self._attention(
            self.norm0(hidden_states), cu_seqlens, max_seqlen, rope_freqs_cis
        )
        return hidden_states + self.mlp(self.norm1(hidden_states))


class KimiK3VisionEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        block_cfg: dict,
        norm_type: str = "rmsnorm",
    ) -> None:
        super().__init__()
        qkv_hidden_size = block_cfg.get("qkv_hidden_size") or block_cfg["hidden_dim"]
        self.rope_dim = qkv_hidden_size // block_cfg["num_heads"]
        self.blocks = nn.ModuleList(
            [KimiK3VisionBlock(**block_cfg) for _ in range(num_layers)]
        )
        if norm_type == "layernorm":
            self.final_layernorm = nn.LayerNorm(hidden_dim)
        elif norm_type == "rmsnorm":
            self.final_layernorm = nn.RMSNorm(hidden_dim)
        else:
            raise NotImplementedError(f"Unsupported norm_type: {norm_type}")

    def forward(
        self, hidden_states: torch.Tensor, grid_thws: torch.Tensor
    ) -> torch.Tensor:
        rope_freqs_cis = rope_2d_freqs_cis(
            grid_thws, self.rope_dim, hidden_states.device
        )
        lengths = torch.cat(
            (
                torch.zeros(1, dtype=grid_thws.dtype, device=grid_thws.device),
                grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
            )
        )
        max_seqlen = int(lengths.max().item())
        cu_seqlens = lengths.to(hidden_states.device).cumsum(dim=0, dtype=torch.int32)

        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis)
        return self.final_layernorm(hidden_states)


def tpool_patch_merge(
    x: torch.Tensor,
    grid_thws: torch.Tensor,
    merge_kernel_size: tuple[int, int] = (2, 2),
) -> torch.Tensor:
    """``sd2_tpool`` merge: 2x2 spatial grouping with temporal mean pooling.

    Returns ``(total_merged_patches, kernel_h * kernel_w * d_model)`` — the
    per-image tensors of the reference implementation concatenated in order,
    which is what the projector consumes.
    """
    kernel_h, kernel_w = merge_kernel_size
    d_model = x.size(-1)
    outputs = []
    offset = 0
    for t, h, w in grid_thws.tolist():
        seq = x[offset : offset + t * h * w]
        new_h, new_w = h // kernel_h, w // kernel_w
        seq = seq.view(t, new_h, kernel_h, new_w, kernel_w, d_model)
        # temporal pooling after grouping each 2x2 spatial block together
        seq = seq.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
        outputs.append(seq.reshape(new_h * new_w, kernel_h * kernel_w * d_model))
        offset += t * h * w
    return torch.cat(outputs, dim=0)


class KimiK3VisionTower(nn.Module):
    """MoonViT3d tower: patch embed -> encoder -> ``sd2_tpool`` merge."""

    def __init__(self, vision_config) -> None:
        super().__init__()
        self.merge_kernel_size = _as_pair(vision_config.merge_kernel_size)
        self.merge_type = getattr(vision_config, "merge_type", "sd2_tpool")
        if self.merge_type != "sd2_tpool":
            raise NotImplementedError(f"Unsupported merge_type: {self.merge_type}")

        hidden_size = vision_config.vt_hidden_size
        self.patch_embed = KimiK3PatchEmbed(
            out_dim=hidden_size,
            patch_size=vision_config.patch_size,
            pos_emb_height=vision_config.init_pos_emb_height,
            pos_emb_width=vision_config.init_pos_emb_width,
            pos_emb_time=vision_config.init_pos_emb_time,
            pos_emb_type=getattr(vision_config, "pos_emb_type", "divided_fixed"),
            patch_embed_proj_bias=getattr(
                vision_config, "patch_embed_proj_bias", False
            ),
            pos_emb_interpolation_mode=getattr(
                vision_config, "pos_emb_interpolation_mode", "bilinear"
            ),
        )
        mlp_type = getattr(vision_config, "mlp_type", "mlp2")
        if mlp_type != "mlp2":
            raise NotImplementedError(f"Unsupported mlp_type: {mlp_type}")
        norm_type = getattr(vision_config, "norm_type", "rmsnorm")
        self.encoder = KimiK3VisionEncoder(
            hidden_dim=hidden_size,
            num_layers=vision_config.vt_num_hidden_layers,
            block_cfg={
                "num_heads": vision_config.vt_num_attention_heads,
                "hidden_dim": hidden_size,
                "qkv_hidden_size": getattr(vision_config, "qkv_hidden_size", None),
                "mlp_dim": vision_config.vt_intermediate_size,
                "norm_type": norm_type,
                "attn_bias": getattr(vision_config, "attn_bias", False),
                "linear_bias": getattr(vision_config, "linear_bias", False),
            },
            norm_type=norm_type,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def forward(
        self, pixel_values: torch.Tensor, grid_thws: torch.Tensor
    ) -> torch.Tensor:
        if grid_thws.ndim != 2 or grid_thws.size(1) != 3:
            raise ValueError(
                f"grid_thws must be (num_images, 3), got {grid_thws.shape}"
            )
        hidden_states = self.patch_embed(
            pixel_values.to(device=self.device, dtype=self.dtype), grid_thws
        )
        hidden_states = self.encoder(hidden_states, grid_thws)
        return tpool_patch_merge(hidden_states, grid_thws, self.merge_kernel_size)


class KimiK3PatchMergerProjector(nn.Module):
    """PatchMergerMLPV2: projects merged vision patches into the text width."""

    def __init__(self, vision_config) -> None:
        super().__init__()
        kernel_h, kernel_w = _as_pair(vision_config.merge_kernel_size)
        mm_hidden_size = (
            getattr(vision_config, "mm_hidden_size", None)
            or vision_config.vt_hidden_size
        )
        self.hidden_size = mm_hidden_size * kernel_h * kernel_w
        text_hidden_size = vision_config.text_hidden_size
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(self.hidden_size, text_hidden_size, bias=False),
        )
        self.post_norm = nn.RMSNorm(
            text_hidden_size, eps=getattr(vision_config, "projector_ln_eps", 1e-5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.post_norm(self.proj(x.view(-1, self.hidden_size)))


def build_vision_modules(vision_config) -> tuple[nn.Module, nn.Module]:
    """Build the ``(vision_tower, mm_projector)`` pair for a Kimi-K3 config."""
    projector_type = getattr(vision_config, "mm_projector_type", "patchmergerv2")
    if projector_type != "patchmergerv2":
        raise NotImplementedError(f"Unsupported mm_projector_type: {projector_type}")
    return KimiK3VisionTower(vision_config), KimiK3PatchMergerProjector(vision_config)
