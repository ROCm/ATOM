# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.3-Flash vision tower (``glm5_next_vision``).

A 24-layer ViT over 14x14 patches (temporal patch 2, so a video frame pair is one
patch row), 2-D rotary position embeddings, packed variable-length attention so a
patch only ever attends inside its own image, then a 2x2 spatial merge and a
projection into the language model's 4096-wide embedding space.

Shapes, for orientation:

    pixel_values [sum(t*h*w), 3*2*14*14]
      -> patch_embed (Conv3d)              [L, 1024]
      -> 24 x block (attn + clamped SwiGLU) [L, 1024]
      -> post_layernorm                     [L, 1024]
      -> 2x2 merge + downsample (Conv2d)    [L/4, 4096]
      -> merger (proj, LN, GELU, SwiGLU)    [L/4, 4096]

The `L/4` output rows are what get scattered onto the image/video placeholder
tokens in the text stream.

Everything is BF16 in the checkpoint and the tower is small next to the language
model, so it is replicated on every TP rank rather than sharded -- the same
choice `kimi_k3_vl.py` makes.
"""

import math
import os
from itertools import pairwise

import torch
import torch.nn.functional as F
from aiter import flash_attn_varlen_func
from torch import nn

__all__ = ["Glm5NextVisionTower", "build_vision_tower"]


class Glm5NextVisionRMSNorm(nn.Module):
    """RMSNorm matching the reference: fp32 accumulate, weight applied after cast."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dtype)


def vision_position_ids(
    grid_thw: torch.Tensor, spatial_merge_size: int
) -> torch.Tensor:
    """(h, w) index per patch, laid out block-major over ``m x m`` merge blocks.

    Port of `transformers.vision_utils.get_vision_position_ids` for the 2-axis
    case. The block-major ordering is what makes the later `view(-1, m, m, C)`
    merge pick up spatially adjacent patches.
    """
    device = grid_thw.device
    out = []
    m = spatial_merge_size
    for t, h, w in grid_thw.tolist():
        hpos, wpos = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        block = (h // m, m, w // m, m)
        hpos = hpos.reshape(block).transpose(1, 2).flatten()
        wpos = wpos.reshape(block).transpose(1, 2).flatten()
        out.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
    return torch.cat(out, dim=0)


def vision_cu_seqlens(grid_thw: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Per-frame attention segments: one segment of ``h*w`` per frame."""
    seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    cu = F.pad(seqlens.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0)
    return cu, int(seqlens.max()) if seqlens.numel() else 0


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q_dtype, k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    return (
        ((q * cos) + (_rotate_half(q) * sin)).to(q_dtype),
        ((k * cos) + (_rotate_half(k) * sin)).to(k_dtype),
    )


# GLM53_VISION_ATTN=torch swaps the packed aiter kernel for per-segment SDPA.
# Only for bisecting numerics against the transformers reference, which takes
# the SDPA path -- it is slower and allocates per segment.
_USE_TORCH_ATTN = os.environ.get("GLM53_VISION_ATTN") == "torch"


def _sdpa_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Non-causal SDPA run separately per packed segment. [L, H, D] in and out."""
    outs = []
    for start, end in pairwise(cu_seqlens.tolist()):
        if end <= start:
            continue
        qs, ks, vs = (t[start:end].transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        o = F.scaled_dot_product_attention(qs, ks, vs, scale=scale, is_causal=False)
        outs.append(o.squeeze(0).transpose(0, 1))
    return torch.cat(outs, dim=0)


def _clamped_swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """GLM's clamped SwiGLU: gate clamped above only, up clamped both ways."""
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


def _fused_gate_up(in_features: int, out_features: int, bias: bool) -> nn.Linear:
    """A `[gate | up]` Linear that ATOM's loader can fill one shard at a time.

    The model's `packed_modules_mapping` rewrites `.gate_proj` / `.up_proj` to
    `.gate_up_proj` by substring, so it applies to this tower too whether or not
    the projections are fused here -- leaving them separate silently drops those
    tensors at load. Fusing is therefore the fix, and a `weight_loader` taking
    `(param, tensor, shard_id)` is all the loader needs. Done with a plain
    nn.Linear rather than `MergedReplicatedLinear` so the tower stays usable
    without an initialised TP group (the parity harness has none).
    """
    layer = nn.Linear(in_features, 2 * out_features, bias=bias)

    def loader(param: nn.Parameter, loaded: torch.Tensor, shard_id: int) -> None:
        size = param.shape[0] // 2
        param.data[shard_id * size : (shard_id + 1) * size].copy_(loaded)

    layer.weight.weight_loader = loader
    if bias:
        layer.bias.weight_loader = loader
    return layer


class Glm5NextVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.in_channels = config.in_channels
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.embed_dim = config.hidden_size
        kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel, stride=kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(x.to(self.proj.weight.dtype)).view(-1, self.embed_dim)


class Glm5NextVisionAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        bias = bool(config.attention_bias)
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
        self.q_norm = Glm5NextVisionRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Glm5NextVisionRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        length = x.shape[0]
        q, k, v = (
            self.qkv(x)
            .reshape(length, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = position_embeddings
        q, k = _apply_rope_vision(q, k, cos, sin)

        scale = 1.0 / math.sqrt(self.head_dim)
        if _USE_TORCH_ATTN:
            out = _sdpa_varlen(q, k, v, cu_seqlens, scale)
        else:
            out = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                softmax_scale=scale,
                causal=False,
            )
            if isinstance(out, tuple):
                out = out[0]
        return self.proj(out.reshape(length, -1))


class Glm5NextVisionMLP(nn.Module):
    """Clamped-SwiGLU FFN.

    See `_fused_gate_up` for why gate and up are fused here.
    """

    def __init__(self, config) -> None:
        super().__init__()
        bias = bool(config.attention_bias)
        self.gate_up_proj = _fused_gate_up(
            config.hidden_size, config.intermediate_size, bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=bias
        )
        self.swiglu_limit = float(config.swiglu_limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(_clamped_swiglu(gate, up, self.swiglu_limit))


class Glm5NextVisionBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = Glm5NextVisionRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = Glm5NextVisionRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Glm5NextVisionAttention(config)
        self.mlp = Glm5NextVisionMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cu_seqlens, max_seqlen, position_embeddings)
        return x + self.mlp(self.norm2(x))


class Glm5NextVisionPatchMerger(nn.Module):
    """Projects merged patches into the language model's embedding space."""

    def __init__(self, dim: int, context_dim: int, swiglu_limit: float) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.post_projection_norm = nn.LayerNorm(dim)
        # Fused for the same reason as Glm5NextVisionMLP.
        self.gate_up_proj = _fused_gate_up(dim, context_dim, bias=False)
        self.down_proj = nn.Linear(context_dim, dim, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = F.gelu(self.post_projection_norm(x))
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(_clamped_swiglu(gate, up, self.swiglu_limit))


class Glm5NextVisionTower(nn.Module):
    """Full tower: patches in, language-model-width embeddings out."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = int(config.spatial_merge_size)
        self.out_hidden_size = int(config.out_hidden_size)
        head_dim = config.hidden_size // config.num_heads
        # 2-D rope: half the head dim per axis, so (h, w) fills head_dim/2 and
        # the cat below widens it back to head_dim.
        self.rotary_dim = head_dim // 2
        theta = float(getattr(config, "rope_theta", 10000.0))
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.patch_embed = Glm5NextVisionPatchEmbed(config)
        self.blocks = nn.ModuleList(
            [Glm5NextVisionBlock(config) for _ in range(config.depth)]
        )
        self.post_layernorm = Glm5NextVisionRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.downsample = nn.Conv2d(
            config.hidden_size,
            config.out_hidden_size,
            kernel_size=self.spatial_merge_size,
            stride=self.spatial_merge_size,
        )
        self.merger = Glm5NextVisionPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.projection_intermediate_size,
            swiglu_limit=float(config.swiglu_limit),
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        grid_thw = grid_thw.to(pixel_values.device)
        position_ids = vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens, max_seqlen = vision_cu_seqlens(grid_thw)

        x = self.patch_embed(pixel_values.to(self.dtype))

        freqs = (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)
        emb = torch.cat((freqs, freqs), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        for block in self.blocks:
            x = block(x, cu_seqlens, max_seqlen, position_embeddings)
        x = self.post_layernorm(x)

        # 2x2 spatial merge. The block-major position layout above is what makes
        # these four rows spatially adjacent.
        m = self.spatial_merge_size
        x = x.view(-1, m, m, x.shape[-1]).permute(0, 3, 1, 2)
        x = self.downsample(x).view(-1, self.out_hidden_size)
        return self.merger(x)


def build_vision_tower(vision_config) -> Glm5NextVisionTower:
    return Glm5NextVisionTower(vision_config)
