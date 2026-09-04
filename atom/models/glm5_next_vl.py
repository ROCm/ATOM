# SPDX-License-Identifier: MIT
"""Native GLM-5.3-Flash vision tower for ATOM."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)).to(
            dtype
        ) * self.weight


def swiglu_clamped(gate, up, limit: float):
    return F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class PackedGateUp(nn.Linear):
    """Replicated gate/up projection with ATOM's packed-weight loader hook."""

    def __init__(self, in_features, hidden_features, bias):
        super().__init__(in_features, 2 * hidden_features, bias=bias)
        self.weight.weight_loader = self._load
        if self.bias is not None:
            self.bias.weight_loader = self._load

    @staticmethod
    def _load(param, loaded_weight, shard_id):
        shard = param.shape[0] // 2
        start = int(shard_id) * shard
        param.data[start : start + shard].copy_(loaded_weight)


class VisionMLP(nn.Module):
    def __init__(self, dim, hidden, limit, bias=True):
        super().__init__()
        self.gate_up_proj = PackedGateUp(dim, hidden, bias=bias)
        self.down_proj = nn.Linear(hidden, dim, bias=bias)
        self.limit = limit

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, -1)
        return self.down_proj(swiglu_clamped(gate, up, self.limit))


class VisionAttention(nn.Module):
    def __init__(self, dim, heads, eps):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.q_norm = RMSNorm(self.head_dim, eps)
        self.k_norm = RMSNorm(self.head_dim, eps)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, cos, sin, lengths):
        n = x.shape[0]
        q, k, v = self.qkv(x).view(n, 3, self.heads, self.head_dim).unbind(1)
        q, k = self.q_norm(q), self.k_norm(k)
        rotary_dim = cos.shape[-1]
        qrot, krot = q[..., :rotary_dim], k[..., :rotary_dim]
        cos, sin = cos[:, None].to(q.dtype), sin[:, None].to(q.dtype)
        q = torch.cat((qrot * cos + rotate_half(qrot) * sin, q[..., rotary_dim:]), -1)
        k = torch.cat((krot * cos + rotate_half(krot) * sin, k[..., rotary_dim:]), -1)
        outputs, start = [], 0
        for length in lengths:
            end = start + int(length)
            qs = q[start:end].transpose(0, 1).unsqueeze(0)
            ks = k[start:end].transpose(0, 1).unsqueeze(0)
            vs = v[start:end].transpose(0, 1).unsqueeze(0)
            output = F.scaled_dot_product_attention(qs, ks, vs)
            outputs.append(output.squeeze(0).transpose(0, 1))
            start = end
        return self.proj(torch.cat(outputs).reshape(n, -1))


class VisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, eps = config.hidden_size, config.rms_norm_eps
        self.norm1, self.norm2 = RMSNorm(d, eps), RMSNorm(d, eps)
        self.attn = VisionAttention(d, config.num_heads, eps)
        self.mlp = VisionMLP(d, config.intermediate_size, config.swiglu_limit)

    def forward(self, x, cos, sin, lengths):
        x = x + self.attn(self.norm1(x), cos, sin, lengths)
        return x + self.mlp(self.norm2(x))


class PatchMerger(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, h = config.out_hidden_size, config.projection_intermediate_size
        self.proj = nn.Linear(d, d, bias=False)
        self.post_projection_norm = nn.LayerNorm(d)
        self.gate_up_proj = PackedGateUp(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)
        self.limit = config.swiglu_limit

    def forward(self, x):
        x = F.gelu(self.post_projection_norm(self.proj(x)))
        gate, up = self.gate_up_proj(x).chunk(2, -1)
        return self.down_proj(swiglu_clamped(gate, up, self.limit))


class Glm5NextVisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size, self.num_heads = config.hidden_size, config.num_heads
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.spatial_merge_size = config.spatial_merge_size
        kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv3d(
            config.in_channels, config.hidden_size, kernel, kernel, bias=True
        )
        self.blocks = nn.ModuleList(VisionBlock(config) for _ in range(config.depth))
        self.post_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.downsample = nn.Conv2d(
            config.hidden_size,
            config.out_hidden_size,
            config.spatial_merge_size,
            config.spatial_merge_size,
        )
        self.merger = PatchMerger(config)

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self):
        return self.patch_embed.proj.weight.device

    def _rotary(self, grids):
        ids, lengths, merge = [], [], self.spatial_merge_size
        for t, h, w in grids:
            t, h, w = int(t), int(h), int(w)
            hp = (
                torch.arange(h)
                .view(h, 1)
                .expand(h, w)
                .reshape(h // merge, merge, w // merge, merge)
                .permute(0, 2, 1, 3)
                .flatten()
            )
            wp = (
                torch.arange(w)
                .view(1, w)
                .expand(h, w)
                .reshape(h // merge, merge, w // merge, merge)
                .permute(0, 2, 1, 3)
                .flatten()
            )
            ids.append(torch.stack((hp, wp), -1).repeat(t, 1))
            lengths.extend([h * w] * t)
        pos = torch.cat(ids).to(self.device)
        rotary_half = (self.hidden_size // self.num_heads) // 4
        inv = 1.0 / (
            10000
            ** (torch.arange(0, rotary_half, device=self.device).float() / rotary_half)
        )
        freq = pos.float()[..., None] * inv
        emb = torch.cat((freq[:, 0], freq[:, 1]), -1)
        return (
            torch.cat((emb.cos(), emb.cos()), -1),
            torch.cat((emb.sin(), emb.sin()), -1),
            lengths,
        )

    def forward(self, x, grid_thw):
        x = x.to(self.device, self.dtype)
        x = self.patch_embed.proj(
            x.view(
                x.shape[0],
                -1,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            )
        ).view(x.shape[0], -1)
        cos, sin, lengths = self._rotary(grid_thw.tolist())
        for block in self.blocks:
            x = block(x, cos, sin, lengths)
        x = (
            self.post_layernorm(x)
            .view(
                -1, self.spatial_merge_size, self.spatial_merge_size, self.hidden_size
            )
            .permute(0, 3, 1, 2)
        )
        return self.merger(self.downsample(x).flatten(1))
