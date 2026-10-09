# SPDX-License-Identifier: MIT
"""V4.1 image encoder, adapted from the released DeepSeek reference.

Each image has its own bidirectional attention domain. Vision RoPE rotates
halves in two dimensions; it is independent of language-side YaRN.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from atom.model_ops.linear import ReplicatedLinear
from atom.model_ops.utils import atom_parameter


@lru_cache(8)
def get_vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float, device):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    hpos = torch.arange(n_h, device=device).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w, device=device).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = atom_parameter(torch.ones(dim, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class PatchEmbed(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.proj = ReplicatedLinear(
            3 * args.patch_size**2, args.hidden_size, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.wqkv = ReplicatedLinear(args.hidden_size, 3 * args.hidden_size, bias=True)
        self.wo = ReplicatedLinear(args.hidden_size, args.hidden_size, bias=True)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        n = x.size(0)
        q, k, v = (
            t.view(n, self.n_heads, self.head_dim)
            for t in self.wqkv(x).chunk(3, dim=-1)
        )
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        )
        return self.wo(o.transpose(0, 1).reshape(n, -1))


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.w1 = ReplicatedLinear(
            args.hidden_size, 2 * args.intermediate_size, bias=False
        )
        self.w2 = ReplicatedLinear(args.intermediate_size, args.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm1 = RMSNorm(args.hidden_size)
        self.attn = Attention(args)
        self.norm2 = RMSNorm(args.hidden_size)
        self.mlp = MLP(args)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """DeepSeek ViT: full bidirectional attention over one image with 2D RoPE."""

    def __init__(self, args):
        super().__init__()
        self.rope_dim = args.hidden_size // args.num_attention_heads // 2
        self.rope_theta = args.rope_theta
        self.patch_embed = PatchEmbed(args)
        self.blocks = nn.ModuleList(
            [Block(args) for _ in range(args.num_hidden_layers)]
        )
        self.norm = RMSNorm(args.hidden_size)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(
            n_h, n_w, self.rope_dim, self.rope_theta, x.device
        )
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class Aligner(nn.Module):
    def __init__(self, args, language_dim):
        super().__init__()
        self.downsample_ratio = args.downsample_ratio
        in_dim = args.hidden_size * self.downsample_ratio**2
        self.w1 = ReplicatedLinear(in_dim, language_dim, bias=True)
        self.w2 = ReplicatedLinear(language_dim, language_dim, bias=True)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))
