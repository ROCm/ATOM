# SPDX-License-Identifier: Apache-2.0
"""Flux Diffusion Transformer model."""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from aiter.dist.parallel_state import get_tensor_model_parallel_world_size
from atom.model_ops.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    MergedColumnParallelLinear,
)
from atom.config import QuantizationConfig, Config
from atom.utils.decorators import support_torch_compile
from atom.models.utils import maybe_prefix


def timestep_embedding(
    t: torch.Tensor, dim: int, max_period: int = 10000
) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_dim = freq_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.freq_dim))


class FluxRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * (
            x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        ).to(x.dtype)


class FluxMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class FluxAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = num_heads // tp_size

        self.qkv = ColumnParallelLinear(
            hidden_size,
            3 * hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
        )
        self.proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
        )
        self.q_norm = FluxRMSNorm(self.head_dim)
        self.k_norm = FluxRMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads_per_partition, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn = F.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.proj(out)


class FluxSingleBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.norm1 = FluxRMSNorm(hidden_size)
        self.attn = FluxAttention(
            hidden_size, num_heads, quant_config, f"{prefix}.attn"
        )
        self.norm2 = FluxRMSNorm(hidden_size)
        self.mlp = FluxMLP(
            hidden_size, int(hidden_size * mlp_ratio), quant_config, f"{prefix}.mlp"
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(temb).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = gate1.unsqueeze(1) * self.attn(h) + x
        h = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        return gate2.unsqueeze(1) * self.mlp(h) + x


class FluxDoubleBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        mlp_dim = int(hidden_size * mlp_ratio)
        self.img_norm1, self.img_norm2 = FluxRMSNorm(hidden_size), FluxRMSNorm(
            hidden_size
        )
        self.txt_norm1, self.txt_norm2 = FluxRMSNorm(hidden_size), FluxRMSNorm(
            hidden_size
        )
        self.img_attn = FluxAttention(
            hidden_size, num_heads, quant_config, f"{prefix}.img_attn"
        )
        self.img_mlp = FluxMLP(hidden_size, mlp_dim, quant_config, f"{prefix}.img_mlp")
        self.txt_mlp = FluxMLP(hidden_size, mlp_dim, quant_config, f"{prefix}.txt_mlp")
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(
        self, img: torch.Tensor, txt: torch.Tensor, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_m = self.img_mod(temb).chunk(6, dim=-1)
        txt_m = self.txt_mod(temb).chunk(6, dim=-1)

        img_h = self.img_norm1(img) * (1 + img_m[1].unsqueeze(1)) + img_m[0].unsqueeze(
            1
        )
        txt_h = self.txt_norm1(txt) * (1 + txt_m[1].unsqueeze(1)) + txt_m[0].unsqueeze(
            1
        )

        joint = torch.cat([txt_h, img_h], dim=1)
        joint_out = self.img_attn(joint)
        txt_out, img_out = joint_out.split([txt_h.shape[1], img_h.shape[1]], dim=1)

        img = img_m[2].unsqueeze(1) * img_out + img
        txt = txt_m[2].unsqueeze(1) * txt_out + txt

        img_h = self.img_norm2(img) * (1 + img_m[4].unsqueeze(1)) + img_m[3].unsqueeze(
            1
        )
        txt_h = self.txt_norm2(txt) * (1 + txt_m[4].unsqueeze(1)) + txt_m[3].unsqueeze(
            1
        )

        return (
            img_m[5].unsqueeze(1) * self.img_mlp(img_h) + img,
            txt_m[5].unsqueeze(1) * self.txt_mlp(txt_h) + txt,
        )


class FluxPatchEmbed(nn.Module):
    def __init__(self, in_ch: int = 16, hidden_size: int = 3072, patch_size: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_ch * patch_size * patch_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        x = (
            x.unfold(2, p, p)
            .unfold(3, p, p)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(B, -1, C * p * p)
        )
        return self.proj(x)


class FluxFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_ch: int):
        super().__init__()
        self.norm = FluxRMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_ch)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(temb).chunk(2, dim=-1)
        return self.linear(self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))


@support_torch_compile
class FluxTransformer(nn.Module):
    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__()
        config = atom_config.hf_config
        quant_config = atom_config.quant_config

        self.hidden_size = getattr(config, "hidden_size", 3072)
        self.num_heads = getattr(config, "num_attention_heads", 24)
        self.patch_size = getattr(config, "patch_size", 2)
        self.in_channels = getattr(config, "in_channels", 16)
        self.out_channels = getattr(config, "out_channels", 16)
        num_double = getattr(config, "num_double_layers", 19)
        num_single = getattr(config, "num_single_layers", 38)
        text_dim = getattr(config, "text_hidden_size", 4096)

        self.img_embed = FluxPatchEmbed(
            self.in_channels, self.hidden_size, self.patch_size
        )
        self.txt_embed = nn.Linear(text_dim, self.hidden_size)
        self.time_embed = TimestepEmbedder(self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                FluxDoubleBlock(
                    self.hidden_size,
                    self.num_heads,
                    quant_config=quant_config,
                    prefix=f"{prefix}.double_blocks.{i}",
                )
                for i in range(num_double)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                FluxSingleBlock(
                    self.hidden_size,
                    self.num_heads,
                    quant_config=quant_config,
                    prefix=f"{prefix}.single_blocks.{i}",
                )
                for i in range(num_single)
            ]
        )
        self.final_layer = FluxFinalLayer(
            self.hidden_size, self.patch_size, self.out_channels
        )

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: torch.Tensor,
        guidance_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, H, W = latents.shape
        img = self.img_embed(latents)
        txt = self.txt_embed(text_embeddings)
        temb = self.time_embed(timesteps)

        for block in self.double_blocks:
            img, txt = block(img, txt, temb)

        x = torch.cat([txt, img], dim=1)
        for block in self.single_blocks:
            x = block(x, temb)

        img = x[:, txt.shape[1] :]
        out = self.final_layer(img, temb)

        p = self.patch_size
        return (
            out.view(B, H // p, W // p, p, p, self.out_channels)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(B, self.out_channels, H, W)
        )


class FluxForImageGeneration(nn.Module):
    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__()
        self.transformer = FluxTransformer(
            atom_config, maybe_prefix(prefix, "transformer")
        )

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: torch.Tensor,
        guidance_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.transformer(latents, timesteps, text_embeddings, guidance_scale)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states
