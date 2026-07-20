# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Native Qwen3.5 Vision Encoder for ATOM.

Pure PyTorch implementation (no vLLM dependencies) of the Qwen3 VisionTransformer.
Uses F.scaled_dot_product_attention for the vision self-attention blocks.
"""

from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Qwen3VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [L, C * temporal_patch_size * patch_size * patch_size]
        L, C = x.shape
        x = x.view(L, -1, self.temporal_patch_size, self.patch_size, self.patch_size)
        x = self.proj(x).view(L, self.hidden_size)
        return x


class Qwen3VisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = True,
        act_fn=F.silu,
    ):
        super().__init__()
        self.linear_fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.linear_fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Qwen3VisionAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Combined QKV projection (matching checkpoint weight name "attn.qkv")
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
    ) -> torch.Tensor:
        # x: [seq_len, 1, embed_dim] (the VisionBlock adds a batch dim)
        seq_len = x.shape[0]
        batch = x.shape[1] if x.dim() == 3 else 1
        x_2d = x.view(seq_len, self.embed_dim)

        qkv = self.qkv(x_2d)
        qkv = qkv.view(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Apply rotary embeddings
        q = self._apply_rotary_emb(q, rotary_pos_emb_cos, rotary_pos_emb_sin)
        k = self._apply_rotary_emb(k, rotary_pos_emb_cos, rotary_pos_emb_sin)

        # Reshape for SDPA: [batch, heads, seq_len, head_dim]
        q = q.unsqueeze(0).transpose(1, 2)  # [1, num_heads, seq_len, head_dim]
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(seq_len, self.embed_dim)
        out = self.proj(out)
        return out.view(seq_len, batch, self.embed_dim)

    @staticmethod
    def _apply_rotary_emb(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        # x: [seq_len, num_heads, head_dim]
        # cos, sin: [seq_len, head_dim]
        # Use rotate_half style matching HF's apply_rotary_pos_emb_vision
        orig_dtype = x.dtype
        x = x.float()
        cos = cos.unsqueeze(-2).float()  # [seq_len, 1, head_dim]
        sin = sin.unsqueeze(-2).float()  # [seq_len, 1, head_dim]
        # rotate_half: [-x2, x1] where x1 = x[..., :dim//2], x2 = x[..., dim//2:]
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        x_rotated = torch.cat((-x2, x1), dim=-1)
        out = (x * cos) + (x_rotated * sin)
        return out.to(orig_dtype)


class Qwen3VisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn=F.silu,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = Qwen3VisionAttention(embed_dim=dim, num_heads=num_heads)
        self.mlp = Qwen3VisionMLP(dim, mlp_hidden_dim, act_fn=act_fn, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
        )
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen3VisionPatchMerger(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.norm = nn.LayerNorm(context_dim, eps=norm_eps)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq_len, 1, context_dim]
        x = self.norm(x).view(-1, self.hidden_size)
        x = self.linear_fc1(x)
        x = self.act_fn(x)
        x = self.linear_fc2(x)
        return x


class Qwen3VisionTransformer(nn.Module):
    def __init__(
        self,
        vision_config,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.num_position_embeddings = vision_config.num_position_embeddings
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.deepstack_visual_indexes = getattr(
            vision_config, "deepstack_visual_indexes", []
        )
        self.num_grid_per_side = int(self.num_position_embeddings**0.5)

        self.patch_embed = Qwen3VisionPatchEmbed(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=vision_config.in_channels,
            hidden_size=self.hidden_size,
        )

        self.pos_embed = nn.Embedding(self.num_position_embeddings, self.hidden_size)

        head_dim = self.hidden_size // self.num_heads
        self.head_dim = head_dim
        self.rotary_dim = head_dim  # full rotation by default

        act_fn_map = {
            "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
            "silu": F.silu,
            "gelu": F.gelu,
        }
        act_fn = act_fn_map.get(getattr(vision_config, "hidden_act", "silu"), F.silu)

        self.blocks = nn.ModuleList(
            [
                Qwen3VisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_hidden_dim=vision_config.intermediate_size,
                    act_fn=act_fn,
                    norm_eps=norm_eps,
                )
                for _ in range(vision_config.depth)
            ]
        )

        self.merger = Qwen3VisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            spatial_merge_size=self.spatial_merge_size,
            norm_eps=norm_eps,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    @staticmethod
    @lru_cache(maxsize=1024)
    def rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
        hpos_ids = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
        h_div = h // spatial_merge_size
        w_div = w // spatial_merge_size
        hpos_ids = hpos_ids.reshape(
            h_div, spatial_merge_size, w_div, spatial_merge_size
        )
        hpos_ids = hpos_ids.transpose(0, 2, 1, 3).flatten()

        wpos_ids = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
        wpos_ids = wpos_ids.reshape(
            h_div, spatial_merge_size, w_div, spatial_merge_size
        )
        wpos_ids = wpos_ids.transpose(0, 2, 1, 3).flatten()

        return torch.from_numpy(np.stack([hpos_ids, wpos_ids], axis=-1))

    def _compute_rotary_emb(self, grid_thw: list[list[int]]):
        """Compute rotary position embeddings for the vision encoder.

        Matches HuggingFace Qwen3_5MoeVisionModel.rot_pos_emb:
        - Uses head_dim // 2 as the rotary dimension (head_dim // 4 frequencies)
        - Concatenates h and w frequency tables → head_dim // 2
        - Doubles: cat((emb, emb)) → head_dim
        - Returns (cos, sin) each of shape (seq_len, head_dim)
        """
        max_grid_size = max(max(h, w) for _, h, w in grid_thw)

        # Rotary dim is head_dim // 2 (matching HF's VisionRotaryEmbedding(head_dim // 2))
        rotary_dim = self.head_dim // 2
        inv_freq = 1.0 / (
            10000.0
            ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        positions = torch.arange(max_grid_size, dtype=torch.float32)
        freq_table = torch.outer(positions, inv_freq)  # (max_hw, head_dim // 4)

        # Build position IDs for each image/frame
        pos_ids = []
        for t, h, w in grid_thw:
            ids = self.rot_pos_ids(h, w, self.spatial_merge_size)
            if t > 1:
                ids = ids.repeat(t, 1)
            pos_ids.append(ids)
        pos_ids = torch.cat(pos_ids, dim=0)  # [total_patches, 2] (CPU)

        # Index into freq table: (seq_len, 2, head_dim // 4) → flatten → (seq_len, head_dim // 2)
        embeddings = freq_table[pos_ids]  # (seq_len, 2, head_dim // 4)
        embeddings = embeddings.flatten(1)  # (seq_len, head_dim // 2)

        # Double the embeddings: cat((emb, emb)) → (seq_len, head_dim)
        emb = torch.cat((embeddings, embeddings), dim=-1)
        cos = emb.cos().to(self.device, dtype=self.dtype)
        sin = emb.sin().to(self.device, dtype=self.dtype)

        return cos, sin

    def _compute_pos_embed(self, grid_thw: list[list[int]]) -> torch.Tensor:
        """Compute interpolated position embeddings."""
        num_grid_per_side = self.num_grid_per_side
        m_size = self.spatial_merge_size
        hidden_dim = self.pos_embed.embedding_dim

        outputs = []
        for t, h, w in grid_thw:
            h_idxs = torch.linspace(
                0, num_grid_per_side - 1, h, dtype=torch.float32, device=self.device
            )
            w_idxs = torch.linspace(
                0, num_grid_per_side - 1, w, dtype=torch.float32, device=self.device
            )

            h_floor = h_idxs.to(torch.long)
            w_floor = w_idxs.to(torch.long)
            h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
            w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
            h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
            h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1 - dh_grid - w01

            h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
            w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
            h_grid_idx = h_grid * num_grid_per_side

            indices = (h_grid_idx + w_grid).reshape(4, -1)
            weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
            weights = weights.to(dtype=self.dtype)

            embeds = self.pos_embed(indices)
            embeds *= weights
            combined = embeds.sum(dim=0)

            combined = combined.reshape(
                h // m_size, m_size, w // m_size, m_size, hidden_dim
            )
            combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
            repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
            outputs.append(repeated)

        return torch.cat(outputs, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: pixel_values tensor [total_patches, C*temporal*patch*patch]
            grid_thw: tensor of shape [num_images, 3] with (t, h, w) for each image
        Returns:
            image embeddings [num_merged_patches, out_hidden_size]
        """
        hidden_states = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
        hidden_states = self.patch_embed(hidden_states)

        if isinstance(grid_thw, torch.Tensor):
            grid_thw_list = grid_thw.tolist()
        else:
            grid_thw_list = grid_thw

        # Position embeddings
        pos_embeds = self._compute_pos_embed(grid_thw_list)
        hidden_states = hidden_states + pos_embeds

        # Add batch dim for attention blocks
        hidden_states = hidden_states.unsqueeze(1)

        # Rotary position embeddings
        rotary_cos, rotary_sin = self._compute_rotary_emb(grid_thw_list)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                rotary_pos_emb_cos=rotary_cos,
                rotary_pos_emb_sin=rotary_sin,
            )

        hidden_states = self.merger(hidden_states)
        return hidden_states
