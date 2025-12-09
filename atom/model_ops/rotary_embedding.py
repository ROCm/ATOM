# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from functools import lru_cache
import torch
from torch import nn
import aiter
from aiter import dtypes
from typing import Union, Optional

def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if dtype is None:
            dtype = torch.get_default_dtype()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cos, sin = self._compute_cos_sin_cache()
        cos = cos.to(dtype)
        sin = sin.to(dtype)
        self.cos_cache: torch.Tensor
        self.sin_cache: torch.Tensor
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

        assert rotary_dim == head_size

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=dtypes.fp32) / self.rotary_dim
            )
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=dtypes.fp32)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos().unsqueeze(-2).unsqueeze(-2)
        sin = freqs.sin().unsqueeze(-2).unsqueeze(-2)
        return cos, sin

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        is_nope_first = False
        self.cos_cache = self.cos_cache.to(query.device, dtype=query.dtype)
        self.sin_cache = self.sin_cache.to(query.device, dtype=query.dtype)
        cos, sin = self.cos_cache, self.sin_cache

        rotate_style = 0 if self.is_neox_style else 1

        num_tokens = positions.numel()

        query_shape = query.shape
        query = query.view(1, num_tokens, -1, self.head_size)
        if key is not None:
            key_shape = key.shape
            key = key.view(1, num_tokens, -1, self.head_size)

        positions = positions.view(*query.shape[:2])

        if not is_nope_first:
            query_ = query[..., : self.rotary_dim]
            key_ = key[..., : self.rotary_dim] if key is not None else None
        else:
            query_ = query[..., -self.rotary_dim :]
            key_ = key[..., -self.rotary_dim :] if key is not None else None
        
        aiter.rope_cached_positions_2c_fwd_inplace(
            query_,
            key_,
            cos,
            sin,
            positions,
            rotate_style,
            reuse_freqs_front_part=True,
            nope_first=is_nope_first,
        )
        query = query.view(query_shape)

        key = key.view(key_shape)

        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
