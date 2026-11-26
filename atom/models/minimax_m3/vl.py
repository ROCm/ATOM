# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Native vision stack for MiniMax-M3.1 (``minimax_m3_vl``) and the VL model.

Pure PyTorch port of the reference CLIP-style tower with 3D RoPE, followed by
the multi-modal projector and the 2x2 patch-merge MLP. Module and parameter
names mirror the checkpoint (``vision_tower.vision_model.*``,
``multi_modal_projector.*``, ``patch_merge_mlp.*``) with one exception: the
attention's q/k/v projections are fused into ``qkv_proj``. The model's packed
mapping already renames every ``.q_proj``/``.k_proj``/``.v_proj`` checkpoint
tensor to ``.qkv_proj``, so the fused parameter carries a shard loader.

The stack is replicated on every TP rank rather than sharded: at ~0.6B params
it is small next to the language model, and replication keeps the image
embeddings identical across ranks (they are scattered into the token
embeddings before the first collective).
"""

import math
from typing import ClassVar

import torch
import torch.nn.functional as F
from aiter import flash_attn_varlen_func
from aiter.dist.parallel_state import get_pp_group
from torch import nn

from atom.config import Config
from atom.models.minimax_m3.image_processing import (
    image_input_refusal,
    image_token_id,
)
from atom.models.minimax_m3.model import (
    MiniMaxM3SparseForConditionalGenerationTextOnly,
)
from atom.models.utils import PPMissingLayer


def _compression(config) -> dict:
    return getattr(config, "img_token_compression_config", None) or {}


def _load_qkv_shard(param: nn.Parameter, loaded_weight: torch.Tensor, shard_id):
    """Copy one of the checkpoint's q/k/v tensors into the fused projection."""
    index = "qkv".index(shard_id)
    size = param.shape[0] // 3
    param.data[index * size : (index + 1) * size].copy_(loaded_weight)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the leading ``rot_dim`` head dims in fp32; the tail passes through."""
    rot_dim = cos.shape[-1]
    x_rot = x[..., :rot_dim].float()
    x_rot = x_rot * cos + _rotate_half(x_rot) * sin
    return torch.cat((x_rot.to(x.dtype), x[..., rot_dim:]), dim=-1)


def _varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    """Non-causal packed attention; a patch attends only within its segment."""
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


class MiniMaxM3VisionEmbeddings(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.num_channels = getattr(config, "num_channels", 3)
        self.patch_size = config.patch_size
        self.temporal_patch_size = _compression(config).get("temporal_patch_size", 2)
        kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.patch_embedding = nn.Conv3d(
            self.num_channels,
            config.hidden_size,
            kernel_size=kernel,
            stride=kernel,
            bias=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Kernel == stride over a single patch, so the conv is one matmul over
        # the processor's flattened (C, T, P, P) patch.
        weight = self.patch_embedding.weight
        return F.linear(pixel_values.to(weight.dtype), weight.flatten(1))


class MiniMaxM3VisionAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.qkv_proj.weight.weight_loader = _load_qkv_shard
        self.qkv_proj.bias.weight_loader = _load_qkv_shard
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(x).view(x.shape[0], 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        out = _varlen_attention(
            q.contiguous(), k.contiguous(), v.contiguous(), cu_seqlens, max_seqlen
        )
        return self.out_proj(out)


class MiniMaxM3VisionMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class MiniMaxM3VisionEncoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.self_attn = MiniMaxM3VisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MiniMaxM3VisionMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x, cos, sin, cu_seqlens, max_seqlen) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x), cos, sin, cu_seqlens, max_seqlen)
        return x + self.mlp(self.layer_norm2(x))


class MiniMaxM3VisionEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            MiniMaxM3VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)
        )


class MiniMaxM3VisionTransformer(nn.Module):
    """CLIP tower with 3D RoPE; returns the last layer (no post-layernorm)."""

    def __init__(self, config) -> None:
        super().__init__()
        self.embeddings = MiniMaxM3VisionEmbeddings(config)
        # "layrnorm" is the checkpoint's spelling.
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = MiniMaxM3VisionEncoder(config)

        self.spatial_merge_size = _compression(config).get("spatial_merge_size", 2)
        self.max_frames = getattr(config, "vision_segment_max_frames", None)
        self.rope_theta = float(getattr(config, "rope_theta", 10000.0))
        head_dim = config.hidden_size // config.num_attention_heads
        # t/h/w each get the same even share of the head dim; leftover dims
        # (2 of 80 here) are not rotated.
        self.axis_dim = 2 * ((2 * (head_dim // 2) // 3) // 2)
        self._inv_freq: torch.Tensor | None = None

    def _segments(self, grid_thw: list[list[int]]) -> list[list[int]]:
        """Videos longer than `vision_segment_max_frames` attend per segment."""
        if not self.max_frames:
            return grid_thw
        segments = []
        for t, h, w in grid_thw:
            for start in range(0, t, self.max_frames):
                segments.append([min(self.max_frames, t - start), h, w])
        return segments

    def _rope_cos_sin(
        self, segments: list[list[int]], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Kept off the module's buffers so a dtype cast of the model cannot
        # round the frequencies to bf16.
        if self._inv_freq is None or self._inv_freq.device != device:
            exponent = (
                torch.arange(0, self.axis_dim, 2, dtype=torch.float32, device=device)
                / self.axis_dim
            )
            self._inv_freq = 1.0 / (self.rope_theta**exponent)
        inv_freq = self._inv_freq
        m = self.spatial_merge_size
        positions = []
        for t, h, w in segments:
            # Patches arrive in merge-window order: (h/m, w/m, m, m) per frame.
            hpos = torch.arange(h, device=device).view(h, 1).expand(h, w)
            wpos = torch.arange(w, device=device).view(1, w).expand(h, w)
            hpos = hpos.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
            wpos = wpos.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
            tpos = torch.arange(t, device=device).repeat_interleave(h * w)
            positions.append(torch.stack((tpos, hpos.repeat(t), wpos.repeat(t)), -1))
        pos = torch.cat(positions).float()
        freqs = (pos[:, :, None] * inv_freq).flatten(1)  # (L, 3 * axis_dim / 2)
        freqs = torch.cat((freqs, freqs), dim=-1)[:, None, :]
        return freqs.cos(), freqs.sin()

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: list[list[int]]
    ) -> torch.Tensor:
        x = self.pre_layrnorm(self.embeddings(pixel_values))
        segments = self._segments(grid_thw)
        cos, sin = self._rope_cos_sin(segments, x.device)
        lengths = [t * h * w for t, h, w in segments]
        cu_seqlens = torch.tensor(
            [0] + lengths, dtype=torch.int32, device=x.device
        ).cumsum(0, dtype=torch.int32)
        max_seqlen = max(lengths)
        for layer in self.encoder.layers:
            x = layer(x, cos, sin, cu_seqlens, max_seqlen)
        return x


class MiniMaxM3VisionTower(nn.Module):
    def __init__(self, vision_config) -> None:
        super().__init__()
        self.vision_model = MiniMaxM3VisionTransformer(vision_config)

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: list[list[int]]
    ) -> torch.Tensor:
        return self.vision_model(pixel_values, grid_thw)


class MiniMaxM3MultiModalProjector(nn.Module):
    def __init__(self, in_size: int, mid_size: int, out_size: int, bias: bool):
        super().__init__()
        self.linear_1 = nn.Linear(in_size, mid_size, bias=bias)
        self.linear_2 = nn.Linear(mid_size, out_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.gelu(self.linear_1(x)))


class MiniMaxM3PatchMerger(MiniMaxM3MultiModalProjector):
    """Concatenates each 2x2 merge window, then projects it to one token."""

    def __init__(self, merge_size: int, hidden: int, mid_size: int, bias: bool):
        super().__init__(hidden * merge_size**2, mid_size, hidden, bias)
        self.merge_size = merge_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.reshape(x.shape[0] // self.merge_size**2, -1))


def build_vision_modules(hf_config) -> tuple[nn.Module, nn.Module, nn.Module]:
    """The tower, projector and patch merger named as in the checkpoint."""
    vision_config = hf_config.vision_config
    text_config = getattr(hf_config, "text_config", None) or hf_config
    text_hidden = text_config.hidden_size
    mid_size = getattr(hf_config, "projector_hidden_size", None) or text_hidden
    merge_size = _compression(vision_config).get(
        "spatial_merge_size", _compression(hf_config).get("spatial_merge_size", 2)
    )
    for name in ("hidden_act", "projector_hidden_act"):
        act = getattr(vision_config, name, None) or getattr(hf_config, name, "gelu")
        if act != "gelu":
            raise ValueError(f"MiniMax-M3 vision expects gelu {name}, got {act!r}")
    tower = MiniMaxM3VisionTower(vision_config)
    projector = MiniMaxM3MultiModalProjector(
        vision_config.hidden_size,
        mid_size,
        text_hidden,
        getattr(hf_config, "multimodal_projector_bias", True),
    )
    merger = MiniMaxM3PatchMerger(
        merge_size, text_hidden, mid_size, getattr(hf_config, "patch_merge_bias", True)
    )
    return tower, projector, merger


class MiniMaxM3SparseForConditionalGenerationVL(
    MiniMaxM3SparseForConditionalGenerationTextOnly
):
    """MiniMax-M3.1 with its vision tower, for native ATOM serving.

    Image embeddings are produced once per request by `ModelRunner` and
    scattered over the `]<]image[>[` runs that `build_minimax_m3_inputs`
    recorded as embedding spans. Where image input is refused
    (`image_input_refusal`, e.g. a text-only export) this degrades to the
    text-only model.
    """

    # Vision weights belong to this model, so nothing is skipped by default.
    # `__init__` re-adds the skips where no tower is built.
    skip_weight_prefixes: ClassVar[list[str]] = []
    vision_weight_prefixes: ClassVar[tuple[str, ...]] = tuple(
        MiniMaxM3SparseForConditionalGenerationTextOnly.skip_weight_prefixes
    )

    def __init__(self, atom_config: Config, prefix: str = "") -> None:
        super().__init__(atom_config, prefix=prefix)
        hf_config = atom_config.hf_config
        self.image_token_id = image_token_id(hf_config)
        # The tower only runs where the token embeddings are produced, and
        # only where image requests are accepted (`build_minimax_m3_inputs`).
        why = image_input_refusal(atom_config)
        self.has_vision_tower = why is None and get_pp_group().is_first_rank
        if not self.has_vision_tower:
            self.vision_tower = PPMissingLayer()
            self.multi_modal_projector = PPMissingLayer()
            self.patch_merge_mlp = PPMissingLayer()
            self.skip_weight_prefixes = list(self.vision_weight_prefixes)
            return

        (
            self.vision_tower,
            self.multi_modal_projector,
            self.patch_merge_mlp,
        ) = build_vision_modules(hf_config)
        self.language_model.embed_before_model = True

    def get_vision_embeddings(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        if not self.has_vision_tower:
            raise RuntimeError(
                "MiniMax-M3 image embeddings were requested on a rank without "
                "a vision tower (image input refused, or a non-first PP rank)."
            )
        features = self.vision_tower(pixel_values, grid_thw.tolist())
        return self.patch_merge_mlp(self.multi_modal_projector(features))

    def merge_multimodal_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        vision_embeds: torch.Tensor,
    ) -> torch.Tensor:
        mask = input_ids == self.image_token_id
        num_placeholders = int(mask.sum())
        if num_placeholders != vision_embeds.shape[0]:
            raise ValueError(
                f"MiniMax-M3 got {vision_embeds.shape[0]} image embeddings for "
                f"{num_placeholders} image tokens; an image prefill without "
                "embedding spans must not be chunked."
            )
        inputs_embeds[mask] = vision_embeds.to(inputs_embeds.dtype)
        return inputs_embeds
