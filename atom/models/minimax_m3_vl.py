# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Native vision stack for MiniMax-M3.1 (``minimax_m3_vl``).

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
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from atom.config import Config

IMAGE_TOKEN = "]<]image[>["
DEFAULT_IMAGE_TOKEN_ID = 200025


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
    from aiter import flash_attn_varlen_func

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


def _image_token_spans(input_ids: list[int], image_token_id: int) -> list[tuple]:
    spans, start = [], None
    for index, token in enumerate(input_ids + [None]):
        if token == image_token_id and start is None:
            start = index
        elif token != image_token_id and start is not None:
            spans.append((start, index - start))
            start = None
    return spans


def build_minimax_m3_inputs(
    atom_config: "Config",
    processor: Any,
    messages: list[dict],
    images: list,
    chat_template_kwargs: dict,
    tools: Any = None,
) -> tuple[list[int], dict]:
    """Render the chat template, then let ``MiniMaxVLProcessor`` expand images.

    The template emits one ``]<]image[>[`` per image; the processor wraps each
    in start/end-of-image tokens and repeats it once per merged patch
    (``t*h*w // 4``). Each run of image tokens becomes one embedding span, so
    image prefills can be chunked and their embeddings cached per request.
    """
    from atom.model_engine.multimodal_runtime import multimodal_cache_seed

    hf_config = atom_config.hf_config
    image_token_id = int(
        getattr(hf_config, "image_token_index", None)
        or getattr(hf_config, "image_token_id", DEFAULT_IMAGE_TOKEN_ID)
    )
    template_kwargs = dict(chat_template_kwargs)
    template_kwargs.pop("tokenize", None)
    template_kwargs.pop("add_generation_prompt", None)
    if tools:
        template_kwargs["tools"] = tools
    # The checkpoint ships its template for the tokenizer only; the server's
    # --chat-template, when given, is installed on the processor.
    text = processor.apply_chat_template(
        messages,
        chat_template=processor.chat_template or processor.tokenizer.chat_template,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    num_placeholders = text.count(IMAGE_TOKEN)
    if num_placeholders != len(images):
        raise ValueError(
            f"Prompt has {num_placeholders} image placeholders but {len(images)} images"
        )
    # The rendered template already carries its own BOS.
    processed = processor(
        text=[text], images=images, add_special_tokens=False, return_tensors="pt"
    )
    input_ids = processed["input_ids"][0].tolist()
    grid_thw = processed["image_grid_thw"]
    spans = _image_token_spans(input_ids, image_token_id)
    merge_length = processor.image_processor.merge_size**2
    expected = [int(n) // merge_length for n in grid_thw.prod(-1).tolist()]
    if [count for _, count in spans] != expected:
        raise ValueError(
            f"Image token runs {[c for _, c in spans]} disagree with the "
            f"processed grids {expected}"
        )
    data = {
        "pixel_values": processed["pixel_values"],
        "image_grid_thw": grid_thw,
        "embedding_spans": tuple(spans),
    }
    data["cache_seed"] = multimodal_cache_seed(data)
    return input_ids, data
