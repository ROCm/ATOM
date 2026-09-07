# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Vision tower, image preprocessing and prompt encoding for
DeepSeek-V4-Flash-Vision.

Pure PyTorch port of the reference ``inference/vision.py`` shipped inside the
checkpoint: a 32-layer ViT with 2D RoPE and full bidirectional attention over
one image, followed by the aligner that 3x3-downsamples the patch grid into the
language model's width.

Module and parameter names mirror the checkpoint exactly (``vision.*`` /
``aligner.*``), so the loader needs no rename rules.

The tower is replicated on every TP rank rather than sharded: at ~0.4B params it
is negligible next to the 43-layer language stack, and replication keeps the
image embeddings bit-identical across ranks (they are scattered into the token
embeddings before the first collective). It is BF16 throughout — the checkpoint
carries no ``.scale`` tensors under either prefix, unlike every ``layers.*``
weight.

Reference geometry for this checkpoint: ``vision_dim=1024``, ``n_heads=16``
(head_dim 64, rope over two 32-wide halves), ``patch_size=14``,
``inter_dim=2816``, ``downsample_ratio=3``, text ``dim=4096``.
"""

import importlib.util
import itertools
import logging
import math
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn

from atom.config import Config


@dataclass(frozen=True)
class DeepseekV4VisionConfig:
    """Vision hyperparameters, read off the V4 HF config's ``vision_*`` keys.

    Split out from ``DeepseekV4Args`` so the tower can be built and tested
    without constructing a full engine ``Config``.
    """

    n_layers: int = 32  # vision_n_layers
    dim: int = 1024  # vision_dim
    n_heads: int = 16  # vision_n_heads
    inter_dim: int = 2816  # vision_inter_dim
    patch_size: int = 14  # vision_patch_size
    rope_theta: float = 10000.0  # vision_rope_theta
    downsample_ratio: int = 3  # vision_downsample_ratio
    text_dim: int = 4096  # hidden_size — aligner output width
    # NOT the language stack's `rms_norm_eps` (1e-20 in this checkpoint). The
    # reference `vision.py` builds every vision RMSNorm with the class default
    # and never threads the config value in, so this is a fixed 1e-6. Wiring
    # `rms_norm_eps` here instead would silently diverge from the reference.
    norm_eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def rope_dim(self) -> int:
        """Width of each rotated half — the 2D table covers half the head."""
        return self.dim // self.n_heads // 2

    @classmethod
    def from_hf_config(cls, hf_config) -> "DeepseekV4VisionConfig":
        def g(key, default):
            value = getattr(hf_config, key, None)
            return default if value is None else value

        return cls(
            n_layers=g("vision_n_layers", 32),
            dim=g("vision_dim", 1024),
            n_heads=g("vision_n_heads", 16),
            inter_dim=g("vision_inter_dim", 2816),
            patch_size=g("vision_patch_size", 14),
            rope_theta=g("vision_rope_theta", 10000.0),
            downsample_ratio=g("vision_downsample_ratio", 3),
            text_dim=g("hidden_size", 4096),
            # norm_eps deliberately left at its default — see the field comment.
        )


def llm_grid_hw(n_vit_h: int, n_vit_w: int, downsample_ratio: int) -> tuple[int, int]:
    """Aligner output grid for a ``n_vit_h x n_vit_w`` patch grid.

    The aligner pads the grid up to a multiple of the downsample ratio before
    unfolding, so each axis rounds UP. Mirrors ``grid_tokens`` in the reference
    ``image_processor.py``, and is what lets the model recover ``(n_llm_h,
    n_llm_w)`` — and from it the N-layout permutation — from nothing but the
    patch grid carried in ``image_grid_thw``.
    """
    ceil_div = -(-n_vit_h // downsample_ratio), -(-n_vit_w // downsample_ratio)
    return ceil_div


@lru_cache(maxsize=64)
def _cos_sin_table(
    n_h: int, n_w: int, rope_dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """2D RoPE cos/sin for one ``n_h x n_w`` patch grid, shaped ``[L, 1, D]``.

    Each position contributes ``rope_dim // 2`` height frequencies followed by
    ``rope_dim // 2`` width frequencies, so a head's first rotated half mixes
    both axes. Cached because a request's images repeat few distinct grids and
    the table is pure geometry.
    """
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
    )
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def get_vision_cos_sin(
    grids: Sequence[tuple[int, int]],
    rope_dim: int,
    theta: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed 2D RoPE tables for a run of images, concatenated in image order."""
    tables = [_cos_sin_table(int(h), int(w), rope_dim, theta) for h, w in grids]
    cos = torch.cat([t[0] for t in tables]).to(device)
    sin = torch.cat([t[1] for t in tables]).to(device)
    return cos, sin


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Rotate the two halves of each head against the 2D frequencies.

    ``x`` is ``[L, n_heads, head_dim]``; ``cos``/``sin`` are ``[L, 1,
    head_dim // 2]`` and broadcast over heads.
    """
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


def _segment_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    """Non-causal packed attention over per-image segments.

    ``cu_seqlens`` bounds each image's patch grid, so a patch only ever attends
    within its own image — matching the reference, which runs one image at a
    time. Returns ``[L, n_heads * head_dim]``.

    Uses aiter's varlen flash attention on GPU. The SDPA fallback exists so the
    tower stays runnable (and numerically checkable against the reference) on a
    plain CPU box with no aiter build; it is not a perf path.
    """
    if q.is_cuda:
        from aiter import flash_attn_varlen_func

        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            softmax_scale=q.shape[-1] ** -0.5,
            causal=False,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out.flatten(start_dim=-2)

    chunks = []
    for start, end in itertools.pairwise(cu_seqlens.tolist()):
        # [S, H, D] -> [H, S, D] for SDPA, then back.
        seg = F.scaled_dot_product_attention(
            q[start:end].transpose(0, 1),
            k[start:end].transpose(0, 1),
            v[start:end].transpose(0, 1),
        )
        chunks.append(seg.transpose(0, 1))
    return torch.cat(chunks).flatten(start_dim=-2)


class RMSNorm(nn.Module):
    """Float32 RMSNorm with a float32 weight, matching the reference exactly.

    Not ATOM's fused `model_ops` RMSNorm: that one keeps the weight in the
    activation dtype, while these checkpoint tensors are stored float32 and the
    reference normalizes in float32 before casting back.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class PatchEmbed(nn.Module):
    """Per-patch linear projection over already-patchified pixels.

    The image processor hands over ``[L, 3, patch, patch]``, so the reference's
    projection is a plain Linear over the flattened patch rather than a Conv2d.
    """

    def __init__(self, cfg: DeepseekV4VisionConfig):
        super().__init__()
        self.proj = nn.Linear(3 * cfg.patch_size**2, cfg.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class Attention(nn.Module):
    def __init__(self, cfg: DeepseekV4VisionConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.wqkv = nn.Linear(cfg.dim, 3 * cfg.dim)
        self.wo = nn.Linear(cfg.dim, cfg.dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        n = x.size(0)
        q, k, v = (
            t.view(n, self.n_heads, self.head_dim)
            for t in self.wqkv(x).chunk(3, dim=-1)
        )
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        return self.wo(_segment_attention(q, k, v, cu_seqlens, max_seqlen))


class MLP(nn.Module):
    """SwiGLU with gate and up fused into a single ``w1`` projection."""

    def __init__(self, cfg: DeepseekV4VisionConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, 2 * cfg.inter_dim, bias=False)
        self.w2 = nn.Linear(cfg.inter_dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg: DeepseekV4VisionConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, cu_seqlens, max_seqlen)
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """DeepSeek ViT: full bidirectional attention within one image, 2D RoPE.

    Unlike the reference — which takes one image per call — this packs a run of
    images into a single sequence and segments attention with ``cu_seqlens``, so
    a multi-image prompt runs the 32 blocks once instead of once per image. The
    per-image math is unchanged.
    """

    def __init__(self, cfg: DeepseekV4VisionConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)

    def forward(
        self, patches: torch.Tensor, grids: Sequence[tuple[int, int]]
    ) -> torch.Tensor:
        """``patches``: ``[sum(h*w), 3, p, p]``; ``grids``: per-image ``(h, w)``."""
        lengths = [int(h) * int(w) for h, w in grids]
        if sum(lengths) != patches.size(0):
            raise ValueError(
                f"patch count {patches.size(0)} does not match grids {list(grids)} "
                f"(expected {sum(lengths)})"
            )
        cos, sin = get_vision_cos_sin(
            grids, self.cfg.rope_dim, self.cfg.rope_theta, patches.device
        )
        cu_seqlens = torch.tensor(
            [0, *lengths], dtype=torch.int32, device=patches.device
        ).cumsum(0, dtype=torch.int32)
        max_seqlen = max(lengths)

        x = self.patch_embed(patches)
        for block in self.blocks:
            x = block(x, cos, sin, cu_seqlens, max_seqlen)
        return self.norm(x)


class Aligner(nn.Module):
    """Projects the ViT grid into the language width, 3x3-downsampling first.

    The grid is padded up (right/bottom) to a multiple of the ratio, then
    unfolded into non-overlapping ``r x r`` blocks — so the feature layout the
    ``w1`` weight expects is ``(channel, kh, kw)``, which is exactly what
    ``F.unfold`` emits. Output rows are row-major over the downsampled grid.
    """

    def __init__(self, cfg: DeepseekV4VisionConfig):
        super().__init__()
        self.downsample_ratio = cfg.downsample_ratio
        self.w1 = nn.Linear(cfg.dim * cfg.downsample_ratio**2, cfg.text_dim)
        self.w2 = nn.Linear(cfg.text_dim, cfg.text_dim)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))


def build_vision_modules(cfg: DeepseekV4VisionConfig) -> tuple[ViT, Aligner]:
    """Build the ``(vision, aligner)`` pair for a DeepSeek-V4 vision config."""
    return ViT(cfg), Aligner(cfg)


# --------------------------------------------------------------------------
# Image preprocessing and prompt encoding
# --------------------------------------------------------------------------
#
# Port of the reference ``inference/image_processor.py`` shipped inside the
# checkpoint, adapted to ATOM's multimodal contract. Three things here are
# specific to this model and worth reading before touching any of it:
#
# **Sentinel token ids live ABOVE the vocabulary.** Each ``<｜deepseek_image｜>``
# placeholder (id 129264, a real token) expands into a run of ``vocab_size +
# type`` ids — 129280..129284 for a 129280-entry vocab. The model detects image
# positions with ``input_ids >= vocab_size`` and never embeds them from the table;
# ``DeepseekV4ForCausalLM`` overwrites every one of them when it carries a
# vision tower. Anything
# that indexes a vocab-sized table by ``input_ids`` (embedding, hash routing) must
# clamp first.
#
# **The N-layout is not row-major.** A row of the aligner grid is followed by an
# ``IMAGE_NEW_LINE``, rows are padded to an even count, and then *pairs of rows
# are transposed* so the sequence walks columns within each row pair. ``perm``
# maps that slot order back onto the aligner's row-major output.
#
# **Image blocks are aligned to the ratio-4 compression grid.** ``compress_pad``
# leading pads put ``IMAGE_START`` at ``pos ≡ 3 (mod 4)``, so image content starts
# exactly on a compression-block boundary of the CSA attention layers.
#
# Prompt formatting is NOT reimplemented here: ``encoding/encoding_dsv4.py`` ships
# inside the checkpoint and is loaded from there, so the template can never drift
# from the weights it was trained with. See :func:`load_checkpoint_encoding`.

logger = logging.getLogger("atom")

# Sentinel types, in the order the model's embedding table expects them. The
# numeric values are checkpoint ABI: they are added to `vocab_size` to form
# token ids, so they cannot be reordered.
IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)

# Image blocks are padded so IMAGE_START lands on the last slot of a 4-token
# compression block — see the module docstring.
COMPRESS_PAD_TO = 4

IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"


# ---------------------------------------------------------------------------
# Prompt encoding: loaded from the checkpoint, not vendored
# ---------------------------------------------------------------------------

_encoding_module = None
_encoding_lock = threading.Lock()


def load_checkpoint_encoding(model_path: str):
    """Import ``encoding/encoding_dsv4.py`` from the checkpoint directory.

    The prompt template (roles, thinking modes, reasoning-effort preambles, DSML
    tool rendering) is versioned with the weights, so it is loaded from the
    checkpoint rather than copied into ATOM — a vendored copy would silently
    serve a stale template after a model update, which shows up as an accuracy
    regression with no error anywhere.

    This executes code from the model directory, the same trust boundary
    ``AutoProcessor(trust_remote_code=True)`` already crosses for Kimi-K3.
    """
    global _encoding_module
    with _encoding_lock:
        if _encoding_module is not None:
            return _encoding_module

        path = f"{model_path}/encoding/encoding_dsv4.py"
        spec = importlib.util.spec_from_file_location("atom_dsv4_encoding", path)
        if spec is None or spec.loader is None:
            raise FileNotFoundError(
                f"DeepSeek-V4 vision needs the checkpoint's prompt encoder at "
                f"{path}, which is missing. It ships with the model repo; a "
                f"partial download or a stripped checkpoint will not serve "
                f"image requests."
            )
        module = importlib.util.module_from_spec(spec)
        # encoding_dsv4 is self-contained (stdlib only), but register it so a
        # traceback inside it resolves its source lines.
        sys.modules["atom_dsv4_encoding"] = module
        spec.loader.exec_module(module)
        logger.info(f"Loaded DeepSeek-V4 prompt encoder from {path}")
        _encoding_module = module
        return module


# ---------------------------------------------------------------------------
# Image geometry — faithful port of the reference
# ---------------------------------------------------------------------------


def grid_tokens(
    best_height: int, best_width: int, patch_size: int, downsample_ratio: int
) -> tuple[int, int, int]:
    """LLM token count for an aligner grid, including row and align padding.

    Counts the N-layout: one ``IMAGE_NEW_LINE`` per row, two for
    ``IMAGE_START``/``IMAGE_END``, a padding row when the row count is odd, and
    a trailing pad pair when the transposed layout ends mid-pair.
    """
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def solve_resize_ratio(
    height: int, width: int, patch_size: int, downsample_ratio: int, max_n_token: int
) -> tuple[int, int, int, int, int]:
    """Largest grid preserving aspect ratio that fits within ``max_n_token``.

    Solves ``h * (w + 1) + 2 <= max_n_token`` under ``h / w == height / width``,
    then falls back to degenerate shapes for extreme aspect ratios.
    """
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        assert max_w > 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(
            max_w * patch_size * downsample_ratio / width,
            max_h * patch_size * downsample_ratio / height,
        )
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def safe_resize(
    height: int,
    width: int,
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int, int, int]:
    """Shrink the grid until its N-layout token count fits the budget.

    The budget is reduced by ``COMPRESS_PAD_TO - 1`` up front to leave room for
    the worst-case leading compression pad, then walked down one token at a time
    because ``grid_tokens`` is not monotonic in the solver's input budget.
    """
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, budget
        )
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


class VisionPreprocessConfig:
    """The ``vision_*`` geometry knobs :func:`preprocess_image` reads."""

    __slots__ = (
        "downsample_ratio",
        "max_n_token",
        "max_wh_ratio",
        "min_pixels",
        "patch_size",
    )

    def __init__(
        self,
        patch_size: int = 14,
        downsample_ratio: int = 3,
        max_n_token: int = 384,
        min_pixels: int = 147456,
        max_wh_ratio: int | None = 8,
    ):
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.max_n_token = max_n_token
        self.min_pixels = min_pixels
        self.max_wh_ratio = max_wh_ratio

    @classmethod
    def from_hf_config(cls, hf_config) -> "VisionPreprocessConfig":
        def g(key, default):
            value = getattr(hf_config, key, None)
            return default if value is None else value

        return cls(
            patch_size=g("vision_patch_size", 14),
            downsample_ratio=g("vision_downsample_ratio", 3),
            max_n_token=g("vision_max_n_token", 384),
            min_pixels=g("vision_min_pixels", 147456),
            max_wh_ratio=g("vision_max_wh_ratio", 8),
        )


def preprocess_image(
    image: "Image.Image", cfg: VisionPreprocessConfig
) -> tuple[torch.Tensor, int, int, int, int]:
    """Resize/pad one image and patchify it.

    Returns ``(patches[L, 3, p, p], n_vit_h, n_vit_w, n_llm_h, n_llm_w)``.

    Note the deliberate asymmetry carried over from the reference: the aspect
    clamp and min-pixel upscale adjust the *geometry* variables `width`/`height`
    used to solve for the target grid, while the choice between `resize` and
    letterbox `pad` re-reads the image's ORIGINAL dimensions. Keep it —
    "tidying" it changes which images get letterboxed and silently shifts
    outputs away from the reference.
    """
    p = cfg.patch_size
    image = image.convert("RGB")
    width, height = image.size
    if cfg.max_wh_ratio is not None and width > height * cfg.max_wh_ratio:
        width = height * cfg.max_wh_ratio
    if 0 < width * height < cfg.min_pixels:
        ratio = (cfg.min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    n_llm_h, n_llm_w, best_height, best_width = safe_resize(
        height, width, best_height, best_width, p, cfg.downsample_ratio, cfg.max_n_token
    )
    n_vit_h, n_vit_w = best_height // p, best_width // p

    if cfg.max_wh_ratio is not None and image.width >= cfg.max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))

    x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    patches = (
        x.reshape(3, n_vit_h, p, n_vit_w, p)
        .permute(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3, p, p)
    )
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


# ---------------------------------------------------------------------------
# N-layout
# ---------------------------------------------------------------------------


def image_block_perm(n_llm_h: int, n_llm_w: int) -> torch.Tensor:
    """Map N-layout IMAGE slot order onto the aligner's row-major output.

    Depends only on the grid — NOT on where the block lands in the prompt, since
    ``start_pos`` only controls how many leading pads precede ``IMAGE_START``.
    That is what lets the model rebuild this from ``image_grid_thw`` alone
    instead of shipping it through ``multimodal_data``.
    """
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    # Walk each pair of rows column-first.
    order = torch.arange(rows * row_len).view(rows // 2, 2, row_len).transpose(1, 2)
    order = order.reshape(-1)
    image_idx = torch.full((rows * row_len,), -1, dtype=torch.int64)
    image_idx.view(rows, row_len)[:n_llm_h, :n_llm_w] = torch.arange(
        n_llm_h * n_llm_w
    ).view(n_llm_h, n_llm_w)
    perm = image_idx[order]
    return perm[perm >= 0]


def build_image_block(
    n_llm_h: int, n_llm_w: int, start_pos: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sentinel types in final token order, plus the aligner-row permutation.

    ``start_pos`` is the number of tokens already emitted, which fixes the
    leading compression pad so ``IMAGE_START`` lands at ``pos ≡ 3 (mod 4)``.
    """
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = torch.tensor(
        ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
        + [IMAGE_PAD] * (row_len * pad_h),
        dtype=torch.int64,
    )
    order = torch.arange(rows * row_len).view(rows // 2, 2, row_len).transpose(1, 2)
    order = order.reshape(-1)
    types = torch.cat(
        [
            torch.full((compress_pad,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_START]),
            types[order],
            torch.full((pad_last,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_END]),
        ]
    )
    return types, image_block_perm(n_llm_h, n_llm_w)


def expand_image_placeholders(
    prompt_tokens: list[int],
    images: list,
    image_token_id: int,
    vocab_size: int,
    cfg: VisionPreprocessConfig,
) -> tuple[list[int], list[torch.Tensor], list[tuple[int, int]]]:
    """Replace each placeholder token with its sentinel block.

    Must run sequentially: each block's leading compression pad is a function of
    how many tokens precede it, so blocks cannot be built independently.

    Returns ``(input_ids, per-image patches, per-image (n_vit_h, n_vit_w))``.
    """
    num_placeholders = sum(1 for token in prompt_tokens if token == image_token_id)
    if num_placeholders != len(images):
        raise ValueError(
            f"prompt has {num_placeholders} image placeholder tokens but "
            f"{len(images)} images were supplied"
        )

    tokens: list[int] = []
    patches_per_image: list[torch.Tensor] = []
    grids: list[tuple[int, int]] = []
    image_iter = iter(images)
    for token in prompt_tokens:
        if token != image_token_id:
            tokens.append(token)
            continue
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = preprocess_image(
            next(image_iter), cfg
        )
        types, _ = build_image_block(n_llm_h, n_llm_w, len(tokens))
        patches_per_image.append(patches)
        grids.append((n_vit_h, n_vit_w))
        tokens += (vocab_size + types).tolist()
    return tokens, patches_per_image, grids


# ---------------------------------------------------------------------------
# ATOM input builder
# ---------------------------------------------------------------------------


def _resolve_thinking(chat_template_kwargs: dict) -> tuple[str, str | None]:
    """Map ATOM's chat-template kwargs onto the encoder's thinking controls."""
    mode = chat_template_kwargs.get("thinking_mode")
    if mode is None:
        enable = chat_template_kwargs.get("enable_thinking")
        mode = "thinking" if enable else "chat"
    if mode not in ("chat", "thinking"):
        raise ValueError(f"thinking_mode must be 'chat' or 'thinking', got {mode!r}")
    return mode, chat_template_kwargs.get("reasoning_effort")


def _messages_for_encoder(messages: list[dict]) -> list[dict]:
    """Rewrite already-loaded image parts into blocks the encoder accepts.

    ``api_server._collect_multimodal_parts`` has already decoded every image into
    a PIL object and collected them in prompt order, so the encoder only needs to
    emit one placeholder per image — it does not need to reload pixels. The
    pixel source stays the caller's ordered `images` list; the traversal order of
    both walks is identical (messages in order, parts in order).
    """
    rewritten = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            rewritten.append(message)
            continue
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("image", "image_url"):
                parts.append({"type": "image", "url": "atom://preloaded"})
            else:
                parts.append(part)
        rewritten.append({**message, "content": parts})
    return rewritten


def build_deepseek_v4_inputs(
    atom_config: Config,
    processor: Any,
    messages: list[dict],
    images: list,
    chat_template_kwargs: dict,
    tools: Any = None,
) -> tuple[list[int], dict] | None:
    """Build DeepSeek-V4 vision inputs, or None on a text-only V4 checkpoint."""
    hf_config = atom_config.hf_config
    if int(getattr(hf_config, "vision_n_layers", 0) or 0) <= 0:
        return None

    encoding = load_checkpoint_encoding(atom_config.model)
    thinking_mode, reasoning_effort = _resolve_thinking(chat_template_kwargs)

    encoder_messages = _messages_for_encoder(messages)
    if tools:
        if not encoder_messages:
            raise ValueError("a request with tools must contain at least one message")
        # The encoder reads tools off the first message, not a separate arg.
        encoder_messages[0] = {**encoder_messages[0], "tools": tools}

    prompt, media = encoding.encode_messages(
        encoder_messages,
        thinking_mode=thinking_mode,
        reasoning_effort=reasoning_effort,
        return_multi_modal_data=True,
    )
    if len(media["images"]) != len(images):
        raise ValueError(
            f"prompt encoder emitted {len(media['images'])} image placeholders "
            f"but {len(images)} images were loaded from the request"
        )

    tokenizer = _resolve_tokenizer(processor, atom_config)
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
    if image_token_id is None or image_token_id == getattr(
        tokenizer, "unk_token_id", None
    ):
        raise ValueError(f"tokenizer has no {IMAGE_PLACEHOLDER} token")

    vocab_size = int(hf_config.vocab_size)
    input_ids, patches_per_image, grids = expand_image_placeholders(
        tokenizer.encode(prompt),
        images,
        image_token_id,
        vocab_size,
        VisionPreprocessConfig.from_hf_config(hf_config),
    )

    multimodal_data = {
        "pixel_values": torch.cat(patches_per_image, dim=0),
        # (t, h, w) with t=1 to match the engine's generic multimodal contract;
        # the vision tower only ever reads (h, w).
        "image_grid_thw": torch.tensor(
            [[1, h, w] for h, w in grids], dtype=torch.int64
        ),
    }
    return input_ids, multimodal_data


def _resolve_tokenizer(processor: Any, atom_config: Config):
    """Get a tokenizer from whatever the caller handed us.

    This checkpoint ships no ``preprocessor_config.json``, so there is no HF
    processor to unwrap — the server passes the tokenizer straight through, but
    accept a processor-shaped object too so the offline example can share this
    path.
    """
    if processor is not None:
        if hasattr(processor, "encode") and hasattr(processor, "convert_tokens_to_ids"):
            return processor
        inner = getattr(processor, "tokenizer", None)
        if inner is not None:
            return inner

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(atom_config.model, trust_remote_code=True)


# ---------------------------------------------------------------------------
# Attention visibility for image spans
# ---------------------------------------------------------------------------


def image_visible_spans(
    input_ids: np.ndarray,
    vocab_size: int,
    max_image_tokens: int,
    seq_lens: "list[int] | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-token visible distance left/right within its ``[START, END]`` span.

    NumPy port of the reference ``model.get_image_visible``. Tokens outside any
    image span get ``(0, 0)``, which collapses the caller's window arithmetic
    back to the plain causal sliding window — that is what keeps text batches
    bit-identical.

    ``input_ids`` is ATOM's flat ragged prefill batch. Spans are resolved per
    sequence rather than across the whole concatenation: a run-on span (an
    ``IMAGE_START`` whose ``IMAGE_END`` was never emitted) would otherwise let
    one request's tokens see the next request's, and the balanced-cumsum trick
    would hide it instead of failing.
    """
    input_ids = np.asarray(input_ids)
    if input_ids.ndim != 1:
        raise ValueError(f"expected a flat token array, got shape {input_ids.shape}")
    total = input_ids.shape[0]
    lens = [total] if seq_lens is None else list(seq_lens)
    if sum(lens) != total:
        raise ValueError(f"seq_lens {lens} do not sum to {total} tokens")

    left = np.zeros(total, dtype=np.int32)
    right = np.zeros(total, dtype=np.int32)
    offset = 0
    for length in lens:
        if length == 0:
            continue
        ids = input_ids[offset : offset + length]
        idx = np.arange(length, dtype=np.int32)
        is_start = ids == vocab_size + IMAGE_START
        is_end = ids == vocab_size + IMAGE_END
        if is_start.sum() != is_end.sum():
            raise ValueError(
                f"unbalanced image span in a {length}-token sequence: "
                f"{int(is_start.sum())} IMAGE_START vs {int(is_end.sum())} "
                "IMAGE_END. Multimodal prefills must not be chunked."
            )
        # Inside a span, or the closing token itself.
        valid = (np.cumsum(is_start) > np.cumsum(is_end)) | is_end
        starts = np.maximum.accumulate(np.where(is_start, idx, 0))
        # Nearest IMAGE_END at or after each position.
        ends = np.minimum.accumulate(np.where(is_end, idx, length)[::-1])[::-1]
        left[offset : offset + length] = (idx - starts) * valid
        right[offset : offset + length] = (ends - idx) * valid
        offset += length

    np.clip(left, None, max_image_tokens - 1, out=left)
    np.clip(right, None, max_image_tokens, out=right)
    return left, right


def image_aware_extend_window(
    positions: np.ndarray,
    img_left: np.ndarray,
    img_right: np.ndarray,
    window_size: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Widen each token's sliding window to cover its whole image span.

    NumPy port of the reference ``model.get_window_topk_idxs_visible``, returned
    as ``(extend_start, extend_count)`` because ATOM's prefill attention consumes
    a per-token contiguous index range rather than a padded matrix.

    The window grows in BOTH directions: ``extend_start`` reaches back past the
    128-token window to the image's first token, and the range runs forward to
    ``position + right`` — past the token itself, which is how in-image
    attention becomes bidirectional. That is only sound because the prefill
    kernel is purely index-driven with no causal mask, and because the scheduler
    never chunks a multimodal prefill, so every future row named here is already
    materialized in the current forward's ``kv`` tensor.

    With ``img_left == img_right == 0`` this reduces exactly to the causal
    ``(max(pos - win + 1, 0), min(pos + 1, win))`` the text path already uses.
    """
    positions = np.asarray(positions, dtype=np.int64)
    left_add = np.maximum(np.asarray(img_left, dtype=np.int64) - (window_size - 1), 0)
    extend_start = np.maximum(positions - (window_size - 1) - left_add, 0)
    # The reference materializes a fixed `width`-column matrix and masks past
    # `pos + right`; the effective last column is whichever comes first.
    extend_last = np.minimum(
        positions + np.asarray(img_right, dtype=np.int64), extend_start + width - 1
    )
    extend_count = extend_last - extend_start + 1
    return extend_start.astype(np.int32), extend_count.astype(np.int32)


def image_window_width(seq_len: int, window_size: int, max_image_tokens: int) -> int:
    """Column budget the reference allots per token: ``min(seqlen, win + max)``.

    Bounds how far :func:`image_aware_extend_window` can widen a single token's
    index list — 512 for this checkpoint (128 window + 384 image tokens) — which
    is what keeps the prefill index buffers from growing without limit.
    """
    return min(seq_len, window_size + max_image_tokens)
