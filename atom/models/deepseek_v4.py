# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
DeepSeek-V4 model for ATOM (PR1: skeleton + tiny-config eager forward).

Architecture reference: /data/DeepSeek-V4-Pro/inference/model.py
Tech report: /app/logs_claude/deepseek_v4/DeepSeek_V4.pdf

This file is the PR1 skeleton. It mirrors the reference implementation's class
structure so dummy state_dicts produced by the reference can be loaded directly
into ATOM modules for numerical parity validation. Production paths (FP8/FP4
weight loading, tensor parallelism, AITER kernels, KV cache integration, MTP
spec decode, torch.compile, server) land in PR2-PR6.
"""

import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from aiter.dist.parallel_state import get_tensor_model_parallel_world_size
from atom.config import Config
from atom.model_ops.embed_head import VocabParallelEmbedding
from atom.model_ops.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.model_ops.moe import FusedMoE
from atom.model_ops.quant_v4 import (
    act_quant_inplace,
    fp4_act_quant_inplace,
    rotate_activation,
)
from atom.model_ops.sparse_attn_v4 import hc_split_sinkhorn, sparse_attn  # noqa: F401

# ---------------------------------------------------------------------------
# Config wrapper
# ---------------------------------------------------------------------------


@dataclass
class DeepseekV4Args:
    """Mirrors `inference/model.py:ModelArgs`. Constructed from `hf_config`.

    Field names match the V4 HuggingFace `config.json` keys where possible;
    aliases are documented inline.
    """

    # Core
    vocab_size: int = 129280
    dim: int = 7168  # hidden_size
    n_layers: int = 61  # num_hidden_layers
    n_mtp_layers: int = 1  # num_nextn_predict_layers
    n_hash_layers: int = 3  # num_hash_layers
    norm_eps: float = 1e-6  # rms_norm_eps
    max_seq_len: int = 1048576  # max_position_embeddings
    max_batch_size: int = 4  # PR1 toy default; PR3 driven by ATOM scheduler

    # Attention (MQA, single shared KV head)
    n_heads: int = 128  # num_attention_heads
    head_dim: int = 512
    rope_head_dim: int = 64  # qk_rope_head_dim
    q_lora_rank: int = 1536
    o_lora_rank: int = 1024
    o_groups: int = 16
    window_size: int = 128  # sliding_window

    # Per-layer attention type: 0=Dense, 4=CSA, 128 (or other large m')=HCA
    compress_ratios: Tuple[int, ...] = field(default_factory=tuple)

    # Indexer (CSA layers only)
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 1024

    # MoE
    moe_inter_dim: int = 3072  # moe_intermediate_size
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    n_activated_experts: int = 6  # num_experts_per_tok
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 2.5  # routed_scaling_factor
    swiglu_limit: float = 10.0

    # Hyper-Connections (mHC)
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # YaRN RoPE
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_factor: float = 16.0  # rope_scaling.factor
    original_seq_len: int = 65536  # rope_scaling.original_max_position_embeddings
    beta_fast: int = 32
    beta_slow: int = 1

    # Quantization (PR1 ignores; PR2+ uses)
    dtype: Literal["bf16", "fp8"] = "bf16"
    expert_dtype: Optional[Literal["fp4", "fp8"]] = None
    scale_fmt: Optional[Literal["ue8m0"]] = None

    # ATOM QuantizationConfig — wired in PR3c so Linear layers auto-build the
    # right (FP8 / FP4 / BF16) weight + scale params for real-checkpoint loading.
    # When None, all Linears are BF16 (used by toy / dummy validation paths).
    quant_config: Optional[Any] = None

    @classmethod
    def from_hf_config(cls, hf_config: Any) -> "DeepseekV4Args":
        # Use getattr with sensible defaults so we work whether the HF config is
        # a real V4 PretrainedConfig (all fields present) or a V3 PretrainedConfig
        # populated with extra V4 attrs (some fields may live only in the raw
        # config_dict, not on the config object — `transformers` strips unknown
        # kwargs unless they're in the schema).
        g = lambda k, default=None: getattr(hf_config, k, default)
        rope_scaling = g("rope_scaling", {}) or {}
        return cls(
            vocab_size=g("vocab_size"),
            dim=g("hidden_size"),
            n_layers=g("num_hidden_layers"),
            n_mtp_layers=g("num_nextn_predict_layers", 1),
            n_hash_layers=g("num_hash_layers", 0),
            norm_eps=g("rms_norm_eps", 1e-6),
            max_seq_len=g("max_position_embeddings", 2048),
            n_heads=g("num_attention_heads"),
            head_dim=g("head_dim", 512),
            rope_head_dim=g("qk_rope_head_dim", 64),
            q_lora_rank=g("q_lora_rank", 1536),
            o_lora_rank=g("o_lora_rank", 256),
            o_groups=g("o_groups", 16),
            window_size=g("sliding_window", 128),
            compress_ratios=tuple(g("compress_ratios", (0,))),
            index_n_heads=g("index_n_heads", 64),
            index_head_dim=g("index_head_dim", 128),
            index_topk=g("index_topk", 1024),
            moe_inter_dim=g("moe_intermediate_size", 2048),
            n_routed_experts=g("n_routed_experts", 256),
            n_shared_experts=g("n_shared_experts", 1),
            n_activated_experts=g("num_experts_per_tok", 6),
            score_func=g("scoring_func", "sqrtsoftplus"),
            route_scale=g("routed_scaling_factor", 1.5),
            swiglu_limit=g("swiglu_limit", 10.0),
            hc_mult=g("hc_mult", 4),
            hc_sinkhorn_iters=g("hc_sinkhorn_iters", 20),
            hc_eps=g("hc_eps", 1e-6),
            rope_theta=g("rope_theta", 10000.0),
            compress_rope_theta=g("compress_rope_theta", 160000.0),
            rope_factor=rope_scaling.get("factor", 1.0),
            original_seq_len=rope_scaling.get("original_max_position_embeddings", 0),
            beta_fast=rope_scaling.get("beta_fast", 32),
            beta_slow=rope_scaling.get("beta_slow", 1),
        )


# ---------------------------------------------------------------------------
# Module-level constants matching reference inference/model.py module globals
# ---------------------------------------------------------------------------

# PR1 always runs single-rank; TP comes in PR3.
_FP4_BLOCK_SIZE = 32  # matches reference's fp4_block_size


# ---------------------------------------------------------------------------
# V4-specific QuantizationConfig — wired by DeepseekV4ForCausalLM in PR3c
# ---------------------------------------------------------------------------


def make_v4_quant_config(hf_config):
    """Build a QuantizationConfig that knows V4's per-layer quant scheme.

    V4 checkpoint layout:
      - Most projections (wq_a/b, wkv, wo_b, indexer.wq_b, etc.): FP8 e4m3 +
        128x128 ue8m0 block scale. Picked up by ATOM's standard "fp8" parser.
      - Routed expert weights (`ffn.experts.{N}.w{1,2,3}`): FP4 e2m1 +
        per-1x32 ue8m0 block scale. Needs explicit per_1x32 override.
      - `wo_a`: FP8 on disk but loaded as BF16 (convert.py:137-141 dequantizes
        because the grouped-LoRA einsum needs BF16; aiter has no FP8 einsum).
      - `Compressor.wkv` / `Compressor.wgate` / `indexer.weights_proj`: BF16
        (or fp32 internally; reference declares dtype= explicitly). Loaded raw.
      - All RMSNorm weights, attn_sink, hc_*: BF16/fp32 raw, no quant.
    """
    from atom.config import LayerQuantConfig, QuantizationConfig, QuantType
    from aiter import dtypes

    base = QuantizationConfig(hf_config)

    fp4_spec = LayerQuantConfig(quant_type=QuantType.per_1x32, quant_dtype=dtypes.fp4x2)
    no_spec = LayerQuantConfig(quant_type=QuantType.No, quant_dtype=torch.bfloat16)
    orig_lookup = base.get_layer_quant_config

    def overridden(layer_name, *, check_children=False):
        # Routed experts → FP4 (NOT shared_experts, which stay FP8).
        # Match both per-expert prefix `layers.N.ffn.experts.M.w{1,2,3}` (used
        # by individual Linear lookups, with trailing `.M.w1`) AND the bare
        # `layers.N.ffn.experts` prefix (used by FusedMoE.__init__ when
        # constructing fused expert params — has NO trailing dot).
        if ".ffn.experts" in layer_name:
            return fp4_spec
        # BF16 / fp32 raw paths
        if (
            ".compressor.wkv" in layer_name
            or ".compressor.wgate" in layer_name
            or ".indexer.weights_proj" in layer_name
        ):
            return no_spec
        # NOTE: wo_a is FP8 on disk but used as BF16 in forward (aiter has no FP8
        # grouped einsum). It's NOT in no_spec — instead we let it allocate as
        # FP8 + e8m0 scale so the standard loader fills both, then
        # DeepseekV4Attention.process_weights_after_loading dequants in place.
        return orig_lookup(layer_name, check_children=check_children)

    base.get_layer_quant_config = overridden
    return base


def _have_current_atom_config() -> bool:
    """Check whether ATOM's global Config has been set.

    `FusedMoE.__init__` calls `get_current_atom_config()` (which asserts non-None)
    to read TP/EP/dtype globals. The toy / dummy validation paths run before any
    ATOM ModelRunner sets it, so MoE falls back to its manual per-expert path
    when this returns False.
    """
    try:
        from atom.config import get_current_atom_config

        get_current_atom_config()
        return True
    except (AssertionError, ImportError):
        return False


def _dequant_fp8_block_to_bf16(w_fp8, scale, block=128):
    """Dequant block-scaled FP8 e4m3 → BF16 (for wo_a load path).

    Mirrors convert.py:137-141. The wo_a weight is stored FP8 on disk but
    used as BF16 in inference because aiter doesn't support FP8 grouped einsum.
    """
    out_dim, in_dim = w_fp8.shape
    w = w_fp8.unflatten(0, (-1, block)).unflatten(-1, (-1, block)).float()
    s = scale.float()
    deq = w * s[:, None, :, None]
    return deq.flatten(2, 3).flatten(0, 1).bfloat16()


# ---------------------------------------------------------------------------
# Small utilities — port of inference/model.py:183-276
# ---------------------------------------------------------------------------


class _RMSNorm(nn.Module):
    """Reference RMSNorm: weight stored in fp32, computation in fp32, output in input dtype.

    Port of inference/model.py:183-196. Distinct from atom.model_ops.layernorm.RMSNorm
    which is wired for TP/quantization; PR1 keeps a self-contained class so dummy
    state_dicts load directly. PR3 may swap to ATOM's RMSNorm with dtype shim.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


@lru_cache(2)
def _precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> torch.Tensor:
    """Precompute complex exponentials for rotary embeddings with YaRN scaling.

    Port of inference/model.py:199-229. When `original_seq_len > 0`, applies YaRN
    frequency interpolation with a smooth linear ramp between beta_fast and
    beta_slow correction ranges.
    """

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_, max_, dim):
        if min_ == max_:
            max_ += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_) / (max_ - min_)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Apply rotary positional embeddings IN-PLACE.

    Port of inference/model.py:232-244. The input tensor `x` is overwritten with
    the rotated values; the same tensor is also returned for chaining.
    `inverse=True` uses the conjugate (un-rotation) — used on the attention
    output to remove absolute-position embedding from the value contribution.
    """
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


@lru_cache(1)
def _get_window_topk_idxs(
    window_size: int,
    bsz: int,
    seqlen: int,
    start_pos: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Per-query topk-style indices into the sliding window KV cache.

    Port of inference/model.py:255-265. Returns [bsz, seqlen, window_size] of
    int positions into the KV buffer. -1 marks "skip" (causal mask, no fill).
    """
    kw = dict(device=device) if device is not None else {}
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat(
            [
                torch.arange(start_pos + 1, window_size, **kw),
                torch.arange(0, start_pos + 1, **kw),
            ],
            dim=0,
        )
    elif start_pos > 0:
        matrix = F.pad(
            torch.arange(start_pos + 1, **kw),
            (0, window_size - start_pos - 1),
            value=-1,
        )
    else:
        base = torch.arange(seqlen, **kw).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(
            min(seqlen, window_size), **kw
        )
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


@lru_cache(2)
def _get_compress_topk_idxs(
    ratio: int,
    bsz: int,
    seqlen: int,
    start_pos: int,
    offset: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Per-query indices into the compressed KV cache (for HCA — no Indexer).

    Port of inference/model.py:269-276. -1 marks compressed blocks that are
    still in the future at this query position.
    """
    kw = dict(device=device) if device is not None else {}
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio, **kw) + offset
    else:
        matrix = torch.arange(seqlen // ratio, **kw).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1, **kw).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


# ---------------------------------------------------------------------------
# Compressor + Indexer — port of inference/model.py:279-433
# ---------------------------------------------------------------------------


class Compressor(nn.Module):
    """Compresses KV cache via learned gated pooling over `compress_ratio` consecutive tokens.

    Port of inference/model.py:279-377. When `overlap=True` (always set when
    ratio==4, used by CSA), the compression uses overlapping windows to smooth
    block boundaries. When `rotate=True` (only the Indexer's compressor),
    output is Hadamard-rotated and FP4-simulated; otherwise non-RoPE dims are
    FP8-simulated.
    """

    def __init__(
        self,
        args: DeepseekV4Args,
        compress_ratio: int = 4,
        head_dim: int = 512,
        rotate: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = head_dim - args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        self.scale_fmt = args.scale_fmt
        self.prefix = prefix
        coff = 1 + self.overlap

        self.ape = nn.Parameter(
            torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32)
        )
        # wkv/wgate stored as fp32 (matches reference's Linear(dtype=fp32) BF16 path).
        # Kept as nn.Linear (not ATOM Linear) because the fp32 path through
        # ATOM's tgemm auto-casts output to BF16 — losing precision the
        # Compressor's softmax-pool step depends on. PR3+ may revisit.
        self.wkv = nn.Linear(
            self.dim, coff * self.head_dim, bias=False, dtype=torch.float32
        )
        self.wgate = nn.Linear(
            self.dim, coff * self.head_dim, bias=False, dtype=torch.float32
        )
        self.norm = _RMSNorm(self.head_dim, args.norm_eps)

        # External tensors — assigned by the owning Attention / Indexer at first forward.
        self.kv_cache: Optional[torch.Tensor] = None
        self.freqs_cis: Optional[torch.Tensor] = None

        # Decode-phase state buffers. With overlap: state[:, :ratio] holds the
        # overlapping window from the previous compression block; state[:, ratio:]
        # holds the current in-progress window.
        self.register_buffer(
            "kv_state",
            torch.zeros(
                args.max_batch_size,
                coff * compress_ratio,
                coff * self.head_dim,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "score_state",
            torch.full(
                (
                    args.max_batch_size,
                    coff * compress_ratio,
                    coff * self.head_dim,
                ),
                float("-inf"),
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def overlap_transform(
        self, tensor: torch.Tensor, value: float = 0.0
    ) -> torch.Tensor:
        """Reshape the [b, s, ratio, 2*d] overlap-projected tensor into [b, s, 2*ratio, d]
        such that the second half (size ratio) holds the current window's
        normal-projection slice and the first half (size ratio) holds the
        previous query position's overlap-projection slice (-inf padded for s=0).
        """
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int) -> Optional[torch.Tensor]:
        """Compress KV for the input tokens. Writes into self.kv_cache when a
        compression block boundary is hit; otherwise just buffers state and returns None.

        Args:
            x: [num_tokens, dim] (2D, ATOM convention) or [B, S, dim] (3D, legacy).
                2D input is treated as a single sequence (B=1 implicit).
            start_pos: starting position in the absolute sequence (0 = prefill).
        Returns:
            Compressed KV slice that was just written ([1, S/ratio, head_dim] in
            prefill, or [1, 1, head_dim] in decode), or None if no compression
            boundary was hit on this call.
        """
        assert self.kv_cache is not None, "compressor.kv_cache must be set by owner"
        assert self.freqs_cis is not None, "compressor.freqs_cis must be set by owner"
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [num_tokens, dim] → [1, num_tokens, dim]
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        overlap = self.overlap
        d = self.head_dim
        rd = self.rope_head_dim
        dtype = x.dtype

        # Compression always done in fp32 for stability.
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)

        if start_pos == 0:
            # ===== Prefill path =====
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0

            # Save the last `ratio` overlap-slice tokens into kv_state for use
            # by the next decode call's overlap window.
            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff - ratio : cutoff]
                self.score_state[:bsz, :ratio] = (
                    score[:, cutoff - ratio : cutoff] + self.ape
                )
            # Save the trailing partial block (remainder tokens) into kv_state.
            if remainder > 0:
                kv, self.kv_state[:bsz, offset : offset + remainder] = kv.split(
                    [cutoff, remainder], dim=1
                )
                self.score_state[:bsz, offset : offset + remainder] = (
                    score[:, cutoff:] + self.ape[:remainder]
                )
                score = score[:, :cutoff]
            # Reshape to [B, num_blocks, ratio, coff*d] and softmax-pool.
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0.0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            # ===== Decode path (start_pos > 0, seqlen == 1) =====
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score = score + self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat(
                        [
                            self.kv_state[:bsz, :ratio, :d],
                            self.kv_state[:bsz, ratio:, d:],
                        ],
                        dim=1,
                    )
                    score_state = torch.cat(
                        [
                            self.score_state[:bsz, :ratio, :d],
                            self.score_state[:bsz, ratio:, d:],
                        ],
                        dim=1,
                    )
                    kv = (kv_state * score_state.softmax(dim=1)).sum(
                        dim=1, keepdim=True
                    )
                    # Roll: the just-completed window becomes the next overlap window.
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (
                        self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)
                    ).sum(dim=1, keepdim=True)

        if not should_compress:
            return None

        kv = self.norm(kv.to(dtype))

        # Apply RoPE to the rope-head-dim tail of compressed entries.
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)
        _apply_rotary_emb(kv[..., -rd:], freqs_cis)

        # QAT round-trip: rotated branch (Indexer) uses Hadamard + FP4 sim;
        # plain branch uses FP8 sim on non-rope dims only.
        if self.rotate:
            kv = rotate_activation(kv)
            fp4_act_quant_inplace(kv, _FP4_BLOCK_SIZE)
        else:
            act_quant_inplace(kv[..., :-rd], 64, self.scale_fmt)

        if start_pos == 0:
            self.kv_cache[:bsz, : seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv


class Indexer(nn.Module):
    """Selects top-k compressed KV positions for sparse attention via learned scoring.

    Port of inference/model.py:380-433. Has its own Compressor (with Hadamard
    rotation + FP4 simulation) to build a separate compressed KV cache used
    only for index scoring; query is also FP4-simulated.
    """

    def __init__(self, args: DeepseekV4Args, compress_ratio: int = 4, prefix: str = ""):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        # TP shards Q heads (wq_b is ColumnParallelLinear); per-rank head count.
        tp_size = get_tensor_model_parallel_world_size()
        assert (
            args.index_n_heads % tp_size == 0
        ), f"index_n_heads={args.index_n_heads} not divisible by tp={tp_size}"
        self.n_local_heads = args.index_n_heads // tp_size
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.compress_ratio = compress_ratio

        qc = args.quant_config
        # Indexer Q heads sharded across TP ranks.
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=qc,
            prefix=f"{prefix}.wq_b",
        )
        # weights_proj: BF16 in reference (dtype=bf16); V4QuantConfig already
        # forces no_spec for ".indexer.weights_proj" so quant_config is fine.
        self.weights_proj = ColumnParallelLinear(
            self.dim,
            self.n_heads,
            bias=False,
            quant_config=qc,
            prefix=f"{prefix}.weights_proj",
        )
        self.softmax_scale = self.head_dim**-0.5

        self.compressor = Compressor(
            args,
            compress_ratio,
            self.head_dim,
            rotate=True,
            prefix=f"{prefix}.compressor",
        )
        self.register_buffer(
            "kv_cache",
            torch.zeros(
                args.max_batch_size,
                args.max_seq_len // compress_ratio,
                self.head_dim,
            ),
            persistent=False,
        )
        self.freqs_cis: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        offset: int,
    ) -> torch.Tensor:
        """Compute sparse top-k indices over the indexer's compressed KV cache.

        Args:
            x: [num_tokens, dim] input hidden states (for compressor + weights_proj).
            qr: [num_tokens, q_lora_rank] latent query shared with main Attention's q_a.
            start_pos: absolute sequence start position.
            offset: offset added to returned indices to land them in the
                concatenated (window || compressed) KV layout consumed by sparse_attn.
        Returns:
            topk_idxs: [1, num_tokens, K] int (B=1 implicit). -1 = future-masked.
        """
        assert self.freqs_cis is not None
        assert x.dim() == 2 and qr.dim() == 2
        seqlen = x.size(0)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen

        # Lazy plumb the indexer's kv_cache + freqs_cis into its compressor.
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis

        # ----- Indexer Q (2D Linear, then add B=1 dim for RoPE / einsum) -----
        q = self.wq_b(qr).view(seqlen, self.n_local_heads, self.head_dim)
        q = q.unsqueeze(0)  # [1, S, H, D]
        _apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = rotate_activation(q)
        fp4_act_quant_inplace(q, _FP4_BLOCK_SIZE)

        # ----- Indexer KV (Compressor takes 2D, mutates kv_cache) -----
        self.compressor(x, start_pos)
        # weights_proj is ATOM Linear → 2D input; restore B=1 dim for einsum.
        weights = (
            self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        ).unsqueeze(0)

        # ----- Index score -----
        index_score = torch.einsum(
            "bshd,btd->bsht", q, self.kv_cache[:1, : end_pos // ratio]
        )
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)

        # ----- Top-k selection over compressed positions -----
        if start_pos == 0:
            mask = (
                torch.arange(seqlen // ratio, device=x.device).repeat(seqlen, 1)
                >= torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            )
            index_score = index_score + torch.where(mask, float("-inf"), 0.0)
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            mask = (
                topk_idxs
                >= torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            )
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs = topk_idxs + offset
        return topk_idxs


# ---------------------------------------------------------------------------
# Stubs — implementations land in tasks #5-#8
# ---------------------------------------------------------------------------


class DeepseekV4Attention(nn.Module):
    """Hybrid attention: MQA + grouped output LoRA + sliding window + attn_sink.

    Port of inference/model.py:436-543. Per-layer behavior driven by
    `compress_ratio` (read from args.compress_ratios[layer_id]):

      - `compress_ratio == 0`: Dense (sliding-window only; no compressor/indexer)
      - `compress_ratio == 4`: CSA (compressor with overlap + indexer for top-k)
      - `compress_ratio >= 8`: HCA (compressor only; topk_idxs pre-computed)

    Layout:
      - Single shared MQA head for KV (head_dim=512). Each query head attends
        to the same compressed/window KV via per-query top-k gather.
      - q_lora_rank low-rank Q projection: wq_a -> q_norm -> wq_b -> RMSNorm-per-head -> RoPE
      - Grouped output LoRA: o_groups groups, each with rank o_lora_rank
      - Sliding window of `args.window_size=128` raw KV entries (BF16, FP8-simulated nope dims)
      - Compressed KV up to `max_seq_len // compress_ratio` entries (when ratio > 0)
      - attn_sink: per-head learnable logit added only to softmax denominator
    """

    def __init__(self, layer_id: int, args: DeepseekV4Args, prefix: str = ""):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        # TP shards heads + groups across ranks. ColumnParallelLinear (wq_b, wo_a)
        # auto-splits output dim, so per-rank counts must be divided by tp_size.
        tp_size = get_tensor_model_parallel_world_size()
        assert (
            args.n_heads % tp_size == 0
        ), f"n_heads={args.n_heads} not divisible by tp={tp_size}"
        assert (
            args.o_groups % tp_size == 0
        ), f"o_groups={args.o_groups} not divisible by tp={tp_size}"
        self.tp_size = tp_size
        self.n_local_heads = args.n_heads // tp_size
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps
        self.scale_fmt = args.scale_fmt

        qc = args.quant_config
        p = prefix  # e.g. "layers.7.attn"

        # ----- Parameters (names mirror reference for state_dict load) -----
        self.attn_sink = nn.Parameter(
            torch.empty(self.n_local_heads, dtype=torch.float32)
        )
        self.wq_a = ReplicatedLinear(
            self.dim,
            self.q_lora_rank,
            bias=False,
            quant_config=qc,
            prefix=f"{p}.wq_a",
        )
        self.q_norm = _RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=qc,
            prefix=f"{p}.wq_b",
        )
        self.wkv = ReplicatedLinear(
            self.dim,
            self.head_dim,
            bias=False,
            quant_config=qc,
            prefix=f"{p}.wkv",
        )
        self.kv_norm = _RMSNorm(self.head_dim, self.eps)
        # wo_a: grouped LoRA — V4QuantConfig forces this BF16 even though disk is FP8.
        # The grouped einsum (`bsgd,grd->bsgr`) needs BF16 weights; aiter has no FP8 einsum.
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * args.o_lora_rank,
            bias=False,
            quant_config=qc,
            prefix=f"{p}.wo_a",
        )
        self.wo_b = RowParallelLinear(
            self.n_groups * args.o_lora_rank,
            self.dim,
            bias=False,
            quant_config=qc,
            prefix=f"{p}.wo_b",
        )
        self.softmax_scale = self.head_dim**-0.5

        # ----- Compressor (and Indexer for CSA) -----
        if self.compress_ratio:
            self.compressor = Compressor(
                args,
                self.compress_ratio,
                self.head_dim,
                prefix=f"{p}.compressor",
            )
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio, prefix=f"{p}.indexer")
            else:
                self.indexer = None
        else:
            self.compressor = None
            self.indexer = None

        # ----- KV cache: [B, window_size + max_seq_len/ratio, head_dim] -----
        kv_cache_size = args.window_size + (
            args.max_seq_len // self.compress_ratio if self.compress_ratio else 0
        )
        self.register_buffer(
            "kv_cache",
            torch.zeros(args.max_batch_size, kv_cache_size, self.head_dim),
            persistent=False,
        )

        # ----- RoPE freqs (own freqs, not shared): YaRN for compressed
        # attention layers (long context), plain rope for dense (window-only) -----
        if self.compress_ratio:
            original_seq_len, rope_theta = (
                args.original_seq_len,
                args.compress_rope_theta,
            )
        else:
            original_seq_len, rope_theta = 0, args.rope_theta
        freqs_cis = _precompute_freqs_cis(
            self.rope_head_dim,
            args.max_seq_len,
            original_seq_len,
            rope_theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def process_weights_after_loading(self) -> None:
        """Dequant wo_a (FP8 + e8m0 block scale) → BF16 in place.

        Called by ATOM's standard loader (atom.model_loader.loader.load_model)
        after all weights are filled. wo_a is allocated as FP8 ColumnParallelLinear
        so both `.weight` (FP8) and `.weight_scale` (e8m0 block scale) load
        correctly via the standard FP8 path. We then dequant to BF16 because
        forward needs `wo_a.weight` as BF16 for the grouped LoRA einsum
        (`bsgd,grd->bsgr`); aiter has no FP8 grouped einsum.

        Idempotent: if wo_a.weight is already BF16 (e.g. dequant was applied
        elsewhere), this is a no-op.
        """
        w = self.wo_a.weight
        if w.dtype == torch.bfloat16:
            return  # already dequanted
        scale = getattr(self.wo_a, "weight_scale", None)
        if w.dtype != torch.float8_e4m3fn or scale is None:
            return  # nothing to do
        # Dequant: w (FP8 [out, in]) × scale (e8m0 [out/128, in/128]) → BF16
        bf16 = _dequant_fp8_block_to_bf16(
            w.data, scale.data.to(torch.float32), block=128
        )
        # Replace the weight tensor with BF16, drop the scale param so future
        # loads / introspection don't try to use a stale FP8 scale.
        self.wo_a.weight = torch.nn.Parameter(bf16, requires_grad=False)
        try:
            delattr(self.wo_a, "weight_scale")
        except AttributeError:
            pass
        # CRITICAL: prevent LinearBase.process_weights_after_loading from
        # `shuffle_weights(self.weight)` on the now-BF16 wo_a. That shuffle
        # is for the FP8 CK GEMM layout; applying it to a plain BF16 matrix
        # consumed by `torch.einsum` corrupts the layout (rows get permuted
        # within 16×16 blocks, only rows aligned to the block boundaries
        # stay in place). Iteration order in load_model is parent-first
        # (DeepseekV4Attention before its child wo_a Linear), so our hook
        # runs BEFORE the shuffle — overriding `quant_type` here makes the
        # subsequent LinearBase post-load a no-op for wo_a.
        #
        # TODO(perf): replace dequant-to-BF16 + einsum with FP8 batched BMM
        # (same path as MLA's `_v_up_proj_and_o_proj`). Steps:
        #   1. Dequant FP8 per-128-block → BF16 (this code)
        #   2. Reshape to [n_local_groups, o_lora_rank, d_per_group]
        #   3. Requant via dynamic_per_batched_tensor_quant → FP8 + scalar scale
        #   4. Forward: _aiter_triton_fp8_bmm(o, W_OA, W_OA_scale, group_size=128)
        # This avoids the dequant + einsum overhead and reuses the proven MLA
        # batched-FP8 kernel. See attention_mla.py:211 for reference.
        from atom.config import QuantType as _QT

        self.wo_a.quant_type = _QT.No

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Compute attention for `x` at absolute position `start_pos`.

        Args:
            x: [num_tokens, dim] flat ATOM convention. Single-sequence
                (B=1 implicit; multi-sequence batching needs attn_metadata,
                deferred until ATOM's scheduler integration).
            start_pos: absolute position of the first token in this batch.
        Returns:
            [num_tokens, dim] attention output (BF16).
        """
        assert (
            x.dim() == 2
        ), f"DeepseekV4Attention expects 2D [num_tokens, dim], got {x.shape}"
        seqlen = x.size(0)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim

        # First-call plumbing: hand the (compressed-half) KV cache + freqs_cis
        # to the compressor / indexer.
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis

        # Reset all KV buffers on new prefill (start_pos==0). The ATOM warmup
        # forward (seqlen=MAX_BATCHED_TOKENS with zeros) fills these buffers
        # with garbage. Real prefill only overwrites a few slots, leaving
        # stale warmup data that poisons decode attention.
        if start_pos == 0:
            self.kv_cache.zero_()
            if self.compress_ratio:
                self.compressor.kv_state.zero_()
                self.compressor.score_state.fill_(float("-inf"))
                if self.indexer is not None:
                    self.indexer.kv_cache.zero_()
                    self.indexer.compressor.kv_state.zero_()
                    self.indexer.compressor.score_state.fill_(float("-inf"))

        # ----- Q: low-rank projection + per-head RMSNorm + partial RoPE -----
        # ATOM TP linears require 2D inputs; subsequent ops (RoPE, sparse_attn)
        # need a [B=1, S, ...] view. We add the singleton batch dim only where
        # required.
        qr = self.q_norm(self.wq_a(x))  # [S, q_lora_rank], shared with Indexer
        q = self.wq_b(qr).view(seqlen, self.n_local_heads, self.head_dim)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        # Insert B=1 dim for RoPE / sparse_attn convention.
        q = q.unsqueeze(0)  # [1, S, H, D]
        _apply_rotary_emb(q[..., -rd:], freqs_cis)

        # ----- Window KV: project, RMSNorm, RoPE on rope dims, FP8-sim on nope -----
        kv = self.wkv(x)  # [S, head_dim]
        kv = self.kv_norm(kv).unsqueeze(0)  # [1, S, head_dim]
        _apply_rotary_emb(kv[..., -rd:], freqs_cis)
        act_quant_inplace(kv[..., :-rd], 64, self.scale_fmt)

        # ----- Build topk_idxs (B=1 implicit) -----
        topk_idxs = _get_window_topk_idxs(win, 1, seqlen, start_pos, device=x.device)
        if self.compress_ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
            else:
                compress_topk_idxs = _get_compress_topk_idxs(
                    ratio, 1, seqlen, start_pos, offset, device=x.device
                )
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # ----- Attention: prefill writes window KV linearly + concats fresh
        # compressed entries; decode writes one slot in window ring buffer
        # then reads from the persistent kv_cache. (kv_cache slot 0 = our
        # implicit B=1.) -----
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:1, :seqlen] = kv
            else:
                cutoff = seqlen % win
                (
                    self.kv_cache[:1, cutoff:win],
                    self.kv_cache[:1, :cutoff],
                ) = kv[
                    :, -win:
                ].split([win - cutoff, cutoff], dim=1)
            if self.compress_ratio:
                if (kv_compress := self.compressor(x, start_pos)) is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            self.kv_cache[:1, start_pos % win] = kv.squeeze(1)
            if self.compress_ratio:
                self.compressor(x, start_pos)
            o = sparse_attn(
                q,
                self.kv_cache[:1],
                self.attn_sink,
                topk_idxs,
                self.softmax_scale,
            )

        # Inverse RoPE on output's rope dims to remove absolute-position contribution
        # carried in by the value-side RoPE of the KV entries.
        _apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)

        # ----- Grouped output LoRA -----
        # o: [1, S, H, D] → drop B; reshape into groups for the einsum.
        o = o.squeeze(0).view(seqlen, self.n_local_groups, -1)  # [S, g, H/g * D]
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("sgd,grd->sgr", o, wo_a)  # [S, g, o_lora_rank]
        x = self.wo_b(o.flatten(1))  # 2D [S, dim]
        return x


class Gate(nn.Module):
    """MoE gate with sqrtsoftplus/sigmoid/softmax scoring + hash routing.

    Port of inference/model.py:546-584. For `layer_id < args.n_hash_layers`,
    routing is by precomputed token-id-to-expert table (`tid2eid`); no scoring,
    no bias. Otherwise routing is by `score_func(W @ x) + bias` topk.

    Bias affects expert SELECTION (added before topk) but NOT routing weights —
    weights come from the original (pre-bias) score gathered at the topk indices.
    """

    def __init__(self, layer_id: int, args: DeepseekV4Args):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        if self.hash:
            self.tid2eid = nn.Parameter(
                torch.empty(
                    args.vocab_size, args.n_activated_experts, dtype=torch.int32
                ),
                requires_grad=False,
            )
            self.bias = None
        else:
            self.bias = nn.Parameter(
                torch.empty(args.n_routed_experts, dtype=torch.float32)
            )

    def forward(
        self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:  # sqrtsoftplus (V4 default)
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            assert input_ids is not None, "hash routing requires input_ids"
            indices = self.tid2eid[input_ids].long()
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights, indices


class Expert(nn.Module):
    """Single MoE expert: SwiGLU FFN (w1, w2, w3). Computation in float32 for stability.

    Port of inference/model.py:587-606. With `swiglu_limit > 0`, clamps both gate
    and up projections (gate clipped above only, up clipped both sides) before
    the SiLU * up product — matches reference behavior exactly.
    """

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        swiglu_limit: float = 0.0,
        quant_config: Optional[Any] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        if quant_config is None:
            self.w1 = nn.Linear(dim, inter_dim, bias=False)
            self.w2 = nn.Linear(inter_dim, dim, bias=False)
            self.w3 = nn.Linear(dim, inter_dim, bias=False)
        else:
            self.w1 = ColumnParallelLinear(
                dim,
                inter_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.w1",
            )
            self.w2 = RowParallelLinear(
                inter_dim,
                dim,
                bias=False,
                quant_config=quant_config,
                reduce_results=reduce_results,
                prefix=f"{prefix}.w2",
            )
            self.w3 = ColumnParallelLinear(
                dim,
                inter_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.w3",
            )
        self.swiglu_limit = swiglu_limit

    def forward(
        self, x: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class MoE(nn.Module):
    """Mixture-of-Experts: top-k routed experts (FusedMoE) + 1 shared expert.

    PR3b: replaces the per-expert nn.Linear list with `FusedMoE` so 384 routed
    experts shard across TP/EP ranks and load FP4 weights via the existing
    `gemm_a4w4_quant` aiter kernel.

    Routing math (`sqrtsoftplus(scores) + bias` topk) is delegated to
    `FusedMoE.select_experts(scoring_func="sqrtsoftplus", e_score_correction_bias=...)`,
    which we extended in atom/model_ops/moe.py to add the V4 path.

    Hash routing for `layer_id < n_hash_layers` (first 3 V4 layers) is NOT yet
    wired through FusedMoE — the `tid2eid` buffer is declared so weight loading
    completes, but inference uses the standard sqrtsoftplus path. Hash layers
    will produce incorrect routing; correct hash routing lands in PR3+.
    """

    def __init__(self, layer_id: int, args: DeepseekV4Args, prefix: str = ""):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.is_hash_layer = layer_id < args.n_hash_layers
        self.routed_scaling_factor = args.route_scale
        qc = args.quant_config
        # FusedMoE requires `get_current_atom_config()` (TP/EP/dtype globals).
        # When that's not set (toy / dummy validation path), fall back to the
        # manual per-expert path which preserves PR1 bit-exact reference parity.
        self.use_fused = qc is not None and _have_current_atom_config()
        self.use_torch_moe = bool(os.environ.get("ATOM_V4_TORCH_MOE"))
        self.swiglu_limit = args.swiglu_limit
        self.tp_size = get_tensor_model_parallel_world_size()

        if self.use_fused:
            # ----- Production path: ReplicatedLinear gate + FusedMoE experts -----
            self.gate = ReplicatedLinear(
                self.dim,
                self.n_routed_experts,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.gate",
            )
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(self.n_routed_experts, dtype=torch.float32)
            )
            if self.is_hash_layer:
                # tid2eid: per-token-id top-k expert lookup table (V4 first 3
                # layers use this in lieu of gate-logit routing).
                self.gate.tid2eid = nn.Parameter(
                    torch.empty(
                        args.vocab_size, args.n_activated_experts, dtype=torch.int32
                    ),
                    requires_grad=False,
                )
                # Cache for input_ids — set by forward() right before the FusedMoE
                # call so the custom routing closure can index tid2eid.
                self._hash_input_ids: Optional[torch.Tensor] = None

            from types import SimpleNamespace

            moe_cfg = SimpleNamespace(
                routed_scaling_factor=self.routed_scaling_factor,
                n_shared_experts=args.n_shared_experts,
            )
            self.experts = FusedMoE(
                num_experts=self.n_routed_experts,
                top_k=self.n_activated_experts,
                hidden_size=self.dim,
                intermediate_size=args.moe_inter_dim,
                reduce_results=False,
                renormalize=True,
                quant_config=qc,
                use_grouped_topk=False,
                prefix=f"{prefix}.experts",
                scoring_func=args.score_func,  # "sqrtsoftplus"
                e_score_correction_bias=self.gate.e_score_correction_bias,
                config=moe_cfg,
            )
            self.experts.swiglu_limit = args.swiglu_limit
            assert args.n_shared_experts == 1
            self.shared_experts = Expert(
                args.dim,
                args.moe_inter_dim,
                swiglu_limit=args.swiglu_limit,
                quant_config=qc,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
            if self.is_hash_layer:
                # Inject hash routing into FusedMoE.select_experts via the
                # custom_routing_function hook (added in atom/model_ops/moe.py).
                self.experts.custom_routing_function = self._hash_topk
        else:
            # ----- Toy / dummy path: manual Gate + per-expert nn.Linear -----
            # Preserves bit-exact reference parity for PR1 verify (no FusedMoE
            # math drift, no requirement on global atom config).
            self.gate = Gate(layer_id, args)
            self.experts = nn.ModuleList(
                [
                    Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)
                    for _ in range(self.n_routed_experts)
                ]
            )
            assert args.n_shared_experts == 1
            self.shared_experts = Expert(
                args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit
            )

    def _hash_topk(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """V4 hash routing for first 3 layers.

        topk_ids = tid2eid[input_ids]  (no gate-based selection)
        topk_weights = sqrtsoftplus(router_logits) gathered at topk_ids
        Then renormalize so weights sum to 1 per token.
        """
        assert (
            self._hash_input_ids is not None
        ), "MoE.forward() must set self._hash_input_ids before calling experts() in hash layers"
        ids = self._hash_input_ids.flatten()
        topk_ids = self.gate.tid2eid[ids].to(torch.int32)  # [N, topk]
        scores = torch.nn.functional.softplus(gating_output.float()).sqrt()
        topk_weights = scores.gather(dim=-1, index=topk_ids.long())
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights, topk_ids

    def _torch_moe_forward(
        self, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor
    ) -> torch.Tensor:
        """Per-expert torch loop using unshuffled FP4 weights from FusedMoE.

        Supports swiglu_limit clamping that the fused kernel path cannot do.
        Requires ATOM_V4_TORCH_MOE=1 (skips weight shuffle in post-load).
        """
        from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

        w13 = self.experts.w13_weight  # [E, 2*inter_tp, H//2] fp4x2
        w2 = self.experts.w2_weight  # [E, H, inter_tp//2] fp4x2
        w13_s = self.experts.w13_weight_scale  # [E, 2*inter_tp, H//32] uint8
        w2_s = self.experts.w2_weight_scale  # [E, H, inter_tp//32] uint8

        E = w13.shape[0]
        inter_tp = w13.shape[1] // 2
        limit = self.swiglu_limit

        y = torch.zeros_like(x, dtype=torch.float32)
        for e_id in range(E):
            mask = topk_ids == e_id
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=False)
            tok_idx = idx[:, 0]
            top_idx = idx[:, 1]
            sub_x = x[tok_idx].float()
            sub_w = topk_weights[tok_idx, top_idx].unsqueeze(-1)

            # Dequant w1/w3 (gate/up) from FP4
            w13_e = mxfp4_to_f32(w13[e_id])  # [2*inter_tp, H]
            w13_s_e = e8m0_to_f32(w13_s[e_id].contiguous().view(torch.float8_e8m0fnu))
            # Apply block scale: w13 [2*inter_tp, H], scale [2*inter_tp, H//32]
            w13_f = w13_e.view(2 * inter_tp, -1, 32) * w13_s_e.view(2 * inter_tp, -1, 1)
            w13_f = w13_f.reshape(2 * inter_tp, -1)
            w1_f = w13_f[:inter_tp]  # gate [inter_tp, H]
            w3_f = w13_f[inter_tp:]  # up   [inter_tp, H]

            gate = sub_x @ w1_f.T  # [N, inter_tp]
            up = sub_x @ w3_f.T

            if limit > 0:
                gate = gate.clamp(max=limit)
                up = up.clamp(-limit, limit)
            act = F.silu(gate) * up * sub_w  # weight before w2

            # Dequant w2 (down) from FP4
            w2_e = mxfp4_to_f32(w2[e_id])  # [H, inter_tp]
            w2_s_e = e8m0_to_f32(w2_s[e_id].contiguous().view(torch.float8_e8m0fnu))
            w2_f = w2_e.view(-1, w2_e.shape[1] // 1, 1) * 1.0  # placeholder
            # Correct: w2 [H, inter_tp], scale [H, inter_tp//32]
            w2_f = w2_e.view(w2_e.shape[0], -1, 32) * w2_s_e.view(
                w2_s_e.shape[0], -1, 1
            )
            w2_f = w2_f.reshape(w2_e.shape[0], -1)

            out = act.to(torch.bfloat16).float() @ w2_f.T  # [N, H]
            y[tok_idx] += out

        return y

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        if self.use_fused and self.use_torch_moe:
            # Torch fallback: use FusedMoE's select_experts for routing,
            # then per-expert torch loop with FP4 dequant + swiglu_limit clamp.
            router_logits = self.gate(x)
            if self.is_hash_layer:
                self._hash_input_ids = input_ids
            topk_weights, topk_ids = FusedMoE.select_experts(
                hidden_states=x,
                router_logits=router_logits,
                top_k=self.n_activated_experts,
                use_grouped_topk=False,
                renormalize=True,
                custom_routing_function=(
                    self._hash_topk if self.is_hash_layer else None
                ),
                scoring_func=self.experts.scoring_func,
                e_score_correction_bias=self.gate.e_score_correction_bias,
                routed_scaling_factor=self.routed_scaling_factor,
            )
            if self.is_hash_layer:
                self._hash_input_ids = None
            y = self._torch_moe_forward(x, topk_weights, topk_ids)
        elif self.use_fused:
            router_logits = self.gate(x)
            if self.is_hash_layer:
                self._hash_input_ids = input_ids
            y = self.experts(hidden_states=x, router_logits=router_logits)
            if self.is_hash_layer:
                self._hash_input_ids = None
        else:
            weights, indices = self.gate(x, input_ids.flatten())
            y = torch.zeros_like(x, dtype=torch.float32)
            counts = torch.bincount(
                indices.flatten(), minlength=self.n_routed_experts
            ).tolist()
            for i in range(self.n_routed_experts):
                if counts[i] == 0:
                    continue
                idx, top = torch.where(indices == i)
                y[idx] = y[idx] + self.experts[i](x[idx], weights[idx, top, None])
        y = y + self.shared_experts(x)
        if self.use_fused and self.tp_size > 1:
            from aiter.dist.communication_op import tensor_model_parallel_all_reduce

            y = tensor_model_parallel_all_reduce(y)
        return y.type_as(x).view(shape)


class Block(nn.Module):
    """Transformer block with Manifold-Constrained Hyper-Connections (mHC).

    Port of inference/model.py:648-701. The residual stream is widened to
    `[B, S, hc_mult, D]`. Each sub-layer (attn / ffn):
      1. `hc_pre`: project `[B, S, hc_mult, D]` -> `[B, S, D]` via Sinkhorn-projected
         pre-weights (also producing post-weights and combination matrix for hc_post).
      2. `attn_norm` + `attn` (or `ffn_norm` + `ffn`): standard sub-layer in `[B, S, D]`.
      3. `hc_post`: expand `[B, S, D]` back to `[B, S, hc_mult, D]` using the
         post-weights (gate on the new contribution) + the combination matrix
         applied to the previous residual.
    """

    def __init__(self, layer_id: int, args: DeepseekV4Args, prefix: str = ""):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = DeepseekV4Attention(layer_id, args, prefix=f"{prefix}.attn")
        self.ffn = MoE(layer_id, args, prefix=f"{prefix}.ffn")
        self.attn_norm = _RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = _RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        # All HC params stored in fp32 (matches reference's `set_dtype(torch.float32)`).
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    # mHC `hc_post_mult_value`: V4 uses `2.0 * sigmoid(post)` for the post gate.
    HC_POST_MULT = 2.0

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reduce [..., hc, D] residual to [..., D] sub-layer input.

        Prefers the fused aiter `mhc_pre` kernel (single ROCm op for RMSNorm +
        hc-fn linear + Sinkhorn projection + weighted reduction). Falls back to
        the torch `hc_split_sinkhorn` reference implementation when aiter is
        unavailable (toy validation paths in PR1).

        Shape-agnostic in leading dims: works for [B, S, hc, D] (legacy 4D) and
        [num_tokens, hc, D] (ATOM 2D-flat convention) alike.

        Returns: (layer_input, post, comb) — order matches torch fallback.
        """
        # Fused path: aiter.mhc_pre takes the residual as [m, hc_mult, dim] and
        # returns (post_mix [m, hc, 1], comb_mix [m, hc, hc], layer_input [m, dim]).
        # When the leading dims are >2 (e.g. [B, S, hc, D]), flatten to [B*S, hc, D]
        # and unflatten the outputs.
        try:
            import aiter as _aiter  # local import; avoids hard dep at module load

            mhc_pre = getattr(_aiter, "mhc_pre", None)
        except ImportError:
            mhc_pre = None
        # aiter mhc_pre kernel asserts hidden % 512 == 0 OR hidden % 256 == 0
        # (mhc_kernels.cu:864 calls __builtin_trap on violation, NOT a raise).
        # Pre-check in Python so toy / small-dim configs gracefully fall through.
        dim = x.shape[-1]
        aiter_ok = (
            mhc_pre is not None and x.is_cuda and (dim % 512 == 0 or dim % 256 == 0)
        )
        if aiter_ok:
            lead = x.shape[:-2]
            r = x.reshape(-1, *x.shape[-2:])  # [M, hc, D]
            post, comb, y = mhc_pre(
                r,
                hc_fn,
                hc_scale,
                hc_base,
                float(self.norm_eps),
                float(self.hc_eps),
                float(self.hc_eps),
                self.HC_POST_MULT,
                int(self.hc_sinkhorn_iters),
            )
            post = post.squeeze(-1)  # aiter: [M, hc, 1] → [M, hc]
            return (
                y.reshape(*lead, y.shape[-1]),
                post.reshape(*lead, post.shape[-1]),
                comb.reshape(*lead, *comb.shape[-2:]),
            )

        # Torch fallback (PR1 toy mode / no-aiter): mirrors the reference math.
        shape, dtype = x.size(), x.dtype
        x = x.flatten(-2).float()  # [..., hc*D]
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt  # [..., mix_hc]
        pre, post, comb = hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=-2)
        return y.to(dtype), post, comb

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Expand [..., D] sub-layer output back to [..., hc, D] residual.

        Prefers fused aiter `mhc_post`; falls back to torch.
        """
        try:
            import aiter as _aiter

            mhc_post = getattr(_aiter, "mhc_post", None)
        except ImportError:
            mhc_post = None
        # Same divisibility constraint as mhc_pre.
        dim = residual.shape[-1]
        aiter_ok = (
            mhc_post is not None and x.is_cuda and (dim % 512 == 0 or dim % 256 == 0)
        )
        if aiter_ok:
            lead = residual.shape[:-2]
            x_ = x.reshape(-1, x.shape[-1])
            r_ = residual.reshape(-1, *residual.shape[-2:])
            post_ = post.reshape(-1, post.shape[-1]).unsqueeze(-1)
            comb_ = comb.reshape(-1, *comb.shape[-2:])
            out = torch.empty_like(r_)
            mhc_post(out, x_, r_, post_, comb_)
            return out.reshape(*lead, *r_.shape[-2:]).type_as(x)

        # Torch fallback.
        # x: [..., D]; residual: [..., hc, D]
        # post.unsqueeze(-1) * x.unsqueeze(-2): [..., hc, D] gating
        # comb.unsqueeze(-1) * residual.unsqueeze(-2): [..., hc, hc, D]; sum over hc-dim
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=-3
        )
        return y.type_as(x)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        input_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # ----- Attention sub-layer with mHC mixing -----
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attn_norm(x)
        x = self.attn(x, start_pos)
        x = self.hc_post(x, residual, post, comb)

        # ----- FFN sub-layer with mHC mixing -----
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x


class ParallelHead(nn.Module):
    """LM head with mHC reduction.

    Port of inference/model.py:704-736. Unlike `Block.hc_pre` (which uses
    Sinkhorn projection on the combination matrix), `hc_head` uses simple
    `Sigmoid(mix*scale + base) + eps` weights to reduce the [B, S, hc, D]
    residual to [B, S, D] before applying the LM head linear projection.

    `get_logits` projects only the last token (decode mode); for prefill the
    caller should slice the desired positions before passing through.
    """

    def __init__(
        self, vocab_size: int, dim: int, norm_eps: float = 1e-6, hc_eps: float = 1e-6
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        # PR1 single-rank: full vocab on this rank.
        self.weight = nn.Parameter(
            torch.empty(self.vocab_size, self.dim, dtype=torch.float32)
        )

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Project the last-token slice of `x` to vocab logits.

        Accepts either:
          - 2D [num_tokens, D]: takes last row → [vocab]
          - 3D [B, S, D] (legacy): takes x[:, -1, :] → [B, vocab]
        """
        if x.dim() == 2:
            return F.linear(x[-1:].float(), self.weight)
        return F.linear(x[:, -1].float(), self.weight)

    def hc_head(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce [..., hc, D] → [..., D] via Sigmoid-gated weighted sum.

        Shape-agnostic in leading dims (mirrors Block.hc_pre / hc_post).
        """
        shape, dtype = x.size(), x.dtype
        x = x.flatten(-2).float()  # [..., hc*D]
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=-2)
        return y.to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm: nn.Module,
    ) -> torch.Tensor:
        x = self.hc_head(x, hc_fn, hc_scale, hc_base)
        logits = self.get_logits(norm(x))
        # PR1 single-rank: skip all_gather
        return logits


class MTPBlock(Block):
    """MTP block: V4 dense block + e_proj/h_proj/enorm/hnorm + own hc_head params + LM head.

    Port of inference/model.py:739-767. Subclass of Block reusing all HC + Attention + FFN
    machinery; adds a token-embed projection (`e_proj`), a hidden-state projection
    (`h_proj`), per-input RMSNorms, and its own `hc_head_fn/base/scale` parameters
    for the final LM head reduction.

    `embed` and `head` are assigned externally by `DeepseekV4Model` (shared with
    the main model's embedding and LM head).
    """

    def __init__(self, layer_id: int, args: DeepseekV4Args, prefix: str = ""):
        super().__init__(layer_id, args, prefix=prefix)
        # e_proj / h_proj are FP8 on disk per index; ATOM Linear with V4QuantConfig
        # picks per_1x128 automatically. nn.Linear at construction works for the
        # toy/dummy path; for real-checkpoint loading, switch to ReplicatedLinear.
        qc = args.quant_config
        if qc is None:
            self.e_proj = nn.Linear(args.dim, args.dim, bias=False)
            self.h_proj = nn.Linear(args.dim, args.dim, bias=False)
        else:
            self.e_proj = ReplicatedLinear(
                args.dim,
                args.dim,
                bias=False,
                quant_config=qc,
                prefix=f"{prefix}.e_proj",
            )
            self.h_proj = ReplicatedLinear(
                args.dim,
                args.dim,
                bias=False,
                quant_config=qc,
                prefix=f"{prefix}.h_proj",
            )
        self.enorm = _RMSNorm(args.dim, args.norm_eps)
        self.hnorm = _RMSNorm(args.dim, args.norm_eps)
        self.norm = _RMSNorm(args.dim, args.norm_eps)
        # Per-MTP hc_head params (distinct from Block's hc_attn/hc_ffn params).
        hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        self.hc_head_fn = nn.Parameter(
            torch.empty(hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        # Externally-assigned by DeepseekV4Model (shared with main model).
        self.embed: Optional[nn.Module] = None
        self.head: Optional[ParallelHead] = None

    def forward(
        self, x: torch.Tensor, start_pos: int, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Forward.

        Args:
            x: residual stream from main model. Either [num_tokens, hc, D]
                (ATOM 2D-flat convention) or [B, S, hc, D] (legacy 4D).
            start_pos: absolute position offset.
            input_ids: matching token ids.
        Returns:
            Logits of the last token (vocab projected by self.head).
        """
        assert (
            self.embed is not None and self.head is not None
        ), "MTPBlock requires .embed and .head to be assigned by the parent model"
        e = self.embed(input_ids)  # [num_tokens, D] or [B, S, D]
        e = self.enorm(e)
        x = self.hnorm(x)
        # Mix embedding + hidden into a fresh residual stream. The unsqueeze
        # adds the hc dim before the trailing D so [num_tokens, D] → [num_tokens, 1, D]
        # broadcasts correctly against x [num_tokens, hc, D]. Same for 4D path.
        x = self.e_proj(e).unsqueeze(-2) + self.h_proj(x)
        x = super().forward(x, start_pos, input_ids)
        logits = self.head(
            x, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm
        )
        return logits


class DeepseekV4Model(nn.Module):
    """Full model: embed -> expand to hc_mult copies -> N blocks -> hc_head -> logits.

    Port of inference/model.py:Transformer (770-810). MTP blocks are constructed
    and have their `.embed` and `.head` linked to the main model's, but they are
    NOT called from the main forward path — PR5 will integrate them into ATOM's
    EagleProposer via `self.mtp[k].forward(...)` from outside.

    PR1 single-rank: uses plain `nn.Embedding` for `self.embed` (state_dict-compatible
    with reference's `ParallelEmbedding` since both store a single `weight` parameter).
    """

    def __init__(self, *, args: DeepseekV4Args):
        super().__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.hc_mult = args.hc_mult

        # VocabParallelEmbedding shards along vocab dim. At TP=1 weight shape
        # equals nn.Embedding's [vocab_size, dim] so dummy state_dicts load
        # directly. At TP>1 each rank holds vocab_size/tp rows.
        self.embed = VocabParallelEmbedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList(
            [
                Block(layer_id, args, prefix=f"layers.{layer_id}")
                for layer_id in range(args.n_layers)
            ]
        )
        self.norm = _RMSNorm(args.dim, self.norm_eps)
        self.head = ParallelHead(args.vocab_size, args.dim, self.norm_eps, self.hc_eps)

        # MTP blocks: constructed and linked, but only invoked externally (PR5).
        self.mtp = nn.ModuleList()
        for layer_id in range(args.n_mtp_layers):
            blk = MTPBlock(args.n_layers + layer_id, args, prefix=f"mtp.{layer_id}")
            blk.embed = self.embed
            blk.head = self.head
            self.mtp.append(blk)

        # Top-level hc_head params used to reduce the final hc_mult residual stack
        # before the LM head linear projection.
        hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        self.hc_head_fn = nn.Parameter(
            torch.empty(hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        **model_kwargs: dict,
    ) -> torch.Tensor:
        """Forward.

        Args:
            input_ids: 1D `[num_tokens]` (ATOM 2D-flat convention) OR 2D
                `[B, S]` (legacy reference convention; treated as a single
                sequence of B*S tokens — only correct for B=1).
            start_pos: absolute position of the first token.
        Returns:
            Logits of the last token: `[vocab]` (1D path) or `[B, vocab]` (2D path).
        """
        # Normalize input_ids to 1D [num_tokens] for the 2D internal convention.
        if input_ids.dim() == 2:
            assert (
                input_ids.size(0) == 1
            ), "B>1 batched input_ids needs attn_metadata; not supported yet"
            input_ids = input_ids.flatten()
        h = self.embed(input_ids)  # [num_tokens, dim]
        # Expand to hc_mult copies for Hyper-Connections: [num_tokens, hc, dim]
        h = h.unsqueeze(-2).repeat(1, self.hc_mult, 1)

        for layer in self.layers:
            h = layer(h, start_pos, input_ids)

        logits = self.head(
            h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm
        )
        return logits


class DeepseekV4ForCausalLM(nn.Module):
    """ATOM model contract wrapper.

    Loads via two paths:
    - `model.load_weights(...)` (this file): used by tests + when ModelRunner
      is bypassed. Handles V4 ckpt naming + FP8 wo_a dequant + FusedMoE expert
      dispatch in one place.
    - `atom.model_loader.loader.load_model(...)` (standard ATOM serving): uses
      the `weights_mapping` class attribute below to rename V4 ckpt names into
      shapes the standard FusedMoE expert mapping understands. Wo_a dequant
      and other special cases are handled by the `process_weights_after_loading`
      path on the relevant Linear modules (TODO PR4).
    """

    # Disk-name → param-name renames applied by atom.model_loader.loader.load_model.
    #
    # We use a `WeightsMapper` (prefix/suffix-anchored) for the `model.` prefix
    # injection because the V4 HF checkpoint stores bare names (`norm.weight`,
    # `head.weight`, `embed.weight`, `layers.X.*`, `hc_head_*`, `mtp.X.*`) and
    # our model lives under `self.model = DeepseekV4Model(...)` so all params
    # are accessed via `model.<name>`. The legacy `weights_mapping` substring
    # dict CANNOT express this safely: `"norm.weight" → "model.norm.weight"`
    # also matches inside `attn_norm.weight` / `ffn_norm.weight` / `q_norm.weight`
    # / `compressor.norm.weight` etc. and silently corrupts the lookup
    # (b87f6f, debugged via the `load_model` post-load WARNING).
    #
    # The substring dict is reserved for the renames that ARE legitimately
    # substring-shaped:
    # - `.gate.bias` → `.gate.e_score_correction_bias` (V4's routed-expert
    #   score correction bias has a different name in our model)
    # - `.scale` → `.weight_scale_inv` (V4 ckpt suffix → ATOM's expected name;
    #   load_model then auto-renames `_inv` → `` so the final param is
    #   `.weight_scale`).
    from atom.model_loader.loader import WeightsMapper as _WeightsMapper

    weights_mapper = _WeightsMapper(
        orig_to_new_prefix={
            "embed.": "model.embed.",
            "layers.": "model.layers.",
            "norm.weight": "model.norm.weight",
            "head.weight": "model.head.weight",
            "hc_head_": "model.hc_head_",
            "mtp.": "model.mtp.",
        }
    )
    weights_mapping = {
        ".gate.bias": ".gate.e_score_correction_bias",
        ".scale": ".weight_scale_inv",
    }

    def __init__(self, config: Config, prefix: str = "") -> None:
        super().__init__()
        self.atom_config = config
        self.hf_config = config.hf_config
        self.args = DeepseekV4Args.from_hf_config(self.hf_config)
        # Build the V4-specific QuantizationConfig (FP8 default + FP4 experts +
        # BF16 wo_a/Compressor) so child Linear layers auto-build the right
        # weight + scale params for real-checkpoint loading. When the HF
        # config lacks `quantization_config` (e.g. dummy / toy validation),
        # this still works — base spec is QuantType.No.
        self.args.quant_config = make_v4_quant_config(self.hf_config)
        self.model = DeepseekV4Model(args=self.args)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[Any] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **model_kwargs: dict,
    ) -> torch.Tensor:
        start_pos = int(positions[0].item()) if positions is not None else 0
        return self.model(input_ids=input_ids, start_pos=start_pos, **model_kwargs)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # In V4, the LM head is fused into DeepseekV4Model.forward (it consumes
        # the hc_mult-expanded residual). So `hidden_states` already IS logits.
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """Return (param_name, weight_name, expert_id, shard_id) tuples for FusedMoE.

        V4 expert weights on disk are named `ffn.experts.{e}.w{1,2,3}`. Pass
        these as the gate/down/up names to FusedMoE.make_expert_params_mapping.
        """
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.args.n_routed_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from an iterable of (name, tensor) pairs.

        Naming conventions (HF V4 checkpoint matches our internal naming 1:1):
            embed.weight
            layers.{i}.attn.{wq_a,q_norm,wq_b,wkv,kv_norm,wo_a,wo_b,attn_sink,...}
            layers.{i}.attn.compressor.{ape,wkv,wgate,norm}
            layers.{i}.attn.indexer.{wq_b,weights_proj}
            layers.{i}.attn.indexer.compressor.{...}
            layers.{i}.ffn.gate.{weight,bias|tid2eid}
            layers.{i}.ffn.experts.{e}.w{1,2,3}
            layers.{i}.ffn.shared_experts.w{1,2,3}
            layers.{i}.{attn_norm,ffn_norm}
            layers.{i}.{hc_attn_*,hc_ffn_*}
            mtp.{i}.{...}                    (same shape as a Block + e_proj/h_proj/...)
            norm.weight, head.weight, hc_head_*

        On-disk quirks:
        - FP8/FP4 scale tensors are named `<param>.scale`; ATOM internally names
          them `<param>.weight_scale`. Remap on lookup.
        - `wo_a` is FP8 + scale on disk but BF16 in our model (V4QuantConfig
          forces no_spec; aiter has no FP8 grouped-einsum). Dequantize the FP8
          weight using the on-disk scale before copying into the BF16 param.

        Returns:
            Set of parameter names successfully loaded.
        """
        loaded: set[str] = set()
        # Index all our params + buffers for fast lookup.
        targets: dict[str, torch.Tensor] = dict(self.model.named_parameters())
        targets.update(dict(self.model.named_buffers()))

        # First pass: bucket on-disk tensors by their candidate target names.
        # Some special-case tensors (wo_a.weight + wo_a.scale → BF16) need to be
        # processed together, so collect all tensors first then resolve.
        scratch: dict[str, torch.Tensor] = {}
        for name, tensor in weights:
            scratch[name] = tensor

        # ----- FusedMoE expert weight dispatch (PR3b) -----
        # Routed expert weights `layers.{i}.ffn.experts.{e}.w{1,2,3}.{weight,scale}`
        # on disk go to FusedMoE's merged `experts.w13_*` / `experts.w2_*` params.
        # The mapping uses substring substitution: `experts.{e}.w1.` (weight_name_part)
        # → `experts.w13_` (param_name_part), keeping the `weight` / `scale` suffix.
        try:
            expert_mapping = self.get_expert_mapping()
        except Exception:
            expert_mapping = []
        # Build longest-first index for unambiguous matching (shared with std loader).
        expert_index: dict[str, tuple[str, int, str]] = {}
        for param_part, weight_part, expert_id, shard_id in expert_mapping:
            expert_index[weight_part] = (param_part, expert_id, shard_id)
        weight_parts_sorted = sorted(expert_index.keys(), key=len, reverse=True)

        consumed: set[str] = set()
        for ckpt_name in list(scratch.keys()):
            if "ffn.experts." not in ckpt_name and "experts." not in ckpt_name:
                continue
            # Skip the routed-gate/non-expert tensors that just live alongside.
            for wpart in weight_parts_sorted:
                if wpart not in ckpt_name:
                    continue
                ppart, expert_id, shard_id = expert_index[wpart]
                tgt_name = ckpt_name.replace(wpart, ppart)
                # FusedMoE expert scales: on-disk `.{shard_id}.scale` → param `_weight_scale`
                # After substring sub `experts.{e}.w1.` → `experts.w13_`, the suffix
                # becomes `_scale`; rename to match FusedMoE's `_weight_scale` param.
                if tgt_name.endswith("_scale"):
                    tgt_name = tgt_name[: -len("_scale")] + "_weight_scale"
                elif tgt_name.endswith(".scale"):
                    tgt_name = tgt_name[: -len(".scale")] + ".weight_scale"
                param = targets.get(tgt_name)
                if param is None:
                    break
                loader = getattr(param, "weight_loader", None)
                if loader is None:
                    break
                tensor = scratch[ckpt_name].to(param.device)
                # Dtype glue:
                # - FP4 packed weights: disk is int8, param is float4_e2m1fn_x2;
                #   FusedMoE._load_w13/w2 already does `.view(torch.uint8)` for fp4x2
                #   params, but only when the loaded tensor dtype matches.
                # - FP8 e8m0 scale: disk is float8_e8m0fnu, param is uint8;
                #   torch's copy_ between mismatched dtypes silently zeros, so
                #   force a uint8 view here.
                if tensor.dtype == torch.float8_e8m0fnu and param.dtype == torch.uint8:
                    tensor = tensor.view(torch.uint8)
                if tensor.dtype == torch.int8 and param.dtype == torch.float4_e2m1fn_x2:
                    tensor = tensor.view(torch.uint8)
                loader(
                    param,
                    tensor,
                    tgt_name,  # weight_name (post-mapping; "scale" substring drives scale dispatch)
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                loaded.add(tgt_name)
                consumed.add(ckpt_name)
                break
        # Drop consumed expert tensors so the second loop doesn't re-process them.
        for k in consumed:
            scratch.pop(k, None)

        for tgt_name, param in targets.items():
            ckpt_name = tgt_name
            # ATOM scale → on-disk scale name
            if ckpt_name.endswith(".weight_scale"):
                alt = ckpt_name.replace(".weight_scale", ".scale")
                if alt in scratch:
                    ckpt_name = alt
            # ATOM `gate.e_score_correction_bias` ↔ on-disk `gate.bias`
            if ckpt_name.endswith(".gate.e_score_correction_bias"):
                alt = ckpt_name.replace(".gate.e_score_correction_bias", ".gate.bias")
                if alt in scratch:
                    ckpt_name = alt
            if ckpt_name not in scratch:
                continue

            # NOTE: previously wo_a had a manual FP8+scale → BF16 dequant special
            # case here. wo_a is now FP8 ColumnParallelLinear in the model so
            # weight + scale load through the standard FP8 path. Dequant happens
            # in DeepseekV4Attention.process_weights_after_loading (called via the
            # post-load hook walk at the end of this method).

            tensor = scratch[ckpt_name].to(param.device)

            # Shape mismatch handling:
            # - When test caps n_routed_experts (e.g. 8 vs disk 384), the on-disk
            #   gate.weight/bias are larger than param. Slice to the first N rows.
            #   Real serving uses full 384 so this is a no-op there.
            # - Other shape mismatches indicate a true wiring bug → skip safely.
            if param.shape != tensor.shape:
                can_slice = param.dim() == tensor.dim() and all(
                    ps <= ts for ps, ts in zip(param.shape, tensor.shape, strict=True)
                )
                if can_slice:
                    slices = tuple(slice(0, s) for s in param.shape)
                    tensor = tensor[slices].contiguous()
                else:
                    continue

            loader = getattr(param, "weight_loader", None)
            if loader is not None:
                loader(param, tensor)
            else:
                if (
                    param.dtype != tensor.dtype
                    and param.dtype == torch.float4_e2m1fn_x2
                ):
                    param.data.view(torch.uint8).copy_(tensor.view(torch.uint8))
                else:
                    param.data.copy_(tensor.to(param.dtype))
            loaded.add(tgt_name)

        # Trigger post-load hooks (e.g. FusedMoE's `process_weights_after_loading`
        # runs `shuffle_weights` so aiter ck_moe sees the right layout). Without
        # this the FP4 ck_moe kernel reads stale layout → HSA crash at forward.
        for module in self.model.modules():
            ppl = getattr(module, "process_weights_after_loading", None)
            if callable(ppl):
                # quant_method.process_weights_after_loading(layer) — quant_method
                # is the FusedMoE attribute, layer is the module itself.
                qm = getattr(module, "quant_method", None)
                if qm is not None and hasattr(qm, "process_weights_after_loading"):
                    qm.process_weights_after_loading(module)
                else:
                    ppl()
        return loaded
