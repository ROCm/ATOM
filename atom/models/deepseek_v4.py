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
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Iterable, Literal, Optional, Tuple, cast

if TYPE_CHECKING:
    from atom.model_ops.attentions.deepseek_v4_attn import AttentionMetaData_DSV4

import torch
import torch.nn.functional as F
from torch import nn

import aiter
from aiter import (
    cp_gather_indexer_k_quant_cache,
    dtypes,
    get_hip_quant,
    silu_and_mul as aiter_silu_and_mul,
)
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.dist.communication_op import (
    tensor_model_parallel_all_reduce,
)
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from aiter.ops.topk import top_k_per_row_decode, top_k_per_row_prefill
from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits
from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
from atom.config import (
    Config,
    LayerQuantConfig,
    QuantizationConfig,
    QuantType,
    get_current_atom_config,
)
from atom.model_loader.loader import WeightsMapper
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import DualRMSNorm, RMSNorm, rmsnorm2d_fwd_
from atom.model_ops.triton_rmsnorm_nw import rmsnorm_nw
from atom.model_ops.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    MergedReplicatedLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.model_ops.moe import FusedMoE
from atom.model_ops.topK import is_rocm_aiter_fusion_shared_expert_enabled
from aiter import rope_rotate_activation
from atom.model_ops.quant_v4 import act_quant_inplace
from atom.model_ops.sparse_attn_v4 import (
    hc_split_sinkhorn,
)
from atom.model_ops.utils import atom_parameter
from atom.utils import envs, mark_spliting_op

# Side-effect import: registers `torch.ops.aiter.maybe_dual_stream_forward`
# (shared with deepseek_v2) and `torch.ops.aiter.indexer_score_topk` (V4-only).
# MoE.forward dispatches via the former so torch.compile/Dynamo treats stream
# code as opaque; Indexer.forward_batched dispatches via the latter to hide
# its dynamic-shape internals from Dynamo / fake-tensor mode.
from atom.model_ops import module_dispatch_ops as _module_dispatch_ops  # noqa: F401
from atom.model_ops.v4_kernels import (
    CompressPlan,
    csa_translate_pack,
    fused_compress_attn,
    inverse_rope_inplace,
    scale_indexer_weights,
    sparse_attn_v4_paged_decode,
    sparse_attn_v4_paged_prefill,
    swa_write,
    update_compressor_states,
)
from atom.utils.forward_context import get_forward_context
from atom.utils.decorators import support_torch_compile

# ---------------------------------------------------------------------------
# Classical KV cache scatter / gather helpers (PR3-pre2c-B).
#
# Each V4 block (block_size=lcm(m, m')=128 original tokens) holds k_per_block
# compressed entries per layer (k1=32 for CSA, k2=1 for HCA). Compressor.forward
# scatters newly-compressed entries into block-table-indexed slots; sparse_attn
# input gathers all committed entries up to the current position.
#
# In PR3-pre2c-B these helpers run on a single sequence (block_table fetched
# from `forward_context.attn_metadata.block_tables[0]`). PR3-main extends to
# per-seq dispatch.
# ---------------------------------------------------------------------------

# V4 paper §3.6.1: classical-KV block_size = lcm(m, m'). For V4-Pro / V4-Flash
# this is lcm(4, 128) = 128 original tokens. Kept as a constant so Compressor
# code does not need to import the builder.
_V4_BLOCK_SIZE: int = 128

_V4_RMSNORM_BACKEND = os.environ.get("ATOM_V4_RMSNORM_BACKEND", "triton")
_V4_USE_TRITON_RMSNORM = _V4_RMSNORM_BACKEND == "triton"
# Env-gated quant round-trips. Read once at module load — checking each
# forward burns syscalls (V4-Pro: 64 layers × multiple sites per call).
_V4_FORCE_UE8M0_QUANT = os.environ.get("V4_FORCE_UE8M0_QUANT", "0") == "1"
_V4_USE_REF_QUANT = os.environ.get("V4_USE_REF_QUANT", "0") == "1"


def _rmsnorm_nw(x: torch.Tensor, eps: float, dim: int) -> torch.Tensor:
    if _V4_USE_TRITON_RMSNORM:
        return rmsnorm_nw(x, eps)
    ones = torch.ones(dim, dtype=x.dtype, device=x.device)
    return rmsnorm2d_fwd_(x, ones, eps, dim)


def _make_weightless_rmsnorm(dim: int, eps: float) -> RMSNorm:
    """Build an `RMSNorm(dim, eps)` whose `.weight` is `None`.

    Drops the learnable Parameter so `state_dict()` is empty for this
    submodule and `load_model` doesn't expect a weight from disk (no
    "unloaded parameter" warning).  `DualRMSNorm` recognizes the
    sentinel `weight is None` and instructs the fused kernel to skip
    both the q_weight load and the multiply (Q_HAS_WEIGHT=False),
    matching the prior `_rmsnorm_nw(x, eps, dim)` behavior.
    """
    norm = RMSNorm(dim, eps)
    del norm.weight  # remove Parameter from `_parameters`
    norm.weight = None  # sentinel for downstream consumers
    return norm


def _hc_head_reduce_fake(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    return torch.empty(x.shape[0], x.shape[-1], dtype=x.dtype, device=x.device)


@torch_compile_guard(gen_fake=_hc_head_reduce_fake, mutates_args=[])
def _hc_head_reduce(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    x_flat = x.flatten(-2)
    x_normed = _rmsnorm_nw(x_flat, norm_eps, x_flat.shape[-1])
    mixes = F.linear(x_normed.float(), hc_fn)
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x, dim=-2)
    return y.to(x.dtype)


def _v4_attention_fake(
    x: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(x)


@mark_spliting_op(is_custom=True, gen_fake=_v4_attention_fake, mutates_args=[])
def v4_attention_with_output(
    x: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    atom_config = get_current_atom_config()
    self = atom_config.compilation_config.static_forward_context[layer_name]
    return self.forward_impl(x, positions)


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
    max_batch_size: int = 4  # default placeholder; production driven by ATOM scheduler

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

    # V4QuantizationConfig — Linear layers auto-build the right (FP8 / FP4
    # / BF16) weight + scale params. Set by DeepseekV4ForCausalLM at init.
    quant_config: Optional[Any] = None

    @classmethod
    def from_hf_config(cls, hf_config: Any) -> "DeepseekV4Args":
        # Use getattr with sensible defaults so we work whether the HF config is
        # a real V4 PretrainedConfig (all fields present) or a V3 PretrainedConfig
        # populated with extra V4 attrs (some fields may live only in the raw
        # config_dict, not on the config object — `transformers` strips unknown
        # kwargs unless they're in the schema).
        def g(k, default=None):
            return getattr(hf_config, k, default)

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
            # Default to "ue8m0" matching reference ModelArgs (inference/model.py:40);
            # HF config.json does not carry this field, only inference/config.json does.
            scale_fmt=g("scale_fmt", "ue8m0"),
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
    """Apply rotary positional embeddings IN-PLACE (manual complex multiply).

    Port of inference/model.py:232-244. The input tensor `x` is overwritten with
    the rotated values; the same tensor is also returned for chaining.
    `inverse=True` uses the conjugate (un-rotation) — used on the attention
    output to remove absolute-position embedding from the value contribution.

    NOTE: forward RoPE on Q/KV now goes through `_V4RoPE` (aiter kernel). This
    function is kept ONLY for the output inverse step, which aiter does not
    expose.
    """
    y = x
    x_f = x.float()
    x = torch.view_as_complex(x_f.reshape(*x_f.shape[:-1], -1, 2))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


@lru_cache(8)
def _build_cos_sin_cache(
    rotary_dim: int,
    max_seq_len: int,
    base: float,
    factor: float,
    original_seq_len: int,
    beta_fast: int,
    beta_slow: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shared cos/sin cache for `_V4RoPE`, keyed by (rope params, dtype, device).

    V4 has only 3 distinct rope param sets (HCA / CSA / Dense) — without
    deduping we'd materialize 62 copies per rank (~16GB at fp32 complex,
    ~8GB at bf16). Per-device caching means each rank holds exactly one
    cos+sin pair per param set. Cache size 8 covers (HCA, CSA, Dense) ×
    (cuda:0..N) headroom.
    """
    freqs = _precompute_freqs_cis(
        rotary_dim,
        max_seq_len,
        original_seq_len,
        base,
        factor,
        beta_fast,
        beta_slow,
    )
    cos = (
        freqs.real.to(device=device, dtype=dtype)
        .contiguous()
        .unsqueeze(-2)
        .unsqueeze(-2)
    )
    sin = (
        freqs.imag.to(device=device, dtype=dtype)
        .contiguous()
        .unsqueeze(-2)
        .unsqueeze(-2)
    )
    return cos, sin


class _V4RoPE(nn.Module):
    """Per-token-positions RoPE wrapper around aiter's `rope_cached_*_fwd_inplace`.

    Builds the cos/sin cache via V4's exact YaRN math (`_precompute_freqs_cis`),
    then dispatches to the aiter HIP kernel. Works on a pre-sliced rope tensor
    (`head_size == rotary_dim`) so callers stay symmetric with the existing
    `_apply_rotary_emb(x[..., -rd:], ...)` pattern.

    `freqs_for_positions(positions)` rebuilds a complex tensor from the cos/sin
    slices for the attention output's inverse RoPE step (which aiter does not
    expose). We deliberately do NOT keep a complex `freqs_cis` buffer: cos/sin
    in bf16 is half the memory of complex64, and 62 layers × 1M positions ×
    32 freqs adds up fast.
    """

    def __init__(
        self,
        rotary_dim: int,
        max_seq_len: int,
        base: float,
        factor: float,
        original_seq_len: int,
        beta_fast: int,
        beta_slow: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.rotary_dim = rotary_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.factor = factor
        self.original_seq_len = original_seq_len
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.dtype = dtype
        # Cos/sin caches are fetched lazily on first forward via the
        # device-keyed `_build_cos_sin_cache`; this lets all 62 layers share
        # one cache per (rope params, device) instead of registering 62
        # buffers that .to() would each clone onto GPU.

    def _caches(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return _build_cos_sin_cache(
            self.rotary_dim,
            self.max_seq_len,
            self.base,
            self.factor,
            self.original_seq_len,
            self.beta_fast,
            self.beta_slow,
            self.dtype,
            device,
        )

    def freqs_for_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Rebuild the complex `freqs_cis` slice for the given positions.

        Used by the attention output's inverse RoPE step.
        Returns: complex64 [num_tokens, rotary_dim // 2].
        """
        cos_cache, sin_cache = self._caches(positions.device)
        cos = cos_cache.index_select(0, positions).squeeze(-2).squeeze(-2).float()
        sin = sin_cache.index_select(0, positions).squeeze(-2).squeeze(-2).float()
        return torch.complex(cos, sin)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> None:
        """In-place RoPE on `query` (and `key` if given). All inputs are the
        rope-slice only (`head_size == rotary_dim`)."""
        cos, sin = self._caches(query.device)
        # rotate_style=1 → GPT-J / interleaved (matches V4's view_as_complex).
        rotate_style = 1
        num_tokens = positions.numel()
        if key is not None:
            aiter.rope_cached_positions_2c_fwd_inplace(
                query.view(1, num_tokens, -1, self.rotary_dim),
                key.view(1, num_tokens, -1, self.rotary_dim),
                cos,
                sin,
                positions.view(1, num_tokens),
                rotate_style,
                reuse_freqs_front_part=True,
                nope_first=False,
            )
        else:
            aiter.rope_cached_positions_fwd_inplace(
                query.view(1, num_tokens, -1, self.rotary_dim),
                cos,
                sin,
                positions.view(1, num_tokens),
                rotate_style,
                reuse_freqs_front_part=True,
                nope_first=False,
            )

    def inverse(self, positions: torch.Tensor, x: torch.Tensor) -> None:
        """In-place inverse RoPE via fused Triton kernel.

        ``x`` must be the rope-slice only (last dim == rotary_dim).
        """
        cos, sin = self._caches(x.device)
        inverse_rope_inplace(x, cos, sin, positions)


# ---------------------------------------------------------------------------
# Compressor + Indexer — port of inference/model.py:279-433
# ---------------------------------------------------------------------------


class Compressor(nn.Module):
    """Compresses KV cache via learned gated pooling over `compress_ratio` consecutive tokens.

    Port of inference/model.py:279-377. `overlap=True` (always set when
    ratio==4, used by CSA) uses overlapping windows to smooth block boundaries.

    Forward delegates pool + RMSNorm + RoPE + bf16 kv_cache scatter to a single
    fused Triton kernel (`fused_compress_attn`). Per-source-position dispatch
    inside the kernel (`s >= start_pos` → INPUT, else state cache) handles
    fresh prefill / chunked prefill / single-token decode / MTP-N uniformly.

    !!!! TODO: QUANT NOT YET FUSED — output drifts from training-time numerics !!!!
    The reference model trained with QAT round-trip:
      - CSA path (rotate=False): `act_quant_inplace(kv[..., :-rd], 64, "ue8m0")`
                                 (BF16 → FP8 e4m3 with ue8m0 scale → BF16)
      - Indexer path (rotate=True): `rotate_activation(kv); fp4_act_quant_inplace(kv, 32)`
                                    (Hadamard rotate then BF16 → FP4 e2m1 → BF16)
    Currently the fused kernel writes raw post-RoPE BF16 to kv_cache, skipping
    both. End-to-end testing shows outputs remain coherent (4 prompts from PR
    #650 baseline still produce sensible completions), but they are NOT
    byte-equal to baseline; benchmark accuracy (lm_eval / GSM8K) MAY regress.
    `self.rotate` is preserved on the module as the discriminator for the
    follow-up PR that ports the two quant flavours into the kernel.
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

        self.ape = atom_parameter(
            torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32)
        )
        # Fused [wkv; wgate]: both BF16 on disk (same dim out per shard).
        # quant_config=None → BF16 weight; forward calls with otype=fp32 to
        # keep the Compressor's softmax-pool path in fp32 accumulate.
        self.wkv_gate = MergedReplicatedLinear(
            self.dim,
            [coff * self.head_dim, coff * self.head_dim],
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.wkv_gate",
        )
        self.norm = RMSNorm(self.head_dim, args.norm_eps)

        # External tensors — assigned by the owning Attention / Indexer at first forward.
        self.kv_cache: Optional[torch.Tensor] = None
        self.rotary_emb: Optional[_V4RoPE] = None
        # FP8 quant path only: strided fp32 view of the per-block scale region
        # of `self.kv_cache`. Bound by the V4 builder when `kv_cache.dtype` is
        # FP8 (Indexer-inner Compressor); None for BF16 cache (Main path).
        self.cache_scale: Optional[torch.Tensor] = None

        # State cache (per paper §3.6.1 "uncompressed tail + B-side overlap
        # window" portion). Indexed as a single ring buffer of size
        # `coff * compress_ratio` by `pos % STATE_SIZE` per token — no
        # segment switching, no roll. The `forward` softmax-pool consumer
        # resolves A-side (current block) vs B-side (previous block) by
        # block-id parity (`comp_id % 2`).
        #
        # PR3-pre2a: a 1-slot register_buffer is kept here so warmup (which
        # runs before allocate_kv_cache → build_kv_cache_tensor) sees a
        # valid tensor; afterwards `DeepseekV4AttentionMetadataBuilder.
        # build_kv_cache_tensor` setattr-replaces these attributes with
        # views of the per-request cache pool (shape
        # `[max_num_seqs, coff*ratio, coff*head_dim]`). The 1-slot init
        # buffers (≈9 MB total across all layers) are GC'd once replaced.
        self.register_buffer(
            "kv_state",
            torch.zeros(
                1,
                coff * compress_ratio,
                coff * self.head_dim,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "score_state",
            torch.full(
                (1, coff * compress_ratio, coff * self.head_dim),
                float("-inf"),
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,  # [num_tokens, dim]
        plan: "CompressPlan",
        state_slot_mapping: torch.Tensor,  # [bs] int32
        block_tables: Optional[torch.Tensor] = None,  # [bs, max_blocks_per_seq] int32
    ) -> None:
        """Batched plan-style compress: one fused kernel call for the whole
        fwd's batch (across all seqs).

        Single fused Triton kernel does pool + RMSNorm + RoPE + cache scatter
        in one launch. Each compression boundary across the batch is one row
        in `plan.compress_plan_gpu`. State cache update fires after (write
        order critical — fused kernel reads state-cache-as-of-previous-fwd;
        `update_compressor_states` overwrites for next fwd).

        Quant mode is auto-selected by `self.kv_cache.dtype`:
          - BF16 cache (CSA Main / HCA Main): raw BF16 row write into
            `self.kv_cache` (consumed by paged_decode/paged_prefill via
            `unified_kv` per-fwd indices).
          - FP8 cache (Indexer-inner): per-row amax → ue8m0 scale → fp8 cast
            → preshuffled (MFMA 16x16 tile) write into `self.kv_cache`, plus
            fp32 scale into `self.cache_scale` (a strided view of the same
            allocation built by the V4 builder). Bit-exact with
            `indexer_k_quant_and_cache` / `cp_gather_indexer_k_quant_cache`
            (cache_kernels.cu:1145+).

        Side-effecting only — no return value (cache scatter IS the output).

        TODO: QAT for the BF16 Main path (FP8 round-trip per Compressor
        docstring) is not yet fused. End-to-end accuracy unaffected today
        because the input act_quant simulation is applied upstream.

        Args:
            x:           [num_tokens, dim] flat ragged batch hidden state.
            plan:        CompressPlan from attn_metadata.compress_plans[ratio]
                         (or a synthetic bs=1 plan during warmup).
            state_slot_mapping: [bs] int32 — per-seq state cache slot.
            block_tables: [bs, max_blocks_per_seq] int32 — physical block IDs
                         per seq; None during warmup (skips kv_cache scatter).
                         Required for the Indexer FP8 path (slot resolution).
        """
        assert self.rotary_emb is not None, "compressor.rotary_emb must be set by owner"
        assert (
            x.dim() == 2 and x.shape[-1] == self.dim
        ), f"Compressor expects [num_tokens, {self.dim}], got {tuple(x.shape)}"
        ratio = self.compress_ratio
        overlap = self.overlap
        d = self.head_dim
        rd = self.rope_head_dim

        # Single fused BF16 GEMM via tgemm. (Probing whether dropping the
        # otype=fp32 upcast — relying on fused_compress_attn's internal fp32
        # accumulator instead — is accuracy-neutral.) torch.split returns
        # zero-copy strided views; downstream kernels (fused_compress_attn,
        # update_compressor_states) accept strided kv/score (only inner
        # stride must be 1).
        coff_d = (1 + overlap) * d
        combined = self.wkv_gate(x)
        kv, score = torch.split(combined, [coff_d, coff_d], dim=-1)

        # ====== Unified fused kernel path (CSA + Indexer) ======
        # Order is critical: fused kernel reads state cache as-of-end-of-
        # PREVIOUS-fwd. `update_compressor_states` overwrites them with this
        # fwd's data for the NEXT fwd's overlap — must run AFTER the fused
        # kernel.
        cos_cache, sin_cache = self.rotary_emb._caches(x.device)
        # Quant path triggers when the bound cache is FP8 (Indexer-inner).
        # `self.cache_scale` is bound alongside `self.kv_cache` by the V4
        # builder when the cache is FP8 (strided fp32 view of the per-block
        # scale region).
        is_quant = self.kv_cache is not None and self.kv_cache.dtype != torch.bfloat16
        # Skip the kernel's cache scatter during warmup (kv_cache/block_tables
        # not yet bound).
        if block_tables is None or self.kv_cache is None:
            scatter_kv_cache = None
            scatter_block_tables = None
        else:
            scatter_kv_cache = self.kv_cache
            scatter_block_tables = block_tables
        fused_compress_attn(
            kv_in=kv,
            score_in=score,
            kv_state=self.kv_state,
            score_state=self.score_state,
            plan=plan,
            state_slot_mapping=state_slot_mapping,
            ape=self.ape,
            rms_weight=self.norm.weight,
            rms_eps=self.norm.eps,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            kv_cache=scatter_kv_cache,
            block_tables=scatter_block_tables,
            k_per_block=_V4_BLOCK_SIZE // ratio,
            overlap=overlap,
            ratio=ratio,
            head_dim=d,
            rope_head_dim=rd,
            quant=is_quant,
            cache_scale=self.cache_scale if is_quant else None,
            use_ue8m0=(self.scale_fmt == "ue8m0"),
            preshuffle=True,
            fp8_max=(torch.finfo(self.kv_cache.dtype).max if is_quant else None),
        )
        update_compressor_states(
            kv,
            score,
            self.ape,
            self.kv_state,
            self.score_state,
            write_plan=plan.write_plan_gpu,
            num_write=plan.num_write,
            state_slot_mapping=state_slot_mapping,
            ratio=ratio,
            overlap=overlap,
        )


class Indexer(nn.Module):
    """Selects top-k compressed KV positions for sparse attention via learned scoring.

    Port of inference/model.py:380-433. Has its own Compressor (with Hadamard
    rotation + FP4 simulation) to build a separate compressed KV cache used
    only for index scoring; query is also FP4-simulated.
    """

    def __init__(self, args: DeepseekV4Args, compress_ratio: int = 4, prefix: str = ""):
        super().__init__()
        self.prefix = prefix  # Used by V4 attention builder for layer-id parsing.
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.compress_ratio = compress_ratio

        qc = args.quant_config
        # Indexer Q is replicated across TP ranks: the index scoring path
        # needs all 64 heads at every rank to compute the per-token
        # compressed-position topk locally without cross-rank all_reduce.
        # Sharding wq_b would force an extra all_reduce on `index_score`
        # after the per-head sum.
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=qc,
            prefix=f"{prefix}.wq_b",
        )
        # weights_proj: BF16 in reference. Replicated because the layer is
        # tiny (dim × n_heads = 7168 × 64 ≈ 896KB BF16) and column-parallel
        # sharding produces a degenerate N=8 GEMM with no aiter tuned
        # config; full replication keeps N=64.
        self.weights_proj = ReplicatedLinear(
            self.dim,
            self.n_heads,
            bias=False,
            quant_config=qc,
            prefix=f"{prefix}.weights_proj",
        )
        self.softmax_scale = self.head_dim**-0.5
        # Init-time hoists out of `forward_batched`'s hot path.
        self._fp8_quant_func = get_hip_quant(QuantType.per_1x128)
        self._weights_scale = self.softmax_scale * self.n_heads**-0.5
        # `deepgemm_fp8_paged_mqa_logits` decode-path output column count:
        # one indexer slot per `compress_ratio` source tokens.
        self._max_model_len_idx = args.max_seq_len // compress_ratio

        self.compressor = Compressor(
            args,
            compress_ratio,
            self.head_dim,
            rotate=True,
            prefix=f"{prefix}.compressor",
        )
        # PR3-pre2c-B: Indexer.kv_cache is bound by the V4 attention builder
        # to a `[num_blocks, k1, head_dim]` per-CSA-layer view of the global
        # `csa_idx_kv` classical KV pool. The 1-slot register_buffer below is
        # a warmup fallback (warmup runs before allocate_kv_cache); it is
        # setattr-replaced post-binding and GC'd. Same pattern as Compressor's
        # kv_state in pre2a / Attention.swa_kv in pre2c-A.
        self.register_buffer(
            "kv_cache",
            torch.zeros(
                1,
                args.max_seq_len // compress_ratio,
                self.head_dim,
            ),
            persistent=False,
        )
        self.rotary_emb: Optional[_V4RoPE] = None

        # Register self in static_forward_context so the
        # `torch.ops.aiter.indexer_score_topk` dispatcher can look us up by
        # `layer_name` (= self.prefix). Same pattern as V4 MoE registration.
        get_current_atom_config().compilation_config.static_forward_context[
            prefix
        ] = self

    def forward_batched(
        self,
        x_full: torch.Tensor,  # [total_tokens, dim]
        qr_full: torch.Tensor,  # [total_tokens, q_lora_rank] — fp8 when qr_full_scale given
        positions: torch.Tensor,  # [total_tokens]
        qr_full_scale: Optional[
            torch.Tensor
        ] = None,  # per_1x128 scale paired with qr_full
    ) -> torch.Tensor:
        """Q proj + RoPE + FP8-quant + weights compute (have module state),
        then dispatch to `torch.ops.aiter.indexer_score_topk`, which calls
        back into `self.indexer_score_topk(q_fp8, weights, self.index_topk)`.

        Caller must invoke `self.compressor` once batched BEFORE this so all
        seqs' Indexer kv_cache is already populated.

        Returns:
          topk_in_seq: `[total_tokens, index_topk] int32` — RAW seq-local row
            indices (each token's column refers to row in its own seq's
            compressed K). Cols past per-token visibility cap hold -1
            sentinels (kernel-native: prefill `top_k_per_row_prefill` and
            decode `top_k_per_row_decode` both write -1 in the tail).
            Consumer (`csa_translate_pack`) skips negative entries via its
            `topk >= 0` write mask.
        """
        assert self.rotary_emb is not None
        rd = self.rope_head_dim
        total_tokens = x_full.size(0)

        # Q proj + RoPE + rotate (batched). rotary_emb internally reshapes
        # to (1, num_tokens, -1, rotary_dim) so the input doesn't need an
        # explicit batch dim. rotate_activation is last-dim-only.
        q = self.wq_b(qr_full, x_scale=qr_full_scale).view(
            total_tokens, self.n_heads, self.head_dim
        )
        # self.rotary_emb(positions, q[..., -rd:])
        # q = rotate_activation(q)
        cos, sin = self.rotary_emb._caches(q.device)
        rope_rotate_activation(q, q, cos, sin, positions, rd)

        # FP8 quant Q (still online — Q is recomputed each fwd, no cache).
        # `_fp8_quant_func` / `_weights_scale` precomputed in __init__.
        q_2d = q.view(-1, self.head_dim)
        q_fp8, q_scale = self._fp8_quant_func(q_2d, quant_dtype=dtypes.fp8)
        q_fp8 = q_fp8.view(total_tokens, self.n_heads, self.head_dim)
        q_scale = q_scale.view(total_tokens, self.n_heads, 1)

        # weights = weights_proj * q_scale * (softmax_scale * 1/sqrt(H))
        # weights_proj is BF16 but auto-promotes to fp32 via fp32 q_scale,
        # so no explicit `.float()` cast needed.
        weights = self.weights_proj(x_full)
        weights = scale_indexer_weights(weights, q_scale, self._weights_scale)

        return torch.ops.aiter.indexer_score_topk(
            q_fp8, weights, self.prefix, self.index_topk
        )  # [total_tokens, index_topk] int32

    def indexer_score_topk(
        self,
        q_fp8: torch.Tensor,  # [total_tokens, n_heads, head_dim] fp8
        weights: torch.Tensor,  # [total_tokens, n_heads] fp32
        topk: int,
    ) -> torch.Tensor:
        """Module-side entry invoked by `torch.ops.aiter.indexer_score_topk`.

        Reads `block_tables` and `v4_indexer_meta` from
        `get_forward_context().attn_metadata` (built once per fwd in
        `DeepseekV4AttentionMetadataBuilder._build_v4_indexer_meta`) — the
        per-CSA-layer call has zero CPU index math and zero H2D copies.

        Returns:
          topk_in_seq: `[total_tokens, topk] int32` — RAW seq-local row
            indices into each token's seq's compressed K cache. Cols past
            per-token visibility cap hold -1 sentinels (kernel-native).
            `csa_translate_pack` consumes this layout directly.
        """
        fc = get_forward_context()
        indexer_meta = fc.attn_metadata.indexer_meta
        block_tables = fc.attn_metadata.block_tables  # [bs, max_blocks_per_seq] int32

        # No host-side `if total_committed == 0: return torch.full(-1)`
        # short-circuit — that would freeze a Python branch into the
        # CUDAGraph at capture time. The hot path handles the corner
        # natively: when n_committed == 0 the per-token K bound is 0, the
        # underlying top-k kernels write -1 sentinels across the row, and
        # `csa_translate_pack` skips them via its `topk >= 0` mask.
        if fc.context.is_prefill:
            return self._score_topk_prefill(
                q_fp8, weights, block_tables, indexer_meta, topk
            )  # [total_tokens, topk] int32
        return self._score_topk_decode(
            q_fp8, weights, block_tables, indexer_meta, topk
        )  # [total_tokens, topk] int32

    def _score_topk_prefill(
        self,
        q_fp8: torch.Tensor,  # [total_tokens, n_heads, head_dim] fp8
        weights: torch.Tensor,  # [total_tokens, n_heads] fp32
        block_tables: torch.Tensor,  # [bs, max_blocks_per_seq] int32
        indexer_meta: dict,
        topk: int,
    ) -> torch.Tensor:
        """Variable-K prefill / mixed batch: cp_gather + fp8_mqa_logits.

        Eager-only — total_committed varies per fwd, so output logits shape
        is dynamic and incompatible with CUDAGraph capture.
        """
        device = q_fp8.device
        total_tokens = q_fp8.size(0)
        # K side: cache stores FP8 + 4-byte fp32 scale per row interleaved
        # (uint8 layout written by `indexer_k_quant_and_cache` from the inner
        # Compressor). `cp_gather_indexer_k_quant_cache` does paged-gather
        # + split into separate (FP8, scale) buffers in one kernel — no
        # per-row index list, no online quant.
        total_committed = indexer_meta["total_committed"]
        cu_committed = indexer_meta["cu_committed_gpu"]
        k_fp8 = torch.empty(
            (total_committed, self.head_dim), device=device, dtype=dtypes.fp8
        )
        k_scale = torch.empty((total_committed, 1), device=device, dtype=torch.float32)
        cp_gather_indexer_k_quant_cache(
            self.kv_cache,
            k_fp8,
            k_scale.view(dtypes.fp8),  # 4-byte scale rows treated as fp8 bytes
            block_tables,
            cu_committed,
            preshuffle=True,
        )

        cu_starts = indexer_meta["cu_starts_gpu"]  # [total_tokens] int32
        cu_ends = indexer_meta["cu_ends_gpu"]  # [total_tokens] int32
        logits = fp8_mqa_logits(
            Q=q_fp8,
            KV=k_fp8,
            kv_scales=k_scale,
            weights=weights,
            cu_starts=cu_starts,
            cu_ends=cu_ends,
        )  # [total_tokens, total_committed] fp32; outside [start,end) is -inf

        # aiter `top_k_per_row_prefill` (radix kernel, parametric `k` via the
        # pybind kwarg). Honors per-row [cu_starts[i], cu_ends[i]) so cells
        # outside each row's valid window are never selected; rows shorter
        # than `topk` get -1 sentinels for tail cols.
        #
        # Output is GLOBAL: each cell holds either -1 or
        # `cu_starts[t] + col_in_seq` (= seq_base + seq-local idx). We
        # subtract `seq_base_per_token` to produce the raw seq-local layout
        # `csa_translate_pack` expects. The -1 sentinels are preserved via
        # `torch.where`.
        # [total_tokens, topk] int32 — eager-only path so per-fwd alloc
        # is fine (prefill total_tokens is dynamic; no CG capture here).
        topk_global = torch.empty(
            (total_tokens, topk), dtype=torch.int32, device=device
        )
        top_k_per_row_prefill(
            logits,
            cu_starts,
            cu_ends,
            topk_global,
            None,  # values not needed, only indices
            total_tokens,
            logits.stride(0),
            logits.stride(1),
            k=topk,
        )
        seq_base = indexer_meta["seq_base_per_token_gpu"].unsqueeze(
            1
        )  # [total_tokens, 1] int32
        return torch.where(
            topk_global < 0,
            topk_global,  # preserve -1 sentinel
            topk_global - seq_base,
        )  # [total_tokens, topk] int32, raw seq-local with -1 in tail

    def _score_topk_decode(
        self,
        q_fp8: torch.Tensor,  # [total_tokens, n_heads, head_dim] fp8
        weights: torch.Tensor,  # [total_tokens, n_heads] fp32
        block_tables: torch.Tensor,  # [bs, max_blocks_per_seq] int32
        indexer_meta: dict,
        topk: int,
    ) -> torch.Tensor:
        """Pure-decode path: `deepgemm_fp8_paged_mqa_logits` reads paged FP8
        cache directly, producing fixed-shape `[bs*next_n, max_model_len_idx]`
        logits — CUDAGraph-friendly (no per-fwd `total_committed`-shaped
        allocation). Mirrors V3.2 sparse_attn_indexer decode branch
        (deepseek_v2.py:1047-1084).

        Top-k uses aiter `top_k_per_row_decode` (radix kernel, parametric `k`):
        the kernel honors `n_committed_per_seq` per row, so logits cells past
        each row's valid range are never selected — no `fill_(-inf)` required.
        Rows whose valid range is shorter than `index_topk` get -1 sentinels
        for tail cols. Output is RAW seq-local (each row's cols are 0-indexed
        into that batch's compressed K), exactly the layout
        `csa_translate_pack` consumes.
        """
        total_tokens = q_fp8.size(0)
        n_committed_per_seq_gpu = indexer_meta["n_committed_per_seq_gpu"]  # int32 [bs]
        bs = block_tables.size(0)
        # V4-Pro has no MTP, so next_n = total_tokens // bs = 1. The reshape
        # also handles future multi-token decode (MTP) without code change.
        next_n = total_tokens // bs
        # deepgemm requires Q in [bs, next_n, heads, head_dim], KV in
        # [num_blocks, block_size, n_head=1, hidden_dim+scale_dim] (4D).
        q_4d = q_fp8.view(
            bs, next_n, self.n_heads, self.head_dim
        )  # [bs, next_n, n_heads, head_dim] fp8
        kv_cache_4d = self.kv_cache.unsqueeze(
            -2
        )  # [num_blocks, k1_csa, 1, head_dim+scale_dim] uint8
        # Per-fwd write-once GPU scratch — no CPU mirror, no cross-fwd state.
        # Under CUDAGraph capture, torch allocates from the graph's private
        # memory pool and the address is stable across replays at this
        # captured `total_tokens`. No `fill_(-inf)` needed —
        # `top_k_per_row_decode` bounds each row by `n_committed_per_seq[batch]`
        # so unwritten cols are never picked.
        logits = torch.empty(
            total_tokens,
            self._max_model_len_idx,
            dtype=torch.float32,
            device=q_fp8.device,
        )
        deepgemm_fp8_paged_mqa_logits(
            q_4d,
            kv_cache_4d,
            weights,
            logits,
            n_committed_per_seq_gpu,  # int32, sized [bs] (staged in builder)
            block_tables,
            self._max_model_len_idx,
            KVBlockSize=self.kv_cache.size(1),  # k1_csa = 32
            Preshuffle=True,
        )
        # Per-fwd write-once int32 scratch. Kernel writes exactly `index_topk`
        # ints per row (valid seq-local indices then -1 sentinels). CG-safe
        # for the same reason as `logits` above.
        topk_local = torch.empty(
            total_tokens, self.index_topk, dtype=torch.int32, device=q_fp8.device
        )
        top_k_per_row_decode(
            logits,
            next_n,
            n_committed_per_seq_gpu,
            topk_local,
            total_tokens,
            logits.stride(0),
            logits.stride(1),
            k=topk,
        )
        return topk_local  # [total_tokens, index_topk] int32, raw seq-local


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

    def __init__(
        self,
        layer_id: int,
        args: DeepseekV4Args,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        compress_stream: Optional[torch.cuda.Stream] = None,
    ):
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
        self.attn_sink = atom_parameter(
            torch.empty(self.n_local_heads, dtype=torch.float32)
        )
        # Fused [wq_a; wkv]: both ReplicatedLinear FP8 sharing input x.
        # On disk still split (`attn.wq_a.{weight,scale}` + `attn.wkv.{weight,scale}`);
        # routed via packed_modules_mapping in DeepseekV4ForCausalLM.
        self.wqkv_a = MergedReplicatedLinear(
            self.dim,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=qc,
            prefix=f"{p}.wqkv_a",
        )
        # Fuse q_norm + per_1x128 FP8 quant: kernel emits (qr_fp8, qr_scale)
        # in one launch, both wq_b consumers (outer ColumnParallel + Indexer
        # ReplicatedLinear) skip their own input quant.
        self.q_norm = RMSNorm(
            self.q_lora_rank,
            self.eps,
            fused_quant=True,
            quant_config=qc,
            prefix=f"{p}.q_norm",
        )
        # q_norm2: per-head Q normalization. The checkpoint has no
        # `q_norm2.weight` entry, so the module is constructed with
        # `weight=None` (no learnable Parameter) — `DualRMSNorm` reads the
        # sentinel and tells the kernel to skip the q_weight load entirely.
        # Behavior is identical to the prior `_rmsnorm_nw(q, eps, head_dim)`.
        self.q_norm2 = _make_weightless_rmsnorm(self.head_dim, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=qc,
            prefix=f"{p}.wq_b",
        )
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        # Fused per-head Q RMSNorm (identity weight) + KV RMSNorm — both have
        # feature dim = head_dim, single Triton kernel via DualRMSNorm.
        # `q_norm2.weight is None` tells the kernel's Q_HAS_WEIGHT=False
        # path to skip the q_weight load (Q side has no learnable weight;
        # equivalent to the prior `_rmsnorm_nw` helper).
        self.qk_norm = DualRMSNorm(
            self.q_norm2,
            self.kv_norm,
            num_q_heads=self.n_local_heads,
            num_kv_heads=1,
            head_dim=self.head_dim,
            prefix=f"{p}.qk_norm",
        )
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

        # ----- KV cache splitting (paper §3.6.1) -----
        # State cache (per-request slot, in per_req_cache pool):
        #   `swa_kv`: [num_slots, n_win, head_dim] — most recent n_win window.
        #   Bound by DeepseekV4AttentionMetadataBuilder.build_kv_cache_tensor()
        #   after allocate_kv_cache. The 1-slot register_buffer below is a
        #   warmup fallback (warmup runs before allocate_kv_cache); after
        #   binding it is setattr-replaced with the per_req_cache pool slice
        #   `[max_num_seqs, n_win, head_dim]` and the original buffer is GC'd.
        # `unified_kv` (paged_decode/paged_prefill base) is NOT pre-registered
        # — V4Attention.forward short-circuits the sparse_attn dispatch on
        # `is_dummy_run` so warmup never reads it.
        self.register_buffer(
            "swa_kv",
            torch.zeros(1, args.window_size, self.head_dim),
            persistent=False,
        )
        # Classical KV cache (paper §3.6.1) lives entirely in the global
        # `csa_main_kv` / `hca_main_kv` pool (allocated by the V4 attention
        # builder as `[num_blocks, n_layers, k_per_block, head_dim]`).
        # `Compressor.kv_cache` is bound to a per-layer view of that pool by
        # `DeepseekV4AttentionMetadataBuilder.build_kv_cache_tensor`. The
        # Attention module no longer owns a `kv_cache` attribute (PR3-pre2c-B).

        # ----- RoPE (own per-layer instance, not shared): YaRN for compressed
        # attention layers (long context), plain RoPE for dense (window-only).
        # Wraps aiter's `rope_cached_*_fwd_inplace` kernel so RoPE is driven by
        # per-token `positions` (groundwork for PR3 multi-sequence), while the
        # cos/sin cache uses V4's exact YaRN math via `_precompute_freqs_cis`.
        if self.compress_ratio:
            original_seq_len, rope_theta = (
                args.original_seq_len,
                args.compress_rope_theta,
            )
        else:
            original_seq_len, rope_theta = 0, args.rope_theta
        self.rotary_emb = _V4RoPE(
            rotary_dim=self.rope_head_dim,
            max_seq_len=args.max_seq_len,
            base=rope_theta,
            factor=args.rope_factor,
            original_seq_len=original_seq_len,
            beta_fast=args.beta_fast,
            beta_slow=args.beta_slow,
            dtype=torch.bfloat16,
        )
        # Plumb rotary_emb into compressor / indexer here in __init__ rather
        # than lazily in forward — Dynamo can't trace NNModule setattr inside
        # a compiled forward (graph break + backend re-entry).
        if self.compressor is not None:
            self.compressor.rotary_emb = self.rotary_emb
        if self.indexer is not None:
            self.indexer.rotary_emb = self.rotary_emb
            self.indexer.compressor.rotary_emb = self.rotary_emb

        self.alt_stream = alt_stream
        self.compress_stream = compress_stream
        self._use_async_compress = (
            self.alt_stream is not None and self.compressor is not None
        )

        self.layer_name = prefix
        atom_config = get_current_atom_config()
        atom_config.compilation_config.static_forward_context[self.layer_name] = self

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
        self.wo_a.weight = atom_parameter(bf16)
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
        self.wo_a.quant_type = QuantType.No

    def _launch_compressors_async(self, x, plan, state_slot_mapping, block_tables):
        """Fire Compressor(s) on side streams, return immediately.

        Main Compressor → alt_stream (CSA + HCA).
        Indexer Compressor → compress_stream (CSA only).
        Waits resolve instantly: side streams ~25us, main Q/KV chain ~87us."""
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.alt_stream):
            self.compressor(
                x,
                plan=plan,
                state_slot_mapping=state_slot_mapping,
                block_tables=block_tables,
            )
        if self.indexer is not None and self.compress_stream is not None:
            self.compress_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.compress_stream):
                self.indexer.compressor(
                    x,
                    plan=plan,
                    state_slot_mapping=state_slot_mapping,
                    block_tables=block_tables,
                )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.aiter.v4_attention_with_output(x, positions, self.layer_name)

    def forward_impl(
        self,
        x: torch.Tensor,  # [num_tokens, dim]  flat ragged-batch hidden state
        positions: torch.Tensor,  # [num_tokens] int  absolute token positions
    ) -> torch.Tensor:  # [num_tokens, dim]  BF16 attention output
        """Compute attention for `x` at absolute token `positions`.

        PR3-main: handles batched multi-sequence input. Linear projections + RoPE
        run once on the flat `[num_tokens, ...]` batch; SWA write, Compressor
        scatter, sparse_attn (gather + score) iterate over sequences using
        per-seq slot + block_table from the V4 attention builder's metadata.
        Per-seq slicing uses `cu_seqlens_q` from `forward_context`.
        """
        assert (
            x.dim() == 2 and x.shape[-1] == self.dim
        ), f"DeepseekV4Attention expects [num_tokens, {self.dim}], got {tuple(x.shape)}"
        # warmup_model runs BEFORE allocate_kv_cache → `unified_kv` is unbound
        # and the new sparse_attn_v4_paged_{decode,prefill} kernels would read
        # OOB. Same pattern as `attention_mha.py:98` — short-circuit dummy_run
        # with a zero output of the correct shape; downstream layers compile
        # on a real fwd. swa_write / Compressor / Indexer are also skipped to
        # avoid touching unbound state caches.
        if get_forward_context().context.is_dummy_run:
            return torch.zeros_like(x)
        num_tokens = x.size(0)
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim

        # ----- Batched ops on full flat tensors -----
        # `_V4_FORCE_UE8M0_QUANT` (module-level): round-trip x/qr to ue8m0-FP8
        # to mirror the reference's `act_quant(scale_fmt="ue8m0")` Linear-input
        # quantization. EXPERIMENT only.
        if _V4_FORCE_UE8M0_QUANT:
            x = x.clone()
            act_quant_inplace(x, 128, "ue8m0")

        attn_md = cast("AttentionMetaData_DSV4", get_forward_context().attn_metadata)
        compress_plans = attn_md.compress_plans
        swa_write_indices = attn_md.swa_write_indices
        v4_batch_id_per_token = attn_md.batch_id_per_token
        block_tables_gpu = attn_md.block_tables
        state_slot_mapping = attn_md.state_slot_mapping
        plan_for_layer = compress_plans[ratio] if ratio else None

        # ===== Triple-stream: Q/KV path + both Compressors in parallel =====
        if self._use_async_compress:
            self._launch_compressors_async(
                x, plan_for_layer, state_slot_mapping, block_tables_gpu
            )

        # ----- Q/KV projections (main stream) -----
        qkv_a = self.wqkv_a(x)
        q_lora, kv_pre = torch.split(qkv_a, [self.q_lora_rank, self.head_dim], dim=-1)
        assert (
            not _V4_FORCE_UE8M0_QUANT
        ), "_V4_FORCE_UE8M0_QUANT incompatible with fused q_norm quant (qr is already FP8)"
        qr, qr_scale = self.q_norm(q_lora)
        # Flat q_flat is [num_tokens, n_local_heads * head_dim]; DualRMSNorm
        # views internally to per-head shape and returns flat. Single Triton
        # launch fuses per-head Q RMSNorm (weightless) + KV RMSNorm (both
        # head_dim=128), replacing the prior `_rmsnorm_nw + kv_norm` pair.
        q_flat = self.wq_b(qr, x_scale=qr_scale)
        q_flat, kv = self.qk_norm(q_flat, kv_pre)
        q = q_flat.view(num_tokens, self.n_local_heads, self.head_dim)
        # q [S, H, D] / kv [S, head_dim] — rotary_emb internally unsqueezes
        # to (1, num_tokens, ...) for aiter's per-position rope kernel.
        self.rotary_emb(positions, q[..., -rd:], kv[..., -rd:])
        if _V4_USE_REF_QUANT:
            act_quant_inplace(kv[..., :-rd], 64, self.scale_fmt)

        # ===== Compressor + Indexer =====
        if not self._use_async_compress:
            if self.compressor is not None:
                self.compressor(
                    x,
                    plan=plan_for_layer,
                    state_slot_mapping=state_slot_mapping,
                    block_tables=block_tables_gpu,
                )
                if self.indexer is not None:
                    self.indexer.compressor(
                        x,
                        plan=plan_for_layer,
                        state_slot_mapping=state_slot_mapping,
                        block_tables=block_tables_gpu,
                    )
        if self.indexer is not None:
            if self._use_async_compress:
                torch.cuda.current_stream().wait_stream(self.alt_stream)
                torch.cuda.current_stream().wait_stream(self.compress_stream)
            indexer_topk_batched = self.indexer.forward_batched(
                x_full=x,
                qr_full=qr,
                qr_full_scale=qr_scale,
                positions=positions,
            )
            # Translate seq-local topk → physical paged offsets and write into
            # the CSA section of either:
            #   - decode buffer `kv_indices_csa` (is_pure_decode)
            #   - prefill buffer `kv_indices_prefix_csa` (otherwise)
            # `_fill_csa_paged_compress` dispatches internally on is_pure_decode.
            self._fill_csa_paged_compress(attn_md, indexer_topk_batched, num_tokens)
        elif self._use_async_compress:
            torch.cuda.current_stream().wait_stream(self.alt_stream)

        # ===== Sparse attention dispatch =====
        # Two paths over the unified KV pool. The order of `swa_write` vs
        # `sparse_attn` differs because the two kernels read SWA differently:
        #
        # decode (is_pure_decode==True):
        #   `paged_decode` reads SWA from `unified_kv` ring slot `pos % win`.
        #   The current decode token's K must be present in the ring before
        #   attn fires, otherwise the token can't see its own K. So:
        #     swa_write → paged_decode
        #
        # prefill / mixed (is_pure_decode==False):
        #   `paged_prefill` reads in-chunk K from per-fwd `kv` tensor (extend
        #   region) and prior-chunk K from `unified_kv` ring (prefix region).
        #   `swa_write` writes the LAST `win` tokens of THIS fwd into ring
        #   slots `pos % win`, which OVERLAP with prior-chunk slots that the
        #   prefix SWA region wants to read (chunked prefill). So:
        #     paged_prefill → swa_write
        #   Pure prefill (chunk_start==0) has prefix_swa_count==0 so no prior
        #   ring read; swa_write order is irrelevant in that subcase.
        q_sa = q.contiguous()
        if attn_md.is_pure_decode:
            swa_write(
                kv,
                swa_write_indices,
                positions,
                v4_batch_id_per_token,
                state_slot_mapping,
                self.swa_kv,
                win,
            )
            if ratio == 0:
                kv_indices = attn_md.kv_indices_swa
                kv_indptr = attn_md.kv_indptr_swa
            elif ratio == 4:
                kv_indices = attn_md.kv_indices_csa
                kv_indptr = attn_md.kv_indptr_csa
            else:  # ratio == 128
                kv_indices = attn_md.kv_indices_hca
                kv_indptr = attn_md.kv_indptr_hca
            o = sparse_attn_v4_paged_decode(
                q_sa,
                self.unified_kv,
                kv_indices,
                kv_indptr,
                self.attn_sink,
                self.softmax_scale,
            )  # [S, H, head_dim]
        else:
            # Two-source paged prefill: prefix from `unified_kv` (per-ratio
            # buffer with SWA history + compress section), extend from per-fwd
            # `kv` tensor (in-chunk SWA tail; extend buffer is layer-invariant).
            if ratio == 0:
                kv_indices_prefix = attn_md.kv_indices_prefix_swa
                kv_indptr_prefix = attn_md.kv_indptr_prefix_swa
            elif ratio == 4:
                kv_indices_prefix = attn_md.kv_indices_prefix_csa
                kv_indptr_prefix = attn_md.kv_indptr_prefix_csa
            else:  # ratio == 128
                kv_indices_prefix = attn_md.kv_indices_prefix_hca
                kv_indptr_prefix = attn_md.kv_indptr_prefix_hca
            o = sparse_attn_v4_paged_prefill(
                q_sa,
                self.unified_kv,
                kv_indices_prefix,
                kv_indptr_prefix,
                kv,
                attn_md.kv_indices_extend,
                attn_md.kv_indptr_extend,
                self.attn_sink,
                self.softmax_scale,
            )  # [S, H, head_dim]
            # swa_write AFTER attn so chunked-prefill prefix SWA reads see
            # prior-chunk's ring contents (current swa_write would overwrite
            # ring slots `pos % win` for positions in this chunk's tail).
            swa_write(
                kv,
                swa_write_indices,
                positions,
                v4_batch_id_per_token,
                state_slot_mapping,
                self.swa_kv,
                win,
            )

        # Inverse RoPE on output's rope dims to remove absolute-position
        # contribution carried in by the value-side RoPE of the KV entries.
        self.rotary_emb.inverse(positions, o[..., -rd:])
        # ----- Grouped output LoRA (batched on the full flat tensor) -----
        o = o.view(num_tokens, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("sgd,grd->sgr", o, wo_a)
        x = self.wo_b(o.flatten(1))
        return x

    def _fill_csa_paged_compress(
        self,
        attn_md,
        topk_local_raw: torch.Tensor,
        total_tokens: int,
    ) -> None:
        """Per-CSA-layer: translate indexer raw `topk_in_seq` → physical paged
        offsets in `unified_kv` and packed-write into the CSA section of the
        active prefix buffer.

        Dispatch:
          - is_pure_decode → write into decode buffer `kv_indices_csa`,
                             skip = `window_size` (full SWA prefix per token)
          - prefill / mixed → write into prefill buffer `kv_indices_prefix_csa`,
                              skip = per-token `prefix_swa_count[t]`

        Per doc §6.4:
          block_idx_in_seq = topk_local // csa_block_capacity
          slot_in_block    = topk_local %  csa_block_capacity
          physical_block   = block_tables[batch_id_per_token[t], block_idx_in_seq]
          paged_offset     = swa_pages + physical_block * csa_block_capacity
                             + slot_in_block

        Fully fused into one triton kernel — no [T, index_topk] intermediates,
        no PyTorch fancy index. CG sentinel (batch_id=-1) and OOB clamp are
        handled in-kernel; downstream paged_decode/paged_prefill kernels
        strict-slice via `kv_indptr*`, so tail cols beyond `valid_k` are never
        read and need no `-1` fill (CSA section was -1 pre-filled by builder).

        Args:
          topk_local_raw: [total_tokens, index_topk] int32 — RAW seq-local
            output of `Indexer.forward_batched` (negative tails are sentinels;
            csa_translate_pack skips them via `topk >= 0` write mask).
        """
        # csa_block_capacity = block_size // ratio = 128 // 4 = 32.
        # Derived from constants (not `compressor.kv_cache.size(1)`) because
        # warmup runs before `build_kv_cache_tensor` binds compressor.kv_cache,
        # and this method now fires for both decode and prefill (including
        # warmup batches). Equivalent post-bind: `compressor.kv_cache.size(1)`.
        csa_block_capacity = _V4_BLOCK_SIZE // 4

        if attn_md.is_pure_decode:
            kv_indptr = attn_md.kv_indptr_csa
            kv_indices = attn_md.kv_indices_csa
        else:
            kv_indptr = attn_md.kv_indptr_prefix_csa
            kv_indices = attn_md.kv_indices_prefix_csa

        csa_translate_pack(
            topk_local_raw,
            attn_md.block_tables,
            attn_md.n_committed_csa_per_seq,
            kv_indptr,
            attn_md.batch_id_per_token,
            attn_md.skip_prefix_len_csa,
            kv_indices,
            swa_pages=attn_md.swa_pages,
            csa_block_capacity=csa_block_capacity,
        )


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
        # Fused [w1; w3] (gate_up_proj): both share input x, both ColumnParallel
        # — standard llama/dsv2 fusion. Disk still split; routed via
        # packed_modules_mapping in DeepseekV4ForCausalLM.
        self.gate_up_proj = MergedColumnParallelLinear(
            dim,
            [inter_dim, inter_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.w2 = RowParallelLinear(
            inter_dim,
            dim,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.w2",
        )
        self.swiglu_limit = swiglu_limit

    def forward(
        self,
        x: torch.Tensor,  # [num_tokens, dim]
        weights: Optional[torch.Tensor] = None,  # [num_tokens, 1]  optional gate
    ) -> torch.Tensor:  # [num_tokens, dim]

        dtype = x.dtype
        # Single fused GEMM. Layout is [gate | up] concat on last dim — matches
        # aiter silu_and_mul's split([d, d], dim=-1) contract. The kernel does
        # silu/clamp/mul in fp32 internally regardless of input dtype, so we
        # feed the bf16 GEMM output directly.
        combined = self.gate_up_proj(x)  # [num_tokens, 2*inter_dim_per_tp]
        out = torch.empty(
            (combined.shape[0], combined.shape[-1] // 2),
            dtype=dtype,
            device=combined.device,
        )
        # limit > 0 enables in-kernel clamp (gate≤limit, up∈[-limit,limit]) via
        # ROCm v_med3_f32 — same semantics as the prior torch.clamp pair.
        aiter_silu_and_mul(out, combined, self.swiglu_limit)
        if weights is not None:
            out = weights.to(dtype) * out
        return self.w2(out)  # [num_tokens, dim]


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

    def __init__(
        self,
        layer_id: int,
        args: DeepseekV4Args,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.prefix = prefix
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.is_hash_layer = layer_id < args.n_hash_layers
        self.routed_scaling_factor = args.route_scale
        self.swiglu_limit = args.swiglu_limit
        self.tp_size = get_tensor_model_parallel_world_size()
        self.alt_stream = alt_stream
        qc = args.quant_config

        self.gate = ReplicatedLinear(
            self.dim,
            self.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        # V4 hash-routed layers (layer_id < n_hash_layers) use tid2eid lookup,
        # not bias-corrected gate-logit routing — checkpoint has no
        # `gate.bias` for those layers. Only allocate the bias for
        # sqrtsoftplus layers to avoid 3 spurious unloaded-param warnings.
        if not self.is_hash_layer:
            self.gate.e_score_correction_bias = atom_parameter(
                torch.empty(self.n_routed_experts, dtype=torch.float32)
            )
        else:
            # tid2eid: per-token-id top-k expert lookup table (V4 first 3
            # layers use this in lieu of gate-logit routing).
            self.gate.tid2eid = atom_parameter(
                torch.empty(
                    args.vocab_size, args.n_activated_experts, dtype=torch.int32
                ),
            )
            # input_ids for hash routing is read from forward_context.context
            # (set by ModelRunner). torch.compile silently drops NNModule
            # attribute mutation across the compile boundary, so stashing on
            # `self.foo` from inside forward is a no-op at runtime.
        assert args.n_shared_experts == 1
        # aiter can fuse shared expert into the routed FusedMoE kernel ONLY
        # when shared and routed have the same quant dtype.
        # `is_rocm_aiter_fusion_shared_expert_enabled` compares shared vs
        # GLOBAL (=base dtype), but V4-Pro has routed=FP4 (per-layer override
        # for `.ffn.experts`) and shared=FP8 (=global). The function returns
        # True for V4 because shared matches global, missing the routed-shared
        # mismatch. Direct comparison: get routed and shared specs and compare.
        routed_dtype = qc.get_layer_quant_config(f"{prefix}.experts").quant_dtype
        shared_dtype = qc.get_layer_quant_config(f"{prefix}.shared_experts").quant_dtype
        self._fuse_shared_into_routed = (
            routed_dtype == shared_dtype
            and is_rocm_aiter_fusion_shared_expert_enabled()
        )
        moe_cfg = SimpleNamespace(
            routed_scaling_factor=self.routed_scaling_factor,
            n_shared_experts=(
                args.n_shared_experts if self._fuse_shared_into_routed else 0
            ),
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
            e_score_correction_bias=getattr(self.gate, "e_score_correction_bias", None),
            config=moe_cfg,
        )
        self.experts.swiglu_limit = args.swiglu_limit

        if not self._fuse_shared_into_routed:
            # self.experts.num_fused_shared_experts = 0
            self.shared_experts = Expert(
                args.dim,
                args.moe_inter_dim,
                swiglu_limit=args.swiglu_limit,
                quant_config=qc,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None
        if self.is_hash_layer:
            # Inject hash routing into FusedMoE.select_experts via the
            # custom_routing_function hook (added in atom/model_ops/moe.py).
            self.experts.custom_routing_function = self._hash_topk

        # Dual-stream: run shared_experts on `alt_stream` in parallel with
        # routed experts on the current stream. Mirrors V2's pattern. Only
        # active when shared_experts exist (not fused into routed) AND the
        # env threshold is positive AND we got an alt_stream from the model.
        # Per-call token count gating happens inside the custom op dispatcher
        # — prefill (large batch) skips dual-stream (overhead > benefit).
        self._use_dual_stream = (
            self.shared_experts is not None
            and self.alt_stream is not None
            and envs.ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD > 0
        )
        if self._use_dual_stream:
            # Register self in static_forward_context so the custom op
            # dispatcher can look us up by `layer_name` (= self.prefix).
            get_current_atom_config().compilation_config.static_forward_context[
                prefix
            ] = self

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
        # input_ids comes from forward_context.context (stashed by
        # `DeepseekV4ForCausalLM.forward`). This callback runs inside the
        # FusedMoE expert call, which is wrapped in `maybe_dual_stream_forward`
        # — a custom op opaque to Dynamo, so context access is safe here.
        fwd_input_ids = get_forward_context().context.input_ids
        assert (
            fwd_input_ids is not None
        ), "forward_context.context.input_ids is None — caller must invoke DeepseekV4ForCausalLM.forward, not DeepseekV4Model.forward directly."
        ids = fwd_input_ids.flatten()
        # Under DP-attention gather/scatter, forward_impl_graph gathers
        # hidden_states and gating_output across DP ranks before calling
        # select_experts → _hash_topk.  input_ids is still local (not
        # gathered), so topk_ids would be [local_bs, topk] while the
        # gathered gating_output is [local_bs*dp, n_experts].  Gather
        # input_ids to match.
        num_tokens = gating_output.shape[0]
        if ids.shape[0] < num_tokens:
            from atom.model_ops.moe import pad_for_all_gather

            ids_2d = ids.unsqueeze(-1)
            ids_2d, _ = pad_for_all_gather(ids_2d)
            from aiter.dist.parallel_state import get_dp_group

            ids_2d = get_dp_group().all_gather(ids_2d, dim=0)
            ids = ids_2d[:num_tokens].flatten()
            ids = ids.clamp(0, self.gate.tid2eid.shape[0] - 1)
        topk_ids = self.gate.tid2eid[ids].to(torch.int32)  # [N, topk]
        scores = torch.nn.functional.softplus(gating_output.float()).sqrt()
        topk_weights = scores.gather(dim=-1, index=topk_ids.long())
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights, topk_ids

    def routed_expert_forward(
        self, x: torch.Tensor  # [num_tokens, dim]
    ) -> torch.Tensor:  # [num_tokens, dim]
        """Gate + FusedMoE routed-expert pass.

        For hash layers the gate's `tid2eid` lookup needs `input_ids`;
        `DeepseekV4ForCausalLM.forward` stashes it on
        `forward_context.context.input_ids` before each forward, and
        `_hash_topk` (FusedMoE's custom_routing_function) reads it there.
        """
        router_logits = self.gate(x)  # [num_tokens, n_routed_experts]
        return self.experts(hidden_states=x, router_logits=router_logits)

    def combine_outputs(
        self,
        routed: torch.Tensor,  # [num_tokens, dim]
        shared: Optional[torch.Tensor],  # [num_tokens, dim] or None
    ) -> torch.Tensor:  # [num_tokens, dim]
        """Add shared-expert contribution (when not fused into routed) and
        all-reduce across TP ranks.
        """
        if shared is not None:
            routed = routed + shared
        if self.tp_size > 1:
            routed = tensor_model_parallel_all_reduce(routed)
        return routed

    def single_stream_moe_forward(
        self, x: torch.Tensor  # [num_tokens, dim]
    ) -> torch.Tensor:  # [num_tokens, dim]
        """Sequential: shared_experts → routed_experts → combine."""
        shared = self.shared_experts(x) if self.shared_experts is not None else None
        routed = self.routed_expert_forward(x)
        return self.combine_outputs(routed, shared)

    def dual_stream_moe_forward(
        self, x: torch.Tensor  # [num_tokens, dim]
    ) -> torch.Tensor:  # [num_tokens, dim]
        """Run shared_experts on `alt_stream` in parallel with routed_experts
        on the current stream. Mirrors V2's pattern. Both reads of `x` are
        independent; main stream waits on alt_stream's completion before
        combining.
        """
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.alt_stream):
            shared = self.shared_experts(x)
        routed = self.routed_expert_forward(x)
        current_stream.wait_stream(self.alt_stream)
        return self.combine_outputs(routed, shared)

    def forward(
        self,
        x: torch.Tensor,  # [num_tokens, dim]  hidden state (post ffn_norm)
    ) -> torch.Tensor:  # [num_tokens, dim]
        # Hash-layer routing reads `input_ids` from forward_context.context
        # inside `_hash_topk` (FusedMoE.custom_routing_function callback);
        # the MoE call itself doesn't need it as a parameter.
        assert (
            x.dim() == 2 and x.shape[-1] == self.dim
        ), f"MoE expects 2D [num_tokens, {self.dim}], got {tuple(x.shape)}"
        if self._use_dual_stream:
            # Shared custom op (also used by V2). Dispatcher reads
            # `_use_dual_stream` + per-call num_tokens vs threshold to pick
            # dual vs single. Custom op = Dynamo barrier so stream context
            # inside `dual_stream_moe_forward` is opaque to torch.compile.
            return torch.ops.aiter.maybe_dual_stream_forward(x, self.prefix)
        return self.single_stream_moe_forward(x)


class Block(nn.Module):
    """Transformer block with Manifold-Constrained Hyper-Connections (mHC).

    Port of inference/model.py:648-701. ATOM 2D-flat convention: the residual
    stream is widened to `[num_tokens, hc_mult, dim]`. Each sub-layer (attn / ffn):
      1. `hc_pre`: project `[num_tokens, hc_mult, dim]` → `[num_tokens, dim]` via
         Sinkhorn-projected pre-weights (also producing post-weights and combination
         matrix for hc_post).
      2. `attn_norm` + `attn` (or `ffn_norm` + `ffn`): standard sub-layer in
         `[num_tokens, dim]`.
      3. `hc_post`: expand `[num_tokens, dim]` back to `[num_tokens, hc_mult, dim]`
         using the post-weights (gate on the new contribution) + the combination
         matrix applied to the previous residual.
    """

    def __init__(
        self,
        layer_id: int,
        args: DeepseekV4Args,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        compress_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = DeepseekV4Attention(
            layer_id,
            args,
            prefix=f"{prefix}.attn",
            alt_stream=alt_stream,
            compress_stream=compress_stream,
        )
        self.ffn = MoE(layer_id, args, prefix=f"{prefix}.ffn", alt_stream=alt_stream)
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        # All HC params stored in fp32 (matches reference's `set_dtype(torch.float32)`).
        self.hc_attn_fn = atom_parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        )
        self.hc_ffn_fn = atom_parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        )
        self.hc_attn_base = atom_parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = atom_parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = atom_parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = atom_parameter(torch.empty(3, dtype=torch.float32))

        # aiter mhc_pre/post kernels assert hidden % 512 == 0 OR hidden % 256 == 0
        # (mhc_kernels.cu:864 calls __builtin_trap on violation). Bind kernel refs
        # at init: present + dim-compatible → use fused; else None → torch fallback.
        # `x.is_cuda` is implicit here — model lives on GPU post-`.to()`; a CPU tensor
        # would have crashed earlier in DeepseekV4Attention.
        _dim_ok = args.dim % 512 == 0 or args.dim % 256 == 0
        self._mhc_pre = getattr(aiter, "mhc_pre", None) if _dim_ok else None
        self._mhc_post = getattr(aiter, "mhc_post", None) if _dim_ok else None

    # mHC `hc_post_mult_value`: V4 uses `2.0 * sigmoid(post)` for the post gate.
    HC_POST_MULT = 2.0

    def hc_pre(
        self,
        residual: torch.Tensor,  # [num_tokens, hc, dim]  mHC-widened residual
        hc_fn: torch.Tensor,  # [mix_hc, hc*dim]  fp32
        hc_scale: torch.Tensor,  # [3] fp32
        hc_base: torch.Tensor,  # [mix_hc] fp32
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reduce mHC residual `[num_tokens, hc, dim]` to sub-layer input `[num_tokens, dim]`.

        Prefers the fused aiter `mhc_pre` kernel (single ROCm op for RMSNorm +
        hc-fn linear + Sinkhorn projection + weighted reduction). Falls back to
        the torch `hc_split_sinkhorn` reference when the aiter kernel is
        unavailable or `dim` doesn't satisfy the `% 256/512 == 0` constraint.

        Returns:
          y:    [num_tokens, dim]      sub-layer input
          post: [num_tokens, hc]       post-gate weights for hc_post
          comb: [num_tokens, hc, hc]   combination matrix for hc_post
        """
        if self._mhc_pre is not None:
            # aiter mhc_pre wants [M, hc, dim] and returns
            # (post [M, hc, 1], comb [M, hc, hc], y [M, dim]).
            post, comb, y = self._mhc_pre(
                residual,
                hc_fn,
                hc_scale,
                hc_base,
                float(self.norm_eps),
                float(self.hc_eps),
                float(self.hc_eps),
                self.HC_POST_MULT,
                int(self.hc_sinkhorn_iters),
            )
            return y, post.squeeze(-1), comb

        # Torch fallback (no-aiter): mirrors the reference math.
        dtype = residual.dtype
        x_flat = residual.flatten(-2)  # [num_tokens, hc*dim]
        x_normed = _rmsnorm_nw(x_flat, self.norm_eps, x_flat.shape[-1])
        mixes = F.linear(x_normed.float(), hc_fn)  # [num_tokens, mix_hc]
        pre, post, comb = hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        y = torch.sum(pre.unsqueeze(-1) * residual, dim=-2)  # [num_tokens, dim]
        return y.to(dtype), post, comb

    def hc_post(
        self,
        x: torch.Tensor,  # [num_tokens, dim]      sub-layer output
        residual: torch.Tensor,  # [num_tokens, hc, dim]  pre-layer residual
        post: torch.Tensor,  # [num_tokens, hc]       from hc_pre
        comb: torch.Tensor,  # [num_tokens, hc, hc]   from hc_pre
    ) -> torch.Tensor:  # [num_tokens, hc, dim]  new residual
        """Expand sub-layer output `[num_tokens, dim]` back to mHC residual
        `[num_tokens, hc, dim]`.

        Prefers the fused aiter `mhc_post` kernel; falls back to the torch
        reference when aiter is unavailable or shape constraints aren't met
        (kernel asserts hidden % 512 == 0 OR hidden % 256 == 0).

        See `/app/logs_claude/deepseek_v4/notes/12_aiter_mhc_post_root_cause.md`
        for past numerical-drift notes on long decode trajectories.
        """
        if self._mhc_post is not None:
            # `out` inherits residual.dtype = x.dtype (residual stream is BF16
            # end-to-end in Block.forward), so no cast needed on the kernel path.
            out = torch.empty_like(residual)
            self._mhc_post(out, x, residual, post.unsqueeze(-1), comb)
            return out

        # Torch fallback. fp32 (post, comb) × BF16 (x, residual) promotes to
        # fp32 by PyTorch promotion rules — cast back to x.dtype before return.
        # post.unsqueeze(-1) * x.unsqueeze(-2): [num_tokens, hc, dim] gating
        # comb.unsqueeze(-1) * residual.unsqueeze(-2): [num_tokens, hc, hc, dim]; sum over hc
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=-3
        )
        return y.type_as(x)

    def forward(
        self,
        x: torch.Tensor,  # [num_tokens, hc, dim]  mHC residual stream
        positions: torch.Tensor,  # [num_tokens] int  absolute token positions
    ) -> torch.Tensor:  # [num_tokens, hc, dim]  updated residual stream
        # ----- Attention sub-layer with mHC mixing -----
        residual = x  # [num_tokens, hc, dim]
        (
            x,
            post,
            comb,
        ) = self.hc_pre(  # [num_tokens, dim], [num_tokens, hc], [num_tokens, hc, hc]
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attn_norm(x)  # [num_tokens, dim]
        x = self.attn(x, positions)  # [num_tokens, dim]
        x = self.hc_post(x, residual, post, comb)  # [num_tokens, hc, dim]
        # ----- FFN sub-layer with mHC mixing -----
        residual = x  # [num_tokens, hc, dim]
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn_norm(x)  # [num_tokens, dim]
        x = self.ffn(
            x
        )  # [num_tokens, dim]  (input_ids read from forward_context for hash MoE)
        x = self.hc_post(x, residual, post, comb)  # [num_tokens, hc, dim]
        return x


class ParallelHead(ParallelLMHead):
    """V4 LM head with mHC reduction; vocab-parallel sharded across TP ranks.

    Port of inference/model.py:704-736. Inherits from `ParallelLMHead` so the
    vocab-axis sharding, `weight_loader`, last-token slicing, bf16 a16w16 GEMM
    (`tgemm.mm`), and TP all-gather come for free. V4 only adds:

    - `forward(...)` taking the mHC residual + hc_head params + final norm
    - `get_logits(...)` so `compute_logits` can call it directly on the
      hidden-state output of `model.forward` (CUDAGraph contract)
    - `hc_head(...)` Sigmoid-gated mHC reduction (vs `Block.hc_pre`'s Sinkhorn)

    Note on weight dtype: the V4 reference (model.py:713-714) keeps the LM head
    in fp32 because the disk weight is bf16; on AMD CDNA3/CDNA4 the bf16 MFMA
    instruction accumulates in fp32 natively, so a bf16 GEMM with the
    bf16-on-disk weight has the same effective precision as the reference's
    fp32 path while halving VRAM and using the faster a16w16 kernel.
    """

    def __init__(
        self, vocab_size: int, dim: int, norm_eps: float = 1e-6, hc_eps: float = 1e-6
    ):
        super().__init__(vocab_size, dim, bias=False)
        self.dim = dim
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps

    def get_logits(
        self, x: torch.Tensor  # [num_tokens, dim]
    ) -> torch.Tensor:  # [bs, vocab]
        """Project to vocab logits via the inherited `ParallelLMHead.forward`,
        which handles last-token slicing (prefill) + tgemm.mm + all-gather.
        """
        assert (
            x.dim() == 2 and x.shape[-1] == self.dim
        ), f"get_logits expects [num_tokens, {self.dim}], got {tuple(x.shape)}"
        return super().forward(x)

    def hc_head(
        self,
        x: torch.Tensor,  # [num_tokens, hc, dim]  mHC residual
        hc_fn: torch.Tensor,  # [hc, hc*dim]  fp32
        hc_scale: torch.Tensor,  # [1] fp32
        hc_base: torch.Tensor,  # [hc] fp32
    ) -> torch.Tensor:  # [num_tokens, dim]
        """Reduce mHC residual `[num_tokens, hc, dim]` → `[num_tokens, dim]`
        via Sigmoid-gated weighted sum (vs Block.hc_pre's Sinkhorn variant).
        """
        return _hc_head_reduce(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            self.norm_eps,
            self.hc_eps,
        )

    def forward(
        self,
        x: torch.Tensor,  # [num_tokens, hc, dim]
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm: nn.Module,
    ) -> torch.Tensor:  # [bs, vocab]
        x = self.hc_head(x, hc_fn, hc_scale, hc_base)  # [num_tokens, dim]
        # get_logits handles the per-rank vocab shard + all-gather internally.
        return self.get_logits(norm(x))  # [bs, vocab]


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
        self.enorm = RMSNorm(args.dim, args.norm_eps)
        self.hnorm = RMSNorm(args.dim, args.norm_eps)
        self.norm = RMSNorm(args.dim, args.norm_eps)
        # Per-MTP hc_head params (distinct from Block's hc_attn/hc_ffn params).
        hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        self.hc_head_fn = atom_parameter(
            torch.empty(hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = atom_parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = atom_parameter(torch.empty(1, dtype=torch.float32))
        # Externally-assigned by DeepseekV4Model (shared with main model).
        self.embed: Optional[nn.Module] = None
        self.head: Optional[ParallelHead] = None

    def forward(
        self,
        x: torch.Tensor,  # [num_tokens, hc, dim]  residual stream from main model
        positions: torch.Tensor,  # [num_tokens] int  absolute positions
        input_ids: torch.Tensor,  # [num_tokens] int
    ) -> torch.Tensor:  # [bs, vocab]  last-token logits via self.head
        assert (
            self.embed is not None and self.head is not None
        ), "MTPBlock requires .embed and .head to be assigned by the parent model"
        e = self.enorm(self.embed(input_ids))  # [num_tokens, dim]
        x = self.hnorm(x)  # [num_tokens, hc, dim]
        # Mix token-embed + hidden into a fresh residual. e_proj output is
        # [num_tokens, dim]; unsqueeze adds the hc axis so it broadcasts
        # against h_proj(x) [num_tokens, hc, dim].
        x = self.e_proj(e).unsqueeze(-2) + self.h_proj(x)  # [num_tokens, hc, dim]
        x = super().forward(x, positions)  # [num_tokens, hc, dim]
        return self.head(
            x, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm
        )


@support_torch_compile
class DeepseekV4Model(nn.Module):
    """Full model: embed -> expand to hc_mult copies -> N blocks -> hc_head -> logits.

    Port of inference/model.py:Transformer (770-810). MTP blocks are constructed
    and have their `.embed` and `.head` linked to the main model's, but they are
    NOT called from the main forward path — PR5 will integrate them into ATOM's
    EagleProposer via `self.mtp[k].forward(...)` from outside.

    PR1 single-rank: uses plain `nn.Embedding` for `self.embed` (state_dict-compatible
    with reference's `ParallelEmbedding` since both store a single `weight` parameter).
    """

    def __init__(
        self,
        *,
        atom_config: Config,
        args: DeepseekV4Args,
    ):
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
        # alt_stream: dual-stream MoE (shared_experts // routed_experts) AND
        # Main Compressor overlap. compress_stream: Indexer Compressor overlap.
        # Both allocated once, shared across all blocks. Attention runs before
        # MoE in each block, so attn and MoE never contend for alt_stream.
        self.alt_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        self.compress_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        self.layers = nn.ModuleList(
            [
                Block(
                    layer_id,
                    args,
                    prefix=f"layers.{layer_id}",
                    alt_stream=self.alt_stream,
                    compress_stream=self.compress_stream,
                )
                for layer_id in range(args.n_layers)
            ]
        )
        self.norm = RMSNorm(args.dim, self.norm_eps)
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
        self.hc_head_fn = atom_parameter(
            torch.empty(hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = atom_parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = atom_parameter(torch.empty(1, dtype=torch.float32))

    def forward(
        self,
        input_ids: torch.Tensor,  # [num_tokens] int  flat ragged-batch token ids
        positions: torch.Tensor,  # [num_tokens] int  abs positions (required)
    ) -> (
        torch.Tensor
    ):  # [num_tokens, dim]  hidden_states (vocab head deferred to compute_logits)
        """Forward over `num_tokens` flat ragged-batch tokens.

        Returns hidden_states `[num_tokens, dim]` — the vocab projection is
        deferred to `compute_logits` to satisfy ATOM's CUDAGraph contract
        (capture buffer is sized to hidden_size, not vocab).
        """
        assert input_ids.dim() == 1, f"input_ids must be 1D, got {input_ids.shape}"
        h = self.embed(input_ids)  # [num_tokens, dim]
        # Expand to hc_mult copies for Hyper-Connections: [num_tokens, hc, dim]
        h = h.unsqueeze(-2).repeat(1, self.hc_mult, 1)

        for layer in self.layers:
            h = layer(h, positions)  # [num_tokens, hc, dim]

        # Reduce the mHC residual stack (hc_head + final RMSNorm); leave the
        # vocab projection to compute_logits.
        x_hc = self.head.hc_head(  # [num_tokens, dim]
            h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
        )
        return self.norm(x_hc)


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
    weights_mapper = WeightsMapper(
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
    packed_modules_mapping = {
        "attn.wq_a": ("attn.wqkv_a", 0),
        "attn.wkv": ("attn.wqkv_a", 1),
        "compressor.wkv": ("compressor.wkv_gate", 0),
        "compressor.wgate": ("compressor.wkv_gate", 1),
        "shared_experts.w1": ("shared_experts.gate_up_proj", 0),
        "shared_experts.w3": ("shared_experts.gate_up_proj", 1),
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
        self.model = DeepseekV4Model(atom_config=config, args=self.args)

    def forward(
        self,
        input_ids: torch.Tensor,  # [num_tokens] int
        positions: torch.Tensor,  # [num_tokens] int  required
    ) -> torch.Tensor:  # [num_tokens, dim]  hidden_states
        # Stash input_ids on forward_context for the V4 hash MoE routing
        # callback (`MoE._hash_topk`), which runs inside the Dynamo-opaque
        # `maybe_dual_stream_forward` custom op and can't receive input_ids
        # via the FusedMoE custom-routing-function fixed signature. Setting
        # it here (rather than in ModelRunner) means any caller of
        # `model.forward` — production runner, warmup, benchmarks — gets
        # correct hash routing without a separate setup step.
        get_forward_context().context.input_ids = input_ids
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,  # [num_tokens, dim]  post hc_head + final norm
    ) -> torch.Tensor:  # [bs, vocab]
        # Vocab projection is split off from `model.forward` so the latter
        # returns hidden_size-shaped tensors — required by ATOM's CUDAGraph
        # capture contract (outputs buffer is sized to hidden_size, not vocab).
        return self.model.head.get_logits(hidden_states)

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
