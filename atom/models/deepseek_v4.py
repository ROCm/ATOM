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

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from aiter import QuantType as _AiterQuantType
from aiter import dtypes, get_hip_quant
from aiter.dist.parallel_state import get_tensor_model_parallel_world_size
from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits
from atom.config import Config
from atom.model_ops.embed_head import VocabParallelEmbedding
from atom.model_ops.layernorm import RMSNorm, rmsnorm2d_fwd_
from atom.model_ops.triton_rmsnorm_nw import rmsnorm_nw
from atom.model_ops.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.model_ops.moe import FusedMoE
from atom.model_ops.quant_v4 import (
    act_quant_inplace,
    rotate_activation,
)
from atom.model_ops.sparse_attn_v4 import (  # noqa: F401
    hc_split_sinkhorn,
    sparse_attn_ragged_varlen,
)
from atom.model_ops.utils import atom_parameter
from atom.model_ops.v4_kernels import (  # noqa: F401
    CompressPlan,
    fused_compress_attn,
    swa_write,
    update_compressor_states,
)
from atom.utils.forward_context import get_forward_context

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
_V4_AITER_HC_POST = os.environ.get("V4_AITER_HC_POST", "") == "1"


def _rmsnorm_nw(x: torch.Tensor, eps: float, dim: int) -> torch.Tensor:
    if _V4_USE_TRITON_RMSNORM:
        return rmsnorm_nw(x, eps)
    ones = torch.ones(dim, dtype=x.dtype, device=x.device)
    return rmsnorm2d_fwd_(x, ones, eps, dim)


def _v4_gather_compressed_batched(
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,  # [bs, max_blocks_per_seq]
    gather_indices: dict,
) -> torch.Tensor:
    """Gather compressed entries from `kv_cache` using pre-built index tensors
    (built once per fwd in `_build_v4_gather_indices`).

    Returns:
      gathered_flat: `[total_committed, head_dim]` concatenated in seq order.
    """
    batch_ids_gpu = gather_indices["batch_ids_gpu"]
    if batch_ids_gpu is None:
        return torch.empty(
            0, kv_cache.size(-1), dtype=kv_cache.dtype, device=kv_cache.device
        )
    physical_blocks = block_tables[
        batch_ids_gpu, gather_indices["block_in_seq_gpu"]
    ].long()
    return kv_cache[physical_blocks, gather_indices["slot_in_block_gpu"]]


def _segment_indices(
    seq_ids: np.ndarray, lens: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """For ragged segments (one per `seq_ids[i]` of length `lens[i]`), return
    flat (per-row seq id, per-row local position) arrays of total length
    `sum(lens)`.
    """
    total = int(lens.sum())
    if total == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    token_seq_ids = np.repeat(seq_ids.astype(np.int64), lens)
    cum = np.concatenate(([0], np.cumsum(lens.astype(np.int64))[:-1]))
    local_pos = np.arange(total, dtype=np.int64) - np.repeat(cum, lens)
    return token_seq_ids, local_pos


def _v4_build_sparse_inputs_batched(
    *,
    kv: torch.Tensor,
    swa_kv: torch.Tensor,
    kv_compress_batched: Optional[torch.Tensor],
    compressor_kv_cache: Optional[torch.Tensor],
    block_tables_gpu: Optional[torch.Tensor],
    window_topk_batched: torch.Tensor,
    indexer_topk_batched: Optional[torch.Tensor],
    pack_meta: dict,
    has_indexer: bool,
    ratio: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GPU-only `kv_sa` + `topk_flat` packer using pre-built index tensors.

    All the per-(fwd, ratio) CPU index math + H2D copies live in
    `DeepseekV4AttentionMetadataBuilder._build_v4_pack_meta_for_ratio`. Here
    we only run the GPU index_copy_/index_select kernels — eliminating the
    ~14 per-layer `torch.as_tensor` synchronous transfers from the legacy
    inline path below.
    """
    head_dim = kv.size(-1)
    device = kv.device
    kv_flat_sa = torch.empty(
        pack_meta["total_kv"], head_dim, dtype=kv.dtype, device=device
    )
    topk_flat = torch.empty(pack_meta["total_topk"], dtype=torch.int32, device=device)

    # Part A: prefill kv slice
    if pack_meta["prefill_kv_dst_gpu"] is not None:
        kv_flat_sa.index_copy_(
            0,
            pack_meta["prefill_kv_dst_gpu"],
            kv.squeeze(0).index_select(0, pack_meta["prefill_kv_src_gpu"]),
        )

    # Part A: decode swa gather
    if pack_meta["decode_state_slots_gpu"] is not None:
        decode_swa = swa_kv.index_select(
            0, pack_meta["decode_state_slots_gpu"]
        ).reshape(-1, head_dim)
        kv_flat_sa.index_copy_(0, pack_meta["decode_swa_dst_gpu"], decode_swa)

    # Part B: prefill compress
    if (
        ratio > 0
        and kv_compress_batched is not None
        and pack_meta["prefill_compress_dst_gpu"] is not None
    ):
        kv_flat_sa.index_copy_(
            0,
            pack_meta["prefill_compress_dst_gpu"],
            kv_compress_batched.index_select(0, pack_meta["prefill_compress_src_gpu"]),
        )

    # Part B: decode compress (gather from compressor's kv_cache)
    if (
        ratio > 0
        and compressor_kv_cache is not None
        and block_tables_gpu is not None
        and pack_meta["decode_compress_seqs_gpu"] is not None
    ):
        decode_block_tables = block_tables_gpu.index_select(
            0, pack_meta["decode_compress_seqs_gpu"]
        )
        gathered_decode_compress = _v4_gather_compressed_batched(
            compressor_kv_cache,
            decode_block_tables,
            pack_meta["decode_compress_gather_indices"],
        )
        kv_flat_sa.index_copy_(
            0, pack_meta["decode_compress_dst_gpu"], gathered_decode_compress
        )

    # Topk window
    if pack_meta["window_topk_dst_gpu"] is not None:
        topk_flat.index_copy_(
            0,
            pack_meta["window_topk_dst_gpu"],
            window_topk_batched.reshape(-1).index_select(
                0, pack_meta["window_topk_src_gpu"]
            ),
        )

    # Topk compress
    if pack_meta["compress_topk_dst_gpu"] is not None:
        if has_indexer:
            assert (
                indexer_topk_batched is not None
                and pack_meta["compress_topk_src_gpu"] is not None
            )
            topk_flat.index_copy_(
                0,
                pack_meta["compress_topk_dst_gpu"],
                indexer_topk_batched.reshape(-1).index_select(
                    0, pack_meta["compress_topk_src_gpu"]
                ),
            )
        else:
            assert pack_meta["compress_topk_values_gpu"] is not None
            topk_flat.index_copy_(
                0,
                pack_meta["compress_topk_dst_gpu"],
                pack_meta["compress_topk_values_gpu"],
            )

    return kv_flat_sa, topk_flat


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
    from atom.config import LayerQuantConfig, QuantizationConfig, QuantType

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
        import aiter as ops

        cos, sin = self._caches(query.device)
        num_tokens = positions.numel()
        # rotate_style=1 → GPT-J / interleaved (matches V4's view_as_complex).
        rotate_style = 1
        q_view = query.view(1, num_tokens, -1, self.rotary_dim)
        positions_view = positions.view(1, num_tokens)
        if key is not None:
            k_view = key.view(1, num_tokens, -1, self.rotary_dim)
            ops.rope_cached_positions_2c_fwd_inplace(
                q_view,
                k_view,
                cos,
                sin,
                positions_view,
                rotate_style,
                reuse_freqs_front_part=True,
                nope_first=False,
            )
        else:
            ops.rope_cached_positions_fwd_inplace(
                q_view,
                cos,
                sin,
                positions_view,
                rotate_style,
                reuse_freqs_front_part=True,
                nope_first=False,
            )


def _build_window_topk_batched(
    positions: torch.Tensor,  # [total_tokens] int (abs token positions)
    start_pos_per_token: torch.Tensor,  # [total_tokens] int (each token's seq start_pos)
    window_size: int,
) -> torch.Tensor:  # [total_tokens, window_size] int32
    """Per-token sliding-window topk indices for the whole batch.

    Three-branch semantics:
      - sp == 0 (fresh prefill): matrix entries = abs positions in the window
        [pos-win+1, pos] clamped to [0, pos]; mask future via -1.
      - 0 < sp < win-1 (prefix mode): all tokens in the seq share a single
        matrix [0..sp, -1, ..., -1] (matches original semantics, including
        MTP-N where the same start_pos is reused).
      - sp >= win-1 (cyclic mode): cyclic ring offsets starting at sp+1 mod win.
    """
    device = positions.device
    total = positions.size(0)
    arange_w = torch.arange(window_size, device=device, dtype=positions.dtype).view(
        1, window_size
    )
    pos_col = positions.view(total, 1)
    sp_col = start_pos_per_token.view(total, 1)
    neg1 = torch.tensor(-1, device=device, dtype=positions.dtype)

    # Case A: sp == 0 (fresh prefill) — abs positions [pos-win+1, pos] clamped.
    case_a = (pos_col - window_size + 1).clamp(min=0) + arange_w
    case_a = torch.where(case_a > pos_col, neg1, case_a)

    # Case B: 0 < sp < win-1 (prefix mode) — shared per-seq matrix.
    case_b = arange_w.expand(total, window_size).clone()
    case_b = torch.where(arange_w > sp_col, neg1, case_b)

    # Case C: sp >= win-1 (cyclic mode) — ring offsets.
    sp_mod = sp_col % window_size
    case_c = (sp_mod + 1 + arange_w) % window_size

    sp_eq_0 = sp_col == 0
    sp_in_prefix = (sp_col > 0) & (sp_col < window_size - 1)

    out = case_c
    out = torch.where(sp_in_prefix.expand_as(out), case_b, out)
    out = torch.where(sp_eq_0.expand_as(out), case_a, out)
    return out.to(torch.int32)


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
        self.norm = RMSNorm(self.head_dim, args.norm_eps)

        # External tensors — assigned by the owning Attention / Indexer at first forward.
        self.kv_cache: Optional[torch.Tensor] = None
        self.rotary_emb: Optional[_V4RoPE] = None
        # Shared compress-output buffer (set by V4 builder.build_kv_cache_tensor
        # to a per-kind torch.empty pre-allocation in `forward_vars`). When
        # present, fused_compress_attn writes into it via `out=` so the GPU
        # pointer stays stable across captures (CUDAGraph requirement); when
        # None, falls back to the eager fresh-allocation path.
        self.compress_out: Optional[torch.Tensor] = None

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
        x: torch.Tensor,
        plan: "CompressPlan",
        state_slot_mapping: torch.Tensor,
        block_tables: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
    ) -> Optional[torch.Tensor]:
        """Batched plan-style compress: one fused kernel call for the whole
        fwd's batch (across all seqs).

        Single fused Triton kernel does pool + RMSNorm + RoPE + bf16 kv_cache
        scatter in one launch. Each compression boundary across the batch is
        one row in `plan.compress_plan_gpu`. State cache update fires after
        (write order critical — fused kernel reads state-cache-as-of-previous-
        fwd; update_compressor_states overwrites for next fwd).

        TODO: quant (FP8 ue8m0 for CSA, FP4 + Hadamard for Indexer) is NOT
        applied; see class docstring.

        Args:
            x:           [num_q_tokens, dim] flat ragged batch.
            plan:        CompressPlan from attn_metadata.compress_plans[ratio]
                         (or a synthetic bs=1 plan during warmup).
            state_slot_mapping: [bs] int32 — per-seq state cache slot
                         (attn_metadata.state_slot_mapping).
            block_tables: [bs, max_blocks_per_seq] int32 — physical block IDs
                         per seq; None during warmup (skips kv_cache scatter).
            out_dtype:   override kernel output dtype. Defaults to bf16.

        Returns:
            Compressed KV `[num_compress, head_dim]` post-norm post-rope BF16,
            in plan order (= per-seq grouped by `plan.cu_compress_cpu`).
            Returns None if `plan.num_compress == 0`.
        """
        assert self.rotary_emb is not None, "compressor.rotary_emb must be set by owner"
        if x.dim() == 3:
            assert x.size(0) == 1, f"3D x must have B=1, got {x.size(0)}"
            x = x.squeeze(0)
        assert x.dim() == 2, f"x must be [num_q_tokens, dim], got {x.shape}"
        ratio = self.compress_ratio
        overlap = self.overlap
        d = self.head_dim
        rd = self.rope_head_dim
        if out_dtype is None:
            out_dtype = torch.bfloat16 if x.dtype == torch.float32 else x.dtype

        # Projection (always fp32 for stability).
        x = x.float()
        kv = self.wkv(x)  # [num_q_tokens, coff*head_dim]
        score = self.wgate(x)

        # ====== Unified fused kernel path (CSA + Indexer) ======
        # Order is critical: fused kernel reads state cache as-of-end-of-
        # PREVIOUS-fwd. `update_compressor_states` overwrites them with this
        # fwd's data for the NEXT fwd's overlap — must run AFTER the fused
        # kernel.
        #
        # The kernel does pool + RMSNorm + RoPE + BF16 store to kv_cache.
        # No quant is applied (FP8 ue8m0 for CSA, FP4 + Hadamard for Indexer
        # are both dropped). Trade-off: removes the QAT round-trip the model
        # was trained with → output quality drops measurably from baseline.
        # TODO: port the quant variants into the kernel for full byte-equal.
        cos_cache, sin_cache = self.rotary_emb._caches(x.device)
        # Warmup runs through the same path: kv_cache is None until the V4
        # builder binds it post-warmup, so the kernel skips the scatter.
        # When real, writes to block 0 are overwritten when slot 0 is later
        # assigned to a real seq (start_pos==0 path discards prior state).
        scatter_kv_cache = self.kv_cache if block_tables is not None else None
        scatter_block_tables = block_tables if scatter_kv_cache is not None else None
        out = fused_compress_attn(
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
            out_dtype=out_dtype,
            out=self.compress_out,
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
        return out


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
        self._fp8_quant_func = get_hip_quant(_AiterQuantType.per_1x128)
        self._weights_scale = self.softmax_scale * self.n_heads**-0.5

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

    def forward_batched(
        self,
        x_full: torch.Tensor,  # [total_tokens, dim]
        qr_full: torch.Tensor,  # [total_tokens, q_lora_rank]
        positions: torch.Tensor,  # [total_tokens]
        block_tables: torch.Tensor,  # [bs, max_blocks_per_seq]
        indexer_meta: dict,
    ) -> torch.Tensor:
        """Batched score+topk across all seqs in one fp8_mqa_logits call.

        Caller must invoke `self.compressor` once batched BEFORE this so all
        seqs' Indexer kv_cache is already populated.

        `indexer_meta` (built once per fwd in
        `DeepseekV4AttentionMetadataBuilder._build_v4_indexer_meta`) carries
        every CPU array and pre-uploaded GPU index tensor — the per-CSA-layer
        call has zero CPU index math and zero H2D copies.

        Returns:
          topk_flat: `[total_tokens, max_K] int32`, padded with -1 to max K
            (max over per-seq K). Per-token usable width = K_per_seq.
        """
        assert self.rotary_emb is not None
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        device = x_full.device
        total_tokens = x_full.size(0)

        max_k = indexer_meta["max_k"]
        if max_k == 0:
            return torch.full((total_tokens, 0), -1, dtype=torch.int32, device=device)

        # Q proj + RoPE + rotate (batched).
        q = self.wq_b(qr_full).view(total_tokens, self.n_heads, self.head_dim)
        q = q.unsqueeze(0)
        self.rotary_emb(positions, q[..., -rd:])
        q = rotate_activation(q)

        # FP8 quant Q + batched gather K + FP8 quant. `_fp8_quant_func`,
        # `_weights_scale` precomputed in __init__.
        q_2d = q.squeeze(0).contiguous().view(-1, self.head_dim)
        q_fp8, q_scale = self._fp8_quant_func(q_2d, quant_dtype=dtypes.fp8)
        q_fp8 = q_fp8.view(total_tokens, self.n_heads, self.head_dim)
        q_scale = q_scale.view(total_tokens, self.n_heads, 1)

        gathered_flat = _v4_gather_compressed_batched(
            self.kv_cache, block_tables, indexer_meta["gather_indices"]
        )
        k_fp8, k_scale = self._fp8_quant_func(gathered_flat, quant_dtype=dtypes.fp8)

        # weights = weights_proj * q_scale * (softmax_scale * 1/sqrt(H))
        weights = (
            (self.weights_proj(x_full).unsqueeze(-1) * q_scale * self._weights_scale)
            .squeeze(-1)
            .float()
        )

        # All per-token broadcast helpers + layer-invariant derivations are
        # pre-built in `_build_v4_indexer_meta`.
        seq_base_per_token = indexer_meta["seq_base_per_token_gpu"]
        cu_starts = indexer_meta["cu_starts_gpu"]
        cu_ends = indexer_meta["cu_ends_gpu"]
        future_threshold = indexer_meta["future_threshold_gpu"]
        width_mask = indexer_meta["width_mask_gpu"]
        offset_per_token = indexer_meta["offset_per_token_gpu"]
        is_prefill_per_token = indexer_meta["is_prefill_per_token_gpu"]

        logits = fp8_mqa_logits(
            Q=q_fp8,
            KV=k_fp8,
            kv_scales=k_scale.view(-1).float(),
            weights=weights,
            cu_starts=cu_starts,
            cu_ends=cu_ends,
        )  # [total_tokens, total_committed] fp32; outside [start,end) is -inf

        # PyTorch topk over -inf-masked logits. aiter `top_k_per_row_prefill`
        # would be the obvious replacement but it hardcodes K=2048 — V4's
        # K=index_topk=64 doesn't fit (the kernel writes 2048 ints/row,
        # overflowing a [tok, 64] indices buffer and corrupting memory).
        topk_global = logits.topk(max_k, dim=-1)[1].to(torch.int32)
        # Global flat index → seq-local compress idx; drop slots past per-seq K.
        topk_local = topk_global - seq_base_per_token.unsqueeze(1)
        topk_local = topk_local.masked_fill(width_mask, -1)

        # Per-seq offset to land indices in the [SWA || compressed] kv_sa
        # layout consumed by sparse_attn (token_num for fresh prefill, win
        # for decode); future-mask only applies in fresh prefill.
        future_mask = is_prefill_per_token & (topk_local >= future_threshold)
        topk_with_offset = topk_local + offset_per_token.unsqueeze(1)
        topk_final = torch.where(
            (topk_local < 0) | future_mask,
            torch.full_like(topk_local, -1),
            topk_with_offset,
        )
        return topk_final


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
        self.attn_sink = atom_parameter(
            torch.empty(self.n_local_heads, dtype=torch.float32)
        )
        self.wq_a = ReplicatedLinear(
            self.dim,
            self.q_lora_rank,
            bias=False,
            quant_config=qc,
            prefix=f"{p}.wq_a",
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.q_norm2 = RMSNorm(self.head_dim, self.eps)
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
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
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
        from atom.config import QuantType as _QT

        self.wo_a.quant_type = _QT.No

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Compute attention for `x` at absolute token `positions`.

        PR3-main: handles batched multi-sequence input. Linear projections + RoPE
        run once on the flat `[S_total, ...]` batch; SWA write, Compressor scatter,
        sparse_attn (gather + score) iterate over sequences using per-seq slot +
        block_table from the V4 attention builder's metadata.

        Args:
            x: [num_tokens, dim] flat ATOM ragged-batch convention.
            positions: [num_tokens] absolute token positions per token.
                Per-seq slicing uses `cu_seqlens_q` from `forward_context`.
        Returns:
            [num_tokens, dim] attention output (BF16).
        """
        assert (
            x.dim() == 2
        ), f"DeepseekV4Attention expects 2D [num_tokens, dim], got {x.shape}"
        seqlen_total = x.size(0)
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim

        # Idempotent one-time plumb of rotary_emb into compressor / indexer
        # (and the indexer's inner compressor). `rotary_emb` is set by the
        # owning layer after __init__, so this can't move into __init__.
        if self.compress_ratio and self.compressor.rotary_emb is None:
            self.compressor.rotary_emb = self.rotary_emb
            if self.indexer is not None:
                self.indexer.rotary_emb = self.rotary_emb
                self.indexer.compressor.rotary_emb = self.rotary_emb

        # ----- Batched ops on full flat tensors -----
        # `_V4_FORCE_UE8M0_QUANT` (module-level): round-trip x/qr to ue8m0-FP8
        # to mirror the reference's `act_quant(scale_fmt="ue8m0")` Linear-input
        # quantization. EXPERIMENT only.
        if _V4_FORCE_UE8M0_QUANT:
            x = x.clone()
            act_quant_inplace(x, 128, "ue8m0")
        qr = self.q_norm(self.wq_a(x))  # [S_total, q_lora_rank], shared with Indexer
        if _V4_FORCE_UE8M0_QUANT:
            qr = qr.clone()
            act_quant_inplace(qr, 128, "ue8m0")
        q = self.wq_b(qr).view(seqlen_total, self.n_local_heads, self.head_dim)
        q = _rmsnorm_nw(q, self.eps, self.head_dim)
        q = q.unsqueeze(0)  # [1, S_total, H, D]

        kv = self.wkv(x)  # [S_total, head_dim]
        kv = self.kv_norm(kv).unsqueeze(0)  # [1, S_total, head_dim]
        self.rotary_emb(positions, q[..., -rd:], kv[..., -rd:])
        if _V4_USE_REF_QUANT:
            act_quant_inplace(kv[..., :-rd], 64, self.scale_fmt)
        # ===== Per-fwd metadata (built once in prepare_prefill/decode). =====
        # All per-fwd state read once. Production prepare_decode/prefill
        # always populates these; warmup goes through the same path
        # (`_populate_state_slot_mapping` falls back to slot 0).
        attn_md = get_forward_context().attn_metadata
        compress_plans = attn_md.compress_plans
        v4_sparse_layouts = attn_md.v4_sparse_layouts
        v4_indexer_meta = attn_md.v4_indexer_meta
        window_topk_batched = attn_md.window_topk_batched
        swa_write_indices = attn_md.swa_write_indices
        swa_positions_filtered = attn_md.swa_positions_filtered
        swa_slot_per_token_filtered = attn_md.swa_slot_per_token_filtered
        block_tables_gpu = attn_md.block_tables
        state_slot_mapping = attn_md.state_slot_mapping

        # ===== Batched compressor + Indexer (ONCE per layer) =====
        # State cache reset on fresh prefill is redundant: SWA's start_pos==0
        # path uses raw seq_kv, and fused_compress's state-cache reads are
        # already masked by `s < 0` for fresh prefill (compress_plan.py:88 +
        # fused_compress.py:124-127).
        plan_for_layer = compress_plans[ratio] if ratio else None
        kv_compress_batched = (
            self.compressor(
                x,
                plan=plan_for_layer,
                state_slot_mapping=state_slot_mapping,
                block_tables=block_tables_gpu,
            )
            if ratio
            else None
        )
        indexer_topk_batched = None
        if self.indexer is not None:
            # Indexer's inner compressor populates the indexer kv_cache; the
            # outer `forward_batched` reads `v4_indexer_meta` (built once per
            # fwd) so it has zero per-layer H2D / CPU index math.
            self.indexer.compressor(
                x,
                plan=plan_for_layer,
                state_slot_mapping=state_slot_mapping,
                block_tables=block_tables_gpu,
            )
            indexer_topk_batched = self.indexer.forward_batched(
                x_full=x,
                qr_full=qr,
                positions=positions,
                block_tables=block_tables_gpu,
                indexer_meta=v4_indexer_meta,
            )

        # ===== SWA write =====
        # `swa_write_indices` (and the filtered positions/slots) are pre-built
        # once per fwd in `_attach_v4_per_fwd_meta` — pre-filtered to the last
        # `win` tokens per seq to avoid intra-seq ring-slot races. None means
        # "nothing to write" (warmup / empty batch).
        if swa_write_indices is not None:
            swa_write(
                kv.squeeze(0)[swa_write_indices].contiguous(),
                swa_positions_filtered,
                swa_slot_per_token_filtered,
                self.swa_kv,
                win,
            )

        # ===== Sparse-attn inputs (kv_sa, topk_flat) + ragged layout =====
        # Pre-built `pack_meta` (built once in `_build_v4_pack_meta_for_ratio`)
        # carries every CPU/GPU index — the per-layer call is just GPU
        # index_copy_/select ops.
        q_sa = q.squeeze(0).contiguous()
        layout = v4_sparse_layouts[ratio]
        kv_sa, topk_flat = _v4_build_sparse_inputs_batched(
            kv=kv,
            swa_kv=self.swa_kv,
            kv_compress_batched=kv_compress_batched,
            compressor_kv_cache=self.compressor.kv_cache if ratio else None,
            block_tables_gpu=block_tables_gpu,
            window_topk_batched=window_topk_batched,
            indexer_topk_batched=indexer_topk_batched,
            pack_meta=layout["pack_meta"],
            has_indexer=self.indexer is not None,
            ratio=ratio,
        )
        topk_starts = layout["topk_starts"]
        topk_lens = layout["topk_lens"]
        kv_offsets = layout["kv_offsets"]
        max_topk = layout["max_topk"]

        o = sparse_attn_ragged_varlen(
            q_sa,
            kv_sa,
            self.attn_sink,
            topk_flat,
            topk_starts,
            topk_lens,
            kv_offsets,
            max_topk,
            self.softmax_scale,
        ).unsqueeze(0)

        # Inverse RoPE on output's rope dims to remove absolute-position
        # contribution carried in by the value-side RoPE of the KV entries.
        # aiter has no inverse-RoPE kernel; rebuild per-token complex freqs
        # from the rotary_emb's cos/sin cache and reuse the manual multiply.
        freqs_slice = self.rotary_emb.freqs_for_positions(positions)
        _apply_rotary_emb(o[..., -rd:], freqs_slice, inverse=True)
        # ----- Grouped output LoRA (batched on the full flat tensor) -----
        o = o.squeeze(0).view(seqlen_total, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("sgd,grd->sgr", o, wo_a)
        x = self.wo_b(o.flatten(1))
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
        self.weight = atom_parameter(torch.empty(args.n_routed_experts, args.dim))
        if self.hash:
            self.tid2eid = atom_parameter(
                torch.empty(
                    args.vocab_size, args.n_activated_experts, dtype=torch.int32
                ),
            )
            self.bias = None
        else:
            self.bias = atom_parameter(
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
            # V4 hash-routed layers (layer_id < n_hash_layers) use tid2eid lookup,
            # not bias-corrected gate-logit routing — checkpoint has no
            # `gate.bias` for those layers. Only allocate the bias for
            # sqrtsoftplus layers to avoid 3 spurious unloaded-param warnings.
            if not self.is_hash_layer:
                self.gate.e_score_correction_bias = atom_parameter(
                    torch.empty(self.n_routed_experts, dtype=torch.float32)
                )
            if self.is_hash_layer:
                # tid2eid: per-token-id top-k expert lookup table (V4 first 3
                # layers use this in lieu of gate-logit routing).
                self.gate.tid2eid = atom_parameter(
                    torch.empty(
                        args.vocab_size, args.n_activated_experts, dtype=torch.int32
                    ),
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
                e_score_correction_bias=getattr(
                    self.gate, "e_score_correction_bias", None
                ),
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
            # `_V4_USE_REF_QUANT` (module-level): round-trip sub_x through FP8
            # ue8m0 to mirror ref Expert.forward (act_quant before fp4_gemm).
            # See notes/17.
            if _V4_USE_REF_QUANT:
                sub_x = sub_x.to(torch.bfloat16)
                act_quant_inplace(sub_x, 128, "ue8m0")
                sub_x = sub_x.float()

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

            act_bf = act.to(torch.bfloat16)
            if _V4_USE_REF_QUANT:
                act_quant_inplace(act_bf, 128, "ue8m0")
            out = act_bf.float() @ w2_f.T  # [N, H]
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
                e_score_correction_bias=getattr(
                    self.gate, "e_score_correction_bias", None
                ),
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
        shared_out = self.shared_experts(x)
        y = y + shared_out
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
        x = x.flatten(-2)  # [..., hc*D]
        hc_dim = x.shape[-1]
        x_normed = _rmsnorm_nw(x, self.norm_eps, hc_dim)
        mixes = F.linear(x_normed.float(), hc_fn)  # [..., mix_hc]
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

        Defaults to the torch reference; aiter `mhc_post` is opt-in via
        `V4_AITER_HC_POST=1`. The aiter kernel produces small numerical drift
        per call that compounds over 30+ autoregressive decode steps and
        collapses long generations into degenerate same-word repetition; the
        torch path is bit-stable across the full decode trajectory. See
        `/app/logs_claude/deepseek_v4/notes/12_aiter_mhc_post_root_cause.md`
        for the diagnosis. Re-enable aiter once the kernel is fixed upstream.
        """
        if _V4_AITER_HC_POST:
            try:
                import aiter as _aiter

                mhc_post = getattr(_aiter, "mhc_post", None)
            except ImportError:
                mhc_post = None
            dim = residual.shape[-1]
            aiter_ok = (
                mhc_post is not None
                and x.is_cuda
                and (dim % 512 == 0 or dim % 256 == 0)
            )
        else:
            aiter_ok = False
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
        positions: torch.Tensor,
        input_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:

        # ----- Attention sub-layer with mHC mixing -----
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attn_norm(x)
        x = self.attn(x, positions)
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
        self.weight = atom_parameter(
            torch.empty(self.vocab_size, self.dim, dtype=torch.float32)
        )

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Project the last-token-per-seq slice of `x` to vocab logits.

        Accepts either:
          - 2D [num_tokens, D] (ATOM ragged-batch convention): picks the
            last token of each sequence using `cu_seqlens_q` from the
            forward_context — returns `[bs, vocab]`. Falls back to `x[-1:]`
            (single-seq) when no forward_context is set (warmup / standalone).
          - 3D [B, S, D] (legacy): takes `x[:, -1, :]` → `[B, vocab]`.
        """
        if x.dim() == 2:
            ctx = get_forward_context()
            cu_seqlens_q = (
                ctx.attn_metadata.cu_seqlens_q
                if ctx is not None and ctx.attn_metadata is not None
                else None
            )
            if cu_seqlens_q is not None and cu_seqlens_q.numel() >= 2:
                # Last-token positions per seq: cu_seqlens_q[1:bs+1] - 1.
                # block_tables tells us the actual scheduled bs (cu_seqlens_q
                # may have trailing zeros from the CpuGpuBuffer pool).
                bs = ctx.attn_metadata.block_tables.size(0)
                last_idx = cu_seqlens_q[1 : bs + 1] - 1
                return F.linear(x.index_select(0, last_idx.long()).float(), self.weight)
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
        x = x.flatten(-2)  # [..., hc*D]
        hc_dim = x.shape[-1]
        x_normed = _rmsnorm_nw(x, self.norm_eps, hc_dim)
        mixes = F.linear(x_normed.float(), hc_fn)
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
        self, x: torch.Tensor, positions: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Forward.

        Args:
            x: residual stream from main model. Either [num_tokens, hc, D]
                (ATOM 2D-flat convention) or [B, S, hc, D] (legacy 4D).
            positions: [num_tokens] absolute token positions.
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
        x = super().forward(x, positions, input_ids)
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

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        **model_kwargs: dict,
    ) -> torch.Tensor:
        """Forward.

        Args:
            input_ids: 1D `[num_tokens]` (ATOM 2D-flat convention) OR 2D
                `[B, S]` (legacy reference convention; treated as a single
                sequence of B*S tokens — only correct for B=1).
            positions: [num_tokens] absolute token positions. If None, defaults
                to `arange(num_tokens)` (i.e. start_pos=0 prefill).
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
        if positions is None:
            positions = torch.arange(
                input_ids.numel(), device=input_ids.device, dtype=torch.long
            )

        for layer in self.layers:
            h = layer(h, positions, input_ids)

        # CUDAGraph contract: model.forward returns hidden_states sized
        # [num_tokens, hidden_size]; compute_logits applies the vocab head.
        # Reduce the hc_mult residual stack here (hc_head + final RMSNorm),
        # leaving the vocab projection to compute_logits.
        x_hc = self.head.hc_head(
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
        return self.model(input_ids=input_ids, positions=positions, **model_kwargs)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # `hidden_states` is post-hc_head + norm (= [num_tokens, dim]); apply
        # the vocab projection here so DeepseekV4Model.forward can stay aligned
        # with ATOM's standard hidden_states-shaped contract (required for
        # CUDAGraph capture: outputs buffer is sized to hidden_size, not vocab).
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
