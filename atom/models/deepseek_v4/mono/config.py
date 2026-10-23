# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications: namespace/dependency extraction for ATOM, 2026-10-13.

"""V4-Pro TP8 geometry, arithmetic profiles, and storage-layout contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MoeMode(str, Enum):
    """Public arithmetic modes for the expert up/gate and down projections."""

    W8A8 = "w8a8"
    W8A16 = "w8a16"
    A16W4 = "a16w4"
    A8W4 = "a8w4"


class ExpertActivation(str, Enum):
    """Activation representation consumed by both expert projections."""

    FP8_BLOCK128 = "fp8_block128"
    MXFP8_BLOCK32 = "mxfp8_block32"
    BF16 = "bf16"


class ExpertWeight(str, Enum):
    """Packed expert-weight representation."""

    FP8_BLOCK128 = "fp8_block128"
    MXFP4_BLOCK32 = "mxfp4_block32"


@dataclass(frozen=True)
class MoeFormat:
    activation: ExpertActivation
    weight: ExpertWeight

    @property
    def activation_group(self) -> int | None:
        if self.activation is ExpertActivation.FP8_BLOCK128:
            return 128
        if self.activation is ExpertActivation.MXFP8_BLOCK32:
            return 32
        return None


MOE_FORMATS = {
    MoeMode.W8A8: MoeFormat(ExpertActivation.FP8_BLOCK128, ExpertWeight.FP8_BLOCK128),
    MoeMode.W8A16: MoeFormat(ExpertActivation.BF16, ExpertWeight.FP8_BLOCK128),
    MoeMode.A16W4: MoeFormat(ExpertActivation.BF16, ExpertWeight.MXFP4_BLOCK32),
    MoeMode.A8W4: MoeFormat(ExpertActivation.MXFP8_BLOCK32, ExpertWeight.MXFP4_BLOCK32),
}


def as_moe_mode(value: MoeMode | str) -> MoeMode:
    """Normalize a public mode argument and report supported values clearly."""

    if isinstance(value, MoeMode):
        return value
    try:
        return MoeMode(value)
    except ValueError as error:
        choices = ", ".join(mode.value for mode in MoeMode)
        raise ValueError(
            f"unsupported MoE mode {value!r}; expected one of: {choices}"
        ) from error


def moe_format(value: MoeMode | str) -> MoeFormat:
    """Return the independent activation and weight formats for a public mode."""

    return MOE_FORMATS[as_moe_mode(value)]


# Shared-KV MQA: one head_dim vector per token is both K and V, RoPE in its tail.
HIDDEN = 7168
Q_LORA = 1536
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM  # 448, the FP8-quantized part of the KV row
O_LORA = 1024
O_GROUPS = 2  # local; 16 global / 8 ranks
WINDOW = 128
# ATOM's V4 KV block size; a block holds BLOCK_TOKENS // ratio compressed entries.
BLOCK_TOKENS = 256

N_EXPERTS = 384
TOP_K = 6
MOE_SLOTS = 1 + TOP_K
SHARED_EXPERT = N_EXPERTS  # the shared expert sits last in the bank
INTER = 384  # local; 3072 global / 8 ranks
ROUTE_SCALE = 2.5
SWIGLU_LIMIT = 10.0

HC_MULT = 4
HC_SINKHORN_ITERS = 20
HC_EPS = 1e-6
HC_MIX = (2 + HC_MULT) * HC_MULT  # pre | post | comb, packed in that order

EPS = 1e-6
SCALE_BM = 128
FP8_MAX = 448.0
SOFTMAX_SCALE = HEAD_DIM**-0.5

SUPPORTED_SAMPLES = tuple(range(1, 9))
# Candidate device shapes do not broaden default serving profiles or q limits.
# S14 holds two complete q7 requests; prepare still gates resident resources.
SUPPORTED_LAUNCH_ROWS = (*SUPPORTED_SAMPLES, 14)
LAUNCH_PHASES = {
    "one": ("one",),
    "two": ("attention", "moe"),
    "three": ("attention_pre", "attention_core", "moe"),
}
SUPPORTED_PEERS = (1, 2, 4, 8)
# The split-attention kernel requires heads % WAVES == 0 and heads <= 16.
SUPPORTED_HEADS = (8, 16)
# Launches per step, each with its own epoch tag (61 layers, up to 4 each with MTP).
MAX_LAYERS_PER_STEP = 256

# V4-Pro compress_ratios: [128, 128] + [4, 128] * 29 + [4] + [0] -- layers 0-1 HCA, then
# CSA iff even; no ratio-0 main layer, the trailing 0 is the MTP block.
COMPRESS_SWA = 0  # sliding window only (the MTP block)
COMPRESS_CSA = 4  # CSA, with the lightning indexer
COMPRESS_HCA = 128  # HCA, dense over compressed entries


COMPRESS_ROPE_THETA = 1.6e5
# Lightning indexer (CSA only). Replicated on every rank, as ATOM: each rank scores with all
# 64 heads, so the scores are bit-identical across ranks with no all-reduce.
INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
INDEX_TOPK = 1024
KEY_BLOCK = 64  # the split-attention key tile; the index list is padded to it


def validate_shard(
    samples: int,
    heads: int,
    rank: int,
    npes: int,
    window: int = WINDOW,
    compress_ratio: int = COMPRESS_SWA,
) -> None:
    """Validate the V4 shard contract before allocating GPU buffers."""
    if samples not in SUPPORTED_LAUNCH_ROWS:
        raise ValueError(
            f"samples must be one of {SUPPORTED_LAUNCH_ROWS}, got {samples}"
        )
    if heads not in SUPPORTED_HEADS:
        raise ValueError(f"heads must be one of {SUPPORTED_HEADS}, got {heads}")
    if npes not in SUPPORTED_PEERS:
        raise ValueError(f"npes must be one of {SUPPORTED_PEERS}, got {npes}")
    if not 0 <= rank < npes:
        raise ValueError(f"rank must be in [0, {npes}), got {rank}")
    if window <= 0 or window % 64:
        raise ValueError(f"window must be a positive multiple of 64, got {window}")


@dataclass
class V4Config:
    """One rank's shard of a DeepSeek-V4 layer. Defaults are V4-Pro at TP8."""

    heads: int = 16  # local; 128 global / 8 ranks
    hidden: int = HIDDEN
    q_lora: int = Q_LORA
    head_dim: int = HEAD_DIM  # K and V share this; rope lives in its tail
    rope_dim: int = ROPE_DIM
    o_groups: int = O_GROUPS  # local; 16 global / 8 ranks
    o_lora: int = O_LORA
    n_experts: int = N_EXPERTS
    top_k: int = TOP_K
    inter: int = INTER  # local; 3072 global / 8 ranks
    window: int = WINDOW
    route_scale: float = ROUTE_SCALE
    swiglu_limit: float = SWIGLU_LIMIT
    eps: float = EPS
    rope_theta: float = 1.0e4
    compress_ratio: int = COMPRESS_SWA  # 0 SWA, 4 CSA, 128 HCA
    # Physical pool capacity, independent of a ragged step's live query width.
    # None preserves the ordinary capacity of c_rows + q - 1.
    compressor_ring_rows: int | None = None
    index_heads: int = (
        INDEX_HEADS  # all of them on every rank (the indexer is replicated)
    )
    index_head_dim: int = INDEX_HEAD_DIM
    index_topk: int = INDEX_TOPK
    compress_rope_theta: float = COMPRESS_ROPE_THETA
    max_seq: int = 4096  # sizes the compressed half of the KV cache
    hc_mult: int = HC_MULT  # 1 = plain residual
    hc_sinkhorn_iters: int = HC_SINKHORN_ITERS
    hc_eps: float = HC_EPS
    # KV rows in ATOM's fp8 layout (NoPE + E8M0 plane, bf16 RoPE plane) instead of one bf16 plane
    kv_fp8: bool = False
    # PR's synthetic BF16 oracle simulates checkpoint QAT. ATOM's ordinary
    # BF16 cache instead writes unquantized NoPE (quant_mode="none"). Keep this
    # mathematical profile explicit and independent of the cache's layout.
    bf16_kv_roundtrip: bool = True
    # Query numerics are independently keyed, but the native FP8 two-buffer
    # path group-quantizes Q even when its caller passes quant_q=False (that
    # argument controls only the BF16 path). Draft binding must match dispatch.
    quant_query_fp8: bool = True
    # Additional keys for block-bidirectional draft attention. Zero preserves
    # target/ordinary geometry; a draft owner freezes its actual runtime K.
    draft_block_width: int = 0
    # ATOM's non-fused shared Expert: BF16 normalized input -> E8M0/128,
    # BF16 gate/up -> BF16 activation -> E8M0/128. Routed experts retain
    # their independent MXFP8/32 profile. False preserves the PR oracle.
    shared_bf16_block128: bool = False
    # Ordinary down materializes weighted routes in BF16 before a BF16
    # routed sum and shared add. Opus UG retains FP32 activation before
    # MXFP8/32 (unlike another FlyDSL stage1); do not round routed UG.
    routed_down_bf16: bool = False
    # The ordinary main/index compressor GEMM writes BF16 before gate bias
    # and ring publication; fused pool/norm/RoPE remains FP32 to cache/quant.
    # False retains the imported PR FP32 projection/BF16 pooling seams.
    compressor_projection_bf16: bool = False
    # Ordinary attention: block128 E8M0 FP8 activations, BF16 projection
    # epilogues, and fused FP32 q_norm -> FP8 shared by query/indexer.
    attention_fp8_bf16: bool = False
    # Hadamard-rotate indexer q/k as the model does; ATOM's cache is unrotated, so serving it needs False
    indexer_hadamard: bool = True
    # CSA layers always consume an index selection, but native V4 may only
    # produce a fresh selection every Nth CSA layer.  Reuse layers share the
    # CSA arena with their producer and read ``i_sel`` with that layer's tag.
    # These are execution-policy fields, not geometry: they must not split the
    # variant scratch arena or the cross-layer selection would be lost.
    index_topk_refresh: bool = True
    index_topk_source_layer: int = -1

    @property
    def hc_mix(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult

    @property
    def rope_base(self) -> float:
        """Per layer, not per consumer: a compressing layer rotates everything on ``compress_rope_theta``."""
        return self.compress_rope_theta if self.compress_ratio else self.rope_theta

    @property
    def overlap(self) -> bool:
        """CSA (ratio 4) pools 2 * ratio tokens per entry at a stride of ratio."""
        return self.compress_ratio == COMPRESS_CSA

    @property
    def indexed(self) -> bool:
        """CSA selects compressed entries with the lightning indexer; HCA takes them all."""
        return self.compress_ratio == COMPRESS_CSA

    @property
    def n_index(self) -> int:
        """Compressed entries the attention can reach in one step."""
        if not self.compress_ratio:
            return 0
        return (
            min(self.index_topk, self.n_compressed)
            if self.indexed
            else self.n_compressed
        )

    @property
    def c_coff(self) -> int:
        """Compressor channel multiplier: overlapping windows carry previous and current halves."""
        return 2 if self.overlap else 1

    @property
    def c_rows(self) -> int:
        """Rows of compressor state: two windows' worth when overlapping."""
        return self.compress_ratio * self.c_coff

    def state_ring_rows(self, tokens_per_seq: int) -> int:
        minimum = self.c_rows + tokens_per_seq - 1
        rows = self.compressor_ring_rows
        if rows is None:
            return minimum
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < minimum:
            raise ValueError(f"compressor ring must have at least {minimum} rows")
        return rows

    @property
    def n_compressed(self) -> int:
        """Compressed cache slots, stored after the ``window`` rows of the same cache."""
        return 0 if self.compress_ratio == 0 else self.max_seq // self.compress_ratio

    @property
    def cache_rows(self) -> int:
        return self.window + self.n_compressed

    @property
    def n_keys(self) -> int:
        """Length of the attention's index list (window + compressed), padded to the key tile."""
        if type(self.draft_block_width) is not int or self.draft_block_width < 0:
            raise ValueError("draft block width must be a nonnegative integer")
        if self.draft_block_width and self.compress_ratio:
            raise ValueError("compressed draft key geometry is not qualified")
        rows = self.window + self.n_index + self.draft_block_width
        return (rows + KEY_BLOCK - 1) // KEY_BLOCK * KEY_BLOCK

    @property
    def hc_rows(self) -> int:
        """hc_mix padded to the MFMA row group (24 -> 32 at hc_mult 4)."""
        return (self.hc_mix + 15) // 16 * 16

    @property
    def nope_dim(self) -> int:
        return self.head_dim - self.rope_dim

    @property
    def group_dim(self) -> int:
        """Slice of the concatenated heads that one ``o_a`` group consumes."""
        return self.heads * self.head_dim // self.o_groups

    @property
    def shared_expert(self) -> int:
        """Last in an FP8 bank; an MXFP4 bank keeps it as FP8 ``w_sug`` / ``w_sdn`` beside it."""
        return self.n_experts

    @property
    def softmax_scale(self) -> float:
        return self.head_dim**-0.5

    def validate(self) -> None:
        assert self.compress_ratio == 0 or self.window % self.compress_ratio == 0
        assert self.nope_dim % 64 == 0, "act_quant blocks the nope part by 64"
        assert self.heads % self.o_groups == 0
        assert self.head_dim % 2 == 0 and self.rope_dim % 2 == 0
        assert self.index_topk_refresh or (
            self.indexed and self.index_topk_source_layer >= 0
        ), "an index reuse layer needs a prior refresh-layer source"


def shape_dims(cfg) -> dict:
    """The shape arguments layout(), stage_tasks() and build_dsv4_kernel() must share identically.

    sources.SOURCES explicitly keys the device plans and all traced helpers.
    """
    return {
        "hidden": cfg.hidden,
        "q_lora": cfg.q_lora,
        "head_dim": cfg.head_dim,
        "o_groups": cfg.o_groups,
        "o_lora": cfg.o_lora,
        "hc_mult": cfg.hc_mult,
        "compress_ratio": cfg.compress_ratio,
        "n_keys": cfg.n_keys,
        "c_coff": cfg.c_coff,
        # 0: no indexer (only CSA has one)
        "index_head_dim": cfg.index_head_dim if cfg.indexed else 0,
        "index_heads": cfg.index_heads if cfg.indexed else 0,
        "max_seq": cfg.max_seq,
        "index_topk": cfg.index_topk if cfg.indexed else 0,
        "kv_fp8": cfg.kv_fp8,
        "shared_bf16_block128": cfg.shared_bf16_block128,
        "routed_down_bf16": cfg.routed_down_bf16,
        "compressor_projection_bf16": cfg.compressor_projection_bf16,
        "attention_fp8_bf16": cfg.attention_fp8_bf16,
        "indexer_hadamard": cfg.indexer_hadamard,
        "n_experts": cfg.n_experts,
        "top_k": cfg.top_k,
        "inter": cfg.inter,
    }


def _variant_key(
    cfg,
    samples,
    npes,
    moe_mode,
    timeline,
    poll_timeout_us,
    tokens_per_seq=1,
    launch_shape="one",
    qkv_batch_limit=None,
    qkv_samples_per_pass=None,
    qkv_norm_handoff=False,
    attention_mode="causal",
):
    """Everything the compiled kernel and the scratch layout depend on."""
    return (
        poll_timeout_us,
        tokens_per_seq,
        cfg.state_ring_rows(tokens_per_seq),
        launch_shape,
        qkv_batch_limit,
        qkv_samples_per_pass,
        qkv_norm_handoff,
        tuple(sorted(shape_dims(cfg).items())),
        cfg.bf16_kv_roundtrip,
        cfg.quant_query_fp8,
        samples,
        cfg.heads,
        npes,
        cfg.window,
        cfg.cache_rows,
        cfg.n_experts,
        cfg.top_k,
        cfg.inter,
        cfg.softmax_scale,
        cfg.swiglu_limit,
        cfg.hc_sinkhorn_iters,
        cfg.hc_eps,
        moe_mode,
        timeline,
        attention_mode,
    )
