# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Analytic cost of DeepSeek-V4-Pro at TP8 with DSpark.

One decode step is four pieces, matching ``DeepseekV4DSpark.forward_spec``
and the target verify that follows it:

* draft backbone: ``dspark_num_layers`` V4 blocks, query length = speculative
  block, attention over the rolling target-KV window
* Markov head: one GEMV per block position
* confidence head
* target verify: a full V4-Pro forward at query length ``K + 1``

``K + 1`` is the full CUDA-graph bucket (``ragged_graph_sizes`` equal to the
block). Acceptance does not shrink that forward. It only decides how many
tokens the step emits, and therefore TPOT and throughput.

KV capacity reuses ``UnifiedPoolGeometry``. Weight bytes follow the sharding
in ``DeepseekV4Attention`` / ``MoE`` / ``DSparkLayer``: column- and
row-parallel tensors divide by TP, replicated tensors do not.

With no price list the seconds are an analytic roofline at the machine
spec's peaks (derate defaults to 1). Passing a ``PriceList`` replaces each
GEMM, sparse-attention, compressor and allreduce with its measured time.
Decode composes them with the Phase-1c communication-overlap calibration
measured from real TP8 full-forward steps.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field, replace

from atom.compass.machine import MachineSpec, mi355x
from atom.compass.price_list import PriceList
from atom.model_ops.attentions.v4_pool_geometry import UnifiedPoolGeometry

# tests/test_v4_pool_geometry.py: the 61-layer V4-Pro trunk. The trailing
# dense entry in that test is the draft slot, not a target layer.
V4_PRO_RATIOS: tuple[int, ...] = tuple([128, 128] + [4, 128] * 29 + [4])

_FP8 = "fp8"
_FP4 = "fp4"
_BF16 = "bf16"
_FP32 = "fp32"
_INT32 = "int32"

# Names written into a price list. They select the AITER kernel, not just
# the storage width: FP8 projections are 128-block scaled, FP4 experts are
# packed e2m1 with a scale every 32 elements.
_PRICE_DTYPE = {
    _FP8: "fp8_block128",
    _FP4: "fp4_block32",
    _BF16: "bf16",
    _FP32: "fp32",
}


class _PriceSink:
    """Measured kernel totals, split so communication can overlap compute."""

    def __init__(self, catalog: PriceList):
        self.catalog = catalog
        self.compute_s = 0.0
        self.collective_s = 0.0

    def add(self, dtype: str, m: int, k: int, n: int, count: int) -> None:
        # GEMM dtypes are translated to the measured layout name. Attention,
        # compress and allreduce pass their op name through unchanged.
        key = _PRICE_DTYPE.get(dtype, dtype)
        seconds = self.catalog.seconds(key, m, k, n) * count
        if key == "allreduce":
            self.collective_s += seconds
        else:
            self.compute_s += seconds


_SINK: ContextVar[_PriceSink | None] = ContextVar("compass_price_sink", default=None)


@dataclass(frozen=True)
class V4ProShape:
    """V4-Pro geometry. Defaults match ``DeepseekV4Args`` plus the Pro ratios."""

    dim: int = 7168
    n_layers: int = 61
    n_heads: int = 128
    head_dim: int = 512
    rope_head_dim: int = 64
    q_lora_rank: int = 1536
    o_lora_rank: int = 1024
    o_groups: int = 16
    window_size: int = 128
    vocab_size: int = 129280
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    moe_inter_dim: int = 3072
    n_hash_layers: int = 3
    hc_mult: int = 4
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 1024
    ratios: tuple[int, ...] = V4_PRO_RATIOS
    block_size: int = 256
    # FP8 KV plane widths from AttentionMetadataBuilder: NoPE is ``head_dim``
    # bytes, RoPE is ``rope_head_dim`` bf16 bytes. Indexer row is
    # ``index_head_dim + 4`` on the FP8 index path.
    nope_row_bytes: int = 512
    rope_row_bytes: int = 128
    indexer_row_bytes: int = 132

    def __post_init__(self) -> None:
        if len(self.ratios) != self.n_layers:
            raise ValueError(
                f"compress_ratios has {len(self.ratios)} entries, n_layers is {self.n_layers}"
            )
        unknown = set(self.ratios) - {0, 4, 128}
        if unknown:
            raise ValueError(f"unsupported compress ratios {sorted(unknown)}")

    @property
    def csa_layers(self) -> int:
        return sum(r == 4 for r in self.ratios)

    @property
    def hca_layers(self) -> int:
        return sum(r == 128 for r in self.ratios)

    @property
    def dense_layers(self) -> int:
        return sum(r == 0 for r in self.ratios)


@dataclass(frozen=True)
class DSparkShape:
    num_stages: int = 3
    block_size: int = 7
    markov_rank: int = 512
    # mtp.0.main_proj injects these target layers. Docstring of deepseek_v4_dspark.
    target_layer_ids: tuple[int, ...] = (58, 59, 60)


@dataclass(frozen=True)
class Work:
    flops_fp8: float = 0.0
    flops_fp4: float = 0.0
    flops_bf16: float = 0.0
    flops_fp32: float = 0.0
    hbm_bytes: float = 0.0
    collective_bytes: float = 0.0

    def add(self, other: "Work") -> "Work":
        return Work(
            self.flops_fp8 + other.flops_fp8,
            self.flops_fp4 + other.flops_fp4,
            self.flops_bf16 + other.flops_bf16,
            self.flops_fp32 + other.flops_fp32,
            self.hbm_bytes + other.hbm_bytes,
            self.collective_bytes + other.collective_bytes,
        )


@dataclass(frozen=True)
class StepTime:
    name: str
    tokens_in: int
    device_s: float
    collective_s: float
    seconds: float
    work: Work


@dataclass(frozen=True)
class Estimate:
    """One configuration. ``seconds`` are analytic, not a measured forward."""

    provenance: str
    tp: int
    acceptance_length: float
    tokens_per_decode_step: float
    weight_bytes_per_rank: int
    kv_bytes_per_rank: int
    draft_window_bytes_per_rank: int
    bytes_per_rank: int
    fits: bool
    prefill: StepTime
    draft: StepTime
    markov: StepTime
    confidence: StepTime
    verify: StepTime
    decode_s: float
    tpot_s: float
    output_tokens_per_s: float
    limits: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        lines = [
            f"DeepSeek-V4-Pro TP{self.tp} + DSpark  ({self.provenance})",
            f"per-rank resident {self.bytes_per_rank / 1e9:.2f} GB"
            f"  weights {self.weight_bytes_per_rank / 1e9:.2f} GB"
            f"  kv {self.kv_bytes_per_rank / 1e9:.2f} GB"
            f"  draft window {self.draft_window_bytes_per_rank / 1e9:.2f} GB"
            f"  fits={self.fits}",
            f"prefill {self.prefill.seconds * 1e3:.3f} ms"
            f"  ({self.prefill.tokens_in} tokens)",
            f"decode  {self.decode_s * 1e3:.3f} ms"
            f"  draft {self.draft.seconds * 1e3:.3f}"
            f"  markov {self.markov.seconds * 1e3:.3f}"
            f"  confidence {self.confidence.seconds * 1e3:.3f}"
            f"  verify {self.verify.seconds * 1e3:.3f}",
            f"acceptance length {self.acceptance_length:.3f}"
            f"  TPOT {self.tpot_s * 1e3:.3f} ms"
            f"  output {self.output_tokens_per_s:.1f} tok/s",
            "limits:",
            *(f"  - {item}" for item in self.limits),
        ]
        return "\n".join(lines)


def _elem_bytes(dtype: str, n: int) -> int:
    if dtype == _FP8:
        return n + n // 128  # ue8m0 scale per 128
    if dtype == _FP4:
        return n // 2 + n // 32  # e2m1 packed, ue8m0 per 32
    if dtype == _BF16:
        return n * 2
    if dtype == _FP32 or dtype == _INT32:
        return n * 4
    raise ValueError(dtype)


def _shard(n: int, tp: int, replicated: bool) -> int:
    return n if replicated else n // tp


def _require_tp(shape: V4ProShape, tp: int) -> None:
    if tp < 1:
        raise ValueError(f"tp must be positive, got {tp}")
    for name, value in (
        ("n_heads", shape.n_heads),
        ("o_groups", shape.o_groups),
        ("moe_inter_dim", shape.moe_inter_dim),
    ):
        if value % tp:
            raise ValueError(f"{name}={value} is not divisible by tp={tp}")


def weight_bytes(shape: V4ProShape, draft: DSparkShape, tp: int) -> dict[str, int]:
    """Parameter bytes resident on one rank. Activations are not included."""
    _require_tp(shape, tp)
    dim = shape.dim
    out: dict[str, int] = {}

    def put(name: str, k: int, n: int, dtype: str, replicated: bool, times: int = 1) -> None:
        local_n = n if replicated else n // tp
        # Row-parallel shards K. Callers pass already-local shapes via `replicated`
        # for column-parallel (shard N) and pass replicated=False with the full N.
        out[name] = times * _elem_bytes(dtype, k * local_n)

    # Attention projections. wqkv_a and the indexer are replicated.
    # wq_b / wo_a shard N, wo_b shards K: store wo_b as K/tp by full N.
    qkv_n = shape.q_lora_rank + shape.head_dim
    put("attn.wqkv_a", dim, qkv_n, _FP8, True, shape.n_layers + draft.num_stages)
    wq_b_n = shape.n_heads * shape.head_dim
    put("attn.wq_b", shape.q_lora_rank, wq_b_n, _FP8, False, shape.n_layers + draft.num_stages)
    wo_a_k = shape.n_heads * shape.head_dim // shape.o_groups
    wo_a_n = shape.o_groups * shape.o_lora_rank
    put("attn.wo_a", wo_a_k, wo_a_n, _BF16, False, shape.n_layers + draft.num_stages)
    wo_b_k = wo_a_n // tp
    put("attn.wo_b", wo_b_k, dim, _FP8, True, shape.n_layers + draft.num_stages)

    # Compressor, target CSA/HCA only. CSA overlaps, so coff = 2; HCA coff = 1.
    def compressor(coff: int, layers: int) -> int:
        return layers * _elem_bytes(_BF16, dim * (2 * coff * shape.head_dim))

    out["compressor.csa"] = compressor(2, shape.csa_layers)
    out["compressor.hca"] = compressor(1, shape.hca_layers)

    # Indexer, CSA only, replicated.
    out["indexer.wq_b"] = shape.csa_layers * _elem_bytes(
        _FP8, shape.q_lora_rank * shape.index_n_heads * shape.index_head_dim
    )
    out["indexer.weights_proj"] = shape.csa_layers * _elem_bytes(
        _BF16, dim * shape.index_n_heads
    )

    # Experts. TP shards the intermediate dim. Count routed + shared, FP4.
    inter = shape.moe_inter_dim // tp
    expert = _elem_bytes(_FP4, dim * (2 * inter) + inter * dim)
    n_experts = shape.n_routed_experts + shape.n_shared_experts
    out["moe.experts"] = (shape.n_layers + draft.num_stages) * n_experts * expert
    out["moe.gate"] = (shape.n_layers + draft.num_stages) * _elem_bytes(
        _BF16, dim * shape.n_routed_experts
    )
    out["moe.tid2eid"] = shape.n_hash_layers * _elem_bytes(
        _INT32, shape.vocab_size * shape.n_activated_experts
    )

    # mHC, fp32, replicated. mix_hc x (hc_mult * dim), attn and ffn.
    mix = (2 + shape.hc_mult) * shape.hc_mult
    hc_dim = shape.hc_mult * dim
    out["mhc"] = (shape.n_layers + draft.num_stages) * 2 * _elem_bytes(_FP32, mix * hc_dim)

    # Embed and lm_head, vocab-parallel, not tied. bf16.
    out["embed_and_head"] = 2 * _elem_bytes(_BF16, (shape.vocab_size // tp) * dim)

    # DSpark extras, replicated.
    out["dspark.main_proj"] = _elem_bytes(
        _FP8, dim * len(draft.target_layer_ids) * dim
    )
    # Both Markov tables are nn.Embedding. Count fp32, which is the module default.
    out["dspark.markov"] = 2 * _elem_bytes(_FP32, shape.vocab_size * draft.markov_rank)
    out["dspark.confidence"] = _elem_bytes(_BF16, (dim + draft.markov_rank) * 1)
    out["dspark.hc_head"] = _elem_bytes(_FP32, shape.hc_mult * hc_dim)
    return out


def kv_bytes(
    shape: V4ProShape,
    *,
    num_blocks: int,
    num_slots: int,
    num_speculative_tokens: int,
) -> dict[str, int]:
    """Bytes of the target KV pool on one rank.

    MQA keeps a single KV head, replicated across TP. ``num_blocks`` and
    ``num_slots`` are the pool's own counts; this does not decide admission.
    """
    if num_speculative_tokens < 0:
        raise ValueError("num_speculative_tokens must be >= 0")
    ring = shape.window_size + num_speculative_tokens
    geo = UnifiedPoolGeometry(
        list(shape.ratios),
        num_blocks=num_blocks,
        num_slots=num_slots,
        ring_slots=ring,
        block_size=shape.block_size,
    )
    row = shape.nope_row_bytes + shape.rope_row_bytes
    # Compressor state is fp32 and lives in the slot, beside the window rows.
    # Shapes follow AttentionMetadataBuilder: CSA ring 8+K, HCA ring 128+K.
    k = num_speculative_tokens
    csa_state = (8 + k) * (2 * shape.head_dim) * 4
    csa_idx_state = (8 + k) * (2 * shape.index_head_dim) * 4
    hca_state = (128 + k) * shape.head_dim * 4
    state_one = (
        shape.csa_layers * 2 * (csa_state + csa_idx_state)
        + shape.hca_layers * 2 * hca_state
    )
    indexer_one_block = (
        shape.csa_layers * (shape.block_size // 4) * shape.indexer_row_bytes
    )
    return {
        "pool": geo.plane_bytes(row),
        "compressor_state": num_slots * state_one,
        "indexer": num_blocks * indexer_one_block,
    }


def draft_window_bytes(
    shape: V4ProShape, draft: DSparkShape, num_slots: int, num_speculative_tokens: int
) -> int:
    """Per-rank bytes of the DSpark rolling windows.

    One window per stage per sequence. Width is ``window_size + K``, the
    same ``write_per_batch`` the draft layer allocates. FP8 KV uses the
    target's two plane widths.
    """
    rows = shape.window_size + num_speculative_tokens
    row_bytes = shape.nope_row_bytes + shape.rope_row_bytes
    return num_slots * draft.num_stages * rows * row_bytes


def _gemm(m: int, k: int, n: int, weight: str, act: str = _BF16, count: int = 1) -> Work:
    """One GEMM on this rank, issued ``count`` times. ``m`` is the token count."""
    if m <= 0 or k <= 0 or n <= 0 or count <= 0:
        return Work()
    sink = _SINK.get()
    if sink is not None:
        sink.add(weight, m, k, n, count)
        return Work()
    flops = 2.0 * m * k * n * count
    weight_bytes = float(_elem_bytes(weight, k * n)) * count
    act_bytes = float(_elem_bytes(act, m * k + m * n)) * count
    moved = weight_bytes + act_bytes
    if weight == _FP4:
        return Work(flops_fp4=flops, hbm_bytes=moved)
    if weight == _FP8:
        return Work(flops_fp8=flops, hbm_bytes=moved)
    if weight == _BF16:
        return Work(flops_bf16=flops, hbm_bytes=moved)
    if weight == _FP32:
        return Work(flops_fp32=flops, hbm_bytes=moved)
    raise ValueError(weight)


def _attention(
    q: int,
    keys: int,
    heads: int,
    head_dim: int,
    row_bytes: int,
    *,
    prefill: bool,
) -> Work:
    """MQA: ``heads`` local heads, ``q`` queries, each attending ``keys`` KV rows.

    A price list replaces this with the measured paged kernel. Decode is
    ``mla_decode_fwd_v4_nm``; prefill is ``pa_sparse_prefill_fp8_opus``.
    """
    if q <= 0 or keys <= 0:
        return Work()
    if _SINK.get() is not None:
        _SINK.get().add("attn_prefill" if prefill else "attn_decode", q, keys, heads, 1)
        return Work()
    flops = 4.0 * q * heads * head_dim * keys  # QK + AV
    moved = float(q * keys * row_bytes)
    return Work(flops_bf16=flops, hbm_bytes=moved)


def _allreduce(tokens: int, dim: int, times: int, world: int) -> Work:
    if tokens <= 0 or times <= 0:
        return Work()
    if _SINK.get() is not None:
        _SINK.get().add("allreduce", tokens, dim, world, times)
        return Work()
    return Work(collective_bytes=float(times * tokens * dim * 2))


def _compress(tokens: int, ratio: int, queries_per_seq: int) -> Work:
    """One fused compressor launch. Only the measured kernel has a time."""
    if tokens <= 0 or _SINK.get() is None:
        return Work()
    _SINK.get().add("compress", tokens, ratio, queries_per_seq, 1)
    return Work()


def _layer_proj(
    shape: V4ProShape, tp: int, tokens: int, *, compressor: int, indexer: bool
) -> Work:
    """Projections for one layer. ``compressor`` is 0, 1 (HCA) or 2 (CSA coff)."""
    dim = shape.dim
    work = _gemm(tokens, dim, shape.q_lora_rank + shape.head_dim, _FP8)
    work = work.add(
        _gemm(tokens, shape.q_lora_rank, shape.n_heads * shape.head_dim // tp, _FP8)
    )
    wo_a_k = shape.n_heads * shape.head_dim // shape.o_groups
    wo_a_n = shape.o_groups * shape.o_lora_rank // tp
    work = work.add(_gemm(tokens, wo_a_k, wo_a_n, _BF16))
    work = work.add(_gemm(tokens, wo_a_n, dim, _FP8))  # K already sharded
    if compressor:
        work = work.add(_gemm(tokens, dim, 2 * compressor * shape.head_dim, _BF16))
    if indexer:
        work = work.add(
            _gemm(
                tokens,
                shape.q_lora_rank,
                shape.index_n_heads * shape.index_head_dim,
                _FP8,
            )
        )
        work = work.add(_gemm(tokens, dim, shape.index_n_heads, _BF16))
    # MoE: each token runs top-k routed experts plus the shared expert.
    # The grouped GEMM streams those experts; the per-rank cost is that many
    # sharded GEMMs, not one expert loaded once.
    active = shape.n_activated_experts + shape.n_shared_experts
    inter = shape.moe_inter_dim // tp
    # One grouped launch per activated expert (top-k routed plus the shared one).
    work = work.add(_gemm(tokens, dim, 2 * inter, _FP4, count=active))
    work = work.add(_gemm(tokens, inter, dim, _FP4, count=active))
    work = work.add(_gemm(tokens, dim, shape.n_routed_experts, _BF16))
    # mHC pre, twice (attn and ffn). The parameter is small; activations dominate.
    mix = (2 + shape.hc_mult) * shape.hc_mult
    hc_dim = shape.hc_mult * dim
    work = work.add(_gemm(tokens, hc_dim, mix, _FP32))
    work = work.add(_gemm(tokens, hc_dim, mix, _FP32))
    # wo_b and the MoE down-proj each all-reduce the hidden state.
    work = work.add(_allreduce(tokens, dim, 2, tp))
    return work


def _target_forward(
    shape: V4ProShape,
    tp: int,
    tokens: int,
    queries_per_seq: int,
    context: int,
    *,
    prefill: bool,
) -> Work:
    """A V4-Pro forward. ``tokens`` is the GEMM token count (batch * query)."""
    if queries_per_seq <= 0:
        raise ValueError("queries_per_seq must be positive")
    work = Work()
    row = shape.nope_row_bytes + shape.rope_row_bytes
    heads = shape.n_heads // tp
    csa_keys = min(shape.index_topk, max(context // 4, 1))
    hca_keys = max(context // 128, 1)
    swa_keys = min(shape.window_size, context)
    for ratio in shape.ratios:
        work = work.add(
            _layer_proj(
                shape,
                tp,
                tokens,
                compressor=2 if ratio == 4 else (1 if ratio == 128 else 0),
                indexer=ratio == 4,
            )
        )
        keys = csa_keys if ratio == 4 else (hca_keys if ratio == 128 else swa_keys)
        work = work.add(
            _attention(tokens, keys, heads, shape.head_dim, row, prefill=prefill)
        )
        if ratio in (4, 128):
            work = work.add(_compress(tokens, ratio, queries_per_seq))
    # Vocab-parallel head.
    work = work.add(_gemm(tokens, shape.dim, shape.vocab_size // tp, _BF16))
    return work


def _draft_forward(
    shape: V4ProShape, draft: DSparkShape, tp: int, batch: int, num_draft: int
) -> Work:
    tokens = batch * num_draft
    work = Work()
    for _ in range(draft.num_stages):
        work = work.add(
            _layer_proj(shape, tp, tokens, compressor=0, indexer=False)
        )
        work = work.add(
            _attention(
                tokens,
                shape.window_size,
                shape.n_heads // tp,
                shape.head_dim,
                shape.nope_row_bytes + shape.rope_row_bytes,
                prefill=False,
            )
        )
    work = work.add(
        _gemm(batch, shape.dim * len(draft.target_layer_ids), shape.dim, _FP8)
    )
    return work


def _markov(shape: V4ProShape, draft: DSparkShape, batch: int, num_draft: int) -> Work:
    # Per position: bias = W1[token] @ W2^T, W2 is [V, r]. Read W2 once per
    # position (the fused sampler does), and do the GEMV per sequence.
    positions = num_draft
    gemv = _gemm(batch * positions, draft.markov_rank, shape.vocab_size, _FP32)
    if _SINK.get() is not None:
        return Work()
    # The weight traffic above is per sequence. Replace it with one read of W2
    # per position plus the per-sequence activation.
    w2 = float(positions * _elem_bytes(_FP32, shape.vocab_size * draft.markov_rank))
    act = float(_elem_bytes(_FP32, batch * positions * (draft.markov_rank + shape.vocab_size)))
    return Work(flops_fp32=gemv.flops_fp32, hbm_bytes=w2 + act)


def _confidence(shape: V4ProShape, draft: DSparkShape, batch: int, num_draft: int) -> Work:
    return _gemm(batch * num_draft, shape.dim + draft.markov_rank, 1, _BF16)


def gemm_shapes(shape: V4ProShape, draft: DSparkShape, tp: int) -> list[tuple[str, int, int]]:
    """Distinct ``(price dtype, K, N)`` a TP8 step multiplies, M aside.

    The measurement campaign crosses this list with the token counts it is
    willing to price. ``M`` is the only axis that changes with the batch.
    """
    _require_tp(shape, tp)
    dim = shape.dim
    wo_a_k = shape.n_heads * shape.head_dim // shape.o_groups
    wo_a_n = shape.o_groups * shape.o_lora_rank // tp
    inter = shape.moe_inter_dim // tp
    mix = (2 + shape.hc_mult) * shape.hc_mult
    shapes = [
        (_PRICE_DTYPE[_FP8], dim, shape.q_lora_rank + shape.head_dim),
        (_PRICE_DTYPE[_FP8], shape.q_lora_rank, shape.n_heads * shape.head_dim // tp),
        (_PRICE_DTYPE[_BF16], wo_a_k, wo_a_n),
        (_PRICE_DTYPE[_FP8], wo_a_n, dim),
        (_PRICE_DTYPE[_BF16], dim, 2 * 2 * shape.head_dim),
        (_PRICE_DTYPE[_BF16], dim, 2 * 1 * shape.head_dim),
        (_PRICE_DTYPE[_FP8], shape.q_lora_rank, shape.index_n_heads * shape.index_head_dim),
        (_PRICE_DTYPE[_BF16], dim, shape.index_n_heads),
        (_PRICE_DTYPE[_FP4], dim, 2 * inter),
        (_PRICE_DTYPE[_FP4], inter, dim),
        (_PRICE_DTYPE[_BF16], dim, shape.n_routed_experts),
        (_PRICE_DTYPE[_FP32], shape.hc_mult * dim, mix),
        (_PRICE_DTYPE[_BF16], dim, shape.vocab_size // tp),
        (_PRICE_DTYPE[_FP8], dim * len(draft.target_layer_ids), dim),
        (_PRICE_DTYPE[_FP32], draft.markov_rank, shape.vocab_size),
        (_PRICE_DTYPE[_BF16], dim + draft.markov_rank, 1),
    ]
    unique: list[tuple[str, int, int]] = []
    seen: set[tuple[str, int, int]] = set()
    for item in shapes:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def attention_keys(context: int, ratio: int, window: int, index_topk: int) -> int:
    """KV rows one query attends, matching ``_target_forward``."""
    if ratio == 4:
        return min(index_topk, max(context // 4, 1))
    if ratio == 128:
        return max(context // 128, 1)
    return min(window, context)


def kernel_shapes(
    shape: V4ProShape,
    draft: DSparkShape,
    tp: int,
    *,
    batch: int,
    prefill_tokens: int,
    context: int,
    num_speculative_tokens: int,
) -> list[tuple[str, int, int, int]]:
    """``(op, M, K, N)`` lookups one priced step makes besides GEMMs.

    Attention ``N`` is the local head count. Compress ``N`` is the query length
    of one sequence. Allreduce ``N`` is the tensor-parallel world size.
    """
    _require_tp(shape, tp)
    heads = shape.n_heads // tp
    verify_q = num_speculative_tokens + 1
    keys: list[tuple[str, int, int, int]] = []

    def add(op: str, m: int, k: int, n: int) -> None:
        item = (op, m, k, n)
        if item not in keys:
            keys.append(item)

    for ratio in set(shape.ratios):
        if ratio == 0:
            continue
        k = attention_keys(prefill_tokens, ratio, shape.window_size, shape.index_topk)
        add("attn_prefill", prefill_tokens, k, heads)
        add("compress", prefill_tokens, ratio, prefill_tokens)
        k = attention_keys(context, ratio, shape.window_size, shape.index_topk)
        add("attn_decode", batch * verify_q, k, heads)
        add("compress", batch * verify_q, ratio, verify_q)
    add("attn_decode", batch * num_speculative_tokens, shape.window_size, heads)
    for tokens in (
        prefill_tokens,
        batch * verify_q,
        batch * num_speculative_tokens,
    ):
        add("allreduce", tokens, shape.dim, tp)
    return keys


def _seconds(work: Work, machine: MachineSpec) -> tuple[float, float]:
    fp8 = work.flops_fp8 / (machine.fp8_flops * machine.compute_derate)
    fp4 = work.flops_fp4 / (machine.fp4_flops * machine.compute_derate)
    bf16 = work.flops_bf16 / (machine.bf16_flops * machine.compute_derate)
    fp32 = work.flops_fp32 / (machine.fp32_flops * machine.compute_derate)
    hbm = work.hbm_bytes / (machine.hbm_bandwidth_bytes_per_s * machine.memory_derate)
    device = max(fp8, fp4, bf16, fp32, hbm)
    collective = work.collective_bytes / (
        machine.interconnect_bandwidth_bytes_per_s * machine.interconnect_derate
    )
    return device, collective


def _step(
    name: str,
    tokens: int,
    work: Work,
    machine: MachineSpec,
    measured_compute_s: float = 0.0,
    measured_collective_s: float = 0.0,
) -> StepTime:
    device, collective = _seconds(work, machine)
    device += measured_compute_s
    collective += measured_collective_s
    return StepTime(
        name, tokens, device, collective, device + collective, work
    )


def _collective_overlap_fraction(batch: int) -> float:
    """gfx950 TP8 decode overlap calibrated from real full-forward steps.

    Phase-1c points:
      B=1: 25.52 ms isolated sum -> 23.60 ms real (32.5% comm overlap)
      B=8: 28.02 ms isolated sum -> 23.41 ms real (57.5% comm overlap)

    B=2..7 interpolate between the two measured graph buckets. This correction
    is decode-only: prefill is chunked by the real scheduler and needs a
    separate step model.
    """
    if not 1 <= batch <= 8:
        raise ValueError(
            f"measured TP8 overlap calibration covers batch 1..8, got {batch}"
        )
    return 0.325 + (batch - 1) * (0.575 - 0.325) / 7


def estimate(
    *,
    batch: int,
    prefill_tokens: int,
    num_speculative_tokens: int = 7,
    acceptance_length: float,
    num_slots: int,
    num_blocks: int,
    context: int = 4096,
    tp: int = 8,
    machine: MachineSpec | None = None,
    shape: V4ProShape | None = None,
    draft: DSparkShape | None = None,
    memory_utilization: float = 0.9,
    price_list: PriceList | None = None,
) -> Estimate:
    """Predict one prefill step and one DSpark decode step.

    ``acceptance_length`` counts the target token, so it lies in
    ``[1, num_speculative_tokens + 1]``. ``context`` is the KV length a verify
    query attends over, in tokens, before compression.
    """
    if batch < 1:
        raise ValueError(f"batch must be positive, got {batch}")
    if prefill_tokens < 1:
        raise ValueError(f"prefill_tokens must be positive, got {prefill_tokens}")
    if num_speculative_tokens < 1:
        raise ValueError("num_speculative_tokens must be positive")
    if num_slots < 1 or num_blocks < 1:
        raise ValueError("num_slots and num_blocks must be positive")
    if context < 1:
        raise ValueError("context must be positive")
    hi = num_speculative_tokens + 1
    if not 1 <= acceptance_length <= hi:
        raise ValueError(
            f"acceptance_length must be in [1, {hi}], got {acceptance_length}"
        )
    if not 0 < memory_utilization <= 1:
        raise ValueError("memory_utilization must be in (0, 1]")

    machine = machine or mi355x()
    shape = shape or V4ProShape()
    draft = draft or DSparkShape(block_size=num_speculative_tokens)
    if draft.block_size < num_speculative_tokens:
        raise ValueError(
            f"dspark block_size {draft.block_size} < num_speculative_tokens "
            f"{num_speculative_tokens}"
        )
    _require_tp(shape, tp)

    weights = weight_bytes(shape, draft, tp)
    kv = kv_bytes(
        shape,
        num_blocks=num_blocks,
        num_slots=num_slots,
        num_speculative_tokens=num_speculative_tokens,
    )
    window = draft_window_bytes(shape, draft, num_slots, num_speculative_tokens)
    resident = sum(weights.values()) + sum(kv.values()) + window
    fits = resident <= int(machine.memory_bytes * memory_utilization)

    verify_q = num_speculative_tokens + 1
    sink = _PriceSink(price_list) if price_list is not None else None
    token = _SINK.set(sink)

    def _take(name: str, tokens_in: int, work: Work) -> StepTime:
        measured_compute_s = 0.0
        measured_collective_s = 0.0
        if sink is not None:
            measured_compute_s = sink.compute_s
            measured_collective_s = sink.collective_s
            sink.compute_s = 0.0
            sink.collective_s = 0.0
        return _step(
            name,
            tokens_in,
            work,
            machine,
            measured_compute_s,
            measured_collective_s,
        )

    try:
        prefill_work = _target_forward(
            shape, tp, prefill_tokens, prefill_tokens, prefill_tokens, prefill=True
        )
        prefill = _take("prefill", prefill_tokens, prefill_work)
        draft_work = _draft_forward(shape, draft, tp, batch, num_speculative_tokens)
        draft_t = _take("draft", batch * num_speculative_tokens, draft_work)
        markov_work = _markov(shape, draft, batch, num_speculative_tokens)
        markov_t = _take("markov", batch * num_speculative_tokens, markov_work)
        conf_work = _confidence(shape, draft, batch, num_speculative_tokens)
        conf_t = _take("confidence", batch * num_speculative_tokens, conf_work)
        verify_work = _target_forward(
            shape, tp, batch * verify_q, verify_q, context, prefill=False
        )
        verify_t = _take("verify", batch * verify_q, verify_work)
    finally:
        _SINK.reset(token)

    if price_list is not None:
        overlap = _collective_overlap_fraction(batch)

        def _overlap(step: StepTime) -> StepTime:
            return replace(
                step,
                seconds=step.device_s + step.collective_s * (1.0 - overlap),
            )

        draft_t = _overlap(draft_t)
        verify_t = _overlap(verify_t)
    decode_s = draft_t.seconds + markov_t.seconds + conf_t.seconds + verify_t.seconds

    if price_list is None:
        provenance = "analytical"
        roofline_limit = (
            "analytic roofline at the machine-spec peaks; derate 1 is not achieved kernel time"
        )
    else:
        provenance = "mixed"
        roofline_limit = (
            f"GEMM, attention, compress and allreduce prices are {price_list.method} "
            f"from {price_list.device}; a kernel measured alone is not the same "
            "kernel inside the step"
        )
    limits = (
        roofline_limit,
        (
            f"decode uses a Phase-1c TP8 overlap calibration "
            f"({_collective_overlap_fraction(batch):.1%} of measured allreduce hidden)"
            if price_list is not None
            else "no kernel-launch floor and no credit for main/alt/compress overlap"
        ),
        (
            "allreduce is measured with synchronized 8-rank RCCL windows"
            if price_list is not None
            else "collectives are bytes/injection-bandwidth, with no link latency"
        ),
        "verify GEMMs are priced at q=K+1, the full graph bucket",
        "acceptance_length is an input; the confidence head is not executed",
        "one prefill step holds the whole prompt; chunked prefill is not split",
        "EPLB, MegaMoE and PD transfer are not in this estimate",
    )
    return Estimate(
        provenance=provenance,
        tp=tp,
        acceptance_length=acceptance_length,
        tokens_per_decode_step=acceptance_length,
        weight_bytes_per_rank=sum(weights.values()),
        kv_bytes_per_rank=sum(kv.values()),
        draft_window_bytes_per_rank=window,
        bytes_per_rank=resident,
        fits=fits,
        prefill=prefill,
        draft=draft_t,
        markov=markov_t,
        confidence=conf_t,
        verify=verify_t,
        decode_s=decode_s,
        tpot_s=decode_s / acceptance_length,
        output_tokens_per_s=batch * acceptance_length / decode_s,
        limits=limits,
    )
