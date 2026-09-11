# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4.1-Flash, text-only.

Reference implementation: ``/data/DeepSeek-V4.1-Flash/inference/model.py``.
Design contract: ``docs/deepseek_v41_design.md``.

DeepSeek-V4 supplies the pieces whose semantics are unchanged -- YaRN RoPE,
the MoE stack, the LM head, the quant-config plumbing and the mHC primitives.
V4.1 brings its own compressor (no ``ape``, no overlap, ratio 1 and 2), an
indexer keyed off the compressor latent with a two-level candidate pool, the
engram memory layers and a one-block shift of the mHC coefficients.

The forward path here is the eager reference: one sequence per call, dense
caches, no paging and no fused attention kernel. It exists to be numerically
right; the paged/kernel bring-up replaces it behind the ``_V41_*`` switches.
"""

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, ClassVar

import aiter
import torch
import torch.nn.functional as F
from aiter import QuantType, dtypes
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from aiter.jit.utils.chip_info import get_gfx
from torch import nn

from atom.config import Config, LayerQuantConfig
from atom.model_loader.loader import WeightsMapper
from atom.model_ops.attentions.pool_layout.v4_pool_geometry import owner_layers
from atom.model_ops.communication_op import tensor_model_parallel_all_reduce
from atom.model_ops.embed_head import VocabParallelEmbedding
from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.model_ops.sparse_attn_v4 import hc_split_sinkhorn
from atom.model_ops.utils import atom_parameter
from atom.model_ops.v4_kernels import (
    csa_translate_pack,
    sparse_attn_v4_paged_decode,
    sparse_attn_v4_paged_prefill,
    swa_write,
)
from atom.models.deepseek_v4 import (
    Block as _V4Block,
)
from atom.models.deepseek_v4 import (
    DeepseekV4Args,
    DeepseekV4ForCausalLM,
    MoE,
    ParallelHead,
    _dequant_fp8_block_to_bf16,
    _V4RoPE,
    make_v4_quant_config,
)
from atom.models.deepseek_v41_engram import EngramLayout, NgramHashState
from atom.utils.forward_context import AttnState, get_forward_context

logger = logging.getLogger(__name__)

# --- switches for paths that are deliberately not the fast one yet ---------
# Round-trip the indexer q/k through MXFP4 the way the reference does. Off for
# first bring-up; the selection is close enough without it to debug the rest.
_V41_INDEXER_MXFP4 = os.environ.get("ATOM_V41_INDEXER_MXFP4", "0") == "1"
# The other two QAT round-trips the reference bakes into its kernels: the
# window KV through fp8 and the compressed latent through fp4. Needed to match
# the reference bit for bit, not needed to bring the model up.
_V41_SIM_QAT_QUANT = os.environ.get("ATOM_V41_SIM_QAT_QUANT", "0") == "1"
# Prefer the AITER engram ops over the torch fallback when they are importable.
_V41_ENGRAM_AITER = os.environ.get("ATOM_V41_ENGRAM_AITER", "1") == "1"
# The fused engram gate op is still being written and its signature is not
# fixed, so it stays off until it can be checked against the torch fallback.
_V41_ENGRAM_FUSED_GATE = os.environ.get("ATOM_V41_ENGRAM_FUSED_GATE", "0") == "1"
# Where a fused mHC coefficient kernel would go. None exists yet: aiter's
# `mhc_pre`, `mhc` and `mhc_post_pre` all return the collapsed layer input with
# their OWN `pre` already applied, and V4.1 needs the previous sub-layer's.
_V41_MHC_FUSED_COEFFS = os.environ.get("ATOM_V41_MHC_FUSED_COEFFS", "0") == "1"

# The eager reference path: one contiguous sequence per call, dense caches,
# no paging. It is the numerical reference the paged path is checked against,
# and it cannot run under the engine (it asserts a single sequence), so the
# default is the paged one.
_V41_EAGER_ATTN = os.environ.get("ATOM_V41_EAGER_ATTN", "0") == "1"

_FP4_BLOCK = 32
_FP8_BLOCK = 32
# The compressed latent quantizes in groups of 16 with an e4m3 scale, unlike
# everything else, which uses 32 with a power-of-two one.
_COMPRESS_FP4_BLOCK = 16
# One e8m0 scale per 32 table columns, which is the layout on disk
# (`engram.embed.scale` is [rows, head_dim // 32]).
_ENGRAM_SCALE_BLOCK = 32
# E2M1 magnitudes, in order of the 3-bit mantissa/exponent code.
_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DeepseekV41Args(DeepseekV4Args):
    """V4.1 shapes on top of V4's. Field names are the text_config JSON keys."""

    kv_source_layer_ids: tuple[int, ...] = field(default_factory=tuple)
    index_source_layer_ids: tuple[int, ...] = field(default_factory=tuple)
    candidate_source_layer_id: int = -1
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 0
    engram_layer_ids: tuple[int, ...] = field(default_factory=tuple)
    engram_num_embeddings: tuple[int, ...] = field(default_factory=tuple)
    engram_max_ngram_size: int = 1
    engram_vocab_size: int = 0
    engram_n_heads: int = 0
    engram_head_dim: int = 0
    engram_pad_token_id: int = 2
    engram_compressed_vocab_size: int = 0
    # Dense fp8 block shape on disk. V4 is [128, 128]; V4.1 is [32, 32].
    weight_block_size: int = 32

    @classmethod
    def from_hf_config(cls, hf_config: Any) -> "DeepseekV41Args":
        args = super().from_hf_config(hf_config)

        def g(key, default=None):
            return getattr(hf_config, key, default)

        # V4-only routing modes that V4.1 does not have.
        args.n_hash_layers = 0
        args.n_mtp_layers = 0
        args.index_topk_freq = 1
        args.index_topk_pattern = None
        args.use_index_cache = False
        # The config list carries the MTP layers as well; they are out of scope.
        args.compress_ratios = tuple(g("compress_ratios", ()))[: args.n_layers]
        args.kv_source_layer_ids = tuple(g("kv_source_layer_ids", ()))
        args.index_source_layer_ids = tuple(g("index_source_layer_ids", ()))
        args.candidate_source_layer_id = int(g("candidate_source_layer_id", -1))
        args.candidate_topk_blocks = int(g("candidate_topk_blocks", 0))
        args.candidate_block_size = int(g("candidate_block_size", 0))
        args.engram_layer_ids = tuple(g("engram_layer_ids", ()))
        args.engram_num_embeddings = tuple(g("engram_num_embeddings", ()))
        args.engram_max_ngram_size = int(g("engram_max_ngram_size", 1))
        args.engram_vocab_size = int(g("engram_vocab_size", 0))
        args.engram_n_heads = int(g("engram_n_heads", 0))
        args.engram_head_dim = int(g("engram_head_dim", 0))
        args.engram_pad_token_id = int(g("engram_pad_token_id", 2))
        args.engram_compressed_vocab_size = int(g("engram_compressed_vocab_size", 0))
        quant = g("quantization_config", None) or {}
        block = quant.get("weight_block_size") if isinstance(quant, dict) else None
        args.weight_block_size = int(block[0]) if block else 128
        args.expert_dtype = (
            quant.get("expert_dtype") if isinstance(quant, dict) else None
        )
        return args


@dataclass(frozen=True)
class V41LayerMap:
    """Which layer produces the compressed KV, the index keys and the candidates.

    A layer whose ratio is non-zero reads the compressed KV of the nearest
    preceding kv source and the top-k of the nearest preceding index source.
    """

    ratios: tuple[int, ...]
    kv_sources: tuple[int, ...]
    index_sources: tuple[int, ...]
    candidate_source: int
    kv_owner: tuple[int | None, ...]
    index_owner: tuple[int | None, ...]

    @classmethod
    def from_args(cls, args: DeepseekV41Args) -> "V41LayerMap":
        """Both owner tables come from `owner_layers`, which is also what the
        row space builds its `compress_bias` from -- one derivation, one
        sentinel (`None` for a layer that keeps no compressed KV)."""
        ratios = tuple(args.compress_ratios)
        n = len(ratios)
        kv_sources = tuple(i for i in args.kv_source_layer_ids if i < n)
        index_sources = tuple(i for i in args.index_source_layer_ids if i < n)
        missing = set(kv_sources) - set(index_sources)
        if missing:
            raise ValueError(
                f"kv sources {sorted(missing)} are not index sources, so nothing "
                "would ever publish their index keys"
            )
        return cls(
            ratios=ratios,
            kv_sources=kv_sources,
            index_sources=index_sources,
            candidate_source=int(args.candidate_source_layer_id),
            kv_owner=owner_layers(ratios, kv_sources),
            index_owner=owner_layers(ratios, index_sources),
        )

    def kv_owner_of(self, layer_id: int) -> int:
        """The layer whose compressed KV `layer_id` reads. Dense layers raise:
        a `-1` reaching an index expression selects the last element."""
        owner = self.kv_owner[layer_id]
        if owner is None:
            raise ValueError(f"layer {layer_id} is dense and reads no compressed KV")
        return owner


class V41SharedRuntime:
    """What an attention layer hands to the ones after it.

    Layers run in order and every source writes before its consumers read, so
    one slot each is enough. Mirrors the reference's ``SharedAttentionRuntime``.
    """

    def __init__(self) -> None:
        self.compress_kv: torch.Tensor | None = None
        self.index_k: torch.Tensor | None = None
        self.topk_idxs: torch.Tensor | None = None
        self.candidates: torch.Tensor | None = None
        # Paged path only. `dummy` is warmup, which runs before the pools
        # exist and wants neither path.
        self.batch: V41Batch | None = None
        self.dummy = False
        self.topk_local: torch.Tensor | None = None
        self.block_candidates: dict[int, torch.Tensor] = {}


class V41Batch:
    """The per-forward facts every attention layer of the step needs.

    Built once, in the model, so the two per-sequence spans it reads back off
    the device are read back once and not forty times.
    """

    __slots__ = (
        "attn_md",
        "block_tables",
        "bs",
        "first_pos",
        "is_decode",
        "last_pos",
        "positions",
        "spans",
    )

    def __init__(self, attn_md, positions: torch.Tensor):
        self.attn_md = attn_md
        self.is_decode = attn_md.state is AttnState.DECODE
        self.positions = positions
        self.block_tables = attn_md.block_tables
        self.bs = int(attn_md.state_slot_out.shape[0])
        cu = attn_md.cu_seqlens_q[: self.bs + 1].tolist()
        self.spans = [(int(cu[b]), int(cu[b + 1])) for b in range(self.bs)]
        ends = torch.tensor(
            [hi - 1 for _, hi in self.spans if hi > 0],
            dtype=torch.long,
            device=positions.device,
        )
        heads = torch.tensor(
            [lo for lo, hi in self.spans if hi > lo],
            dtype=torch.long,
            device=positions.device,
        )
        if ends.numel():
            edge = torch.stack(
                (
                    positions.index_select(0, heads),
                    positions.index_select(0, ends),
                )
            ).tolist()
        else:
            edge = [[], []]
        first, last = iter(edge[0]), iter(edge[1])
        self.first_pos = [next(first) if hi > lo else -1 for lo, hi in self.spans]
        self.last_pos = [next(last) if hi > lo else -1 for lo, hi in self.spans]


def _index_scores(
    q: torch.Tensor, keys: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """``[n, C]`` indexer logits: rectified per head, then combined.

    One head at a time rather than one einsum: the joint form materializes
    ``[n, heads, C]``, which at a long context is the largest tensor in the
    forward by an order of magnitude.
    """
    out = None
    for h in range(q.shape[1]):
        part = torch.mm(q[:, h], keys.t()).relu_().mul_(weights[:, h : h + 1])
        out = part if out is None else out.add_(part)
    return out


# ---------------------------------------------------------------------------
# Small numeric helpers (the eager reference path)
# ---------------------------------------------------------------------------


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm in fp32, exactly as the reference does it."""
    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (weight.float() * xf).to(dtype)


def _rope_(x: torch.Tensor, freqs: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate the last dim of ``x`` in place, adjacent elements as one complex.

    ``x`` is ``[num_tokens, ..., rotary_dim]`` and ``freqs`` is
    ``[num_tokens, rotary_dim // 2]`` complex.
    """
    out = x
    # .float() on an fp32 slice hands back the slice, and view_as_complex needs
    # even strides all the way up, so make the copy explicit.
    xc = torch.view_as_complex(x.float().contiguous().unflatten(-1, (-1, 2)))
    if inverse:
        freqs = freqs.conj()
    while freqs.ndim < xc.ndim:
        freqs = freqs.unsqueeze(-2)
    out.copy_(torch.view_as_real(xc * freqs).flatten(-2))
    return out


def _to_e2m1(mag: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Round magnitudes onto the E2M1 codebook, ties to even, as the cast does."""
    upper = torch.searchsorted(grid, mag.contiguous(), right=True).clamp(
        1, grid.numel() - 1
    )
    lower = upper - 1
    low, high = grid[lower], grid[upper]
    mid = (low + high) / 2
    return torch.where(
        mag > mid,
        high,
        torch.where(mag < mid, low, torch.where(lower % 2 == 0, low, high)),
    )


def _fp4_roundtrip(
    x: torch.Tensor, block: int = _FP4_BLOCK, e4m3_scale: bool = False
) -> torch.Tensor:
    """Quantize to FP4 (E2M1 plus one scale per ``block``) and back.

    The indexer uses groups of 32 with a power-of-two scale; the compressed KV
    uses groups of 16 with an e4m3 one. Both are what the reference trained
    against, so a selection or an attention output only matches through them.
    """
    shape = x.shape
    xf = x.float().unflatten(-1, (-1, block))
    amax = xf.abs().amax(-1, keepdim=True)
    if e4m3_scale:
        amax = amax.clamp_min(6.0 * 2.0**-9)
        scale = (amax / 6.0).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6.0 * 2.0**-126)
        # The scale rounds UP to a power of two, so amax / scale <= 6.
        scale = torch.exp2(torch.ceil(torch.log2(amax / 6.0)).clamp(-127, 127))
    grid = torch.tensor(_E2M1_GRID, dtype=torch.float32, device=x.device)
    mag = (xf.abs() / scale).clamp(max=6.0)
    q = _to_e2m1(mag, grid) * torch.sign(xf) * scale
    return q.flatten(-2).reshape(shape).to(x.dtype)


def _fp8_roundtrip(x: torch.Tensor, block: int = _FP8_BLOCK) -> torch.Tensor:
    """Quantize to FP8 e4m3 with one ue8m0 scale per ``block`` and back."""
    shape = x.shape
    xf = x.float().unflatten(-1, (-1, block))
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127))
    q = (xf / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * scale
    return q.flatten(-2).reshape(shape).to(x.dtype)


def _sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """One softmax over the gathered latents; ``attn_sink`` only in the denominator.

    q ``[N, h, d]``, kv ``[T, d]``, topk_idxs ``[N, k]`` with -1 for an empty
    slot, attn_sink ``[h]``.
    """
    valid = topk_idxs >= 0
    gathered = kv.index_select(0, topk_idxs.clamp_min(0).reshape(-1).long())
    gathered = gathered.view(*topk_idxs.shape, kv.shape[-1]).float()
    logits = torch.einsum("nhd,nkd->nhk", q.float(), gathered) * scale
    logits = logits.masked_fill(~valid.unsqueeze(1), float("-inf"))
    sink = attn_sink.float().view(1, -1)
    peak = torch.maximum(logits.amax(-1), sink)
    weights = torch.exp(logits - peak.unsqueeze(-1))
    denom = weights.sum(-1) + torch.exp(sink - peak)
    out = torch.einsum("nhk,nkd->nhd", weights, gathered) / denom.unsqueeze(-1)
    return out.to(q.dtype)


def _window_topk_idxs(
    window: int, num_tokens: int, start_pos: int, device: torch.device
) -> torch.Tensor:
    """Which sliding-window slots each query reads; -1 marks a slot holding nothing.

    Prefill indexes the fresh chunk directly, decode indexes the ring buffer.
    """
    if start_pos == 0:
        end = torch.arange(num_tokens, device=device).unsqueeze(1)
        idxs = (end - window + 1).clamp(0) + torch.arange(
            min(num_tokens, window), device=device
        )
        idxs = torch.where(idxs > end, -1, idxs)
    else:
        oldest = start_pos % window + 1
        idxs = torch.cat(
            [
                torch.arange(oldest, window, device=device),
                torch.arange(oldest, device=device),
            ]
        )
        idxs = torch.where(idxs > start_pos, -1, idxs).unsqueeze(0)
    return idxs.int()


def _grow(
    cache: torch.Tensor | None, need: int, width: int, ref: torch.Tensor
) -> torch.Tensor:
    """Dense append-only cache that doubles when it runs out of rows."""
    if cache is not None and cache.shape[0] >= need:
        return cache
    rows = max(need, 2 * (cache.shape[0] if cache is not None else 128))
    grown = torch.zeros(rows, width, dtype=ref.dtype, device=ref.device)
    if cache is not None:
        grown[: cache.shape[0]] = cache
    return grown


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: torch.Tensor | int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the two-level top-k: keep the best ``topk_blocks`` blocks.

    Positions the query cannot reach are already -inf, so a block scoring -inf
    is unreachable. Returns a bool mask shaped like ``logits``.
    """
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=float("-inf"))
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    # The block holding the query's newest position is only partly filled; pin
    # it in so an older full block cannot outscore it.
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last, float("inf")
    )
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > float("-inf")
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


# ---------------------------------------------------------------------------
# Compressor / Indexer
# ---------------------------------------------------------------------------


class Compressor(nn.Module):
    """Pools ``compress_ratio`` consecutive tokens into one KV latent.

    Ratio 1 is a plain projection. Above 1 the pooling runs in fp32 with a
    learned softmax gate over the group, and a trailing partial group waits in
    ``kv_state`` / ``score_state`` until the next call completes it. The latent
    comes out before RoPE because the indexer needs the unrotated form.
    """

    def __init__(self, args: DeepseekV41Args, ratio: int, prefix: str = ""):
        super().__init__()
        self.compress_ratio = ratio
        self.head_dim = args.head_dim
        self.eps = args.norm_eps
        qc = args.quant_config
        self.wkv = ReplicatedLinear(
            args.dim, args.head_dim, bias=False, quant_config=qc, prefix=f"{prefix}.wkv"
        )
        if ratio > 1:
            self.wgate = ReplicatedLinear(
                args.dim,
                args.head_dim,
                bias=False,
                quant_config=qc,
                prefix=f"{prefix}.wgate",
            )
            self.register_buffer(
                "kv_state",
                torch.zeros(ratio, args.head_dim, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "score_state",
                torch.full((ratio, args.head_dim), float("-inf"), dtype=torch.float32),
                persistent=False,
            )
        self.norm = RMSNorm(args.head_dim, args.norm_eps)
        self.prefix = prefix
        # The per-request ring the paged path pools a straddling group from,
        # bound by `DeepseekV4AttentionMetadataBuilder.build_kv_cache_tensor`.
        # Named apart from the eager `kv_state` because the two have different
        # shapes: the eager one is this sequence's, the paged one every
        # request's.
        self.paged_kv_state: torch.Tensor | None = None
        self.paged_score_state: torch.Tensor | None = None

    def pool_paged(
        self,
        x: torch.Tensor,
        plan,
        state_slot_in: torch.Tensor,
        state_slot_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Plan-driven batched pooling; returns pre-RoPE latents, row ids, seqs.

        One row of `plan.compress_plan_gpu` per compression boundary in the
        forward: `[ragged_id, batch_id, position, window_len]`, where the
        leading `window_len` of the group's `ratio` sources come from the
        request's ring rather than from this forward's tokens.
        """
        ratio = self.compress_ratio
        device = x.device
        if ratio == 1:
            kv_all, score_all = self.wkv(x), None
        else:
            # The reference holds both weights in fp32 at ratio > 1 and pools
            # in fp32; the eager path does the same.
            xf = x.float()
            kv_all = F.linear(xf, self.wkv.weight.float())
            score_all = F.linear(xf, self.wgate.weight.float())
        n = int(plan.num_compress)
        if n == 0:
            self._write_state_paged(plan, kv_all, score_all, state_slot_out)
            empty = torch.zeros(0, dtype=torch.long, device=device)
            return (
                torch.zeros(0, self.head_dim, dtype=x.dtype, device=device),
                empty,
                empty,
            )
        rows = plan.compress_plan_gpu[:n].long()
        ragged, bid, pos, wlen = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
        if ratio == 1:
            pooled = kv_all.index_select(0, ragged)
        else:
            assert self.paged_kv_state is not None, "compressor ring is unbound"
            d = self.head_dim
            ring_size = self.paged_kv_state.shape[1]
            k = torch.arange(ratio, device=device)
            src_row = (ragged - (ratio - 1)).unsqueeze(1) + k
            src_pos = (pos - (ratio - 1)).unsqueeze(1) + k
            take = k.unsqueeze(0) >= wlen.unsqueeze(1)
            slot = (
                state_slot_in.long()
                .index_select(0, bid)
                .unsqueeze(1)
                .expand(-1, ratio)
                .reshape(-1)
            )
            ring = src_pos.remainder(ring_size).reshape(-1)
            flat = src_row.clamp_min(0).reshape(-1)
            kv_g = torch.where(
                take.unsqueeze(-1),
                kv_all.index_select(0, flat).view(n, ratio, d),
                self.paged_kv_state[slot, ring, :d].view(n, ratio, d),
            )
            score_g = torch.where(
                take.unsqueeze(-1),
                score_all.index_select(0, flat).view(n, ratio, d),
                self.paged_score_state[slot, ring, :d].view(n, ratio, d),
            )
            pooled = (kv_g * score_g.softmax(dim=1)).sum(dim=1)
        # After the read: the ring this forward writes is the next one's.
        self._write_state_paged(plan, kv_all, score_all, state_slot_out)
        return _rms(pooled.to(x.dtype), self.norm.weight, self.eps), pos // ratio, bid

    def _write_state_paged(self, plan, kv_all, score_all, state_slot_out) -> None:
        """Carry this forward's trailing raw rows into each request's ring."""
        if self.compress_ratio == 1 or self.paged_kv_state is None:
            return
        m = int(plan.num_write)
        if m == 0:
            return
        rows = plan.write_plan_gpu[:m].long()
        ragged, bid, pos = rows[:, 0], rows[:, 1], rows[:, 2]
        slot = state_slot_out.long().index_select(0, bid)
        ring = pos.remainder(self.paged_kv_state.shape[1])
        d = self.head_dim
        self.paged_kv_state[slot, ring, :d] = kv_all.index_select(0, ragged).to(
            self.paged_kv_state.dtype
        )
        self.paged_score_state[slot, ring, :d] = score_all.index_select(0, ragged).to(
            self.paged_score_state.dtype
        )

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        """``x`` ``[num_tokens, dim]`` -> ``[num_groups, head_dim]``, or None."""
        num_tokens = x.shape[0]
        ratio, dtype = self.compress_ratio, x.dtype
        if ratio == 1:
            return _rms(self.wkv(x), self.norm.weight, self.eps)

        # The reference holds these two weights in fp32 for ratio > 1 and pools
        # in fp32; matching that needs an fp32 matmul, not an fp32 input.
        xf = x.float()
        kv = F.linear(xf, self.wkv.weight.float())
        score = F.linear(xf, self.wgate.weight.float())
        if start_pos == 0:
            should_compress = num_tokens >= ratio
            remainder = num_tokens % ratio
            cutoff = num_tokens - remainder
            if remainder:
                self.kv_state[:remainder] = kv[cutoff:]
                self.score_state[:remainder] = score[cutoff:]
                kv, score = kv[:cutoff], score[:cutoff]
            kv = kv.unflatten(0, (-1, ratio))
            score = score.unflatten(0, (-1, ratio))
            kv = (kv * score.softmax(dim=1)).sum(dim=1)
        else:
            assert num_tokens == 1, "chunked prefill is not supported yet"
            should_compress = (start_pos + 1) % ratio == 0
            self.kv_state[start_pos % ratio] = kv[0]
            self.score_state[start_pos % ratio] = score[0]
            if should_compress:
                pooled = (self.kv_state * self.score_state.softmax(dim=0)).sum(dim=0)
                kv = pooled.unsqueeze(0)
        if not should_compress:
            return None
        return _rms(kv.to(dtype), self.norm.weight, self.eps)


class Indexer(nn.Module):
    """Keeps the ``index_topk`` best compressed positions per query.

    A small side attention: replicated fp8 query heads against one shared key
    per compressed position, rectified per head then combined by
    ``weights_proj``. Only a layer that compresses its own KV can build the
    keys, because they are a projection of the compressor latent.
    """

    def __init__(
        self,
        args: DeepseekV41Args,
        layer_id: int,
        layer_map: V41LayerMap,
        prefix: str = "",
    ):
        super().__init__()
        self.owns_k = layer_id in layer_map.kv_sources
        self.is_candidate_source = layer_id == layer_map.candidate_source
        self.uses_candidates = 0 <= layer_map.candidate_source < layer_id
        self.compress_ratio = args.compress_ratios[layer_id]
        self.candidate_topk_blocks = args.candidate_topk_blocks
        self.candidate_block_size = args.candidate_block_size
        self.n_heads = args.index_n_heads
        self.index_head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.eps = args.norm_eps
        qc = args.quant_config
        # Replicated, as in V4: every rank needs all heads to pick the same
        # positions without an all-reduce on the score.
        self.wq_b = ReplicatedLinear(
            args.q_lora_rank,
            self.n_heads * self.index_head_dim,
            bias=False,
            quant_config=qc,
            prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = ReplicatedLinear(
            args.dim,
            self.n_heads,
            bias=False,
            quant_config=qc,
            prefix=f"{prefix}.weights_proj",
        )
        self._weights_scale = self.index_head_dim**-0.5 * self.n_heads**-0.5
        if self.owns_k:
            self.wk = ReplicatedLinear(
                args.head_dim,
                self.index_head_dim,
                bias=False,
                quant_config=qc,
                prefix=f"{prefix}.wk",
            )
            self.k_norm = RMSNorm(self.index_head_dim, args.norm_eps)
            self.k_cache: torch.Tensor | None = None
        self.rotary_emb: _V4RoPE | None = None
        self.prefix = prefix
        # Paged: this layer's KV owner's slice of the index-K pool,
        # `[num_blocks, pool_rows_per_block, index_head_dim]`, and how many
        # rows of a block it holds. Bound by the V4 attention builder.
        self.kv_cache: torch.Tensor | None = None
        self.rows_per_block = 0

    def publish_keys_paged(
        self,
        latent: torch.Tensor,
        positions: torch.Tensor,
        compress_ids: torch.Tensor,
        batch_ids: torch.Tensor,
        block_tables: torch.Tensor,
        layer_rows_per_block: int,
    ) -> None:
        """Page this forward's index keys into the owner's pool slice.

        `layer_rows_per_block` is the owning class's rows per block, which is
        what maps a compressed id onto a block; the pool's own row count is
        sized for the denser of the two classes, so the tail of a coarse
        class's block simply goes unwritten.
        """
        assert self.rotary_emb is not None and self.kv_cache is not None
        k = _rms(self.wk(latent), self.k_norm.weight, self.eps)
        _rope_(
            k[..., -self.rope_head_dim :],
            self.rotary_emb.freqs_for_positions(positions),
        )
        if _V41_INDEXER_MXFP4:
            k = _fp4_roundtrip(k)
        phys = block_tables[batch_ids, compress_ids // layer_rows_per_block].long()
        dest = phys * self.rows_per_block + compress_ids % layer_rows_per_block
        self.kv_cache.view(-1, self.index_head_dim).index_copy_(
            0, dest, k.to(self.kv_cache.dtype)
        )

    def topk_paged(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        batch: "V41Batch",
        layer_rows_per_block: int,
        shared: V41SharedRuntime,
    ) -> torch.Tensor:
        """Seq-local top-k compressed rows per token, `[T, k]` int32.

        Column `j` of row `t` is meaningful only for `j < min(visible(t),
        index_topk)`; the translator recovers that bound from the index
        buffer's own per-token span, so the tail is left at -1.
        """
        assert self.rotary_emb is not None and self.kv_cache is not None
        ratio = self.compress_ratio
        positions = batch.positions
        num_tokens = x.shape[0]
        q = self.wq_b(qr).view(num_tokens, self.n_heads, self.index_head_dim)
        _rope_(
            q[..., -self.rope_head_dim :],
            self.rotary_emb.freqs_for_positions(positions),
        )
        if _V41_INDEXER_MXFP4:
            q = _fp4_roundtrip(q)
        q = q.float()
        weights = self.weights_proj(x).float() * self._weights_scale
        visible = (positions + 1) // ratio
        widest = max(
            (min(self.index_topk, (p + 1) // ratio) for p in batch.last_pos),
            default=0,
        )
        out = torch.full(
            (num_tokens, max(widest, 1)), -1, dtype=torch.int32, device=x.device
        )
        pool = self.kv_cache.view(-1, self.index_head_dim)
        for b, (lo, hi) in enumerate(batch.spans):
            length = (batch.last_pos[b] + 1) // ratio
            if hi <= lo or length <= 0:
                continue
            cids = torch.arange(length, device=x.device)
            phys = batch.block_tables[b, cids // layer_rows_per_block].long()
            keys = pool.index_select(
                0, phys * self.rows_per_block + cids % layer_rows_per_block
            ).float()
            vis = visible[lo:hi].unsqueeze(-1)
            score = _index_scores(q[lo:hi], keys, weights[lo:hi])
            score = score.masked_fill(cids.unsqueeze(0) >= vis, float("-inf"))
            if self.is_candidate_source:
                shared.block_candidates[b] = select_candidate_blocks(
                    score, vis, self.candidate_topk_blocks, self.candidate_block_size
                )
            elif self.uses_candidates:
                keep = shared.block_candidates[b]
                score = score.masked_fill(~keep[:, :length], float("-inf"))
            kk = min(self.index_topk, length)
            idxs = score.topk(kk, dim=-1, sorted=False).indices.sort(dim=-1).values
            out[lo:hi, :kk] = torch.where(idxs < vis, idxs, -1).int()
        return out

    def publish_keys(self, latent: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Project the pre-RoPE latent to index keys and append them to the cache."""
        ratio = self.compress_ratio
        first = start_pos // ratio
        k = _rms(self.wk(latent), self.k_norm.weight, self.eps)
        positions = (
            first + torch.arange(k.shape[0], device=k.device, dtype=torch.long)
        ) * ratio
        assert self.rotary_emb is not None
        freqs = self.rotary_emb.freqs_for_positions(positions)
        _rope_(k[..., -self.rope_head_dim :], freqs)
        if _V41_INDEXER_MXFP4:
            k = _fp4_roundtrip(k)
        self.k_cache = _grow(self.k_cache, first + k.shape[0], self.index_head_dim, k)
        self.k_cache[first : first + k.shape[0]] = k
        return self.k_cache

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        start_pos: int,
        offset: int,
        shared: V41SharedRuntime,
    ) -> torch.Tensor:
        """Returns ``[num_tokens, topk]`` int32 indices into the concatenated KV."""
        num_tokens = x.shape[0]
        ratio = self.compress_ratio
        end_pos = start_pos + num_tokens
        assert self.rotary_emb is not None and shared.index_k is not None

        q = self.wq_b(qr).view(num_tokens, self.n_heads, self.index_head_dim)
        _rope_(
            q[..., -self.rope_head_dim :],
            self.rotary_emb.freqs_for_positions(positions),
        )
        if _V41_INDEXER_MXFP4:
            q = _fp4_roundtrip(q)

        index_k = shared.index_k[: end_pos // ratio]
        weights = self.weights_proj(x).float() * self._weights_scale
        score = torch.einsum("nhd,td->nht", q.float(), index_k.float())
        score = (score.relu() * weights.unsqueeze(-1)).sum(dim=1)

        if start_pos == 0:
            compress_lens = (
                torch.arange(1, num_tokens + 1, device=x.device) // ratio
            ).unsqueeze(-1)
            score = score.masked_fill(
                torch.arange(num_tokens // ratio, device=x.device) >= compress_lens,
                float("-inf"),
            )
        else:
            compress_lens = end_pos // ratio

        if self.is_candidate_source:
            shared.candidates = select_candidate_blocks(
                score,
                compress_lens,
                self.candidate_topk_blocks,
                self.candidate_block_size,
            )
        elif self.uses_candidates:
            assert shared.candidates is not None
            score = score.masked_fill(~shared.candidates, float("-inf"))

        topk = min(self.index_topk, end_pos // ratio)
        idxs = score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < compress_lens, idxs + offset, -1).int()


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class DeepseekV41Attention(nn.Module):
    """Latent MQA over a 128-slot sliding window plus up to 512 gathered
    compressed latents, in one softmax. Q and the output projection are both
    low-rank, the output one grouped over ``o_groups``.
    """

    def __init__(
        self,
        layer_id: int,
        args: DeepseekV41Args,
        layer_map: V41LayerMap,
        shared: V41SharedRuntime,
        prefix: str = "",
        **_ignored: Any,
    ):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        assert args.n_heads % tp_size == 0, f"n_heads {args.n_heads} vs tp {tp_size}"
        assert args.o_groups % tp_size == 0, f"o_groups {args.o_groups} vs tp {tp_size}"
        self.layer_id = layer_id
        self.tp_size = tp_size
        self.tp_rank = get_tp_group().rank_in_group
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // tp_size
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.o_lora_rank = args.o_lora_rank
        self.n_groups = args.o_groups
        self.n_local_groups = args.o_groups // tp_size
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps
        self.softmax_scale = args.head_dim**-0.5
        self.shared = shared
        self.is_kv_source = layer_id in layer_map.kv_sources
        self.is_index_source = layer_id in layer_map.index_sources
        self.wo_a_block = args.weight_block_size

        qc = args.quant_config
        self.attn_sink = atom_parameter(
            torch.empty(self.n_local_heads, dtype=torch.float32)
        )
        self.attn_sink.weight_loader = self._sink_loader
        # wq_a and wkv stay separate linears: the [32, 32] block scale makes a
        # merged allocation harder to reason about than the GEMM it saves.
        self.wq_a = ReplicatedLinear(
            self.dim,
            args.q_lora_rank,
            bias=False,
            quant_config=qc,
            prefix=f"{prefix}.wq_a",
        )
        self.q_norm = RMSNorm(args.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            args.q_lora_rank,
            args.n_heads * args.head_dim,
            bias=False,
            quant_config=qc,
            prefix=f"{prefix}.wq_b",
        )
        self.wkv = ReplicatedLinear(
            self.dim,
            args.head_dim,
            bias=False,
            quant_config=qc,
            prefix=f"{prefix}.wkv",
        )
        self.kv_norm = RMSNorm(args.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            args.n_heads * args.head_dim // args.o_groups,
            args.o_groups * args.o_lora_rank,
            bias=False,
            quant_config=qc,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_b = RowParallelLinear(
            args.o_groups * args.o_lora_rank,
            self.dim,
            bias=False,
            quant_config=qc,
            prefix=f"{prefix}.wo_b",
        )

        self.compressor = (
            Compressor(args, self.compress_ratio, prefix=f"{prefix}.compressor")
            if self.is_kv_source
            else None
        )
        self.indexer = (
            Indexer(args, layer_id, layer_map, prefix=f"{prefix}.indexer")
            if self.is_index_source
            else None
        )

        # Ratio 0 is window-only, so it keeps the base theta and no YaRN.
        if self.compress_ratio:
            original_seq_len = args.original_seq_len
            rope_theta = args.compress_rope_theta
        else:
            original_seq_len, rope_theta = 0, args.rope_theta
        self.rotary_emb = _V4RoPE(
            rotary_dim=args.rope_head_dim,
            max_seq_len=args.max_seq_len,
            base=rope_theta,
            factor=args.rope_factor,
            original_seq_len=original_seq_len,
            beta_fast=args.beta_fast,
            beta_slow=args.beta_slow,
            dtype=torch.bfloat16,
        )
        if self.indexer is not None:
            self.indexer.rotary_emb = self.rotary_emb

        self.register_buffer(
            "window_kv_cache",
            torch.zeros(args.window_size, args.head_dim),
            persistent=False,
        )
        self.compress_kv_cache: torch.Tensor | None = None

        # ----- Paged bindings -----
        # All filled by `DeepseekV4AttentionMetadataBuilder`
        # `.build_kv_cache_tensor`, which runs after `allocate_kv_cache`.
        # `unified_kv` being None is also how the paged forward recognizes
        # warmup, which runs before that.
        self.unified_kv: torch.Tensor | None = None
        self.swa_plane: torch.Tensor | None = None
        self.swa_window = None
        self.envelope_rows = 0
        # Rows from this layer's own compressed rows to the ones it reads;
        # non-zero exactly where it reuses another layer's compressor.
        self.compress_bias = 0
        self.rows_per_block = 0
        # Views of the layers that read this one's compressed rows. A reuse
        # layer keeps its own envelope rows and its own window ring, and the
        # window is addressed by one class-wide formula relative to a layer's
        # own base -- so the only redirection that leaves both halves
        # addressable is to put a copy of the latent in each reader's rows.
        self.mirror_kv: list[torch.Tensor] = []
        # Which of the two compressed classes this layer is in, and so which
        # of the two index buffers carries its window prefix.
        self.is_fine_class = False

    def _sink_loader(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        param.data.copy_(
            loaded.narrow(0, self.tp_rank * self.n_local_heads, self.n_local_heads).to(
                param.dtype
            )
        )

    def process_weights_after_loading(self) -> None:
        """Dequantize ``wo_a`` to BF16 for the grouped-LoRA einsum.

        V4's hook is 128-block only -- both its mxscale fast path and its
        dequant assume it -- while V4.1 ships a [32, 32] block scale, so the
        block size comes from the config and the mxscale path is skipped.
        """
        weight = self.wo_a.weight
        if weight.dtype == torch.bfloat16:
            return
        scale = getattr(self.wo_a, "weight_scale", None)
        if (
            weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
            or scale is None
        ):
            return
        raw = scale.data
        if raw.dtype == torch.uint8:
            # A bare biased exponent, which is what an e8m0 byte is.
            values = torch.exp2(raw.float() - 127.0)
        else:
            values = raw.float()
        self.wo_a.weight = atom_parameter(
            _dequant_fp8_block_to_bf16(weight.data, values, block=self.wo_a_block)
        )
        try:
            delattr(self.wo_a, "weight_scale")
        except AttributeError:
            pass
        # Stop LinearBase's post-load from shuffling a matrix that torch.einsum
        # now reads row-major.
        self.wo_a.quant_type = QuantType.No
        self.wo_a.need_normalize_e4m3fn_to_e4m3fnuz = False

    def _window_kv(
        self, x: torch.Tensor, positions: torch.Tensor, start_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = x.shape[0]
        win = self.window_size
        kv = _rms(self.wkv(x), self.kv_norm.weight, self.eps)
        _rope_(
            kv[..., -self.rope_head_dim :],
            self.rotary_emb.freqs_for_positions(positions),
        )
        if _V41_SIM_QAT_QUANT:
            kv = _fp8_roundtrip(kv, _FP8_BLOCK)
        if start_pos == 0:
            if num_tokens <= win:
                self.window_kv_cache[:num_tokens] = kv
            else:
                cutoff = num_tokens % win
                tail = kv[-win:]
                self.window_kv_cache[cutoff:win] = tail[: win - cutoff]
                self.window_kv_cache[:cutoff] = tail[win - cutoff :]
            window_kv = kv
        else:
            self.window_kv_cache[start_pos % win] = kv[0]
            window_kv = self.window_kv_cache
        return window_kv, _window_topk_idxs(win, num_tokens, start_pos, x.device)

    def _compress_kv(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        start_pos: int,
        offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = x.shape[0]
        ratio = self.compress_ratio
        compress_len = (start_pos + num_tokens) // ratio
        latent = None
        if self.is_kv_source:
            latent = self.compressor(x, start_pos)
            if latent is not None and self.indexer is not None:
                self.shared.index_k = self.indexer.publish_keys(latent, start_pos)

        if self.is_index_source and compress_len:
            self.shared.topk_idxs = self.indexer(
                x, qr, positions, start_pos, offset, self.shared
            )
        elif self.is_index_source:
            self.shared.topk_idxs = torch.empty(
                num_tokens, 0, dtype=torch.int32, device=x.device
            )
        idxs = self.shared.topk_idxs
        assert idxs is not None, "no index source ran before this layer"

        if latent is not None:
            first = start_pos // ratio
            group_positions = (
                first + torch.arange(latent.shape[0], device=x.device, dtype=torch.long)
            ) * ratio
            _rope_(
                latent[..., -self.rope_head_dim :],
                self.rotary_emb.freqs_for_positions(group_positions),
            )
            if _V41_SIM_QAT_QUANT:
                latent = _fp4_roundtrip(latent, _COMPRESS_FP4_BLOCK, True)
            self.compress_kv_cache = _grow(
                self.compress_kv_cache, first + latent.shape[0], self.head_dim, latent
            )
            self.compress_kv_cache[first : first + latent.shape[0]] = latent
        if self.is_kv_source:
            # Every step, not only the ones that close a group: a ratio-2
            # layer closes nothing on half its decode steps, and leaving the
            # slot alone there hands it whatever the last compressed layer of
            # the PREVIOUS forward put in it.
            self.shared.compress_kv = self.compress_kv_cache
        assert self.shared.compress_kv is not None
        return self.shared.compress_kv[:compress_len], idxs

    # ---------------------------------------------------------------- #
    # Paged path (the engine)                                           #
    # ---------------------------------------------------------------- #

    def _paged_compress(self, x: torch.Tensor, batch: "V41Batch") -> None:
        """Owner layers only: pool, publish the index keys, page the rows."""
        attn_md = batch.attn_md
        plan = attn_md.compress_plans[self.compress_ratio]
        latent, compress_ids, batch_ids = self.compressor.pool_paged(
            x, plan, attn_md.state_slot_in, attn_md.state_slot_out
        )
        if latent.shape[0] == 0:
            return
        # RoPE at the group's FIRST token, which is what the reference rotates
        # a compressed latent by -- not at the boundary the plan names.
        group_positions = compress_ids * self.compress_ratio
        if self.indexer is not None and self.indexer.owns_k:
            # Before the rotation: the index keys project the unrotated latent.
            self.indexer.publish_keys_paged(
                latent,
                group_positions,
                compress_ids,
                batch_ids,
                batch.block_tables,
                self.rows_per_block,
            )
        _rope_(
            latent[..., -self.rope_head_dim :],
            self.rotary_emb.freqs_for_positions(group_positions),
        )
        if _V41_SIM_QAT_QUANT:
            latent = _fp4_roundtrip(latent, _COMPRESS_FP4_BLOCK, True)
        phys = batch.block_tables[batch_ids, compress_ids // self.rows_per_block].long()
        rows = phys * self.envelope_rows + compress_ids % self.rows_per_block
        latent = latent.to(self.unified_kv.dtype)
        self.unified_kv.index_copy_(0, rows, latent)
        for view in self.mirror_kv:
            view.index_copy_(0, rows, latent)

    def _paged_index_buffers(self, batch: "V41Batch"):
        """This layer's (indices, indptr, skip, window) for the paged kernels."""
        md = batch.attn_md
        if batch.is_decode:
            if self.compress_ratio == 0:
                return md.kv_indices_swa, md.kv_indptr_swa, None, 0
            if self.is_fine_class:
                return md.kv_indices_csa, md.kv_indptr_csa, None, self.window_size
            return md.kv_indices_hca, md.kv_indptr_hca, None, self.window_size
        skip = md.skip_prefix_len_csa
        if self.compress_ratio == 0:
            return md.kv_indices_prefix_swa, md.kv_indptr_prefix_swa, skip, 0
        if self.is_fine_class:
            return md.kv_indices_prefix_csa, md.kv_indptr_prefix_csa, skip, 0
        return md.kv_indices_prefix_hca, md.kv_indptr_prefix_hca, skip, 0

    def forward_paged(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        num_tokens = x.shape[0]
        batch = self.shared.batch
        assert batch is not None, "DeepseekV41Model.forward did not open the batch"
        attn_md = batch.attn_md
        rd = self.rope_head_dim
        freqs = self.rotary_emb.freqs_for_positions(positions)

        qr = _rms(self.wq_a(x), self.q_norm.weight, self.eps)
        q = self.wq_b(qr).view(num_tokens, self.n_local_heads, self.head_dim)
        _rope_(q[..., -rd:], freqs)
        kv = _rms(self.wkv(x), self.kv_norm.weight, self.eps)
        _rope_(kv[..., -rd:], freqs)
        if _V41_SIM_QAT_QUANT:
            kv = _fp8_roundtrip(kv, _FP8_BLOCK)
        kv = kv.contiguous()

        if self.compress_ratio:
            # The compressed rows this forward closes are visible to its own
            # queries, so both writes precede the attention.
            if self.compressor is not None:
                self._paged_compress(x, batch)
            if self.indexer is not None:
                self.shared.topk_local = self.indexer.topk_paged(
                    x, qr, batch, self.rows_per_block, self.shared
                )

        indices, indptr, skip, window = self._paged_index_buffers(batch)
        if self.compress_ratio:
            topk_local = self.shared.topk_local
            assert topk_local is not None, "no index source ran before this layer"
            csa_translate_pack(
                topk_local,
                batch.block_tables,
                positions,
                indptr,
                attn_md.batch_id_per_q_token,
                skip,
                indices,
                envelope_rows=self.envelope_rows,
                csa_block_capacity=self.rows_per_block,
                compress_bias=self.compress_bias,
                window_size=window,
            )

        if batch.is_decode:
            # Decode reads its own token out of the ring, so the write is
            # first; prefill's prefix must NOT see this chunk, so its write is
            # after the attention.
            swa_write(
                kv,
                positions,
                attn_md.cu_seqlens_q,
                attn_md.state_slot_out,
                self.swa_plane,
                self.swa_window,
                max(1, min(attn_md.max_seqlen_q, self.swa_window.ring_slots)),
            )
            o = sparse_attn_v4_paged_decode(
                q,
                self.unified_kv,
                indices,
                indptr,
                self.attn_sink,
                self.softmax_scale,
            )
        else:
            o = sparse_attn_v4_paged_prefill(
                q,
                self.unified_kv,
                indices,
                indptr,
                kv,
                attn_md.kv_indices_extend,
                attn_md.kv_indptr_extend,
                self.attn_sink,
                self.softmax_scale,
            )
            swa_write(
                kv,
                positions,
                attn_md.cu_seqlens_q,
                attn_md.state_slot_out,
                self.swa_plane,
                self.swa_window,
                max(1, min(self.window_size, attn_md.max_seqlen_q)),
            )

        o = o.reshape(num_tokens, self.n_local_heads, self.head_dim)
        _rope_(o[..., -rd:], freqs, inverse=True)
        o = o.view(num_tokens, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("sgd,grd->sgr", o.to(wo_a.dtype), wo_a)
        return self.wo_b(o.flatten(1))

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.shared.batch is not None:
            return self.forward_paged(x, positions)
        if self.shared.dummy:
            # Warmup: the pools it would read are not bound yet.
            return torch.zeros_like(x)
        return self.forward_eager(x, positions)

    def forward_eager(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        num_tokens = x.shape[0]
        start_pos = int(positions[0])
        assert (
            int(positions[-1]) - start_pos == num_tokens - 1
        ), "the eager V4.1 path takes one contiguous sequence per call"
        assert start_pos == 0 or num_tokens == 1, "chunked prefill is not supported yet"
        rd = self.rope_head_dim

        qr = _rms(self.wq_a(x), self.q_norm.weight, self.eps)
        q = self.wq_b(qr).view(num_tokens, self.n_local_heads, self.head_dim)
        freqs = self.rotary_emb.freqs_for_positions(positions)
        _rope_(q[..., -rd:], freqs)

        kv, topk_idxs = self._window_kv(x, positions, start_pos)
        if self.compress_ratio:
            compress_kv, compress_idxs = self._compress_kv(
                x, qr, positions, start_pos, kv.shape[0]
            )
            kv = torch.cat([kv, compress_kv], dim=0)
            topk_idxs = torch.cat(
                [topk_idxs.expand(num_tokens, -1), compress_idxs], dim=-1
            )
        else:
            topk_idxs = topk_idxs.expand(num_tokens, -1)

        o = _sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        _rope_(o[..., -rd:], freqs, inverse=True)

        # wo_a is block diagonal over groups, so an einsum and not a Linear.
        o = o.view(num_tokens, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("sgd,grd->sgr", o.to(wo_a.dtype), wo_a)
        return self.wo_b(o.flatten(1))


# ---------------------------------------------------------------------------
# Engram
# ---------------------------------------------------------------------------


def _aiter_op(name: str):
    """Resolve an AITER engram op at call time; None while it is being written."""
    if not _V41_ENGRAM_AITER:
        return None
    return getattr(aiter, name, None)


class EngramEmbedding(nn.Module):
    """The n-gram hash table, fp8 with one e8m0 scale per 32 columns.

    Sharded over rows: a rank returns zero for a row it does not own, and the
    caller all-reduces. Kept fp8 on disk and dequantized on lookup -- at
    384M rows there is no other option.
    """

    def __init__(self, num_embeddings: int, dim: int, block: int = 32):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tp_group().rank_in_group
        self.num_embeddings = num_embeddings
        self.block = block
        # The released row counts (384006168, 384016682) divide by neither 4
        # nor 8, so the shard is a ceiling and the last rank is clamped to the
        # rows that exist. An id nobody owns reads as zero, so the all-reduce
        # is still exact.
        shard = (num_embeddings + tp_size - 1) // tp_size
        self.row_offset = self.tp_rank * shard
        self.num_rows = max(0, min(shard, num_embeddings - self.row_offset))
        self.weight = atom_parameter(
            torch.zeros(self.num_rows, dim, dtype=torch.float8_e4m3fn)
        )
        self.weight_scale = atom_parameter(
            torch.zeros(self.num_rows, dim // block, dtype=torch.uint8)
        )
        self.weight.weight_loader = self._loader
        self.weight_scale.weight_loader = self._loader

    def _loader(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        rows = max(0, min(self.num_rows, loaded.shape[0] - self.row_offset))
        if rows <= 0:
            return
        chunk = loaded.narrow(0, self.row_offset, rows)
        if chunk.dtype != param.dtype:
            chunk = chunk.view(param.dtype) if chunk.element_size() == 1 else chunk
        param.data[:rows].copy_(chunk.to(param.device))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """``indices`` ``[N, cols]`` global row ids -> ``[N, cols, dim]`` bf16."""
        if self.num_rows == 0:  # a rank past the end of a ragged last shard
            return torch.zeros(
                *indices.shape,
                self.weight.shape[-1],
                dtype=torch.bfloat16,
                device=indices.device,
            )
        fused = _aiter_op("engram_embedding_lookup")
        if fused is not None:
            return fused(
                indices,
                self.weight,
                self.weight_scale,
                row_offset=self.row_offset,
                num_rows=self.num_rows,
            )
        local = indices - self.row_offset
        owned = (local >= 0) & (local < self.num_rows)
        local = local.clamp(0, self.num_rows - 1).reshape(-1)
        raw = self.weight.view(torch.uint8).index_select(0, local)
        values = raw.view(torch.float8_e4m3fn).float()
        scale = torch.exp2(self.weight_scale.index_select(0, local).float() - 127.0)
        values = (values.unflatten(-1, (-1, self.block)) * scale.unsqueeze(-1)).flatten(
            -2
        )
        values = values.view(*indices.shape, -1).to(torch.bfloat16)
        return values * owned.unsqueeze(-1)


class Engram(nn.Module):
    """An n-gram lookup written into the residual stream, gated by how well it
    matches that stream. One key per hc copy plus a shared value.
    """

    def __init__(self, args: DeepseekV41Args, layer_id: int, layout: EngramLayout):
        super().__init__()
        self.layer_id = layer_id
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.dim = args.dim
        self.hc_mult = args.hc_mult
        self.eps = args.norm_eps
        self.clamp_value = 1e-6
        self.tp_size = get_tensor_model_parallel_world_size()
        self.embed = EngramEmbedding(
            layout.num_embeddings[self.layer_hash_index],
            layout.head_dim,
            block=_ENGRAM_SCALE_BLOCK,
        )
        self.wkv = ReplicatedLinear(
            layout.n_hash_cols * layout.head_dim,
            args.dim * (args.hc_mult + 1),
            bias=False,
            quant_config=args.quant_config,
            prefix=f"layers.{layer_id}.engram.wkv",
        )
        self.q_weight = atom_parameter(torch.ones(args.hc_mult, args.dim))
        self.k_weight = atom_parameter(torch.ones(args.hc_mult, args.dim))

    def forward(
        self,
        x: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``x`` ``[N, hc, dim]``, ``hash_ids`` ``[N, n_hash_cols]``."""
        rows = self.embed(hash_ids)
        if self.tp_size > 1:
            rows = tensor_model_parallel_all_reduce(rows)
        kv = self.wkv(rows.flatten(-2))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        fused = _aiter_op("engram_gate_apply") if _V41_ENGRAM_FUSED_GATE else None
        if fused is not None:
            return fused(x, key, value, self.q_weight, self.k_weight, self.eps)
        weight = self.q_weight.float() * self.k_weight.float()
        h = x.float()
        # Normalized per (token, hc copy) over dim, NOT jointly over the copies.
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(
            key.square().mean(-1) + self.eps
        )
        dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
        gate = torch.sigmoid(
            torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot)
        )
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


# ---------------------------------------------------------------------------
# Block (mHC with the one-block coefficient shift)
# ---------------------------------------------------------------------------


@dataclass
class V41HCState:
    """The mHC residual stream plus the ``pre`` the next sub-layer will use."""

    residual: torch.Tensor
    pre_mix: torch.Tensor


def _mhc_post_dim_ok(dim: int) -> bool:
    """Whether aiter's ``mhc_post`` has a tile for this hidden size.

    It picks the largest residual block that divides ``dim`` and then asks for
    two of them to prefetch, so a small ``dim`` is rejected by the kernel
    rather than falling back to a smaller tile.
    """
    if get_gfx() != "gfx942" and dim % 1024 == 0:
        block = 1024
    elif dim % 512 == 0:
        block = 512
    elif dim % 256 == 0:
        block = 256
    else:
        return False
    return dim >= 2 * block


def _hc_post_torch(
    x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
) -> torch.Tensor:
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=-3
    )
    return y.type_as(x)


class DeepseekV41Block(_V4Block):
    """V4's block, with V4.1's attention and V4.1's single-pass mHC.

    V4 applies the ``pre`` coefficients a sub-layer computed to that same
    sub-layer. V4.1 shifts them by one: a sub-layer mixes its input with the
    PREVIOUS sub-layer's ``pre``, so ``forward`` returns its own ffn ``pre``
    for the next block and the first block is bootstrapped one-hot on stream 0.
    That is why ``hc_pre`` (whose fused kernel does not expose ``pre``) is
    replaced here by ``hc_mixes`` plus an explicit weighted sum.
    """

    def __init__(
        self,
        layer_id: int,
        args: DeepseekV41Args,
        layer_map: V41LayerMap,
        shared: V41SharedRuntime,
        engram_layout: EngramLayout | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ):
        nn.Module.__init__(self)
        self.prefix = prefix
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        self.attn = DeepseekV41Attention(
            layer_id, args, layer_map, shared, prefix=f"{prefix}.attn"
        )
        self.ffn = MoE(layer_id, args, prefix=f"{prefix}.ffn", alt_stream=alt_stream)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        self.engram = (
            Engram(args, layer_id, engram_layout)
            if engram_layout is not None and layer_id in engram_layout.layer_ids
            else None
        )
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * args.dim
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
        self._moe_merge_enabled = False
        # V4's __init__ is bypassed, so bind what its inherited `hc_post` reads.
        # The fused pre/post pair is not usable here: it applies the `pre` it
        # just computed, which is the shift this block exists to undo.
        self._mhc_post = (
            getattr(aiter, "mhc_post", None) if _mhc_post_dim_ok(args.dim) else None
        )
        self._mhc_pre = None
        self._mhc_fused_post_pre = None
        self.enable_fused_hc = False

    def hc_mixes(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``pre`` / ``post`` / ``comb`` from one projection of the flat stream."""
        if _V41_MHC_FUSED_COEFFS:
            fused = getattr(aiter, "mhc_split_coeffs", None)
            if fused is None:
                raise RuntimeError(
                    "ATOM_V41_MHC_FUSED_COEFFS is set, but aiter has no op that "
                    "returns `pre` without applying it -- mhc_pre / mhc / "
                    "mhc_post_pre all fold it into the layer input, which is "
                    "the coefficient V4.1 carries to the next sub-layer."
                )
            return fused(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                self.norm_eps,
                self.hc_eps,
                self.hc_sinkhorn_iters,
            )
        flat = x.flatten(-2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flat, hc_fn) * rsqrt
        return hc_split_sinkhorn(
            mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
        )

    @staticmethod
    def hc_apply_pre(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        """Collapse the hc copies into one sub-layer input."""
        return torch.sum(pre.unsqueeze(-1) * x.float(), dim=-2).to(x.dtype)

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        # The aiter kernel is GPU-only and the eager path is also run on CPU.
        if not x.is_cuda:
            return _hc_post_torch(x, residual, post, comb)
        return super().hc_post(x, residual, post, comb)

    def forward(self, hc_state: V41HCState, positions: torch.Tensor) -> V41HCState:
        residual = hc_state.residual
        attn_pre, attn_post, attn_comb = self.hc_mixes(
            residual, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.hc_apply_pre(residual, hc_state.pre_mix)
        x = _rms(x, self.attn_norm.weight, self.norm_eps)
        x = self.attn(x, positions)
        residual = self.hc_post(x, residual, attn_post, attn_comb)

        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(
            residual, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.hc_apply_pre(residual, attn_pre)
        x = _rms(x, self.ffn_norm.weight, self.norm_eps)
        x = self.ffn(x)
        residual = self.hc_post(x, residual, ffn_post, ffn_comb)
        return V41HCState(residual=residual, pre_mix=ffn_pre)


# ---------------------------------------------------------------------------
# Quantization config
# ---------------------------------------------------------------------------


def make_v41_quant_config(hf_config, model_path=None, online_quant_config=None):
    """V4's quant config plus the two things V4.1 states differently.

    The routed experts are FP4 even though the checkpoint's top-level
    ``quant_method`` reads ``fp8`` -- ``expert_dtype`` inside
    ``quantization_config`` is the declaration -- and the compressor / indexer
    weights that ship without a ``.scale`` companion stay BF16.
    """
    # TODO: the dense fp8 layers still resolve to per_1x128 because ATOM has no
    # [32, 32] block spec yet. That work is separate; when it lands the base
    # parser produces the right spec and nothing here has to change.
    base = make_v4_quant_config(
        hf_config, model_path=model_path, online_quant_config=online_quant_config
    )
    quant = getattr(hf_config, "quantization_config", None) or {}
    expert_dtype = (
        str(quant.get("expert_dtype") or "").lower() if isinstance(quant, dict) else ""
    )
    fp4_spec = LayerQuantConfig(quant_type=QuantType.per_1x32, quant_dtype=dtypes.fp4x2)
    fp8_block_spec = LayerQuantConfig(
        quant_type=QuantType.per_1x128, quant_dtype=dtypes.fp8
    )
    no_spec = LayerQuantConfig(quant_type=QuantType.No, quant_dtype=torch.bfloat16)
    bf16_on_disk = (".compressor.", ".indexer.wk", ".indexer.weights_proj")
    inner = base.get_layer_quant_config

    def overridden(layer_name, use_online_quant=False, *, check_children=False):
        if ".ffn.experts" in layer_name:
            if "fp4" in expert_dtype:
                return fp4_spec
            if "fp8" in expert_dtype:
                return fp8_block_spec
            if not quant:
                # Nothing declared at all, e.g. the bf16 parity checkpoint.
                # V4's detector falls back to FP4 here, which is a guess.
                return no_spec
        if any(part in layer_name for part in bf16_on_disk):
            return no_spec
        return inner(
            layer_name,
            use_online_quant=use_online_quant,
            check_children=check_children,
        )

    base.get_layer_quant_config = overridden
    return base


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DeepseekV41Model(nn.Module):
    """embed -> hc_mult copies -> engram at L1/L14 -> blocks -> collapse."""

    def __init__(self, *, atom_config: Config, args: DeepseekV41Args):
        super().__init__()
        self.args = args
        self.hc_mult = args.hc_mult
        self.layer_map = V41LayerMap.from_args(args)
        self.shared = V41SharedRuntime()
        self.engram_layout = EngramLayout.from_args(args)
        self.engram_hash: NgramHashState | None = None
        self.model_path = getattr(atom_config, "model", None)

        self.embed = VocabParallelEmbedding(args.vocab_size, args.dim, prefix="embed")
        self.alt_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.layers = nn.ModuleList(
            [
                DeepseekV41Block(
                    layer_id,
                    args,
                    self.layer_map,
                    self.shared,
                    engram_layout=self.engram_layout,
                    prefix=f"layers.{layer_id}",
                    alt_stream=self.alt_stream,
                )
                for layer_id in range(args.n_layers)
            ]
        )
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.head = ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
        lookback = max(1, args.engram_max_ngram_size - 1)
        self.register_buffer(
            "engram_slots",
            torch.full((1, lookback), NgramHashState.DEAD, dtype=torch.int64),
            persistent=False,
        )
        per_rank = args.moe_inter_dim // max(1, get_tensor_model_parallel_world_size())
        if per_rank % 128:
            logger.info(
                "MoE intermediate %d per rank is not 128-aligned; FusedMoE pads "
                "it to %d.",
                per_rank,
                math.ceil(per_rank / 128) * 128,
            )

    def build_engram_hash(self, tokenizer=None) -> None:
        """Bind the tokenizer-derived n-gram hash state. Idempotent.

        With no tokenizer given, one is loaded from the model path.
        """
        if self.engram_layout is None or self.engram_hash is not None:
            return
        from atom.models.deepseek_v41_engram import cached_compressed_token_map

        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
        lookup, size = cached_compressed_token_map(tokenizer, self.model_path or "")
        self.engram_hash = NgramHashState(
            self.engram_layout, lookup, size, self.args.engram_pad_token_id
        )

    def _engram_hashes(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor | None:
        """One sequence per call: `engram_slots` holds a single lookback row."""
        if self.engram_layout is None:
            return None
        if self.engram_hash is None:
            self.build_engram_hash()
        if int(positions[0]) == 0:
            self.engram_slots.fill_(NgramHashState.DEAD)
        cu = torch.tensor([0, input_ids.numel()], device=input_ids.device)
        history = self.engram_hash.build_history(
            input_ids, positions, cu, self.engram_slots
        )
        return self.engram_hash(input_ids, positions, history)

    def _engram_hashes_paged(
        self, input_ids: torch.Tensor, positions: torch.Tensor, batch: V41Batch
    ) -> torch.Tensor | None:
        """The same lookback, one row per REQUEST rather than one per model.

        The n-gram history a step needs reaches back past the tokens the step
        holds, so it has to be per-request state. It is keyed by the request's
        V4 state slot -- the same slot the compressor rings and the sliding
        windows use, allocated and freed with the request -- rather than by a
        field of its own in the state pool, which would move the checkpoint
        layout for a table of three integers.
        """
        if self.engram_layout is None:
            return None
        if self.engram_hash is None:
            self.build_engram_hash()
        slots = batch.attn_md.state_slot_out[: batch.bs].long()
        self._grow_engram_slots(int(slots.max().item()) + 1 if batch.bs else 0)
        carried = self.engram_slots.index_select(0, slots)
        # A request whose first token this step is position 0 starts from no
        # history; the slot it took may hold a finished request's.
        fresh = torch.tensor(
            [p == 0 for p in batch.first_pos], device=input_ids.device
        ).unsqueeze(1)
        dead = torch.full_like(carried, NgramHashState.DEAD)
        carried = torch.where(fresh, dead, carried)
        history = self.engram_hash.build_history(
            input_ids,
            positions,
            batch.attn_md.cu_seqlens_q[: batch.bs + 1],
            carried,
        )
        self.engram_slots.index_copy_(0, slots, carried)
        return self.engram_hash(input_ids, positions, history)

    def _grow_engram_slots(self, rows: int) -> None:
        if rows <= self.engram_slots.shape[0]:
            return
        grown = torch.full(
            (rows, self.engram_slots.shape[1]),
            NgramHashState.DEAD,
            dtype=self.engram_slots.dtype,
            device=self.engram_slots.device,
        )
        grown[: self.engram_slots.shape[0]] = self.engram_slots
        self.engram_slots = grown

    @staticmethod
    def _open_batch(positions: torch.Tensor) -> tuple[V41Batch | None, bool]:
        """The forward's attention metadata, and whether this is warmup.

        No metadata on every path the paged attention cannot serve: the eager
        switch, and a bare model built outside the engine as
        `tools/dsv41/run_parity.py` builds one. Warmup has metadata but runs
        before `allocate_kv_cache`, so it gets neither path.
        """
        if _V41_EAGER_ATTN:
            return None, False
        try:
            fc = get_forward_context()
        except AssertionError:
            # get_forward_context asserts when no context is set, which is the
            # bare-model case.
            return None, False
        attn_md = getattr(fc, "attn_metadata", None)
        ctx = getattr(fc, "context", None)
        if attn_md is None or ctx is None:
            return None, False
        if ctx.is_dummy_run or getattr(attn_md, "state_slot_out", None) is None:
            return None, True
        return V41Batch(attn_md, positions), False

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """``[num_tokens]`` ids -> ``[num_tokens, dim]`` collapsed hidden state."""
        assert input_ids.dim() == 1, f"input_ids must be 1D, got {input_ids.shape}"
        batch, dummy = self._open_batch(positions)
        self.shared.batch = batch
        self.shared.dummy = dummy
        self.shared.block_candidates = {}
        self.shared.topk_local = None
        if batch is not None:
            hashes = self._engram_hashes_paged(input_ids, positions, batch)
        else:
            hashes = self._engram_hashes(input_ids, positions)
        h = self.embed(input_ids)
        h = h.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        pre_mix = h.new_zeros(h.shape[0], self.hc_mult, dtype=torch.float32)
        pre_mix[:, 0] = 1.0
        state = V41HCState(residual=h, pre_mix=pre_mix)
        for layer in self.layers:
            if layer.engram is not None:
                state.residual = layer.engram(
                    state.residual, hashes[:, layer.engram.layer_hash_index, :]
                )
            state = layer(state, positions)
        return DeepseekV41Block.hc_apply_pre(state.residual, state.pre_mix)


class DeepseekV41ForCausalLM(nn.Module):
    """ATOM model contract wrapper for DeepSeek-V4.1-Flash, text-only."""

    # Prefix-anchored: a substring rule for "norm.weight" also matches inside
    # attn_norm.weight / compressor.norm.weight and corrupts the lookup.
    weights_mapper = WeightsMapper(
        orig_to_new_prefix={
            "embed.": "model.embed.",
            "layers.": "model.layers.",
            "norm.weight": "model.norm.weight",
            "head.weight": "model.head.weight",
        },
        # The VL routing bias is never read on the text-only path.
        orig_to_new_suffix={".gate.bias_vl": None},
    )
    weights_mapping: ClassVar[dict[str, str]] = {
        ".gate.bias": ".gate.e_score_correction_bias",
        ".scale": ".weight_scale_inv",
    }
    packed_modules_mapping: ClassVar[dict[str, tuple[str, int]]] = {
        "shared_experts.w1": ("shared_experts.gate_up_proj", 0),
        "shared_experts.w3": ("shared_experts.gate_up_proj", 1),
    }
    skip_weight_prefixes: ClassVar[list[str]] = [
        "vision.",
        "aligner.",
        "mtp.",
        "image_start",
        "image_end",
        "image_newline",
    ]

    # Same disk names and the same FusedMoE dispatch as V4.
    get_expert_mapping = DeepseekV4ForCausalLM.get_expert_mapping
    load_weights = DeepseekV4ForCausalLM.load_weights
    disable_fused_shared_loading = DeepseekV4ForCausalLM.disable_fused_shared_loading

    def __init__(self, config: Config, prefix: str = "") -> None:
        super().__init__()
        if not getattr(config, "enforce_eager", False):
            raise NotImplementedError(
                "DeepSeek-V4.1 runs eager only for now. Its indexer walks the "
                "batch in python and reads each sequence's span back off the "
                "device, so a captured graph would replay one step's shapes "
                "and one step's spans forever. Pass --enforce-eager."
            )
        self.atom_config = config
        self.hf_config = config.hf_config
        self.args = DeepseekV41Args.from_hf_config(self.hf_config)
        self.args.quant_config = make_v41_quant_config(
            self.hf_config,
            model_path=getattr(config, "model", None),
            online_quant_config=getattr(config, "online_quant_config", None),
        )
        self.atom_config.quant_config = self.args.quant_config
        self.model = DeepseekV41Model(atom_config=config, args=self.args)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # V4.1 has no hc_head: the stream is already collapsed with the last
        # block's ffn `pre` by DeepseekV41Model.forward.
        return self.model.head.get_logits(self.model.norm(hidden_states))
