# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``DenseDecoderRanks4``: the communicators of the dense decoder layers on four TP ranks (built collectively, after
the weights load, before any CUDA graph capture) and the calls that use them. Each call picks the code layout its
consumer GEMM reads (``gemm.mxfp8_common.prequant_code_tile_rows`` of the consumer's (N, K)): the caller passes the
consumer's weight shape, not a tile."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

H = 6144
FC_K = 5 * H  # the input fc reads 5 taps
FC_N = 1536  # the fc's output columns per rank
WORLD = 4
MAX_ROWS = 2048  # the communicators' row bound (the fc goes in pieces of this many rows)


def _tile(m: int, consumer_shape) -> int:
    from amd_kernels.gemm.mxfp8_common import gemm as G

    n, k = consumer_shape
    return G.prequant_code_tile_rows(m, n, k, torch.bfloat16)


@dataclass(frozen=True)
class DenseLayer:
    """One layer's weights for ``DenseDecoderRanks4.forward`` (MXFP8 weights preshuffled)."""

    input_norm: torch.Tensor  # [6144] bf16
    qkv_shape: tuple[int, int]  # (N, K) of the qkv GEMM that reads the input norm's codes
    post_norm: torch.Tensor  # [6144] bf16
    gate_up_w: torch.Tensor
    gate_up_ws: torch.Tensor
    down_w: torch.Tensor
    down_ws: torch.Tensor


def stack_blocker(world: int, max_rows: int, on_library: Sequence[bool], fc_weight: torch.Tensor,
                  embedding: torch.Tensor, layers: Sequence[DenseLayer]) -> str | None:
    """Why ``DenseDecoderRanks4`` cannot run this layer stack (None: it can): ``world`` ranks, up to ``max_rows`` rows
    per call, ``on_library`` per linear whether it runs ``gemm.mxfp8_common`` on this block's codes, the replicated
    fc weight, this rank's embedding shard and every layer's weights."""
    from .mlp import mlp_supported

    if world != WORLD:
        return f"{world} ranks (four-rank communicators)"
    if not 1 <= max_rows <= MAX_ROWS:
        return f"{max_rows} rows per call (1-{MAX_ROWS})"
    if not all(on_library):
        return "a linear does not run gemm.mxfp8_common on this block's codes"
    if tuple(fc_weight.shape) != (H, FC_K) or fc_weight.dtype != torch.bfloat16:
        return f"fc weight {tuple(fc_weight.shape)} {fc_weight.dtype} (bf16 [{H}, {FC_K}])"
    e = embedding
    if e.dim() != 2 or e.shape[1] != H or e.dtype != torch.bfloat16 or not e.is_contiguous():
        return f"embedding {tuple(e.shape)} {e.dtype} (contiguous bf16 [V / 4, {H}])"
    if not layers:
        return "no layers"
    for i, layer in enumerate(layers):
        if not mlp_supported(layer.gate_up_w, layer.down_w):
            return f"layer {i}: FFN weights (MXFP8 gate_up [6144, 6144], down [6144, 3072])"
        if layer.qkv_shape[1] != H:
            return f"layer {i}: qkv K {layer.qkv_shape[1]}"
        for w in (layer.input_norm, layer.post_norm):
            if tuple(w.shape) != (H,) or w.dtype != torch.bfloat16 or not w.is_contiguous():
                return f"layer {i}: norm weight {tuple(w.shape)} {w.dtype} (contiguous bf16 [{H}])"
    return None


def context_kv_blocker(proj_on_library: bool, proj_w: torch.Tensor, rotary_dim: int, neox: bool,
                       cos_sin_cache: torch.Tensor, kv_caches: list[torch.Tensor], k_scales: list[torch.Tensor],
                       v_scales: list[torch.Tensor]) -> str | None:
    """Why ``store_context_kv`` cannot store these layers' context K / V (None: it can): the projection on
    ``gemm.mxfp8_common`` with every layer's K / V rows, the RoPE (``rope_blocker``), the caches and scales
    (``supports_multilayer``)."""
    from amd_kernels.attention.norm_rope_fp8_insert_d128_rot64 import rope_blocker
    from amd_kernels.attention.norm_rope_fp8_insert_d128_rot64.kv_norm_rope_insert_multilayer import (
        supports_multilayer,
    )

    if not proj_on_library:
        return "the context projection does not run gemm.mxfp8_common"
    why = rope_blocker(rotary_dim, neox, cos_sin_cache)
    if why is not None:
        return why
    if not supports_multilayer(kv_caches, k_scales, v_scales):
        return "the KV caches (equal e4m3 [pages, kv_heads, page, 256]) or their scales (fp32)"
    width = len(kv_caches) * 2 * kv_caches[0].shape[1] * 128
    if tuple(proj_w.shape) != (width, H):
        return f"context projection {tuple(proj_w.shape)} (every layer's K / V rows: [{width}, {H}])"
    return None


class DenseDecoderRanks4:
    """The input fc split by N (``bf16_concat_linear_n1536_k30720``'s ``ConcatLinearComm``; eager), the owner read of
    the four vocabulary shards (``vocab_gather_h6144``) and the all-reduce + Gemma norm tails
    (``allreduce_gemma_bf16_mxfp8_h6144``'s ``Bf16ArNormComm``). Built and warmed up collectively: every rank, the
    same order. ``fc_weight`` is the replicated [6144, 30720] weight (this rank keeps its 1536 rows, packed; the
    caller may free the replicated one), ``embedding`` this rank's vocabulary shard [V / 4, 6144]."""

    def __init__(self, rank: int, group, device, *, fc_weight: torch.Tensor, embedding: torch.Tensor,
                 max_rows: int = MAX_ROWS):
        from amd_kernels.embedding.vocab_gather_gemma_mxfp8_h6144 import warmup_vocab_gather_gemma_mxfp8
        from amd_kernels.embedding.vocab_gather_h6144 import VocabGather
        from amd_kernels.gemm.bf16_concat_linear_n1536_k30720 import pack_weight
        from amd_kernels.gemm.bf16_concat_linear_n1536_k30720.comm import ConcatLinearComm
        from amd_kernels.norm.allreduce_gemma_bf16_mxfp8_h6144 import Bf16ArNormComm

        assert tuple(fc_weight.shape) == (H, FC_K) and fc_weight.dtype == torch.bfloat16
        assert 1 <= max_rows <= MAX_ROWS
        self.rank, self.group, self.device, self.max_rows = rank, group, torch.device(device), max_rows
        self.fc_weight = pack_weight(fc_weight[rank * FC_N:(rank + 1) * FC_N])
        self.fc_comm = ConcatLinearComm(self.fc_weight, rank, group, max_rows=max_rows)
        self.vocab = VocabGather(embedding, rank, group)
        self.ar = Bf16ArNormComm(rank, group, self.device)
        # collective warm-up (real exchanges on every rank): the fc tiers, the lookups, the 4 row tiers of the
        # all-reduce + norm in both layouts and residual forms
        w = torch.ones(H, dtype=torch.bfloat16, device=self.device)
        self.fc_comm.warm_up()
        self.vocab.warm_up()
        warmup_vocab_gather_gemma_mxfp8(self.vocab, w)
        self.ar.warm_up(w)
        torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------ the input fc
    def fc(self, taps: list[torch.Tensor], num_tokens: int, out: torch.Tensor) -> None:
        """out[:num_tokens] = cat(taps)[:num_tokens] @ fc^T (each [>= num_tokens, 6144] bf16, unit column stride,
        row stride a multiple of 8, read in place); every rank the same call, eager, in pieces of ``max_rows``."""
        assert len(taps) == 5, len(taps)
        for t in taps:
            assert t.dtype == torch.bfloat16 and t.dim() == 2 and t.shape[1] == H and t.shape[0] >= num_tokens
            assert t.stride(1) == 1 and t.stride(0) % 8 == 0, t.stride()
        for i in range(0, num_tokens, self.max_rows):
            j = min(i + self.max_rows, num_tokens)
            self.fc_comm([t[i:j] for t in taps], out[i:j])

    def fc_cat(self, x: torch.Tensor) -> torch.Tensor:
        """fc of the concatenated taps ``x`` [rows, 30720] bf16."""
        m = x.shape[0]
        out = torch.empty((m, H), dtype=x.dtype, device=x.device)
        self.fc([x[:, i * H:(i + 1) * H] for i in range(5)], m, out)
        return out

    # ------------------------------------------------------------ embedding, all-reduce tails
    def embed_norm_quant(self, ids: torch.Tensor, norm_weight: torch.Tensor, eps: float, consumer_shape):
        """(codes, scales, residual): the embedding rows of ``ids`` (the owner reads, no all-reduce), their Gemma
        norm and its MXFP8 codes in the layout of the consumer GEMM (N, K = ``consumer_shape``)."""
        from amd_kernels.embedding.vocab_gather_gemma_mxfp8_h6144 import vocab_gather_gemma_mxfp8

        return vocab_gather_gemma_mxfp8(ids, self.vocab, norm_weight, eps, _tile(ids.numel(), consumer_shape))

    def ar_norm_quant(self, x: torch.Tensor, residual: torch.Tensor, norm_weight: torch.Tensor, eps: float,
                      consumer_shape):
        """(codes, scales, residual) of all-reduce(x) + residual -> Gemma norm -> MXFP8 in the consumer's layout;
        the residual stays on each rank's owned rows (pass the previous tail's residual)."""
        return self.ar.ar_add_rmsnorm_mxfp8(x, residual, norm_weight, eps, _tile(x.shape[0], consumer_shape),
                                            owned_residual=True)

    def ar_norm(self, x: torch.Tensor, residual: torch.Tensor, norm_weight: torch.Tensor, eps: float):
        """(bf16 norm output, residual) of the last tail."""
        return self.ar.ar_add_rmsnorm(x, residual, norm_weight, eps, owned_residual=True)

    def forward(self, ids: torch.Tensor, layers: Sequence[DenseLayer],
                attn: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor], final_norm: torch.Tensor, eps: float,
                swiglu: tuple[float, float, float], form: str = "bf16_chain") -> torch.Tensor:
        """The layer stack on ``ids`` (1-2048): embedding + norm + quant, then per layer ``attn(i, codes, scales)``
        (the caller's attention + o_proj on the qkv GEMM's codes, its TP partial [m, 6144] bf16), the all-reduce +
        norm + quant tail, the MLP block (``swiglu`` = (alpha, beta, limit), ``form``) and the next tail; the last
        tail's bf16 norm output [m, 6144] with ``final_norm``."""
        from .mlp import mxfp8_swiglu_oai_mlp

        ids = ids.reshape(-1)
        m = ids.numel()
        q, s, residual = self.embed_norm_quant(ids, layers[0].input_norm, eps, layers[0].qkv_shape)
        for i, layer in enumerate(layers):
            o = attn(i, q, s)
            q, s, residual = self.ar_norm_quant(o, residual, layer.post_norm, eps, layer.gate_up_w.shape)
            d = mxfp8_swiglu_oai_mlp(q, s, m, layer.gate_up_w, layer.gate_up_ws, layer.down_w, layer.down_ws,
                                     *swiglu, form=form)
            if i + 1 == len(layers):
                return self.ar_norm(d, residual, final_norm, eps)[0]
            nxt = layers[i + 1]
            q, s, residual = self.ar_norm_quant(d, residual, nxt.input_norm, eps, nxt.qkv_shape)
        raise ValueError("no layers")

    @staticmethod
    def store_context_kv(states: torch.Tensor, hidden_norm: torch.Tensor, eps: float, proj_w: torch.Tensor,
                         proj_ws: torch.Tensor, positions: torch.Tensor, k_norm: torch.Tensor, k_eps: float,
                         cos_sin_cache: torch.Tensor, kv_caches: list[torch.Tensor], slot_mapping: torch.Tensor,
                         k_scales: list[torch.Tensor], v_scales: list[torch.Tensor]) -> None:
        """The context K / V of every layer into its fp8 cache: ``gemma_rmsnorm_mxfp8_linear(states, hidden_norm,
        eps, proj_w, proj_ws)`` (all layers' K / V rows), then
        ``attention/norm_rope_fp8_insert_d128_rot64``'s ``kv_norm_rope_insert_multilayer`` at ``slot_mapping``."""
        from amd_kernels.attention.norm_rope_fp8_insert_d128_rot64.kv_norm_rope_insert_multilayer import (
            kv_norm_rope_insert_multilayer,
        )

        from .norm_linear import gemma_rmsnorm_mxfp8_linear

        if states.shape[0] == 0:
            return
        kv = gemma_rmsnorm_mxfp8_linear(states, hidden_norm, eps, proj_w, proj_ws)
        kv_norm_rope_insert_multilayer(kv, positions, k_norm, cos_sin_cache, kv_caches,
                                       [slot_mapping] * len(kv_caches), k_scales, v_scales, k_eps)

    def capture(self):
        """Around the CUDA graph capture of the calls (every rank, the same graphs): the all-reduce tails recorded
        inside read their peers' inputs in place."""
        return self.ar.capture()

    def close(self) -> None:
        for c in (self.fc_comm, self.vocab, self.ar):
            c.close()
