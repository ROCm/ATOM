# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The hand-offs of a sequence-parallel decoder layer on four ranks (hidden 6144, 2048 attention-output columns per
rank): the multi-kernel chains between the head-sharded parts of a layer (qkv, attention, the FFN of every token) and
the token-sharded residual stream (rank r holds ``rows`` consecutive tokens, ``rows`` a multiple of 32).

``SeqParallelHandoffs`` owns the four communicators of ``pull_collectives_ranks4`` (and the pools of the unpacking
all-to-all), the per-phase pool slots, the MXFP8 pack of the attention output
(``mxfp8_pack_q2048``) and the warm-up of every kernel the phases launch (``amd_kernels.warm_up.entry``). The caller
passes tensors: this rank's rows, the layer's weights, epsilons. Per layer:

1. ``input_gather(x)``: the first layer's normed bf16 rows -> MXFP8 codes + AITER-shuffled scales of every token
   (``gemm/mxfp8_common``'s quantizer on the shard, then ``AllGatherComm``).
2. ``attention_moe(attn_output, o_weight, o_scale, residual, norm_weight, eps, moe_weights)``: this rank's heads of
   every token -> the pack into the pool slot, ``FencedAllGatherComm.all_to_all_unpack_shuffled`` (the whole o_proj
   input of this rank's rows, scales in the GEMM's layout), o_proj_full on ``mxfp8_gemm``, ``gemma_add_rmsnorm``
   (bf16), ``prepare_mxfp4_moe`` + its lossless compact payload, gathered (``AllGatherComm``) -> the
   ``MoeInputCompact`` of every token for ``mxfp4_moe(..., return_partials=True)``, and the new residual shard.
   ``attention_dense``: the same o_proj, then ``NormAllGatherComm`` (add + Gemma RMSNorm + MXFP8 quant of the fp32
   norm + gather in one launch) -> the dense MLP's codes + scales of every token, the residual shard.
3. ``ffn_norm_gather(ffn_out, residual, norm_weight, eps)``: the FFN output of every token (MoE: its ``(slots,
   shared)`` partials; dense: the bf16 partial) -> the reduce-scatter (fp32 in rank order), the next layer's input
   norm + MXFP8 quant on this rank's rows, gathered: ``SlotsReduceScatterComm.slots4_reduce_scatter_add_rmsnorm_mxfp8``
   + ``AllGatherComm``, or the dense reduce-scatter + ``NormAllGatherComm``.
   After the last layer ``ffn_reduce_scatter(ffn_out)`` gives this rank's summed rows for the model's final norm, and
   ``gather(x)`` gathers bf16 rows (the normed output, an aux hidden state).

Every collective moves bf16 partial sums or quantized next-GEMM inputs (no quantized reduction, no atomics). The
phases run on the caller's current stream, eagerly (the pool slots carry host-side sequence state; not inside CUDA
graph capture); every rank calls them with the same shapes in the same order.

Warm-up (root README rule 4): the communicators compile their transport kernels at construction (a collective);
``warm_up(o_weight, o_scale, moe_weights)`` makes the first call of each compute mode (the quantizer, o_proj, o_proj +
norm + MoE preparation, the pack, the compact partials mode of ``mxfp4_moe``), each of which compiles every variant of
its rows range first (``ROWS``; ``amd_kernels.warm_up``). The compute modes run no collective, so a rank's warm-up
never waits for another.
"""

import torch

from amd_kernels.warm_up import entry

from .mxfp8_pack_q2048 import ROW_BYTES, mxfp8_pack_q2048

WORLD = 4
H = 6144
Q = 2048  # attention output per rank: 16 heads x 128
ROW_ALIGN = 32  # shard rows: whole 32-row scale groups (NormAllGatherComm, the GEMMs' scale layout)
# The compact MoE payload's gather: 8192 rows per rank (gather_mxfp4_moe's bound).
MAX_TOKENS = 32768
MAX_ROWS = MAX_TOKENS // WORLD
# The served range starts at 4096 tokens (1024 rows per rank): the warm-up covers MIN_ROWS..MAX_ROWS.
MIN_TOKENS = 4096
MIN_ROWS = MIN_TOKENS // WORLD
_FP8 = torch.float8_e4m3fn

# MoeInputCompact's fields in gather order: (columns, dtype)
_COMPACT = (
    (H // 32, torch.uint8),  # routed (MXFP4) scales
    (4, torch.float32),  # top-4 weights (routed scale included)
    (4, torch.int32),  # top-4 ids
    (H, torch.uint8),  # shared-expert MXFP8 codes
    (H // 32, torch.uint8),  # their scales
    (H // 8, torch.uint8),  # one correction bit per value (exact MXFP4 codes)
)


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def shard_rows(num_tokens: int) -> int:
    """Rows per rank of a ``num_tokens`` call (``ROW_ALIGN``-aligned, the last ranks' rows padded with zeros)."""
    return _round_up(-(-num_tokens // WORLD), ROW_ALIGN)


def shard(x: torch.Tensor, rank: int) -> torch.Tensor:
    """Rank ``rank``'s ``shard_rows`` rows of every token's ``x`` ``[t, ...]``, zero padded past ``t``."""
    rows = shard_rows(x.shape[0])
    start = rank * rows
    n = min(rows, max(0, x.shape[0] - start))
    if n == rows:
        return x[start : start + n].contiguous()
    out = x.new_zeros((rows, *x.shape[1:]))
    if n:
        out[:n].copy_(x[start : start + n])
    return out


# The shard compute of the phases (no collective): one registered entry, a mode per kind.


def _shard_compute(kind, x, scales=None, o_weight=None, o_scale=None, residual=None, norm_weight=None, eps=0.0,
                   moe_weights=None):  # fmt: skip
    """``input``: bf16 rows ``x`` -> MXFP8 codes + shuffled scales. ``dense``: o_proj_full of the unpacked codes ``x``
    + ``scales`` -> bf16. ``moe``: that o_proj, the bf16 add + Gemma RMSNorm with ``residual``, the MoE preparation ->
    (compact payload, new residual)."""
    from amd_kernels.gemm.mxfp8_common.gemm import mxfp8_e4m3_quantize_flydsl, mxfp8_gemm

    if kind == "input":
        return mxfp8_e4m3_quantize_flydsl(x)
    h = mxfp8_gemm(x, scales, o_weight, o_scale, torch.bfloat16, activation_scale_preshuffled=True)
    if kind == "dense":
        return h
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import prepare_mxfp4_moe
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm.host import gemma_add_rmsnorm

    y, r = gemma_add_rmsnorm(h, residual, norm_weight, eps)
    return prepare_mxfp4_moe(y, moe_weights).compact(), r


def _shard_mode(kind, *args, moe_weights=None, **kwargs):
    """What a call fixes about its kernels besides the row count (``amd_kernels.warm_up``)."""
    if kind != "moe":
        return (kind,)
    return kind, moe_weights.shared_weights is not None, moe_weights.router_weight_lines is not None


def _gemm_rows() -> tuple[int, ...]:
    """One row count per o_proj_full variant (``mxfp8_gemm``'s table row and its four-wave remainder plan) over the
    served rows: what ``warmup_mxfp8_gemm_weight`` walks, for this one shape and row-major codes."""
    from amd_kernels.gemm.mxfp8_common.gemm import _FOURWAVE_KERNELS, _fourwave_plan, _select_config

    seen = {}
    for m in range(MIN_ROWS, MAX_ROWS + 1, ROW_ALIGN):
        config = _select_config(m, H, WORLD * Q, torch.bfloat16, False)
        key: tuple = (config,)
        if config.kernel in _FOURWAVE_KERNELS:
            key += tuple(tiles for tiles, _, _ in _fourwave_plan(m, H, config))
        seen.setdefault(key, m)
    return tuple(sorted(seen.values()))


def _shard_sizes(mode) -> tuple[int, ...]:
    # the quantizer's variant is the same from 1024 rows (rows per wave by row count)
    return (MIN_ROWS,) if mode[0] == "input" else _gemm_rows()


def _shard_inputs(mode, n, kind, x, scales=None, o_weight=None, o_scale=None, residual=None, norm_weight=None,
                  eps=0.0, moe_weights=None):  # fmt: skip
    """One warm-up call of the mode at ``n`` rows (zeros: the dispatch depends on shapes only)."""
    dev = x.device
    if kind == "input":
        return ("input", torch.zeros((n, H), dtype=torch.bfloat16, device=dev)), {}
    codes = torch.zeros((n, WORLD * Q), dtype=_FP8, device=dev)
    sc = torch.full((_round_up(n, 32), WORLD * Q // 32), 127, dtype=torch.uint8, device=dev)
    if kind == "dense":
        return (kind, codes, sc, o_weight, o_scale), {}
    res = torch.zeros((n, H), dtype=torch.bfloat16, device=dev)
    return (kind, codes, sc, o_weight, o_scale, res, norm_weight, eps), {"moe_weights": moe_weights}


shard_compute = entry("collectives/seq_parallel_handoffs_h6144_q2048_ranks4", mode=_shard_mode, sizes=_shard_sizes,
                      inputs=_shard_inputs)(_shard_compute)  # fmt: skip


def _pack_mode(x, out=None, **kwargs):
    return tuple(sorted(kwargs.items()))


def _pack_inputs(mode, n, x, out=None, **kwargs):
    return (torch.zeros((n, Q), dtype=torch.bfloat16, device=x.device),), kwargs


# one variant (rows are a runtime argument)
pack = entry("collectives/seq_parallel_handoffs_h6144_q2048_ranks4:pack", mode=_pack_mode, sizes=lambda mode: (32,),
             inputs=_pack_inputs)(mxfp8_pack_q2048)  # fmt: skip


class SeqParallelHandoffs:
    """The four communicators of the phases, one of each per rank, sized for ``max_tokens`` tokens per call (a
    server's batch-token limit, clamped to ``MIN_TOKENS``..``MAX_TOKENS``). Collective: every rank of the four-rank
    ``cpu_group`` (gloo) constructs it at the same point; it compiles the transport kernels (``warm_up`` of each
    communicator, real collectives on zero payloads)."""

    def __init__(self, rank: int, cpu_group, device: torch.device, max_tokens: int = MAX_TOKENS):
        from amd_kernels.collectives.pull_collectives_ranks4 import (
            AllGatherComm,
            FencedAllGatherComm,
            NormAllGatherComm,
            Pools,
            SlotsReduceScatterComm,
        )
        from amd_kernels.collectives.pull_collectives_ranks4.pull_unpack_q2048_shuffled import FENCE_MAX_TOKENS

        max_tokens = min(max(max_tokens, MIN_TOKENS), MAX_TOKENS)
        device = torch.device(device)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.max_tokens = max_tokens
        self.rows = shard_rows(max_tokens)
        self.padded = WORLD * self.rows
        # the transport warm-up's larger unpack (FENCE_MAX_TOKENS // 4 + 32 rows per rank)
        a2a_tokens = max(self.padded, FENCE_MAX_TOKENS + WORLD * 32)
        self.pools = Pools(rank, WORLD, cpu_group, device, _round_up(a2a_tokens * ROW_BYTES, 256), sets=1)
        self.a2a = FencedAllGatherComm(self.pools)
        # the widest field set: one bf16 [rows, H] (aux / final output); the MoE payload is 7328 B/row in six 256 B
        # aligned fields
        self.ag = AllGatherComm(rank, cpu_group, device, _round_up(self.rows * 2 * H + 6 * 256, 256))
        self.rs = SlotsReduceScatterComm(rank, cpu_group, device, max_rows=self.padded)
        self.norm_gather = NormAllGatherComm(rank, cpu_group, device, max_rows=self.rows)
        for c in (self.a2a, self.ag, self.rs, self.norm_gather):
            c.warm_up()
        torch.accelerator.synchronize()

    def supports(self, num_tokens: int) -> bool:
        """Whether a ``num_tokens`` call fits the communicators."""
        return 0 < num_tokens <= self.max_tokens

    def warm_up(self, o_weight: torch.Tensor, o_scale: torch.Tensor, moe_weights=None) -> None:
        """Compile every compute variant the phases launch, before serving (module docstring): ``o_weight`` /
        ``o_scale`` a layer's preshuffled o_proj_full (``gemm.mxfp8_common``), ``moe_weights`` a MoE layer's ``MoeWeights``
        (the dispatch depends on shapes only, so one layer serves all)."""
        dev = self.device
        x = torch.zeros((MIN_ROWS, H), dtype=torch.bfloat16, device=dev)
        shard_compute("input", x)
        codes = torch.zeros((MIN_ROWS, WORLD * Q), dtype=_FP8, device=dev)
        sc = torch.full((MIN_ROWS, WORLD * Q // 32), 127, dtype=torch.uint8, device=dev)
        shard_compute("dense", codes, sc, o_weight, o_scale)
        pack(torch.zeros((ROW_ALIGN, Q), dtype=torch.bfloat16, device=dev))
        if moe_weights is not None:
            from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import mxfp4_moe

            w = torch.zeros(H, dtype=torch.bfloat16, device=dev)
            compact, _ = shard_compute("moe", codes, sc, o_weight, o_scale, x, w, 1e-6, moe_weights=moe_weights)
            # the consumer of the gathered payload: its compact partials mode
            mxfp4_moe(compact, moe_weights, return_partials=True)
        torch.accelerator.synchronize()

    def _gathered(self, rows: int, cols: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.empty((WORLD * rows, cols), dtype=dtype, device=self.device)

    def input_gather(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """This rank's normed bf16 rows ``[rows, H]`` -> MXFP8 codes ``[4 rows, H]`` e4m3 + AITER-shuffled scales
        ``[4 rows, H / 32]`` of every token (the quantizer of the non-SP first layer, gathered)."""
        rows = x.shape[0]
        codes, scales = self._gathered(rows, H, _FP8), self._gathered(rows, H // 32, torch.uint8)
        q, s = shard_compute("input", x)
        self.ag.all_gather([codes, scales], [q, s])
        return codes, scales

    def _unpacked(self, attn_output: torch.Tensor, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        """This rank's heads of every token ``[t, Q]`` -> the o_proj input of this rank's rows: codes ``[rows, 4 Q]``,
        scales in o_proj_full's GEMM layout."""
        from amd_kernels.collectives.pull_collectives_ranks4 import Collectives

        t, d = attn_output.shape
        padded = WORLD * rows
        assert d == Q and t <= padded, (attn_output.shape, rows)
        coll = Collectives(self.pools, 0)
        coll.bind(torch.cuda.current_stream(self.device))
        send = coll.alloc((padded, ROW_BYTES), torch.uint8)
        assert send is not None, "the all-to-all slot is sized for every call up to max_tokens"
        pack(attn_output, out=send[:t])
        send[t:].zero_()
        return self.a2a.all_to_all_unpack_shuffled(coll, send)

    def attention_moe(self, attn_output, o_weight, o_scale, residual, norm_weight, eps, moe_weights):
        """The o_proj, the post-attention norm and the MoE input preparation of this rank's rows (``residual`` its
        ``[rows, H]`` residual shard) -> (``MoeInputCompact`` of every token, the new residual shard)."""
        from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import MoeInputCompact

        rows = residual.shape[0]
        outs = [self._gathered(rows, cols, dtype) for cols, dtype in _COMPACT]
        xq, xs = self._unpacked(attn_output, rows)
        local, r = shard_compute("moe", xq, xs, o_weight, o_scale, residual, norm_weight, eps, moe_weights=moe_weights)
        self.ag.all_gather(outs, list(local.tensors()))
        return MoeInputCompact(*outs), r

    def attention_dense(self, attn_output, o_weight, o_scale, residual, norm_weight, eps):
        """The o_proj and the post-attention norm + MXFP8 quant of this rank's rows -> (codes ``[4 rows, H]``, scales,
        the new residual shard): the dense MLP's input of every token."""
        rows = residual.shape[0]
        codes, scales = self._gathered(rows, H, _FP8), self._gathered(rows, H // 32, torch.uint8)
        xq, xs = self._unpacked(attn_output, rows)
        h = shard_compute("dense", xq, xs, o_weight, o_scale)
        _, _, r = self.norm_gather.norm_all_gather(h, residual, norm_weight, eps, q_out=codes, scale_out=scales)
        return codes, scales, r

    def ffn_norm_gather(self, ffn_out, residual, norm_weight, eps):
        """The FFN output of every token (MoE: ``(slots [4 rows, 4, H], shared [4 rows, H])``; dense: the bf16 partial
        ``[4 rows, H]``) -> reduce-scatter, the next layer's input norm + MXFP8 quant on this rank's rows, gathered:
        (codes ``[4 rows, H]``, scales, the new residual shard)."""
        rows = residual.shape[0]
        codes, scales = self._gathered(rows, H, _FP8), self._gathered(rows, H // 32, torch.uint8)
        if isinstance(ffn_out, tuple):
            q, s, r = self.rs.slots4_reduce_scatter_add_rmsnorm_mxfp8(*ffn_out, residual, norm_weight, eps)
            self.ag.all_gather([codes, scales], [q, s])
        else:
            _, _, r = self.norm_gather.norm_all_gather(
                self.rs.reduce_scatter(ffn_out), residual, norm_weight, eps, q_out=codes, scale_out=scales
            )
        return codes, scales, r

    def ffn_reduce_scatter(self, ffn_out) -> torch.Tensor:
        """The FFN output of every token (as ``ffn_norm_gather``) -> this rank's summed bf16 rows ``[rows, H]``."""
        if isinstance(ffn_out, tuple):
            return self.rs.slots4_reduce_scatter(*ffn_out)
        return self.rs.reduce_scatter(ffn_out)

    def gather(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's bf16 rows ``[rows, H]`` -> every token's ``[4 rows, H]``."""
        out = self._gathered(x.shape[0], x.shape[1], x.dtype)
        self.ag.all_gather([out], [x])
        return out
