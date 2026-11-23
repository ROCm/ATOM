# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sequence-parallel (SP) prefill for MiniMax-M3 under TP.

A prefill-only forward of at least ``ATOM_M3_SP_PREFILL_MIN_TOKENS`` tokens keeps
1/tp of the residual stream on each rank (rank r: rows [r*rows, (r+1)*rows), the
last rank zero padded). Per layer the two TP all-reduces become hand-offs::

    add + Gemma RMSNorm of the rank's rows -> all-gather
    -> qkv + attention of every token (this rank's heads, as under TP)
    -> all-to-all of the attention output -> the whole o_proj on the rank's rows
    -> add + Gemma RMSNorm -> all-gather -> the FFN of every token (TP partial)
    -> reduce-scatter

``o_proj_full`` is the checkpoint's o_proj replicated on every rank. Without it
(``ATOM_M3_SP_PREFILL_OPROJ=tp``) the TP o_proj's partial is reduce-scattered
instead. Decode, smaller prefills and CUDA-graph steps keep the TP path.

Wires: the reduce-scatters sum bf16 partials, by default with amd_kernels' pull
collectives (fp32 sums in rank order, one rounding, like TP's custom all-reduce;
RCCL's ring rounds at every hop). Gathers carry bf16, or with
``ATOM_M3_SP_PREFILL_QGATHER=1`` the MXFP8 codes + scales the consuming GEMM
would compute from those bf16 rows itself (per-row 1x32 blocks, so the same
bytes either side of the wire).

P1 switches (p1/prefill-sp-tbo, default off; each keeps the bytes of the path it
replaces, see /shared_nfs/ygan/p1/prefill-sp-tbo/NOTES.md):

- ATOM_P1_SP_HANDOFFS=1: the communicators of amd_kernels'
  ``SeqParallelHandoffs`` (all-gather, reduce-scatter, the unpacking all-to-all),
  shards of a multiple of 32 rows (more zero padding rows, sliced off as before).
  Needed by the three below.
- ATOM_P1_SP_QG_AK=1 (with QGATHER): the shard is quantized by amd_kernels'
  quantizer (the one ``mxfp8_linear`` runs on bf16 rows at prefill sizes), codes
  and shuffled scales gathered in one call, qkv on ``mxfp8_gemm`` without a
  re-quant (QGATHER alone routes qkv to AITER's GEMM).
- ATOM_P1_SP_A2A_UNPACK=1 (with QGATHER + o_proj_full + ATOM_P1_SP_OPROJ_AK): the
  attention output is packed to MXFP8 (``mxfp8_pack_q2048``, the same quantizer
  rule) and the unpacking all-to-all writes o_proj_full's codes and shuffled
  scales; replaces pad + quantize + 2 RCCL all-to-alls + 2 permute copies.
- ATOM_P1_SP_MOE_SLOTS=1: amd_kernels MoE layers return their partials (slots +
  shared) and ``slots4_reduce_scatter`` sums them inside the reduce-scatter
  (fp32 ((((s0+s1)+s2)+s3)+shared) rounded once, then the rank-order fp32 sum:
  the reduced mode's arithmetic followed by the pull reduce-scatter's).
- ATOM_P1_SP_MOE_COMPACT=1 (with MOE_SLOTS): the MoE input is routed and
  quantized on this rank's rows (amd_kernels' prepare kernels, with the router
  K split of the full-M call, so the routing and the quantized bytes equal the
  bf16 entry's) and the lossless compact payload (7328 B/token: shared MXFP8
  codes + scales, MXFP4 scales, one correction bit per value, top-4) is
  gathered instead of the bf16 rows (12288 B/token), as amd_kernels'
  ``SeqParallelHandoffs.attention_moe`` does (sp_moe_compact.py), from
  ATOM_P1_SP_MOE_COMPACT_MIN_TOKENS (default 4096) padded tokens.
- ATOM_P1_SP_SHARED_LOCAL=1 (with MOE_SLOTS; p1/sp-local-shared): every rank
  holds the full shared expert (ak_moe gathers the TP shards at load) and runs it
  on its OWN rows; the gather carries the routed MXFP4 payload only (3296
  B/token: codes, scales, top-4); the reduce-scatter's partials carry the slot
  sums only and the owner adds its shared output in the final fp32 sum
  (p1_sls_shared.sum5). Routed bytes and slots equal the compact path's; the
  layer output does not (one fp32 sum over I 3072, rounded once, instead of four
  bf16 TP partials folded into the partials): GSM8K gate.
  From ATOM_P1_SP_SHARED_LOCAL_MIN_TOKENS (default 2048) padded tokens; side
  stream from ATOM_P1_SP_SHARED_LOCAL_SIDE_ROWS own rows (default 1; 0: serial).
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from aiter import rmsnorm2d_fwd_with_add
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)

from atom.model_ops.blockscale import quantize_fp8
from atom.model_ops.embed_head import masked_embedding
from atom.model_ops.linear import ReplicatedLinear
from atom.utils import envs
from atom.utils.forward_context import get_forward_context


def enabled() -> bool:
    return (
        envs.ATOM_M3_SP_PREFILL
        and get_tensor_model_parallel_world_size() > 1
        and get_pp_group().world_size == 1
    )


def make_o_proj_full(o_proj, input_size: int, output_size: int, quant_config, prefix):
    """The whole o_proj on this rank, loaded from the same checkpoint tensors as
    the TP ``o_proj`` (its loaders feed both), or None when SP prefill is off."""
    if not enabled() or envs.ATOM_M3_SP_PREFILL_OPROJ != "full":
        return None
    full = ReplicatedLinear(
        input_size, output_size, bias=False, quant_config=quant_config, prefix=prefix
    )
    for name, param in o_proj.named_parameters(recurse=False):
        twin = getattr(full, name, None)
        if twin is None:
            continue

        def load_both(p, w, *args, _load=param.weight_loader, _twin=twin):
            _load(p, w, *args)
            _twin.weight_loader(_twin, w)

        param.weight_loader = load_both
    return full


class ShardedOutput(tuple):
    """A decoder layer's SP return: (reduced FFN rows, residual rows) of this rank."""

    sp: "SeqParallelPrefill"


class SeqParallelPrefill:
    def __init__(self, model, max_tokens: int):
        self.model = model
        tp = get_tp_group()
        self.tp, self.world, self.rank = tp, tp.world_size, tp.rank_in_group
        self.min_tokens = envs.ATOM_M3_SP_PREFILL_MIN_TOKENS
        self.qgather = envs.ATOM_M3_SP_PREFILL_QGATHER
        self.max_rows = -(-max_tokens // self.world)
        self.pull = envs.ATOM_M3_SP_PREFILL_COMM == "pull" and self._pull_available()
        self._ag = self._rs = None
        # P1 switches (module docstring)
        self.p1 = envs.ATOM_P1_SP_HANDOFFS and self.pull
        self.p1_qg_ak = self.p1 and self.qgather and envs.ATOM_P1_SP_QG_AK
        self.p1_unpack = (
            self.p1
            and self.qgather
            and envs.ATOM_P1_SP_A2A_UNPACK
            and envs.ATOM_P1_SP_OPROJ_AK
            and envs.ATOM_M3_SP_PREFILL_OPROJ == "full"
        )
        self.p1_slots = self.p1 and envs.ATOM_P1_SP_MOE_SLOTS
        self.p1_compact = self.p1_slots and envs.ATOM_P1_SP_MOE_COMPACT
        self._compact = None
        self.compact_min = envs.ATOM_P1_SP_MOE_COMPACT_MIN_TOKENS
        self.p1_shared_local = self.p1_slots and envs.ATOM_P1_SP_SHARED_LOCAL
        self.shared_local_min = envs.ATOM_P1_SP_SHARED_LOCAL_MIN_TOKENS
        self.shared_local_side = envs.ATOM_P1_SP_SHARED_LOCAL_SIDE_ROWS
        if self.p1_compact or self.p1_shared_local:
            from atom.models.minimax_m3.sp_moe_compact import CompactMoeHandoff

            self._compact = CompactMoeHandoff()
        self._ho = None
        self.n = self.rows = 0
        self.dummy = False

    @classmethod
    def create(
        cls, model, consumes_per_token_fp8, max_tokens: int
    ) -> "SeqParallelPrefill | None":
        """For a ``MiniMaxM3Model`` whose layers all have an ``attend`` (indexed
        sparse attention) and MXFP8 / bf16 linears, else None."""
        if not enabled():
            return None
        for layer in model.layers[model.start_layer : model.end_layer]:
            attn = layer.self_attn
            if not hasattr(attn, "attend") or any(
                consumes_per_token_fp8(m)
                for m in (attn.qkv_proj, attn.o_proj)
                + (() if layer.is_moe_layer else (layer.mlp.gate_up_proj,))
            ):
                return None
        return cls(model, max_tokens)

    def supports(self, input_ids, positions, intermediate_tensors, inputs_embeds):
        if (
            intermediate_tensors is not None
            or inputs_embeds is not None
            or input_ids.numel() < self.min_tokens
            or self.model.aux_hidden_state_layers
            or torch.cuda.is_current_stream_capturing()
        ):
            return False
        fc = get_forward_context()
        ctx = fc.context
        if self.p1 and input_ids.numel() > 32768:
            return False  # SeqParallelHandoffs serves <= 32768 tokens
        return (
            ctx is not None
            and ctx.is_prefill
            and not (ctx.forward_mode is not None and ctx.forward_mode.token_graph)
            and fc.ubatch_slices is None
            and fc.tr_graph is None
        )

    # -- collectives ------------------------------------------------------------

    def pad_tokens(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.rows * self.world - x.shape[0]
        return F.pad(x, (0, 0, 0, pad)) if pad else x

    def _pull_available(self) -> bool:
        try:
            import amd_kernels.collectives.pull_collectives_ranks4  # noqa: F401
        except ImportError:
            import logging

            logging.getLogger("atom").warning(
                "SP prefill: amd_kernels is not importable, falling back to RCCL "
                "(its ring reduce-scatter rounds the bf16 partial at every hop)"
            )
            return False
        return self.world == 4

    def _pull_comms(self, hidden: int):
        """ATOM_M3_SP_PREFILL_COMM=pull: amd_kernels' bf16 pull all-gather /
        reduce-scatter (fp32 sums in rank order), built at the first SP forward
        (a collective: every rank runs the same step)."""
        if self._ag is None and self.p1:
            from amd_kernels.collectives.seq_parallel_handoffs_h6144_q2048_ranks4 import (
                SeqParallelHandoffs,
            )

            dev = torch.device("cuda", torch.cuda.current_device())
            self._ho = SeqParallelHandoffs(
                self.rank, self.tp.cpu_group, dev, max_tokens=self.world * self.max_rows
            )
            self._ag, self._rs = self._ho.ag, self._ho.rs
        if self._ag is None:
            from amd_kernels.collectives.pull_collectives_ranks4 import (
                AllGatherComm,
                SlotsReduceScatterComm,
            )

            dev = torch.device("cuda", torch.cuda.current_device())
            cpu = self.tp.cpu_group
            slot = -(-self.max_rows * hidden * 2 // 256) * 256
            self._ag = AllGatherComm(self.rank, cpu, dev, slot)
            self._rs = SlotsReduceScatterComm(
                self.rank, cpu, dev, max_rows=self.world * self.max_rows
            )
            for c in (self._ag, self._rs):
                c.warm_up()
        return self._ag, self._rs

    def gather(self, x: torch.Tensor) -> torch.Tensor:
        """Every rank's rows ``[rows, ...]`` -> ``[world * rows, ...]``."""
        x = x.contiguous()
        out = x.new_empty((self.world * x.shape[0], *x.shape[1:]))
        if self.pull:
            self._pull_comms(self.model.config.hidden_size)[0].all_gather([out], [x])
        else:
            dist.all_gather_into_tensor(out, x, group=self.tp.device_group)
        return out

    def reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        """Every token's TP partial ``[world * rows, H]`` -> this rank's summed rows."""
        x = x.contiguous()
        if self.pull:
            return self._pull_comms(x.shape[1])[1].reduce_scatter(x)
        out = x.new_empty((x.shape[0] // self.world, *x.shape[1:]))
        dist.reduce_scatter_tensor(out, x, group=self.tp.device_group)
        return out

    def gather_gemm_input(self, x: torch.Tensor):
        """This rank's normed bf16 rows -> (every token's GEMM input, its MXFP8
        scales or None for the GEMM to quantize the bf16 itself)."""
        if not self.qgather:
            return self.gather(x)[: self.n], None
        if self.p1_qg_ak:
            from amd_kernels.gemm.mxfp8_common.gemm import mxfp8_e4m3_quantize_aiter

            # x: [rows, H], rows % 32 == 0 -> codes [rows, H], scales [rows, H/32]
            # shuffled per 32-row block, so the gathered blocks are the shuffle of
            # every token's scales
            q, s = mxfp8_e4m3_quantize_aiter(x)
            codes = q.new_empty((self.world * q.shape[0], q.shape[1]))
            scales = s.new_empty((self.world * s.shape[0], s.shape[1]))
            self._pull_comms(x.shape[1])[0].all_gather([codes, scales], [q, s])
            scales._p1_ak_preshuffled = True  # linear.py routes it to mxfp8_gemm
            return codes[: self.n], scales
        q, s = quantize_fp8(x)
        return (
            self.gather(q.view(torch.uint8)).view(q.dtype)[: self.n],
            self.gather(s.view(torch.uint8)).view(s.dtype)[: self.n],
        )

    def heads_to_rows(self, attn_out: torch.Tensor):
        """This rank's heads of every token ``[n, q]`` -> every head of this rank's
        rows ``[rows, world * q]`` (+ MXFP8 scales under QGATHER: the 1x32 blocks
        never straddle two ranks' heads)."""
        world, rows = self.world, self.rows

        def a2a(t):
            out = torch.empty_like(t)
            dist.all_to_all_single(out, t.contiguous(), group=self.tp.device_group)
            return out.view(world, rows, -1).transpose(0, 1).reshape(rows, -1)

        if self.p1_unpack and isinstance(attn_out, torch.Tensor):
            # pack (MXFP8 codes + scales per row) + the unpacking all-to-all: the
            # whole o_proj input of this rank's rows, scales in the GEMM's layout
            self._pull_comms(self.model.config.hidden_size)
            xq, xs = self._ho._unpacked(attn_out.contiguous(), rows)
            xs._p1_ak_preshuffled = True
            return xq, xs
        attn_out = self.pad_tokens(attn_out)
        if not self.qgather:
            return a2a(attn_out), None
        q, s = quantize_fp8(attn_out)
        return a2a(q.view(torch.uint8)).view(q.dtype), a2a(s.view(torch.uint8)).view(
            s.dtype
        )

    @staticmethod
    def add_norm(x: torch.Tensor, residual: torch.Tensor, norm):
        """TP's large-batch tail (aiter's all-reduce + this kernel) on the rows."""
        out, res = torch.empty_like(x), torch.empty_like(x)
        rmsnorm2d_fwd_with_add(
            out, x, residual, res, norm.weight, norm.variance_epsilon, True, 0
        )
        return out, res

    # -- forward ----------------------------------------------------------------

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """This rank's rows of the vocab-parallel embedding: every token's masked
        local lookup, reduce-scattered (one rank holds each row: an exact sum)."""
        emb = self.model.embed_tokens
        ids = self.pad_tokens(input_ids.unsqueeze(-1)).squeeze(-1)
        return self.reduce_scatter(
            masked_embedding(ids, emb.weight, emb.vocab_start_idx, emb.vocab_end_idx)
        )

    def layer_forward(self, layer, positions, hidden, residual) -> ShardedOutput:
        attn = layer.self_attn
        if residual is None:
            residual = hidden
            x = layer.input_layernorm(hidden)
        else:
            x, residual = self.add_norm(hidden, residual, layer.input_layernorm)
        attn_out = attn.attend(positions, *self.gather_gemm_input(x))
        if self.dummy:
            # attention skips the dummy run and returns torch.empty: inf / NaN
            # rows would reach the router and pick out-of-range experts
            attn_out = torch.zeros_like(attn_out)
        if attn.o_proj_full is not None:
            h = attn.o_proj_full(*self.heads_to_rows(attn_out))
        else:
            h = self.reduce_scatter(self.pad_tokens(attn.o_proj(attn_out)))
        x, residual = self.add_norm(h, residual, layer.post_attention_layernorm)
        # The FFN runs every token incl. the zero padding rows: its partial is
        # then reduce-scattered without a copy.
        ffn = layer.block_sparse_moe if layer.is_moe_layer else layer.mlp
        ak = getattr(ffn, "ak_moe", None) if layer.is_moe_layer else None
        if self.p1_slots and ak is not None and ak.runs(self.world * self.rows):
            from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import mxfp4_moe

            if (
                self.p1_shared_local
                and ak.shared_full is not None
                and self.world * self.rows >= self.shared_local_min
            ):
                from p1_sls_shared.sum5 import slots4_reduce_scatter_owner_shared

                slots, shared = self._compact.local_shared(
                    x.contiguous(),
                    ak.weights,
                    ak.shared_full,
                    self.world,
                    self._pull_comms(x.shape[1])[0].all_gather,
                    self.shared_local_side,
                )
                red = slots4_reduce_scatter_owner_shared(
                    self._pull_comms(x.shape[1])[1], slots, shared
                )
                out = ShardedOutput((red, residual))
                out.sp = self
                return out
            if (
                self.p1_compact
                and self._compact is not None
                and self.world * self.rows >= self.compact_min
            ):
                slots, shared = self._compact(
                    x.contiguous(),
                    ak.weights,
                    self.world,
                    self._pull_comms(x.shape[1])[0].all_gather,
                )
            else:
                slots, shared = mxfp4_moe(
                    self.gather(x).contiguous(), ak.weights, return_partials=True
                )
            red = self._pull_comms(x.shape[1])[1].slots4_reduce_scatter(slots, shared)
            out = ShardedOutput((red, residual))
            out.sp = self
            return out
        out = ShardedOutput((self.reduce_scatter(ffn(self.gather(x))), residual))
        out.sp = self
        return out

    def aux_hidden_state(self, output: ShardedOutput) -> torch.Tensor:
        """A DSpark tap: every token's residual stream after the layer."""
        return self.gather(output[0] + output[1])[: self.n]

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        model = self.model
        self.n = input_ids.shape[0]
        self.rows = -(-self.n // self.world)
        if self.p1:
            self.rows = -(-self.rows // 32) * 32  # SeqParallelHandoffs' shard rows
        self.dummy = get_forward_context().context.is_dummy_run
        hidden, residual = self.embed(input_ids), None
        for idx in range(model.start_layer, model.end_layer):
            hidden, residual = model.layers[idx](positions, hidden, residual, sp=self)
        hidden, _ = self.add_norm(hidden, residual, model.norm)
        return self.gather(hidden)[: self.n]


def install_dump_hooks(causal_lm, model, layers: bool) -> None:
    """ATOM_M3_SP_DUMP=<dir> (debug): rank 0 saves every token's residual stream
    after each decoder layer (``layers``; the TP path needs an eager model for
    that) and the model output of each prefill forward of >= MIN_TOKENS tokens,
    as <dir>/f<k>_L<layer>.pt / f<k>_final.pt, k counting those forwards. Large
    forwards keep every 5th layer. One engine per path, compared offline: running
    two paths in one step re-runs attention, which a step does not support."""
    import os

    path = envs.ATOM_M3_SP_DUMP
    os.makedirs(path, exist_ok=True)
    tp = get_tp_group()
    state = {"k": -1}

    def eligible(n):
        ctx = get_forward_context().context
        return (
            n >= envs.ATOM_M3_SP_PREFILL_MIN_TOKENS
            and ctx is not None
            and ctx.is_prefill
            and not ctx.is_dummy_run
            and not ctx.is_draft
            and not torch.cuda.is_current_stream_capturing()
        )

    def save(t, name):
        if tp.rank_in_group == 0:
            torch.save(t.detach().cpu(), f"{path}/f{state['k']}_{name}.pt")

    def on_layer(idx):
        def hook(module, args, output):
            sharded = isinstance(output, ShardedOutput)
            n = output.sp.n if sharded else output[1].shape[0]
            if not eligible(n) or (n > 8192 and idx % 5 != 4):
                return
            if sharded:
                stream = output.sp.aux_hidden_state(output)
            else:
                stream = tensor_model_parallel_all_reduce(output[0].clone()) + output[1]
            save(stream, f"L{idx}")

        return hook

    def pre(module, args, kwargs):
        ids = args[0] if args else kwargs.get("input_ids")
        if ids is not None and eligible(ids.shape[0]):
            state["k"] += 1

    def final(module, args, kwargs, output):
        ids = args[0] if args else kwargs.get("input_ids")
        if ids is not None and eligible(ids.shape[0]) and isinstance(output, torch.Tensor):
            save(output, "final")

    causal_lm.register_forward_pre_hook(pre, with_kwargs=True)
    causal_lm.register_forward_hook(final, with_kwargs=True)
    if layers:
        for idx in range(model.start_layer, model.end_layer):
            model.layers[idx].register_forward_hook(on_layer(idx))
