# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A decode step of 17..64 tokens: a mono front half, a back half, the FFN all-reduce.

Per MoE layer, on the (ar, res) pair mono's own layers hand each other (``ar``
the reduced FFN output, ``res`` the residual stream)::

    front half   (ar, res) -> input rmsnorm, QKV, heads / rope, K / V / index
                 insert, index score, select, split-KV attention + merge,
                 o_proj, all-reduce + residual + post-attention rmsnorm
                 -> (res_mid, x_moe)
    back half    x_moe -> router, routed MXFP4 MoE, unfused MXFP8 shared expert,
                 the FFN all-reduce -> the next layer's ``ar`` (res_mid is its ``res``)

The first three layers (dense MLP, sparse attention on M3.1) run on the original
modules as the n <= 16 path does, and the final norm takes the last layer's pair.

Each half is picked per pipeline (``PIPELINES``): ``prod`` the original ops (the
front: the attention half of ``MiniMaxM3DecoderLayer``; the back: ``MiniMaxM3MoE.forward``
itself + the plain all-reduce, so its numerics are the original's), ``mono`` the front-half
kernel (``front64.Front64``), ``full`` the whole layer in ONE persistent kernel (K4 built with
``moe_back=True``: front half, router, routed + shared MoE, FFN all-reduce) with ``fused`` as
its (empty) back half.
    hybrid   prod front + prod back now (the stand-in); mono front + prod back
             once the front kernel is accepted
    wide     full + fused: the full mono layer of 32 / 64 rows, one launch a layer
"""

from __future__ import annotations

import logging
import os

import torch

from atom.model_ops.communication_op import tensor_model_parallel_all_reduce
from atom.model_ops.layernorm import fused_allreduce_gemma_rms_norm
from atom.models.minimax_m3.mono.config import HIDDEN
from atom.models.minimax_m3.mono.layout import N_SPLIT
from atom.models.minimax_m3.mono.policy import HYBRID_MAX_ROWS
from atom.models.minimax_m3.mono.runner import (
    K4_ABI,
    N_DENSE,
    _ptr,
    _run_compiled,
    to_page16_shuffle,
    token_rows,
)
from atom.mono.runtime.compile import kernel_names
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")

# (front, back) of each pipeline; ATOM_MONO_HYBRID_FRONT overrides the hybrid's front.
PIPELINES = {"hybrid": ("prod", "prod"), "wide": ("full", "fused")}


class _Parts:
    """The buffers the mono halves share, per runner: layer i reads ``ars[i % 2]``
    (its input ``ar``), writes ``ars[(i + 1) % 2]`` (the back half's output) and
    ``hiddens[i % 2]`` (the front half's residual); ``x_moe`` is between them."""

    def __init__(self, runner) -> None:
        shape = (HYBRID_MAX_ROWS, HIDDEN)
        new = lambda: torch.empty(shape, dtype=torch.bfloat16, device=runner.dev)
        self.ars, self.hiddens, self.x_moe = [new(), new()], [new(), new()], new()
        self.halves: dict[str, object] = {}  # built on first use, shared by pipelines


class ProductionFront:
    """The original attention half of a decoder layer on the (ar, res) pair."""

    needs_rows = False

    def prepare(self, n: int) -> bool:
        return True

    def begin_step(self) -> None:
        pass

    def __call__(self, i, lw, layer, positions, ar, res, want_entering, **_):
        """-> (res_mid, x_moe, the residual stream entering the layer)."""
        h, entering = layer.input_layernorm(ar, res)
        attn = layer.self_attn(positions=positions, hidden_states=h)
        x_moe, res_mid = fused_allreduce_gemma_rms_norm(
            attn, entering, layer.post_attention_layernorm
        )
        return res_mid, x_moe, entering


def front_options(r, n: int) -> dict:
    """The build options of an n-row front half, as ``Front64._build`` picks them (measured per size:
    32 rows keep 8 split tasks a token, wider 4; the group pipeline and 13 copy rounds in flight everywhere).
    """
    return {
        "pipe": r.attn_mxfp8,
        "copy_batch": 13,
        "n_split": 4 if n > 32 and r.nvfp4 else 8,
    }


def symbol_tokens(r, n: int, whole: bool) -> set[str]:
    """What the n-row kernel's symbol must spell (``post_attn.py``: ``sym_extra``; a default is not spelled):
    the same build options as the front half, ``mb1`` for the whole layer, ``fr1`` for a front-only build.
    """
    o = front_options(r, n)
    want = {f"s{n}", "mb1" if whole else "fr1"}
    want |= {"pp1"} if o["pipe"] else set()
    want |= {f"cb{o['copy_batch']}"} if o["copy_batch"] != 8 else set()
    want |= {f"ns{o['n_split']}"} if o["n_split"] != N_SPLIT else set()
    return want


class MonoFront:
    """The front-half kernel (``front64.Front64``): one launch a layer, writing
    ``hidden`` (the residual after attention) and ``x_moe`` (the MoE input). The
    kernel inserts the step's K / V / index entries itself, so the original
    attention does not run; it reads ``ar`` from ``parts.ars``."""

    needs_rows = True

    def __init__(self, runner, parts, front_cls=None) -> None:
        from atom.models.minimax_m3.mono.front64 import Front64

        # the front build selects blocks in every layer (no layer reuses an earlier selection)
        if not all(lw.index_topk for lw in runner.sparse):
            raise RuntimeError("a layer reuses an earlier layer's block selection")
        self.parts = parts
        self.front = (front_cls or Front64)(runner, parts.ars)

    whole = False  # MonoFull: the build holds the MoE half too

    def prepare(self, n: int) -> bool:
        """Built, and spelling the options this run measured (a front kernel built without them is
        another kernel: refused, loudly)."""
        if not self.front.prepare(n):
            return False
        names = kernel_names(self.front.builds[n])
        want = symbol_tokens(self.front.runner, n, self.whole)
        bad = [nm for nm in names if not want <= set(nm.split("_"))]
        if bad:
            logger.error(
                "MiniMax-M3 mono %d-row kernel %s lacks %s: width refused",
                n,
                bad,
                sorted(want),
            )
            return False
        logger.info(
            "MiniMax-M3 mono %d-row %s kernel: %s",
            n,
            "whole-layer" if self.whole else "front",
            names,
        )
        return True

    def begin_step(self) -> None:
        self.front.begin_step()

    def __call__(self, i, lw, layer, positions, ar, res, want_entering, fwd, n, rows):
        p = self.parts
        src = p.ars[i % 2][:n]
        if ar.data_ptr() != src.data_ptr():  # layer 0's, from the prefix
            src.copy_(ar)
        hidden, x_moe = p.hiddens[i % 2][:n], p.x_moe[:n]
        self.front.run_layer(i, lw, fwd, positions, res, rows, hidden, x_moe)
        return hidden, x_moe, ((ar + res) if want_entering else None)


def launch_full_layer(front, i, lw, fwd, positions, res, rows, hidden, ar_out) -> None:
    """One launch of the whole-layer K4 (``moe_back=True``) over ``front``'s buffers: the arguments
    of ``Front64.run_layer`` with the layer's real MoE tensors, ``ar_out`` the final reduced FFN output.
    """
    runner, n = front.runner, res.shape[0]
    sparse_md = (
        getattr(fwd.attn_metadata, "sparse_attention_metadata", None)
        or fwd.attn_metadata
    )
    block_table, seq_lens = rows
    kv = fwd.kv_cache_data[f"layer_{lw.layer_id}"]
    if runner.nvfp4:
        k16 = v16 = kv.k_cache
        ks16, vs16 = lw.attn_impl.nvfp4_k_scale, lw.attn_impl.nvfp4_v_scale
    else:
        k16, v16, ks16, vs16 = to_page16_shuffle(
            kv.k_cache, kv.v_cache, kv.k_scale, kv.v_scale
        )
    args = {
        "h_in": _ptr(front.h), "q": _ptr(front.q), "block_table": _ptr(block_table),
        "seq_lens": _ptr(seq_lens), "k_cache": _ptr(k16), "v_cache": _ptr(v16),
        "k_scale": _ptr(ks16), "v_scale": _ptr(vs16), "w_o": _ptr(lw.w_o), "s_o": _ptr(lw.s_o),
        "g_post": _ptr(lw.g_post), "w_gate": _ptr(lw.gate), "bias": _ptr(lw.bias),
        "w13": _ptr(lw.w13), "s13": _ptr(lw.s13), "w2": _ptr(lw.w2), "s2": _ptr(lw.s2),
        "h_mid": _ptr(hidden), "ar_out": _ptr(ar_out), "scratch": _ptr(front.scratch4),
        **front.peers.kernel_args(), "layer": lw.layer_id, "bt_width": block_table.shape[1],
        "q_len": sparse_md.decode.max_query_len, "tl": 0, "k1_args": _ptr(front.k1_args[i]),
        "positions": _ptr(positions), "slot_mapping": _ptr(sparse_md.slot_mapping),
        "res": _ptr(res), "batch_ids": _ptr(fwd.attn_metadata.batch_id_per_q_token),
        "w13s": _ptr(lw.w13s), "s13s": _ptr(lw.s13s), "w2s": _ptr(lw.w2s), "s2s": _ptr(lw.s2s),
    }  # fmt: skip
    _run_compiled(front.builds[n], *K4_ABI.pack(args), torch.cuda.current_stream())


class MonoFull(MonoFront):
    """The whole layer in one launch (K4 built with ``moe_back=True``, 32 / 64 rows): the front
    half's inputs plus the layer's MoE tensors; writes ``hidden`` and the next layer's ``ar``
    (``parts.ars[(i + 1) % 2]``). ``x_moe`` stays inside the kernel."""

    whole = True

    def __init__(self, runner, parts) -> None:
        from atom.models.minimax_m3.mono.front64 import Front64

        if (
            not runner.shared_mxfp8
        ):  # the full kernel's shared expert is the unfused MXFP8 one
            raise RuntimeError(
                "the whole-layer kernel needs the unfused MXFP8 shared expert"
            )

        class Full64(Front64):
            def _build(self, n: int):
                # moe_a4w4 is the back half's convention; the front options as the front-only build's
                return self.runner._build_k4(
                    n,
                    index_topk=True,
                    moe_back=True,
                    moe_a4w4=True,
                    **front_options(self.runner, n),
                )

            def run_layer(self, i, lw, fwd, positions, res, rows, hidden, ar_out):
                launch_full_layer(
                    self, i, lw, fwd, positions, res, rows, hidden, ar_out
                )

        super().__init__(runner, parts, Full64)

    def __call__(self, i, lw, layer, positions, ar, res, want_entering, fwd, n, rows):
        p = self.parts
        src = p.ars[i % 2][:n]
        if ar.data_ptr() != src.data_ptr():  # layer 0's, from the prefix
            src.copy_(ar)
        hidden, ar_out = p.hiddens[i % 2][:n], p.ars[(i + 1) % 2][:n]
        self.front.run_layer(i, lw, fwd, positions, res, rows, hidden, ar_out)
        return hidden, None, ((ar + res) if want_entering else None)


class ProductionBack:
    """The original MoE (router, aiter fused MoE, unfused shared expert) and the plain all-reduce."""

    def prepare(self, n: int) -> bool:
        return True

    def begin_step(self) -> None:
        pass

    def __call__(self, i, lw, layer, x_moe, fwd, n, ar_out):
        return tensor_model_parallel_all_reduce(layer.block_sparse_moe(x_moe)).view(
            n, HIDDEN
        )


class FusedBack:
    """The back half the full kernel already did: its output is the next layer's ``ar``."""

    def prepare(self, n: int) -> bool:
        return True

    def begin_step(self) -> None:
        pass

    def __call__(self, i, lw, layer, x_moe, fwd, n, ar_out):
        return ar_out


def _half(runner, parts, side: str, kind: str):
    """The pipeline half ``side`` (front | back) of ``kind``, made once per runner."""
    key = f"{side}:{kind}"
    if key not in parts.halves:
        make = {
            "front:prod": lambda: ProductionFront(),
            "front:mono": lambda: MonoFront(runner, parts),
            "front:full": lambda: MonoFull(runner, parts),
            "back:prod": lambda: ProductionBack(),
            "back:fused": lambda: FusedBack(),
        }.get(key)
        if make is None:
            raise ValueError(f"pipeline half {key!r}: no such half")
        parts.halves[key] = make()
    return parts.halves[key]


class HybridDecode:
    """Runs the steps of one pipeline (``hybrid`` | ``wide``) of one ``MonoDecodeRunner``'s model."""

    def __init__(self, runner, name: str = "hybrid") -> None:
        self.r = runner
        # no quantized (quick) all-reduce anywhere on this path: its numerics are not the original's
        quick = os.environ.get("AITER_QUICK_REDUCE_QUANTIZATION", "")
        if quick and quick.upper() != "NONE":
            raise RuntimeError(f"AITER_QUICK_REDUCE_QUANTIZATION={quick} is set")
        if getattr(runner, "parts", None) is None:
            runner.parts = _Parts(runner)
        self.parts = runner.parts
        front, back = PIPELINES[name]
        if name == "hybrid" and envs.ATOM_MONO_HYBRID_FRONT:
            front = envs.ATOM_MONO_HYBRID_FRONT
        self.name = name
        self.front = _half(runner, self.parts, "front", front)
        self.back = _half(runner, self.parts, "back", back)
        self.layers = [runner.model.layers[lw.layer_id] for lw in runner.sparse]
        logger.info(
            "MiniMax-M3 %s decode: %s front half, %s back half, quick all-reduce off",
            name, front, back,
        )  # fmt: skip

    def prepare(self, n: int) -> bool:
        """Whether every rank holds the n-row halves (collective)."""
        return self.front.prepare(n) and self.back.prepare(n)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        """The final hidden states (and the aux layers' as ``MonoDecodeRunner.forward``)."""
        r = self.r
        model, fwd = r.model, get_forward_context()
        aux_layers = model.aux_hidden_state_layers
        aux: list[torch.Tensor] = []
        n = input_ids.numel()
        rows = token_rows(fwd, n) if self.front.needs_rows else None
        self.front.begin_step()
        self.back.begin_step()
        ar, res = self._prefix(input_ids, positions, aux_layers, aux)
        check = envs.ATOM_MONO_CHECK
        for i, (lw, layer) in enumerate(zip(r.sparse, self.layers)):
            if check:
                ar_in, res_in = ar.clone(), res.clone()
            res, ar = self.layer(i, lw, layer, fwd, positions, ar, res, rows, aux, n)
            if check:
                self.compare(lw, layer, positions, ar_in, res_in, res, ar)
        hidden, residual = model.norm(ar, res)
        r._write_aux(r.sparse[-1].layer_id, residual)
        return (hidden, aux) if aux else hidden

    def layer(self, i, lw, layer, fwd, positions, ar, res, rows, aux, n):
        """Sparse layer i: (ar, res) -> (res, ar) of the next layer."""
        r = self.r
        tap_aux = r.aux is not None and lw.layer_id - 1 in r.aux.layer_ids
        tap_eagle = lw.layer_id in r.model.aux_hidden_state_layers
        res, x_moe, entering = self.front(
            i,
            lw,
            layer,
            positions,
            ar,
            res,
            tap_aux or tap_eagle,
            fwd=fwd,
            n=n,
            rows=rows,
        )
        if tap_eagle:
            aux.append(entering.clone())
        if tap_aux:
            r._write_aux(lw.layer_id - 1, entering)
        ar_out = self.parts.ars[(i + 1) % 2][:n]
        return res, self.back(i, lw, layer, x_moe, fwd, n, ar_out)

    def compare(self, lw, layer, positions, ar_in, res_in, res, ar) -> None:
        """Debug (``ATOM_MONO_CHECK``, eager): the original decoder layer on the same
        (ar, res) -- fed ``ar`` on rank 0 and zeros elsewhere, so its fused all-reduce
        reproduces it -- against this layer's result, per token (rel L2); rank 0
        logs the median and max over the step's tokens, with the original layer run
        twice (``noise``: its own run-to-run difference, the aiter MoE's atomic
        accumulation is not deterministic). Both write the same cache bytes, and
        this layer ran first, so it read only what it inserted."""
        r, n = self.r, res.shape[0]
        h_in = ar_in if r.rank == 0 else torch.zeros_like(ar_in)
        ref_partial, ref_res = layer(positions, h_in, res_in)
        ref_ar = tensor_model_parallel_all_reduce(ref_partial).view(n, HIDDEN)
        again, _ = layer(positions, h_in, res_in)
        again = tensor_model_parallel_all_reduce(again).view(n, HIDDEN)

        def rel(a, b):
            a, b = a.float(), b.float().view(n, HIDDEN)
            d = ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-12)).sort().values
            return f"{d[n // 2].item():.3e} / {d[-1].item():.3e}"

        if r.rank == 0:
            logger.info(
                "%s check layer %d n=%d: residual rel L2 median / max %s | "
                "reduced out %s | original vs itself (noise) %s",
                self.name, lw.layer_id, n, rel(res, ref_res), rel(ar, ref_ar), rel(again, ref_ar),
            )  # fmt: skip

    def _prefix(self, input_ids, positions, aux_layers, aux):
        """Embedding and the dense-MLP prefix layers on the original modules ->
        (the first MoE layer's ar, residual)."""
        r, n = self.r, input_ids.numel()
        hidden = r.model.get_input_embeddings(input_ids)
        residual = None
        for i in range(N_DENSE):
            if i in aux_layers:
                hidden, residual, tap = r.model.layers[i](
                    positions, hidden, residual, capture_aux=True
                )
                aux.append(tap)
            else:
                hidden, residual = r.model.layers[i](positions, hidden, residual)
        ar = tensor_model_parallel_all_reduce(hidden).view(n, HIDDEN)
        return ar, residual.view(n, HIDDEN)
