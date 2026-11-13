# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The mono front half of a sparse MoE layer for 17..MAX_ROWS rows, as launches the runner side calls.

One launch per layer and rank: K1 (input norm of ``ar + res``, QKV, heads, K/V + index-K cache
insert, index scores), selection, split-KV attention + merge, o_proj with the in-kernel
reduce-scatter / all-gather all-reduce, then the post-attention residual and norm
(``build_post_attn_kernel(front=True)``). It writes ``hidden`` and ``x_moe`` ([n, HIDDEN] bf16)
and nothing of the MoE half: the caller's back half takes ``x_moe`` and ``hidden``.

``Front64`` owns its scratch and peer buffers sized for MAX_ROWS rows (collective: every rank
builds it together, after the runner's own handshake). The step's mailboxes are zeroed once
(``begin_step``); the tags carry the layer id, so layers need no reset between them."""

import logging

import torch
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from atom.model_ops.minimax_m3.sparse_attn import to_page16_shuffle
from atom.models.minimax_m3.mono.config import (
    HEAD_DIM,
    HIDDEN,
    LOCAL_Q_HEADS,
    MAX_ROWS,
    MAX_TOKENS,
)
from atom.models.minimax_m3.mono.kernels.common import KV_PAGE_BYTES
from atom.models.minimax_m3.mono.kernels.post_attn import K4_ABI
from atom.models.minimax_m3.mono.kernels.pre_attn import K1_ARGS, k1_scratch
from atom.models.minimax_m3.mono.layout import scratch_layout, sym_layout
from atom.models.minimax_m3.mono.weights import SparseMoeLayer
from atom.mono.runtime.consensus import MonoUnsupported
from atom.mono.runtime.lifecycle import owned_peer_buffer
from atom.mono.runtime.mailboxes import StepMailboxes
from atom.mono.runtime.uncached import uncached_bytes
from atom.mono.runtime.widths import WidthBuilds

logger = logging.getLogger("atom")


def _ptr(t: torch.Tensor) -> int:
    return t.data_ptr()


class Front64:
    """``runner``: the ``MonoDecodeRunner`` (its weights, config and TP group are reused);
    ``ars``: two [MAX_ROWS, HIDDEN] bf16 buffers, layer i's ``ar`` input (the previous layer's
    all-reduced FFN output) being ``ars[i % 2]``."""

    def __init__(self, runner, ars) -> None:
        if len(ars) != 2 or any(a.shape[0] < MAX_ROWS for a in ars):
            raise MonoUnsupported(f"two [{MAX_ROWS}, {HIDDEN}] ar buffers")
        if runner.index_heads != 1:
            raise MonoUnsupported(
                "the front half serves one index head (no indexer CP)"
            )
        from aiter.dist.parallel_state import get_tp_group

        group = get_tp_group().cpu_group
        self.runner, self.ars = runner, list(ars)
        dev = runner.dev
        self.rows = MAX_ROWS
        self.scratch1 = uncached_bytes(k1_scratch(self.rows)["BYTES"], dev)
        self.scratch4 = uncached_bytes(scratch_layout(self.rows)["_bytes"], dev)
        self.peers, self._finalizer = owned_peer_buffer(
            self,
            sym_layout(runner.npes, self.rows)["_bytes"],
            group,
            runner.rank,
            runner.npes,
            dev,
        )
        self.mailboxes = StepMailboxes(
            self.peers, self.scratch1, self.scratch4, debug=runner.debug
        )
        bf16 = torch.bfloat16
        # K1's residual out (read by the layer's post stage), q and the index q
        self.h = torch.empty(self.rows, HIDDEN, dtype=bf16, device=dev)
        self.q = torch.empty(
            self.rows, LOCAL_Q_HEADS * HEAD_DIM, dtype=bf16, device=dev
        )
        self.iq = torch.empty(self.rows, 1, HEAD_DIM, dtype=bf16, device=dev)
        self.k1_args = []
        for i, lw in enumerate(runner.sparse):
            ptrs = {
                "ar": _ptr(self.ars[i % 2]), "g_in": _ptr(lw.g_in), "w_qkv": _ptr(lw.w_qkv),
                "s_qkv": _ptr(lw.s_qkv), "g_q": _ptr(lw.g_q), "g_k": _ptr(lw.g_k),
                "g_iq": _ptr(lw.g_iq), "g_ik": _ptr(lw.g_ik), "cos_sin": _ptr(lw.cos_sin),
                "index_cache": _ptr(lw.attn_impl.index_cache), "iq_out": _ptr(self.iq),
                "scratch": _ptr(self.scratch1),
            }  # fmt: skip
            self.k1_args.append(
                torch.tensor([ptrs[a] for a in K1_ARGS], dtype=torch.int64, device=dev)
            )
        self.builds = WidthBuilds(
            self._build, lambda built: [(built, K4_ABI)], group, "MiniMax-M3 mono front"
        )

    def _build(self, n: int):
        if not MAX_TOKENS < n <= self.rows:
            raise MonoUnsupported(f"front half for {n} rows")
        # measured per size (INTERFACE.md): 32 rows keep 8 split tasks a token, wider widths 4 (one
        # round at 64 rows); pipe and 13 copy rounds in flight help every width
        r = self.runner
        return r._build_k4(
            n, index_topk=True, front=True, pipe=r.attn_mxfp8, copy_batch=13,
            n_split=4 if n > 32 and r.nvfp4 else 8,
        )  # fmt: skip

    def prepare(self, n: int) -> bool:
        """Whether every rank holds the n-row front kernel (compiled here, before a graph capture)."""
        return self.builds.prepare(n)

    def begin_step(self) -> None:
        """Zero this object's mailboxes and fence the peers: once per step, before the first layer's launch."""
        self.mailboxes.begin_step()

    def close(self) -> None:
        self._finalizer()

    def run_layer(
        self,
        i: int,
        lw: SparseMoeLayer,
        fwd,
        positions: torch.Tensor,
        res: torch.Tensor,
        rows,
        hidden: torch.Tensor,
        x_moe: torch.Tensor,
    ) -> None:
        """Sparse layer ``i``'s front half: reads ``self.ars[i % 2][:n]`` (the previous layer's
        all-reduced FFN output) and ``res`` ([n, HIDDEN], the residual stream), writes ``hidden``
        and ``x_moe`` ([>= n, HIDDEN] bf16, caller-owned). ``rows``: ``runner.token_rows(fwd, n)``.
        """
        n = res.shape[0]
        runner = self.runner
        md = fwd.attn_metadata
        sparse_md = getattr(md, "sparse_attention_metadata", None) or md
        block_table, seq_lens = rows
        stream = torch.cuda.current_stream()
        kv = fwd.kv_cache_data[f"layer_{lw.layer_id}"]
        if runner.nvfp4:
            pages = kv.k_cache
            if pages.stride() != (KV_PAGE_BYTES, 1) or pages.dtype != torch.uint8:
                raise MonoUnsupported(
                    f"NVFP4 KV pages {pages.dtype} {tuple(pages.shape)} {pages.stride()}"
                )
            k16 = v16 = pages
            ks16, vs16 = lw.attn_impl.nvfp4_k_scale, lw.attn_impl.nvfp4_v_scale
        else:
            k16, v16, ks16, vs16 = to_page16_shuffle(
                kv.k_cache, kv.v_cache, kv.k_scale, kv.v_scale
            )
        args = {
            "h_in": _ptr(self.h), "q": _ptr(self.q), "block_table": _ptr(block_table),
            "seq_lens": _ptr(seq_lens), "k_cache": _ptr(k16), "v_cache": _ptr(v16),
            "k_scale": _ptr(ks16), "v_scale": _ptr(vs16), "w_o": _ptr(lw.w_o),
            "s_o": _ptr(lw.s_o), "g_post": _ptr(lw.g_post),
            # a front build reads the post-attention norm weight from the w_gate slot (post_norm)
            "w_gate": _ptr(lw.g_post), "bias": 0, "w13": 0, "s13": 0, "w2": 0, "s2": 0,
            "h_mid": _ptr(hidden), "ar_out": _ptr(x_moe), "scratch": _ptr(self.scratch4),
            **self.peers.kernel_args(), "layer": lw.layer_id,
            "bt_width": block_table.shape[1],
            "q_len": sparse_md.decode.max_query_len, "tl": 0,
            "k1_args": _ptr(self.k1_args[i]), "positions": _ptr(positions),
            "slot_mapping": _ptr(sparse_md.slot_mapping), "res": _ptr(res),
            "batch_ids": _ptr(md.batch_id_per_q_token), "w13s": 0, "s13s": 0, "w2s": 0, "s2s": 0,
        }  # fmt: skip
        _run_compiled(self.builds[n], *K4_ABI.pack(args), stream)
