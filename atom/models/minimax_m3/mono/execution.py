# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared sparse execution for native ATOM and tensor-library callers."""

import torch

from atom.models.minimax_m3.mono.config import (
    HEAD_DIM,
    HIDDEN,
    LOCAL_Q_HEADS,
    MAX_TOKENS,
)
from atom.models.minimax_m3.mono.kernels.post_attn import K4_ABI
from atom.models.minimax_m3.mono.kernels.pre_attn import K1_ARGS
from atom.models.minimax_m3.mono.kernels.pre_attn import (
    SCRATCH_BYTES as K1_SCRATCH_BYTES,
)
from atom.models.minimax_m3.mono.layout import SCRATCH_BYTES as K4_SCRATCH_BYTES
from atom.models.minimax_m3.mono.layout import sym_layout
from atom.mono.runtime.consensus import bind_agreed
from atom.mono.runtime.lifecycle import owned_peer_buffer
from atom.mono.runtime.mailboxes import StepMailboxes


def _ptr(t):
    return t.data_ptr()


class SparseExecution:
    def __init__(self, weights, index_caches, group, rank, npes, device, debug=False):
        self.dev, self.rank, self.npes, self.debug = device, rank, npes, debug
        self.weights = weights
        # Every rank finishes local allocation before the collective IPC handshake.
        bind_agreed(lambda: self._allocate_local(weights, index_caches), group)
        self.peers, self._finalizer = owned_peer_buffer(
            self, sym_layout(npes)["_bytes"], group, rank, npes, device
        )
        try:
            bind_agreed(lambda: self._make_mailboxes(debug), group)
        except Exception:
            self.close()
            raise

    def _make_mailboxes(self, debug):
        self.mailboxes = StepMailboxes(
            self.peers, self.scratch1, self.scratch4, debug=debug
        )

    def _allocate_local(self, weights, index_caches):
        dev = self.dev
        self.scratch1 = torch.zeros(K1_SCRATCH_BYTES, dtype=torch.uint8, device=dev)
        self.scratch4 = torch.zeros(K4_SCRATCH_BYTES, dtype=torch.uint8, device=dev)
        bf16 = torch.bfloat16
        # row k = token k of the step; sparse layer i reads ars[i % 2] (the
        # previous layer's output) and writes ars[(i + 1) % 2], likewise h_mids
        self.ars = [
            torch.empty(MAX_TOKENS, HIDDEN, dtype=bf16, device=dev) for _ in range(2)
        ]
        self.h_mids = [
            torch.empty(MAX_TOKENS, HIDDEN, dtype=bf16, device=dev) for _ in range(2)
        ]
        self.h = torch.empty(MAX_TOKENS, HIDDEN, dtype=bf16, device=dev)
        self.q = torch.empty(
            MAX_TOKENS, LOCAL_Q_HEADS * HEAD_DIM, dtype=bf16, device=dev
        )
        self.iq = torch.empty(MAX_TOKENS, 1, HEAD_DIM, dtype=bf16, device=dev)
        # layer 0's residual input: its K1 takes (embedding, 0), acc = embedding
        self.zero_res = torch.zeros(MAX_TOKENS, HIDDEN, dtype=bf16, device=dev)
        # each layer's K1 pointers that do not change per step (K1_ARGS order)
        self.k1_args = []
        for i, lw in enumerate(weights):
            ptrs = {
                "ar": _ptr(self.ars[i % 2]), "g_in": _ptr(lw.g_in), "w_qkv": _ptr(lw.w_qkv),
                "s_qkv": _ptr(lw.s_qkv), "g_q": _ptr(lw.g_q), "g_k": _ptr(lw.g_k),
                "g_iq": _ptr(lw.g_iq), "g_ik": _ptr(lw.g_ik), "cos_sin": _ptr(lw.cos_sin),
                "index_cache": _ptr(index_caches[i]), "iq_out": _ptr(self.iq),
                "scratch": _ptr(self.scratch1),
            }  # fmt: skip
            self.k1_args.append(
                torch.tensor([ptrs[a] for a in K1_ARGS], dtype=torch.int64, device=dev)
            )

    @property
    def resources(self):
        return [
            *self.ars,
            *self.h_mids,
            self.h,
            self.q,
            self.iq,
            self.zero_res,
            *self.k1_args,
            *self.mailboxes.resources,
        ]

    def close(self):
        self._finalizer()

    def forward_layer(
        self,
        index,
        hidden,
        residual,
        positions,
        cache,
        rows,
        slots,
        batch_ids,
        query_len,
        kernel,
        timeline=0,
        cache_args=0,
    ):
        n = hidden.shape[0]
        if hidden.data_ptr() != self.ars[index % 2].data_ptr():
            self.ars[index % 2][:n].copy_(hidden)
        lw = self.weights[index]
        block_table, seq_lens = rows
        k16, v16, ks16, vs16 = cache
        args = {
            "h_in": _ptr(self.h),
            "q": _ptr(self.q),
            "block_table": _ptr(block_table),
            "seq_lens": _ptr(seq_lens),
            "k_cache": _ptr(k16),
            "v_cache": _ptr(v16),
            "k_scale": _ptr(ks16),
            "v_scale": _ptr(vs16),
            "w_o": _ptr(lw.w_o),
            "s_o": _ptr(lw.s_o),
            "g_post": _ptr(lw.g_post),
            "w_gate": _ptr(lw.gate),
            "bias": _ptr(lw.bias),
            "w13": _ptr(lw.w13),
            "s13": _ptr(lw.s13),
            "w2": _ptr(lw.w2),
            "s2": _ptr(lw.s2),
            "h_mid": _ptr(self.h_mids[(index + 1) % 2]),
            "ar_out": _ptr(self.ars[(index + 1) % 2]),
            "scratch": _ptr(self.scratch4),
            **self.peers.kernel_args(),
            "layer": lw.layer_id,
            "bt_width": block_table.shape[1],
            "q_len": query_len,
            "tl": timeline,
            "k1_args": _ptr(self.k1_args[index]),
            "positions": _ptr(positions),
            "slot_mapping": _ptr(slots),
            "res": _ptr(residual),
            "batch_ids": _ptr(batch_ids),
        }
        kernel(
            *K4_ABI.pack(args),
            stream=torch.cuda.current_stream(self.dev),
            cache_args=cache_args,
        )
        return self.ars[(index + 1) % 2][:n], self.h_mids[(index + 1) % 2][:n]
