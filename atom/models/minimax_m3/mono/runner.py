# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One decode step of 1..MAX_TOKENS tokens on the mono kernels (a speculative
verify's q tokens of a request count as q, see ``token_rows``).

Dense layers 0..2 run the original modules. Every sparse MoE layer is one
launch of K4 with the layer's K1 fused in front of its stages::

    K1 part        (ar, res) -> h, q; K / V / index cache insert; indexer scores
    K4 part        top-k, attention .. FFN, both all-reduces in-kernel
                   -> (ar, h_mid)

with ``(ar, res)`` the (reduced partials, residual) pair the original path hands
its fused all-reduce + RMSNorm, so a layer computes the same values either way.
A launch reads the previous layer's (ar, h_mid) while it writes its own, so both
alternate between two buffers by layer parity.
"""

import torch
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)

import atom.model_ops.topK as topk_meta
from atom.distributed.indexer_cp import indexer_cp_enabled
from atom.model_ops.communication_op import tensor_model_parallel_all_reduce
from atom.models.minimax_m3.mono.config import (
    BLOCKS,
    HEAD_DIM,
    HIDDEN,
    LOCAL_Q_HEADS,
    MAX_TOKENS,
    N_ROUTED,
    ONE_INDEX_HEAD,
    TOP_K,
    TOPK_BLOCKS,
    TP,
    IndexHeads,
    MonoUnsupported,
)
from atom.models.minimax_m3.mono.kernels.post_attn import build_post_attn_kernel
from atom.models.minimax_m3.mono.kernels.pre_attn import K1_ARGS
from atom.models.minimax_m3.mono.kernels.pre_attn import (
    SCRATCH_BYTES as K1_SCRATCH_BYTES,
)
from atom.models.minimax_m3.mono.layout import SCRATCH_BYTES as K4_SCRATCH_BYTES
from atom.models.minimax_m3.mono.layout import sym_layout
from atom.models.minimax_m3.mono.peer_buffer import PeerBuffer
from atom.models.minimax_m3.mono.timeline import LayerTimeline
from atom.models.minimax_m3.mono.weights import SparseMoeLayer
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

N_DENSE = 3


def _ptr(t: torch.Tensor) -> int:
    return t.data_ptr()


def _sparse_metadata(fwd):
    md = fwd.attn_metadata
    return getattr(md, "sparse_attention_metadata", None) or md


def token_rows(fwd, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(block table, seq_lens) with one row per token of the step: a request of q
    query tokens (speculative verify) becomes q rows, its token j seeing
    ``seq_len - (q - 1 - j)`` keys, so the kernels treat every token alike."""
    decode = _sparse_metadata(fwd).decode
    q = decode.max_query_len
    if q == 1:
        return decode.block_table, decode.seq_lens
    reqs = n // q
    back = torch.arange(
        q - 1, -1, -1, dtype=decode.seq_lens.dtype, device=decode.seq_lens.device
    )
    return (
        decode.block_table[:reqs].repeat_interleave(q, dim=0),
        (decode.seq_lens[:reqs, None] - back).reshape(-1),
    )


class MonoDecodeRunner:
    """Owns the per-layer weight views, scratch, peer buffers and compiled kernels."""

    def __init__(self, causal_lm) -> None:
        model = causal_lm.model
        config = causal_lm.config
        npes = get_tensor_model_parallel_world_size()
        if npes != TP:
            raise MonoUnsupported(f"TP {npes}")
        layers = model.layers
        if len(layers) <= N_DENSE or any(
            getattr(layers[i].self_attn, "is_indexed_sparse_attention", False)
            or layers[i].is_moe_layer
            for i in range(N_DENSE)
        ):
            raise MonoUnsupported("layers 0..2 must be dense full-attention layers")
        self.model = model
        self.rank = get_tensor_model_parallel_rank()
        # indexer context parallelism: the fused projection holds every index q
        # head, the rank's own being its TP rank
        heads = IndexHeads(TP, self.rank) if indexer_cp_enabled() else ONE_INDEX_HEAD
        self.sparse = [
            SparseMoeLayer.from_layer(layers[i], i, heads)
            for i in range(N_DENSE, len(layers))
        ]
        impl = self.sparse[0].attn_impl
        experts = layers[N_DENSE].block_sparse_moe.experts
        eps = config.rms_norm_eps
        route_scale = float(layers[N_DENSE].block_sparse_moe.routed_scaling_factor)
        # the fused shared expert's routing weight is whatever the topK metadata holds
        # read through the module: the metadata is assigned after this file is imported
        total_w, total_ids = topk_meta.aiter_topK_meta_data
        if int(total_ids[0, TOP_K].item()) != N_ROUTED:
            raise MonoUnsupported("fused shared expert id")
        shared_weight = float(total_w[0, TOP_K].item())
        self.sm_scale = impl.scale
        if abs(self.sm_scale - HEAD_DIM**-0.5) > 1e-12:
            raise MonoUnsupported(f"softmax scale {self.sm_scale}")
        if impl.topk != TOPK_BLOCKS:
            raise MonoUnsupported(f"indexer top-{impl.topk} blocks")

        dev = torch.device("cuda", torch.cuda.current_device())
        cus = torch.cuda.get_device_properties(dev).multi_processor_count
        if cus != BLOCKS:
            raise MonoUnsupported(
                f"{cus} CUs: the kernels need all {BLOCKS} CTAs co-resident"
            )
        tl_prefix = envs.ATOM_MONO_TIMELINE
        self.timeline = (
            LayerTimeline(tl_prefix, len(self.sparse), self.rank, dev)
            if tl_prefix
            else None
        )
        # one layer kernel per decode batch size (every graph captures its own)
        self.k4 = {
            s: build_post_attn_kernel(
                npes,
                self.sm_scale,
                eps,
                route_scale,
                shared_weight,
                experts.swiglu_limit,
                impl.init_blocks,
                impl.local_blocks,
                s,
                fuse_k1=True,
                timeline=self.timeline is not None,
                heads=heads,
            )
            for s in range(1, MAX_TOKENS + 1)
        }
        self.scratch1 = torch.zeros(K1_SCRATCH_BYTES, dtype=torch.uint8, device=dev)
        self.scratch4 = torch.zeros(K4_SCRATCH_BYTES, dtype=torch.uint8, device=dev)
        self.peers = PeerBuffer(
            sym_layout(npes)["_bytes"], get_tp_group().cpu_group, self.rank, npes, dev
        )
        self.step = torch.zeros(1, dtype=torch.int32, device=dev)
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
        # each layer's K1 pointers that do not change per step (K1_ARGS order)
        self.k1_args = []
        for i, lw in enumerate(self.sparse):
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

    def close(self) -> None:
        self.peers.close()

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """The final hidden states; with Eagle3 aux layers set, also the residual
        stream entering each of them, as the original model returns."""
        aux_layers = self.model.aux_hidden_state_layers
        aux: list[torch.Tensor] = []
        res = self.run_dense(input_ids, positions, aux_layers, aux)
        fwd = get_forward_context()
        rows = token_rows(fwd, res.shape[0])
        for i, lw in enumerate(self.sparse):
            res = self.run_sparse_layer(i, lw, fwd, positions, res, rows)
            if lw.layer_id in aux_layers:
                aux.append(self.h[: res.shape[0]].clone())
        hidden = self.run_final_norm(res)
        if self.timeline is not None:
            self.timeline.step_done(res.shape[0])
        return (hidden, aux) if aux else hidden

    def run_dense(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        aux_layers: tuple[int, ...] = (),
        aux: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Embedding and the dense layers on the original modules; leaves the first
        sparse layer's (ar, res) in ``self.ars[0][:S]`` and the returned residual.
        Appends the aux hidden state of the dense ones among ``aux_layers``."""
        n = input_ids.numel()
        hidden = self.model.get_input_embeddings(input_ids)
        residual = None
        for i in range(N_DENSE):
            if i in aux_layers:
                hidden, residual, aux_hidden = self.model.layers[i](
                    positions, hidden, residual, capture_aux=True
                )
                aux.append(aux_hidden)
            else:
                hidden, residual = self.model.layers[i](positions, hidden, residual)
        # the all-reduce layer N_DENSE's fused AR + RMSNorm would do, without the norm
        self.ars[0][:n].copy_(tensor_model_parallel_all_reduce(hidden).view(n, HIDDEN))
        return residual.view(n, HIDDEN)

    def run_sparse_layer(
        self,
        i: int,
        lw: SparseMoeLayer,
        fwd,
        positions: torch.Tensor,
        res: torch.Tensor,
        rows: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Sparse layer i: (ars[i % 2][:S], res) -> (ars[(i + 1) % 2][:S], returned
        residual). ``rows``: ``token_rows``."""
        n = res.shape[0]
        sparse_md = _sparse_metadata(fwd)
        block_table, seq_lens = rows
        stream = torch.cuda.current_stream()
        impl = lw.attn_impl
        kv = fwd.kv_cache_data[f"layer_{lw.layer_id}"]
        k16, v16, ks16, vs16 = impl._to_page16_shuffle(
            kv.k_cache, kv.v_cache, kv.k_scale, kv.v_scale
        )
        self.k4[n](
            _ptr(self.h), _ptr(self.q), _ptr(block_table), _ptr(seq_lens), _ptr(k16), _ptr(v16),
            _ptr(ks16), _ptr(vs16), _ptr(lw.w_o), _ptr(lw.s_o), _ptr(lw.g_post), _ptr(lw.gate),
            _ptr(lw.bias), _ptr(lw.w13), _ptr(lw.s13), _ptr(lw.w2), _ptr(lw.s2),
            _ptr(self.h_mids[(i + 1) % 2]), _ptr(self.ars[(i + 1) % 2]), _ptr(self.scratch4),
            self.peers.local, _ptr(self.peers.addresses), _ptr(self.step), self.rank, lw.layer_id,
            block_table.shape[1], sparse_md.decode.max_query_len,
            self.timeline.ptr(i) if self.timeline else 0,
            _ptr(self.k1_args[i]), _ptr(positions),
            _ptr(sparse_md.slot_mapping), _ptr(res), stream=stream,
        )  # fmt: skip
        return self.h_mids[(i + 1) % 2][:n]

    def run_final_norm(self, res: torch.Tensor) -> torch.Tensor:
        ar = self.ars[len(self.sparse) % 2]
        hidden, _ = self.model.norm(ar[: res.shape[0]], res)
        self.step.add_(1)
        return hidden
