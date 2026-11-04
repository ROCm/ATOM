# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A model's DSA indexers on the mono indexer kernel (``kernel``).

Whether the kernel serves a deployment is decided once, from static
configuration every rank shares (no vote needed): the kernel waits in-kernel
on every peer, so a per-rank or per-layer fallback would hang its peers. When
it does not, the indexers keep their own forward and the rest of the model's
mono is unaffected.

Per step a model's mono runner serves the indexer on a decode -- plain (one
query token a request) or a uniform speculative verify (``IxBuild.qlen`` rows a
request, each row its own causal window); every other step -- prefill, a dummy
run -- calls the indexer's own forward on every rank alike.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from atom.config import get_current_atom_config
from atom.distributed.dcp_utils import get_dcp_rank
from atom.distributed.pcp_utils import pcp_is_enabled
from atom.mono.dsa_indexer import kernel as kix
from atom.mono.dsa_indexer.plan import (
    IndexerShape,
    unsupported,
)
from atom.utils.custom_register import direct_register_custom_op
from atom.utils.forward_context import (
    get_forward_context,
)

logger = logging.getLogger("atom")


@dataclass(frozen=True)
class IndexerBinding:
    modules: dict  # DSA layer id -> its Indexer
    shape: IndexerShape
    world: int  # the TP ranks splitting a row's keys
    max_ctx: int  # a row's keys at most: max_model_len
    replicated: bool = False  # no DCP: every rank's cache holds every key
    # where its peer regions start in the peer buffer it shares (bytes): past
    # the other kernels' (``kernel.peer_layout``), a launch argument
    peer_offset: int = 0

    def key(self, rows: int, qlen: int = 1) -> kix.IxBuild:
        return kix.IxBuild(
            self.shape, rows, self.world, self.max_ctx, self.replicated, qlen=qlen
        )

    def serves(self, rows: int, qlen: int, ilog) -> bool:
        """Whether the kernel has a build for this step and, replicated, the
        step's ILOG fits ``ilog`` (``logits_plane``)."""
        key = self.key(rows, qlen)
        if kix.unfit(key):
            return False
        return not self.replicated or key.ilog_words <= ilog.numel()


def logits_plane(metadata_builder):
    """The production indexer's decode logits plane, flat fp32: a replicated
    step's ILOG (no DCP: the plane exists exactly then, ``LogitsWorkspace``),
    free while the mono launch stands in for that indexer. None under DCP."""
    workspace = getattr(metadata_builder, "index_logits_workspace", None)
    if workspace is None:
        return None
    return workspace.logits(1, workspace.capacity).view(-1)


def shape_of(module, block_rows: int) -> IndexerShape:
    """An Indexer's numbers and storage, as the kernel reads them."""
    w = module.wq_b.weight
    plain = w.dtype == torch.bfloat16 and module.wq_b.quant_type.value == 0
    return IndexerShape(
        heads=module.n_head,
        head_dim=module.head_dim,
        topk=module.topk_tokens,
        q_lora=module.q_lora_rank,
        wq_dtype="bf16" if plain else f"{w.dtype}-{module.wq_b.quant_type.name}",
        kv_dtype="fp4" if module._indexer_fp4 else "fp8",
        block_rows=block_rows,
        rope_dim=module.rope_dim,
        rope_neox=bool(module.rope_dim and module.rotary_emb.is_neox_style),
    )


def _refusal(cfg, tp: int, rank: int, indexers: dict) -> str:
    """Why the kernel cannot serve these indexers, or ''."""
    why = _static_refusal(cfg, tp, rank, indexers)
    if why:
        return why
    shape = shape_of(next(iter(indexers.values())), cfg.kv_cache_block_size)
    for layer_id, m in indexers.items():
        why = _cache_refusal(m, shape)
        if why:
            return f"layer {layer_id}: {why}"
    return ""


def routing_refusal(cfg, indexers: dict) -> str:
    """This TP rank's ``_refusal`` but the index caches (allocated after the
    model is wrapped): what the wrapper checks before routing a layer, so a
    deployment the kernel cannot serve never has its indexers routed (a
    compiled stack keeps the routing op it traced)."""
    return _static_refusal(
        cfg,
        get_tensor_model_parallel_world_size(),
        get_tensor_model_parallel_rank(),
        indexers,
    )


def _static_refusal(cfg, tp: int, rank: int, indexers: dict) -> str:
    if not indexers:
        return "no DSA layers"
    shapes = {shape_of(m, cfg.kv_cache_block_size) for m in indexers.values()}
    if len(shapes) > 1:
        return f"the DSA layers differ: {sorted(s.tag for s in shapes)}"
    (shape,) = shapes
    why = unsupported(shape)
    if why:
        return why
    dcp = cfg.decode_context_parallel_size
    if dcp > 1 and dcp != tp:
        return f"DCP {dcp} / TP {tp}: the peer exchange runs on the TP group"
    if dcp > 1 and get_dcp_rank() != rank:
        return "DCP rank is not the TP rank: the peer exchange runs on the TP group"
    if dcp > 1 and cfg.dcp_config.interleave_size != 1:
        return f"DCP interleave {cfg.dcp_config.interleave_size}, the kernel's is 1"
    if pcp_is_enabled():
        return "prefill context parallel"
    for layer_id, m in indexers.items():
        why = _operand_refusal(m, shape)
        if why:
            return f"layer {layer_id}: {why}"
    return ""


def _layer_refusal(m, shape: IndexerShape) -> str:
    """Why one Indexer's operands or index cache are not what the kernel
    reads, or ''."""
    return _operand_refusal(m, shape) or _cache_refusal(m, shape)


def _operand_refusal(m, shape: IndexerShape) -> str:
    if not m.external_k_weights:
        return "k and head weights projected inside the indexer"
    if m.quantized_q:
        return "q comes quantized with a scale, the kernel reads bf16"
    if m.rope_dim and not m.use_qk_rope_cache_fusion:
        return "RoPE outside the fused writer, whose numerics the kernel has"
    w = m.wq_b.weight
    if tuple(w.shape) != (shape.qrows, shape.q_lora) or not w.is_contiguous():
        return f"wq_b {tuple(w.shape)}"
    return ""


def _cache_refusal(m, shape: IndexerShape) -> str:
    cache = m.k_cache.kv_cache[0]
    if cache.numel() % shape.block_bytes or not cache.is_contiguous():
        return f"index cache {tuple(cache.shape)}"
    return ""


def indexer_modules(inner) -> dict:
    """A layer stack's DSA indexers: layer id -> Indexer (a layer that reuses
    another's selection has none)."""
    return {
        i: layer.self_attn.indexer
        for i, layer in enumerate(inner.layers)
        if getattr(getattr(layer, "self_attn", None), "indexer", None) is not None
    }


def bind_indexers(inner, tp: int, rank: int) -> IndexerBinding | None:
    """The DSA layers' indexers, or None (with the reason logged) when the
    kernel cannot serve them."""
    indexers = indexer_modules(inner)
    cfg = get_current_atom_config()
    why = _refusal(cfg, tp, rank, indexers)
    if why:
        logger.info("mono DSA indexer off: %s", why)
        return None
    # without DCP every rank stores every key, the TP ranks split the scoring
    replicated = cfg.decode_context_parallel_size == 1
    shape = shape_of(next(iter(indexers.values())), cfg.kv_cache_block_size)
    return IndexerBinding(indexers, shape, tp, cfg.max_model_len, replicated)


# an Indexer's ``(hook, layer id)`` while hooked
_HOOKED = "_mono_indexer_hook"


class IndexerHook:
    """The DSA layers' indexers routed through the mono launch on a step a
    runner serves with it (``serving``), their own forward on any other.

    Installed on every layer when made; ``remove`` gives each its class's
    forward back. The patched forward is one opaque op (``mono_dsa_indexer``)
    whose eager body decides, so a layer stack ``torch.compile`` traces keeps
    the op as a leaf and the decision is the step's, not the trace's. That
    holds only for a trace made after the routing: a stack compiled later
    (a production one, at its warmup) is hooked when the model is wrapped,
    and the runner ``attach``-es its buffers once it exists. It holds the
    step's buffers, not the runner, so the layers it patches keep no runner
    alive."""

    def __init__(self, modules: dict):
        self._modules = modules
        self._buffers = None
        self._tag0 = 0
        self._launcher = None
        for layer_id, module in self._modules.items():
            module.__dict__[_HOOKED] = (self, layer_id)
            module.forward = _routed_forward(module)

    def attach(self, binding: IndexerBinding, scratch, plain, ilog, peers, rank,
               tag0):  # fmt: skip
        """The runner's step buffers (``ilog`` a replicated build's,
        ``logits_plane``), its peer regions' start and its tags."""
        self._buffers = (
            binding.replicated, scratch, plain, ilog, peers, binding.peer_offset,
            rank,
        )  # fmt: skip
        self._tag0 = tag0

    def remove(self) -> None:
        for module in self._modules.values():
            module.__dict__.pop("forward", None)
            module.__dict__.pop(_HOOKED, None)

    @contextmanager
    def serving(self, launcher):
        """A step's forward with the layers on ``launcher`` (None: their own)."""
        self._launcher = launcher
        try:
            yield
        finally:
            self._launcher = None

    def dispatch(self, layer_id, module, hidden_states, qr, qr_scale, positions, k,
                 weights) -> None:  # fmt: skip
        """One layer's indexer this step: the mono launch, or its own forward."""
        if self._launcher is None:
            type(module).forward(
                module,
                hidden_states,
                qr,
                qr_scale,
                positions,
                projected_k_weights=(k, weights),
            )
            return
        args = launch_args(
            module, qr, positions, k, weights, *self._buffers, self._tag0 + layer_id
        )
        if args is not None:
            _run_compiled(self._launcher, *args, torch.cuda.current_stream())


def _routed_forward(module):
    """``module``'s forward while hooked: its operands into the opaque op."""

    def forward(hidden_states, qr, qr_scale, positions, rotary_emb=None, *,
                projected_k_weights=None):  # fmt: skip
        k, weights = projected_k_weights
        torch.ops.aiter.mono_dsa_indexer(
            hidden_states,
            qr,
            qr_scale,
            positions,
            k,
            weights,
            module.prefix,
            module.k_cache.kv_cache[0],
            module.sparse_kv_indices_buffer,
            module.dcp_sparse_kv_indptr_buffer,
            module.dcp_owned_counts_buffer,
        )

    return forward


def mono_dsa_indexer(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    qr_scale: torch.Tensor | None,
    positions: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    layer_name: str,
    kv_cache: torch.Tensor,
    sparse_kv_indices_buffer: torch.Tensor,
    dcp_sparse_kv_indptr_buffer: torch.Tensor,
    dcp_owned_counts_buffer: torch.Tensor,
) -> None:
    """A hooked layer's indexer (``IndexerHook.dispatch``). The buffers it
    writes are arguments only to be declared mutated: the MLA reads them next,
    and an undeclared write would let a compiler order that read first."""
    module = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    hook, layer_id = module.__dict__[_HOOKED]
    hook.dispatch(layer_id, module, hidden_states, qr, qr_scale, positions, k, weights)


direct_register_custom_op(
    op_name="mono_dsa_indexer",
    op_func=mono_dsa_indexer,
    mutates_args=[
        "kv_cache",
        "sparse_kv_indices_buffer",
        "dcp_sparse_kv_indptr_buffer",
        "dcp_owned_counts_buffer",
    ],
    fake_impl=lambda *_args, **_kwargs: None,
)


def launch_args(
    module, qr, positions, k, weights, replicated, scratch, plain, ilog, peers,
    peer_off, rank, tag,
) -> list | None:  # fmt: skip
    """The kernel's arguments for one DSA layer's decode step, or None on a
    dummy run (the original op skips those too: nothing is written)."""
    fwd = get_forward_context()
    if fwd.context.is_dummy_run:
        return None
    md = fwd.attn_metadata
    # a request's context: each rank's share of a row follows from it
    # (``select.row_kind``), so no second published count can disagree
    ctx = md.context_lens
    if ctx.dtype != torch.int32:
        raise RuntimeError(f"mono indexer: context_lens {ctx.dtype}, not int32")
    # the kernel reads a slot's low word (``2 t``)
    if md.slot_mapping.dtype != torch.int64:
        raise RuntimeError(f"mono indexer: slot_mapping {md.slot_mapping.dtype}")
    if k.stride(1) != 1 or weights.stride(1) != 1 or k.stride(0) != weights.stride(0):
        raise RuntimeError("mono indexer: k and head weights must share a row stride")
    # a request's row each; a verify's token reads its request's through its
    # batch id (a plain decode's token r is request r: none published)
    bt = md.block_tables
    bids = md.batch_id_per_q_token
    if md.max_seqlen_q > 1 and (bids is None or bids.dtype != torch.int32):
        raise RuntimeError(f"mono indexer: batch ids {bids}, not int32")
    # replicated: the whole selection at the MLA's own indptr (read); under
    # DCP the kernel writes this rank's owned slots' indptr and counts
    out_ptr = md.sparse_kv_indptr if replicated else module.dcp_sparse_kv_indptr_buffer
    if out_ptr.dtype != torch.int32:
        raise RuntimeError(f"mono indexer: indptr {out_ptr.dtype}, not int32")
    p = torch.Tensor.data_ptr
    rope = {"pos": 0, "cos": 0, "sin": 0, "cs_ld": 0, "maxpos": 0}
    if module.rope_dim:
        if positions.dtype != torch.int64:
            raise RuntimeError(f"mono indexer: positions {positions.dtype}, not int64")
        # the original's tables (``Indexer.forward_impl``): [positions, half]
        cos = module.rotary_emb.cos_cache.squeeze(-2).squeeze(-2)
        sin = module.rotary_emb.sin_cache.squeeze(-2).squeeze(-2)
        if cos.dtype != torch.bfloat16 or cos.stride() != sin.stride():
            raise RuntimeError(f"mono indexer: RoPE tables {cos.dtype} {cos.stride()}")
        rope = {"pos": p(positions), "cos": p(cos), "sin": p(sin),
                "cs_ld": cos.stride(0), "maxpos": cos.size(0)}  # fmt: skip
    return kix.ABI.pack({
        "qr": p(qr), "qr_ld": qr.stride(0), "w_qb": p(module.wq_b.weight),
        "kp": p(k), "wp": p(weights), "kw_ld": k.stride(0),
        "kn_w": p(module.k_norm.weight), "kn_b": p(module.k_norm.bias),
        "slots": p(md.slot_mapping), **rope, "cache": p(module.k_cache.kv_cache[0]),
        "ctx": p(ctx), "bids": p(bids) if md.max_seqlen_q > 1 else 0,
        "bt": p(bt), "bt_ld": bt.stride(0),
        "out_idx": p(module.sparse_kv_indices_buffer),
        "out_ptr": p(out_ptr),
        "out_cnt": p(module.dcp_owned_counts_buffer),
        "scratch": p(scratch), "plain": p(plain),
        "ilog": p(ilog) if replicated else 0,
        "peers": peers.addresses.data_ptr(), "peer_off": peer_off, "rank": rank,
        "layer": tag,
    })  # fmt: skip
