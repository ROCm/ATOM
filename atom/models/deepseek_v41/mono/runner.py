# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One TP rank's V4.1 mono decode: the bound weights, the step's buffers, and the
forward that runs them.

The forward is the model's own layer loop with each layer in two mono
kernels: K1 (``kernels.attn_pre``) from the attention seam to the rotated query
and the window row; K2 (``kernels.layer_post``) K2a (``kernels.attn_post``: a
selecting layer's indexer, the sparse decode attention, the output projections,
their all-reduce (in-kernel, over this runner's peer buffer), the FFN seam and
ffn_norm) and K2b (``kernels.moe``: the MoE and its all-reduce) in one launch.
``ATOM_MONO_CHECK`` launches K2a and K2b apart, to compare each. Between the
kernels the original modules still run the compressor and Engram.
``draft_runner`` runs the DSpark draft's layers on the same kernels.
"""

import logging
import struct
from dataclasses import dataclass

import torch
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)

from atom.model_ops.deepseek_v41.candidate_table import candidate_block_table
from atom.model_ops.deepseek_v41.mhc import SinglePassHCState
from atom.model_ops.v4_kernels.paged_decode import LOG2E
from atom.models.deepseek_v41.config import AttentionMode
from atom.models.deepseek_v41.mono import index_plan as ip
from atom.models.deepseek_v41.mono.check import (
    check_front,
    check_moe,
    check_post,
    check_selection,
)
from atom.models.deepseek_v41.mono.config import MAX_ROWS
from atom.models.deepseek_v41.mono.kernels import attn_post as k2a
from atom.models.deepseek_v41.mono.kernels import attn_pre as k1
from atom.models.deepseek_v41.mono.kernels import debug as kernel_debug
from atom.models.deepseek_v41.mono.kernels import layer_post as k2
from atom.models.deepseek_v41.mono.kernels import moe as k2b
from atom.models.deepseek_v41.mono.kernels.dims import Dims
from atom.models.deepseek_v41.mono.kernels.index_query import (
    QS_BYTES,
    QS_M_TILES,
)
from atom.models.deepseek_v41.mono.kernels.moe_shape import EXPERTS, TOPK
from atom.models.deepseek_v41.mono.state import bind_aux_taps
from atom.models.deepseek_v41.mono.step_meta import KMETA, write_step_meta
from atom.models.deepseek_v41.mono.walk_plan import walk_plan_name
from atom.models.deepseek_v41.mono.weights import bind_model
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed
from atom.mono.runtime.debug import DIAG_BYTES, given_up_waits, raise_if_given_up
from atom.mono.runtime.lifecycle import owned_peer_buffer
from atom.mono.runtime.mailboxes import StepMailboxes
from atom.mono.runtime.timeline import LayerTimeline
from atom.mono.runtime.widths import WidthBuilds
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")

# K2a's and K2b's mailbox tags start past every layer's K1 and K2a
POST_TAG0 = 64
MOE_TAG0 = 128
# the step's row buffers, allocated for the widest step (``_allocate``) and
# narrowed to a step's rows (``_use_rows``): each with its row dim
_ROW_BUFFERS = (
    ("res_attn", 1), ("res_ffn", 1), ("gates_attn", 1), ("gates_ffn", 1),
    ("ffn_normed", 1), ("ffn_out", 1), ("q", 1), ("normed", 1), ("qr", 1),
    ("qr_scale", 1), ("ring_rel", 0), ("key_meta", 0), ("ilog", 0), ("ibmax", 0),
    ("isel", 0), ("irow", 0), ("icand", 0), ("no_bound", 0), ("iq", 0),
    ("iq_scale", 0), ("iw", 0),
)  # fmt: skip


def _rows(x, dim, s):
    """``x`` (a tensor, or a tuple / dict of them) narrowed to s rows."""
    if isinstance(x, tuple):
        return tuple(_rows(v, dim, s) for v in x)
    if isinstance(x, dict):
        return {k: _rows(v, dim, s) for k, v in x.items()}
    return x.narrow(dim, 0, s)


@dataclass(frozen=True)
class RowsBuild:
    """The kernels of an S-row step: K1's build a layer and their launchers,
    and K2 (or, with ``check``, K2a and K2b apart)."""

    keys: list
    launchers: dict
    layer_post: object = None
    attn_post: object = None
    moe: object = None


# the layers whose attention norm has other readers (the compressor, the indexer)
_FEEDS_INDEX = (AttentionMode.FULL, AttentionMode.REINDEX)


def f32_bits(x: float) -> int:
    """``x`` rounded to fp32, as the int32 of its bits (a kernel's Int32 arg)."""
    return struct.unpack("<i", struct.pack("<f", x))[0]


class V41MonoDecodeRunner:
    """Binds every rank's weights and buffers, or none: a rank that cannot is a
    refusal every rank agrees on (``tp_agree``) before anything collective.

    A step is up to ``max_rows`` rows, whatever requests they belong to (a
    row's window comes from its own batch id, ``write_step_meta``); the buffers
    hold the widest and every width has its own kernel builds. The rows, the
    routing, the K1 builds and where RoPE positions come from are the target's
    here; ``draft_runner.DraftMonoRunner`` sets its own."""

    max_rows = MAX_ROWS
    experts, topk = EXPERTS, TOPK
    ug_groups = 2
    # the index plane is FP4: K1 quantizes the indexer query for, and K2a
    # scores with, ``index_score_fp4``
    index_fp4 = False

    def __init__(self, model, drafter, max_model_len: int, index_fp4: bool) -> None:
        # the indexer's widest bound: a ratio-1 layer sees every position. No
        # token's bound may exceed it (positions stay below max_model_len): its
        # selection would never reach a final level nor publish IRDY
        self.index_bound_max = ip.cdiv(max_model_len, ip.CHUNK) * ip.CHUNK
        self.index_fp4 = index_fp4
        self._bind_agreed(lambda: self._bind(model, drafter))

    def _bind_agreed(self, bind) -> None:
        """``bind()`` on every rank, then, once every rank agreed it did, the
        collective peer buffer (the attention and FFN partials' landing zone)."""
        group = get_tp_group().cpu_group
        bind_agreed(bind, group)
        self.peers, self._finalizer = owned_peer_buffer(
            self,
            k2a.peer_bytes(self.max_rows, self.tp)
            + k2b.peer_bytes(self.max_rows, self.tp),
            group,
            self.rank,
            self.tp,
            self.scratch.device,
            debug=self.debug,
        )
        self.mailboxes = StepMailboxes(self.peers, self.scratch)
        self.widths = WidthBuilds(
            self._rows_build, self._row_kernels, group, "V4.1 mono"
        )

    def prepare(self, rows: int) -> bool:
        """Whether every rank holds the ``rows``-row kernels (``WidthBuilds``)."""
        return 0 < rows <= self.max_rows and self.widths.prepare(rows)

    def _bind(self, model, drafter) -> None:
        if drafter is None:
            raise MonoUnsupported("no drafter")
        self.model = model
        self._bind_rank()
        self.layers = bind_model(model, self.tp)
        self.aux = bind_aux_taps(drafter, model.config.hidden_size)
        for rope in (model.window_rope, model.global_rope):
            for table in (rope.cos_cache, rope.sin_cache):
                if table.dtype != torch.bfloat16 or table.shape[-1] != k1.HALF:
                    raise MonoUnsupported(
                        f"rope table {table.dtype} {tuple(table.shape)}"
                    )
        self.check = envs.ATOM_MONO_CHECK
        # check mode compares K2a's and K2b's outputs: two launches
        self.split_k2 = self.check
        self.debug = envs.ATOM_MONO_DEBUG
        tl_prefix = envs.ATOM_MONO_TIMELINE
        dev = torch.device("cuda", torch.cuda.current_device())
        post_points = (
            (("k2a", k2a.TL_POINTS), ("k2b", k2b.TL_POINTS))
            if self.check
            else (("k2", k2.TL_POINTS),)
        )
        self.timelines = (
            {
                name: LayerTimeline(
                    f"{tl_prefix}_{name}", len(model.layers), points, self.rank, dev
                )
                for name, points in (("k1", k1.TL_POINTS), *post_points)
            }
            if tl_prefix
            else {}
        )
        self.timeline = bool(self.timelines)
        # the layers that select (FULL / REINDEX): each its selection buffer
        self.selecting = [
            spec.layer_id
            for spec, block in zip(model.topology, model.layers)
            if block.attn.indexer is not None
        ]
        self._allocate(model.config)

    def index_query(self):
        """K1's indexer query as the original scorer takes it quantized: e4m3,
        or the FP4 (values, packed scales) of one-row sequences."""
        if not self.index_fp4:
            return self.iq
        rows = self.iq.shape[0]
        return (
            self.iq.view(rows, 1, *self.iq.shape[1:]),
            self.iq_scale.view(rows, 1, 1, ip.DIM // 32, 16, QS_M_TILES),
        )

    def _bind_rank(self) -> None:
        """This rank and the TP size every build is for (``config_refusal``
        refused the ones the kernels are not built for)."""
        self.rank = get_tensor_model_parallel_rank()
        self.tp = get_tensor_model_parallel_world_size()
        self.dims = Dims(self.tp)

    def _layer_keys(self, s: int) -> list:
        """K1's build of every layer for an s-row step."""
        # layer 0 starts from the embeddings and an Engram layer settles first:
        # neither seam has an owed post to fold
        return [
            k1.AttnPreBuild(
                tokens=s,
                fold=spec.layer_id != 0 and block.engram is None,
                feeds_index=spec.mode in _FEEDS_INDEX,
                index_fp4=self.index_fp4,
                aux=spec.layer_id in self.aux.layer_ids,
                tp=self.tp,
                timeline=self.timeline,
                diag_off=self.diag_off,
            )
            for spec, block in zip(self.model.topology, self.model.layers)
        ]

    def _post_key(self, s: int) -> k2.LayerPostBuild:
        return k2.LayerPostBuild(
            tokens=s,
            index_bound_max=self.index_bound_max,
            index_fp4=self.index_fp4,
            experts=self.experts,
            topk=self.topk,
            tp=self.tp,
            timeline=self.timeline,
            diag_off=self.diag_off,
            ug_groups=self.ug_groups,
        )

    def _rows_build(self, s: int) -> RowsBuild:
        """The s-row step's kernels."""
        keys = self._layer_keys(s)
        launchers = {key: k1.build_attn_pre(key) for key in set(keys)}
        post_key = self._post_key(s)
        if self.split_k2:
            key_a, key_b = post_key.halves()
            return RowsBuild(
                keys,
                launchers,
                attn_post=k2a.build_attn_post(key_a),
                moe=k2b.build_moe(key_b),
            )
        return RowsBuild(keys, launchers, layer_post=k2.build_layer_post(post_key))

    @staticmethod
    def _row_kernels(build: RowsBuild) -> list:
        """Every kernel of ``build``, each with its ABI."""
        kernels = [(launcher, k1.ABI) for launcher in build.launchers.values()]
        if build.layer_post is not None:
            return kernels + [(build.layer_post, k2.ABI)]
        return kernels + [(build.attn_post, k2a.ABI), (build.moe, k2b.ABI)]

    def _use_rows(self, s: int) -> None:
        """This step is s rows: its kernels, and every row buffer narrowed."""
        if not 0 < s <= self.max_rows:
            raise ValueError(f"{s} rows is not a step of this runner")
        self.build = self.widths[s]
        for name, dim in _ROW_BUFFERS:
            setattr(self, name, _rows(self._full[name], dim, s))

    def _allocate(self, config) -> None:
        dev = torch.device("cuda", torch.cuda.current_device())
        s, hc, h = self.max_rows, config.hc_mult, config.hidden_size
        bf = torch.bfloat16
        # explicit dtypes: the runner is built inside a warmup forward, whose
        # default dtype is the model's (bf16)
        f32 = torch.float32

        def gates():
            return (
                torch.empty(1, s, hc, device=dev, dtype=f32),  # pre
                torch.empty(1, s, hc, device=dev, dtype=f32),  # post
                torch.empty(1, s, hc, hc, device=dev, dtype=f32),  # comb
            )

        # each seam's residual and gates: the attention seam reads the FFN seam's
        # and the FFN seam the attention seam's, so one of each suffices
        self.res_attn = torch.empty(1, s, hc, h, device=dev, dtype=bf)
        self.res_ffn = torch.empty(1, s, hc, h, device=dev, dtype=bf)
        self.gates_attn, self.gates_ffn = gates(), gates()
        self.ffn_normed = torch.empty(1, s, h, device=dev, dtype=bf)
        # the FFN output, reduced: the next layer's owed post
        self.ffn_out = torch.empty(1, s, h, device=dev, dtype=bf)
        self.q = torch.empty(1, s, self.dims.heads, k1.HEAD_DIM, device=dev, dtype=bf)
        self.normed = torch.empty(1, s, h, device=dev, dtype=bf)
        self.qr = torch.empty(1, s, k1.Q_RANK, device=dev, dtype=torch.float8_e4m3fn)
        self.qr_scale = torch.empty(
            1, s, k1.Q_RANK // 32, device=dev, dtype=torch.float8_e8m0fnu
        )
        self.ring_rel = torch.empty(s, device=dev, dtype=torch.int32)
        # each token's window metadata (``attention._keys``)
        self.key_meta = torch.empty(s, KMETA, device=dev, dtype=torch.int32)
        # the indexer: its logits and block maxima, each selecting layer's
        # selection, the candidate blocks, a zero bound for a layer without keys
        # from a selection, and each selection's bound this step
        width = self.index_bound_max
        self.ilog = torch.empty(s, width, device=dev, dtype=f32)
        self.ibmax = torch.empty(s, ip.blocks(width), device=dev, dtype=f32)
        self.isel = {
            layer: torch.empty(s, ip.TOPK, device=dev, dtype=torch.int32)
            for layer in self.selecting
        }
        # each selection's pool rows, the attention's keys (every layer sharing
        # a selection shares its KV owner: ``index_score._selected_rows``)
        self.irow = {
            layer: torch.empty(s, ip.TOPK, device=dev, dtype=torch.int32)
            for layer in self.selecting
        }
        self.icand = torch.empty(s, ip.TOPK_BLOCKS, device=dev, dtype=torch.int32)
        self.no_bound = torch.zeros(s, device=dev, dtype=torch.int32)
        self.bounds = {}
        self.candidate_table = None
        # the indexer's query, its scales and the scaled head weights: e4m3 and
        # a scale a head, or E2M1 and E8M0s in the FP4 scorer's packed order
        # (its padding never written: zero, as ``pack_q_scales`` pads)
        heads, dim = config.index_n_heads, config.index_head_dim
        if self.index_fp4:
            self.iq = torch.empty(s, heads, dim // 2, device=dev, dtype=torch.uint8)
            self.iq_scale = torch.zeros(s, QS_BYTES, device=dev, dtype=torch.uint8)
        else:
            self.iq = torch.empty(s, heads, dim, device=dev, dtype=torch.float8_e4m3fn)
            self.iq_scale = torch.empty(s, heads, device=dev, dtype=f32)
        self.iw = torch.empty(s, heads, device=dev, dtype=f32)
        self.diag_off = -1
        post_key = self._post_key(s)
        key_a, key_b = post_key.halves()
        nbytes = max(
            k1.scratch_bytes(s),
            k2a.scratch_bytes(key_a),
            k2b.scratch_bytes(key_b),
            k2.scratch_bytes(post_key),
        )
        # a debug build's wait records: past every kernel's regions, cleared
        # with them every step
        if self.debug:
            self.diag_off = nbytes
            nbytes += DIAG_BYTES
        self.scratch = torch.zeros(nbytes, device=dev, dtype=torch.uint8)
        self._full = {name: getattr(self, name) for name, _ in _ROW_BUFFERS}

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        model = self.model
        metadata = get_forward_context().attn_metadata
        cache, step = metadata.cache, metadata.step
        if cache.packed:
            raise MonoUnsupported("packed KV cache")
        rows = input_ids.numel()
        self._use_rows(rows)
        self.mailboxes.begin_step()
        hidden = model.embed(input_ids).view(1, rows, -1)
        model.begin_forward(hidden, None)
        # Engram staging's side-stream all-gather spins on CUs until every
        # rank joins it: beside a kernel that needs every CU resident it
        # deadlocks, so it ends before the first one
        rows_staged = metadata.engram_embeddings
        if getattr(rows_staged, "stage", None) is not None:
            rows_staged.join()
        write_step_meta(
            step,
            cache.geometry,
            cache.geometry.window(0, cache.num_pages),
            self.ring_rel,
            self.key_meta,
        )
        state = SinglePassHCState.from_embeddings(hidden, model.config.hc_mult)
        for spec, block, weights, key in zip(
            model.topology, model.layers, self.layers, self.build.keys
        ):
            rope = model.global_rope if spec.ratio else model.window_rope
            if block.engram is not None:
                state = state.settle()
                state = SinglePassHCState(
                    block.engram_forward(state.residual, None, None), state.pre_mix
                )
            residual, gates = self._front(
                spec.layer_id, key, weights, state, cache, step, rope
            )
            if self.check:
                check_front(
                    self,
                    spec.layer_id,
                    block,
                    state,
                    rope,
                    cache,
                    step,
                    residual,
                    gates,
                )
            if block.attn.indexer is not None:
                block.attn._compress_batch(self.normed, cache, step, rope)
                if self.check:
                    # the original scorer, the reference the selection is held to
                    block.attn.indexer.score_quantized(
                        self.index_query(), self.iw, cache, step
                    )
            post = (spec, block, weights, residual, gates, step, cache, rope)
            if self.split_k2:
                self._attn_post(*post)
                if not self.check:
                    self._moe(spec.layer_id, weights)
            else:
                self._layer_post(*post)
            if spec.produces_candidates:
                # the REINDEX layers' block table over this layer's candidates
                self.candidate_table = candidate_block_table(
                    self.icand,
                    step.block_tables,
                    step.batch_ids,
                    cache.geometry.index_blocks_per_page(spec.ratio),
                    step.visible[spec.ratio],
                    rows_per_block=ip.BLOCK_ROWS,
                )
            if self.check:
                if block.attn.indexer is not None:
                    check_selection(self, spec, step)
                # the original index build, for the reference attention alone
                keys = cache.attention_indices(spec, step)[:2]
                check_post(self, block, keys, residual, gates, rope, cache, step)
                self._moe(spec.layer_id, weights)
                check_moe(self, block)
            pre, post, comb = self.gates_ffn
            state = SinglePassHCState(self.res_ffn, pre, self.ffn_out, post, comb)
        hidden = state.collapse()
        model.end_forward(hidden, None)
        self._raise_on_given_up_waits(rows)
        for timeline in self.timelines.values():
            timeline.step_done(rows)
        return hidden.squeeze(0)

    def _front(self, layer_id, key, weights, state, cache, step, rope):
        """K1 -> (the residual after the owed post, the seam's (pre, post, comb));
        the query, the window row and the indexer's inputs land in this runner's
        buffers."""
        if (state.pending is not None) != key.fold:
            raise MonoUnsupported(f"layer {layer_id}: owed post is not as built")
        res_out = self.res_attn if key.fold else state.residual
        pre, post, comb = self.gates_attn
        aux = self.aux.buffers[self.aux.layer_ids.index(layer_id)] if key.aux else None
        p = torch.Tensor.data_ptr
        fold = key.fold
        args = k1.ABI.pack({
            "res_in": p(state.residual), "pend": p(state.pending) if fold else 0,
            "post_in": p(state.post_mix) if fold else 0,
            "comb_in": p(state.combination) if fold else 0, "pre_in": p(state.pre_mix),
            "hc_fn": p(weights["hc_attn_fn"]), "hc_scale": p(weights["hc_attn_scale"]),
            "hc_base": p(weights["hc_attn_base"]),
            "attn_w": p(weights["attn_norm.weight"]),
            "wqkv": p(weights["attn.wqkv_a.weight"]),
            "wqkv_s": p(weights["attn.wqkv_a.weight_scale"]),
            "qn_w": p(weights["attn.q_norm.weight"]),
            "kvn_w": p(weights["attn.kv_norm.weight"]),
            "wqb": p(weights["attn.wq_b.weight"]),
            "wqb_s": p(weights["attn.wq_b.weight_scale"]),
            "cos": p(rope.cos_cache), "sin": p(rope.sin_cache),
            "pos": p(self._rope_positions(cache, step)), "ring": p(cache.pool),
            "ring_rows": p(self.ring_rel),
            "ring_off": cache.geometry.window(layer_id, cache.num_pages).ring_start,
            "res_out": p(res_out), "post_out": p(post), "comb_out": p(comb),
            "pre_out": p(pre), "q_out": p(self.q), "normed": p(self.normed),
            "qr_out": p(self.qr), "qrs_out": p(self.qr_scale),
            "aux": 0 if aux is None else p(aux), **self._index_projection_args(key, weights),
            "scratch": p(self.scratch), "layer": layer_id,
            "tl": self._timeline_ptr("k1", layer_id),
        })  # fmt: skip
        # the caller's stream: a FlyDSL launcher left to its default submits on the
        # NULL stream, outside a graph capture
        self.build.launchers[key](*args, stream=torch.cuda.current_stream())
        return res_out, (pre, post, comb)

    def _index_projection_args(self, key, weights) -> dict:
        """K1's indexer projections: FULL / REINDEX builds only."""
        names = ("iq_w", "iq_ws", "iw_w", "iq_out", "iqs_out", "iw_out")
        if not key.feeds_index:
            return dict.fromkeys(names, 0)
        p = torch.Tensor.data_ptr
        return {
            "iq_w": p(weights["attn.indexer.wq_b.weight"]),
            "iq_ws": p(weights["attn.indexer.wq_b.weight_scale"]),
            "iw_w": p(weights["attn.indexer.weights_proj.weight"]),
            "iq_out": p(self.iq), "iqs_out": p(self.iq_scale),
            "iw_out": p(self.iw),
        }  # fmt: skip

    def _layer_post(self, spec, block, weights, residual, gates, step, cache, rope):
        """K2: K2a's work, then K2b's, in one launch."""
        layer_id = spec.layer_id
        args = k2.ABI.pack({
            **self._attn_post_args(spec, block, weights, residual, gates, step, cache, rope),
            **self._moe_args(weights),
            **self._peer_args(),
            "layer": POST_TAG0 + layer_id, "moe_layer": MOE_TAG0 + layer_id,
            "tl": self._timeline_ptr("k2", layer_id),
        })  # fmt: skip
        self.build.layer_post(*args, stream=torch.cuda.current_stream())

    def _attn_post(self, spec, block, weights, residual, gates, step, cache, rope):
        """K2a: the decode sparse attention over this layer's keys (its class's
        ``key_meta``, the selection of its top-k owner, the window), the output
        projections and their all-reduce, folded into the residual by the FFN
        seam -> ``res_ffn``, the FFN gates, ``ffn_normed``."""
        layer_id = spec.layer_id
        args = k2a.ABI.pack({
            **self._attn_post_args(spec, block, weights, residual, gates, step, cache, rope),
            **self._peer_args(),
            # its own tags: K1's regions of this layer share the scratch
            "layer": POST_TAG0 + layer_id,
            "tl": self._timeline_ptr("k2a", layer_id),
        })  # fmt: skip
        self.build.attn_post(*args, stream=torch.cuda.current_stream())

    def _peer_args(self) -> dict:
        p = torch.Tensor.data_ptr
        return {"scratch": p(self.scratch), **self.peers.kernel_args()}

    def _attn_post_args(self, spec, block, weights, residual, gates, step, cache, rope):
        """K2a's own arguments."""
        layer_id = spec.layer_id
        p = torch.Tensor.data_ptr
        pre, post, comb = gates
        pre_out, post_out, comb_out = self.gates_ffn
        attn = block.attn
        geometry = cache.geometry
        meta = self.key_meta
        index = self._selection_args(spec, attn, cache, step)
        owner = spec.topk_owner if spec.ratio else None
        return {
            **index,
            "q": p(self.q), "pool": p(cache.pool),
            # a layer without a selection reads a dummy row it never uses
            "sel": p(meta if owner is None else self.irow[owner]),
            "topk": 1 if owner is None else ip.TOPK,
            "sbound": p(self.no_bound if owner is None else self.bounds[owner]),
            # a window layer's keys are all ring rows: it reads no block table
            "kmeta": p(meta),
            "table": p(step.block_tables) if spec.ratio else p(meta),
            "rows_per_page": geometry.rows_per_page(spec.ratio or 1),
            "page_rows": geometry.page_bytes // geometry.row_bytes,
            "main_off": geometry.main_offset(spec.kv_owner) if spec.ratio else 0,
            "ring_off": geometry.window(layer_id, cache.num_pages).ring_start,
            "ring_slots": geometry.window(layer_id, cache.num_pages).ring_slots,
            "sink": p(attn.attn_sink),
            "qk_scale": f32_bits(attn.softmax_scale * LOG2E),
            "pos": p(self._rope_positions(cache, step)),
            "cos": p(rope.cos_cache), "sin": p(rope.sin_cache),
            "woa": p(weights["attn.wo_a.weight"]),
            "woa_s": p(weights["attn.wo_a.weight_scale"]),
            "wob": p(weights["attn.wo_b.weight"]),
            "wob_s": p(weights["attn.wo_b.weight_scale"]),
            "res_in": p(residual), "post_in": p(post), "comb_in": p(comb),
            "pre_in": p(pre), "hc_fn": p(weights["hc_ffn_fn"]),
            "hc_scale": p(weights["hc_ffn_scale"]), "hc_base": p(weights["hc_ffn_base"]),
            "ffn_w": p(weights["ffn_norm.weight"]), "res_out": p(self.res_ffn),
            "post_out": p(post_out), "comb_out": p(comb_out), "pre_out": p(pre_out),
            "normed": p(self.ffn_normed),
        }  # fmt: skip

    def _moe(self, layer_id, weights) -> None:
        """K2b: router, routed and shared experts and their all-reduce ->
        ``ffn_out``."""
        key_a, _ = self._post_key(self.ffn_normed.shape[1]).halves()
        args = k2b.ABI.pack({
            "x": self.ffn_normed.data_ptr(), **self._moe_args(weights),
            **self._peer_args(), "layer": MOE_TAG0 + layer_id,
            "tl": self._timeline_ptr("k2b", layer_id),
            # where the merged launch's layout puts K2b's regions
            "scratch": self.scratch.data_ptr() + k2.moe_base(key_a),
        })  # fmt: skip
        self.build.moe(*args, stream=torch.cuda.current_stream())

    def _moe_args(self, weights) -> dict:
        """K2b's own arguments but its input."""
        p = torch.Tensor.data_ptr
        return {
            "gate_w": p(weights["ffn.gate.weight"]),
            "bias": p(weights["ffn.gate.e_score_correction_bias"]),
            "w13": p(weights["ffn.experts.w13_weight"]),
            "w13_s": p(weights["ffn.experts.w13_weight_scale"]),
            "w2": p(weights["ffn.experts.w2_weight"]),
            "w2_s": p(weights["ffn.experts.w2_weight_scale"]),
            "sgu": p(weights["ffn.shared_experts.gate_up_proj.weight"]),
            "sgu_s": p(weights["ffn.shared_experts.gate_up_proj.weight_scale"]),
            "sw2": p(weights["ffn.shared_experts.w2.weight"]),
            "sw2_s": p(weights["ffn.shared_experts.w2.weight_scale"]),
            "out": p(self.ffn_out),
        }  # fmt: skip

    def _raise_on_given_up_waits(self, rows: int) -> None:
        """A debug build's eager step: raise with every mailbox wait that gave up
        (a graph replay runs no Python: run it with ``--enforce-eager``)."""
        if not self.debug or torch.cuda.is_current_stream_capturing():
            return
        regions = {
            *k1.scratch_layout(rows),
            *k2.scratch_layout(self._post_key(rows)),
        }
        diag = self.scratch[self.diag_off : self.diag_off + DIAG_BYTES]
        waits = given_up_waits(diag, kernel_debug.region_names(regions))
        raise_if_given_up("V4.1 mono", self.rank, rows, waits, self.peers)

    def _rope_positions(self, cache, step) -> torch.Tensor:
        """Each row's position for RoPE (int64 [S])."""
        return cache.rope_positions(step)

    def _timeline_ptr(self, kernel, layer_id) -> int:
        timeline = self.timelines.get(kernel)
        return timeline.ptr(layer_id) if timeline is not None else 0

    def _selection_args(self, spec, attn, cache, step) -> dict:
        """K2a's indexer arguments: a selecting layer's scorer inputs, its bound
        (the one ``index_plan`` derives every cell from, recorded for the layers
        that reuse its selection) and outputs; dummies elsewhere."""
        p = torch.Tensor.data_ptr
        dummy = p(self.no_bound)
        args = {
            "iact": 0, "iq": dummy, "iw": dummy, "plane": dummy, "iqs": dummy,
            "pscale": dummy, "iplan": dummy, "itab": dummy,
            "itab_stride": 0, "itab_len": 1, "ibatch": dummy, "ishift": 0,
            "ibound": dummy, "ilog": p(self.ilog),
            "lstride": self.ilog.shape[1], "isel": dummy, "icand": dummy, "ilift": 0,
            "iprod": 0, "ibmax": p(self.ibmax), "bstride": self.ibmax.shape[1],
            "icout": dummy, "irow": dummy,
        }  # fmt: skip
        if attn.indexer is None:
            return args
        reindex = spec.candidate_source is not None
        shift = 0
        if reindex:
            table, bound = self.candidate_table
        elif self.index_fp4:
            # the request's PAGE table, an entry 2 ** shift pages of the plane
            per_page = cache.geometry.index_blocks_per_page(spec.ratio)
            assert per_page & (per_page - 1) == 0, per_page
            shift = per_page.bit_length() - 1
            table, bound = step.block_tables, step.visible[spec.ratio]
        else:
            table, bound = cache.unit_tiles(step, spec.ratio), step.visible[spec.ratio]
        batch_ids = step.batch_ids
        assert batch_ids.dtype == torch.int32 and batch_ids.stride(0) == 1
        self.bounds[spec.layer_id] = bound
        units = cache.index_units[spec.kv_owner]
        if self.index_fp4:
            args.update({
                "iqs": p(self.iq_scale), "pscale": p(units.scales),
                "iplan": (
                    dummy if reindex else p(step.planned[walk_plan_name(spec.ratio)])
                ),
            })  # fmt: skip
        args.update({
            "iact": 1, "iq": p(self.iq), "iw": p(self.iw),
            "plane": p(units.values), "itab": p(table),
            "itab_stride": table.stride(0), "itab_len": table.shape[1],
            "ibatch": p(batch_ids), "ishift": shift,
            "ibound": p(bound), "isel": p(self.isel[spec.layer_id]),
            "icand": p(self.icand), "ilift": int(reindex),
            "iprod": int(spec.produces_candidates), "icout": p(self.icand),
            "irow": p(self.irow[spec.layer_id]),
        })  # fmt: skip
        return args
