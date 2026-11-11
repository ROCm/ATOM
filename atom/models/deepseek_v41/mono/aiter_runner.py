# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One TP rank's V4.1 mono decode on AITER's routed decode-layer ops
(``ATOM_DSV41_MONO_BACKEND=aiter``, ``aiter_gate``).

The forward is ``runner.V41MonoDecodeRunner``'s layer loop with its two
launches a layer replaced by AITER's two ops (``aiter_args`` maps the
arguments):

    attn_pre(layer_index=i)      K1: the seam .. the rotated query, the window row
    compressor, indexer          ATOM's own modules, as the in-tree check mode runs them
    layer_post(layer_index=i)    K2: attention .. the reduced MoE output

What the contracts leave to the caller runs between the two calls, on the same
stream: the KV compressor (``Attention._compress_batch``, on K1's normed row)
and the indexer's scoring and top-k, whose ``step.selected`` columns become the
contract's ``topk_rows`` / ``topk_count`` (``aiter_args.selection_rows``). On an
FP8 index plane the scorer takes K1's FP8 index query and scaled head weights
(``Indexer.score_quantized``); on an FP4 plane, which the v1 contract has no
query for, ``Indexer.project`` re-derives it from K1's normed row and q latent
and ``Indexer.score`` selects -- the original decode's path, on K1's inputs.

A step begins at ``attn_pre`` with ``layer_index`` 0 (the provider's step
fence); every op runs on the current stream with the TP group coordinator as
``group``. The first step of a width runs eagerly (ATOM's warm-up forward
before that width's capture): ``prepare`` first asks AITER, without running
anything, whether the provider takes every layer's two calls at that width
(``aiter.plugin.explain``), agrees the verdict over the TP group and logs each
layer's route once; the step's served calls then compile the provider's
kernels and map its peer memory, so the capture that follows only records. A
width that reaches a capture without that eager step is refused (strict: an
error), and every call's recorded route is checked after it ran.

Not covered (refused at install, ``aiter_gate.refusal``): DP attention, expert
parallel. The DSpark draft keeps ATOM's in-tree mono draft runner.
"""

import logging

import torch
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)

from atom.config import get_current_atom_config
from atom.distributed.group_agreement import group_agree
from atom.model_ops.deepseek_v41.mhc import SinglePassHCState
from atom.models.deepseek_v41.mono import aiter_args as args
from atom.models.deepseek_v41.mono import aiter_gate
from atom.models.deepseek_v41.mono.aiter_args import ATTN_PRE, LAYER_POST
from atom.models.deepseek_v41.mono.config import MAX_ROWS
from atom.models.deepseek_v41.mono.hi_gate import log_once
from atom.models.deepseek_v41.mono.kernels.dims import Dims
from atom.models.deepseek_v41.mono.step_meta import KMETA, write_step_meta
from atom.models.deepseek_v41.mono.weights import bind_model
from atom.mono.runtime.aux_taps import AuxTaps, bind_aux_taps
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")

# the step's row buffers, allocated for the widest step and narrowed to a
# step's rows (``_use_rows``): each with its row dim
_ROW_BUFFERS = (
    ("res_attn", 1), ("res_ffn", 1), ("gates_attn", 1), ("gates_ffn", 1),
    ("ffn_out", 1), ("q", 1), ("normed", 1), ("qr", 1), ("qr_scale", 1),
    ("iq", 0), ("iq_scale", 0), ("iw", 0), ("ring_rel", 0), ("key_meta", 0),
)  # fmt: skip


_SHORT = {ATTN_PRE: "attn_pre", LAYER_POST: "layer_post"}


def _rows(x, dim, s):
    if isinstance(x, tuple):
        return tuple(_rows(v, dim, s) for v in x)
    return x.narrow(dim, 0, s)


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class V41AiterDecodeRunner:
    """Binds every rank's weights and buffers, or none (``bind_agreed``)."""

    max_rows = MAX_ROWS

    def __init__(self, model, drafter, ops: aiter_gate.AiterOps | None = None) -> None:
        self.ops = ops if ops is not None else aiter_gate.load_ops()
        self.group = get_tp_group()
        self.strict = aiter_gate.strict()
        bind_agreed(lambda: self._bind(model, drafter), self.group.cpu_group)
        # widths preflighted (and agreed on) by every rank
        self._served: set[int] = set()
        self._ring_starts = None

    # -- binding -----------------------------------------------------------

    def _bind(self, model, drafter) -> None:
        self.model = model
        self.tp = get_tensor_model_parallel_world_size()
        self.rank = get_tensor_model_parallel_rank()
        self.layers = bind_model(model, Dims(self.tp, 1))
        self.aux = (
            AuxTaps((), ())
            if drafter is None
            else bind_aux_taps(drafter, model.config.hidden_size)
        )
        for rope in (model.window_rope, model.global_rope):
            for table in (rope.cos_cache, rope.sin_cache):
                if table.dtype != torch.bfloat16 or table.shape[-1] != args.ROPE_HALF:
                    raise MonoUnsupported(
                        f"rope table {table.dtype} {tuple(table.shape)}"
                    )
        config = get_current_atom_config()
        self.index_fp4 = config.index_cache_dtype == "fp4"
        self.static = []
        for spec, block, weights in zip(model.topology, model.layers, self.layers):
            feeds_index = block.attn.indexer is not None
            pre = args.attn_pre_weights(weights, feeds_index)
            post = args.layer_post_weights(weights)
            want_pre, want_post = args.weight_shapes(self.tp, feeds_index)
            errors = args.shape_errors(pre, want_pre) + args.shape_errors(
                post, want_post
            )
            if errors:
                raise MonoUnsupported(
                    f"layer {spec.layer_id}: not the AITER contract's: "
                    + "; ".join(errors)
                )
            self.static.append((pre, post))
        self._allocate(model.config, torch.device("cuda", torch.cuda.current_device()))

    def _allocate(self, config, dev) -> None:
        s, hc, h = self.max_rows, config.hc_mult, config.hidden_size
        bf, f32 = torch.bfloat16, torch.float32

        def gates():
            return (
                torch.empty(1, s, hc, device=dev, dtype=f32),
                torch.empty(1, s, hc, device=dev, dtype=f32),
                torch.empty(1, s, hc, hc, device=dev, dtype=f32),
            )

        self.res_attn = torch.empty(1, s, hc, h, device=dev, dtype=bf)
        self.res_ffn = torch.empty(1, s, hc, h, device=dev, dtype=bf)
        self.gates_attn, self.gates_ffn = gates(), gates()
        self.ffn_out = torch.empty(1, s, h, device=dev, dtype=bf)
        heads = config.num_attention_heads // self.tp
        self.q = torch.empty(1, s, heads, args.HEAD_DIM, device=dev, dtype=bf)
        # K1's index group (FULL / REINDEX layers): the normed row (the
        # compressor's input), the ROUND-rule q latent, the FP8 index query
        self.normed = torch.empty(1, s, h, device=dev, dtype=bf)
        self.qr = torch.empty(1, s, args.Q_RANK, device=dev, dtype=args.FP8)
        self.qr_scale = torch.empty(
            1, s, args.Q_RANK // 32, device=dev, dtype=torch.float8_e8m0fnu
        )
        ih, idim = config.index_n_heads, config.index_head_dim
        self.iq = torch.empty(s, ih, idim, device=dev, dtype=args.FP8)
        self.iq_scale = torch.empty(s, ih, device=dev, dtype=f32)
        self.iw = torch.empty(s, ih, device=dev, dtype=f32)
        # ``write_step_meta``'s: each token's ring row before the layer's ring
        # start, and its window key metadata
        self.ring_rel = torch.empty(s, device=dev, dtype=torch.int32)
        self.key_meta = torch.empty(s, KMETA, device=dev, dtype=torch.int32)
        self._full = {name: getattr(self, name) for name, _ in _ROW_BUFFERS}

    def _use_rows(self, s: int) -> None:
        if not 0 < s <= self.max_rows:
            raise ValueError(f"{s} rows is not a step of this runner")
        for name, dim in _ROW_BUFFERS:
            setattr(self, name, _rows(self._full[name], dim, s))

    # -- widths ------------------------------------------------------------

    def prepare(self, rows: int) -> bool:
        """Whether this ``rows``-row step runs here. A width's first step is
        eager: it is preflighted (``_preflight``) and agreed over the TP group;
        one first seen under a capture is refused (strict: an error)."""
        if not 0 < rows <= self.max_rows:
            return False
        if rows in self._served:
            return True
        if _capturing():
            why = (
                f"a {rows}-row step reached a graph capture before an eager "
                "step of that width warmed the provider"
            )
            if self.strict:
                raise RuntimeError(f"V4.1 mono aiter backend: {why}")
            log_once(logging.WARNING, "V4.1 mono aiter backend declines: %s", why)
            return False
        metadata = get_forward_context().attn_metadata
        ok, lines = self._preflight(rows, metadata.cache, metadata.step)
        agreed = group_agree(ok, self.group.cpu_group)
        for line in lines:
            log_once(logging.INFO, "V4.1 mono aiter %d rows: %s", rows, line)
        if not agreed:
            first = next((x for x in lines if "not served" in x), "another TP rank")
            why = f"{rows} rows: the provider does not serve every call ({first})"
            if self.strict:
                raise RuntimeError(
                    f"V4.1 mono aiter backend: {why}; ATOM_DSV41_MONO_AITER_STRICT=0 "
                    "runs AITER's native fallback instead"
                )
            log_once(
                logging.WARNING,
                "V4.1 mono aiter backend: %s; AITER falls back to its native "
                "reference for those calls (ATOM_DSV41_MONO_AITER_STRICT=0)",
                why,
            )
        self._served.add(rows)
        return True

    def _preflight(self, rows: int, cache, step) -> tuple[bool, list[str]]:
        """Every layer's two calls at ``rows`` rows asked of AITER without
        running them (the provider reads shapes and dtypes only: the step
        integers stand in as empty tensors). ``(all served, lines)``: one line
        per run of layers with the same routes."""
        self._use_rows(rows)
        self._ensure_ring_starts(cache)
        dev = self.ring_rel.device
        kv_slots = torch.empty(rows, device=dev, dtype=torch.int32)
        window = torch.empty(rows, args.KMETA, device=dev, dtype=torch.int32)
        selected = torch.empty(rows, args.TOPK, device=dev, dtype=torch.int32)
        count = torch.empty(rows, device=dev, dtype=torch.int32)
        no_sel = args.no_selection(rows, dev)
        positions = self._positions(cache, step, rows)
        stream = torch.cuda.current_stream()
        routes, all_ok = [], True
        for i, (spec, block) in enumerate(zip(self.model.topology, self.model.layers)):
            # the state a layer meets: an owed fold from the FFN seam's buffers,
            # or (layer 0, an Engram layer) a residual of its own, which
            # ``res_attn`` stands in for; K2 reads ``res_attn`` either way
            fold = i != 0 and block.engram is None
            state = (
                SinglePassHCState(self.res_ffn, self.gates_ffn[0], self.ffn_out,
                                  *self.gates_ffn[1:])
                if fold
                else SinglePassHCState(self.res_attn, self.gates_ffn[0])
            )  # fmt: skip
            pre = self._attn_pre_kwargs(
                i, spec, block, state, cache, positions, kv_slots, stream
            )
            topk = (selected, count) if spec.ratio else no_sel
            post = self._layer_post_kwargs(
                i, spec, block, self.res_attn, cache, positions, topk, window, stream
            )
            line = []
            for op, kwargs in ((ATTN_PRE, pre), (LAYER_POST, post)):
                ok, why = aiter_gate.explain(self.ops, op, kwargs)
                if not ok and self._step_order_only(op, kwargs):
                    ok = True
                    why = (
                        f"{aiter_gate.provider_name()} in step order "
                        "(dry query: NO_WORKLOAD)"
                    )
                all_ok &= ok
                verdict = "served by" if ok else "not served:"
                line.append(f"{_SHORT[op]} {verdict} {why}")
            routes.append("; ".join(line))
        return all_ok, _runs(routes)

    def _step_order_only(self, op: str, kwargs: dict) -> bool:
        """Whether the provider's only objection to this call is that no step is
        open here (``NO_WORKLOAD``: "could serve, but is not ready for this
        call").

        ``_preflight`` asks every layer's two calls without running any, so the
        provider's step cursor -- a call is served only once the step's served
        ``attn_pre(layer_index=0)`` opened it -- declines everything but layer
        0's ``attn_pre``. Outside a graph capture (``prepare`` refuses a width
        first seen under one) that cursor is the provider's only ``NO_WORKLOAD``
        rule, so a dry ``NO_WORKLOAD`` says the call is covered and would be
        served in step order, while ``NO_SCHEMA`` / ``OUT_OF_CONTRACT`` -- what
        this preflight exists to catch -- stay declines. Nothing is relaxed:
        every call's recorded route is still checked after it ran (``routed``),
        so a call that really goes native is an error under strict.
        """

        return aiter_gate.provider_reason(self.ops, op, kwargs) == "NO_WORKLOAD"

    def _ensure_ring_starts(self, cache) -> None:
        """Each layer's ring start in pool rows (fixed once the cache is
        allocated), as a device tensor made outside any capture."""
        key = cache.num_pages
        if self._ring_starts is not None and self._ring_starts[0] == key:
            return
        starts = [
            cache.geometry.window(spec.layer_id, cache.num_pages).ring_start
            for spec in self.model.topology
        ]
        tensor = torch.tensor(starts, dtype=torch.int64, device=self.ring_rel.device)
        self._ring_starts = (key, tensor)

    # -- the step ----------------------------------------------------------

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        model = self.model
        metadata = get_forward_context().attn_metadata
        cache, step = metadata.cache, metadata.step
        if cache.packed:
            raise MonoUnsupported("packed KV cache")
        rows = input_ids.numel()
        self._use_rows(rows)
        self._ensure_ring_starts(cache)
        hidden = model.embed(input_ids).view(1, rows, -1)
        model.begin_forward(hidden, None)
        # Engram staging's side-stream all-gather spins on CUs until every
        # rank joins it: beside a persistent kernel it deadlocks (``runner``)
        rows_staged = metadata.engram_embeddings
        if getattr(rows_staged, "stage", None) is not None:
            rows_staged.join()
        write_step_meta(
            step,
            cache.geometry,
            cache.geometry.window(0, cache.num_pages),
            self.ring_rel,
            self.key_meta,
            {},
            (0, rows),
        )
        ring_starts = self._ring_starts[1]
        kv_slots = args.layer_kv_slots(self.ring_rel, ring_starts)
        windows = args.layer_window_meta(self.key_meta, ring_starts)
        no_sel = args.no_selection(rows, self.ring_rel.device)
        selections = {}
        positions = self._positions(cache, step, rows)
        stream = torch.cuda.current_stream()
        state = SinglePassHCState.from_embeddings(hidden, model.config.hc_mult)
        for i, (spec, block) in enumerate(zip(model.topology, model.layers)):
            rope = model.global_rope if spec.ratio else model.window_rope
            if block.engram is not None:
                state = state.settle()
                state = SinglePassHCState(
                    block.engram_forward(state.residual, None, None), state.pre_mix
                )
            pre = self._attn_pre_kwargs(
                i, spec, block, state, cache, positions, kv_slots[i], stream
            )
            self._call(ATTN_PRE, i, pre)
            # the seam's residual: K1's fold, or the state's own without one
            residual = self.res_attn if state.pending is not None else state.residual
            if block.attn.indexer is not None:
                block.attn._compress_batch(self.normed, cache, step, rope)
                self._score(block, cache, step, rope)
            topk = (
                self._selection(spec, step, cache, selections) if spec.ratio else no_sel
            )
            post = self._layer_post_kwargs(
                i, spec, block, residual, cache, positions, topk, windows[i], stream
            )
            self._call(LAYER_POST, i, post)
            pre_m, post_m, comb = self.gates_ffn
            state = SinglePassHCState(self.res_ffn, pre_m, self.ffn_out, post_m, comb)
        hidden = state.collapse()
        model.end_forward(hidden, None)
        return hidden.squeeze(0)

    def _call(self, op: str, layer: int, kwargs: dict) -> None:
        """One routed call, then its recorded route held to the provider."""
        self.ops.call(op)(**kwargs)
        ok, why = aiter_gate.routed(self.ops, op)
        if ok:
            return
        if self.strict:
            raise RuntimeError(
                f"V4.1 mono aiter backend: layer {layer} {op} did not run on "
                f"{aiter_gate.provider_name()}: {why} "
                "(ATOM_DSV41_MONO_AITER_STRICT=1)"
            )
        log_once(
            logging.WARNING,
            "V4.1 mono aiter backend: layer %d %s not on the provider: %s",
            layer,
            op,
            why,
        )

    def _score(self, block, cache, step, rope) -> None:
        """The indexer's scoring and top-k into ``step.selected`` (and a
        candidate producer's ``step.candidates``), on K1's outputs."""
        indexer = block.attn.indexer
        if self.index_fp4:
            # no FP4 query in the v1 contract: the original projection of
            # K1's normed row and q latent (``check._index_projections``)
            query, weights = indexer.project(
                self.normed, self.qr, self.qr_scale, cache, step, rope
            )
            indexer.score(query, weights, cache, step)
        else:
            # K1's per-(row, head) FP8 query and its scale-folded head weights,
            # as the in-tree check mode scores them
            indexer.score_quantized(self.iq, self.iw, cache, step)

    def _selection(self, spec, step, cache, selections: dict) -> tuple:
        """This layer's ``(topk_rows, topk_count)``: its top-k owner's
        selection as pool rows of its KV owner, once a step each."""
        key = (spec.topk_owner, spec.kv_owner, spec.ratio)
        if key not in selections:
            geometry = cache.geometry
            selections[key] = args.selection_rows(
                step.selected[spec.topk_owner][0],
                step.batch_ids,
                step.block_tables,
                geometry.rows_per_page(spec.ratio),
                geometry.page_bytes // geometry.row_bytes,
                geometry.main_offset(spec.kv_owner),
            )
        return selections[key]

    @staticmethod
    def _positions(cache, step, rows) -> torch.Tensor:
        positions = cache.rope_positions(step).narrow(0, 0, rows)
        return positions if positions.dtype == torch.int64 else positions.long()

    def _attn_pre_kwargs(
        self, i, spec, block, state, cache, positions, kv_slots, stream
    ):
        fold = state.pending is not None
        if fold != (i != 0 and block.engram is None):
            raise MonoUnsupported(f"layer {spec.layer_id}: owed post is not as built")
        rope = self.model.global_rope if spec.ratio else self.model.window_rope
        aux = (
            self.aux.buffers[self.aux.layer_ids.index(spec.layer_id)][
                : kv_slots.shape[0]
            ]
            if spec.layer_id in self.aux.layer_ids
            else None
        )
        index = (
            {
                "normed": self.normed, "qr": self.qr, "qr_scale": self.qr_scale,
                "iq": self.iq, "iq_scale": self.iq_scale, "iw": self.iw,
            }
            if block.attn.indexer is not None
            else None
        )  # fmt: skip
        return args.attn_pre_kwargs(
            self.static[i][0],
            layer_index=i,
            residual=state.residual,
            pre_mix=state.pre_mix,
            fold=(state.pending, state.post_mix, state.combination) if fold else None,
            residual_out=self.res_attn if fold else None,
            rope=rope,
            positions=positions,
            kv_cache=cache.pool,
            kv_slots=kv_slots,
            gates_out=self.gates_attn,
            q_out=self.q,
            index_out=index,
            aux_out=aux,
            group=self.group,
            stream=stream,
        )

    def _layer_post_kwargs(
        self, i, spec, block, residual, cache, positions, topk, window, stream
    ):
        rope = self.model.global_rope if spec.ratio else self.model.window_rope
        ring = cache.geometry.window(spec.layer_id, cache.num_pages)
        return args.layer_post_kwargs(
            self.static[i][1],
            layer_index=i,
            q=self.q,
            kv_cache=cache.pool,
            topk_rows=topk[0],
            topk_count=topk[1],
            window_meta=window,
            positions=positions,
            rope=rope,
            residual=residual,
            gates_in=self.gates_attn,
            residual_out=self.res_ffn,
            gates_out=self.gates_ffn,
            ffn_out=self.ffn_out,
            softmax_scale=block.attn.softmax_scale,
            ring_slots=ring.ring_slots,
            group=self.group,
            stream=stream,
        )


def _runs(routes: list[str]) -> list[str]:
    """``routes`` (one a layer) as runs of consecutive layers with one route."""
    lines, start = [], 0
    for i in range(1, len(routes) + 1):
        if i == len(routes) or routes[i] != routes[start]:
            span = f"layer {start}" if i - 1 == start else f"layers {start}-{i - 1}"
            lines.append(f"{span}: {routes[start]}")
            start = i
    return lines


def create_runner(model, drafter) -> V41AiterDecodeRunner:
    """The runner ``LazyRunner`` creates on the first eligible step. Strict:
    a refusal of the loaded model is an error, not the original path."""
    try:
        return V41AiterDecodeRunner(model, drafter)
    except (MonoUnsupported, aiter_gate.Unavailable) as why:
        if aiter_gate.strict():
            raise RuntimeError(
                f"ATOM_DSV41_MONO_BACKEND=aiter, but the model cannot run on it: {why}"
            ) from why
        if isinstance(why, MonoUnsupported):
            raise
        raise MonoUnsupported(str(why)) from why
