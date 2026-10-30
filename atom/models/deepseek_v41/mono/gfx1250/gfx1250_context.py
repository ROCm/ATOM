# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx950 peer/mailbox lifecycle shared by every gfx1250 decode layer."""

import torch
from aiter.dist.parallel_state import get_tp_group

from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_merged_post import (
    ARGUMENTS,
    _merged_launcher,
    merged_scratch_specs,
    prepare_merged_kernel,
)
from atom.models.deepseek_v41.mono.kernels.peer_layout import peer_bytes
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_attn_pre import (
    ARG_COUNT as K1_ARG_COUNT, _launcher as k1_launcher, prepare_attn_pre_kernel,
)
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed
from atom.mono.runtime.lifecycle import owned_peer_buffer
from atom.mono.runtime.mailboxes import StepMailboxes
from atom.mono.runtime.widths import WidthBuilds
from atom.models.deepseek_v41.mono.gfx1250.kernels import gfx1250_timeline as timeline_ops


def _layout(rows, tp, index_bound_max=0):
    specs = {
        **merged_scratch_specs(rows, tp, index_bound_max),
        "attention": ((rows, 5120), torch.bfloat16),
        "ffn": ((rows, 5120), torch.bfloat16),
        "residual": ((rows, 4, 5120), torch.bfloat16),
        "pre": ((rows, 4), torch.float32),
        "post": ((rows, 4), torch.float32),
        "comb": ((rows, 4, 4), torch.float32),
        "normed": ((rows, 5120), torch.bfloat16),
    }
    offset, layout = 0, {}
    for name, (shape, dtype) in specs.items():
        offset = (offset + 15) // 16 * 16
        count = 1
        for dim in shape:
            count *= dim
        size = count * torch.empty((), dtype=dtype, device="meta").element_size()
        layout[name] = (offset, size, shape, dtype)
        offset += size
    return (offset + 15) // 16 * 16, layout


class Gfx1250DecodeContext:
    """One model's uncached peer buffer, shared scratch and step fence.

    Parameter binding is a TP CPU agreement before the IPC handshake, exactly
    as V41MonoDecodeRunner. The model's native layer loop and compressor/index
    metadata remain the owner of layer order. No per-layer peer allocation or
    independent epoch protocol is introduced.
    """

    def __init__(self, model, cache, step=None):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("[gfx1250 K2] bind the model before graph capture")
        group = get_tp_group()
        self.tp, self.rank = group.world_size, group.rank_in_group
        self.cache = cache
        self.model = model
        self._prepared = set()
        self.argument_tables = {}
        self.rope_tables = {}
        self.rows = {}
        self.index_bound_max = ((model.max_length + 127) // 128) * 128

        def bind():
            from atom.models.deepseek_v41.mono.gfx1250.gfx1250_runner import LocalK1

            if self.tp != 2 or len(model.layers) != 40:
                raise MonoUnsupported("gfx1250 runtime requires TP2 and 40 layers")
            if cache.packed or cache.geometry.index_fp4:
                raise MonoUnsupported("gfx1250 K2 requires BF16 split KV and FP8 index")
            if cache.geometry.speculative_tokens:
                raise MonoUnsupported("gfx1250 K2 does not bind speculative decode")
            if self.index_bound_max > 8192:
                raise MonoUnsupported("gfx1250 in-K2 indexer currently supports max_length <= 8192")
            if step is None:
                raise MonoUnsupported("gfx1250 binding needs the decode metadata")
            # Native V4.1 RoPE stores BF16. K2's pointer ABI reads FP32:
            # bind compact FP32 tables once for both modes, shared by 40 layers.
            for rope in (model.window_rope, model.global_rope):
                self.rope_tables[id(rope)] = (
                    rope.cos_cache.float().contiguous(),
                    rope.sin_cache.float().contiguous(),
                )
            runners = []
            for layer in model.layers:
                runner = layer.__dict__.get("_gfx1250_k1")
                if runner is None:
                    runner = LocalK1(layer)
                if runner.k2 is None:
                    raise MonoUnsupported("merged K2 requires ATOM_DSV41_GFX1250_K2=1")
                runners.append(runner)
            sizes = [_layout(rows, self.tp, self.index_bound_max)[0] for rows in (1, 2)]
            self.scratch = torch.zeros(
                max(sizes), dtype=torch.uint8, device=cache.pool.device
            )
            # Keep the first local wait's evidence across deferred acceptance
            # and step scratch clears. A failed context cannot be reused.
            self.sync_detail = torch.zeros(16, dtype=torch.int32, device=cache.pool.device)
            self.timeline = (
                torch.zeros(40, 320, timeline_ops.POINTS, dtype=torch.int64, device=cache.pool.device)
                if timeline_ops.ENABLED else None
            )
            self.metadata = [
                torch.tensor(
                    [self.rank, layer.attn.spec.layer_id],
                    dtype=torch.int32,
                    device=cache.pool.device,
                )
                for layer in model.layers
            ]
            for rows in (1, 2):
                _, layout = _layout(rows, self.tp, self.index_bound_max)
                views = {
                    name: self.scratch.narrow(0, off, size).view(dtype).view(shape)
                    for name, (off, size, shape, dtype) in layout.items()
                }
                names = merged_scratch_specs(rows, self.tp, self.index_bound_max)
                self.rows[rows] = {
                    "scratch": {n: views[n] for n in names},
                    "outputs": {n: views[n] for n in views if n not in names},
                }
                self.rows[rows]["scratch"]["sync_detail"] = self.sync_detail
            self.runners = runners
            self.rings = {
                layer.attn.spec.layer_id: ring_source(cache, layer.attn.spec.layer_id)
                for layer in model.layers
            }
            self.index_config = {}
            for layer, runner in zip(model.layers, runners):
                spec = layer.attn.spec
                ratio = spec.ratio
                units = cache.index_units[spec.kv_owner].values if ratio else cache.backing[:0]
                self.index_config[spec.layer_id] = torch.tensor([
                    int(runner.feeds), int(spec.candidate_source is not None),
                    int(spec.produces_candidates), cache.geometry.rows_per_page(ratio) if ratio else 1,
                    cache.geometry.page_bytes // cache.geometry.row_bytes,
                    cache.geometry.main_offset(spec.kv_owner) if ratio else 0,
                    units.shape[0], cache.geometry.window(spec.layer_id, cache.num_pages).ring_slots,
                    step.block_tables.numel(), ratio, self.index_bound_max,
                ], dtype=torch.int32, device=cache.pool.device)
            for rows in (1, 2):
                dev = cache.pool.device
                self.rows[rows]["index"] = {
                    "prefix": torch.empty(rows * 512, dtype=torch.int32, device=dev),
                    "pptr": torch.empty(rows + 1, dtype=torch.int32, device=dev),
                    "extend": torch.empty(rows * cache.geometry.window_size, dtype=torch.int32, device=dev),
                    "eptr": torch.empty(rows + 1, dtype=torch.int32, device=dev),
                    "positions": torch.empty(rows, dtype=torch.int64, device=dev),
                    "table": torch.empty_like(step.block_tables),
                    "visible": {ratio: torch.empty(rows, dtype=torch.int32, device=dev) for ratio in (1, 2)},
                    "selection": {layer.attn.spec.layer_id: torch.empty(rows, 512, dtype=torch.int32, device=dev)
                                  for layer, runner in zip(model.layers, runners) if runner.feeds},
                    "candidates": torch.empty(rows, 2048, dtype=torch.int32, device=dev),
                }
            # One K1 scratch/output set per width. K2's residual and pending
            # remain separate; the next K1 reads them before K2 overwrites them.
            self.k1_buffers = {
                rows: runners[0]._buffers(rows, cache.pool.device) for rows in (1, 2)
            }
            for runner in runners:
                runner.buffers = self.k1_buffers

        bind_agreed(bind, group.cpu_group)
        self.peers, self._finalizer = owned_peer_buffer(
            self,
            peer_bytes(2, self.tp),
            group.cpu_group,
            self.rank,
            self.tp,
            cache.pool.device,
        )
        self.mailboxes = StepMailboxes(
            self.peers, self.scratch, debug=True, gfx1250=True
        )
        self.widths = WidthBuilds(
            self._rows_build, self._row_kernels, group.cpu_group, "gfx1250 K1/K2"
        )
        for layer, runner in zip(model.layers, self.runners):
            runner.k2.context = self
            layer.__dict__["_gfx1250_k1"] = runner

    def begin_step(self, rows, step):
        if rows not in self.rows:
            raise ValueError("[gfx1250 K2] runtime supports rows1/2")
        if rows not in self._prepared:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("[gfx1250 K2] prepare the row width before capture")
            if not self.widths.prepare(rows):
                raise MonoUnsupported("gfx1250 K2 row width refused on a TP rank")
            # As in the original runner, both ranks agree before anything that
            # can wait on the other. HIP occupancy is queried without dispatch.
            dummy_table = torch.empty(
                len(ARGUMENTS.names), dtype=torch.int64, device=self.cache.pool.device
            )
            bind_agreed(
                lambda: [
                    prepare_attn_pre_kernel(key[1:]) if key[0] == "k1"
                    else prepare_merged_kernel(key[1:], dummy_table)
                    for key in self.widths[rows]
                ],
                get_tp_group().cpu_group,
            )
            # Compilation/residency alone does not prepare each layer's
            # real addresses or HIP eager graph. Complete both and the guards
            # collectively before begin_step can submit a peer-dependent wait.
            bind_agreed(lambda: self._prepare_bindings(rows), get_tp_group().cpu_group)
            self._prepared.add(rows)
        self.mailboxes.begin_step()
        self.k1_buffers[rows]["state"].zero_()
        inputs = self.rows[rows]["index"]
        inputs["positions"].copy_(step.positions.reshape(rows))
        inputs["table"].copy_(step.block_tables)
        for ratio, target in inputs["visible"].items():
            if ratio in step.visible:
                target.copy_(step.visible[ratio].reshape(rows))
            else:
                target.zero_()

    def _prepare_bindings(self, rows):
        from atom.models.deepseek_v41.mono.gfx1250.gfx1250_runner import _diagnostics

        b = self.k1_buffers[rows]
        o = self.rows[rows]["outputs"]
        codes, _ = _diagnostics(self.cache, rows)
        for block, runner in zip(self.model.layers, self.runners):
            fold = block.attn.spec.layer_id > 0 and block.engram is None
            rope = self.model.global_rope if block.attn.spec.ratio else self.model.window_rope
            if fold:
                inputs = (o["residual"], o["ffn"], o["pre"], o["post"], o["comb"])
            else:
                # Initial embeddings and Engram's settled output are staged
                # into this same input set by LocalK1.forward on every step.
                inputs = (b["res_input"], b["dummy_y"], b["pre_input"],
                          b["dummy_post"], b["dummy_comb"])
            binding = runner.prepare_binding(rows, fold, inputs, self.cache, rope)
            residual, pre, post, comb, query, _ = binding.outputs
            runner.guard(rows, self.cache, codes, residual, pre, post, comb, prepare=True)
            runner.k2.prepare_binding(query, residual, pre, post, comb, self.cache, rope, codes)

    def _rows_build(self, rows):
        keys = {}
        for block in self.model.layers:
            attn = block.attn
            rope = self.model.global_rope if attn.spec.ratio else self.model.window_rope
            ring = ring_source(self.cache, attn.spec.layer_id)
            key = (
                rows,
                self.tp,
                rope.cos_cache.shape[0],
                self.cache.pool.shape[0],
                ring.shape[0],
                rows * self.cache.geometry.batch_topk(attn.spec.ratio),
                rows * self.cache.geometry.window_size,
                float(attn.softmax_scale),
                block._gfx1250_k1.k2.interleaved,
                self.index_bound_max,
            )
            keys[("k2", *key)] = _merged_launcher(*key)
            runner = block._gfx1250_k1
            for fold in (False, True):
                k1_key = (
                    rows, attn.heads, fold, runner.feeds, ring.shape[0], 0,
                    True, True,
                )
                keys[("k1", *k1_key)] = k1_launcher(*k1_key)
        return keys

    @staticmethod
    def _row_kernels(build):
        return [
            (launcher, _PointerAbi(K1_ARG_COUNT if key[0] == "k1" else 1))
            for key, launcher in build.items()
        ]


class _PointerAbi:
    """WidthBuilds' compile-only argument values for the compact pointer ABI."""

    def __init__(self, count=1):
        self.count = count

    def zeros(self):
        import flydsl.compiler as flyc
        import flydsl.expr as fx

        values = [flyc.from_c_void_p(fx.Uint8, 0)] * self.count
        return values + ([fx.Stream(None)] if self.count > 1 else [])


def ring_source(cache, layer):
    """One layer's legal slot span, independent of its offset in the pool."""
    window = cache.geometry.window(layer, cache.num_pages)
    extent = (cache.num_slots - 1) * window.slot_rows + window.ring_slots
    source = cache.ring_view(layer)
    if extent <= 0 or extent > source.shape[0]:
        raise ValueError("[gfx1250 K2] ring slot span is outside the pool")
    return source[:extent]


def bind_gfx1250_context(model, metadata):
    """Bind once during eager warmup, before any merged layer may dispatch."""
    import os

    if os.getenv("ATOM_DSV41_GFX1250_K2_MERGED", "0") != "1":
        return
    step = metadata.step
    if not step.requests or not step.decode or step.width not in (1, 2):
        return
    context = getattr(model, "_gfx1250_decode", None)
    if context is None:
        context = Gfx1250DecodeContext(model, metadata.cache, step)
        model._gfx1250_decode = context
        metadata.cache._gfx1250_decode = context
    elif context.cache is not metadata.cache:
        raise RuntimeError("[gfx1250 K2] model peer context cannot change live caches")
