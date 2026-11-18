# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""V4.1 decode indexer, independent of the layer's attention and FFN path.

The replicated MXFP8 projection, tail RoPE, query quantization and paged
selections share the stages used by the layer Mono runner. Prefill and
unsupported decode widths continue through Indexer.project/score.
"""

import logging

import torch
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from torch import nn

from atom.model_ops.deepseek_v41.index_plane import FP4_MFMA
from atom.models.deepseek_v41.mono import index_plan as ip
from atom.models.deepseek_v41.mono.kernels import indexer as kernel
from atom.models.deepseek_v41.mono.kernels.index_query import QS_BYTES
from atom.models.deepseek_v41.mono.step_meta import KMETA, write_step_meta
from atom.mono.plan.execution import BLOCKS
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed
from atom.mono.runtime.route import LazyRunner
from atom.mono.runtime.uncached import uncached_bytes
from atom.mono.runtime.widths import WidthBuilds
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


class MonoIndexer:
    max_rows = kernel.MAX_ROWS

    def __init__(self, model, max_model_len, fp4, layout, group=None):
        self.model, self.fp4 = model, fp4
        self.layout = layout if fp4 else "fp8"
        self.bound_max = ip.cdiv(max_model_len, ip.CHUNK) * ip.CHUNK
        bind_agreed(self._bind, group)
        self.widths = WidthBuilds(
            self._build, lambda k: [(k, kernel.ABI)], group, "V4.1 Mono indexer"
        )
        self.active = False

    def _bind(self):
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if (
            "gfx950" not in getattr(props, "gcnArchName", "")
            or props.multi_processor_count != BLOCKS
        ):
            raise MonoUnsupported("V4.1 indexer requires 256 gfx950 CUs")
        self.weights = {}
        for block in self.model.layers:
            indexer = block.attn.indexer
            if indexer is None:
                continue
            if (
                indexer.heads,
                indexer.head_dim,
                indexer.topk,
                indexer.block_size,
                indexer.topk_blocks,
            ) != (ip.HEADS, ip.DIM, ip.TOPK, ip.BLOCK_ROWS, ip.TOPK_BLOCKS):
                raise MonoUnsupported("V4.1 indexer geometry")
            expected = (
                (indexer.wq_b.weight, (4096, 1280), torch.float8_e4m3fn),
                (indexer.wq_b.weight_scale, (128, 40), torch.float8_e8m0fnu),
                (indexer.weights_proj.weight, (32, 5120), torch.bfloat16),
            )
            if not getattr(indexer.wq_b.weight, "is_shuffled", False):
                raise MonoUnsupported(
                    "V4.1 indexer requires (16,16) shuffled MXFP8 weights"
                )
            for value, shape, dtype in expected:
                if (
                    value.shape != shape
                    or value.dtype != dtype
                    or not value.is_contiguous()
                ):
                    raise MonoUnsupported(
                        f"V4.1 index weight {value.shape}, {value.dtype}"
                    )
            self.weights[indexer.spec.layer_id] = tuple(v for v, _, _ in expected)
        if not self.weights:
            raise MonoUnsupported("no V4.1 indexers")
        for rope in (self.model.global_rope, self.model.window_rope):
            if any(
                t.dtype != torch.bfloat16 or t.shape[-1] != 32 or not t.is_contiguous()
                for t in (rope.cos_cache, rope.sin_cache)
            ):
                raise MonoUnsupported("V4.1 indexer RoPE table")
        if self.layout not in ("fp8", "mfma", "preshuffle"):
            raise MonoUnsupported(f"V4.1 index layout {self.layout}")
        dev = torch.device("cuda", torch.cuda.current_device())
        if any(t.device != dev for values in self.weights.values() for t in values):
            raise MonoUnsupported("V4.1 index weights must be on this rank's GPU")
        if any(
            t.device != dev
            for rope in (self.model.global_rope, self.model.window_rope)
            for t in (rope.cos_cache, rope.sin_cache)
        ):
            raise MonoUnsupported("V4.1 indexer RoPE must be on this rank's GPU")
        s, width = self.max_rows, self.bound_max
        self.scratch = uncached_bytes(kernel.scratch_bytes(self._key(s)), dev)
        # Projection and scoring exchange these plain rows within one launch
        # across XCDs. As with the tagged scratch, keep them uncached: draining
        # the producer's stores alone does not invalidate a consumer's cache.
        self.iq = (
            uncached_bytes(s * ip.HEADS * ip.DIM // (2 if self.fp4 else 1), dev)
            .view(torch.uint8 if self.fp4 else torch.float8_e4m3fn)
            .view(s, ip.HEADS, ip.DIM // (2 if self.fp4 else 1))
        )
        self.iqs = (
            uncached_bytes(s * (QS_BYTES if self.fp4 else ip.HEADS * 4), dev)
            .view(torch.uint8 if self.fp4 else torch.float32)
            .view(s, QS_BYTES if self.fp4 else ip.HEADS)
        )
        self.iw = (
            uncached_bytes(s * ip.HEADS * 4, dev).view(torch.float32).view(s, ip.HEADS)
        )
        self.ilog = torch.empty(s, width, device=dev, dtype=torch.float32)
        self.ibmax = torch.empty(s, ip.blocks(width), device=dev, dtype=torch.float32)
        self.selected = {
            i: torch.empty(s, ip.TOPK, device=dev, dtype=torch.int32)
            for i in self.weights
        }
        self.irow = torch.empty(s, ip.TOPK, device=dev, dtype=torch.int32)
        self.candidates = torch.empty(s, ip.TOPK_BLOCKS, device=dev, dtype=torch.int32)
        self.ring_rel = torch.empty(s, device=dev, dtype=torch.int32)
        self.kmeta = torch.empty(s, KMETA, device=dev, dtype=torch.int32)
        self.walk_plans = (
            {
                r: torch.empty(BLOCKS, 4, device=dev, dtype=torch.int32)
                for r in sorted(
                    {
                        b.attn.spec.ratio
                        for b in self.model.layers
                        if b.attn.indexer is not None
                        and b.attn.spec.candidate_source is None
                    }
                )
            }
            if self.fp4
            else {}
        )

    def _key(self, rows):
        return kernel.IndexerBuild(
            rows, self.bound_max, self.fp4, self.layout == FP4_MFMA
        )

    def _build(self, rows):
        return kernel.build_indexer(self._key(rows))

    def prepare(self, rows):
        return 0 < rows <= self.max_rows and self.widths.prepare(rows)

    def begin_step(self, rows, cache, step):
        self.rows = rows
        layout = kernel.scratch_layout(self._key(rows))
        resets = tuple(
            self.scratch[off : off + n].view(torch.int32)
            for name, (off, n) in layout.items()
            if name in ("qdone", "qready", "isf", "irdy")
            or "ready" in name
            or (not self.fp4 and name in ("iq", "iw"))
        )
        write_step_meta(
            step,
            cache.geometry,
            cache.geometry.window(0, cache.num_pages),
            self.ring_rel[:rows],
            self.kmeta[:rows],
            self.walk_plans,
            (0, rows),
            resets,
            parallel_plans=True,
        )

    def run(self, indexer, hidden, qr, qr_scale, cache, step, rope):
        """Return False before any launch if this layer's inputs do not fit."""
        if not self.active:
            return False
        rows, spec, geometry = self.rows, indexer.spec, cache.geometry
        if not (
            hidden.shape == (1, rows, 5120)
            and hidden.dtype == torch.bfloat16
            and hidden.is_contiguous()
            and qr.shape == (1, rows, 1280)
            and qr.dtype == torch.float8_e4m3fn
            and qr.is_contiguous()
            and qr_scale.shape == (1, rows, 40)
            and qr_scale.dtype == torch.float8_e8m0fnu
            and qr_scale.is_contiguous()
            and step.batch_ids.dtype == torch.int32
            and step.batch_ids.is_contiguous()
            and step.positions.dtype == torch.int64
            and step.positions.is_contiguous()
            and not cache.packed
        ):
            return False
        units = cache.index_units[spec.kv_owner]
        if geometry.index_fp4 != self.fp4 or (self.fp4 and units.layout != self.layout):
            return False
        reindex = spec.candidate_source is not None
        shift = 0
        if reindex:
            candidates = step.candidates[spec.candidate_source]
            bound_candidates = cache.candidate_blocks(
                step, spec.candidate_source, spec.ratio
            )
            table, bound = bound_candidates.table, bound_candidates.context
        elif self.fp4:
            per_page = geometry.index_blocks_per_page(spec.ratio)
            if per_page & (per_page - 1):
                return False
            shift = per_page.bit_length() - 1
            table, bound = step.block_tables, step.visible[spec.ratio]
            candidates = self.candidates[:rows]
        else:
            table, bound = cache.unit_tiles(step, spec.ratio), step.visible[spec.ratio]
            candidates = self.candidates[:rows]
        positions = cache.rope_positions(step)
        inputs = (
            hidden,
            qr,
            qr_scale,
            table,
            bound,
            step.batch_ids,
            positions,
            units.values,
            units.scales,
            rope.cos_cache,
            rope.sin_cache,
        )
        if (
            any(t.device != self.iq.device for t in inputs)
            or table.ndim != 2
            or table.dtype != torch.int32
            or table.stride(1) != 1
            or table.shape[1] == 0
            or bound.shape != (rows,)
            or bound.dtype != torch.int32
            or not bound.is_contiguous()
            or step.batch_ids.shape != (rows,)
            or positions.shape != (rows,)
            or positions.dtype != torch.int64
            or not positions.is_contiguous()
            or not units.values.is_contiguous()
            or not units.scales.is_contiguous()
        ):
            return False
        p = torch.Tensor.data_ptr
        iq_w, iq_ws, iw_w = self.weights[spec.layer_id]
        selected = self.selected[spec.layer_id][:rows]
        args = kernel.ABI.pack(
            {
                "hidden": p(hidden),
                "qr": p(qr),
                "qr_scale": p(qr_scale),
                "iq_w": p(iq_w),
                "iq_ws": p(iq_ws),
                "iw_w": p(iw_w),
                "pos": p(positions),
                "cos": p(rope.cos_cache),
                "sin": p(rope.sin_cache),
                "iq": p(self.iq),
                "iqs": p(self.iqs),
                "iw": p(self.iw),
                "plane": p(units.values),
                "pscale": p(units.scales) if self.fp4 else 0,
                "iplan": (
                    p(self.walk_plans[spec.ratio]) if self.fp4 and not reindex else 0
                ),
                "itab": p(table),
                "itab_stride": table.stride(0),
                "itab_len": table.shape[1],
                "ibatch": p(step.batch_ids),
                "ishift": shift,
                "ibound": p(bound),
                "ilog": p(self.ilog),
                "lstride": self.ilog.shape[1],
                "isel": p(selected),
                "icand": p(candidates),
                "ilift": int(reindex),
                "iprod": int(spec.produces_candidates),
                "ibmax": p(self.ibmax),
                "bstride": self.ibmax.shape[1],
                "icout": p(self.candidates),
                "irow": p(self.irow),
                "kmeta": p(self.kmeta),
                "table": p(step.block_tables),
                "rows_per_page": geometry.rows_per_page(spec.ratio),
                "page_rows": (
                    geometry.rows_per_page(spec.ratio)
                    if cache.kv_fp8
                    else geometry.page_bytes // geometry.row_bytes
                ),
                "main_off": geometry.main_offset(spec.kv_owner),
                "scratch": p(self.scratch),
                "layer": spec.layer_id,
            }
        )
        _run_compiled(self.widths[rows], *args, torch.cuda.current_stream())
        indexer._publish(
            selected, self.candidates[:rows] if spec.produces_candidates else None, step
        )
        return True


def step_supported(input_ids, inputs_embeds):
    rows = input_ids.numel()
    fwd = get_forward_context()
    step = getattr(fwd.attn_metadata, "step", None)
    return (
        inputs_embeds is None
        and 0 < rows <= kernel.MAX_ROWS
        and fwd.context is not None
        and not fwd.context.is_prefill
        and fwd.ubatch_slices is None
        and step is not None
        and step.decode
        and step.running_tokens == rows
        and getattr(fwd.attn_metadata, "image_mask", None) is None
    )


class MonoIndexerModel(nn.Module):
    """Activate the native layer hook for one supported decode forward."""

    def __init__(self, wrapped, raw, config):
        super().__init__()
        from aiter.dist.parallel_state import get_tp_group

        self.model = wrapped
        self._mono = LazyRunner(
            lambda: MonoIndexer(
                raw,
                config.max_model_len,
                config.index_cache_dtype == "fp4",
                config.index_cache_layout,
                get_tp_group().cpu_group,
            ),
            "V4.1 Mono indexer",
        )
        for block in raw.layers:
            if block.attn.indexer is not None:
                block.attn.indexer._mono = self._mono

    def forward(self, input_ids, positions, inputs_embeds=None):
        # The layer runner owns its standalone launch and buffers. Preparing
        # the native hook here would allocate and clear an unused second set.
        uses_layer = getattr(self.model, "uses_layer_mono", None)
        if uses_layer is not None and uses_layer(input_ids, inputs_embeds):
            return self.model(input_ids, positions)
        active = step_supported(input_ids, inputs_embeds) and self._mono.ready(
            input_ids.numel()
        )
        if active:
            md = get_forward_context().attn_metadata
            self._mono.runner.begin_step(input_ids.numel(), md.cache, md.step)
            self._mono.runner.active = True
        try:
            if inputs_embeds is None:
                return self.model(input_ids, positions)
            return self.model(input_ids, positions, inputs_embeds=inputs_embeds)
        finally:
            if active:
                self._mono.runner.active = False

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
