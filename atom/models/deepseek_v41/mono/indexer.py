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

from atom.model_ops.deepseek_v41.index_plane import FP4_MFMA
from atom.models.deepseek_v41.mono.kernels.dims import HIDDEN, Q_RANK
from atom.mono.dsa_indexer.paged import build, kernel
from atom.mono.dsa_indexer.paged import plan as ip
from atom.mono.dsa_indexer.paged import tp_plan as tp
from atom.mono.dsa_indexer.paged.metadata import KMETA, write_step_meta
from atom.mono.plan.execution import BLOCKS
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed
from atom.mono.runtime.lifecycle import owned_peer_buffer
from atom.mono.runtime.mailboxes import StepMailboxes
from atom.mono.runtime.step_begin import FENCE_BYTES
from atom.mono.runtime.widths import WidthBuilds

logger = logging.getLogger("atom")


class MonoIndexer:
    max_rows = build.MAX_ROWS

    def __init__(
        self,
        model,
        max_model_len,
        fp4,
        layout,
        group=None,
        *,
        replicated=True,
        workspace,
    ):
        if not fp4:
            raise MonoUnsupported("standalone V4.1 indexer requires FP4")
        self.workspace = workspace
        self.model, self.fp4 = model, fp4
        self.layout = layout
        self.bound_max = ip.cdiv(max_model_len, ip.CHUNK) * ip.CHUNK
        self.world = (
            torch.distributed.get_world_size(group)
            if fp4
            and replicated
            and group is not None
            and torch.distributed.is_initialized()
            else 1
        )
        self.rank = torch.distributed.get_rank(group) if self.world > 1 else 0
        self.peer_plain = (
            tp.layout_bytes(tp.peer_layout(self.max_rows, self.world)) + FENCE_BYTES
            if self.world > 1
            else 0
        )
        self.local_scratch = None
        self.full_scratch = None
        bind_agreed(self._bind, group)
        if self.world > 1:
            self.peers, self._finalizer = owned_peer_buffer(
                self,
                self.peer_plain - FENCE_BYTES,
                group,
                self.rank,
                self.world,
                self.scratch.device,
                tp.layout_bytes(tp.plain_layout(self.max_rows)),
            )
            self.mailboxes = StepMailboxes(self.peers)
        self.widths = WidthBuilds(
            self._build, lambda k: [(k, build.ABI)], group, "V4.1 Mono indexer"
        )
        self.local_widths = (
            WidthBuilds(
                lambda rows: kernel.build_indexer(build.reindex_build(self._key(rows))),
                lambda k: [(k, build.ABI)],
                group,
                "V4.1 local REINDEX",
            )
            if self.local_scratch is not None
            else None
        )
        self.full_widths = (
            WidthBuilds(
                lambda rows: kernel.build_indexer(self._full_key(rows)),
                lambda k: [(k, build.ABI)],
                group,
                "V4.1 FULL indexer",
            )
            if self.full_scratch is not None
            else None
        )
        self.active = False
        self.engram_staged = None

    def _join_staging(self):
        # The native backbone starts this fork after the outer wrapper's
        # begin_step. Join it immediately before the first all-CU launch.
        if self.engram_staged is not None:
            if getattr(self.engram_staged, "stage", None) is not None:
                self.engram_staged.join()
            self.engram_staged = None

    def _bind(self):
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if "gfx950" not in props.gcnArchName or props.multi_processor_count != BLOCKS:
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
        if self.layout not in ("mfma", "preshuffle"):
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
        s = self.max_rows
        workspace = self.workspace
        self.scratch, self.local_scratch, self.full_scratch = (
            workspace.scratch,
            workspace.local_scratch,
            workspace.full_scratch,
        )
        self.full_ratio = min(
            (
                block.attn.indexer.spec.ratio
                for block in self.model.layers
                if block.attn.indexer is not None
                and block.attn.indexer.spec.candidate_source is None
                and not block.attn.indexer.spec.produces_candidates
            ),
            default=None,
        )
        # All paths share the uncached producer/consumer projection rows.
        self.iq, self.iqs, self.iw = workspace.iq, workspace.iqs, workspace.iw
        self.ilog, self.ibmax, self.selected = (
            workspace.ilog,
            workspace.ibmax,
            workspace.selected,
        )
        self.irow = torch.empty(s, ip.TOPK, device=dev, dtype=torch.int32)
        self.candidates = torch.empty(s, ip.TOPK_BLOCKS, device=dev, dtype=torch.int32)
        self.ring_rel = torch.empty(s, device=dev, dtype=torch.int32)
        self.kmeta = torch.empty(s, KMETA, device=dev, dtype=torch.int32)
        self.ix_meta = torch.empty(s, 4, device=dev, dtype=torch.int32)
        self.walk_plans = {
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

    def _key(self, rows):
        world = tp.kernel_world(self.world, rows, self.bound_max)
        return build.IndexerBuild(
            rows,
            self.bound_max,
            self.fp4,
            self.layout == FP4_MFMA,
            world=world,
            peer_plain=self.peer_plain if world > 1 else 0,
            hidden_size=HIDDEN,
            query_rank=Q_RANK,
        )

    def _build(self, rows):
        return kernel.build_indexer(self._key(rows))

    def _full_key(self, rows):
        return build.full_build(self._key(rows), self.full_ratio)

    def prepare(self, rows):
        return (
            0 < rows <= self.max_rows
            and self.widths.prepare(rows)
            and (self.local_widths is None or self.local_widths.prepare(rows))
            and (self.full_widths is None or self.full_widths.prepare(rows))
        )

    def _launch_context(self, spec, rows):
        if self.local_widths is not None and spec.candidate_source is not None:
            return self.local_widths[rows], self.local_scratch
        if (
            self.full_widths is not None
            and spec.candidate_source is None
            and not spec.produces_candidates
        ):
            return self.full_widths[rows], self.full_scratch
        return self.widths[rows], self.scratch

    def begin_step(self, rows, cache, step):
        self.rows = rows
        key = self._key(rows)
        world = key.world
        layout = build.scratch_layout(key)
        resets = tuple(
            self.scratch[off : off + n].view(torch.int32)
            for name, (off, n) in layout.items()
            if name in ("qdone", "qready", "isf", "irdy", "ix_fenced")
            or "ready" in name
            or (not self.fp4 and name in ("iq", "iw"))
        )
        if self.local_scratch is not None:
            local_layout = build.scratch_layout(build.reindex_build(self._key(rows)))
            resets += tuple(
                self.local_scratch[off : off + n].view(torch.int32)
                for name, (off, n) in local_layout.items()
                if name in ("qdone", "qready", "isf", "irdy") or "ready" in name
            )
        if self.full_scratch is not None:
            full_layout = build.scratch_layout(self._full_key(rows))
            resets += tuple(
                self.full_scratch[off : off + n].view(torch.int32)
                for name, (off, n) in full_layout.items()
                if name in ("qdone", "qready", "isf", "irdy", "ix_fenced")
                or "ready" in name
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
            ix_meta=self.ix_meta[:rows] if world > 1 else None,
            tp_world=world,
            tp_rank=self.rank if world > 1 else 0,
            peer_reset=(
                self.peers.bytes[: tp.layout_bytes(tp.peer_layout(rows, world))].view(
                    torch.int32
                )
                if world > 1
                else None
            ),
            split_reset_buffers=tuple(
                self.scratch[off : off + n].view(torch.int32)
                for name, (off, n) in layout.items()
                if name == "ix_rowhist"
            ),
            peer_nonsplit_bytes=(
                tp.peer_layout(rows, world)["ix_hist"][0] if world > 1 else None
            ),
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
        launcher, scratch = self._launch_context(spec, rows)
        world = self._key(rows).world
        shift = 0
        if reindex:
            candidates = step.candidates[spec.candidate_source]
            bound_candidates = cache.candidate_blocks(
                step, spec.candidate_source, spec.ratio
            )
            table, bound = bound_candidates.table, bound_candidates.context
        else:
            per_page = geometry.index_blocks_per_page(spec.ratio)
            if per_page & (per_page - 1):
                return False
            shift = per_page.bit_length() - 1
            table, bound = step.block_tables, step.visible[spec.ratio]
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
        args = build.ABI.pack(
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
                "pscale": p(units.scales),
                "iplan": (p(self.walk_plans[spec.ratio]) if not reindex else 0),
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
                "scratch": p(scratch),
                "layer": spec.layer_id,
                "peers": self.peers.addresses.data_ptr() if world > 1 else 0,
                "rank": self.rank,
                "ix_meta": p(self.ix_meta),
                "step_state": self.mailboxes._state.data_ptr() if world > 1 else 0,
                "step_fence": int(world > 1 and spec.layer_id == min(self.weights)),
            }
        )
        self._join_staging()
        _run_compiled(launcher, *args, torch.cuda.current_stream())
        indexer._publish(
            selected, self.candidates[:rows] if spec.produces_candidates else None, step
        )
        return True
