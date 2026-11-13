# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications: namespace/dependency extraction for ATOM, 2026-10-13.

"""Production KV/state bindings and exact-row metadata publication for V4 Mono."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from atom.models.deepseek_v4.mono.binding import LayerStep
from atom.models.deepseek_v4.mono.config import (
    BLOCK_TOKENS,
    SUPPORTED_LAUNCH_ROWS,
    SUPPORTED_SAMPLES,
)
from atom.mono.runtime.consensus import MonoUnsupported

__all__ = ["V4CacheAdapter"]

# Whole-step capacity is distinct from a device launch's 1..8 live rows.
# Bound independent metadata/peer lanes while bringing up the low-C DSpark grid.
_MAX_VERIFY_REQUESTS = 16
_MAX_VERIFY_ROWS = _MAX_VERIFY_REQUESTS * max(SUPPORTED_SAMPLES)


def _window_in_bounds(params, slot, start, stop, rows):
    """Exact bounds for a half-open token interval in a strided ring.

    Production window addresses increase within each ring cycle. Split a
    wrapping interval into at most two monotone ranges; a whole cycle needs
    only its endpoints. Keep the literal check for nonmonotone geometries.
    This checks the same live window, not a more permissive allocation bound.
    """
    if start >= stop:
        return True
    if params.run_rows < params.ring_stride:
        return all(0 <= params.index(slot, p) < rows for p in range(start, stop))
    ring = params.ring_slots
    length = stop - start
    first = start % ring
    if length >= ring:
        endpoints = (0, ring - 1)
    elif first + length <= ring:
        endpoints = (first, first + length - 1)
    else:
        endpoints = (first, ring - 1, 0, (stop - 1) % ring)
    return all(0 <= params.index(slot, p) < rows for p in endpoints)


@dataclass(frozen=True)
class _V4Bucket:
    """Stable metadata and layer views for one exact rows/q launch."""

    width: int
    ordinal: int
    positions: torch.Tensor
    tokens: torch.Tensor
    slots: torch.Tensor
    slots_in: torch.Tensor
    tables: torch.Tensor
    metadata: dict
    layers: tuple[LayerStep, ...]
    tokens_per_seq: int = 1
    compressed_lookup: dict = field(default_factory=dict)


class V4CacheAdapter:
    """Exact-row views of ATOM's production cache and request metadata.

    No request state is allocated or cleared. The engine publishes metadata
    before eager execution or graph replay, not through pointers to rotating
    H2D buffers. Wider batches use prepared whole-request buckets, including
    ordinary C4's exact S3 tail; padding rows are never bound to a bucket.
    The first profile accepts only in-place state updates,
    not state forks. Ragged construction prepares q-specific shapes without
    changing deployment admission; its live offsets come from the native
    cu_seqlens_q mirror and never include graph padding requests.
    """

    def __init__(
        self,
        model,
        builder,
        configs,
        *,
        max_rows: int = 16,
        tokens_per_seq: int = 1,
        max_requests: int | None = None,
        ragged: bool = False,
        launch_rows: tuple[int, ...] = SUPPORTED_SAMPLES,
        fused_metadata: bool = False,
        request_widths: tuple[int, ...] = (1, 2, 4, 8),
    ):
        from atom.model_ops.attentions.pool_layout.v4_pool_fields import (
            FP4_GFX950_OPUS,
            FP4_GFX950_PRESHUFFLE,
        )

        self.model, self.builder = model, builder
        self.configs = tuple(configs)
        self.fused_metadata = fused_metadata
        if len(model.layers) != len(self.configs) or not self.configs:
            raise MonoUnsupported("one configuration per real target layer is required")
        max_requests = self._validate_capacity(max_rows, tokens_per_seq, max_requests)
        self.max_rows = max_rows
        self.ragged = ragged
        self.tokens_per_seq = int(tokens_per_seq)
        self.max_requests = max_requests
        if (
            not launch_rows
            or any(
                type(rows) is not int or rows not in SUPPORTED_LAUNCH_ROWS
                for rows in launch_rows
            )
            or len(set(launch_rows)) != len(launch_rows)
        ):
            raise MonoUnsupported("invalid explicit device launch rows")
        self.launch_rows = tuple(launch_rows)
        if (
            not request_widths
            or any(
                type(width) is not int or not 1 <= width <= _MAX_VERIFY_REQUESTS
                for width in request_widths
            )
            or len(set(request_widths)) != len(request_widths)
        ):
            raise MonoUnsupported("invalid prepared request widths")
        self.request_widths = tuple(sorted(request_widths))
        self.geometry = builder.pool_geometry
        self.sources, layer_bases = [], []
        tables = builder.model_runner.forward_vars["block_tables"].gpu
        self.table_cols = tables.shape[1]
        self.device = tables.device
        ropes = {}

        def view(owner, name):
            value = getattr(owner, name, None)
            if not isinstance(value, torch.Tensor) or value.device.type != "cuda":
                raise MonoUnsupported(f"real cache is not bound: {name}")
            signature = (
                value.data_ptr(),
                tuple(value.shape),
                tuple(value.stride()),
                value.dtype,
            )
            self.sources.append((owner, name, value, signature))
            return value.detach()

        for block, cfg in zip(model.layers, self.configs):
            attn, cmp = block.attn, block.attn.compressor
            kv_owner = view(attn, "unified_kv")
            if (
                kv_owner.ndim != 2
                or kv_owner.shape[1] != cfg.head_dim
                or not kv_owner.is_contiguous()
            ):
                raise MonoUnsupported("main unified plane layout")
            k_pb = BLOCK_TOKENS // cfg.compress_ratio
            if self.geometry.layer_class(block.layer_id).block_rows != k_pb:
                raise MonoUnsupported("main compressed cache geometry/format")
            compressed = view(cmp, "kv_cache")
            swa = view(attn, "swa_plane")
            expected_stride = (
                self.geometry.envelope_rows * cfg.head_dim,
                cfg.head_dim,
                1,
            )
            if (
                compressed.data_ptr() != kv_owner.data_ptr()
                or tuple(compressed.shape) != (builder.num_blocks, k_pb, cfg.head_dim)
                or compressed.stride() != expected_stride
                or swa.data_ptr() != kv_owner.data_ptr()
                or attn.swa_window != self.geometry.window_params(cfg.compress_ratio)
            ):
                raise MonoUnsupported("ordinary and Mono main-cache views disagree")
            if cfg.kv_fp8:
                # The ordinary pool is typed as torch float8, while the Mono
                # ABI loads and stores the exact packed bytes.  A dtype view is
                # zero-copy and keeps the inline E8M0 bytes untouched.
                if (
                    kv_owner.element_size() != 1
                    or compressed.dtype != kv_owner.dtype
                    or cmp.quant_mode != "group_fp8"
                ):
                    raise MonoUnsupported("FP8 NoPE/E8M0 plane layout")
                kv_nope = (
                    kv_owner
                    if kv_owner.dtype == torch.uint8
                    else kv_owner.view(torch.uint8)
                )
                if not kv_nope.is_contiguous():
                    raise MonoUnsupported("FP8 byte view must stay contiguous")
                kv_rope = view(attn, "unified_kv_rope")
                compressed_rope = view(cmp, "kv_cache_rope")
                if (
                    kv_rope.dtype != torch.bfloat16
                    or kv_rope.ndim != 2
                    or kv_rope.shape != (kv_owner.shape[0], cfg.rope_dim)
                    or not kv_rope.is_contiguous()
                    or compressed_rope.data_ptr() != kv_rope.data_ptr()
                    or tuple(compressed_rope.shape)
                    != (builder.num_blocks, k_pb, cfg.rope_dim)
                    or compressed_rope.stride()
                    != (self.geometry.envelope_rows * cfg.rope_dim, cfg.rope_dim, 1)
                    or view(attn, "swa_plane_rope").data_ptr() != kv_rope.data_ptr()
                ):
                    raise MonoUnsupported("FP8 RoPE plane layout")
                kv = (kv_nope, kv_rope)
            else:
                if (
                    kv_owner.dtype != torch.bfloat16
                    or compressed.dtype != torch.bfloat16
                    or cmp.quant_mode != "none"
                ):
                    raise MonoUnsupported("BF16 unified plane layout")
                kv = kv_owner
            state = {
                "kv_state": view(cmp, "kv_state"),
                "score_state": view(cmp, "score_state"),
            }
            state["st_kv"] = state["kv_state"].stride(0)
            state["st_sc"] = state["score_state"].stride(0)
            if cfg.indexed:
                idx, icmp = attn.indexer, attn.indexer.compressor
                if (
                    not idx._indexer_fp4
                    or idx.indexer_layout
                    not in (FP4_GFX950_PRESHUFFLE, FP4_GFX950_OPUS)
                    or idx.indexer_layout != cfg.indexer_layout
                ):
                    raise MonoUnsupported("real FP4 index pool layout")
                codes, scales = view(idx, "kv_cache"), view(idx, "kv_scale")
                nb = codes.shape[0]
                if (
                    codes.dtype != torch.uint8
                    or scales.dtype != torch.uint8
                    or nb != builder.num_blocks
                    or not codes.is_contiguous()
                    or not scales.is_contiguous()
                    or codes.numel() != nb * 4096
                    or scales.numel() != nb * 256
                ):
                    raise MonoUnsupported(
                        "FP4 index block must be 4096 code / 256 scale bytes"
                    )
                if (
                    view(icmp, "kv_cache").data_ptr() != codes.data_ptr()
                    or view(icmp, "cache_scale").data_ptr() != scales.data_ptr()
                ):
                    raise MonoUnsupported(
                        "index compressor/reader cache views disagree"
                    )
                if nb >= (1 << 32) // 4096:
                    raise MonoUnsupported(
                        "FP4 index pool exceeds the device resource reach"
                    )
                state.update(
                    i_kv_state=view(icmp, "kv_state"),
                    i_score_state=view(icmp, "score_state"),
                    i_cache=codes,
                    i_cache_s=scales,
                    st_ic=0,
                )
                state["st_i"] = state["i_kv_state"].stride(0)
                state["st_isc"] = state["i_score_state"].stride(0)
            for prefix, width in (("", cfg.head_dim), ("i_", cfg.index_head_dim)):
                if prefix and not cfg.indexed:
                    continue
                values, scores = (
                    state[prefix + "kv_state"],
                    state[prefix + "score_state"],
                )
                expected = (
                    cfg.state_ring_rows(self.tokens_per_seq),
                    cfg.c_coff * width,
                )
                if (
                    values.dtype != torch.float32
                    or scores.dtype != torch.float32
                    or tuple(values.shape[1:]) != expected
                    or scores.shape != values.shape
                    or values.stride()[1:] != (expected[1], 1)
                    or scores.stride()[1:] != (expected[1], 1)
                    or any(
                        not expected[0] * expected[1] <= tensor.stride(0) < (1 << 31)
                        for tensor in (values, scores)
                    )
                ):
                    raise MonoUnsupported(
                        f"real {prefix}compressor ring/stride must be {expected}; "
                        f"values={tuple(values.shape)}/{values.stride()}/{values.dtype}, "
                        f"scores={tuple(scores.shape)}/{scores.stride()}/{scores.dtype}"
                    )
            rope = attn.rotary_emb
            cos, sin = view(rope, "cos_cache"), view(rope, "sin_cache")
            rk = (cos.data_ptr(), sin.data_ptr())
            if rk not in ropes:
                if (
                    cos.shape[-1] != cfg.rope_dim // 2
                    or cos.shape != sin.shape
                    or cos.shape[0] < cfg.max_seq
                ):
                    raise MonoUnsupported("real YaRN table geometry")
                ropes[rk] = tuple(
                    t[: cfg.max_seq].reshape(cfg.max_seq, -1).float().contiguous()
                    for t in (cos, sin)
                )
            layer_bases.append((cfg, kv, ropes[rk], state))
        self._layer_bases = tuple(layer_bases)
        self._prepare_buckets()
        self.published_buckets: tuple[tuple[int, _V4Bucket], ...] = ()
        self.published_contract: tuple[
            tuple[int | None, int, int, tuple[int, ...]], ...
        ] = ()
        self.published_rows = 0
        self.published_layout = ()

    @staticmethod
    def _validate_capacity(max_rows, tokens_per_seq, max_requests):
        if type(max_rows) is not int or not 1 <= max_rows <= _MAX_VERIFY_ROWS:
            raise MonoUnsupported(
                f"the production adapter supports at most {_MAX_VERIFY_ROWS} live rows"
            )
        if (
            type(tokens_per_seq) is not int
            or tokens_per_seq not in SUPPORTED_SAMPLES
            or max_rows % tokens_per_seq
        ):
            raise MonoUnsupported(
                "the production adapter requires whole fixed-width request spans"
            )
        requests = max_rows // tokens_per_seq if max_requests is None else max_requests
        if (
            type(requests) is not int
            or not 1 <= requests <= _MAX_VERIFY_REQUESTS
            or requests * tokens_per_seq != max_rows
        ):
            raise MonoUnsupported(
                "request and row capacities disagree or exceed the 16-request budget"
            )
        return requests

    def _prepare_buckets(self):
        """Allocate stable whole-request lanes from explicit device candidates."""
        launch_rows = getattr(self, "launch_rows", SUPPORTED_SAMPLES)
        request_widths = getattr(self, "request_widths", (1, 2, 4, 8))
        self.prepared_widths = tuple(
            request_width * self.tokens_per_seq
            for request_width in request_widths
            if request_width <= self.max_requests
            and request_width * self.tokens_per_seq in launch_rows
        )
        if not self.prepared_widths:
            raise MonoUnsupported(
                "no exact Mono launch width serves this request profile"
            )
        if self.ragged:
            self.prepared_shapes = tuple(
                (count * q, q)
                for q in range(1, self.tokens_per_seq + 1)
                for count in request_widths
                if count <= self.max_requests and count * q in launch_rows
            )
            self._shape_buckets = {
                (width, q): tuple(
                    self._make_bucket(width, lane, tokens_per_seq=q)
                    for lane in range(self.max_requests // (width // q))
                )
                for width, q in self.prepared_shapes
            }
        else:
            self.prepared_shapes = tuple(
                (width, self.tokens_per_seq) for width in self.prepared_widths
            )
            largest = max(self.prepared_widths)
            self._shape_buckets = {
                (width, self.tokens_per_seq): tuple(
                    self._make_bucket(width, lane)
                    for lane in range(
                        self.max_rows // largest if width == largest else 1
                    )
                )
                for width in self.prepared_widths
            }
        self._buckets = {
            width: self._shape_buckets[(width, self.tokens_per_seq)]
            for width in self.prepared_widths
        }

    def _make_bucket(
        self, width: int, ordinal: int, *, tokens_per_seq=None
    ) -> _V4Bucket:
        positions = torch.zeros(width, dtype=torch.int32, device="cuda")
        tokens = torch.zeros_like(positions)
        slots = torch.zeros_like(positions)
        slots_in = torch.zeros_like(positions)
        tables = torch.zeros((width, self.table_cols), dtype=torch.int32, device="cuda")
        metadata = {}
        compressed_lookup = {}
        for cfg in self.configs:
            if cfg.compress_ratio not in metadata:
                metadata[cfg.compress_ratio] = (
                    torch.zeros((2, width), dtype=torch.int32, device="cuda"),
                    torch.full(
                        (width, cfg.n_keys), -1, dtype=torch.int32, device="cuda"
                    ),
                    torch.arange(cfg.window, dtype=torch.int32, device="cuda"),
                    torch.arange(cfg.n_index, dtype=torch.int32, device="cuda"),
                )
                if cfg.n_index and not cfg.indexed:
                    # Allocation-local constants, not request cache contents.
                    # Keep live block-table gathers and visibility checks in
                    # publication; graph replay may change both every step.
                    ids = metadata[cfg.compress_ratio][3]
                    kpb = BLOCK_TOKENS // cfg.compress_ratio
                    compressed_lookup[cfg.compress_ratio] = (
                        ids.div(kpb, rounding_mode="floor").long()[None, :],
                        ids.remainder(kpb)[None, :],
                    )
        layers = []
        for cfg, kv, rope, state_base in self._layer_bases:
            dest, indices, _, _ = metadata[cfg.compress_ratio]
            state = dict(state_base, state_slots=slots, state_slots_in=slots_in)
            layers.append(
                LayerStep(
                    positions,
                    kv,
                    dest,
                    indices,
                    *rope,
                    tables,
                    self.geometry.envelope_rows,
                    state,
                )
            )
        return _V4Bucket(
            width,
            ordinal,
            positions,
            tokens,
            slots,
            slots_in,
            tables,
            metadata,
            tuple(layers),
            self.tokens_per_seq if tokens_per_seq is None else tokens_per_seq,
            compressed_lookup,
        )

    def _layout(self, context, *, padded_rows):
        from atom.models.deepseek_v4.mono.verify_plan import (
            build_verify_plan,
            build_verify_plan_from_offsets,
        )

        requests, rows = self._actual_shape(context)
        var = self.builder.model_runner.forward_vars
        args = (
            var["positions"].np[:rows],
            var["v4_meta_state_slot_in"].np[:requests],
            var["v4_meta_state_slot_out"].np[:requests],
        )
        if getattr(self, "ragged", False):
            return build_verify_plan_from_offsets(
                var["cu_seqlens_q"].np[: requests + 1],
                *args,
                padded_rows=padded_rows,
                max_seqlen_q=self.tokens_per_seq,
            )
        return build_verify_plan(
            (self.tokens_per_seq,) * requests,
            *args,
            padded_rows=padded_rows,
            max_seqlen_q=self.tokens_per_seq,
        )

    def _ragged_plan(self, layout):
        from atom.models.deepseek_v4.mono.verify_plan import plan_verify_launches

        launches = plan_verify_launches(
            [segment.rows for segment in layout.segments], self.prepared_shapes
        )
        result = []
        for launch in launches:
            buckets = self._shape_buckets[(launch.rows, launch.tokens_per_seq)]
            if launch.lane >= len(buckets):
                raise MonoUnsupported("ragged publication exceeds prepared arena lanes")
            result.append((launch.offset, buckets[launch.lane]))
        return tuple(result)

    def bucket(self, width: int, ordinal: int = 0) -> _V4Bucket:
        return self._buckets[width][ordinal]

    def bucket_count(self, width: int) -> int:
        """Stable arena lanes needed when this width repeats in one wider step."""
        return len(self._buckets[width])

    def shape_buckets(self, rows: int, q: int) -> tuple[_V4Bucket, ...]:
        return self._shape_buckets[(rows, q)]

    def capture_lane_candidates(self):
        """Prepared lanes plus a published same-shape dummy metadata source.

        Capture runs lanes sequentially and restores the published write set
        between them. Repeated lanes borrow the source's values, never storage
        addresses or request ownership. This is not a live publication plan.
        """
        candidates = {}
        for _, source in self.published_buckets:
            key = (source.width, source.tokens_per_seq)
            lanes = self.shape_buckets(*key)
            if not any(source is lane for lane in lanes):
                raise MonoUnsupported("capture source is not a prepared arena lane")
            for lane in lanes:
                candidates.setdefault((*key, lane.ordinal), (source, lane))
        return tuple(candidates.values())

    @staticmethod
    def stage_capture_lane(source, target):
        """Copy dummy metadata values into a same-shape fixed-address lane."""
        if (
            source.width != target.width
            or source.tokens_per_seq != target.tokens_per_seq
            or source.metadata.keys() != target.metadata.keys()
        ):
            raise MonoUnsupported("capture metadata source shape differs from lane")
        copies = [
            (getattr(target, name), getattr(source, name))
            for name in ("tokens", "positions", "slots", "slots_in", "tables")
        ]
        for ratio in source.metadata:
            copies.extend(zip(target.metadata[ratio][:2], source.metadata[ratio][:2]))
        # Resolve every copy before touching the first destination.
        if any(
            dst.shape != src.shape or dst.dtype != src.dtype or dst.device != src.device
            for dst, src in copies
        ):
            raise MonoUnsupported("capture metadata source storage differs from lane")
        for dst, src in copies:
            if dst is not src:
                dst.copy_(src)

    def _plan(self, rows: int) -> tuple[tuple[int, _V4Bucket], ...]:
        if not 1 <= rows <= self.max_rows:
            raise MonoUnsupported(
                f"decode rows must be in [1, {self.max_rows}], got {rows}"
            )
        result, offset = [], 0
        q = self.tokens_per_seq
        if rows % q:
            raise MonoUnsupported("live rows split a request's verify span")
        from atom.models.deepseek_v4.mono.verify_plan import plan_verify_launches

        try:
            launches = plan_verify_launches(
                (q,) * (rows // q), [(width, q) for width in self.prepared_widths]
            )
        except (TypeError, ValueError) as error:
            raise MonoUnsupported(str(error)) from error
        for launch in launches:
            if launch.lane >= self.bucket_count(launch.rows):
                raise MonoUnsupported("verify publication exceeds prepared arena lanes")
            result.append((launch.offset, self.bucket(launch.rows, launch.lane)))
            offset += launch.rows
        if offset != rows:
            raise MonoUnsupported("prepared buckets do not cover whole request spans")
        return tuple(result)

    def assert_current(self):
        if self.builder.pool_geometry is not self.geometry:
            raise MonoUnsupported(
                "KV geometry rebound; retire graphs and reprepare Mono"
            )
        for owner, name, source, signature in self.sources:
            current = getattr(owner, name, None)
            if (
                current is not source
                or (
                    current.data_ptr(),
                    tuple(current.shape),
                    tuple(current.stride()),
                    current.dtype,
                )
                != signature
            ):
                raise MonoUnsupported(f"cache/RoPE owner rebound: {name}")

    def capture_write_views(self) -> tuple[torch.Tensor, ...]:
        """Bounded restore coverage for one already-published capture step.

        Use agreed host contracts, never synchronize metadata tensors here.
        KV saves individual destination rows; compressor state saves touched
        slots; FP4 index saves touched blocks to cover its interleaved layout.
        These views are a conservative write-set cover, not request history
        snapshots. The lifecycle owner must save them before warmup/capture
        and restore on the same stream after all captured work has retired.
        Runner epoch/scratch/peer storage needs separate restoration.
        """
        self.assert_current()
        contracts = self.published_contract
        if not contracts or len(contracts) != self.published_rows:
            raise MonoUnsupported("capture restore requires a live published contract")
        views = {}

        def add(tensor, row):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.ndim < 1
                or not 0 <= row < tensor.shape[0]
            ):
                raise MonoUnsupported("capture write view exceeds bound cache/state")
            view = tensor[row : row + 1]
            key = (view.data_ptr(), view.shape, view.stride(), view.dtype, view.device)
            views.setdefault(key, view)

        for cfg, kv, _, state in self._layer_bases:
            planes = kv if isinstance(kv, tuple) else (kv,)
            params = self.geometry.window_params(cfg.compress_ratio)
            slots, compressed = set(), set()
            for _, position, slot, table in contracts:
                if (
                    type(position) is not int
                    or type(slot) is not int
                    or not 0 <= position < cfg.max_seq
                    or slot < 0
                ):
                    raise MonoUnsupported(
                        "capture write contract has invalid position/slot"
                    )
                slots.add(slot)
                for plane in planes:
                    add(plane, params.index(slot, position))
                ratio = cfg.compress_ratio
                if ratio and (position + 1) % ratio == 0:
                    entry = position // ratio
                    kpb = BLOCK_TOKENS // ratio
                    table_col = entry // kpb
                    if table_col >= len(table):
                        raise MonoUnsupported(
                            "capture write contract lacks block table"
                        )
                    block = table[table_col]
                    if (
                        type(block) is not int
                        or not 0 <= block < self.builder.num_blocks
                    ):
                        raise MonoUnsupported(
                            "capture write contract has invalid block"
                        )
                    compressed.add((block, entry % kpb))
            for block, within in sorted(compressed):
                for plane in planes:
                    add(plane, block * self.geometry.envelope_rows + within)
                if cfg.indexed and cfg.index_topk_refresh:
                    for name in ("i_cache", "i_cache_s"):
                        # Native FP4 pool views retain one complete block in
                        # their first dimension (4096 code / 256 scale bytes).
                        add(state[name], block)
            names = ("kv_state", "score_state") if cfg.compress_ratio else ()
            if cfg.indexed and cfg.index_topk_refresh:
                names += ("i_kv_state", "i_score_state")
            for slot in sorted(slots):
                for name in names:
                    add(state[name], slot)
        return tuple(views.values())

    def _actual_shape(self, context) -> tuple[int, int]:
        from atom.utils.forward_context import AttnState

        fc, md = context.context, context.attn_metadata
        if (
            fc is None
            or fc.is_prefill
            or context.ubatch_slices is not None
            or md is None
            or md.state != AttnState.DECODE
            or (
                not 1 <= md.max_seqlen_q <= self.tokens_per_seq
                if getattr(self, "ragged", False)
                else md.max_seqlen_q != self.tokens_per_seq
            )
        ):
            return (0, 0)
        requests = int(fc.scheduled_bs)
        rows = int(fc.scheduled_tokens)
        if (
            not 1 <= requests <= self.max_requests
            or (
                not requests <= rows <= requests * self.tokens_per_seq
                if getattr(self, "ragged", False)
                else rows != requests * self.tokens_per_seq
            )
            or not 1 <= rows <= self.max_rows
            or fc.running_bs < requests
        ):
            return (0, 0)
        return requests, rows

    def supported(self, context) -> bool:
        """Host-only bounds/fork check using the engine's published CPU mirrors."""
        requests, rows = self._actual_shape(context)
        if not rows:
            return False
        md = context.attn_metadata
        var = self.builder.model_runner.forward_vars
        slots = var["v4_meta_state_slot_out"].np[:requests]
        slots_in = var["v4_meta_state_slot_in"].np[:requests]
        positions = var["positions"].np[:rows]
        if (
            len(slots) != requests
            or len(slots_in) != requests
            or len(positions) != rows
            or md.block_tables is None
            or md.block_tables.ndim != 2
            or md.block_tables.shape[0] < requests
            or md.block_tables.shape[1] != self.table_cols
        ):
            return False
        try:
            plan = self._layout(context, padded_rows=rows)
            if plan.live_rows != rows or plan.max_seqlen_q > md.max_seqlen_q:
                return False
            if getattr(self, "ragged", False):
                self._ragged_plan(plan)
        except (TypeError, ValueError, MonoUnsupported):
            return False
        # Layout support does not authorize forks: ordinary/MTP3 still require
        # the original in-place state contract. DSpark transaction admission is
        # a separate qualification, not a consequence of accepting packed rows.
        if not plan.in_place:
            return False
        if any(
            t is None
            or t.device != self.device
            or t.dtype != torch.int32
            or t.ndim != 1
            or t.numel() < requests
            or not t.is_contiguous()
            for t in (md.state_slot_in, md.state_slot_out)
        ):
            return False
        if (
            md.block_tables.device != self.device
            or md.block_tables.dtype != torch.int32
        ):
            return False
        if getattr(self, "ragged", False):
            cu = getattr(md, "cu_seqlens_q", None)
            if (
                cu is None
                or cu.device != self.device
                or cu.dtype != torch.int32
                or cu.ndim != 1
                or cu.numel() < requests + 1
                or not cu.is_contiguous()
            ):
                return False
        tables = var["block_tables"].np[:requests]
        # Resolve every layer's live capacities on this invocation. Identical
        # address formulas/capacities have identical bounds; no result or
        # geometry survives this call, and a smaller late-layer view stays a
        # distinct check. State capacities are independent of token position.
        state_capacities = []
        windows = {}
        for c, kv_cache, _, state in self._layer_bases:
            state_capacities.extend(
                state[name].shape[0]
                for name in ("kv_state", "i_kv_state")
                if name in state
            )
            params = self.geometry.window_params(c.compress_ratio)
            kv_rows = (
                kv_cache[0].shape[0]
                if isinstance(kv_cache, tuple)
                else kv_cache.shape[0]
            )
            key = (
                params.ring_start,
                params.slot_rows,
                params.ring_slots,
                params.ring_stride,
                params.run_rows,
                c.window,
                c.compress_ratio,
                kv_rows,
            )
            windows.setdefault(key, (c, params, kv_rows))
        if state_capacities:
            state_capacity = min(state_capacities)
            if any(not 0 <= int(slot) < state_capacity for slot in slots):
                return False
        # The layout has already checked contiguous positions per request.
        # Table prefixes are nested; the final row's prefix includes every
        # earlier row's. Sliding windows form one exact interval union.
        # This amortizes q-row checks, without retaining results across calls.
        for request, segment in enumerate(plan.segments):
            slot = int(slots[request])
            first = segment.position
            last = first + segment.rows - 1
            if not (0 <= first <= last < self.configs[0].max_seq):
                return False
            table = tables[request]
            needed = (last + BLOCK_TOKENS) // BLOCK_TOKENS
            if needed > len(table) or any(
                not 0 <= int(b) < self.builder.num_blocks for b in table[:needed]
            ):
                return False
            max_block = int(max(table[:needed])) if needed else 0
            for c, params, kv_rows in windows.values():
                if not _window_in_bounds(
                    params, slot, max(0, first + 1 - c.window), last + 1, kv_rows
                ):
                    return False
                if needed and (
                    max_block * self.geometry.envelope_rows
                    + BLOCK_TOKENS // c.compress_ratio
                    > kv_rows
                ):
                    return False
        return True

    def continuation_contract(self, context):
        """Return an exact-request host frontier without synchronizing the GPU.

        Every worker receives the same ``ScheduledBatch`` broadcast, including
        positions, state slots and versioned block tables. A fully agreed batch
        publication can therefore authorize later ordinary or speculative
        windows from these CPU mirrors. Actual target and draft tokens stay
        device-resident and follow the engine's existing TP-uniform sampling
        contract.

        This deliberately checks only values that can change during a stable
        decode.  Cache/weight owners and the 61 layer ABI are checked by the
        anchor publication and invalidated by their lifecycle hooks.
        """
        if getattr(self, "ragged", False):
            return None  # ragged membership requires a new layout publication
        requests, rows = self._actual_shape(context)
        exact_tail = (
            self.tokens_per_seq == 1
            and requests == 3
            and context.context.running_bs == 4
        )
        if not exact_tail and (
            rows not in self.prepared_widths or context.context.running_bs != requests
        ):
            return None
        var = self.builder.model_runner.forward_vars
        slots = var["v4_meta_state_slot_out"].np[:requests]
        slots_in = var["v4_meta_state_slot_in"].np[:requests]
        positions = var["positions"].np[:rows]
        if (
            len(slots) != requests
            or len(slots_in) != requests
            or len(positions) != rows
            or any(int(a) != int(b) for a, b in zip(slots, slots_in))
        ):
            return None
        q = self.tokens_per_seq
        if any(
            int(positions[start + offset]) != int(positions[start]) + offset
            for start in range(0, rows, q)
            for offset in range(q)
        ):
            return None
        result = []
        tables = var["block_tables"].np[:requests]
        for request, (slot_value, table) in enumerate(zip(slots, tables)):
            slot = int(slot_value)
            start = int(positions[request * q])
            end = int(positions[(request + 1) * q - 1])
            if not (0 <= start <= end < self.configs[0].max_seq):
                return None
            needed = (end + BLOCK_TOKENS) // BLOCK_TOKENS
            if needed > len(table):
                return None
            blocks = tuple(int(block) for block in table[:needed])
            if any(not 0 <= block < self.builder.num_blocks for block in blocks):
                return None
            result.append((start, slot, blocks))
        return tuple(result)

    def write(
        self,
        input_ids,
        positions,
        context,
        *,
        validate_static=True,
        validate_device_values=True,
        supported=None,
        publish_metadata=True,
    ):
        """Publish stable exact-row buckets before eager forward; no cache writes.

        ``validate_device_values=False`` is used only by an already-agreed batch
        continuation. It avoids synchronizing token/position/slot tensors back
        to the host; the broadcast CPU mirrors and the previous TP-agreed
        frontier are its source of truth.
        """
        if validate_static:
            self.assert_current()
        if getattr(self, "ragged", False) and not validate_device_values:
            raise MonoUnsupported(
                "ragged steps require a full agreed layout publication"
            )
        requests, rows = self._actual_shape(context)
        if (
            not rows
            or input_ids.numel() != positions.numel()
            or not rows <= input_ids.numel() <= self.max_rows
            or any(
                t.device != self.device
                or t.ndim != 1
                or t.dtype not in (torch.int32, torch.int64)
                or not t.is_contiguous()
                for t in (input_ids, positions)
            )
            or not (self.supported(context) if supported is None else supported)
        ):
            raise MonoUnsupported(
                "only a real prepared in-place decode layout is supported"
            )
        md = context.attn_metadata
        var = self.builder.model_runner.forward_vars
        expected_positions = [int(v) for v in var["positions"].np[:rows]]
        expected_slots = [int(v) for v in var["v4_meta_state_slot_out"].np[:requests]]
        if validate_device_values:
            # The anchor/transition path is deliberately expensive: validate
            # actual staged device values before raw cache loads, then agree the
            # complete contract across TP ranks. Stable replay does not take
            # these D2H synchronizations every token.
            ragged = getattr(self, "ragged", False)
            packed_mirrors = (
                getattr(self, "fused_metadata", False) and input_ids.is_cuda
            )
            if packed_mirrors:
                from atom.models.deepseek_v4.mono.step_meta import read_decode_mirrors

                tokens_cpu, positions_cpu, slots_in, slots_out, offsets_cpu = (
                    read_decode_mirrors(
                        input_ids,
                        positions,
                        md.state_slot_in,
                        md.state_slot_out,
                        rows,
                        requests,
                        md.cu_seqlens_q if ragged else None,
                    )
                )
            else:
                tokens_cpu = [int(v) for v in input_ids[:rows].cpu().tolist()]
                positions_cpu = [int(v) for v in positions[:rows].cpu().tolist()]
                slots_in = [int(v) for v in md.state_slot_in[:requests].cpu().tolist()]
                slots_out = [
                    int(v) for v in md.state_slot_out[:requests].cpu().tolist()
                ]
            if (
                any(
                    not 0 <= token < self.model.embed.num_embeddings
                    for token in tokens_cpu
                )
                or positions_cpu != expected_positions
                or slots_in != slots_out
                or slots_out != expected_slots
            ):
                raise MonoUnsupported(
                    "actual token/position/state values violate the decode contract"
                )
            if ragged:
                expected_cu = [
                    int(value) for value in var["cu_seqlens_q"].np[: requests + 1]
                ]
                actual_cu = (
                    offsets_cpu
                    if packed_mirrors
                    else md.cu_seqlens_q[: requests + 1].cpu().tolist()
                )
                if actual_cu != expected_cu:
                    raise MonoUnsupported(
                        "actual verify offsets violate the packed layout"
                    )
        else:
            tokens_cpu = [None] * rows
            positions_cpu = expected_positions
            slots_out = expected_slots
        layout = self._layout(context, padded_rows=input_ids.numel())
        plan = (
            self._ragged_plan(layout)
            if getattr(self, "ragged", False)
            else self._plan(rows)
        )
        self._write_buckets(
            input_ids,
            positions,
            md,
            layout,
            plan,
            tokens_cpu,
            publish_metadata=publish_metadata,
        )
        return tuple(bucket.layers for _, bucket in plan)

    def _write_buckets(
        self,
        input_ids,
        positions,
        md,
        layout,
        plan,
        tokens_cpu,
        *,
        publish_metadata=True,
    ):
        """Publish a validated packed layout into stable per-shape metadata.

        Every bucket must start/end on request boundaries and contain one q.
        No cache mutation occurs here. Routing, peer consensus, state leases
        and graph-shape qualification remain the session's responsibility.
        """
        if not publish_metadata and not getattr(self, "fused_metadata", False):
            raise MonoUnsupported(
                "deferred metadata requires the DSpark device publisher"
            )
        by_start = {segment.start: i for i, segment in enumerate(layout.segments)}
        by_stop = {segment.stop: i + 1 for i, segment in enumerate(layout.segments)}
        checked = []
        at = 0
        for offset, bucket in plan:
            end = offset + bucket.width
            q = getattr(bucket, "tokens_per_seq", self.tokens_per_seq)
            if offset != at or offset not in by_start or end not in by_stop:
                raise MonoUnsupported("a Mono bucket split a request's verify span")
            start_req, end_req = by_start[offset], by_stop[end]
            if any(segment.rows != q for segment in layout.segments[start_req:end_req]):
                raise MonoUnsupported("a Mono bucket mixes request verify widths")
            checked.append((offset, end, bucket, q, start_req, end_req))
            at = end
        if at != layout.live_rows:
            raise MonoUnsupported("Mono buckets do not cover the live verify layout")
        contracts = []
        for offset, end, bucket, q, request_start, request_end in checked:
            bucket_requests = request_end - request_start
            if getattr(self, "fused_metadata", False):
                from atom.models.deepseek_v4.mono.step_meta import write_bucket_inputs

                write_bucket_inputs(
                    bucket, input_ids, positions, md, offset, request_start
                )
            else:
                bucket.tokens.copy_(input_ids[offset:end])
                bucket.positions.copy_(positions[offset:end])
                bucket.slots.view(bucket_requests, q).copy_(
                    md.state_slot_out[request_start:request_end, None].expand(-1, q)
                )
                bucket.slots_in.view(bucket_requests, q).copy_(
                    md.state_slot_in[request_start:request_end, None].expand(-1, q)
                )
                bucket.tables.view(bucket_requests, q, self.table_cols).copy_(
                    md.block_tables[request_start:request_end, None, :].expand(
                        -1, q, -1
                    )
                )
            for ratio, (
                dest,
                indices,
                window_ids,
                compressed_ids,
            ) in bucket.metadata.items():
                if not publish_metadata:
                    continue
                cfg = next(c for c in self.configs if c.compress_ratio == ratio)
                if getattr(self, "fused_metadata", False):
                    from atom.models.deepseek_v4.mono.step_meta import (
                        write_bucket_metadata,
                    )

                    write_bucket_metadata(bucket, cfg, self.geometry)
                    continue
                params = self.geometry.window_params(ratio)
                wp = bucket.positions[:, None] + 1 - cfg.window + window_ids[None, :]
                ring_index = wp.remainder(params.ring_slots)
                window_rows = (
                    bucket.slots[:, None] * params.slot_rows
                    + params.ring_start
                    + ring_index.div(params.ring_stride, rounding_mode="floor")
                    * params.run_rows
                    + ring_index.remainder(params.ring_stride)
                )
                indices[:, : cfg.window].copy_(torch.where(wp >= 0, window_rows, -1))
                dest[0].copy_(window_rows[:, -1])
                dest[1].zero_()  # compressed rows are relative to the layer view
                if cfg.n_index and not cfg.indexed:
                    block_ids, within_block = bucket.compressed_lookup[ratio]
                    blocks = bucket.tables.gather(1, block_ids.expand(bucket.width, -1))
                    ci = blocks * self.geometry.envelope_rows + within_block
                    visible = (bucket.positions[:, None] + 1).div(
                        ratio, rounding_mode="floor"
                    )
                    indices[:, cfg.window : cfg.window + cfg.n_index].copy_(
                        torch.where(compressed_ids[None, :] < visible, ci, -1)
                    )
            var = self.builder.model_runner.forward_vars
            for request in range(request_start, request_end):
                segment = layout.segments[request]
                table = var["block_tables"].np[request]
                for row in range(segment.start, segment.stop):
                    pos = segment.position + row - segment.start
                    needed = (pos + BLOCK_TOKENS) // BLOCK_TOKENS
                    contracts.append(
                        (
                            tokens_cpu[row],
                            pos,
                            segment.slot_out,
                            tuple(int(b) for b in table[:needed]),
                        )
                    )
        self.published_contract = tuple(contracts)
        self.published_buckets = plan
        self.published_rows = layout.live_rows
        self.published_layout = tuple(
            (
                segment.start,
                segment.stop,
                segment.slot_in,
                launch_offset,
                bucket.width,
                bucket.ordinal,
            )
            for launch_offset, _, bucket, _, start_req, end_req in checked
            for segment in layout.segments[start_req:end_req]
        )

    def publish_device_metadata(self):
        """Refresh fixed integer destinations inside FULL capture or eager forward.

        Only the DSpark fused publisher uses this boundary. Live input/slot/
        table bounds and TP agreement still precede forward. The graph reads
        current bucket buffers, never capture-time positions or slot values.
        Lane composition invokes this before replaying backbone-only graphs.
        """
        if not getattr(self, "fused_metadata", False):
            raise MonoUnsupported("device metadata requires the DSpark fused publisher")
        from atom.models.deepseek_v4.mono.step_meta import write_bucket_metadata

        for _, bucket in self.published_buckets:
            for ratio in bucket.metadata:
                cfg = next(c for c in self.configs if c.compress_ratio == ratio)
                write_bucket_metadata(bucket, cfg, self.geometry)

    def publication_consensus_tensor(self) -> torch.Tensor:
        """Fixed-size CPU encoding of the complete published host contract.

        Full admission must prove that every TP rank received the same token,
        position, state slot and effective block-table prefix.  An object
        collective did that exactly, but paid Python pickle and Gloo object
        setup on the first live request.  This dense encoding preserves the
        same exact comparison with one ordinary tensor collective.

        ``-1`` is an unambiguous sentinel: admitted tokens, positions, slots
        and block ids are all non-negative.  The explicit block count keeps a
        short table distinct from a longer one with sentinel-valued padding.
        """
        import numpy as np

        stride = 4 + self.table_cols
        layout_base = 1 + self.max_rows * stride
        # This is CPU metadata. Fill one owned array and expose it as a tensor
        # once, avoiding a tiny tensor allocation/copy for every live verify
        # row. The exact wire layout and all per-publication checks stay intact.
        payload = np.full(
            (layout_base + 1 + self.max_rows * 6,),
            -1,
            dtype=np.int64,
        )
        payload[0] = self.published_rows
        if len(self.published_contract) != self.published_rows:
            raise MonoUnsupported("publication contract row count is inconsistent")
        for row, (token, position, slot, blocks) in enumerate(self.published_contract):
            if row >= self.max_rows or len(blocks) > self.table_cols:
                raise MonoUnsupported("publication contract exceeds its fixed ABI")
            base = 1 + row * stride
            payload[base : base + 4] = (
                -1 if token is None else int(token),
                int(position),
                int(slot),
                len(blocks),
            )
            if blocks:
                payload[base + 4 : base + 4 + len(blocks)] = blocks
        # The same row values can describe different request boundaries and
        # mailbox launches. Agree the input state, segmentation and lane too;
        # otherwise equal tokens/positions do not prove equal TP execution.
        layout = self.published_layout
        payload[layout_base] = len(layout)
        if len(layout) > self.max_rows:
            raise MonoUnsupported("publication layout exceeds its fixed ABI")
        at = 0
        for request, values in enumerate(layout):
            start, stop, slot_in, offset, width, lane = values
            if (
                start != at
                or not start < stop <= self.published_rows
                or slot_in < 0
                or not 0
                <= offset
                <= start
                < stop
                <= offset + width
                <= self.published_rows
                or lane < 0
            ):
                raise MonoUnsupported("publication layout is inconsistent")
            base = layout_base + 1 + request * 6
            payload[base : base + 6] = values
            at = stop
        if at != self.published_rows:
            raise MonoUnsupported("publication layout does not cover live rows")
        return torch.from_numpy(payload)

    def graph_layout(self) -> tuple:
        """Captured operation topology, excluding mutable token/slot values."""
        return (
            self.published_rows,
            tuple(
                (start, stop, offset, width, lane)
                for start, stop, _, offset, width, lane in self.published_layout
            ),
            tuple(
                (offset, bucket.width, bucket.tokens_per_seq, bucket.ordinal)
                for offset, bucket in self.published_buckets
            ),
        )

    def clear_publication(self):
        self.published_contract = ()
        self.published_buckets = ()
        self.published_rows = 0
        self.published_layout = ()
