# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications: namespace/dependency extraction for ATOM, 2026-10-13.

"""Shared variant arenas and per-layer step metadata; no model weights or golden imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from atom.models.deepseek_v4.mono.config import (
    BLOCK_TOKENS,
    MoeMode,
    _variant_key,
    as_moe_mode,
    shape_dims,
    validate_shard,
)
from atom.models.deepseek_v4.mono.contract import layout, stage_tasks
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed

__all__ = ["Dsv4Variant", "LayerStep", "V4CacheAdapter", "fp4_pool_shapes"]


def fp4_pool_shapes(cfg, samples: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Codes/scales of the gfx950 paged FP4 index pool, in runs of blocks."""
    k_pb = BLOCK_TOKENS // cfg.compress_ratio
    nb = -(-cfg.n_compressed // k_pb)
    ng = cfg.index_head_dim // 32
    return (samples, nb, ng * k_pb * 16), (samples, nb, ng * k_pb)


@dataclass(frozen=True)
class LayerStep:
    """The caller's real KV/state views and addresses for one layer.

    No synthetic RoPE, contiguous paging assumption, cache allocation or state
    reset is made here. Target and draft must supply distinct owning states.
    ``state`` is mandatory when the layer does not own its rolling buffers.
    """

    positions: torch.Tensor
    kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    dest_rows: torch.Tensor
    indices: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    block_tables: torch.Tensor | None = None
    env_rows: int | None = None
    state: dict | None = None


@dataclass(frozen=True)
class _V4Bucket:
    """Stable metadata and layer views for one exact 1/2/4-row launch."""

    width: int
    ordinal: int
    positions: torch.Tensor
    tokens: torch.Tensor
    slots: torch.Tensor
    tables: torch.Tensor
    metadata: dict
    layers: tuple[LayerStep, ...]


class V4CacheAdapter:
    """Exact-row views of ATOM's production cache and request metadata.

    No request state is allocated or cleared. The engine publishes metadata
    before the eager Mono stack, not through pointers to rotating H2D buffers.
    Wider batches are decomposed into stable 8/4/2/1 buckets; padding rows are never bound
    to a bucket. The first profile accepts only in-place state updates, not
    state forks.
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
    ):
        from atom.model_ops.attentions.pool_layout.v4_pool_fields import (
            FP4_GFX950_PRESHUFFLE,
        )

        self.model, self.builder = model, builder
        self.configs = tuple(configs)
        if len(model.layers) != len(self.configs) or not self.configs:
            raise MonoUnsupported("one configuration per real target layer is required")
        if not 1 <= max_rows <= 16:
            raise MonoUnsupported(
                "the production adapter supports at most 16 live rows"
            )
        if tokens_per_seq <= 0 or max_rows % tokens_per_seq:
            raise MonoUnsupported(
                "the production adapter requires whole fixed-width request spans"
            )
        self.max_rows = max_rows
        self.tokens_per_seq = int(tokens_per_seq)
        self.max_requests = (
            max_rows // self.tokens_per_seq
            if max_requests is None
            else int(max_requests)
        )
        if self.max_requests * self.tokens_per_seq != max_rows:
            raise MonoUnsupported("request and row capacities disagree")
        self.geometry = builder.pool_geometry
        self.sources, layer_bases = [], []
        self.table_cols = builder.model_runner.forward_vars["block_tables"].gpu.shape[1]
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
            if cfg.indexed:
                idx, icmp = attn.indexer, attn.indexer.compressor
                if not idx._indexer_fp4 or idx.indexer_layout != FP4_GFX950_PRESHUFFLE:
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
            for prefix, width in (("", cfg.head_dim), ("i_", cfg.index_head_dim)):
                if prefix and not cfg.indexed:
                    continue
                values, scores = (
                    state[prefix + "kv_state"],
                    state[prefix + "score_state"],
                )
                expected = (
                    cfg.c_rows + self.tokens_per_seq - 1,
                    cfg.c_coff * width,
                )
                if (
                    values.dtype != torch.float32
                    or scores.dtype != torch.float32
                    or tuple(values.shape[1:]) != expected
                    or scores.shape != values.shape
                    or values.stride()[1:] != (expected[1], 1)
                    or scores.stride() != values.stride()
                ):
                    raise MonoUnsupported(
                        f"real {prefix}compressor ring/stride must be {expected}"
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
        self.prepared_widths = tuple(
            request_width * self.tokens_per_seq
            for request_width in (1, 2, 4, 8)
            if request_width <= self.max_requests
            and request_width * self.tokens_per_seq in (1, 2, 4, 8)
        )
        if not self.prepared_widths:
            raise MonoUnsupported(
                "no exact Mono launch width serves this request profile"
            )
        self._buckets = {
            width: tuple(
                self._make_bucket(width, ordinal)
                for ordinal in range(max_rows // 8 if width == 8 else 1)
            )
            for width in self.prepared_widths
        }
        # Keep the legacy names as compatibility aliases for existing probes;
        # the MTP profile starts at its exact S4 verify width.
        s1 = self._buckets[self.prepared_widths[0]][0]
        self.positions, self.tokens, self.slots, self.tables = (
            s1.positions,
            s1.tokens,
            s1.slots,
            s1.tables,
        )
        self.metadata, self.layers = s1.metadata, s1.layers
        self.published_buckets: tuple[tuple[int, _V4Bucket], ...] = ()
        self.published_contract: tuple[
            tuple[int | None, int, int, tuple[int, ...]], ...
        ] = ()
        self.published_rows = 0

    def _make_bucket(self, width: int, ordinal: int) -> _V4Bucket:
        positions = torch.zeros(width, dtype=torch.int32, device="cuda")
        tokens = torch.zeros_like(positions)
        slots = torch.zeros_like(positions)
        tables = torch.zeros((width, self.table_cols), dtype=torch.int32, device="cuda")
        metadata = {}
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
        layers = []
        for cfg, kv, rope, state_base in self._layer_bases:
            dest, indices, _, _ = metadata[cfg.compress_ratio]
            state = dict(state_base, state_slots=slots)
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
            width, ordinal, positions, tokens, slots, tables, metadata, tuple(layers)
        )

    def bucket(self, width: int, ordinal: int = 0) -> _V4Bucket:
        return self._buckets[width][ordinal]

    def bucket_count(self, width: int) -> int:
        """Stable arena lanes needed when this width repeats in one wider step."""
        return len(self._buckets[width])

    def _plan(self, rows: int) -> tuple[tuple[int, _V4Bucket], ...]:
        if not 1 <= rows <= self.max_rows:
            raise MonoUnsupported(
                f"decode rows must be in [1, {self.max_rows}], got {rows}"
            )
        result, offset = [], 0
        counts = (
            (8, rows // 8),
            (4, (rows % 8) // 4),
            (2, (rows % 4) // 2),
            (1, rows % 2),
        )
        for width, count in counts:
            for ordinal in range(count):
                result.append((offset, self.bucket(width, ordinal)))
                offset += width
        if offset != rows:
            raise AssertionError("8/4/2/1 bucket plan did not cover every live row")
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

    def _actual_shape(self, context) -> tuple[int, int]:
        from atom.utils.forward_context import AttnState

        fc, md = context.context, context.attn_metadata
        if (
            fc is None
            or fc.is_prefill
            or context.ubatch_slices is not None
            or md is None
            or md.state != AttnState.DECODE
            or md.max_seqlen_q != self.tokens_per_seq
        ):
            return (0, 0)
        requests = int(fc.scheduled_bs)
        rows = int(fc.scheduled_tokens)
        if (
            not 1 <= requests <= self.max_requests
            or rows != requests * self.tokens_per_seq
            or not 1 <= rows <= self.max_rows
            or fc.running_bs < requests
        ):
            return (0, 0)
        return requests, rows

    def _actual_rows(self, context) -> int:
        return self._actual_shape(context)[1]

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
            or any(int(a) != int(b) for a, b in zip(slots, slots_in))
            or md.block_tables is None
            or md.block_tables.ndim != 2
            or md.block_tables.shape[0] < requests
            or md.block_tables.shape[1] != self.table_cols
        ):
            return False
        q = self.tokens_per_seq
        if any(
            int(positions[start + offset]) != int(positions[start]) + offset
            for start in range(0, rows, q)
            for offset in range(q)
        ):
            return False
        if any(
            t is None
            or t.device != self.positions.device
            or t.dtype != torch.int32
            or t.ndim != 1
            or t.numel() < requests
            or not t.is_contiguous()
            for t in (md.state_slot_in, md.state_slot_out)
        ):
            return False
        if (
            md.block_tables.device != self.tables.device
            or md.block_tables.dtype != torch.int32
        ):
            return False
        tables = var["block_tables"].np[:requests]
        for row, pos_value in enumerate(positions):
            request = row // q
            slot, pos = int(slots[request]), int(pos_value)
            if not (0 <= pos < self.configs[0].max_seq):
                return False
            table = tables[request]
            needed = (pos + BLOCK_TOKENS) // BLOCK_TOKENS
            if needed > len(table) or any(
                not 0 <= int(b) < self.builder.num_blocks for b in table[:needed]
            ):
                return False
            for c, kv_cache, _, state in self._layer_bases:
                if any(
                    not 0 <= slot < state[name].shape[0]
                    for name in ("kv_state", "i_kv_state")
                    if name in state
                ):
                    return False
                params = self.geometry.window_params(c.compress_ratio)
                kv_rows = (
                    kv_cache[0].shape[0]
                    if isinstance(kv_cache, tuple)
                    else kv_cache.shape[0]
                )
                if any(
                    not 0 <= params.index(slot, p) < kv_rows
                    for p in range(max(0, pos + 1 - c.window), pos + 1)
                ):
                    return False
                if needed and (
                    int(max(table[:needed])) * self.geometry.envelope_rows
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
        requests, rows = self._actual_shape(context)
        if rows not in self.prepared_widths or context.context.running_bs != requests:
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

    def c1_continuation_contract(self, context):
        """Compatibility view of the one-row continuation frontier."""
        contract = self.continuation_contract(context)
        return contract[0] if contract is not None and len(contract) == 1 else None

    def write(
        self,
        input_ids,
        positions,
        context,
        *,
        validate_static=True,
        validate_device_values=True,
        supported=None,
    ):
        """Publish stable exact-row buckets before eager forward; no cache writes.

        ``validate_device_values=False`` is used only by an already-agreed batch
        continuation. It avoids synchronizing token/position/slot tensors back
        to the host; the broadcast CPU mirrors and the previous TP-agreed
        frontier are its source of truth.
        """
        if validate_static:
            self.assert_current()
        requests, rows = self._actual_shape(context)
        if (
            not rows
            or input_ids.numel() != positions.numel()
            or not rows <= input_ids.numel() <= self.max_rows
            or any(
                t.device != self.positions.device
                or t.ndim != 1
                or t.dtype not in (torch.int32, torch.int64)
                or not t.is_contiguous()
                for t in (input_ids, positions)
            )
            or not (self.supported(context) if supported is None else supported)
        ):
            raise MonoUnsupported(
                "only a real fixed-width in-place decode profile is supported"
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
            tokens_cpu = [int(v) for v in input_ids[:rows].cpu().tolist()]
            positions_cpu = [int(v) for v in positions[:rows].cpu().tolist()]
            slots_in = [int(v) for v in md.state_slot_in[:requests].cpu().tolist()]
            slots_out = [int(v) for v in md.state_slot_out[:requests].cpu().tolist()]
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
        else:
            tokens_cpu = [None] * rows
            positions_cpu = expected_positions
            slots_out = expected_slots
        contracts = []
        plan = self._plan(rows)
        q = self.tokens_per_seq
        for offset, bucket in plan:
            end = offset + bucket.width
            if offset % q or bucket.width % q:
                raise MonoUnsupported("a Mono bucket split a request's verify span")
            request_start, request_end = offset // q, end // q
            bucket_requests = request_end - request_start
            bucket.tokens.copy_(input_ids[offset:end])
            bucket.positions.copy_(positions[offset:end])
            bucket.slots.view(bucket_requests, q).copy_(
                md.state_slot_out[request_start:request_end, None].expand(-1, q)
            )
            bucket.tables.view(bucket_requests, q, self.table_cols).copy_(
                md.block_tables[request_start:request_end, None, :].expand(-1, q, -1)
            )
            for ratio, (
                dest,
                indices,
                window_ids,
                compressed_ids,
            ) in bucket.metadata.items():
                cfg = next(c for c in self.configs if c.compress_ratio == ratio)
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
                    kpb = BLOCK_TOKENS // ratio
                    block_ids = compressed_ids.div(kpb, rounding_mode="floor").long()[
                        None, :
                    ]
                    blocks = bucket.tables.gather(1, block_ids.expand(bucket.width, -1))
                    ci = blocks * self.geometry.envelope_rows + compressed_ids[
                        None, :
                    ].remainder(kpb)
                    visible = (bucket.positions[:, None] + 1).div(
                        ratio, rounding_mode="floor"
                    )
                    indices[:, cfg.window : cfg.window + cfg.n_index].copy_(
                        torch.where(compressed_ids[None, :] < visible, ci, -1)
                    )
            for row in range(offset, end):
                pos = positions_cpu[row]
                request = row // q
                needed = (pos + BLOCK_TOKENS) // BLOCK_TOKENS
                table = var["block_tables"].np[request]
                contracts.append(
                    (
                        tokens_cpu[row],
                        pos,
                        slots_out[request],
                        tuple(int(b) for b in table[:needed]),
                    )
                )
        self.published_contract = tuple(contracts)
        self.published_buckets = plan
        self.published_rows = rows
        return tuple(bucket.layers for _, bucket in plan)

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
        stride = 4 + self.table_cols
        payload = torch.full(
            (1 + self.max_rows * stride,),
            -1,
            dtype=torch.int64,
            device="cpu",
        )
        payload[0] = self.published_rows
        if len(self.published_contract) != self.published_rows:
            raise MonoUnsupported("publication contract row count is inconsistent")
        for row, (token, position, slot, blocks) in enumerate(self.published_contract):
            if row >= self.max_rows or len(blocks) > self.table_cols:
                raise MonoUnsupported("publication contract exceeds its fixed ABI")
            base = 1 + row * stride
            payload[base : base + 4] = torch.tensor(
                [
                    -1 if token is None else int(token),
                    int(position),
                    int(slot),
                    len(blocks),
                ],
                dtype=torch.int64,
            )
            if blocks:
                payload[base + 4 : base + 4 + len(blocks)] = torch.tensor(
                    blocks, dtype=torch.int64
                )
        return payload

    def clear_publication(self):
        self.published_contract = ()
        self.published_buckets = ()
        self.published_rows = 0


class Dsv4Variant:
    """The compiled kernel, scratch and symmetric buffer shared by all layers of one attention variant.

    Scratch cannot be shared across variants (different mailbox layouts). Within one, fresh epoch
    tags make sharing safe, so ``step`` lives here: one counter per scratch, advanced once per step.
    """

    _cache: ClassVar[dict] = {}
    _adv_cache: ClassVar[dict] = {}

    def __init__(
        self,
        cfg,
        samples: int,
        rank: int = 0,
        npes: int = 1,
        group=None,
        timeline: bool = False,
        moe_mode: MoeMode | str = MoeMode.A8W4,
        poll_timeout_us: int | None = None,
        tokens_per_seq: int = 1,
        launch_shape: str = "one",
        qkv_batch_limit: int | None = None,
        qkv_samples_per_pass: int | None = None,
        qkv_norm_handoff: bool = False,
        finite_observer_layer: int = -1,
    ):
        from atom.models.deepseek_v4.mono.kernels.layer import (
            build_advance_step,
            build_dsv4_kernel,
        )
        from atom.mono.runtime.peer_memory import SymmetricPeerBuffer

        moe_mode = as_moe_mode(moe_mode)
        validate_shard(samples, cfg.heads, rank, npes, cfg.window, cfg.compress_ratio)
        if tokens_per_seq <= 0 or samples % tokens_per_seq:
            raise ValueError(
                f"{samples} samples are not whole runs of {tokens_per_seq} tokens per sequence"
            )
        self.poll_timeout_us = poll_timeout_us
        self.tokens_per_seq = tokens_per_seq
        if launch_shape not in ("one", "two"):
            raise ValueError(f"launch_shape must be one/two, got {launch_shape!r}")
        if launch_shape == "one" and (
            qkv_batch_limit is not None
            or qkv_samples_per_pass is not None
            or qkv_norm_handoff
        ):
            raise ValueError("the legacy ONE launch does not take split-QKV tuning")
        self.launch_shape = launch_shape
        self.qkv_batch_limit = qkv_batch_limit
        self.qkv_samples_per_pass = qkv_samples_per_pass
        self.qkv_norm_handoff = qkv_norm_handoff
        self.finite_observer_layer = finite_observer_layer
        self.key = _variant_key(
            cfg,
            samples,
            npes,
            moe_mode,
            timeline,
            poll_timeout_us,
            tokens_per_seq,
            launch_shape,
            qkv_batch_limit,
            qkv_samples_per_pass,
            qkv_norm_handoff,
            finite_observer_layer,
        )
        dims = shape_dims(cfg)
        dev = torch.device("cuda", torch.cuda.current_device())
        self.scr_layout, self.sym_layout = layout(
            samples,
            cfg.heads,
            npes,
            cfg.window,
            moe_mode,
            qkv_norm_handoff=qkv_norm_handoff,
            **dims,
        )
        phases = ("one",) if launch_shape == "one" else ("attention", "moe")
        refresh_modes = (True, False) if cfg.indexed else (True,)
        self.launch_sets = {}
        for refresh in refresh_modes:
            cache_key = (self.key, refresh)
            built = Dsv4Variant._cache.get(cache_key)
            if built is None:
                built = tuple(
                    build_dsv4_kernel(
                        samples,
                        cfg.heads,
                        npes,
                        cfg.window,
                        scale=cfg.softmax_scale,
                        timeline=timeline,
                        moe_mode=moe_mode,
                        poll_timeout_us=poll_timeout_us,
                        tokens_per_seq=tokens_per_seq,
                        swiglu_limit=cfg.swiglu_limit,
                        hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
                        hc_eps=cfg.hc_eps,
                        bf16_kv_roundtrip=cfg.bf16_kv_roundtrip,
                        window_rows=cfg.cache_rows,
                        index_topk_refresh=refresh,
                        phase=phase,
                        qkv_batch_limit=qkv_batch_limit,
                        qkv_samples_per_pass=qkv_samples_per_pass,
                        qkv_norm_handoff=qkv_norm_handoff,
                        finite_observer_layer=finite_observer_layer,
                        **dims,
                    )
                    for phase in phases
                )
                Dsv4Variant._cache[cache_key] = built
            elif not isinstance(built, tuple):
                built = (built,)
            self.launch_sets[refresh] = built
        self.launches = self.launch_sets[
            cfg.index_topk_refresh if cfg.indexed else True
        ]
        # the step advance clears mailbox pairs; the plain-f32 "xqd" region sits last and is skipped
        self.scr_pairs = self.scr_layout["xqd"] // 8
        self.sym_pairs = self.sym_layout["_bytes"] // 8
        adv_key = (self.scr_pairs, self.sym_pairs)
        if adv_key not in Dsv4Variant._adv_cache:
            Dsv4Variant._adv_cache[adv_key] = build_advance_step(*adv_key)
        self._advance = Dsv4Variant._adv_cache[adv_key]
        self.stages = stage_tasks(
            samples,
            cfg.heads,
            window=cfg.window,
            qkv_norm_handoff=qkv_norm_handoff,
            index_topk_refresh=cfg.index_topk_refresh,
            **dims,
        )
        import torch.distributed as dist

        group = dist.group.WORLD if npes > 1 and group is None else group

        def allocate():
            self.scratch = torch.zeros(
                self.scr_layout["_bytes"], dtype=torch.uint8, device=dev
            )
            self.step = torch.zeros(1, dtype=torch.int32, device=dev)
            # Slot 0 is the bounded-poll timeout. Diagnostic builds add NaN
            # boundary bits without inserting kernels between full layers.
            self.hang = torch.zeros(
                9 if finite_observer_layer >= 0 else 1,
                dtype=torch.int32,
                device=dev,
            )

        bind_agreed(allocate, group)
        self.peer_buffer = SymmetricPeerBuffer(
            self.sym_layout["_bytes"], rank=rank, npes=npes, group=group
        )
        self.sym = self.peer_buffer.local_address
        self.peers = self.peer_buffer.addresses

    def hang_detected(self) -> bool:
        """Whether a launch timed out a poll (needs ``poll_timeout_us``). Synchronizes."""
        return bool(self.hang[0].item())

    def advance_step(self):
        """``step += 1`` and clear a slice of stale mailbox pairs so tags can wrap; graph-capturable."""
        self._advance(
            self.step.data_ptr(),
            self.scratch.data_ptr(),
            self.sym,
            stream=torch.cuda.current_stream(),
        )

    def close(self):
        self.peer_buffer.close()
