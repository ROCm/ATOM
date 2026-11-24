# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Engram n-gram row indices and cursor rows, computed where the tokens are.

The table lookup has been on the device since UVA; what stayed on the host was
the integer arithmetic naming the rows, and its two inputs are this forward's
token ids and each request's committed history -- one readback each.

`rolling` never carries a sign bit, which is what lets these kernels use the
hardware's `%` where the reference uses numpy's. It holds by construction and
`tests/model_ops/engram/test_hash_bounds.py` pins it, on the host, where CI runs.
"""

from dataclasses import dataclass

import numpy as np
import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class EngramBatch:
    """One forward as the kernels see it: every field already on device.

    `compressed` is derived from the very tensor the model embeds, so the rows
    Engram looks up and the tokens the model runs cannot disagree. `history` is
    any `[n, max_ngram_size - 1]` int64 plane with `history_index` naming a row
    per request, which is how the committed cursor is read where it lies rather
    than gathered out first.
    """

    compressed: torch.Tensor
    batch_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    history: torch.Tensor
    history_index: torch.Tensor


@triton.jit
def _compressed(input_ids, lookup, dead_mask, at, valid, HAS_MASK: tl.constexpr):
    """Token ``at``'s compressed id, DEAD (-1) where the mask says so; 0 off
    ``valid``."""
    raw = tl.load(input_ids + at, mask=valid, other=0)
    value = tl.load(lookup + raw, mask=valid, other=0)
    if HAS_MASK:
        dead = tl.load(dead_mask + at, mask=valid, other=0)
        value = tl.where(dead != 0, -1, value)
    return value


@triton.jit
def _compress_kernel(
    input_ids,
    lookup,
    dead_mask,
    out,
    tokens,
    live_count,
    HAS_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
    COUNTED: tl.constexpr,
):
    """Raw ids to compressed ids, DEAD where the mask says so.

    Once per forward rather than once per lookback per layer: both kernels
    below walk the same ids, and neither needs the vocabulary table to do it.
    ``COUNTED``: the count is read from ``live_count`` (a replay's, not the
    capture's).
    """
    if COUNTED:
        tokens = tl.load(live_count)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = offsets < tokens
    value = _compressed(input_ids, lookup, dead_mask, offsets, live, HAS_MASK)
    tl.store(out + offsets, value, mask=live)


@triton.jit
def _hash_step(rolling, blocked, source, multiplier, PAD: tl.constexpr):
    """One lookback of the rolling hash. `blocked` latches, and the order is
    the reference's -- latch, then substitute -- so the lookback that finds
    DEAD is itself `PAD`."""
    blocked = blocked | (source == -1)
    value = tl.where(blocked, PAD, source).to(tl.int64)
    return rolling ^ (value * multiplier), blocked


@triton.jit
def _engram_hash_kernel(
    compressed,
    batch_ids,
    cu_seqlens,
    history,
    history_index,
    multipliers,
    head_sizes,
    head_offsets,
    out,
    history_stride,
    out_stride,
    PAD: tl.constexpr,
    NGRAM: tl.constexpr,
    NHEADS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One program per token; `NGRAM` lookbacks, `NHEADS` row ids per lookback.

    The lookback walks `[history | this request's compressed ids]` by address
    rather than materializing the reference's `combined`.
    """
    token = tl.program_id(0)
    batch = tl.load(batch_ids + token)
    local = token - tl.load(cu_seqlens + batch)
    row = tl.load(history_index + batch)

    rolling = tl.zeros((), dtype=tl.int64)
    blocked = tl.zeros((), dtype=tl.int1)
    # This lookback's own heads: a block masked on the layer's total would
    # spill into the next lookback's columns, against this one's accumulator.
    heads = tl.arange(0, BLOCK_H)
    live = heads < NHEADS

    for shift in tl.static_range(NGRAM):
        back = local - shift
        from_chunk = back >= 0
        source = tl.where(
            from_chunk,
            tl.load(compressed + token - shift, mask=from_chunk, other=0),
            tl.load(
                history + row * history_stride + (NGRAM - 1 + back),
                mask=back < 0,
                other=-1,
            ),
        )
        rolling, blocked = _hash_step(
            rolling, blocked, source, tl.load(multipliers + shift), PAD
        )
        if shift:
            column = (shift - 1) * NHEADS + heads
            size = tl.load(head_sizes + column, mask=live, other=1)
            offset = tl.load(head_offsets + column, mask=live, other=0)
            tl.store(out + token * out_stride + column, rolling % size + offset, live)


@triton.jit
def _lookback(
    compressed, input_ids, lookup, dead_mask, history, token, local, row, live,
    shift, history_stride, NGRAM: tl.constexpr, COMPRESS: tl.constexpr,
    HAS_MASK: tl.constexpr,
):  # fmt: skip
    """Lookback ``shift`` of ``token``: this request's compressed id inside the
    step (``COMPRESS``: compressed here from the raw ids, as ``_compress_kernel``
    does), past its start the committed history; -1 off ``live``."""
    back = local - shift
    inside = live & (back >= 0)
    if COMPRESS:
        step_id = tl.where(
            inside,
            _compressed(input_ids, lookup, dead_mask, token - shift, inside, HAS_MASK),
            -1,
        )
    else:
        step_id = tl.load(compressed + token - shift, inside, -1)
    return tl.where(
        back >= 0,
        step_id,
        tl.load(
            history + row * history_stride + NGRAM - 1 + back, live & (back < 0), -1
        ),
    )


@triton.jit
def _engram_snapshot_kernel(
    compressed,
    input_ids,
    lookup,
    dead_mask,
    batch_ids,
    cu_seqlens,
    history,
    history_index,
    out,
    tokens,
    padded_tokens,
    history_stride,
    cursor_positions,
    cursor_out,
    cursor_slot_stride,
    cursor_row_stride,
    live_count,
    multipliers,
    head_sizes,
    head_offsets,
    row_ids,
    multiplier_stride,
    head_stride,
    row_layer_stride,
    row_token_stride,
    NGRAM: tl.constexpr,
    BLOCK: tl.constexpr,
    STAGE_CURSOR: tl.constexpr,
    COUNTED: tl.constexpr,
    COMPRESS: tl.constexpr,
    HAS_MASK: tl.constexpr,
    LAYERS: tl.constexpr,
    PAD: tl.constexpr,
    NHEADS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """A block of tokens (``program_id(0)``) x a layer (``program_id(1)``).
    Layer 0's program: the tokens' frozen lookbacks (``out`` [token][NGRAM],
    newest first; -2 a padding row) and a verify step's cursor candidates,
    ``COMPRESS``: with the step's compressed ids made here from the raw ones
    (each token's also stored to ``compressed``) instead of read. With
    ``LAYERS`` stacked layers, every program also hashes its layer's row ids
    from the lookbacks into ``row_ids`` [layer][token][heads] (-1 a padding
    row), as ``_engram_hash_kernel`` hashes a token: the layers' int64
    modulos on their own programs."""
    if COUNTED:
        tokens = tl.load(live_count)
    token = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    layer = tl.program_id(1)
    live = token < tokens
    batch = tl.load(batch_ids + token, live, 0)
    local = token - tl.load(cu_seqlens + batch, live, 0)
    row = tl.load(history_index + batch, live, 0)
    if layer == 0:
        if COMPRESS:
            own = _compressed(input_ids, lookup, dead_mask, token, live, HAS_MASK)
            tl.store(compressed + token, own, live)
        for shift in tl.static_range(NGRAM):
            source = _lookback(
                compressed, input_ids, lookup, dead_mask, history, token, local,
                row, live, shift, history_stride, NGRAM, COMPRESS, HAS_MASK,
            )  # fmt: skip
            # -1 is a real DEAD token; -2 marks padding with no table read.
            tl.store(
                out + token * NGRAM + shift,
                tl.where(live, source, -2),
                token < padded_tokens,
            )
            if STAGE_CURSOR and shift < NGRAM - 1:
                # Snapshot is newest-first; a cursor stores oldest-first
                # history AFTER this token. Both use these same lookbacks.
                tl.store(
                    cursor_out
                    + batch * cursor_slot_stride
                    + local * cursor_row_stride
                    + NGRAM
                    - 1
                    - shift,
                    source,
                    live,
                )
        if STAGE_CURSOR:
            position = tl.load(cursor_positions + token - local, live, 0)
            tl.store(
                cursor_out + batch * cursor_slot_stride + local * cursor_row_stride,
                position + local + 1,
                live,
            )
    if LAYERS:
        heads = tl.arange(0, BLOCK_H)
        own_heads = heads < NHEADS
        stored = (token < padded_tokens)[:, None] & own_heads[None, :]
        rolling = tl.zeros((BLOCK,), dtype=tl.int64)
        blocked = tl.zeros((BLOCK,), dtype=tl.int1)
        for shift in tl.static_range(NGRAM):
            source = _lookback(
                compressed, input_ids, lookup, dead_mask, history, token, local,
                row, live, shift, history_stride, NGRAM, COMPRESS, HAS_MASK,
            )  # fmt: skip
            rolling, blocked = _hash_step(
                rolling, blocked, source,
                tl.load(multipliers + layer * multiplier_stride + shift), PAD,
            )  # fmt: skip
            if shift:
                column = (shift - 1) * NHEADS + heads
                size = tl.load(head_sizes + layer * head_stride + column, own_heads, 1)
                offset = tl.load(
                    head_offsets + layer * head_stride + column, own_heads, 0
                )
                result = tl.where(
                    live[:, None],
                    rolling[:, None] % size[None, :] + offset[None, :],
                    -1,
                )
                tl.store(
                    row_ids + layer * row_layer_stride
                    + token[:, None] * row_token_stride + column[None, :],
                    result,
                    stored,
                )  # fmt: skip


SNAPSHOT_BLOCK = 128  # tokens a snapshot program
SNAPSHOT_HASH_BLOCK = 16  # tokens a program that also hashes its layer


def engram_snapshot(
    tables,
    batch,
    out,
    *,
    cursor_positions=None,
    cursor_out=None,
    live=None,
    compress=None,
    rows=None,
):
    """Freeze lookbacks before the committed cursor advances.

    `live`: the live leading tokens on the device; without it every token of
    `batch.batch_ids` is. `compress`: `(input_ids, dead_mask)`, the step's
    compressed ids made in the launch into `batch.compressed`
    (`engram_compress`'s) instead of read from it. `rows`: `(layer_ids,
    row_ids)`, every layer's row ids hashed in the launch too, into `row_ids`
    [layers, out rows, heads] (consecutive layers of the stacked tables,
    `EngramHashTables.run`); padding rows -1, so the UVA gather reads zeros.
    """
    _check_plane(batch.history, tables)
    tokens = batch.batch_ids.numel()
    if out.shape[1:] != (tables.ngram,) or out.shape[0] < tokens:
        raise ValueError("Engram snapshot must cover all tokens and lookbacks")
    if not out.is_contiguous():
        raise ValueError("Engram snapshot must be contiguous")
    stage_cursor = cursor_out is not None
    if stage_cursor:
        _check_plane(batch.history, tables, cursor_out)
        if (
            cursor_out.ndim != 3
            or cursor_out.shape[0] < batch.history_index.numel()
            or cursor_out.shape[-1] != tables.ngram
            or cursor_positions is None
            or cursor_positions.numel() < tokens
        ):
            raise ValueError("Engram snapshot requires per-request cursor candidates")
        if (
            cursor_out.untyped_storage().data_ptr()
            == batch.history.untyped_storage().data_ptr()
        ):
            raise ValueError("Engram snapshot cursor candidates must not alias history")
    input_ids, dead_mask = compress if compress is not None else (None, None)
    layers, row_ids, stacked = 0, None, (None, None, None)
    if rows is not None:
        layer_ids, row_ids = rows
        layers = len(layer_ids)
        if (
            row_ids.shape != (layers, out.shape[0], tables.heads)
            or row_ids.stride(-1) != 1
        ):
            raise ValueError("Invalid Engram row buffer")
        stacked = tables.run(layer_ids)
    heads = tables.heads // tables.width
    # tokens a program: fewer where it hashes, its int64 modulos the cost
    block = SNAPSHOT_HASH_BLOCK if layers else SNAPSHOT_BLOCK
    if out.shape[0]:
        _engram_snapshot_kernel[(triton.cdiv(out.shape[0], block), max(layers, 1))](
            batch.compressed,
            input_ids,
            tables.lookup,
            dead_mask,
            batch.batch_ids,
            batch.cu_seqlens,
            batch.history,
            batch.history_index,
            out,
            tokens,
            out.shape[0],
            batch.history.stride(0),
            cursor_positions,
            cursor_out,
            cursor_out.stride(0) if stage_cursor else 0,
            cursor_out.stride(1) if stage_cursor else 0,
            live,
            *stacked,
            row_ids,
            stacked[0].stride(0) if layers else 0,
            stacked[1].stride(0) if layers else 0,
            row_ids.stride(0) if layers else 0,
            row_ids.stride(1) if layers else 0,
            NGRAM=tables.ngram,
            BLOCK=block,
            STAGE_CURSOR=stage_cursor,
            COUNTED=live is not None,
            COMPRESS=compress is not None,
            HAS_MASK=dead_mask is not None,
            LAYERS=layers,
            PAD=tables.pad_id,
            NHEADS=heads,
            BLOCK_H=triton.next_power_of_2(heads),
        )
    return out


@triton.jit
def _engram_cursor_kernel(
    compressed,
    cu_seqlens,
    positions,
    history,
    history_index,
    out,
    history_stride,
    out_slot_stride,
    out_row_stride,
    live_count,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    ALL: tl.constexpr,
    COUNTED: tl.constexpr,
):
    """A request's cursor after accepting `index + 1` of this forward's tokens.

    A row is `[position, history...]`, the history after a prefix of length `L`
    being `[committed | compressed][L : L + WIDTH]` -- what `advance_history`
    takes. No latching here: DEAD travels into the history as itself and only
    `_engram_hash_kernel` substitutes `PAD` for it.

    The whole row is loaded before any of it is stored. Without that this reads
    a row it is overwriting: at `ALL=False` the destination IS the history
    plane offset by one column, so the windows overlap for every `L < WIDTH`.
    """
    if COUNTED:  # noqa: SIM102 (a constexpr, then a device load)
        if tl.load(live_count) == 0:
            return
    batch = tl.program_id(0)
    start = tl.load(cu_seqlens + batch)
    length = tl.load(cu_seqlens + batch + 1) - start
    index = tl.program_id(1) if ALL else length - 1
    # a padded request (length 0) must not write prefix 0 into slot 0
    if (index >= length) | (length == 0):
        return
    prefix = index + 1
    slot = tl.load(history_index + batch)

    # Column 0 is the position; column `c` holds `combined[prefix + c - 1]`.
    column = tl.arange(0, BLOCK)
    live = column < WIDTH + 1
    at = prefix + column - 1
    carried = (column > 0) & (at < WIDTH)
    written = (column > 0) & (at >= WIDTH)
    value = tl.where(
        column == 0,
        tl.load(positions + start).to(tl.int64) + prefix,
        tl.where(
            carried,
            tl.load(history + slot * history_stride + at, live & carried, 0),
            tl.load(compressed + start + at - WIDTH, live & written, 0),
        ).to(tl.int64),
    )
    base = (
        out + batch * out_slot_stride + index * out_row_stride
        if ALL
        else out + slot * out_row_stride
    )
    tl.store(base + column, value, mask=live)


@dataclass(frozen=True)
class EngramHashTables:
    """The constants both kernels read, resident and per layer.

    Under a megabyte: only the compressed-vocab table scales with anything, at
    `vocab_size` int32. The embedding tables are not here and do not move.

    `stacked` holds every layer's constants as rows, in `order`; `layers` is
    each layer's rows of it, pre-sliced so a forward does no tensor arithmetic
    to find its own, and a run of consecutive layers hashes in one launch.
    """

    lookup: torch.Tensor
    stacked: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    order: tuple[int, ...]
    layers: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    pad_id: int
    ngram: int
    heads: int

    @property
    def width(self):
        return self.ngram - 1

    def run(self, layer_ids):
        """The stacked tables from ``layer_ids``' first row, one row a layer:
        ``layer_ids`` must be consecutive layers of ``order``."""
        first = self.order.index(layer_ids[0])
        if tuple(layer_ids) != self.order[first : first + len(layer_ids)]:
            raise ValueError(f"layers {layer_ids} are not a run of {self.order}")
        return tuple(t[first:] for t in self.stacked)

    @classmethod
    def from_mapping(cls, mapping, device):
        table = np.asarray(mapping.tokenizer.lookup_table, dtype=np.int64)
        if table.min() < 0:
            raise ValueError("Compressed ids must be non-negative")
        if table.max() > np.iinfo(np.int32).max:
            raise ValueError("Compressed vocabulary does not fit an int32 table")

        order = tuple(mapping.config.layer_ids)

        def resident(per_layer):
            rows = [np.asarray(per_layer[layer], dtype=np.int64) for layer in order]
            return torch.as_tensor(np.stack(rows)).to(device)

        stacked = (
            resident(mapping.layer_multipliers),
            resident(mapping.head_vocab_sizes),
            resident(mapping.head_offsets),
        )
        return cls(
            lookup=torch.as_tensor(table, dtype=torch.int32).to(device),
            stacked=stacked,
            order=order,
            layers={
                layer: tuple(t[i] for t in stacked) for i, layer in enumerate(order)
            },
            pad_id=int(mapping.pad_id),
            ngram=int(mapping.config.max_ngram_size),
            heads=int(mapping.config.num_hash_heads),
        )


def engram_compress(tables, input_ids, dead_mask=None, out=None, live=None):
    """`[tokens]` int32 compressed ids, `-1` where `dead_mask` is set.

    `dead_mask` is true where a token carries no id of its own -- an image row,
    which the attention metadata already publishes in exactly this sense.
    `live`: the live leading tokens on the device, the rest untouched.
    """
    tokens = input_ids.numel()
    if out is None:
        out = torch.empty(tokens, dtype=torch.int32, device=input_ids.device)
    if tokens:
        block = 1024
        _compress_kernel[(triton.cdiv(tokens, block),)](
            input_ids,
            tables.lookup,
            dead_mask,
            out,
            tokens,
            live,
            HAS_MASK=dead_mask is not None,
            BLOCK=block,
            COUNTED=live is not None,
        )
    return out


def _check_plane(history, tables, *planes):
    """Both kernels walk a row, and neither carries a column stride."""
    if history.shape[-1] != tables.width:
        raise ValueError(f"history is {history.shape[-1]} wide, want {tables.width}")
    for plane in (history, *planes):
        if plane.stride(-1) != 1:
            raise ValueError("cursor and history rows must be contiguous")


def engram_row_indices(
    tables, layer_id, compressed, batch_ids, cu_seqlens, history, history_index, out
):
    """`[tokens, num_hash_heads]` int64 absolute rows of `layer_id`'s table.

    `history` is any `[n, width]` int64 plane and `history_index` names a row
    of it per request, so the committed cursor is read where it lies -- a
    strided field of the STATE arena -- rather than copied out to be indexed.

    `out` is reused across layers: a layer's rows are consumed by its own
    gather before the next layer is hashed.
    """
    tokens = batch_ids.numel()
    if not tokens:
        return out
    _check_plane(history, tables)
    if out.shape != (tokens, tables.heads):
        raise ValueError(f"row buffer is {tuple(out.shape)}, want {tokens} rows")
    multipliers, sizes, offsets = tables.layers[layer_id]
    _engram_hash_kernel[(tokens,)](
        compressed,
        batch_ids,
        cu_seqlens,
        history,
        history_index,
        multipliers,
        sizes,
        offsets,
        out,
        history.stride(0),
        out.stride(0),
        PAD=tables.pad_id,
        NGRAM=tables.ngram,
        NHEADS=tables.heads // tables.width,
        BLOCK_H=triton.next_power_of_2(tables.heads // tables.width),
    )
    return out


def engram_cursor_rows(
    tables,
    compressed,
    cu_seqlens,
    positions,
    history,
    history_index,
    out,
    candidates=0,
    live=None,
):
    """Write each request's next cursor row, or every prefix a verify may take.

    `candidates` is the verify width: 0 writes one row per request into `out`
    addressed by `history_index` -- the cursor itself, so `history` is its own
    `[:, 1:]` view -- and a positive width writes `out[request, prefix - 1]`
    for every prefix, which is the plane `commit_tentative` selects from.

    After the checkpoint store, whose image pairs the ring and cursor of the
    step before. `live`: the forward's live tokens on the device; 0 writes
    nothing.
    """
    requests = history_index.numel()
    if not requests:
        return out
    _check_plane(history, tables, out)
    if out.shape[-1] != tables.ngram:
        raise ValueError(f"cursor rows are {out.shape[-1]} wide, want {tables.ngram}")
    _engram_cursor_kernel[(requests, max(candidates, 1))](
        compressed,
        cu_seqlens,
        positions,
        history,
        history_index,
        out,
        history.stride(0),
        out.stride(0) if candidates else 0,
        out.stride(1) if candidates else out.stride(0),
        live,
        WIDTH=tables.width,
        BLOCK=triton.next_power_of_2(tables.ngram),
        ALL=bool(candidates),
        COUNTED=live is not None,
    )
    return out


def engram_row_indices_reference(mapping, layer_id, tokens, histories, masks=None):
    """The published chain, one request at a time, rows concatenated.

    `EngramPrefetcher.row_indices` with the request objects taken out: the same
    `compress_tokens` -> `hash_layer` -> `to_row_indices` over the same
    per-request history, which is what the kernels have to match to the bit.
    """
    out = []
    for index, ids in enumerate(tokens):
        mask = None if masks is None else np.asarray([masks[index]])
        compressed = mapping.compress_tokens(np.asarray([ids], dtype=np.int64), mask)
        hashes = mapping.hash_layer(
            compressed,
            layer_id,
            compress=False,
            history=np.asarray([histories[index]], dtype=np.int64),
        )
        out.append(mapping.to_row_indices(hashes, layer_id)[0])
    return np.concatenate(out) if out else np.empty((0, 0), dtype=np.int64)


def engram_cursor_rows_reference(mapping, tokens, histories, positions, masks=None):
    """`[prefix, 1 + width]` per request: `advance_history` at every prefix."""
    width = mapping.config.max_ngram_size - 1
    out = []
    for index, ids in enumerate(tokens):
        mask = None if masks is None else np.asarray([masks[index]])
        compressed = mapping.compress_tokens(np.asarray([ids], dtype=np.int64), mask)
        combined = np.concatenate((np.asarray(histories[index]), compressed[0]))
        out.append(
            np.stack(
                [
                    np.concatenate(
                        ([positions[index] + prefix], combined[prefix : prefix + width])
                    )
                    for prefix in range(1, len(ids) + 1)
                ]
            )
        )
    return out
