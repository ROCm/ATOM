# SPDX-License-Identifier: MIT
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


@triton.jit
def _compress_kernel(
    input_ids,
    lookup,
    dead_mask,
    out,
    tokens,
    HAS_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Raw ids to compressed ids, DEAD where the mask says so.

    Once per forward rather than once per lookback per layer: both kernels
    below walk the same ids, and neither needs the vocabulary table to do it.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = offsets < tokens
    raw = tl.load(input_ids + offsets, mask=live, other=0)
    value = tl.load(lookup + raw, mask=live, other=0)
    if HAS_MASK:
        dead = tl.load(dead_mask + offsets, mask=live, other=0)
        value = tl.where(dead != 0, -1, value)
    tl.store(out + offsets, value, mask=live)


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
    SNAPSHOT: tl.constexpr = False,
):
    """One program per token; `NGRAM` lookbacks, `NHEADS` row ids per lookback.

    The lookback walks `[history | this request's compressed ids]` by address
    rather than materializing the reference's `combined`.

    `blocked` latches, and the order is the reference's -- latch, then
    substitute -- so the lookback that finds DEAD is itself `PAD`.
    """
    token = tl.program_id(0)
    if SNAPSHOT:
        padded = tl.load(compressed + token * NGRAM) == -2
    else:
        batch = tl.load(batch_ids + token)
        local = token - tl.load(cu_seqlens + batch)
        row = tl.load(history_index + batch)
        padded = False

    rolling = tl.zeros((), dtype=tl.int64)
    blocked = tl.zeros((), dtype=tl.int1)
    # This lookback's own heads: a block masked on the layer's total would
    # spill into the next lookback's columns, against this one's accumulator.
    heads = tl.arange(0, BLOCK_H)
    live = heads < NHEADS

    for shift in tl.static_range(NGRAM):
        if SNAPSHOT:
            source = tl.load(compressed + token * NGRAM + shift)
        else:
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
        blocked = blocked | (source == -1)
        value = tl.where(blocked, PAD, source).to(tl.int64)
        rolling = rolling ^ (value * tl.load(multipliers + shift))
        if shift:
            column = (shift - 1) * NHEADS + heads
            size = tl.load(head_sizes + column, mask=live, other=1)
            offset = tl.load(head_offsets + column, mask=live, other=0)
            result = tl.where(padded, -1, rolling % size + offset)
            tl.store(out + token * out_stride + column, result, live)


@triton.jit
def _engram_snapshot_kernel(
    compressed,
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
    NGRAM: tl.constexpr,
    BLOCK: tl.constexpr,
    STAGE_CURSOR: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = token < tokens
    batch = tl.load(batch_ids + token, live, 0)
    local = token - tl.load(cu_seqlens + batch, live, 0)
    row = tl.load(history_index + batch, live, 0)
    for shift in tl.static_range(NGRAM):
        back = local - shift
        source = tl.where(
            back >= 0,
            tl.load(compressed + token - shift, live & (back >= 0), -1),
            tl.load(
                history + row * history_stride + NGRAM - 1 + back,
                live & (back < 0),
                -1,
            ),
        )
        # -1 is a real DEAD token; -2 marks padding with no table read.
        tl.store(
            out + token * NGRAM + shift,
            tl.where(live, source, -2),
            token < padded_tokens,
        )
        if STAGE_CURSOR and shift < NGRAM - 1:
            # Snapshot is newest-first; a cursor stores oldest-first history
            # AFTER this token. Both use these same committed lookbacks.
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


def engram_snapshot(tables, batch, out, *, cursor_positions=None, cursor_out=None):
    """Freeze lookbacks before the runner advances the committed cursor.

    The output has a stable address across graph buckets and replays. Only
    this small input copy runs in prepare; per-layer hashing runs in forward.
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
    if out.shape[0]:
        _engram_snapshot_kernel[(triton.cdiv(out.shape[0], 128),)](
            batch.compressed,
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
            NGRAM=tables.ngram,
            BLOCK=128,
            STAGE_CURSOR=stage_cursor,
        )
    return out


def engram_snapshot_indices(tables, layer_id, snapshot, out):
    """Hash frozen lookbacks; padding produces -1 so UVA gathers zeros."""
    tokens = snapshot.shape[0]
    if snapshot.shape[1:] != (tables.ngram,) or not snapshot.is_contiguous():
        raise ValueError("Invalid Engram snapshot layout")
    if out.shape != (tokens, tables.heads) or out.stride(-1) != 1:
        raise ValueError("Invalid Engram row buffer")
    if tokens:
        multipliers, sizes, offsets = tables.layers[layer_id]
        _engram_hash_kernel[(tokens,)](
            snapshot,
            None,
            None,
            None,
            None,
            multipliers,
            sizes,
            offsets,
            out,
            0,
            out.stride(0),
            PAD=tables.pad_id,
            NGRAM=tables.ngram,
            NHEADS=tables.heads // tables.width,
            BLOCK_H=triton.next_power_of_2(tables.heads // tables.width),
            SNAPSHOT=True,
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
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    ALL: tl.constexpr,
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
    batch = tl.program_id(0)
    start = tl.load(cu_seqlens + batch)
    length = tl.load(cu_seqlens + batch + 1) - start
    index = tl.program_id(1) if ALL else length - 1
    if index >= length:
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

    `layers` is pre-sliced per layer so a forward does no tensor arithmetic to
    find its own row.
    """

    lookup: torch.Tensor
    layers: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    pad_id: int
    ngram: int
    heads: int

    @property
    def width(self):
        return self.ngram - 1

    @classmethod
    def from_mapping(cls, mapping, device):
        table = np.asarray(mapping.tokenizer.lookup_table, dtype=np.int64)
        if table.min() < 0:
            raise ValueError("Compressed ids must be non-negative")
        if table.max() > np.iinfo(np.int32).max:
            raise ValueError("Compressed vocabulary does not fit an int32 table")

        def resident(values):
            return torch.as_tensor(np.asarray(values, dtype=np.int64)).to(device)

        return cls(
            lookup=torch.as_tensor(table, dtype=torch.int32).to(device),
            layers={
                layer: (
                    resident(mapping.layer_multipliers[layer]),
                    resident(mapping.head_vocab_sizes[layer]),
                    resident(mapping.head_offsets[layer]),
                )
                for layer in mapping.config.layer_ids
            },
            pad_id=int(mapping.pad_id),
            ngram=int(mapping.config.max_ngram_size),
            heads=int(mapping.config.num_hash_heads),
        )


def engram_compress(tables, input_ids, dead_mask=None, out=None):
    """`[tokens]` int32 compressed ids, `-1` where `dead_mask` is set.

    `dead_mask` is true where a token carries no id of its own -- an image row,
    which the attention metadata already publishes in exactly this sense.
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
            HAS_MASK=dead_mask is not None,
            BLOCK=block,
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
    tables, compressed, cu_seqlens, positions, history, history_index, out, candidates=0
):
    """Write each request's next cursor row, or every prefix a verify may take.

    `candidates` is the verify width: 0 writes one row per request into `out`
    addressed by `history_index` -- the cursor itself, so `history` is its own
    `[:, 1:]` view -- and a positive width writes `out[request, prefix - 1]`
    for every prefix, which is the plane `commit_tentative` selects from.

    Launched where `advance_cursor` ran, and for its reason: a checkpoint store
    runs ahead of the batch, so the image it takes pairs the ring and cursor of
    the step before this one.
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
        WIDTH=tables.width,
        BLOCK=triton.next_power_of_2(tables.ngram),
        ALL=bool(candidates),
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
