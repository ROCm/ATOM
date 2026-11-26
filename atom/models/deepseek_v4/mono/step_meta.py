# SPDX-License-Identifier: Apache-2.0
"""Exact integer publication for V4's strided window and compressed pool.

The caller has already validated live requests, slots and block tables. This
publisher changes neither admission nor cache contents; it replaces only the
elementwise metadata chain, reading current device values on every replay.
"""

import triton
import triton.language as tl


@triton.jit
def _publish_bucket_inputs(
    tokens,
    positions,
    source_slots,
    target_slots,
    tables,
    out_tokens,
    out_positions,
    out_source_slots,
    out_target_slots,
    out_tables,
    OFFSET: tl.constexpr,
    REQUEST_START: tl.constexpr,
    Q: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    TABLE_COL_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // COLS
    col = i % COLS
    live = row < ROWS
    request = REQUEST_START + row // Q
    value = tl.load(
        tables + request * TABLE_STRIDE + col * TABLE_COL_STRIDE, mask=live, other=0
    )
    tl.store(out_tables + i, value, mask=live)
    # One lane per row owns the scalar planes. All pointers are validated by
    # the adapter; these are exact integer copies, not a support decision.
    scalar = live & (col == 0)
    token = tl.load(tokens + OFFSET + row, mask=scalar, other=0)
    pos = tl.load(positions + OFFSET + row, mask=scalar, other=0)
    src = tl.load(source_slots + request, mask=scalar, other=0)
    dst = tl.load(target_slots + request, mask=scalar, other=0)
    tl.store(out_tokens + row, token, mask=scalar)
    tl.store(out_positions + row, pos, mask=scalar)
    tl.store(out_source_slots + row, src, mask=scalar)
    tl.store(out_target_slots + row, dst, mask=scalar)


def write_bucket_inputs(bucket, tokens, positions, metadata, offset, request_start):
    """One launch for the existing five exact integer publication copies.

    Request boundaries, live sizes and input storage are checked upstream.
    No frontier readback, static check or TP consensus is removed here.
    Sources are read on every invocation/replay, including prefix fork slots.
    """
    cols = bucket.tables.shape[1]
    _publish_bucket_inputs[(triton.cdiv(bucket.width * cols, 256),)](
        tokens,
        positions,
        metadata.state_slot_in,
        metadata.state_slot_out,
        metadata.block_tables,
        bucket.tokens,
        bucket.positions,
        bucket.slots_in,
        bucket.slots,
        bucket.tables,
        OFFSET=offset,
        REQUEST_START=request_start,
        Q=bucket.tokens_per_seq,
        ROWS=bucket.width,
        COLS=cols,
        TABLE_STRIDE=metadata.block_tables.stride(0),
        TABLE_COL_STRIDE=metadata.block_tables.stride(1),
        BLOCK=256,
    )


@triton.jit
def _read_decode_mirrors(
    tokens,
    positions,
    slots_in,
    slots_out,
    offsets,
    result,
    ROWS: tl.constexpr,
    REQUESTS: tl.constexpr,
    OFFSETS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    token = tl.load(tokens + i, mask=i < ROWS, other=0).to(tl.int64)
    position = tl.load(positions + i, mask=i < ROWS, other=0).to(tl.int64)
    source = tl.load(slots_in + i, mask=i < REQUESTS, other=0).to(tl.int64)
    target = tl.load(slots_out + i, mask=i < REQUESTS, other=0).to(tl.int64)
    tl.store(result + i, token, mask=i < ROWS)
    tl.store(result + ROWS + i, position, mask=i < ROWS)
    tl.store(result + 2 * ROWS + i, source, mask=i < REQUESTS)
    tl.store(result + 2 * ROWS + REQUESTS + i, target, mask=i < REQUESTS)
    if OFFSETS:
        offset = tl.load(offsets + i, mask=i < REQUESTS + 1, other=0).to(tl.int64)
        tl.store(result + 2 * ROWS + 2 * REQUESTS + i, offset, mask=i < REQUESTS + 1)


def read_decode_mirrors(
    tokens, positions, slots_in, slots_out, rows, requests, offsets=None
):
    """One packed D2H for the original exact device/CPU mirror validation.

    The adapter has already checked contiguous layouts and live capacities.
    This is not a support verdict: every returned value must still be compared
    with the scheduler mirrors before metadata publication or cache writes.
    """
    import torch

    size = 2 * rows + 2 * requests + (requests + 1 if offsets is not None else 0)
    result = torch.empty(size, dtype=torch.int64, device=tokens.device)
    _read_decode_mirrors[(1,)](
        tokens,
        positions,
        slots_in,
        slots_out,
        offsets if offsets is not None else slots_in,
        result,
        ROWS=rows,
        REQUESTS=requests,
        OFFSETS=offsets is not None,
        BLOCK=triton.next_power_of_2(max(rows, requests + 1)),
    )
    values = result.tolist()
    return (
        values[:rows],
        values[rows : 2 * rows],
        values[2 * rows : 2 * rows + requests],
        values[2 * rows + requests : 2 * rows + 2 * requests],
        values[2 * rows + 2 * requests :] if offsets is not None else None,
    )


@triton.jit
def _read_frontier(
    anchors,
    current,
    snapshot,
    segments,
    requests,
    result,
    CURRENT_ROWS: tl.constexpr,
    SNAPSHOT_ROWS: tl.constexpr,
    WRITE_TAIL: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    live = i < N
    request = tl.load(requests + i, mask=live, other=0)
    start = tl.load(segments + request, mask=live, other=0)
    stop = tl.load(segments + request + 1, mask=live, other=0)
    row = tl.load(anchors + i, mask=live, other=-1).to(tl.int64)
    valid = (
        live
        & (row >= tl.maximum(start, stop - WRITE_TAIL))
        & (row < stop)
        & (row >= 0)
        & (row < CURRENT_ROWS)
        & (row < SNAPSHOT_ROWS)
    )
    now = tl.load(current + row, mask=valid, other=-1).to(tl.int64)
    saved = tl.load(snapshot + row, mask=valid, other=-1).to(tl.int64)
    # Status 1 is a row/tail violation; 2 is a changed producer position.
    status = tl.where(valid, tl.where(now == saved, 0, 2), 1)
    tl.store(result + i * 2, status, mask=live)
    tl.store(result + i * 2 + 1, saved, mask=live)


def read_draft_frontier(
    anchors, current, snapshot, segments, requests, tail, *, workspace=None
):
    """One device validation and one packed D2H, without unsafe gathers.

    ``requests`` maps already validated host physical slots to producer
    segments. No cache read or backbone launch is authorized by this helper;
    the session must interpret every status under its TP failure boundary.
    """
    import torch

    n = anchors.numel()
    if workspace is None:
        request_ids = torch.tensor(requests, dtype=torch.int32, device=anchors.device)
        result = torch.empty((n, 2), dtype=torch.int64, device=anchors.device)
    else:
        request_ids, result = workspace
    _read_frontier[(1,)](
        anchors,
        current,
        snapshot,
        segments,
        request_ids,
        result,
        CURRENT_ROWS=current.numel(),
        SNAPSHOT_ROWS=snapshot.numel(),
        WRITE_TAIL=tail,
        N=n,
        BLOCK=triton.next_power_of_2(n),
    )
    return result.tolist()


@triton.jit
def _publish_metadata(
    positions,
    slots,
    tables,
    dest,
    indices,
    TABLE_COLS: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    ROWS: tl.constexpr,
    WINDOW: tl.constexpr,
    N_INDEX: tl.constexpr,
    KEYS_PER_BLOCK: tl.constexpr,
    ENVELOPE_ROWS: tl.constexpr,
    RATIO: tl.constexpr,
    SLOT_ROWS: tl.constexpr,
    RING_START: tl.constexpr,
    RING_SLOTS: tl.constexpr,
    RING_STRIDE: tl.constexpr,
    RUN_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    key = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    pos = tl.load(positions + row).to(tl.int64)
    slot = tl.load(slots + row).to(tl.int64)
    wp = pos + 1 - WINDOW + key
    # PyTorch remainder is nonnegative, including the invisible initial keys.
    ring = ((wp % RING_SLOTS) + RING_SLOTS) % RING_SLOTS
    address = (
        slot * SLOT_ROWS
        + RING_START
        + (ring // RING_STRIDE) * RUN_ROWS
        + ring % RING_STRIDE
    )
    tl.store(
        indices + row * INDEX_STRIDE + key,
        tl.where(wp >= 0, address, -1),
        mask=key < WINDOW,
    )
    if N_INDEX:
        compressed = key - WINDOW
        live = (compressed >= 0) & (compressed < N_INDEX)
        block = tl.load(
            tables + row * TABLE_COLS + compressed // KEYS_PER_BLOCK,
            mask=live,
            other=0,
        ).to(tl.int64)
        ci = block * ENVELOPE_ROWS + compressed % KEYS_PER_BLOCK
        tl.store(
            indices + row * INDEX_STRIDE + key,
            tl.where(compressed < (pos + 1) // RATIO, ci, -1),
            mask=live,
        )
    if tl.program_id(1) == 0:
        write_ring = pos % RING_SLOTS
        write_row = (
            slot * SLOT_ROWS
            + RING_START
            + (write_ring // RING_STRIDE) * RUN_ROWS
            + write_ring % RING_STRIDE
        )
        tl.store(dest + row, write_row)
        tl.store(dest + ROWS + row, 0)


def write_bucket_metadata(bucket, cfg, geometry) -> None:
    """Publish one ratio's existing int32 buffers without temporary tensors.

    Indexed layers retain their native indexer contract: only their window
    segment is published here, just as in the original torch chain.
    """
    from atom.models.deepseek_v4.mono.config import BLOCK_TOKENS

    dest, indices, _, _ = bucket.metadata[cfg.compress_ratio]
    params = geometry.window_params(cfg.compress_ratio)
    n_index = cfg.n_index if not cfg.indexed else 0
    block = 256
    _publish_metadata[(bucket.width, triton.cdiv(cfg.window + n_index, block))](
        bucket.positions,
        bucket.slots,
        bucket.tables,
        dest,
        indices,
        TABLE_COLS=bucket.tables.stride(0),
        INDEX_STRIDE=indices.stride(0),
        ROWS=bucket.width,
        WINDOW=cfg.window,
        N_INDEX=n_index,
        KEYS_PER_BLOCK=BLOCK_TOKENS // max(1, cfg.compress_ratio),
        ENVELOPE_ROWS=geometry.envelope_rows,
        RATIO=cfg.compress_ratio,
        SLOT_ROWS=params.slot_rows,
        RING_START=params.ring_start,
        RING_SLOTS=params.ring_slots,
        RING_STRIDE=params.ring_stride,
        RUN_ROWS=params.run_rows,
        BLOCK=block,
    )
