# SPDX-License-Identifier: MIT
"""The UVA gather: this rank's hash heads, read from host memory by the device.

The page-locking, and the host gather this path replaces, stay in `..tables`:
that module is what the prefetch runtime imports, and it has to keep importing
on a machine with no Triton.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _uva_lookup_kernel(
    weight,
    scales,
    ids,
    out,
    num_rows,
    vocab_start,
    vocab_end,
    ids_stride_t,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    """Gather this rank's hash heads from a host table over UVA, dequantize, store.

    `weight`/`scales` address page-locked HOST memory holding only this rank's
    shard, so a row is addressed by `index - vocab_start`. A head this rank does
    not own writes zeros, which is what makes the all-gather that follows a plain
    concatenation. The ue8m0 scale byte IS an fp32 exponent field, so its decode
    is a shift.

    The grid is persistent and sized to the device rather than to the batch: the
    table dwarfs any TLB, so the win is in reusing warmed translations, not in
    one program per row.
    """
    cols = tl.arange(0, DIM)
    for base in tl.range(
        tl.program_id(0) * BLOCK_R, num_rows, tl.num_programs(0) * BLOCK_R
    ):
        rows = base + tl.arange(0, BLOCK_R)
        valid = rows < num_rows
        head = HEAD_START + rows % LOCAL_HEADS
        token = (rows // LOCAL_HEADS).to(tl.int64)
        index = tl.load(
            ids + token * ids_stride_t + head,
            mask=valid & (head < TOTAL_HEADS),
            other=-1,
        ).to(tl.int64)
        owned = valid & (head < TOTAL_HEADS)
        owned &= (index >= vocab_start) & (index < vocab_end)
        local = tl.where(owned, index - vocab_start, 0)
        values = tl.load(
            weight + local[:, None] * DIM + cols[None, :],
            mask=owned[:, None],
            other=0.0,
        ).to(tl.float32)
        if HAS_SCALE:
            scale = tl.load(
                scales
                + local[:, None] * (DIM // QUANT_BLOCK)
                + (cols // QUANT_BLOCK)[None, :],
                mask=owned[:, None],
                other=0,
            )
            values = values * (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        tl.store(out + rows[:, None] * DIM + cols[None, :], values, mask=valid[:, None])


# ---- device row cache ------------------------------------------------------
# A small set-associative cache in device memory of the table rows recently
# read over the bus: their raw fp8 bytes and e8m0 scale bytes, so a row served
# from it dequantizes to exactly what the bus read would have. It fills itself:
# a row that missed claims a way of its set (the least recently used one) and
# is written there after the gather, so the n-grams a request keeps repeating
# -- and the prompt n-grams its decode copies -- stop crossing the bus.
#
# Four launches per gather, all on the gather's stream, each covering one table
# or two (a decode step's tables in one go: `uva_gather_pair_into`):
#   device  -- device memory only: tags; hit: read the cache, stamp the way;
#              miss: claim the victim way (atomic max of the row id) and list
#              the row. A program per BLOCK_R rows: nothing here waits on the
#              bus, so the rows are spread over the whole device.
#   host    -- the listed misses only: read the host, stage the raw bytes,
#              write the row. Grid and rows per program as the plain gather
#              (`_uva_lookup_kernel`): the bus sees no more requests at once.
#   fill    -- the claim's winner copies its staged bytes in and sets the tag;
#   release -- every claim goes back to empty, the use clock advances and the
#              miss lists empty.
# Nothing reads the cache while it is filled (stream order), and the fill only
# reads claims, which nobody writes until the release: a winner is decided once
# however many threads Triton has load the same claim. Concurrent writers of
# one way carry the same row, so they write the same bytes.
#
# Two tables share one row space: [table 0's rows | padding to the device
# pass's block | table 1's rows]; a block never straddles the two, and the
# padding rows are no row of either.

ROW_CACHE_SLOTS_LOG2 = 22
# rows per program of the device pass (also the table 0 padding of a pair)
_DEVICE_ROWS = 64


@triton.jit
def _uva_cached_device_kernel(
    ids0,
    out0,
    tag0,
    stamp0,
    claim0,
    val0,
    cscale0,
    clock0,
    vocab_start0,
    vocab_end0,
    ids1,
    out1,
    tag1,
    stamp1,
    claim1,
    val1,
    cscale1,
    clock1,
    vocab_start1,
    vocab_end1,
    rows0,
    rows0_pad,
    num_rows,
    ids_stride_t,
    mslot,
    mrow,
    mlist,
    mcount,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    SETS_LOG2: tl.constexpr,
):
    """The device-memory half of a cached gather: one block of rows."""
    NSB: tl.constexpr = DIM // QUANT_BLOCK
    cols = tl.arange(0, DIM)
    sblk = cols // QUANT_BLOCK
    start = tl.program_id(0) * BLOCK_R
    if start >= rows0_pad:
        ids, out = ids1, out1
        ctag, cstamp, cclaim, cval, cscale = tag1, stamp1, claim1, val1, cscale1
        clock = tl.load(clock1)
        vocab_start, vocab_end = vocab_start1, vocab_end1
        base = start - rows0_pad
        nrows = num_rows - rows0_pad
        count = mcount + 1
        listed = mlist + rows0_pad
    else:
        ids, out = ids0, out0
        ctag, cstamp, cclaim, cval, cscale = tag0, stamp0, claim0, val0, cscale0
        clock = tl.load(clock0)
        vocab_start, vocab_end = vocab_start0, vocab_end0
        base = start
        nrows = rows0
        count = mcount
        listed = mlist
    rows = base + tl.arange(0, BLOCK_R)
    valid = rows < nrows
    head = HEAD_START + rows % LOCAL_HEADS
    token = (rows // LOCAL_HEADS).to(tl.int64)
    index = tl.load(
        ids + token * ids_stride_t + head,
        mask=valid & (head < TOTAL_HEADS),
        other=-1,
    ).to(tl.int64)
    owned = valid & (head < TOTAL_HEADS)
    owned &= (index >= vocab_start) & (index < vocab_end)
    local = tl.where(owned, index - vocab_start, 0)
    key = local.to(tl.int32)
    way0 = (((local * 2654435761) & 0xFFFFFFFF) >> (32 - SETS_LOG2)) * 2
    tag_a = tl.load(ctag + way0, mask=owned, other=-1)
    tag_b = tl.load(ctag + way0 + 1, mask=owned, other=-1)
    hit0 = owned & (tag_a == key)
    hit1 = owned & (tag_b == key) & (tag_a != key)
    hit = hit0 | hit1
    miss = owned & (tag_a != key) & (tag_b != key)
    hslot = way0 + hit1.to(tl.int64)
    age0 = tl.load(cstamp + way0, mask=miss, other=0)
    age1 = tl.load(cstamp + way0 + 1, mask=miss, other=0)
    vslot = way0 + (age1 < age0).to(tl.int64)
    tl.store(cstamp + hslot, tl.zeros_like(key) + clock, mask=hit)
    srow = start + tl.arange(0, BLOCK_R)
    tl.store(mslot + srow, tl.where(miss, vslot, -1).to(tl.int32), mask=valid)
    tl.store(mrow + srow, key, mask=valid)
    tl.atomic_max(cclaim + vslot, key, mask=miss)
    # a miss is listed for the host pass, which writes its row
    flag = miss.to(tl.int32)
    first = tl.atomic_add(count, tl.sum(flag, axis=0))
    tl.store(listed + first + tl.cumsum(flag, axis=0) - 1, rows, mask=miss)
    # a hit from the cache; a row nobody owns is zeros, as the bus path writes it
    raw = tl.load(
        cval + hslot[:, None] * DIM + cols[None, :], mask=hit[:, None], other=0
    )
    sc = tl.load(
        cscale + hslot[:, None] * NSB + sblk[None, :], mask=hit[:, None], other=0
    )
    values = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    values = values * (sc.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    tl.store(
        out + rows[:, None] * DIM + cols[None, :],
        values,
        mask=(valid & ~miss)[:, None],
    )


@triton.jit
def _uva_cached_host_kernel(
    weight0,
    scales0,
    out0,
    weight1,
    scales1,
    out1,
    rows0,
    rows0_pad,
    num_rows,
    shard_rows0,
    shard_rows1,
    mrow,
    mval,
    mscale,
    mlist,
    mcount,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    NUM_TABLES: tl.constexpr,
):
    """The bus half: the listed misses of each table, read from the host.

    The miss count lives in device memory, so it is bounded here by the
    table's row count and every listed row and host row id by its range: a
    count left stale cannot send this kernel to rows or host addresses
    outside the gather's."""
    NSB: tl.constexpr = DIM // QUANT_BLOCK
    cols = tl.arange(0, DIM)
    sblk = cols // QUANT_BLOCK
    first = (cols % QUANT_BLOCK) == 0
    for t in tl.static_range(NUM_TABLES):
        if t == 0:
            weight, scales, out, offset = weight0, scales0, out0, 0
            rows_t, shard_rows = rows0, shard_rows0
        else:
            weight, scales, out, offset = weight1, scales1, out1, rows0_pad
            rows_t, shard_rows = num_rows - rows0_pad, shard_rows1
        n = tl.minimum(tl.maximum(tl.load(mcount + t), 0), rows_t)
        for base in tl.range(
            tl.program_id(0) * BLOCK_R, n, tl.num_programs(0) * BLOCK_R
        ):
            k = base + tl.arange(0, BLOCK_R)
            live = k < n
            rows = tl.load(mlist + offset + k, mask=live, other=0)
            live &= (rows >= 0) & (rows < rows_t)
            srow = (rows + offset).to(tl.int64)
            local = tl.load(mrow + srow, mask=live, other=0).to(tl.int64)
            live &= (local >= 0) & (local < shard_rows)
            raw = tl.load(
                weight + local[:, None] * DIM + cols[None, :],
                mask=live[:, None],
                other=0,
            )
            sc = tl.load(
                scales + local[:, None] * NSB + sblk[None, :],
                mask=live[:, None],
                other=0,
            )
            tl.store(
                mval + srow[:, None] * DIM + cols[None, :], raw, mask=live[:, None]
            )
            tl.store(
                mscale + srow[:, None] * NSB + sblk[None, :],
                sc,
                mask=live[:, None] & first[None, :],
            )
            values = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            values = values * (sc.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            tl.store(
                out + rows[:, None] * DIM + cols[None, :], values, mask=live[:, None]
            )


@triton.jit
def _uva_cache_fill_kernel(
    tag0,
    stamp0,
    claim0,
    val0,
    cscale0,
    clock0,
    tag1,
    stamp1,
    claim1,
    val1,
    cscale1,
    clock1,
    rows0,
    rows0_pad,
    num_rows,
    mslot,
    mrow,
    mval,
    mscale,
    DIM: tl.constexpr,
    NSB: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    cols = tl.arange(0, DIM)
    scols = tl.arange(0, NSB)
    c0 = tl.load(clock0)
    c1 = tl.load(clock1)
    for base in tl.range(
        tl.program_id(0) * BLOCK_R, num_rows, tl.num_programs(0) * BLOCK_R
    ):
        rows = base + tl.arange(0, BLOCK_R)
        second = rows >= rows0_pad
        valid = (rows < rows0) | (second & (rows < num_rows))
        slot = tl.load(mslot + rows, mask=valid, other=-1)
        key = tl.load(mrow + rows, mask=valid, other=-1)
        cand = valid & (slot >= 0)
        s64 = tl.where(cand, slot, 0).to(tl.int64)
        claim_a = tl.load(claim0 + s64, mask=cand & ~second, other=-2)
        claim_b = tl.load(claim1 + s64, mask=cand & second, other=-2)
        win = cand & (tl.where(second, claim_b, claim_a) == key)
        win0 = win & ~second
        win1 = win & second
        rows64 = rows.to(tl.int64)
        v = tl.load(
            mval + rows64[:, None] * DIM + cols[None, :], mask=win[:, None], other=0
        )
        tl.store(val0 + s64[:, None] * DIM + cols[None, :], v, mask=win0[:, None])
        tl.store(val1 + s64[:, None] * DIM + cols[None, :], v, mask=win1[:, None])
        sv = tl.load(
            mscale + rows64[:, None] * NSB + scols[None, :],
            mask=win[:, None],
            other=0,
        )
        tl.store(cscale0 + s64[:, None] * NSB + scols[None, :], sv, mask=win0[:, None])
        tl.store(cscale1 + s64[:, None] * NSB + scols[None, :], sv, mask=win1[:, None])
        tl.store(tag0 + s64, key, mask=win0)
        tl.store(tag1 + s64, key, mask=win1)
        tl.store(stamp0 + s64, tl.zeros_like(key) + c0, mask=win0)
        tl.store(stamp1 + s64, tl.zeros_like(key) + c1, mask=win1)


@triton.jit
def _uva_cache_release_kernel(
    claim0,
    clock0,
    claim1,
    clock1,
    rows0,
    rows0_pad,
    num_rows,
    mslot,
    mcount,
    BLOCK_R: tl.constexpr,
    NUM_TABLES: tl.constexpr,
):
    if tl.program_id(0) == 0:
        tl.store(clock0, tl.load(clock0) + 1)
        if NUM_TABLES == 2:
            tl.store(clock1, tl.load(clock1) + 1)
        tl.store(mcount + tl.arange(0, 2), tl.zeros([2], tl.int32))
    for base in tl.range(
        tl.program_id(0) * BLOCK_R, num_rows, tl.num_programs(0) * BLOCK_R
    ):
        rows = base + tl.arange(0, BLOCK_R)
        second = rows >= rows0_pad
        valid = (rows < rows0) | (second & (rows < num_rows))
        slot = tl.load(mslot + rows, mask=valid, other=-1)
        target = tl.where(slot >= 0, slot, 0).to(tl.int64)
        empty = tl.full([BLOCK_R], -1, tl.int32)
        tl.store(claim0 + target, empty, mask=(slot >= 0) & ~second)
        tl.store(claim1 + target, empty, mask=(slot >= 0) & second)


class UvaMissStaging:
    """Per-row scratch the cached gathers of one rank share (they run in turn
    on one stream): the claimed way, the row id and the raw bytes of a miss,
    and each table's list of missed rows (`count` is zero between gathers)."""

    def __init__(self, max_rows: int, head_dim: int, quant_block: int, device):
        self.slot = torch.empty(max_rows, dtype=torch.int32, device=device)
        self.row = torch.empty(max_rows, dtype=torch.int32, device=device)
        self.val = torch.empty(max_rows, head_dim, dtype=torch.uint8, device=device)
        self.scale = torch.empty(
            max_rows, head_dim // quant_block, dtype=torch.uint8, device=device
        )
        self.misses = torch.empty(max_rows, dtype=torch.int32, device=device)
        self.count = torch.zeros(2, dtype=torch.int32, device=device)

    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (
                self.slot,
                self.row,
                self.val,
                self.scale,
                self.misses,
                self.count,
            )
        )


class UvaRowCache:
    """The device row cache of one table shard (see the block comment above)."""

    def __init__(
        self,
        head_dim: int,
        quant_block: int,
        staging: UvaMissStaging,
        device,
        slots_log2: int = ROW_CACHE_SLOTS_LOG2,
    ):
        slots = 1 << slots_log2
        self.sets_log2 = slots_log2 - 1  # two ways per set
        self.staging = staging
        self.tag = torch.full((slots,), -1, dtype=torch.int32, device=device)
        self.stamp = torch.zeros(slots, dtype=torch.int32, device=device)
        self.claim = torch.full((slots,), -1, dtype=torch.int32, device=device)
        self.val = torch.zeros(slots, head_dim, dtype=torch.uint8, device=device)
        self.scale = torch.zeros(
            slots, head_dim // quant_block, dtype=torch.uint8, device=device
        )
        self.clock = torch.ones(1, dtype=torch.int32, device=device)

    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (
                self.tag,
                self.stamp,
                self.claim,
                self.val,
                self.scale,
                self.clock,
            )
        )

    @staticmethod
    def fits(table) -> bool:
        """Only quantized tables are cached, and their row ids must fit int32."""
        _, _, row_start, row_end = table.registered_shard()
        return table.quantized and row_end - row_start < 2**31


def uva_gather_into(
    table,
    ids: torch.Tensor,
    out: torch.Tensor,
    *,
    head_start: int,
    local_heads: int,
    total_heads: int,
    cache: "UvaRowCache | None" = None,
) -> None:
    """Fill `out` ([tokens, local_heads, head_dim], device) from `ids`.

    `ids` is the FULL `[tokens, total_heads]` index matrix -- every rank sees
    every index and skips the ones outside its shard, so no index exchange is
    needed. Heads this rank does not own are left as zeros for the caller's
    all-gather. Requires `table.enable_uva()`.

    `cache` serves recently read rows from device memory (`UvaRowCache`); the
    output is bit-identical with or without it.
    """
    tokens = ids.shape[0]
    num_rows = tokens * local_heads
    if num_rows == 0:
        return
    if out.shape != (tokens, local_heads, table.head_dim):
        raise ValueError(
            f"engram UVA output is {tuple(out.shape)}, expected "
            f"{(tokens, local_heads, table.head_dim)}"
        )
    weight, scales, row_start, row_end = table.registered_shard()
    num_sms = torch.cuda.get_device_properties(out.device).multi_processor_count
    block = 16
    grid = min(-(-num_rows // block), num_sms)
    if cache is not None and num_rows <= cache.staging.slot.numel():
        _cached_gather(
            (table,), (cache,), (ids,), (out,), head_start, local_heads, total_heads
        )
        return
    _uva_lookup_kernel[(grid,)](
        weight,
        scales,
        ids,
        out,
        num_rows,
        row_start,
        row_end,
        ids.stride(0),
        HEAD_START=head_start,
        LOCAL_HEADS=local_heads,
        TOTAL_HEADS=total_heads,
        DIM=table.head_dim,
        QUANT_BLOCK=table.block_size or 1,
        BLOCK_R=block,
        HAS_SCALE=table.quantized,
    )


def uva_pair_fits(tables, caches, tokens: int, local_heads: int) -> bool:
    """Whether two tables' cached gathers of `tokens` rows go in one row space
    (`uva_gather_pair_into`): both cached on one miss scratch, laid out alike,
    and both tables' rows fit the scratch (a decode step, not a long prefill)."""
    if len(tables) != 2 or len(caches) != 2 or any(c is None for c in caches):
        return False
    (t0, t1), (k0, k1) = tables, caches
    rows = tokens * local_heads
    return (
        k0 is not k1
        and k0.staging is k1.staging
        and (t0.head_dim, t0.block_size, k0.sets_log2)
        == (t1.head_dim, t1.block_size, k1.sets_log2)
        and -(-rows // _DEVICE_ROWS) * _DEVICE_ROWS + rows <= k0.staging.slot.numel()
    )


def uva_gather_pair_into(
    tables,
    ids,
    outs,
    caches,
    *,
    head_start: int,
    local_heads: int,
    total_heads: int,
) -> bool:
    """`uva_gather_into` of two cached tables in one go: the same rows, one
    launch per pass instead of one per table. Returns False, having launched
    nothing, where they do not fit one row space (`uva_pair_fits`); the caller
    gathers them one by one."""
    tokens = ids[0].shape[0]
    if not uva_pair_fits(tables, caches, tokens, local_heads):
        return False
    if ids[1].shape[0] != tokens or ids[0].stride(0) != ids[1].stride(0):
        return False
    for table, out in zip(tables, outs):
        if out.shape != (tokens, local_heads, table.head_dim):
            raise ValueError(
                f"engram UVA output is {tuple(out.shape)}, expected "
                f"{(tokens, local_heads, table.head_dim)}"
            )
    if tokens * local_heads:
        _cached_gather(tables, caches, ids, outs, head_start, local_heads, total_heads)
    return True


def _cached_gather(tables, caches, ids, outs, head_start, local_heads, total_heads):
    """One or two tables through their row caches (see the block comment)."""
    pair = len(tables) == 2
    t0, t1 = tables[0], tables[-1]
    k0, k1 = caches[0], caches[-1]
    staging = k0.staging
    rows0 = ids[0].shape[0] * local_heads
    rows0_pad = -(-rows0 // _DEVICE_ROWS) * _DEVICE_ROWS if pair else rows0
    num_rows = rows0_pad + rows0 if pair else rows0
    weight0, scales0, start0, end0 = t0.registered_shard()
    weight1, scales1, start1, end1 = t1.registered_shard()
    quant_block = t0.block_size
    num_sms = torch.cuda.get_device_properties(outs[0].device).multi_processor_count
    # device memory only: a program per block of rows
    _uva_cached_device_kernel[(-(-num_rows // _DEVICE_ROWS),)](
        ids[0],
        outs[0],
        k0.tag,
        k0.stamp,
        k0.claim,
        k0.val,
        k0.scale,
        k0.clock,
        start0,
        end0,
        ids[-1],
        outs[-1],
        k1.tag,
        k1.stamp,
        k1.claim,
        k1.val,
        k1.scale,
        k1.clock,
        start1,
        end1,
        rows0,
        rows0_pad,
        num_rows,
        ids[0].stride(0),
        staging.slot,
        staging.row,
        staging.misses,
        staging.count,
        HEAD_START=head_start,
        LOCAL_HEADS=local_heads,
        TOTAL_HEADS=total_heads,
        DIM=t0.head_dim,
        QUANT_BLOCK=quant_block,
        BLOCK_R=_DEVICE_ROWS,
        SETS_LOG2=k0.sets_log2,
    )
    # the bus: grid and rows per program as `_uva_lookup_kernel`
    block = 16
    _uva_cached_host_kernel[(min(-(-num_rows // block), num_sms),)](
        weight0.view(torch.uint8),
        scales0,
        outs[0],
        weight1.view(torch.uint8),
        scales1,
        outs[-1],
        rows0,
        rows0_pad,
        num_rows,
        end0 - start0,
        end1 - start1,
        staging.row,
        staging.val,
        staging.scale,
        staging.misses,
        staging.count,
        DIM=t0.head_dim,
        QUANT_BLOCK=quant_block,
        BLOCK_R=block,
        NUM_TABLES=2 if pair else 1,
    )
    fill_block = 32
    _uva_cache_fill_kernel[(min(-(-num_rows // fill_block), num_sms),)](
        k0.tag,
        k0.stamp,
        k0.claim,
        k0.val,
        k0.scale,
        k0.clock,
        k1.tag,
        k1.stamp,
        k1.claim,
        k1.val,
        k1.scale,
        k1.clock,
        rows0,
        rows0_pad,
        num_rows,
        staging.slot,
        staging.row,
        staging.val,
        staging.scale,
        DIM=t0.head_dim,
        NSB=t0.head_dim // quant_block,
        BLOCK_R=fill_block,
    )
    release_block = 256
    _uva_cache_release_kernel[(min(-(-num_rows // release_block), num_sms),)](
        k0.claim,
        k0.clock,
        k1.claim,
        k1.clock,
        rows0,
        rows0_pad,
        num_rows,
        staging.slot,
        staging.count,
        BLOCK_R=release_block,
        NUM_TABLES=2 if pair else 1,
    )
