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
# Three launches per gather, all on the gather's stream:
#   lookup  -- hit: read the cache; miss: read the host, stage the raw bytes,
#              and claim the victim way (atomic max of the row id);
#   fill    -- the claim's winner copies its staged bytes in and sets the tag;
#   release -- every claim goes back to empty and the use clock advances.
# Nothing reads the cache while it is filled (stream order), and the fill only
# reads claims, which nobody writes until the release: a winner is decided once
# however many threads Triton has load the same claim. Concurrent writers of
# one way carry the same row, so they write the same bytes.

ROW_CACHE_SLOTS_LOG2 = 22


@triton.jit
def _uva_lookup_cached_kernel(
    weight,
    scales,
    ids,
    out,
    num_rows,
    vocab_start,
    vocab_end,
    ids_stride_t,
    ctag,
    cstamp,
    cclaim,
    cval,
    cscale,
    cclock,
    mslot,
    mrow,
    mval,
    mscale,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    SETS_LOG2: tl.constexpr,
):
    """`_uva_lookup_kernel` for a quantized table, through a 2-way row cache."""
    NSB: tl.constexpr = DIM // QUANT_BLOCK
    cols = tl.arange(0, DIM)
    sblk = cols // QUANT_BLOCK
    clock = tl.load(cclock)
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
        key = local.to(tl.int32)
        way0 = (((local * 2654435761) & 0xFFFFFFFF) >> (32 - SETS_LOG2)) * 2
        tag0 = tl.load(ctag + way0, mask=owned, other=-1)
        tag1 = tl.load(ctag + way0 + 1, mask=owned, other=-1)
        hit0 = owned & (tag0 == key)
        hit1 = owned & (tag1 == key) & (tag0 != key)
        hit = hit0 | hit1
        miss = owned & (tag0 != key) & (tag1 != key)
        hslot = way0 + hit1.to(tl.int64)
        age0 = tl.load(cstamp + way0, mask=miss, other=0)
        age1 = tl.load(cstamp + way0 + 1, mask=miss, other=0)
        vslot = way0 + (age1 < age0).to(tl.int64)
        tl.store(cstamp + hslot, tl.zeros_like(key) + clock, mask=hit)
        raw_c = tl.load(
            cval + hslot[:, None] * DIM + cols[None, :], mask=hit[:, None], other=0
        )
        raw_h = tl.load(
            weight + local[:, None] * DIM + cols[None, :], mask=miss[:, None], other=0
        )
        sc_c = tl.load(
            cscale + hslot[:, None] * NSB + sblk[None, :], mask=hit[:, None], other=0
        )
        sc_h = tl.load(
            scales + local[:, None] * NSB + sblk[None, :], mask=miss[:, None], other=0
        )
        rows64 = rows.to(tl.int64)
        tl.store(
            mval + rows64[:, None] * DIM + cols[None, :], raw_h, mask=miss[:, None]
        )
        first = (cols % QUANT_BLOCK) == 0
        tl.store(
            mscale + rows64[:, None] * NSB + sblk[None, :],
            sc_h,
            mask=miss[:, None] & first[None, :],
        )
        tl.store(mslot + rows, tl.where(miss, vslot, -1).to(tl.int32), mask=valid)
        tl.store(mrow + rows, key, mask=valid)
        tl.atomic_max(cclaim + vslot, key, mask=miss)
        raw = tl.where(hit[:, None], raw_c, raw_h)
        sc = tl.where(hit[:, None], sc_c, sc_h)
        values = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        values = values * (sc.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        tl.store(out + rows[:, None] * DIM + cols[None, :], values, mask=valid[:, None])


@triton.jit
def _uva_cache_fill_kernel(
    num_rows,
    ctag,
    cstamp,
    cclaim,
    cval,
    cscale,
    cclock,
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
    clock = tl.load(cclock)
    for base in tl.range(
        tl.program_id(0) * BLOCK_R, num_rows, tl.num_programs(0) * BLOCK_R
    ):
        rows = base + tl.arange(0, BLOCK_R)
        valid = rows < num_rows
        slot = tl.load(mslot + rows, mask=valid, other=-1)
        key = tl.load(mrow + rows, mask=valid, other=-1)
        cand = valid & (slot >= 0)
        s64 = tl.where(cand, slot, 0).to(tl.int64)
        win = cand & (tl.load(cclaim + s64, mask=cand, other=-2) == key)
        rows64 = rows.to(tl.int64)
        v = tl.load(
            mval + rows64[:, None] * DIM + cols[None, :], mask=win[:, None], other=0
        )
        tl.store(cval + s64[:, None] * DIM + cols[None, :], v, mask=win[:, None])
        sv = tl.load(
            mscale + rows64[:, None] * NSB + scols[None, :], mask=win[:, None], other=0
        )
        tl.store(cscale + s64[:, None] * NSB + scols[None, :], sv, mask=win[:, None])
        tl.store(ctag + s64, key, mask=win)
        tl.store(cstamp + s64, tl.zeros_like(key) + clock, mask=win)


@triton.jit
def _uva_cache_release_kernel(num_rows, cclaim, cclock, mslot, BLOCK_R: tl.constexpr):
    if tl.program_id(0) == 0:
        tl.store(cclock, tl.load(cclock) + 1)
    for base in tl.range(
        tl.program_id(0) * BLOCK_R, num_rows, tl.num_programs(0) * BLOCK_R
    ):
        rows = base + tl.arange(0, BLOCK_R)
        slot = tl.load(mslot + rows, mask=rows < num_rows, other=-1)
        tl.store(
            cclaim + tl.where(slot >= 0, slot, 0).to(tl.int64),
            tl.full([BLOCK_R], -1, tl.int32),
            mask=slot >= 0,
        )


class UvaMissStaging:
    """Per-row scratch the cached gathers of one rank share (they run in turn
    on one stream): the claimed way, the row id and the raw bytes of a miss."""

    def __init__(self, max_rows: int, head_dim: int, quant_block: int, device):
        self.slot = torch.empty(max_rows, dtype=torch.int32, device=device)
        self.row = torch.empty(max_rows, dtype=torch.int32, device=device)
        self.val = torch.empty(max_rows, head_dim, dtype=torch.uint8, device=device)
        self.scale = torch.empty(
            max_rows, head_dim // quant_block, dtype=torch.uint8, device=device
        )

    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.slot, self.row, self.val, self.scale)
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
            table,
            cache,
            ids,
            out,
            num_rows,
            grid,
            num_sms,
            block,
            head_start,
            local_heads,
            total_heads,
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


def _cached_gather(
    table,
    cache,
    ids,
    out,
    num_rows,
    grid,
    num_sms,
    block,
    head_start,
    local_heads,
    total_heads,
):
    weight, scales, row_start, row_end = table.registered_shard()
    quant_block = table.block_size
    staging = cache.staging
    _uva_lookup_cached_kernel[(grid,)](
        weight.view(torch.uint8),
        scales,
        ids,
        out,
        num_rows,
        row_start,
        row_end,
        ids.stride(0),
        cache.tag,
        cache.stamp,
        cache.claim,
        cache.val,
        cache.scale,
        cache.clock,
        staging.slot,
        staging.row,
        staging.val,
        staging.scale,
        HEAD_START=head_start,
        LOCAL_HEADS=local_heads,
        TOTAL_HEADS=total_heads,
        DIM=table.head_dim,
        QUANT_BLOCK=quant_block,
        BLOCK_R=block,
        SETS_LOG2=cache.sets_log2,
    )
    fill_block = 32
    _uva_cache_fill_kernel[(min(-(-num_rows // fill_block), num_sms),)](
        num_rows,
        cache.tag,
        cache.stamp,
        cache.claim,
        cache.val,
        cache.scale,
        cache.clock,
        staging.slot,
        staging.row,
        staging.val,
        staging.scale,
        DIM=table.head_dim,
        NSB=table.head_dim // quant_block,
        BLOCK_R=fill_block,
    )
    release_block = 256
    _uva_cache_release_kernel[(min(-(-num_rows // release_block), num_sms),)](
        num_rows, cache.claim, cache.clock, staging.slot, BLOCK_R=release_block
    )
