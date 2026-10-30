# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""FP8 FULL/REINDEX selection and key expansion within the gfx1250 K2.

The short-context build covers up to 8192 compressed rows. At this bound
there are at most 1024 candidate blocks, so candidate top-2048 is all blocks
in index order, as index_plan specifies. Larger bounds need its merge tree.
FP8 values are exactly representable in BF16; the score uses BF16 WMMA with
FP32 accumulation, followed by the original key scale and folded head weight.
"""

import flydsl.expr as fx
from flydsl.expr import range_constexpr

from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_attention import _wmma
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_k1 import _await, _publish
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_mhc import _typed
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_topk import lds_topk, lds_words
from atom.mono.device.ops import lane_gather, traced

TOPK, BLOCKS, TILE_BYTES = 512, 2048, 1056


def index_lds_bytes(bound_max):
    per_thread = max(4, ((bound_max + 1023) // 1024) * 4)
    return max(16384, sum(lds_words(per_thread).values()) * 4)


def _fp8_bf16(words):
    # E4M3FN normals and subnormals are exact in BF16. The pinned gfx1250
    # compiler cannot lower FlyDSL's vector<fp8> extension or the gfx950
    # packed conversion opcode; construct the IEEE FP32 bits directly.
    values = []
    for j in range(16):
        byte = (fx.Int32(words[j // 4]) >> (8 * (j % 4))) & 255
        exp, mant = (byte >> 3) & 15, byte & 7
        sign = (byte & 128) << 24
        normal = ((exp + 120) << 23) | (mant << 20)
        sub = (mant.to(fx.Float32) * fx.Float32(1.0 / 512.0)).bitcast(fx.Int32)
        bits = (exp == 0).select(sub, normal) | sign
        bits = ((exp == 15) & (mant == 7)).select(fx.Int32(0x7FC00000), bits)
        values.append(bits.bitcast(fx.Float32).to(fx.BFloat16))
    return fx.Vector.from_elements(values, fx.BFloat16)


@traced
def _error(diag, code, bad):
    if bad:
        fx.atomic_cas(
            diag, fx.Int32(0), fx.Int32(code) + fx.Int32(fx.block_idx.x),
            success_ordering=fx.AtomicOrdering.Release,
            failure_ordering=fx.AtomicOrdering.Monotonic, syncscope="agent",
        )


@traced
def _lift(a, row, ident, reindex):
    value = ident
    if reindex & (ident >= 0):
        candidate = fx.ptr_load(_typed(a("i_candidates"), fx.Int32) + row * BLOCKS + ident // 8)
        value = candidate * 8 + ident % 8
    return value


@traced
def _page(a, row, ident, rpp, diag, table_entries):
    offset = fx.ptr_load(_typed(a("i_keymeta"), fx.Int32) + row * 4 + 2)
    at = offset + fx.max(ident, 0) // rpp
    valid = (ident >= 0) & (at >= 0) & (at < table_entries)
    page = fx.Int32(-1)
    if valid:
        page = fx.ptr_load(_typed(a("i_table"), fx.Int32) + at)
    _error(diag, 34_000_000, (ident >= 0) & ((~valid) | (page < 0)))
    return page


@traced
def score_chunk(a, row, chunk, bound, p, lds, diag, tid):
    lane, wave = tid % 32, tid // 32
    lane16, half = lane % 16, lane // 16
    reindex, rpp, tiles, entries = p[1] != 0, p[3], p[6], p[8]
    column = chunk * 128 + wave * 16 + lane16
    ident = _lift(a, row, column, reindex)
    live = column < bound
    page = _page(a, row, live.select(ident, fx.Int32(-1)), rpp, diag, entries)
    page_valid = (page >= 0) & (page < tiles // (rpp // 8))
    page = page_valid.select(page, fx.Int32(-1))
    tile = page * (rpp // 8) + fx.max(ident, 0) % rpp // 8
    valid = live & (ident >= 0) & (tile >= 0) & (tile < tiles)
    _error(diag, 35_000_000, live & (~valid))
    key_scale = fx.Float32(0.0)
    plane = fx.Int64(fx.ptrtoint(a("i_plane"))) + fx.Int64(fx.max(tile, 0)) * TILE_BYTES
    if valid:
        key_scale = fx.ptr_load(
            fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type, 1, 4), plane + 1024 + (ident % 8) * 4)
        )
    partial = _typed(lds, fx.Float32)
    for ht in range_constexpr(2):
        head = ht * 16 + lane16
        accum = fx.make_rmem_tensor(8, fx.Float32)
        accum.store(fx.full(8, 0.0, fx.Float32))
        for kt in range_constexpr(4):
            d = kt * 32 + half * 16
            query_words = fx.Vector(fx.ptr_load(
                _typed(a("i_query"), fx.Int32) + ((row * 32 + head) * 128 + d) // 4,
                result_type=fx.Vector.make_type(4, fx.Int32),
            ))
            query = _fp8_bf16(query_words)
            key = fx.Vector.filled(16, 0.0, fx.BFloat16)
            if valid:
                ptr = fx.inttoptr(
                    fx.PointerType.get(fx.Int32.ir_type, 1, 16),
                    plane + (d // 16 * 8 + ident % 8) * 16,
                )
                key_words = fx.Vector(fx.ptr_load(ptr, result_type=fx.Vector.make_type(4, fx.Int32)))
                key = _fp8_bf16(key_words)
            _wmma(accum, query, key)
        weights = fx.ptr_load(_typed(a("i_weights"), fx.Float32) + row * 32 + head)
        got = accum.load()
        for j in range_constexpr(8):
            # WMMA's output key is half*8+j, rather than this lane's key.
            ks = lane_gather(key_scale.bitcast(fx.Int32), half * 8 + j).bitcast(fx.Float32)
            term = fx.max(got[j] * ks, fx.Float32(0.0)) * weights
            fx.ptr_store(term, partial + (wave * 16 + half * 8 + j) * 32 + head)
    fx.gpu.barrier()
    if tid < 128:
        total = fx.Float32(0.0)
        for h in range_constexpr(32):
            total = total + fx.ptr_load(partial + tid * 32 + h)
        col = chunk * 128 + tid
        # Every padded position of the last chunk is also written.
        fx.ptr_store((col < bound).select(total, fx.Float32(float("-inf"))),
                     _typed(a("i_logits"), fx.Float32) + row * p[10] + col)
    fx.gpu.barrier()


@traced
def run_index(a, c, bound_max, index_flags):
    rows, tid, bid = c["rows"], c["tid"], c["bid"]
    flags, diag, epoch = c["flags"], c["diag"], c["epoch"]
    local, status = c["local"], c["status"]
    failure_detail = c.get("failure_detail")
    p = [fx.ptr_load(_typed(a("i_config"), fx.Int32) + i) for i in range(11)]
    chunks = (bound_max + 127) // 128
    selecting, ratio = p[0] != 0, p[9]
    if selecting & (bid < rows * chunks):
        row, chunk = bid // chunks, bid % chunks
        bound = fx.ptr_load(_typed(a("i_visible"), fx.Int32) + row)
        _error(diag, 36_000_000, (bound < 0) | (bound > bound_max))
        if (bound > TOPK) & (bound <= bound_max) & (chunk * 128 < bound):
            ready = _await(flags, 0, 0, epoch, diag, local, tid, 320, 36, status=status,
                           failure_detail=failure_detail)
            if ready:
                score_chunk(a, row, chunk, bound, p, c["arena"].base_ptr, diag, tid)
                _publish(flags, index_flags + bid, epoch, synchronized=True)
    if bid < rows:
        row = bid
        bound = fx.Int32(0)
        if ratio != 0:
            bound = fx.ptr_load(_typed(a("i_visible"), fx.Int32) + row)
        _error(diag, 36_000_000, (bound < 0) | (bound > bound_max))
        bound = fx.min(fx.max(bound, 0), bound_max)
        if selecting:
            if bound > TOPK:
                ready = _await(flags, index_flags + row * chunks, chunks,
                               epoch, diag, local, tid, 320, 37, status=status,
                               active_count=(bound + 127) // 128,
                               failure_detail=failure_detail)
                if ready:
                    per_thread = max(4, ((bound_max + 1023) // 1024) * 4)
                    sizes = lds_words(per_thread)
                    keys = local
                    hist = keys + sizes["keys"]
                    buf = hist + sizes["hist"]

                    def load(q, i):
                        ptr = _typed(a("i_logits"), fx.Int32) + row * bound_max + i
                        v = fx.Vector(fx.ptr_load(ptr, result_type=fx.Vector.make_type(4, fx.Int32)))
                        return [fx.Int32(v[j]) for j in range(4)]

                    def emit(pos, d, i, extra):
                        ident = fx.Int32(-1) if d is None else _lift(a, row, i, p[1] != 0)
                        fx.ptr_store(ident, _typed(a("i_selection"), fx.Int32) + row * TOPK + pos)

                    lds_topk(bound, TOPK, load, emit, {"keys": keys, "hist": hist, "buf": buf}, tid, per_thread)
            else:
                for r in range_constexpr(2):
                    col = tid + r * 256
                    ident = fx.Int32(-1)
                    if col < bound:
                        ident = _lift(a, row, col, p[1] != 0)
                    fx.ptr_store(ident, _typed(a("i_selection"), fx.Int32) + row * TOPK + col)
            if p[2] != 0:
                for r in range_constexpr(BLOCKS // 256):
                    col = tid + r * 256
                    value = (col < (bound + 7) // 8).select(col, fx.Int32(-1))
                    fx.ptr_store(value, _typed(a("i_candidates"), fx.Int32) + row * BLOCKS + col)
            fx.gpu.barrier()
        # Each layer expands its own pool owner and shared window metadata.
        if tid == 0:
            for i in range_constexpr(rows + 1):
                fx.ptr_store((ratio != 0).select(fx.Int32(i * TOPK), fx.Int32(0)),
                             _typed(a("prefix_indptr"), fx.Int32) + i)
                fx.ptr_store(fx.Int32(i * c["window_size"]), _typed(a("extend_indptr"), fx.Int32) + i)
        if ratio != 0:
            for r in range_constexpr(2):
                col = tid + r * 256
                ident = fx.ptr_load(_typed(a("i_selection"), fx.Int32) + row * TOPK + col)
                page = _page(a, row, ident, p[3], diag, p[8])
                slot = fx.Int64(page) * fx.Int64(p[4]) + fx.Int64(p[5]) + fx.Int64(fx.max(ident, 0) % p[3])
                live = (ident >= 0) & (slot >= 0) & (slot < c["pool_rows"])
                _error(diag, 38_000_000, (ident >= 0) & (~live))
                fx.ptr_store(live.select(slot, fx.Int64(-1)).to(fx.Int32), _typed(a("prefix"), fx.Int32) + row * TOPK + col)
        meta = _typed(a("i_keymeta"), fx.Int32) + row * 4
        first, count, base = fx.ptr_load(meta), fx.ptr_load(meta + 1), fx.ptr_load(meta + 3)
        for r in range_constexpr((c["window_size"] + 255) // 256):
            col = tid + r * 256
            if col < c["window_size"]:
                slot = base + (first + col) % p[7]
                live = (col < count) & (slot >= 0) & (slot < c["ring_rows"])
                _error(diag, 38_000_000, (col < count) & (~live))
                fx.ptr_store(live.select(slot, fx.Int32(-1)), _typed(a("extend"), fx.Int32) + row * c["window_size"] + col)
        _publish(flags, index_flags + rows * chunks + row, epoch)
