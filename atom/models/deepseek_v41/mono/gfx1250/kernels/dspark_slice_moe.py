# SPDX-License-Identifier: MIT
"""c8 routed MoE as expert K128 slices: UG and its down slice on one CTA.

A unit is (union slot u, K128 slice s of the intermediate). Its CTA runs the
two original UG tasks of the slice's four FP4 groups for every pick tile of
the slot (unchanged ``pipeline``/``epilogue``: same MID rounding), then the
down K128 step s of those MID rows against all 320 output N16 tiles, and
stores the FP32 partial per (pick, s). No other CTA waits on its MID. After
every CTA has published its units, each column owner sums a pick's slices
(the original two K chains: even and odd steps), applies the route weight,
rounds to BF16 and runs the original shared/top-k finish and TP2.

The down K order differs from one full-K WMMA chain only in FP32 association
(each slice's WMMA starts from zero); every BF16 rounding point is kept.
"""
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_down import *
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_native_support import (
    BUFFERS, B_WORDS, SLOT_WORDS, jobs, prime, pipeline, tensor, ready, retire_reads,
)
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_native_ug import epilogue, X8_WORDS
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_round_column_down import (
    bulk_finish, operands,
)
from atom.models.deepseek_v41.mono.gfx1250.kernels.reference_sync import publish_plain as publish
from flydsl._mlir.dialects import llvm
from atom.models.deepseek_v41.mono.gfx1250.kernels.reference_sync import memrealtime

import os as _os
_SKIP = _os.environ.get('SLICE_DEBUG_SKIP', '')  # TEMP resource bisection
# TEMP diagnosis: per CTA and layer, int64 clock stamps after the partials
STAMPS = _os.environ.get('SLICE_STAMPS', '0') == '1'
STAMP_SLOTS = 8
STAMP_WORDS = 64 * BLOCKS * STAMP_SLOTS * 2
# TEMP diagnosis: K2 stage stamps (``dspark_mhtjstest_native_k2``), after the slice stamps
K2_STAMPS = _os.environ.get('K2_STAMPS', '0') == '1'
K2_STAMP_SLOTS = 16
K2_STAMP_WORDS = 64 * BLOCKS * K2_STAMP_SLOTS * 2
K_SLICES = 5          # K128 steps of the padded intermediate (640)
OUT_TILES = HIDDEN // 16
PAIRS_PER_WAVE = OUT_TILES // 2 // WAVES  # output N16 tile pairs a wave owns


def partial_words(rs):
    """FP32 words of the per-(pick, slice) down partials."""
    return rs.picks * K_SLICES * HIDDEN


def unit_count(c):
    return c['rs'].n_union(c['tab'], c['S']) * K_SLICES


@traced
def gather_issue(c, first, count):
    tid = c['tid']
    live = fx.min(count, fx.Int32(16))
    def token(row):return c['rs'].pick_at(c['tab'], first + row) // c['rs'].topk
    if tid < 16:
        c['poll']([(c['x8rdy'], token(fx.min(tid, live - 1)), 1)])
    gpu.barrier()
    # A full token is contiguous; the four-dword LDS pad is at row end.
    # Each wave owns two rows. Do not gather all column parts onto one CTA:
    # retain gfx950's dynamic task supply and complete per-task input.
    for half in range_constexpr(2):
        row = c['wave'] * 2 + half
        if row < live:
            address = c['x8'].value + fx.Int64(token(row)) * X8_WORDS * 4
            pointer = fx.inttoptr(fx.PointerType.get(fx.Int32.ir_type,
                fx.AddressSpace.Global, 16), address)
            src = tensor(pointer, (5, 256), (256, 1))
            atom = fx.rocdl.make_tdm_atom(src, [5, 256], strides=[256, None],
                num_warps=1, cache_modifier=CM_DEV)
            dst = tensor(c['native_x'] + row * lds_row(X8_WORDS),
                         (5, 256), (256, 1))
            fx.copy(atom, src, dst)
    # Full A/SA lifecycle: pack each four group32 codes once per CTA tile.
    steps = HIDDEN // 128
    for begin in range_constexpr(0, 16 * steps, THREADS):
        q = begin + tid
        row, step = q // steps, q % steps
        if (q < 16 * steps) & (row < live):
            codes = fx.Vector(bo.buffer_load(rsrc(c['x8s'].value), token(row) * (HIDDEN // 32) + step * 4,
                vec_width=4, dtype=T.i32, cache_modifier=CM_DEV))
            code = codes[0] | (codes[1] << 8) | (codes[2] << 16) | (codes[3] << 24)
            fx.ptr_store(code, c['native_s'] + row * steps + step)
    # No wait here: the caller primes its weights behind these x rows and
    # waits for the rows alone (TDM completes in issue order).



def ug_job(c, e, s, half):
    part0 = (2 * s + half) * c['rs'].ug_groups
    _, tile, ks, _ = ug_wave_pair(c, part0)
    job = jobs(c, c['args']['w13'], c['args']['w13_s'], local_expert(c, e), tile,
               HIDDEN, 2 * c['d'].inter, HIDDEN // 32)
    return part0, job, ks


def down_prime(c, e, s):
    """The wave's first ring-full of down tile pairs (``down_slice``)."""
    retire_reads()
    for k in range_constexpr(BUFFERS):
        down_issue(c, e, s, k, k)


@traced
def ug_slice(c, e, first, count, s):
    """The slice's UG tasks (parts 2 s, 2 s + 1; the last slice has one) for
    picks first .. first + count (at most 16), MID to the scratch rows as the
    original epilogue writes it. One x gather serves both; the next weights
    (the second task's, then the down slice's) stream in before each epilogue."""
    row = fx.min(c['lane'] % 16, count - 1)
    part0, job, ks = ug_job(c, e, s, 0)
    retire_reads()
    gather_issue(c, first, count)
    prime(c, job, ks)
    # this wave's x rows landed (the primed weights may still be in flight)
    ready(2 * BUFFERS)
    gpu.barrier()
    acc = pipeline(c, job, HIDDEN // 256, row, HIDDEN, ks)
    if s < K_SLICES - 1:
        part1, job1, ks1 = ug_job(c, e, s, 1)
        retire_reads()
        prime(c, job1, ks1)
        epilogue(c, part0, first, count, acc)
        acc1 = pipeline(c, job1, HIDDEN // 256, row, HIDDEN, ks1)
        down_prime(c, e, s)
        epilogue(c, part1, first, count, acc1)
    else:
        down_prime(c, e, s)
        epilogue(c, part0, first, count, acc)


def down_job(c, e, s, pair):
    """Output tiles 2 pair, 2 pair + 1 at K128 step s: their 1 KB weight rows
    (5 KB apart, the native [N16][K128] layout) and their n32k4 scale row."""
    d = c['d']
    words = (fx.Int64(e) * OUT_TILES + fx.Int64(2 * pair)) * (K_SLICES * 256) + fx.Int64(s) * 256
    pointer = fx.inttoptr(fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 16),
                          c['args']['w2'] + words * 4)
    bw = tensor(pointer, (2, 256), (K_SLICES * 256, 1))
    ba = fx.rocdl.make_tdm_atom(bw, [2, 256], strides=[K_SLICES * 256, None],
                                num_warps=1, cache_modifier=CM_NT)
    words = (fx.Int64(e) * (HIDDEN // 32) + fx.Int64(pair)) * (d.down_scale_cols * 8) + fx.Int64(s) * 32
    pointer = fx.inttoptr(fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 16),
                          c['args']['w2_s'] + words * 4)
    sw = tensor(pointer, (1, 32), (32, 1))
    sa = fx.rocdl.make_tdm_atom(sw, [1, 32], strides=[32, None], num_warps=1, cache_modifier=CM_NT)
    return ba, bw, sa, sw


def down_issue(c, e, s, k, slot):
    """This wave's k-th tile pair into its ring slot."""
    ba, bw, sa, sw = down_job(c, e, s, c['wave'] * PAIRS_PER_WAVE + k)
    base = c['native_ring'] + c['wave'] * (BUFFERS * SLOT_WORDS) + slot * SLOT_WORDS
    fx.copy(ba, bw, tensor(base, (2, 256), (256, 1)))
    fx.copy(sa, sw, tensor(base + B_WORDS, (1, 32), (32, 1)))


@traced
def down_slice(c, e, first, count, s):
    """Down K128 step s of MID rows first .. first + count (at most 16) against
    the wave's output tile pairs; FP32 partials stored by pick."""
    lane, d = c['lane'], c['d']
    atom = fx.make_mma_atom(fx.rocdl.WMMAScale(16, 16, 128,
        fx.Float4E2M1FN, fx.Float8E4M3FN, fx.Float32))
    # the first BUFFERS pairs were issued by ``ug_slice`` (``down_prime``)
    # A operand: the 16 MID rows' K128 step s, as the column down reads it.
    row = first + fx.min(lane % 16, count - 1)
    chunks = [fx.Vector(bo.buffer_load(rsrc(c['mid'].value),
                  row * d.mid_words + s * 32 + lane // 16 * 4 + h * 8,
                  vec_width=4, dtype=T.i32, cache_modifier=CM_DEV)) for h in range(4)]
    x = fx.make_rmem_tensor(16, fx.Int32)
    x.store(fx.Vector.from_elements([chunks[h][q] for h in range(4) for q in range(4)], fx.Int32))
    sx = fx.Int32(0)
    for group in range_constexpr(4):
        code = fx.Int32(bo.buffer_load(rsrc(c['mids'].value), mid_code_index(d, row, s * 4 + group),
                                       vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
        sx = sx | (code << (8 * group))
    pick = c['rs'].pick_at(c['tab'], row)
    live = (lane % 16) < count
    at = lane // 16 * 64 + lane % 16 * 4

    def tile_pair(k, outstanding, refill):
        """Pair k from its ring slot: wait, read B/SB, refill the slot, compute,
        store both tiles' partials."""
        slot = k % BUFFERS
        ready(2 * outstanding)
        base = c['native_ring'] + c['wave'] * (BUFFERS * SLOT_WORDS) + slot * SLOT_WORDS
        w = [fx.make_rmem_tensor(8, fx.Int32) for _ in range(2)]
        for tile in range_constexpr(2):
            halves = [fx.Vector(fx.ptr_load(base + tile * 256 + at + p * 128,
                        result_type=fx.Vector.make_type(4, fx.Int32))) for p in range(2)]
            w[tile].store(halves[0].shuffle(halves[1], list(range(8))))
        scales = [fx.ptr_load(base + B_WORDS + tile * 16 + lane % 16) for tile in range(2)]
        if const_expr(refill):
            retire_reads()
            down_issue(c, e, s, k + BUFFERS, slot)
        pair = c['wave'] * PAIRS_PER_WAVE + k
        for tile in range_constexpr(2):
            acc = fx.make_rmem_tensor(8, fx.Float32)
            acc.store(fx.full(8, 0., fx.Float32))
            fx.gemm(atom, acc, w[tile], x, acc, scale_a=scales[tile], scale_b=sx)
            col = (2 * pair + tile) * 16 + 8 * (lane // 16)
            if live:
                value = acc.load()
                for h in range_constexpr(2):
                    bo.buffer_store(value.shuffle(value, list(range(4 * h, 4 * h + 4))),
                                    rsrc(c['partial']), (pick * K_SLICES + s) * HIDDEN + col + 4 * h,
                                    cache_modifier=CM_DEV)

    # rolled steady state (one ring slot refilled a pair), unrolled drain
    for k in range(PAIRS_PER_WAVE - BUFFERS):
        tile_pair(k, BUFFERS - 1, True)
    for k in range_constexpr(PAIRS_PER_WAVE - BUFFERS, PAIRS_PER_WAVE):
        tile_pair(fx.Int32(k), PAIRS_PER_WAVE - 1 - k, False)
    ready(0)


@traced
def slice_units(c, bid):
    """Units (pick tile i, slice s), j = i K_SLICES + s: the CTA's first is its
    bid, then the launch's queue (``take_ug_task``, as the UG stage takes its
    tasks): a unit is one pick tile, so a hot expert's tiles spread over CTAs."""
    tab = c['tab']
    total = c['rs'].tile_first(tab, c['rs'].n_union(tab, c['S'])) * K_SLICES
    j = fx.Int32(bid)
    while j < total:
        i, s = j // K_SLICES, j % K_SLICES
        u = c['rs'].tile_slot(tab, i)
        e = c['rs'].union_expert(tab, c['S'], u)
        first = mplan.tile_pick0(c['rs'].slot_first(tab, u), i, c['rs'].tile_first(tab, u))
        count = fx.min(fx.Int32(16), c['rs'].slot_first(tab, u + 1) - first)
        if const_expr(_SKIP != 'ug'):
            ug_slice(c, e, first, count, s)
        if j == bid:
            stamp(c, 6)
        # MID rows are this CTA's own stores: drain them before any wave
        # reads them back (gfx1250's store counter is separate)
        llvm.inline_asm(None, [], "s_wait_storecnt 0", "~{memory}", has_side_effects=True)
        gpu.barrier()
        if const_expr(_SKIP != 'down'):
            down_slice(c, e, first, count, s)
        gpu.barrier()
        if j == bid:
            stamp(c, 7)
        j = (total > BLOCKS).select(take_ug_task(c) + BLOCKS, total)


@traced
def slice_contributions(c, tasks):
    """Each pick's BF16 routed contribution to the owned columns from its
    slices (the original even/odd K chains), as ``matrix_tile`` stores it."""
    per_task = c['rs'].picks * 8
    for begin in range_constexpr(0, len(tasks) * per_task, THREADS):
        idx = c['tid'] + begin
        if idx < len(tasks) * per_task:
            j, rem = idx // per_task, idx % per_task
            pick, rp = rem // 8, rem % 8
            task = tasks[0]
            if const_expr(len(tasks) == 2):
                task = (j == 0).select(tasks[0], tasks[1])
            col = task * ROWS + 2 * rp
            parts = [fx.Vector(bo.buffer_load(rsrc(c['partial']), (pick * K_SLICES + s) * HIDDEN + col,
                                              vec_width=2, dtype=T.f32, cache_modifier=CM_DEV))
                     for s in range(K_SLICES)]
            v = [(parts[0][q] + parts[2][q] + parts[4][q]) + (parts[1][q] + parts[3][q]) for q in range(2)]
            w = c['rs'].route_w(c['tab'], c['S'], pick // c['rs'].topk, pick % c['rs'].topk)
            contrib_store(c, j, pick * 8 + rp, bf16_pair(v[0] * w, v[1] * w).bitcast(fx.Int32))
    gpu.barrier()



@traced
def stamp(c, slot):
    if const_expr(STAMPS):
        if c['tid'] == 0:
            v = memrealtime(True)
            layer = scalar_word(fx.ptr_load(c['late_meta'] + 219)) - 1
            at = ((layer % 64) * BLOCKS + c['bid']) * STAMP_SLOTS + slot
            stamps = c['partial'] + fx.Int64(partial_words(c['rs']) * 4)
            bo.buffer_store(fx.Vector.from_elements([fx.Int32(v), fx.Int32(v >> 32)], fx.Int32),
                            rsrc(stamps), at * 2)


@traced
def run_slice_moe(root, key, bid):
    """K2's routed UG and down as slice units, then the original finish."""
    assert key.tokens == 48 and key.ep == 1 and not key.tok_par and key.ug_groups == 2
    c = down_context(root, key)
    # down_context's args are the down weights; the UG ones too (``ug_context``)
    c['args'].update(w13=late_pointer(root, 240), w13_s=late_pointer(root, 242))
    c.update(native_x=root['native_x'], native_s=root['native_s'], native_ring=root['native_ring'],
             sxl=root['native_x'], sxsl=root['native_s'], sdl=root['native_sdl'],
             contrib=root['column_contrib'], partial=root['slice_partial'])
    c['wave'] = uniform(c['tid'] >> 5)
    c['bid'] = bid
    load_route(c, c['tab'])
    stamp(c, 0)
    slice_units(c, bid)
    stamp(c, 1)
    done = sreg(late_pointer(root, 24), root['scratch_layout']['ugc'][0], 'ugc')
    publish(c['put'], done, bid, 1, c['tid'] == 0)
    task = first_task(bid, DOWN0)
    arrival = take_down_rank(c)
    # The owners' shared down overlaps the other CTAs' units.
    if arrival < DOWN_TASKS - BLOCKS:
        down_shared(c, [task, BLOCKS + arrival])
        stamp(c, 2)
        c['poll']([(done, fx.min(c['lane'] + 32 * h + 64 * q, fx.Int32(BLOCKS - 1)), 1)
                   for q in range(BLOCKS // 64) for h in range(2)])
        gpu.barrier()
        stamp(c, 3)
        slice_contributions(c, [task, BLOCKS + arrival])
        stamp(c, 4)
        bulk_finish(c, [task, BLOCKS + arrival])
        stamp(c, 5)
    else:
        down_shared(c, [task])
        stamp(c, 2)
        c['poll']([(done, fx.min(c['lane'] + 32 * h + 64 * q, fx.Int32(BLOCKS - 1)), 1)
                   for q in range(BLOCKS // 64) for h in range(2)])
        gpu.barrier()
        stamp(c, 3)
        slice_contributions(c, [task])
        stamp(c, 4)
        bulk_finish(c, [task])
        stamp(c, 5)
    flag = sreg(late_pointer(root, 24), root['scratch_layout']['dnf'][0], 'dnf')
    publish(c['put'], flag, bid, 1, c['tid'] == 0)
    for t in range(first_task(fresh(bid), RED0), RED_TASKS, BLOCKS):
        stage_reduce(c, t)
