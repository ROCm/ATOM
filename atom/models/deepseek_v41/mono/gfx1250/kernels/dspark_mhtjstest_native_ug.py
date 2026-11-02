# SPDX-License-Identifier: MIT
"""M16 N64 UG, KS2, native fragments through activation/MXFP8 publication."""
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_ug import *
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_native_support import jobs
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_native_support import prime
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_native_support import pipeline
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_native_support import RING_WORDS
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_native_support import tensor
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_native_support import ready
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_native_support import retire_reads


@traced
def gather(c, first, count):
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
    # Shared tensorcnt also covers all six primed B/SB jobs. This may
    # sacrifice overlap; full K2 directly measures that cost.
    ready(0)
    gpu.barrier()


@traced
def epilogue(c, part0, first, count, acc):
    lane, wave = c['lane'], c['wave']
    p, bb, ks = wave // 4, wave % 2, wave // 2 % 2
    group = p * 2 + bb
    at = group * 512 + lane * 8
    # The K1 half alone is written, in native lane/register order. K0 stays
    # in registers; no old logical-wave64 RED restoration or all-partial read.
    if ks == 1:
        for tile in range_constexpr(2):
            fx.ptr_store(acc[tile], c['red'] + at + tile * 256)
    gpu.barrier()
    vals = fx.Vector.filled(8, 0., fx.Float32)
    if ks == 0:
        partial = [fx.Vector(fx.ptr_load(c['red'] + at + tile * 256,
            result_type=fx.Vector.make_type(8, fx.Float32))) for tile in range(2)]
        gate, up = acc[0] + partial[0], acc[1] + partial[1]
        vv=[]
        for j in range_constexpr(8):
            gc = fx.min(gate[j], fx.Float32(LIMIT))
            uc = fx.max(fx.min(up[j], fx.Float32(LIMIT)), fx.Float32(-LIMIT))
            vv.append(bf16_round(gc * hw_rcp(fx.Float32(1.) + hw_exp2(gc * fx.Float32(-LOG2E))) * uc))
        vals = fx.Vector.from_elements(vv, fx.Float32)
        amax = fx.Float32(AMAX_FLOOR)
        for v in vv:amax=fx.max(amax, abs(v))
        amax=fx.max(amax, amax.shuffle_xor(16, 32))
        if lane < 16:fx.ptr_store(amax, c['red'] + 2048 + group * 16 + lane)
    gpu.barrier()
    if ks == 0:
        row, part = lane % 16, part0 + p
        amax = fx.max(fx.ptr_load(c['red'] + 2048 + p * 32 + row),
                      fx.ptr_load(c['red'] + 2048 + p * 32 + 16 + row))
        code = code_ceil(amax * fx.Float32(1. / FP8_MAX))
        inv = rcp_pow2(code)
        if row < count:
            for j in range_constexpr(0, 8, 4):
                q = [clamp_fp8(vals[j+d] * inv) for d in range(4)]
                word = fp8_pack4(*q)
                offset = bb * 4 + (lane // 16) * 2 + j // 4
                st_dev(word, c['mid'], (first + row) * c['d'].mid_words + part * 8 + offset)
            if (bb == 0) & (lane < 16):
                st_dev(code, c['mids'], mid_code_index(c['d'], first + row, part))
    gpu.barrier()


@traced
def run_ug(root, key, bid):
    assert key.tokens == 48 and key.ug_groups == 2 and not key.tok_par
    c = ug_context(root, key)
    c.update(native_x=root['native_x'], native_s=root['native_s'], native_ring=root['native_ring'])
    c['wave'] = uniform(c['tid'] >> 5)
    load_route(c, c['tab'])
    tiles=c['rs'].tile_first(c['tab'], c['rs'].n_union(c['tab'], c['S']))
    total=mplan.ug_tasks(tiles, c['rs'].ug_parts)
    task=first_task(bid, c['rs'].ug0)
    if task < total:
        # CT1-M3-A1 (from tp4-mono/ct1-m3-ugfront): the first task's weights are primed
        # before the loop; every later task's by the previous iteration, before its epilogue.
        e0, part00, _, _ = ug_tile_task(c, task)
        _, tile0, ks0, _ = ug_wave_pair(c, part00)
        prime(c, jobs(c, c['args']['w13'], c['args']['w13_s'], local_expert(c, e0), tile0,
                      HIDDEN, 2 * c['d'].inter, HIDDEN // 32), ks0)
    while task < total:
        e, part0, first, count = ug_tile_task(c, task)
        _, tile, ks, _ = ug_wave_pair(c, part0)
        job = jobs(c, c['args']['w13'], c['args']['w13_s'], local_expert(c, e), tile,
                   HIDDEN, 2 * c['d'].inter, HIDDEN // 32)
        gather(c, first, count)
        acc=pipeline(c, job, HIDDEN // 256, fx.min(c['lane'] % 16, count - 1), HIDDEN, ks)
        # take the next task now and start its weight stream into the retired ring
        # before this task's epilogue
        nxt=(total > BLOCKS).select(take_ug_task(c) + BLOCKS, total)
        if nxt < total:
            e1, part01, _, _ = ug_tile_task(c, nxt)
            _, tile1, ks1, _ = ug_wave_pair(c, part01)
            retire_reads()
            prime(c, jobs(c, c['args']['w13'], c['args']['w13_s'], local_expert(c, e1), tile1,
                          HIDDEN, 2 * c['d'].inter, HIDDEN // 32), ks1)
        epilogue(c, part0, first, count, acc)
        publish(c['put'], c['ugf'], task, 1, c['tid'] == 0)
        task=nxt
    return c
