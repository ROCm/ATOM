# SPDX-License-Identifier: MIT
"""gfx950 column owners; gfx1250 full-expert TDM and native operands.

An owner retains its one/two N16 routed contributions in LDS, computes the
shared expert before routed waits, and combines/publishes its own columns.
Wave ownership follows the branch contract; native two-bank operands remain. B/SB survive all M16 tiles of that expert;
MID/SA are read directly into native register operands after that tile's UGF.
There is no global DCON handoff, column counter, or foreign finish owner.
"""
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_down import *
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_native_support import tensor
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_native_support import ready
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_native_support import retire_reads
from atom.models.deepseek_v41.mono.gfx1250.kernels.reference_sync import publish_plain as publish


def weight_words(inter):
    return WAVES * 2 * (inter // 128) * (256 + 16)


@traced
def issue_expert(c, tasks, expert, slot, slots):
    """Each wave's complete K weights/scales, issued once per expert."""
    d, steps = c['d'], c['d'].inter // 128
    base = c['column_weights'] + c['wave'] * (2 * steps * (256 + 16))
    for j in range_constexpr(len(tasks)):
        task = tasks[j]
        offset = fx.Int64(expert) * HIDDEN * d.inter // 8 + fx.Int64(task) * d.inter * 2
        pointer = fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Global, 16),
                             c['args']['w2'] + offset * 4)
        src = tensor(pointer, (steps, 256), (256, 1))
        atom = fx.rocdl.make_tdm_atom(src, [steps, 256], strides=[256, None],
                                    num_warps=1, cache_modifier=CM_NT)
        fx.copy(atom, src, tensor(base + (j * slots + slot) * steps * 256,
                                 (steps, 256), (256, 1)))
        # Native n32k4 is [K128][N16 half][16 scale dwords]. Select one
        # half with a strided source, then pack only that N16's scales in LDS.
        offset = ((fx.Int64(expert) * HIDDEN // 32 + fx.Int64(task // 2))
                  * (d.down_scale_cols * 8) + (task % 2) * 16)
        pointer = fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Global, 16),
                             c['args']['w2_s'] + offset * 4)
        src = tensor(pointer, (steps, 16), (32, 1))
        atom = fx.rocdl.make_tdm_atom(src, [steps, 16], strides=[32, None],
                                    num_warps=1, cache_modifier=CM_NT)
        fx.copy(atom, src, tensor(base + 2 * steps * 256 + (j * slots + slot) * steps * 16,
                                 (steps, 16), (16, 1)))


def operands(n):
    return dict(x=fx.make_rmem_tensor(16, fx.Int32),
                w=[fx.make_rmem_tensor(8, fx.Int32) for _ in range(n)],
                sx=fx.make_rmem_tensor(1, fx.Int32),
                sw=fx.make_rmem_tensor(n, fx.Int32))


@traced
def load_operands(c, tasks, first, count, step, out, slot, slots):
    lane, d = c['lane'], c['d']
    row = first + fx.min(lane % 16, count - 1)
    chunks = [fx.Vector(bo.buffer_load(rsrc(c['mid'].value),
                  row * d.mid_words + step * 32 + lane // 16 * 4 + h * 8,
                  vec_width=4, dtype=T.i32, cache_modifier=CM_DEV)) for h in range(4)]
    out['x'].store(fx.Vector.from_elements(
        [chunks[h][q] for h in range(4) for q in range(4)], fx.Int32))
    sx = fx.Int32(0)
    for group in range_constexpr(4):
        code = fx.Int32(bo.buffer_load(rsrc(c['mids'].value),
            mid_code_index(d, row, step * 4 + group), vec_width=1,
            dtype=T.i32, cache_modifier=CM_DEV))
        sx = sx | (code << (8 * group))
    out['sx'].store(fx.Vector.from_elements([sx], fx.Int32))
    steps = d.inter // 128
    base = c['column_weights'] + c['wave'] * (2 * steps * (256 + 16))
    at = lane // 16 * 64 + lane % 16 * 4
    scales = []
    for j in range_constexpr(len(tasks)):
        halves = [fx.Vector(fx.ptr_load(base + (j * slots + slot) * steps * 256 + step * 256
                     + at + h * 128, result_type=fx.Vector.make_type(4, fx.Int32)))
                  for h in range(2)]
        out['w'][j].store(halves[0].shuffle(halves[1], list(range(8))))
        scales.append(fx.ptr_load(base + 2 * steps * 256 + (j * slots + slot) * steps * 16
                                  + step * 16 + lane % 16))
    out['sw'].store(fx.Vector.from_elements(scales, fx.Int32))


@traced
def matrix_tile(c, tasks, first, count, slot, slots):
    """Original two interleaved K chains and current/next register operands."""
    atom = fx.make_mma_atom(fx.rocdl.WMMAScale(16, 16, 128,
        fx.Float4E2M1FN, fx.Float8E4M3FN, fx.Float32))
    acc = [[fx.make_rmem_tensor(8, fx.Float32) for _ in range(2)] for _ in tasks]
    for pair in acc:
        for a in pair:
            a.store(fx.full(8, 0., fx.Float32))
    cur, nxt = operands(len(tasks)), operands(len(tasks))
    load_operands(c, tasks, first, count, fx.Int32(0), cur, slot, slots)
    for step in range_constexpr(c['d'].inter // 128 - 1):
        load_operands(c, tasks, first, count, step + 1, nxt, slot, slots)
        sx, sw = cur['sx'].load()[0], cur['sw'].load()
        for j in range_constexpr(len(tasks)):
            fx.gemm(atom, acc[j][step % 2], cur['w'][j], cur['x'], acc[j][step % 2],
                    scale_a=sw[j], scale_b=sx)
        for name in ('x', 'sx', 'sw'):
            cur[name].store(nxt[name].load())
        for j in range_constexpr(len(tasks)):
            cur['w'][j].store(nxt['w'][j].load())
    sx, sw = cur['sx'].load()[0], cur['sw'].load()
    for j in range_constexpr(len(tasks)):
        fx.gemm(atom, acc[j][(c['d'].inter // 128 - 1) % 2], cur['w'][j], cur['x'], acc[j][(c['d'].inter // 128 - 1) % 2],
                scale_a=sw[j], scale_b=sx)
    lane, rs = c['lane'], c['rs']
    row = lane % 16
    pick = rs.pick_at(c['tab'], first + fx.min(row, count - 1))
    w = rs.route_w(c['tab'], c['S'], pick // rs.topk, pick % rs.topk)
    if row < count:
        for j in range_constexpr(len(tasks)):
            value = acc[j][0].load() + acc[j][1].load()
            for q in range_constexpr(0, 8, 2):
                r = 8 * (lane // 16) + q
                word = bf16_pair(value[q] * w, value[q + 1] * w).bitcast(fx.Int32)
                contrib_store(c, j, pick * 8 + r // 2, word)


@traced
def routed_columns(c, tasks):
    # gfx950 stage_down: task count times expert slots is always two.
    tab = c['tab']
    nu = c['rs'].n_union(tab, c['S'])
    slots = 2 // len(tasks)
    for rnd in range(0, (nu + WAVES * slots - 1) // (WAVES * slots)):
        us = [c['wave'] + WAVES * (slots * rnd + i) for i in range(slots)]
        ucs = [fx.min(u, nu - 1) for u in us]
        # Retire old operands once before either bank can be overwritten.
        retire_reads()
        for i in range_constexpr(slots):
            if us[i] < nu:
                e = local_expert(c, c['rs'].union_expert(tab, c['S'], us[i]))
                issue_expert(c, tasks, e, i, slots)
        # Complete all current-round slot/tile/part producers before MID.
        # Fold gfx950's wave64 flags across both logical halves of wave32.
        c['poll']([(c['ugf'], f, 1) for f in down_ug_flags(c, ucs)])
        ready(0)
        for i in range_constexpr(slots):
            if us[i] < nu:
                first = c['rs'].slot_first(tab, us[i])
                count = c['rs'].slot_first(tab, us[i] + 1) - first
                for offset in range(0, count, 16):
                    matrix_tile(c, tasks, first + offset,
                                fx.min(fx.Int32(16), count - offset), i, slots)


@traced
def owner_columns(c, tasks):
    # gfx950: independent shared work hides routed input waits. Only SDL
    # survives shared's temporaries; weight/contribution arenas start later.
    down_shared(c, tasks)
    routed_columns(c, tasks)
    gpu.barrier()
    # Original top-k FP32 sum of BF16 routed results, then shared, then BF16
    # rounding; original tagged full-row TP2 publication and rank reduction.
    bulk_finish(c, tasks)


@traced
def run_down(root, key, bid):
    c = down_context(root, key)
    assert key.tokens == 48 and key.ep == 1 and not key.tok_par
    c.update(sxl=root['native_x'], sxsl=root['native_s'],
             sdl=root['native_sdl'], contrib=root['column_contrib'],
             column_weights=root['column_weights'])
    task = first_task(bid, DOWN0)
    arrival = take_down_rank(c)
    if arrival < DOWN_TASKS - BLOCKS:
        owner_columns(c, [task, BLOCKS + arrival])
    else:
        owner_columns(c, [task])
    flag = sreg(late_pointer(root, 24), root['scratch_layout']['dnf'][0], 'dnf')
    publish(c['put'], flag, bid, 1, c['tid'] == 0)
    for t in range(first_task(fresh(bid), RED0), RED_TASKS, BLOCKS):
        stage_reduce(c, t)


@traced
def bulk_finish(c, tasks):
    """All 48 rows/owned columns together, original top-k order and TP ABI.

    Source is the owner's LDS contribution, already complete at the preceding
    CTA barrier. There is no per-M16 pair arena or intermediate handoff. Issue
    all routed words together, sum two BF16 values, then directly publish the
    same tagged BF16 pair used by the established P7 finish.
    """
    per_task = c['S'] * 8
    for begin in range_constexpr(0, len(tasks) * per_task, THREADS):
        idx = c['tid'] + begin
        if idx < len(tasks) * per_task:
            j, rem = idx // per_task, idx % per_task
            t, rp = rem // 8, rem % 8
            task = tasks[0]
            if const_expr(len(tasks) == 2):
                task = (j == 0).select(tasks[0], tasks[1])
            words = [fx.ptr_load(c['contrib'] + j * c['rs'].picks * 8
                       + (t * c['rs'].topk + k) * 8 + rp)
                     for k in range(c['rs'].topk)]
            r0, r1 = fx.Float32(0.), fx.Float32(0.)
            for k in range_constexpr(c['rs'].topk):
                r0 = r0 + bf_lo(words[k])
                r1 = r1 + bf_hi(words[k])
            at = (j * c['S'] + t) * ROWS + 2 * rp
            sh0 = fx.ptr_load(c['sdl'] + at).bitcast(fx.Float32)
            sh1 = fx.ptr_load(c['sdl'] + at + 1).bitcast(fx.Float32)
            v0, v1 = bf16_round(r0 + sh0), bf16_round(r1 + sh1)
            col = task * ROWS + 2 * rp
            for p in range_constexpr(c['d'].ranks):
                dst = preg(c['peer_addr'](p), c['peer']['ffn'], 'ffn')
                c['put_bf'](dst, 2 * region_pair(c['rank'], t, c['S'], col), [v0, v1])
    gpu.barrier()
