# SPDX-License-Identifier: MIT
"""gfx950 column owners; gfx1250 full-expert TDM and native operands.

An owner retains its one/two N16 routed contributions in LDS, computes the
shared expert before routed waits, and combines/publishes its own columns.
A wave takes two experts a round (one-task owners) or one (two-task owners).
B/SB survive all M16 tiles of that expert; MID/SA are read into native
register operands after its UGF, every K step at once. Rounds are pipelined:
the next round's weights (second bank) and UGF loads are in flight while a
round computes.
There is no global DCON handoff, column counter, or foreign finish owner.
"""
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_down import *
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_native_support import tensor
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_native_support import ready
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_native_support import retire_reads
from atom.models.deepseek_v41.mono.gfx1250.kernels.reference_sync import publish_plain as publish
from atom.mono.device.sync import _noted, _spin, preg
from atom.mono.device.ops import bf2_f32, bf16_pair, rsrc


# A wave's weight arena: two banks of (owned tasks x expert slots) == 2 B/SB
# regions each. Rounds alternate banks so the next round's weights stream in
# while this round computes.
REGIONS = 4


def _slots(tasks):
    return 2 // len(tasks)  # experts a wave takes per round


def weight_words(inter):
    return WAVES * REGIONS * (inter // 128) * (256 + 16)


def _arena(c):
    return c['column_weights'] + c['wave'] * (REGIONS * (c['d'].inter // 128) * (256 + 16))


def _region(tasks, bank, j, slot):
    return bank * 2 + j * _slots(tasks) + slot


@traced
def issue_round(c, tasks, experts, bank):
    """Each wave's complete K weights/scales of a round's experts."""
    d, steps = c['d'], c['d'].inter // 128
    base = _arena(c)
    for i in range_constexpr(_slots(tasks)):
        for j in range_constexpr(len(tasks)):
            region = _region(tasks, bank, j, i)
            task = tasks[j]
            offset = fx.Int64(experts[i]) * HIDDEN * d.inter // 8 + fx.Int64(task) * d.inter * 2
            pointer = fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Global, 16),
                                 c['args']['w2'] + offset * 4)
            src = tensor(pointer, (steps, 256), (256, 1))
            atom = fx.rocdl.make_tdm_atom(src, [steps, 256], strides=[256, None],
                                        num_warps=1, cache_modifier=CM_NT)
            fx.copy(atom, src, tensor(base + region * steps * 256, (steps, 256), (256, 1)))
            # Native n32k4 is [K128][N16 half][16 scale dwords]. Select one
            # half with a strided source, then pack only that N16's scales in LDS.
            offset = ((fx.Int64(experts[i]) * HIDDEN // 32 + fx.Int64(task // 2))
                      * (d.down_scale_cols * 8) + (task % 2) * 16)
            pointer = fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Global, 16),
                                 c['args']['w2_s'] + offset * 4)
            src = tensor(pointer, (steps, 16), (32, 1))
            atom = fx.rocdl.make_tdm_atom(src, [steps, 16], strides=[32, None],
                                        num_warps=1, cache_modifier=CM_NT)
            fx.copy(atom, src, tensor(base + REGIONS * steps * 256 + region * steps * 16,
                                     (steps, 16), (16, 1)))


def operands(n):
    return dict(x=fx.make_rmem_tensor(16, fx.Int32),
                w=[fx.make_rmem_tensor(8, fx.Int32) for _ in range(n)],
                sx=fx.make_rmem_tensor(1, fx.Int32),
                sw=fx.make_rmem_tensor(n, fx.Int32))


@traced
def load_tile_scales(c, first, count):
    steps = c['d'].inter // 128
    row = first + fx.min(c['lane'] % 16, count - 1)
    packed = fx.make_rmem_tensor(steps, fx.Int32)
    packed.store(fx.full(steps, 0, fx.Int32))
    for group in range_constexpr(4):
        codes = []
        for begin in range_constexpr(0, steps, 4):
            width = min(4, steps - begin)
            values = bo.buffer_load(rsrc(c['mids'].value),
                row * (4 * steps) + group * steps + begin,
                vec_width=width, dtype=T.i32, cache_modifier=CM_DEV)
            if const_expr(width == 1):
                codes.append(fx.Int32(values))
            else:
                vec = fx.Vector(values)
                for j in range_constexpr(width):
                    codes.append(vec[j])
        old = packed.load()
        packed.store(fx.Vector.from_elements([
            old[i] | (codes[i] << (8 * group)) for i in range_constexpr(steps)], fx.Int32))
    return packed


def flag_loader(c, ucs):
    """The round's UGF loads and stale-tag test, split so the loads can be in
    flight while the previous round computes (same loads as ``Mailbox.poll``)."""
    mb = c['poll'].__self__
    specs = [(c['ugf'], f, 1) for f in down_ug_flags(c, ucs)]
    cms = [mb.scope_cm[a.space] for a, _, _ in specs]
    bases = [_noted(a, 'poll') for a, _, _ in specs]
    tag = mb.tag
    nw = sum(2 * n for _, _, n in specs)

    def load_all():
        words = []
        for b, cm, (_, i, n) in zip(bases, cms, specs):
            w = fx.Vector(bo.buffer_load(rsrc(b), fx.Int32(i) * 2, vec_width=2 * n,
                                         dtype=T.i32, cache_modifier=cm))
            words += [w[e] for e in range(2 * n)]
        return fx.Vector.from_elements(words, fx.Int32)

    def pending(v):
        bad = v[1] != tag
        for e in range_constexpr(3, nw, 2):
            bad = bad | (v[e] != tag)
        return bad
    return load_all, pending


@traced
def load_x(c, first, count, step, out):
    lane, d = c['lane'], c['d']
    row = first + fx.min(lane % 16, count - 1)
    chunks = [fx.Vector(bo.buffer_load(rsrc(c['mid'].value),
                  row * d.mid_words + step * 32 + lane // 16 * 4 + h * 8,
                  vec_width=4, dtype=T.i32, cache_modifier=CM_DEV)) for h in range(4)]
    out.store(fx.Vector.from_elements(
        [chunks[h][q] for h in range(4) for q in range(4)], fx.Int32))


@traced
def load_weights(c, tasks, step, out, bank, slot):
    lane, steps = c['lane'], c['d'].inter // 128
    base = _arena(c)
    at = lane // 16 * 64 + lane % 16 * 4
    scales = []
    for j in range_constexpr(len(tasks)):
        region = _region(tasks, bank, j, slot)
        halves = [fx.Vector(fx.ptr_load(base + region * steps * 256 + step * 256
                     + at + h * 128, result_type=fx.Vector.make_type(4, fx.Int32)))
                  for h in range(2)]
        out['w'][j].store(halves[0].shuffle(halves[1], list(range(8))))
        scales.append(fx.ptr_load(base + REGIONS * steps * 256 + region * steps * 16
                                  + step * 16 + lane % 16))
    out['sw'].store(fx.Vector.from_elements(scales, fx.Int32))


def preload(c, first, count):
    """A tile's scale codes and every K step's MID, issued together (after its UGF)."""
    steps = c['d'].inter // 128
    codes = load_tile_scales(c, first, count)
    xs = [fx.make_rmem_tensor(16, fx.Int32) for _ in range(steps)]
    for step in range_constexpr(steps):
        load_x(c, first, count, step, xs[step])
    return codes, xs


@traced
def compute_tile(c, tasks, first, count, bank, slot, codes, xs):
    """Original two interleaved K chains; route weight, BF16 contribution."""
    steps = c['d'].inter // 128
    atom = fx.make_mma_atom(fx.rocdl.WMMAScale(16, 16, 128,
        fx.Float4E2M1FN, fx.Float8E4M3FN, fx.Float32))
    acc = [[fx.make_rmem_tensor(8, fx.Float32) for _ in range(2)] for _ in tasks]
    for pair in acc:
        for a in pair:
            a.store(fx.full(8, 0., fx.Float32))
    code = codes.load()
    cur, nxt = operands(len(tasks)), operands(len(tasks))
    load_weights(c, tasks, 0, cur, bank, slot)
    for step in range_constexpr(steps):
        if const_expr(step + 1 < steps):
            load_weights(c, tasks, step + 1, nxt, bank, slot)
        sw = cur['sw'].load()
        for j in range_constexpr(len(tasks)):
            fx.gemm(atom, acc[j][step % 2], cur['w'][j], xs[step], acc[j][step % 2],
                    scale_a=sw[j], scale_b=code[step])
        if const_expr(step + 1 < steps):
            cur['sw'].store(nxt['sw'].load())
            for j in range_constexpr(len(tasks)):
                cur['w'][j].store(nxt['w'][j].load())
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
def extra_tile(c, tasks, first, count, bank, slot):
    codes, xs = preload(c, first, count)
    compute_tile(c, tasks, first, count, bank, slot, codes, xs)


def _units(c, r, slots):
    return [c['wave'] + WAVES * (slots * r + i) for i in range(slots)]


def _experts(c, nu, r, slots):
    return [local_expert(c, c['rs'].union_expert(c['tab'], c['S'], fx.min(u, nu - 1)))
            for u in _units(c, r, slots)]


def _clamped(nu, units):
    return [fx.min(u, nu - 1) for u in units]


@traced
def routed_round(c, tasks, nu, nr, rnd):
    slots = _slots(tasks)
    bank = rnd % 2
    us = _units(c, rnd, slots)
    ucs = _clamped(nu, us)
    firsts = [c['rs'].slot_first(c['tab'], u) for u in ucs]
    counts = [c['rs'].slot_first(c['tab'], u + 1) - f for u, f in zip(ucs, firsts)]
    pre = [preload(c, firsts[i], fx.min(fx.Int32(16), counts[i])) for i in range(slots)]
    nxt = fx.min(rnd + 1, nr - 1)
    retire_reads()
    issue_round(c, tasks, _experts(c, nu, nxt, slots), 1 - bank)
    load_all, pending = flag_loader(c, _clamped(nu, _units(c, nxt, slots)))
    early = load_all()
    ready(2 * len(tasks) * slots)
    for i in range_constexpr(slots):
        if us[i] < nu:
            compute_tile(c, tasks, firsts[i], fx.min(fx.Int32(16), counts[i]), bank, i,
                         pre[i][0], pre[i][1])
            for offset in range(16, counts[i], 16):
                extra_tile(c, tasks, firsts[i] + offset,
                           fx.min(fx.Int32(16), counts[i] - offset), bank, i)
    if pending(early):
        _spin(load_all, pending)


@traced
def routed_columns(c, tasks):
    slots = _slots(tasks)
    nu = c['rs'].n_union(c['tab'], c['S'])
    nr = (nu + WAVES * slots - 1) // (WAVES * slots)
    retire_reads()
    issue_round(c, tasks, _experts(c, nu, fx.Int32(0), slots), 0)
    c['poll']([(c['ugf'], f, 1) for f in down_ug_flags(c, _clamped(nu, _units(c, fx.Int32(0), slots)))])
    for rnd in range(0, nr):
        routed_round(c, tasks, nu, nr, rnd)
    ready(0)


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
        stage_reduce_packed(c, t)


# TP2 partials travel as 16-byte groups (three bf16-pair words and one tag):
# a 16-column owner task's 48 tokens x 3 groups are one contiguous 2304 B,
# 256 B aligned run per rank (W28), instead of 64 B tagged-pair fragments.
FFN_GROUPS = 3
FFN_TASKS = 320


def ffn_group_word(src, task16, t, s, g):
    return ((fx.Int32(src) * FFN_TASKS + task16) * s + t) * (FFN_GROUPS * 4) + g * 4


@traced
def bulk_finish(c, tasks):
    mb = c['put'].__self__
    per_task = c['S'] * FFN_GROUPS
    for begin in range_constexpr(0, len(tasks) * per_task, THREADS):
        idx = c['tid'] + begin
        if idx < len(tasks) * per_task:
            j, rem = idx // per_task, idx % per_task
            t, g = rem // FFN_GROUPS, rem % FFN_GROUPS
            task = tasks[0]
            if const_expr(len(tasks) == 2):
                task = (j == 0).select(tasks[0], tasks[1])
            words = []
            for k in range_constexpr(3):
                rp = fx.min(g * 3 + k, 7)
                ws = [fx.ptr_load(c['contrib'] + j * c['rs'].picks * 8
                       + (t * c['rs'].topk + q) * 8 + rp) for q in range(c['rs'].topk)]
                r0, r1 = fx.Float32(0.), fx.Float32(0.)
                for q in range_constexpr(c['rs'].topk):
                    r0 = r0 + bf_lo(ws[q])
                    r1 = r1 + bf_hi(ws[q])
                at = (j * c['S'] + t) * ROWS + 2 * rp
                sh0 = fx.ptr_load(c['sdl'] + at).bitcast(fx.Float32)
                sh1 = fx.ptr_load(c['sdl'] + at + 1).bitcast(fx.Float32)
                v0, v1 = bf16_round(r0 + sh0), bf16_round(r1 + sh1)
                words.append(bf16_pair(v0, v1).bitcast(fx.Int32))
            words.append(fx.Int32(mb.tag))
            vec = fx.Vector.from_elements(words, fx.Int32)
            index = ffn_group_word(c['rank'], task, t, c['S'], g)
            for p in range_constexpr(c['d'].ranks):
                dst = preg(c['peer_addr'](p), c['peer']['ffn'], 'ffn')
                bo.buffer_store(vec, rsrc(_noted(dst, 'put')), index,
                                cache_modifier=mb.scope_cm[dst.space])
    gpu.barrier()


@traced
def reduce_pair_packed(c, task, idx):
    s, tp = c['S'], c['d'].ranks
    a = c['args']
    if idx < s * RED_COLS // 2:
        t = idx // (RED_COLS // 2)
        col = task * RED_COLS + 2 * (idx % (RED_COLS // 2))
        own = preg(c['sym'], c['peer']['ffn'], 'ffn')
        mb = c['poll'].__self__
        cm = mb.scope_cm[own.space]
        base = _noted(own, 'poll')
        task16 = col // 16
        rp = (col % 16) // 2
        g = fx.min(rp // 3, FFN_GROUPS - 1)
        k = rp - g * 3
        tag = fx.Int32(mb.tag)

        def load_all():
            words = []
            for src in range_constexpr(tp):
                w = fx.Vector(bo.buffer_load(rsrc(base), ffn_group_word(src, task16, t, s, g),
                                             vec_width=4, dtype=T.i32, cache_modifier=cm))
                words += [w[e] for e in range(4)]
            return fx.Vector.from_elements(words, fx.Int32)

        def pending(v):
            bad = v[3] != tag
            for src in range_constexpr(1, tp):
                bad = bad | (v[4 * src + 3] != tag)
            return bad

        v = _spin(load_all, pending)
        acc0, acc1 = fx.Float32(0.), fx.Float32(0.)
        for src in range_constexpr(tp):
            w = (k == 0).select(v[4 * src], (k == 1).select(v[4 * src + 1], v[4 * src + 2]))
            lo, hi = bf2_f32(fx.Int32(w))
            if const_expr(src == 0):
                acc0, acc1 = lo, hi
            else:
                acc0 = acc0 + lo
                acc1 = acc1 + hi
        bo.buffer_store(acc0.to(fx.BFloat16), rsrc(a['out']), t * HIDDEN + col)
        bo.buffer_store(acc1.to(fx.BFloat16), rsrc(a['out']), t * HIDDEN + col + 1)


@traced
def stage_reduce_packed(c, task):
    """``stage_reduce`` over the packed TP2 layout written by ``bulk_finish``."""
    for pas in range_constexpr(0, c["S"] * RED_COLS // 2, THREADS):
        reduce_pair_packed(c, task, pas + c["tid"])
