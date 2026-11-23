# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K3's distributed radix applied directly to V4.1's paged FP4 scores.

All segments stay resident, combine their histograms and exchange four
8-bit digits. Contiguous rank windows and segment scans preserve V4.1's
ascending global index tie rule, including its raw FP32 bit ordering.
"""

import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

from atom.mono.device.ops import (
    block_excl_scan,
    ld_dev,
    read_lane_i32,
    st_dev,
    traced,
    uniform,
    wave_incl_scan,
    wave_max,
    wave_sum,
)
from atom.mono.device.select import u32_gt
from atom.mono.device.sync import preg, publish
from atom.mono.dsa_indexer import select as k3
from atom.mono.dsa_indexer.paged import plan as ip
from atom.mono.dsa_indexer.paged import tp_plan as tp
from atom.mono.dsa_indexer.paged.score import _wait_chunks
from atom.mono.dsa_indexer.paged.topk import order_key
from atom.mono.dsa_indexer.paged.tp_select import _publish_count
from atom.mono.dsa_indexer.paged.tp_work import row_work
from atom.mono.dsa_indexer.selection_plan import DEFAULT_RADIX
from atom.mono.plan.execution import BLOCKS, THREADS, first_task

CHUNK = k3.CHUNK
PT = CHUNK // THREADS


def _no_stamp(*args):
    return None


def _marks(live, keys, prefix, whole_bucket):
    greater, equal = [], []
    ng, ne = fx.Int32(0), fx.Int32(0)
    for i in range_constexpr(len(keys)):
        same = live[i] & (keys[i] == prefix)
        gt = live[i] & (u32_gt(keys[i], prefix) | (whole_bucket & same))
        eq = same & ~whole_bucket
        greater.append(gt)
        equal.append(eq)
        ng = ng + gt.select(1, 0)
        ne = ne + eq.select(1, 0)
    return greater, equal, (ng << 16) | ne


def histogram_pairs(rows, world):
    return len(tp.KINDS) * rows * DEFAULT_RADIX.passes * world * DEFAULT_RADIX.bins


@traced
def run_splits(c, kind, base=0):
    """Choose a segment size that keeps every participating CTA resident."""
    lane, rows = c["lane"], c["S"]
    row = fx.min(lane, rows - 1)
    mode, _owner, _first, count = row_work(c, row)
    count = ip.kind_bound(kind, count)
    active = (lane < rows) & (mode == tp.SPLIT)
    segment = fx.Int32(CHUNK)
    while (
        uniform(
            wave_sum(active.select(fx.max(ip.cdiv(count, segment), 1), fx.Int32(0)))
        )
        > BLOCKS
    ):
        segment = segment * 2
    slots = active.select(fx.max(ip.cdiv(count, segment), 1), fx.Int32(0))
    end = wave_incl_scan(slots, lane)
    start = end - slots
    total = uniform(read_lane_i32(end, rows - 1))
    for task in range(first_task(c["bid"], base), total, BLOCKS):
        hit = active & (task >= start) & (task < end)
        selected_row = uniform(wave_max(hit.select(lane, fx.Int32(-1))))
        index = task - uniform(read_lane_i32(start, selected_row))
        stage_segment(c, selected_row, index, segment, kind)
    return base + total


@traced
def _keys(c, row, first, lo, here, kind):
    source, stride = (
        (c["args"]["ilog"], c["args"]["lstride"])
        if kind == "sel"
        else (c["args"]["ibmax"], c["args"]["bstride"])
    )
    live, keys = [], []
    for quad in range_constexpr(PT // 4):
        offset = c["tid"] * PT + quad * 4
        at = fx.min(offset, fx.max(here - 1, 0) & ~3)
        words = fx.Vector(ld_dev(source, row * stride + first + lo + at, 4))
        for element in range_constexpr(4):
            valid = offset + element < here
            live.append(valid)
            keys.append(valid.select(order_key(fx.Int32(words[element])), fx.Int32(0)))
    return live, keys


@traced
def _emit(c, row, first, lo, gt, eq, at, run, ties, kind):
    k = ip.KINDS[kind]
    target = c["ix_pairs"][kind]
    for element in range_constexpr(PT):
        chosen = gt[element] | (eq[element] & (run < ties))
        if chosen:
            ident = first + lo + c["tid"] * PT + element
            st_dev(ident, target, row * k * 2 + at)
        at = at + chosen.select(1, 0)
        run = run + eq[element].select(1, 0)


@traced
def stage_segment(c, row, g, segment, kind):
    code = tp.KINDS.index(kind)
    radix_row = code * c["S"] + row
    _mode, _owner, key_first, raw = row_work(c, row)
    first = key_first if kind == "sel" else key_first // ip.BLOCK_ROWS
    n = ip.kind_bound(kind, raw)
    global_n = ip.kind_bound(kind, c["load_bound"](row))
    k = ip.KINDS[kind]
    nseg = fx.max(ip.cdiv(n, segment), 1)
    lo = g * segment
    here = fx.max(fx.min(n - lo, segment), 0)
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    slot = radix_row * BLOCKS + g
    total = fx.Int32(0)

    if global_n <= k:
        for i in range(tid, here, THREADS):
            st_dev(first + lo + i, c["ix_pairs"][kind], row * k * 2 + lo + i)
        total = n
    else:
        flags_first = (
            key_first + (lo if kind == "sel" else lo * ip.BLOCK_ROWS)
        ) // ip.CHUNK
        flag_rows = here if kind == "sel" else here * ip.BLOCK_ROWS
        _wait_chunks(
            c, row, ip.chunks(c["bound_max"]), flags_first, ip.chunks(flag_rows), 2
        )
        co = {
            **c,
            "radix": DEFAULT_RADIX,
            "hist": c["sel_lds"]["hist"],
            "sel": c["radix_sel"],
            "scan": c["radix_scan"],
            "before": c["radix_before"],
            "rowhist": c["ix_rowhist"],
            "phist": [
                preg(c["peer_addr"](p), c["ix_peer_layout"]["ix_hist"][0], "ix_hist")
                for p in range(c["world"])
            ],
            "phist_own": preg(
                c["peer_addr"](c["rank"]), c["ix_peer_layout"]["ix_hist"][0], "ix_hist"
            ),
            "stamp": _no_stamp,
            "tl_pass": 0,
            "split_q": True,
            "capture_bucket": True,
        }
        if tid == 0:
            fx.ptr_store(fx.Int32(k), co["sel"] + k3.SEL_NEED)
        gpu.barrier()
        live, keys = _keys(c, row, first, lo, here, kind)
        nck = ip.cdiv(here, CHUNK)
        prefix = fx.Int32(0)
        take = fx.Int32(0)
        done = fx.Boolean(False)
        for phase in range_constexpr(DEFAULT_RADIX.passes):
            if ~done:
                k3._zero_hist(co)
                k3._count(co, live, keys, phase, prefix)
                for chunk in range(1, nck):
                    clive, ckeys = _keys(
                        c, row, first, lo + chunk * CHUNK, here - chunk * CHUNK, kind
                    )
                    k3._count(co, clive, ckeys, phase, prefix)
                gpu.barrier()
                if nseg > 1:
                    k3._row_add(co, radix_row, phase, g)
                if g == 0:
                    if nseg > 1:
                        k3._row_counts(co, radix_row, phase, nseg)
                    k3._lead_push(co, radix_row, phase)
                    k3._decide(co, radix_row, phase, nseg, fx.Boolean(True))
                    if tid == 0:
                        phase_at = (radix_row * DEFAULT_RADIX.passes + phase) * 3
                        need = fx.ptr_load(co["sel"] + k3.SEL_NEED)
                        whole = fx.ptr_load(co["sel"] + k3.SEL_BUCKET) == need
                        ties = (
                            fx.ptr_load(co["sel"] + k3.SEL_TAKE)
                            if phase == DEFAULT_RADIX.passes - 1
                            else fx.Int32(0)
                        )
                        state = whole.select(fx.Int32(65536), ties)
                        c["put_words"](
                            c["ix_phase_ready"],
                            phase_at,
                            [fx.ptr_load(co["sel"] + k3.SEL_DIGIT), need],
                        )
                        c["put"](c["ix_phase_ready"], phase_at + 2, state)
                # Only the leader polls peer histograms. The other segments
                # consume three local words instead of repeating those reads.
                phase_at = (radix_row * DEFAULT_RADIX.passes + phase) * 3
                values = c["poll"](
                    [
                        (c["ix_phase_ready"], phase_at, 2),
                        (c["ix_phase_ready"], phase_at + 2, 1),
                    ]
                )
                prefix = (prefix << DEFAULT_RADIX.bits[phase]) | values[0][0]
                take = values[1][0] & 0xFFFF
                done = (values[1][0] & 65536) != 0
                if done:
                    prefix = prefix << DEFAULT_RADIX.shifts[phase]
        gt, eq, packed = _marks(live, keys, prefix, done)
        excl, chunk_total = block_excl_scan(packed, lane, wave, co["scan"])
        seg_gt, seg_eq = (chunk_total >> 16) & 0xFFFF, chunk_total & 0xFFFF
        for chunk in range(1, nck):
            clive, ckeys = _keys(
                c, row, first, lo + chunk * CHUNK, here - chunk * CHUNK, kind
            )
            _, _, cpacked = _marks(clive, ckeys, prefix, done)
            _, ctotal = block_excl_scan(cpacked, lane, wave, co["scan"])
            seg_gt = seg_gt + ((ctotal >> 16) & 0xFFFF)
            seg_eq = seg_eq + (ctotal & 0xFFFF)
        if tid == 0:
            c["put_words"](c["ix_segment_ready"], 2 * slot, [seg_gt, seg_eq])
        if tid < g:
            previous = c["poll"]([(c["ix_segment_ready"], 2 * (slot - g + tid), 2)])
            fx.ptr_store(fx.Int32(previous[0][0]), co["before"] + 2 * tid)
            fx.ptr_store(fx.Int32(previous[0][1]), co["before"] + 2 * tid + 1)
        gpu.barrier()
        gt_before, eq_before = k3._sum_before(co, g)
        ties = fx.min(fx.max(take - eq_before, 0), seg_eq)
        base = gt_before + fx.min(take, eq_before)
        total = gt_before + seg_gt + take
        _emit(
            c,
            row,
            first,
            lo,
            gt,
            eq,
            base + ((excl >> 16) & 0xFFFF) + fx.min(excl & 0xFFFF, ties),
            excl & 0xFFFF,
            ties,
            kind,
        )
        gt_seen, eq_seen = (chunk_total >> 16) & 0xFFFF, chunk_total & 0xFFFF
        for chunk in range(1, nck):
            clive, ckeys = _keys(
                c, row, first, lo + chunk * CHUNK, here - chunk * CHUNK, kind
            )
            cgt, ceq, cpacked = _marks(clive, ckeys, prefix, done)
            cx, ct = block_excl_scan(cpacked, lane, wave, co["scan"])
            run = eq_seen + (cx & 0xFFFF)
            _emit(
                c,
                row,
                first,
                lo + chunk * CHUNK,
                cgt,
                ceq,
                base + gt_seen + ((cx >> 16) & 0xFFFF) + fx.min(run, ties),
                run,
                ties,
                kind,
            )
            gt_seen = gt_seen + ((ct >> 16) & 0xFFFF)
            eq_seen = eq_seen + (ct & 0xFFFF)
        if (g == nseg - 1) & (nseg > 1):
            k3._row_clear(co, radix_row)
    if g != nseg - 1:
        publish(c["put"], c["ix_emitted_ready"], slot, 1, tid == 0)
    else:
        if tid < g:
            c["poll"]([(c["ix_emitted_ready"], slot - g + tid, 1)])
        _publish_count(c, row, code, total)
