"""PLE stages (layer 1): P1 key/value projections, P2 gate, conv and residual.

P1, task ``t`` (160): every task packs the embedding row (MXFP8) into LDS.
Waves 0-3 then run key n-block ``4 t + w`` as one MFMA chain over k_iters
0-19; wave 4 runs value n-block ``t`` as 4 chains (chain ``q`` takes k_iters
``q, q + 4, ...``) summed ``((c0 + c1) + c2) + c3``, the native reduction.
The bf16 outputs go out as pairs (``pk``: the key's 4 streams, ``pv``).

P2, task ``t`` (80): columns ``32 t .. 32 t + 31`` of every stream. Wave ``w``
holds stream ``j = w % 4``, lane ``l`` its vectors of 8 ``l + 64 m`` (``m <
5``): the 5 vectors PyTorch's virtual threads ``l`` and ``l + 64`` reduce in
the gate dot. Waves 0-3 normalize the key and the query (the layer input) and
take the dot product and the scalar chain to the gate ``sigma_j``; waves 4-7
then normalize the gated value ``bf16(sigma_j v)`` of their stream
(``norm_conv``). Thread ``tid < 128`` finally owns channel ``ch =
2560 (tid / 32) + 32 t + tid % 32``: the conv over its state taps, the
activation, the gated add and the residual, and the next state. ``h'`` goes
out as pairs (``ph``, laid out as C1's ``w``) with the square sums of its
16-column halves (``phs``, as ``whs``) for the attention HC.

The PLE state (the embedding row, the weights, the norm weights, the conv
state) comes from the table behind the ``ple`` argument (:data:`STATE`),
loaded at the start of each stage.

Batched builds (``c.rows`` requests): the ``ple`` table holds a
:data:`STATE` row per request (the rows share the weights). P1 packs every
row's embedding into its LDS row and runs them as the MFMA rows (request
``r``'s key and value in ``acc[r]``). P2 runs one task set per request row
(``80 r + t``), in rounds past ``GRID`` tasks.
"""

from types import SimpleNamespace

import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.qwen4_exp.mono.kernels import buffer_ops as bo
from atom.models.qwen4_exp.mono.kernels import mailbox as mbx
from atom.models.qwen4_exp.mono.kernels import p2
from atom.models.qwen4_exp.mono.kernels.config import DEFAULT as C
from atom.models.qwen4_exp.mono.kernels.config import GRID
from atom.models.qwen4_exp.mono.kernels.device_ops import bpermute_i32, rsrc, xred
from atom.models.qwen4_exp.mono.kernels.layout import PAIR, REGIONS, SC, TASKS
from atom.models.qwen4_exp.mono.kernels.stages.common import (
    HC_WORDS,
    a_operand,
    add,
    bf16_bits,
    bf16r,
    bf16x8,
    mb_ix,
    mfma_operands,
    p2_issue,
    p2_mma,
    pick_row,
    row_ix,
    task_index,
    task_rounds,
)
from atom.models.qwen4_exp.mono.kernels.stages.delta import _theirs
from atom.models.qwen4_exp.mono.kernels.stages.hc import (
    EPS,
    INV_HIDDEN,
    OOB,
    _f32,
    fma,
    natural_sum,
    sigmoid_acc,
)
from atom.models.qwen4_exp.mono.kernels.stages.qsa import _entry, _request

P_KI = p2.k_iters(C.hidden)  # 20
P_GROUPS = P_KI // 4  # 5
KEY_NB = C.hc_width // 16  # 640: past the last key n-block, so out of range
STREAM_VECS = C.hidden // 8  # 320 vectors of 8 per stream
PV = STREAM_VECS // 64  # 5 per lane
ROW = C.hidden // 2  # bf16 words per stream
TAPS = (C.ple_conv - 1) * 3  # 9 conv state positions per channel (dilation 3)
INV_SQRT_H = 0x3CA1E89B  # fl(1 / sqrt(2560))
PV_FROM_PK = (SC["pv"] - SC["pk"]) // PAIR
# Seq builds: row r's P2 publishes its ``nrm`` (bf16 pairs per channel pair)
# for the later rows' conv state in Q3's ``isc`` region (no Q stage runs in a
# PLE layer, and a new region would move every row's mailboxes).
PV_FROM_PN = (SC["pv"] - SC["isc"]) // PAIR
assert REGIONS["isc"] >= C.hc_width // 2 and PV_FROM_PN > 0

# Byte sizes behind the resources: a load at ``OOB`` (int32 elements) returns
# zeros. The value weights take the key's sizes (its n-blocks stay in range).
KEY_W_BYTES = C.hc_width * C.hidden // 2
KEY_S_BYTES = C.hc_width // 16 * P_GROUPS * 64 * 4
STREAM_BYTES = C.hc_width * 2
CONV_W_BYTES = C.hc_width * C.ple_conv * 2
CONV_STATE_BYTES = C.hc_width * TAPS * 2

# The table behind the ``ple`` argument: one int64 per entry. ``emb`` is the
# gathered embedding row (bf16 ``[2560]``); the weights are MXFP4 tiles with
# ``scales_k4`` scales; ``nq``/``nk``/``nc`` the folded (shifted) norm
# weights (bf16 ``[10240]``); ``conv_w`` bf16 ``[10240, 4]``; ``conv_in`` the
# conv state and ``conv_out`` the next one (bf16 ``[10240, 9]``).
STATE = (
    "emb",
    "key_w",
    "key_s",
    "val_w",
    "val_s",
    "nq",
    "nk",
    "nc",
    "conv_w",
    "conv_in",
    "conv_out",
)
RECORDS = {
    "key_w": KEY_W_BYTES,
    "key_s": KEY_S_BYTES,
    "val_w": KEY_W_BYTES,
    "val_s": KEY_S_BYTES,
    "nq": STREAM_BYTES,
    "nk": STREAM_BYTES,
    "nc": STREAM_BYTES,
    "conv_w": CONV_W_BYTES,
    "conv_in": CONV_STATE_BYTES,
    "conv_out": CONV_STATE_BYTES,
}


def ple_state(p):
    """The :data:`STATE` addresses (wave-uniform int64). Call at the start of
    a stage, outside runtime branches."""
    r = rsrc(p.a_ple)
    return SimpleNamespace(**{name: _entry(r, i) for i, name in enumerate(STATE)})


def ple_row(p, r, name):
    """Request row ``r``'s address ``name`` (``r`` a Python int or
    wave-uniform)."""
    return _entry(rsrc(p.a_ple), row_ix(STATE.index(name), r, len(STATE)))


def _res(addr, name):
    return bo.create_buffer_resource_from_addr(addr, num_records_bytes=RECORDS[name])


# ------------------------------------------------------------- P1


def _value_chains(units, x8, xe, lane, rows):
    """The value n-block of each request row: chain ``q`` over k_iters ``q,
    q + 4, ...``, then the native fixed reduction of the 4 chains."""
    accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(4)]
    for g in range_constexpr(P_GROUPS):
        ws, sw = units[g]
        for q in range_constexpr(4):
            kit = g * 4 + q
            ea = fx.ptr_load(xe + (kit * 4 + lane // 16))
            accs[q] = p2.mfma_p2(accs[q], a_operand(x8, kit, lane), ws[q], ea, sw, 0, q)
    return [((accs[0][r] + accs[1][r]) + accs[2][r]) + accs[3][r] for r in range(rows)]


def _put_block(c, region, nb, v, r=0):
    """The 16 bf16 outputs of n-block ``nb`` (lanes 0-15) as 8 pairs, in
    request row ``r``."""
    lane = c.lane
    out = bf16r(v)
    other = xred(out, 1, _theirs)
    if (lane < 16) & (lane % 2 == 0):
        mbx.put(region, mb_ix(nb * 8 + lane // 2, r), mbx.bf16_pair(out, other), c.tag)


_put_block = ASTRewriter.transform(_put_block)


def _pack_vec(x):
    """MXFP8 pack of thread ``tid``'s vector ``x``: (2 fp8 words, its
    32-block's exponent); a 32-block sits on 4 adjacent lanes."""
    y = bf16x8([x[e] for e in range(4)])
    amax = fmath.absf(y[0])
    for e in range_constexpr(1, 8):
        amax = fx.max(amax, fmath.absf(y[e]))
    amax = xred(xred(amax, 1, fx.max), 2, fx.max)
    ex, scale = p2.e8m0_fp8(amax)
    words = fx.Vector.from_elements(
        [p2.fp8_word(*y[0:4], scale), p2.fp8_word(*y[4:8], scale)], fx.Int32
    )
    return words, ex


def ple_proj(c, p):
    tid, lane, wave = c.tid, c.lane, c.wave
    rows = c.rows
    t = task_index(c, "p1")
    s = ple_state(p)
    x8 = c.lds_i + c.L.lds["p_x8"]
    xe = c.lds_i + c.L.lds["p_xe"]
    sx, se = c.L.stride["p_x8"], c.L.stride["p_xe"]
    xr, er = mfma_operands(c, x8, xe, "p_x8", "p_xe")
    key = wave < 4
    nb = key.select(t * 4 + wave, (wave == 4).select(t, KEY_NB))
    r_w = _res(key.select(s.key_w, s.val_w), "key_w")
    r_s = _res(key.select(s.key_s, s.val_s), "key_s")
    r_emb = [rsrc(s.emb)] + [rsrc(ple_row(p, r, "emb")) for r in range(1, rows)]

    if t < c.L.tasks["p1"]:
        # The embedding rows first: the barrier keeps every wave's row loads
        # ahead of the weight streams in the CU's load pipe.
        vec = fx.min(tid, STREAM_VECS - 1) * 4
        xs = [
            fx.Vector(bo.buffer_load(r_emb[r], vec, vec_width=4, dtype=T.i32))
            for r in range(rows)
        ]
        rocdl.sched_barrier(0)
        gpu.barrier()
        # Waves 5-7 read out of range (n-block 640): zeros, no memory traffic.
        units = [p2_issue(r_w, r_s, nb, P_KI, g, lane) for g in range(P_GROUPS)]

        # The pack: thread ``tid < 320`` (waves 0-4) owns vector ``tid`` of
        # every row.
        for rp in range_constexpr(rows):
            words, ex = _pack_vec(xs[rp])
            if wave < 5:
                fx.ptr_store(words, x8 + row_ix(tid * 2, rp, sx))
                if lane % 4 == 0:
                    fx.ptr_store(ex, xe + row_ix(tid // 4, rp, se))
        gpu.barrier()

        if wave < 4:
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for g in range_constexpr(P_GROUPS):
                acc = p2_mma(acc, units[g], g, xr, er, lane)
            for rk in range_constexpr(rows):
                _put_block(c, c.mb_pk, nb, acc[rk], rk)
        if wave == 4:
            vs = _value_chains(units, xr, er, lane, rows)
            for rv in range_constexpr(rows):
                _put_block(c, c.mb_pv, nb, vs[rv], rv)


ple_proj = ASTRewriter.transform(ple_proj)


# ------------------------------------------------------------- P2


def _sq_chunk(y):
    """Square sum of a 32-chunk in the natural pairing tree; this lane's 8
    values, the chunk's 4 vectors on 4 adjacent lanes."""
    q = [v * v for v in y]
    s = ((q[0] + q[1]) + (q[2] + q[3])) + ((q[4] + q[5]) + (q[6] + q[7]))
    return xred(xred(s, 1, add), 2, add)


def stream_inv(ys, lane):
    """Inverse RMS of the stream whose vector ``l + 64 m`` is ``ys[m]``
    on lane ``l``. Chunk ``C[16 m + l / 4]`` lands on lanes ``4 (l / 4) ..``;
    lane ``l`` gathers ``a[l] = C[l] (+ C[l + 64] if l < 16)``."""
    src = (lane % 16) * 16  # byte address of lane 4 (l % 16)
    g = [
        bpermute_i32(src, _sq_chunk(y).bitcast(fx.Int32)).bitcast(fx.Float32)
        for y in ys
    ]
    q = lane // 16
    a = (q == 0).select(g[0], (q == 1).select(g[1], (q == 2).select(g[2], g[3])))
    a = (lane < 16).select(a + g[4], a)
    stab = natural_sum(a) * _f32(INV_HIDDEN) + _f32(EPS)
    return (fx.Float64(1.0) / fx.Float64(fmath.sqrt(stab.to(fx.Float64)))).to(
        fx.Float32
    )


def _vecs(vs):
    """dwordx4 vectors of bf16 words -> lists of 8 f32."""
    return [bf16x8([v[e] for e in range(4)]) for v in vs]


def _normalize(xs, inv, ws):
    return [[bf16r((x * inv) * w) for x, w in zip(xv, wv)] for xv, wv in zip(xs, ws)]


def _gate(kn, qn):
    """The gate: the dot product of one stream in PyTorch's order, then the
    scalar chain to ``bf16(sigma_o(g))``."""
    pr = [[bf16r(k * q) for k, q in zip(kv, qv)] for kv, qv in zip(kn, qn)]
    zero = fx.Float32(0.0)
    lo = hi = None
    for e in range_constexpr(8):
        a = ((zero + pr[0][e]) + pr[2][e]) + pr[4][e]  # virtual thread l
        b = (zero + pr[1][e]) + pr[3][e]  # virtual thread l + 64
        lo = a if lo is None else lo + a
        hi = b if hi is None else hi + b
    red = natural_sum(lo + hi)
    s = bf16r(bf16r(red) * _f32(INV_SQRT_H))
    mag = fmath.absf(s)
    cl = bf16r((mag != mag).select(mag, fx.max(mag, _f32(EPS))))
    root = bf16r(fmath.sqrt(cl))
    sgn = (s > zero).select(fx.Float32(1.0), (s < zero).select(fx.Float32(-1.0), zero))
    return bf16r(sigmoid_acc(bf16r(root * sgn)))


def _half(w, half):
    """bf16 ``half`` (0 low, 1 high) of word ``w`` in the low 16 bits."""
    return w & 0xFFFF if half == 0 else (w >> 16) & 0xFFFF


def _bf(bits):
    return (bits << 16).bitcast(fx.Float32)


def _pack(lo, hi):
    return lo | (hi << 16)


def _odd_half(odd, w):
    """The f32 of the bf16 half of ``w`` at the column's parity."""
    return odd.select(w & fx.Int32(-65536), w << 16).bitcast(fx.Float32)


def _half_sum(v):
    """Sums over 16 lanes in the pairing tree of the HC chunk sums (a half
    of a 32-chunk, see ``hc._chunk_sum``)."""
    for off in (1, 2, 4, 8):
        v = xred(v, off, add)
    return v


def _mix_specs(c, pbase, lane, col, r):
    """Poll specs of a P2 thread of request row ``r``: its 5 vectors (key
    stream or value) as 2 two-pair loads each, then the value word of its
    conv column."""
    specs = []
    for m in range(PV):
        v = pbase + 4 * (lane + 64 * m)
        specs += [(c.mb_pk, mb_ix(v, r), 2), (c.mb_pk, mb_ix(v + 2, r), 2)]
    return specs + [(c.mb_pv, mb_ix(col // 2, r), 1)]


def _next_state(odd, taps, nrm):
    """Words of the next state (taps 1-8, then ``nrm``) this thread writes:
    4 at ``sw + odd`` and 1 at ``sw + 4``. An even channel's last word holds
    the next channel's first element."""
    n = taps[1:] + [bf16_bits(nrm)]
    n0_next = xred(n[0], 1, _theirs)
    words = [
        odd.select(_pack(n[2 * i + 1], n[2 * i + 2]), _pack(n[2 * i], n[2 * i + 1]))
        for i in range(4)
    ]
    last = odd.select(_pack(n[7], n[8]), _pack(n[8], n0_next))
    return fx.Vector.from_elements(words, fx.Int32), last


def _seq_taps(c, rq, ch, col, odd, taps):
    """Seq builds: row ``rq``'s 9 state elements (bf16 bits), the last 9 of
    row 0's state (``taps``) then the ``nrm`` of rows 0 .. rq - 1. Polls
    only rows before ``rq`` (the others' specs repeat this row's own ``pv``
    word, already published): a later row waits on this one."""
    specs = [
        (
            c.mb_isc,
            (rq > q).select(mb_ix(ch // 2, q), mb_ix(PV_FROM_PN + col // 2, rq)),
            1,
        )
        for q in range(c.rows - 1)
    ]
    got = mbx.poll(specs, c.tag)
    hist = taps + [odd.select(_half(w[0], 1), _half(w[0], 0)) for w in got]
    return [pick_row(rq, [hist[r + k] for r in range(c.rows)]) for k in range(TAPS)]


def ple_mix(c, p):
    t = task_index(c, "p2")
    for k in range_constexpr(task_rounds(c, "p2")):
        if k > 0:
            gpu.barrier()  # after the last round's LDS reads
        _ple_mix(c, p, t if k == 0 else t + k * GRID)


def _ple_mix(c, p, t):
    tid, lane, wave = c.tid, c.lane, c.wave
    rq, tq = _request(c, t, TASKS["p2"])
    s = ple_state(p)
    sig_l = c.lds_f + c.L.lds["p_sig"]
    inv_l = c.lds_f + c.L.lds["p_inv"]
    j = wave % 4
    key = wave < 4
    r_nq = _res(s.nq, "nq")
    r_nk = _res(s.nk, "nk")
    r_nc = _res(s.nc, "nc")
    r_cw = _res(s.conv_w, "conv_w")
    if const_expr(c.seq):
        # Every row starts from row 0's state and stores its own snapshot.
        cin, cout = s.conv_in, ple_row(p, rq, "conv_out")
    else:
        cin = s.conv_in if c.rows == 1 else ple_row(p, rq, "conv_in")
        cout = s.conv_out if c.rows == 1 else ple_row(p, rq, "conv_out")
    r_cin = _res(cin, "conv_in")
    r_cout = _res(cout, "conv_out")
    # Stream words of this lane's vectors; waves 4-7 read the key side's
    # operands out of range.
    vw = [key.select(j * ROW + 4 * (lane + 64 * m), OOB) for m in range(PV)]
    # The pairs polled: the key's stream j (waves 0-3) or the value (4-7).
    pbase = key.select(j * ROW, PV_FROM_PK)
    # Conv channel of thread tid < 128 (the others read out of range).
    conv = tid < 128
    cj = (tid // 32) % 4
    col = tq * 32 + tid % 32
    ch = cj * C.hidden + col
    odd = (tid % 2) == 1  # parity of ch and of its first state element
    sw = ch * TAPS // 2

    def hint(m):
        return c.mb_pv, mb_ix(m * 8 + 7, rq)

    if t < c.L.tasks["p2"]:
        hv = [
            fx.Vector(
                bo.buffer_load(
                    p.h_in, row_ix(w, rq, HC_WORDS), vec_width=4, dtype=T.i32
                )
            )
            for w in vw
        ]
        qw = [fx.Vector(bo.buffer_load(r_nq, w, vec_width=4, dtype=T.i32)) for w in vw]
        kw = [fx.Vector(bo.buffer_load(r_nk, w, vec_width=4, dtype=T.i32)) for w in vw]
        cw = fx.Vector(
            bo.buffer_load(r_cw, conv.select(ch * 2, OOB), vec_width=2, dtype=T.i32)
        )
        st = fx.Vector(
            bo.buffer_load(r_cin, conv.select(sw, OOB), vec_width=4, dtype=T.i32)
        )
        st4 = fx.Int32(
            bo.buffer_load(r_cin, conv.select(sw + 4, OOB), vec_width=1, dtype=T.i32)
        )
        hw = fx.Int32(
            bo.buffer_load(
                p.h_in,
                conv.select(row_ix(ch // 2, rq, HC_WORDS), OOB),
                vec_width=1,
                dtype=T.i32,
            )
        )
        ncw = fx.Int32(
            bo.buffer_load(r_nc, conv.select(ch // 2, OOB), vec_width=1, dtype=T.i32)
        )

        # The query norm does not wait on P1 (waves 4-7 normalize zeros).
        hs = _vecs(hv)
        qn = _normalize(hs, stream_inv(hs, lane), _vecs(qw))

        mbx.pre_poll(hint, c.L.tasks["p1"], c.tag)
        got = mbx.poll(_mix_specs(c, pbase, lane, col, rq), c.tag)
        xs = [bf16x8(got[2 * m] + got[2 * m + 1]) for m in range(PV)]

        if wave < 4:
            kn = _normalize(xs, stream_inv(xs, lane), _vecs(kw))
            sigma = _gate(kn, qn)
            if lane == 0:
                fx.ptr_store(sigma, sig_l + j)
        gpu.barrier()
        if wave >= 4:
            gated = [[bf16r(fx.ptr_load(sig_l + j) * x) for x in xv] for xv in xs]
            ginv = stream_inv(gated, lane)
            if lane == 0:
                fx.ptr_store(ginv, inv_l + j)
        gpu.barrier()

        if wave < 2:
            gv = bf16r(fx.ptr_load(sig_l + cj) * _odd_half(odd, got[2 * PV][0]))
            nrm = bf16r((gv * fx.ptr_load(inv_l + cj)) * _odd_half(odd, ncw))
            # State element k (bf16 bits): half ``odd + k`` of the 5 words.
            sv = [st[0], st[1], st[2], st[3], st4]
            taps = [
                odd.select(
                    _half(sv[(k + 1) // 2], (k + 1) % 2), _half(sv[k // 2], k % 2)
                )
                for k in range(TAPS)
            ]
            if const_expr(c.seq):
                nrm_other = xred(nrm, 1, _theirs)
                if lane % 2 == 0:
                    mbx.put(
                        c.mb_isc,
                        mb_ix(ch // 2, rq),
                        mbx.bf16_pair(nrm, nrm_other),
                        c.tag,
                    )
                taps = _seq_taps(c, rq, ch, col, odd, taps)
            w0, w1 = mbx.bf2_f32(cw[0])
            w2, w3 = mbx.bf2_f32(cw[1])
            acc = fma(_bf(taps[0]), w0, fx.Float32(0.0))
            acc = fma(_bf(taps[3]), w1, acc)
            acc = fma(_bf(taps[6]), w2, acc)
            acc = fma(nrm, w3, acc)
            rc = bf16r(acc)
            act = bf16r(rc * sigmoid_acc(rc))
            out = bf16r(_odd_half(odd, hw) + bf16r(gv + act))

            nxt, last = _next_state(odd, taps, nrm)
            bo.buffer_store(nxt, r_cout, sw + tid % 2)
            bo.buffer_store(last, r_cout, sw + 4)

            other = xred(out, 1, _theirs)
            sq = _half_sum(out * out)
            if lane % 2 == 0:
                mbx.put(c.mb_ph, mb_ix(ch // 2, rq), mbx.bf16_pair(out, other), c.tag)
            if lane % 16 == 0:
                mbx.put(
                    c.mb_phs,
                    mb_ix(cj * c.L.tasks["c1"] + tq * 2 + (tid % 32) // 16, rq),
                    sq,
                    c.tag,
                )


_ple_mix = ASTRewriter.transform(_ple_mix)
