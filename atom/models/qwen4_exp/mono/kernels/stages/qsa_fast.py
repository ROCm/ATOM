"""Q3 and Q4 of the QSA layers: the index blocks' scores and the 512 best
blocks, for contexts up to :data:`MAX_CONTEXT` tokens.

- Q3 runs on every CTA (``GRID / rows`` tasks per request row). Waves 0-3
  first build index query head ``w`` in LDS (zero-centered norm, RoPE at row
  ``L``). The blocks the step's first token's context completes take their
  compressed keys from ``index_cmp``, where the step that completed each
  wrote it (ATOM's own steps, prefill included, write theirs there too),
  :data:`UC` blocks per wave at a time, the batch's loads first
  (:func:`_cached_scores`). The blocks after them, at most one per wave, are
  keyed from their 4 raw index key rows (:func:`_block_key`: row ``L`` from
  the mailbox; pooled, normalized and rotated at row ``4 b``), and the wave
  of the block row ``L`` completes stores its key in ``index_cmp``. A score
  (:func:`_score`) takes the 4 dot products in f64 (rounded to f32) and is
  published in ``isc``; a key Q3 wrote gives the bits its raw keys give. The
  64-bit wave sums swap the upper halves of the wave in VALU (``permlane``).
  Each task ends with its hint (``fisch``).
- Q4 runs in two levels per request row: :data:`CHUNKS` chunk tasks, each
  selecting the 512 best of its :data:`FAST_CHUNK` blocks, and a final task
  selecting the 512 best of their candidates and ranking them, ties to the
  lower block. The 512 best of all blocks are among their chunk's 512 best.
  Up to :data:`FAST_CHUNK` blocks the final task selects from the scores
  itself, and up to 512 takes them all.

Selection (:func:`_select`): each thread holds its keys in registers --
folded score bits and ``0xFFFF - block``, unique -- and the radix passes
(:data:`RADIX`) count the keys matching the prefix so far in one of three
LDS histograms in turn, so that a pass needs one barrier: every wave scans
the histogram it just filled while the one two passes on is cleared. The
candidates go to LDS as 64-bit keys, and a bitonic sort of them, one per
thread, ranks them (:func:`_rank`).

The scores and this module's mailboxes (:data:`FSC`) sit in the scratch
buffer's second half (``layout.scratch_bytes``), :data:`SCORE_SHIFT` bytes
past the first: each row's scores at its ``isc`` offset, up to
:data:`MAX_BLOCKS` of them, and the mailboxes before them.
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from atom.models.qwen4_exp.mono.kernels import buffer_ops as bo
from atom.models.qwen4_exp.mono.kernels import mailbox as mbx
from atom.models.qwen4_exp.mono.kernels.config import DEFAULT as C
from atom.models.qwen4_exp.mono.kernels.config import GRID, THREADS
from atom.models.qwen4_exp.mono.kernels.device_ops import (
    read_lane_i32,
    rsrc,
    wave_umax_dpp,
    xshfl,
)
from atom.models.qwen4_exp.mono.kernels.dpp_utils import update_dpp_i32
from atom.models.qwen4_exp.mono.kernels.layout import (
    BASE,
    MAX_ROWS,
    PAIR,
    SC,
    SCRATCH_BYTES,
    STRIDE,
    TASKS,
    WAVES,
)
from atom.models.qwen4_exp.mono.kernels.stages.common import bf16r, mb_ix, task_index
from atom.models.qwen4_exp.mono.kernels.stages.hc import _f32
from atom.models.qwen4_exp.mono.kernels.stages.qsa import (
    INDEX_WORDS,
    P_IK,
    PAGE_BLOCKS,
    Q1_INDEX_TASKS,
    R128,
    ROPE_WORDS,
    SEL,
    SIGN,
    SQRT128,
    _i32,
    _lds_i4,
    _request,
    norm,
    page_block,
    qsa_row,
    row_len,
    unpack,
    word,
)

# Index blocks of a request row (16 bits of the radix key hold the block), and
# the contexts they hold.
MAX_BLOCKS = 1 << 16
MAX_CONTEXT = MAX_BLOCKS * C.index_compress  # 262144
# The scores' region sits this many bytes past ``isc``, in the scratch's second
# half (wide builds add their rows past ``MAX_ROWS``, ``kernel._stage_mailboxes``).
SCORE_SHIFT = MAX_ROWS * SCRATCH_BYTES
assert (
    SC["isc"] + MAX_BLOCKS * PAIR <= SCRATCH_BYTES
)  # a row's scores end within its row

UC = 8  # Q3: blocks per wave per batch, from their compressed keys
SPAN = 64 * PAGE_BLOCKS  # Q3: a wave's blocks per load of their page entries
assert UC % PAGE_BLOCKS == 0  # a batch takes whole pages
_S = 6 - (4 * UC).bit_length() + 1  # Q3: the butterfly steps after the scatter
assert 4 * UC == 64 >> _S and C.index_heads == 4
FAST_CHUNK = 8192  # Q4: blocks per chunk task
CHUNKS = MAX_BLOCKS // FAST_CHUNK  # 8
Q4_TASKS = CHUNKS + 1  # per request row: the chunk tasks, then the final one
Q4_BATCH = 8  # scores polled per thread and retry
KEYS_CHUNK = FAST_CHUNK // THREADS  # 16 keys per thread
KEYS_FINAL = CHUNKS * SEL // THREADS  # 8 candidates per thread
assert (
    KEYS_FINAL == CHUNKS and SEL == THREADS
)  # thread ``tid`` holds slot ``tid`` of each chunk
assert FAST_CHUNK % (THREADS * Q4_BATCH) == 0 and FAST_CHUNK > SEL
assert TASKS["q3"] == GRID and TASKS["q4"] == Q4_TASKS
assert BASE["q4"] == BASE["q7"] + TASKS["q7"]  # Q4 on the CTAs after Q7's

# Q4's radix passes over the key (score bits, 0xFFFF - block): (key word,
# shift of the 8-bit digit).
RADIX = ((0, 24), (0, 16), (0, 8), (0, 0), (1, 8), (1, 0))

# Mailboxes, bytes into each row's second half: Q3's hints (one per task, at
# most ``GRID`` per row), the chunk tasks' hints, then their candidates (two
# pairs each: folded score bits, block).
_FAST_REGIONS = {"fisch": GRID, "fch": CHUNKS, "fcand": CHUNKS * SEL * 2}
FSC = {}
_off = 0
for _name, _pairs in _FAST_REGIONS.items():
    FSC[_name] = _off
    _off += (_pairs * PAIR + 255) // 256 * 256
assert _off <= SC["isc"]  # before the row's scores

# LDS words from ``q4_keys``: the three radix histograms, the waves' counts,
# the candidates' 64-bit keys (two words each) and blocks, then pass 0's
# histogram in ``COPIES`` copies, lane ``l`` counting into copy ``l % COPIES``:
# its keys' digits are few (the scores' exponents), and lanes adding to one
# LDS word wait on each other. Copies ``H0_STRIDE`` words apart sit on
# distinct banks.
H_WORDS = 256
CNT = 3 * H_WORDS
CK = CNT + 256
CB = CK + 2 * SEL
COPIES = 8
H0_STRIDE = H_WORDS + 4
H0 = CB + SEL
H0_CLEAR = -(-COPIES * H0_STRIDE // THREADS) * THREADS  # whole rounds of THREADS
LDS_WORDS = H0 + H0_CLEAR
assert H0 % 4 == 0 and H0_STRIDE % 4 == 0  # _lds_i4's 16-byte loads
assert LDS_WORDS <= STRIDE["q4_keys"]


def _mailboxes(c):
    """Move this stage's ``isc`` to the second half and add :data:`FSC`'s
    (``_run`` sets every mailbox afresh before the next stage)."""
    c.mb_isc = c.mb_isc + SCORE_SHIFT
    half = c.mb_isc - SC["isc"]
    for name, off in FSC.items():
        setattr(c, "mb_" + name, half + off)


# ------------------------------------------------------------------ Q3


def _permswap(a, b, off):
    """``v_permlane32_swap`` / ``v_permlane16_swap`` (``off`` 32 / 16) of
    words ``a`` and ``b``: ``a``'s upper lanes (``l & off``) trade places with
    ``b``'s lower ones, so of lanes ``l`` and ``l ^ off`` the lower gets
    (its ``a``, the upper's ``a``) and the upper (the lower's ``b``, its ``b``)."""
    swap = rocdl.permlane32_swap if off == 32 else rocdl.permlane16_swap
    pair = swap(
        llvm.StructType.get_literal([T.i32, T.i32]),
        as_ir_value(fx.Int32(a)),
        as_ir_value(fx.Int32(b)),
        False,
        False,
    )
    return tuple(fx.Int32(llvm.extractvalue(T.i32, pair, [k])) for k in range(2))


def _swap(v, off):
    """(the lower lane's ``v``, the upper lane's) of lanes ``l`` and ``l ^
    off`` (32 or 16), in VALU, as ``device_ops.xred``."""
    return _permswap(v, v, off)


def _words(v):
    return fx.Vector.from_elements([v], fx.Float64).bitcast(fx.Int32)


def _f64(words):
    return fx.Vector.from_elements(list(words), fx.Int32).bitcast(fx.Float64)[0]


def _wave_sum64(v):
    """Butterfly sum of an f64 over the wave; every lane gets the same bits.
    The 32 and 16 steps swap in VALU (:func:`_swap`)."""
    for off in (32, 16, 8, 4, 2, 1):
        w = _words(v)
        if off >= 16:
            halves = [_swap(w[0], off), _swap(w[1], off)]
            v = _f64([h[0] for h in halves]) + _f64([h[1] for h in halves])
        else:
            v = v + _f64([xshfl(w[0], off), xshfl(w[1], off)])
    return v


# Row shifts by 8 / 4 (right, then left) and the banks of the lanes with that
# bit set (``device_ops.xshfl``'s DPP).
_DPP_HALF = {8: (0x118, 0x108, 0xC), 4: (0x114, 0x104, 0xA)}


def _scatter_step(lo, hi, off, lane):
    """One step of :func:`_wave_sum64`'s butterfly over many values at once:
    lane ``l`` and lane ``l ^ off`` each keep half of them (the lower lane
    ``lo``, the upper ``hi``) and add the partner's copy, which halves the
    values per lane. Each sum adds the butterfly's two values (the 32 / 16
    steps in its order), so it has its bits."""
    out = []
    for a, b in zip(lo, hi):
        wa, wb = _words(a), _words(b)
        if off >= 16:
            pairs = [_permswap(wa[k], wb[k], off) for k in range(2)]
            out.append(_f64([q[0] for q in pairs]) + _f64([q[1] for q in pairs]))
            continue
        upper = (lane & off) != 0
        if off in _DPP_HALF:
            shr, shl, banks = _DPP_HALF[off]
            got = []
            for k in range(2):
                r = fx.Int32(update_dpp_i32(wb[k], wb[k], shr, 0xF, banks, False))
                got.append(
                    fx.Int32(update_dpp_i32(r, wa[k], shl, 0xF, 0xF ^ banks, False))
                )
        else:
            send = _words(upper.select(a, b))
            got = [xshfl(send[k], off) for k in range(2)]
        out.append(upper.select(b, a) + _f64(got))
    return out


def _scatter_sums(vals, lane):
    """The wave sums (:func:`_wave_sum64`'s bits) of the ``2^n`` (``n`` <= 6)
    values each lane holds in ``vals``: lane ``l`` ends with the sum of value
    ``l >> (6 - n)``."""
    off = 32
    while len(vals) > 1:
        half = len(vals) // 2
        vals = _scatter_step(vals[:half], vals[half:], off, lane)
        off //= 2
    v = vals[0]
    while off:
        w = _words(v)
        v = v + _f64([xshfl(w[0], off), xshfl(w[1], off)])
        off //= 2
    return v


def _rope16(y, cs, sn, lane, active):
    """``qsa.rope`` with ``off`` 16, the partner from a VALU swap."""
    neg = ((lane & 16) == 0).select(fx.Int32(SIGN), fx.Int32(0))
    out = []
    for v, c, s in zip(y, cs, sn):
        lo, hi = _swap(v.bitcast(fx.Int32), 16)
        partner = ((lane & 16) == 0).select(hi, lo)
        partner = (partner ^ neg).bitcast(fx.Float32)
        out.append(active.select(bf16r(bf16r(v * c) + bf16r(partner * s)), v))
    return out


def _block_key(raw, wk, kc, ks, lane):
    """A block's compressed key (this lane's pair) from its four raw index
    key rows ``raw`` (this lane's pairs), ``wk`` the norm weight and ``kc`` /
    ``ks`` the RoPE words of the block's first token: the pooled mean, the
    zero-centred norm, RoPE. bf16 values: ``index_cmp``'s row holds them."""
    pooled = [
        bf16r(((raw[0][e] + raw[1][e]) + (raw[2][e] + raw[3][e])) * fx.Float32(0.25))
        for e in range(2)
    ]
    y = norm(pooled, wk, R128, True)
    return _rope16(y, unpack([kc]), unpack([ks]), lane, lane < ROPE_WORDS)


def _score(q, kn):
    """A block's score from the index heads' queries ``q`` and its key ``kn``
    (this lane's pairs): the relu of each head's dot product (a 64-bit wave
    sum), their sum over sqrt(128)."""
    relu = []
    for h in range(C.index_heads):
        part = (q[2 * h] * kn[0]).to(fx.Float64) + (q[2 * h + 1] * kn[1]).to(fx.Float64)
        relu.append(fx.max(_wave_sum64(part).to(fx.Float32), fx.Float32(0.0)))
    return ((relu[0] + relu[1]) + (relu[2] + relu[3])) / _f32(SQRT128)


def _fma64(a, b, c):
    return fx.Float64(
        llvm.call_intrinsic(
            T.f64, "llvm.fma.f64", [fx.Float64(v).ir_value() for v in (a, b, c)], [], []
        )
    )


def _cached_scores(c, rq, q, r_pg, r_cmp, first, nc, nw):
    """Row ``rq``'s scores of the blocks below ``nc`` from their
    ``index_cmp`` rows. Those rows hold the bits Q3 computes from the raw
    keys: the step that last ran a block's fourth token wrote them, after
    its other tokens' last runs (a step that reruns a token reruns every
    later one), and a step runs only tokens past every block below ``nc``.

    While no wave has two blocks (``nc`` up to ``nw``: contexts to ~8K at
    one row), wave ``first`` takes block ``first`` with :func:`_score`.
    Past that, every wave of the row (the test is the row's: a wave left on
    block ``first`` would leave blocks no wave scores, and Q4 would wait on
    them) takes ``nb`` consecutive blocks from ``first nb``, :data:`UC` per
    batch: its page entries come in one load per :data:`SPAN` blocks (lane
    ``l`` the ``l``-th page), so a batch's key loads wait on no other load.
    The batch's ``4 UC`` head sums are reduce-scattered
    (:func:`_scatter_sums`), each pair's two products one f64 multiply and
    fma: the f32 products of bf16 ``q`` and keys are exact, so these are
    :func:`_score`'s sums. Lane ``l`` ends with head ``(l >> S) & 3`` of
    block ``l >> (S + 2)``, the relus add as :func:`_score`'s, and one store
    puts the batch's scores."""
    lane = c.lane
    if nc <= nw:
        if first < nc:
            g = page_block(r_pg, first, True)
            score = _score(q, unpack([word(r_cmp, g * INDEX_WORDS + lane)]))
            if lane == 0:
                mbx.put(c.mb_isc, mb_ix(first, rq), score, c.tag)
    else:
        qd = [v.to(fx.Float64) for v in q]
        nb = (nc + nw * UC - 1) // (nw * UC) * UC
        start = first * nb
        end = fx.min(start + nb, nc)
        last = (end - 1) // PAGE_BLOCKS
        for s0 in range(start, end, SPAN):
            pg = word(r_pg, fx.min(s0 // PAGE_BLOCKS + lane, last))
            # Here, so the batches' reads of pg wait on no batch's store.
            rocdl.s_waitcnt(vmcnt=0)
            for b0 in range(s0, fx.min(s0 + SPAN, end), UC):
                at = (b0 - s0) // PAGE_BLOCKS
                pages = [read_lane_i32(pg, at + k) for k in range(UC // PAGE_BLOCKS)]
                gs = [
                    pages[u // PAGE_BLOCKS] * PAGE_BLOCKS + u % PAGE_BLOCKS
                    for u in range(UC)
                ]
                kws = [word(r_cmp, g * INDEX_WORDS + lane) for g in gs]
                parts = []
                for u in range_constexpr(UC):
                    kd = [v.to(fx.Float64) for v in unpack([kws[u]])]
                    for h in range_constexpr(C.index_heads):
                        parts.append(_fma64(qd[2 * h + 1], kd[1], qd[2 * h] * kd[0]))
                relu = fx.max(
                    _scatter_sums(parts, lane).to(fx.Float32), fx.Float32(0.0)
                )
                pair = relu + xshfl(relu, 1 << _S)
                score = pair + xshfl(pair, 2 << _S)
                b = b0 + (lane >> (_S + 2))
                if ((lane & ((4 << _S) - 1)) == 0) & (b < end):
                    mbx.put(c.mb_isc, mb_ix(b, rq), score / _f32(SQRT128), c.tag)


_cached_scores = ASTRewriter.transform(_cached_scores)


def q3(c, p):
    _mailboxes(c)
    lane, wave = c.lane, c.wave
    t = task_index(c, "q3")
    iqn = c.lds_f + c.L.lds["iqn"]
    per = c.L.tasks["q3"] // c.rows
    rq, tq = _request(c, t, per)
    L = row_len(p, rq)
    a_pages = qsa_row(p, rq, "pages")
    a_index_key = qsa_row(p, rq, "index_key")
    a_index_cmp = qsa_row(p, rq, "index_cmp")
    a_cos = qsa_row(p, rq, "cos")
    a_sin = qsa_row(p, rq, "sin")
    cb = (L + 1) // 4

    # Tasks with no block skip the stage; Q4 waits on the others' hints, and
    # on task 0's at any context (a row at positions 0-2 has no block).
    if (t < c.L.tasks["q3"]) & (tq * WAVES < fx.max(cb, 1)):
        r_pg = rsrc(a_pages)
        r_ik = rsrc(a_index_key)
        r_cmp = rsrc(a_index_cmp)
        r_cos = rsrc(a_cos)
        r_sin = rsrc(a_sin)
        rope_word = fx.min(lane, ROPE_WORDS - 1)
        wq = unpack([word(rsrc(p.a_iq_norm), lane)])
        wk = unpack([word(rsrc(p.a_ik_norm), lane)])
        cs = unpack([word(r_cos, L * ROPE_WORDS + rope_word)])
        sn = unpack([word(r_sin, L * ROPE_WORDS + rope_word)])
        mbx.pre_poll(lambda m: (c.mb_qp, mb_ix(32 * m, rq)), Q1_INDEX_TASKS, c.tag)

        if wave < C.index_heads:
            x = mbx.poll([(c.mb_qp, mb_ix(INDEX_WORDS * wave + lane, rq), 1)], c.tag)[0]
            y = norm(unpack(x), wq, R128, True)
            y = _rope16(y, cs, sn, lane, lane < ROPE_WORDS)
            for k in range_constexpr(2):
                fx.ptr_store(y[k], iqn + (wave * C.index_dim + 2 * lane + k))
        if const_expr(c.seq):
            got = mbx.poll(
                [(c.mb_qp, mb_ix(P_IK + lane, q), 1) for q in range(c.rows)], c.tag
            )
            iks = [v[0] for v in got]
        else:
            ik = mbx.poll([(c.mb_qp, mb_ix(P_IK + lane, rq), 1)], c.tag)[0][0]
        gpu.barrier()
        q = [
            fx.ptr_load(iqn + (h * C.index_dim + 2 * lane + e))
            for h in range(4)
            for e in range(2)
        ]

        # Wave ``w`` of task ``tq`` is wave ``first`` of the row's ``nw``. The
        # blocks the step's first token's context completes are keyed by the
        # rows earlier steps wrote to ``index_cmp`` (:func:`_cached_scores`);
        # the rest, at most one per wave, from their raw keys, and the row
        # that completes one writes its compressed key.
        nw = per * WAVES
        first = tq * WAVES + wave
        nc = L // 4
        if const_expr(c.seq):
            nc = p.Ls[0] // 4
        _cached_scores(c, rq, q, r_pg, r_cmp, first, nc, nw)
        b = nc + first
        if b < cb:
            g = page_block(r_pg, b, True)
            kws = [word(r_ik, (4 * g + r) * INDEX_WORDS + lane) for r in range(4)]
            at = 4 * b * ROPE_WORDS + rope_word
            kc, ks = word(r_cos, at), word(r_sin, at)
            raw = []
            for r in range_constexpr(4):
                row = 4 * b + r
                kw = kws[r]
                if const_expr(c.seq):
                    for s in range_constexpr(c.rows):
                        kw = (row == p.Ls[s]).select(iks[s], kw)
                    raw.append(unpack([fx.Int32(kw)]))
                else:
                    raw.append(unpack([fx.Int32((row == L).select(ik, kw))]))
            kn = _block_key(raw, wk, kc, ks, lane)
            score = _score(q, kn)
            if lane == 0:
                mbx.put(c.mb_isc, mb_ix(b, rq), score, c.tag)
            # Row L completes block b: its key for ATOM's compressed cache.
            if 4 * b + 3 == L:
                bo.buffer_store(
                    mbx.bf16_pair(kn[0], kn[1]), r_cmp, g * INDEX_WORDS + lane
                )

        gpu.barrier()
        if c.tid == 0:
            mbx.put(c.mb_fisch, mb_ix(tq, rq), 1, c.tag)


q3 = ASTRewriter.transform(q3)


# ------------------------------------------------------------------ Q4


def _clear(c, k0):
    """Histograms 0 and 1 and pass 0's copies to zero; a barrier must follow
    before the first pass."""
    fx.ptr_store(fx.Int32(0), k0 + fx.Int32(c.tid))
    for e in range_constexpr(H0_CLEAR // THREADS):
        fx.ptr_store(fx.Int32(0), k0 + (H0 + THREADS * e + c.tid))


def _digit(c, h, which, shift, prefix, remain, copies=1):
    """From the counts in histogram ``h`` (the sum of its ``copies``), the
    digit at ``shift`` of the ``remain``-th largest key among those matching
    ``prefix``: the extended prefix, the count still to take and whether that
    is all of the digit's keys (1 or 0). In every wave: lane ``l`` scans
    digits ``255 - 4 l ..``; the one lane holding the digit packs the three."""
    lane = c.lane
    v = _lds_i4(h, 252 - 4 * lane)
    for k in range_constexpr(1, copies):
        w = _lds_i4(h + H0_STRIDE * k, 252 - 4 * lane)
        v = [a + b for a, b in zip(v, w)]
    cnt = [v[3], v[2], v[1], v[0]]  # digits 255 - 4 l, 254 - 4 l, ...
    tot = (cnt[0] + cnt[1]) + (cnt[2] + cnt[3])
    above = fx.coop.warp_inclusive_scan(tot, fx.ReductionOp.ADD, width=64) - tot
    pick = fx.Int32(0)
    for k in range_constexpr(4):
        hit = (above < remain) & (above + cnt[k] >= remain)
        whole = (above + cnt[k] == remain).select(fx.Int32(1 << 24), fx.Int32(0))
        pick = hit.select(whole | ((255 - 4 * lane - k) << 16) | (remain - above), pick)
        above = above + cnt[k]
    got = wave_umax_dpp(pick)
    return prefix | (((got >> 16) & 255) << shift), got & 0xFFFF, got >> 24


def _count(c, k0, hi, lo, k, prefix, mask):
    """Radix pass ``k``'s histogram: the keys matching ``prefix`` under
    ``mask`` (pass 0: every key but ``hi`` -1) counted by their digit; then
    the barrier, and the histogram two passes on cleared."""
    tid = fx.Int32(c.tid)
    which, shift = RADIX[k]
    h = k0 + H_WORDS * (k % 3)
    if const_expr(k == 0):
        h = k0 + (H0 + H0_STRIDE * (c.lane & (COPIES - 1)))
    for e in range_constexpr(len(hi)):
        m = hi[e] >= 0
        if const_expr(k > 0):
            m = ((hi[e] & _i32(mask[0])) == prefix[0]) & (
                (lo[e] & _i32(mask[1])) == prefix[1]
            )
        if m:
            d = ((hi[e], lo[e])[which] >> shift) & 255
            fx.atomic_add(h + d, fx.Int32(1), syncscope="workgroup")
    gpu.barrier()
    # The histogram of pass k + 2, read by every wave in pass k - 1.
    if const_expr(k + 2 < len(RADIX)):
        if tid < H_WORDS:
            fx.ptr_store(fx.Int32(0), k0 + (H_WORDS * ((k + 2) % 3) + tid))


_count = ASTRewriter.transform(_count)


def _select(c, k0, hi, lo):
    """The 512th largest of the keys ``(hi, lo)`` this CTA's threads hold,
    as ``[hi, lo]``; a key ``hi`` -1 holds no block, and no pass counts it
    (the first tests it, the others' prefixes clear its top bit). Once a
    pass takes its digit's keys whole, the rest are skipped: the prefix,
    its lower bits 0, keeps the same keys (the keys are unique).
    Histograms 0 and 1 and pass 0's copies cleared."""
    p0, p1 = fx.Int32(0), fx.Int32(0)
    mask = [0, 0]
    remain = fx.Int32(SEL)
    live = fx.Int32(1)  # the same in every thread
    for k in range_constexpr(len(RADIX)):
        which, shift = RADIX[k]
        h, copies = (k0 + H0, COPIES) if k == 0 else (k0 + H_WORDS * (k % 3), 1)
        if live != 0:
            _count(c, k0, hi, lo, k, [p0, p1], mask)
            got, remain, whole = _digit(
                c, h, which, shift, [p0, p1][which], remain, copies
            )
            if const_expr(which == 0):
                p0 = got
            else:
                p1 = got
            live = whole ^ 1
        mask[which] |= 0xFF << shift
    return [p0, p1]


_select = ASTRewriter.transform(_select)


def _slots(c, k0, keep):
    """This thread's first slot among the kept keys of the CTA, in thread
    order (a barrier)."""
    lane, wave = c.lane, c.wave
    n = fx.Int32(0)
    for kk in keep:
        n = n + kk.select(fx.Int32(1), fx.Int32(0))
    incl = fx.coop.warp_inclusive_scan(n, fx.ReductionOp.ADD, width=64)
    if lane == 63:
        fx.ptr_store(incl, k0 + (CNT + wave))
    gpu.barrier()
    before = fx.Int32(0)
    for w in range_constexpr(WAVES):
        before = before + (wave > w).select(fx.ptr_load(k0 + (CNT + w)), fx.Int32(0))
    return before + incl - n


_slots = ASTRewriter.transform(_slots)


def _kept(hi, lo, thr, every=None):
    """Which keys are among the 512 best: at or above ``thr``, or all
    (``every``; none with ``hi`` -1, below every threshold)."""
    if every is not None:
        thr = [every.select(fx.Int32(0), t) for t in thr]
    return [(h > thr[0]) | ((h == thr[0]) & (lo_ >= thr[1])) for h, lo_ in zip(hi, lo)]


def _put_key(k0, i, hi, lo, blk):
    """Candidate slot ``i`` in LDS: its 64-bit key ``hi << 16 | lo``, its block."""
    fx.ptr_store((hi << 16) | lo, k0 + (CK + 2 * i))
    fx.ptr_store(hi >> 16, k0 + (CK + 2 * i + 1))
    fx.ptr_store(blk, k0 + (CB + i))


def _scores(c, base, n, rq):
    """Thread ``tid``'s blocks ``base + tid + 512 e`` (``e`` <
    :data:`KEYS_CHUNK`; ``hi`` -1 from ``base + n``): keys and blocks."""
    tid = fx.Int32(c.tid)
    hi, lo, blk = [], [], []
    for e0 in range_constexpr(0, KEYS_CHUNK, Q4_BATCH):
        at = [fx.min(tid + THREADS * (e0 + k), n - 1) for k in range(Q4_BATCH)]
        got = mbx.poll([(c.mb_isc, mb_ix(base + a, rq), 1) for a in at], c.tag)
        for k in range_constexpr(Q4_BATCH):
            j = tid + THREADS * (e0 + k)
            hi.append((j < n).select(got[k][0] & 0x7FFFFFFF, fx.Int32(-1)))
            lo.append(0xFFFF - (base + j))
            blk.append(base + j)
    return hi, lo, blk


def _candidates(c, cb, rq):
    """Thread ``tid``'s candidates: slot ``tid`` of each chunk (``hi`` -1
    where the chunk has none; those poll chunk 0's, which always has it)."""
    tid = fx.Int32(c.tid)
    hi, lo, blk = [], [], []
    for e0 in range_constexpr(0, CHUNKS, 4):
        specs, have = [], []
        for e in range(e0, e0 + 4):
            have.append(tid < fx.min(cb - e * FAST_CHUNK, SEL))
            specs.append(
                (c.mb_fcand, mb_ix(2 * have[-1].select(e * SEL + tid, tid), rq), 2)
            )
        got = mbx.poll(specs, c.tag)
        for k in range_constexpr(4):
            hi.append(have[k].select(got[k][0], fx.Int32(-1)))
            blk.append(got[k][1])
            lo.append(0xFFFF - got[k][1])
    return hi, lo, blk


def _keep(c, k0, hi, lo, blk):
    """The 512 best of the held keys to LDS slots (:func:`_put_key`)."""
    thr = _select(c, k0, hi, lo)
    keep = _kept(hi, lo, thr)
    slot = _slots(c, k0, keep)
    for e in range_constexpr(len(hi)):
        if keep[e]:
            _put_key(k0, slot, hi[e], lo[e], blk[e])
        slot = slot + keep[e].select(fx.Int32(1), fx.Int32(0))


_keep = ASTRewriter.transform(_keep)


def _chunk(c, j, cb, busy, per, k0, rq):
    """Chunk task ``j``: the 512 best of blocks ``j FAST_CHUNK ..`` (all of
    them up to 512) to ``fcand``, then its hint."""
    tid = fx.Int32(c.tid)
    base = j * FAST_CHUNK
    n = fx.min(cb - base, FAST_CHUNK)
    _clear(c, k0)
    mbx.pre_poll(lambda m: (c.mb_fisch, mb_ix(fx.min(m, busy - 1), rq)), per, c.tag)
    hi, lo, blk = _scores(c, base, n, rq)
    thr = _select(c, k0, hi, lo)
    keep = _kept(hi, lo, thr, n <= SEL)
    slot = _slots(c, k0, keep)
    for e in range_constexpr(KEYS_CHUNK):
        if keep[e]:
            at = mb_ix(2 * (j * SEL + slot), rq)
            mbx.put2(c.mb_fcand, at, hi[e], blk[e], c.tag)
        slot = slot + keep[e].select(fx.Int32(1), fx.Int32(0))
    gpu.barrier()
    if tid == 0:
        mbx.put(c.mb_fch, mb_ix(j, rq), 1, c.tag)


_chunk = ASTRewriter.transform(_chunk)


def _xchg64(c, k0, key, stride, buf):
    """The int64 ``key`` of thread ``tid ^ stride``: in-wave (VALU swaps)
    below 64, else through LDS words ``buf ..`` (one key per thread, after a
    barrier)."""
    w = fx.Vector.from_elements([key], fx.Int64).bitcast(fx.Int32)
    if const_expr(stride >= 64):
        at = k0 + (buf + 2 * c.tid)
        fx.ptr_store(w[0], at)
        fx.ptr_store(w[1], at + 1)
        gpu.barrier()
        at = k0 + (buf + 2 * (c.tid ^ stride))
        got = [fx.ptr_load(at), fx.ptr_load(at + 1)]
    elif const_expr(stride >= 16):
        upper = (c.lane & stride) != 0
        got = []
        for k in range_constexpr(2):
            lo_, hi_ = _swap(w[k], stride)
            got.append(upper.select(lo_, hi_))
    else:
        got = [xshfl(w[k], stride) for k in range(2)]
    return fx.Vector.from_elements(got, fx.Int32).bitcast(fx.Int64)[0]


def _sort(c, k0, key):
    """The CTA's keys (one per thread: unique, or -1) largest first, thread
    ``tid`` getting the ``tid``-th: a bitonic sort, the exchanges across
    waves through LDS words 0 .. 1024 and ``CK ..`` in turn (free once the
    keys are loaded: every exchange's barrier follows the reads before)."""
    tid = fx.Int32(c.tid)
    bufs = (0, CK)
    n = 0
    size = 2
    while size <= THREADS:
        stride = size // 2
        while stride:
            other = _xchg64(c, k0, key, stride, bufs[n % 2])
            if stride >= 64:
                n += 1
            # The lower thread of a pair keeps the larger key in the blocks
            # sorting down (``tid & size`` 0: all of them at the last size).
            down = (
                (tid >> (stride.bit_length() - 1)) ^ (tid >> (size.bit_length() - 1))
            ) & 1
            up = (other > key).select(fx.Int32(1), fx.Int32(0))
            key = (up == (down ^ 1)).select(other, key)
            stride //= 2
        size *= 2
    return key


def _rank(c, k0, n, rq):
    """The ``n`` candidates in LDS to ``sel`` at their ranks (the keys
    above each): sorted (:func:`_sort`), thread ``tid`` puts the ``tid``-th
    one's block; then the hint."""
    tid = fx.Int32(c.tid)
    gpu.barrier()
    have = tid < n
    w0 = have.select(
        fx.ptr_load(k0 + (CK + 2 * tid)), fx.Int32(-1)
    )  # -1: below every key
    w1 = have.select(fx.ptr_load(k0 + (CK + 2 * tid + 1)), fx.Int32(-1))
    key = fx.Vector.from_elements([w0, w1], fx.Int32).bitcast(fx.Int64)[0]
    key = _sort(c, k0, key)
    if have:
        lo = fx.Vector.from_elements([key], fx.Int64).bitcast(fx.Int32)[0] & 0xFFFF
        mbx.put(c.mb_sel, mb_ix(tid, rq), 0xFFFF - lo, c.tag)
    gpu.barrier()
    if tid == 0:
        mbx.put(c.mb_selh, mb_ix(0, rq), 1, c.tag)


_rank = ASTRewriter.transform(_rank)


def _final(c, cb, busy, per, nch, k0, rq):
    """The final task: the 512 best of the chunks' candidates (past
    :data:`FAST_CHUNK` blocks) or of the scores, ranked to ``sel``."""
    tid = fx.Int32(c.tid)
    _clear(c, k0)
    if cb > FAST_CHUNK:
        mbx.pre_poll(lambda m: (c.mb_fch, mb_ix(fx.min(m, nch - 1), rq)), CHUNKS, c.tag)
        hi, lo, blk = _candidates(c, cb, rq)
        _keep(c, k0, hi, lo, blk)
    else:
        mbx.pre_poll(lambda m: (c.mb_fisch, mb_ix(fx.min(m, busy - 1), rq)), per, c.tag)
        if cb > SEL:
            hi, lo, blk = _scores(c, 0, cb, rq)
            _keep(c, k0, hi, lo, blk)
        else:
            if tid < cb:
                sv = mbx.poll([(c.mb_isc, mb_ix(tid, rq), 1)], c.tag)[0][0]
                _put_key(k0, tid, sv & 0x7FFFFFFF, 0xFFFF - tid, tid)
    _rank(c, k0, fx.min(cb, SEL), rq)


_final = ASTRewriter.transform(_final)


def q4(c, p):
    _mailboxes(c)
    t = task_index(c, "q4")
    k0 = c.lds_i + c.L.lds["q4_keys"]
    per = c.L.tasks["q3"] // c.rows  # Q3 tasks of a request row
    rq, j = _request(c, t, Q4_TASKS)
    L = row_len(p, rq)

    if t < c.L.tasks["q4"]:
        cb = (L + 1) // 4
        busy = fx.max(fx.min((cb + WAVES - 1) // WAVES, per), 1)  # Q3 tasks with blocks
        nch = (cb + FAST_CHUNK - 1) // FAST_CHUNK
        if j < CHUNKS:
            if (cb > FAST_CHUNK) & (j < nch):
                _chunk(c, j, cb, busy, per, k0, rq)
        else:
            _final(c, cb, busy, per, nch, k0, rq)


q4 = ASTRewriter.transform(q4)
