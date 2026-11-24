# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The contract of ``top_p_mask``: vLLM's split top-p pipeline (``v1/sample/ops/topk_topp_triton.py``
``_apply_topp_split``: ``_topp_sb_stats_kernel``, five ``_topp_sb_step_kernel`` rounds, ``_topp_sb_mask_kernel``) on
fp32 logits rows of vocab 200064 with top-p only (no top-k), bit for bit as Triton 3.8 builds it for gfx950 (8
waves of 64; the vLLM ROCm nightly image).

Per row (p < 1), S slices of L = 200064 / S columns (S 32 at <= 8 rows, else 16):
1. Slice stats over 8192-column chunks: running max m, ``exp_sum = exp_sum * rescale + sum(e)`` (``rescale = exp(m -
   m_new)``, ``e = exp(x - m_new)`` of finite x), min and count of finite x.
2. Row stats: M = max m_s, Z = sum(exp_sum_s * exp(m_s - M)), the min and the count; rows with Z <= 0, no finite
   value or ``1/Z <= max(exp(mn - M)/Z, 1e-30) * 1.00000095`` stay as they are.
3. Five rounds: 8 pivots ``lo * exp(log(hi / lo) * (f + 1) / 9)`` between (lo, hi) = (min prob, max prob); each slice
   sums, per 2048-column chunk, the probabilities ``exp(x - M) / Z`` above each pivot (mass), their minimum and the
   count within 1e-9 of it, merged chunk by chunk; the slices combine into the bracket update or the final pivot.
4. Mask: keep ``prob > pivot`` or ``x == M``; boundary duplicates keep the first ``numkeep`` in column order (the
   lower slices' share from the finalizing round's partials).

The fp32 operations and their association are Triton's: ``tl.exp(y)`` is ``exp2(y * log2e)`` with the
denormal-range fma (``exp_f32``), ``/`` is IEEE division (torch divides a tensor by a Python scalar as a multiply by
its reciprocal, so every divisor here is a tensor), ``tl.log`` is LLVM's ``llvm.log.f32`` expansion
(``log_f32``), each ``tl.sum`` is the xor butterfly of its layout (``*_OFFSETS``: the in-thread pairs, the wave's
DPP rows and row broadcasts, the waves), and the two products the compiler fuses into an fma stay fused: the
combine's ``mass - gmin * nmin`` and, at S 16, the first level of Z (a lane adds its own unrounded product to its
partner's rounded one). So at S 16 the lanes hold one of two Z values (lane bit 3, ``GROUP_BITS``) and everything
derived from Z (probabilities, lo / hi, the pivots) follows the lane that computes it: a round compares a column's
probability (its lane in the 4-column load layout) with the pivots of its lane in the reduction layout. The pipeline
runs on the GPU in torch, the fmas exactly on the host (``top_p_mask_reference``); tests compare exactly. Rows with
p >= 1 are not touched.
"""

from fractions import Fraction

import torch

V, F, ROUNDS = 200064, 8, 5
SPLIT_MAX_BATCH = 64
_LOG2E = 1.4426950408889634  # fp32 0x3FB8AA3B
# tl.sum butterflies (xor offsets in order) of each layout:
# stats chunk of 8192, column c: thread (c % 2048) // 4, register 4 (c // 2048) + c % 4
STATS_OFFSETS = (2, 2048, 4096, 1, 32, 16, 8, 4, 64, 128, 1024, 512, 256)
# round chunk of 2048, column c: thread c % 512, register c // 512
STEP_OFFSETS = (1024, 512, 8, 4, 2, 1, 16, 32, 256, 128, 64)
# the slices' combine of each round
COMBINE_OFFSETS = {16: (4, 2, 1, 8), 32: (4, 2, 1, 16, 8)}
# Z: S 32 is a plain butterfly (16, 8, 4, 2, 1); S 16 fuses its first level (8) and then sums 4, 2, 1 per group
Z_OFFSETS = {16: (4, 2, 1), 32: (16, 8, 4, 2, 1)}
# Z group of a column offset c in its slice: bit 5 of c in the 4-column load layouts (round loads, mask), bit 3 in
# a round's reduction layout (the pivots)
GROUP_BITS = {"load": 5, "pivot": 3}


def splits_for(rows: int, vocab: int = V, num_cu: int = 256) -> int:
    """``_topp_split_count``: slices per row."""
    s = 1
    while s < 32 and (s * 2 * rows <= num_cu or s * 2 * 7168 <= vocab):
        s *= 2
    return s


def _f32(x: float) -> float:
    return float(torch.tensor(x, dtype=torch.float32))


def _round_f32(q: Fraction) -> float:
    """``q`` rounded to the nearest fp32 (ties to even); finite results only."""
    if q == 0:
        return 0.0
    sign, q = (-1.0, -q) if q < 0 else (1.0, q)
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if Fraction(2) ** e > q:
        e -= 1
    ulp = Fraction(2) ** max(e - 23, -149)
    n, rem = divmod(q, ulp)
    if rem * 2 > ulp or (rem * 2 == ulp and n % 2 == 1):
        n += 1
    return sign * float(n * ulp)


def fma_f32(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``a * b + c`` with one fp32 rounding (finite inputs), elementwise on the host."""
    a, b, c = torch.broadcast_tensors(a, b, c)
    out = [_round_f32(Fraction(x) * Fraction(y) + Fraction(z))
           for x, y, z in zip(a.flatten().tolist(), b.flatten().tolist(), c.flatten().tolist())]
    return torch.tensor(out, dtype=torch.float32, device=a.device).view(a.shape)


def exp_f32(d: torch.Tensor) -> torch.Tensor:
    """Triton's fp32 ``tl.exp`` on gfx950: ``llvm.exp2.f32(d * log2e)``; below 2^-126 the expansion adds 64 in one fma
    with the product, then scales the result by 2^-64."""
    t = d * _f32(_LOG2E)
    small = t < -126.0
    adj = (d.double() * _f32(_LOG2E) + 64.0).float()
    r = torch.exp2(torch.where(small, adj, t))
    return torch.where(small, r * 2.0**-64, r)


_LN2_HI = Fraction(float(torch.tensor(0x3F317217, dtype=torch.int32).view(torch.float32)))
_LN2_LO = Fraction(float(torch.tensor(0x3377D1CF, dtype=torch.int32).view(torch.float32)))


def log_f32(x: torch.Tensor) -> torch.Tensor:
    """Triton's fp32 ``tl.log`` on gfx950 (LLVM's ``llvm.log.f32`` expansion with fp contraction) for normal x > 0:
    ``y = v_log_f32(x)`` (torch's log2 on the GPU), ``r = y * ln2_hi``, then ``fma(y, ln2_hi, fma(y, ln2_lo,
    fma(y, ln2_hi, -r)))`` with ln2 as 0x3f317217 + 0x3377d1cf; the fmas are rounded exactly on the host."""
    assert bool((x >= 2.0**-126).all()) and bool(torch.isfinite(x).all())
    out = []
    for y in torch.log2(x).flatten().tolist():
        y = Fraction(y)
        r = Fraction(_round_f32(y * _LN2_HI))
        f0 = Fraction(_round_f32(y * _LN2_HI - r))
        f1 = Fraction(_round_f32(y * _LN2_LO + f0))
        out.append(_round_f32(y * _LN2_HI + f1))
    return torch.tensor(out, dtype=torch.float32, device=x.device).view(x.shape)


def butterfly(v: torch.Tensor, offsets) -> torch.Tensor:
    """Sum over the last dim by a xor butterfly: per offset o, ``v[i] = v[i] + v[i ^ o]``; the value of element 0."""
    idx = torch.arange(v.shape[-1], device=v.device)
    for o in offsets:
        v = v + v[..., idx ^ o]
    return v[..., 0]


def _ladder(lo, hi):
    """[..., F] pivots ``lo * exp(log(hi / lo) * (f + 1) / 9)``."""
    fi = torch.arange(1, F + 1, device=lo.device, dtype=torch.float32)
    y = log_f32(hi / lo)[..., None] * fi
    return lo[..., None] * exp_f32(y / torch.full_like(y, 9.0))  # a tensor divisor (a scalar one is a reciprocal)


def slice_stats(x: torch.Tensor, S: int):
    """[n, S] running max, exp_sum, min, count (``_topp_sb_stats_kernel``)."""
    n = x.shape[0]
    L = V // S
    nch = -(-L // 8192)
    pad = torch.full((n, S, nch * 8192), float("-inf"), device=x.device)
    pad[..., :L] = x.view(n, S, L)
    m = torch.full((n, S), float("-inf"), device=x.device)
    es = torch.zeros(n, S, device=x.device)
    mn = torch.full((n, S), float("inf"), device=x.device)
    cnt = torch.zeros(n, S, device=x.device)
    for q in range(nch):
        xc = pad[..., q * 8192:(q + 1) * 8192]
        fin = xc > float("-inf")
        nm = torch.maximum(m, xc.amax(-1))
        rescale = torch.where(m == float("-inf"), torch.zeros_like(m), exp_f32(m - nm))
        e = torch.where(fin, exp_f32(xc - nm[..., None]), torch.zeros_like(xc))
        es = es * rescale + butterfly(e, STATS_OFFSETS)
        m = nm
        mn = torch.minimum(mn, torch.where(fin, xc, torch.full_like(xc, float("inf"))).amin(-1))
        cnt = cnt + fin.sum(-1).float()
    return m, es, mn, cnt


def row_z(m: torch.Tensor, es: torch.Tensor, S: int):
    """(M [n], Z [n, 2]): Z of the lanes of group 0 and group 1 (equal at S 32)."""
    M = m.amax(1)
    w = torch.where(m == float("-inf"), torch.zeros_like(m), exp_f32(m - M[:, None]))
    prod = es * w
    if S == 32:
        z = butterfly(prod, Z_OFFSETS[32])
        return M, torch.stack([z, z], 1)
    l1 = fma_f32(es, w, prod[:, torch.arange(16, device=m.device) ^ 8])
    return M, torch.stack([butterfly(l1[:, :8], Z_OFFSETS[16]), butterfly(l1[:, 8:], Z_OFFSETS[16])], 1)


def _group(S: int, which: str, device) -> torch.Tensor:
    """[V] Z group of each column (by its offset in the slice); all 0 at S 32."""
    if S == 32:
        return torch.zeros(V, dtype=torch.long, device=device)
    off = torch.arange(V, device=device) % (V // S)
    return (off >> GROUP_BITS[which]) & 1


def round_parts(prob_load: torch.Tensor, piv: torch.Tensor, gpiv: torch.Tensor, S: int):
    """One round's slice partials mass, minl, nmin [n, S, F] (2048-column chunks merged in column order).
    ``prob_load`` [n, V]: each column's probability (its load lane's Z); ``piv`` [n, 2, F] per group; ``gpiv`` [V]."""
    n = prob_load.shape[0]
    L = V // S
    nq = -(-L // 2048)
    pad = torch.zeros(n, S, nq * 2048, device=prob_load.device)
    pad[..., :L] = prob_load.view(n, S, L)
    gp = torch.zeros(S, nq * 2048, dtype=torch.long, device=prob_load.device)
    gp[:, :L] = gpiv.view(S, L)
    mass = torch.zeros(n, S, F, device=prob_load.device)
    minl = torch.ones(n, S, F, device=prob_load.device)
    nmin = torch.zeros(n, S, F, device=prob_load.device)
    eps = _f32(1e-9)
    for q in range(nq):
        p2 = pad[..., None, q * 2048:(q + 1) * 2048]  # [n, S, 1, 2048]
        g = gp[None, :, None, q * 2048:(q + 1) * 2048]
        pv = torch.where(g == 1, piv[:, 1, None, :, None], piv[:, 0, None, :, None])  # [n, S, F, 2048]
        above = p2 > pv
        mass = mass + butterfly(torch.where(above, p2, torch.zeros_like(p2)), STEP_OFFSETS)
        tmin = torch.where(above, p2, torch.full_like(p2, float("inf"))).amin(-1)
        tcnt = (above & ((p2 - tmin[..., None]).abs() < eps)).sum(-1).float()
        is_new = tmin < minl
        is_same = (tmin - minl).abs() < eps
        nmin = torch.where(is_new, tcnt, nmin + torch.where(is_same, tcnt, torch.zeros_like(tcnt)))
        minl = torch.minimum(minl, tmin)
    return mass, minl, nmin


class _State:
    """Per row and Z group [n, 2]: the search state ``_topp_sb_combine`` carries."""

    def __init__(self, lo, hi):
        z = torch.zeros_like(lo)
        self.lo, self.hi = lo, hi
        self.best, self.best_mass, self.best_minl, self.best_nmin = z - 1, z.clone(), z + 1, z.clone()
        self.done, self.nomask, self.pivot = z.clone(), z.clone(), z.clone()
        self.dup, self.numdup, self.numkeep = z + 1, z + 1, z + 1
        self.fin, self.best_fin = z - 1, z - 1


def _combine(st, live, mass_s, minl_s, nmin_s, Z, p, S, is_last, j):
    """One ``_topp_sb_combine`` on the (row, group) pairs ``live``, in place."""
    eps = _f32(1e-9)
    mass = butterfly(mass_s.transpose(1, 2), COMBINE_OFFSETS[S])  # [n, F]
    gmin = minl_s.amin(1)
    nmin = torch.where((minl_s - gmin[:, None]).abs() < eps, nmin_s, torch.zeros_like(nmin_s)).sum(1)
    pivs = _ladder(st.lo, st.hi)  # [n, 2, F]
    fidx = torch.arange(F, device=p.device)
    pc = p[:, None]
    ok = (mass >= pc) & (fma_f32(-gmin, nmin, mass) < pc)
    ok_idx = torch.where(ok, fidx, -1).amax(1)[:, None].expand(-1, 2)  # [n, 2]
    ge = (mass >= pc)[:, None, :].expand(-1, 2, -1)
    ge_idx = torch.where(ge, fidx, -1).amax(2)
    take = lambda t, i: t.gather(-1, i.clamp(min=0)[..., None])[..., 0]  # noqa: E731
    mass2, gmin2, nmin2 = (t[:, None, :].expand(-1, 2, -1) for t in (mass, gmin, nmin))
    bp = take(pivs, ge_idx)
    upd = live & (ge_idx >= 0) & (bp > st.best)
    st.best = torch.where(upd, bp, st.best)
    st.best_mass = torch.where(upd, take(mass2, ge_idx), st.best_mass)
    st.best_minl = torch.where(upd, take(gmin2, ge_idx), st.best_minl)
    st.best_nmin = torch.where(upd, take(nmin2, ge_idx), st.best_nmin)
    st.best_fin = torch.where(upd, (j * F + ge_idx).float(), st.best_fin)
    hit = live & (ok_idx >= 0)
    new_lo = torch.maximum(st.lo, torch.where(ge, pivs, torch.full_like(pivs, float("-inf"))).amax(2))
    new_hi = torch.minimum(st.hi, torch.where(~ge, pivs, torch.full_like(pivs, float("inf"))).amin(2))
    collapsed = torch.ones_like(hit) if is_last else ~(new_hi > new_lo * _f32(1.0 + 1e-7))
    stop = live & ~hit & collapsed
    fall = stop & (st.best > 0.0)
    zero, one = torch.zeros_like(st.best), torch.ones_like(st.best)
    pivot = torch.where(hit, take(pivs, ok_idx), torch.where(fall, st.best, zero))
    sel_mass = torch.where(hit, take(mass2, ok_idx), torch.where(fall, st.best_mass, zero))
    dup = torch.where(hit, take(gmin2, ok_idx), torch.where(fall, st.best_minl, one))
    numdup = torch.where(hit, take(nmin2, ok_idx), torch.where(fall, st.best_nmin, one))
    chosen = hit | fall
    nomask = torch.where(chosen, (pivot * Z >= 1.0).float(), (stop & ~fall).float())
    q = torch.clamp(sel_mass - pc, min=0.0) / dup
    qi = torch.nan_to_num(q, nan=0.0).clamp(0, 4294967295.0).to(torch.int64).float()  # v_cvt_u32_f32 saturates
    numkeep = torch.where(chosen, torch.minimum(torch.maximum(numdup - qi, one), numdup), one)
    adv = live & ~hit
    st.lo = torch.where(adv, new_lo, st.lo)
    st.hi = torch.where(adv, new_hi, st.hi)
    st.fin = torch.where(hit, (j * F + ok_idx).float(), st.fin)
    for name, val in (("done", (hit | stop).float()), ("nomask", nomask), ("pivot", pivot), ("dup", dup),
                      ("numdup", numdup), ("numkeep", numkeep)):
        setattr(st, name, torch.where(live, val, getattr(st, name)))


def top_p_mask_reference(logits: torch.Tensor, p: torch.Tensor, debug: dict | None = None) -> torch.Tensor:
    """fp32 [n, 200064] logits (1 <= n <= 64), fp32 [n] top-p -> masked logits (a new tensor; -inf outside the kept
    set). ``debug`` gets the slice stats and every round's slice partials (Triton's STATS / PARTS layout) and the
    rounds each row ran."""
    n = logits.shape[0]
    assert logits.dtype == torch.float32 and logits.shape[1] == V and 1 <= n <= SPLIT_MAX_BATCH
    S = splits_for(n)
    L = V // S
    x = logits.contiguous()
    p = p.float().contiguous()
    m, es, mn_s, cnt_s = slice_stats(x, S)
    M, Z = row_z(m, es, S)  # Z [n, 2]
    mn = mn_s.amin(1)
    cnt = cnt_s.sum(1)
    hi = torch.ones_like(Z) / Z
    lo = torch.clamp(exp_f32(mn - M)[:, None] / Z, min=_f32(1e-30))
    run = ((p < 1.0) & ~(cnt < 1.0))[:, None] & ~(Z <= 0.0) & ~(hi <= lo * _f32(1.0 + 1e-6))
    if debug is not None:
        debug["S"] = S
        debug["stats"] = torch.stack([m, es, mn_s, cnt_s], -1)
        debug["parts"] = torch.zeros(n, ROUNDS, S, F, 3, device=x.device)
        debug["rounds"] = torch.zeros(n, dtype=torch.int32, device=x.device)
        debug["divergent"] = torch.zeros(n, dtype=torch.bool, device=x.device)
    st = _State(lo, hi)
    prob = exp_f32(x - M[:, None])[:, None, :] / Z[..., None]  # [n, 2, V]
    gload, gpiv = _group(S, "load", x.device), _group(S, "pivot", x.device)
    prob_load = torch.where(gload[None] == 1, prob[:, 1], prob[:, 0])
    parts = []
    for j in range(ROUNDS):
        live = run & (st.done == 0.0)
        if debug is not None:
            debug["divergent"] |= live[:, 0] != live[:, 1]
        if not bool(live.any()):
            break
        mass_s, minl_s, nmin_s = round_parts(prob_load, _ladder(st.lo, st.hi), gpiv, S)
        parts.append((minl_s, nmin_s))
        if debug is not None:
            lr = live.any(1)
            debug["parts"][:, j] = torch.where(lr[:, None, None, None], torch.stack([mass_s, minl_s, nmin_s], -1), 0.0)
            debug["rounds"] += lr.int()
        _combine(st, live, mass_s, minl_s, nmin_s, Z, p, S, j == ROUNDS - 1, j)
    # the mask kernel: per column, the state of its lane's group
    col = lambda t: torch.where(gload[None] == 1, t[:, 1, None], t[:, 0, None])  # noqa: E731  [n, 2] -> [n, V]
    masked = col((run & (st.nomask == 0.0)).float()) > 0
    keep = (prob_load > col(st.pivot)) | (x == M[:, None])
    ties = col((st.numkeep < st.numdup).float()) > 0
    if bool((ties & masked).any()):
        fin = torch.where(st.fin < 0, st.best_fin, st.fin)  # [n, 2]
        fr = (fin / F).to(torch.int64).clamp(0, len(parts) - 1)
        fs = (fin - fr.float() * F).to(torch.int64).clamp(0, F - 1)
        minl_all = torch.stack([pm for pm, _ in parts], 1)  # [n, rounds, S, F]
        nmin_all = torch.stack([pn for _, pn in parts], 1)
        ar = torch.arange(n, device=x.device)[:, None]
        mf = minl_all[ar, fr].gather(3, fs[:, :, None, None].expand(n, 2, S, 1))[..., 0]  # [n, 2, S]
        nf = nmin_all[ar, fr].gather(3, fs[:, :, None, None].expand(n, 2, S, 1))[..., 0]
        dupc = torch.where((mf - st.dup[..., None]).abs() < _f32(1e-9), nf, torch.zeros_like(nf))
        preceding = dupc.cumsum(2) - dupc  # [n, 2, S] exact small integers
        sl = torch.arange(V, device=x.device) // L
        prec = torch.where(gload[None] == 1, preceding[:, 1, sl], preceding[:, 0, sl])  # [n, V]
        dmask = prob_load == col(st.dup)
        rank = dmask.view(n, S, L).int().cumsum(2).view(n, V).float() + prec
        keep_dup = dmask & (rank <= col(st.numkeep))
        keep = torch.where(ties, keep & (~dmask | keep_dup), keep)
    return torch.where(masked & ~keep, torch.full_like(x, float("-inf")), x)
