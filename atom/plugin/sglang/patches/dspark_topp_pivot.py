# SPDX-License-Identifier: MIT
"""Exact top-p pivots for SGLang's ``top_p_renorm_probs_triton`` without a
full-vocabulary sort and without a host synchronisation.

SGLang 0.5.22 (ROCm) renormalises a ``[R, V]`` fp32 probability table for
top-p as

    sorted = torch.sort(probs, -1).values          # ascending
    cdf = torch.cumsum(sorted, -1)                  # fp32
    cut = torch.searchsorted(cdf, (1.0 - top_p)[:, None], right=False)
    pivot = sorted[cut.clamp(max=V - 1)]
    out = _renorm_from_pivots(probs, pivot)         # keep probs >= pivot, renormalise

The output depends on ``probs`` only through ``pivot``: the same pivot,
renormalised by the same kernel, gives bitwise the same table. This module
finds that pivot. It reproduces PyTorch's fp32 arithmetic, not an
exact-arithmetic idealisation of it:

* ``torch.cumsum`` along the last dim of a 2-D CUDA/HIP tensor (PyTorch
  ``ScanUtils.cuh``, ``tensor_kernel_scan_innermost_dim``) walks each row in
  chunks of ``C = 2 * num_threads_x`` -- 1024 for up to 256 rows of
  V = 129280, 512 for 257..1024 rows, 256 from 1025 (``torch_scan_chunk``;
  this module serves up to ``MAX_ROWS`` = 512) -- adds the
  running total to the chunk's first entry and runs a Sklansky scan in the
  chunk. ``_sklansky`` performs the same additions in the same order.
* ``torch.searchsorted`` is ``lower_bound`` with the test ``cdf[mid] >= v``;
  it is replayed probe by probe, so even a non-monotone fp32 cdf gives the
  same index.

Fast path (``_topp_head_kernel``, one program per row): the top ``K`` = 1280
values (``torch.topk``) are the last ``K`` ascending positions and, because
``(V - K) % C == 0``, whole chunks. The running total entering them is the
cumsum of the other ``V - K`` values, which is not computed: their exact sum
``S`` (fp64) and a rigorous bound ``E`` on PyTorch's fp32 rounding of it give
an interval. Every addition is monotone in the running total, so the head
chunks scanned from both ends of the interval bracket each head cdf value. A
row is settled when the interval's top is below ``1 - top_p`` (every probe
below the head goes right) and each probe that lands in the head is decided
by both ends of its bracket; its pivot is the head value at the found index.
``top_p >= 1`` (threshold <= 0) settles to the row minimum, as the reference
finds index 0.

Fallback (``_topp_fallback_kernel``, one program per row, returns at once for
a settled row): sorts the row (bitonic, padded to 2**17), replays the chunked
cumsum from zero and the probes exactly. It serves only rows the head did not
settle -- flat rows whose cut lies below the head, thresholds inside the
rounding bracket -- at about 5 ms of one compute unit per row. Nothing reads
the flags on the host.

Rounding bound (first order in u = 2**-24, doubled): the running total after
chunk ``c`` takes ``L + 1`` additions on partial sums no larger than the exact
prefix ``S_(c+1)C`` (``L = log2 C``), on top of pairwise block sums of depth
<= ``L - 1``; the sum of the ``c * C`` smallest values is at most ``c / n_c``
of the sum ``S`` of all ``n_c`` chunks below the head. Hence
``E <= 2 u S ((L + 1)(n_c + 1) / 2 + 3 L + 2)`` plus ``1e-9 * row_sum`` for
the fp64 sums: ``E ~ 8.6e-5 * S`` for C = 1024, S being the probability mass
outside the top 1280.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

HEAD_K = 1280
# torch's scan chunk for V = 129280 is 1024 up to 256 rows, 512 for 257..1024
# rows and 256 from 1025; 512 is a conservative cap (85 requests x 6 rows), not
# a requirement of the replay.
MAX_ROWS = 512
_LOGT = 12  # bitonic tile 4096
FALLBACK_WARPS = 16
_U = 2.0**-24


def torch_scan_chunk(num_rows: int, row_size: int) -> int:
    """``2 * num_threads_x`` of PyTorch's innermost-dim scan for a
    ``[num_rows, row_size]`` tensor (``get_log_num_threads_x_inner_scan``,
    uint32 arithmetic)."""
    lx = 0
    while (1 << lx) < row_size:
        lx += 1
    ly = 0
    while (1 << ly) < num_rows:
        ly += 1
    v = ((9 + lx - ly) & 0xFFFFFFFF) // 2
    return 2 << min(max(4, v), 9)


def rounding_coef(chunk: int, row_size: int, head_k: int) -> float:
    lg = chunk.bit_length() - 1
    n_c = (row_size - head_k) // chunk
    return 2.0 * _U * ((lg + 1) * (n_c + 1) / 2.0 + 3 * lg + 2)


def supports(num_rows: int, row_size: int, head_k: int = HEAD_K) -> bool:
    """Whether the exact pivots below cover a ``[num_rows, row_size]`` table."""
    if num_rows < 2 or num_rows > MAX_ROWS or row_size <= head_k:
        return False
    if row_size > (1 << 17) or row_size < (1 << _LOGT):
        return False
    c = torch_scan_chunk(num_rows, row_size)
    return c in (512, 1024) and (row_size - head_k) % c == 0


@triton.jit
def _sklansky(v, C: tl.constexpr, L: tl.constexpr):
    # PyTorch's in-chunk Sklansky scan of v (the running total already added to v[0]): at
    # level m every entry in the upper half of an aligned 2**(m+1) block adds the last entry
    # of the lower half, buf[ti] + buf[si] -- the same additions in the same order.
    for m in tl.static_range(L):
        v3 = tl.permute(tl.reshape(v, [C >> (m + 1), 2, 1 << m]), [0, 2, 1])
        lo, hi = tl.split(v3)
        sel = tl.arange(0, 1 << m)[None, :] == (1 << m) - 1
        last = tl.max(tl.where(sel, lo, float("-inf")), axis=1)
        v = tl.reshape(tl.permute(tl.join(lo, hi + last[:, None]), [0, 2, 1]), [C])
    return v


@triton.jit
def _pick(vec, idx, pos):
    # vec[pos] for a runtime scalar pos, exact (one selected entry, the rest -inf)
    return tl.max(tl.where(idx == pos, vec, float("-inf")))


@triton.jit
def _chunk_cdf(x, carry, lidx, C: tl.constexpr, L: tl.constexpr):
    # torch.cumsum over one chunk: the running total goes into the first entry, then Sklansky
    return _sklansky(tl.where(lidx == 0, x + carry, x), C, L)


@triton.jit
def _topp_head_kernel(
    probs_ptr,
    head_ptr,
    top_ps_ptr,
    pivot_ptr,
    flag_ptr,
    cnt_ptr,
    rows_per_req,
    coef,
    V: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    NHC: tl.constexpr,
    NHCP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    r64 = row.to(tl.int64)
    probs = probs_ptr + r64 * V
    head = head_ptr + r64 * K
    thr = 1.0 - tl.load(top_ps_ptr + row // rows_per_req)

    acc = tl.zeros([BLOCK], dtype=tl.float64)
    mn = tl.full([BLOCK], float("inf"), tl.float32)
    for b in range(0, V, BLOCK):
        idx = b + tl.arange(0, BLOCK)
        msk = idx < V
        xv = tl.load(probs + idx, mask=msk, other=0.0)
        acc += xv.to(tl.float64)
        mn = tl.minimum(mn, tl.where(msk, xv, float("inf")))
    total = tl.sum(acc)

    if thr <= 0.0:
        # every cdf entry is >= 0 >= threshold: lower_bound returns 0, the row minimum
        tl.store(pivot_ptr + row, tl.min(mn))
        tl.store(flag_ptr + row, 0)
        return

    jj = tl.arange(0, KP)
    hsum = tl.sum(tl.load(head + jj, mask=jj < K, other=0.0).to(tl.float64))
    rest = tl.maximum(total - hsum, 0.0)
    err = rest * coef + 1e-9 * total
    lo32 = (rest - err).to(tl.float32)
    hi32 = (rest + err).to(tl.float32)
    lo_bits = lo32.to(tl.int32, bitcast=True) - 1
    carry_lo = tl.where(lo32 > 0.0, lo_bits.to(tl.float32, bitcast=True), 0.0)
    carry_hi = (hi32.to(tl.int32, bitcast=True) + 1).to(tl.float32, bitcast=True)

    if carry_hi >= thr:
        # the cut may lie below the head
        tl.store(flag_ptr + row, 1)
        tl.atomic_add(cnt_ptr + 0, 1)
        return

    lidx = tl.arange(0, C)
    hrow = tl.arange(0, NHCP)[:, None]
    codes = tl.zeros([NHCP, C], dtype=tl.int32)
    clo = carry_lo
    chi = carry_hi
    for h in tl.static_range(NHC):
        j = h * C + lidx
        x = tl.load(head + (K - 1 - j), mask=j < K, other=0.0)  # ascending
        vlo = _chunk_cdf(x, clo, lidx, C, L)
        vhi = _chunk_cdf(x, chi, lidx, C, L)
        code = tl.where(vlo >= thr, 1, tl.where(vhi < thr, 0, 2))
        codes = tl.where(hrow == h, code[None, :], codes)
        clo = _pick(vlo, lidx, C - 1)
        chi = _pick(vhi, lidx, C - 1)

    lcol = lidx[None, :]
    start = 0
    end = V
    und = 0
    for _ in tl.static_range(18):
        act = start < end
        mid = start + ((end - start) >> 1)
        q = mid - (V - K)
        hit = (hrow == q // C) & (lcol == q % C)
        code = tl.where(act & (q >= 0), tl.max(tl.where(hit, codes, -1)), 0)
        und = und | (code == 2).to(tl.int32)
        start = tl.where(act & (code == 0), mid + 1, start)
        end = tl.where(act & (code == 1), mid, end)
    cut = tl.minimum(start, V - 1)
    if und != 0:
        tl.store(flag_ptr + row, 1)
        tl.atomic_add(cnt_ptr + 1, 1)
    else:
        tl.store(pivot_ptr + row, tl.load(head + (V - 1 - cut)))
        tl.store(flag_ptr + row, 0)


@triton.jit
def _signed_sort(x, desc, dim: tl.constexpr):
    # ascending, or descending when desc; exact for values in [0, +inf]
    y = tl.where(desc, -x, x)
    y = tl.sort(y, dim)
    return tl.where(desc, -y, y)


@triton.jit
def _topp_fallback_kernel(
    probs_ptr,
    top_ps_ptr,
    pivot_ptr,
    flag_ptr,
    buf_ptr,
    car_ptr,
    cnt_ptr,
    rows_per_req,
    V: tl.constexpr,
    LOGN: tl.constexpr,
    LOGT: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    NC: tl.constexpr,
    CARS: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(flag_ptr + row) == 0:
        return
    tl.atomic_add(cnt_ptr + 2, 1)
    r64 = row.to(tl.int64)
    N: tl.constexpr = 1 << LOGN
    T: tl.constexpr = 1 << LOGT
    NT: tl.constexpr = N // T
    src = probs_ptr + r64 * V
    buf = buf_ptr + r64 * N
    car = car_ptr + r64 * CARS
    thr = 1.0 - tl.load(top_ps_ptr + row // rows_per_req)

    # bitonic sort of the row, ascending, padded with +inf to 2**LOGN
    offs = tl.arange(0, T)
    for t in range(NT):
        idx = t * T + offs
        xs = tl.load(src + idx, mask=idx < V, other=float("inf"))
        xs = _signed_sort(xs, (t & 1) == 1, 0)
        tl.store(buf + idx, xs)
    tl.debug_barrier()
    for lv in tl.static_range(LOGT + 1, LOGN + 1):
        # strides 2**(lv-1) .. T: each strided group is bitonic; sorting it is the merge
        for it in range(NT):
            seg = it // (1 << (lv - LOGT))
            lo0 = (it % (1 << (lv - LOGT))) * (T >> (lv - LOGT))
            gi = tl.arange(0, 1 << (lv - LOGT))[None, :]
            wi = tl.arange(0, T >> (lv - LOGT))[:, None]
            idx2 = seg * (1 << lv) + gi * T + lo0 + wi
            xs2 = tl.load(buf + idx2)
            xs2 = _signed_sort(xs2, ((seg & 1) == 1) & (lv != LOGN), 1)
            tl.store(buf + idx2, xs2)
        tl.debug_barrier()
        for t in range(NT):
            idx = t * T + offs
            xs = tl.load(buf + idx)
            xs = _signed_sort(xs, (((t >> (lv - LOGT)) & 1) == 1) & (lv != LOGN), 0)
            tl.store(buf + idx, xs)
        tl.debug_barrier()

    # torch.cumsum's running total entering each chunk
    lidx = tl.arange(0, C)
    carry = tl.min(tl.zeros([C], dtype=tl.float32))
    for c in range(NC):
        tl.store(car + c, carry)
        x = tl.load(buf + c * C + lidx, mask=c * C + lidx < V, other=0.0)
        carry = _pick(_chunk_cdf(x, carry, lidx, C, L), lidx, C - 1)
    tl.debug_barrier()

    # torch.searchsorted's lower_bound, probe by probe, on the exact cdf
    start = 0
    end = V
    for _ in tl.static_range(18):
        act = start < end
        mid = tl.where(act, start + ((end - start) >> 1), 0)
        c = mid // C
        x = tl.load(buf + c * C + lidx, mask=c * C + lidx < V, other=0.0)
        val = _pick(_chunk_cdf(x, tl.load(car + c), lidx, C, L), lidx, mid % C)
        ge = val >= thr
        start = tl.where(act & ~ge, mid + 1, start)
        end = tl.where(act & ge, mid, end)
    cut = tl.minimum(start, V - 1)
    tl.store(pivot_ptr + row, tl.load(buf + cut))


class ToppPivotCounters:
    """Device-side counts, never read with a host sync: [0] rows whose cut may
    lie below the head, [1] rows whose threshold fell inside the head's
    rounding bracket, [2] rows the fallback sorted. ``calls`` counts
    ``exact_top_p_pivots`` calls on the host; ``budget_ms`` is a host-side
    running total the caller may add to (the patch's guard does)."""

    def __init__(self, device):
        self.dev = torch.zeros(4, dtype=torch.int64, device=device)
        self._host = torch.zeros(4, dtype=torch.int64, pin_memory=True)
        self._pending = None  # (event, calls, budget_ms) when the copy was issued
        self.last = None  # (calls, budget_ms, counts) of the newest completed copy
        self.calls = 0
        self.budget_ms = 0.0

    def poll(self):
        """Publish a finished copy, start a new one if none is in flight, and
        return ``last``. Only queries an event; never waits for the device."""
        if self._pending is not None:
            event, calls, budget = self._pending
            if not event.query():
                return self.last
            self.last = (calls, budget, [int(v) for v in self._host.tolist()])
            self._pending = None
        self._host.copy_(self.dev, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._pending = (event, self.calls, self.budget_ms)
        return self.last


def exact_top_p_pivots(
    probs: torch.Tensor,
    top_ps: torch.Tensor,
    rows_per_req: int,
    counters: ToppPivotCounters | None = None,
    head_k: int = HEAD_K,
) -> torch.Tensor:
    """The pivots ``top_p_renorm_probs_triton(probs, top_ps.repeat_interleave(
    rows_per_req))`` derives, bitwise. ``probs``: contiguous fp32 ``[R, V]``
    CUDA tensor with ``supports(R, V)``; ``top_ps``: fp32 ``[R // rows_per_req]``."""
    R, V = probs.shape
    C = torch_scan_chunk(R, V)
    L = C.bit_length() - 1
    dev = probs.device
    if counters is None:
        cnt = torch.zeros(4, dtype=torch.int64, device=dev)
    else:
        cnt = counters.dev
    head = torch.topk(probs, head_k, dim=-1, sorted=True).values
    pivots = torch.empty(R, dtype=torch.float32, device=dev)
    flags = torch.empty(R, dtype=torch.int32, device=dev)
    kp = 1 << (head_k - 1).bit_length()
    nhc = (head_k + C - 1) // C
    _topp_head_kernel[(R,)](
        probs,
        head,
        top_ps,
        pivots,
        flags,
        cnt,
        rows_per_req,
        rounding_coef(C, V, head_k),
        V=V,
        K=head_k,
        KP=kp,
        C=C,
        L=L,
        NHC=nhc,
        NHCP=1 << (nhc - 1).bit_length(),
        BLOCK=4096,
        num_warps=8,
    )
    logn = (V - 1).bit_length()
    nc = (V + C - 1) // C
    buf = torch.empty((R, 1 << logn), dtype=torch.float32, device=dev)
    car = torch.empty((R, nc), dtype=torch.float32, device=dev)
    _topp_fallback_kernel[(R,)](
        probs,
        top_ps,
        pivots,
        flags,
        buf,
        car,
        cnt,
        rows_per_req,
        V=V,
        LOGN=logn,
        LOGT=_LOGT,
        C=C,
        L=L,
        NC=nc,
        CARS=nc,
        num_warps=FALLBACK_WARPS,
    )
    if counters is not None:
        counters.calls += 1
    return pivots
