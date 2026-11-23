"""p1/L3-glue: drop-in for torch.cat(taps, dim=-1) of NT row-contiguous bf16 [M, H] taps -> [M, NT*H] (DSpark concat_taps).
A pure copy (bit-identical by construction): one program per (row, tap, 2048-column chunk), 16-byte vector loads/stores.
torch's CatArrayBatchedCopy_contig<OpaqueType<2>> moves 2-byte elements and costs 5.8-7.3 us per call (graph replay, 3 taps,
G1-16) against ~2.3 us for a same-size elementwise add."""
import torch
import triton
import triton.language as tl


@triton.jit
def _l3g_cat_rows_kernel(T0, T1, T2, T3, T4, T5, T6, T7, OUT, H, s0, s1, s2, s3, s4, s5, s6, s7, so,
                         NT: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nch = H // BLOCK
    row = pid // (NT * nch)
    rem = pid - row * (NT * nch)
    t = rem // nch
    c = (rem - t * nch) * BLOCK + tl.arange(0, BLOCK)
    row64 = row.to(tl.int64)
    v = tl.zeros((BLOCK,), dtype=tl.bfloat16)
    if t == 0:
        v = tl.load(T0 + row64 * s0 + c)
    if NT > 1:
        if t == 1:
            v = tl.load(T1 + row64 * s1 + c)
    if NT > 2:
        if t == 2:
            v = tl.load(T2 + row64 * s2 + c)
    if NT > 3:
        if t == 3:
            v = tl.load(T3 + row64 * s3 + c)
    if NT > 4:
        if t == 4:
            v = tl.load(T4 + row64 * s4 + c)
    if NT > 5:
        if t == 5:
            v = tl.load(T5 + row64 * s5 + c)
    if NT > 6:
        if t == 6:
            v = tl.load(T6 + row64 * s6 + c)
    if NT > 7:
        if t == 7:
            v = tl.load(T7 + row64 * s7 + c)
    tl.store(OUT + row64 * so + t * H + c, v)


def cat_rows(taps, out=None, block=2048):
    """torch.cat(taps, dim=-1) for 1 < len(taps) <= 8 bf16 [M, H] taps with unit column stride, H % block == 0."""
    nt = len(taps)
    m, h = taps[0].shape
    ok = (1 < nt <= 8 and all(t.dtype == torch.bfloat16 and t.dim() == 2 and t.shape == (m, h) and t.stride(1) == 1
                              for t in taps) and h % block == 0)
    if not ok:
        return torch.cat(tuple(taps), dim=-1)
    if out is None:
        out = torch.empty((m, nt * h), dtype=taps[0].dtype, device=taps[0].device)
    if m == 0:
        return out
    ts = list(taps) + [taps[0]] * (8 - nt)
    _l3g_cat_rows_kernel[(m * nt * (h // block),)](*ts, out, h, *[t.stride(0) for t in ts], out.stride(0),
                                                   NT=nt, BLOCK=block, num_warps=4)
    return out
