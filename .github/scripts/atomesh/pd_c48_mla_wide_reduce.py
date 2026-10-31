"""Wide-grid replacement for aiter's persistent-MLA split reduce (kn_mla_reduce_v1) at <= 16 output rows.

aiter launches heads x min(qlen, ...) workgroups that each walk every split serially, so at batch 1 with 256
splits (C1 decode) the reduce is latency-bound (~44 us/layer on MI355X). This kernel takes the same metadata
(reduce_indptr / reduce_final_map / reduce_partial_map) and the same fp32 LSE merge, but one program per
(head x 32-wide dv slice, q row, tile) that loads up to 256 splits in one chunk (~12-14 us at 32K-384K).
"""
import torch
import triton
import triton.language as tl

# (BLOCK_D, BLOCK_S, num_warps); MI355X qlen 8: 3x faster at bs 1, 1.2x at bs 2, even at bs 4, slower at bs 16-32.
BLOCKS = (32, 256, 4)
GATE_ROWS = 16


@triton.jit
def _wide_reduce_kernel(partial_o, partial_lse, reduce_indptr, reduce_final_map, reduce_partial_map, out, out_lse,
                        stride_s_o, stride_h_o, H: tl.constexpr, DV: tl.constexpr, BLOCK_D: tl.constexpr,
                        BLOCK_S: tl.constexpr, HAS_LSE: tl.constexpr):
    # workgroups go round-robin over the 8 XCDs by linear id, so the (mostly dead) tile axis must vary slowest
    hd = tl.program_id(0)
    sq = tl.program_id(1)
    tile = tl.program_id(2)
    head = hd // (DV // BLOCK_D)
    dblk = hd % (DV // BLOCK_D)
    start = tl.load(reduce_indptr + tile)
    n = tl.load(reduce_indptr + tile + 1) - start
    first = tl.load(reduce_partial_map + start, mask=n > 0, other=-1)
    q_start = tl.load(reduce_final_map + tile * 2)
    q_end = tl.load(reduce_final_map + tile * 2 + 1)
    seq = q_start + sq
    # n == 1 with slot -1: the decode kernel already wrote this tile's output
    if (n < 1) | ((n == 1) & (first == -1)) | (seq >= q_end):
        return
    offs_s = tl.arange(0, BLOCK_S)
    offs_d = dblk * BLOCK_D + tl.arange(0, BLOCK_D)
    m = float("-inf")
    total = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for s0 in range(0, n, BLOCK_S):
        mask = s0 + offs_s < n
        slot = tl.load(reduce_partial_map + start + s0 + offs_s, mask=mask, other=0).to(tl.int64)
        row = (slot + sq) * H + head
        lse = tl.load(partial_lse + row, mask=mask, other=float("-inf"))
        p = tl.load(partial_o + row[:, None] * DV + offs_d[None, :], mask=mask[:, None], other=0.0)
        m_new = tl.maximum(m, tl.max(lse, 0))
        m_ref = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m - m_ref)
        w = tl.where(mask, tl.exp(lse - m_ref), 0.0)
        acc = acc * alpha + tl.sum(p * w[:, None], 0)
        total = total * alpha + tl.sum(w, 0)
        m = m_new
    # aiter: zero/NaN sum -> lse +inf and every scale 0
    empty = (total == 0.0) | (total != total)
    g_lse = tl.where(empty, float("inf"), tl.log(total) + m)
    o = tl.where(empty, 0.0, acc / total)
    tl.store(out + seq.to(tl.int64) * stride_s_o + head * stride_h_o + offs_d, o.to(out.dtype.element_ty))
    if HAS_LSE:
        if dblk == 0:
            tl.store(out_lse + seq.to(tl.int64) * H + head, g_lse.to(out_lse.dtype.element_ty))


def supported(partial_output, partial_lse, final_output, final_lse, max_seqlen_q) -> bool:
    return (final_output.shape[0] <= GATE_ROWS
            and partial_output.dtype == torch.float32 and partial_lse.dtype == torch.float32
            and partial_output.is_contiguous() and partial_lse.is_contiguous()
            and final_output.dim() == 3 and final_output.stride(-1) == 1
            and partial_output.shape[-2] == final_output.shape[1]
            and partial_output.shape[-1] == final_output.shape[2] and final_output.shape[2] % 128 == 0
            and (final_lse is None or final_lse.is_contiguous())
            and isinstance(max_seqlen_q, int) and 1 <= max_seqlen_q <= 64)


def wide_reduce(partial_output, partial_lse, reduce_indptr, reduce_final_map, reduce_partial_map, max_seqlen_q,
                final_output, final_lse):
    total_s, h, dv = final_output.shape
    block_d, block_s, warps = BLOCKS
    # every tile owns >= 1 output row, so total_s bounds the live tiles without reading the metadata on the host
    tiles = min(reduce_final_map.shape[0], total_s)
    grid = (h * (dv // block_d), max_seqlen_q, tiles)
    _wide_reduce_kernel[grid](partial_output, partial_lse, reduce_indptr, reduce_final_map, reduce_partial_map,
                              final_output, final_lse if final_lse is not None else final_output,
                              final_output.stride(0), final_output.stride(1), H=h, DV=dv, BLOCK_D=block_d,
                              BLOCK_S=block_s, HAS_LSE=final_lse is not None, num_warps=warps)


def install(mla):
    orig = mla._mla_decode_reduce_v1_dispatch

    def dispatch(partial_output, partial_lse, reduce_indptr, reduce_final_map, reduce_partial_map,
                 max_seqlen_q, num_kv_splits, final_output, final_lse):
        if reduce_final_map is not None and supported(partial_output, partial_lse, final_output, final_lse,
                                                      max_seqlen_q):
            return wide_reduce(partial_output, partial_lse, reduce_indptr, reduce_final_map,
                               reduce_partial_map, max_seqlen_q, final_output, final_lse)
        return orig(partial_output, partial_lse, reduce_indptr, reduce_final_map, reduce_partial_map,
                    max_seqlen_q, num_kv_splits, final_output, final_lse)

    mla._mla_decode_reduce_v1_dispatch = dispatch
