"""Small-batch top-k softmax router with fused sigmoid shared-expert gates.

Matches aiter ``topk_softmax(..., need_renorm=True, num_shared, "sigmoid")``:
routed weights are the renormalised softmax over the selected top-k logits,
ties go to the smaller expert id, and shared expert ``s`` gets
``sigmoid(logit[num_routed + s])`` in output column ``k + s``.

One wave handles one row. A bf16 logit is mapped to an order-preserving
16-bit key and packed with ``0xFFFF - expert`` so each selection round is a
single integer max reduction.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _topk_softmax_shared_kernel(
    logits_ptr,
    w_ptr,
    ids_ptr,
    stride_l,
    stride_w,
    stride_i,
    NUM_ROUTED: tl.constexpr,
    NUM_SHARED: tl.constexpr,
    TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, NUM_ROUTED)
    bits = tl.load(logits_ptr + row * stride_l + cols).to(tl.int16, bitcast=True)
    b = bits.to(tl.int32) & 0xFFFF
    # Negative floats: flip all bits; non-negative: set the sign bit.
    ordered = tl.where((b & 0x8000) != 0, b ^ 0xFFFF, b | 0x8000)
    key = (ordered << 16) | (0xFFFF - cols)
    key = key.to(tl.uint32, bitcast=True)

    out_k = tl.arange(0, 16)
    sel_keys = tl.zeros([16], dtype=tl.uint32)
    for k in tl.static_range(TOPK):
        best = tl.max(key, axis=0)
        sel_keys = tl.where(out_k == k, best, sel_keys)
        key = tl.where(key == best, 0, key)

    sel = sel_keys.to(tl.int32, bitcast=True)
    expert = 0xFFFF - (sel & 0xFFFF)
    ob = (sel >> 16) & 0xFFFF
    vb = tl.where((ob & 0x8000) != 0, ob ^ 0x8000, ob ^ 0xFFFF)
    val = (vb << 16).to(tl.float32, bitcast=True)
    valid = out_k < TOPK
    vmax = tl.max(tl.where(valid, val, float("-inf")), axis=0)
    e = tl.where(valid, tl.exp(val - vmax), 0.0)
    w = e / tl.sum(e, axis=0)
    tl.store(w_ptr + row * stride_w + out_k, w, mask=valid)
    tl.store(ids_ptr + row * stride_i + out_k, expert, mask=valid)

    sh = tl.arange(0, 16)
    sh_mask = sh < NUM_SHARED
    sl = tl.load(
        logits_ptr + row * stride_l + NUM_ROUTED + sh, mask=sh_mask, other=0.0
    ).to(tl.float32)
    tl.store(w_ptr + row * stride_w + TOPK + sh, tl.sigmoid(sl), mask=sh_mask)


def topk_softmax_shared(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    num_shared: int,
) -> None:
    rows, total = gating_output.shape
    num_routed = total - num_shared
    _topk_softmax_shared_kernel[(rows,)](
        gating_output,
        topk_weights,
        topk_ids,
        gating_output.stride(0),
        topk_weights.stride(0),
        topk_ids.stride(0),
        NUM_ROUTED=num_routed,
        NUM_SHARED=num_shared,
        TOPK=topk,
        num_warps=1,
    )


def topk_softmax_shared_supported(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_shared: int,
    scoring_func: str,
) -> bool:
    if gating_output.dim() != 2 or gating_output.dtype != torch.bfloat16:
        return False
    if gating_output.stride(1) != 1 or not renormalize or scoring_func != "sigmoid":
        return False
    num_routed = gating_output.shape[1] - num_shared
    return (
        0 < num_shared <= 16
        and 0 < topk <= 16
        and num_routed in (256, 512, 1024)
    )
