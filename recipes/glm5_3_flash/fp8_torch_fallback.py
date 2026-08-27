"""Pure-torch stand-in for the `finegrained-fp8` hub Triton kernel.

The hub kernel (`kernels-community/finegrained-fp8`) loads fine on ROCm but its
Triton compile asserts on gfx950:

    llvm/ADT/Sequence.h:275: iota_range(T, T, bool): Assertion `Begin <= End' failed.

This module installs a torch-only bundle in its place. It dequantises the block-FP8
weights on the fly and runs an ordinary matmul -- slow, but this is only used to
produce a correctness oracle, so being obviously-right matters more than being fast.

Install by calling `install()` before the first forward.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _dequant(weight: torch.Tensor, scale_inv: torch.Tensor, block_size) -> torch.Tensor:
    """Expand block scales over a (possibly batched) FP8 weight.

    `weight` is [..., N, K] FP8; `scale_inv` is [..., ceil(N/bn), ceil(K/bk)] fp32.
    The trailing block may be partial, hence the crop after repeat_interleave.
    """
    bn, bk = block_size if block_size is not None else (128, 128)
    n, k = weight.shape[-2], weight.shape[-1]
    scale = scale_inv.to(torch.float32)
    scale = scale.repeat_interleave(bn, dim=-2).repeat_interleave(bk, dim=-1)
    scale = scale[..., :n, :k]
    return (weight.to(torch.float32) * scale).to(torch.bfloat16)


def _matmul(
    input,
    weight,
    weight_scale_inv,
    block_size=None,
    out_dtype=None,
    activation_scale=None,
):
    """2-D FP8 linear: [..., K] @ [N, K]^T -> [..., N]."""
    if activation_scale is not None:
        raise NotImplementedError(
            "static activation scales are not used by GLM-5.3-Flash"
        )
    out_dtype = out_dtype or input.dtype
    w = _dequant(weight, weight_scale_inv, block_size)
    return F.linear(input.to(torch.bfloat16), w).to(out_dtype)


def _batched_matmul(x, weight, weight_scale, block_size=None, expert_ids=None):
    """Per-token expert-indexed matmul: x [S, K], weight [E, N, K] -> [S, N].

    `expert_ids` may contain EP sentinels (>= E); the real kernel leaves those rows
    uninitialised and the caller masks them, so anything finite is fine here.
    """
    num_experts, n, _ = weight.shape
    out = x.new_zeros((x.shape[0], n), dtype=torch.bfloat16)
    valid = expert_ids < num_experts
    for e in torch.unique(expert_ids[valid]).tolist():
        rows = (expert_ids == e).nonzero(as_tuple=True)[0]
        w = _dequant(weight[e], weight_scale[e], block_size)
        out[rows] = F.linear(x[rows].to(torch.bfloat16), w)
    return out.to(x.dtype)


def _grouped_matmul(
    x, w, scale_inv, offsets=None, tokens_per_expert=None, block_size=None, **_
):
    """Grouped matmul over expert-sorted rows: x [S, K], w [E, N, K] -> [S, N].

    `offsets` is the int32 cumsum of `tokens_per_expert`; rows past `offsets[-1]`
    are EP sentinels the real kernel skips, so they stay zero here.
    """
    n = w.shape[-2]
    out = x.new_zeros((x.shape[0], n), dtype=torch.bfloat16)
    ends = offsets.tolist()
    start = 0
    for e, end in enumerate(ends):
        end = int(end)
        if end > start:
            weight = _dequant(w[e], scale_inv[e], block_size)
            out[start:end] = F.linear(x[start:end].to(torch.bfloat16), weight)
        start = end
    return out.to(x.dtype)


def install() -> None:
    from transformers.integrations import finegrained_fp8 as ff

    ff._FINEGRAINED_FP8 = ff.FineGrainedFP8(
        matmul=_matmul,
        batched_matmul=_batched_matmul,
        grouped_matmul=_grouped_matmul,
    )
    print("[fp8-fallback] installed torch-only finegrained-fp8 bundle", flush=True)
