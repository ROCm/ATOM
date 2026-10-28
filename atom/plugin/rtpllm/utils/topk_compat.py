"""RTP-only compatibility for the pinned AITER top-k softmax signature.

RTP-LLM pins amd-aiter 0.1.23.dev134+g8449f416b, whose topk_softmax has
the seven-argument ABI. Remove this shim once RTP-LLM ships an AITER wheel
whose topk_softmax accepts the eighth num_valid_rows argument.
"""

from __future__ import annotations

import inspect
from functools import lru_cache

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=2)
def _accepts_valid_rows(topk_softmax) -> bool:
    try:
        return "num_valid_rows" in inspect.signature(topk_softmax).parameters
    except (TypeError, ValueError):
        # Older compiled wrappers can lack a Python text signature. They use
        # the seven-argument ABI, so mask Graph padding after that call.
        return False


@triton.jit
def _mask_padding(
    weights,
    ids,
    valid_rows,
    stride_weights,
    stride_ids,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= tl.load(valid_rows):
        columns = tl.arange(0, BLOCK)
        mask = columns < TOPK
        tl.store(weights + row * stride_weights + columns, 0.0, mask=mask)
        tl.store(ids + row * stride_ids + columns, -1, mask=mask)


def rtp_topk_softmax(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_fused_shared_experts: int = 0,
    num_routing_experts: int = 0,
    fused_shared_experts_scoring_func: str | None = None,
    num_valid_rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter import topk_softmax

    from atom.model_ops import topK

    del num_routing_experts
    token = gating_output.shape[0]
    if num_fused_shared_experts > 0:
        total_weights, total_ids = topK.topk_meta_rows(token)
        weights, ids = total_weights, total_ids[:, :topk]
    else:
        ids = torch.empty((token, topk), dtype=torch.int32, device=gating_output.device)
        weights = torch.empty(
            (token, topk), dtype=torch.float32, device=gating_output.device
        )
    token_ids = torch.empty(
        (token, topk), dtype=torch.int32, device=gating_output.device
    )
    shared = (
        num_fused_shared_experts if fused_shared_experts_scoring_func is not None else 0
    )
    scoring = fused_shared_experts_scoring_func or ""
    args = (weights, ids, token_ids, gating_output, renormalize, shared, scoring)
    if _accepts_valid_rows(topk_softmax):
        topk_softmax(*args, num_valid_rows)
    else:
        topk_softmax(*args)
        if num_valid_rows is not None:
            _mask_padding[(token,)](
                weights,
                ids,
                num_valid_rows,
                weights.stride(0),
                ids.stride(0),
                TOPK=topk,
                BLOCK=triton.next_power_of_2(topk),
            )
    if num_fused_shared_experts > 0:
        return total_weights, total_ids
    return weights, ids


def install_rtp_topk_compat() -> None:
    """Patch RTP's process at model build; retain native behavior elsewhere."""
    from atom.model_ops import topK
    from atom.plugin.prepare import is_rtpllm

    original = topK.rocm_aiter_topk_softmax_impl
    if getattr(original, "_rtp_topk_compat", False):
        return

    def dispatch(*args, **kwargs):
        if is_rtpllm():
            return rtp_topk_softmax(*args, **kwargs)
        return original(*args, **kwargs)

    dispatch._rtp_topk_compat = True
    topK.rocm_aiter_topk_softmax_impl = dispatch
