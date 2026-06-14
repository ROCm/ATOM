from __future__ import annotations

from typing import Optional

import torch


def _is_stream_capturing() -> bool:
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def _is_uniform_full_graph_batch() -> bool:
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    if not is_forward_context_available():
        return False
    forward_context = get_forward_context()
    batch_descriptor = forward_context.batch_descriptor
    return (
        forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
        and batch_descriptor is not None
        and batch_descriptor.uniform
    )


def _try_get_exact_valid_rows(dispatch_recv_token_num: torch.Tensor) -> Optional[int]:
    if dispatch_recv_token_num.numel() == 0 or _is_stream_capturing():
        return None
    return int(dispatch_recv_token_num.reshape(-1)[0].item())


def trim_vllm_mori_dispatch_tensors(
    dispatch_a1: torch.Tensor,
    dispatch_scale: torch.Tensor | None,
    dispatch_ids: torch.Tensor,
    dispatch_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    ep_world_size: int,
    dispatch_recv_token_num: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    # Only trim in full-cudagraph uniform-decode settings.
    # All DP/TP ranks are padded to a common token count only under full-graph
    # settings. In piecewise or eager batches, token counts per rank can differ
    if _is_uniform_full_graph_batch() and ep_world_size > 0:
        num_local_tokens, topk = topk_ids.shape[0], topk_ids.shape[1]
        valid_rows = num_local_tokens * topk * ep_world_size
    else:
        exact = _try_get_exact_valid_rows(dispatch_recv_token_num)
        if exact is None:
            return dispatch_a1, dispatch_scale, dispatch_ids, dispatch_weights
        valid_rows = exact

    valid_rows = max(0, min(valid_rows, dispatch_a1.shape[0]))
    if valid_rows == 0 or valid_rows >= dispatch_a1.shape[0]:
        return dispatch_a1, dispatch_scale, dispatch_ids, dispatch_weights

    dispatch_a1 = dispatch_a1[:valid_rows]
    dispatch_ids = dispatch_ids[:valid_rows]
    dispatch_weights = dispatch_weights[:valid_rows]
    if dispatch_scale is not None:
        dispatch_scale = dispatch_scale[:valid_rows]
    return dispatch_a1, dispatch_scale, dispatch_ids, dispatch_weights
