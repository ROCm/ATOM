from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class MoriDispatchRuntimeMeta:
    exact_valid_rows: Optional[int]


def _is_stream_capturing() -> bool:
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def _try_get_exact_valid_rows(dispatch_recv_token_num: torch.Tensor) -> Optional[int]:
    if dispatch_recv_token_num.numel() == 0 or _is_stream_capturing():
        return None
    return int(dispatch_recv_token_num.reshape(-1)[0].item())


def get_mori_dispatch_runtime_meta(
    dispatch_recv_token_num: torch.Tensor,
) -> MoriDispatchRuntimeMeta:
    return MoriDispatchRuntimeMeta(
        exact_valid_rows=_try_get_exact_valid_rows(dispatch_recv_token_num),
    )


def trim_vllm_mori_dispatch_tensors(
    dispatch_a1: torch.Tensor,
    dispatch_scale: torch.Tensor | None,
    dispatch_ids: torch.Tensor,
    dispatch_weights: torch.Tensor,
    dispatch_recv_token_num: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    meta = get_mori_dispatch_runtime_meta(dispatch_recv_token_num)
    if meta.exact_valid_rows is None:
        # During stream capture MORI's exact recv rows are not host-readable.
        # Keep the fixed-capacity receive buffers intact and rely on
        # `dispatch_recv_token_num` as the valid-prefix contract for fused MoE
        return dispatch_a1, dispatch_scale, dispatch_ids, dispatch_weights

    valid_rows = meta.exact_valid_rows
    valid_rows = max(0, min(valid_rows, dispatch_a1.shape[0]))
    if valid_rows == 0 or valid_rows >= dispatch_a1.shape[0]:
        return dispatch_a1, dispatch_scale, dispatch_ids, dispatch_weights

    dispatch_a1 = dispatch_a1[:valid_rows]
    dispatch_ids = dispatch_ids[:valid_rows]
    dispatch_weights = dispatch_weights[:valid_rows]
    if dispatch_scale is not None:
        dispatch_scale = dispatch_scale[:valid_rows]
    return dispatch_a1, dispatch_scale, dispatch_ids, dispatch_weights
