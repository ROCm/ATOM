"""atom-vllm plugin patches for the MORI all-to-all MoE path.

The native fused-moe code (``atom/model_ops/fused_moe/``) is frontend-agnostic:
it makes no ``is_vllm()`` decision and pulls in nothing from ``atom.plugin``.
The two places where atom-vllm needs different behavior are isolated behind
overridable methods and injected here, so native files stay clean:

* ``MoriPrepareAndFinalize._get_dispatch_config`` (MORI launch config) -- vLLM
  has no stable prefill/decode flag at that call site, so select by a
  token-count threshold instead.
* ``FusedMoEModularKernel._maybe_trim_dispatch_output`` (dispatch-buffer trim)
  -- vLLM DP+EP mixed batches need an exact received-token trim; the native
  graph_bs bound under-counts recv on a decoding rank and reads past the
  buffer -> illegal memory access.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

import atom.model_ops.fused_moe.modular_kernel as mk
from atom.model_ops.fused_moe.mori_prepare_finalize import MoriPrepareAndFinalize
from atom.plugin.config import VLLM_MORI_LAUNCH_CONFIG_TOKEN_THRESHOLD
from aiter.jit.utils.chip_info import get_cu_num

_MORI_PATCH_APPLIED = False


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


def apply_vllm_mori_patch() -> None:
    """Monkeypatch the MORI MoE seams with atom-vllm-specific behavior."""
    global _MORI_PATCH_APPLIED
    if _MORI_PATCH_APPLIED:
        return

    original_get_dispatch_config = MoriPrepareAndFinalize._get_dispatch_config

    @functools.wraps(original_get_dispatch_config)
    def vllm_get_dispatch_config(self, num_tokens=None):
        # vLLM does not expose a stable prefill/decode flag here, so use a
        # token-count threshold to keep MORI warmup and runtime selection
        # deterministic in atom-vllm mode.
        assert (
            num_tokens is not None
        ), "num_tokens is required to choose MORI launch config in vLLM mode."
        # Cap block_num at the device CU count: mori's IntraNode grid-wide
        # barrier requires all gridDim.x blocks co-resident; >CU blocks (e.g.
        # 128 on the 80-CU MI308X) deadlock at warmup. Mirrors the native
        # MoriPrepareAndFinalize._get_dispatch_config cap.
        mp = get_cu_num()
        if num_tokens >= VLLM_MORI_LAUNCH_CONFIG_TOKEN_THRESHOLD:
            return min(128, mp), 16
        return min(64, mp), 4

    setattr(vllm_get_dispatch_config, "_atom_vllm_mori_patched", True)
    MoriPrepareAndFinalize._get_dispatch_config = vllm_get_dispatch_config

    original_trim = mk.FusedMoEModularKernel._maybe_trim_dispatch_output

    @functools.wraps(original_trim)
    def vllm_maybe_trim_dispatch_output(
        self,
        dispatch_a1,
        dispatch_scale,
        dispatch_ids,
        dispatch_weights,
        topk_ids,
        expert_tokens_meta,
    ):
        # Exact-recv trim. trim_vllm_mori_dispatch_tensors trims to the
        # graph_bs*topk*ep bound only under a uniform FULL-cudagraph batch
        # (where that bound >= recv by construction), skips trimming during
        # graph capture, and otherwise trims to the exact received-token count.
        return trim_vllm_mori_dispatch_tensors(
            dispatch_a1,
            dispatch_scale,
            dispatch_ids,
            dispatch_weights,
            topk_ids,
            self.prepare_finalize.num_dispatchers(),
            expert_tokens_meta.expert_num_tokens,
        )

    setattr(vllm_maybe_trim_dispatch_output, "_atom_vllm_mori_patched", True)
    mk.FusedMoEModularKernel._maybe_trim_dispatch_output = (
        vllm_maybe_trim_dispatch_output
    )

    _MORI_PATCH_APPLIED = True
