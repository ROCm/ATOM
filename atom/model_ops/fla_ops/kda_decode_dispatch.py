# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Conservative Kimi-KDA adapter for AITER's gfx1250 write-back decode."""

import logging
from functools import lru_cache

import torch

from atom.model_ops.fla_ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update as _atom_decode,
)
from atom.utils import envs

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_aiter_decode():
    # Checking the public function alone is insufficient: older AITER versions
    # have it but lack the write-back path's negative-slot/no-write contract.
    try:
        from aiter.ops.triton._gluon_kernels.gfx1250.attention import kda_decode
        from aiter.ops.triton.attention.kda import fused_recurrent_kda
        from aiter.ops.triton.attention.kda_writeback import try_writeback

        if not callable(try_writeback) or not hasattr(
            kda_decode, "kda_writeback_decode_kernel"
        ):
            return None
        return fused_recurrent_kda
    except (ImportError, AttributeError):
        logger.warning(
            "AITER KDA write-back unavailable; retaining ATOM KDA decode",
            exc_info=True,
        )
        return None


def _same_index_view(a, b):
    # ATOM slices read/write indices separately, even without a state fork.
    # Compare metadata only; torch.equal()/item() would synchronize the GPU.
    return a is None or (
        b is not None
        and a.device == b.device
        and a.dtype == b.dtype
        and a.shape == b.shape
        and a.stride() == b.stride()
        and a.data_ptr() == b.data_ptr()
    )


def _supported(q, k, v, a, b, A_log, dt_bias, state, starts, slots, out):
    if any(x is None for x in (state, starts, slots, A_log, dt_bias)):
        return False
    if not q.is_cuda or q.ndim != 4:
        return False
    arch = torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0]
    if arch not in ("gfx1250", "gfx1250-strict"):
        return False
    n = q.shape[1]
    if not (
        n > 0
        and q.shape == k.shape == v.shape == a.shape == (1, n, 12, 128)
        and b.shape == (1, n, 12)
        and state.ndim == 4
        and state.shape[1:] == (12, 128, 128)
        and state.stride()[1:] == (16384, 128, 1)
        and state.stride(0) % 128 == 0
        and state.dtype == torch.float32
        and all(x.dtype == torch.bfloat16 for x in (q, k, v, a, b))
        and all(x.is_contiguous() for x in (q, k, v, A_log, dt_bias, starts))
        and a.stride()[1:] == (1536, 128, 1)
        and b.stride(-1) == 1
        and A_log.numel() == 12
        and dt_bias.numel() == 1536
        and A_log.dtype == dt_bias.dtype == torch.float32
        and starts.dtype == slots.dtype == torch.int32
        and starts.ndim == slots.ndim == 1
        and starts.numel() == n + 1
        and slots.numel() == n
    ):
        return False
    tensors = (q, k, v, a, b, A_log, dt_bias, state, starts, slots)
    if not all(x.device == q.device for x in tensors):
        return False
    if any(x.data_ptr() % 16 for x in (q, k, v, a, dt_bias, state)):
        return False
    return out is None or (
        out.shape == (n, 12, 128)
        and out.device == q.device
        and out.dtype == v.dtype
        and out.is_contiguous()
        and out.data_ptr() % 16 == 0
        and all(
            out.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
            for x in tensors
        )
    )


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
    ssm_state_indices_in: torch.Tensor | None = None,
    lower_bound: float | None = None,
):
    if (
        envs.ATOM_USE_AITER_KDA_WRITEBACK
        and is_kda
        and lower_bound == -5.0
        and beta == 1.0
        and threshold == 20.0
        and (scale is None or scale == 128**-0.5)
        and inplace_final_state
        and use_qk_l2norm_in_kernel
        and num_accepted_tokens is None
        and _same_index_view(ssm_state_indices_in, ssm_state_indices)
        and _supported(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            initial_state,
            cu_seqlens,
            ssm_state_indices,
            o,
        )
    ):
        decode = _get_aiter_decode()
        if decode is not None:
            # Do not catch launch failures: state may already have been updated.
            result, final_state = decode(
                q=q,
                k=k,
                v=v,
                g=a,
                beta=b,
                A_log=A_log,
                dt_bias=dt_bias,
                initial_state=initial_state,
                scale=scale,
                inplace_final_state=True,
                state_v_first=True,
                cu_seqlens=cu_seqlens,
                ssm_state_indices=ssm_state_indices,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                lower_bound=-5.0,
                cache_state_updates=False,
                pad_slot_guard=False,
                out=None if o is None else o.unsqueeze(0),
            )
            return (result.squeeze(0) if o is None else o), final_state

    return _atom_decode(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        o=o,
        beta=beta,
        threshold=threshold,
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=inplace_final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        is_kda=is_kda,
        ssm_state_indices_in=ssm_state_indices_in,
        lower_bound=lower_bound,
    )
