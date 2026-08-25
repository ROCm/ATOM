# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Inference-only KDA forward using ATOM's checkpoint-aware state kernel.

The preprocessing and output kernels remain upstream FLA kernels. Only the
recurrent state scan is local, because that is where prefix-cache snapshots
must be scattered from the live fp32 accumulator. Keeping the public call
compatible with ``fla.ops.kda.chunk_kda`` lets Kimi-K3 switch at the existing
opaque custom-op boundary without tracing this stateful path.
"""

from __future__ import annotations

import warnings

import torch

from fla.modules.l2norm import l2norm_fwd
from fla.ops.common.gate import fused_beta_sigmoid
from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.index import prepare_chunk_indices

from .chunk_delta_h_kda import chunk_gated_delta_rule_fwd_h


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    cp_context=None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    chunk_size: int = 64,
    ckpt: torch.Tensor | None = None,
    ckpt_slots: torch.LongTensor | None = None,
    ckpt_base: torch.LongTensor | None = None,
    ckpt_every: int = 0,
    **kwargs,
):
    """Run chunked KDA and optionally scatter recurrent-state checkpoints.

    This is an inference path: backward state is intentionally not retained.
    Arguments through ``cp_context`` mirror upstream FLA's public API; the four
    ``ckpt*`` arguments are ATOM extensions.
    """
    if "transpose_state_layout" in kwargs:
        if state_v_first:
            raise ValueError(
                "Cannot pass both `state_v_first` and "
                "the deprecated `transpose_state_layout`."
            )
        warnings.warn(
            "`transpose_state_layout` is deprecated and renamed to " "`state_v_first`.",
            DeprecationWarning,
            stacklevel=2,
        )
        state_v_first = kwargs.pop("transpose_state_layout")
    if kwargs:
        raise TypeError(f"unexpected KDA arguments: {sorted(kwargs)}")
    if cp_context is not None:
        raise NotImplementedError(
            "ATOM's checkpoint-aware KDA path does not support CP"
        )
    if disable_recompute:
        raise ValueError("disable_recompute is a training option, not supported here")

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                "The batch size must be 1 for flattened variable-length input, "
                f"got {q.shape[0]}."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                "initial_state sequence count must match cu_seqlens: "
                f"{initial_state.shape[0]} vs {len(cu_seqlens) - 1}"
            )
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be float32"

    if chunk_size not in (32, 64):
        raise ValueError(f"chunk_size must be 32 or 64, got {chunk_size}")
    if safe_gate and use_gate_in_kernel:
        if lower_bound is None:
            raise ValueError("lower_bound is required when safe_gate=True")
        if not -5 <= lower_bound < 0:
            raise ValueError(f"lower_bound must be in [-5, 0), got {lower_bound}")
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError(
            "allow_neg_eigval=True requires use_beta_sigmoid_in_kernel=True"
        )

    B, T, H, K, HV = *q.shape, v.shape[2]
    assert q.shape == k.shape
    assert K <= 256
    assert HV % H == 0
    assert g.shape == (B, T, HV, K)
    assert beta.shape == (B, T, HV)
    if scale is None:
        scale = K**-0.5

    if use_qk_l2norm_in_kernel:
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)
    if use_beta_sigmoid_in_kernel:
        beta = fused_beta_sigmoid(beta, scale=2.0 if allow_neg_eigval else 1.0)

    chunk_indices = None
    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(
            cu_seqlens,
            chunk_size,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )

    if use_gate_in_kernel:
        assert A_log is not None, "A_log is required when use_gate_in_kernel=True"
        g_cumsum = kda_gate_chunk_cumsum(
            g=g,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=RCP_LN2,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            lower_bound=lower_bound,
        )
    else:
        g_cumsum = chunk_local_cumsum(
            g=g,
            scale=RCP_LN2,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    w, u, _, kg, Aqk, _ = chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g_cumsum,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
        disable_recompute=False,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g_cumsum,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        state_v_first=state_v_first,
        ckpt=ckpt,
        ckpt_slots=ckpt_slots,
        ckpt_base=ckpt_base,
        ckpt_every=ckpt_every,
    )
    o = chunk_gla_fwd_o_gk(
        q=q,
        v=v_new,
        g=g_cumsum,
        A=Aqk,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
    ).type_as(q)
    if return_intermediate_states:
        assert (
            torch.is_inference_mode_enabled()
        ), "return_intermediate_states is only allowed in inference mode"
        return o, final_state, h
    return o, final_state
