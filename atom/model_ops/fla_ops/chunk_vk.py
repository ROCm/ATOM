# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import warnings

import torch

# vk variant of chunk_gated_delta_rule. Ported verbatim from vLLM upstream
# (vllm.model_executor.layers.fla.ops.chunk). Differs from ATOM's existing
# chunk.py only in the per-head [V, K] (vk) state layout vs ATOM's existing
# [K, V] (kv) layout. The prologue kernels (cumsum, KKT, solve_tril,
# recompute_w_u, l2norm) are layout-agnostic and shared with the kv path.
# Only chunk_delta_h and chunk_o have layout-sensitive code, so we import
# the vk variants of those two.
from .chunk_delta_h_vk import chunk_gated_delta_rule_fwd_h_vk
from .chunk_o_vk import chunk_fwd_o_vk
from .chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from .cumsum import chunk_local_cumsum
from .l2norm import l2norm_fwd
from .solve_tril import solve_tril
from .utils import SUPPRESS_LEVEL, input_guard
from .wy_fast import recompute_w_u_fwd



def chunk_gated_delta_rule_fwd_vk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    o: torch.Tensor | None = None,
):
    """ATOM-native vk prefill: K1-K6 all run as separate Triton kernels.

    See `chunk_gated_delta_rule_fwd_vk_flydsl` for the flydsl-K5 variant.
    """
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g=g, cu_seqlens=cu_seqlens, output_dtype=torch.float32
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h_vk(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o_vk(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        o=o,
    )
    if SUPPRESS_LEVEL < 3:
        return g, o, A, final_state, None, None, None
    elif SUPPRESS_LEVEL >= 3:
        return g, o, A, final_state, w, h, v_new


class ChunkGatedDeltaRuleFunctionVk(torch.autograd.Function):
    @staticmethod
    @input_guard
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        o: torch.Tensor | None = None,
    ):
        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        # NOTE: input_guard calls .contiguous() on every Tensor arg including o.
        # For our intended caller (a contiguous output buffer) that is a no-op
        # and returns the same storage, so the inplace contract is preserved.
        # chunk_fwd_o_vk asserts contiguity again as a backstop.
        g, o, A, final_state, w, h, v_new = chunk_gated_delta_rule_fwd_vk(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            o=o,
        )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        # Skip the dtype cast when it's a no-op so the caller's buffer is
        # the literal returned tensor (preserves the inplace contract).
        if o.dtype != q.dtype:
            o = o.to(q.dtype)
        return o, final_state


@torch.compiler.disable
def chunk_gated_delta_rule_vk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    o: torch.Tensor | None = None,
):
    r"""
    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) Gating tensor (in log space!) of shape `[B, T, H]`.
        beta (torch.Tensor):
            Betas of shape `[B, T, H]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, V, K]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, V, K]`. Default: `False`.
        cu_seqlens (torch.Tensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, V, K]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, V, K, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.int32)
        >>> o_var, ht_var = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, (
        "ChunkGatedDeltaRuleFunctionVk does not support float32. Please use bfloat16."
    )
    assert len(beta.shape) == 3, "beta must be of shape [B, T, H]."
    if q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...].",
            stacklevel=2,
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if o is not None:
        # Pre-check the inplace contract HERE — input_guard inside
        # ChunkGatedDeltaRuleFunctionVk.forward will call .contiguous() on
        # every Tensor arg including o, silently cloning a non-contiguous
        # caller buffer and writing the kernel output into the clone instead
        # of the caller's storage. Asserting here, before .apply(), is the
        # only place where we can catch that misuse loudly.
        assert o.shape == v.shape, (
            f"chunk_gated_delta_rule_vk: o.shape {tuple(o.shape)} != v.shape "
            f"{tuple(v.shape)}"
        )
        assert o.dtype == v.dtype, (
            f"chunk_gated_delta_rule_vk: o.dtype {o.dtype} != v.dtype {v.dtype}"
        )
        assert o.is_contiguous(), (
            "chunk_gated_delta_rule_vk: caller-provided o must be contiguous"
        )

    # Optional: dispatch to aiter's end-to-end flydsl prefill pipeline
    # (K1+K2 fused, K3+K4 fused, K5 flydsl, K6 chunk_fwd_o_opt_vk).
    # Enabled via ATOM_USE_FLYDSL_GDR_PREFILL=1 and only when the aiter
    # flydsl-prefill package is importable. We dispatch HERE (not inside
    # the autograd Function) because flydsl_gdr_prefill is a complete
    # pipeline that handles l2norm and contiguity itself — bypassing
    # input_guard and the autograd machinery is cleaner than threading
    # the dispatch through them. The inplace o= contract is forwarded
    # directly to aiter's flydsl_gdr_prefill via its `o=` parameter
    # (which threads into chunk_fwd_o_opt_vk via the same mechanism).
    from atom.utils import envs

    if False and _HAS_FLYDSL_GDR_PREFILL:
        return flydsl_gdr_prefill(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            o=o,
        )

    o, final_state = ChunkGatedDeltaRuleFunctionVk.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        o,
    )
    return o, final_state
