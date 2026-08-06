# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project
# (version 0.5.2, fla/ops/kda/chunk.py). The original source code was licensed
# under the MIT license and included the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Adapted for ATOM:
#   - No torch.autograd.Function: inference only.
#   - No @input_guard: it calls .contiguous() on every tensor argument, which
#     silently clones the rearrange views this path passes and would clone a
#     caller-provided `o` out from under them. Contiguity is asserted instead.
#   - No @dispatch('kda'): flash_kda is not installed and tilelang needs nvcc,
#     so no backend is reachable on ROCm; the indirection would only make the
#     parity test ambiguous about what it compared.
#   - No trailing `o.type_as(q)` (fla/ops/kda/chunk.py:119). The returned dtype
#     therefore follows `v`, not `q`: with a caller-provided `o` we assert
#     o.dtype == v.dtype, and on the o=None path the buffer is allocated from
#     `v`. Nothing here ties v.dtype to q.dtype, so this is a real divergence
#     from upstream whenever they differ -- benign on the Kimi-K3 path, where
#     q/k/v are all bf16. Dropping it is deliberate: with a caller buffer,
#     type_as under a dtype mismatch returns a silent copy and the result stops
#     aliasing the caller's `o`, which is the opposite of this wrapper's
#     contract. Better to return v's dtype than to break the aliasing promise.
#   - A_log / dt_bias / chunk_size are explicit parameters rather than **kwargs,
#     and the deprecated `transpose_state_layout` alias is not accepted.

import torch
from fla.modules.l2norm import l2norm_fwd
from fla.ops.common.gate import fused_beta_sigmoid
from fla.ops.utils.index import prepare_chunk_indices

from .chunk_fwd import chunk_kda_fwd


@torch.compiler.disable
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
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    chunk_size: int = 64,
    h0_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = False,
    o: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunked KDA forward, fused for ATOM's Kimi-K3 prefill path.

    Same semantics as ``fla.ops.kda.chunk_kda`` with these additions:

    ``h0_indices``
        1D per-sequence cache-slot index into ``initial_state``'s first
        dimension. When given, the kernel reads the initial state from
        ``initial_state[h0_indices[i]]`` instead of ``initial_state[i]``, so
        the caller does not gather. ``-1`` (PAD_SLOT_ID) skips both read and
        write.
    ``has_initial_state``
        1D per-sequence bool. False means the sequence starts from a zero
        state; the kernel skips the load rather than loading and zeroing
        afterwards.
    ``inplace_final_state``
        Write the final state back into the same indexed slots of
        ``initial_state``. The returned ``final_state`` *is* ``initial_state``.
    ``o``
        Caller-provided output buffer, written in place and returned.

        CALLER CONTRACT (inherited from ``chunk_gla_fwd_o_gk``): when ``o``
        is passed in the varlen case, rows at ``t >= cu_seqlens[-1]`` are NOT
        written by the kernel — the grid covers only ``NT = len(chunk_indices)``
        tiles, which tile ``[cu_seqlens[0], cu_seqlens[-1])``. The caller MUST
        zero (or otherwise own) any padding rows beyond the last sequence end
        before calling. Passing ``o=None`` (the default) allocates a
        ``torch.zeros_like(v)`` inside the kernel, which does not have this
        constraint.

    With all four at their defaults this is bit-identical to upstream.
    """
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when "
            f"using `cu_seqlens`. Please flatten variable-length inputs first."
        )
    if initial_state is not None and initial_state.dtype != torch.float32:
        raise ValueError(f"initial_state must be float32, got {initial_state.dtype}.")
    if chunk_size not in (32, 64):
        raise ValueError(
            f"`chunk_size` must be either 32 or 64 for KDA, got {chunk_size}."
        )
    if use_gate_in_kernel and A_log is None:
        raise ValueError("A_log must be provided when use_gate_in_kernel=True.")
    if safe_gate and use_gate_in_kernel:
        if lower_bound is None:
            raise ValueError(
                "`lower_bound` must be specified when `safe_gate=True` and "
                "`use_gate_in_kernel=True`."
            )
        if not -5 <= lower_bound < 0:
            raise ValueError(
                f"`lower_bound` must be in the safe range [-5, 0), got "
                f"{lower_bound}."
            )
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError(
            "`allow_neg_eigval=True` requires `use_beta_sigmoid_in_kernel=True`."
        )

    B, T, H, K, HV = *q.shape, v.shape[2]
    if q.shape != k.shape:
        raise ValueError(f"q and k must match, got {q.shape} vs {k.shape}")
    if K > 256:
        raise ValueError(f"KDA supports key headdim <= 256, got {K}.")
    if HV % H != 0:
        raise ValueError(f"num_v_heads ({HV}) must be divisible by ({H}).")
    if tuple(g.shape) != (B, T, HV, K):
        raise ValueError(f"g must be {(B, T, HV, K)}, got {tuple(g.shape)}")
    if tuple(beta.shape) != (B, T, HV):
        raise ValueError(f"beta must be {(B, T, HV)}, got {tuple(beta.shape)}")

    if h0_indices is not None and h0_indices.ndim != 1:
        raise ValueError(
            f"h0_indices must be 1D, got shape {tuple(h0_indices.shape)}. 2D "
            f"spec-decode indices are not supported on the prefill path; the "
            f"decode kernel handles those."
        )
    if inplace_final_state and h0_indices is None:
        raise ValueError("inplace_final_state requires h0_indices.")
    if inplace_final_state and not output_final_state:
        raise ValueError("inplace_final_state requires output_final_state.")
    if has_initial_state is not None and h0_indices is None:
        raise ValueError("has_initial_state requires h0_indices.")
    if o is not None:
        # Without @input_guard nothing clones a bad buffer silently, so these
        # turn a wrong-strides bug into an error instead of corruption.
        if tuple(o.shape) != (B, T, HV, v.shape[-1]):
            raise ValueError(
                f"o must be {(B, T, HV, v.shape[-1])}, got {tuple(o.shape)}"
            )
        if o.dtype != v.dtype:
            raise ValueError(f"o.dtype {o.dtype} != v.dtype {v.dtype}")
        if not o.is_contiguous():
            raise ValueError("o must be contiguous")

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
            cu_seqlens, chunk_size, cu_seqlens_cpu=cu_seqlens_cpu
        )

    out, final_state = chunk_kda_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        disable_recompute=disable_recompute,
        h0_indices=h0_indices,
        has_initial_state=has_initial_state,
        inplace_final_state=inplace_final_state,
        o=o,
    )
    return out, final_state
