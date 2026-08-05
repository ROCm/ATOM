# SPDX-License-Identifier: MIT
"""Paged-state KDA forward for Kimi linear attention.

``fla.ops.kda.chunk_kda`` cannot serve the SSM state cache: its state is
batch-indexed, so a prefill must gather the pool before the call and scatter
after, and there is no way to read one slot and write another. That is exactly
what ``state_indices`` / ``dst_indices`` / ``h0_mask`` exist for on the GDN
path.

Rather than fork the whole KDA op (~2900 lines across chunk/intra/gate/wy_fast),
this reuses fla's stages unchanged and swaps in ATOM's paged
``chunk_gated_delta_rule_fwd_h`` for the one call that touches the recurrent
state. The KDA and GDN chunk forwards already share that function upstream —
KDA passes ``gk=g`` where GDN passes ``g=g`` — so the paging support written
for GDN applies here with no kernel change beyond ``state_v_first``.

Inference only: no autograd wrapper, and the intermediates fla keeps for
backward are dropped. Add a ``torch.autograd.Function`` here if training ever
needs the paged path.
"""

from __future__ import annotations

import torch

from .chunk_delta_h import chunk_gated_delta_rule_fwd_h

# Kimi's KDA state is [HV, V, K]; GDN's is [HV, K, V]. fla calls this
# `state_v_first` (older releases: `transpose_state_layout`). Everything the
# state cache touches — the pool, `h`, and the checkpoint copies — has to agree
# on it, so it is threaded rather than assumed.
STATE_V_FIRST = True


def chunk_kda_paged(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool = True,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    state_indices: torch.Tensor | None = None,
    dst_indices: torch.Tensor | None = None,
    h0_mask: torch.Tensor | None = None,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    chunk_size: int = 64,
    return_intermediate_states: bool = False,
):
    """KDA chunked forward that reads and writes the state pool by slot.

    ``initial_state`` is the whole pool ``[num_slots, HV, V, K]``;
    ``state_indices`` picks each sequence's source slot and ``dst_indices``
    its destination. They may be equal — a prefill chunk advances its own
    runtime slot in place.

    Returns ``(o, final_state, h)``. ``h`` is the per-chunk intermediates the
    state cache slices interior checkpoints from, or None when not requested.
    """
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.common.gate import fused_beta_sigmoid
    from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
    from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
    from fla.ops.kda.gate import kda_gate_chunk_cumsum
    from fla.ops.utils.constant import RCP_LN2
    from fla.ops.utils.index import prepare_chunk_indices

    if scale is None:
        scale = k.shape[-1] ** -0.5
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )

    # `use_qk_l2norm_in_kernel=True` upstream, and the model passes it on the
    # non-paged path. ChunkKDAFunction.forward applies this BEFORE
    # chunk_kda_fwd, so reusing fla's stages directly means doing it here —
    # omitting it is silent: q/k just carry their raw magnitudes into the
    # delta rule and every prefill under the state cache is wrong by ~27%
    # relative, with no error anywhere.
    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)

    # Gate activation + chunk cumsum, fused as the model expects
    # (`use_gate_in_kernel=True` in KimiKDAAttention._run_kda).
    g = kda_gate_chunk_cumsum(
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=RCP_LN2,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        lower_bound=lower_bound,
    )
    # `use_beta_sigmoid_in_kernel=True` upstream; fp32 in, matching the model's
    # `beta.float()` (a bf16 sigmoid erodes the delta-rule write strength).
    beta = fused_beta_sigmoid(beta)

    w, u, _qg, kg, Aqk, _Akk = chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
        # STORE_QG specialization: fla 0.5.1's default recompute path is
        # non-deterministic for long packed gfx950 prefills (see
        # KimiKDAAttention._run_kda). Same reason, kept here.
        disable_recompute=True,
    )

    # The one call that touches recurrent state — ATOM's paged version.
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        state_indices=state_indices,
        dst_indices=dst_indices,
        h0_mask=h0_mask,
        state_v_first=STATE_V_FIRST,
    )

    o = chunk_gla_fwd_o_gk(
        q=q,
        v=v_new,
        g=g,
        A=Aqk,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        state_v_first=STATE_V_FIRST,
    )
    return o, final_state, (h if return_intermediate_states else None)
