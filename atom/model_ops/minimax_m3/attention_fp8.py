# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pre-head-exchange FP8 transport for the MiniMax-M3 attention output.

The fused ``aiter.a2a_gemm_fused`` v1 op takes the rank's **pre-exchange** FP8
send buffer ``[4*M_real, 2048]`` plus the per-token scales ``[M_real]``, i.e.
exactly what ATOM's existing o-proj quantizer produces for the SP head
exchange, with the head exchange itself folded into the routed kernel:

* one BF16 row maximum per head shard (``head_amax``),
* those maxima all-gathered across the SP group, so the scale covers *all*
  8192 channels of a token (the o-proj quantizer contract),
* ``scale = amax / 448`` and the FP32 E4M3 quantization ATOM already uses
  (``sp_attention.quantize_with_gathered_amax``).

This is the same rule the frozen kernel's harness applies
(``a2a_gemm_flydsl-20261111/repo/tests/opt_r1_common.py::build_activation``:
per-row amax over 2048 all-gathered, ``scale_rows = amax_all / 448``,
``a_scale = scale_rows[rank*M:(rank+1)*M]``), so the fused seam reuses ATOM's
two existing helpers rather than restating the quantization.

Torch/triton/AITER are imported lazily inside the calls: the module must stay
importable from the CPU test gate, and the helpers live in modules that need
the GPU stack at import time.
"""

import torch

# The o-proj quantizer's FP8 format (E4M3FN on gfx950).
_FP8 = torch.float8_e4m3fn


def _sp_attention():
    """The SP attention module holding ATOM's amax/quantize helpers."""
    from atom.model_ops import sp_attention

    return sp_attention


def _sp_group():
    """The SP group (AITER PCP rank dimension) as the fused op executes it."""
    from aiter.dist.parallel_state import get_pcp_group

    return get_pcp_group()


def gather_amax_and_quantize(
    x: torch.Tensor, padded_tokens: int, *, out=None, scale=None
):
    """Quantize the pre-exchange BF16 head shard into the SP4 send buffer.

    ``x`` is ``[real_tokens, 2048]`` BF16 (this rank's head shard over the
    global token rows; ``real_tokens <= padded_tokens``, the divisibility pad
    quantizes as zero rows). Returns ``(q, scale)`` with ``q`` fp8
    ``[padded_tokens, 2048]`` -- the frozen op's ``x`` -- and ``scale`` fp32
    ``[padded_tokens // world, 1]`` -- its per-token scales, one per token over
    all 8192 channels. ``out``/``scale``, when given, are caller-owned buffers
    of exactly those shapes (the provider's registered producer view and the
    capture-time scale cell); the quantizer writes into them so the routed
    call's pointer identity is stable across an eager pre-bind and a CUDA-graph
    capture/replay.
    """
    sp = _sp_attention()
    group = _sp_group()
    world = int(group.world_size)
    rank = int(group.rank_in_group)
    padded_tokens = int(padded_tokens)
    assert padded_tokens % world == 0, "SP send buffer is world-divisible"
    amax = group.device_communicator.all_gatherv(sp.head_amax(x, padded_tokens))
    if scale is None:
        scale = torch.empty(
            (padded_tokens // world, 1), device=x.device, dtype=torch.float32
        )
    q = sp.quantize_with_gathered_amax(x, amax, world, rank, scale, out=out)
    return q, scale


def pre_exchange_send_ok(x) -> bool:
    """Whether ``x`` is the pre-exchange BF16 head-shard output the seam takes."""
    return (
        x is not None
        and getattr(x, "dim", None) is not None
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and bool(getattr(x, "is_cuda", False))
        and x.stride(1) == 1
    )
