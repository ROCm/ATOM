# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routed MXFP4 H6144 I768 E128 top4: one entry, shape-based dispatch."""
import os
from dataclasses import dataclass

import torch

from amd_kernels.warm_up import entry

H, I, E, TOPK = 6144, 768, 128, 4
# upper token bound -> expert GEMM variant; one Quark-even MXFP4 activation quant at every M
DISPATCH = (
    (512, "a4w4_small"),
    (2047, "a4w4_bm128"),
    (4096, "a4w4_bm128_router_quant"),
    (87381, "a4w4_router_quant"),
)
# fused router + quant from here, every call site: routing prep -3.7..-8.6 us at 1025-2047, +1.0..+4.5 at 513-1024
ROUTER_QUANT_FROM = 1025


@dataclass
class MoeWeights:
    """The router (FP32 [E, H] row-major and its 3 MiB line layout), the bias FP32 [E] and the experts in the kernel
    layouts (gate / up interleaved MXFP4 values and e8m0 scales); built once by ``from_row_major`` before graph capture.
    """
    router_weight: torch.Tensor
    correction_bias: torch.Tensor
    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    a4w4_scales: object
    routed_scaling_factor: float = 2.0
    # the entry needs both (``from_row_major`` builds the gate's line layout; ``shared`` is the caller's)
    router_weight_lines: torch.Tensor | None = None
    shared_weights: object | None = None

    @classmethod
    def from_row_major(cls, router_weight, correction_bias, w13, w13_scale, w2, w2_scale, *, shared=None,
                       routed_scaling_factor=2.0):
        """The checkpoint layout (row-major uint8 MXFP4 values / e8m0 scales, gate rows then up rows; a
        ``SharedWeights``, which ``mxfp4_moe`` needs) -> the entry's layouts, built once before graph capture."""
        from .layout import w13_layout, w2_layout, router_line_layout
        from .prefill_h6144_i768 import Mxfp4PrefillScales
        scales = Mxfp4PrefillScales.from_row_major(w13_scale, w2_scale)
        return cls(router_weight, correction_bias, w13_layout(w13), scales.w13,
                   w2_layout(w2), scales.w2, scales, float(routed_scaling_factor),
                   router_weight_lines=router_line_layout(router_weight), shared_weights=shared)


def variant_for(n_tokens):
    for bound, name in DISPATCH:
        if 0 < n_tokens <= bound:
            return name
    raise ValueError(f"unsupported token count {n_tokens}")


def _check_weights(w):
    if w.shared_weights is None or w.router_weight_lines is None:
        raise ValueError("needs MoeWeights with the shared expert and the line-layout gate: "
                         "MoeWeights.from_row_major(..., shared=SharedWeights...)")


def _route_parts(parts, w):
    from .router_topk import splitk_biased_topk
    m = parts.shape[0]
    ids = torch.empty((m, TOPK), dtype=torch.int32, device=parts.device)
    wt = torch.empty((m, TOPK), dtype=torch.float32, device=parts.device)
    splitk_biased_topk(parts, w.correction_bias, wt, ids, w.routed_scaling_factor)
    return wt, ids


def _route_quant_small(x, w, *, s8_tiles=False):
    """Below ``ROUTER_QUANT_FROM`` rows: the split-K router, then the top-4 with the MXFP4 + MXFP8 input quant in one
    launch (at ``router_dual_quant_for`` rows the quant workgroups join the router launch) -> ((q4, s4), (q8, s8),
    topk weights, topk ids); ``s8_tiles``: the FP8 scales in the shared GEMMs' tiled layout."""
    from .router_gemm import router_gemm_partials
    from .router_gemm_topk_mxfp4_mxfp8_quant import router_dual_quant_for, router_gemm_topk_mxfp4_mxfp8_quant
    from .router_topk_mxfp4_mxfp8_quant import splitk_biased_topk_mxfp4_mxfp8_quant
    n = x.shape[0]
    if router_dual_quant_for(n):
        q4, q8, wt, ids, _ = router_gemm_topk_mxfp4_mxfp8_quant(
            x, w.router_weight, w.correction_bias, w.routed_scaling_factor)
        return q4, q8, wt, ids
    parts = router_gemm_partials(x, w.router_weight, w.router_weight_lines)
    if parts is None:
        raise ValueError("dual input quant requires the split-K router range")
    ids = torch.empty((n, TOPK), dtype=torch.int32, device=x.device)
    wt = torch.empty((n, TOPK), dtype=torch.float32, device=x.device)
    q4, q8 = splitk_biased_topk_mxfp4_mxfp8_quant(parts, w.correction_bias, wt, ids, w.routed_scaling_factor, x,
                                                  s8_tiles=s8_tiles)
    return q4, q8, wt, ids


def _prepare(x, w):
    """From ``ROUTER_QUANT_FROM`` rows, where the shared input quant runs apart: the fused router + MXFP4 input quant on
    the line-layout gate, then the split-K top-4 -> ((q4, s4), topk weights, topk ids)."""
    from .router_mxfp4_quant_line_layout import router_mxfp4_quant_line_layout
    q4, s4, parts = router_mxfp4_quant_line_layout(x, w.router_weight_lines)
    wt, ids = _route_parts(parts, w)
    return (q4, s4), wt, ids


def route(x, w):
    """The BF16 entry's routing (topk weights, topk ids), exposed for numerical checks."""
    if x.shape[0] < ROUTER_QUANT_FROM:
        return _route_quant_small(x, w)[2:]
    return _prepare(x, w)[1:]


def mxfp4_moe_experts(x_quant, topk_weights, topk_ids, w):
    """The routed experts from 513 rows on the MXFP4 input ``x_quant`` = (codes, scales), already routed
    (``topk_weights`` include the routed scale) -> the BF16 [M, 4, H] slots."""
    n = x_quant[0].shape[0]
    variant = variant_for(n)
    if variant == "a4w4_small":
        raise ValueError("below 513 rows the routed experts run with the shared expert folded in")
    if "bm128" in variant:
        from .experts_a4w4_bm128 import a4w4_experts
    else:
        from .experts_a4w4 import a4w4_experts
    return a4w4_experts(x_quant, w.w13, w.w2, w.a4w4_scales, topk_weights, topk_ids)


def mxfp4_moe(x, w, *, return_partials=False):
    """Normed BF16 [M, 6144], or a ``MoeInput`` / ``MoeInputCompact`` (routed and quantized before a sequence-parallel
    gather) -> the routed experts plus the MXFP8 shared expert.

    Partials mode returns the slots BF16 [M, 4, H] and the shared output BF16 [M, H]; reduced mode sums slots 0..3,
    then the shared output, in FP32.
    """
    from .prepared_input import MoeInput, MoeInputCompact, _mxfp4_moe_prepared
    _check_weights(w)
    if isinstance(x, MoeInputCompact):
        x = x.expand()
    if isinstance(x, MoeInput):
        return _mxfp4_moe_prepared(x, w, return_partials=return_partials)
    variant_for(x.shape[0])
    if x.ndim != 2 or x.shape[1] != H or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("expected contiguous BF16 [M,6144]")
    from .shared_expert import (shared_expert, shared_expert_parallel, shared_parallel_for, shared_dual_quant_for,
                                shared_folded_for, shared_in_persistent_for, SHARED_QUANT_REDUCED_ONLY)
    from .experts_a4w4_small import id_rows_for
    n = x.shape[0]
    sw = w.shared_weights
    current = torch.cuda.current_stream(x.device)
    shared_persistent = shared_in_persistent_for(n)
    dual = shared_dual_quant_for(n) or shared_persistent
    # the shared tiles join the routed 16-row launches; otherwise their own launches
    folded = shared_folded_for(n) and dual and variant_for(n) == "a4w4_small" and not shared_persistent
    parallel = shared_parallel_for(n) and not folded and not shared_persistent
    if return_partials and not shared_persistent and SHARED_QUANT_REDUCED_ONLY[0] <= n <= SHARED_QUANT_REDUCED_ONLY[1]:
        dual = False
    tiled_shared = n >= ROUTER_QUANT_FROM
    if dual:
        if tiled_shared:
            from .router_mxfp4_mxfp8_quant_line_layout import router_mxfp4_mxfp8_quant_line_layout
            prepared, xq8, parts_router = router_mxfp4_mxfp8_quant_line_layout(x, w.router_weight_lines)
            if n > 8192:
                wt, ids = _route_parts(parts_router, w)
        else:
            # the combined launch (449-1024) reads the shared scales tiled
            prepared, xq8, wt, ids = _route_quant_small(x, w, s8_tiles=shared_persistent)
        if folded or shared_persistent:
            shared = None
        elif parallel:
            shared = shared_expert_parallel(x, sw, current, x_quant=xq8, scales_tiled=tiled_shared)
        else:
            shared = shared_expert(x, sw, x_quant=xq8, scales_tiled=tiled_shared)
    elif parallel:
        shared = shared_expert_parallel(x, sw, current)
    else:
        shared = shared_expert(x, sw)
    if not dual:
        prepared, wt, ids = _prepare(x, w)
    elif tiled_shared and n <= 8192:
        # to 8192 rows the shared stream forks before the top-k, above after it (measured overlap)
        wt, ids = _route_parts(parts_router, w)
    if folded:
        from .experts_a4w4_small import a4w4_experts
        # 16-96 rows: GEMM1 builds its rows from topk_ids
        parts, shared = a4w4_experts(prepared, w.w13, w.w2, w.a4w4_scales, wt, ids, shared=(*xq8, sw),
                                     id_rows=id_rows_for(n))
    elif shared_persistent:
        from .experts_a4w4_bm128 import a4w4_experts
        parts, shared = a4w4_experts(prepared, w.w13, w.w2, w.a4w4_scales, wt, ids, shared=(*xq8, sw))
    else:
        parts = mxfp4_moe_experts(prepared, wt, ids, w)
    if parallel:
        current.wait_stream(sw.stream)
        shared.record_stream(current)
    if return_partials:
        return parts, shared
    from .common.reduce_slots_shared import reduce_slots_shared
    return reduce_slots_shared(parts, shared)


# One shape per router wave/split tier, sort tier and expert GEMM variant (33: the router without
# quant tasks at 33-57; 64: the router with appended quant tasks at 58-64;
# the id-rows GEMMs of the BF16 entry with shared weights at 16-96: 16 / 17 / 33 their 64 / 128 /
# 256 pair slots, 65 their 512 slots; 16 and 17 the two shared down-tile widths).
WARM_UP_TOKENS = (2, 8, 9, 12, 15, 16, 17, 33, 64, 65, 128, 129, 192, 193, 256, 257, 385, 512,
                  513, 641, 769, 1408, 1536, 1537, 2048, 2049, 3584, 3841, 4096, 4097, 5377, 7681, 8192, 8193, 10881, 16384, 16385,
                  32768)


def _mode(x, w, *, return_partials=False):
    """What a call fixes about its kernels besides the token count (``amd_kernels.warm_up``)."""
    from .prepared_input import MoeInput, MoeInputCompact
    kind = "compact" if isinstance(x, MoeInputCompact) else "prepared" if isinstance(x, MoeInput) else "bf16"
    return kind, w.shared_weights is not None, w.router_weight_lines is not None, bool(return_partials)


def _inputs(mode, n, x, w, *, return_partials=False):
    """One warm-up call of the mode at ``n`` tokens (zeros: the dispatch depends on shapes only)."""
    from .prepared_input import prepare_mxfp4_moe
    xn = torch.zeros((n, H), device=w.router_weight.device, dtype=torch.bfloat16)
    if mode[0] != "bf16":
        xn = prepare_mxfp4_moe(xn, w)
        if mode[0] == "compact":
            xn = xn.compact()
    return (xn, w), {"return_partials": return_partials}


mxfp4_moe = entry("moe/mxfp4_h6144_i768_e128_top4", mode=_mode, sizes=lambda mode: WARM_UP_TOKENS,
                  inputs=_inputs)(mxfp4_moe)


_loaded: set = set()


def _node_workers() -> int:
    # every GPU of a node warms up at once: an eighth of this process's CPUs each
    return max(1, min(16, len(os.sched_getaffinity(0)) // 8))


def warm_up_mxfp4_moe(w, *, partials=False, workers=None):
    """Loads every variant of the BF16-input reduced mode, and of the partials mode with ``partials``, into this
    process before serving, once per mode and device: the first call compiles them all (``amd_kernels.warm_up``,
    ``workers`` processes, default ``_node_workers()``; the count stays for the process), then one call per
    ``WARM_UP_TOKENS`` size; one layer's weights serve every layer."""
    from amd_kernels import warm_up

    dev = w.router_weight.device
    todo = [(dev, _mode(None, w, return_partials=p)) for p in ((False, True) if partials else (False,))]
    todo = [key for key in todo if key not in _loaded]
    if not todo:
        return
    warm_up.set_workers(_node_workers() if workers is None else workers)
    with torch.inference_mode():
        for key in todo:
            for m in WARM_UP_TOKENS:
                mxfp4_moe(torch.zeros((m, H), device=dev, dtype=torch.bfloat16), w, return_partials=key[1][3])
            _loaded.add(key)
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()  # the warm-up's buffers back to the device before the caller sizes its caches
