# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""MiniMax-M3.1 decode QKV front chain (``ATOM_M3_QKV_CHAIN=1``, default off).

In pure-decode steps of ``MIN_ROWS``..``max_rows()`` rows (17-64: one aiter launch; 65-128 with an aiter that has the split
chain: split-3 GEMM + reduce-insert, two launches) (TP4, NVFP4 K/V and index caches) a sparse attention layer's
``qkv_proj`` runs only the MXFP8 activation quantizer (codes in the decode GEMM's (16, 16) tile layout, AITER-shuffled
E8M0 scales), and the attention op's ``rope_cache`` runs aiter's one-launch ``qkv_chain_cooperative_insert`` (QKV MXFP8
GEMM + q / k / index RMSNorm + partial RoPE + NVFP4 K / V / index-K insert) on the parked codes. Both decisions are taken
at run time inside the two custom ops (concrete row count and forward context) by the same predicate ``take``, so the
compiled graph is the same for every row count; the codes travel from the linear to the attention op through
a per-thread hand-off slot (same forward, same stream, consumed by the next attention op of the same layer).

Techniques and the kernel's numerics (bitwise the unfused split-3 chain) are amd-kernels'
``mxfp8_qkv_norm_rope_nvfp4_insert``, ported into aiter (aiter.ops.flydsl.kernels.qkv_chain).
"""

import functools
import importlib.util
import logging
import threading

import torch
from aiter.jit.utils.torch_guard import torch_compile_guard

from atom.utils import envs

logger = logging.getLogger("atom")

N, K = 2560, 6144
MIN_ROWS = 17
COOP_MAX_ROWS = 64  # the one-launch chain
SPLIT_MAX_ROWS = (
    128  # the split-3 GEMM + reduce-insert chain (aiter ``qkv_chain_insert``)
)

# (rows, codes, scales, weight, chain_scale) of the last qkv_front that took the chain, per thread: TBO decode runs
# its two micro-batches in two threads, each doing linear -> attention for its own layer.
_STATE = threading.local()


def _get():
    return getattr(_STATE, "pending", None)


def _set(value):
    _STATE.pending = value


def enabled() -> bool:
    """The flag is on, aiter has the kernels, and the GPU is gfx950 (the only arch the chain kernel supports)."""
    return bool(envs.ATOM_M3_QKV_CHAIN) and available() and _is_gfx950()


@functools.cache
def _is_gfx950() -> bool:
    try:
        from aiter.jit.utils.chip_info import get_gfx

        return get_gfx() == "gfx950"
    except Exception:  # noqa: BLE001
        # no GPU / aiter probe failure: keep the existing path
        return False


@functools.cache
def available() -> bool:
    """aiter has the chain kernel and its activation quantizer."""
    try:
        return all(
            importlib.util.find_spec(m) is not None
            for m in (
                "aiter.ops.flydsl.kernels.qkv_chain",
                "aiter.ops.flydsl.kernels.mxfp8_quant_shuffled",
            )
        )
    except ImportError:
        return False


@functools.cache
def split_available() -> bool:
    """aiter's chain entry also serves 65-128 rows (``qkv_chain_insert`` / ``code_rows``)."""
    if not available():
        return False
    try:
        from aiter.ops.flydsl.kernels import qkv_chain as _c

        return hasattr(_c, "qkv_chain_insert") and hasattr(_c, "code_rows")
    except ImportError:
        return False


def max_rows() -> int:
    return SPLIT_MAX_ROWS if split_available() else COOP_MAX_ROWS


def code_rows(m: int) -> int:
    """Rows of (16, 16)-layout codes the chain reads at ``m`` rows (the quantizer's block)."""
    if split_available():
        from aiter.ops.flydsl.kernels.qkv_chain import code_rows as _code_rows

        return _code_rows(m)
    return 16 * ((m + 15) // 16)


def layer_ok(
    num_heads,
    num_kv_heads,
    num_idx_heads,
    rotary_dim,
    kv_dtype,
    index_dtype,
    skip_index_topk,
    per_token_fp8_out,
):
    """Static (init-time) eligibility of a sparse attention layer: the kernel's TP4 head layout and NVFP4 caches."""
    return (
        enabled()
        and (num_heads, num_kv_heads, num_idx_heads, rotary_dim) == (16, 1, 1, 64)
        and kv_dtype == "nvfp4"
        and index_dtype == "nvfp4"
        and not skip_index_topk
        and not per_token_fp8_out
    )


def linear_ok(linear) -> bool:
    """Static eligibility of the qkv linear's weight path: a native group32 (1x32) MXFP8 weight with its own FP8 GEMM
    (``_qkv_chain_scale`` exists only on that path); online-requantized, per-token, 32x32, MXFP4 or BF16-fallback
    weights keep their existing linear."""
    return (
        getattr(linear, "native_a8_group_rows", None) == 1
        and getattr(linear, "params_dtype", None) == torch.float8_e4m3fn
        and not getattr(linear, "runs_bf16", False)
        and getattr(linear, "bias", None) is None
    )


# One-shot activity log. Counters are plain Python ints touched only where Python runs (warm-up, eager steps and
# CUDA-graph *capture*; a graph replay never enters Python), so the replay hot path costs nothing. The first call logs
# the layer census and the row window; after _LOG_AFTER decisions the per-reason tally is logged once and counting stops.
_LOG_AFTER = 1000
_REASONS = ("taken", "prefill", "dummy", "token_graph", "out_of_window")
_stats = {r: 0 for r in _REASONS}
_stats.update(calls=0, layers_on=0, layers_not_preshuffled=0, done=False)


def aiter_features() -> str:
    """Which chain entries the installed aiter has (stage 1: ``qkv_chain_cooperative_insert``, 17-64 rows; the
    65-128 row stage adds ``qkv_chain_insert``), for the one-shot log only."""
    try:
        import aiter.ops.flydsl.kernels.qkv_chain as k

        names = (
            "qkv_chain_cooperative_insert",
            "qkv_chain_insert",
            "qkv_chain_split_insert",
        )
        have = [n for n in names if hasattr(k, n)]
    except Exception as e:  # noqa: BLE001 - log-only probe, never raises into the step
        return f"unavailable ({type(e).__name__})"
    if importlib.util.find_spec("aiter.ops.flydsl.kernels.mxfp8_quant_shuffled"):
        have.append("mxfp8_quantize_shuffled")
    return ",".join(have) or "none"


def note_layer(chain_on: bool) -> None:
    """Load time (qkv linear prepare): count a chain layer, or one that fell back for lack of a preshuffled weight."""
    _stats["layers_on" if chain_on else "layers_not_preshuffled"] += 1


def _count(reason: str) -> None:
    st = _stats
    if st["done"]:
        return
    st[reason] += 1
    st["calls"] += 1
    if st["calls"] == 1:
        logger.info(
            "ATOM_M3_QKV_CHAIN active: %d layer(s) with the chain, %d layer(s) off (weight not preshuffled); "
            "row window %d..%d rows (min_rows=%d, max_rows=%d); aiter features: %s; decode token-graph steps "
            "are skipped",
            st["layers_on"],
            st["layers_not_preshuffled"],
            MIN_ROWS,
            max_rows(),
            MIN_ROWS,
            max_rows(),
            aiter_features(),
        )
    if st["calls"] >= _LOG_AFTER:
        st["done"] = True
        logger.info(
            "ATOM_M3_QKV_CHAIN decisions over the first %d python-side calls (warm-up/eager/capture; graph "
            "replays are not counted): %s",
            st["calls"],
            ", ".join(f"{r}={st[r]}" for r in _REASONS),
        )


def take(m: int) -> bool:
    """Run time: this forward's qkv linear + insert take the chain.

    Never inside a token graph (TR_HYBRID decode, the level-3 default): replay runs the captured quantizer without
    the Python hand-off and the attention op (an eager region) would read the uncomputed QKV.
    """
    from atom.utils.forward_context import get_forward_context

    fctx = get_forward_context()
    # TR_HYBRID record/replay of a token graph: the capture runs the quantizer without the Python hand-off (record
    # has forward_mode None, only ForwardContext.tr_graph marks it)
    if getattr(fctx, "tr_graph", None) is not None:
        _count("token_graph")
        return False
    c = fctx.context
    if c is None or c.is_prefill:
        _count("prefill")
        return False
    if c.is_dummy_run:
        _count("dummy")
        return False
    mode = getattr(c, "forward_mode", None)
    if mode is not None and getattr(mode, "token_graph", False):
        _count("token_graph")
        return False
    if not (MIN_ROWS <= m <= max_rows()):
        _count("out_of_window")
        return False
    _count("taken")
    return True


def chain_weight_scale(weight_scale: torch.Tensor) -> torch.Tensor:
    """Row-major E8M0 weight scale [2560, 192] -> AITER's shuffled layout the chain's GEMM reads."""
    from aiter import dtypes
    from aiter.ops.shuffle import shuffle_scale_a16w4

    return shuffle_scale_a16w4(
        weight_scale.view(dtypes.fp8_e8m0).contiguous(), 1, False
    )


def _front_fake(x, weight, weight_scale, chain_scale, weight_preshuffled):
    return torch.empty(
        (*x.shape[:-1], weight.shape[0]), dtype=torch.bfloat16, device=x.device
    )


@torch_compile_guard(gen_fake=_front_fake, mutates_args=[])
def qkv_front(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    chain_scale: torch.Tensor,
    weight_preshuffled: bool,
) -> torch.Tensor:
    """The qkv linear of a chain-eligible layer. Chain rows: MXFP8 quant only, codes parked in the hand-off slot, the returned
    qkv is NOT computed (the attention op's chain never reads it). Other rows: today's linear.
    """
    x2 = x.reshape(-1, K)
    m = x2.shape[0]
    if weight_preshuffled and take(m):
        from aiter.ops.flydsl.kernels.mxfp8_quant_shuffled import (
            mxfp8_quantize_shuffled,
        )

        codes, scales = mxfp8_quantize_shuffled(x2, code_rows(m))
        _set((m, codes, scales, weight, chain_scale))
        return torch.empty((*x.shape[:-1], N), dtype=torch.bfloat16, device=x.device)
    _set(None)
    from atom.model_ops.blockscale import native_quant_linear

    return native_quant_linear(
        x,
        weight,
        weight_scale,
        x_scale=None,
        weight_group_rows=1,
        weight_preshuffled=weight_preshuffled,
        dtype=torch.bfloat16,
    )


def pop(m: int):
    """The attention op's side: the parked codes of this layer's linear (or None = today's path)."""
    p = _get()
    _set(None)
    if p is None:
        return None
    assert p[0] == m, f"qkv_chain: parked codes for {p[0]} rows, attention has {m}"
    return p


def run_chain(
    pend,
    q_out,
    index_q_out,
    q_norm_w,
    k_norm_w,
    iq_norm_w,
    ik_norm_w,
    cos_sin_cache,
    positions,
    slot_mapping,
    kv_pages,
    index_pages,
    k_scale,
    v_scale,
    num_heads,
    num_kv_heads,
    num_index_heads,
    eps,
):
    from aiter.ops.flydsl.kernels import qkv_chain as _chain

    m, codes, scales, weight, chain_scale = pend
    entry = (
        _chain.qkv_chain_insert
        if split_available()
        else _chain.qkv_chain_cooperative_insert
    )
    entry(
        codes.view(torch.float8_e4m3fn),
        scales,
        weight,
        chain_scale,
        m,
        q_norm_w,
        k_norm_w,
        iq_norm_w,
        ik_norm_w,
        cos_sin_cache,
        positions,
        slot_mapping,
        slot_mapping,
        kv_pages,
        index_pages,
        k_scale,
        v_scale,
        q_out,
        index_q_out,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_index_heads=num_index_heads,
        eps=eps,
    )


def warm(weight: torch.Tensor, chain_scale: torch.Tensor) -> None:
    """Eager calls at load (QKV linear prepare): compile the chain's row tiers and create its counter pool outside graph
    capture (a first call inside capture raises). Every slot -1: no page is written."""
    from aiter.ops.flydsl.kernels import qkv_chain as _chain
    from aiter.ops.flydsl.kernels.mxfp8_quant_shuffled import mxfp8_quantize_shuffled

    entry = (
        _chain.qkv_chain_insert
        if split_available()
        else _chain.qkv_chain_cooperative_insert
    )
    dev = weight.device
    nw = torch.zeros(128, dtype=torch.bfloat16, device=dev)
    cs = torch.zeros((128, 64), dtype=torch.bfloat16, device=dev)
    kv = torch.zeros((1, 2, 128, 72), dtype=torch.uint8, device=dev)
    idx = torch.zeros((1, 128, 72), dtype=torch.uint8, device=dev)
    one = torch.ones(1, dtype=torch.float32, device=dev)
    # one row per tier: the one-launch chain (17-64) and, with the split chain, its three 65-128 row configs
    # (80 / 96 / 128 rows of codes), which compile separately
    for m in (MIN_ROWS, 33, 49) + ((65, 81, 97) if split_available() else ()):
        codes, scales = mxfp8_quantize_shuffled(
            torch.zeros((m, K), dtype=torch.bfloat16, device=dev), code_rows(m)
        )
        pos = torch.zeros(m, dtype=torch.int64, device=dev)
        skip = torch.full((m,), -1, dtype=torch.int64, device=dev)
        q = torch.empty((m, 2048), dtype=torch.bfloat16, device=dev)
        iq = torch.empty((m, 128), dtype=torch.float8_e4m3fn, device=dev)
        entry(
            codes.view(torch.float8_e4m3fn),
            scales,
            weight,
            chain_scale,
            m,
            nw,
            nw,
            nw,
            nw,
            cs,
            pos,
            skip,
            skip,
            kv,
            idx,
            one,
            one,
            q,
            iq,
            num_heads=16,
            num_kv_heads=1,
            num_index_heads=1,
            eps=1e-6,
        )
    torch.cuda.synchronize(dev)
