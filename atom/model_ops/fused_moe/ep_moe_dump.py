# SPDX-License-Identifier: Apache-2.0
"""TEMPORARY: dump post-dispatch MoE inputs so the EP MoE path can be
benchmarked offline against a real token->expert distribution.

Delete this file and its single call site in ``modular_kernel.py`` when
``_test/bench_ep_moe.py`` no longer needs fresh captures.

Everything here is a no-op unless ``ATOM_EP_MOE_DUMP_DIR`` is set.

Why capture instead of synthesise: the routing cost depends entirely on how the
top-k choices land across the 48 local experts, and a uniform random draw gets
that wrong in both directions -- it over-states how evenly loaded the experts
are, and it cannot reproduce the ``M >> num_local_tokens`` over-allocation of
the mori receive buffer. Both drive the kernels under test.

Usage (eager mode is REQUIRED -- see below)::

    ATOM_EP_MOE_DUMP_DIR=/app/_test/ep_dump \\
    ATOM_USE_TRITON_MOE_EP=1 ENFORCE_EAGER=1 \\
        ./run_dsv4_pro_ep_server.sh
    # then send any request, e.g. ./run_dsv4_pro_ep_curl.sh

Works equally well under a real benchmark client (``bench_no_warmup.sh``:
conc=64, ISL=128, OSL=256).

Writes ``ep_moe_layer{LL}_{regime}_rank0.pt``, one file per (layer, regime),
keeping the pass with the MOST live rows in each -- see ``_keep()`` and
``_regime()``. Bucketing by regime is what lets ONE bench run yield both prefill
and decode shapes: prefill carries ~1000x more rows, so a single per-layer
watermark meant prefill always won and decode was never captured.

Rank 0 only (override with ``ATOM_EP_MOE_DUMP_RANK``): every rank sees a
statistically equivalent token->expert distribution, so the other seven were pure
duplication. Row-indexed tensors are also trimmed to the last live row before
saving -- see ``_live_rows()``. Together these take a capture set from ~64 GB to
a few MB. ``M_orig`` records the untrimmed buffer height so the trim factor stays
visible in the data.

Under CUDAGraph the Python body of a captured region is not re-executed on
replay, so a dump placed here would silently never fire for captured shapes.
Run with ``ENFORCE_EAGER=1``; the capture guard below is a backstop for the
uncaptured-prefill case.
"""

import os

import torch

_CALLS = 0
_PASSES = 0
# layer -> best (num_local_tokens) written so far. See _keep() for why max.
_BEST = {}


def _shape(t):
    return None if t is None else tuple(t.shape)


def _dtype(t):
    return None if t is None else str(t.dtype)


def _regime():
    """(is_dummy, regime) for the pass in flight.

    ``regime`` names which branch of ``_maybe_trim_dispatch_output`` the pass
    takes, because that is what determines the buffer height the MoE stages are
    sized by:

      "prefill" trimmed to cu_tokens_across_dp_cpu[-1], only if
                ATOM_EP_TRIM_PREFILL is set
      "decode"  trimmed to graph_bs * topk * dp_size, always
      "mixed"   non-uniform batch: falls through BOTH trims unless
                ATOM_EP_TRIM_PREFILL is set. Worth capturing precisely because
                it is the case that keeps the full mbt * ep_size buffer.

    ``is_dummy`` marks the startup profile run and the cudagraph warmups. Those
    push a synthetic max_num_batched_tokens batch -- the LARGEST the server ever
    sees -- so a max-live-rows watermark latches onto warmup and is never beaten
    by real traffic. That is exactly what happened: 1.8 GB captures appeared
    before any request was sent.
    """
    from atom.utils.forward_context import get_forward_context

    try:
        ctx = getattr(get_forward_context(), "context", None)
    except AssertionError:
        # get_forward_context asserts when no context is set. Treat that as real
        # traffic rather than dropping every capture.
        return False, "unknown"
    if ctx is None:
        return False, "unknown"
    if getattr(ctx, "is_dummy_run", False):
        return True, "dummy"
    if getattr(ctx, "is_prefill", False):
        return False, "prefill"
    return False, "decode" if getattr(ctx, "dp_uniform_decode", True) else "mixed"


def _keep(key, rows):
    """Keep the pass with the MOST live rows seen for this (layer, regime).

    Keyed by regime, not just layer: prefill outnumbers decode rows by ~1000x, so
    a single per-layer watermark means prefill always wins and decode is never
    captured. One bucket per regime gets every shape from a single bench run.

    Max-rows within a bucket, because a real benchmark drains -- requests retire
    at slightly different steps, so the final passes carry a nearly empty batch.
    Last-pass-wins would hand the bench the least loaded sample of the whole run.

    The watermark is the SAVED row count, not num_local_tokens. Those can
    disagree, and using the latter locked in empty captures: a pass with a large
    num_local_tokens but zero live rows would raise the watermark while writing a
    0-byte file, after which every real pass was rejected as "smaller".
    """
    if rows <= _BEST.get(key, -1):
        return False
    _BEST[key] = rows
    return True


def _live_rows(dispatch_ids, expert_map, num_local_tokens):
    """One past the last row holding a gate this rank owns.

    The mori receive buffer is over-allocated -- M=131072 against 12 live rows was
    measured -- and every row past this bound is garbage that no backend reads.
    Trimming before saving takes a capture from ~1.8 GB to well under a megabyte.

    Returns a bound <= num_local_tokens, never above it. A row only arrives here
    because this rank owns one of its experts, so in practice the two are equal;
    computing it rather than assuming it means a stale or over-stated
    num_local_tokens shows up as a smaller file instead of as garbage rows.

    ``num_local_tokens`` stays a device tensor so the row comparison broadcasts on
    the GPU -- the single sync is on the result.
    """
    n = expert_map.numel()
    local = expert_map[dispatch_ids.long().clamp_(0, n - 1)]  # (M, topk)
    rows = torch.arange(dispatch_ids.shape[0], device=dispatch_ids.device)
    live = (local >= 0) & (rows.unsqueeze(1) < num_local_tokens)
    nz = live.any(dim=1).nonzero()
    return 0 if nz.numel() == 0 else int(nz.max()) + 1


def _cfg():
    d = os.getenv("ATOM_EP_MOE_DUMP_DIR")
    if not d:
        return None
    layers = os.getenv("ATOM_EP_MOE_DUMP_LAYERS", "3,17,31,45,60")
    return (
        d,
        {int(x) for x in layers.split(",") if x.strip()},
        # DeepSeek-V4-Pro is MoE in every layer, so the MoE call index within a
        # forward pass IS the layer index. Overridable for other models.
        int(os.getenv("ATOM_EP_MOE_DUMP_NUM_LAYERS", "61")),
    )


def maybe_dump_dispatched(
    dispatch_a1,
    dispatch_scale,
    dispatch_ids,
    dispatch_weights,
    num_local_tokens,
    expert_map,
    expert_mask,
    num_local_experts,
    w1,
    w2,
    fwd_kwargs,
):
    """Snapshot the post-dispatch state, keyed by MoE call index within a pass.

    Records the *shapes* of w1/w2 plus every scalar knob both backends need, but
    never the expert weights themselves -- 48 experts of fp4 is tens of GB, and
    the bench only needs weights that are self-consistent, not the real ones.
    """
    global _CALLS, _PASSES
    cfg = _cfg()
    if cfg is None:
        return
    out_dir, want_layers, n_layers = cfg

    layer = _CALLS % n_layers
    _CALLS += 1
    if layer == n_layers - 1:
        _PASSES += 1
    if layer not in want_layers:
        return
    if torch.cuda.is_current_stream_capturing():
        return  # host work is illegal mid-capture; see module docstring
    is_dummy, regime = _regime()
    if is_dummy:
        return  # startup profile run / cudagraph warmup, not real traffic

    # One rank is enough: the whole point is a representative token->expert
    # distribution, and every rank sees a statistically equivalent one. Dumping
    # all 8 multiplied the output by 8 for no extra information.
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank != int(os.getenv("ATOM_EP_MOE_DUMP_RANK", "0")):
        return

    # Trim the dead tail of the over-allocated receive buffer, THEN compare -- the
    # watermark has to be the row count actually saved. Syncs once on the result;
    # acceptable here and only here, this being an eager-mode debug path.
    keep = _live_rows(dispatch_ids, expert_map, num_local_tokens)
    if not _keep((layer, regime), keep):
        return
    r = int(num_local_tokens.item())
    os.makedirs(out_dir, exist_ok=True)

    def cpu(t, trim=True):
        if t is None:
            return None
        t = t[:keep] if trim else t
        return t.detach().to("cpu", copy=True)

    torch.save(
        {
            "dispatch_a1": cpu(dispatch_a1),
            "dispatch_scale": cpu(dispatch_scale),
            "dispatch_ids": cpu(dispatch_ids),
            "dispatch_weights": cpu(dispatch_weights),
            # Clamped to the trimmed row count, so a consumer's `row <
            # num_local_tokens` mask can never index past the saved rows.
            "num_local_tokens": torch.tensor(
                [min(r, keep)], dtype=num_local_tokens.dtype
            ),
            # Not row-indexed -- must not be trimmed.
            "expert_map": cpu(expert_map, trim=False),
            "expert_mask": cpu(expert_mask, trim=False),
            "num_local_experts": num_local_experts,
            "layer": layer,
            "pass_idx": _PASSES,
            "live_tokens": r,
            "regime": regime,
            # M is the row count actually saved. M_orig is the untrimmed receive
            # buffer height, kept because the gap between them is itself a result:
            # everything after the sort (quant, reduce) is sized by the buffer,
            # not by live rows, so M_orig/M is the factor ATOM_EP_TRIM_PREFILL
            # recovers (measured 11x at M_orig=131072, live=12).
            "M": keep,
            "M_orig": int(dispatch_a1.shape[0]),
            # Shapes/dtypes only -- the bench synthesises its own weights and
            # derives both the flydsl and triton layouts from one source.
            "w1_shape": tuple(w1.shape),
            "w1_dtype": str(w1.dtype),
            "w2_shape": tuple(w2.shape),
            "w2_dtype": str(w2.dtype),
            # The FlyDSL-shuffled scales, recorded for shape/dtype cross-check
            # against what the bench reconstructs from its own source scales.
            "w1_scale_shape": _shape(fwd_kwargs.get("w1_scale")),
            "w1_scale_dtype": _dtype(fwd_kwargs.get("w1_scale")),
            "w2_scale_shape": _shape(fwd_kwargs.get("w2_scale")),
            "w2_scale_dtype": _dtype(fwd_kwargs.get("w2_scale")),
            # activation / quant_type are enums; store the raw value plus repr so
            # the bench can round-trip them without importing ATOM's enum module.
            **{
                f"cfg_{k}": (v if isinstance(v, (int, float, str, bool)) else repr(v))
                for k, v in fwd_kwargs.items()
                if not isinstance(v, torch.Tensor)
            },
        },
        os.path.join(out_dir, f"ep_moe_layer{layer:02d}_{regime}_rank{rank}.pt"),
    )
