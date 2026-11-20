"""ATOM_FOLD_POSTNORM (default off): the verify pass's post-attention RMSNorm folded away (Qwen3.5 / 3.6, gfx1250).

The layer's last projection (GDN out_proj / attention o_proj, an MXFP8 1x32 Linear) adds the residual in its
epilogue and stores, besides the residual sum s = bf16(o) + r (the next residual, as the norm's), the per-tile
fp32 sums of squares of s's rows. The MoE block's fold D router kernel (aiter v2, <= 16 rows) then takes s itself:
router and shared gate/up weights pre-scaled by (1 + w) along K, accumulators scaled by rstd, and its route tiles
normalize their row before the MXFP4 expert-activation quant. One launch (the norm) fewer a layer.

Not bitwise the unfolded path: the router / shared-expert GEMMs see bf16(s) @ bf16(W * (1 + w)) * rstd instead of
bf16(s * rstd * (1 + w)) @ W, and the experts' input is normalized from bf16(s) instead of the fp32 sum.

ATOM_FOLD_POSTNORM_MAX_ROWS (default 16, at most 16: the v2 kernel's narrow variant) bounds the rows.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import torch

ENABLED = os.environ.get("ATOM_FOLD_POSTNORM", "0") == "1"
MAX_ROWS = min(16, int(os.environ.get("ATOM_FOLD_POSTNORM_MAX_ROWS", "16")))

_stats = {"folded": 0, "materialized": 0}


class PreNormed(NamedTuple):
    """The post-attention norm's input after the residual add, not yet normalized."""

    x: torch.Tensor  # (M, K) bf16 residual sum (also the layer's next residual)
    ssq: torch.Tensor  # (M, NSSQ) fp32 per-tile sums of squares of x's rows (unrounded)
    norm: torch.nn.Module  # the GemmaRMSNorm folded away

    @property
    def shape(self):
        return self.x.shape

    def materialize(self) -> torch.Tensor:
        """The normalized rows (fallback): the norm on x without residual (x is the rounded sum)."""
        _stats["materialized"] += 1
        return self.norm(self.x)


def request(proj, residual: torch.Tensor) -> None:
    """Ask the next forward of `proj` (an MXFP8 1x32 Linear) for the residual + ssq epilogue."""
    proj.__dict__["_pn_residual"] = residual


def take(proj):
    """The ssq of `proj`'s last forward if it took the request, else None (and the request is dropped)."""
    proj.__dict__.pop("_pn_residual", None)
    return proj.__dict__.pop("_pn_ssq", None)


def linear_wants(layer, m: int):
    """The residual `layer`'s forward should add (and consume the request), or None."""
    r = layer.__dict__.pop("_pn_residual", None)
    if (
        r is None
        or m == 0
        or m > MAX_ROWS
        or r.shape[0] != m
        or r.dtype != torch.bfloat16
    ):
        return None
    return r


def prescaled(block, norm) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(router weight padded, shared gate_up weight), both * (1 + norm.weight) along K, cached on the block;
    None while capturing a graph before they exist (the caller then runs the unfolded path).
    """
    cache = block.__dict__.get("_pn_weights")
    wr = block._router_weight_padded()
    wgu = block.shared_expert.gate_up_proj.weight
    key = (wr.data_ptr(), wgu.data_ptr(), norm.weight.data_ptr())
    if cache is not None and cache[0] == key:
        return cache[1], cache[2]
    if torch.cuda.is_current_stream_capturing():
        return None
    s = norm.weight.float() + 1.0
    pwr = (wr.float() * s[None, :]).to(wr.dtype).contiguous()
    pwgu = (wgu.float() * s[None, :]).to(wgu.dtype).contiguous()
    block.__dict__["_pn_weights"] = (key, pwr, pwgu)
    return pwr, pwgu
