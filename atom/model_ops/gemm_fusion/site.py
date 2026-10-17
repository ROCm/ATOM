# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``GemmFusionSite``: one GEMM followed by a declared epilogue chain.

Model-agnostic apart from the QKV split sizes. ``resolve`` asks aiter which
stage prefix a profile covers for a token bucket (cached, consensus-checked);
``run`` executes the fused prefix through the aiter plan and the remainder with
today's ATOM code. Every call yields a ``Decision`` record (evidence).
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from . import hooks
from .stages import aiter_gf

logger = logging.getLogger("atom.gemm_fusion")


class GemmFusionContractError(RuntimeError):
    """A profile row covers the spec but its contract fails (all ranks)."""


@dataclass(frozen=True)
class Resolution:
    plan: object  # aiter Plan or None
    n_fused: int  # stages covered (0 = all native)
    vid: str = "native"
    reason: str = ""


@dataclass
class Decision:
    site: str
    tokens: int
    vid: str
    n_fused: int
    path: str  # "native" | "fused"
    reason: str = ""


def _plan_vid(plan) -> str:
    if plan is None:
        return "native"
    for obj in (plan, getattr(plan, "info", None)):
        vid = getattr(obj, "vid", None)
        if vid:
            return vid
    return type(plan).__name__


class GemmFusionSite:
    def __init__(
        self,
        name,
        linear,
        stages,
        native_fns,
        *,
        attn,
        q_size,
        kv_size,
        index_q_size=0,
        index_k_size=0,
        sparse=False,
        skip_index=False,
        register=True,
    ):
        gf = aiter_gf()
        self.name = name
        self.linear = linear
        self.stages = tuple(stages)
        self.native = native_fns
        self.attn = attn
        self.q_size, self.kv_size = q_size, kv_size
        self.index_q_size, self.index_k_size = index_q_size, index_k_size
        self.sparse, self.skip_index = sparse, skip_index
        gemm = gf.GemmSpec(
            quant="fp8_ptpc",
            n=getattr(linear, "output_size"),
            k=getattr(linear, "input_size"),
        )
        self.spec = gf.FusionSpec(gemm, self.stages)
        self._by_tokens: dict = {}
        if register:
            hooks.register_site(self)

    # -- layout helpers --------------------------------------------------
    def split_qkv(self, qkv):
        """Views of q/k/v as today's model forward produces them."""
        if self.sparse:
            q, k, v, _, _ = qkv.split(
                [self.q_size, self.kv_size, self.kv_size, self.index_q_size, self.index_k_size],
                dim=-1,
            )
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return q, k, v

    def packed_width(self, impl) -> int:
        """Columns of the exchanged rows (== ``_exchange_columns`` output)."""
        w = impl.num_heads * impl.head_dim + 2 * impl.num_kv_heads * impl.head_dim
        if self.sparse:
            w += impl.index_q_size + impl.index_head_dim
        return w

    def bind_caches(self, kv_cache_data) -> None:
        key = f"layer_{self.attn.impl.layer_num}"
        if key not in kv_cache_data:
            raise KeyError(f"gemm_fusion site {self.name}: no KV cache entry {key}")

    # -- resolution ------------------------------------------------------
    def resolve(self, tokens: int, graph_mode: str = "eager") -> Resolution:
        r = self._by_tokens.get(tokens)
        if r is not None:
            if isinstance(r, Exception):
                raise r
            return r
        res = hooks.resources()
        if res is None:
            r = Resolution(None, 0, "native", "profile_off")
        else:
            gf = aiter_gf()
            contract_err = getattr(gf, "ContractError", ())
            errored, plan, n, msg = False, None, 0, ""
            try:
                plan, n = gf.match_prefix(
                    self.spec, res.match_ctx(res.local_m(tokens), graph_mode)
                )
            except contract_err as e:  # covered but contract fails
                errored, msg = True, str(e)
            outcome = res.agree(self.name, tokens, plan, n, errored=errored)
            if outcome == "error":
                r = GemmFusionContractError(msg or "contract error on all ranks")
                self._by_tokens[tokens] = r
                raise r
            if outcome == "disagree" or plan is None:
                r = Resolution(
                    None, 0, "native", "rank_disagree" if outcome == "disagree" else "uncovered"
                )
            else:
                res.ensure(plan.resources())
                r = Resolution(plan, n, _plan_vid(plan), "profile_hit")
        self._by_tokens[tokens] = r
        return r

    # -- execution -------------------------------------------------------
    def run(self, x, x_scale, ctx):
        """-> ((output, scale), Decision)."""
        tokens = ctx.padded_tokens
        r = self.resolve(tokens)
        reason = r.reason
        if r.plan is not None and ctx.is_dummy_run:
            r, reason = Resolution(None, 0, "native", "dummy_run"), "dummy_run"
        state = None
        if r.plan is not None and r.n_fused >= 2:
            state = ctx.prologue_state()
            if state is None:
                r, reason = Resolution(None, 0, "native", "prologue_unsupported"), "prologue_unsupported"
        if r.plan is None:
            out = self.native.full(self, x, x_scale, ctx)
            d = Decision(self.name, tokens, "native", 0, "native", reason)
            hooks.record_decision(d)
            return out, d

        outs = ctx.alloc_outputs(
            [s.kind for s in self.stages[: r.n_fused]], self, x.device, _out_dtype()
        )
        meta = dict(state or {})
        if state is not None and "v_cache_shuffle" in state:
            meta["v_cache"] = state["v_cache_shuffle"]  # prologue writes the shuffle view
        meta["positions"] = ctx.positions
        meta["real_tokens"] = ctx.real_tokens  # kernel must not touch pad rows
        r.plan.run(
            x=x,
            x_scale=x_scale,
            w=self.linear.weight,
            w_scale=self.linear.weight_scale,
            outs=outs,
            meta=meta,
            res=hooks.resources().bound,
            stream=torch.cuda.current_stream(),
        )
        if r.n_fused >= 2:
            out = self.native.after_prologue(self, outs, state, ctx)
        else:
            out = self.native.after_exchange(self, outs["packed"], ctx)
        d = Decision(self.name, tokens, r.vid, r.n_fused, "fused", reason)
        hooks.record_decision(d)
        return out, d


def _out_dtype():
    from atom.config import get_current_atom_config

    return get_current_atom_config().torch_dtype
