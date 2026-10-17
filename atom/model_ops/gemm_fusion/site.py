# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``GemmFusionSite``: one GEMM followed by a declared epilogue chain.

Model-agnostic apart from the QKV split sizes. ``resolve`` asks aiter which
stage prefix a profile covers for a token bucket (cached, consensus-checked);
``run`` executes the fused prefix through the aiter plan and the remainder with
today's ATOM code. Every call yields a ``Decision`` record (evidence).
``run_prologue`` stops after the norm/rope/cache stage (no attention); it is the
entry the golden-harness candidate adapter uses.
"""

import logging
from dataclasses import dataclass
from types import SimpleNamespace

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
    return getattr(plan, "vid", None) or type(plan).__name__


def _rows2d(t):
    """[max_pos, 1, 1, rot/2] (ATOM rotary) or [max_pos, rot/2] -> [max_pos, rot/2]."""
    return t if t.dim() == 2 else t.reshape(t.shape[0], -1)


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
        self._wcache: dict = {}
        self._staging: dict = {}
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
                reason = "rank_disagree" if outcome == "disagree" else "uncovered"
                r = Resolution(None, 0, "native", reason)
            elif not res.can_ensure(plan):
                # windows are frozen (finalize ran); rank-uniform by construction
                r = Resolution(None, 0, "native", "after_finalize")
            else:
                res.ensure(plan)
                r = Resolution(plan, n, _plan_vid(plan), "profile_hit")
        self._by_tokens[tokens] = r
        return r

    # -- persistent staging (aiter binds raw pointers once per tensor set) ------
    def _stage(self, tokens, device, dtype):
        """Per-bucket buffers whose addresses never move: aiter's plan binds
        outs/x_scale/positions/slot_mapping/caches by pointer outside capture, and
        the per-call tensors ATOM receives are not address-stable. Each call
        copies the small per-call inputs into these (device-side, capturable).
        """
        st = self._staging.get(tokens)
        if st is None:
            impl = self.attn.impl
            m = tokens // max(1, hooks.world_or_1())
            outs = {"q": torch.empty((tokens, impl.num_heads, impl.head_dim), dtype=dtype, device=device)}
            if self.sparse and not self.skip_index:  # indexer: index_q, NO k/v
                outs["index_q"] = torch.empty(
                    (tokens, impl.index_q_size // impl.index_head_dim, impl.index_head_dim),
                    dtype=dtype, device=device,
                )
            else:  # dense / reuse
                kv = (tokens, impl.num_kv_heads, impl.head_dim)
                outs["k"] = torch.empty(kv, dtype=dtype, device=device)
                outs["v"] = torch.empty(kv, dtype=dtype, device=device)
            st = SimpleNamespace(
                outs=outs,
                xs=torch.empty((m, 1), dtype=torch.float32, device=device),
                pos=torch.empty((tokens,), dtype=torch.int64, device=device),
                slot=torch.empty((tokens,), dtype=torch.int64, device=device),
                prepared=False,
            )
            self._staging[tokens] = st
        return st

    def _f32(self, key, t):
        c = self._wcache.get(key)
        if c is None or c[0] is not t:
            c = (t, t.float().contiguous())
            self._wcache[key] = c
        return c[1]

    def build_meta(self, st, state) -> dict:
        """``Plan.run(meta=...)``: the authoritative aiter sp_qkv.small.v1 key
        contract (aiter-integration-notes.md, S2-2b-ii)."""
        impl = self.attn.impl
        rot = impl.rotary_emb
        meta = {
            "positions": st.pos,
            "slot_mapping": st.slot,
            "k_cache": state["k_cache"],
            # the prologue writes the SHUFFLE view of V (== what rope_cache writes)
            "v_cache": state.get("v_cache_shuffle", state["v_cache"]),
            "k_scale": state.get("k_scale_view", state.get("k_scale")),
            "v_scale": state.get("v_scale_view", state.get("v_scale")),
            "cos_cache": _rows2d(rot.cos_cache),
            "sin_cache": _rows2d(rot.sin_cache),
            "q_norm_w": self._f32("qw", impl.q_norm.weight),
            "k_norm_w": self._f32("kw", impl.k_norm.weight),
        }
        if self.sparse and not self.skip_index:
            meta["index_cache"] = impl.index_cache
            meta["index_q_norm_w"] = self._f32("iqw", impl.index_q_norm.weight)
            meta["index_k_norm_w"] = self._f32("ikw", impl.index_k_norm.weight)
        return meta

    def prepare_bucket(self, tokens, device, kv_cache_data, out_dtype, pad_rows=False):
        """Bind the plan for one (padded == real) bucket OUTSIDE graph capture.

        Called after ``bind_kv_cache`` and the bucket's resolution, before capture.
        """
        r = self.resolve(tokens)
        if r.plan is None:
            return False
        if r.n_fused < 2:
            # 1-stage prefix (large M): the plan takes raw per-launch pointers, no
            # staging; just attach + init control after finalize, before capture.
            hooks.resources().finalize()
            prep = getattr(r.plan, "prepare", None)
            if prep is not None:
                prep(res=hooks.resources().bound)
            return True
        if pad_rows:
            return False  # run time falls back (pad_rows); nothing to bind
        st = self._stage(tokens, device, out_dtype)
        if st.prepared:
            return True
        state = self.attn.impl.gemm_fusion_prologue_state(_ShimFwdCtx(kv_cache_data, st.slot, self.attn.impl))
        if state is None:
            return False
        x = torch.zeros((st.xs.shape[0], self.linear.input_size), dtype=torch.float8_e4m3fn, device=device)
        res = hooks.resources()
        res.finalize()
        prep = getattr(r.plan, "prepare", None)
        if prep is not None:
            prep(x=x, x_scale=st.xs, w_scale=self.linear.weight_scale, outs=st.outs,
                 meta=self.build_meta(st, state), res=res.bound)
        st.prepared = True
        return True

    # -- execution -------------------------------------------------------
    def _fallback_reason(self, ctx, x, x_scale, n_fused=2):
        """Why the fused prefix cannot run for this call (None = it can)."""
        if ctx.is_dummy_run:
            return "dummy_run"
        if n_fused >= 2 and ctx.real_tokens != ctx.padded_tokens:
            # small-M kernel needs T == 4*M with every row live. The 1-stage large-M
            # exchange moves padded rows too and the native remainder drops them
            # (attend_packed slices [:real]), so it has no such constraint.
            return "pad_rows"
        if x_scale is None or x.dtype != torch.float8_e4m3fn:
            return "unquantized_input"
        if x_scale.dim() != 2 or x_scale.shape != (x.shape[0], 1) or x_scale.dtype != torch.float32:
            return "xscale_layout"
        return None

    def _prepare(self, ctx, x, x_scale):
        """-> (Resolution, reason, state, staging)."""
        r = self.resolve(ctx.padded_tokens)
        reason = r.reason
        if r.plan is None:
            return r, reason, None, None
        bad = self._fallback_reason(ctx, x, x_scale, r.n_fused)
        if bad is None and r.n_fused >= 2:
            state = ctx.prologue_state()
            if state is None:
                bad = "prologue_unsupported"
        if bad is not None:
            return Resolution(None, 0, "native", bad), bad, None, None
        st = None
        if r.n_fused >= 2:
            st = self._stage(ctx.padded_tokens, x.device, _out_dtype_from(ctx))
        return r, reason, (state if r.n_fused >= 2 else None), st

    def _execute(self, r, x, x_scale, ctx, state, st, out_dtype):
        """Run the fused prefix; return its ``outs`` dict."""
        res = hooks.resources()
        res.finalize()  # collective + idempotent; before any plan executes
        if r.n_fused >= 2:
            n = ctx.real_tokens
            st.xs.copy_(x_scale)
            st.pos[:n].copy_(ctx.positions[:n])
            st.slot[:n].copy_(state["slot_mapping"][:n])
            outs, meta, xs = st.outs, self.build_meta(st, state), st.xs
        else:
            outs, meta, xs = {}, {}, x_scale  # large M: raw per-launch pointers
        r.plan.run(
            x=x, x_scale=xs, w=self.linear.weight, w_scale=self.linear.weight_scale,
            outs=outs, meta=meta, res=res.bound, stream=torch.cuda.current_stream(),
        )
        if r.n_fused < 2:
            # Zero-copy view of the plan's receive window; valid until this rank
            # launches the next epoch of the SAME plan, i.e. past the native
            # remainder that consumes it (same stream, in order). Reuse layers get
            # the [T,2560] row-stride-2304 view: the IQ|IK columns alias the next
            # row and are never read (skip branch), and rope_cache's
            # qkv.contiguous() copies the view with those semantics intact.
            plan = r.plan
            if getattr(plan, "mode", None) == "reuse" and hasattr(plan, "reuse_view"):
                view = plan.reuse_view(res.bound)
            else:
                view = plan.recv_view(res.bound)
            outs = {"packed": view}
        return outs

    def _decision(self, ctx, r, reason, path):
        d = Decision(self.name, ctx.padded_tokens, r.vid, r.n_fused, path, reason)
        hooks.record_decision(d)
        return d

    def run(self, x, x_scale, ctx):
        """Full QKV+attention. -> ((output, scale), Decision)."""
        r, reason, state, st = self._prepare(ctx, x, x_scale)
        if r.plan is None:
            out = self.native.full(self, x, x_scale, ctx)
            return out, self._decision(ctx, r, reason, "native")
        outs = self._execute(r, x, x_scale, ctx, state, st, _out_dtype())
        if r.n_fused >= 2:
            out = self.native.after_prologue(self, outs, state, ctx)
        else:
            out = self.native.after_exchange(self, outs["packed"], ctx)
        return out, self._decision(ctx, r, reason, "fused")

    def run_prologue(self, x, x_scale, ctx, native_full, native_after_exchange, out_dtype):
        """GEMM + exchange + norm/rope/cache only (no attention).

        Returns ``(kind, payload, Decision)``: ``("outs", dict)`` when the fused
        prefix covered the norm/rope/cache stage, else ``("native", result)``
        from ``native_full(x, x_scale)`` / ``native_after_exchange(packed)``.
        """
        r, reason, state, st = self._prepare(ctx, x, x_scale)
        if r.plan is None:
            return "native", native_full(x, x_scale), self._decision(ctx, r, reason, "native")
        outs = self._execute(r, x, x_scale, ctx, state, st, out_dtype)
        if r.n_fused >= 2:
            return "outs", outs, self._decision(ctx, r, reason, "fused")
        return "native", native_after_exchange(outs["packed"]), self._decision(ctx, r, reason, "fused")


class _ShimFwdCtx:
    """Just enough forward context for ``gemm_fusion_prologue_state`` before
    capture: the bound KV caches and a slot-mapping holder."""

    def __init__(self, kv_cache_data, slot_mapping, impl):
        self.kv_cache_data = kv_cache_data
        self.attn_metadata = SimpleNamespace(slot_mapping=slot_mapping)


def _out_dtype():
    from atom.config import get_current_atom_config

    return get_current_atom_config().torch_dtype


def _out_dtype_from(ctx):
    return _out_dtype()
