# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``GemmFusionSite``: one GEMM followed by a declared epilogue chain.

Model-agnostic apart from the QKV split sizes. ``resolve`` asks aiter which
stage prefix a profile covers for a token bucket (cached, consensus-checked);
``run`` executes the fused prefix through the aiter plan and the remainder with
today's ATOM code. Every call yields a ``Decision`` record.

Binding: the per-call tensors ATOM receives (x_scale, positions, slot_mapping)
are not address-stable, so aiter's pointer bindings are made once, outside graph
capture, against site-owned staging buffers; each call copies the small inputs
into them (device-side, capturable).
"""

import logging
import re
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from . import hooks
from .resources import plan_owner
from .stages import aiter_gf

logger = logging.getLogger("atom.gemm_fusion")


class GemmFusionContractError(RuntimeError):
    """A profile row covers the spec but its contract fails (all ranks)."""


class GemmFusionBindError(RuntimeError):
    """An invariant the fused path relies on was violated (KV page mismatch, a
    re-bound KV pool, a consumer on the wrong stream). Never downgraded to a
    one-rank fallback."""


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
    bind_mode: str = ""  # "staged" | "raw" (1-stage plans take raw pointers)


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

    def declared_page(self):
        """KV page size encoded in the site's HeadNormRopeCache.kv_cache string
        (``shuffle_page{N}_{dtype}_ptok``); aiter reads the page only from there."""
        for st in self.stages:
            kv = getattr(st, "kv_cache", None)
            if kv:
                m = re.search(r"_page(\d+)_", kv)
                return int(m.group(1)) if m else None
        return None

    def bind_caches(self, kv_cache_data) -> None:
        key = f"layer_{self.attn.impl.layer_num}"
        if key not in kv_cache_data:
            raise KeyError(f"gemm_fusion site {self.name}: no KV cache entry {key}")
        # The page in the signature must be the page of the cache aiter will write.
        # K shuffle layout: [blocks, kv_heads, head_dim//x, page, x] -> page = shape[3].
        # Sparse layers re-view the pool as page-16 (any multiple of 16 is fine);
        # dense layers use the pool's own page.
        declared = self.declared_page()
        k = getattr(kv_cache_data[key], "k_cache", None)
        if declared is not None and k is not None and k.dim() == 5:
            pool_page = int(k.shape[3])
            ok = (pool_page % declared == 0) if self.sparse else (pool_page == declared)
            if not ok:
                raise GemmFusionBindError(
                    f"{self.name}: signature declares KV page {declared} but the bound cache "
                    f"has page {pool_page} ({'sparse' if self.sparse else 'dense'} layer)"
                )

    # -- resolution ------------------------------------------------------
    def _static_prologue_ok(self) -> bool:
        """Static (config/env) eligibility of the fused norm/rope/cache stage,
        folded into the cross-rank agreement so it can never differ per call."""
        impl = getattr(self.attn, "impl", None)
        fn = getattr(impl, "gemm_fusion_prologue_supported", None)
        # Fail safe: an impl that cannot say it supports a fused prologue (no impl,
        # or no method) is not eligible -> native on every rank (via the agreement).
        return False if fn is None else bool(fn())

    def resolve(self, tokens: int, graph_mode: str = "eager") -> Resolution:
        """Resolve a bucket. Cache key is ``(tokens, graph_mode)``.

        Only the collective ``preparing`` phase runs match/consensus/ensure. Before
        it every call is native (``not_prepared``); after it a bucket that was not
        resolved is native (``late_bucket``) -- both decided identically on every
        rank without a collective inside the forward.
        """
        key = (tokens, graph_mode)
        r = self._by_tokens.get(key)
        if r is not None:
            if isinstance(r, Exception):
                raise r
            return r
        res = hooks.resources()
        phase = hooks.phase()
        if res is None:
            return Resolution(None, 0, "native", "profile_off")
        if phase == "pre":
            return Resolution(None, 0, "native", "not_prepared")  # not cached
        if phase == "ready":
            r = Resolution(None, 0, "native", "late_bucket")
            self._by_tokens[key] = r
            return r
        gf = aiter_gf()
        contract_err = getattr(gf, "ContractError", ())
        errored, plan, n, msg = False, None, 0, ""
        try:
            # profile passed EXPLICITLY (config.gemm_fusion.profile via hooks), never left to
            # aiter's AITER_GEMM_FUSION_PROFILE env fallback (absent in spawned workers)
            plan, n = gf.match_prefix(
                self.spec, res.match_ctx(res.local_m(tokens), graph_mode), hooks.profile()
            )
        except contract_err as e:  # covered but contract fails
            errored, msg = True, str(e)
        unsupported = False
        if plan is not None and n >= 2 and not self._static_prologue_ok():
            plan, n, unsupported = None, 0, True  # agreed below => native on every rank
        outcome = res.agree(self.name, tokens, plan, n, errored=errored)
        if outcome == "error":
            r = GemmFusionContractError(msg or "contract error on all ranks")
            self._by_tokens[key] = r
            raise r
        if outcome == "disagree" or plan is None:
            if outcome == "disagree":
                reason = "rank_disagree"
            else:
                # a profile row covered the spec but this layer's prologue cannot be
                # fused (static eligibility, identical on every rank): say so, do not
                # report it as a profile miss
                reason = "prologue_unsupported" if unsupported else "uncovered"
            r = Resolution(None, 0, "native", reason)
        elif not res.can_ensure(plan):
            r = Resolution(None, 0, "native", "after_finalize")  # uniform: same order on all ranks
        else:
            # ONE plan object per (site, bucket), shared by the eager and capture keys.
            # match_prefix builds a NEW plan each call and a small-M plan keeps its
            # pointer bindings in a per-INSTANCE dict, so a plan prepared for "eager"
            # would be a different, unbound object when the capture/replay path asks
            # for "capture" ("new binding required during capture"). The profile row
            # does not depend on graph_mode, so the first match is the one to keep.
            other = self._by_tokens.get((tokens, "capture" if graph_mode == "eager" else "eager"))
            if (
                other is not None
                and not isinstance(other, Exception)
                and other.plan is not None
                and other.n_fused == n
                and _plan_vid(other.plan) == _plan_vid(plan)
                and plan_owner(other.plan) == plan_owner(plan)
            ):
                plan = other.plan
            res.ensure(plan)
            hooks.register_plan(plan)
            r = Resolution(plan, n, _plan_vid(plan), "profile_hit")
        self._by_tokens[key] = r
        return r

    @staticmethod
    def _runtime_mode() -> str:
        return "capture" if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing() else "eager"

    def _prepared_ok(self, tokens: int, r: Resolution, prepared: bool) -> bool:
        """This rank's local verdict that bucket ``tokens`` is safe to launch for resolution
        ``r``. Every rank-local invariant the per-call path would otherwise re-check (and raise
        on) is folded in here, at prepare time, and then AGREED across ranks by
        ``agree_prepared``: a failure downgrades the bucket to native on ALL ranks instead
        of raising on one rank in the middle of a step.
          * prepared    : this rank bound the plan (copies into its own staging buffers)
          * plan instance : the plan prepared is the plan this resolution will launch
                            (an unshared instance would need a new binding at run/capture)
        """
        if not prepared:
            return False
        if r.n_fused < 2:
            return True  # 1-stage large-M: raw per-launch pointers, nothing bound per site
        st = self._staging.get(tokens)
        if st is None or not st.prepared:
            return False
        prepared_plan = st.plan
        if prepared_plan is not None and prepared_plan is not r.plan:
            logger.error("%s: bucket %d resolves to a plan instance other than the one prepared", self.name, tokens)
            return False
        return True

    def agree_prepared(self, tokens: int, prepared: bool) -> None:
        """Collective: every rank must have prepared (or not) the same buckets.
        A mismatch downgrades the bucket to native on ALL ranks. This is the ONLY place the
        "can this bucket launch" invariants are decided; the per-call path trusts it."""
        res = hooks.resources()
        for mode in ("eager", "capture"):
            r = self._by_tokens.get((tokens, mode))
            if r is None or isinstance(r, Exception) or r.plan is None:
                continue
            ok = self._prepared_ok(tokens, r, prepared)
            outcome = res.agree(self.name + "#prepared", tokens, None, int(bool(ok)))
            if outcome != "agree" or not ok:
                self._by_tokens[(tokens, mode)] = Resolution(
                    None, 0, "native", "prepare_disagree" if outcome != "agree" else "unprepared"
                )

    # -- KV pointer guard (bind time, collective) --------------------------------
    def kv_ptrs(self, kv_cache_data):
        """Addresses of this layer's KV caches and scales: what a binding is made on."""
        layer = kv_cache_data.get(f"layer_{self.attn.impl.layer_num}")
        if layer is None:
            return None
        return tuple(self._ptr(getattr(layer, n, None)) for n in ("k_cache", "v_cache", "k_scale", "v_scale"))

    def kv_moved(self, kv_cache_data) -> bool:
        """True when a bucket was already prepared against KV caches whose addresses differ
        from ``kv_cache_data``'s (a re-allocated pool). The plan keeps raw pointers from
        prepare, so launching it would write the OLD cache. Evaluated by ``hooks.bind_kv_cache``
        and agreed across ranks there; it is never evaluated per call."""
        now = self.kv_ptrs(kv_cache_data)
        return any(
            st.prepared and st.kv_ptrs not in (None, now) for st in self._staging.values()
        )

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
                plan=None,
                kv_ptrs=None,
                m=m,
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
        """``Plan.run(meta=...)``: the authoritative aiter sp_qkv.small.v1 key contract."""
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

    @staticmethod
    def _ptr(t):
        return 0 if t is None else t.data_ptr()

    @staticmethod
    def decide_path(*, resolution, is_dummy_run, real_tokens, padded_tokens, x_dtype_is_fp8,
                    x_scale_ok):
        """The per-call fused/native decision as a PURE function of rank-uniform inputs
        (resolution agreed at resolve, the broadcast batch's dummy flag, global token counts,
        and the input's dtype/shape facts). No rank, no env, no device state:
        the same inputs give the same ``(path, reason)`` on every rank. -> (path, reason)."""
        if resolution.plan is None:
            return "native", resolution.reason
        if is_dummy_run:
            return "native", "dummy_run"
        if resolution.n_fused >= 2 and real_tokens != padded_tokens:
            return "native", "pad_rows"  # small-M kernel needs T == 4*M with every row live; the 1-stage large-M exchange moves padded rows and the native remainder drops them, so it has no such constraint
        if not x_dtype_is_fp8:
            return "native", "unquantized_input"
        if not x_scale_ok:
            return "native", "xscale_layout"
        return "fused", resolution.reason

    @staticmethod
    def input_facts(x, x_scale):
        """The two rank-uniform input facts ``decide_path`` needs, read from the tensors:
        (x is fp8 e4m3 with a scale, x_scale is a [rows, 1] float32 column)."""
        fp8 = x_scale is not None and x.dtype == torch.float8_e4m3fn
        ok = (
            x_scale is not None
            and x_scale.dim() == 2
            and tuple(x_scale.shape) == (x.shape[0], 1)
            and x_scale.dtype == torch.float32
        )
        return fp8, ok

    # -- binding -----------------------------------------------------------------
    def _bind_plan(self, r, tokens, st, state, device, kv_cache_data):
        """Create aiter's pointer binding for a 2-stage plan outside capture and mark the
        bucket prepared. ``st.xs/pos/slot`` are the buffers the binding is made on."""
        x = torch.zeros((st.m, self.linear.input_size), dtype=torch.float8_e4m3fn, device=device)
        res = hooks.resources()
        res.finalize()
        prep = getattr(r.plan, "prepare", None)
        if prep is not None:
            prep(x=x, x_scale=st.xs, w_scale=self.linear.weight_scale, outs=st.outs,
                 meta=self.build_meta(st, state), res=res.bound)
        st.prepared = True
        st.plan = r.plan  # the exact plan instance whose binding was just created
        st.kv_ptrs = self.kv_ptrs(kv_cache_data) if kv_cache_data else None  # for kv_moved()
        return True

    @staticmethod
    def _prepare_one_stage(r):
        """1-stage prefix (large M): the plan takes raw per-launch pointers, no staging; just
        attach + init control after finalize, before capture."""
        hooks.resources().finalize()
        prep = getattr(r.plan, "prepare", None)
        if prep is not None:
            prep(res=hooks.resources().bound)
        return True

    def prepare_bucket(self, tokens, device, kv_cache_data, out_dtype):
        """Bind the plan for one bucket OUTSIDE graph capture (padded == real).

        Called after ``bind_kv_cache`` and the bucket's resolution, before capture."""
        r = self.resolve(tokens, "eager")
        if r.plan is None:
            return False
        if r.n_fused < 2:
            return self._prepare_one_stage(r)
        st = self._stage(tokens, device, out_dtype)
        if st.prepared:
            return True
        state = self.attn.impl.gemm_fusion_prologue_state(
            _ShimFwdCtx(kv_cache_data, st.slot, self.attn.impl)
        )
        if state is None:
            return False
        return self._bind_plan(r, tokens, st, state, device, kv_cache_data)

    # -- execution -------------------------------------------------------
    def _prepare(self, ctx, x, x_scale):
        """-> (Resolution, reason, state, staging).

        PER-CALL DECISION. ``decide_path`` is a pure function of rank-uniform inputs, so every
        rank takes the same path for the same call:
          resolution   : agreed at resolve (collective consensus), cached, and downgraded
                         uniformly by ``agree_prepared`` (prepared / plan instance are folded
                         into THAT agreement, at prepare time).
          dummy_run    : ``batch.is_dummy_run`` comes from ONE pickled batch that
                         ``AsyncIOProcManager.call_func`` broadcasts to every worker of the
                         engine core (rpc_broadcast_mq.enqueue); SP ranks share that core.
          real/padded  : ``real_tokens`` = positions.shape[-1] (global, positions stay whole
                         under SP), ``padded_tokens`` = local rows * world, equal on every rank.
          x dtype / x_scale shape : follow from the model (SPMD) and equal local rows.
        There is NO rank-local raise in this method: the "unprepared bucket", "prologue state
        unavailable" and "other plan instance" conditions are unreachable after
        ``agree_prepared`` (they downgrade the bucket to native on every rank, or fail
        ``hooks.bind_kv_cache`` on every rank), so nothing here can make one rank leave a step
        while its peers wait in a fused kernel.

        What this does NOT make safe: a non-uniform decision caused by a bug or a fault (a
        one-rank exception before launch). The step-end digest in ``hooks.check_step`` only
        sees steps that COMPLETE on every rank; a divergence that blocks mid-step (one rank in
        the native RCCL all_to_all, the others in the fused CCO kernel) never reaches it. Such a
        divergence is FAIL-STOP at the process-group timeout.
        """
        r = self.resolve(ctx.padded_tokens, self._runtime_mode())
        reason = r.reason
        if r.plan is None:
            return r, reason, None, None
        fp8, scale_ok = self.input_facts(x, x_scale)
        path, why = self.decide_path(
            resolution=r,
            is_dummy_run=ctx.is_dummy_run,
            real_tokens=ctx.real_tokens,
            padded_tokens=ctx.padded_tokens,
            x_dtype_is_fp8=fp8,
            x_scale_ok=scale_ok,
        )
        if path == "native":
            return Resolution(None, 0, "native", why), why, None, None
        if r.n_fused < 2:
            return r, reason, None, None
        st = self._staging[ctx.padded_tokens]  # prepared on every rank (agree_prepared)
        state = ctx.prologue_state()  # eligibility was agreed at resolve; never None here
        return r, reason, state, st

    def _stage_inputs(self, x_scale, ctx, state, st):
        """Copy the per-call inputs into the buffers the plan is bound to (device-side,
        capturable)."""
        n = ctx.real_tokens
        st.xs.copy_(x_scale)
        st.pos[:n].copy_(ctx.positions[:n])
        st.slot[:n].copy_(state["slot_mapping"][:n])

    def _execute(self, r, x, x_scale, ctx, state, st, out_dtype):
        """Run the fused prefix; return its ``outs`` dict."""
        res = hooks.resources()
        res.finalize()  # collective + idempotent; before any plan executes
        if r.n_fused >= 2:
            self._stage_inputs(x_scale, ctx, state, st)
            # the plan keeps its pointer bindings per INSTANCE; that the plan launched here is the
            # one prepared is verified at prepare time (`_prepared_ok` via `agree_prepared`),
            # not per call, so no rank can raise here while its peers launch.
            outs, meta, xs = st.outs, self.build_meta(st, state), st.xs
        else:
            outs, meta, xs = {}, {}, x_scale  # large M: raw per-launch pointers
        r.plan.run(
            x=x, x_scale=xs, w=self.linear.weight, w_scale=self.linear.weight_scale,
            outs=outs, meta=meta, res=res.bound, stream=torch.cuda.current_stream(),
        )
        if r.n_fused < 2:
            # Zero-copy view of the plan's receive window; valid until this rank
            # launches the next large epoch (of ANY plan under a shared window), i.e. past
            # the native remainder that consumes it (same stream, in order). Reuse layers get
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

    def _bind_label(self, r, path, st):
        return "raw" if (path == "fused" and r.n_fused < 2) else "staged"

    def _decision(self, ctx, r, reason, path, st=None):
        d = Decision(self.name, ctx.padded_tokens, r.vid, r.n_fused, path, reason,
                     self._bind_label(r, path, st))
        hooks.record_decision(d)
        return d

    def _remainder_after_exchange(self, r, outs, ctx, consumer):
        """Run the native remainder on a large-M recv_view. The view is only valid on the
        launching stream until this rank's next large launch, so the launch and every
        consumer must be on ``torch.cuda.current_stream()``."""
        s0 = torch.cuda.current_stream()
        out = consumer(outs["packed"])
        if torch.cuda.current_stream() != s0:
            raise GemmFusionBindError("recv_view consumer ran on a different stream")
        return out

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
            out = self._remainder_after_exchange(
                r, outs, ctx, lambda packed: self.native.after_exchange(self, packed, ctx)
            )
        return out, self._decision(ctx, r, reason, "fused", st)


class _ShimFwdCtx:
    """Just enough forward context for ``gemm_fusion_prologue_state`` before
    capture: the bound KV caches and a slot-mapping holder."""

    def __init__(self, kv_cache_data, slot_mapping, impl):
        self.kv_cache_data = kv_cache_data
        self.attn_metadata = SimpleNamespace(slot_mapping=slot_mapping)


def _out_dtype():
    from atom.config import get_current_atom_config

    return get_current_atom_config().torch_dtype
