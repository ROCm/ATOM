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
import re
from dataclasses import dataclass
from types import SimpleNamespace

import torch

import os

from . import faults, hooks
from .resources import plan_owner
from .stages import aiter_gf

logger = logging.getLogger("atom.gemm_fusion")


class GemmFusionContractError(RuntimeError):
    """A profile row covers the spec but its contract fails (all ranks)."""


class GemmFusionBindError(RuntimeError):
    """An invariant the fused path relies on was violated (unprepared bucket,
    moved KV cache, wrong stream). Never downgraded to a one-rank fallback."""


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
    bind_mode: str = ""  # "direct" | "staged" | "raw" (1-stage plans take raw pointers)


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
        bind_mode="staged",
        direct_provider=None,
    ):
        gf = aiter_gf()
        if bind_mode not in ("direct", "staged"):
            raise ValueError(f"bind_mode must be 'direct' or 'staged', got {bind_mode!r}")
        # 'staged': per-call copies into site-owned buffers (eager, unstable
        # addresses). 'direct': plans bind the caller's own x_scale/positions/
        # slot_mapping/cache tensors at prepare time, no per-call copies; the
        # caller guarantees stable addresses (graph static buffers, harness
        # cases). A changed address is never used silently: reason 'rebind_needed'.
        self.bind_mode = bind_mode
        # optional callable(tokens) -> {"x_scale","positions","slot_mapping","state"}
        self.direct_provider = direct_provider
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
        self._bind_dump: dict = {}  # tokens -> (where, binding_key) from the first dump
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
        """Resolve a bucket. Cache key is ``(tokens, graph_mode)`` (L-2).

        Only the collective ``preparing`` phase runs match/consensus/ensure. Before
        it every call is native (``not_prepared``); after it a bucket that was not
        resolved is native (``late_bucket``) -- both decided identically on every
        rank without a collective inside the forward (L-3, M-1).
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

    def _runtime_mode(self) -> str:
        return "capture" if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing() else "eager"

    def _prepared_ok(self, tokens: int, r: Resolution, prepared: bool) -> bool:
        """This rank's local verdict that bucket ``tokens`` is safe to launch for resolution
        ``r``. Every rank-local invariant the per-call path used to re-check (and raise on)
        is folded in here, at prepare time, and then AGREED across ranks by
        ``agree_prepared``: a failure downgrades the bucket to native on ALL ranks instead
        of raising on one rank in the middle of a step.
          * prepared            : this rank bound the plan (staged: copies into its buffers;
                                  direct: caller tensors)
          * plan instance      : the plan prepared is the plan this resolution will launch
                                  (an unshared instance would need a new binding at run/capture)
          * direct KV pointers : the caller's KV cache/scale addresses are still the ones the
                                  plan was bound to (``kv_moved``)
        """
        if not prepared:
            return False
        if r.n_fused < 2:
            return True  # 1-stage large-M: raw per-launch pointers, nothing bound per site
        st = self._staging.get(tokens)
        if st is None or not st.prepared:
            return False
        prepared_plan = getattr(st, "plan", None)
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

    # -- staged-mode KV pointer guard (bind time, collective) ------------------
    def kv_ptrs(self, kv_cache_data):
        """Addresses of this layer's KV caches and scales: what a staged binding is made on."""
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
            getattr(st, "prepared", False) and getattr(st, "kv_ptrs", None) not in (None, now)
            for st in self._staging.values()
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
            staged = self.bind_mode == "staged"
            st = SimpleNamespace(
                outs=outs,
                # direct mode: xs/pos/slot are the caller's tensors, set at prepare
                xs=torch.empty((m, 1), dtype=torch.float32, device=device) if staged else None,
                pos=torch.empty((tokens,), dtype=torch.int64, device=device) if staged else None,
                slot=torch.empty((tokens,), dtype=torch.int64, device=device) if staged else None,
                prepared=False,
                rebound=False,
                ptrs=None,
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

    @staticmethod
    def _ptr(t):
        return 0 if t is None else t.data_ptr()

    @staticmethod
    def decide_path(*, resolution, is_dummy_run, real_tokens, padded_tokens, x_dtype_is_fp8,
                    x_scale_ok):
        """The per-call fused/native decision as a PURE function of rank-uniform inputs
        (resolution agreed at resolve, the broadcast batch's dummy flag, global token counts,
        and the input's dtype/shape facts). No rank, no env, no device state, no fault seam:
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

    def binding_key(self, st, state, x_scale=None) -> dict:
        """Named data_ptr of every tensor aiter's small plan keys its binding on
        (variants.py ``_binding``: x_scale, w_scale, norms, cos/sin, positions,
        slot_mapping, q out, caches, k/v/index_q outs, index_cache) plus the row count.
        Used by the debug dump below: diffing the dump at prepare against the dump at
        run/capture names the tensor whose address moved."""
        meta = self.build_meta(st, state)
        d = {"x_scale": self._ptr(x_scale if x_scale is not None else st.xs),
             "w_scale": self._ptr(self.linear.weight_scale)}
        for k in ("q_norm_w", "k_norm_w", "index_q_norm_w", "index_k_norm_w", "cos_cache", "sin_cache",
                  "positions", "slot_mapping", "k_cache", "v_cache", "k_scale", "v_scale", "index_cache"):
            if k in meta:
                d[k] = self._ptr(meta[k])
        for k, v in st.outs.items():
            d["out_" + k] = self._ptr(v)
        d["rows"] = st.m
        return d

    def _dump_binding(self, where, tokens, st, state, x_scale=None):
        """``ATOM_GEMM_FUSION_DEBUG_BIND=1``: log the binding key at prepare vs run."""
        if os.environ.get("ATOM_GEMM_FUSION_DEBUG_BIND") != "1":
            return
        key = self.binding_key(st, state, x_scale)
        first = self._bind_dump.setdefault(tokens, (where, key))
        diff = {k: (first[1].get(k), v) for k, v in key.items() if first[1].get(k) != v}
        logger.warning(
            "gemm_fusion bind[%s] %s tokens=%d plan=%s mode=%s %s",
            where, self.name, tokens, id(st), self._runtime_mode(),
            "SAME as " + first[0] if not diff else "DIFFERS from " + first[0] + ": " + repr(diff),
        )

    def _live_ptrs(self, x_scale, positions, state):
        """Addresses the plan binds (small-M); compared at run time in direct mode."""
        return (
            self._ptr(x_scale),
            self._ptr(positions),
            self._ptr(state["slot_mapping"]),
            self._ptr(state["k_cache"]),
            self._ptr(state.get("v_cache_shuffle", state["v_cache"])),
            self._ptr(state.get("k_scale_view", state.get("k_scale"))),
            self._ptr(state.get("v_scale_view", state.get("v_scale"))),
        )

    def prepare_bucket(self, tokens, device, kv_cache_data, out_dtype, pad_rows=False, direct=None):
        """Bind the plan for one (padded == real) bucket OUTSIDE graph capture.

        Called after ``bind_kv_cache`` and the bucket's resolution, before capture.
        ``direct`` (direct mode only; else ``direct_provider(tokens)``):
        ``{"x_scale","positions","slot_mapping","state"}`` -- the caller's stable
        tensors, bound as-is.
        """
        r = self.resolve(tokens, "eager")
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
        if self.bind_mode == "direct":
            if direct is None and self.direct_provider is not None:
                direct = self.direct_provider(tokens)
            if direct is None:
                return st.prepared  # no stable tensors to bind; agree_prepared() downgrades uniformly
            if st.prepared:
                # Already bound. The KV cache/scale addresses are fixed at KV allocation: if the
                # caller's differ from the ones bound, the plan would write a stale cache. Report
                # "not prepared" so agree_prepared() downgrades the bucket to native on ALL ranks
                # (moved x_scale/positions/slot are self-healed by copy at run time instead).
                live = self._live_ptrs(direct["x_scale"], direct["positions"], direct["state"])
                return live[3:] == st.ptrs[3:]
            state = direct["state"]
            st.xs, st.pos, st.slot = direct["x_scale"], direct["positions"], direct["slot_mapping"]
            st.ptrs = self._live_ptrs(st.xs, st.pos, state)
        else:
            if st.prepared:
                return True
            state = self.attn.impl.gemm_fusion_prologue_state(
                _ShimFwdCtx(kv_cache_data, st.slot, self.attn.impl)
            )
            if state is None:
                return False
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
        self._dump_binding("prepare", tokens, st, state)
        return True

    # -- execution -------------------------------------------------------
    def _prepare(self, ctx, x, x_scale):
        """-> (Resolution, reason, state, staging).

        PER-CALL DECISION, M-1 / E-3. ``decide_path`` is a pure function of rank-uniform
        inputs, so every rank takes the same path for the same call:
          resolution   : agreed at resolve (collective consensus), cached, and downgraded
                         uniformly by ``agree_prepared`` (prepared / plan instance / direct KV
                         pointers are all folded into THAT agreement, at prepare time).
          dummy_run    : ``batch.is_dummy_run`` comes from ONE pickled batch that
                         ``AsyncIOProcManager.call_func`` broadcasts to every worker of the
                         engine core (rpc_broadcast_mq.enqueue); SP ranks share that core.
          real/padded  : ``real_tokens`` = positions.shape[-1] (global, positions stay whole
                         under SP), ``padded_tokens`` = local rows * world, equal on every rank.
          x dtype / x_scale shape : follow from the model (SPMD) and equal local rows.
        There is NO rank-local raise in this method any more: the "unprepared bucket",
        "prologue state unavailable", "KV address moved" and "other plan instance" conditions
        are unreachable after ``agree_prepared`` (they would downgrade the bucket to native on
        every rank, or fail ``hooks.bind_kv_cache`` on every rank), so nothing here can make
        one rank leave a step while its peers wait in a fused kernel.

        What this does NOT make safe: a non-uniform decision caused by a bug or a fault (the
        ``rank_native`` seam, a one-rank exception before launch). The step-end digest in
        ``hooks.check_step`` only sees steps that COMPLETE on every rank; a divergence that blocks
        mid-step (one rank in the native RCCL all_to_all, the others in the fused CCO kernel)
        never reaches it. Such a divergence is FAIL-STOP at the process-group timeout (E-3).
        """
        r = self.resolve(ctx.padded_tokens, self._runtime_mode())
        reason = r.reason
        if r.plan is None:
            return r, reason, None, None
        if faults.active("rank_native", hooks.sp_rank()) is not None:
            # TEST SEAM: deliberately non-uniform. Fail-stop by design (E-3): the peers block in
            # their fused kernels, so the step-end digest is not reached.
            return Resolution(None, 0, "native", "fault_rank_native"), "fault_rank_native", None, None
        path, why = self.decide_path(
            resolution=r,
            is_dummy_run=ctx.is_dummy_run,
            real_tokens=ctx.real_tokens,
            padded_tokens=ctx.padded_tokens,
            x_dtype_is_fp8=(x_scale is not None and x.dtype == torch.float8_e4m3fn),
            x_scale_ok=(
                x_scale is not None
                and x_scale.dim() == 2
                and tuple(x_scale.shape) == (x.shape[0], 1)
                and x_scale.dtype == torch.float32
            ),
        )
        if path == "native":
            return Resolution(None, 0, "native", why), why, None, None
        if r.n_fused < 2:
            return r, reason, None, None
        st = self._staging[ctx.padded_tokens]  # prepared on every rank (agree_prepared)
        state = ctx.prologue_state()  # eligibility was agreed at resolve; never None here
        if self.bind_mode == "direct":
            # x_scale / positions / slot_mapping may have moved since prepare: refreshed by copy
            # in _execute (a device-side copy, no collective, no fallback). The KV addresses were
            # checked at prepare (agree_prepared), not here.
            live = self._live_ptrs(x_scale, ctx.positions, state)
            st.rebound = live[:3] != st.ptrs[:3]
        return r, reason, state, st
    def _execute(self, r, x, x_scale, ctx, state, st, out_dtype):
        """Run the fused prefix; return its ``outs`` dict."""
        res = hooks.resources()
        res.finalize()  # collective + idempotent; before any plan executes
        if r.n_fused >= 2:
            if self.bind_mode == "staged" or getattr(st, "rebound", False):
                # staged: always. direct: only when x_scale/positions/slot_mapping
                # moved since prepare -- the plan is bound to the held buffers, so
                # refresh them in place (self-heal) rather than fall back on one rank.
                n = ctx.real_tokens
                st.xs.copy_(x_scale)
                st.pos[:n].copy_(ctx.positions[:n])
                st.slot[:n].copy_(state["slot_mapping"][:n])
            # direct, unmoved: st.xs/pos/slot ARE the caller's tensors; no copies
            outs, meta, xs = st.outs, self.build_meta(st, state), st.xs
            # the plan keeps its pointer bindings per INSTANCE; that the plan launched here is the
            # one prepared is verified at prepare time (`_prepared_ok` via `agree_prepared`),
            # not per call, so no rank can raise here while its peers launch.
            self._dump_binding("run", ctx.padded_tokens, st, state, xs)
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

    def _decision(self, ctx, r, reason, path, st=None):
        bm = "raw" if (path == "fused" and r.n_fused < 2) else self.bind_mode
        if path == "fused" and st is not None and getattr(st, "rebound", False):
            bm += "+copy"  # direct bind refreshed by copy (rank-local, not in the digest)
        d = Decision(self.name, ctx.padded_tokens, r.vid, r.n_fused, path, reason, bm)
        hooks.record_decision(d)
        return d

    def _remainder_after_exchange(self, r, outs, ctx, consumer):
        """Run the native remainder on a large-M recv_view. L-4: the view is only
        valid on the launching stream until this rank's next epoch of the plan, so
        the launch and every consumer must be on ``torch.cuda.current_stream()``."""
        s0 = torch.cuda.current_stream()
        out = consumer(outs["packed"])
        if torch.cuda.current_stream() != s0:
            raise GemmFusionBindError("recv_view consumer ran on a different stream")
        delay = faults.active("slow_remainder", hooks.sp_rank())
        if delay is not None:  # TEST SEAM (E5): widen the WAR window on this rank
            _spin_gpu_us(int(delay or "2000"))
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
            return "outs", outs, self._decision(ctx, r, reason, "fused", st)
        out = self._remainder_after_exchange(r, outs, ctx, native_after_exchange)
        return "native", out, self._decision(ctx, r, reason, "fused", st)


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


def _spin_gpu_us(us: int) -> None:
    """Test seam: keep the current stream busy ~``us`` microseconds (E5)."""
    try:
        torch.cuda._sleep(int(us * 1000))  # cycles; order-of-magnitude is enough
    except Exception:
        torch.cuda.synchronize()
