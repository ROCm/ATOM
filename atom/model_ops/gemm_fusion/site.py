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

from . import faults, hooks
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
    def _static_prologue_ok(self) -> bool:
        """Static (config/env) eligibility of the fused norm/rope/cache stage,
        folded into the cross-rank agreement so it can never differ per call."""
        impl = getattr(self.attn, "impl", None)
        fn = getattr(impl, "gemm_fusion_prologue_supported", None)
        return True if fn is None else bool(fn())

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
            plan, n = gf.match_prefix(self.spec, res.match_ctx(res.local_m(tokens), graph_mode))
        except contract_err as e:  # covered but contract fails
            errored, msg = True, str(e)
        if plan is not None and n >= 2 and not self._static_prologue_ok():
            plan, n = None, 0  # agreed below => native on every rank
        outcome = res.agree(self.name, tokens, plan, n, errored=errored)
        if outcome == "error":
            r = GemmFusionContractError(msg or "contract error on all ranks")
            self._by_tokens[key] = r
            raise r
        if outcome == "disagree" or plan is None:
            reason = "rank_disagree" if outcome == "disagree" else "uncovered"
            r = Resolution(None, 0, "native", reason)
        elif not res.can_ensure(plan):
            r = Resolution(None, 0, "native", "after_finalize")  # uniform: same order on all ranks
        else:
            res.ensure(plan)
            hooks.register_plan(plan)
            r = Resolution(plan, n, _plan_vid(plan), "profile_hit")
        self._by_tokens[key] = r
        return r

    def _runtime_mode(self) -> str:
        return "capture" if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing() else "eager"

    def agree_prepared(self, tokens: int, prepared: bool) -> None:
        """Collective: every rank must have prepared (or not) the same buckets.
        A mismatch downgrades the bucket to native on ALL ranks."""
        res = hooks.resources()
        for mode in ("eager", "capture"):
            r = self._by_tokens.get((tokens, mode))
            if r is None or isinstance(r, Exception) or r.plan is None:
                continue
            outcome = res.agree(self.name + "#prepared", tokens, None, int(bool(prepared)))
            if outcome != "agree" or not prepared:
                self._by_tokens[(tokens, mode)] = Resolution(
                    None, 0, "native", "prepare_disagree" if outcome != "agree" else "unprepared"
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
        if st.prepared:
            return True
        if self.bind_mode == "direct":
            if direct is None and self.direct_provider is not None:
                direct = self.direct_provider(tokens)
            if direct is None:
                return False  # no stable tensors to bind; agree_prepared() downgrades uniformly
            state = direct["state"]
            st.xs, st.pos, st.slot = direct["x_scale"], direct["positions"], direct["slot_mapping"]
            st.ptrs = self._live_ptrs(st.xs, st.pos, state)
        else:
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
        """-> (Resolution, reason, state, staging).

        Rank-uniformity of every fallback decided here (M-1). A one-rank-only
        fallback would hang or poison the flag chain, so each is either uniform by
        construction or impossible:
          dummy_run   : ``batch.is_dummy_run`` comes from ONE pickled batch that
                        ``AsyncIOProcManager.call_func`` broadcasts to every worker
                        of the engine core (rpc_broadcast_mq.enqueue); SP ranks share
                        that engine core. Uniform.
          pad_rows    : ``real_tokens`` = positions.shape[-1] (global, positions stay
                        whole under SP) and ``padded_tokens`` = local rows * world with
                        equal local rows on every rank. Uniform.
          unquantized_input / xscale_layout: x dtype and [M,1] shape follow from the
                        model (SPMD) and equal local rows. Uniform.
          prologue support: static impl/env eligibility, agreed at resolve time.
          prepared/bound : agreed collectively at prepare (``agree_prepared``).
        ``check_step`` additionally all-reduces a digest of every decision made in a
        step, turning any violation into a recovered, raised error (not a hang).
        """
        r = self.resolve(ctx.padded_tokens, self._runtime_mode())
        reason = r.reason
        if r.plan is None:
            return r, reason, None, None
        if faults.active("rank_native", hooks.sp_rank()) is not None:
            # TEST SEAM: deliberately non-uniform; the step digest must catch it
            return Resolution(None, 0, "native", "fault_rank_native"), "fault_rank_native", None, None
        bad = self._fallback_reason(ctx, x, x_scale, r.n_fused)
        if bad is not None:
            return Resolution(None, 0, "native", bad), bad, None, None
        if r.n_fused < 2:
            return r, reason, None, None
        state = ctx.prologue_state()
        if state is None:  # static eligibility was agreed at resolve: this is a bug
            raise GemmFusionBindError(f"{self.name}: prologue state unavailable after agreed resolve")
        st = self._staging.get(ctx.padded_tokens)
        if self.bind_mode == "staged":
            if st is None or not st.prepared:
                raise GemmFusionBindError(f"{self.name}: bucket {ctx.padded_tokens} not prepared")
            return r, reason, state, st
        # direct: bound at prepare to the caller's tensors
        if st is None or not st.prepared:
            raise GemmFusionBindError(f"{self.name}: direct bucket {ctx.padded_tokens} not prepared")
        live = self._live_ptrs(x_scale, ctx.positions, state)
        if live[3:] != st.ptrs[3:]:
            # KV cache / scale addresses are fixed at KV allocation; a change means
            # bind_kv_cache must run again. Never launch against stale pointers.
            raise GemmFusionBindError(f"{self.name}: KV cache address changed after prepare")
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
